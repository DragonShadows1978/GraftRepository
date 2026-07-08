#!/usr/bin/env python3
"""P0 attribution profiler for the opt-in GQAArenaCache CUDA route bridge.

New file, read-only w.r.t. product code — this is a profiling harness, not a
product change. It reuses the exact same setup path as
`scripts/grm_gqa_cuda_bridge_smoke.py` (same native store build, same capture
loading, same GQAArenaCache construction) but instead of timing one call per
query, it:

  1. Attaches the CUDA route bank once (first `arena.route()` call — cold,
     includes `configure_cuda_gqa_route_bank` + signature hash).
  2. Repeats the SAME reused route call `--profile-repeats` times inside a
     single `cProfile.Profile()` region, so steady-state reused-route cost
     dominates the profile (the P0 ask: isolate reused calls from first-call
     bank setup).
  3. Also records manual `time.perf_counter_ns()` wall time per reused call,
     both under cProfile and in an uninstrumented control run, so cProfile's
     own dispatch overhead can be sanity-checked against a clean baseline.

Deliverable: pstats table (top cumulative-time functions), attributed to the
plan's named suspects, written under artifacts/grm_gqa_cuda_bridge/.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.graft_arena import GQAArenaCache  # noqa: E402
from core.grm_native import NativeGraftStore  # noqa: E402
from scripts import grm_gqa_router_benchmark as bench  # noqa: E402
from scripts import grm_router_baseline as baseline  # noqa: E402
from scripts.grm_gqa_cuda_bridge_smoke import (  # noqa: E402
    _build_native_store,
    _make_arena,
    _populate_store,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=sorted(bench.PRESETS), default="qwen35-2b-attn")
    p.add_argument("--capture-dir", type=Path, required=True)
    p.add_argument("--capture-role", default="source")
    p.add_argument("--capture-layer", type=int, default=3)
    p.add_argument("--capture-limit", type=int, default=128)
    p.add_argument("--node-count", type=int, default=32)
    p.add_argument("--query-tokens", type=int, default=4)
    p.add_argument("--token-limit", type=int, default=0)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--profile-repeats", type=int, default=200,
                    help="reused arena.route() calls inside the cProfile region")
    p.add_argument("--control-repeats", type=int, default=200,
                    help="reused arena.route() calls in an UNINSTRUMENTED "
                         "control run (no cProfile), for overhead sanity check")
    p.add_argument("--probe-text", default="plain probe")
    p.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    p.add_argument("--out-prefix", type=Path,
                    default=Path("artifacts/grm_gqa_cuda_bridge/profile"))
    p.add_argument("--top-n", type=int, default=25)
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    preset = bench.PRESETS[args.preset]
    routes, capture_stats = bench.load_capture_routes(
        args.capture_dir,
        role=args.capture_role,
        layer=args.capture_layer,
        limit=args.capture_limit,
        token_limit=args.token_limit,
        kv_heads=int(preset["kv_heads"]),
        head_dim=int(preset["head_dim"]),
    )
    if routes.shape[1] != 1:
        raise RuntimeError("CUDA bridge profile expects one route key per node")
    if routes.shape[0] < args.node_count:
        raise RuntimeError(
            f"requested {args.node_count} nodes but loaded {routes.shape[0]}")
    routes = np.ascontiguousarray(routes[:args.node_count], dtype=np.float32)
    # single fixed query reused for every repeat -- steady state, no query
    # variety needed since we are attributing HOST overhead, not routing
    # correctness (parity is already receipted by the smoke script).
    queries = bench.make_capture_queries(
        routes, query_count=1, query_heads=int(preset["query_heads"]),
        query_tokens=int(args.query_tokens))
    query = queries[0]
    if int(args.topk) > 16:
        raise RuntimeError("CUDA bridge profile is limited by sidecar topk <= 16")

    old_env = os.environ.get("GRM_GQA_CUDA_ROUTE")
    os.environ["GRM_GQA_CUDA_ROUTE"] = "1"
    result: dict = {"node_count": int(args.node_count)}
    try:
        with tempfile.TemporaryDirectory(prefix="grm-gqa-cuda-bridge-profile-") as td:
            build_dir = Path(td)
            lib_path = baseline.build_native_lib(build_dir, cxx=args.cxx, openmp=True)
            with _build_native_store(lib_path, preset, args.capture_layer) as store:
                node_ids = _populate_store(store, routes)
                arena = _make_arena(store, node_ids, routes)
                arena._probe_key = lambda _text, q=query: q

                # --- cold call: attaches the CUDA bank (bank build + hash
                # signature). NOT part of the reused-route attribution.
                t0 = time.perf_counter_ns()
                got_cold = arena.route(args.probe_text, exclude=set(), limit=int(args.topk))
                cold_wall_ms = (time.perf_counter_ns() - t0) / 1.0e6
                if arena.last_route_backend != "cuda":
                    raise RuntimeError(
                        f"bridge used {arena.last_route_backend}, not cuda "
                        "(cuda sidecar unavailable?)")
                result["cold_wall_ms"] = cold_wall_ms
                result["cold_route"] = [int(x) for x in got_cold]

                # --- uninstrumented control: clean reused-call wall times,
                # no cProfile dispatch overhead, for cross-check.
                control_wall_ms = []
                for _ in range(int(args.control_repeats)):
                    t0 = time.perf_counter_ns()
                    arena.route(args.probe_text, exclude=set(), limit=int(args.topk))
                    control_wall_ms.append((time.perf_counter_ns() - t0) / 1.0e6)
                result["control_wall_ms_runs"] = control_wall_ms
                result["control_wall_ms_min"] = float(np.min(control_wall_ms))
                result["control_wall_ms_mean"] = float(np.mean(control_wall_ms))
                result["control_wall_ms_median"] = float(np.median(control_wall_ms))

                # --- profiled region: N reused calls under cProfile.
                profiler = cProfile.Profile()
                profiled_wall_ms = []
                profiler.enable()
                for _ in range(int(args.profile_repeats)):
                    t0 = time.perf_counter_ns()
                    arena.route(args.probe_text, exclude=set(), limit=int(args.topk))
                    profiled_wall_ms.append((time.perf_counter_ns() - t0) / 1.0e6)
                profiler.disable()
                result["profiled_wall_ms_runs"] = profiled_wall_ms
                result["profiled_wall_ms_min"] = float(np.min(profiled_wall_ms))
                result["profiled_wall_ms_mean"] = float(np.mean(profiled_wall_ms))
                result["profile_repeats"] = int(args.profile_repeats)

                out_prefix = args.out_prefix
                out_prefix.parent.mkdir(parents=True, exist_ok=True)
                raw_path = out_prefix.with_name(
                    out_prefix.name + f"_{args.node_count}n.pstats")
                profiler.dump_stats(str(raw_path))
                result["pstats_path"] = str(raw_path)

                # pstats text table, cumulative-time sorted, per-call ms
                # (divide raw seconds*1000/repeats since these are the sums
                # over --profile-repeats calls).
                buf = io.StringIO()
                stats = pstats.Stats(profiler, stream=buf)
                stats.sort_stats("cumulative")
                stats.print_stats(int(args.top_n))
                txt_path = out_prefix.with_name(
                    out_prefix.name + f"_{args.node_count}n_cumulative.txt")
                txt_path.write_text(buf.getvalue(), encoding="utf-8")
                result["pstats_text_path"] = str(txt_path)

                # Also grab raw stats dict for programmatic top-N extraction
                # (function, cumulative seconds, own calls).
                top_rows = []
                stats2 = pstats.Stats(profiler)
                for func, (cc, nc, tt, ct, callers) in stats2.stats.items():
                    top_rows.append({
                        "file": func[0], "line": func[1], "func": func[2],
                        "ncalls_primitive": cc, "ncalls_total": nc,
                        "tottime_s": tt, "cumtime_s": ct,
                    })
                top_rows.sort(key=lambda r: -r["cumtime_s"])
                result["top_by_cumtime"] = top_rows[: int(args.top_n) * 2]
    finally:
        if old_env is None:
            os.environ.pop("GRM_GQA_CUDA_ROUTE", None)
        else:
            os.environ["GRM_GQA_CUDA_ROUTE"] = old_env
    return result


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    result = run(args)
    import json
    print(json.dumps(
        {k: v for k, v in result.items() if k != "top_by_cumtime"},
        indent=2, sort_keys=True, default=str))
    print("\n--- top by cumulative time (profiled region, sum over "
          f"{result['profile_repeats']} repeats) ---")
    for row in result["top_by_cumtime"][: args.top_n]:
        per_call_ms = row["cumtime_s"] * 1000.0 / result["profile_repeats"]
        print(f"{per_call_ms:9.5f} ms/call  cum={row['cumtime_s']*1000:9.3f}ms "
              f"calls={row['ncalls_total']:>6}  "
              f"{row['file']}:{row['line']}({row['func']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
