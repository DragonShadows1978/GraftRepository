#!/usr/bin/env python3
"""P0 production-path profiler for MLA `ArenaCache.route()`.

New file, read-only w.r.t. product code — a profiling harness, not a product
change. Answers the plan's registered suspicion (docs/GRM_MLA_CUDA_ROUTE_PLAN.md):
the receipted 1M-node 23.88ms p50 bounded-staging number
(docs/GRM_GEMV_ROUTER_PLAN.md P3 notes) timed `NativeRouter.route()` — a bare
ctypes call directly into the C ABI (see scripts/grm_router_baseline.py). It
never went through `ArenaCache.route()`, which real turn execution calls
(`core/graft_arena.py::step()` line ~1224: `self.route(user_text,
exclude=live_idx, limit=route_limit)`). That wrapper does, per call:
  - `cand = [i for i in range(len(self.grafts)) if ...]`            O(N) Python
  - `qrare = self._rare_tokens(bare_text)`                          O(len(text))
  - `_native_route_order`: builds `native_to_idx` by walking `cand` again
    (O(N) Python dict build), calls `store.route(...)` (the native call),
    then re-maps `routed_native` back through `native_to_idx`            O(topk)
  - the `len(routed) != len(cand)` equivalence check (implicit O(N) via the
    prior dict) with a Python-scan fallback if native declines.

This harness builds a real MLA `NativeGraftStore`-backed `ArenaCache` (same
harvested route corpus + native lib build as `scripts/grm_router_baseline.py`)
and cProfiles reused, steady-state `arena.route(...)` calls at the requested
node counts, both flat fp32 and `GRM_ROUTER_INT4=1` bounded-staging (matching
the receipted M=128 operating point). `_probe_key` is stubbed to a fixed
harvested vector (as the GQA bridge profiler does) so the profile measures
ROUTING wrapper cost, not the unrelated model-forward cost of building a probe
key from text — that forward pass is a separate, already-understood cost
outside this plan's scope (P0 is about the routing wrapper, not model
inference).

Deliverable: pstats dumps + an attribution table (named Python buckets vs
native call) under artifacts/grm_mla_route/.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.graft_arena import ArenaCache  # noqa: E402
from core.grm_native import NativeGraftStore  # noqa: E402
from scripts import grm_router_baseline as baseline  # noqa: E402


def _median_ms(values_ns: list[int]) -> float:
    if not values_ns:
        return float("nan")
    return statistics.median(values_ns) / 1.0e6


def _p95_ms(values_ns: list[int]) -> float:
    if not values_ns:
        return float("nan")
    if len(values_ns) < 2:
        return values_ns[0] / 1.0e6
    return statistics.quantiles(values_ns, n=20, method="inclusive")[18] / 1.0e6


def _build_native_store(lib_path: Path, *, dim: int, route_layer: int = 0) -> NativeGraftStore:
    # Mirrors scripts/grm_router_baseline.py NativeRouter's MLA store shape
    # (model_type, num_layers, hidden_dim, vals_per_tok_layer, route_layer,
    # latent_rank, rope_dim) -- these are ABI metadata, not enforced against
    # the route vector length, which is whatever is passed to set_route().
    return NativeGraftStore(
        str(lib_path),
        model_type="RouterBaselineMLA",
        num_layers=1,
        hidden_dim=2048,
        vals_per_tok_layer=576,
        route_layer=route_layer,
        payload_kind="mla",
        latent_rank=512,
        rope_dim=64,
    )


def _populate_store(store: NativeGraftStore, routes: np.ndarray) -> list[int]:
    node_ids: list[int] = []
    for idx in range(routes.shape[0]):
        node_id = store.add_node(f"mla route node {idx}", b"", ntok=1)
        store.set_route(
            node_id, np.asarray(routes[idx], dtype=np.float32),
            lexical_keys=baseline.lexical_keys_for_node(idx))
        node_ids.append(int(node_id))
    return node_ids


def _make_arena(store: NativeGraftStore, node_ids: list[int],
                 routes: np.ndarray) -> ArenaCache:
    arena = ArenaCache.__new__(ArenaCache)
    arena.native_store = store
    arena.last_route_backend = "python"
    arena.grafts = [
        {
            "native_node_id": int(node_id),
            "cent": np.asarray(routes[idx], dtype=np.float32),
            "rare": set(),
            "text": f"mla route node {idx}",
            "kind": "turn",
        }
        for idx, node_id in enumerate(node_ids)
    ]
    return arena


def profile_one(*, arena: ArenaCache, probe_text: str, topk: int,
                 profile_repeats: int, control_repeats: int,
                 node_count: int, mode: str, out_prefix: Path,
                 top_n: int) -> dict:
    result: dict = {"node_count": int(node_count), "mode": mode}

    # cold call: first native_to_idx build, any first-touch cost.
    t0 = time.perf_counter_ns()
    got_cold = arena.route(probe_text, exclude=set(), limit=int(topk))
    cold_wall_ms = (time.perf_counter_ns() - t0) / 1.0e6
    result["cold_wall_ms"] = cold_wall_ms
    result["cold_route"] = [int(x) for x in got_cold]
    result["backend"] = arena.last_route_backend

    # uninstrumented control: clean reused wall times, no cProfile dispatch.
    control_wall_ms = []
    for _ in range(int(control_repeats)):
        t0 = time.perf_counter_ns()
        arena.route(probe_text, exclude=set(), limit=int(topk))
        control_wall_ms.append((time.perf_counter_ns() - t0) / 1.0e6)
    result["control_wall_ms_min"] = float(np.min(control_wall_ms)) if control_wall_ms else float("nan")
    result["control_wall_ms_median"] = float(np.median(control_wall_ms)) if control_wall_ms else float("nan")
    result["control_wall_ms_p95"] = (
        float(np.percentile(control_wall_ms, 95)) if control_wall_ms else float("nan"))

    # profiled region: N reused calls under cProfile.
    profiler = cProfile.Profile()
    profiled_wall_ms = []
    profiler.enable()
    for _ in range(int(profile_repeats)):
        t0 = time.perf_counter_ns()
        arena.route(probe_text, exclude=set(), limit=int(topk))
        profiled_wall_ms.append((time.perf_counter_ns() - t0) / 1.0e6)
    profiler.disable()
    result["profiled_wall_ms_min"] = float(np.min(profiled_wall_ms))
    result["profiled_wall_ms_median"] = float(np.median(profiled_wall_ms))
    result["profile_repeats"] = int(profile_repeats)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    tag = f"{node_count}n_{mode}"
    raw_path = out_prefix.with_name(out_prefix.name + f"_{tag}.pstats")
    profiler.dump_stats(str(raw_path))
    result["pstats_path"] = str(raw_path)

    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf)
    stats.sort_stats("cumulative")
    stats.print_stats(int(top_n))
    txt_path = out_prefix.with_name(out_prefix.name + f"_{tag}_cumulative.txt")
    txt_path.write_text(buf.getvalue(), encoding="utf-8")
    result["pstats_text_path"] = str(txt_path)

    top_rows = []
    stats2 = pstats.Stats(profiler)
    for func, (cc, nc, tt, ct, callers) in stats2.stats.items():
        top_rows.append({
            "file": func[0], "line": func[1], "func": func[2],
            "ncalls_primitive": cc, "ncalls_total": nc,
            "tottime_s": tt, "cumtime_s": ct,
        })
    top_rows.sort(key=lambda r: -r["cumtime_s"])
    result["top_by_cumtime"] = top_rows[: int(top_n) * 2]
    return result


def attribute(result: dict) -> dict:
    """Split the profiled cumulative time into named buckets vs the native
    ctypes call, using pstats OWN tottime per function (additive across the
    whole profiled region; cumtime double counts callees so is not used for
    the additive table). Built-in methods report file '~' in pstats, not a
    real path -- matched on function name alone.

    Named buckets map onto the plan's suspects:
      - route_own_cand_build: route()'s own frame tottime. The ONLY Python
        work in that frame besides dispatch is the
        `cand = [i for i in range(len(self.grafts)) if ...]` list
        comprehension -- an O(N) walk of every graft on every call.
      - native_route_order_own: _native_route_order()'s own frame tottime:
        the `native_to_idx = {}` build loop (O(N) over cand, one dict
        setitem + up to two dict .get per graft) plus loop/dispatch
        overhead. Does NOT include the remap loop's dict.get cost (that is
        counted separately below since it's a distinct built-in frame).
      - remap_dict_get: `{method 'get' of 'dict' objects}` -- the
        native_to_idx.get(node_id) calls in _native_route_order's build
        loop (g.get(...) x2/graft) AND its post-route remap loop (one
        native_to_idx.get() per id in routed_native). Because
        _native_route_order requests topk=len(self.grafts) from the native
        call (see module docstring), routed_native has N entries, so this
        remap alone is a full O(N) Python walk on every call.
      - remap_list_append: `{method 'append' of 'list' objects}` -- building
        `routed` (up to N appends) plus `cand` list comp appends (N) plus
        `scored` appends in the python-fallback path (0 when native
        backend is used).
    """
    repeats = result["profile_repeats"]
    buckets = {
        "native_call (store.route/_lib.grm_store_route*)": 0.0,
        "route_own_cand_build (route(): O(N) `cand` list comp)": 0.0,
        "native_route_order_own (native_to_idx build loop, O(N))": 0.0,
        "remap_dict_get ({method 'get' of 'dict'}: O(N), inflated by topk=len(grafts))": 0.0,
        "remap_list_append ({method 'append' of 'list'}: O(N))": 0.0,
        "rare_tokens (_rare_tokens)": 0.0,
        "python_fallback (_vector_route_scores/_cent_score/_normalize_scores/_lex_bonus)": 0.0,
        "ctypes_marshal (_float_array/_lexical_blob/_filter_blob/_check)": 0.0,
        "other": 0.0,
    }
    named_total = 0.0
    for row in result["top_by_cumtime"]:
        func = row["func"]
        file_ = row["file"]
        tot = row["tottime_s"]
        if "grm_native.py" in file_ and func == "route":
            buckets["native_call (store.route/_lib.grm_store_route*)"] += tot
        elif func == "<method 'get' of 'dict' objects>":
            buckets["remap_dict_get ({method 'get' of 'dict'}: O(N), inflated by topk=len(grafts))"] += tot
        elif func == "<method 'append' of 'list' objects>":
            buckets["remap_list_append ({method 'append' of 'list'}: O(N))"] += tot
        elif "graft_arena.py" in file_ and func == "route":
            buckets["route_own_cand_build (route(): O(N) `cand` list comp)"] += tot
        elif "graft_arena.py" in file_ and func == "_native_route_order":
            buckets["native_route_order_own (native_to_idx build loop, O(N))"] += tot
        elif "graft_arena.py" in file_ and func == "_rare_tokens":
            buckets["rare_tokens (_rare_tokens)"] += tot
        elif "graft_arena.py" in file_ and func in (
                "_vector_route_scores", "_cent_score", "_normalize_scores",
                "_lex_bonus"):
            buckets["python_fallback (_vector_route_scores/_cent_score/_normalize_scores/_lex_bonus)"] += tot
        elif "grm_native.py" in file_ and func in (
                "_float_array", "_lexical_blob", "_filter_blob", "_check"):
            buckets["ctypes_marshal (_float_array/_lexical_blob/_filter_blob/_check)"] += tot
        else:
            buckets["other"] += tot
        named_total += tot
    per_call_ms = {k: (v * 1000.0 / repeats) for k, v in buckets.items()}
    total_tottime_s = sum(buckets.values())
    return {
        "buckets_ms_per_call": per_call_ms,
        "total_named_tottime_ms_per_call": total_tottime_s * 1000.0 / repeats,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--node-counts", nargs="+", type=int, default=[100000, 1000000])
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--max-vectors", type=int, default=8192)
    p.add_argument("--max-files", type=int, default=2048)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--profile-repeats", type=int, default=8)
    p.add_argument("--control-repeats", type=int, default=8)
    p.add_argument("--int4", action="store_true", help="also profile GRM_ROUTER_INT4=1 bounded-staging")
    p.add_argument("--refine-m", type=int, default=128)
    p.add_argument("--openmp", action="store_true", default=True)
    p.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    p.add_argument("--probe-text", default="mla production route profile probe")
    p.add_argument("--out-prefix", type=Path, default=Path("artifacts/grm_mla_route/profile"))
    p.add_argument("--top-n", type=int, default=30)
    p.add_argument("--loadavg-note", default="")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    bank, harvest = baseline.harvest_centroid_bank(
        ROOT, dim=args.dim, max_vectors=args.max_vectors, max_files=args.max_files)

    all_results = []
    old_int4 = os.environ.get("GRM_ROUTER_INT4")
    old_refine = os.environ.get("GRM_ROUTER_INT4_REFINE_M")
    try:
        with tempfile.TemporaryDirectory(prefix="grm-mla-route-profile-") as td:
            lib_path = baseline.build_native_lib(
                Path(td), cxx=args.cxx, openmp=bool(args.openmp))
            for node_count in args.node_counts:
                routes = baseline.make_route_matrix(bank, int(node_count))
                probe = np.asarray(routes[0], dtype=np.float32)
                modes = [("fp32", False)]
                if args.int4:
                    modes.append(("int4_bounded_staging", True))
                for mode_name, use_int4 in modes:
                    if use_int4:
                        os.environ["GRM_ROUTER_INT4"] = "1"
                        os.environ["GRM_ROUTER_INT4_REFINE_M"] = str(int(args.refine_m))
                    else:
                        os.environ.pop("GRM_ROUTER_INT4", None)
                        os.environ.pop("GRM_ROUTER_INT4_REFINE_M", None)
                    with _build_native_store(lib_path, dim=args.dim) as store:
                        node_ids = _populate_store(store, routes)
                        arena = _make_arena(store, node_ids, routes)
                        arena._probe_key = lambda _text, q=probe: q
                        res = profile_one(
                            arena=arena, probe_text=args.probe_text,
                            topk=args.topk,
                            profile_repeats=args.profile_repeats,
                            control_repeats=args.control_repeats,
                            node_count=node_count, mode=mode_name,
                            out_prefix=args.out_prefix, top_n=args.top_n)
                        res["attribution"] = attribute(res)
                        res["refine_m"] = int(args.refine_m) if use_int4 else None
                        all_results.append(res)
    finally:
        if old_int4 is None:
            os.environ.pop("GRM_ROUTER_INT4", None)
        else:
            os.environ["GRM_ROUTER_INT4"] = old_int4
        if old_refine is None:
            os.environ.pop("GRM_ROUTER_INT4_REFINE_M", None)
        else:
            os.environ["GRM_ROUTER_INT4_REFINE_M"] = old_refine

    output = {
        "schema": "grm_mla_route_profile_v1",
        "harvest": harvest,
        "loadavg_note": args.loadavg_note,
        "results": [
            {k: v for k, v in r.items() if k != "top_by_cumtime"}
            for r in all_results
        ],
    }
    out_json = args.out_prefix.with_name(args.out_prefix.name + "_summary.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2, sort_keys=True, default=str),
                        encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True, default=str))
    for r in all_results:
        print(f"\n--- {r['node_count']}n {r['mode']} attribution (ms/call) ---")
        for k, v in r["attribution"]["buckets_ms_per_call"].items():
            print(f"  {v:9.5f}  {k}")
        print(f"  control median wall: {r['control_wall_ms_median']:.5f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
