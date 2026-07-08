#!/usr/bin/env python3
"""P0 production wall-curve: through-Python `ArenaCache.route()` at
1k/10k/100k/1M, for comparison against the historic native-only curve in
docs/GRM_GEMV_ROUTER_PLAN.md (P3 bounded-staging: 1M nodes, 23.8805ms p50 /
25.4883ms p95, native-only ctypes call, never through ArenaCache.route()).

New file, read-only w.r.t. product code. Reuses scripts/grm_router_baseline.py
harvest/native-lib-build machinery and scripts/grm_mla_route_profile.py's
arena-construction helpers (same MLA NativeGraftStore shape, same `_probe_key`
stub so the model-forward cost of a real probe is excluded -- this measures
routing-wrapper wall time only, uninstrumented, no cProfile).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_router_baseline as baseline  # noqa: E402
from scripts.grm_mla_route_profile import (  # noqa: E402
    _build_native_store,
    _populate_store,
    _make_arena,
)


def _median_ms(values_ns: list[int]) -> float:
    return statistics.median(values_ns) / 1.0e6


def _p95_ms(values_ns: list[int]) -> float:
    if len(values_ns) < 2:
        return values_ns[0] / 1.0e6
    return statistics.quantiles(values_ns, n=20, method="inclusive")[18] / 1.0e6


def run_one(node_count: int, *, dim: int, max_vectors: int, max_files: int,
            topk: int, warmup: int, queries: int, lib_path: Path,
            int4: bool, refine_m: int) -> dict:
    bank, harvest = baseline.harvest_centroid_bank(
        ROOT, dim=dim, max_vectors=max_vectors, max_files=max_files)
    routes = baseline.make_route_matrix(bank, int(node_count))
    probes = baseline.make_queries(bank, max(queries, warmup + 1))

    old_int4 = os.environ.get("GRM_ROUTER_INT4")
    old_refine = os.environ.get("GRM_ROUTER_INT4_REFINE_M")
    if int4:
        os.environ["GRM_ROUTER_INT4"] = "1"
        os.environ["GRM_ROUTER_INT4_REFINE_M"] = str(int(refine_m))
    else:
        os.environ.pop("GRM_ROUTER_INT4", None)
        os.environ.pop("GRM_ROUTER_INT4_REFINE_M", None)
    try:
        with _build_native_store(lib_path, dim=dim) as store:
            node_ids = _populate_store(store, routes)
            arena = _make_arena(store, node_ids, routes)
            wall_ns = []
            backend = None
            for i, probe in enumerate(probes):
                arena._probe_key = lambda _t, q=probe: q
                if i < warmup:
                    arena.route("warmup probe", exclude=set(), limit=int(topk))
                    continue
                t0 = time.perf_counter_ns()
                arena.route("wall curve probe", exclude=set(), limit=int(topk))
                wall_ns.append(time.perf_counter_ns() - t0)
                backend = arena.last_route_backend
    finally:
        if old_int4 is None:
            os.environ.pop("GRM_ROUTER_INT4", None)
        else:
            os.environ["GRM_ROUTER_INT4"] = old_int4
        if old_refine is None:
            os.environ.pop("GRM_ROUTER_INT4_REFINE_M", None)
        else:
            os.environ["GRM_ROUTER_INT4_REFINE_M"] = old_refine

    return {
        "node_count": int(node_count),
        "mode": "int4_bounded_staging" if int4 else "fp32",
        "refine_m": int(refine_m) if int4 else None,
        "backend": backend,
        "queries": len(wall_ns),
        "route_ms_p50": _median_ms(wall_ns),
        "route_ms_p95": _p95_ms(wall_ns),
        "route_ms_min": min(wall_ns) / 1.0e6,
        "route_ms_max": max(wall_ns) / 1.0e6,
        "harvest": harvest,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--node-counts", nargs="+", type=int,
                   default=[1000, 10000, 100000, 1000000])
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--max-vectors", type=int, default=8192)
    p.add_argument("--max-files", type=int, default=2048)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--queries", type=int, default=12)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--int4", action="store_true")
    p.add_argument("--refine-m", type=int, default=128)
    p.add_argument("--openmp", action="store_true", default=True)
    p.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    p.add_argument("--out", type=Path,
                   default=Path("artifacts/grm_mla_route/wall_curve.json"))
    p.add_argument("--loadavg-note", default="")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="grm-mla-wall-curve-") as td:
        lib_path = baseline.build_native_lib(
            Path(td), cxx=args.cxx, openmp=bool(args.openmp))
        results = []
        for node_count in args.node_counts:
            row = run_one(
                node_count, dim=args.dim, max_vectors=args.max_vectors,
                max_files=args.max_files, topk=args.topk, warmup=args.warmup,
                queries=args.queries, lib_path=lib_path, int4=False,
                refine_m=args.refine_m)
            results.append(row)
            if args.int4:
                row2 = run_one(
                    node_count, dim=args.dim, max_vectors=args.max_vectors,
                    max_files=args.max_files, topk=args.topk,
                    warmup=args.warmup, queries=args.queries,
                    lib_path=lib_path, int4=True, refine_m=args.refine_m)
                results.append(row2)
    output = {
        "schema": "grm_mla_route_wall_curve_v1",
        "loadavg_note": args.loadavg_note,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True, default=str),
                        encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
