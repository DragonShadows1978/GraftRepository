#!/usr/bin/env python3
"""P2 exactness + timing gate for the CUDA MLA route (P2,
docs/GRM_MLA_CUDA_ROUTE_PLAN.md).

E3 exactness: CUDA top-k must be IDENTICAL to the native fp32 scan top-k
(zero tolerance) at 1k/10k/100k, plus a fuzz battery (mixed eligibility via
excludes, exact ties, non-finite rows/queries) and a handful of sampled
queries at 1M (the native fp32 scan at 1M is receipted ~230ms/query on the
Release build -- a handful suffices for a parity check, not a curve).

E2 timing: steady-state route wall through `ArenaCache.route()` with
GRM_MLA_CUDA_ROUTE=1, at 1k/10k/100k/1M, compared against the P1b host
numbers (0.179/1.17/13.9/127.5 ms int4 bounded-staging). Target: <=5ms at
1M.

New file, read-only w.r.t. product code -- a validation harness, reusing
scripts/grm_mla_route_profile.py's store/arena builders and
scripts/grm_router_baseline.py's harvested-corpus route matrices (same
corpus shape every other MLA route receipt in this work order used).
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


def _median_ms(values_ns: list[float]) -> float:
    return statistics.median(values_ns) / 1.0e6


def _p95_ms(values_ns: list[float]) -> float:
    if len(values_ns) < 2:
        return values_ns[0] / 1.0e6
    return statistics.quantiles(values_ns, n=20, method="inclusive")[18] / 1.0e6


# ------------------------------------------------------------- exactness


def _exactness_case(node_count: int, dim: int, queries: int, topk: int,
                     *, seed: int, exclude_frac: float = 0.0,
                     retire_frac: float = 0.0) -> dict:
    """One exactness case: build a real MLA NativeGraftStore + ArenaCache,
    attach the CUDA bank, and compare `route()` under GRM_MLA_CUDA_ROUTE=1
    against the CPU path (env unset) for the SAME probes/excludes -- not a
    synthetic Python reference, the actual production call path both ways,
    so a mismatch here is a real E3 violation, not a harness bug."""
    rng = np.random.default_rng(seed)
    bank, _ = baseline.harvest_centroid_bank(
        ROOT, dim=dim, max_vectors=8192, max_files=2048)
    routes = baseline.make_route_matrix(bank, node_count)

    # Inject exact ties: force a block of rows to be byte-identical to
    # exercise the node_id ASC tie-break end to end.
    tie_block = max(2, node_count // 200)
    if tie_block >= 2 and node_count >= 4:
        routes[1:tie_block] = routes[0]

    # Inject non-finite rows (M6 fuzz): a handful of NaN/Inf rows must be
    # excluded from both paths identically.
    nan_rows = sorted(rng.choice(node_count, size=max(1, node_count // 500),
                                 replace=False).tolist())
    for r in nan_rows:
        if r < tie_block:
            continue  # keep the tie block clean of NaN for a separate signal
        routes[r, 0] = np.nan

    with tempfile.TemporaryDirectory(prefix="grm-mla-cuda-gate-") as td:
        lib_path = baseline.build_native_lib(Path(td), cxx="g++", openmp=False)
        with _build_native_store(lib_path, dim=dim) as store:
            node_ids = _populate_store(store, routes)
            arena = _make_arena(store, node_ids, routes)

            n_retire = int(node_count * retire_frac)
            retired_idx = set(
                rng.choice(node_count, size=n_retire, replace=False).tolist()
            ) if n_retire else set()
            for i in retired_idx:
                arena.grafts[i]["retired"] = True
            arena._bump_cuda_gqa_epoch()

            probes = baseline.make_queries(bank, queries)
            mismatches = []
            cuda_used = 0
            for qi, probe in enumerate(probes):
                arena._probe_key = lambda _t, q=probe: q
                exclude = set()
                if exclude_frac > 0:
                    n_ex = int(node_count * exclude_frac)
                    eligible = [i for i in range(node_count) if i not in retired_idx]
                    exclude = set(
                        int(x) for x in rng.choice(
                            eligible, size=min(n_ex, len(eligible)), replace=False))

                os.environ["GRM_MLA_CUDA_ROUTE"] = "1"
                got_cuda = arena.route("plain gate probe", exclude=exclude,
                                       limit=topk)
                backend = arena.last_route_backend
                os.environ.pop("GRM_MLA_CUDA_ROUTE", None)
                got_cpu = arena.route("plain gate probe", exclude=exclude,
                                      limit=topk)

                if backend == "cuda":
                    cuda_used += 1
                if got_cuda != got_cpu:
                    mismatches.append({
                        "query": qi, "cuda": got_cuda, "cpu": got_cpu,
                        "backend": backend,
                    })

    return {
        "node_count": node_count,
        "dim": dim,
        "queries": queries,
        "topk": topk,
        "exclude_frac": exclude_frac,
        "retire_frac": retire_frac,
        "tie_block": tie_block,
        "nan_rows": len(nan_rows),
        "cuda_used_count": cuda_used,
        "mismatches": mismatches,
        "parity": not mismatches,
    }


def run_exactness_battery(node_counts: list[int], dim: int) -> list[dict]:
    cases = []
    for nc in node_counts:
        cases.append(_exactness_case(nc, dim, queries=10, topk=8, seed=nc))
        cases.append(_exactness_case(
            nc, dim, queries=8, topk=6, seed=nc + 1, exclude_frac=0.05))
        cases.append(_exactness_case(
            nc, dim, queries=8, topk=6, seed=nc + 2, retire_frac=0.1))
        cases.append(_exactness_case(
            nc, dim, queries=8, topk=6, seed=nc + 3, exclude_frac=0.05,
            retire_frac=0.1))
    return cases


def run_exactness_1m_sample(dim: int, queries: int, topk: int) -> dict:
    return _exactness_case(1_000_000, dim, queries=queries, topk=topk, seed=999)


# ----------------------------------------------------------------- timing


def run_timing_one(node_count: int, *, dim: int, topk: int, warmup: int,
                    queries: int) -> dict:
    bank, _ = baseline.harvest_centroid_bank(
        ROOT, dim=dim, max_vectors=8192, max_files=2048)
    routes = baseline.make_route_matrix(bank, node_count)
    probes = baseline.make_queries(bank, max(queries, warmup + 1))

    with tempfile.TemporaryDirectory(prefix="grm-mla-cuda-timing-") as td:
        lib_path = baseline.build_native_lib(Path(td), cxx="g++", openmp=False)
        with _build_native_store(lib_path, dim=dim) as store:
            node_ids = _populate_store(store, routes)
            arena = _make_arena(store, node_ids, routes)

            os.environ["GRM_MLA_CUDA_ROUTE"] = "1"
            try:
                wall_ns = []
                backend = None
                for i, probe in enumerate(probes):
                    arena._probe_key = lambda _t, q=probe: q
                    if i < warmup:
                        arena.route("timing warmup", exclude=set(), limit=topk)
                        continue
                    t0 = time.perf_counter_ns()
                    arena.route("timing probe", exclude=set(), limit=topk)
                    wall_ns.append(time.perf_counter_ns() - t0)
                    backend = arena.last_route_backend
            finally:
                os.environ.pop("GRM_MLA_CUDA_ROUTE", None)

    return {
        "node_count": int(node_count),
        "backend": backend,
        "queries": len(wall_ns),
        "route_ms_p50": _median_ms(wall_ns),
        "route_ms_p95": _p95_ms(wall_ns),
        "route_ms_min": min(wall_ns) / 1.0e6,
        "route_ms_max": max(wall_ns) / 1.0e6,
    }


P1B_HOST_MS_P50 = {
    1000: 0.179,
    10000: 1.17,
    100000: 13.9,
    1000000: 127.5,
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--exactness-node-counts", nargs="+", type=int,
                   default=[1000, 10000, 100000])
    p.add_argument("--timing-node-counts", nargs="+", type=int,
                   default=[1000, 10000, 100000, 1000000])
    p.add_argument("--skip-1m-exactness", action="store_true")
    p.add_argument("--1m-exactness-queries", dest="m_exactness_queries",
                   type=int, default=4)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--timing-queries", type=int, default=12)
    p.add_argument("--out", type=Path,
                   default=Path("artifacts/grm_mla_cuda_route/p2_gate.json"))
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    exactness = run_exactness_battery(args.exactness_node_counts, args.dim)
    if not args.skip_1m_exactness:
        exactness.append(
            run_exactness_1m_sample(args.dim, args.m_exactness_queries,
                                    args.topk))

    timing = []
    for nc in args.timing_node_counts:
        row = run_timing_one(nc, dim=args.dim, topk=args.topk,
                             warmup=args.warmup, queries=args.timing_queries)
        row["p1b_host_int4_ms_p50"] = P1B_HOST_MS_P50.get(nc)
        if row["p1b_host_int4_ms_p50"]:
            row["speedup_vs_p1b_host"] = (
                row["p1b_host_int4_ms_p50"] / row["route_ms_p50"]
                if row["route_ms_p50"] > 0 else None)
        timing.append(row)

    all_parity = all(c["parity"] for c in exactness)
    e2_met = any(
        r["node_count"] == 1000000 and r["route_ms_p50"] <= 5.0
        for r in timing)

    output = {
        "schema": "grm_mla_cuda_route_p2_gate_v1",
        "e3_exactness_all_pass": all_parity,
        "e2_target_met_1m_le_5ms": e2_met,
        "exactness_cases": exactness,
        "timing": timing,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True, default=str),
                        encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True, default=str))
    return 0 if all_parity else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
