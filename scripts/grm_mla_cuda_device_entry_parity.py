#!/usr/bin/env python3
"""Device-pointer MLA route entry validation (P2,
docs/GRM_MLA_CUDA_ROUTE_PLAN.md).

Proves `grm_mla_cuda_arena_route_device` (device-resident query pointer, no
H2D copy inside the call) and `grm_mla_cuda_arena_route` (host pointer, the
existing entry) produce IDENTICAL top-k on the same bank/queries. Uses the
self-contained test-only harness (`CudaProbe.upload_queries` ->
`grm_mla_cuda_upload_queries`/`grm_mla_cuda_free_device_ptr` in the sidecar
source itself), mirroring scripts/grm_gqa_cuda_device_entry_parity.py.

New file, product code untouched by running it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_router_baseline as baseline  # noqa: E402
from scripts.grm_mla_cuda_probe import build_probe, CudaProbe  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--node-count", type=int, default=2000)
    p.add_argument("--queries", type=int, default=6)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--nvcc", default="nvcc")
    p.add_argument("--out", type=Path,
                   default=Path(
                       "artifacts/grm_mla_cuda_route/device_entry_parity.json"))
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    nvcc = shutil.which(args.nvcc)
    if not nvcc:
        raise RuntimeError(f"nvcc not found: {args.nvcc}")

    bank, _ = baseline.harvest_centroid_bank(
        ROOT, dim=args.dim, max_vectors=8192, max_files=2048)
    rows = baseline.make_route_matrix(bank, args.node_count)
    query_count = max(1, int(args.queries))
    queries = baseline.make_queries(bank, query_count)
    queries = np.ascontiguousarray(queries, dtype=np.float32)

    with tempfile.TemporaryDirectory(prefix="grm-mla-cuda-device-parity-") as td:
        build_dir = Path(td)
        lib = build_probe(build_dir, nvcc)
        probe = CudaProbe(lib)
        arena = probe.create_arena(rows)
        try:
            host_topk, host_device_ms, host_wall_ms = arena.route_topk(
                queries, warmup=0, topk=int(args.topk))

            device_ptr = probe.upload_queries(queries)
            try:
                dev_topk, dev_device_ms, dev_wall_ms = arena.route_topk_device(
                    device_ptr, warmup=0, topk=int(args.topk))
            finally:
                device_ptr.close()

            match = bool(np.array_equal(host_topk, dev_topk))
        finally:
            arena.close()

    result = {
        "schema": "grm-mla-cuda-device-entry-parity-v1",
        "nodes": int(rows.shape[0]),
        "dim": int(args.dim),
        "queries": int(query_count),
        "topk": int(args.topk),
        "host_topk": host_topk.tolist(),
        "device_topk": dev_topk.tolist(),
        "match": match,
        "host_device_ms": float(host_device_ms),
        "host_wall_ms": float(host_wall_ms),
        "device_device_ms": float(dev_device_ms),
        "device_wall_ms": float(dev_wall_ms),
    }
    if not match:
        raise RuntimeError(
            f"device-entry parity FAILED: host={host_topk.tolist()} "
            f"device={dev_topk.tolist()}")
    return result


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
