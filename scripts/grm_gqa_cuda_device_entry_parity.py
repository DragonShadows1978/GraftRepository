#!/usr/bin/env python3
"""W2 device-pointer route entry validation (GRM_CUDA_BRIDGE_OVERHEAD_PLAN P2).

Proves `grm_gqa_cuda_arena_route_device` (device-resident query pointer, no
H2D copy inside the call) and `grm_gqa_cuda_arena_route` (host pointer, the
existing entry) produce IDENTICAL top-k on the same bank/queries. Uses the
self-contained test-only harness (`CudaProbe.upload_queries` ->
`grm_gqa_cuda_upload_queries`/`grm_gqa_cuda_free_device_ptr` in the sidecar
source itself) rather than a cross-repo tensor_cuda dependency — per the
plan's "Prefer the self-contained harness" note.

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

from scripts import grm_gqa_router_benchmark as bench  # noqa: E402
from scripts.grm_gqa_cuda_probe import build_probe, CudaProbe  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=sorted(bench.PRESETS), default="qwen35-2b-attn")
    p.add_argument("--capture-dir", type=Path, required=True)
    p.add_argument("--capture-role", default="source")
    p.add_argument("--capture-layer", type=int, default=3)
    p.add_argument("--capture-limit", type=int, default=128)
    p.add_argument("--node-count", type=int, default=32)
    p.add_argument("--queries", type=int, default=4)
    p.add_argument("--query-tokens", type=int, default=4)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--nvcc", default="nvcc")
    p.add_argument("--out", type=Path,
                   default=Path("artifacts/grm_gqa_cuda_bridge/device_entry_parity.json"))
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    nvcc = shutil.which(args.nvcc)
    if not nvcc:
        raise RuntimeError(f"nvcc not found: {args.nvcc}")
    preset = bench.PRESETS[args.preset]
    routes, capture_stats = bench.load_capture_routes(
        args.capture_dir,
        role=args.capture_role,
        layer=args.capture_layer,
        limit=args.capture_limit,
        token_limit=0,
        kv_heads=int(preset["kv_heads"]),
        head_dim=int(preset["head_dim"]),
    )
    if routes.shape[1] != 1:
        raise RuntimeError("device-entry parity check expects one route key per node")
    if routes.shape[0] < args.node_count:
        raise RuntimeError(
            f"requested {args.node_count} nodes but only loaded {routes.shape[0]}")
    keys = np.ascontiguousarray(routes[:args.node_count, 0], dtype=np.float32)
    query_count = max(1, int(args.queries))
    queries = bench.make_capture_queries(
        keys[:, None, :, :, :], query_count=query_count,
        query_heads=int(preset["query_heads"]), query_tokens=args.query_tokens)
    queries = np.ascontiguousarray(queries, dtype=np.float32)

    with tempfile.TemporaryDirectory(prefix="grm-gqa-cuda-device-parity-") as td:
        build_dir = Path(td)
        lib = build_probe(build_dir, nvcc)
        probe = CudaProbe(lib)
        arena = probe.create_arena(keys)
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
        "schema": "grm-gqa-cuda-device-entry-parity-v1",
        "preset": args.preset,
        "capture": capture_stats,
        "nodes": int(keys.shape[0]),
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
