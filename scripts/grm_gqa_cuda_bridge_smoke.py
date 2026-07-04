#!/usr/bin/env python3
"""Live smoke for the opt-in GQAArenaCache CUDA route bridge.

This validates the runtime bridge, not just the standalone CUDA probe:

- builds the normal dependency-free native GRM C ABI
- loads real Qwen3.5 capture K banks read-only
- syncs those banks into a NativeGraftStore
- routes through GQAArenaCache with GRM_GQA_CUDA_ROUTE=1
- checks the bridge result against the batched Python raw q.k law
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.graft_arena import GQAArenaCache  # noqa: E402
from core.grm_native import NativeGraftStore  # noqa: E402
from scripts import grm_gqa_router_benchmark as bench  # noqa: E402
from scripts import grm_router_baseline as baseline  # noqa: E402


def _write_markdown(result: dict, path: Path) -> None:
    receipt = result.get("last_cuda_receipt") or {}
    lines = [
        "# GRM GQA CUDA Bridge Smoke",
        "",
        "Runtime bridge validation for `GRM_GQA_CUDA_ROUTE=1`.",
        "",
        "## Result",
        "",
        f"- parity: {str(result['parity']).lower()}",
        f"- backend: `{result['backend']}`",
        f"- nodes: {result['nodes']}",
        f"- queries: {result['queries']}",
        f"- topk: {result['topk']}",
        f"- capture: `{result['capture']['capture_dir']}`",
        f"- layer/key: {result['capture']['layer']} / `{result['capture']['key_name']}`",
        f"- key shape: {result['shape']['key_bank']}",
        f"- query shape: {result['shape']['queries']}",
        f"- first bridge wall ms: {result['bridge_wall_ms_first']:.4f}",
        f"- min bridge wall ms: {result['bridge_wall_ms_min']:.4f}",
        f"- last direct CUDA route wall ms: {receipt.get('route_wall_ms', 0.0):.4f}",
        f"- last direct CUDA device ms/query: "
        f"{receipt.get('device_route_ms_per_query', 0.0):.4f}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_native_store(lib_path: Path, preset: dict, route_layer: int) -> NativeGraftStore:
    return NativeGraftStore(
        str(lib_path),
        model_type=preset["model_type"],
        num_layers=int(preset["num_layers"]),
        hidden_dim=int(preset["hidden_dim"]),
        vals_per_tok_layer=int(preset["vals_per_tok_layer"]),
        route_layer=int(route_layer),
        payload_kind="gqa",
        num_kv_heads=int(preset["kv_heads"]),
        head_dim=int(preset["head_dim"]),
    )


def _populate_store(store: NativeGraftStore, routes: np.ndarray) -> list[int]:
    node_ids: list[int] = []
    for idx, keys in enumerate(routes):
        node_id = store.add_node(f"cuda bridge route node {idx}", b"", ntok=1)
        store.set_route_key_list(node_id, [np.asarray(keys[0], dtype=np.float32)])
        node_ids.append(int(node_id))
    return node_ids


def _make_arena(store: NativeGraftStore, node_ids: list[int], routes: np.ndarray) -> GQAArenaCache:
    arena = GQAArenaCache.__new__(GQAArenaCache)
    arena.native_store = store
    arena.last_route_backend = "python"
    arena._cuda_gqa_route_unavailable = False
    arena.grafts = [
        {
            "native_node_id": int(node_id),
            "cent": np.asarray(routes[idx, 0], dtype=np.float32),
            "rare": set(),
            "text": f"cuda bridge route node {idx}",
            "kind": "doc",
        }
        for idx, node_id in enumerate(node_ids)
    ]
    return arena


def run_smoke(args: argparse.Namespace) -> dict:
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
        raise RuntimeError("CUDA bridge smoke expects one route key per node")
    if routes.shape[0] < args.node_count:
        raise RuntimeError(
            f"requested {args.node_count} nodes but loaded {routes.shape[0]}")
    routes = np.ascontiguousarray(routes[:args.node_count], dtype=np.float32)
    query_count = max(int(args.queries), int(args.warmup) + 1)
    queries = bench.make_capture_queries(
        routes,
        query_count=query_count,
        query_heads=int(preset["query_heads"]),
        query_tokens=int(args.query_tokens),
    )
    if int(args.topk) > 16:
        raise RuntimeError("CUDA bridge smoke is limited by sidecar topk <= 16")

    old_env = os.environ.get("GRM_GQA_CUDA_ROUTE")
    os.environ["GRM_GQA_CUDA_ROUTE"] = "1"
    bridge_wall_ms: list[float] = []
    bridge_results: list[list[int]] = []
    direct_results: list[list[int]] = []
    expected_results: list[list[int]] = []
    mismatches: list[dict] = []
    last_receipt = None
    try:
        with tempfile.TemporaryDirectory(prefix="grm-gqa-cuda-bridge-") as td:
            build_dir = Path(td)
            lib_path = baseline.build_native_lib(
                build_dir, cxx=args.cxx, openmp=True)
            with _build_native_store(lib_path, preset, args.capture_layer) as store:
                node_ids = _populate_store(store, routes)
                arena = _make_arena(store, node_ids, routes)
                for qi, query in enumerate(queries):
                    expected = bench.python_gqa_reference_route(
                        routes, query, (), int(args.topk), "batched")
                    arena._probe_key = lambda _text, q=query: q
                    t0 = time.perf_counter_ns()
                    got = arena.route(
                        args.probe_text, exclude=set(), limit=int(args.topk))
                    wall_ms = (time.perf_counter_ns() - t0) / 1.0e6
                    if arena.last_route_backend != "cuda":
                        raise RuntimeError(
                            f"bridge used {arena.last_route_backend}, not cuda")
                    direct, receipt = store.route_gqa_cuda(
                        query,
                        topk=int(args.topk),
                        warmup=0,
                        route_repeats=int(args.route_repeats),
                        return_receipt=True,
                    )
                    bridge_wall_ms.append(float(wall_ms))
                    bridge_results.append([int(x) for x in got])
                    direct_results.append([int(x) for x in direct])
                    expected_results.append([int(x) for x in expected])
                    last_receipt = receipt
                    if got != expected or direct != expected:
                        mismatches.append({
                            "query": int(qi),
                            "expected": [int(x) for x in expected],
                            "bridge": [int(x) for x in got],
                            "direct_cuda": [int(x) for x in direct],
                        })
    finally:
        if old_env is None:
            os.environ.pop("GRM_GQA_CUDA_ROUTE", None)
        else:
            os.environ["GRM_GQA_CUDA_ROUTE"] = old_env

    result = {
        "schema": "grm_gqa_cuda_bridge_smoke_v1",
        "preset": args.preset,
        "capture": capture_stats,
        "nodes": int(routes.shape[0]),
        "queries": int(queries.shape[0]),
        "topk": int(args.topk),
        "backend": "cuda",
        "parity": not mismatches,
        "mismatches": mismatches,
        "shape": {
            "routes": [int(x) for x in routes.shape],
            "key_bank": [int(x) for x in routes[:, 0].shape],
            "queries": [int(x) for x in queries.shape],
        },
        "bridge_wall_ms_first": float(bridge_wall_ms[0]) if bridge_wall_ms else 0.0,
        "bridge_wall_ms_min": float(np.min(bridge_wall_ms)) if bridge_wall_ms else 0.0,
        "bridge_wall_ms_runs": bridge_wall_ms,
        "bridge_results": bridge_results,
        "direct_cuda_results": direct_results,
        "expected_results": expected_results,
        "last_cuda_receipt": asdict(last_receipt) if last_receipt is not None else None,
    }
    if mismatches:
        raise RuntimeError(f"CUDA bridge parity failed: {mismatches[:2]}")
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=sorted(bench.PRESETS), default="qwen35-2b-attn")
    p.add_argument("--capture-dir", type=Path, required=True)
    p.add_argument("--capture-role", default="source")
    p.add_argument("--capture-layer", type=int, default=3)
    p.add_argument("--capture-limit", type=int, default=128)
    p.add_argument("--node-count", type=int, default=32)
    p.add_argument("--queries", type=int, default=2)
    p.add_argument("--query-tokens", type=int, default=4)
    p.add_argument("--token-limit", type=int, default=0)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--route-repeats", type=int, default=2)
    p.add_argument("--probe-text", default="plain probe")
    p.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    p.add_argument("--out", type=Path)
    p.add_argument("--markdown-out", type=Path)
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    result = run_smoke(args)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if args.markdown_out:
        _write_markdown(result, args.markdown_out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
