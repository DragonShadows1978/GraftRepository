#!/usr/bin/env python3
"""Dense GQA route-card CPU/CUDA stable-top-k parity receipt."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.grm_native import NativeGraftStore  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--slots", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--query-tokens", type=int, default=3)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--out", type=Path,
        default=Path("artifacts/gqa_route_card/dev/cpu_cuda_parity.json"))
    return parser.parse_args()


def build_native(root: Path) -> tuple[str, tempfile.TemporaryDirectory]:
    temp = tempfile.TemporaryDirectory(prefix="grm-route-card-native-")
    lib = Path(temp.name) / "libgrm_runtime.so"
    subprocess.run([
        "g++", "-std=c++17", "-O3", "-shared", "-fPIC", "-pthread",
        "-I", str(root / "cpp"), str(root / "cpp" / "grm_runtime.cpp"),
        "-o", str(lib),
    ], check=True)
    return str(lib), temp


def main() -> int:
    args = parse_args()
    if args.query_heads % args.kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    rng = np.random.default_rng(args.seed)
    cards = rng.normal(size=(
        args.nodes, args.slots, args.kv_heads, args.head_dim)).astype(np.float32)
    # Spread norms so top-k margins are not dominated by exact random ties.
    cards *= np.linspace(0.75, 1.25, args.nodes, dtype=np.float32)[:, None, None, None]
    queries = rng.normal(size=(
        args.queries, args.query_heads, args.query_tokens,
        args.head_dim)).astype(np.float32)

    lib, temp = build_native(ROOT)
    store = NativeGraftStore(
        lib, model_type="RouteCardParity", num_layers=1,
        hidden_dim=args.query_heads * args.head_dim,
        vals_per_tok_layer=args.kv_heads * args.head_dim * 2,
        route_layer=0, payload_kind="gqa", num_kv_heads=args.kv_heads,
        head_dim=args.head_dim)
    node_ids = []
    try:
        for node in range(args.nodes):
            node_id = store.add_node(f"route card parity node {node}", b"", 1)
            node_ids.append(node_id)
            store.set_route_card(
                node_id, cards[node], state="ready",
                dialect_profile_hash="d" * 64,
                source_text_sha256=f"{node:064x}",
                prompt_version="gqa-route-card-v1",
                generator_version="parity-v1")
        cpu_started = time.perf_counter()
        cpu = [store.route_gqa(q, topk=args.topk) for q in queries]
        cpu_ms = (time.perf_counter() - cpu_started) * 1000.0

        cuda_bank = np.ascontiguousarray(
            np.transpose(cards, (0, 2, 1, 3)), dtype=np.float32)
        store.configure_cuda_gqa_route_bank(
            cuda_bank, np.asarray(node_ids, dtype=np.uint64))
        cuda, receipt = store.route_gqa_cuda(
            queries, topk=args.topk, warmup=2, route_repeats=5,
            return_receipt=True)
        receipt_json = (asdict(receipt) if is_dataclass(receipt)
                        else dict(getattr(receipt, "__dict__", {})))
        parity = [list(a) == list(b) for a, b in zip(cpu, cuda)]
        payload = {
            "schema": "grm_gqa_route_card_cpu_cuda_parity_v1",
            "evidence_tier": "development-visible-synthetic-parity",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "nodes": args.nodes,
            "queries": args.queries,
            "slots": args.slots,
            "kv_heads": args.kv_heads,
            "query_heads": args.query_heads,
            "head_dim": args.head_dim,
            "query_tokens": args.query_tokens,
            "topk": args.topk,
            "parity_count": int(sum(parity)),
            "parity_total": len(parity),
            "all_equal": bool(all(parity)),
            "cpu_total_ms": cpu_ms,
            "cuda_receipt": receipt_json,
            "mismatches": [{
                "query": i, "cpu": cpu[i], "cuda": cuda[i]}
                for i, equal in enumerate(parity) if not equal],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps({
            "out": str(args.out), "all_equal": payload["all_equal"],
            "parity": f"{payload['parity_count']}/{payload['parity_total']}",
            "cpu_total_ms": cpu_ms, "cuda_receipt": receipt_json,
        }, indent=2, default=str))
        return 0 if payload["all_equal"] else 1
    finally:
        store.close()
        temp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
