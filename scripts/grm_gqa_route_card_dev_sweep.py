#!/usr/bin/env python3
"""One-shot development-visible slot sweep over existing Qwen3.5 captures.

This is not a fresh gate. It samples already-visible target K/Q shards and
compares self-source recall@3 under the full ragged raw-|q.k| law against
source-only fixed cards of sizes 2, 4, and 8.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.graft_repository import GraftRepository  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capture-dir", type=Path,
        default=Path("/mnt/ForgeRealm/qwen35_graft_translation_poc/captures"))
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--nodes", type=int, default=32)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--query-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--out", type=Path,
        default=Path("artifacts/gqa_route_card/dev/qwen35_slot_sweep.json"))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def raw_score(query: np.ndarray, key: np.ndarray) -> float:
    q_heads, _, dim = query.shape
    kv_heads = key.shape[0]
    repeated = np.repeat(key, q_heads // kv_heads, axis=0)
    scores = np.einsum(
        "hqd,hkd->hqk", query.astype(np.float32, copy=False),
        repeated.astype(np.float32, copy=False), optimize=True)
    return float(np.abs(scores).max(axis=(1, 2)).mean() / np.sqrt(dim))


def ranking(query: np.ndarray, banks: list[np.ndarray]) -> list[int]:
    rows = [(raw_score(query, key), node_id)
            for node_id, key in enumerate(banks)]
    rows = [row for row in rows if np.isfinite(row[0])]
    rows.sort(key=lambda item: (-item[0], item[1]))
    return [node_id for _, node_id in rows]


def main() -> int:
    args = parse_args()
    paths = sorted(args.capture_dir.glob("target_*.npz"))
    if len(paths) < args.nodes:
        raise RuntimeError(
            f"need {args.nodes} target captures, found {len(paths)}")
    rng = np.random.default_rng(args.seed)
    chosen_idx = np.sort(rng.choice(
        len(paths), size=int(args.nodes), replace=False))
    chosen = [paths[int(i)] for i in chosen_idx]
    key_name = f"l{args.layer}_k"
    query_name = f"l{args.layer}_q"
    keys, queries = [], []
    for path in chosen:
        with np.load(path, allow_pickle=False) as shard:
            key = np.asarray(shard[key_name][0], dtype=np.float32)
            query = np.asarray(shard[query_name][0], dtype=np.float32)
        keys.append(key)
        queries.append(query)

    q_count = min(int(args.queries), len(chosen))
    query_nodes = np.sort(rng.choice(len(chosen), size=q_count, replace=False))
    full_hits = 0
    full_rankings = []
    query_payload = []
    started = time.perf_counter()
    for ordinal, node_id in enumerate(query_nodes):
        query_bank = queries[int(node_id)]
        width = min(max(1, int(args.query_tokens)), query_bank.shape[1])
        start = (ordinal * 17 + int(node_id) * 7) % (
            query_bank.shape[1] - width + 1)
        query = np.ascontiguousarray(query_bank[:, start:start + width])
        ranked = ranking(query, keys)
        rank = ranked.index(int(node_id)) + 1
        full_hits += int(rank <= 3)
        full_rankings.append(ranked[:10])
        query_payload.append((int(node_id), int(start), query))

    sizes = {}
    for slots in (2, 4, 8):
        cards = []
        selected_positions = []
        for key in keys:
            logical, positions = GraftRepository._select_route_card_keys(
                key, slots)
            cards.append(np.ascontiguousarray(np.transpose(logical, (1, 0, 2))))
            selected_positions.append(positions.tolist())
        hits = 0
        source_ranks = []
        top3_equal = 0
        for q_ord, (node_id, _start, query) in enumerate(query_payload):
            ranked = ranking(query, cards)
            rank = ranked.index(node_id) + 1
            source_ranks.append(rank)
            hits += int(rank <= 3)
            top3_equal += int(ranked[:3] == full_rankings[q_ord][:3])
        sizes[str(slots)] = {
            "source_recall_at_3": hits,
            "full_source_recall_at_3": full_hits,
            "no_worse_than_full": hits >= full_hits,
            "source_ranks": source_ranks,
            "exact_full_top3_count": top3_equal,
            "selected_positions_sha256": hashlib.sha256(json.dumps(
                selected_positions, separators=(",", ":")).encode()).hexdigest(),
        }

    passing = [int(size) for size, result in sizes.items()
               if result["no_worse_than_full"]]
    payload = {
        "schema": "grm_gqa_route_card_qwen35_dev_sweep_v1",
        "evidence_tier": "development-visible",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "capture_dir": str(args.capture_dir),
        "layer": int(args.layer),
        "seed": int(args.seed),
        "node_count": len(chosen),
        "query_count": q_count,
        "query_tokens": int(args.query_tokens),
        "full_source_recall_at_3": full_hits,
        "sizes": sizes,
        "smallest_qwen_only_pass": min(passing) if passing else None,
        "development_winner": None,
        "development_gate_complete": False,
        "incomplete_reasons": [
            "GPT-OSS card reranking still requires replayable probe Q tensors",
            "hybrid alias-card comparison is not represented by this source-only sweep",
            "CPU/CUDA card parity is a separate gate",
        ],
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "capture_files": [{
            "path": str(path), "sha256": sha256_file(path)} for path in chosen],
        "queries": [{"source_node": node, "token_start": start}
                    for node, start, _ in query_payload],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "full_recall_at_3": full_hits,
        "sizes": {k: v["source_recall_at_3"] for k, v in sizes.items()},
        "smallest_qwen_only_pass": payload["smallest_qwen_only_pass"],
        "development_gate_complete": False,
        "elapsed_ms": payload["elapsed_ms"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
