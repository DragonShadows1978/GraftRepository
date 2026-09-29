#!/usr/bin/env python3
"""Qualified one-shot Qwen3.5-9B hybrid route-card development replay.

This continuation consumes the exact capture paths and query locations frozen
by ``grm_gqa_route_card_dev_sweep.py``.  ``qualify`` only checks that the first
two nodes produce grounded JSON aliases and layer-3 native K.  ``full`` reuses
those receipts, completes all nodes, and scores the registered {2,4,8} cards.
Neither phase opens fresh evidence.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TENSOR_CUDA_ROOT = Path("/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
for path in (ROOT, TENSOR_CUDA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import tensor_cuda as tc  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from core.graft_repository import GraftRepository  # noqa: E402
from core.qwen35_tc import Qwen35_TC  # noqa: E402


DEFAULT_MODEL = Path(
    "/home/vader/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.5-9B/snapshots/"
    "c202236235762e1c871ad0ccb60c8ee5ba337b9a")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("qualify", "full"))
    parser.add_argument(
        "--source-sweep", type=Path,
        default=Path(
            "artifacts/gqa_route_card/dev/qwen35_slot_sweep.json"))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--aliases", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--qualification-nodes", type=int, default=2)
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(
            "artifacts/gqa_route_card/dev/qwen35_hybrid_v1"))
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")


def sha256_bytes(value) -> str:
    return hashlib.sha256(bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    temp.replace(path)


def clear_transients():
    gc.collect()
    if hasattr(tc, "empty_cache"):
        tc.empty_cache()


def chat_ids(tokenizer, prompt: str) -> np.ndarray:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    return tokenizer(
        rendered, return_tensors="np", add_special_tokens=False,
    ).input_ids.astype(np.int64)


def greedy_json(model, tokenizer, prompt: str, max_new_tokens: int):
    ids = chat_ids(tokenizer, prompt)
    started = time.perf_counter()
    out = []
    parsed = None
    with tc.no_grad():
        logits, caches = model(ids, last_token_only=True)
        next_token = int(logits.float().numpy()[0, -1].argmax())
        offset = int(ids.shape[1])
        for _ in range(int(max_new_tokens)):
            if next_token == int(model.config.eos_token_id):
                break
            out.append(next_token)
            text = tokenizer.decode(
                out, skip_special_tokens=True,
                clean_up_tokenization_spaces=False).strip()
            try:
                parsed = GraftRepository._route_card_json_payload(text)
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed = None
            if isinstance(parsed, (list, dict)):
                break
            logits, caches = model(
                np.asarray([[next_token]], dtype=np.int64), caches=caches,
                position_offset=offset, last_token_only=True)
            offset += 1
            next_token = int(logits.float().numpy()[0, -1].argmax())
    tc.synchronize()
    elapsed = time.perf_counter() - started
    text = tokenizer.decode(
        out, skip_special_tokens=True,
        clean_up_tokenization_spaces=False).strip()
    del logits, caches
    clear_transients()
    return text, {
        "prompt_tokens": int(ids.shape[1]),
        "generated_tokens": len(out),
        "elapsed_sec": elapsed,
        "json_shape": (type(parsed).__name__ if parsed is not None else None),
    }


def capture_layer_k(model, token_ids: np.ndarray, layer: int) -> np.ndarray:
    block = model.layers[int(layer)]
    attention = getattr(block, "self_attn", None)
    if attention is None and getattr(block, "is_attn", False):
        attention = getattr(block, "mixer", None)
    if attention is None:
        raise RuntimeError(f"Qwen3.5 layer {layer} is not an attention layer")
    attention._capture = True
    attention._captured = None
    result = None
    try:
        with tc.no_grad():
            result = model(
                np.asarray([token_ids], dtype=np.int64),
                last_token_only=True, max_layers=int(layer) + 1)
        captured = attention._captured
        if captured is None:
            raise RuntimeError(f"Qwen3.5 layer {layer} produced no K capture")
        key = np.ascontiguousarray(captured[0][0], dtype=np.float32)
    finally:
        attention._capture = False
        attention._captured = None
        if result is not None:
            del result
        clear_transients()
    return key


def raw_score(query: np.ndarray, key: np.ndarray) -> float:
    q_heads, _, dim = query.shape
    repeated = np.repeat(key, q_heads // key.shape[0], axis=0)
    scores = np.einsum(
        "hqd,hkd->hqk", query.astype(np.float32, copy=False),
        repeated.astype(np.float32, copy=False), optimize=True)
    return float(np.abs(scores).max(axis=(1, 2)).mean() / np.sqrt(dim))


def ranking(query: np.ndarray, banks: list[np.ndarray]) -> list[int]:
    rows = [(raw_score(query, bank), node_id)
            for node_id, bank in enumerate(banks)]
    rows = [row for row in rows if np.isfinite(row[0])]
    rows.sort(key=lambda item: (-item[0], item[1]))
    return [node_id for _, node_id in rows]


def provisional_card(source_key: np.ndarray, slots: int) -> np.ndarray:
    logical, _ = GraftRepository._select_route_card_keys(source_key, slots)
    return np.ascontiguousarray(np.transpose(logical, (1, 0, 2)))


def hybrid_card(source_key: np.ndarray, alias_key: np.ndarray | None,
                slots: int) -> np.ndarray:
    if alias_key is None or alias_key.shape[1] == 0:
        return provisional_card(source_key, slots)
    source_all, _ = GraftRepository._select_route_card_keys(source_key, slots)
    alias_slots = slots // 2
    source_slots = slots - alias_slots
    alias_all, _ = GraftRepository._select_route_card_keys(
        alias_key, alias_slots)
    distinct_alias_slots = min(alias_slots, int(alias_key.shape[1]))
    pieces = [source_all[:source_slots], alias_all[:distinct_alias_slots]]
    fill = slots - sum(int(piece.shape[0]) for piece in pieces)
    if fill:
        pieces.append(source_all[source_slots:source_slots + fill])
    logical = np.ascontiguousarray(np.concatenate(pieces, axis=0))
    return np.ascontiguousarray(np.transpose(logical, (1, 0, 2)))


def load_sweep(path: Path):
    payload = json.loads(path.read_text())
    if payload.get("schema") != "grm_gqa_route_card_qwen35_dev_sweep_v1":
        raise RuntimeError("source sweep has an unexpected schema")
    if int(payload.get("seed", -1)) != 20260711:
        raise RuntimeError("source sweep seed is not the frozen 20260711 seed")
    captures = payload.get("capture_files", ())
    if len(captures) != int(payload.get("node_count", -1)) or len(captures) != 32:
        raise RuntimeError("source sweep is not the frozen 32-node selection")
    if len(payload.get("queries", ())) != 20:
        raise RuntimeError("source sweep is not the frozen 20-query selection")
    return payload


def protocol_payload(args, sweep_path: Path):
    return {
        "schema": "grm_gqa_route_card_hybrid_protocol_v1",
        "source_sweep": str(sweep_path.resolve()),
        "source_sweep_sha256": sha256_file(sweep_path),
        "model_dir": str(args.model_dir.resolve()),
        "layer": int(args.layer),
        "prompt_version": "gqa-route-card-v1",
        "aliases": int(args.aliases),
        "max_new_tokens": int(args.max_new_tokens),
        "selector": "raw-maxnorm-diverse-v1",
        "validation": "exact-evidence-source-identifiers-v1",
        "card_policy": "half-source-half-alias-v1",
    }


def load_receipt(path: Path, protocol_hash: str):
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    if receipt.get("protocol_sha256") != protocol_hash:
        raise RuntimeError(f"receipt protocol mismatch: {path}")
    return receipt


def process_node(model, tokenizer, validator, capture_entry, node_id: int,
                 args, protocol_hash: str):
    capture_path = Path(capture_entry["path"])
    actual_sha = sha256_file(capture_path)
    if actual_sha != capture_entry["sha256"]:
        raise RuntimeError(f"capture hash changed: {capture_path}")
    with np.load(capture_path, allow_pickle=False) as shard:
        token_ids = np.asarray(shard["token_ids"], dtype=np.int64).reshape(-1)
    source_text = tokenizer.decode(
        token_ids.tolist(), skip_special_tokens=True,
        clean_up_tokenization_spaces=False)
    prompt = validator._route_card_prompt(source_text)
    generated = ""
    accepted = []
    validation_receipts = []
    generation = {}
    error = None
    alias_banks = []
    alias_offsets = [0]
    started = time.perf_counter()
    try:
        generated, generation = greedy_json(
            model, tokenizer, prompt, args.max_new_tokens)
        accepted, validation_receipts = validator._validate_route_card_aliases(
            generated, source_text)
        if not accepted:
            raise ValueError("librarian produced no validated aliases")
        for alias in accepted:
            ids = np.asarray(tokenizer(
                alias["query"], add_special_tokens=False).input_ids,
                dtype=np.int64)
            if ids.size == 0:
                raise RuntimeError("accepted alias tokenized to an empty query")
            key = capture_layer_k(model, ids, args.layer)
            alias_banks.append(key)
            alias_offsets.append(alias_offsets[-1] + int(key.shape[1]))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    alias_key = (np.ascontiguousarray(np.concatenate(alias_banks, axis=1))
                 if alias_banks else None)
    key_path = args.out_dir / f"node_{node_id:03d}_alias_k.npz"
    if alias_key is not None:
        np.savez_compressed(
            key_path, alias_k=alias_key,
            alias_offsets=np.asarray(alias_offsets, dtype=np.int64))
    elif key_path.exists():
        key_path.unlink()
    receipt = {
        "schema": "grm_gqa_route_card_hybrid_node_v1",
        "created_at": utc_now(),
        "protocol_sha256": protocol_hash,
        "node_id": int(node_id),
        "capture_path": str(capture_path),
        "capture_sha256": actual_sha,
        "source_text_sha256": sha256_bytes(source_text.encode("utf-8")),
        "token_ids_sha256": sha256_bytes(
            np.ascontiguousarray(token_ids).view(np.uint8)),
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "generated_text": generated,
        "generated_text_sha256": sha256_bytes(generated.encode("utf-8")),
        "generation": generation,
        "accepted_aliases": accepted,
        "accepted_alias_count": len(accepted),
        "validation_receipts": validation_receipts,
        "alias_k_path": str(key_path) if alias_key is not None else None,
        "alias_k_shape": list(alias_key.shape) if alias_key is not None else None,
        "alias_offsets": alias_offsets if alias_key is not None else [],
        "qualification_pass": bool(
            error is None and accepted and alias_key is not None and
            alias_key.ndim == 3 and alias_key.shape[0] == 4 and
            alias_key.shape[2] == 256),
        "error": error,
        "elapsed_sec": time.perf_counter() - started,
    }
    write_json_atomic(args.out_dir / f"node_{node_id:03d}.json", receipt)
    return receipt


def score_full(args, sweep, receipts, protocol_hash):
    key_name = f"l{args.layer}_k"
    query_name = f"l{args.layer}_q"
    source_keys = []
    source_queries = []
    alias_keys = []
    for node_id, entry in enumerate(sweep["capture_files"]):
        capture_path = Path(entry["path"])
        with np.load(capture_path, allow_pickle=False) as shard:
            source_keys.append(np.asarray(
                shard[key_name][0], dtype=np.float32))
            source_queries.append(np.asarray(
                shard[query_name][0], dtype=np.float32))
        receipt = receipts[node_id]
        alias_path = receipt.get("alias_k_path")
        if alias_path:
            with np.load(alias_path, allow_pickle=False) as alias_shard:
                alias_keys.append(np.asarray(
                    alias_shard["alias_k"], dtype=np.float32))
        else:
            alias_keys.append(None)

    query_payload = []
    full_hits = 0
    full_top3 = []
    full_ranks = []
    for item in sweep["queries"]:
        node_id = int(item["source_node"])
        start = int(item["token_start"])
        query = np.ascontiguousarray(
            source_queries[node_id][:, start:start + int(sweep["query_tokens"])])
        ranked = ranking(query, source_keys)
        rank = ranked.index(node_id) + 1
        full_hits += int(rank <= 3)
        full_ranks.append(rank)
        full_top3.append(ranked[:3])
        query_payload.append((node_id, query))
    if full_hits != int(sweep["full_source_recall_at_3"]):
        raise RuntimeError("full-ragged replay no longer matches source receipt")

    sizes = {}
    for slots in (2, 4, 8):
        provisional = [provisional_card(key, slots) for key in source_keys]
        hybrid = [hybrid_card(source_keys[i], alias_keys[i], slots)
                  for i in range(len(source_keys))]
        provisional_hits = 0
        hybrid_hits = 0
        provisional_ranks = []
        hybrid_ranks = []
        hybrid_top3_equal = 0
        for ordinal, (node_id, query) in enumerate(query_payload):
            provisional_ranked = ranking(query, provisional)
            hybrid_ranked = ranking(query, hybrid)
            provisional_rank = provisional_ranked.index(node_id) + 1
            hybrid_rank = hybrid_ranked.index(node_id) + 1
            provisional_ranks.append(provisional_rank)
            hybrid_ranks.append(hybrid_rank)
            provisional_hits += int(provisional_rank <= 3)
            hybrid_hits += int(hybrid_rank <= 3)
            hybrid_top3_equal += int(hybrid_ranked[:3] == full_top3[ordinal])
        expected_provisional = int(
            sweep["sizes"][str(slots)]["source_recall_at_3"])
        if provisional_hits != expected_provisional:
            raise RuntimeError(
                f"{slots}-slot provisional replay changed: "
                f"{provisional_hits} vs {expected_provisional}")
        sizes[str(slots)] = {
            "full_source_recall_at_3": full_hits,
            "provisional_source_recall_at_3": provisional_hits,
            "hybrid_source_recall_at_3": hybrid_hits,
            "hybrid_no_worse_than_full": hybrid_hits >= full_hits,
            "hybrid_no_worse_than_provisional": (
                hybrid_hits >= provisional_hits),
            "passes_qwen_hybrid_gate": (
                hybrid_hits >= full_hits and hybrid_hits >= provisional_hits),
            "provisional_source_ranks": provisional_ranks,
            "hybrid_source_ranks": hybrid_ranks,
            "hybrid_exact_full_top3_count": hybrid_top3_equal,
        }

    accepted = sum(int(r.get("accepted_alias_count", 0)) for r in receipts)
    reasons = {}
    for receipt in receipts:
        for validation in receipt.get("validation_receipts", ()):
            reason = str(validation.get("reason", "unknown"))
            reasons[reason] = reasons.get(reason, 0) + 1
    passing = [int(slots) for slots, value in sizes.items()
               if value["passes_qwen_hybrid_gate"]]
    result = {
        "schema": "grm_gqa_route_card_qwen35_hybrid_dev_v1",
        "evidence_tier": "development-visible",
        "created_at": utc_now(),
        "protocol_sha256": protocol_hash,
        "source_sweep": str(args.source_sweep),
        "node_count": len(receipts),
        "query_count": len(query_payload),
        "layer": int(args.layer),
        "full_source_recall_at_3": full_hits,
        "full_source_ranks": full_ranks,
        "accepted_alias_count": accepted,
        "nodes_with_accepted_aliases": sum(
            int(r.get("accepted_alias_count", 0) > 0) for r in receipts),
        "validation_reason_counts": reasons,
        "unsupported_identifier_acceptances": 0,
        "sizes": sizes,
        "smallest_qwen_hybrid_pass": min(passing) if passing else None,
        "line_survives_qwen_continuation": bool(passing),
        "development_gate_complete": False,
        "incomplete_reasons": [
            "GPT-OSS replayable probe Q tensors remain unavailable",
            "fresh Qwen3.5 and GPT-OSS gates remain sealed",
        ],
        "node_receipts": [
            str(args.out_dir / f"node_{i:03d}.json")
            for i in range(len(receipts))],
    }
    write_json_atomic(args.out_dir / "qwen35_hybrid_sweep.json", result)
    return result


def main() -> int:
    args = parse_args()
    if args.layer != 3 or args.aliases != 8 or args.max_new_tokens != 384:
        raise RuntimeError(
            "the registered continuation fixes layer=3, aliases=8, "
            "and max_new_tokens=384")
    sweep = load_sweep(args.source_sweep)
    protocol = protocol_payload(args, args.source_sweep)
    protocol_hash = sha256_bytes(canonical_json(protocol))
    protocol["protocol_sha256"] = protocol_hash
    protocol["registered_at"] = utc_now()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = args.out_dir / "protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text())
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("existing hybrid protocol does not match")
    else:
        write_json_atomic(protocol_path, protocol)

    if args.phase == "full":
        qualification_path = args.out_dir / "qualification.json"
        if not qualification_path.exists() or not json.loads(
                qualification_path.read_text()).get("qualification_pass"):
            raise RuntimeError(
                "run the registered qualify phase successfully before full")

    target_count = (int(args.qualification_nodes)
                    if args.phase == "qualify" else len(sweep["capture_files"]))
    if args.qualification_nodes != 2:
        raise RuntimeError("the registered qualification count is two nodes")
    receipts = []
    missing = []
    for node_id in range(target_count):
        receipt_path = args.out_dir / f"node_{node_id:03d}.json"
        receipt = load_receipt(receipt_path, protocol_hash)
        receipts.append(receipt)
        if receipt is None:
            missing.append(node_id)

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir), local_files_only=True)
    model = None
    if missing:
        print(f"loading Qwen3.5-9B for {len(missing)} node(s)", flush=True)
        model, info = Qwen35_TC.from_pretrained(str(args.model_dir))
        print(json.dumps({"model": info, "protocol_sha256": protocol_hash},
                         indent=2, default=str), flush=True)
        validator = GraftRepository.__new__(GraftRepository)
        validator.route_card_aliases = int(args.aliases)
        try:
            for ordinal, node_id in enumerate(missing, start=1):
                print(
                    f"node {node_id:03d} ({ordinal}/{len(missing)}): "
                    "generate, validate, harvest", flush=True)
                receipt = process_node(
                    model, tokenizer, validator,
                    sweep["capture_files"][node_id], node_id, args,
                    protocol_hash)
                receipts[node_id] = receipt
                print(json.dumps({
                    "node_id": node_id,
                    "accepted_alias_count": receipt["accepted_alias_count"],
                    "qualification_pass": receipt["qualification_pass"],
                    "error": receipt["error"],
                    "generation": receipt["generation"],
                    "elapsed_sec": receipt["elapsed_sec"],
                }, indent=2), flush=True)
        finally:
            del model
            clear_transients()

    qualification = receipts[:2]
    qualification_pass = (
        len(qualification) == 2 and
        all(r is not None and r.get("qualification_pass")
            for r in qualification))
    qualification_result = {
        "schema": "grm_gqa_route_card_hybrid_qualification_v1",
        "created_at": utc_now(),
        "protocol_sha256": protocol_hash,
        "nodes": [{
            "node_id": int(r["node_id"]),
            "accepted_alias_count": int(r["accepted_alias_count"]),
            "alias_k_shape": r.get("alias_k_shape"),
            "qualification_pass": bool(r.get("qualification_pass")),
            "error": r.get("error"),
        } for r in qualification if r is not None],
        "qualification_pass": qualification_pass,
        "retrieval_scored": False,
    }
    write_json_atomic(args.out_dir / "qualification.json", qualification_result)
    if not qualification_pass:
        print(json.dumps(qualification_result, indent=2), flush=True)
        return 2
    if args.phase == "qualify":
        print(json.dumps(qualification_result, indent=2), flush=True)
        return 0

    if any(receipt is None for receipt in receipts):
        raise RuntimeError("full phase ended with missing node receipts")
    result = score_full(args, sweep, receipts, protocol_hash)
    print(json.dumps({
        "out": str(args.out_dir / "qwen35_hybrid_sweep.json"),
        "accepted_alias_count": result["accepted_alias_count"],
        "nodes_with_accepted_aliases": result["nodes_with_accepted_aliases"],
        "full_recall_at_3": result["full_source_recall_at_3"],
        "hybrid_recall_at_3": {
            slots: value["hybrid_source_recall_at_3"]
            for slots, value in result["sizes"].items()},
        "smallest_qwen_hybrid_pass": result["smallest_qwen_hybrid_pass"],
        "line_survives_qwen_continuation": (
            result["line_survives_qwen_continuation"]),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
