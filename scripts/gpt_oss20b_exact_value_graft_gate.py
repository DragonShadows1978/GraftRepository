#!/usr/bin/env python3
"""GPT-OSS exact code/name/number cold-graft recall gate.

This H6 helper plants multi-token exact values into one real-token capture,
captures pre-RoPE K/V graft shards, then asks fact-local generated-answer
questions with and without the mounted graft. It covers the Phase 5
code/name/number axis that single-token color facts do not exercise.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

THIS_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(THIS_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(THIS_REPO_ROOT))

from scripts.gpt_oss20b_multifact_graft_gate import (
    ASSISTANT_FINAL,
    REPO_ROOT,
    SNAPSHOT,
    SYSTEM_PREFIX,
    read_corpus,
    run_child,
    sha256_bytes,
    stream_capture_cmd,
    summarize_greedy,
    token_ids,
    write_json,
)


DEFAULT_RECORDS = [
    {
        "id": "asset_code",
        "letter": "A",
        "kind": "code",
        "label": "GPT-OSS asset code",
        "answer": "ZX-47B",
    },
    {
        "id": "operator_name",
        "letter": "B",
        "kind": "name",
        "label": "GPT-OSS operator name",
        "answer": "MIRA SOL",
    },
    {
        "id": "access_number",
        "letter": "C",
        "kind": "number",
        "label": "GPT-OSS access number",
        "answer": "7391-2048",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GPT-OSS exact code/name/number graft gate."
    )
    parser.add_argument("--model-dir", default=SNAPSHOT)
    parser.add_argument("--corpus-dir", type=Path, default=REPO_ROOT / "docs")
    parser.add_argument("--target-tokens", type=int, default=4096)
    parser.add_argument("--graft-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--attention-mode",
        choices=("standard", "apa_selective"),
        default="apa_selective",
    )
    parser.add_argument("--apa-layer-scope", choices=("full", "all"), default="full")
    parser.add_argument("--refine-percentile", type=float, default=0.15)
    parser.add_argument("--bulk-bits", type=int, default=8)
    parser.add_argument(
        "--expert-mode",
        choices=("resident_packed_mxfp4", "packed_mxfp4", "dequant"),
        default="resident_packed_mxfp4",
    )
    parser.add_argument("--route-detail", choices=("full", "summary"), default="summary")
    parser.add_argument("--expert-empty-cache-interval", type=int, default=0)
    parser.add_argument("--skip-capture", action="store_true")
    parser.add_argument("--skip-control", action="store_true")
    parser.add_argument(
        "--record-id",
        action="append",
        default=None,
        help="run only the selected record id; repeat for multiple records",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build prompts and receipts without launching streamed GPU runs",
    )
    return parser.parse_args()


def default_output() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "artifacts" / "gpt_oss_20b" / f"h6_exact_value_{stamp}.json"


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def record_needle(item: dict[str, Any]) -> str:
    return (
        f"\nEXACT VALUE FACT {item['letter']}. The exact {item['label']} is "
        f"{item['answer']}.\nEND EXACT VALUE FACT {item['letter']}.\n"
    )


def forced_exact_prompt(item: dict[str, Any]) -> str:
    user = (
        f"Turn 50. Previous context is cleared. In EXACT VALUE FACT {item['letter']}, "
        f"what is the exact {item['label']}? Answer with the exact stored "
        f"{item['kind']}."
    )
    return f"{SYSTEM_PREFIX}{user}{ASSISTANT_FINAL}"


def exact_greedy_cmd(
    *,
    args: argparse.Namespace,
    model_dir: Path,
    prompt: str,
    output: Path,
    graft_dir: Path | None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "gpt_oss20b_stream_greedy_smoke.py"),
        "--model-dir",
        str(model_dir),
        "--prompt",
        prompt,
        "--steps",
        str(args.steps),
        "--max-tokens",
        str(args.max_tokens),
        "--top-k",
        str(args.top_k),
        "--attention-mode",
        args.attention_mode,
        "--apa-layer-scope",
        args.apa_layer_scope,
        "--refine-percentile",
        str(args.refine_percentile),
        "--bulk-bits",
        str(args.bulk_bits),
        "--expert-mode",
        args.expert_mode,
        "--route-detail",
        args.route_detail,
        "--expert-empty-cache-interval",
        str(args.expert_empty_cache_interval),
        "--output",
        str(output),
    ]
    if graft_dir is not None:
        cmd.extend(["--mount-graft-dir", str(graft_dir)])
    return cmd


def validate_items(tokenizer) -> list[dict[str, Any]]:
    items = []
    answers = set()
    for raw in DEFAULT_RECORDS:
        answer = str(raw["answer"])
        if answer in answers:
            raise ValueError(f"duplicate answer {answer!r}")
        answers.add(answer)
        ids = token_ids(tokenizer, answer)
        if not ids:
            raise ValueError(f"answer {answer!r} tokenized to zero tokens")
        item = {**raw, "answer_token_ids": [int(x) for x in ids]}
        item["needle"] = record_needle(item)
        items.append(item)
    return items


def select_run_items(
    items: list[dict[str, Any]],
    record_ids: list[str] | None,
) -> list[dict[str, Any]]:
    if not record_ids:
        return items
    by_id = {str(item["id"]): item for item in items}
    missing = [record_id for record_id in record_ids if record_id not in by_id]
    if missing:
        valid = ", ".join(sorted(by_id))
        raise ValueError(f"unknown --record-id {missing!r}; valid ids: {valid}")
    seen = set()
    selected = []
    for record_id in record_ids:
        if record_id in seen:
            continue
        seen.add(record_id)
        selected.append(by_id[record_id])
    return selected


def build_capture_ids(
    *,
    tokenizer,
    corpus_text: str,
    items: list[dict[str, Any]],
    target_tokens: int,
) -> tuple[str, list[int], dict[str, Any]]:
    if target_tokens < 768:
        raise ValueError("--target-tokens must be at least 768 for exact-value gate")
    corpus_ids = token_ids(tokenizer, corpus_text)
    if not corpus_ids:
        raise ValueError("corpus tokenized to zero tokens")
    repeated = (corpus_ids * ((target_tokens // len(corpus_ids)) + 4))[: target_tokens * 2]
    needle_ids = [token_ids(tokenizer, str(item["needle"])) for item in items]
    total_needle = sum(len(ids) for ids in needle_ids)
    if total_needle + len(items) * 80 >= target_tokens:
        raise ValueError("target token count too small for exact-value needles")

    available_fill = target_tokens - total_needle
    slot_fill = available_fill // (len(items) + 1)
    ids: list[int] = []
    cursor = 0
    placements = []

    def take_fill(count: int) -> None:
        nonlocal cursor
        ids.extend(repeated[cursor : cursor + count])
        cursor += count

    for idx, item in enumerate(items):
        take_fill(slot_fill)
        offset = len(ids)
        ids.extend(needle_ids[idx])
        placements.append(
            {
                "id": item["id"],
                "kind": item["kind"],
                "label": item["label"],
                "letter": item["letter"],
                "answer": item["answer"],
                "answer_token_ids": item["answer_token_ids"],
                "needle_offset_tokens": int(offset),
                "needle_tokens": len(needle_ids[idx]),
                "needle_sha256": sha256_bytes(str(item["needle"]).encode("utf-8")),
            }
        )
    take_fill(target_tokens - len(ids))
    ids = ids[:target_tokens]
    text = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
    return text, ids, {
        "target_tokens": int(target_tokens),
        "actual_tokens": len(ids),
        "input_mode": "direct_token_ids",
        "placements": placements,
        "total_needle_tokens": int(total_needle),
    }


def placements_by_id(capture_info: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in capture_info.get("placements") or []}


def summarize_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = []
    for row in rows:
        control = (row.get("control") or {}).get("summary") or {}
        mount = (row.get("mount") or {}).get("summary") or {}
        control_top = (control.get("top_token") or {}).get("text")
        mount_top = (mount.get("top_token") or {}).get("text")
        exact_hit = bool(mount.get("hit")) and control_top != row["answer"]
        generated_hit = bool(mount.get("generated_contains_value")) and not bool(
            control.get("generated_contains_value")
        )
        summary.append(
            {
                "id": row["id"],
                "kind": row["kind"],
                "answer": row["answer"],
                "control_top": control_top,
                "mount_top": mount_top,
                "mount_answer_rank": mount.get("answer_rank"),
                "mount_answer_logit": mount.get("answer_logit"),
                "control_generated_text": control.get("generated_text"),
                "mount_generated_text": mount.get("generated_text"),
                "control_generated_contains_value": control.get(
                    "generated_contains_value"
                ),
                "mount_generated_contains_value": mount.get(
                    "generated_contains_value"
                ),
                "hit": exact_hit,
                "generated_hit": generated_hit,
            }
        )
    exact_hits = sum(1 for row in summary if row["hit"])
    generated_hits = sum(1 for row in summary if row["generated_hit"])
    counts = {
        "hit_count": int(exact_hits),
        "generated_hit_count": int(generated_hits),
        "item_count": len(summary),
        "classification": "pass" if exact_hits == len(summary) else "fail",
        "generated_classification": (
            "pass" if generated_hits == len(summary) else "fail"
        ),
    }
    return summary, counts


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    out = args.output.expanduser().resolve() if args.output else default_output()
    run_dir = out.parent / out.stem
    graft_dir = (
        args.graft_dir.expanduser().resolve()
        if args.graft_dir is not None
        else run_dir / "graft"
    )
    capture_prompt_file = run_dir / "capture_prompt.txt"
    capture_ids_file = run_dir / "capture_ids.json"
    capture_artifact = run_dir / "capture_forward.json"

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    items = validate_items(tokenizer)
    run_items = select_run_items(items, args.record_id)
    corpus_text, corpus_info = read_corpus(args.corpus_dir)
    capture_text, capture_ids, capture_info = build_capture_ids(
        tokenizer=tokenizer,
        corpus_text=corpus_text,
        items=items,
        target_tokens=args.target_tokens,
    )
    placements = placements_by_id(capture_info)

    capture_prompt_file.parent.mkdir(parents=True, exist_ok=True)
    capture_prompt_file.write_text(capture_text, encoding="utf-8")
    capture_ids_file.write_text(json.dumps(capture_ids), encoding="utf-8")

    payload: dict[str, Any] = {
        "schema": "gpt_oss_20b_exact_value_graft_gate_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(model_dir),
        "target_tokens": int(args.target_tokens),
        "graft_dir": str(graft_dir),
        "capture_prompt_file": str(capture_prompt_file),
        "capture_ids_file": str(capture_ids_file),
        "capture_artifact": str(capture_artifact),
        "corpus": corpus_info,
        "capture_prompt": capture_info,
        "items": items,
        "run_item_ids": [str(item["id"]) for item in run_items],
        "steps": int(args.steps),
        "max_tokens": int(args.max_tokens),
        "top_k": int(args.top_k),
        "attention_mode": args.attention_mode,
        "apa_layer_scope": args.apa_layer_scope,
        "refine_percentile": float(args.refine_percentile),
        "bulk_bits": int(args.bulk_bits),
        "expert_mode": args.expert_mode,
        "route_detail": args.route_detail,
        "expert_empty_cache_interval": int(args.expert_empty_cache_interval),
        "script_args": jsonable_args(args),
        "status": "dry_run" if args.dry_run else "running",
        "runs": {"capture": None, "items": []},
    }
    write_json(out, payload)

    if args.dry_run:
        print(f"artifact={out}", flush=True)
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "target_tokens": payload["target_tokens"],
                    "items": [
                        {
                            "id": item["id"],
                            "kind": item["kind"],
                            "answer": item["answer"],
                            "answer_token_ids": item["answer_token_ids"],
                        }
                        for item in items
                    ],
                }
            ),
            flush=True,
        )
        return 0

    if not args.skip_capture:
        payload["runs"]["capture"] = run_child(
            stream_capture_cmd(
                args=args,
                model_dir=model_dir,
                prompt_file=capture_prompt_file,
                input_ids_file=capture_ids_file,
                output=capture_artifact,
                graft_dir=graft_dir,
            )
        )
        write_json(out, payload)
        if payload["runs"]["capture"]["returncode"] != 0:
            payload["status"] = "capture_error"
            write_json(out, payload)
            print(f"artifact={out}", flush=True)
            return int(payload["runs"]["capture"]["returncode"]) or 1

    for item in run_items:
        prompt = forced_exact_prompt(item)
        item_dir = run_dir / str(item["id"])
        control_path = item_dir / "control.json"
        mount_path = item_dir / "mount.json"
        row: dict[str, Any] = {
            "id": item["id"],
            "kind": item["kind"],
            "letter": item["letter"],
            "label": item["label"],
            "answer": item["answer"],
            "answer_token_ids": item["answer_token_ids"],
            "placement": placements.get(str(item["id"])),
            "prompt": prompt,
            "control": None,
            "mount": None,
        }
        payload["runs"]["items"].append(row)
        if not args.skip_control:
            control_run = run_child(
                exact_greedy_cmd(
                    args=args,
                    model_dir=model_dir,
                    prompt=prompt,
                    output=control_path,
                    graft_dir=None,
                )
            )
            row["control"] = {
                "run": control_run,
                "summary": summarize_greedy(control_path, str(item["answer"]))
                if control_run["returncode"] == 0
                else None,
            }
            write_json(out, payload)
            if control_run["returncode"] != 0:
                payload["status"] = "control_error"
                payload["failed_item"] = item["id"]
                write_json(out, payload)
                print(f"artifact={out}", flush=True)
                return int(control_run["returncode"]) or 1

        mount_run = run_child(
            exact_greedy_cmd(
                args=args,
                model_dir=model_dir,
                prompt=prompt,
                output=mount_path,
                graft_dir=graft_dir,
            )
        )
        row["mount"] = {
            "run": mount_run,
            "summary": summarize_greedy(mount_path, str(item["answer"]))
            if mount_run["returncode"] == 0
            else None,
        }
        write_json(out, payload)
        if mount_run["returncode"] != 0:
            payload["status"] = "mount_error"
            payload["failed_item"] = item["id"]
            write_json(out, payload)
            print(f"artifact={out}", flush=True)
            return int(mount_run["returncode"]) or 1

    summary, counts = summarize_rows(payload["runs"]["items"])
    payload.update({"status": "ok", "summary": summary, **counts})
    write_json(out, payload)
    print(f"artifact={out}", flush=True)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "classification": payload["classification"],
                "generated_classification": payload["generated_classification"],
                "hits": payload["hit_count"],
                "generated_hits": payload["generated_hit_count"],
                "item_count": payload["item_count"],
                "summary": summary,
            }
        ),
        flush=True,
    )
    return 0 if payload["generated_classification"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
