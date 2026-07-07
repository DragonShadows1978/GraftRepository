#!/usr/bin/env python3
"""GPT-OSS repeated cold-graft recall drift gate.

This Phase 5 helper repeatedly clears live context by launching independent
greedy probes, remounts the same graft, and asks the same retained instruction
again. It tests whether a cold-graft memory remains stable across repeated
turns rather than only working once.
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

from scripts.gpt_oss20b_exact_value_graft_gate import (
    exact_greedy_cmd,
    records_for_mode,
    select_run_items,
    summarize_rows,
    validate_items,
)
from scripts.gpt_oss20b_multifact_graft_gate import (
    ASSISTANT_FINAL,
    REPO_ROOT,
    SNAPSHOT,
    SYSTEM_PREFIX,
    run_child,
    summarize_greedy,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT-OSS repetition/drift gate.")
    parser.add_argument("--model-dir", default=SNAPSHOT)
    parser.add_argument("--record-mode", default="instruction_retention")
    parser.add_argument("--record-id", action="append", default=None)
    parser.add_argument("--graft-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, default=8)
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
    parser.add_argument("--skip-control", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_output() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "artifacts" / "gpt_oss_20b" / f"h6_repetition_drift_{stamp}.json"


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def repeat_user_prompt(item: dict[str, Any], *, repeat_index: int, repeats: int) -> str:
    base = item.get("user_prompt")
    if not base:
        base = (
            f"Turn 50. Previous context is cleared. In EXACT VALUE FACT {item['letter']}, "
            f"what is the exact {item['label']}? Answer with the exact stored "
            f"{item['kind']}."
        )
    return (
        f"Repetition probe {repeat_index + 1} of {repeats}. "
        f"{base} Do not infer from this prompt; use only mounted memory if any."
    )


def repeat_prompt(item: dict[str, Any], *, repeat_index: int, repeats: int) -> str:
    return f"{SYSTEM_PREFIX}{repeat_user_prompt(item, repeat_index=repeat_index, repeats=repeats)}{ASSISTANT_FINAL}"


def summarize_drift(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary, counts = summarize_rows(rows)
    repeats = {}
    for row, item in zip(rows, summary):
        item["repeat_index"] = row.get("repeat_index")
        item["record_id"] = row.get("id")
        repeats.setdefault(str(row.get("id")), []).append(item)

    stable_records = 0
    record_summaries = []
    for record_id, items in sorted(repeats.items()):
        generated_hits = sum(1 for item in items if item["generated_hit"])
        record_summaries.append(
            {
                "id": record_id,
                "repeat_count": len(items),
                "generated_hit_count": generated_hits,
                "generated_classification": (
                    "pass" if generated_hits == len(items) else "fail"
                ),
                "mount_generated_texts": [
                    item.get("mount_generated_text") for item in items
                ],
            }
        )
        if generated_hits == len(items):
            stable_records += 1
    counts["record_count"] = len(record_summaries)
    counts["stable_record_count"] = stable_records
    counts["drift_classification"] = (
        "pass"
        if record_summaries and stable_records == len(record_summaries)
        else "fail"
    )
    counts["record_summaries"] = record_summaries
    return summary, counts


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    model_dir = Path(args.model_dir).expanduser().resolve()
    graft_dir = args.graft_dir.expanduser().resolve()
    out = args.output.expanduser().resolve() if args.output else default_output()
    run_dir = out.parent / out.stem

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    record_ids = args.record_id or ["retention_instruction_a"]
    items = select_run_items(
        validate_items(tokenizer, records_for_mode(args.record_mode)),
        record_ids,
    )

    payload: dict[str, Any] = {
        "schema": "gpt_oss_20b_repetition_drift_gate_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(model_dir),
        "record_mode": args.record_mode,
        "record_ids": [str(item["id"]) for item in items],
        "graft_dir": str(graft_dir),
        "repeats": int(args.repeats),
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
        "runs": {"items": []},
    }
    write_json(out, payload)

    planned_rows = []
    for item in items:
        for repeat_index in range(args.repeats):
            planned_rows.append(
                {
                    "id": item["id"],
                    "kind": item["kind"],
                    "label": item["label"],
                    "answer": item["answer"],
                    "repeat_index": repeat_index,
                    "user_prompt": repeat_user_prompt(
                        item, repeat_index=repeat_index, repeats=args.repeats
                    ),
                    "prompt": repeat_prompt(
                        item, repeat_index=repeat_index, repeats=args.repeats
                    ),
                }
            )
    if args.dry_run:
        payload["planned_rows"] = planned_rows
        write_json(out, payload)
        print(f"artifact={out}", flush=True)
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "record_ids": payload["record_ids"],
                    "repeats": args.repeats,
                    "rows": [
                        {
                            "id": row["id"],
                            "repeat_index": row["repeat_index"],
                            "answer": row["answer"],
                            "user_prompt": row["user_prompt"],
                        }
                        for row in planned_rows
                    ],
                }
            ),
            flush=True,
        )
        return 0

    for row in planned_rows:
        item_dir = run_dir / str(row["id"]) / f"repeat_{int(row['repeat_index']):02d}"
        control_path = item_dir / "control.json"
        mount_path = item_dir / "mount.json"
        run_row = {
            **row,
            "control": None,
            "mount": None,
        }
        payload["runs"]["items"].append(run_row)
        if not args.skip_control:
            control_run = run_child(
                exact_greedy_cmd(
                    args=args,
                    model_dir=model_dir,
                    prompt=str(row["prompt"]),
                    output=control_path,
                    graft_dir=None,
                )
            )
            run_row["control"] = {
                "run": control_run,
                "summary": summarize_greedy(control_path, str(row["answer"]))
                if control_run["returncode"] == 0
                else None,
            }
            write_json(out, payload)
            if control_run["returncode"] != 0:
                payload["status"] = "control_error"
                payload["failed_row"] = {
                    "id": row["id"],
                    "repeat_index": row["repeat_index"],
                }
                write_json(out, payload)
                print(f"artifact={out}", flush=True)
                return int(control_run["returncode"]) or 1

        mount_run = run_child(
            exact_greedy_cmd(
                args=args,
                model_dir=model_dir,
                prompt=str(row["prompt"]),
                output=mount_path,
                graft_dir=graft_dir,
            )
        )
        run_row["mount"] = {
            "run": mount_run,
            "summary": summarize_greedy(mount_path, str(row["answer"]))
            if mount_run["returncode"] == 0
            else None,
        }
        write_json(out, payload)
        if mount_run["returncode"] != 0:
            payload["status"] = "mount_error"
            payload["failed_row"] = {
                "id": row["id"],
                "repeat_index": row["repeat_index"],
            }
            write_json(out, payload)
            print(f"artifact={out}", flush=True)
            return int(mount_run["returncode"]) or 1

    summary, counts = summarize_drift(payload["runs"]["items"])
    payload.update({"status": "ok", "summary": summary, **counts})
    write_json(out, payload)
    print(f"artifact={out}", flush=True)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "classification": payload["classification"],
                "generated_classification": payload["generated_classification"],
                "drift_classification": payload["drift_classification"],
                "hits": payload["hit_count"],
                "generated_hits": payload["generated_hit_count"],
                "item_count": payload["item_count"],
                "stable_records": payload["stable_record_count"],
                "record_count": payload["record_count"],
            }
        ),
        flush=True,
    )
    return 0 if payload["drift_classification"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
