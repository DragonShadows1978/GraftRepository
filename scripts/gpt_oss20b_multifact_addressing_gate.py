#!/usr/bin/env python3
"""Fact-local addressing gate for GPT-OSS multi-fact graft recall.

This helper reuses an existing multi-fact graft receipt, then probes whether a
more explicit addressing policy can extract each planted fact. It is meant to
turn prompt-recoverable misses into a repeatable GRM policy test instead of an
ad hoc diagnostic.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from scripts.gpt_oss20b_grm_output_eval import canonical_value, find_value_rank
except ModuleNotFoundError:  # direct execution from scripts/
    from gpt_oss20b_grm_output_eval import canonical_value, find_value_rank


REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = (
    "/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/"
    "snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
)

SYSTEM_PREFIX = (
    "<|start|>system<|message|>You are ChatGPT. Reasoning: low. "
    "Valid channel: final.<|end|><|start|>user<|message|>"
)
ASSISTANT_FINAL = "<|end|><|start|>assistant<|channel|>final<|message|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fact-local addressing probes against a multi-fact GPT-OSS graft."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="multi-fact gate JSON artifact to reuse",
    )
    parser.add_argument("--model-dir", default=SNAPSHOT)
    parser.add_argument(
        "--graft-dir",
        type=Path,
        default=None,
        help="override graft dir; defaults to source artifact graft_dir",
    )
    parser.add_argument(
        "--variants",
        default="fact_local",
        help=(
            "comma-separated variants: fact_local, metadata_card, "
            "conversational_slot, conversational_continuity"
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=160)
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def default_output(source: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        REPO_ROOT
        / "artifacts"
        / "gpt_oss_20b"
        / f"{source.stem}_addressing_{stamp}.json"
    )


def forced_final_prompt(user_prompt: str) -> str:
    return f"{SYSTEM_PREFIX}{user_prompt}{ASSISTANT_FINAL}"


def parse_variants(raw: str) -> list[str]:
    known = {
        "fact_local",
        "metadata_card",
        "conversational_slot",
        "conversational_continuity",
    }
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in variants if item not in known]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; known variants: {sorted(known)}")
    if not variants:
        raise SystemExit("at least one variant is required")
    return variants


def load_source(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "gpt_oss_20b_multifact_graft_gate_v1":
        raise SystemExit(f"{path} is not a multi-fact graft gate artifact")
    facts = payload.get("facts") or []
    if not facts:
        raise SystemExit(f"{path} does not contain facts")
    graft_dir = payload.get("graft_dir")
    if not graft_dir:
        raise SystemExit(f"{path} does not contain graft_dir")
    return payload


def fact_letter(fact: dict[str, Any], placement: dict[str, Any] | None) -> str:
    needle = str(fact.get("needle") or "")
    match = re.search(r"CLASSIFIED FACT\s+([A-Z])", needle)
    if match:
        return match.group(1)
    if placement:
        needle_sha = str(placement.get("needle_sha256") or "")
        if needle_sha:
            # Fallback is deterministic and visible; most current artifacts
            # have the literal letter in the source fact text.
            return str(fact.get("id", "?"))[:1].upper()
    return str(fact.get("id", "?"))[:1].upper()


def placements_by_id(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in (source.get("capture_prompt") or {}).get("facts") or []:
        out[str(item.get("id"))] = item
    return out


def user_prompt_for(variant: str, fact: dict[str, Any], placement: dict[str, Any] | None) -> str:
    letter = fact_letter(fact, placement)
    label = str(fact["label"])
    fact_id = str(fact["id"])
    if variant == "fact_local":
        return (
            f"Turn 50. Previous context is cleared. In CLASSIFIED FACT {letter}, "
            f"what is the {label}? Return only the stored value. No sentence. "
            f"No punctuation."
        )
    if variant == "metadata_card":
        offset = "unknown" if placement is None else placement.get("needle_offset_tokens", "unknown")
        return (
            "Turn 50. Previous context is cleared. Use the mounted memory record "
            f"with fact_id={fact_id}, fact_marker=CLASSIFIED FACT {letter}, "
            f"label={label}, token_offset={offset}. Return exactly the stored "
            "value for that record. No sentence. No punctuation."
        )
    if variant == "conversational_slot":
        return (
            "I'm checking the mounted memory after clearing the chat. "
            f"In CLASSIFIED FACT {letter}, what was the {label}?"
        )
    if variant == "conversational_continuity":
        return (
            "We cleared the prior conversation, but GRM mounted memory is "
            f"available. What value did CLASSIFIED FACT {letter} store for "
            f"the {label}?"
        )
    raise AssertionError(variant)


def run_child(cmd: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-4000:],
        "wall_seconds": time.perf_counter() - started,
    }


def greedy_cmd(
    *,
    args: argparse.Namespace,
    prompt: str,
    output: Path,
    graft_dir: Path | None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "gpt_oss20b_stream_greedy_smoke.py"),
        "--model-dir",
        str(args.model_dir),
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


def summarize_greedy(path: Path, answer: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    step = (payload.get("steps") or [{}])[0]
    child_path = step.get("child_artifact")
    top_tokens = []
    if child_path:
        child_payload = json.loads(Path(child_path).read_text(encoding="utf-8"))
        top_tokens = child_payload.get("top_tokens") or []
    rank = None
    logit = None
    for idx, row in enumerate(top_tokens):
        if row.get("text") == answer:
            rank = idx
            logit = row.get("logit")
            break
    normalized = find_value_rank(top_tokens, answer)
    top = step.get("top_token") or {}
    return {
        "artifact": str(path),
        "status": payload.get("status"),
        "top_token": top,
        "answer": answer,
        "normalized_answer": canonical_value(answer),
        "normalized_top_token": canonical_value(top.get("text")),
        "answer_rank": rank,
        "answer_logit": logit,
        "normalized_answer_rank": normalized["rank"],
        "normalized_answer_logit": normalized["logit"],
        "normalized_answer_text": normalized["text"],
        "hit": top.get("text") == answer,
        "normalized_hit": canonical_value(top.get("text")) == canonical_value(answer),
        "top_tokens": top_tokens,
        "wall_seconds": payload.get("wall_seconds"),
        "child_artifact": child_path,
    }


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def main() -> int:
    args = parse_args()
    source_path = args.source.expanduser().resolve()
    source = load_source(source_path)
    variants = parse_variants(args.variants)
    graft_dir = (
        args.graft_dir.expanduser().resolve()
        if args.graft_dir is not None
        else Path(str(source["graft_dir"])).expanduser().resolve()
    )
    out = args.output.expanduser().resolve() if args.output else default_output(source_path)
    run_dir = out.parent / out.stem
    placement_lookup = placements_by_id(source)
    facts = source.get("facts") or []

    payload: dict[str, Any] = {
        "schema": "gpt_oss_20b_multifact_addressing_gate_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(source_path),
        "source_classification": source.get("classification"),
        "source_target_tokens": source.get("target_tokens"),
        "model_dir": str(args.model_dir),
        "graft_dir": str(graft_dir),
        "variants": variants,
        "attention_mode": args.attention_mode,
        "apa_layer_scope": args.apa_layer_scope,
        "refine_percentile": float(args.refine_percentile),
        "bulk_bits": int(args.bulk_bits),
        "expert_mode": args.expert_mode,
        "route_detail": args.route_detail,
        "expert_empty_cache_interval": int(args.expert_empty_cache_interval),
        "script_args": jsonable_args(args),
        "status": "dry_run" if args.dry_run else "running",
        "runs": [],
    }
    write_json(out, payload)

    planned = []
    for variant in variants:
        for fact in facts:
            placement = placement_lookup.get(str(fact["id"]))
            user_prompt = user_prompt_for(variant, fact, placement)
            planned.append(
                {
                    "variant": variant,
                    "id": fact["id"],
                    "label": fact["label"],
                    "answer": fact["answer"],
                    "placement": placement,
                    "user_prompt": user_prompt,
                    "prompt": forced_final_prompt(user_prompt),
                }
            )
    payload["planned_runs"] = planned
    write_json(out, payload)
    if args.dry_run:
        print(f"artifact={out}", flush=True)
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "variants": variants,
                    "facts": [
                        {
                            "variant": row["variant"],
                            "id": row["id"],
                            "answer": row["answer"],
                            "user_prompt": row["user_prompt"],
                        }
                        for row in planned
                    ],
                }
            ),
            flush=True,
        )
        return 0

    started = time.perf_counter()
    for planned_row in planned:
        row: dict[str, Any] = {
            **planned_row,
            "control": None,
            "mount": None,
        }
        payload["runs"].append(row)
        fact_dir = run_dir / str(row["variant"]) / str(row["id"])
        if not args.skip_control:
            control_path = fact_dir / "control.json"
            control_run = run_child(
                greedy_cmd(
                    args=args,
                    prompt=str(row["prompt"]),
                    output=control_path,
                    graft_dir=None,
                )
            )
            row["control"] = {
                "run": control_run,
                "summary": summarize_greedy(control_path, str(row["answer"]))
                if control_run["returncode"] == 0
                else None,
            }
            write_json(out, payload)
            if control_run["returncode"] != 0:
                payload.update(
                    {
                        "status": "control_error",
                        "failed_variant": row["variant"],
                        "failed_fact": row["id"],
                        "wall_seconds": time.perf_counter() - started,
                    }
                )
                write_json(out, payload)
                print(f"artifact={out}", flush=True)
                return int(control_run["returncode"]) or 1

        mount_path = fact_dir / "mount.json"
        mount_run = run_child(
            greedy_cmd(
                args=args,
                prompt=str(row["prompt"]),
                output=mount_path,
                graft_dir=graft_dir,
            )
        )
        row["mount"] = {
            "run": mount_run,
            "summary": summarize_greedy(mount_path, str(row["answer"]))
            if mount_run["returncode"] == 0
            else None,
        }
        write_json(out, payload)
        if mount_run["returncode"] != 0:
            payload.update(
                {
                    "status": "mount_error",
                    "failed_variant": row["variant"],
                    "failed_fact": row["id"],
                    "wall_seconds": time.perf_counter() - started,
                }
            )
            write_json(out, payload)
            print(f"artifact={out}", flush=True)
            return int(mount_run["returncode"]) or 1

    summary = []
    for row in payload["runs"]:
        control = (row.get("control") or {}).get("summary") or {}
        mount = (row.get("mount") or {}).get("summary") or {}
        control_top = (control.get("top_token") or {}).get("text")
        mount_top = (mount.get("top_token") or {}).get("text")
        hit = bool(mount.get("hit")) and control_top != row["answer"]
        summary.append(
            {
                "variant": row["variant"],
                "id": row["id"],
                "answer": row["answer"],
                "control_top": control_top,
                "mount_top": mount_top,
                "mount_answer_rank": mount.get("answer_rank"),
                "mount_answer_logit": mount.get("answer_logit"),
                "hit": hit,
            }
        )
    hits = sum(1 for item in summary if item["hit"])
    payload.update(
        {
            "status": "ok",
            "summary": summary,
            "hit_count": int(hits),
            "probe_count": len(summary),
            "classification": "pass" if hits == len(summary) else "fail",
            "wall_seconds": time.perf_counter() - started,
        }
    )
    write_json(out, payload)
    print(f"artifact={out}", flush=True)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "classification": payload["classification"],
                "hits": hits,
                "probe_count": len(summary),
                "summary": summary,
            }
        ),
        flush=True,
    )
    return 0 if payload["classification"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
