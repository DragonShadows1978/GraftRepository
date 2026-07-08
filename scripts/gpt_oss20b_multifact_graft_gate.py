#!/usr/bin/env python3
"""Multi-fact GPT-OSS cold-graft recall gate.

This H6 helper plants several one-token facts into one real-token capture,
captures pre-RoPE K/V graft shards, then asks strict forced-final questions with
and without the mounted graft. It is deliberately stricter than the earlier
single-keyword H5/H6 gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

try:
    from scripts.gpt_oss20b_grm_output_eval import (
        canonical_value,
        contains_value_text,
        find_value_rank,
        generated_suffix_from_payload,
    )
except ModuleNotFoundError:  # direct execution from scripts/
    from gpt_oss20b_grm_output_eval import (
        canonical_value,
        contains_value_text,
        find_value_rank,
        generated_suffix_from_payload,
    )


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

DEFAULT_FACTS = [
    {
        "id": "vault_keyword",
        "label": "GPT-OSS vault keyword",
        "answer": "BLUE",
        "needle": (
            "\nCLASSIFIED FACT A. The GPT-OSS vault keyword is BLUE.\n"
            "END FACT A.\n"
        ),
    },
    {
        "id": "relay_marker",
        "label": "GPT-OSS relay marker",
        "answer": "EMBER",
        "needle": (
            "\nCLASSIFIED FACT B. The GPT-OSS relay marker is EMBER.\n"
            "END FACT B.\n"
        ),
    },
    {
        "id": "archive_color",
        "label": "GPT-OSS archive color",
        "answer": "GRAY",
        "needle": (
            "\nCLASSIFIED FACT C. The GPT-OSS archive color is GRAY.\n"
            "END FACT C.\n"
        ),
    },
    {
        "id": "tool_metal",
        "label": "GPT-OSS tool metal",
        "answer": "IRON",
        "needle": (
            "\nCLASSIFIED FACT D. The GPT-OSS tool metal is IRON.\n"
            "END FACT D.\n"
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT-OSS multi-fact graft gate.")
    parser.add_argument("--model-dir", default=SNAPSHOT)
    parser.add_argument("--corpus-dir", type=Path, default=REPO_ROOT / "docs")
    parser.add_argument("--target-tokens", type=int, default=4096)
    parser.add_argument("--graft-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-layers", type=int, default=None)
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
        "--dry-run",
        action="store_true",
        help="build prompts and receipts without launching streamed GPU runs",
    )
    return parser.parse_args()


def default_output() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "artifacts" / "gpt_oss_20b" / f"h6_multifact_{stamp}.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_corpus(root: Path) -> tuple[str, dict[str, Any]]:
    root = root.expanduser().resolve()
    files = sorted(root.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"no markdown corpus files found in {root}")
    parts: list[str] = []
    h = hashlib.sha256()
    for path in files:
        data = path.read_bytes()
        rel = str(path.relative_to(root))
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(data)
        h.update(b"\0")
        parts.append(data.decode("utf-8", errors="ignore"))
    return "\n\n".join(parts), {
        "kind": "directory",
        "path": str(root),
        "file_count": len(files),
        "sha256": h.hexdigest(),
    }


def token_ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def forced_final_prompt(label: str) -> str:
    user = (
        f"Turn 50. Previous context is cleared. Return exactly the stored "
        f"{label}. No sentence. No punctuation."
    )
    return f"{SYSTEM_PREFIX}{user}{ASSISTANT_FINAL}"


def validate_facts(tokenizer) -> list[dict[str, Any]]:
    facts = []
    seen_answers = set()
    for item in DEFAULT_FACTS:
        answer = str(item["answer"])
        answer_ids = token_ids(tokenizer, answer)
        if len(answer_ids) != 1:
            raise ValueError(f"answer {answer!r} must tokenize to one token, got {answer_ids}")
        if answer in seen_answers:
            raise ValueError(f"duplicate answer {answer!r}")
        seen_answers.add(answer)
        facts.append({**item, "answer_token_id": int(answer_ids[0])})
    return facts


def build_capture_ids(
    *,
    tokenizer,
    corpus_text: str,
    facts: list[dict[str, Any]],
    target_tokens: int,
) -> tuple[str, list[int], dict[str, Any]]:
    if target_tokens < 512:
        raise ValueError("--target-tokens must be at least 512 for the multi-fact gate")
    corpus_ids = token_ids(tokenizer, corpus_text)
    if not corpus_ids:
        raise ValueError("corpus tokenized to zero tokens")
    repeated = (corpus_ids * ((target_tokens // len(corpus_ids)) + 4))[: target_tokens * 2]
    needle_ids = [token_ids(tokenizer, str(fact["needle"])) for fact in facts]
    total_needle = sum(len(ids) for ids in needle_ids)
    if total_needle + len(facts) * 64 >= target_tokens:
        raise ValueError("target token count too small for all needles")

    available_fill = target_tokens - total_needle
    slot_fill = available_fill // (len(facts) + 1)
    ids: list[int] = []
    cursor = 0
    placements = []

    def take_fill(count: int) -> None:
        nonlocal cursor
        ids.extend(repeated[cursor : cursor + count])
        cursor += count

    for idx, fact in enumerate(facts):
        take_fill(slot_fill)
        offset = len(ids)
        ids.extend(needle_ids[idx])
        placements.append(
            {
                "id": fact["id"],
                "label": fact["label"],
                "answer": fact["answer"],
                "answer_token_id": fact["answer_token_id"],
                "needle_offset_tokens": int(offset),
                "needle_tokens": len(needle_ids[idx]),
                "needle_sha256": sha256_bytes(str(fact["needle"]).encode("utf-8")),
            }
        )
    take_fill(target_tokens - len(ids))
    ids = ids[:target_tokens]
    text = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
    return text, ids, {
        "target_tokens": int(target_tokens),
        "actual_tokens": len(ids),
        "input_mode": "direct_token_ids",
        "facts": placements,
        "total_needle_tokens": int(total_needle),
    }


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


def stream_capture_cmd(
    *,
    args: argparse.Namespace,
    model_dir: Path,
    prompt_file: Path,
    input_ids_file: Path,
    output: Path,
    graft_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "gpt_oss20b_stream_forward_smoke.py"),
        "--model-dir",
        str(model_dir),
        "--prompt-file",
        str(prompt_file),
        "--input-ids-file",
        str(input_ids_file),
        "--max-tokens",
        str(args.target_tokens),
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
        "--skip-lm-head",
        "--capture-graft-dir",
        str(graft_dir),
        "--output",
        str(output),
    ]
    if args.max_layers is not None:
        cmd.extend(["--max-layers", str(args.max_layers)])
    return cmd


def greedy_cmd(
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
        "1",
        "--max-tokens",
        "160",
        "--top-k",
        "10",
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
    generated_text = generated_suffix_from_payload(payload)
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
        "generated_text": generated_text,
        "generated_contains_value": contains_value_text(generated_text, answer),
        "top_tokens": top_tokens,
        "wall_seconds": payload.get("wall_seconds"),
        "child_artifact": child_path,
    }


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
    facts = validate_facts(tokenizer)
    corpus_text, corpus_info = read_corpus(args.corpus_dir)
    capture_text, capture_ids, capture_info = build_capture_ids(
        tokenizer=tokenizer,
        corpus_text=corpus_text,
        facts=facts,
        target_tokens=args.target_tokens,
    )
    placements_by_id = {
        str(item["id"]): item for item in capture_info.get("facts", [])
    }
    capture_prompt_file.parent.mkdir(parents=True, exist_ok=True)
    capture_prompt_file.write_text(capture_text, encoding="utf-8")
    capture_ids_file.write_text(json.dumps(capture_ids), encoding="utf-8")

    payload: dict[str, Any] = {
        "schema": "gpt_oss_20b_multifact_graft_gate_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(model_dir),
        "target_tokens": int(args.target_tokens),
        "graft_dir": str(graft_dir),
        "capture_prompt_file": str(capture_prompt_file),
        "capture_ids_file": str(capture_ids_file),
        "capture_artifact": str(capture_artifact),
        "corpus": corpus_info,
        "capture_prompt": capture_info,
        "facts": facts,
        "attention_mode": args.attention_mode,
        "apa_layer_scope": args.apa_layer_scope,
        "refine_percentile": float(args.refine_percentile),
        "bulk_bits": int(args.bulk_bits),
        "expert_mode": args.expert_mode,
        "route_detail": args.route_detail,
        "expert_empty_cache_interval": int(args.expert_empty_cache_interval),
        "script_args": jsonable_args(args),
        "status": "dry_run" if args.dry_run else "running",
        "runs": {"capture": None, "facts": []},
    }
    write_json(out, payload)

    if args.dry_run:
        print(f"artifact={out}", flush=True)
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "target_tokens": payload["target_tokens"],
                    "facts": [
                        {
                            "id": fact["id"],
                            "answer": fact["answer"],
                            "answer_token_id": fact["answer_token_id"],
                        }
                        for fact in facts
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

    for fact in facts:
        prompt = forced_final_prompt(str(fact["label"]))
        fact_dir = run_dir / str(fact["id"])
        control_path = fact_dir / "control.json"
        mount_path = fact_dir / "mount.json"
        row: dict[str, Any] = {
            "id": fact["id"],
            "label": fact["label"],
            "answer": fact["answer"],
            "placement": placements_by_id.get(str(fact["id"])),
            "prompt": prompt,
            "control": None,
            "mount": None,
        }
        payload["runs"]["facts"].append(row)
        if not args.skip_control:
            control_run = run_child(
                greedy_cmd(
                    args=args,
                    model_dir=model_dir,
                    prompt=prompt,
                    output=control_path,
                    graft_dir=None,
                )
            )
            row["control"] = {
                "run": control_run,
                "summary": summarize_greedy(control_path, str(fact["answer"]))
                if control_run["returncode"] == 0
                else None,
            }
            write_json(out, payload)
            if control_run["returncode"] != 0:
                payload["status"] = "control_error"
                payload["failed_fact"] = fact["id"]
                write_json(out, payload)
                print(f"artifact={out}", flush=True)
                return int(control_run["returncode"]) or 1

        mount_run = run_child(
            greedy_cmd(
                args=args,
                model_dir=model_dir,
                prompt=prompt,
                output=mount_path,
                graft_dir=graft_dir,
            )
        )
        row["mount"] = {
            "run": mount_run,
            "summary": summarize_greedy(mount_path, str(fact["answer"]))
            if mount_run["returncode"] == 0
            else None,
        }
        write_json(out, payload)
        if mount_run["returncode"] != 0:
            payload["status"] = "mount_error"
            payload["failed_fact"] = fact["id"]
            write_json(out, payload)
            print(f"artifact={out}", flush=True)
            return int(mount_run["returncode"]) or 1

    summary = []
    for row in payload["runs"]["facts"]:
        control = (row.get("control") or {}).get("summary") or {}
        mount = (row.get("mount") or {}).get("summary") or {}
        summary.append(
            {
                "id": row["id"],
                "answer": row["answer"],
                "control_top": (control.get("top_token") or {}).get("text"),
                "mount_top": (mount.get("top_token") or {}).get("text"),
                "mount_answer_rank": mount.get("answer_rank"),
                "hit": bool(mount.get("hit"))
                and (control.get("top_token") or {}).get("text") != row["answer"],
            }
        )
    hits = sum(1 for row in summary if row["hit"])
    payload["summary"] = summary
    payload["hit_count"] = int(hits)
    payload["fact_count"] = len(summary)
    payload["classification"] = "pass" if hits == len(summary) else "fail"
    payload["status"] = "ok"
    write_json(out, payload)
    print(f"artifact={out}", flush=True)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "classification": payload["classification"],
                "hits": hits,
                "fact_count": len(summary),
                "summary": summary,
            }
        ),
        flush=True,
    )
    return 0 if payload["classification"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
