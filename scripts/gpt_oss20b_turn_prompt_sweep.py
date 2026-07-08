#!/usr/bin/env python3
"""Prompt-sensitivity sweep for GPT-OSS cold-graft greedy recall.

This H6 helper runs short forced-final prompts through the existing streamed
greedy smoke, once with no graft and once with a mounted graft. It is meant to
separate graft-recall signal from live prompt/protocol priors.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


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

VARIANT_USER_PROMPTS = {
    "plain": "What is the GPT-OSS vault keyword? Reply with only the keyword.",
    "turn50_original": (
        "Conversation turn 50. You do not have previous chat context. "
        "What is the GPT-OSS vault keyword? Reply with only the keyword."
    ),
    "turn50_compact": (
        "Turn 50. Previous context is cleared. "
        "What is the GPT-OSS vault keyword? Reply with only the keyword."
    ),
    "turn50_strict": (
        "Turn 50. Previous context is cleared. Return exactly the stored "
        "GPT-OSS vault keyword. No sentence. No punctuation."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GPT-OSS forced-final prompt variants with/without a graft."
    )
    parser.add_argument("--model-dir", default=SNAPSHOT)
    parser.add_argument("--graft-dir", type=Path, required=True)
    parser.add_argument(
        "--variants",
        default="plain,turn50_original,turn50_compact,turn50_strict",
        help="comma-separated variant names",
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
    parser.add_argument("--expert-mode", default="resident_packed_mxfp4")
    parser.add_argument("--route-detail", choices=("full", "summary"), default="summary")
    parser.add_argument("--expert-empty-cache-interval", type=int, default=0)
    parser.add_argument("--skip-control", action="store_true")
    return parser.parse_args()


def default_output() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "artifacts" / "gpt_oss_20b" / f"h6_prompt_sweep_{stamp}.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def forced_final_prompt(user_prompt: str) -> str:
    return f"{SYSTEM_PREFIX}{user_prompt}{ASSISTANT_FINAL}"


def parse_variants(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in names if name not in VARIANT_USER_PROMPTS]
    if unknown:
        valid = ", ".join(sorted(VARIANT_USER_PROMPTS))
        raise SystemExit(f"unknown variants {unknown}; valid variants: {valid}")
    return names


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


def top_summary(artifact: Path, target: str = "BLUE") -> dict[str, Any]:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    step_payloads = payload.get("steps") or []
    top_token = (step_payloads[0] or {}).get("top_token") if step_payloads else None
    child_path = None
    top_tokens = []
    if step_payloads:
        child_path = step_payloads[0].get("child_artifact")
    if child_path:
        child_payload = json.loads(Path(child_path).read_text(encoding="utf-8"))
        top_tokens = child_payload.get("top_tokens") or []
    target_rank = None
    target_logit = None
    for idx, row in enumerate(top_tokens):
        if row.get("text") == target:
            target_rank = idx
            target_logit = row.get("logit")
            break
    return {
        "artifact": str(artifact),
        "status": payload.get("status"),
        "top_token": top_token,
        "target_text": target,
        "target_rank": target_rank,
        "target_logit": target_logit,
        "top_tokens": top_tokens,
        "wall_seconds": payload.get("wall_seconds"),
    }


def greedy_cmd(
    *,
    args: argparse.Namespace,
    prompt: str,
    output: Path,
    mount: bool,
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
    if mount:
        cmd.extend(["--mount-graft-dir", str(args.graft_dir)])
    return cmd


def main() -> int:
    args = parse_args()
    variants = parse_variants(args.variants)
    out = args.output or default_output()
    run_dir = out.parent / out.stem
    payload: dict[str, Any] = {
        "schema": "gpt_oss_20b_turn_prompt_sweep_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(args.model_dir),
        "graft_dir": str(args.graft_dir),
        "variants": variants,
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
        "status": "running",
        "runs": [],
    }
    write_json(out, payload)

    started = time.perf_counter()
    for variant in variants:
        user_prompt = VARIANT_USER_PROMPTS[variant]
        prompt = forced_final_prompt(user_prompt)
        row: dict[str, Any] = {
            "variant": variant,
            "user_prompt": user_prompt,
            "prompt": prompt,
            "control": None,
            "mount": None,
        }
        for mode in ("control", "mount"):
            if mode == "control" and args.skip_control:
                continue
            artifact = run_dir / f"{variant}_{mode}.json"
            cmd = greedy_cmd(args=args, prompt=prompt, output=artifact, mount=(mode == "mount"))
            receipt = run_child(cmd)
            row[mode] = {"run": receipt, "summary": None}
            if receipt["returncode"] == 0:
                row[mode]["summary"] = top_summary(artifact)
            write_json(out, payload)
            if receipt["returncode"] != 0:
                payload.update(
                    {
                        "status": "error",
                        "failed_variant": variant,
                        "failed_mode": mode,
                        "wall_seconds": time.perf_counter() - started,
                    }
                )
                write_json(out, payload)
                print(f"artifact={out}", flush=True)
                return int(receipt["returncode"]) or 1
        payload["runs"].append(row)
        write_json(out, payload)

    summary = []
    for row in payload["runs"]:
        mount = row.get("mount") or {}
        ms = mount.get("summary") or {}
        top = ms.get("top_token") or {}
        summary.append(
            {
                "variant": row["variant"],
                "mount_top_text": top.get("text"),
                "mount_top_logit": top.get("logit"),
                "mount_blue_rank": ms.get("target_rank"),
                "mount_blue_logit": ms.get("target_logit"),
                "mount_hit": top.get("text") == "BLUE",
            }
        )
    payload.update(
        {
            "status": "ok",
            "summary": summary,
            "wall_seconds": time.perf_counter() - started,
        }
    )
    write_json(out, payload)
    print(f"artifact={out}", flush=True)
    print(json.dumps({"status": "ok", "summary": summary}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
