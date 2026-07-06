#!/usr/bin/env python3
"""Short greedy decode smoke using the streamed GPT-OSS forward harness.

This intentionally reruns the full streamed forward for each generated token.
It is not a fast decode path and does not use KV cache. Its job is to prove
that the current streamed TensorCUDA path can be driven in a greedy loop.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short streamed greedy smoke.")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--expert-mode", default="resident_packed_mxfp4")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def output_path(path: Path | None) -> Path:
    if path is not None:
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("artifacts") / "gpt_oss_20b" / f"stream_greedy_{stamp}.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out = output_path(args.output)
    root = Path(__file__).resolve().parents[1]
    child_dir = out.parent / f"{out.stem}_steps"
    child_dir.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt
    payload: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "initial_prompt": args.prompt,
        "steps_requested": int(args.steps),
        "max_tokens": int(args.max_tokens),
        "top_k": int(args.top_k),
        "expert_mode": args.expert_mode,
        "note": "greedy smoke via repeated streamed full forwards; no KV cache",
        "status": "running",
        "steps": [],
    }
    write_json(out, payload)

    started = time.perf_counter()
    for step in range(int(args.steps)):
        child = child_dir / f"step_{step:02d}.json"
        cmd = [
            sys.executable,
            str(root / "scripts" / "gpt_oss20b_stream_forward_smoke.py"),
            "--prompt",
            prompt,
            "--max-tokens",
            str(args.max_tokens),
            "--expert-mode",
            args.expert_mode,
            "--top-k",
            str(args.top_k),
            "--output",
            str(child),
        ]
        step_started = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        child_payload = json.loads(child.read_text(encoding="utf-8"))
        top = (child_payload.get("top_tokens") or [None])[0]
        step_info = {
            "step": step,
            "prompt_before": prompt,
            "child_artifact": str(child),
            "returncode": proc.returncode,
            "stdout": proc.stdout[-2000:],
            "top_token": top,
            "wall_seconds": time.perf_counter() - step_started,
        }
        payload["steps"].append(step_info)
        if proc.returncode != 0 or not top:
            payload.update(
                {
                    "status": "error",
                    "failed_step": step,
                    "wall_seconds": time.perf_counter() - started,
                }
            )
            write_json(out, payload)
            print(f"artifact={out}", flush=True)
            return proc.returncode or 1
        prompt = prompt + top["text"]
        payload["current_text"] = prompt
        write_json(out, payload)

    payload.update(
        {
            "status": "ok",
            "final_text": prompt,
            "wall_seconds": time.perf_counter() - started,
        }
    )
    write_json(out, payload)
    print(f"artifact={out}", flush=True)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "final_text": payload["final_text"],
                "steps": [s["top_token"] for s in payload["steps"]],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
