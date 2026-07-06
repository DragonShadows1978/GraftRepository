#!/usr/bin/env python3
"""Phase 1 Ollama smoke probe for gpt-oss:20b.

This is an official-runtime check, not a TensorCUDA validation. It records a
short generation response plus GPU memory samples while Ollama loads/runs the
model.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "gpt-oss:20b"
DEFAULT_PROMPT = "Answer with exactly one sentence: what is the capital of France?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT-OSS-20B Ollama smoke probe.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--num-predict", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--think",
        choices=("unset", "true", "false"),
        default="false",
        help="Ollama thinking flag. Default false keeps the smoke answer visible.",
    )
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=900.0)
    return parser.parse_args()


def output_path(path: Path | None) -> Path:
    if path is not None:
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("artifacts") / "gpt_oss_20b" / f"ollama_smoke_{stamp}.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def nvidia_sample() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    first = out.splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    return {
        "timestamp": parts[0],
        "name": parts[1],
        "memory_used_mib": int(parts[2]),
        "memory_total_mib": int(parts[3]),
        "utilization_gpu_percent": int(parts[4]),
    }


def post_json(base_url: str, path: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read().decode("utf-8")
        return {
            "status_code": resp.status,
            "body": json.loads(payload) if payload else None,
        }


def gpu_sampler(
    samples: list[dict[str, Any]],
    stop: threading.Event,
    interval: float,
) -> None:
    while not stop.is_set():
        try:
            sample = nvidia_sample()
            sample["t_monotonic"] = time.perf_counter()
            samples.append(sample)
        except Exception as exc:
            samples.append(
                {
                    "t_monotonic": time.perf_counter(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        stop.wait(interval)


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [s for s in samples if "memory_used_mib" in s]
    if not valid:
        return {"sample_count": len(samples), "valid_sample_count": 0}
    mem = [int(s["memory_used_mib"]) for s in valid]
    util = [int(s["utilization_gpu_percent"]) for s in valid]
    return {
        "sample_count": len(samples),
        "valid_sample_count": len(valid),
        "baseline_memory_used_mib": mem[0],
        "peak_memory_used_mib": max(mem),
        "final_memory_used_mib": mem[-1],
        "peak_delta_mib": max(mem) - mem[0],
        "peak_utilization_gpu_percent": max(util),
    }


def add_think_flag(body: dict[str, Any], value: str) -> None:
    if value == "true":
        body["think"] = True
    elif value == "false":
        body["think"] = False


def main() -> int:
    args = parse_args()
    out = output_path(args.output)
    payload: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "prompt": args.prompt,
        "prompt_format": "Ollama model template for gpt-oss:20b",
        "base_url": args.base_url,
        "request": {
            "stream": False,
            "think": args.think,
            "options": {
                "num_predict": args.num_predict,
                "temperature": args.temperature,
            },
        },
    }
    write_json(out, payload)

    try:
        payload["ollama_show"] = post_json(
            args.base_url,
            "/api/show",
            {"model": args.model},
            timeout=min(args.timeout, 120.0),
        )
    except urllib.error.HTTPError as exc:
        payload["ollama_show"] = {
            "status_code": exc.code,
            "error": exc.read().decode("utf-8", errors="replace"),
        }
    except Exception as exc:
        payload["ollama_show"] = {
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    write_json(out, payload)

    samples: list[dict[str, Any]] = []
    stop = threading.Event()
    sampler = threading.Thread(
        target=gpu_sampler,
        args=(samples, stop, args.sample_interval),
        daemon=True,
    )
    sampler.start()
    started = time.perf_counter()

    try:
        body = {
            "model": args.model,
            "prompt": args.prompt,
            "stream": False,
            "options": {
                "num_predict": args.num_predict,
                "temperature": args.temperature,
            },
        }
        add_think_flag(body, args.think)
        payload["generate"] = post_json(
            args.base_url,
            "/api/generate",
            body,
            timeout=args.timeout,
        )
        payload["status"] = "ok"
    except urllib.error.HTTPError as exc:
        payload["generate"] = {
            "status_code": exc.code,
            "error": exc.read().decode("utf-8", errors="replace"),
        }
        payload["status"] = "http_error"
    except Exception as exc:
        payload["generate"] = {
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        payload["status"] = "error"
    finally:
        stop.set()
        sampler.join(timeout=2.0)

    payload["wall_seconds"] = time.perf_counter() - started
    payload["gpu_samples"] = samples
    payload["gpu_summary"] = summarize_samples(samples)
    write_json(out, payload)
    print(f"artifact={out}", flush=True)
    print(f"status={payload['status']}", flush=True)
    print(json.dumps(payload["gpu_summary"], sort_keys=True), flush=True)
    response = payload.get("generate", {}).get("body", {}).get("response")
    if response:
        print(response.strip()[:500], flush=True)
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
