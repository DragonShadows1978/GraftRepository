#!/usr/bin/env python3
"""GPT-OSS real-text PPL gate using the streamed TensorCUDA forward path.

This runner exists to move GPT-OSS beyond the six-target toy PPL smoke. It
scores fixed real-text windows by invoking `gpt_oss20b_stream_forward_smoke.py`
with `--score-ppl`, then aggregates weighted NLL/PPL and observed memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
DEFAULT_CORPUS = REPO_ROOT / "docs" / "GRM_Primer.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT-OSS real-text PPL gate.")
    parser.add_argument("--model-dir", default=SNAPSHOT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--window-tokens", type=int, default=64)
    parser.add_argument("--n-windows", type=int, default=2)
    parser.add_argument("--stride-tokens", type=int, default=64)
    parser.add_argument(
        "--settings",
        default="standard,apa_r0.15,apa_r0.10",
        help="Comma list: standard and/or apa_r0.xx entries.",
    )
    parser.add_argument("--apa-layer-scope", choices=("full", "all"), default="full")
    parser.add_argument("--bulk-bits", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument(
        "--expert-mode",
        choices=("resident_packed_mxfp4", "packed_mxfp4", "dequant"),
        default="resident_packed_mxfp4",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def output_path(path: Path | None) -> Path:
    if path is not None:
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "artifacts" / "gpt_oss_20b" / f"realtext_ppl_{stamp}.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_setting(raw: str) -> dict[str, Any]:
    item = raw.strip()
    if item == "standard":
        return {
            "name": "standard",
            "attention_mode": "standard",
            "refine_percentile": None,
        }
    if item.startswith("apa_r"):
        refine = float(item[len("apa_r") :])
        return {
            "name": f"apa_r{refine:.2f}",
            "attention_mode": "apa_selective",
            "refine_percentile": refine,
        }
    raise ValueError(f"unsupported setting {raw!r}")


def parse_settings(raw: str) -> list[dict[str, Any]]:
    settings = [parse_setting(x) for x in raw.split(",") if x.strip()]
    if not settings:
        raise ValueError("at least one setting is required")
    return settings


def parse_gpu_memory_mib(raw: str | None) -> int | None:
    if not raw:
        return None
    first = raw.splitlines()[0]
    parts = [x.strip() for x in first.split(",")]
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def max_observed_memory_mib(artifact: dict[str, Any]) -> int | None:
    values: list[int] = []
    for key in ("gpu_before", "gpu_before_lm_head", "gpu_after"):
        mem = parse_gpu_memory_mib(artifact.get(key))
        if mem is not None:
            values.append(mem)
    for layer in artifact.get("layers", []):
        mem = parse_gpu_memory_mib(layer.get("gpu_after_layer"))
        if mem is not None:
            values.append(mem)
    return max(values) if values else None


def summarize_setting(runs: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [run for run in runs if run.get("status") == "ok" and run.get("ppl")]
    total_tokens = sum(int(run["ppl"]["token_count"]) for run in ok)
    total_nll = sum(
        float(run["ppl"]["mean_nll"]) * int(run["ppl"]["token_count"]) for run in ok
    )
    max_mem = max(
        (int(run["max_observed_memory_mib"]) for run in ok if run.get("max_observed_memory_mib") is not None),
        default=None,
    )
    mean_nll = (total_nll / total_tokens) if total_tokens else None
    return {
        "status": "ok" if len(ok) == len(runs) else "partial" if ok else "error",
        "ok_windows": len(ok),
        "total_windows": len(runs),
        "scored_tokens": total_tokens,
        "mean_nll": mean_nll,
        "ppl": math.exp(mean_nll) if mean_nll is not None else None,
        "max_observed_memory_mib": max_mem,
    }


def build_windows(
    tokenizer,
    text: str,
    *,
    window_tokens: int,
    n_windows: int,
    stride_tokens: int,
) -> tuple[list[dict[str, Any]], int]:
    ids = tokenizer(text, return_tensors="np").input_ids[0].astype(int).tolist()
    windows = []
    for idx in range(n_windows):
        start = idx * stride_tokens
        chunk = ids[start : start + window_tokens]
        if len(chunk) < window_tokens:
            break
        windows.append(
            {
                "window_index": idx,
                "token_start": start,
                "token_count": len(chunk),
                "text": tokenizer.decode(chunk, skip_special_tokens=False),
            }
        )
    return windows, len(ids)


def run_forward_window(
    *,
    model_dir: Path,
    setting: dict[str, Any],
    window: dict[str, Any],
    output: Path,
    window_tokens: int,
    apa_layer_scope: str,
    bulk_bits: int,
    top_k: int,
    expert_mode: str,
    max_layers: int | None,
) -> dict[str, Any]:
    script = REPO_ROOT / "scripts" / "gpt_oss20b_stream_forward_smoke.py"
    cmd = [
        sys.executable,
        str(script),
        "--model-dir",
        str(model_dir),
        "--prompt",
        window["text"],
        "--max-tokens",
        str(window_tokens),
        "--attention-mode",
        setting["attention_mode"],
        "--expert-mode",
        expert_mode,
        "--top-k",
        str(top_k),
        "--score-ppl",
        "--output",
        str(output),
    ]
    if setting["attention_mode"] == "apa_selective":
        cmd.extend(
            [
                "--apa-layer-scope",
                apa_layer_scope,
                "--refine-percentile",
                str(setting["refine_percentile"]),
                "--bulk-bits",
                str(bulk_bits),
            ]
        )
    if max_layers is not None:
        cmd.extend(["--max-layers", str(max_layers)])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPYCACHEPREFIX"] = "/tmp/codex_pycache"
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}:/mnt/ForgeRealm/Project-Tensor/tensor_cuda:"
        f"{env.get('PYTHONPATH', '')}"
    )
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, text=True, capture_output=True)
    artifact = None
    if output.exists():
        artifact = json.loads(output.read_text(encoding="utf-8"))
    result = {
        "window_index": window["window_index"],
        "token_start": window["token_start"],
        "command": cmd,
        "artifact": str(output),
        "returncode": proc.returncode,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-2000:],
        "wall_seconds": time.perf_counter() - started,
    }
    if artifact is None:
        result.update({"status": "error", "error": "artifact was not written"})
        return result
    result.update(
        {
            "status": artifact.get("status"),
            "ppl": artifact.get("ppl"),
            "top_tokens": artifact.get("top_tokens"),
            "attention_layers_used_apa": artifact.get("attention_layers_used_apa"),
            "attention_layers_used_standard": artifact.get(
                "attention_layers_used_standard"
            ),
            "gpu_before": artifact.get("gpu_before"),
            "gpu_after": artifact.get("gpu_after"),
            "max_observed_memory_mib": max_observed_memory_mib(artifact),
        }
    )
    if artifact.get("status") != "ok":
        result["error"] = artifact.get("error")
        result["error_type"] = artifact.get("error_type")
    return result


def main() -> int:
    args = parse_args()
    if args.window_tokens < 2:
        raise ValueError("--window-tokens must be >= 2")
    if args.n_windows < 1:
        raise ValueError("--n-windows must be >= 1")
    if args.stride_tokens < 1:
        raise ValueError("--stride-tokens must be >= 1")

    out = output_path(args.output)
    run_dir = out.with_suffix("")
    model_dir = Path(args.model_dir).expanduser().resolve()
    corpus = args.corpus.expanduser().resolve()
    settings = parse_settings(args.settings)

    from transformers import AutoTokenizer

    text = corpus.read_text(encoding="utf-8", errors="ignore")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    windows, corpus_token_count = build_windows(
        tokenizer,
        text,
        window_tokens=args.window_tokens,
        n_windows=args.n_windows,
        stride_tokens=args.stride_tokens,
    )
    if not windows:
        raise ValueError("corpus did not produce any complete windows")

    payload: dict[str, Any] = {
        "schema": "gpt_oss20b_realtext_ppl_gate_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
        "model_dir": str(model_dir),
        "corpus": str(corpus),
        "corpus_sha256": sha256_file(corpus),
        "corpus_token_count": corpus_token_count,
        "window_tokens": int(args.window_tokens),
        "n_windows_requested": int(args.n_windows),
        "stride_tokens": int(args.stride_tokens),
        "settings_requested": [s["name"] for s in settings],
        "apa_layer_scope": args.apa_layer_scope,
        "bulk_bits": int(args.bulk_bits),
        "expert_mode": args.expert_mode,
        "max_layers": args.max_layers,
        "windows": [
            {k: v for k, v in window.items() if k != "text"} for window in windows
        ],
        "runs": [],
        "summary": {},
    }
    write_json(out, payload)

    for setting in settings:
        setting_runs = []
        for window in windows:
            artifact = (
                run_dir
                / f"{setting['name']}_w{int(window['window_index']):02d}.json"
            )
            run = run_forward_window(
                model_dir=model_dir,
                setting=setting,
                window=window,
                output=artifact,
                window_tokens=args.window_tokens,
                apa_layer_scope=args.apa_layer_scope,
                bulk_bits=args.bulk_bits,
                top_k=args.top_k,
                expert_mode=args.expert_mode,
                max_layers=args.max_layers,
            )
            setting_runs.append(run)
            payload["runs"].append({"setting": setting["name"], **run})
            payload["summary"][setting["name"]] = summarize_setting(setting_runs)
            write_json(out, payload)

    payload["status"] = (
        "ok"
        if all(item.get("status") == "ok" for item in payload["summary"].values())
        else "partial"
    )
    write_json(out, payload)
    print(f"artifact={out}", flush=True)
    print(json.dumps({"status": payload["status"], "summary": payload["summary"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
