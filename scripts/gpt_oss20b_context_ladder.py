#!/usr/bin/env python3
"""GPT-OSS real-token context ladder.

Runs one streamed TensorCUDA process per (setting, context length) using prompt
files built from real corpus tokens. This is the H4 evidence runner: it records
pass/fail/OOM, actual token length, per-layer memory, and APA backend usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = (
    "/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/"
    "snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
)
DEFAULT_CORPUS_DIR = REPO_ROOT / "docs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT-OSS real-token context ladder.")
    parser.add_argument("--model-dir", default=SNAPSHOT)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--lengths", default="128,256,512")
    parser.add_argument("--settings", default="standard,apa_r0.15")
    parser.add_argument("--apa-layer-scope", choices=("full", "all"), default="full")
    parser.add_argument("--bulk-bits", type=int, default=8)
    parser.add_argument(
        "--expert-mode",
        choices=("resident_packed_mxfp4", "packed_mxfp4", "dequant"),
        default="resident_packed_mxfp4",
    )
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument(
        "--with-lm-head",
        action="store_true",
        help="include final lm_head/top-k; default is prefill-only skip-lm-head",
    )
    parser.add_argument(
        "--route-detail",
        choices=("full", "summary"),
        default="summary",
        help="MoE routing receipt detail passed to stream runner",
    )
    parser.add_argument(
        "--expert-empty-cache-interval",
        type=int,
        default=0,
        help="inner expert-loop tc.empty_cache interval; 0 disables it",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def output_path(path: Path | None) -> Path:
    if path is not None:
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "artifacts" / "gpt_oss_20b" / f"context_ladder_{stamp}.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_lengths(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("at least one context length is required")
    bad = [x for x in vals if x < 2]
    if bad:
        raise ValueError(f"context lengths must be >= 2, got {bad}")
    return vals


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


def gpu_memory_mib() -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
        vals = [int(x.strip()) for x in out.splitlines() if x.strip()]
        return vals[0] if vals else None
    except Exception:
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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_corpus(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if args.corpus is not None:
        path = args.corpus.expanduser().resolve()
        data = path.read_bytes()
        return data.decode("utf-8", errors="ignore"), {
            "kind": "file",
            "path": str(path),
            "sha256": sha256_bytes(data),
        }

    root = args.corpus_dir.expanduser().resolve()
    parts = []
    h = hashlib.sha256()
    files = sorted(root.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"no .md files found in {root}")
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


def build_prompt_text(tokenizer, ids: list[int], target_len: int) -> str:
    if len(ids) < target_len:
        raise ValueError(
            f"corpus has only {len(ids)} tokens; cannot build {target_len}-token prompt"
        )
    return tokenizer.decode(ids[:target_len], skip_special_tokens=False)


def classify_run(artifact: dict[str, Any] | None, returncode: int, stderr: str) -> str:
    text = (stderr or "").lower()
    if artifact is not None:
        text += " " + str(artifact.get("error", "")).lower()
        if artifact.get("status") == "ok" and returncode == 0:
            return "pass"
    if "out of memory" in text or "cuda" in text and "memory" in text:
        return "oom"
    return "fail"


def summarize_ladder(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_setting: dict[str, dict[str, Any]] = {}
    for run in runs:
        item = by_setting.setdefault(
            run["setting"],
            {
                "passes": [],
                "failures": [],
                "ooms": [],
                "first_failure": None,
                "first_oom": None,
                "max_pass_tokens": None,
            },
        )
        if run["classification"] == "pass":
            item["passes"].append(run["target_tokens"])
            item["max_pass_tokens"] = max(item["passes"])
        elif run["classification"] == "oom":
            item["ooms"].append(run["target_tokens"])
            item["first_oom"] = item["first_oom"] or run["target_tokens"]
            item["first_failure"] = item["first_failure"] or run["target_tokens"]
        else:
            item["failures"].append(run["target_tokens"])
            item["first_failure"] = item["first_failure"] or run["target_tokens"]
    return by_setting


def run_one(
    *,
    model_dir: Path,
    prompt_file: Path,
    target_len: int,
    setting: dict[str, Any],
    output: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    script = REPO_ROOT / "scripts" / "gpt_oss20b_stream_forward_smoke.py"
    cmd = [
        sys.executable,
        str(script),
        "--model-dir",
        str(model_dir),
        "--prompt-file",
        str(prompt_file),
        "--max-tokens",
        str(target_len),
        "--attention-mode",
        setting["attention_mode"],
        "--expert-mode",
        args.expert_mode,
        "--route-detail",
        args.route_detail,
        "--expert-empty-cache-interval",
        str(args.expert_empty_cache_interval),
        "--output",
        str(output),
    ]
    if not args.with_lm_head:
        cmd.append("--skip-lm-head")
    if setting["attention_mode"] == "apa_selective":
        cmd.extend(
            [
                "--apa-layer-scope",
                args.apa_layer_scope,
                "--refine-percentile",
                str(setting["refine_percentile"]),
                "--bulk-bits",
                str(args.bulk_bits),
            ]
        )
    if args.max_layers is not None:
        cmd.extend(["--max-layers", str(args.max_layers)])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPYCACHEPREFIX"] = "/tmp/codex_pycache"
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}:/mnt/ForgeRealm/Project-Tensor/tensor_cuda:"
        f"{env.get('PYTHONPATH', '')}"
    )
    started = time.perf_counter()
    mem_samples: list[int] = []
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    while proc.poll() is None:
        mem = gpu_memory_mib()
        if mem is not None:
            mem_samples.append(mem)
        time.sleep(0.5)
    stdout, stderr = proc.communicate()
    mem = gpu_memory_mib()
    if mem is not None:
        mem_samples.append(mem)
    artifact = None
    if output.exists():
        artifact = json.loads(output.read_text(encoding="utf-8"))
    backend_counts = None
    if artifact is not None:
        backend_counts = dict(Counter(x.get("attention_backend") for x in artifact.get("layers", [])))
    artifact_max_mem = None if artifact is None else max_observed_memory_mib(artifact)
    monitor_peak = max(mem_samples) if mem_samples else None
    classification = classify_run(artifact, proc.returncode, stderr)
    return {
        "setting": setting["name"],
        "target_tokens": int(target_len),
        "artifact": str(output),
        "prompt_file": str(prompt_file),
        "command": cmd,
        "returncode": proc.returncode,
        "classification": classification,
        "status": None if artifact is None else artifact.get("status"),
        "actual_input_tokens": None if artifact is None else artifact.get("input_ids_shape", [None, None])[1],
        "raw_token_count": None if artifact is None else artifact.get("raw_token_count"),
        "completed_layers": None if artifact is None else artifact.get("completed_layers"),
        "attention_layers_used_apa": None if artifact is None else artifact.get("attention_layers_used_apa"),
        "attention_layers_used_standard": None if artifact is None else artifact.get("attention_layers_used_standard"),
        "attention_backend_counts": backend_counts,
        "max_observed_memory_mib": artifact_max_mem,
        "monitor_peak_memory_mib": monitor_peak,
        "monitor_sample_count": len(mem_samples),
        "gpu_before": None if artifact is None else artifact.get("gpu_before"),
        "gpu_after": None if artifact is None else artifact.get("gpu_after"),
        "error_type": None if artifact is None else artifact.get("error_type"),
        "error": None if artifact is None else artifact.get("error"),
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> int:
    args = parse_args()
    out = output_path(args.output)
    run_dir = out.with_suffix("")
    prompt_dir = run_dir / "prompts"
    model_dir = Path(args.model_dir).expanduser().resolve()
    lengths = parse_lengths(args.lengths)
    settings = parse_settings(args.settings)

    from transformers import AutoTokenizer

    text, source = read_corpus(args)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    ids = tokenizer(text, return_tensors="np").input_ids[0].astype(int).tolist()
    if len(ids) < max(lengths):
        raise ValueError(
            f"corpus token count {len(ids)} is below requested max length {max(lengths)}"
        )

    payload: dict[str, Any] = {
        "schema": "gpt_oss20b_context_ladder_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
        "model_dir": str(model_dir),
        "corpus_source": source,
        "corpus_token_count": len(ids),
        "lengths": lengths,
        "settings": [s["name"] for s in settings],
        "apa_layer_scope": args.apa_layer_scope,
        "bulk_bits": int(args.bulk_bits),
        "expert_mode": args.expert_mode,
        "route_detail": args.route_detail,
        "expert_empty_cache_interval": int(args.expert_empty_cache_interval),
        "with_lm_head": bool(args.with_lm_head),
        "max_layers": args.max_layers,
        "runs": [],
        "summary": {},
    }
    write_json(out, payload)

    for setting in settings:
        for target_len in lengths:
            prompt_text = build_prompt_text(tokenizer, ids, target_len)
            prompt_file = prompt_dir / f"prompt_{target_len}.txt"
            prompt_file.parent.mkdir(parents=True, exist_ok=True)
            prompt_file.write_text(prompt_text, encoding="utf-8")
            artifact = run_dir / f"{setting['name']}_{target_len}.json"
            run = run_one(
                model_dir=model_dir,
                prompt_file=prompt_file,
                target_len=target_len,
                setting=setting,
                output=artifact,
                args=args,
            )
            payload["runs"].append(run)
            payload["summary"] = summarize_ladder(payload["runs"])
            write_json(out, payload)
            if run["classification"] in {"oom", "fail"}:
                break

    payload["status"] = (
        "ok"
        if all(run["classification"] == "pass" for run in payload["runs"])
        else "partial"
    )
    write_json(out, payload)
    print(f"artifact={out}", flush=True)
    print(json.dumps({"status": payload["status"], "summary": payload["summary"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
