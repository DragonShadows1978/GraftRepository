#!/usr/bin/env python3
"""Phase 0 metadata snapshot for openai/gpt-oss-20b.

This script intentionally does not download model weight bodies. It records the
repo revision, selected config/tokenizer metadata, safetensors index metadata,
and safetensors header summaries via HTTP range requests. It also checks whether
the installed tokenizer can render the required Harmony chat template.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError


MODEL_ID = "openai/gpt-oss-20b"
METADATA_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors.index.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record GPT-OSS-20B Phase 0 metadata without downloading weights."
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def output_path(path: Path | None) -> Path:
    if path is not None:
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("artifacts") / "gpt_oss_20b" / f"phase0_snapshot_{stamp}.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def command_path(name: str) -> str | None:
    return shutil.which(name)


def nvidia_smi() -> str | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
        return out.strip()
    except Exception:
        return None


def load_json_file(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def safe_download(model_id: str, filename: str, revision: str | None) -> dict[str, Any]:
    try:
        path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            revision=revision,
            local_files_only=False,
        )
        stat = os.stat(path)
        return {
            "status": "ok",
            "path": path,
            "size_bytes": stat.st_size,
        }
    except EntryNotFoundError as exc:
        return {"status": "missing", "error": str(exc)}
    except Exception as exc:
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def tensor_kind(name: str) -> str:
    if ".mlp.experts." in name:
        return "experts"
    if ".self_attn." in name:
        return "attention"
    if ".mlp.router." in name:
        return "router"
    if "embed_tokens" in name or name == "lm_head.weight":
        return "embed_lmhead"
    if "norm" in name or "layernorm" in name:
        return "norm"
    return "other"


def read_safetensors_header(url: str) -> dict[str, Any]:
    first = requests.get(url, headers={"Range": "bytes=0-7"}, timeout=60)
    first.raise_for_status()
    header_len = struct.unpack("<Q", first.content[:8])[0]
    end = 8 + header_len - 1
    header_resp = requests.get(url, headers={"Range": f"bytes=8-{end}"}, timeout=60)
    header_resp.raise_for_status()
    header = json.loads(header_resp.content)

    dtypes: Counter[str] = Counter()
    bytes_by_dtype: Counter[str] = Counter()
    bytes_by_kind: Counter[str] = Counter()
    examples: dict[str, dict[str, Any]] = {}
    tensor_count = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        tensor_count += 1
        dtype = str(meta["dtype"])
        nbytes = int(meta["data_offsets"][1]) - int(meta["data_offsets"][0])
        kind = tensor_kind(name)
        dtypes[dtype] += 1
        bytes_by_dtype[dtype] += nbytes
        bytes_by_kind[kind] += nbytes
        if kind not in examples:
            examples[kind] = {
                "name": name,
                "dtype": dtype,
                "shape": meta["shape"],
                "size_bytes": nbytes,
            }

    return {
        "url": url,
        "header_len_bytes": header_len,
        "tensor_count": tensor_count,
        "dtypes": dict(sorted(dtypes.items())),
        "bytes_by_dtype": dict(sorted(bytes_by_dtype.items())),
        "gib_by_dtype": {
            k: round(v / (1024 ** 3), 6) for k, v in sorted(bytes_by_dtype.items())
        },
        "bytes_by_kind": dict(sorted(bytes_by_kind.items())),
        "gib_by_kind": {
            k: round(v / (1024 ** 3), 6) for k, v in sorted(bytes_by_kind.items())
        },
        "examples_by_kind": examples,
    }


def summarize_headers(model_id: str, revision: str, index: dict[str, Any]) -> dict[str, Any]:
    files = sorted(set(index.get("weight_map", {}).values()))
    summaries = []
    totals_dtype: Counter[str] = Counter()
    totals_kind: Counter[str] = Counter()
    for filename in files:
        url = f"https://huggingface.co/{model_id}/resolve/{revision}/{filename}"
        summary = read_safetensors_header(url)
        summary["filename"] = filename
        summaries.append(summary)
        totals_dtype.update(summary["bytes_by_dtype"])
        totals_kind.update(summary["bytes_by_kind"])
    return {
        "files": summaries,
        "total_bytes_by_dtype": dict(sorted(totals_dtype.items())),
        "total_gib_by_dtype": {
            k: round(v / (1024 ** 3), 6) for k, v in sorted(totals_dtype.items())
        },
        "total_bytes_by_kind": dict(sorted(totals_kind.items())),
        "total_gib_by_kind": {
            k: round(v / (1024 ** 3), 6) for k, v in sorted(totals_kind.items())
        },
    }


def tokenizer_harmony_probe(model_id: str, revision: str | None) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return {
            "status": "ok",
            "tokenizer_class": tokenizer.__class__.__name__,
            "has_chat_template": bool(getattr(tokenizer, "chat_template", None)),
            "rendered_prefix": rendered[:400],
            "contains_start_token": "<|start|>" in rendered,
            "contains_channel_token": "<|channel|>" in rendered,
            "contains_message_end": "<|message|>" in rendered or "<|end|>" in rendered,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def main() -> int:
    args = parse_args()
    out = output_path(args.output)
    api = HfApi()
    started = time.perf_counter()

    info = api.model_info(args.model_id, revision=args.revision, files_metadata=True)
    revision = info.sha

    payload: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_id": args.model_id,
        "requested_revision": args.revision,
        "resolved_revision": revision,
        "repo_private": getattr(info, "private", None),
        "tags": getattr(info, "tags", None),
        "siblings": [
            {
                "rfilename": s.rfilename,
                "size": getattr(s, "size", None),
            }
            for s in getattr(info, "siblings", [])
        ],
        "environment": {
            "python": sys.version,
            "transformers": package_version("transformers"),
            "huggingface_hub": package_version("huggingface_hub"),
            "safetensors": package_version("safetensors"),
            "vllm": package_version("vllm"),
            "openai_harmony": package_version("openai-harmony"),
            "ollama_path": command_path("ollama"),
            "nvidia_smi": nvidia_smi(),
        },
        "metadata_files": {},
    }
    write_json(out, payload)

    for filename in METADATA_FILES:
        payload["metadata_files"][filename] = safe_download(
            args.model_id, filename, revision
        )
        write_json(out, payload)

    cfg_entry = payload["metadata_files"].get("config.json", {})
    if cfg_entry.get("status") == "ok":
        config = load_json_file(cfg_entry["path"])
        payload["config_summary"] = {
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "hidden_size": config.get("hidden_size"),
            "intermediate_size": config.get("intermediate_size"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "head_dim": config.get("head_dim"),
            "num_local_experts": config.get("num_local_experts"),
            "num_experts_per_tok": config.get("num_experts_per_tok"),
            "experts_per_token": config.get("experts_per_token"),
            "sliding_window": config.get("sliding_window"),
            "max_position_embeddings": config.get("max_position_embeddings"),
            "initial_context_length": config.get("initial_context_length"),
            "rope_theta": config.get("rope_theta"),
            "rope_scaling": config.get("rope_scaling"),
            "vocab_size": config.get("vocab_size"),
            "tie_word_embeddings": config.get("tie_word_embeddings"),
            "layer_types": config.get("layer_types"),
            "quantization_config": config.get("quantization_config"),
        }

    idx_entry = payload["metadata_files"].get("model.safetensors.index.json", {})
    if idx_entry.get("status") == "ok":
        index = load_json_file(idx_entry["path"])
        payload["safetensors_index"] = {
            "metadata": index.get("metadata"),
            "weight_file_count": len(set(index.get("weight_map", {}).values())),
            "tensor_count": len(index.get("weight_map", {})),
        }
        write_json(out, payload)
        payload["safetensors_headers"] = summarize_headers(
            args.model_id, revision, index
        )

    payload["harmony_probe"] = tokenizer_harmony_probe(args.model_id, revision)
    payload["wall_seconds"] = time.perf_counter() - started
    write_json(out, payload)
    print(f"artifact={out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
