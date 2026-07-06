#!/usr/bin/env python3
"""One-layer GPT-OSS TensorCUDA smoke.

This is a loader/runtime shape check for real GPT-OSS tensors. It runs the
embedding row gather plus one attention/residual block. It intentionally skips
MoE and lm_head, so it is not a language-model behavior test.
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

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensor_cuda as tc  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from core.gpt_oss20b_tc import (  # noqa: E402
    GptOss20BConfig,
    GptOssAttentionBlockTC,
    GptOssRowEmbedding,
    build_safetensors_map,
    gpt_oss_yarn_rope_tables,
)
from core.mistral7b_tc import BlockTC, QuantLinearTC, RMSNormTC  # noqa: E402


SNAPSHOT = (
    "/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/"
    "snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one GPT-OSS attention layer smoke.")
    parser.add_argument("--model-dir", default=SNAPSHOT)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--prompt", default="The capital of France is Paris.")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def output_path(path: Path | None) -> Path:
    if path is not None:
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("artifacts") / "gpt_oss_20b" / f"layer_smoke_{stamp}.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def nvidia_smi() -> str | None:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def tensor_shape(t) -> list[int]:
    return [int(x) for x in t.shape]


def main() -> int:
    args = parse_args()
    out = output_path(args.output)
    model_dir = Path(args.model_dir).expanduser().resolve()
    payload: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(model_dir),
        "layer": int(args.layer),
        "prompt": args.prompt,
        "max_tokens": int(args.max_tokens),
        "note": "attention/residual one-layer smoke only; MoE and lm_head skipped",
        "gpu_before": nvidia_smi(),
    }
    write_json(out, payload)

    BlockTC.COMPUTE_DTYPE = "bfloat16"
    QuantLinearTC.FUSED_DECODE = True
    RMSNormTC.USE_FUSED = True

    started = time.perf_counter()
    cfg = GptOss20BConfig.from_model_dir(model_dir)
    where = build_safetensors_map(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    ids = tokenizer(args.prompt, return_tensors="np").input_ids.astype(np.int64)
    if ids.shape[1] > args.max_tokens:
        ids = ids[:, : args.max_tokens]

    with tc.no_grad():
        embed = GptOssRowEmbedding(where)
        block = GptOssAttentionBlockTC.from_safetensors(cfg, where, args.layer)
        h = embed(ids)
        cos, sin = gpt_oss_yarn_rope_tables(cfg, ids.shape[1])
        h2, kv = block(h, cos, sin)
        tc.synchronize()

    payload.update(
        {
            "status": "ok",
            "input_ids_shape": list(ids.shape),
            "hidden_shape": tensor_shape(h),
            "output_shape": tensor_shape(h2),
            "kv_shapes": [tensor_shape(kv[0]), tensor_shape(kv[1])],
            "layer_type": cfg.layer_types[int(args.layer)],
            "full_attention_indices": cfg.full_attention_indices(),
            "sliding_attention_indices": cfg.sliding_attention_indices(),
            "wall_seconds": time.perf_counter() - started,
            "gpu_after": nvidia_smi(),
        }
    )
    write_json(out, payload)
    print(f"artifact={out}", flush=True)
    print(json.dumps({k: payload[k] for k in ("status", "output_shape", "kv_shapes", "gpu_after")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
