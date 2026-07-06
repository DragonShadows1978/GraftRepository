#!/usr/bin/env python3
"""GPT-OSS TensorCUDA selected-expert MoE diagnostic smoke.

This runs real GPT-OSS tensors through embedding, attention, post-attention
norm, and the routed MoE for a tiny token batch. It can run either the exact
selected-expert dequant bridge or the packed MXFP4 TensorCUDA expert kernel.
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
    GptOssDiagnosticBlockTC,
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
    parser = argparse.ArgumentParser(
        description="Run one GPT-OSS attention+selected-expert-MoE diagnostic smoke."
    )
    parser.add_argument("--model-dir", default=SNAPSHOT)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--prompt", default="The capital of France is Paris.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1,
        help="tiny by default because selected-expert dequant is diagnostic only",
    )
    parser.add_argument(
        "--expert-mode",
        choices=("dequant", "packed_mxfp4"),
        default="dequant",
        help="selected-expert exact dequant bridge or packed MXFP4 kernel path",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def output_path(path: Path | None) -> Path:
    if path is not None:
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("artifacts") / "gpt_oss_20b" / f"moe_diag_smoke_{stamp}.json"


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


def tensor_stats(t) -> dict[str, float]:
    arr = t.float().numpy().astype(np.float32, copy=False)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "sum": float(arr.sum()),
    }


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
        "expert_mode": args.expert_mode,
        "note": (
            "attention plus selected-expert GPT-OSS MoE diagnostic; "
            "no lm_head and no generation"
        ),
        "gpu_before": nvidia_smi(),
    }
    write_json(out, payload)

    try:
        BlockTC.COMPUTE_DTYPE = "bfloat16"
        QuantLinearTC.FUSED_DECODE = True
        RMSNormTC.USE_FUSED = True

        started = time.perf_counter()
        cfg = GptOss20BConfig.from_model_dir(model_dir)
        where = build_safetensors_map(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        ids = tokenizer(args.prompt, return_tensors="np").input_ids.astype(np.int64)
        ids = ids[:, : max(1, int(args.max_tokens))]

        with tc.no_grad():
            embed = GptOssRowEmbedding(where)
            block = GptOssDiagnosticBlockTC.from_safetensors(
                cfg, where, args.layer, expert_mode=args.expert_mode
            )
            h = embed(ids)
            cos, sin = gpt_oss_yarn_rope_tables(cfg, ids.shape[1])
            h2, kv, route_info = block(h, cos, sin)
            tc.synchronize()
            if hasattr(tc, "empty_cache"):
                tc.empty_cache()

        payload.update(
            {
                "status": "ok",
                "input_ids_shape": list(ids.shape),
                "hidden_shape": tensor_shape(h),
                "output_shape": tensor_shape(h2),
                "output_stats": tensor_stats(h2),
                "kv_shapes": [tensor_shape(kv[0]), tensor_shape(kv[1])],
                "layer_type": cfg.layer_types[int(args.layer)],
                "route_info": route_info,
                "wall_seconds": time.perf_counter() - started,
                "gpu_after": nvidia_smi(),
            }
        )
        write_json(out, payload)
        print(f"artifact={out}", flush=True)
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "expert_mode": payload["expert_mode"],
                    "output_shape": payload["output_shape"],
                    "unique_experts": route_info["unique_experts"],
                    "gpu_after": payload["gpu_after"],
                }
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        payload.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "gpu_after": nvidia_smi(),
            }
        )
        write_json(out, payload)
        print(f"artifact={out}", flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
