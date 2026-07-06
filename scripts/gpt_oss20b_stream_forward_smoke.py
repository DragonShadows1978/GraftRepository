#!/usr/bin/env python3
"""Streamed GPT-OSS TensorCUDA full-stack forward smoke.

This is a loader/runtime smoke, not a quality claim. It streams layers one at a
time so only one layer's packed experts are resident on GPU, then attaches final
norm and an INT4 TensorCUDA lm_head for a next-token top-k receipt.
"""

from __future__ import annotations

import argparse
import gc
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
    load_tensor_np,
)
from core.mistral7b_tc import BlockTC, QuantLinearTC, RMSNormTC  # noqa: E402


SNAPSHOT = (
    "/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/"
    "snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a streamed GPT-OSS forward smoke.")
    parser.add_argument("--model-dir", default=SNAPSHOT)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--expert-mode",
        choices=("resident_packed_mxfp4", "packed_mxfp4", "dequant"),
        default="resident_packed_mxfp4",
    )
    parser.add_argument("--skip-lm-head", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def output_path(path: Path | None) -> Path:
    if path is not None:
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("artifacts") / "gpt_oss_20b" / f"stream_forward_{stamp}.json"


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
        "prompt": args.prompt,
        "max_tokens": int(args.max_tokens),
        "max_layers": args.max_layers,
        "expert_mode": args.expert_mode,
        "skip_lm_head": bool(args.skip_lm_head),
        "note": (
            "streamed full-stack forward smoke; quantized TensorCUDA lm_head; "
            "not PPL, not generation quality, not APA/GRM evidence"
        ),
        "status": "starting",
        "layers": [],
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
        n_layers = cfg.num_layers if args.max_layers is None else min(args.max_layers, cfg.num_layers)

        with tc.no_grad():
            embed = GptOssRowEmbedding(where)
            h = embed(ids)
            cos, sin = gpt_oss_yarn_rope_tables(cfg, ids.shape[1])
            payload.update(
                {
                    "status": "running_layers",
                    "input_ids": ids.tolist(),
                    "input_ids_shape": list(ids.shape),
                    "hidden_shape_initial": tensor_shape(h),
                    "resolved_layers": int(n_layers),
                }
            )
            write_json(out, payload)

            for layer_idx in range(n_layers):
                layer_started = time.perf_counter()
                block = GptOssDiagnosticBlockTC.from_safetensors(
                    cfg, where, layer_idx, expert_mode=args.expert_mode
                )
                h, kv, route_info = block(h, cos, sin)
                tc.synchronize()
                layer_info = {
                    "layer": int(layer_idx),
                    "layer_type": cfg.layer_types[layer_idx],
                    "hidden_shape": tensor_shape(h),
                    "kv_shapes": [tensor_shape(kv[0]), tensor_shape(kv[1])],
                    "route_info": route_info,
                    "wall_seconds": time.perf_counter() - layer_started,
                    "gpu_after_layer": nvidia_smi(),
                }
                payload["layers"].append(layer_info)
                payload["completed_layers"] = layer_idx + 1
                write_json(out, payload)
                del block, kv
                gc.collect()
                if hasattr(tc, "empty_cache"):
                    tc.empty_cache()

            norm = RMSNormTC(cfg.hidden_dim, cfg.rms_norm_eps)
            norm.weight = tc.tensor(load_tensor_np(where, "model.norm.weight"), dtype="float32")
            h = norm(h)
            h = h.half() if BlockTC.COMPUTE_DTYPE == "float16" else h.astype(BlockTC.COMPUTE_DTYPE)
            last = h.slice(1, h.shape[1] - 1, 1)

            payload.update(
                {
                    "status": "finished_layers" if args.skip_lm_head else "loading_lm_head",
                    "final_hidden_shape": tensor_shape(h),
                    "final_hidden_stats": tensor_stats(last),
                    "gpu_before_lm_head": nvidia_smi(),
                }
            )
            write_json(out, payload)

            if args.skip_lm_head:
                payload.update(
                    {
                        "status": "ok",
                        "top_tokens": [],
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
                            "completed_layers": payload["completed_layers"],
                            "skip_lm_head": True,
                            "gpu_after": payload["gpu_after"],
                        }
                    ),
                    flush=True,
                )
                return 0

            lm_head = QuantLinearTC(
                load_tensor_np(where, "lm_head.weight"),
                group_size=cfg.group_size,
            )
            logits = lm_head(last).reshape([cfg.vocab_size])
            vals, idx = logits.float().topk(int(args.top_k), True)
            vals_np = vals.numpy().astype(np.float32)
            idx_np = idx.numpy().astype(np.int64)
            top_tokens = []
            for rank, (tok, val) in enumerate(zip(idx_np.tolist(), vals_np.tolist())):
                top_tokens.append(
                    {
                        "rank": rank,
                        "token_id": int(tok),
                        "logit": float(val),
                        "text": tokenizer.decode([int(tok)]),
                    }
                )
            tc.synchronize()

        payload.update(
            {
                "status": "ok",
                "top_tokens": top_tokens,
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
                    "completed_layers": payload["completed_layers"],
                    "top_tokens": top_tokens[:3],
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
