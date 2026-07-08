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
import os
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
    gpt_oss_grm_dialect_kwargs,
    gpt_oss_yarn_rope_tables,
    load_tensor_np,
    resolve_gpt_oss_attention_mode,
)
from core.mistral7b_tc import BlockTC, QuantLinearTC, RMSNormTC  # noqa: E402
from core.graft_quant import is_packed_payload, unpack_kv_arrays  # noqa: E402


SNAPSHOT = (
    "/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/"
    "snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a streamed GPT-OSS forward smoke.")
    parser.add_argument("--model-dir", default=SNAPSHOT)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="read prompt text from a file; overrides --prompt before chat-template rendering",
    )
    parser.add_argument(
        "--input-ids-file",
        type=Path,
        default=None,
        help=(
            "read a JSON list of token ids and use it directly; overrides "
            "text tokenization"
        ),
    )
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="render prompt as a single user message with tokenizer chat template",
    )
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--candidate-text",
        action="append",
        default=[],
        help=(
            "score the first token of a candidate continuation at the final "
            "position; repeat for gold/decoys"
        ),
    )
    parser.add_argument(
        "--attention-mode",
        choices=("standard", "apa_selective"),
        default="standard",
    )
    parser.add_argument(
        "--apa-layer-scope",
        choices=("full", "all"),
        default="full",
        help="which GPT-OSS layers may use APA when --attention-mode apa_selective",
    )
    parser.add_argument("--refine-percentile", type=float, default=0.15)
    parser.add_argument("--bulk-bits", type=int, default=8)
    parser.add_argument(
        "--score-ppl",
        action="store_true",
        help="score next-token NLL/PPL over the tokenized prompt instead of only last-token top-k",
    )
    parser.add_argument(
        "--expert-mode",
        choices=("resident_packed_mxfp4", "packed_mxfp4", "dequant"),
        default="resident_packed_mxfp4",
    )
    parser.add_argument(
        "--route-detail",
        choices=("full", "summary"),
        default="full",
        help="full stores per-token MoE routing; summary stores compact histograms",
    )
    parser.add_argument(
        "--expert-empty-cache-interval",
        type=int,
        default=1,
        help=(
            "call tc.empty_cache every N routed expert calls; 0 disables the "
            "inner-loop trim"
        ),
    )
    parser.add_argument("--skip-lm-head", action="store_true")
    parser.add_argument(
        "--capture-graft-dir",
        type=Path,
        default=None,
        help=(
            "write per-layer pre-RoPE K/V graft shards while running the "
            "forward pass"
        ),
    )
    parser.add_argument(
        "--mount-graft-dir",
        type=Path,
        default=None,
        help=(
            "mount per-layer pre-RoPE K/V graft shards before the live prompt; "
            "expects layer_XXX.npz files from --capture-graft-dir"
        ),
    )
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


def graft_layer_path(root: Path, layer_idx: int) -> Path:
    return root / f"layer_{int(layer_idx):03d}.npz"


def load_graft_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    first = graft_layer_path(root, 0)
    if not first.exists():
        raise FileNotFoundError(f"missing graft manifest and {first}")
    with np.load(first) as z:
        return {
            "schema": "gpt_oss_20b_graft_manifest_inferred_v1",
            "token_count": int(z["k"].shape[2]),
            "layers": [{"layer": 0, "path": str(first)}],
        }


def load_graft_layer(root: Path, layer_idx: int, dtype: str):
    """Load one graft shard's (k, v). Transparently detects a packed
    (quantized-at-rest) shard — P3 format-2 hook: written by
    `pack_graft_dir.py`, marked by a `format_version` field
    (`core.graft_quant.is_packed_payload`) — and dequantizes it to fp16
    before upload, exactly as a plain fp16 shard would be. Default path
    (plain {"k","v"} fp16 shard, no format_version field) is UNCHANGED
    byte-for-byte from before this hook existed."""
    path = graft_layer_path(root, layer_idx)
    if not path.exists():
        raise FileNotFoundError(f"missing GPT-OSS graft shard {path}")
    with np.load(path) as z:
        if is_packed_payload(z):
            arrays = unpack_kv_arrays(z, ["k", "v"])
            k_np = np.ascontiguousarray(arrays["k"])
            v_np = np.ascontiguousarray(arrays["v"])
        else:
            k_np = np.ascontiguousarray(z["k"])
            v_np = np.ascontiguousarray(z["v"])
    k = tc.tensor(k_np).astype(dtype)
    v = tc.tensor(v_np).astype(dtype)
    return k, v, int(k_np.shape[2]), int(k_np.nbytes + v_np.nbytes)


def write_graft_manifest(root: Path, payload: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "manifest.json", payload)


def main() -> int:
    args = parse_args()
    out = output_path(args.output)
    model_dir = Path(args.model_dir).expanduser().resolve()
    prompt_source = "arg"
    prompt_text_input = args.prompt
    if args.prompt_file is not None:
        prompt_path = args.prompt_file.expanduser().resolve()
        prompt_text_input = prompt_path.read_text(encoding="utf-8", errors="ignore")
        prompt_source = str(prompt_path)
    if args.input_ids_file is not None and args.use_chat_template:
        raise ValueError("--input-ids-file cannot be combined with --use-chat-template")
    payload: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(model_dir),
        "prompt": prompt_text_input,
        "prompt_source": prompt_source,
        "input_ids_file": None if args.input_ids_file is None else str(args.input_ids_file),
        "use_chat_template": bool(args.use_chat_template),
        "max_tokens": int(args.max_tokens),
        "max_layers": args.max_layers,
        "candidate_texts": list(args.candidate_text),
        "expert_mode": args.expert_mode,
        "route_detail": args.route_detail,
        "expert_empty_cache_interval": int(args.expert_empty_cache_interval),
        "attention_mode": args.attention_mode,
        "apa_layer_scope": args.apa_layer_scope,
        "refine_percentile": float(args.refine_percentile),
        "bulk_bits": int(args.bulk_bits),
        "capture_graft_dir": None if args.capture_graft_dir is None else str(args.capture_graft_dir),
        "mount_graft_dir": None if args.mount_graft_dir is None else str(args.mount_graft_dir),
        "sink_aware_apa_available": bool(hasattr(tc, "apa_blend_softmax_sink")),
        "fused_sink_apa_available": bool(hasattr(tc, "apa_selective_attention_sink")),
        "skip_lm_head": bool(args.skip_lm_head),
        "score_ppl": bool(args.score_ppl),
        "note": (
            "streamed full-stack forward path with quantized TensorCUDA lm_head; "
            "score_ppl writes a teacher-forced window receipt, but this artifact "
            "alone is not generation quality, context, or GRM evidence"
        ),
        "status": "starting",
        "layers": [],
        "gpu_before": nvidia_smi(),
    }
    write_json(out, payload)

    try:
        if args.attention_mode == "apa_selective" and not hasattr(
            tc, "apa_blend_softmax_sink"
        ):
            raise RuntimeError(
                "GPT-OSS apa_selective requires Project-Tensor "
                "tc.apa_blend_softmax_sink"
            )

        BlockTC.COMPUTE_DTYPE = "bfloat16"
        QuantLinearTC.FUSED_DECODE = True
        RMSNormTC.USE_FUSED = True

        started = time.perf_counter()
        cfg = GptOss20BConfig.from_model_dir(model_dir)
        where = build_safetensors_map(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        capture_dir = (
            args.capture_graft_dir.expanduser().resolve()
            if args.capture_graft_dir is not None
            else None
        )
        mount_dir = (
            args.mount_graft_dir.expanduser().resolve()
            if args.mount_graft_dir is not None
            else None
        )
        mount_manifest = load_graft_manifest(mount_dir) if mount_dir is not None else None
        mounted_graft_tokens = (
            int(mount_manifest.get("token_count", 0)) if mount_manifest else 0
        )
        if mounted_graft_tokens:
            payload["mount_graft_manifest"] = mount_manifest
            payload["mount_graft_tokens"] = int(mounted_graft_tokens)
            payload["mount_position_note"] = (
                "live tokens are RoPE-shifted after mounted graft seats; "
                "GPT-OSS sliding layers remain limited by their sliding window"
            )
        if capture_dir is not None:
            capture_dir.mkdir(parents=True, exist_ok=True)
            capture_manifest = {
                "schema": "gpt_oss_20b_prerope_graft_manifest_v1",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "model_dir": str(model_dir),
                "grm_dialect": gpt_oss_grm_dialect_kwargs(cfg),
                "token_count": None,
                "dtype": BlockTC.COMPUTE_DTYPE,
                "layers": [],
                "note": (
                    "per-layer pre-RoPE K/V shards captured by the streamed "
                    "GPT-OSS TensorCUDA harness; remount with --mount-graft-dir"
                ),
            }
            write_graft_manifest(capture_dir, capture_manifest)
            payload["capture_graft_manifest"] = str(capture_dir / "manifest.json")
        prompt_text = prompt_text_input
        if args.use_chat_template:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text_input}],
                tokenize=False,
                add_generation_prompt=True,
            )
        input_ids_source = "tokenizer"
        if args.input_ids_file is not None:
            ids_path = args.input_ids_file.expanduser().resolve()
            raw = json.loads(ids_path.read_text(encoding="utf-8"))
            if raw and isinstance(raw[0], list):
                raw = raw[0]
            raw_ids = np.asarray([[int(x) for x in raw]], dtype=np.int64)
            input_ids_source = str(ids_path)
            prompt_text = tokenizer.decode(
                raw_ids[0].tolist(),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        else:
            raw_ids = tokenizer(prompt_text, return_tensors="np").input_ids.astype(np.int64)
        payload["rendered_prompt"] = prompt_text
        payload["input_ids_source"] = input_ids_source
        write_json(out, payload)
        raw_token_count = int(raw_ids.shape[1])
        if args.score_ppl:
            full_ids = raw_ids[:, : max(2, int(args.max_tokens))]
            if full_ids.shape[1] < 2:
                raise ValueError("score_ppl requires at least two tokens")
            ids = full_ids[:, :-1]
            targets = full_ids[:, 1:]
        else:
            full_ids = raw_ids
            ids = full_ids[:, : max(1, int(args.max_tokens))]
            targets = None
        n_layers = cfg.num_layers if args.max_layers is None else min(args.max_layers, cfg.num_layers)
        planned_attention = [
            resolve_gpt_oss_attention_mode(
                cfg,
                layer_idx,
                args.attention_mode,
                apa_layer_scope=args.apa_layer_scope,
            )
            for layer_idx in range(n_layers)
        ]
        planned_apa_layers = [
            int(item["layer"]) for item in planned_attention if item["apa_active"]
        ]
        planned_standard_layers = [
            int(item["layer"]) for item in planned_attention if not item["apa_active"]
        ]

        with tc.no_grad():
            embed = GptOssRowEmbedding(where)
            h = embed(ids)
            rope_len = int(ids.shape[1] + mounted_graft_tokens)
            cos, sin = gpt_oss_yarn_rope_tables(cfg, rope_len)
            payload.update(
                {
                    "status": "running_layers",
                    "raw_token_count": raw_token_count,
                    "input_truncated": bool(raw_token_count > int(full_ids.shape[1])),
                    "full_input_ids": full_ids.tolist(),
                    "input_ids": ids.tolist(),
                    "target_ids": None if targets is None else targets.tolist(),
                    "input_ids_shape": list(ids.shape),
                    "hidden_shape_initial": tensor_shape(h),
                    "rope_table_length": int(rope_len),
                    "resolved_layers": int(n_layers),
                    "grm_dialect": gpt_oss_grm_dialect_kwargs(cfg),
                    "attention_audit": {
                        "requested_attention_mode": args.attention_mode,
                        "apa_layer_scope": args.apa_layer_scope,
                        "refine_percentile": float(args.refine_percentile),
                        "bulk_bits": int(args.bulk_bits),
                        "sink_aware_apa_available": bool(
                            hasattr(tc, "apa_blend_softmax_sink")
                        ),
                        "fused_sink_apa_available": bool(
                            hasattr(tc, "apa_selective_attention_sink")
                        ),
                        "planned_layers": planned_attention,
                    },
                    "attention_layers_planned_apa": planned_apa_layers,
                    "attention_layers_planned_standard": planned_standard_layers,
                    "attention_layers_used_apa": [],
                    "attention_layers_used_standard": [],
                }
            )
            write_json(out, payload)

            for layer_idx in range(n_layers):
                layer_started = time.perf_counter()
                attention_audit = planned_attention[layer_idx]
                block = GptOssDiagnosticBlockTC.from_safetensors(
                    cfg, where, layer_idx, expert_mode=args.expert_mode
                )
                block.self_attn.attention_mode = attention_audit[
                    "effective_attention_mode"
                ]
                block.self_attn.refine_percentile = float(args.refine_percentile)
                block.self_attn.bulk_bits = int(args.bulk_bits)
                block.mlp.route_detail = args.route_detail
                block.mlp.empty_cache_interval = int(args.expert_empty_cache_interval)
                graft_mount_info = None
                if mount_dir is not None:
                    kg, vg, graft_tokens, graft_bytes = load_graft_layer(
                        mount_dir, layer_idx, BlockTC.COMPUTE_DTYPE
                    )
                    block.self_attn.inject_kv = (kg, vg, 1.0)
                    block.self_attn.graft_seats = int(graft_tokens)
                    graft_mount_info = {
                        "path": str(graft_layer_path(mount_dir, layer_idx)),
                        "token_count": int(graft_tokens),
                        "host_bytes": int(graft_bytes),
                        "sliding_window_limited": bool(
                            cfg.is_sliding_attention(layer_idx)
                        ),
                    }
                if capture_dir is not None:
                    block.self_attn._capture = True
                    block.self_attn._captured = None
                h, kv, route_info = block(h, cos, sin)
                graft_capture_info = None
                if capture_dir is not None:
                    cap = getattr(block.self_attn, "_captured", None)
                    if cap is None:
                        raise RuntimeError(f"GPT-OSS layer {layer_idx} did not capture K/V")
                    k_np = np.ascontiguousarray(cap[0].astype(np.float16, copy=False))
                    v_np = np.ascontiguousarray(cap[1].astype(np.float16, copy=False))
                    shard = graft_layer_path(capture_dir, layer_idx)
                    np.savez(shard, k=k_np, v=v_np)
                    graft_capture_info = {
                        "path": str(shard),
                        "token_count": int(k_np.shape[2]),
                        "host_bytes": int(k_np.nbytes + v_np.nbytes),
                        "k_shape": [int(x) for x in k_np.shape],
                        "v_shape": [int(x) for x in v_np.shape],
                    }
                    capture_manifest["token_count"] = int(k_np.shape[2])
                    capture_manifest["layers"].append(
                        {
                            "layer": int(layer_idx),
                            "layer_type": cfg.layer_types[layer_idx],
                            **graft_capture_info,
                        }
                    )
                    capture_manifest["total_host_bytes"] = int(
                        sum(item["host_bytes"] for item in capture_manifest["layers"])
                    )
                    write_graft_manifest(capture_dir, capture_manifest)
                    block.self_attn._capture = False
                    block.self_attn._captured = None
                tc.synchronize()
                layer_info = {
                    "layer": int(layer_idx),
                    "layer_type": cfg.layer_types[layer_idx],
                    "hidden_shape": tensor_shape(h),
                    "kv_shapes": [tensor_shape(kv[0]), tensor_shape(kv[1])],
                    "graft_mount": graft_mount_info,
                    "graft_capture": graft_capture_info,
                    "attention": attention_audit,
                    "attention_backend": block.self_attn.last_attention_backend,
                    "route_info": route_info,
                    "wall_seconds": time.perf_counter() - layer_started,
                    "gpu_after_layer": nvidia_smi(),
                }
                payload["layers"].append(layer_info)
                payload["completed_layers"] = layer_idx + 1
                if attention_audit["apa_active"]:
                    payload["attention_layers_used_apa"].append(int(layer_idx))
                else:
                    payload["attention_layers_used_standard"].append(int(layer_idx))
                write_json(out, payload)
                del block, kv
                if graft_mount_info is not None:
                    del kg, vg
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
            head_input = h if args.score_ppl else last
            logits = lm_head(head_input)
            logits_2d = logits.float().numpy().astype(np.float32, copy=False).reshape(
                [-1, cfg.vocab_size]
            )
            idx_np = np.argsort(-logits_2d[-1])[: int(args.top_k)].astype(np.int64)
            vals_np = logits_2d[-1, idx_np].astype(np.float32)
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
            candidate_scores = []
            if args.candidate_text:
                final_row = logits_2d[-1]
                row_max = float(final_row.max())
                row_lse = row_max + float(np.log(np.exp(final_row - row_max).sum()))
                for cand in args.candidate_text:
                    cand_ids = tokenizer(
                        cand, add_special_tokens=False
                    ).input_ids
                    if cand_ids and isinstance(cand_ids[0], list):
                        cand_ids = cand_ids[0]
                    if not cand_ids:
                        candidate_scores.append(
                            {
                                "candidate_text": cand,
                                "token_ids": [],
                                "error": "empty_tokenization",
                            }
                        )
                        continue
                    first = int(cand_ids[0])
                    candidate_scores.append(
                        {
                            "candidate_text": cand,
                            "token_ids": [int(x) for x in cand_ids],
                            "first_token_id": first,
                            "first_token_text": tokenizer.decode([first]),
                            "first_token_logit": float(final_row[first]),
                            "first_token_logprob": float(final_row[first] - row_lse),
                        }
                    )
            ppl_payload = None
            if args.score_ppl:
                target_flat = targets.reshape(-1)
                nlls = []
                for row, target in zip(logits_2d, target_flat):
                    mx = float(row.max())
                    lse = mx + float(np.log(np.exp(row - mx).sum()))
                    nlls.append(lse - float(row[int(target)]))
                mean_nll = float(np.mean(nlls))
                ppl_payload = {
                    "token_count": int(len(nlls)),
                    "mean_nll": mean_nll,
                    "ppl": float(np.exp(mean_nll)),
                    "target_ids": target_flat.astype(int).tolist(),
                    "target_text": [tokenizer.decode([int(t)]) for t in target_flat],
                }
            tc.synchronize()

        payload.update(
            {
                "status": "ok",
                "top_tokens": top_tokens,
                "candidate_scores": candidate_scores,
                "ppl": ppl_payload,
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
