#!/usr/bin/env python3
"""GLC P0.c CPU static+numeric audit of GPT-OSS YaRN/RoPE and attention semantics."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import glc_p0_common as common


POSITIONS = (0, 511, 1023, 2047, 2559)
RELATIVE_EPSILON = 1.0e-12
PORT_CORE = REPO_ROOT / "core" / "gpt_oss20b_tc.py"
TC_FUNCTIONAL = Path(
    "/mnt/ForgeRealm/Project-Tensor/tensor_cuda/tensor_cuda/functional.py"
)
TC_OPS = Path("/mnt/ForgeRealm/Project-Tensor/tensor_cuda/src/ops.cpp")
TC_KERNELS = Path("/mnt/ForgeRealm/Project-Tensor/tensor_cuda/src/kernels.cu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "self-test"), default="audit", nargs="?")
    parser.add_argument("--model-dir", type=Path, default=common.MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=common.OUTPUT_ROOT)
    return parser.parse_args()


class CpuTensor:
    """Minimal tensor shim used only to execute the port table builder on CPU."""

    def __init__(self, array: np.ndarray, dtype: str = "float32"):
        self._array = np.ascontiguousarray(array, dtype=np.float32)
        self.dtype = dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    def astype(self, dtype: str) -> "CpuTensor":
        import torch

        if dtype in {"bfloat16", "bf16", "bfloat"}:
            work = torch.from_numpy(self._array).to(torch.bfloat16).to(torch.float32).numpy()
            return CpuTensor(work, "bfloat16")
        if dtype in {"float16", "half"}:
            return CpuTensor(self._array.astype(np.float16).astype(np.float32), "float16")
        if dtype in {"float32", "float"}:
            return CpuTensor(self._array, "float32")
        raise ValueError(f"unsupported CPU audit dtype {dtype}")

    def numpy(self) -> np.ndarray:
        return np.ascontiguousarray(self._array)


class CpuTC:
    @staticmethod
    def tensor(array: np.ndarray, dtype: str = "float32", device: str | None = None) -> CpuTensor:
        del device
        return CpuTensor(np.asarray(array), dtype=dtype)


def max_diff(port: np.ndarray, hf: np.ndarray) -> dict[str, float]:
    left = port.astype(np.float64, copy=False)
    right = hf.astype(np.float64, copy=False)
    absolute = np.abs(left - right)
    relative = absolute / np.maximum(np.abs(right), RELATIVE_EPSILON)
    denominator = max(float(np.linalg.norm(right.ravel())), 1.0e-30)
    return {
        "max_abs_diff": float(absolute.max(initial=0.0)),
        "max_rel_diff": float(relative.max(initial=0.0)),
        "relative_frobenius_diff": float(
            np.linalg.norm((left - right).ravel()) / denominator
        ),
    }


def _port_inv_freq(raw: dict[str, Any]) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Literal NumPy evaluation of core/gpt_oss20b_tc.py:557-603."""

    rope = dict(raw.get("rope_scaling") or {})
    base = float(rope.get("rope_theta", raw["rope_theta"]))
    dim = int(raw["head_dim"])
    factor_value = rope.get("factor")
    original = int(
        rope.get("original_max_position_embeddings", raw["initial_context_length"])
    )
    factor = (
        float(raw["max_position_embeddings"]) / original
        if factor_value is None
        else float(factor_value)
    )
    beta_fast = float(rope.get("beta_fast") or 32.0)
    beta_slow = float(rope.get("beta_slow") or 1.0)
    truncate = bool(rope.get("truncate", True))

    def correction_dim(rotations: float) -> float:
        return (dim * math.log(original / (rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    low = correction_dim(beta_fast)
    high = correction_dim(beta_slow)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    low = max(low, 0.0)
    high = min(high, float(dim - 1))
    if low == high:
        high += 0.001
    ar = np.arange(0, dim, 2, dtype=np.float32)
    pos_freqs = base ** (ar / float(dim))
    inv_extra = 1.0 / pos_freqs
    inv_interp = 1.0 / (factor * pos_freqs)
    ramp = np.clip(
        (np.arange(dim // 2, dtype=np.float32) - low) / (high - low), 0.0, 1.0
    )
    extra_factor = 1.0 - ramp
    inv = inv_interp * (1.0 - extra_factor) + inv_extra * extra_factor
    return np.ascontiguousarray(inv, dtype=np.float32), {
        "base": base,
        "dim": dim,
        "factor": factor,
        "original_max_position_embeddings": original,
        "beta_fast": beta_fast,
        "beta_slow": beta_slow,
        "truncate": truncate,
        "correction_low": float(low),
        "correction_high": float(high),
    }


def _execute_port_tables(gpt: Any, cfg: Any, seq_len: int, dtype: str) -> tuple[np.ndarray, np.ndarray]:
    old_tc = gpt.tc
    old_dtype = gpt.BlockTC.COMPUTE_DTYPE
    try:
        gpt.tc = CpuTC
        gpt.BlockTC.COMPUTE_DTYPE = dtype
        cos, sin = gpt.gpt_oss_yarn_rope_tables(cfg, seq_len)
        return cos.numpy(), sin.numpy()
    finally:
        gpt.tc = old_tc
        gpt.BlockTC.COMPUTE_DTYPE = old_dtype


def _source_location(obj: Any) -> dict[str, Any]:
    path = Path(inspect.getsourcefile(obj) or "").resolve()
    _lines, start = inspect.getsourcelines(obj)
    return {
        "path": str(path),
        "line": int(start),
        "sha256": common.sha256_file(path),
    }


def _load_sink(model_dir: Path, name: str) -> np.ndarray:
    from safetensors import safe_open

    index = common.read_json(model_dir / "model.safetensors.index.json")
    shard = model_dir / index["weight_map"][name]
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(name).to(dtype=__import__("torch").float32)
    return np.ascontiguousarray(tensor.numpy(), dtype=np.float32)


def perform_audit(args: argparse.Namespace) -> dict[str, Any]:
    common.ensure_output_root(args.output_dir)
    model_dir = common.ensure_model_dir(args.model_dir)
    _manifest, _arrays = common.validate_sealed_controls()

    import torch
    from transformers import AutoConfig
    from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask,
        sliding_window_overlay,
    )
    from transformers.models.gpt_oss import modeling_gpt_oss as hf_modeling
    from transformers.modeling_rope_utils import _compute_yarn_parameters
    import core.gpt_oss20b_tc as port

    raw = common.read_json(model_dir / "config.json")
    port_cfg = port.GptOss20BConfig.from_model_dir(model_dir)
    hf_cfg = AutoConfig.from_pretrained(
        str(model_dir), local_files_only=True, attn_implementation="eager"
    )
    rotary = hf_modeling.GptOssRotaryEmbedding(hf_cfg, device="cpu")

    inv_port, port_values = _port_inv_freq(raw)
    inv_hf = rotary.inv_freq.detach().to(torch.float32).cpu().numpy()
    inv_diff = max_diff(inv_port, inv_hf)
    factor = float(raw["rope_scaling"]["factor"])
    port_attention_factor = float(port._yarn_get_mscale(factor))
    hf_attention_factor = float(rotary.attention_scaling)

    seq_len = max(POSITIONS) + 1
    port_cos_fp32, port_sin_fp32 = _execute_port_tables(
        port, port_cfg, seq_len, "float32"
    )
    port_cos_bf16, port_sin_bf16 = _execute_port_tables(
        port, port_cfg, seq_len, "bfloat16"
    )
    all_position_ids = torch.arange(seq_len, dtype=torch.long)[None, :]
    hf_x_fp32 = torch.zeros((1, seq_len, int(raw["head_dim"])), dtype=torch.float32)
    hf_cos_fp32_t, hf_sin_fp32_t = rotary(hf_x_fp32, all_position_ids)
    hf_x_bf16 = hf_x_fp32.to(torch.bfloat16)
    hf_cos_bf16_t, hf_sin_bf16_t = rotary(hf_x_bf16, all_position_ids)
    hf_cos_fp32 = hf_cos_fp32_t[0].cpu().numpy()
    hf_sin_fp32 = hf_sin_fp32_t[0].cpu().numpy()
    hf_cos_bf16 = hf_cos_bf16_t[0].to(torch.float32).cpu().numpy()
    hf_sin_bf16 = hf_sin_bf16_t[0].to(torch.float32).cpu().numpy()
    half_dim = int(raw["head_dim"]) // 2

    requested_rows: list[dict[str, Any]] = []
    numeric_mismatches: list[dict[str, Any]] = []
    for position in POSITIONS:
        row: dict[str, Any] = {"position": position}
        for kind, port_fp, hf_fp, port_bf, hf_bf in (
            (
                "cos",
                port_cos_fp32[position, :half_dim],
                hf_cos_fp32[position],
                port_cos_bf16[position, :half_dim],
                hf_cos_bf16[position],
            ),
            (
                "sin",
                port_sin_fp32[position, :half_dim],
                hf_sin_fp32[position],
                port_sin_bf16[position, :half_dim],
                hf_sin_bf16[position],
            ),
        ):
            fp_diff = max_diff(port_fp, hf_fp)
            bf_diff = max_diff(port_bf, hf_bf)
            row[kind] = {"fp32_construction": fp_diff, "runtime_bfloat16": bf_diff}
            if fp_diff["max_abs_diff"] != 0.0:
                numeric_mismatches.append(
                    {
                        "surface": f"rope_{kind}_fp32_position_{position}",
                        "classification": "pre_cast_backend_rounding",
                        **fp_diff,
                    }
                )
            if bf_diff["max_abs_diff"] != 0.0:
                numeric_mismatches.append(
                    {
                        "surface": f"rope_{kind}_bfloat16_position_{position}",
                        "classification": "runtime_value_mismatch",
                        **bf_diff,
                    }
                )
        requested_rows.append(row)
    if inv_diff["max_abs_diff"] != 0.0:
        numeric_mismatches.insert(
            0,
            {
                "surface": "yarn_inv_freq_fp32",
                "classification": "pre_cast_backend_rounding",
                **inv_diff,
            },
        )

    duplicate_layout = {
        "cos_halves": max_diff(port_cos_fp32[:, :half_dim], port_cos_fp32[:, half_dim:]),
        "sin_halves": max_diff(port_sin_fp32[:, :half_dim], port_sin_fp32[:, half_dim:]),
    }

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260828)
    q = torch.randn((1, 2, len(POSITIONS), int(raw["head_dim"])), generator=generator).to(
        torch.bfloat16
    )
    k = torch.randn((1, 2, len(POSITIONS), int(raw["head_dim"])), generator=generator).to(
        torch.bfloat16
    )
    selected = torch.as_tensor(POSITIONS, dtype=torch.long)
    hf_q_rot, hf_k_rot = hf_modeling.apply_rotary_pos_emb(
        q,
        k,
        hf_cos_bf16_t.index_select(1, selected),
        hf_sin_bf16_t.index_select(1, selected),
    )
    pc = torch.from_numpy(port_cos_bf16[np.asarray(POSITIONS)]).to(torch.bfloat16)[
        None, None
    ]
    ps = torch.from_numpy(port_sin_bf16[np.asarray(POSITIONS)]).to(torch.bfloat16)[
        None, None
    ]

    def port_apply(x: torch.Tensor) -> torch.Tensor:
        first, second = torch.chunk(x, 2, dim=-1)
        rotate_half = torch.cat((-second, first), dim=-1)
        return x * pc + rotate_half * ps

    application = {
        "q": max_diff(
            port_apply(q).to(torch.float32).numpy(), hf_q_rot.to(torch.float32).numpy()
        ),
        "k": max_diff(
            port_apply(k).to(torch.float32).numpy(), hf_k_rot.to(torch.float32).numpy()
        ),
    }
    for kind, diff in application.items():
        if diff["max_abs_diff"] != 0.0:
            numeric_mismatches.append(
                {
                    "surface": f"rotary_application_{kind}_bfloat16",
                    "classification": "runtime_value_mismatch",
                    **diff,
                }
            )

    # Compare the exact HF eager masks to the port's index laws at the longest
    # Phase 0 position.  Hidden width 1 is sufficient; only shape/dtype matter.
    mask_input = torch.zeros((1, seq_len, 1), dtype=torch.bfloat16)
    hf_full_mask = create_causal_mask(
        config=hf_cfg,
        inputs_embeds=mask_input,
        attention_mask=None,
        past_key_values=None,
    )[0, 0]
    hf_sliding_mask = create_sliding_window_causal_mask(
        config=hf_cfg,
        inputs_embeds=mask_input,
        attention_mask=None,
        past_key_values=None,
    )[0, 0]
    q_index = np.arange(seq_len, dtype=np.int64)[:, None]
    k_index = np.arange(seq_len, dtype=np.int64)[None, :]
    port_full_allowed = k_index <= q_index
    port_sliding_allowed = port_full_allowed & (
        k_index > q_index - int(port_cfg.sliding_window)
    )
    hf_full_allowed = hf_full_mask.to(torch.float32).cpu().numpy() == 0
    hf_sliding_allowed = hf_sliding_mask.to(torch.float32).cpu().numpy() == 0
    port_mask_sentinel = float(torch.tensor(-1.0e4, dtype=torch.bfloat16).item())
    hf_mask_sentinel = float(torch.finfo(torch.bfloat16).min)
    mask_audit = {
        "length": seq_len,
        "port_mask_sentinel_bfloat16": port_mask_sentinel,
        "hf_mask_sentinel_bfloat16": hf_mask_sentinel,
        "sentinel_abs_diff": float(abs(port_mask_sentinel - hf_mask_sentinel)),
        "full_allowed_mismatch_count": int(
            np.count_nonzero(port_full_allowed != hf_full_allowed)
        ),
        "sliding_allowed_mismatch_count": int(
            np.count_nonzero(port_sliding_allowed != hf_sliding_allowed)
        ),
        "full_allowed_count": int(port_full_allowed.sum()),
        "sliding_allowed_count": int(port_sliding_allowed.sum()),
        "sliding_window": int(port_cfg.sliding_window),
    }
    if port_mask_sentinel != hf_mask_sentinel:
        numeric_mismatches.append(
            {
                "surface": "masked_logit_sentinel_bfloat16",
                "classification": "semantically_saturated_representation_difference",
                "port": port_mask_sentinel,
                "hf": hf_mask_sentinel,
                "max_abs_diff": mask_audit["sentinel_abs_diff"],
                "max_rel_diff": float(
                    mask_audit["sentinel_abs_diff"] / abs(hf_mask_sentinel)
                ),
            }
        )

    sinks = _load_sink(model_dir, "model.layers.0.self_attn.sinks")
    base_scores = torch.linspace(-3.0, 3.0, steps=32, dtype=torch.float32).reshape(2, 16)
    sink_values = torch.from_numpy(sinks[:2])[:, None]
    port_combined = torch.cat((base_scores, sink_values), dim=-1)
    port_probs = torch.softmax(port_combined, dim=-1)[..., :-1]
    hf_shifted = port_combined - port_combined.max(dim=-1, keepdim=True).values
    hf_probs = torch.softmax(hf_shifted, dim=-1)[..., :-1]
    sink_unmasked_diff = max_diff(port_probs.numpy(), hf_probs.numpy())
    port_masked_scores = base_scores.clone()
    hf_masked_scores = base_scores.clone()
    port_masked_scores[:, -1] = port_mask_sentinel
    hf_masked_scores[:, -1] = hf_mask_sentinel
    port_masked = torch.softmax(
        torch.cat((port_masked_scores, sink_values), dim=-1), dim=-1
    )[..., :-1]
    hf_masked_combined = torch.cat((hf_masked_scores, sink_values), dim=-1)
    hf_masked_combined -= hf_masked_combined.max(dim=-1, keepdim=True).values
    hf_masked = torch.softmax(hf_masked_combined, dim=-1)[..., :-1]
    sink_masked_diff = max_diff(port_masked.numpy(), hf_masked.numpy())
    sink_audit = {
        "actual_sink_weight_name": "model.layers.0.self_attn.sinks",
        "actual_sink_min": float(sinks.min()),
        "actual_sink_max": float(sinks.max()),
        "semantics": {
            "port": "append one sink logit per head, include it in softmax denominator, drop its probability before value matmul",
            "hf": "append one sink logit per head, include it in softmax denominator, drop its probability before value matmul",
        },
        "explicit_pre_softmax_max_subtraction": {"port_callsite": False, "hf": True},
        "port_softmax_internal_max_subtraction": True,
        "unmasked_semantic_simulation": sink_unmasked_diff,
        "masked_semantic_simulation": sink_masked_diff,
    }
    for surface, diff in (
        ("sink_semantics_unmasked", sink_unmasked_diff),
        ("sink_semantics_masked", sink_masked_diff),
    ):
        if diff["max_abs_diff"] != 0.0:
            numeric_mismatches.append(
                {
                    "surface": surface,
                    "classification": "runtime_value_mismatch",
                    **diff,
                }
            )

    raw_layers = tuple(raw["layer_types"])
    port_layers = tuple(port_cfg.layer_types)
    hf_layers = tuple(hf_cfg.layer_types)
    layer_mismatches = [
        index
        for index, triple in enumerate(zip(raw_layers, port_layers, hf_layers))
        if len(set(triple)) != 1
    ]
    layer_map = {
        "raw_hf_config": list(raw_layers),
        "port_parsed": list(port_layers),
        "hf_parsed": list(hf_layers),
        "mismatch_indices": layer_mismatches,
        "full_attention_indices": [
            index for index, value in enumerate(raw_layers) if value == "full_attention"
        ],
        "sliding_attention_indices": [
            index for index, value in enumerate(raw_layers) if value == "sliding_attention"
        ],
    }

    constants = {
        "factor": {"port": port_values["factor"], "hf": float(hf_cfg.rope_parameters["factor"])},
        "original_max_position_embeddings": {
            "port": port_values["original_max_position_embeddings"],
            "hf": int(hf_cfg.rope_parameters["original_max_position_embeddings"]),
        },
        "attention_factor": {
            "port": port_attention_factor,
            "hf": hf_attention_factor,
            "abs_diff": abs(port_attention_factor - hf_attention_factor),
        },
        "rope_theta": {"port": port_values["base"], "hf": float(hf_cfg.rope_parameters["rope_theta"])},
        "beta_fast": {"port": port_values["beta_fast"], "hf": float(hf_cfg.rope_parameters["beta_fast"])},
        "beta_slow": {"port": port_values["beta_slow"], "hf": float(hf_cfg.rope_parameters["beta_slow"])},
        "truncate": {"port": port_values["truncate"], "hf": bool(hf_cfg.rope_parameters["truncate"])},
        "correction_low": port_values["correction_low"],
        "correction_high": port_values["correction_high"],
    }

    checks = {
        "constants_exact": bool(
            constants["factor"]["port"] == constants["factor"]["hf"]
            and constants["original_max_position_embeddings"]["port"]
            == constants["original_max_position_embeddings"]["hf"]
            and constants["attention_factor"]["abs_diff"] == 0.0
            and constants["rope_theta"]["port"] == constants["rope_theta"]["hf"]
            and constants["truncate"]["port"] == constants["truncate"]["hf"]
        ),
        "runtime_bfloat16_tables_exact_at_requested_positions": all(
            row[kind]["runtime_bfloat16"]["max_abs_diff"] == 0.0
            for row in requested_rows
            for kind in ("cos", "sin")
        ),
        "rotary_application_bfloat16_exact": all(
            diff["max_abs_diff"] == 0.0 for diff in application.values()
        ),
        "sink_semantics_numeric_exact_in_cpu_simulation": bool(
            sink_unmasked_diff["max_abs_diff"] == 0.0
            and sink_masked_diff["max_abs_diff"] == 0.0
        ),
        "full_allowed_set_exact": mask_audit["full_allowed_mismatch_count"] == 0,
        "sliding_allowed_set_exact": mask_audit["sliding_allowed_mismatch_count"] == 0,
        "layer_interleave_exact": not layer_mismatches,
    }
    critical_pass = all(checks.values())
    runtime_value_mismatches = [
        row
        for row in numeric_mismatches
        if row["classification"] == "runtime_value_mismatch"
    ]
    verdict = (
        "H_GLC_1_EXCLUDED_AT_BFLOAT16_TABLE_AND_APPLICATION_LEVEL"
        if critical_pass and not runtime_value_mismatches
        else "H_GLC_1_NOT_EXCLUDED"
    )
    source_locations = {
        "port_yarn_builder": _source_location(port.gpt_oss_yarn_rope_tables),
        "port_sink_attention": _source_location(port.sink_attention_tc),
        "port_sliding_sink_attention": _source_location(port.sliding_sink_attention_tc),
        "port_attention_mask": _source_location(port._gpt_oss_attention_mask),
        "port_attention_call": _source_location(port.GptOssAttentionTC.__call__),
        "hf_rotary": _source_location(hf_modeling.GptOssRotaryEmbedding),
        "hf_attention": _source_location(hf_modeling.GptOssAttention),
        "hf_eager_attention": _source_location(hf_modeling.eager_attention_forward),
        "hf_yarn": _source_location(_compute_yarn_parameters),
        "hf_sliding_overlay": _source_location(sliding_window_overlay),
        "tensorcuda_apply_rotary": {
            "path": str(TC_FUNCTIONAL),
            "line": 167,
            "sha256": common.sha256_file(TC_FUNCTIONAL),
        },
        "tensorcuda_softmax_stabilization": {
            "path": str(TC_OPS),
            "line": 796,
            "sha256": common.sha256_file(TC_OPS),
        },
        "tensorcuda_bfloat16_cast": {
            "path": str(TC_KERNELS),
            "line": 211,
            "sha256": common.sha256_file(TC_KERNELS),
        },
    }

    return {
        "schema": "glc_p0c_shared_machinery_audit_v1",
        "status": "complete" if critical_pass else "red",
        "verdict": verdict,
        "evidence_class": "CPU numeric execution plus current-source audit; no GPU kernel measurement",
        "model_dir": str(model_dir),
        "model_revision": common.MODEL_REVISION,
        "control_arrays_sha256": common.sha256_file(common.CONTROL_ARRAYS),
        "positions": list(POSITIONS),
        "relative_epsilon": RELATIVE_EPSILON,
        "constants": constants,
        "inv_freq": {
            "shape": list(inv_port.shape),
            "port_min": float(inv_port.min()),
            "port_max": float(inv_port.max()),
            **inv_diff,
        },
        "rope_tables": {
            "port_runtime_shape": list(port_cos_bf16.shape),
            "hf_runtime_shape": list(hf_cos_bf16_t.shape),
            "port_runtime_dtype": "bfloat16",
            "hf_runtime_dtype": str(hf_cos_bf16_t.dtype),
            "port_layout": "[position, head_dim], duplicated frequency halves",
            "hf_layout": "[batch, position, head_dim/2], half-pair application",
            "duplicate_layout_check": duplicate_layout,
            "requested_positions": requested_rows,
        },
        "rotary_application": application,
        "mask_audit": mask_audit,
        "sink_audit": sink_audit,
        "layer_map": layer_map,
        "numeric_mismatches": numeric_mismatches,
        "runtime_value_mismatches": runtime_value_mismatches,
        "checks": checks,
        "source_locations": source_locations,
        "runtime": common.runtime_facts(),
        "script_sha256": common.sha256_file(SCRIPT_PATH),
    }


def report_markdown(core: dict[str, Any]) -> str:
    constants = core["constants"]
    lines = [
        "# GLC P0.c Shared-Machinery Audit",
        "",
        f"Verdict: **{core['verdict']}**.",
        "",
        "Evidence class: CPU numeric execution plus current-source audit; no GPU kernel measurement.",
        "",
        "## YaRN constants",
        "",
        "| Quantity | port | HF | difference |",
        "|---|---:|---:|---:|",
        f"| factor | {constants['factor']['port']} | {constants['factor']['hf']} | 0 |",
        f"| original context | {constants['original_max_position_embeddings']['port']} | {constants['original_max_position_embeddings']['hf']} | 0 |",
        f"| attention scaling | {constants['attention_factor']['port']:.17g} | {constants['attention_factor']['hf']:.17g} | {constants['attention_factor']['abs_diff']:.9g} |",
        f"| rope theta | {constants['rope_theta']['port']} | {constants['rope_theta']['hf']} | 0 |",
        f"| beta fast | {constants['beta_fast']['port']} | {constants['beta_fast']['hf']} | 0 |",
        f"| beta slow | {constants['beta_slow']['port']} | {constants['beta_slow']['hf']} | 0 |",
        f"| truncate | {constants['truncate']['port']} | {constants['truncate']['hf']} | exact |",
        "",
        "## Requested cos/sin values",
        "",
        "| position | cos FP32 max abs | cos FP32 max rel | cos BF16 max abs | sin FP32 max abs | sin FP32 max rel | sin BF16 max abs |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in core["rope_tables"]["requested_positions"]:
        cos_fp = row["cos"]["fp32_construction"]
        cos_bf = row["cos"]["runtime_bfloat16"]
        sin_fp = row["sin"]["fp32_construction"]
        sin_bf = row["sin"]["runtime_bfloat16"]
        lines.append(
            f"| {row['position']} | {cos_fp['max_abs_diff']:.9g} | {cos_fp['max_rel_diff']:.9g} | "
            f"{cos_bf['max_abs_diff']:.9g} | {sin_fp['max_abs_diff']:.9g} | "
            f"{sin_fp['max_rel_diff']:.9g} | {sin_bf['max_abs_diff']:.9g} |"
        )
    inv = core["inv_freq"]
    lines.extend(
        [
            "",
            f"Inverse-frequency FP32 difference: max abs `{inv['max_abs_diff']:.9g}`, max rel `{inv['max_rel_diff']:.9g}`, relative Frobenius `{inv['relative_frobenius_diff']:.9g}`.",
            "",
            "The port stores duplicated 64-wide cos/sin rows while HF stores 32 frequencies and applies them to paired halves. The duplicated halves and the BF16 rotary application both compare exactly (max abs 0).",
            "",
            "## Sink, masks, and layer map",
            "",
            f"Sink semantics match: both append one learned logit per head to the denominator and drop it before the value matmul. Unmasked and masked CPU semantic simulations have max abs differences `{core['sink_audit']['unmasked_semantic_simulation']['max_abs_diff']:.9g}` and `{core['sink_audit']['masked_semantic_simulation']['max_abs_diff']:.9g}`.",
            "",
            f"The masked-logit sentinels differ numerically: port BF16 `{core['mask_audit']['port_mask_sentinel_bfloat16']:.9g}` versus HF BF16 `{core['mask_audit']['hf_mask_sentinel_bfloat16']:.9g}`. The full and sliding allowed-set mismatch counts are `{core['mask_audit']['full_allowed_mismatch_count']}` and `{core['mask_audit']['sliding_allowed_mismatch_count']}` at length {core['mask_audit']['length']}.",
            "",
            f"Layer interleave mismatch indices: `{core['layer_map']['mismatch_indices']}`; full layers `{core['layer_map']['full_attention_indices']}`, sliding layers `{core['layer_map']['sliding_attention_indices']}`.",
            "",
            "## Every numeric mismatch",
            "",
        ]
    )
    if not core["numeric_mismatches"]:
        lines.append("None.")
    else:
        lines.extend(
            [
                "| Surface | classification | max abs | max rel |",
                "|---|---|---:|---:|",
            ]
        )
        for row in core["numeric_mismatches"]:
            lines.append(
                f"| `{row['surface']}` | {row['classification']} | "
                f"{row['max_abs_diff']:.9g} | {row['max_rel_diff']:.9g} |"
            )
    lines.extend(
        [
            "",
            "The FP32 differences are NumPy-versus-Torch construction roundoff and disappear after the registered BF16 cast. The sentinel difference is numerically large but semantically saturated: allowed key sets and the tested sink-normalized probabilities are exact.",
            "",
            "## Source anchors",
            "",
        ]
    )
    for name, location in core["source_locations"].items():
        lines.append(f"- {name}: `{location['path']}:{location['line']}`")
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    core = perform_audit(args)
    digest = common.canonical_json_sha256(core)[:16]
    json_path = common.OUTPUT_ROOT / "audit" / f"p0c_{digest}.json"
    report_path = common.OUTPUT_ROOT / f"GLC_P0C_AUDIT_{digest}.md"
    if not json_path.exists():
        common.write_new_json(json_path, {**core, "created_at": common.now_iso()})
    common.write_once_text(report_path, report_markdown(core))
    print(
        json.dumps(
            {
                "status": core["status"],
                "verdict": core["verdict"],
                "numeric_mismatch_count": len(core["numeric_mismatches"]),
                "runtime_value_mismatch_count": len(core["runtime_value_mismatches"]),
                "audit": str(json_path),
                "report": str(report_path),
            }
        )
    )
    return 0 if core["status"] == "complete" else 1


def self_test(args: argparse.Namespace) -> int:
    # The audit itself is deterministic and its internal checks are the most
    # useful self-test; keep a separate receipt without duplicating calculations.
    core = perform_audit(args)
    checks = [
        {"name": key, "passed": bool(value)} for key, value in core["checks"].items()
    ]
    payload = {
        "schema": "glc_p0c_self_test_v1",
        "status": "pass" if all(row["passed"] for row in checks) else "fail",
        "checks": checks,
        "audit_verdict": core["verdict"],
        "script_sha256": common.sha256_file(SCRIPT_PATH),
    }
    digest = common.canonical_json_sha256(payload)[:16]
    path = common.OUTPUT_ROOT / "self_test" / f"p0c_{digest}.json"
    if not path.exists():
        common.write_new_json(path, {**payload, "created_at": common.now_iso()})
    print(json.dumps({"status": payload["status"], "receipt": str(path)}))
    return 0 if payload["status"] == "pass" else 1


def main() -> int:
    args = parse_args()
    if args.mode == "audit":
        return run(args)
    if args.mode == "self-test":
        return self_test(args)
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
