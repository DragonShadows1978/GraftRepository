"""GPT-OSS-20B TensorCUDA scaffold.

This module holds the Phase 3A correctness primitives for the GPT-OSS port.
It is deliberately smaller than a full loader: the first job is to lock down
the GPT-OSS-specific math before claiming model support.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.mistral7b_tc import BlockTC, QuantLinearTC, _repeat_kv, tc


MODEL_ID = "openai/gpt-oss-20b"
DEFAULT_GROUP_SIZE = 64

FP4_VALUES = np.asarray(
    [
        +0.0,
        +0.5,
        +1.0,
        +1.5,
        +2.0,
        +3.0,
        +4.0,
        +6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)


@dataclass
class GptOss20BConfig:
    vocab_size: int = 201088
    hidden_dim: int = 2880
    intermediate_dim: int = 2880
    num_layers: int = 24
    num_heads: int = 64
    num_kv_heads: int = 8
    head_dim: int = 64
    num_local_experts: int = 32
    num_experts_per_tok: int = 4
    sliding_window: int = 128
    max_position_embeddings: int = 131072
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling: dict[str, Any] | None = None
    rms_norm_eps: float = 1e-5
    swiglu_limit: float = 7.0
    swiglu_alpha: float = 1.702
    attention_bias: bool = True
    tie_word_embeddings: bool = False
    eos_token_id: int | list[int] | None = 200002
    pad_token_id: int | None = 199999
    layer_types: tuple[str, ...] = ()
    group_size: int = DEFAULT_GROUP_SIZE
    model_dir: str | None = None
    repository: str | None = None
    revision: str | None = None

    @classmethod
    def from_model_dir(cls, model_dir: str | os.PathLike[str]) -> "GptOss20BConfig":
        root = Path(model_dir).expanduser().resolve()
        with (root / "config.json").open(encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("model_type") != "gpt_oss":
            raise ValueError(f"expected model_type gpt_oss, got {data.get('model_type')!r}")
        parts = root.parts
        repo = revision = None
        if len(parts) >= 3 and parts[-2] == "snapshots":
            repo_dir = parts[-3]
            repo = (
                repo_dir[len("models--"):].replace("--", "/")
                if repo_dir.startswith("models--")
                else repo_dir
            )
            revision = parts[-1]
        return cls(
            vocab_size=int(data["vocab_size"]),
            hidden_dim=int(data["hidden_size"]),
            intermediate_dim=int(data["intermediate_size"]),
            num_layers=int(data["num_hidden_layers"]),
            num_heads=int(data["num_attention_heads"]),
            num_kv_heads=int(data["num_key_value_heads"]),
            head_dim=int(data["head_dim"]),
            num_local_experts=int(data["num_local_experts"]),
            num_experts_per_tok=int(
                data.get("num_experts_per_tok", data.get("experts_per_token", 4))
            ),
            sliding_window=int(data["sliding_window"]),
            max_position_embeddings=int(data["max_position_embeddings"]),
            initial_context_length=int(data.get("initial_context_length", 4096)),
            rope_theta=float(data.get("rope_theta", cls.rope_theta)),
            rope_scaling=dict(data.get("rope_scaling") or {}),
            rms_norm_eps=float(data.get("rms_norm_eps", cls.rms_norm_eps)),
            swiglu_limit=float(data.get("swiglu_limit", cls.swiglu_limit)),
            attention_bias=bool(data.get("attention_bias", True)),
            tie_word_embeddings=bool(data.get("tie_word_embeddings", False)),
            eos_token_id=data.get("eos_token_id"),
            pad_token_id=data.get("pad_token_id"),
            layer_types=tuple(data.get("layer_types") or ()),
            model_dir=str(root),
            repository=repo,
            revision=revision,
        )

    @property
    def num_heads_per_kv(self) -> int:
        return self.num_heads // self.num_kv_heads

    def is_full_attention(self, layer_idx: int) -> bool:
        return self.layer_types[int(layer_idx)] == "full_attention"

    def is_sliding_attention(self, layer_idx: int) -> bool:
        return self.layer_types[int(layer_idx)] == "sliding_attention"

    def full_attention_indices(self) -> list[int]:
        return [i for i in range(self.num_layers) if self.is_full_attention(i)]

    def sliding_attention_indices(self) -> list[int]:
        return [i for i in range(self.num_layers) if self.is_sliding_attention(i)]


class BiasedQuantLinearTC:
    """Quantized linear with an optional bias add.

    GPT-OSS attention projections are biased, while the shared QuantLinearTC is
    intentionally weight-only. This wrapper keeps the shared class unchanged.
    """

    def __init__(
        self,
        weight_fp32: np.ndarray,
        bias_fp32: np.ndarray | None = None,
        group_size: int = DEFAULT_GROUP_SIZE,
        bits: int | None = None,
    ):
        self.linear = QuantLinearTC(
            np.ascontiguousarray(weight_fp32), group_size=group_size, bits=bits
        )
        self.out_features = self.linear.out_features
        self.in_features = self.linear.in_features
        self.group_size = group_size
        self.bits = self.linear.bits
        self.bias = None
        self._bias_bytes = 0
        if bias_fp32 is not None:
            if int(np.prod(bias_fp32.shape)) != self.out_features:
                raise ValueError(
                    f"bias has {bias_fp32.shape}, expected ({self.out_features},)"
                )
            dt = BlockTC.COMPUTE_DTYPE
            self.bias = tc.tensor(np.ascontiguousarray(bias_fp32.astype(np.float32))).astype(dt)
            self._bias_bytes = self.out_features * 2

    @property
    def packed(self):
        return self.linear.packed

    @property
    def scales(self):
        return self.linear.scales

    @property
    def zeros(self):
        return self.linear.zeros

    def __call__(self, x):
        y = self.linear(x)
        if self.bias is None:
            return y
        bias = self.bias if self.bias.dtype == y.dtype else self.bias.astype(y.dtype)
        return y + bias

    def vram_bytes(self) -> int:
        return self.linear.vram_bytes() + self._bias_bytes


def mxfp4_dequantize_blocks_np(
    blocks: np.ndarray,
    scales: np.ndarray,
    *,
    out_dtype=np.float32,
) -> np.ndarray:
    """Exact HF-compatible MXFP4 block dequantization.

    Input shapes match GPT-OSS safetensors:
    `blocks == [experts, out_features, groups, 16]` and
    `scales == [experts, out_features, groups]`.

    HF expands each 16-byte group to 32 values, applies E8M0-style exponent
    scales (`scale - 127`), flattens groups, then transposes dimensions 1 and 2
    to produce `[experts, in_features, out_features]`.
    """

    b = np.asarray(blocks, dtype=np.uint8)
    s = np.asarray(scales, dtype=np.uint8)
    if b.ndim != 4:
        raise ValueError(f"blocks must be rank-4, got {b.shape}")
    if s.shape != b.shape[:-1]:
        raise ValueError(f"scales shape {s.shape} does not match blocks prefix {b.shape[:-1]}")

    lo = b & 0x0F
    hi = b >> 4
    vals = np.empty(b.shape[:-1] + (b.shape[-1] * 2,), dtype=np.float32)
    vals[..., 0::2] = FP4_VALUES[lo]
    vals[..., 1::2] = FP4_VALUES[hi]
    exp = s.astype(np.int32) - 127
    vals = np.ldexp(vals, exp[..., None])
    experts, out_features, groups, expanded = vals.shape
    dense = vals.reshape(experts, out_features, groups * expanded)
    return np.swapaxes(dense, 1, 2).astype(out_dtype, copy=False)


def gpt_oss_expert_activation_np(
    gate_up: np.ndarray,
    *,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> np.ndarray:
    """Reference GPT-OSS expert activation for tests and diagnostics."""

    x = np.asarray(gate_up, dtype=np.float32)
    gate = x[..., 0::2]
    up = x[..., 1::2]
    gate = np.minimum(gate, float(limit))
    up = np.clip(up, -float(limit), float(limit))
    sigmoid = 1.0 / (1.0 + np.exp(-(gate * float(alpha))))
    return ((up + 1.0) * gate * sigmoid).astype(np.float32, copy=False)


def gpt_oss_expert_activation_tc(gate_up, *, alpha: float = 1.702, limit: float = 7.0):
    """TensorCUDA GPT-OSS expert activation.

    TensorCUDA has no strided even/odd slice helper, so the input is reshaped to
    `(..., hidden, 2)` and split along the final dimension.
    """

    last = gate_up.shape[-1]
    if last % 2:
        raise ValueError(f"gate_up last dimension must be even, got {last}")
    paired = gate_up.reshape(list(gate_up.shape[:-1]) + [last // 2, 2])
    gate = paired.slice(-1, 0, 1).reshape(list(gate_up.shape[:-1]) + [last // 2])
    up = paired.slice(-1, 1, 1).reshape(list(gate_up.shape[:-1]) + [last // 2])
    gate = gate.clamp(-1.0e30, float(limit))
    up = up.clamp(-float(limit), float(limit))
    return (up + 1.0) * gate * (gate * float(alpha)).sigmoid()


def sink_attention_tc(
    query,
    key,
    value,
    sinks,
    *,
    scale: float,
    attention_mask=None,
    num_heads_per_kv: int = 1,
):
    """GPT-OSS sink-aware attention for the standard path.

    The sink logit is included in the softmax denominator but contributes no
    value vector; after softmax, the sink column is dropped.
    """

    key_h = _repeat_kv(key, num_heads_per_kv)
    value_h = _repeat_kv(value, num_heads_per_kv)
    scores = tc.matmul(query, key_h, alpha=scale, trans_b=True)
    if attention_mask is not None:
        scores = scores + attention_mask
    B, H, L, S = scores.shape
    sink_logits = sinks.reshape([1, H, 1, 1]).expand([B, H, L, 1])
    combined = tc.cat([scores, sink_logits], dim=-1)
    probs = combined.softmax(-1)
    weights = probs.slice(-1, 0, S)
    return tc.matmul(weights, value_h)
