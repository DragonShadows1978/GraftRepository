"""GPT-OSS-20B TensorCUDA scaffold.

This module holds the Phase 3A correctness primitives for the GPT-OSS port.
It is deliberately smaller than a full loader: the first job is to lock down
the GPT-OSS-specific math before claiming model support.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.mistral7b_tc import BlockTC, F, QuantLinearTC, RMSNormTC, _repeat_kv, tc


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


def sink_apa_blend_attention_tc(
    query,
    key,
    key_quant,
    value,
    sinks,
    *,
    scale: float,
    zthr: float,
    attention_mask=None,
    num_heads_per_kv: int = 1,
    attn_block: int = 1024,
):
    """GPT-OSS sink-aware APA blend attention.

    `key_quant` is the APA bulk key approximation. Selection is done by
    TensorCUDA's APA blend kernel, then GPT-OSS sink logits are included in the
    softmax denominator while no sink value column is returned.
    """

    if not hasattr(tc, "apa_blend_softmax_sink"):
        raise RuntimeError(
            "sink-aware GPT-OSS APA requires Project-Tensor "
            "tc.apa_blend_softmax_sink"
        )
    key_h = _repeat_kv(key, num_heads_per_kv)
    keyq_h = _repeat_kv(key_quant, num_heads_per_kv)
    value_h = _repeat_kv(value, num_heads_per_kv)
    L = query.shape[2]
    outs = []
    blk = max(1, int(attn_block))
    for i in range(0, L, blk):
        e = min(i + blk, L)
        Qi = query.slice(2, i, e - i)
        bulk = tc.matmul(Qi, keyq_h, alpha=scale, trans_b=True)
        rank = tc.matmul(Qi, key_h, alpha=scale, trans_b=True)
        if attention_mask is not None:
            mask_i = attention_mask.slice(2, i, e - i)
            bulk = bulk + mask_i
            rank = rank + mask_i
        weights = tc.apa_blend_softmax_sink(bulk, rank, sinks, float(zthr))
        outs.append(tc.matmul(weights, value_h))
    return tc.cat(outs, dim=2)


def build_safetensors_map(model_dir: str | os.PathLike[str]) -> dict[str, str]:
    """Map tensor name to safetensors shard path."""

    from safetensors import safe_open

    root = Path(model_dir).expanduser().resolve()
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards found in {root}")
    where: dict[str, str] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as fh:
            for key in fh.keys():
                where[key] = str(shard)
    return where


def load_tensor_np(where: dict[str, str], name: str, *, dtype=np.float32) -> np.ndarray:
    """Load a tensor as a contiguous NumPy array."""

    import torch
    from safetensors import safe_open

    with safe_open(where[name], framework="pt", device="cpu") as fh:
        t = fh.get_tensor(name)
    if dtype is np.float32:
        t = t.to(torch.float32)
    arr = t.numpy()
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return np.ascontiguousarray(arr)


class GptOssRowEmbedding:
    """Host row-sliced embedding gather for GPT-OSS.

    The full BF16 embedding is about 1.08 GiB. For loader smokes and GRM
    turn-wise operation, gather only requested rows on CPU and upload the small
    `(B, L, H)` result.
    """

    def __init__(self, where: dict[str, str], name: str = "model.embed_tokens.weight"):
        self.where = where
        self.name = name

    def __call__(self, ids_np: np.ndarray):
        import torch
        from safetensors import safe_open

        ids = np.asarray(ids_np, dtype=np.int64)
        flat = ids.reshape(-1).tolist()
        with safe_open(self.where[self.name], framework="pt", device="cpu") as fh:
            rows = fh.get_slice(self.name)[flat].to(torch.float32).numpy()
        rows = np.ascontiguousarray(rows.reshape(ids.shape + (rows.shape[-1],)))
        dt = BlockTC.COMPUTE_DTYPE
        return tc.tensor(rows, dtype="float32").astype(dt)


def _yarn_get_mscale(scale: float, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def gpt_oss_yarn_rope_tables(cfg: GptOss20BConfig, seq_len: int):
    """Build HF-compatible GPT-OSS YARN RoPE tables in TensorCUDA layout."""

    rope = dict(cfg.rope_scaling or {})
    base = float(rope.get("rope_theta", cfg.rope_theta))
    dim = int(cfg.head_dim)
    factor = rope.get("factor")
    original = int(rope.get("original_max_position_embeddings", cfg.initial_context_length))
    if factor is None:
        factor = cfg.max_position_embeddings / original
    factor = float(factor)

    attention_factor = rope.get("attention_factor")
    if attention_factor is None:
        mscale = rope.get("mscale")
        mscale_all_dim = rope.get("mscale_all_dim")
        if mscale and mscale_all_dim:
            attention_factor = _yarn_get_mscale(factor, float(mscale)) / _yarn_get_mscale(
                factor, float(mscale_all_dim)
            )
        else:
            attention_factor = _yarn_get_mscale(factor)
    attention_factor = float(attention_factor)

    beta_fast = float(rope.get("beta_fast") or 32.0)
    beta_slow = float(rope.get("beta_slow") or 1.0)
    truncate = bool(rope.get("truncate", True))

    def correction_dim(num_rotations):
        return (dim * math.log(original / (num_rotations * 2 * math.pi))) / (
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
    ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - low) / (high - low), 0.0, 1.0)
    extra_factor = 1.0 - ramp
    inv = inv_interp * (1.0 - extra_factor) + inv_extra * extra_factor

    pos = np.arange(int(seq_len), dtype=np.float32)[:, None] * inv[None, :]
    emb = np.concatenate([pos, pos], axis=-1)
    c = (np.cos(emb) * attention_factor).astype(np.float32)
    s = (np.sin(emb) * attention_factor).astype(np.float32)
    dt = BlockTC.COMPUTE_DTYPE
    return tc.tensor(c, dtype="float32").astype(dt), tc.tensor(s, dtype="float32").astype(dt)


def _gpt_oss_attention_mask(L: int, S: int, *, sliding_window: int | None, dtype: str):
    q_abs = np.arange(S - L, S, dtype=np.int64)[:, None]
    k_abs = np.arange(S, dtype=np.int64)[None, :]
    allowed = k_abs <= q_abs
    if sliding_window is not None:
        allowed &= k_abs > (q_abs - int(sliding_window))
    mask = np.where(allowed, 0.0, -1.0e4).astype(np.float32)
    return tc.tensor(mask.reshape(1, 1, L, S), dtype="float32").astype(dtype)


class GptOssAttentionTC:
    def __init__(self, cfg: GptOss20BConfig, layer_idx: int):
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
        self.layer_type = cfg.layer_types[self.layer_idx]
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.num_heads_per_kv = cfg.num_heads_per_kv
        self.scaling = cfg.head_dim ** -0.5
        self.sliding_window = cfg.sliding_window if self.layer_type == "sliding_attention" else None
        self.q_proj = self.k_proj = self.v_proj = self.o_proj = None
        self.sinks = None
        self.attention_mode = "standard"
        self.refine_percentile = 0.15
        self.bulk_bits = 8
        self.attn_block = 1024

    @classmethod
    def from_safetensors(
        cls,
        cfg: GptOss20BConfig,
        where: dict[str, str],
        layer_idx: int,
    ) -> "GptOssAttentionTC":
        obj = cls(cfg, layer_idx)
        p = f"model.layers.{layer_idx}.self_attn"
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            w = load_tensor_np(where, f"{p}.{attr}.weight")
            b = load_tensor_np(where, f"{p}.{attr}.bias")
            setattr(obj, attr, BiasedQuantLinearTC(w, b, group_size=cfg.group_size))
            del w, b
        obj.sinks = tc.tensor(load_tensor_np(where, f"{p}.sinks"), dtype="float32").astype(
            BlockTC.COMPUTE_DTYPE
        )
        return obj

    def __call__(self, x, cos, sin, position_offset: int = 0, kv_cache=None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape([B, L, self.num_heads, self.head_dim]).transpose(1, 2)
        k = self.k_proj(x).reshape([B, L, self.num_kv_heads, self.head_dim]).transpose(1, 2)
        v = self.v_proj(x).reshape([B, L, self.num_kv_heads, self.head_dim]).transpose(1, 2)

        cseg = cos.slice(0, position_offset, L)
        sseg = sin.slice(0, position_offset, L)
        q = F.apply_rotary(q, cseg, sseg)
        k = F.apply_rotary(k, cseg, sseg)

        if kv_cache is not None:
            k = tc.cat([kv_cache[0], k], dim=2)
            v = tc.cat([kv_cache[1], v], dim=2)
        S = k.shape[2]
        mask = _gpt_oss_attention_mask(
            L,
            S,
            sliding_window=self.sliding_window,
            dtype=q.dtype,
        )
        if self.attention_mode == "apa_selective":
            from tensor_cuda.quant import _norm_ppf, _quantize_keys, _tables

            dev = q.device.split(":")[0]
            R, CB, BND = _tables(
                self.head_dim, self.bulk_bits, self.num_kv_heads, True, dev
            )
            kq = _quantize_keys(k, R, CB, BND)
            z = _norm_ppf(1.0 - max(0.0, min(1.0, self.refine_percentile)))
            attn = sink_apa_blend_attention_tc(
                q,
                k,
                kq,
                v,
                self.sinks,
                scale=self.scaling,
                zthr=float(z),
                attention_mask=mask,
                num_heads_per_kv=self.num_heads_per_kv,
                attn_block=self.attn_block,
            )
        else:
            attn = sink_attention_tc(
                q,
                k,
                v,
                self.sinks,
                scale=self.scaling,
                attention_mask=mask,
                num_heads_per_kv=self.num_heads_per_kv,
            )
        out = attn.transpose(1, 2).reshape([B, L, self.num_heads * self.head_dim])
        return self.o_proj(out), (k, v)


class GptOssAttentionBlockTC:
    """One GPT-OSS decoder block through attention/residual only.

    The MoE half is intentionally absent here; this is the Phase 3A layer-runtime
    scaffold used before packed MXFP4 expert math is available.
    """

    def __init__(self, cfg: GptOss20BConfig, layer_idx: int):
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
        self.input_layernorm = RMSNormTC(cfg.hidden_dim, cfg.rms_norm_eps)
        self.self_attn = GptOssAttentionTC(cfg, layer_idx)

    @classmethod
    def from_safetensors(
        cls,
        cfg: GptOss20BConfig,
        where: dict[str, str],
        layer_idx: int,
    ) -> "GptOssAttentionBlockTC":
        obj = cls(cfg, layer_idx)
        p = f"model.layers.{layer_idx}"
        obj.input_layernorm.weight = tc.tensor(
            load_tensor_np(where, f"{p}.input_layernorm.weight"), dtype="float32"
        )
        obj.self_attn = GptOssAttentionTC.from_safetensors(cfg, where, layer_idx)
        return obj

    def __call__(self, x, cos, sin, position_offset: int = 0, kv_cache=None):
        dt = BlockTC.COMPUTE_DTYPE
        normed = self.input_layernorm(x)
        normed = normed.half() if dt == "float16" else normed.astype(dt)
        attn, kv = self.self_attn(normed, cos, sin, position_offset, kv_cache)
        return x + attn, kv


class GptOssMoEDiagnosticTC:
    """Selected-expert GPT-OSS MoE diagnostic.

    This class is deliberately not the final memory-viable expert path. It uses
    the real GPT-OSS router and exact MXFP4 dequantization, but only for the
    routed experts of the requested token batch. The packed MXFP4 kernel remains
    the required production route.
    """

    def __init__(
        self,
        cfg: GptOss20BConfig,
        layer_idx: int,
        *,
        expert_mode: str = "dequant",
    ):
        if expert_mode not in {"dequant", "packed_mxfp4", "resident_packed_mxfp4"}:
            raise ValueError(f"unsupported GPT-OSS expert_mode {expert_mode!r}")
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
        self.expert_mode = expert_mode
        self.router_weight = None
        self.router_bias = None
        self.gate_up_blocks = None
        self.gate_up_scales = None
        self.gate_up_bias = None
        self.down_blocks = None
        self.down_scales = None
        self.down_bias = None
        self.gate_up_blocks_tc = None
        self.gate_up_scales_tc = None
        self.gate_up_bias_tc = None
        self.down_blocks_tc = None
        self.down_scales_tc = None
        self.down_bias_tc = None

    @classmethod
    def from_safetensors(
        cls,
        cfg: GptOss20BConfig,
        where: dict[str, str],
        layer_idx: int,
        *,
        expert_mode: str = "dequant",
    ) -> "GptOssMoEDiagnosticTC":
        if expert_mode == "packed_mxfp4" and not hasattr(tc, "mxfp4_linear"):
            raise RuntimeError(
                "packed_mxfp4 expert mode requires Project-Tensor tc.mxfp4_linear"
            )
        if expert_mode == "resident_packed_mxfp4" and not hasattr(
            tc, "mxfp4_linear_expert"
        ):
            raise RuntimeError(
                "resident_packed_mxfp4 expert mode requires "
                "Project-Tensor tc.mxfp4_linear_expert"
            )
        obj = cls(cfg, layer_idx, expert_mode=expert_mode)
        p = f"model.layers.{layer_idx}.mlp"
        obj.router_weight = tc.tensor(
            load_tensor_np(where, f"{p}.router.weight"), dtype="float32"
        )
        obj.router_bias = tc.tensor(
            load_tensor_np(where, f"{p}.router.bias"), dtype="float32"
        )
        ep = f"{p}.experts"
        obj.gate_up_blocks = load_tensor_np(
            where, f"{ep}.gate_up_proj_blocks", dtype=np.uint8
        )
        obj.gate_up_scales = load_tensor_np(
            where, f"{ep}.gate_up_proj_scales", dtype=np.uint8
        )
        obj.gate_up_bias = load_tensor_np(where, f"{ep}.gate_up_proj_bias")
        obj.down_blocks = load_tensor_np(where, f"{ep}.down_proj_blocks", dtype=np.uint8)
        obj.down_scales = load_tensor_np(where, f"{ep}.down_proj_scales", dtype=np.uint8)
        obj.down_bias = load_tensor_np(where, f"{ep}.down_proj_bias")
        if expert_mode == "resident_packed_mxfp4":
            obj.gate_up_blocks_tc = tc.tensor(
                np.ascontiguousarray(obj.gate_up_blocks), dtype="uint8"
            )
            obj.gate_up_scales_tc = tc.tensor(
                np.ascontiguousarray(obj.gate_up_scales), dtype="uint8"
            )
            obj.gate_up_bias_tc = tc.tensor(
                np.ascontiguousarray(obj.gate_up_bias), dtype="float32"
            )
            obj.down_blocks_tc = tc.tensor(
                np.ascontiguousarray(obj.down_blocks), dtype="uint8"
            )
            obj.down_scales_tc = tc.tensor(
                np.ascontiguousarray(obj.down_scales), dtype="uint8"
            )
            obj.down_bias_tc = tc.tensor(
                np.ascontiguousarray(obj.down_bias), dtype="float32"
            )
        return obj

    def _route(self, x_flat):
        logits = tc.matmul(x_flat.float(), self.router_weight, trans_b=True)
        bias = (
            self.router_bias
            if self.router_bias.dtype == logits.dtype
            else self.router_bias.astype(logits.dtype)
        )
        topv, topi = (logits + bias).topk(self.cfg.num_experts_per_tok, True)
        return topv.softmax(-1), topi

    def _selected_expert(self, expert_idx: int):
        e = int(expert_idx)
        gate_w = mxfp4_dequantize_blocks_np(
            self.gate_up_blocks[e : e + 1],
            self.gate_up_scales[e : e + 1],
        )[0]
        down_w = mxfp4_dequantize_blocks_np(
            self.down_blocks[e : e + 1],
            self.down_scales[e : e + 1],
        )[0]
        return (
            gate_w,
            np.ascontiguousarray(self.gate_up_bias[e].astype(np.float32, copy=False)),
            down_w,
            np.ascontiguousarray(self.down_bias[e].astype(np.float32, copy=False)),
        )

    def _selected_expert_packed(self, expert_idx: int):
        e = int(expert_idx)
        return (
            tc.tensor(np.ascontiguousarray(self.gate_up_blocks[e]), dtype="uint8"),
            tc.tensor(np.ascontiguousarray(self.gate_up_scales[e]), dtype="uint8"),
            np.ascontiguousarray(self.gate_up_bias[e].astype(np.float32, copy=False)),
            tc.tensor(np.ascontiguousarray(self.down_blocks[e]), dtype="uint8"),
            tc.tensor(np.ascontiguousarray(self.down_scales[e]), dtype="uint8"),
            np.ascontiguousarray(self.down_bias[e].astype(np.float32, copy=False)),
        )

    @staticmethod
    def _bias_add(y, bias_np: np.ndarray):
        bias = (
            tc.tensor(bias_np, dtype="float32")
            if isinstance(bias_np, np.ndarray)
            else bias_np
        )
        bias = bias if bias.dtype == y.dtype else bias.astype(y.dtype)
        return y + bias

    def _expert_dequant(self, xt, expert_idx: int, weight: float):
        dt = BlockTC.COMPUTE_DTYPE
        gate_w, gate_b, down_w, down_b = self._selected_expert(expert_idx)
        gate_tc = tc.tensor(np.ascontiguousarray(gate_w), dtype="float32").astype(dt)
        gate_up = self._bias_add(tc.matmul(xt, gate_tc), gate_b)
        act = gpt_oss_expert_activation_tc(gate_up)
        down_tc = tc.tensor(np.ascontiguousarray(down_w), dtype="float32").astype(dt)
        y = self._bias_add(tc.matmul(act, down_tc), down_b) * weight
        del gate_w, gate_b, down_w, down_b, gate_tc, gate_up, act, down_tc
        return y

    def _expert_packed_mxfp4(self, xt, expert_idx: int, weight: float):
        if self.expert_mode == "resident_packed_mxfp4":
            e = int(expert_idx)
            gate_n = self.gate_up_blocks.shape[1]
            down_n = self.down_blocks.shape[1]
            gate_b = self.gate_up_bias_tc.slice(0, e, 1).reshape([gate_n])
            down_b = self.down_bias_tc.slice(0, e, 1).reshape([down_n])
            gate_up = self._bias_add(
                tc.mxfp4_linear_expert(
                    xt, self.gate_up_blocks_tc, self.gate_up_scales_tc, e
                ),
                gate_b,
            )
            act = gpt_oss_expert_activation_tc(gate_up)
            y = self._bias_add(
                tc.mxfp4_linear_expert(act, self.down_blocks_tc, self.down_scales_tc, e),
                down_b,
            )
            y = y * weight
            del gate_b, down_b, gate_up, act
            return y

        (
            gate_blocks,
            gate_scales,
            gate_b,
            down_blocks,
            down_scales,
            down_b,
        ) = self._selected_expert_packed(expert_idx)
        gate_up = self._bias_add(tc.mxfp4_linear(xt, gate_blocks, gate_scales), gate_b)
        act = gpt_oss_expert_activation_tc(gate_up)
        y = self._bias_add(tc.mxfp4_linear(act, down_blocks, down_scales), down_b)
        y = y * weight
        del gate_blocks, gate_scales, gate_b, down_blocks, down_scales, down_b
        del gate_up, act
        return y

    def __call__(self, x):
        B, L, H = x.shape
        x_flat = x.reshape([B * L, H])
        topw, topi = self._route(x_flat)
        topw_np = topw.numpy().astype(np.float32)
        topi_np = topi.numpy().astype(np.int64)

        routed = []
        for t in range(B * L):
            xt = x_flat.slice(0, t, 1)
            acc = None
            for slot in range(self.cfg.num_experts_per_tok):
                e = int(topi_np[t, slot])
                w = float(topw_np[t, slot])
                if self.expert_mode in {"packed_mxfp4", "resident_packed_mxfp4"}:
                    y = self._expert_packed_mxfp4(xt, e, w)
                else:
                    y = self._expert_dequant(xt, e, w)
                acc = y if acc is None else acc + y
                del y
                if hasattr(tc, "empty_cache"):
                    tc.empty_cache()
            routed.append(acc)
        out = tc.cat(routed, dim=0).reshape([B, L, H])
        route_info = {
            "expert_mode": self.expert_mode,
            "top_indices": topi_np.tolist(),
            "top_weights": topw_np.tolist(),
            "unique_experts": sorted({int(x) for x in topi_np.reshape(-1)}),
        }
        return out, route_info


class GptOssDiagnosticBlockTC:
    """One GPT-OSS decoder block with attention plus selected-expert MoE."""

    def __init__(
        self,
        cfg: GptOss20BConfig,
        layer_idx: int,
        *,
        expert_mode: str = "dequant",
    ):
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
        self.expert_mode = expert_mode
        self.input_layernorm = RMSNormTC(cfg.hidden_dim, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNormTC(cfg.hidden_dim, cfg.rms_norm_eps)
        self.self_attn = GptOssAttentionTC(cfg, layer_idx)
        self.mlp = GptOssMoEDiagnosticTC(cfg, layer_idx, expert_mode=expert_mode)

    @classmethod
    def from_safetensors(
        cls,
        cfg: GptOss20BConfig,
        where: dict[str, str],
        layer_idx: int,
        *,
        expert_mode: str = "dequant",
    ) -> "GptOssDiagnosticBlockTC":
        obj = cls(cfg, layer_idx, expert_mode=expert_mode)
        p = f"model.layers.{layer_idx}"
        obj.input_layernorm.weight = tc.tensor(
            load_tensor_np(where, f"{p}.input_layernorm.weight"), dtype="float32"
        )
        obj.post_attention_layernorm.weight = tc.tensor(
            load_tensor_np(where, f"{p}.post_attention_layernorm.weight"), dtype="float32"
        )
        obj.self_attn = GptOssAttentionTC.from_safetensors(cfg, where, layer_idx)
        obj.mlp = GptOssMoEDiagnosticTC.from_safetensors(
            cfg, where, layer_idx, expert_mode=expert_mode
        )
        return obj

    def __call__(self, x, cos, sin, position_offset: int = 0, kv_cache=None):
        dt = BlockTC.COMPUTE_DTYPE
        normed = self.input_layernorm(x)
        normed = normed.half() if dt == "float16" else normed.astype(dt)
        attn, kv = self.self_attn(normed, cos, sin, position_offset, kv_cache)
        h = x + attn
        mlp_in = self.post_attention_layernorm(h)
        mlp_in = mlp_in.half() if dt == "float16" else mlp_in.astype(dt)
        mlp_out, route_info = self.mlp(mlp_in)
        return h + mlp_out, kv, route_info
