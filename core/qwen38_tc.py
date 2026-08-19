"""Qwen3.8-27B text-only INT3 adapter for the tensor_cuda engine.

This is a parameterized extension of :mod:`core.qwen35_tc`.  It preserves the
validated Qwen3.5-9B path and adds only the 27B release deltas:

* the attention output gate defaults to sigmoid; the checkpoint's
  output_gate_type metadata is not used to select this gate;
* DeltaNet 48/16 and attention 24/4 head ratios remain config-derived;
* every large linear, including lm_head, is cached and resident as INT3;
* the embedding is a host-side mmap in raw BF16 form;
* vision and MTP tensors are never selected by the cache or loader.

The cache is deliberately NumPy/safetensors-only.  It is safe to build on a
CPU-only host and processes one tensor in bounded row chunks.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from core.mistral7b_tc import (BlockTC, F, LinearTC, QuantLinearTC, RMSNormTC,
                               GROUP_SIZE, tc)
from core.qwen35_tc import (F32Linear, GatedDeltaNetTC, Qwen35AttentionTC,
                            Qwen35BlockTC, Qwen35Config, Qwen35_TC, SwiGLUTC,
                            _cast, _per_head_rmsnorm, _repeat_kv)


DEFAULT_MODEL_DIR = "/mnt/ForgeRealm/models/Qwen3.8-27B"
DEFAULT_CACHE_DIR = "artifacts/qwen38_int3_cache"
DEFAULT_MAX_CONTEXT = 4096
DEFAULT_PREFILL_CHUNK = 64
DEFAULT_KV_BLOCK = 256
# The pre-LC1 single-shot SDPA path was registered at this context capacity.
# Larger configured windows use bounded tiled attention so their transient does
# not scale with the total KV length.
LEGACY_ATTENTION_MAX_CONTEXT = DEFAULT_MAX_CONTEXT
# 6,208 rows divide the 248,320-row vocabulary exactly into 40 chunks and
# bound the two-stage BF16 dequantized weight to 60.625 MiB at hidden=5,120.
DEFAULT_LM_HEAD_CHUNK_ROWS = 6208
DEFAULT_ATTENTION_OUTPUT_GATE = "sigmoid"
CACHE_SCHEMA = 1
VRAM_CEILING_MIB = 12000
# Calibrated from the 2026-08-19 whole-device receipt after subtracting the
# computed resident tensors, DeltaNet state, seq=160 BF16 KV, and the old
# 317,849,600-byte lm-head dequant buffer.  It deliberately includes the
# ~350 MiB desktop plus allocator/activation/kernel workspace high-water.
CALIBRATED_FIXED_PEAK_OVERHEAD_MIB = 1185


def pack_int3_vectorized(codes: np.ndarray) -> np.ndarray:
    """Pack eight 3-bit codes into three bytes, with no Python row loop.

    TensorCUDA's layout is little-endian in a per-row bit stream.  Qwen linear
    input dimensions are divisible by 128, hence also by eight.
    """
    q = np.asarray(codes, dtype=np.uint8)
    if q.ndim != 2:
        raise ValueError(f"codes must be rank-2, got {q.shape}")
    rows, cols = q.shape
    if cols % 8:
        raise ValueError(f"INT3 vector pack requires cols%8==0, got {cols}")
    if q.size and int(q.max()) > 7:
        raise ValueError("INT3 codes must be in [0, 7]")
    g = q.reshape(rows, cols // 8, 8).astype(np.uint32, copy=False)
    word = (
        g[:, :, 0]
        | (g[:, :, 1] << 3)
        | (g[:, :, 2] << 6)
        | (g[:, :, 3] << 9)
        | (g[:, :, 4] << 12)
        | (g[:, :, 5] << 15)
        | (g[:, :, 6] << 18)
        | (g[:, :, 7] << 21)
    )
    out = np.empty((rows, (cols // 8) * 3), dtype=np.uint8)
    out[:, 0::3] = word.astype(np.uint8)
    out[:, 1::3] = (word >> 8).astype(np.uint8)
    out[:, 2::3] = (word >> 16).astype(np.uint8)
    return out


def unpack_int3_vectorized(packed: np.ndarray, in_features: int) -> np.ndarray:
    """Inverse of :func:`pack_int3_vectorized`, used by CPU G2."""
    p = np.asarray(packed, dtype=np.uint8)
    if p.ndim != 2 or in_features % 8:
        raise ValueError(f"bad packed shape/in_features: {p.shape}, {in_features}")
    if p.shape[1] != in_features * 3 // 8:
        raise ValueError("packed byte width mismatch")
    b = p.reshape(p.shape[0], in_features // 8, 3).astype(np.uint32)
    word = b[:, :, 0] | (b[:, :, 1] << 8) | (b[:, :, 2] << 16)
    out = np.empty((p.shape[0], in_features // 8, 8), dtype=np.uint8)
    for i in range(8):
        out[:, :, i] = ((word >> (3 * i)) & 7).astype(np.uint8)
    return out.reshape(p.shape[0], in_features)


def gate_int3_pack(seed: int = 3801) -> dict:
    """Bit-exact CPU gate against the canonical engine reference packer."""
    from tensor_cuda.quantization import pack_lowbit

    rng = np.random.default_rng(seed)
    cases = [(1, 128), (7, 256), (31, 512), (257, 128)]
    checked = 0
    for rows, cols in cases:
        q = rng.integers(0, 8, size=(rows, cols), dtype=np.uint8)
        got = pack_int3_vectorized(q)
        expected = pack_lowbit(q, 3)
        if not np.array_equal(got, expected):
            delta = np.flatnonzero(got != expected)[0]
            raise AssertionError(
                f"vector INT3 pack mismatch case={rows}x{cols} byte={delta}"
            )
        checked += q.size
    return {"cases": len(cases), "codes_checked": checked, "max_byte_diff": 0}


def quantize_kv_int8_np(x: np.ndarray):
    """Symmetric per-token, per-head INT8 reference packer.

    ``x`` is ``(..., head_dim)``. Codes use uint8 storage with zero-point 128;
    one float32 scale is returned for every vector. This CPU reference is the
    bit-exact contract tested independently of CUDA.
    """
    a = np.asarray(x, dtype=np.float32)
    if a.ndim < 1 or a.shape[-1] <= 0:
        raise ValueError(f"bad KV shape {a.shape}")
    scale = np.max(np.abs(a), axis=-1, keepdims=True) / np.float32(127.0)
    scale = np.maximum(scale, np.float32(1e-8))
    signed = np.clip(np.rint(a / scale), -127, 127).astype(np.int16)
    return (signed + 128).astype(np.uint8), scale.astype(np.float32)


def dequantize_kv_int8_np(packed: np.ndarray, scale: np.ndarray):
    """Exact inverse arithmetic for :func:`quantize_kv_int8_np`."""
    q = np.asarray(packed, dtype=np.uint8)
    s = np.asarray(scale, dtype=np.float32)
    if s.shape != q.shape[:-1] + (1,):
        raise ValueError(f"KV scale shape {s.shape} does not match codes {q.shape}")
    return (q.astype(np.int16).astype(np.float32) - 128.0) * s


def gate_kv_int8_pack(seed: int = 3811) -> dict:
    """CPU-only bit-exact pack/unpack gate for the INT8 KV format."""
    rng = np.random.default_rng(seed)
    cases = [(1, 4, 1, 256), (2, 3, 7, 64), (1, 1, 19, 8)]
    values = 0
    for shape in cases:
        x = rng.standard_normal(shape).astype(np.float32)
        # Exercise both ordinary and deliberately massive-activation vectors.
        x.reshape(-1, shape[-1])[0, 0] *= np.float32(4096.0)
        packed, scale = quantize_kv_int8_np(x)
        expected_codes = (np.clip(np.rint(x / scale), -127, 127)
                          .astype(np.int16) + 128).astype(np.uint8)
        np.testing.assert_array_equal(packed, expected_codes)
        expected = (expected_codes.astype(np.int16).astype(np.float32) - 128.0) * scale
        np.testing.assert_array_equal(dequantize_kv_int8_np(packed, scale), expected)
        values += x.size
    return {"cases": len(cases), "values_checked": values,
            "code_byte_mismatches": 0, "unpack_bit_exact": True}


def select_qwen38_attn_path(max_context: int, kv_int8: bool = False,
                            kv_host: bool = False,
                            force_tiled: bool = False) -> str:
    """Select one attention arithmetic path for the complete model load."""
    if force_tiled or kv_int8 or kv_host:
        return "tiled"
    return ("legacy" if int(max_context) <= LEGACY_ATTENTION_MAX_CONTEXT
            else "tiled")


def online_softmax_tiled_np(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                            tile_rows: int, position_offset: int = 0):
    """FP32 NumPy oracle for the TensorCUDA tiled online-softmax recurrence."""
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    if (q.ndim != 4 or k.ndim != 4 or v.ndim != 4
            or q.shape[:2] != k.shape[:2] or k.shape != v.shape
            or q.shape[-1] != k.shape[-1]):
        raise ValueError(f"incompatible q/k/v shapes: {q.shape}, {k.shape}, {v.shape}")
    if tile_rows <= 0:
        raise ValueError("tile_rows must be positive")
    _, _, query_rows, head_dim = q.shape
    key_rows = k.shape[2]
    if key_rows <= 0:
        raise ValueError("attention cache is empty")
    q_positions = np.arange(
        int(position_offset), int(position_offset) + query_rows)[:, None]
    m = denom = accum = None
    scale = np.float32(head_dim ** -0.5)
    for lo in range(0, key_rows, int(tile_rows)):
        hi = min(lo + int(tile_rows), key_rows)
        scores = np.matmul(q, np.swapaxes(k[:, :, lo:hi], -1, -2)) * scale
        if query_rows > 1:
            key_positions = np.arange(lo, hi)[None, :]
            scores = scores + np.where(
                key_positions <= q_positions, np.float32(0.0),
                np.float32(-1e30)).reshape(1, 1, query_rows, hi - lo)
        block_max = scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores - block_max)
        block_denom = weights.sum(axis=-1, keepdims=True)
        block_accum = np.matmul(weights, v[:, :, lo:hi])
        if m is None:
            m, denom, accum = block_max, block_denom, block_accum
        else:
            merged = np.maximum(m, block_max)
            old_scale = np.exp(m - merged)
            new_scale = np.exp(block_max - merged)
            denom = denom * old_scale + block_denom * new_scale
            accum = accum * old_scale + block_accum * new_scale
            m = merged
    return (accum / denom).astype(np.float32, copy=False)


def gate_online_softmax_tiled_np(seed: int = 3821) -> dict:
    """Prove the tiled recurrence against ordinary FP32 stable softmax."""
    rng = np.random.default_rng(seed)
    specs = [
        # (batch, heads, query rows, key rows, head dim, tile rows)
        (1, 2, 4, 4, 7, 4),       # one tile
        (2, 3, 5, 9, 8, 5),       # two tiles, four-row remainder
        (1, 4, 7, 20, 11, 3),     # seven tiles, two-row remainder
    ]
    results = []
    for batch, heads, query_rows, key_rows, head_dim, tile_rows in specs:
        q = rng.standard_normal(
            (batch, heads, query_rows, head_dim), dtype=np.float32)
        k = rng.standard_normal(
            (batch, heads, key_rows, head_dim), dtype=np.float32)
        v = rng.standard_normal(
            (batch, heads, key_rows, head_dim), dtype=np.float32)
        position_offset = key_rows - query_rows
        scores = (np.matmul(q, np.swapaxes(k, -1, -2))
                  * np.float32(head_dim ** -0.5))
        key_positions = np.arange(key_rows)[None, :]
        query_positions = np.arange(
            position_offset, position_offset + query_rows)[:, None]
        scores = scores + np.where(
            key_positions <= query_positions, np.float32(0.0),
            np.float32(-1e30)).reshape(1, 1, query_rows, key_rows)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        expected = np.matmul(weights / weights.sum(axis=-1, keepdims=True), v)
        got = online_softmax_tiled_np(
            q, k, v, tile_rows, position_offset=position_offset)
        delta = got - expected
        rel_l2 = float(np.linalg.norm(delta) /
                       max(float(np.linalg.norm(expected)), 1e-30))
        max_abs = float(np.max(np.abs(delta), initial=0.0))
        if not np.isfinite(rel_l2) or rel_l2 > 1e-6:
            raise AssertionError(
                f"online softmax mismatch tiles={math.ceil(key_rows / tile_rows)} "
                f"rel_l2={rel_l2:.9g} max_abs={max_abs:.9g}")
        results.append({
            "shape": [batch, heads, query_rows, key_rows, head_dim],
            "tile_rows": tile_rows,
            "tiles": math.ceil(key_rows / tile_rows),
            "remainder": key_rows % tile_rows,
            "rel_l2": rel_l2,
            "max_abs": max_abs,
        })
    return {"status": "PASS", "tolerance_rel_l2": 1e-6,
            "defect_caught": None, "cases": results}


class Qwen38Config(Qwen35Config):
    output_gate_type = DEFAULT_ATTENTION_OUTPUT_GATE
    hidden_act = "silu"
    mamba_ssm_dtype = "float32"
    max_position_embeddings = 262144

    def __init__(self, **kwargs):
        self.output_gate_type = kwargs.pop("output_gate_type", type(self).output_gate_type)
        self.hidden_act = kwargs.pop("hidden_act", type(self).hidden_act)
        self.mamba_ssm_dtype = kwargs.pop("mamba_ssm_dtype", type(self).mamba_ssm_dtype)
        self.max_position_embeddings = int(
            kwargs.pop("max_position_embeddings", type(self).max_position_embeddings)
        )
        super().__init__(**kwargs)
        self._validate_qwen38()

    @classmethod
    def from_model_dir(cls, model_dir):
        base = Qwen35Config.from_model_dir(model_dir)
        with open(os.path.join(base.model_dir, "config.json")) as fh:
            raw = json.load(fh)
        text = dict(raw.get("text_config") or raw)
        return cls(
            **vars(base),
            # Empirically adjudicated for the 27B attention path.  The
            # checkpoint's output_gate_type="swish" field is not consumed by
            # installed Transformers attention and does not identify this gate.
            output_gate_type=DEFAULT_ATTENTION_OUTPUT_GATE,
            hidden_act=text.get("hidden_act", "silu"),
            mamba_ssm_dtype=text.get("mamba_ssm_dtype", "float32"),
            max_position_embeddings=text.get("max_position_embeddings", 262144),
        )

    def _validate_qwen38(self):
        if self.output_gate_type not in ("sigmoid", "swish", "silu"):
            raise ValueError(f"unsupported output_gate_type={self.output_gate_type!r}")
        if self.hidden_act not in ("silu", "swish"):
            raise ValueError(f"unsupported hidden_act={self.hidden_act!r}")
        if self.mamba_ssm_dtype != "float32":
            raise ValueError("Qwen3.8 DeltaNet requires mamba_ssm_dtype=float32")
        if self.n_v_heads % self.n_k_heads:
            raise ValueError("linear value heads must be divisible by key heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("attention heads must be divisible by KV heads")
        for name, dim in (("hidden_dim", self.hidden_dim),
                          ("intermediate_dim", self.intermediate_dim),
                          ("n_v_heads*d_v", self.n_v_heads * self.d_v),
                          ("n_k_heads*d_k", self.n_k_heads * self.d_k)):
            if dim % GROUP_SIZE:
                raise ValueError(f"{name}={dim} is not INT3 group aligned")

    def as_printable_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "intermediate_dim": self.intermediate_dim,
            "num_layers": self.num_layers,
            "layer_types": list(self.layer_types),
            "attention_layers": self.attention_layer_indices(),
            "full_attention_interval": self.full_attention_interval,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "gqa_ratio": self.num_heads // self.num_kv_heads,
            "head_dim": self.head_dim,
            "rope_theta": self.rope_theta,
            "partial_rotary_dim": self.partial_rotary_dim,
            "n_k_heads": self.n_k_heads,
            "n_v_heads": self.n_v_heads,
            "deltanet_head_ratio": self.n_v_heads // self.n_k_heads,
            "d_k": self.d_k,
            "d_v": self.d_v,
            "conv_kernel": self.conv_kernel,
            "rms_norm_eps": self.rms_norm_eps,
            "output_gate_type": self.output_gate_type,
            "hidden_act": self.hidden_act,
            "mamba_ssm_dtype": self.mamba_ssm_dtype,
            "max_position_embeddings": self.max_position_embeddings,
            "eos_token_id": self.eos_token_id,
            "tie_word_embeddings": self.tie_word_embeddings,
            "model_dir": self.model_dir,
        }


def _device_zeros(*shape, dtype="bfloat16"):
    """Allocate zero storage without TensorCUDA's FP32 BF16 staging path."""
    zeros = tc.zeros(*shape, dtype="uint8")
    return zeros if dtype == "uint8" else zeros.astype(dtype)


def _float_to_bf16_u16(x):
    """Preserve already-BF16 values as their raw host uint16 representation."""
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    return (a.view(np.uint32) >> 16).astype(np.uint16)


def _bf16_u16_to_float(x):
    a = np.ascontiguousarray(np.asarray(x, dtype=np.uint16))
    return (a.astype(np.uint32) << 16).view(np.float32)


class Qwen38DeviceKVCache:
    """Fixed-capacity device KV cache with BF16 or symmetric INT8 storage."""

    def __init__(self, batch_size, cfg, capacity, int8=False):
        self.capacity = int(capacity)
        self.count = 0
        self.int8 = bool(int8)
        self.key_stats = None
        shape = (int(batch_size), cfg.num_kv_heads, self.capacity, cfg.head_dim)
        if self.int8:
            self.kb = _device_zeros(*shape, dtype="uint8")
            self.vb = _device_zeros(*shape, dtype="uint8")
            scale_shape = shape[:-1] + (1,)
            self.ks = _device_zeros(*scale_shape)
            self.vs = _device_zeros(*scale_shape)
        else:
            self.kb = _device_zeros(*shape)
            self.vb = _device_zeros(*shape)
            self.ks = self.vs = None

    @staticmethod
    def _pack(x):
        scale = x.abs().max([-1], True) * (1.0 / 127.0) + 1e-8
        q = (x / scale).round().clamp(-127.0, 127.0) + 128.0
        return q.astype("uint8"), _cast(scale)

    @staticmethod
    def _unpack(q, scale):
        return (q.astype("bfloat16") - 128.0) * scale

    def append(self, k, v):
        n = int(k.shape[2])
        if self.count + n > self.capacity:
            raise MemoryError(
                f"KV cache capacity exceeded: need={self.count + n} "
                f"capacity={self.capacity}")
        with tc.no_grad():
            if self.int8:
                if self.key_stats is None:
                    kn = k.float().numpy().astype(np.float32, copy=False)
                    flat = kn.reshape(-1, kn.shape[-1])
                    vmax = np.max(np.abs(flat), axis=-1)
                    vrms = np.sqrt(np.mean(flat * flat, axis=-1))
                    ratio = vmax / np.maximum(vrms, np.float32(1e-12))
                    self.key_stats = {
                        "vectors": int(flat.shape[0]),
                        "abs_p99": float(np.percentile(np.abs(flat), 99.0)),
                        "abs_p999": float(np.percentile(np.abs(flat), 99.9)),
                        "abs_max": float(vmax.max(initial=0.0)),
                        "max_over_rms_p99": float(np.percentile(ratio, 99.0)),
                        "max_over_rms_max": float(ratio.max(initial=0.0)),
                    }
                pk, sk = self._pack(k)
                pv, sv = self._pack(v)
                tc.write_rows(self.kb, pk, self.count)
                tc.write_rows(self.ks, sk, self.count)
                tc.write_rows(self.vb, pv, self.count)
                tc.write_rows(self.vs, sv, self.count)
            else:
                tc.write_rows(self.kb, k, self.count)
                tc.write_rows(self.vb, v, self.count)
        self.count += n

    def block(self, lo, n):
        if self.int8:
            return (self._unpack(self.kb.slice(2, lo, n), self.ks.slice(2, lo, n)),
                    self._unpack(self.vb.slice(2, lo, n), self.vs.slice(2, lo, n)))
        return self.kb.slice(2, lo, n), self.vb.slice(2, lo, n)

    def all(self):
        return self.block(0, self.count)

    @property
    def storage_bytes(self):
        B, KV, _, D = self.kb.shape
        per_token = B * KV * ((2 * D + 4) if self.int8 else (4 * D))
        return self.capacity * per_token


class Qwen38HostKVCache:
    """Raw-BF16 host KV with a fixed-size device staging window at attention."""

    def __init__(self, batch_size, cfg, capacity):
        self.capacity = int(capacity)
        self.count = 0
        shape = (int(batch_size), cfg.num_kv_heads, self.capacity, cfg.head_dim)
        # np.empty reserves virtual address space; pages commit only as tokens
        # are appended. Values are raw BF16 so host storage is lossless.
        self.kh = np.empty(shape, dtype=np.uint16)
        self.vh = np.empty(shape, dtype=np.uint16)

    def append(self, k, v):
        n = int(k.shape[2])
        if self.count + n > self.capacity:
            raise MemoryError(
                f"host KV capacity exceeded: need={self.count + n} "
                f"capacity={self.capacity}")
        lo = self.count
        self.kh[:, :, lo:lo + n] = _float_to_bf16_u16(k.float().numpy())
        self.vh[:, :, lo:lo + n] = _float_to_bf16_u16(v.float().numpy())
        self.count += n

    def block(self, lo, n):
        k = _cast(tc.tensor(_bf16_u16_to_float(self.kh[:, :, lo:lo + n])))
        v = _cast(tc.tensor(_bf16_u16_to_float(self.vh[:, :, lo:lo + n])))
        return k, v

    @property
    def storage_bytes(self):
        return self.kh.nbytes + self.vh.nbytes


class Qwen38AttentionTC(Qwen35AttentionTC):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Correctness baseline for a new checkpoint. APA can be characterized
        # separately after standard attention is parity-gated.
        self.attention_mode = "standard"

    def _apply_output_gate(self, attn, gate):
        kind = self.cfg.output_gate_type
        if kind == "sigmoid":
            return attn * gate.sigmoid()
        if kind in ("swish", "silu"):
            return attn * gate.silu()
        raise ValueError(f"unsupported output gate {kind!r}")

    def _streaming_standard_attention(self, q, cache, position_offset, block_rows):
        """Exact standard GQA attention with fixed-size KV/score staging.

        This is an online log-sum-exp decomposition of the same softmax used by
        standard attention. It changes storage/reduction tiling, not the causal
        attention definition, and never materializes repeated full-context KV.
        """
        B, H, L, D = q.shape
        KV = self.cfg.num_kv_heads
        rep = H // KV
        # Fold query heads and query rows into GEMM M so the unexpanded
        # (B,KV,S,D) cache shares exactly the (B,KV) batch dimensions.
        qg = q.reshape([B, KV, rep * L, D])
        m = denom = accum = None
        q_positions = np.arange(position_offset, position_offset + L)[:, None]
        for lo in range(0, cache.count, block_rows):
            n = min(block_rows, cache.count - lo)
            k, v = cache.block(lo, n)
            scores = tc.matmul(qg, k, alpha=D ** -0.5, trans_b=True)
            scores = scores.reshape([B, KV, rep, L, n]).float()
            if L > 1:
                key_positions = np.arange(lo, lo + n)[None, :]
                bias = np.where(key_positions <= q_positions, 0.0, -1e30)
                bias = bias.astype(np.float32).reshape(1, 1, 1, L, n)
                scores = scores + tc.tensor(np.ascontiguousarray(bias))
            block_max = scores.max([-1], True)
            weights = (scores - block_max).exp()
            block_denom = weights.sum([-1], True)
            block_accum = tc.matmul(
                weights.reshape([B, KV, rep * L, n]), v.float())
            block_accum = block_accum.reshape([B, KV, rep, L, D])
            if m is None:
                m, denom, accum = block_max, block_denom, block_accum
            else:
                merged = m.maximum(block_max)
                old_scale = (m - merged).exp()
                new_scale = (block_max - merged).exp()
                denom = denom * old_scale + block_denom * new_scale
                accum = accum * old_scale + block_accum * new_scale
                m = merged
            del k, v, scores, weights, block_accum
        if accum is None:
            raise RuntimeError("attention cache is empty")
        return _cast((accum / denom).reshape([B, H, L, D]))

    def __call__(self, x, cos, sin, position_offset=0, kv_cache=None):
        if not isinstance(kv_cache, (Qwen38DeviceKVCache, Qwen38HostKVCache)):
            return super().__call__(x, cos, sin, position_offset, kv_cache)
        cfg = self.cfg
        B, L, _ = x.shape
        H, KV, D, R = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim,
                       cfg.partial_rotary_dim)
        qg = self.q_proj(x).reshape([B, L, H, 2 * D])
        q = qg.slice(3, 0, D).reshape([B, L, H * D])
        gate = qg.slice(3, D, D).reshape([B, L, H * D])
        q = _per_head_rmsnorm(q, self.q_norm_w, cfg.rms_norm_eps, B, L, H, D)
        k = _per_head_rmsnorm(self.k_proj(x), self.k_norm_w,
                              cfg.rms_norm_eps, B, L, KV, D)
        q = q.reshape([B, L, H, D]).transpose(1, 2)
        k = k.reshape([B, L, KV, D]).transpose(1, 2)
        v = self.v_proj(x).reshape([B, L, KV, D]).transpose(1, 2)
        cseg = cos.slice(0, position_offset, L)
        sseg = sin.slice(0, position_offset, L)
        q = tc.cat([F.apply_rotary(q.slice(3, 0, R), cseg, sseg),
                    q.slice(3, R, D - R)], dim=3)
        k = tc.cat([F.apply_rotary(k.slice(3, 0, R), cseg, sseg),
                    k.slice(3, R, D - R)], dim=3)
        kv_cache.append(k, v)
        # Original pre-LC1 code path. Keep this block bit-for-bit arithmetic:
        # full cache slices, head expansion, then the functional SDPA call.
        if self.attn_path == "legacy":
            ka, va = kv_cache.all()
            attn = F.scaled_dot_product_attention(
                q, _repeat_kv(ka, H // KV), _repeat_kv(va, H // KV),
                is_causal=(L > 1))
        else:
            block_rows = int(getattr(self, "kv_block_rows", DEFAULT_KV_BLOCK))
            attn = self._streaming_standard_attention(
                q, kv_cache, position_offset, block_rows)
        attn = attn.transpose(1, 2).reshape([B, L, H * D])
        attn = self._apply_output_gate(attn, gate)
        return self.o_proj(_cast(attn)), kv_cache


class Qwen38BlockTC(Qwen35BlockTC):
    def __init__(self, cfg, layer_idx):
        self.is_attn = cfg.is_attention(layer_idx)
        self.input_layernorm = RMSNormTC(cfg.hidden_dim, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNormTC(cfg.hidden_dim, cfg.rms_norm_eps)
        self.mixer = Qwen38AttentionTC(cfg) if self.is_attn else GatedDeltaNetTC(cfg)
        self.mlp = SwiGLUTC()


def _atomic_json(path: Path, value: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(value, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


class INT3PackCache:
    """Resumable, source-fingerprinted, mmap-friendly Qwen3.8 pack cache."""

    def __init__(self, model_dir=DEFAULT_MODEL_DIR, cache_dir=DEFAULT_CACHE_DIR,
                 bits=3, group_size=GROUP_SIZE, row_chunk=1024):
        self.model_dir = Path(model_dir).resolve()
        self.cache_dir = Path(cache_dir).resolve()
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.row_chunk = int(row_chunk)
        if self.bits != 3:
            raise ValueError("Qwen3.8 cache is intentionally INT3-only")
        with open(self.model_dir / "model.safetensors.index.json") as fh:
            self.weight_map = json.load(fh)["weight_map"]
        self.cfg = Qwen38Config.from_model_dir(str(self.model_dir))
        self.qlinear_names = self._qlinear_names()
        self.embedding_name = "model.language_model.embed_tokens.weight"
        self.required_names = set(self.qlinear_names) | {self.embedding_name}

    @staticmethod
    def _is_qlinear(name):
        if name == "lm_head.weight":
            return True
        if not name.startswith("model.language_model.layers.") or not name.endswith(".weight"):
            return False
        leaf = name.rsplit(".", 2)[-2]
        return leaf in {
            "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv",
            "in_proj_z", "out_proj", "gate_proj", "up_proj", "down_proj",
        }

    def _qlinear_names(self):
        return sorted(n for n in self.weight_map if self._is_qlinear(n))

    def _source_files(self):
        out = {}
        for rel in sorted(set(self.weight_map[n] for n in self.required_names)):
            st = (self.model_dir / rel).stat()
            out[rel] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
        cfg_st = (self.model_dir / "config.json").stat()
        out["config.json"] = {"size": cfg_st.st_size, "mtime_ns": cfg_st.st_mtime_ns}
        return out

    def _base_manifest(self):
        return {
            "schema": CACHE_SCHEMA,
            "model_dir": str(self.model_dir),
            "bits": self.bits,
            "group_size": self.group_size,
            "source_files": self._source_files(),
            "required_qlinears": len(self.qlinear_names),
            "entries": {},
            "complete": False,
        }

    @property
    def manifest_path(self):
        return self.cache_dir / "manifest.json"

    def _load_or_reset_manifest(self):
        expected = self._base_manifest()
        try:
            with open(self.manifest_path) as fh:
                found = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return expected
        identity = ("schema", "model_dir", "bits", "group_size", "source_files")
        if any(found.get(k) != expected[k] for k in identity):
            return expected
        expected["entries"] = dict(found.get("entries") or {})
        expected["complete"] = bool(found.get("complete"))
        return expected

    @staticmethod
    def _key(name):
        return hashlib.sha256(name.encode()).hexdigest()[:20]

    def _entry_files_exist(self, entry):
        return all((self.cache_dir / entry[k]).is_file()
                   for k in ("packed", "scales", "zeros"))

    def _quantize_tensor(self, tensor, name):
        shape = tuple(int(x) for x in tensor.shape)
        if len(shape) != 2:
            raise ValueError(f"qlinear {name} is not a matrix: {shape}")
        rows, cols = shape
        if cols % self.group_size or cols % 8:
            raise ValueError(f"unaligned qlinear {name}: {shape}")
        groups = cols // self.group_size
        key = self._key(name)
        rel = {
            "packed": f"{key}.packed.npy",
            "scales": f"{key}.scales.npy",
            "zeros": f"{key}.zeros.npy",
        }
        temp = {k: self.cache_dir / (v + ".tmp") for k, v in rel.items()}
        packed = np.lib.format.open_memmap(
            temp["packed"], mode="w+", dtype=np.uint8,
            shape=(rows, cols * 3 // 8))
        scales = np.lib.format.open_memmap(
            temp["scales"], mode="w+", dtype=np.float16, shape=(rows, groups))
        zeros = np.lib.format.open_memmap(
            temp["zeros"], mode="w+", dtype=np.float16, shape=(rows, groups))
        for lo in range(0, rows, self.row_chunk):
            hi = min(rows, lo + self.row_chunk)
            w = tensor[lo:hi].float().numpy()
            wg = w.reshape(hi - lo, groups, self.group_size)
            mn = wg.min(axis=2)
            mx = wg.max(axis=2)
            sc = (mx - mn) / 7.0
            sc = np.where(sc == 0, 1.0, sc)
            codes = np.clip(np.round((wg - mn[:, :, None]) / sc[:, :, None]),
                            0, 7).astype(np.uint8)
            packed[lo:hi] = pack_int3_vectorized(codes.reshape(hi - lo, cols))
            scales[lo:hi] = sc.astype(np.float16)
            zeros[lo:hi] = mn.astype(np.float16)
            del w, wg, mn, mx, sc, codes
        for arr in (packed, scales, zeros):
            arr.flush()
        del packed, scales, zeros
        for k, relpath in rel.items():
            os.replace(temp[k], self.cache_dir / relpath)
        return {
            "kind": "qlinear",
            "shape": list(shape),
            "bits": self.bits,
            "group_size": self.group_size,
            "source": self.weight_map[name],
            **rel,
        }

    def _cache_embedding(self, tensor):
        if str(tensor.dtype) != "torch.bfloat16":
            raise ValueError(f"expected BF16 embedding, got {tensor.dtype}")
        rows, cols = (int(x) for x in tensor.shape)
        rel = "embedding.bfloat16_u16.npy"
        tmp = self.cache_dir / (rel + ".tmp")
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint16,
                                        shape=(rows, cols))
        for lo in range(0, rows, self.row_chunk):
            hi = min(rows, lo + self.row_chunk)
            out[lo:hi] = tensor[lo:hi].view(__import__("torch").uint16).numpy()
        out.flush()
        del out
        os.replace(tmp, self.cache_dir / rel)
        return {
            "kind": "host_embedding",
            "shape": [rows, cols],
            "storage_dtype": "bfloat16-as-uint16",
            "source": self.weight_map[self.embedding_name],
            "file": rel,
        }

    def build(self, progress=True):
        """Build or resume the complete cache, one source shard at a time."""
        from safetensors import safe_open

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        pack_gate = gate_int3_pack()
        manifest = self._load_or_reset_manifest()
        manifest["pack_gate"] = pack_gate
        manifest["complete"] = False
        _atomic_json(self.manifest_path, manifest)
        by_shard = {}
        for name in sorted(self.required_names):
            by_shard.setdefault(self.weight_map[name], []).append(name)
        started = time.perf_counter()
        done = 0
        total = len(self.required_names)
        for rel, names in sorted(by_shard.items()):
            with safe_open(self.model_dir / rel, framework="pt", device="cpu") as fh:
                for name in names:
                    old = manifest["entries"].get(name)
                    if old is not None:
                        files_ok = ((self.cache_dir / old["file"]).is_file()
                                    if old["kind"] == "host_embedding"
                                    else self._entry_files_exist(old))
                        if files_ok:
                            done += 1
                            continue
                    tensor = fh.get_tensor(name)
                    if name == self.embedding_name:
                        entry = self._cache_embedding(tensor)
                    else:
                        entry = self._quantize_tensor(tensor, name)
                    manifest["entries"][name] = entry
                    _atomic_json(self.manifest_path, manifest)
                    done += 1
                    if progress:
                        print(f"CACHE {done}/{total} {name}", flush=True)
                    del tensor
                    gc.collect()
        missing = sorted(self.required_names - set(manifest["entries"]))
        if missing:
            raise RuntimeError(f"cache incomplete, missing {missing[:3]}")
        manifest["complete"] = True
        manifest["elapsed_seconds"] = time.perf_counter() - started
        _atomic_json(self.manifest_path, manifest)
        return manifest

    def ensure_complete(self, progress=True):
        manifest = self._load_or_reset_manifest()
        entries = manifest.get("entries", {})
        if manifest.get("complete") and set(entries) >= self.required_names:
            files_ok = True
            for entry in entries.values():
                if entry["kind"] == "host_embedding":
                    files_ok &= (self.cache_dir / entry["file"]).is_file()
                else:
                    files_ok &= self._entry_files_exist(entry)
            if files_ok:
                return manifest
        return self.build(progress=progress)

    def load_manifest(self):
        with open(self.manifest_path) as fh:
            manifest = json.load(fh)
        expected = self._base_manifest()
        for k in ("schema", "model_dir", "bits", "group_size", "source_files"):
            if manifest.get(k) != expected[k]:
                raise RuntimeError(f"stale INT3 cache: manifest field {k}")
        if not manifest.get("complete"):
            raise RuntimeError("INT3 cache is incomplete")
        return manifest

    def load_quant(self, name, manifest=None):
        manifest = manifest or self.load_manifest()
        entry = manifest["entries"][name]
        return (
            entry,
            np.load(self.cache_dir / entry["packed"], mmap_mode="r"),
            np.load(self.cache_dir / entry["scales"], mmap_mode="r"),
            np.load(self.cache_dir / entry["zeros"], mmap_mode="r"),
        )


class HostBFloat16Embedding:
    """Host-resident raw-BF16 mmap; only selected rows are converted/uploaded."""

    def __init__(self):
        self.weight_u16 = None

    @property
    def weight(self):
        return self.weight_u16

    def __call__(self, ids_np):
        rows = np.ascontiguousarray(self.weight_u16[ids_np.reshape(-1)])
        fp32 = (rows.astype(np.uint32) << 16).view(np.float32)
        fp32 = fp32.reshape(ids_np.shape + (rows.shape[-1],))
        return _cast(tc.tensor(np.ascontiguousarray(fp32)))


class CachedQuantLinearTC(QuantLinearTC):
    """QuantLinearTC constructed from cache arrays without requantization."""

    @classmethod
    def from_arrays(cls, entry, packed, scales, zeros, cache_name):
        self = cls.__new__(cls)
        self.out_features, self.in_features = (int(x) for x in entry["shape"])
        self.group_size = int(entry["group_size"])
        self.bits = int(entry["bits"])
        self.cache_name = cache_name
        self.packed = tc.tensor(np.ascontiguousarray(packed), dtype="uint8")
        self.scales = tc.tensor(np.ascontiguousarray(scales), dtype="float16")
        self.zeros = tc.tensor(np.ascontiguousarray(zeros), dtype="float16")
        self._vram = packed.nbytes + scales.nbytes + zeros.nbytes
        return self


class PackedRowChunkedQuantLinearTC:
    """Low-bit linear whose packed output rows are resident in bounded chunks.

    This is intentionally narrower than :class:`QuantLinearTC`: it is the
    Qwen3.8 lm_head OOM guard.  The ordinary two-stage INT3 kernel dequantizes
    its complete packed argument, so presenting one output-row chunk per call
    bounds the temporary BF16 weight to ``chunk_rows * in_features * 2``.
    No full-size floating-point lm_head weight is ever constructed.
    """

    def __init__(self, entry, packed, scales, zeros, cache_name,
                 chunk_rows=DEFAULT_LM_HEAD_CHUNK_ROWS):
        self.out_features, self.in_features = (int(x) for x in entry["shape"])
        self.group_size = int(entry["group_size"])
        self.bits = int(entry["bits"])
        self.cache_name = cache_name
        self.chunk_rows = int(chunk_rows)
        if self.chunk_rows <= 0:
            raise ValueError(f"lm_head chunk_rows must be positive, got {chunk_rows}")
        expected = (
            (self.out_features, self.in_features * self.bits // 8),
            (self.out_features, self.in_features // self.group_size),
            (self.out_features, self.in_features // self.group_size),
        )
        found = (tuple(packed.shape), tuple(scales.shape), tuple(zeros.shape))
        if found != expected:
            raise ValueError(f"bad packed lm_head arrays: expected={expected} found={found}")
        self.chunks = []
        for lo, hi in self.row_ranges(self.out_features, self.chunk_rows):
            # Slice the mmap-backed packed representation on the host, then
            # upload only that packed row range.  This avoids first creating a
            # monolithic device lm_head that would need to be sliced/copied.
            self.chunks.append((
                tc.tensor(np.ascontiguousarray(packed[lo:hi]), dtype="uint8"),
                tc.tensor(np.ascontiguousarray(scales[lo:hi]), dtype="float16"),
                tc.tensor(np.ascontiguousarray(zeros[lo:hi]), dtype="float16"),
            ))
        self._vram = int(packed.nbytes + scales.nbytes + zeros.nbytes)

    @staticmethod
    def row_ranges(rows, chunk_rows=DEFAULT_LM_HEAD_CHUNK_ROWS):
        rows = int(rows)
        chunk_rows = int(chunk_rows)
        if rows < 0 or chunk_rows <= 0:
            raise ValueError(f"invalid row chunking rows={rows} chunk_rows={chunk_rows}")
        return [(lo, min(rows, lo + chunk_rows))
                for lo in range(0, rows, chunk_rows)]

    def __call__(self, x):
        parts = []
        for packed, scales, zeros in self.chunks:
            # Deliberately never select intn_linear_fused here: the canonical
            # INT3 fused parity defect requires the validated two-stage path.
            parts.append(tc.intn_linear(
                x, packed, scales, zeros, self.bits,
                self.in_features, self.group_size))
        return parts[0] if len(parts) == 1 else tc.cat(parts, dim=-1)

    def vram_bytes(self):
        return self._vram

    def dequant_transient_bytes(self):
        return min(self.chunk_rows, self.out_features) * self.in_features * 2


class Qwen38_TC(Qwen35_TC):
    def __init__(self, cfg=None, lm_head_chunk_rows=DEFAULT_LM_HEAD_CHUNK_ROWS,
                 max_context=DEFAULT_MAX_CONTEXT, kv_int8=False, kv_host=False,
                 prefill_chunk=DEFAULT_PREFILL_CHUNK,
                 kv_block_rows=DEFAULT_KV_BLOCK, force_tiled_attention=False):
        self.config = cfg or Qwen38Config.from_model_dir(DEFAULT_MODEL_DIR)
        self.lm_head_chunk_rows = int(lm_head_chunk_rows)
        if self.lm_head_chunk_rows <= 0:
            raise ValueError("lm_head_chunk_rows must be positive")
        self.max_context = int(max_context)
        self.kv_int8 = bool(kv_int8)
        self.kv_host = bool(kv_host)
        self.attn_path = select_qwen38_attn_path(
            self.max_context, self.kv_int8, self.kv_host,
            force_tiled=force_tiled_attention)
        self.prefill_chunk = int(prefill_chunk)
        self.kv_block_rows = int(kv_block_rows)
        if not 1 <= self.max_context <= self.config.max_position_embeddings:
            raise ValueError(
                f"max_context must be in [1,{self.config.max_position_embeddings}], "
                f"got {self.max_context}")
        if self.kv_int8 and self.kv_host:
            raise ValueError("--kv-int8 and --kv-host are mutually exclusive")
        if self.prefill_chunk <= 0 or self.kv_block_rows <= 0:
            raise ValueError("prefill_chunk and kv_block_rows must be positive")
        self.embed_tokens = HostBFloat16Embedding()
        self.layers = [Qwen38BlockTC(self.config, i)
                       for i in range(self.config.num_layers)]
        for layer in self.layers:
            if layer.is_attn:
                layer.mixer.kv_block_rows = self.kv_block_rows
                layer.mixer.attn_path = self.attn_path
        self.norm = RMSNormTC(self.config.hidden_dim, self.config.rms_norm_eps)
        self.lm_head = None
        self._rope_len = 0
        self.extend_rope(self.max_context)
        self.vram_map = None
        self._preallocated_caches = None

    def new_caches(self, batch_size=1):
        """Preallocate the configured KV capacity before prompt execution."""
        cache_cls = Qwen38HostKVCache if self.kv_host else Qwen38DeviceKVCache
        out = []
        for layer in self.layers:
            if layer.is_attn:
                if self.kv_host:
                    out.append(cache_cls(batch_size, self.config, self.max_context))
                else:
                    out.append(cache_cls(batch_size, self.config, self.max_context,
                                         int8=self.kv_int8))
            else:
                out.append(None)
        return out

    def take_preallocated_caches(self, batch_size=1):
        """Consume the load-time batch-1 cache allocation, then allocate fresh."""
        if batch_size == 1 and self._preallocated_caches is not None:
            caches = self._preallocated_caches
            self._preallocated_caches = None
            return caches
        return self.new_caches(batch_size)

    def __call__(self, input_ids_np, caches=None, position_offset=0,
                 last_token_only=False, max_layers=None):
        """Run Qwen3.8 with bounded prefill and final-position logits."""
        _, total = input_ids_np.shape
        if position_offset + total > self.max_context:
            raise ValueError(
                f"request exceeds configured context: offset={position_offset} "
                f"tokens={total} max_context={self.max_context}")
        if max_layers is None and total > self.prefill_chunk:
            if caches is None:
                caches = self.new_caches(input_ids_np.shape[0])
            logits = None
            outputs = []
            for s0 in range(0, total, self.prefill_chunk):
                seg = np.ascontiguousarray(
                    input_ids_np[:, s0:s0 + self.prefill_chunk])
                logits, caches = self._forward(
                    seg, caches, position_offset + s0,
                    last_token_only=last_token_only)
                if not last_token_only:
                    outputs.append(logits)
                tc.empty_cache()
            return (logits if last_token_only else tc.cat(outputs, dim=1)), caches
        return self._forward(input_ids_np, caches, position_offset,
                             last_token_only, max_layers)

    def _forward(self, input_ids_np, caches=None, position_offset=0,
                 last_token_only=False, max_layers=None):
        _, L = input_ids_np.shape
        if caches is None:
            caches = self.new_caches(input_ids_np.shape[0])
        shift = 0
        for layer in self.layers:
            if not getattr(layer, "is_attn", False):
                continue
            att = layer.mixer
            live_shift = getattr(att, "live_shift", None)
            if live_shift is None:
                live_shift = getattr(att, "graft_seats", 0)
            shift = max(shift, int(live_shift or 0))
        if position_offset + shift + L > self.max_context:
            raise ValueError("graft position shift exceeds configured max_context")
        h = self.embed_tokens(input_ids_np)
        new_caches = []
        run = self.layers if max_layers is None else self.layers[:max_layers]
        for i, layer in enumerate(run):
            cache = caches[i] if caches is not None else None
            h, c = layer(h, self.rope_cos, self.rope_sin, position_offset, cache)
            new_caches.append(c)
        if max_layers is not None:
            return None, new_caches, h
        h = _cast(self.norm(h))
        if last_token_only and h.shape[1] > 1:
            # Prefill and greedy decode consume only next-token logits.  Slice
            # before lm_head so prompt length never multiplies vocab logits.
            h = h.slice(1, h.shape[1] - 1, 1)
        return self.lm_head(h), new_caches

    def _computed_vram_map(self, context_tokens=128, batch_size=1):
        cfg = self.config
        body = 0
        for layer in self.layers:
            mx = layer.mixer
            projs = ([mx.q_proj, mx.k_proj, mx.v_proj, mx.o_proj]
                     if layer.is_attn else
                     [mx.in_proj_qkv, mx.in_proj_z, mx.out_proj])
            projs += [layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj]
            body += sum(p.vram_bytes() for p in projs)
        lm = self.lm_head.vram_bytes()
        standard_norms = (2 * cfg.num_layers * cfg.hidden_dim + cfg.hidden_dim) * 4
        qk_norms = len(cfg.attention_layer_indices()) * 2 * cfg.head_dim * 4
        delta_layers = cfg.num_layers - len(cfg.attention_layer_indices())
        delta_aux = delta_layers * (
            2 * cfg.hidden_dim * cfg.n_v_heads * 4
            + (2 * cfg.n_k_heads * cfg.d_k + cfg.n_v_heads * cfg.d_v)
              * cfg.conv_kernel * 4
            + 3 * cfg.n_v_heads * 4
            + cfg.d_v * 4
        )
        rope = 2 * self.max_context * cfg.partial_rotary_dim * 2
        recurrent = (batch_size * delta_layers * cfg.n_v_heads
                     * cfg.d_k * cfg.d_v * 4)
        conv_state = (batch_size * delta_layers * (cfg.conv_kernel - 1)
                      * (2 * cfg.n_k_heads * cfg.d_k + cfg.n_v_heads * cfg.d_v) * 4)
        kv_tokens = self.max_context
        if self.kv_host:
            kv = 0
        elif self.kv_int8:
            kv = (batch_size * len(cfg.attention_layer_indices()) * 2
                  * cfg.num_kv_heads * kv_tokens * (cfg.head_dim + 2))
        else:
            kv = (batch_size * len(cfg.attention_layer_indices()) * 2
                  * cfg.num_kv_heads * kv_tokens * cfg.head_dim * 2)
        host_kv = (batch_size * len(cfg.attention_layer_indices()) * 2
                   * cfg.num_kv_heads * kv_tokens * cfg.head_dim * 2
                   if self.kv_host else 0)
        resident = body + lm + standard_norms + qk_norms + delta_aux + rope
        runtime = recurrent + conv_state + kv
        return {
            "int3_body": body,
            "int3_lm_head": lm,
            "fp32_norms_qknorm_delta_aux": standard_norms + qk_norms + delta_aux,
            f"bf16_rope_seq{self.max_context}": rope,
            "host_bf16_embedding_ram": int(self.embed_tokens.weight_u16.nbytes),
            f"host_bf16_kv_seq{self.max_context}": host_kv,
            "fp32_deltanet_state_batch1": recurrent + conv_state,
            f"device_kv_prealloc_seq{self.max_context}": kv,
            "resident_vram": resident,
            f"total_vram_batch1_seq{self.max_context}": resident + runtime,
        }

    @staticmethod
    def print_vram_map(vram_map):
        print("PER-COMPONENT MEMORY MAP (GiB; computed from loaded tensors)")
        for name, value in vram_map.items():
            location = "HOST" if name.startswith("host_") else "VRAM"
            print(f"  {name:42s} {value / 2**30:9.4f} GiB  {location}")

    def load_weights(self, model_dir=None, cache_dir=DEFAULT_CACHE_DIR, progress=True,
                     cache_read_only=False):
        from safetensors import safe_open

        d = os.path.abspath(model_dir or DEFAULT_MODEL_DIR)
        cfg = self.config
        cache = INT3PackCache(d, cache_dir, bits=3, group_size=GROUP_SIZE)
        # Gate/receipt runs must never enter the resumable cache builder: an
        # empirical CUDA OOM may interrupt model loading, but the pack cache is
        # already complete and is opened read-only for these runs.
        manifest = (cache.load_manifest() if cache_read_only else
                    cache.ensure_complete(progress=progress))
        where = {n: os.path.join(d, rel) for n, rel in cache.weight_map.items()}

        def gf(name):
            with safe_open(where[name], framework="pt", device="cpu") as fh:
                return fh.get_tensor(name).float().numpy()

        def qlinear(name):
            entry, packed, scales, zeros = cache.load_quant(name, manifest)
            if name == "lm_head.weight":
                return PackedRowChunkedQuantLinearTC(
                    entry, packed, scales, zeros, name,
                    chunk_rows=self.lm_head_chunk_rows)
            return CachedQuantLinearTC.from_arrays(entry, packed, scales, zeros, name)

        emb_entry = manifest["entries"][cache.embedding_name]
        self.embed_tokens.weight_u16 = np.load(
            cache.cache_dir / emb_entry["file"], mmap_mode="r")
        P = "model.language_model"
        for i in range(cfg.num_layers):
            p = f"{P}.layers.{i}"
            layer = self.layers[i]
            layer.input_layernorm.weight = tc.tensor(np.ascontiguousarray(
                1.0 + gf(f"{p}.input_layernorm.weight")), dtype="float32")
            layer.post_attention_layernorm.weight = tc.tensor(np.ascontiguousarray(
                1.0 + gf(f"{p}.post_attention_layernorm.weight")), dtype="float32")
            mx = layer.mixer
            if layer.is_attn:
                a = f"{p}.self_attn"
                mx.q_proj = qlinear(f"{a}.q_proj.weight")
                mx.k_proj = qlinear(f"{a}.k_proj.weight")
                mx.v_proj = qlinear(f"{a}.v_proj.weight")
                mx.o_proj = qlinear(f"{a}.o_proj.weight")
                mx.q_norm_w = tc.tensor(np.ascontiguousarray(
                    1.0 + gf(f"{a}.q_norm.weight")), dtype="float32")
                mx.k_norm_w = tc.tensor(np.ascontiguousarray(
                    1.0 + gf(f"{a}.k_norm.weight")), dtype="float32")
            else:
                a = f"{p}.linear_attn"
                mx.in_proj_qkv = qlinear(f"{a}.in_proj_qkv.weight")
                mx.in_proj_z = qlinear(f"{a}.in_proj_z.weight")
                mx.in_proj_a = F32Linear(gf(f"{a}.in_proj_a.weight"))
                mx.in_proj_b = F32Linear(gf(f"{a}.in_proj_b.weight"))
                mx.out_proj = qlinear(f"{a}.out_proj.weight")
                cw = gf(f"{a}.conv1d.weight").reshape(-1, cfg.conv_kernel)
                mx.conv_w = [tc.tensor(np.ascontiguousarray(
                    cw[:, j].reshape(1, 1, -1))) for j in range(cfg.conv_kernel)]
                mx.neg_A = tc.tensor(np.ascontiguousarray(
                    (-np.exp(gf(f"{a}.A_log"))).reshape(1, 1, -1)))
                mx.dt_bias = tc.tensor(np.ascontiguousarray(
                    gf(f"{a}.dt_bias").reshape(1, 1, -1)))
                mx.norm_w = tc.tensor(np.ascontiguousarray(
                    gf(f"{a}.norm.weight")), dtype="float32")
            mlp = f"{p}.mlp"
            layer.mlp.gate_proj = qlinear(f"{mlp}.gate_proj.weight")
            layer.mlp.up_proj = qlinear(f"{mlp}.up_proj.weight")
            layer.mlp.down_proj = qlinear(f"{mlp}.down_proj.weight")
            gc.collect()
            if progress:
                print(f"LOAD layer {i + 1}/{cfg.num_layers}", flush=True)
        self.norm.weight = tc.tensor(np.ascontiguousarray(
            1.0 + gf(f"{P}.norm.weight")), dtype="float32")
        self.lm_head = qlinear("lm_head.weight")
        self.vram_map = self._computed_vram_map()
        self.print_vram_map(self.vram_map)
        return {
            "loaded": "INT3 Qwen3.8 hybrid from pack cache",
            "framework": "tensor_cuda Qwen3.8",
            "weight_bits": 3,
            "cache_dir": str(cache.cache_dir),
            "cache_access": "read_only" if cache_read_only else "read_or_build",
            "model_dir": d,
            "layers": cfg.num_layers,
            "attention_layers": cfg.attention_layer_indices(),
            "output_gate_type": cfg.output_gate_type,
            "lm_head_chunk_rows": self.lm_head_chunk_rows,
            "lm_head_chunks": len(self.lm_head.chunks),
            "lm_head_dequant_transient_bytes": self.lm_head.dequant_transient_bytes(),
            "max_context": self.max_context,
            "prefill_chunk": self.prefill_chunk,
            "kv_block_rows": self.kv_block_rows,
            "kv_mode": "host_bf16" if self.kv_host else
                       ("device_int8" if self.kv_int8 else "device_bf16"),
            "attn_path": self.attn_path,
            "vram_map": self.vram_map,
        }

    @classmethod
    def from_pretrained(cls, model_dir=DEFAULT_MODEL_DIR,
                        cache_dir=DEFAULT_CACHE_DIR, run_fused_gate=False,
                        lm_head_chunk_rows=DEFAULT_LM_HEAD_CHUNK_ROWS,
                        max_context=DEFAULT_MAX_CONTEXT, kv_int8=False,
                        kv_host=False, prefill_chunk=DEFAULT_PREFILL_CHUNK,
                        kv_block_rows=DEFAULT_KV_BLOCK,
                        force_tiled_attention=False,
                        cache_read_only=False,
                        output_gate_type=None):
        BlockTC.COMPUTE_DTYPE = "bfloat16"
        LinearTC.DTYPE = "bfloat16"
        QuantLinearTC.set_weight_bits(3)
        RMSNormTC.USE_FUSED = True
        fused_status = "disabled: canonical INT3 parity defect"
        fused_ok = False
        if run_fused_gate:
            try:
                gate_int3_fused_real_shapes()
                fused_ok = True
                fused_status = "enabled: real-shape parity PASS"
            except Exception as exc:
                fused_status = f"disabled: {type(exc).__name__}: {exc}"
                print("INT3_FUSED_FALLBACK " + fused_status, flush=True)
        QuantLinearTC.FUSED_DECODE = fused_ok
        with tc.no_grad():
            cfg = Qwen38Config.from_model_dir(model_dir)
            if output_gate_type is not None:
                cfg.output_gate_type = output_gate_type
                cfg._validate_qwen38()
            model = cls(
                cfg, lm_head_chunk_rows=lm_head_chunk_rows,
                max_context=max_context, kv_int8=kv_int8, kv_host=kv_host,
                prefill_chunk=prefill_chunk, kv_block_rows=kv_block_rows,
                force_tiled_attention=force_tiled_attention)
            info = model.load_weights(
                model_dir, cache_dir, cache_read_only=cache_read_only)
            # LC1 contract: a successful load has already reserved the complete
            # configured KV capacity; prompt execution consumes this first set.
            model._preallocated_caches = model.new_caches(1)
            info["kv_preallocated"] = True
        info["fused_decode"] = fused_status
        return model, info


def gate_int3_fused_real_shapes(seed=3802):
    """GPU gate fused INT3 against two-stage at representative real shapes."""
    rng = np.random.default_rng(seed)
    shapes = [(6144, 5120), (5120, 17408), (248320, 5120)]
    results = []
    for out_features, in_features in shapes:
        packed = rng.integers(
            0, 256, size=(out_features, in_features * 3 // 8), dtype=np.uint8)
        groups = in_features // GROUP_SIZE
        scales = (rng.random((out_features, groups), dtype=np.float32)
                  * 0.02 + 1e-4).astype(np.float16)
        zeros = (rng.standard_normal((out_features, groups)).astype(np.float32)
                 * 0.01).astype(np.float16)
        x = rng.standard_normal((1, in_features)).astype(np.float32) * 0.1
        with tc.no_grad():
            xt = tc.tensor(x, dtype="float32").astype("bfloat16")
            pt = tc.tensor(packed, dtype="uint8")
            st = tc.tensor(scales, dtype="float16")
            zt = tc.tensor(zeros, dtype="float16")
            two = tc.intn_linear(xt, pt, st, zt, 3, in_features, GROUP_SIZE).float().numpy()
            fused = tc.intn_linear_fused(
                xt, pt, st, zt, 3, in_features, GROUP_SIZE).float().numpy()
        diff = np.abs(two - fused)
        rel_l2 = float(np.linalg.norm(diff) / (np.linalg.norm(two) + 1e-12))
        max_abs = float(diff.max(initial=0))
        if not np.isfinite(rel_l2) or rel_l2 > 1e-3:
            QuantLinearTC.FUSED_DECODE = False
            raise AssertionError(
                f"INT3 fused parity FAIL shape=1x{in_features}x{out_features} "
                f"rel_l2={rel_l2:.6g} max_abs={max_abs:.6g}"
            )
        results.append({"shape": [1, in_features, out_features],
                        "rel_l2": rel_l2, "max_abs": max_abs})
        del packed, scales, zeros, x, xt, pt, st, zt, two, fused
        gc.collect()
    print("INT3_FUSED_GATE " + json.dumps(results, sort_keys=True), flush=True)
    return results


def dequantized_cache_linear(x, cache: INT3PackCache, name, manifest=None,
                             row_chunk=1024):
    """Bounded-RAM fp32 CPU linear over cached INT3 weights (G2 oracle)."""
    entry, packed, scales, zeros = cache.load_quant(name, manifest)
    rows, cols = (int(v) for v in entry["shape"])
    x2 = np.asarray(x, dtype=np.float32).reshape(-1, cols)
    out = np.empty((x2.shape[0], rows), dtype=np.float32)
    groups = cols // entry["group_size"]
    for lo in range(0, rows, row_chunk):
        hi = min(rows, lo + row_chunk)
        q = unpack_int3_vectorized(packed[lo:hi], cols).astype(np.float32)
        w = q.reshape(hi - lo, groups, entry["group_size"])
        w = w * scales[lo:hi].astype(np.float32)[:, :, None]
        w = w + zeros[lo:hi].astype(np.float32)[:, :, None]
        out[:, lo:hi] = x2 @ w.reshape(hi - lo, cols).T
    return out.reshape(np.asarray(x).shape[:-1] + (rows,))


def compute_qwen38_memory_budget(
        model_dir=DEFAULT_MODEL_DIR, context_tokens=DEFAULT_MAX_CONTEXT,
        batch_size=1, group_size=GROUP_SIZE, kv_int8=False, kv_host=False,
        lm_head_chunk_rows=DEFAULT_LM_HEAD_CHUNK_ROWS,
        prefill_chunk=DEFAULT_PREFILL_CHUNK, kv_block_rows=DEFAULT_KV_BLOCK):
    """Compute persistent storage and a receipt-calibrated whole-device peak."""
    from safetensors import safe_open

    cache = INT3PackCache(model_dir, DEFAULT_CACHE_DIR, 3, group_size)
    cfg = cache.cfg
    shapes = {}
    by_shard = {}
    for name in cache.required_names:
        by_shard.setdefault(cache.weight_map[name], []).append(name)
    for rel, names in by_shard.items():
        with safe_open(cache.model_dir / rel, framework="pt", device="cpu") as fh:
            for name in names:
                shapes[name] = tuple(int(x) for x in fh.get_slice(name).get_shape())
    body = lm = 0
    for name in cache.qlinear_names:
        rows, cols = shapes[name]
        size = rows * (cols * 3 // 8) + rows * (cols // group_size) * 4
        if name == "lm_head.weight":
            lm = size
        else:
            body += size
    standard_norms = (2 * cfg.num_layers * cfg.hidden_dim + cfg.hidden_dim) * 4
    attn_layers = len(cfg.attention_layer_indices())
    delta_layers = cfg.num_layers - attn_layers
    qk_norms = attn_layers * 2 * cfg.head_dim * 4
    delta_aux = delta_layers * (
        2 * cfg.hidden_dim * cfg.n_v_heads * 4
        + (2 * cfg.n_k_heads * cfg.d_k + cfg.n_v_heads * cfg.d_v)
          * cfg.conv_kernel * 4
        + 3 * cfg.n_v_heads * 4 + cfg.d_v * 4)
    context_tokens = int(context_tokens)
    if not 1 <= context_tokens <= cfg.max_position_embeddings:
        raise ValueError(
            f"max context {context_tokens} outside checkpoint window "
            f"[1,{cfg.max_position_embeddings}]")
    if kv_int8 and kv_host:
        raise ValueError("kv_int8 and kv_host are mutually exclusive")
    if lm_head_chunk_rows <= 0 or prefill_chunk <= 0 or kv_block_rows <= 0:
        raise ValueError("chunk sizes must be positive")
    rope = 2 * context_tokens * cfg.partial_rotary_dim * 2
    recurrent = (batch_size * delta_layers * cfg.n_v_heads
                 * cfg.d_k * cfg.d_v * 4)
    conv_state = (batch_size * delta_layers * (cfg.conv_kernel - 1)
                  * (2 * cfg.n_k_heads * cfg.d_k + cfg.n_v_heads * cfg.d_v) * 4)
    bf16_kv = (batch_size * attn_layers * 2 * cfg.num_kv_heads * context_tokens
               * cfg.head_dim * 2)
    int8_kv = (batch_size * attn_layers * 2 * cfg.num_kv_heads * context_tokens
               * (cfg.head_dim + 2))
    device_kv = 0 if kv_host else (int8_kv if kv_int8 else bf16_kv)
    host_kv = bf16_kv if kv_host else 0
    host = math.prod(shapes[cache.embedding_name]) * 2
    resident = body + lm + standard_norms + qk_norms + delta_aux + rope
    lm_transient = min(int(lm_head_chunk_rows), cfg.vocab_size) * cfg.hidden_dim * 2
    # The online standard-attention implementation stages one KV block and a
    # fixed query-block score tile. Use fp32 score/accumulator bytes here.
    qrows = min(int(prefill_chunk), context_tokens)
    brows = min(int(kv_block_rows), context_tokens)
    streaming_workspace = (
        2 * batch_size * cfg.num_kv_heads * brows * cfg.head_dim * 2
        + batch_size * cfg.num_heads * qrows * brows * 4
        + batch_size * cfg.num_heads * qrows * cfg.head_dim * 4
    )
    fixed_overhead = CALIBRATED_FIXED_PEAK_OVERHEAD_MIB * 2**20
    projected_peak = (resident + recurrent + conv_state + device_kv
                      + lm_transient + streaming_workspace + fixed_overhead)
    kv_mode = "host_bf16" if kv_host else ("device_int8" if kv_int8 else
                                             "device_bf16")
    return {
        "int3_body": body,
        "int3_lm_head": lm,
        "fp32_norms_qknorm_delta_aux": standard_norms + qk_norms + delta_aux,
        f"bf16_rope_seq{context_tokens}": rope,
        "host_bf16_embedding_ram": host,
        f"host_bf16_kv_seq{context_tokens}": host_kv,
        "fp32_deltanet_state_batch1": recurrent + conv_state,
        f"{kv_mode}_kv_seq{context_tokens}": device_kv,
        "lm_head_bf16_dequant_transient": lm_transient,
        "bounded_attention_workspace": streaming_workspace,
        "calibrated_fixed_peak_overhead_including_desktop": fixed_overhead,
        "resident_vram": resident,
        f"persistent_vram_batch1_seq{context_tokens}": (
            resident + recurrent + conv_state + device_kv),
        "projected_whole_device_peak": projected_peak,
    }


def validate_qwen38_memory_budget(budget, ceiling_mib=VRAM_CEILING_MIB):
    peak = int(budget["projected_whole_device_peak"])
    ceiling = int(ceiling_mib) * 2**20
    if peak > ceiling:
        raise MemoryError(
            f"projected whole-device peak {peak / 2**20:.1f} MiB exceeds "
            f"fixed ceiling {ceiling_mib} MiB; select --kv-int8 or --kv-host, "
            f"or reduce --max-context")
    return peak
