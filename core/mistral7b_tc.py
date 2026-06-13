"""Mistral 7B implemented natively on the tensor_cuda C++/CUDA engine.

No CuPy, no PyTorch in the forward path. INT4 group-quantized projections via the
engine's native int4_linear kernel; embedding + lm_head kept fp16; RMSNorm fp32.
Switchable attention backend, both native tensor_cuda ops:

  - "standard" : tc.functional.scaled_dot_product_attention (full fp16 SDPA)
  - "apa"      : tc.apa_quant_attention (APA-Quant: 2-bit bulk keys + z-score refine)

This mirrors core/mistral7b.py (the CuPy reference) one-for-one so the two engines
can be compared at identical model state. The weight quantization scheme is byte-
identical to core/quantized_linear.py; only the forward (dequant+matmul) runs on
tensor_cuda instead of CuPy.
"""

import sys
import os
import gc
from typing import Optional

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
import tensor_cuda as tc
from tensor_cuda import functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adapters.tokenizer import MistralConfig

ATTENTION_MODES = ("standard", "apa", "apa_selective")
GROUP_SIZE = 128


def _normalize_mode(mode) -> str:
    if isinstance(mode, bool):
        return "apa" if mode else "standard"
    if mode in ("cupy_apa", "apa_quant"):
        mode = "apa"
    if mode not in ATTENTION_MODES:
        raise ValueError(f"Unknown attention mode {mode!r}; expected {ATTENTION_MODES}")
    return mode


def _quantize_int4(w_fp32: np.ndarray, group_size: int = GROUP_SIZE):
    """Per-group 4-bit min/max quantization. Byte-identical to QuantizedLinear.

    Returns (packed uint8 (N, K/2), scales fp16 (N, G), zeros fp16 (N, G)).
    """
    out_features, in_features = w_fp32.shape
    if in_features % group_size != 0 or in_features % 2 != 0:
        raise ValueError(f"bad dims for int4: ({out_features}, {in_features})")
    num_groups = in_features // group_size
    w_grouped = w_fp32.reshape(out_features, num_groups, group_size)
    mins = w_grouped.min(axis=2)
    maxs = w_grouped.max(axis=2)
    scales = (maxs - mins) / 15.0
    scales = np.where(scales == 0, np.ones_like(scales), scales)
    zeros = mins
    w_norm = (w_grouped - zeros[:, :, None]) / scales[:, :, None]
    w_int4 = np.clip(np.round(w_norm), 0, 15).astype(np.uint8)
    w_int4_flat = w_int4.reshape(out_features, in_features)
    even = w_int4_flat[:, 0::2]
    odd = w_int4_flat[:, 1::2]
    packed = (even | (odd << 4)).astype(np.uint8)
    return packed, scales.astype(np.float16), zeros.astype(np.float16)


class QuantLinearTC:
    """INT4 group-quantized linear on tensor_cuda. Weights live on-device as
    packed uint8 + fp16 scales/zeros; forward is the native int4_linear kernel."""

    # Opt-in fused dequant-GEMM (#12): avoids materializing the full (K,N) fp16
    # weight transient by dequantizing inside the GEMM. False = the validated
    # cuBLAS two-stage path. Flip to True to benchmark / use the fused path.
    USE_FUSED = False
    # Shape-aware variant: fused ONLY at GEMV-like shapes (M <= threshold).
    # At M~1 (decode) two-stage dequantizes the whole weight to multiply one
    # row (~3GB of dequant writes/token on MiniCPM3); fused reads packed INT4
    # directly. At large M (prefill) fused has no tensor-core path and LOSES
    # to two-stage cuBLAS — so the pick must be per-call, not global.
    FUSED_DECODE = False
    FUSED_M_MAX = 8

    def __init__(self, weight_fp32: np.ndarray, group_size: int = GROUP_SIZE):
        self.out_features, self.in_features = weight_fp32.shape
        self.group_size = group_size
        packed, scales, zeros = _quantize_int4(weight_fp32, group_size)
        self.packed = tc.tensor(packed, dtype="uint8")
        self.scales = tc.tensor(scales, dtype="float16")
        self.zeros = tc.tensor(zeros, dtype="float16")
        self._vram = packed.nbytes + scales.nbytes + zeros.nbytes

    def __call__(self, x):
        fused = QuantLinearTC.USE_FUSED
        if not fused and QuantLinearTC.FUSED_DECODE:
            M = 1
            for d in x.shape[:-1]:
                M *= d
            fused = M <= QuantLinearTC.FUSED_M_MAX
        if fused:
            return tc.int4_linear_fused(x, self.packed, self.scales, self.zeros, self.group_size)
        return tc.int4_linear(x, self.packed, self.scales, self.zeros, self.group_size)

    def vram_bytes(self) -> int:
        return self._vram


class LinearTC:
    """Plain linear (no quantization): y = x @ W^T (+ bias). For unquantized
    models. bias optional (Qwen has q/k/v bias; Llama/Mistral don't). dtype can be
    'float16' or 'bfloat16' (bf16 needed for models whose activations overflow
    fp16's ~65504 range, e.g. Qwen2.5)."""

    DTYPE = "float16"  # class default; set to "bfloat16" for bf16-native models

    def __init__(self, weight_fp32: np.ndarray, bias_fp32: np.ndarray = None):
        self.out_features, self.in_features = weight_fp32.shape
        dt = LinearTC.DTYPE
        # store W^T (in,out) so forward is a direct matmul, no transpose per call
        self.wT = tc.tensor(np.ascontiguousarray(weight_fp32.T.astype(np.float32))).astype(dt)
        self._vram = self.out_features * self.in_features * 2
        self.bias = None
        if bias_fp32 is not None:
            self.bias = tc.tensor(np.ascontiguousarray(bias_fp32.astype(np.float32))).astype(dt)
            self._vram += self.out_features * 2

    def __call__(self, x):
        y = tc.matmul(x, self.wT)
        return y + self.bias if self.bias is not None else y

    def vram_bytes(self) -> int:
        return self._vram


class EmbeddingTC:
    """FP16 embedding via the engine's native embedding gather."""

    def __init__(self, num_embeddings, embedding_dim):
        self.weight = tc.tensor(
            np.zeros((num_embeddings, embedding_dim), np.float16), dtype="float16")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def __call__(self, idx_int64):
        return tc.embedding(self.weight, idx_int64)


class RMSNormTC:
    """RMSNorm in fp32 (weight fp32); caller casts the result back to fp16.

    USE_FUSED routes inference through the engine's single-kernel rms_norm
    (fp32 accumulate, output in x's dtype — the caller's cast becomes a
    no-op). Falls back to the unfused chain under grad (fused has no
    backward) or on engines without the op.
    """

    USE_FUSED = False

    def __init__(self, dim, eps=1e-5):
        self.weight = tc.tensor(np.ones(dim, np.float32), dtype="float32")
        self.eps = eps

    def __call__(self, x):
        if (self.USE_FUSED and hasattr(tc, "rms_norm")
                and not tc.is_grad_enabled()):
            return tc.rms_norm(x, self.weight, self.eps)
        xf = x.float()  # compute in fp32 for stability
        ms = (xf * xf).mean([-1], True)
        return xf * (ms + self.eps).pow(-0.5) * self.weight


def _repeat_kv(x, rep):
    """(B, KVH, L, D) -> (B, KVH*rep, L, D), each kv head repeated rep times."""
    if rep == 1:
        return x
    B, KVH, L, D = x.shape
    return x.unsqueeze(2).expand([B, KVH, rep, L, D]).reshape([B, KVH * rep, L, D])


def _kv_quant(x):
    """Symmetric INT8 quantize (B,H,S,D) over the last dim. Returns (uint8, scale
    fp16). Stored as 1 byte/elem + one fp16 scale per (B,H,S) vector — halves the
    KV cache footprint vs fp16. q = round(x/scale)+128, scale = max|x|/127."""
    scale = x.abs().max([-1], True) * (1.0 / 127.0)
    scale = (scale + 1e-8)  # avoid div-by-zero on all-zero vectors
    q = (x / scale + 128.0).astype("uint8")   # device round+clamp to [0,255]
    # scale carried in the compute dtype so dequant matches the model (bf16 for
    # Qwen/OLMoE, fp16 for Llama) — a hardcoded fp16 here breaks bf16 models on
    # the KV-cache decode path (matmul dtype mismatch).
    cdt = BlockTC.COMPUTE_DTYPE
    scale = scale.half() if cdt == "float16" else scale.astype(cdt)
    return q, scale


def _kv_dequant(q, scale):
    """Inverse of _kv_quant -> compute dtype (B,H,S,D). Must match the model's
    compute dtype (bf16 for Qwen/OLMoE) or the attention matmul dtype-mismatches."""
    cdt = BlockTC.COMPUTE_DTYPE
    qd = q.half() if cdt == "float16" else q.astype(cdt)
    return (qd - 128.0) * scale




def _cublas_blend_attention(q, kH, kq_kv, v_kv, group, scale, z, causal, blk):
    """Selective APA via cuBLAS matmuls + the fused apa_blend_softmax kernel,
    tiled over query blocks. q (B,H,L,D); kH expanded to H heads; kq_kv/v_kv at
    KVH heads (expanded here for the GEMM). ~6x the per-row fused kernel.

    bulk = q @ kq^T, rank = q @ k^T (cuBLAS), then one fused kernel does
    threshold+select+softmax, then weights @ v (cuBLAS). Causal masking is baked
    into the score blocks as large-negative additive bias.
    """
    B, H, L, D = q.shape
    S = kH.shape[2]
    if kH.shape[1] == 1 and H > 1:
        # MQA de-expansion (Gemma 4 globals: 1 KV head x 16 q heads):
        # fold q heads into rows against the UNEXPANDED cache — pure
        # reshapes. The expansion path materialized 16 copies of K,
        # quantized-K AND V (3 x ~200MB per chunk at 12K; OOM-proven,
        # the 2x-ceiling loss in the first rematch probe).
        kqT1 = kq_kv.transpose(-2, -1)            # (B,1,D,S)
        kT1 = kH.transpose(-2, -1)
        outs = []
        for i in range(0, L, blk):
            e = min(i + blk, L)
            bl = e - i
            Qf = q.slice(2, i, bl).reshape([B, 1, H * bl, D])
            bulk = (tc.matmul(Qf, kqT1).reshape([B, H, bl, S])
                    * scale)
            rank = (tc.matmul(Qf, kT1).reshape([B, H, bl, S])
                    * scale)
            if causal:
                ca = F._causal_mask(L, S, q.device.split(":")[0],
                                    bulk.dtype).slice(0, i, bl)
                bulk = bulk + ca
                rank = rank + ca
            w = tc.apa_blend_softmax(bulk, rank, z)
            outs.append(tc.matmul(w.reshape([B, 1, H * bl, S]),
                                  v_kv).reshape([B, H, bl, D]))
        return tc.cat(outs, dim=2)
    kqH = _repeat_kv(kq_kv, group)
    vH = _repeat_kv(v_kv, group)
    kqT = kqH.transpose(-2, -1)
    kT = kH.transpose(-2, -1)
    outs = []
    for i in range(0, L, blk):
        e = min(i + blk, L)
        Qi = q.slice(2, i, e - i)
        bulk = tc.matmul(Qi, kqT) * scale
        rank = tc.matmul(Qi, kT) * scale
        if causal:
            # ONE canonical mask: rows [i, e) of functional._causal_mask
            # (bottom-right aligned, cached per shape). This path used to
            # build its OWN triu(k=i+1) — the top-left rectangle bug that
            # _causal_mask's docstring records FIXING in the standard
            # path was alive here the whole time because the logic was
            # duplicated (caught 2026-06-12 by the Gemma 4 refine ppl
            # sweep: ppl 121 -> 11M on cached chunks; every prior APA
            # gate ran square shapes where the two constructions agree).
            ca = F._causal_mask(L, S, q.device.split(":")[0],
                                bulk.dtype).slice(0, i, e - i)
            bulk = bulk + ca
            rank = rank + ca
        w = tc.apa_blend_softmax(bulk, rank, z)
        outs.append(tc.matmul(w, vH))
    return tc.cat(outs, dim=2)


class GQAAttentionTC:
    def __init__(self, config: MistralConfig, attention_mode: str = "standard"):
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_heads_per_kv = config.num_heads_per_kv
        self.attention_mode = _normalize_mode(attention_mode)
        self.refine_percentile = 0.10
        self.bulk_bits = 2          # 2-bit bulk works for normalized-key models;
                                    # outlier-key models (Qwen) need more (4-8)
        self.quant_kv_cache = True  # store KV cache as INT8 (halves its footprint)
        self.q_norm = None          # optional QK-norm (OLMoE); RMSNormTC or None
        self.k_norm = None
        self.fast_attn = True       # cuBLAS-blend selective path (~6x the fused kernel)
        self.fast_max_seq = 4096    # use fast path up to this S; fused above (for ceiling)
        self.attn_block = 1024      # query-block size for the tiled cuBLAS-blend path
        # Decode amortization (#9): in incremental decode the growing key cache is
        # re-rotated+re-quantized for APA every step even though a written key's
        # quantization never changes. When enabled, cache the APA-quantized keys
        # and quantize ONLY the newly appended keys. Off by default so the
        # prefill/perplexity/ceiling benchmarks (validated) are byte-identical;
        # the incremental generate path turns it on.
        self.cache_apa_kq = True
        self.q_proj = self.k_proj = self.v_proj = self.o_proj = None

    def __call__(self, x, cos, sin, position_offset=0, kv_cache=None):
        B, L, _ = x.shape
        qp = self.q_proj(x)
        kp = self.k_proj(x)
        # Optional QK-norm (OLMoE etc.): RMSNorm over the full projection before
        # splitting into heads. self.q_norm/k_norm are RMSNormTC or None.
        if getattr(self, "q_norm", None) is not None:
            qp = self.q_norm(qp)
            kp = self.k_norm(kp)
            cdt = BlockTC.COMPUTE_DTYPE
            qp = qp.half() if cdt == "float16" else qp.astype(cdt)
            kp = kp.half() if cdt == "float16" else kp.astype(cdt)
        # Qwen3-style PER-HEAD QK-norm: RMSNorm over head_dim, applied independently
        # to each head's vector (weight shape (head_dim,)), in fp32. Different from
        # the OLMoE path above (which norms the full hidden projection).
        if getattr(self, "_use_qwen3_qknorm", False):
            cdt = BlockTC.COMPUTE_DTYPE
            def _phn(t, w, nh):
                tr = t.reshape([B, L, nh, self.head_dim]).float()
                ms = (tr * tr).mean([-1], True)
                tn = tr * (ms + self._qk_eps).pow(-0.5) * w
                tn = tn.half() if cdt == "float16" else tn.astype(cdt)
                return tn.reshape([B, L, nh * self.head_dim])
            qp = _phn(qp, self._qk_w, self.num_heads)
            kp = _phn(kp, self._k_qk_w, self.num_kv_heads)
        q = qp.reshape([B, L, self.num_heads, self.head_dim]).transpose(1, 2)
        k = kp.reshape([B, L, self.num_kv_heads, self.head_dim]).transpose(1, 2)
        v = self.v_proj(x).reshape([B, L, self.num_kv_heads, self.head_dim]).transpose(1, 2)

        # KV-Graft capture: stash this layer's k,v BEFORE RoPE (no position baked
        # in), plus v. Capturing pre-RoPE means the harvested keys are
        # position-neutral and can be re-placed at the injection prefix's
        # positions, instead of carrying the document's original positions (which
        # scrambles the relative-position phase when injected elsewhere).
        if getattr(self, "_capture", False):
            self._captured = (k.numpy(), v.numpy())
        # Router capture: pre-RoPE post-norm queries, same position-neutral
        # space as harvested graft keys — dot(q_cap, k_harvest) is the
        # Graft-Repository routing score (E1).
        if getattr(self, "_capture_q", False):
            self._captured_q = q.numpy()

        # KV-Graft inject: splice harvested document k,v in FRONT, as a real
        # positional prefix. The grafted keys get RoPE at positions 0..Sg-1 and
        # the real tokens shift to Sg.., so relative position between query and
        # graft is consistent (the cat happens AFTER RoPE below). inject_kv =
        # (k_graft_preRoPE, v_graft, scale) or None.
        inj = getattr(self, "inject_kv", None)
        kg_pre = None
        if inj is not None:
            kg_pre, vg, gscale = inj

        # RoPE on q,k for this segment's absolute positions. The real tokens sit
        # AFTER the graft's Sg seats, so shift their positions by Sg — on EVERY
        # step the graft is resident, not just the injection step. `graft_seats`
        # is set once when the graft is mounted (prefill) and persists through
        # cached decode (where kg_pre is None because the cat already happened);
        # deriving Sg from kg_pre alone collapsed the shift to 0 on decode and
        # collided live tokens onto the graft's seats (multi-turn cache bug).
        Sg = getattr(self, "graft_seats", 0)
        if kg_pre is not None:
            Sg = kg_pre.shape[2]
        # Arena mode (live_shift set): live positions shift by the FIXED
        # sink+arena width — mounts occupy a PREFIX of the arena and the
        # unused remainder is a positional hole, so swapping mounts never
        # moves live tokens (same law as MLAAttentionTC). Without
        # live_shift: legacy behavior, shift = mount size.
        shift = getattr(self, "live_shift", None)
        if shift is None:
            shift = Sg
        cseg = cos.slice(0, position_offset + shift, L)
        sseg = sin.slice(0, position_offset + shift, L)
        q = F.apply_rotary(q, cseg, sseg)
        k = F.apply_rotary(k, cseg, sseg)

        if inj is not None and kv_cache is None:
            # PREFILL-ONLY injection: the graft enters the K/V stream exactly
            # once, here. On cached decode steps (kv_cache is not None) the
            # graft is ALREADY inside the cached span — re-catting it every
            # step duplicated the graft and scrambled positions (the doc's
            # prerequisite-#1 footgun). The +Sg position shift above still
            # applies on every step: the graft keeps its seats 0..Sg-1.
            # RoPE the grafted keys at prefix positions 0..Sg-1, then cat in front.
            cg = cos.slice(0, 0, Sg)
            sg = sin.slice(0, 0, Sg)
            kg = F.apply_rotary(kg_pre, cg, sg)
            if gscale != 1.0:
                kg = kg * gscale
            k = tc.cat([kg, k], dim=2)
            v = tc.cat([vg, v], dim=2)

        # The APA-quantized keys for the PRIOR cached span (carried as a 5th/3rd
        # cache element when caching is on) — keeps decode from re-quantizing the
        # whole growing cache every step. `k_new` is just this segment's keys.
        cached_kq = None
        k_new = k
        if kv_cache is not None:
            # Cache may be stored INT8-quantized (k_u8, k_scale, v_u8, v_scale[,
            # apa_kq]) or plain fp16 (k, v[, apa_kq]). Dequant the quantized form.
            if len(kv_cache) >= 4:
                ck = _kv_dequant(kv_cache[0], kv_cache[1])
                cv = _kv_dequant(kv_cache[2], kv_cache[3])
                cached_kq = kv_cache[4] if len(kv_cache) >= 5 else None
            else:
                ck, cv = kv_cache[0], kv_cache[1]
                cached_kq = kv_cache[2] if len(kv_cache) >= 3 else None
            k = tc.cat([ck, k_new], dim=2)
            v = tc.cat([cv, v], dim=2)

        causal = (L > 1)
        if self.attention_mode == "apa_selective":
            # GQA-aware sparse selective APA: the fused kernel indexes the right KV
            # head per query head, so k/v/kq stay at num_kv_heads (8) — NOT expanded
            # to num_heads (32). Avoids materializing k_rep/v_rep and builds the
            # quantized keys at 8 heads instead of 32 (saves ~3 large transients).
            from tensor_cuda.quant import _tables, _quantize_keys, _norm_ppf
            import math as _m
            Dh = q.shape[3]
            KVH = k.shape[1]
            dev = q.device.split(":")[0]
            R, CB, BND = _tables(Dh, self.bulk_bits, KVH, True, dev)
            # #9: quantize only the NEW keys and concat onto the cached APA-kq when
            # available; otherwise quantize the whole span (prefill / no cache).
            if self.cache_apa_kq and cached_kq is not None:
                kq_new = _quantize_keys(k_new, R, CB, BND)
                kq = tc.cat([cached_kq, kq_new], dim=2)
            else:
                kq = _quantize_keys(k, R, CB, BND)   # (B, KVH, S, D) — unexpanded
            z = _norm_ppf(1.0 - max(0.0, min(1.0, self.refine_percentile)))
            scale = 1.0 / _m.sqrt(Dh)
            S = k.shape[2]
            # Two selective backends with a clean tradeoff:
            #  - cuBLAS-blend (fast_attn): ~6x faster kernel, but materializes
            #    several full H-head L-sized transients -> caps context ~8K.
            #  - fused GQA kernel: O(L) memory -> reaches ~24K, but slower.
            # Auto-pick: fast when the sequence is short enough that its transients
            # fit comfortably; fused above that to preserve the long-context ceiling.
            use_fast = self.fast_attn and S <= self.fast_max_seq
            if use_fast:
                Hq = q.shape[1]
                budget = 300 * 1024 * 1024  # ~300MB per score tensor
                blk = max(64, min(self.attn_block, int(budget // (Hq * S * 2))))
                attn = _cublas_blend_attention(
                    q, _repeat_kv(k, self.num_heads_per_kv), kq, v,
                    self.num_heads_per_kv, scale, float(z), causal, blk)
            else:
                attn = tc.apa_selective_attention(q, k, kq, v, scale, float(z), causal)
        elif self.attention_mode == "apa":
            k_r = _repeat_kv(k, self.num_heads_per_kv)
            v_r = _repeat_kv(v, self.num_heads_per_kv)
            attn = tc.apa_quant_attention(
                q, k_r, v_r, bulk_bits=2, refine_percentile=self.refine_percentile,
                is_causal=causal)
        else:
            k_r = _repeat_kv(k, self.num_heads_per_kv)
            v_r = _repeat_kv(v, self.num_heads_per_kv)
            attn = F.scaled_dot_product_attention(q, k_r, v_r, is_causal=causal)

        # Build the cache for the next step. Store KV INT8-quantized (halves its
        # footprint) when enabled. In apa_selective with caching on, also stash
        # the full-span APA-quantized keys so the next decode step only quantizes
        # its single new key (#9). kq for the whole span is exactly what we just
        # computed above.
        stash_kq = (self.attention_mode == "apa_selective"
                    and self.cache_apa_kq and L >= 1)
        if self.quant_kv_cache:
            kq8, ks = _kv_quant(k)
            vq8, vs = _kv_quant(v)
            new_kv = (kq8, ks, vq8, vs, kq) if stash_kq else (kq8, ks, vq8, vs)
        else:
            new_kv = (k, v, kq) if stash_kq else (k, v)

        attn = attn.transpose(1, 2).reshape([B, L, self.num_heads * self.head_dim])
        attn = attn.half() if BlockTC.COMPUTE_DTYPE == "float16" else attn.astype(BlockTC.COMPUTE_DTYPE)
        # Drop the large attention transients before o_proj so the caching
        # allocator can immediately reuse their device blocks for the projection
        # and the next layer, rather than growing peak VRAM. Each del decrements
        # the only remaining Python ref, so the Storage dtor returns the block to
        # the pool. Tensors still referenced by new_kv are NOT dropped: when the
        # KV cache is stored INT8 (the default), k/v were re-quantized into
        # new_kv and the fp16 originals are dead; otherwise new_kv aliases k/v so
        # they are left alone.
        del q, qp, kp
        try:
            del kq            # only bound in the apa_selective branch
        except NameError:
            pass
        try:
            del k_r, v_r      # only bound in the apa / standard branches
        except NameError:
            pass
        if self.quant_kv_cache:
            del k, v
        out = self.o_proj(attn)
        del attn
        return out, new_kv


class SwiGLU_TC:
    # FFN is position-wise (each token independent), so chunking over the sequence
    # is mathematically identical but never materializes the full (B, L, 14336)
    # gate/up intermediates (~448MB each at L=16k). Chunk when L exceeds this.
    FFN_CHUNK = 2048

    def __init__(self, hidden_dim, intermediate_dim):
        self.gate_proj = self.up_proj = self.down_proj = None

    def __call__(self, x):
        L = x.shape[1]
        if L <= self.FFN_CHUNK:
            return self.down_proj(self.gate_proj(x).silu() * self.up_proj(x))
        outs = []
        for i in range(0, L, self.FFN_CHUNK):
            end = min(i + self.FFN_CHUNK, L)
            xc = x.slice(1, i, end - i)
            outs.append(self.down_proj(self.gate_proj(xc).silu() * self.up_proj(xc)))
        return tc.cat(outs, dim=1)


class BlockTC:
    # Compute dtype for the residual stream / projections. "float16" for most
    # models; "bfloat16" for models whose activations exceed fp16's ~65504 range
    # by mid-network (Qwen2.5 etc.) — bf16 has fp32's exponent range.
    COMPUTE_DTYPE = "float16"

    def __init__(self, config: MistralConfig, attention_mode="standard"):
        self.input_layernorm = RMSNormTC(config.hidden_dim, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormTC(config.hidden_dim, config.rms_norm_eps)
        self.self_attn = GQAAttentionTC(config, attention_mode)
        self.mlp = SwiGLU_TC(config.hidden_dim, config.intermediate_dim)

    def _cast(self, t):
        dt = BlockTC.COMPUTE_DTYPE
        return t.half() if dt == "float16" else t.astype(dt)

    def __call__(self, x, cos, sin, position_offset=0, kv_cache=None):
        normed = self._cast(self.input_layernorm(x))
        attn, new_kv = self.self_attn(normed, cos, sin, position_offset, kv_cache)
        h = x + attn
        normed2 = self._cast(self.post_attention_layernorm(h))
        out = h + self.mlp(normed2)
        return out, new_kv


class Mistral7B_TC:
    """Full Mistral-7B on tensor_cuda with INT4 weights and switchable attention."""

    def __init__(self, config: MistralConfig, attention_mode="standard"):
        self.config = config
        self.attention_mode = _normalize_mode(attention_mode)
        self.embed_tokens = EmbeddingTC(config.vocab_size, config.hidden_dim)
        self.layers = [BlockTC(config, self.attention_mode) for _ in range(config.num_layers)]
        self.norm = RMSNormTC(config.hidden_dim, config.rms_norm_eps)
        self.lm_head = None
        self._build_rope(min(config.max_position_embeddings, 4096))

    def _build_rope(self, seq_len):
        hd = self.config.head_dim
        inv = 1.0 / (self.config.rope_theta ** (np.arange(0, hd, 2, np.float32) / hd))
        pos = np.arange(seq_len, dtype=np.float32)[:, None] * inv[None, :]
        emb = np.concatenate([pos, pos], axis=-1)
        dt = BlockTC.COMPUTE_DTYPE  # match the residual/attention dtype (fp16 or bf16)
        self.rope_cos = tc.tensor(np.cos(emb).astype(np.float32)).astype(dt)
        self.rope_sin = tc.tensor(np.sin(emb).astype(np.float32)).astype(dt)
        self._rope_len = seq_len

    def extend_rope(self, new_len):
        if new_len > self._rope_len:
            self._build_rope(new_len)

    def set_attention_mode(self, mode, refine_percentile=0.10):
        mode = _normalize_mode(mode)
        self.attention_mode = mode
        for layer in self.layers:
            layer.self_attn.attention_mode = mode
            layer.self_attn.refine_percentile = refine_percentile

    def __call__(self, input_ids_np, kv_caches=None, position_offset=0,
                 last_token_only=False):
        B, L = input_ids_np.shape
        self.extend_rope(position_offset + L)
        idx = tc.tensor(np.ascontiguousarray(input_ids_np.astype(np.int64)), dtype="int64")
        h = self.embed_tokens(idx)
        new_caches = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            h, kv = layer(h, self.rope_cos, self.rope_sin, position_offset, cache)
            new_caches.append(kv)
        h = self.norm(h)
        h = h.half() if BlockTC.COMPUTE_DTYPE == "float16" else h.astype(BlockTC.COMPUTE_DTYPE)
        if last_token_only and h.shape[1] > 1:
            # Generation/prefill only needs the final position's logits. Avoids the
            # (1, L, vocab) logits tensor (~750MB at L=12k, vocab=32k) — the
            # dominant peak-memory transient. Full logits are only needed for
            # training / per-token perplexity (last_token_only=False).
            h = h.slice(1, h.shape[1] - 1, 1)
        logits = self.lm_head(h)
        return logits, new_caches

    def measure_vram(self) -> dict:
        embed = self.embed_tokens.weight.numpy().nbytes
        proj = 0
        norm = 0
        for layer in self.layers:
            for p in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                      layer.self_attn.v_proj, layer.self_attn.o_proj,
                      layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
                if isinstance(p, QuantLinearTC):
                    proj += p.vram_bytes()
            norm += layer.input_layernorm.weight.numpy().nbytes
            norm += layer.post_attention_layernorm.weight.numpy().nbytes
        norm += self.norm.weight.numpy().nbytes
        lm = self.lm_head.vram_bytes() if isinstance(self.lm_head, QuantLinearTC) else 0
        total = embed + proj + norm + lm
        return {
            "embedding_mb": embed / 1024**2,
            "projections_mb": proj / 1024**2,
            "norms_mb": norm / 1024**2,
            "lm_head_mb": lm / 1024**2,
            "total_mb": total / 1024**2,
        }

    def load_weights_quantized(self, model_dir=None, group_size=GROUP_SIZE):
        from adapters.safetensors_loader import load_mistral_weights
        buf = {}
        load_mistral_weights(model_dir, on_tensor=lambda n, a: buf.__setitem__(n, a))

        orig_bytes = 0
        quant_bytes = 0
        loaded = 0

        embed_w = buf.pop("model.embed_tokens.weight")
        self.embed_tokens.weight = tc.tensor(
            np.ascontiguousarray(embed_w.astype(np.float16)), dtype="float16")
        quant_bytes += embed_w.astype(np.float16).nbytes
        loaded += 1
        del embed_w

        for i in range(self.config.num_layers):
            p = f"model.layers.{i}"
            layer = self.layers[i]
            ln1 = buf.pop(f"{p}.input_layernorm.weight")
            layer.input_layernorm.weight = tc.tensor(
                np.ascontiguousarray(ln1.astype(np.float32)), dtype="float32")
            ln2 = buf.pop(f"{p}.post_attention_layernorm.weight")
            layer.post_attention_layernorm.weight = tc.tensor(
                np.ascontiguousarray(ln2.astype(np.float32)), dtype="float32")
            loaded += 2

            for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
                w = buf.pop(f"{p}.self_attn.{nm}.weight").astype(np.float32)
                orig_bytes += w.nbytes
                ql = QuantLinearTC(np.ascontiguousarray(w), group_size)
                setattr(layer.self_attn, nm, ql)
                quant_bytes += ql.vram_bytes()
                loaded += 1
                del w
            for nm in ("gate_proj", "up_proj", "down_proj"):
                w = buf.pop(f"{p}.mlp.{nm}.weight").astype(np.float32)
                orig_bytes += w.nbytes
                ql = QuantLinearTC(np.ascontiguousarray(w), group_size)
                setattr(layer.mlp, nm, ql)
                quant_bytes += ql.vram_bytes()
                loaded += 1
                del w
            gc.collect()
            if (i + 1) % 8 == 0:
                print(f"    Layer {i+1}/{self.config.num_layers} done")

        norm_w = buf.pop("model.norm.weight")
        self.norm.weight = tc.tensor(
            np.ascontiguousarray(norm_w.astype(np.float32)), dtype="float32")
        loaded += 1

        lm_w = buf.pop("lm_head.weight").astype(np.float32)
        orig_bytes += lm_w.nbytes
        self.lm_head = QuantLinearTC(np.ascontiguousarray(lm_w), group_size)
        quant_bytes += self.lm_head.vram_bytes()
        loaded += 1
        del lm_w

        buf.clear()
        gc.collect()
        comp = orig_bytes / quant_bytes if quant_bytes else 0
        return {
            "loaded": loaded,
            "original_bytes": orig_bytes,
            "quantized_bytes": quant_bytes,
            "compression_ratio": comp,
        }

    @classmethod
    def from_pretrained(cls, model_dir=None, attention_mode="standard",
                        config: Optional[MistralConfig] = None, group_size=GROUP_SIZE):
        if config is None:
            config = MistralConfig()
        with tc.no_grad():
            model = cls(config, attention_mode)
            info = model.load_weights_quantized(model_dir, group_size)
        return model, info
