"""Gemma 4 12B Unified (text path) on tensor_cuda, INT4.

Text-only port of google/gemma-4-12B-it: 48 layers = 40 sliding-window
attention (window 1024, GQA 16q/8kv head_dim 256, RoPE theta 1e4) +
8 global attention at i%6==5 (MQA 16q/1kv head_dim 512, K=V shared
projection, p-RoPE theta 1e6 factor 0.25), gated GeLU 15360, vocab
262144 TIED, final logit softcap 30. Ledger: docs/GEMMA4_PORT_LEDGER.md.

Parity traps honored here (ledger has sources for each):
  - ALL RMSNorms are PLAIN w (Gemma 3's (1+w) is GONE). No bake.
  - layer_scalar multiplies the ENTIRE post-block hidden state
    (residual included); ckpt values 0.0045..0.886 — load-bearing.
  - attention scaling = 1.0 (qk-norm does the conditioning, no 1/sqrt d).
  - embed scale sqrt(3840) on the embedding output only — lm_head is the
    same tied matrix UNSCALED.
  - global K=V: one k_proj; K path k_norm(scaled)+p-RoPE, V path
    scale-free RMSNorm, no RoPE. Sliding V also gets the scale-free norm.
  - p-RoPE: inv_freq = [theta^(-2i/512) for i<64] ++ zeros(192) — full
    512-wide rotary apply, zero-freq dims are identity.
  - sliding mask: kv in (q-1024, q]; sandwich norms (post-norms on the
    branch OUTPUT before the residual add).

Cache protocol (per layer, uniform):
  global  -> (k, v): (B, 1, S, 512) compute dtype, append-only
  sliding -> (k, v): (B, 8, <=1023, 256) compute dtype, ring-trimmed to
             the last window-1 keys (the priors for the next step)
Both are exactly the model's visible state -> lossless save/restore
(pure-KV model: no recurrent state anywhere — seats compose).
"""
import gc
import os

import numpy as np

from core.mistral7b_tc import (BlockTC, LinearTC, QuantLinearTC, RMSNormTC,
                               _repeat_kv, F, tc)
from core.qwen35_tc import HostEmbedding, _cast


class Gemma4Config:
    vocab_size = 262144
    hidden_dim = 3840
    intermediate_dim = 15360
    num_layers = 48
    num_heads = 16
    # sliding layers (40 of 48)
    head_dim = 256
    num_kv_heads = 8
    sliding_window = 1024
    rope_theta_local = 10000.0
    # global layers (8 of 48, at i % 6 == 5)
    global_head_dim = 512
    num_global_kv_heads = 1
    rope_theta_global = 1000000.0
    p_rope_angles = 64               # of 256 freq slots; rest identity
    rms_norm_eps = 1e-6
    logit_softcap = 30.0
    eos_token_ids = (1, 106, 50)
    bos_token_id = 2

    @staticmethod
    def is_global(i):
        return i % 6 == 5


_MODEL_DIR = "/mnt/ForgeRealm/models/gemma-4-12B-it"
_QAT_GGUF = ("/mnt/ForgeRealm/models/gemma-4-12B-it-qat/"
             "gemma-4-12b-it-qat-q4_0.gguf")

_band_cache = {}


def _q40_repack(raw, N, K):
    """GGUF q4_0 blocks -> engine packed layout, EXACT (no requant).

    q4_0 block = 18 bytes: fp16 scale d + 16 qs bytes where
    x[j] = (qs[j] & 0xF) - 8 and x[j+16] = (qs[j] >> 4) - 8. The raw
    nibble IS x+8, which is exactly the engine's q under the
    symmetric-8 convention (w = q*s - 8*s, empty zeros tensor).
    Only the nibble ORDER differs (GGUF j/j+16 interleave vs engine
    even/odd pairs)."""
    blk = np.ascontiguousarray(raw).reshape(N * (K // 32), 18)
    scales = blk[:, :2].copy().view(np.float16).reshape(N, K // 32)
    qs = blk[:, 2:]
    q = np.concatenate([qs & 0x0F, qs >> 4], axis=1)     # (B, 32) = x+8
    q = q.reshape(N, K)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    return packed, scales


class Q40LinearTC:
    """Exact q4_0 import: packed nibbles + fp16 scales at group 32,
    EMPTY zeros tensor (engine symmetric-8 path, z = -8*s in-register).
    Dispatch mirrors QuantLinearTC (GEMV/tile/two-stage)."""

    def __init__(self, packed_np, scales_np):
        self.out_features = packed_np.shape[0]
        self.in_features = packed_np.shape[1] * 2
        self.group_size = 32
        self.packed = tc.tensor(packed_np, dtype="uint8")
        self.scales = tc.tensor(np.ascontiguousarray(scales_np),
                                dtype="float16")
        self.zeros = tc.tensor(np.zeros((0,), np.float16), dtype="float16")

    def __call__(self, x):
        # ALWAYS the fused path: int4_linear_fused dequantizes inside
        # shared-memory tiles (M=1 takes the GEMV kernel). The two-stage
        # path's whole-weight transient (113MB per MLP matmul) does not
        # fit next to the ~6.8GB QAT-exact body — prefill OOMs (longctx
        # gate). Cold-prefill speed regression vs cuBLAS is accepted;
        # registered as a speed-pass lever (column-blocked two-stage).
        return tc.int4_linear_fused(x, self.packed, self.scales,
                                    self.zeros, self.group_size)


def _zeros(*shape):
    """Allocate zeros directly in the compute dtype — no fp32 numpy
    intermediate and no fp32 device transient (the _cast(tc.tensor(
    np.zeros(fp32))) pattern transiently needs 2x the final bf16 size,
    which OOMed kqb allocation at 8K on a near-full card)."""
    cdt = BlockTC.COMPUTE_DTYPE
    return tc.zeros(*shape, dtype=cdt)


_RING_BLOCK = 2048


def _grow_cap(need):
    """Capacity policy: double while small (cheap, amortizes the realloc),
    then switch to fixed +2048 blocks once large. Pure power-of-two
    doubling wastes up to 2x at high context — next_pow2(8193)=16384
    held 8192 tokens at 8K, a 50% overallocation that OOMed BOTH modes
    (and doubled APA's kqb). Block growth bounds waste to one block."""
    if need <= _RING_BLOCK:
        cap = 64
        while cap < need:
            cap *= 2
        return cap
    return ((need + _RING_BLOCK - 1) // _RING_BLOCK) * _RING_BLOCK


class KVRing:
    """Mutable decode cache (2026-06-12 ring rework): sliding layers get
    a FIXED (B,KV,window,D) buffer with in-place ring writes; global
    layers get capacity-doubling append. Steady-state decode cache
    copies: ZERO (was ~670MB/token of cat+trim churn — measured 0.8s/tok
    past 1K context). Invalid rows are masked by an additive bias buffer
    (-1e4, zeroed as rows fill; buffers zero-init so scores over unwritten
    rows stay finite). OWNERSHIP CONTRACT: the decode loop owns these
    buffers — sharing (save/mount/branch) must COPY. Created lazily from
    the prefill tuple at the first decode step, per layer (the consumed-
    list contract frees each tuple as its ring is built).
    Attention math is permutation-invariant over keys (position lives in
    the roped values), so a wrapped ring needs no reordering — at the
    sliding steady state the ring content IS the window, maskless."""

    __slots__ = ("kb", "vb", "bias", "count", "cap", "ring", "window",
                 "kqb", "kq_count")

    def __init__(self, k, v, ring_cap=None):
        B, KV, S, D = k.shape
        # APA quantized-key ring (lazy): kqb mirrors kb position-for-
        # position, quantized once per key and never recomputed (a key's
        # quantization is fixed once roped+written). None until APA first
        # engages (apa_min_context). kq_count = rows already quantized.
        self.kqb = None
        self.kq_count = 0
        # capacity-double from small for BOTH kinds — a fixed-size ring
        # costs ~370MB per cache set regardless of context (the state
        # gate holds three sets and OOMed at 48 tokens). Rings cap at
        # the window and only then wrap; globals grow unbounded.
        self.window = ring_cap                 # None for globals
        self.ring = ring_cap is not None
        self.cap = _grow_cap(S + 1)
        if self.ring:
            self.cap = min(self.cap, ring_cap)
        self.kb = _zeros(B, KV, self.cap, D)
        self.vb = _zeros(B, KV, self.cap, D)
        bias = np.full((self.cap, 1), -1e4, np.float32)
        bias[:S] = 0.0
        self.bias = _cast(tc.tensor(bias))
        # rings are inference-only structures: guard the in-place writes
        # so construction works from any scope (load_caches runs outside
        # the caller's no_grad)
        with tc.no_grad():
            tc.write_rows(self.kb, k, 0)
            tc.write_rows(self.vb, v, 0)
        self.count = S

    @property
    def full(self):
        return (self.ring and self.cap == self.window
                and self.count >= self.cap)

    def append(self, k1, v1, zero_row):
        """k1/v1: (B,KV,1,D) post-norm/rope. zero_row: cached (1,1) zero
        tensor in compute dtype (bias unmask write)."""
        at_cap = self.count == self.cap
        can_grow = (not self.ring) or self.cap < self.window
        if at_cap and can_grow:
            old_n = self.count
            B, KV, _, D = self.kb.shape
            self.cap = (min(_grow_cap(self.cap + 1), self.window)
                        if self.ring else _grow_cap(self.cap + 1))

            # Grow ONE buffer at a time, freeing each old before the next
            # new — the doubling otherwise holds old+new for kb, vb AND
            # kqb simultaneously (3x the spike), which OOMed APA at 8K on
            # a near-full card. _grow1 allocates bf16 directly (no fp32
            # device transient from _cast) and drops the old reference.
            def _grow1(buf):
                nb = _zeros(B, KV, self.cap, D)
                with tc.no_grad():
                    tc.write_rows(nb, buf, 0)
                return nb
            self.kb = _grow1(self.kb)
            self.vb = _grow1(self.vb)
            if self.kqb is not None:
                self.kqb = _grow1(self.kqb)
            tc.empty_cache()       # release the old buffers' device blocks
            bias = np.full((self.cap, 1), -1e4, np.float32)
            bias[:old_n] = 0.0
            self.bias = _cast(tc.tensor(bias))
        pos = (self.count % self.cap
               if self.ring and self.cap == self.window else self.count)
        with tc.no_grad():
            tc.write_rows(self.kb, k1, pos)
            tc.write_rows(self.vb, v1, pos)
            if not self.full:
                tc.write_rows(self.bias, zero_row, pos)
        self.count += 1

    def quantized_keys(self, quantize_fn):
        """APA incremental kq cache (#9, ported from GQAAttentionTC). A
        key's quantization is fixed once it is roped + written, so kq is
        cached in `kqb` and only NEWLY-appended rows [kq_count:count) are
        quantized each call — O(D) per step instead of re-quantizing the
        whole O(S*D) cache. quantize_fn(k_slice) -> kq_slice (same shape).
        Globals never wrap (ring=False), so positions are stable.
        Returns the valid kq view (B, KV, count, D)."""
        new = self.count - self.kq_count
        if new > 0:
            if self.kqb is None:
                B, KV, _, D = self.kb.shape
                self.kqb = _zeros(B, KV, self.cap, D)
            # CHUNK the quantize: the first call (cold start) quantizes
            # the whole prefill span, recreating the ~80-96MB O(S*D)
            # transient this cache exists to avoid — once, but enough to
            # OOM at 8K. _quantize_keys is row-independent (verified
            # bit-exact), so quantizing in CHUNK-row slices and writing
            # each into kqb is identical and caps the transient to
            # O(CHUNK*D). Steady-state decode (new==1) takes one pass.
            CHUNK = 512
            with tc.no_grad():
                for s0 in range(self.kq_count, self.count, CHUNK):
                    n = min(CHUNK, self.count - s0)
                    kq_s = quantize_fn(self.kb.slice(2, s0, n))
                    tc.write_rows(self.kqb, kq_s, s0)
                    if new > CHUNK:
                        tc.empty_cache()
            self.kq_count = self.count
        return self.kqb.slice(2, 0, self.count)

    def ordered(self):
        """Valid rows as (k, v) COPIES in LOGICAL order (oldest first) —
        the prefill-continuation / export form. Band masks assume
        temporal order, so a wrapped ring is unrolled via two slices.
        Attention itself never needs this (permutation-invariant)."""
        n = min(self.count, self.cap)
        if self.ring and self.count > self.cap:
            cut = self.count % self.cap
            if cut == 0:
                return (self.kb.slice(2, 0, self.cap),
                        self.vb.slice(2, 0, self.cap))
            k = tc.cat([self.kb.slice(2, cut, self.cap - cut),
                        self.kb.slice(2, 0, cut)], dim=2)
            v = tc.cat([self.vb.slice(2, cut, self.cap - cut),
                        self.vb.slice(2, 0, cut)], dim=2)
            return k, v
        return self.kb.slice(2, 0, n), self.vb.slice(2, 0, n)


def _band_mask(L, S, window, device, dtype):
    """Additive sliding-window mask, BOTTOM-RIGHT aligned like the engine's
    causal mask: query row i is absolute key index (S-L+i); col j visible
    iff (S-L+i)-window < j <= (S-L+i)."""
    key = (L, S, window, device, dtype)
    m = _band_cache.get(key)
    if m is None:
        i = np.arange(L, dtype=np.int64)[:, None] + (S - L)
        j = np.arange(S, dtype=np.int64)[None, :]
        vis = (j <= i) & (j > i - window)
        bias = np.where(vis, 0.0, -1e4).astype(np.float32)
        m = tc.tensor(np.ascontiguousarray(bias), device=device)
        if dtype in ("float16", "bfloat16"):
            m = m.astype(dtype)
        _band_cache[key] = m
    return m


_ones_cache = {}


def _ones(D):
    t = _ones_cache.get(D)
    if t is None:
        t = tc.tensor(np.ones(D, np.float32), dtype="float32")
        _ones_cache[D] = t
    return t


_zero_cache = {}


def _zero_row():
    """Cached (1,1) zero in compute dtype — the bias unmask write."""
    cdt = BlockTC.COMPUTE_DTYPE
    t = _zero_cache.get(cdt)
    if t is None:
        t = _cast(tc.tensor(np.zeros((1, 1), np.float32)))
        _zero_cache[cdt] = t
    return t


def _head_rmsnorm(x, w, eps, B, L, H, D):
    """Per-head RMSNorm, PLAIN w (None = the scale-free v_norm).
    Inference takes the engine's fused rms_norm kernel (one launch vs
    the ~6-launch composed chain — 3 of these per attention layer)."""
    if hasattr(tc, "rms_norm") and not tc.is_grad_enabled():
        wt = w if w is not None else _ones(D)
        return tc.rms_norm(x.reshape([B, L * H, D]), wt,
                           eps).reshape([B, L, H * D])
    xf = x.reshape([B, L, H, D]).float()
    ms = (xf * xf).mean([-1], True)
    xf = xf * (ms + eps).pow(-0.5)
    if w is not None:
        xf = xf * w
    return _cast(xf)


class Gemma4AttentionTC:
    def __init__(self, cfg, layer_idx):
        self.cfg = cfg
        self.is_global = Gemma4Config.is_global(layer_idx)
        self.head_dim = cfg.global_head_dim if self.is_global else cfg.head_dim
        self.kv_heads = (cfg.num_global_kv_heads if self.is_global
                         else cfg.num_kv_heads)
        self.q_proj = self.k_proj = self.v_proj = self.o_proj = None
        self.qkv_proj = None      # concat [q|k(|v)] — one GEMV launch
        self.q_norm_w = self.k_norm_w = None     # fp32 PLAIN w
        # APA dials (qk-norm family -> bulk_bits 4 per the measured law;
        # 6th architecture test). cuBLAS blend path only — fused kernel's
        # head_dim templates stop at 128, global layers are 512. Sliding
        # layers never need APA: their keys are bounded at 1024.
        self.attention_mode = "standard"
        self.refine_percentile = 0.15
        self.bulk_bits = 4
        self.attn_block = 1024
        # apa_selective engages the blend only past this context: APA's
        # job is bounded transients at LONG context, and below this the
        # blend's per-token whole-cache requantization is pure overhead
        # (profiler: 2.49 ms/global-layer at S=700; APA==standard at
        # short S per the zero-flip gate). Gates force this to 0 to
        # keep the blend machinery itself tested.
        self.apa_min_context = 2048

    def __call__(self, x, cos, sin, position_offset=0, kv_cache=None):
        cfg = self.cfg
        B, L, _ = x.shape
        H, KV, D = cfg.num_heads, self.kv_heads, self.head_dim

        if self.qkv_proj is not None:
            # one GEMV launch for [q|k(|v)] instead of 2-3
            qkv = self.qkv_proj(x)
            q_lin = qkv.slice(2, 0, H * D)
            kraw = qkv.slice(2, H * D, KV * D)
            vsrc = (kraw if self.is_global
                    else qkv.slice(2, (H + KV) * D, KV * D))
        else:
            q_lin = self.q_proj(x)
            kraw = self.k_proj(x)
            # V source: global layers share k_proj (K=V projection);
            # sliding layers have their own v_proj.
            vsrc = kraw if self.is_global else self.v_proj(x)

        q = _head_rmsnorm(q_lin, self.q_norm_w, cfg.rms_norm_eps,
                          B, L, H, D)
        q = q.reshape([B, L, H, D]).transpose(1, 2)          # (B,H,L,D)
        k = _head_rmsnorm(kraw, self.k_norm_w, cfg.rms_norm_eps, B, L, KV, D)
        k = k.reshape([B, L, KV, D]).transpose(1, 2)
        # V: scale-free RMSNorm, NO RoPE.
        v = _head_rmsnorm(vsrc, None, cfg.rms_norm_eps, B, L, KV, D)
        v = v.reshape([B, L, KV, D]).transpose(1, 2)

        if hasattr(tc, "rope_apply") and not tc.is_grad_enabled():
            # one launch per tensor vs ~8 for the composed chain
            q = tc.rope_apply(q, cos, sin, position_offset)
            k = tc.rope_apply(k, cos, sin, position_offset)
        else:
            cseg = cos.slice(0, position_offset, L)
            sseg = sin.slice(0, position_offset, L)
            q = F.apply_rotary(q, cseg, sseg)
            k = F.apply_rotary(k, cseg, sseg)

        ring_cache = None
        # ---- DECODE (L==1): ring path, zero cache copies ------------
        # In-place append into a KVRing (lazy-converted from the prefill
        # tuple on first decode; the consumed-list contract frees the
        # tuple as the ring is built). Invalid rows are bias-masked;
        # attention reads the FULL buffer — permutation-invariant, so a
        # wrapped sliding ring needs no reordering and, at steady state,
        # no mask either.
        if L == 1 and kv_cache is not None:
            if isinstance(kv_cache, tuple):
                kv_cache = KVRing(
                    kv_cache[0], kv_cache[1],
                    ring_cap=None if self.is_global else cfg.sliding_window)
            kv_cache.append(k, v, _zero_row())
            S_all = min(kv_cache.count, kv_cache.cap)
            apa_active = (self.is_global
                          and self.attention_mode == "apa_selective"
                          and S_all > self.apa_min_context)
            if apa_active:
                # INCREMENTAL APA decode (#9): the kq cache quantizes only
                # the new key (O(D)) instead of re-quantizing the whole
                # O(S*D) cache every step — the measured 8K-decode OOM
                # driver. kb/vb are the unexpanded MQA cache (globals
                # never wrap, so slice(0,count) is logical order). The
                # blend folds q-heads into rows, no 16x expansion.
                from tensor_cuda.quant import (_norm_ppf, _quantize_keys,
                                               _tables)
                from core.mistral7b_tc import _cublas_blend_attention
                dev = q.device.split(":")[0]
                R_t, C_t, B_t = _tables(D, self.bulk_bits, KV, True, dev)
                kq = kv_cache.quantized_keys(
                    lambda ks: _quantize_keys(ks, R_t, C_t, B_t))
                kk = kv_cache.kb.slice(2, 0, kv_cache.count)
                vv = kv_cache.vb.slice(2, 0, kv_cache.count)
                z_ = _norm_ppf(1.0 - max(0.0, min(1.0,
                                                  self.refine_percentile)))
                S_a = kv_cache.count
                blk = max(64, min(self.attn_block,
                                  int(300 * 1024 * 1024 // (H * S_a * 2))))
                attn = _cublas_blend_attention(
                    q, kk, kq, vv, H // KV, 1.0, float(z_), False, blk)
                attn = attn.transpose(1, 2).reshape([B, L, H * D])
                return self.o_proj(_cast(attn)), kv_cache
            else:
                rep = H // KV
                qg = q.reshape([B, KV, rep, D])
                sc = tc.matmul(qg, kv_cache.kb, alpha=1.0, trans_b=True)
                if not kv_cache.full:
                    sc = sc + kv_cache.bias.reshape([1, 1, 1, kv_cache.cap])
                # fused single-kernel softmax (at L==1 causal == full
                # row); reshape to L=1 rows first — causal_softmax reads
                # dim -2 as L and would mask keys off earlier q heads.
                if hasattr(tc, "causal_softmax"):
                    p = tc.causal_softmax(
                        sc.reshape([B, KV * rep, 1, kv_cache.cap]))
                    p = p.reshape([B, KV, rep, kv_cache.cap])
                else:
                    p = sc.softmax(-1)
                attn = tc.matmul(p, kv_cache.vb).reshape([B, L, H * D])
                return self.o_proj(_cast(attn)), kv_cache
        else:
            if isinstance(kv_cache, KVRing):
                # prefill continuation onto a decode cache (multi-turn /
                # ladder probes): unwrap to logically-ordered tuples —
                # band masks need temporal order. One-time copy at
                # prefill rates.
                kv_cache = kv_cache.ordered()
            if kv_cache is not None:
                k = tc.cat([kv_cache[0], k], dim=2)
                v = tc.cat([kv_cache[1], v], dim=2)
        S_all = k.shape[2]

        apa_active = (self.is_global
                      and self.attention_mode == "apa_selective"
                      and S_all > self.apa_min_context)

        if self.is_global:
            new_kv = ring_cache if ring_cache is not None else (k, v)
            if apa_active:
                from tensor_cuda.quant import (_norm_ppf, _quantize_keys,
                                               _tables)
                from core.mistral7b_tc import _cublas_blend_attention
                dev = q.device.split(":")[0]
                R_t, C_t, B_t = _tables(D, self.bulk_bits, KV, True, dev)
                kq = _quantize_keys(k, R_t, C_t, B_t)
                z_ = _norm_ppf(1.0 - max(0.0, min(1.0,
                                                  self.refine_percentile)))
                blk = max(64, min(self.attn_block,
                                  int(300 * 1024 * 1024 // (H * S_all * 2))))
                # UNEXPANDED k: the blend's MQA path folds q heads
                # into rows instead of expanding the cache 16x
                attn = _cublas_blend_attention(
                    q, k, kq, v, H // KV, 1.0, float(z_), L > 1, blk)
            else:
                # MQA prefill de-expansion (KV=1): fold the 16 q-heads
                # into rows — (B,1,H*L,D) against the UNEXPANDED cache,
                # pure reshapes. repeat_kv here materialized 2x ~200MB
                # per layer-chunk at 12K context (OOM-proven); this
                # path allocates nothing but scores.
                qf = q.reshape([B, 1, H * L, D])
                sc = tc.matmul(qf, k, alpha=1.0, trans_b=True)
                sc = sc.reshape([B, H, L, S_all])
                if hasattr(tc, "causal_softmax") and not tc.is_grad_enabled():
                    # bottom-right causal is built into the kernel —
                    # NO mask tensors. Adaptive chunking gives every
                    # chunk a unique (L,S), so materialized masks
                    # churned ~200MB of cache at 12K (OOM-proven).
                    p_ = tc.causal_softmax(sc)
                else:
                    dev = q.device.split(":")[0]
                    p_ = (sc + F._causal_mask(L, S_all, dev,
                                              sc.dtype)).softmax(-1)
                attn = tc.matmul(p_.reshape([B, 1, H * L, S_all]),
                                 v).reshape([B, H, L, D])
        else:
            W = cfg.sliding_window
            if S_all <= W:
                attn = F.scaled_dot_product_attention(
                    q, _repeat_kv(k, H // KV), _repeat_kv(v, H // KV),
                    is_causal=(L > 1), scale=1.0)
            else:
                dev = q.device.split(":")[0]
                attn = F.scaled_dot_product_attention(
                    q, _repeat_kv(k, H // KV), _repeat_kv(v, H // KV),
                    attn_mask=_band_mask(L, S_all, W, dev, q.dtype),
                    scale=1.0)
            # ring-trim: keep the last window-1 keys (priors for next
            # step); slice only when the window binds
            keep = min(W - 1, S_all)
            new_kv = ((k, v) if keep == S_all else
                      (k.slice(2, S_all - keep, keep),
                       v.slice(2, S_all - keep, keep)))

        attn = attn.transpose(1, 2).reshape([B, L, H * D])
        return self.o_proj(_cast(attn)), new_kv


class GegluTC:
    def __init__(self):
        self.gate_proj = self.up_proj = self.down_proj = None
        self.gu_proj = None        # concat [gate|up] — one GEMV launch

    def __call__(self, x):
        # gelu_pytorch_tanh — engine gelu IS the tanh approximation
        if self.gu_proj is not None:
            gu = self.gu_proj(x)
            F2 = gu.shape[-1] // 2
            return self.down_proj(gu.slice(2, 0, F2).gelu()
                                  * gu.slice(2, F2, F2))
        return self.down_proj(self.gate_proj(x).gelu() * self.up_proj(x))


class Gemma4BlockTC:
    def __init__(self, cfg, layer_idx):
        self.is_global = Gemma4Config.is_global(layer_idx)
        e = cfg.rms_norm_eps
        self.input_layernorm = RMSNormTC(cfg.hidden_dim, e)
        self.post_attention_layernorm = RMSNormTC(cfg.hidden_dim, e)
        self.pre_feedforward_layernorm = RMSNormTC(cfg.hidden_dim, e)
        self.post_feedforward_layernorm = RMSNormTC(cfg.hidden_dim, e)
        self.layer_scalar = 1.0          # ckpt scalar, load-bearing (trap 2)
        self.mixer = Gemma4AttentionTC(cfg, layer_idx)
        self.mlp = GegluTC()

    def __call__(self, x, ropes, position_offset=0, cache=None):
        cos, sin = ropes[1] if self.is_global else ropes[0]
        # sandwich norms: post-norms apply to the branch OUTPUT, then add
        a, new_cache = self.mixer(_cast(self.input_layernorm(x)), cos, sin,
                                  position_offset, cache)
        h = x + _cast(self.post_attention_layernorm(a))
        mo = self.mlp(_cast(self.pre_feedforward_layernorm(h)))
        h = h + _cast(self.post_feedforward_layernorm(mo))
        return h * self.layer_scalar, new_cache


class Gemma4_TC:
    def __init__(self, cfg=None):
        self.config = cfg or Gemma4Config()
        self.embed_tokens = HostEmbedding()   # rows pre-scaled sqrt(3840)
        self.layers = [Gemma4BlockTC(self.config, i)
                       for i in range(self.config.num_layers)]
        self.norm = RMSNormTC(self.config.hidden_dim,
                              self.config.rms_norm_eps)
        self.lm_head = None                   # tied matrix, UNSCALED
        self._rope_len = 0
        self.extend_rope(4096)

    def extend_rope(self, seq_len):
        if seq_len <= self._rope_len:
            return
        cfg = self.config
        pos = np.arange(seq_len, dtype=np.float32)[:, None]
        # sliding: standard full-width rotary over 256
        d = cfg.head_dim
        inv_l = 1.0 / (cfg.rope_theta_local
                       ** (np.arange(0, d, 2, np.float32) / d))
        emb_l = np.concatenate([pos * inv_l, pos * inv_l], axis=-1)
        # global p-RoPE over 512: first 64 freqs AS PARAMETERIZED FOR the
        # full 512-dim space, remaining 192 freq slots zero (identity)
        D = cfg.global_head_dim
        inv_g = 1.0 / (cfg.rope_theta_global
                       ** (np.arange(0, 2 * cfg.p_rope_angles, 2,
                                     np.float32) / D))
        inv_g = np.concatenate([inv_g,
                                np.zeros(D // 2 - cfg.p_rope_angles,
                                         np.float32)])
        emb_g = np.concatenate([pos * inv_g, pos * inv_g], axis=-1)
        self.ropes = tuple(
            (_cast(tc.tensor(np.cos(e).astype(np.float32))),
             _cast(tc.tensor(np.sin(e).astype(np.float32))))
            for e in (emb_l, emb_g))
        self._rope_len = seq_len

    # Auto-chunked prefill: the QAT-exact body is ~6.8GB resident; a
    # single-shot long prefill's transients (per-layer score tensors,
    # two-stage dequant, pool fragmentation) OOM the card past a few
    # hundred tokens (longctx gate caught it at 1200). Chunking bounds
    # every transient; cache math is identical by construction (the
    # longctx B-leg gates single-shot vs chunked equivalence).
    PREFILL_CHUNK = 512

    def __call__(self, input_ids_np, caches=None, position_offset=0,
                 last_token_only=False, max_layers=None):
        B, L = input_ids_np.shape
        C = self.PREFILL_CHUNK
        if max_layers is None and L > C:
            outs = []
            lg = None
            off = position_offset
            s0 = 0
            while s0 < L:
                # ADAPTIVE chunk: per-chunk global score transients are
                # 16*step*S_all*2B (x2 for the mask add) — fixed steps
                # OOM past ~3-4K context on the ~7GB baseline. Hold the
                # transient under ~32MB; floor 64 keeps GEMM shapes sane.
                S_ctx = off + C
                step = min(C, max(64,
                                  (32 * 1024 * 1024) // (32 * S_ctx)))
                step = max(64, (step // 64) * 64)
                seg = input_ids_np[:, s0:s0 + step]
                # last_token_only passes through UNCHANGED: when the
                # caller wants only the final logits, every chunk must
                # compute 1-row logits (a wasted GEMV per non-final
                # chunk, ~1ms) — NOT full chunk logits (512x262144 fp32
                # = 0.8GB of discarded transient; OOMed the longctx
                # gate).
                lg, caches = self._forward(seg, caches, off,
                                           last_token_only=last_token_only)
                off += seg.shape[1]
                s0 += seg.shape[1]
                if not last_token_only:
                    outs.append(lg)
                # trim the allocator pool at chunk boundaries: chunked
                # long prefills create distinct transient shapes per
                # chunk and the pooled high-water alone tips the ~7.1GB
                # baseline over the card (longctx gate OOMed in a 25MB
                # softmax). Decode is untouched — constant shapes reuse
                # the pool correctly.
                tc.empty_cache()
            return (lg if last_token_only else tc.cat(outs, dim=1)), caches
        return self._forward(input_ids_np, caches, position_offset,
                             last_token_only, max_layers)

    def _forward(self, input_ids_np, caches=None, position_offset=0,
                 last_token_only=False, max_layers=None):
        B, L = input_ids_np.shape
        self.extend_rope(position_offset + L)
        h = self.embed_tokens(input_ids_np)
        new_caches = []
        run = self.layers if max_layers is None else self.layers[:max_layers]
        for i, layer in enumerate(run):
            cache = caches[i] if caches is not None else None
            h, c = layer(h, self.ropes, position_offset, cache)
            new_caches.append(c)
            # CONTRACT: the input caches LIST is consumed — entries are
            # released as their successors are built. Holding old+new
            # cache sets simultaneously (~2x335MB at long context) does
            # not fit next to the ~6.8GB QAT body. Callers that need to
            # branch from a cache set must hold their own reference to
            # the TENSORS (tuples), not rely on the passed list.
            if caches is not None:
                caches[i] = None
        if max_layers is not None:
            return None, new_caches, h
        h = _cast(self.norm(h))
        if last_token_only and h.shape[1] > 1:
            h = h.slice(1, h.shape[1] - 1, 1)
        # lm_head in GEMV-sized chunks: at M>8 the two-stage path would
        # materialize the full 2GB bf16 weight transient — on top of the
        # ~6.3GB resident body that OOMs the card. Chunks of 8 rows stay
        # on the packed-read GEMV path (FUSED_DECODE).
        if h.shape[1] > 8:
            parts = []
            for s0 in range(0, h.shape[1], 8):
                seg = h.slice(1, s0, min(8, h.shape[1] - s0))
                parts.append(self.lm_head(seg))
            lg = tc.cat(parts, dim=1)
        else:
            lg = self.lm_head(h)
        c = self.config.logit_softcap
        logits = (lg.float() * (1.0 / c)).tanh() * c
        return logits, new_caches

    # ---- weights ------------------------------------------------------
    def load_weights(self, model_dir=None, progress=True):
        from safetensors import safe_open
        d = model_dir or _MODEL_DIR
        cfg = self.config
        path = os.path.join(d, "model.safetensors")

        def gf(name, f):
            import torch
            return f.get_tensor(name).to(torch.float32).numpy()

        P = "model.language_model"
        with safe_open(path, framework="pt") as f:
            emb = np.ascontiguousarray(gf(f"{P}.embed_tokens.weight", f))
            # tied head quantizes the UNSCALED matrix; the host embedding
            # then takes the sqrt(hidden) scale in place (fp32, trap 4)
            self.lm_head = QuantLinearTC(emb)
            emb *= np.float32(cfg.hidden_dim ** 0.5)
            self.embed_tokens.weight = emb
            gc.collect()

            for i in range(cfg.num_layers):
                p = f"{P}.layers.{i}"
                Lr = self.layers[i]
                for nm in ("input_layernorm", "post_attention_layernorm",
                           "pre_feedforward_layernorm",
                           "post_feedforward_layernorm"):
                    getattr(Lr, nm).weight = tc.tensor(np.ascontiguousarray(
                        gf(f"{p}.{nm}.weight", f)), dtype="float32")
                Lr.layer_scalar = float(gf(f"{p}.layer_scalar", f)[0])
                mx = Lr.mixer
                a = f"{p}.self_attn"
                mx.q_proj = QuantLinearTC(np.ascontiguousarray(
                    gf(f"{a}.q_proj.weight", f)))
                mx.k_proj = QuantLinearTC(np.ascontiguousarray(
                    gf(f"{a}.k_proj.weight", f)))
                if not mx.is_global:
                    mx.v_proj = QuantLinearTC(np.ascontiguousarray(
                        gf(f"{a}.v_proj.weight", f)))
                mx.o_proj = QuantLinearTC(np.ascontiguousarray(
                    gf(f"{a}.o_proj.weight", f)))
                mx.q_norm_w = tc.tensor(np.ascontiguousarray(
                    gf(f"{a}.q_norm.weight", f)), dtype="float32")  # PLAIN
                mx.k_norm_w = tc.tensor(np.ascontiguousarray(
                    gf(f"{a}.k_norm.weight", f)), dtype="float32")  # PLAIN
                g = f"{p}.mlp"
                Lr.mlp.gate_proj = QuantLinearTC(np.ascontiguousarray(
                    gf(f"{g}.gate_proj.weight", f)))
                Lr.mlp.up_proj = QuantLinearTC(np.ascontiguousarray(
                    gf(f"{g}.up_proj.weight", f)))
                Lr.mlp.down_proj = QuantLinearTC(np.ascontiguousarray(
                    gf(f"{g}.down_proj.weight", f)))
                gc.collect()
                if progress and i % 6 == 5:
                    print(f"    layer {i + 1}/{cfg.num_layers}", flush=True)

            self.norm.weight = tc.tensor(np.ascontiguousarray(
                gf(f"{P}.norm.weight", f)), dtype="float32")        # PLAIN
        gc.collect()
        return {"loaded": "INT4 pure-KV",
                "framework": "tensor_cuda Gemma4-12B (40 SWA + 8 global)"}

    # ---- QAT weights (the production path) ----------------------------
    # Post-hoc INT4-g128 of the bf16 release COLLAPSES on this model
    # (host-sim proven: L47 cos 0.28 from quantization noise alone; that
    # is why Google ships a QAT release). The q4_0 QAT grid is imported
    # EXACTLY: nibbles repacked, scales kept fp16, engine symmetric-8
    # kernels (empty zeros). The only post-hoc tensor is the tied head
    # (token_embd is Q6_K in the official file): requantized at g32.
    def load_weights_qat(self, gguf_path=None, progress=True):
        from gguf import GGUFReader
        from gguf.quants import dequantize
        cfg = self.config
        r = GGUFReader(gguf_path or _QAT_GGUF)
        T = {t.name: t for t in r.tensors}

        def q40_parts(names_dims):
            """Concat several q4_0 tensors row-wise into ONE Q40LinearTC
            (per-row packing makes row concat exact): one GEMV launch
            where there were 2-3."""
            packs, scs = [], []
            for name, N, K in names_dims:
                t = T[name]
                assert int(t.tensor_type) == 2, \
                    f"{name}: expected Q4_0 (2), got {t.tensor_type}"
                p, s = _q40_repack(np.asarray(t.data), N, K)
                packs.append(p)
                scs.append(s)
            return Q40LinearTC(np.concatenate(packs, axis=0),
                               np.concatenate(scs, axis=0))

        def q40(name, N, K):
            return q40_parts([(name, N, K)])

        def f32(name):
            return np.asarray(T[name].data, np.float32)

        emb = dequantize(T["token_embd.weight"].data,
                         T["token_embd.weight"].tensor_type
                         ).astype(np.float32)
        # the tied head is the ONLY post-hoc requant (token_embd ships
        # Q6_K); g128 here, not g32 — saves 100MB on a knife-edge VRAM
        # budget, and the parity margin gate adjudicates the cost (it
        # passed WITH g32; re-gated after this change).
        self.lm_head = QuantLinearTC(np.ascontiguousarray(emb))
        emb *= np.float32(cfg.hidden_dim ** 0.5)
        self.embed_tokens.weight = np.ascontiguousarray(emb)
        del emb
        gc.collect()

        Hd, Dg, Ds = cfg.hidden_dim, cfg.global_head_dim, cfg.head_dim
        for i in range(cfg.num_layers):
            Lr = self.layers[i]
            g = Lr.mixer.is_global
            D, KV = (Dg, 1) if g else (Ds, cfg.num_kv_heads)
            b = f"blk.{i}"
            for ours, theirs in (
                    ("input_layernorm", "attn_norm"),
                    ("post_attention_layernorm", "post_attention_norm"),
                    ("pre_feedforward_layernorm", "ffn_norm"),
                    ("post_feedforward_layernorm", "post_ffw_norm")):
                getattr(Lr, ours).weight = tc.tensor(np.ascontiguousarray(
                    f32(f"{b}.{theirs}.weight")), dtype="float32")
            Lr.layer_scalar = float(f32(f"{b}.layer_output_scale.weight")[0])
            mx = Lr.mixer
            parts = [(f"{b}.attn_q.weight", cfg.num_heads * D, Hd),
                     (f"{b}.attn_k.weight", KV * D, Hd)]
            if not g:
                parts.append((f"{b}.attn_v.weight", KV * D, Hd))
            mx.qkv_proj = q40_parts(parts)
            mx.o_proj = q40(f"{b}.attn_output.weight", Hd,
                            cfg.num_heads * D)
            mx.q_norm_w = tc.tensor(np.ascontiguousarray(
                f32(f"{b}.attn_q_norm.weight")), dtype="float32")
            mx.k_norm_w = tc.tensor(np.ascontiguousarray(
                f32(f"{b}.attn_k_norm.weight")), dtype="float32")
            Lr.mlp.gu_proj = q40_parts(
                [(f"{b}.ffn_gate.weight", cfg.intermediate_dim, Hd),
                 (f"{b}.ffn_up.weight", cfg.intermediate_dim, Hd)])
            Lr.mlp.down_proj = q40(f"{b}.ffn_down.weight", Hd,
                                   cfg.intermediate_dim)
            gc.collect()
            if progress and i % 6 == 5:
                print(f"    layer {i + 1}/{cfg.num_layers}", flush=True)

        self.norm.weight = tc.tensor(np.ascontiguousarray(
            f32("output_norm.weight")), dtype="float32")
        gc.collect()
        return {"loaded": "QAT q4_0 exact (symmetric-8 g32)",
                "framework": "tensor_cuda Gemma4-12B (40 SWA + 8 global)"}

    @classmethod
    def from_pretrained(cls, model_dir=None, qat=True):
        BlockTC.COMPUTE_DTYPE = "bfloat16"
        LinearTC.DTYPE = "bfloat16"
        # decode-speed defaults proven on the Qwen3.5 ladder: int4 GEMV at
        # M<=8 and the fused rms_norm kernel (plain-w weights, op unchanged)
        QuantLinearTC.FUSED_DECODE = True
        RMSNormTC.USE_FUSED = True
        with tc.no_grad():
            m = cls()
            info = (m.load_weights_qat(model_dir) if qat
                    else m.load_weights(model_dir))
        return m, info


# ---- GRM surface: lossless pure-KV save/restore ----------------------
# Uniform (k, v) per layer in compute dtype, fp32 round-trip in the npz
# (bf16 -> fp32 is exact). Sliding entries are the <=1023-key ring; global
# entries are the full prefix. position_offset rides along.

def save_caches(caches, path, position_offset):
    """Tuple caches save their tensors; KVRings save RAW buffers + meta
    (count/cap/ring) — exact regardless of wrap state. Saving is a COPY
    (host round-trip), honoring the ring ownership contract."""
    arrs = {"position_offset": np.array([position_offset], np.int64)}
    for i, c in enumerate(caches):
        if isinstance(c, KVRing):
            # save VALID ROWS ONLY in logical order (a full sliding
            # buffer is 1024 rows of mostly zeros for short contexts —
            # the buffer-dump version OOMed the state gate's restore
            # next to the live set). The ring flag is the only meta
            # needed: restore rebuilds capacity/bias from the rows.
            k, v = c.ordered()
            arrs[f"l{i}_k"] = k.float().numpy()
            arrs[f"l{i}_v"] = v.float().numpy()
            arrs[f"l{i}_ring"] = np.array([int(c.ring)], np.int64)
        else:
            k, v = c
            arrs[f"l{i}_k"] = k.float().numpy()
            arrs[f"l{i}_v"] = v.float().numpy()
    tmp = path + ".tmp"
    np.savez(tmp, **arrs)
    os.replace(tmp + ".npz", path)


def load_caches(path, cfg=None):
    cfg = cfg or Gemma4Config()
    z = np.load(path)
    caches = []
    for i in range(cfg.num_layers):
        k = _cast(tc.tensor(z[f"l{i}_k"]))
        v = _cast(tc.tensor(z[f"l{i}_v"]))
        if f"l{i}_ring" in z.files:
            # rebuild the ring from valid rows: KVRing.__init__ writes
            # them at [0:count) and sets the bias. A saved-wrapped ring
            # arrives with count == cap, so the next append overwrites
            # the logical-oldest row — FIFO semantics preserved.
            ring = bool(int(z[f"l{i}_ring"][0]))
            cfg_w = cfg.sliding_window if ring else None
            caches.append(KVRing(k, v, ring_cap=cfg_w))
        else:
            caches.append((k, v))
    return caches, int(z["position_offset"][0])
