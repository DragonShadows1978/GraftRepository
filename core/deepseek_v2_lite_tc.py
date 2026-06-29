"""DeepSeek-V2-Lite on tensor_cuda, INT4 weights + MLA attention.

This is the "baby" port: it reuses the Project-Tensor/GraftRepository INT4
weight path (`QuantLinearTC`) and the MiniCPM3 MLA dataflow, then adds
DeepSeekMoE routing. The MoE dispatch is deliberately simple and inference-first:
it routes token-by-token from the GPU-computed top-k gate. That is slow but keeps
the first correctness surface small; grouped expert dispatch is the obvious next
optimization.
"""
import gc
import glob
import json
import math
import os

import numpy as np

from core.mistral7b_tc import (BlockTC, EmbeddingTC, QuantLinearTC, RMSNormTC,
                               _cublas_blend_attention, F, tc)
from core.graft_arena import ArenaCache


_SNAP_GLOB = ("/home/vader/.cache/huggingface/hub/"
              "models--deepseek-ai--DeepSeek-V2-Lite/snapshots/*")


def _snap():
    hits = sorted(glob.glob(_SNAP_GLOB))
    if not hits:
        raise FileNotFoundError("DeepSeek-V2-Lite snapshot not found")
    return hits[-1]


def _yarn_get_mscale(scale=1.0, mscale=1.0):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_find_correction_dim(num_rotations, dim, base=10000.0,
                              max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings /
                           (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(low_rot, high_rot, dim, base=10000.0,
                                max_position_embeddings=2048):
    low = math.floor(_yarn_find_correction_dim(
        low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(
        high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(lo, hi, dim):
    if lo == hi:
        hi += 0.001
    ramp = (np.arange(dim, dtype=np.float32) - lo) / (hi - lo)
    return np.clip(ramp, 0.0, 1.0)


class DeepSeekV2LiteConfig:
    vocab_size = 102400
    hidden_dim = 2048
    intermediate_dim = 10944
    moe_intermediate_dim = 1408
    num_layers = 27
    first_k_dense_replace = 1
    moe_layer_freq = 1
    num_heads = 16
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    max_position_embeddings = 163840
    original_max_position_embeddings = 4096
    rope_theta = 10000.0
    rope_factor = 40.0
    rope_beta_fast = 32
    rope_beta_slow = 1
    rope_mscale = 0.707
    rope_mscale_all_dim = 0.707
    rms_norm_eps = 1e-6
    n_routed_experts = 64
    n_shared_experts = 2
    num_experts_per_tok = 6
    routed_scaling_factor = 1.0
    group_size = 64
    bos_token_id = 100000
    eos_token_id = 100001


def _cast_compute(t):
    dt = BlockTC.COMPUTE_DTYPE
    return t.half() if dt == "float16" else t.astype(dt)


def _deepseek_rope_prep(x):
    """Match HF DeepSeek's pre-RoPE pair permutation before rotate_half."""
    *prefix, d = x.shape
    return x.reshape(list(prefix) + [d // 2, 2]).transpose(-1, -2).reshape(list(prefix) + [d])


def _cache_zeros(*shape, dtype=None):
    if dtype == "uint8":
        return tc.zeros(*shape, dtype="uint8")
    dt = dtype or BlockTC.COMPUTE_DTYPE
    if dt == "bfloat16":
        z = tc.zeros(*shape, dtype="float16")
        out = z.astype("bfloat16")
        del z
        tc.empty_cache()
        return out
    return tc.zeros(*shape, dtype=dt)


class DeepSeekMLACache:
    """Mutable MLA latent cache: post-norm latent + roped shared key.

    The old tuple path used cat(old, new) every decode step, which transiently
    held two full caches. This owns capacity and appends in-place instead.
    """
    __slots__ = ("cbuf", "cpack", "cscale", "kbuf", "count", "cap",
                 "latent_bits", "group")

    def __init__(self, c_n, k_pe, extra=1024, latent_bits=16, group=32):
        B, S, R = c_n.shape
        _, _, S2, ROPE = k_pe.shape
        if S != S2:
            raise ValueError("DeepSeekMLACache latent/key length mismatch")
        if latent_bits not in (2, 4, 16):
            raise ValueError("DeepSeekMLACache latent_bits must be 2, 4, or 16")
        self.count = S
        self.cap = max(S, S + max(0, int(extra)))
        self.latent_bits = latent_bits
        self.group = int(group)
        self.cbuf = None
        self.cpack = None
        self.cscale = None
        if latent_bits == 4:
            self.cpack = _cache_zeros(B, 1, self.cap, R // 2, dtype="uint8")
            self.cscale = _cache_zeros(
                B, 1, self.cap, R // self.group, dtype=BlockTC.COMPUTE_DTYPE)
            p, s = self._pack_latent(c_n)
            with tc.no_grad():
                tc.write_rows(self.cpack, p, 0)
                tc.write_rows(self.cscale, s, 0)
        else:
            self.cbuf = _cache_zeros(B, self.cap, R)
        self.kbuf = _cache_zeros(B, 1, self.cap, ROPE)
        with tc.no_grad():
            tc.write_rows(self.kbuf, k_pe, 0)
            if self.cbuf is not None:
                tc.write_rows(self.cbuf, self._store_latent(c_n), 0)

    @classmethod
    def empty(cls, cfg, seq_len, extra=1024, latent_bits=16, group=32):
        if latent_bits not in (2, 4, 16):
            raise ValueError("DeepSeekMLACache latent_bits must be 2, 4, or 16")
        obj = cls.__new__(cls)
        obj.count = seq_len
        obj.cap = max(seq_len, seq_len + max(0, int(extra)))
        obj.latent_bits = latent_bits
        obj.group = int(group)
        obj.cbuf = None
        obj.cpack = None
        obj.cscale = None
        if latent_bits == 4:
            obj.cpack = _cache_zeros(
                1, 1, obj.cap, cfg.kv_lora_rank // 2, dtype="uint8")
            obj.cscale = _cache_zeros(
                1, 1, obj.cap, cfg.kv_lora_rank // obj.group,
                dtype=BlockTC.COMPUTE_DTYPE)
        else:
            obj.cbuf = _cache_zeros(1, obj.cap, cfg.kv_lora_rank)
        obj.kbuf = _cache_zeros(1, 1, obj.cap, cfg.qk_rope_head_dim)
        return obj

    def _fake_int2_latent(self, c_n):
        B, S, R = c_n.shape
        g = self.group
        if R % g != 0:
            raise ValueError("INT2 latent group must divide rank")
        x = _cast_compute(c_n).reshape([B, S, R // g, g])
        mn = -((-x).max([-1], True))
        mx = x.max([-1], True)
        scale = (mx - mn) * (1.0 / 3.0)
        scale = scale + 1e-8
        q = ((x - mn) / scale).round().clamp(0.0, 3.0)
        return (q * scale + mn).reshape([B, S, R])

    def _store_latent(self, c_n):
        if self.latent_bits == 2:
            return self._fake_int2_latent(c_n)
        return c_n

    def _pack_latent(self, c_n):
        B, S, R = c_n.shape
        c4 = _cast_compute(c_n).reshape([B, 1, S, R])
        return tc.kv_int4_pack(c4, self.group)

    def _latent_slice(self, lo=0, n=0):
        if n == 0:
            n = self.count - lo
        if self.latent_bits == 4:
            B, _, _, half_R = self.cpack.shape
            R = half_R * 2
            c4 = tc.kv_int4_unpack(
                self.cpack, self.cscale, self.group, lo, n,
                BlockTC.COMPUTE_DTYPE)
            return c4.reshape([B, n, R])
        return self.cbuf.slice(1, lo, n)

    def _grow(self, need):
        old = self.count
        new_cap = max(need, self.cap + max(1024, self.cap // 8))
        if self.latent_bits == 4:
            B, _, _, half_R = self.cpack.shape
            R = half_R * 2
        else:
            B, _, R = self.cbuf.shape
        _, _, _, ROPE = self.kbuf.shape
        if self.latent_bits == 4:
            pnew = _cache_zeros(B, 1, new_cap, R // 2, dtype="uint8")
            snew = _cache_zeros(
                B, 1, new_cap, R // self.group, dtype=BlockTC.COMPUTE_DTYPE)
            with tc.no_grad():
                # uint8 tensors do not support slice(); copying the full old
                # capacity is still bounded and preserves valid rows.
                tc.write_rows(pnew, self.cpack, 0)
                tc.write_rows(snew, self.cscale, 0)
            self.cpack = pnew
            self.cscale = snew
            tc.empty_cache()
        else:
            cnew = _cache_zeros(B, new_cap, R)
            with tc.no_grad():
                tc.write_rows(cnew, self.cbuf.slice(1, 0, old), 0)
            self.cbuf = cnew
            tc.empty_cache()
        knew = _cache_zeros(B, 1, new_cap, ROPE)
        with tc.no_grad():
            tc.write_rows(knew, self.kbuf.slice(2, 0, old), 0)
        self.kbuf = knew
        self.cap = new_cap
        tc.empty_cache()

    def append(self, c_new, k_new):
        n = c_new.shape[1]
        if self.count + n > self.cap:
            self._grow(self.count + n)
        if self.latent_bits == 4:
            p, s = self._pack_latent(c_new)
        with tc.no_grad():
            if self.latent_bits == 4:
                tc.write_rows(self.cpack, p, self.count)
                tc.write_rows(self.cscale, s, self.count)
            else:
                tc.write_rows(self.cbuf, self._store_latent(c_new), self.count)
            tc.write_rows(self.kbuf, k_new, self.count)
        self.count += n
        return self

    def tensors(self):
        return self._latent_slice(0, self.count), self.kbuf.slice(2, 0, self.count)

    def __getitem__(self, idx):
        return self.tensors()[idx]

    def __iter__(self):
        return iter(self.tensors())


class DeepSeekMLATC:
    def __init__(self, cfg):
        self.cfg = cfg
        self.q_proj = self.kv_a = self.kv_b = self.o_proj = None
        self.kv_a_norm = RMSNormTC(cfg.kv_lora_rank, cfg.rms_norm_eps)
        self.attention_mode = "standard"
        self.refine_percentile = 0.10
        self.bulk_bits = 4
        self.fast_attn = True
        self.fast_max_seq = 4096
        self.attn_block = 1024
        self.absorbed_decode = True
        self.latent_cache_bits = 16
        self.latent_cache_group = 32
        self._abs_w = None
        self.debug_counts = {
            "absorbed_decode": 0,
            "absorbed_exact_prefill": 0,
            "apa_cublas_blend": 0,
            "apa_fused_selective": 0,
            "standard_sdpa": 0,
        }
        self.debug_last = None
        self.softmax_scale = cfg.q_head_dim ** -0.5
        self.softmax_scale *= _yarn_get_mscale(
            cfg.rope_factor, cfg.rope_mscale_all_dim) ** 2

    def _debug_mark(self, branch, L, S):
        self.debug_counts[branch] = self.debug_counts.get(branch, 0) + 1
        self.debug_last = {
            "branch": branch,
            "L": int(L),
            "S": int(S),
            "refine": float(self.refine_percentile),
            "bulk_bits": int(self.bulk_bits),
            "latent_bits": int(self.latent_cache_bits),
        }

    def _absorbed_weights(self):
        """kv_b split as per-head K/V projection blocks for MLA decode.

        For L==1 decode, scores can be computed exactly as:
            (q_nope W_k) · latent + q_pe · k_pe
        and the weighted latent sum is projected by W_v. This is the real MLA
        memory path; expanding full per-head K/V for the whole cache defeats it.
        """
        if self._abs_w is None:
            cfg = self.cfg
            H, NOPE, V, R = (cfg.num_heads, cfg.qk_nope_head_dim,
                             cfg.v_head_dim, cfg.kv_lora_rank)
            eye = tc.tensor(np.eye(R, dtype=np.float32)).astype(
                BlockTC.COMPUTE_DTYPE)
            wt = self.kv_b(eye).reshape([R, H, NOPE + V])
            w_k = wt.slice(2, 0, NOPE).transpose(0, 1).transpose(1, 2)
            w_v = wt.slice(2, NOPE, V).transpose(0, 1).transpose(1, 2)
            self._abs_w = (w_k, w_v)
        return self._abs_w

    def __call__(self, x, cos, sin, position_offset=0, kv_cache=None):
        cfg = self.cfg
        B, L, _ = x.shape
        H, NOPE, ROPE, V = (cfg.num_heads, cfg.qk_nope_head_dim,
                            cfg.qk_rope_head_dim, cfg.v_head_dim)

        q = self.q_proj(x).reshape([B, L, H, cfg.q_head_dim]).transpose(1, 2)
        q_nope = q.slice(3, 0, NOPE)
        q_pe = q.slice(3, NOPE, ROPE)
        # GRM router capture: position-neutral composite queries, matching the
        # pre-RoPE latent/key space used by harvested MLA grafts.
        if getattr(self, "_capture_q", False):
            self._captured_q = tc.cat([q_nope, q_pe], dim=3).numpy()

        ckv = self.kv_a(x)
        c_kv = ckv.slice(2, 0, cfg.kv_lora_rank)
        k_pe = ckv.slice(2, cfg.kv_lora_rank, ROPE)
        k_pe = k_pe.reshape([B, L, 1, ROPE]).transpose(1, 2)
        c_n = _cast_compute(self.kv_a_norm(c_kv))
        # GRM payload capture: DeepSeek MLA grafts are the post-norm latent plus
        # the unrotated shared RoPE key. Only k_pe is re-seated at injection.
        if getattr(self, "_capture", False):
            self._captured = (c_n.numpy(), k_pe.numpy())

        inj = getattr(self, "inject_kv", None)
        Sg = getattr(self, "graft_seats", 0)
        if inj is not None:
            Sg = int(inj[0].shape[1])
        shift = getattr(self, "live_shift", None)
        if shift is None:
            shift = Sg

        cseg = cos.slice(0, position_offset + shift, L)
        sseg = sin.slice(0, position_offset + shift, L)
        q_pe = F.apply_rotary(_deepseek_rope_prep(q_pe), cseg, sseg)
        k_pe = F.apply_rotary(_deepseek_rope_prep(k_pe), cseg, sseg)

        if inj is not None and kv_cache is None:
            # Prefill-only latent injection. The graft's k_pe is stored
            # unrotated, then re-RoPE'd into arena seats 0..Sg-1 before the
            # live segment. Cached decode sees the graft as ordinary history.
            c_g, kpe_g = inj
            kpe_g = F.apply_rotary(
                _deepseek_rope_prep(kpe_g),
                cos.slice(0, 0, Sg),
                sin.slice(0, 0, Sg))
            c_n = tc.cat([c_g, c_n], dim=1)
            k_pe = tc.cat([kpe_g, k_pe], dim=2)

        new_kv = None
        if isinstance(kv_cache, DeepSeekMLACache):
            new_kv = kv_cache.append(c_n, k_pe)
            c_n, k_pe = new_kv.tensors()
        elif kv_cache is not None and L == 1 and B == 1:
            new_kv = DeepSeekMLACache(
                kv_cache[0], kv_cache[1], extra=1024,
                latent_bits=self.latent_cache_bits,
                group=self.latent_cache_group)
            new_kv.append(c_n, k_pe)
            c_n, k_pe = new_kv.tensors()
        elif kv_cache is not None:
            c_n = tc.cat([kv_cache[0], c_n], dim=1)
            k_pe = tc.cat([kv_cache[1], k_pe], dim=2)
            new_kv = (c_n, k_pe)
        else:
            new_kv = (c_n, k_pe)
        S_all = c_n.shape[1]

        if (self.absorbed_decode and L == 1 and B == 1
                and kv_cache is not None):
            # Decode is already O(S); use exact absorbed MLA instead of APA's
            # expanded-key path. APA remains the long-prefill/chunk mechanism.
            w_k, w_v = self._absorbed_weights()
            qn = q_nope.reshape([H, 1, NOPE])
            qp = q_pe.reshape([H, 1, ROPE])
            c2 = c_n.reshape([S_all, cfg.kv_lora_rank])
            k2 = k_pe.reshape([S_all, ROPE])
            q_lat = tc.matmul(qn, w_k)
            scores = (tc.matmul(q_lat, c2, alpha=self.softmax_scale,
                                trans_b=True) +
                      tc.matmul(qp, k2, alpha=self.softmax_scale,
                                trans_b=True))
            weights = tc.causal_softmax(scores)
            ctx_lat = tc.matmul(weights, c2)
            attn = tc.matmul(ctx_lat, w_v, trans_b=True)
            attn = attn.reshape([B, H, 1, V]).transpose(1, 2).reshape(
                [B, 1, H * V])
            self._debug_mark("absorbed_decode", L, S_all)
            return self.o_proj(_cast_compute(attn)), new_kv

        if self.attention_mode == "absorbed_exact":
            if B != 1:
                raise RuntimeError("absorbed_exact currently supports B=1")
            w_k, w_v = self._absorbed_weights()
            qn = q_nope.reshape([H, L, NOPE])
            qp = q_pe.reshape([H, L, ROPE])
            c2 = c_n.reshape([S_all, cfg.kv_lora_rank])
            k2 = k_pe.reshape([S_all, ROPE])
            q_lat = tc.matmul(qn, w_k)
            scores = (tc.matmul(q_lat, c2, alpha=self.softmax_scale,
                                trans_b=True) +
                      tc.matmul(qp, k2, alpha=self.softmax_scale,
                                trans_b=True))
            weights = tc.causal_softmax(scores) if L > 1 else tc.causal_softmax(scores)
            ctx_lat = tc.matmul(weights, c2)
            attn = tc.matmul(ctx_lat, w_v, trans_b=True)
            attn = attn.reshape([B, H, L, V]).transpose(1, 2).reshape(
                [B, L, H * V])
            self._debug_mark("absorbed_exact_prefill", L, S_all)
            if kv_cache is None:
                new_kv = DeepSeekMLACache(
                    c_n, k_pe, extra=1024,
                    latent_bits=self.latent_cache_bits,
                    group=self.latent_cache_group)
            return self.o_proj(_cast_compute(attn)), new_kv

        kv = self.kv_b(c_n)
        kv = kv.reshape([B, S_all, H, NOPE + V]).transpose(1, 2)
        k_nope = kv.slice(3, 0, NOPE)
        v = kv.slice(3, NOPE, V)
        k_pe_x = k_pe.expand([B, H, S_all, ROPE])

        q_full = tc.cat([q_nope, q_pe], dim=3)
        k_full = tc.cat([k_nope, k_pe_x], dim=3)
        causal = L > 1

        if self.attention_mode == "apa_selective":
            from tensor_cuda.quant import _norm_ppf, _quantize_keys, _tables
            dev = q_full.device.split(":")[0]
            R, CB, BND = _tables(cfg.q_head_dim, self.bulk_bits, H, True, dev)
            kq = _quantize_keys(k_full, R, CB, BND)
            z = _norm_ppf(1.0 - max(0.0, min(1.0, self.refine_percentile)))
            if self.fast_attn and S_all <= self.fast_max_seq:
                budget = 300 * 1024 * 1024
                blk = max(64, min(self.attn_block, int(budget // (H * S_all * 2))))
                self._debug_mark("apa_cublas_blend", L, S_all)
                attn = _cublas_blend_attention(
                    q_full, k_full, kq, v, 1, self.softmax_scale, float(z),
                    causal, blk)
            else:
                self._debug_mark("apa_fused_selective", L, S_all)
                attn = tc.apa_selective_attention(
                    q_full, k_full, kq, v, self.softmax_scale, float(z), causal)
        else:
            self._debug_mark("standard_sdpa", L, S_all)
            attn = F.scaled_dot_product_attention(
                q_full, k_full, v, is_causal=causal, scale=self.softmax_scale)

        attn = attn.transpose(1, 2).reshape([B, L, H * V])
        if kv_cache is None:
            new_kv = DeepSeekMLACache(
                c_n, k_pe, extra=1024,
                latent_bits=self.latent_cache_bits,
                group=self.latent_cache_group)
        return self.o_proj(_cast_compute(attn)), new_kv


class DeepSeekMLAArenaCache(ArenaCache):
    """GRM arena dialect for DeepSeek-V2-Lite MLA payloads."""

    VALS_PER_TOK_LAYER = (DeepSeekV2LiteConfig.kv_lora_rank +
                          DeepSeekV2LiteConfig.qk_rope_head_dim)
    ROPE_PAIR_SWAP = True


class DeepSeekDenseMLPTC:
    # Same law as Mistral's SwiGLU_TC: the FFN is position-wise, so sequence
    # chunks are equivalent while bounding gate/up activation transients during
    # consolidation prefills.
    FFN_CHUNK = 2048

    def __init__(self):
        self.gate_proj = self.up_proj = self.down_proj = None

    def __call__(self, x):
        L = x.shape[1] if len(x.shape) == 3 else 1
        if L <= self.FFN_CHUNK:
            return self.down_proj(self.gate_proj(x).silu() * self.up_proj(x))
        outs = []
        for i in range(0, L, self.FFN_CHUNK):
            end = min(i + self.FFN_CHUNK, L)
            xc = x.slice(1, i, end - i)
            outs.append(self.down_proj(self.gate_proj(xc).silu() * self.up_proj(xc)))
        return tc.cat(outs, dim=1)


class DeepSeekMoETC:
    def __init__(self, cfg):
        self.cfg = cfg
        self.gate_weight = None
        self.experts = [DeepSeekDenseMLPTC() for _ in range(cfg.n_routed_experts)]
        self.shared_experts = DeepSeekDenseMLPTC()

    def _route(self, x_flat):
        scores = tc.matmul(x_flat.float(), self.gate_weight, trans_b=True).softmax(-1)
        return scores.topk(self.cfg.num_experts_per_tok, True)

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
                w = float(topw_np[t, slot] * self.cfg.routed_scaling_factor)
                y = self.experts[e](xt) * w
                acc = y if acc is None else acc + y
            routed.append(acc)
        y = tc.cat(routed, dim=0).reshape([B, L, H])
        return y + self.shared_experts(x)


class DeepSeekV2LiteBlockTC:
    def __init__(self, cfg, layer_idx):
        self.input_layernorm = RMSNormTC(cfg.hidden_dim, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNormTC(cfg.hidden_dim, cfg.rms_norm_eps)
        self.self_attn = DeepSeekMLATC(cfg)
        self.mlp = (DeepSeekMoETC(cfg)
                    if layer_idx >= cfg.first_k_dense_replace
                    and layer_idx % cfg.moe_layer_freq == 0
                    else DeepSeekDenseMLPTC())

    def __call__(self, x, cos, sin, position_offset=0, kv_cache=None):
        attn, new_kv = self.self_attn(
            _cast_compute(self.input_layernorm(x)), cos, sin, position_offset,
            kv_cache)
        h = x + attn
        out = h + self.mlp(_cast_compute(self.post_attention_layernorm(h)))
        return out, new_kv


class DeepSeekV2Lite_TC:
    def __init__(self, cfg=None, attention_mode="standard", max_layers=None,
                 latent_cache_bits=16, latent_cache_group=32):
        self.config = cfg or DeepSeekV2LiteConfig()
        self.max_layers = max_layers
        n_layers = self.config.num_layers if max_layers is None else max_layers
        self.embed_tokens = EmbeddingTC(self.config.vocab_size, self.config.hidden_dim)
        self.layers = [DeepSeekV2LiteBlockTC(self.config, i) for i in range(n_layers)]
        self.norm = RMSNormTC(self.config.hidden_dim, self.config.rms_norm_eps)
        self.lm_head = None
        self._rope_len = 0
        self.extend_rope(4096)
        self.set_attention_mode(attention_mode)
        self.set_latent_cache_bits(latent_cache_bits, latent_cache_group)

    def _build_rope(self, seq_len):
        cfg = self.config
        dim = cfg.qk_rope_head_dim
        base = cfg.rope_theta
        freq_extra = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        freq_inter = 1.0 / (cfg.rope_factor * base ** (
            np.arange(0, dim, 2, dtype=np.float32) / dim))
        low, high = _yarn_find_correction_range(
            cfg.rope_beta_fast, cfg.rope_beta_slow, dim, base,
            cfg.original_max_position_embeddings)
        mask = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2)
        inv = freq_inter * (1.0 - mask) + freq_extra * mask
        pos = np.arange(seq_len, dtype=np.float32)[:, None] * inv[None, :]
        emb = np.concatenate([pos, pos], axis=-1)
        mscale = (_yarn_get_mscale(cfg.rope_factor, cfg.rope_mscale) /
                  _yarn_get_mscale(cfg.rope_factor, cfg.rope_mscale_all_dim))
        dt = BlockTC.COMPUTE_DTYPE
        self.rope_cos = tc.tensor((np.cos(emb) * mscale).astype(np.float32)).astype(dt)
        self.rope_sin = tc.tensor((np.sin(emb) * mscale).astype(np.float32)).astype(dt)
        self._rope_len = seq_len

    def extend_rope(self, seq_len):
        if seq_len > self._rope_len:
            self._build_rope(seq_len)

    def set_attention_mode(self, mode, refine_percentile=0.10, bulk_bits=4):
        for layer in self.layers:
            layer.self_attn.attention_mode = mode
            layer.self_attn.refine_percentile = refine_percentile
            layer.self_attn.bulk_bits = bulk_bits

    def set_latent_cache_bits(self, bits=16, group=32):
        if bits not in (2, 4, 16):
            raise ValueError("latent cache bits must be 2, 4, or 16")
        for layer in self.layers:
            layer.self_attn.latent_cache_bits = bits
            layer.self_attn.latent_cache_group = int(group)

    def set_value_int4(self, enabled=True, group=32):
        """Experiment switch: INT4-store the MLA latent used as decode values.

        In absorbed MLA decode the cached latent is the value source before
        W_v. Quantizing it is the DeepSeek equivalent of quantized V storage.
        """
        self.set_latent_cache_bits(4 if enabled else 16, group=group)

    def warm_absorbed_decode(self):
        """Materialize absorbed MLA decode projections before loading grafts.

        Fresh repository resume can otherwise hit the lazy `_absorbed_weights`
        path only after all active graft payloads have been re-uploaded, which
        leaves less transient room for the kv_b(identity) projection.
        """
        for layer in self.layers:
            with tc.no_grad():
                layer.self_attn._absorbed_weights()
            tc.empty_cache()

    def __call__(self, input_ids_np, kv_caches=None, position_offset=0,
                 last_token_only=False, max_layers=None):
        B, L = input_ids_np.shape
        self.extend_rope(position_offset + L)
        idx = tc.tensor(np.ascontiguousarray(input_ids_np.astype(np.int64)),
                        dtype="int64")
        h = self.embed_tokens(idx)
        new_caches = []
        run_layers = self.layers if max_layers is None else self.layers[:max_layers]
        for i, layer in enumerate(run_layers):
            cache = kv_caches[i] if kv_caches is not None else None
            h, kv = layer(h, self.rope_cos, self.rope_sin, position_offset, cache)
            new_caches.append(kv)
        if max_layers is not None or self.lm_head is None:
            return None, new_caches
        h = _cast_compute(self.norm(h))
        if last_token_only and h.shape[1] > 1:
            h = h.slice(1, h.shape[1] - 1, 1)
        return self.lm_head(h), new_caches

    def _tensor_map(self, model_dir):
        from safetensors import safe_open
        shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if not shards:
            raise FileNotFoundError(f"No safetensors shards found in {model_dir}")
        where = {}
        for shard in shards:
            with safe_open(shard, framework="pt") as f:
                for name in f.keys():
                    where[name] = shard
        return where

    def load_weights(self, model_dir=None, load_lm_head=True, progress=True):
        import torch
        from safetensors import safe_open

        d = model_dir or _snap()
        where = self._tensor_map(d)

        def gf(name):
            with safe_open(where[name], framework="pt") as f:
                return f.get_tensor(name).to(torch.float32).numpy()

        cfg = self.config
        group = cfg.group_size
        loaded = 0
        quant_bytes = 0
        orig_bytes = 0
        source_bf16_linear_bytes = 0
        unquantized_weight_bytes = 0

        emb = gf("model.embed_tokens.weight")
        self.embed_tokens.weight = tc.tensor(np.ascontiguousarray(emb)).astype("bfloat16")
        unquantized_weight_bytes += emb.size * 2
        loaded += 1
        del emb

        for i, layer in enumerate(self.layers):
            p = f"model.layers.{i}"
            w = np.ascontiguousarray(gf(f"{p}.input_layernorm.weight"))
            layer.input_layernorm.weight = tc.tensor(
                w, dtype="float32")
            unquantized_weight_bytes += w.nbytes
            w = np.ascontiguousarray(gf(f"{p}.post_attention_layernorm.weight"))
            layer.post_attention_layernorm.weight = tc.tensor(
                w, dtype="float32")
            unquantized_weight_bytes += w.nbytes
            del w
            loaded += 2

            A = layer.self_attn
            for attr, name in (
                ("q_proj", "q_proj"),
                ("kv_a", "kv_a_proj_with_mqa"),
                ("kv_b", "kv_b_proj"),
                ("o_proj", "o_proj"),
            ):
                w = np.ascontiguousarray(gf(f"{p}.self_attn.{name}.weight"))
                orig_bytes += w.nbytes
                source_bf16_linear_bytes += w.size * 2
                ql = QuantLinearTC(w, group)
                setattr(A, attr, ql)
                quant_bytes += ql.vram_bytes()
                loaded += 1
                del w
            w = np.ascontiguousarray(gf(f"{p}.self_attn.kv_a_layernorm.weight"))
            A.kv_a_norm.weight = tc.tensor(
                w, dtype="float32")
            unquantized_weight_bytes += w.nbytes
            del w
            loaded += 1

            if isinstance(layer.mlp, DeepSeekMoETC):
                M = layer.mlp
                w = np.ascontiguousarray(gf(f"{p}.mlp.gate.weight"))
                M.gate_weight = tc.tensor(
                    w, dtype="float32")
                unquantized_weight_bytes += w.nbytes
                del w
                loaded += 1
                for base, obj in (("shared_experts", M.shared_experts),):
                    for nm in ("gate_proj", "up_proj", "down_proj"):
                        w = np.ascontiguousarray(gf(f"{p}.mlp.{base}.{nm}.weight"))
                        orig_bytes += w.nbytes
                        source_bf16_linear_bytes += w.size * 2
                        ql = QuantLinearTC(w, group)
                        setattr(obj, nm, ql)
                        quant_bytes += ql.vram_bytes()
                        loaded += 1
                        del w
                for e, expert in enumerate(M.experts):
                    for nm in ("gate_proj", "up_proj", "down_proj"):
                        w = np.ascontiguousarray(
                            gf(f"{p}.mlp.experts.{e}.{nm}.weight"))
                        orig_bytes += w.nbytes
                        source_bf16_linear_bytes += w.size * 2
                        ql = QuantLinearTC(w, group)
                        setattr(expert, nm, ql)
                        quant_bytes += ql.vram_bytes()
                        loaded += 1
                        del w
            else:
                for nm in ("gate_proj", "up_proj", "down_proj"):
                    w = np.ascontiguousarray(gf(f"{p}.mlp.{nm}.weight"))
                    orig_bytes += w.nbytes
                    source_bf16_linear_bytes += w.size * 2
                    ql = QuantLinearTC(w, group)
                    setattr(layer.mlp, nm, ql)
                    quant_bytes += ql.vram_bytes()
                    loaded += 1
                    del w

            gc.collect()
            if progress:
                print(f"    layer {i + 1}/{len(self.layers)}", flush=True)

        w = np.ascontiguousarray(gf("model.norm.weight"))
        self.norm.weight = tc.tensor(w, dtype="float32")
        unquantized_weight_bytes += w.nbytes
        del w
        loaded += 1

        if load_lm_head:
            w = np.ascontiguousarray(gf("lm_head.weight"))
            orig_bytes += w.nbytes
            source_bf16_linear_bytes += w.size * 2
            self.lm_head = QuantLinearTC(w, group)
            quant_bytes += self.lm_head.vram_bytes()
            loaded += 1
            del w

        gc.collect()
        resident_weight_bytes = quant_bytes + unquantized_weight_bytes
        return {
            "loaded": loaded,
            "framework": "tensor_cuda DeepSeek-V2-Lite INT4",
            "group_size": group,
            "original_bytes": orig_bytes,
            "source_bf16_linear_bytes": source_bf16_linear_bytes,
            "quantized_bytes": quant_bytes,
            "quantized_linear_bytes": quant_bytes,
            "unquantized_weight_bytes": unquantized_weight_bytes,
            "estimated_resident_weight_bytes": resident_weight_bytes,
            "estimated_resident_weight_gib": resident_weight_bytes / (1024 ** 3),
            "compression_ratio": (orig_bytes / quant_bytes) if quant_bytes else 0.0,
            "layers_loaded": len(self.layers),
            "lm_head_loaded": bool(load_lm_head),
        }

    @classmethod
    def from_pretrained(cls, model_dir=None, attention_mode="standard",
                        max_layers=None, load_lm_head=True, progress=True,
                        latent_cache_bits=16, latent_cache_group=32):
        BlockTC.COMPUTE_DTYPE = "bfloat16"
        QuantLinearTC.FUSED_DECODE = True
        RMSNormTC.USE_FUSED = True
        with tc.no_grad():
            model = cls(attention_mode=attention_mode, max_layers=max_layers,
                        latent_cache_bits=latent_cache_bits,
                        latent_cache_group=latent_cache_group)
            info = model.load_weights(model_dir, load_lm_head=load_lm_head,
                                      progress=progress)
        return model, info


def describe_snapshot(model_dir=None):
    d = model_dir or _snap()
    idx = os.path.join(d, "model.safetensors.index.json")
    out = {"model_dir": d}
    if os.path.exists(idx):
        with open(idx) as f:
            j = json.load(f)
        out["total_size"] = j.get("metadata", {}).get("total_size")
        out["num_tensors"] = len(j.get("weight_map", {}))
    return out
