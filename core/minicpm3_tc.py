"""MiniCPM3-4B on tensor_cuda — first MLA (Multi-head Latent Attention) model.

MLA dataflow (per the modeling file, implementation-exact):
  q  = q_b( RMSNorm( q_a(x) ) )                # 2560 -> 768 -> 40*96
       split per head: q_nope[:64] | q_pe[64:96]
  ckv= kv_a(x)                                  # 2560 -> 288
       split: c_kv[:256] | k_pe[256:288]        # k_pe is SHARED across heads (MQA-style)
  kv = kv_b( RMSNorm( c_kv ) )                  # 256 -> 40*(64+64)
       split per head: k_nope[:64] | v[64:128]
  RoPE on q_pe and k_pe ONLY (32 dims); k_pe broadcast to all 40 heads.
  q = [q_nope|q_pe] (96), k = [k_nope|k_pe] (96), attn scale = 96^-0.5, v = 64.
  o_proj: 40*64 -> 2560.

muP scalings (silent-garbage-if-missed):
  embeddings * scale_emb (12);
  every residual add * scale_depth/sqrt(62) (~0.1778);
  hidden / (hidden_size/dim_model_base) (=10) BEFORE the tied lm_head.

Rope: "longrope" with per-dim factor table, but max_pos == original_max (32768)
so mscale degenerates to 1.0 — just divide each inv_freq dim by its factor.

This implementation assembles full K/V per forward (correctness-first, same
numbers as the reference). Caching the 288-d LATENT instead — the actual MLA
memory win (~4x less KV than Qwen3-4B) — is the follow-up optimization.
"""
import os
import glob
import math
import numpy as np

from core.mistral7b_tc import (QuantLinearTC, RMSNormTC, EmbeddingTC, BlockTC,
                               F, tc)
from core.qwen_tc import _TiedLMHead


class MiniCPM3Config:
    vocab_size = 73448
    hidden_dim = 2560
    intermediate_dim = 6400
    num_layers = 62
    num_heads = 40
    q_lora_rank = 768
    kv_lora_rank = 256
    qk_nope_head_dim = 64
    qk_rope_head_dim = 32
    v_head_dim = 64                  # hidden/heads = 2560/40
    q_head_dim = 96                  # nope + rope
    max_position_embeddings = 32768
    rope_theta = 10000.0
    rms_norm_eps = 1e-5
    scale_emb = 12.0
    scale_depth = 1.4
    dim_model_base = 256
    rope_factors = [1.0591234137867171, 1.1241891283591912, 1.2596935748670968,
                    1.5380380402321725, 2.093982484148734, 3.1446935121267696,
                    4.937952647693647, 7.524541999994549, 10.475458000005451,
                    13.062047352306353, 14.85530648787323, 15.906017515851266,
                    16.461961959767827, 16.740306425132907, 16.87581087164081,
                    16.940876586213285]


_SNAP_GLOB = "/home/vader/.cache/huggingface/hub/models--openbmb--MiniCPM3-4B/snapshots/*"


def _snap():
    hits = sorted(glob.glob(_SNAP_GLOB))
    if not hits:
        raise FileNotFoundError("MiniCPM3-4B snapshot not found")
    return hits[-1]


class MLAAttentionTC:
    def __init__(self, cfg):
        self.cfg = cfg
        self.q_a = self.q_b = self.kv_a = self.kv_b = self.o_proj = None
        self.q_a_norm = RMSNormTC(cfg.q_lora_rank, cfg.rms_norm_eps)
        self.kv_a_norm = RMSNormTC(cfg.kv_lora_rank, cfg.rms_norm_eps)
        # APA on MLA: composite 96-dim keys, 64-dim values (D_v != D_qk).
        # The cuBLAS-blend path is dim-agnostic (used for S <= fast_max_seq);
        # the fused kernel assumes one D, so beyond that we zero-pad V to 96
        # and slice the output back to 64.
        self.attention_mode = "standard"
        self.refine_percentile = 0.10
        self.bulk_bits = 8
        self.fast_max_seq = 4096
        self.attn_block = 1024
        # Absorbed decode (DeepSeek-V2 style): at L==1 score the LATENT
        # directly — (W_uk^T q_nope)·c_n + q_pe·k_pe — and apply W_uv after
        # the attention-weighted latent sum. No kv_b re-expansion of the
        # span per step. Off by default until gated; expanded path is the
        # validated reference.
        self.absorbed_decode = False
        self._abs_w = None

    def _absorbed_weights(self):
        """Dequantized per-head kv_b blocks for the absorbed-decode path:
        Wuk (H, NOPE, R) and Wuv (H, V, R). Built once, lazily, by running
        kv_b on the identity — the EXACT dequantization the INT4 kernel
        applies, so absorbed scores reassociate the same numbers. ~2.6MB
        fp16 per layer."""
        if self._abs_w is None:
            cfg = self.cfg
            H, NOPE, V = cfg.num_heads, cfg.qk_nope_head_dim, cfg.v_head_dim
            R = cfg.kv_lora_rank
            eye = tc.tensor(np.eye(R, dtype=np.float32)).astype(BlockTC.COMPUTE_DTYPE)
            WT = self.kv_b(eye).reshape([R, H, NOPE + V])     # == W^T, kernel-exact
            Wuk = WT.slice(2, 0, NOPE).transpose(0, 1).transpose(1, 2)   # (H,NOPE,R)
            Wuv = WT.slice(2, NOPE, V).transpose(0, 1).transpose(1, 2)   # (H,V,R)
            self._abs_w = (Wuk, Wuv)
        return self._abs_w

    def __call__(self, x, cos, sin, position_offset=0, kv_cache=None):
        cfg = self.cfg
        B, L, _ = x.shape
        H, NOPE, ROPE, V = cfg.num_heads, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
        cdt = BlockTC.COMPUTE_DTYPE

        # ---- q path: down-project, norm, up-project, split nope|pe
        qa = self.q_a_norm(self.q_a(x)).astype(cdt)
        q = self.q_b(qa).reshape([B, L, H, cfg.q_head_dim]).transpose(1, 2)
        q_nope = q.slice(3, 0, NOPE)
        q_pe = q.slice(3, NOPE, ROPE)
        # Router capture (Graft Repository): composite PRE-RoPE queries
        # [q_nope|q_pe] (B,H,L,96) — position-neutral, dot against harvested
        # pre-RoPE graft keys for routing (E1).
        if getattr(self, "_capture_q", False):
            self._captured_q = tc.cat([q_nope, q_pe], dim=3).numpy()

        # ---- kv path: joint down-project, split latent | shared rope-key
        ckv = self.kv_a(x)                                  # (B, L, 256+32)
        c_kv = ckv.slice(2, 0, cfg.kv_lora_rank)
        k_pe = ckv.slice(2, cfg.kv_lora_rank, ROPE)
        k_pe = k_pe.reshape([B, L, 1, ROPE]).transpose(1, 2)  # (B,1,L,32)
        c_n = self.kv_a_norm(c_kv).astype(cdt)              # post-norm latent

        # KV-Graft capture (MLA): the LATENT graft = position-free post-norm
        # latent + PRE-RoPE shared rope-key, 288 vals/token/layer — ~22x
        # smaller artifacts than full per-head K/V. Only k_pe needs re-RoPE
        # at injection; c_n re-seats for free.
        if getattr(self, "_capture", False):
            self._captured = (c_n.numpy(), k_pe.numpy())

        # ---- RoPE: q for this segment's absolute positions; k_pe likewise.
        # Live tokens sit AFTER the graft's Sg seats on EVERY step the graft
        # is resident (graft_seats persists through cached decode — same
        # position law as GQAAttentionTC).
        inj = getattr(self, "inject_kv", None)
        Sg = getattr(self, "graft_seats", 0)
        if inj is not None:
            Sg = inj[0].shape[1]
        # Arena mode (live_shift set): live positions shift by the FIXED
        # arena width — mounts occupy a PREFIX of the arena and the unused
        # remainder is a positional hole, so swapping mounts never moves
        # live tokens. Without live_shift: legacy behavior, shift = mount size.
        shift = getattr(self, "live_shift", None)
        if shift is None:
            shift = Sg
        cseg = cos.slice(0, position_offset + shift, L)
        sseg = sin.slice(0, position_offset + shift, L)
        q_pe = F.apply_rotary(q_pe, cseg, sseg)
        k_pe = F.apply_rotary(k_pe, cseg, sseg)

        if inj is not None and kv_cache is None:
            # PREFILL-ONLY latent injection: re-RoPE the graft's shared key at
            # seats 0..Sg-1 and cat (latent, k_pe) in front. Enters new_kv, so
            # the graft lives in the persistent cache from then on; the kv_b
            # expansion below covers the graft span like any cached token.
            c_g, kpe_g = inj
            kpe_g = F.apply_rotary(kpe_g, cos.slice(0, 0, Sg), sin.slice(0, 0, Sg))
            c_n = tc.cat([c_g, c_n], dim=1)
            k_pe = tc.cat([kpe_g, k_pe], dim=2)

        # ---- MLA LATENT CACHE: the cache holds (post-norm latent, RoPE'd shared
        # key) = 288 values/token/layer — ~18x smaller than the assembled
        # per-head K/V (40 heads x 160 dims). New tokens append; K/V for the
        # whole span are re-expanded from the latent below (assembled
        # transiently, freed after the step — the RESIDENT footprint is latent).
        if kv_cache is not None:
            c_n = tc.cat([kv_cache[0], c_n], dim=1)          # (B, S, 256)
            k_pe = tc.cat([kv_cache[1], k_pe], dim=2)        # (B, 1, S, 32)
        new_kv = (c_n, k_pe)
        S_all = c_n.shape[1]

        if (getattr(self, "absorbed_decode", False) and L == 1 and B == 1
                and kv_cache is not None
                and self.attention_mode != "apa_selective"):
            # ---- ABSORBED DECODE: attention in latent space, no expansion.
            # scores = scale*(q_nope W_uk)·c_n + scale*q_pe·k_pe   (per head;
            # c_n/k_pe are head-shared — MQA over 256+32 dims). Output =
            # (softmax · c_n) W_uv^T. Exactly the expanded math, reassociated.
            Wuk, Wuv = self._absorbed_weights()
            scale = cfg.q_head_dim ** -0.5
            qn = q_nope.reshape([H, 1, NOPE])
            qp = q_pe.reshape([H, 1, ROPE])
            c2 = c_n.reshape([S_all, cfg.kv_lora_rank])
            k2 = k_pe.reshape([S_all, ROPE])
            qa = tc.matmul(qn, Wuk)                              # (H,1,R)
            s = tc.matmul(qa, c2, alpha=scale, trans_b=True) \
              + tc.matmul(qp, k2, alpha=scale, trans_b=True)    # (H,1,S)
            w = tc.causal_softmax(s)                             # L=1: full row
            ctxl = tc.matmul(w, c2)                              # (H,1,R)
            attn = tc.matmul(ctxl, Wuv, trans_b=True)            # (H,1,V)
            attn = attn.reshape([B, H, 1, V]).transpose(1, 2).reshape([B, 1, H * V])
            attn = attn.half() if cdt == "float16" else attn.astype(cdt)
            return self.o_proj(attn), new_kv

        # ---- expand latent -> per-head k_nope | v for the WHOLE span
        kv = self.kv_b(c_n)
        kv = kv.reshape([B, S_all, H, NOPE + V]).transpose(1, 2)  # (B,H,S,128)
        k_nope = kv.slice(3, 0, NOPE)
        v = kv.slice(3, NOPE, V)
        k_pe_x = k_pe.expand([B, H, S_all, ROPE])            # share across heads

        # ---- assemble composite q/k and attend (scale = 96^-0.5). With a cache,
        # queries are the LAST L of S_all keys — the bottom-right causal mask
        # (engine fix 63960a4) makes the rectangular case correct.
        q_full = tc.cat([q_nope, q_pe], dim=3)
        k_full = tc.cat([k_nope, k_pe_x], dim=3)
        causal = (L > 1)
        if self.attention_mode == "apa_selective":
            # APA on MLA: quantize the composite 96-dim keys (TurboQuant) and
            # refine only the top fraction. MHA here (KVH == H, group 1).
            from tensor_cuda.quant import _tables, _quantize_keys, _norm_ppf
            from core.mistral7b_tc import _cublas_blend_attention
            dev = q_full.device.split(":")[0]
            R, CB, BND = _tables(cfg.q_head_dim, self.bulk_bits, H, True, dev)
            kq = _quantize_keys(k_full, R, CB, BND)
            z = _norm_ppf(1.0 - max(0.0, min(1.0, self.refine_percentile)))
            scale = cfg.q_head_dim ** -0.5
            S = k_full.shape[2]
            if S <= self.fast_max_seq:
                # cuBLAS-blend path is dim-agnostic (D_v=64 != D_qk=96 is fine)
                budget = 300 * 1024 * 1024
                blk = max(64, min(self.attn_block, int(budget // (H * S * 2))))
                attn = _cublas_blend_attention(q_full, k_full, kq, v, 1,
                                               scale, float(z), causal, blk)
            else:
                # fused kernel assumes one D for keys AND values: zero-pad V to
                # 96, slice the output back to 64.
                pad = tc.zeros(B, H, S, cfg.q_head_dim - V,
                               dtype=BlockTC.COMPUTE_DTYPE)
                vpad = tc.cat([v, pad], dim=3)
                attn = tc.apa_selective_attention(q_full, k_full, kq, vpad,
                                                  scale, float(z), causal)
                attn = attn.slice(3, 0, V)
        else:
            attn = F.scaled_dot_product_attention(q_full, k_full, v, is_causal=causal)

        attn = attn.transpose(1, 2).reshape([B, L, H * V])
        attn = attn.half() if cdt == "float16" else attn.astype(cdt)
        return self.o_proj(attn), new_kv


class SwiGLUTC:
    def __init__(self):
        self.gate_proj = self.up_proj = self.down_proj = None

    def __call__(self, x):
        return self.down_proj(self.gate_proj(x).silu() * self.up_proj(x))


class MiniCPM3BlockTC:
    def __init__(self, cfg):
        self.input_layernorm = RMSNormTC(cfg.hidden_dim, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNormTC(cfg.hidden_dim, cfg.rms_norm_eps)
        self.self_attn = MLAAttentionTC(cfg)
        self.mlp = SwiGLUTC()
        self.res_scale = cfg.scale_depth / math.sqrt(cfg.num_layers)

    def _cast(self, t):
        cdt = BlockTC.COMPUTE_DTYPE
        return t.half() if cdt == "float16" else t.astype(cdt)

    def __call__(self, x, cos, sin, position_offset=0, kv_cache=None):
        a, new_kv = self.self_attn(self._cast(self.input_layernorm(x)), cos, sin,
                                   position_offset, kv_cache)
        h = x + a * self.res_scale
        mo = self.mlp(self._cast(self.post_attention_layernorm(h)))
        return h + mo * self.res_scale, new_kv


class MiniCPM3_TC:
    def __init__(self, cfg):
        self.config = cfg
        self.embed_tokens = EmbeddingTC(cfg.vocab_size, cfg.hidden_dim)
        self.layers = [MiniCPM3BlockTC(cfg) for _ in range(cfg.num_layers)]
        self.norm = RMSNormTC(cfg.hidden_dim, cfg.rms_norm_eps)
        self.lm_head = None
        self._rope_len = 0
        self.extend_rope(4096)

    def extend_rope(self, seq_len):
        if seq_len <= self._rope_len:
            return
        cfg = self.config
        d = cfg.qk_rope_head_dim
        inv = 1.0 / (cfg.rope_theta ** (np.arange(0, d, 2, np.float32) / d))
        inv = inv / np.array(cfg.rope_factors, np.float32)   # longrope; mscale=1
        pos = np.arange(seq_len, dtype=np.float32)[:, None] * inv[None, :]
        emb = np.concatenate([pos, pos], axis=-1)
        dt = BlockTC.COMPUTE_DTYPE
        self.rope_cos = tc.tensor(np.cos(emb).astype(np.float32)).astype(dt)
        self.rope_sin = tc.tensor(np.sin(emb).astype(np.float32)).astype(dt)
        self._rope_len = seq_len

    def __call__(self, input_ids_np, kv_caches=None, position_offset=0,
                 last_token_only=False, max_layers=None):
        cfg = self.config
        B, L = input_ids_np.shape
        self.extend_rope(position_offset + L)
        idx = tc.tensor(np.ascontiguousarray(input_ids_np.astype(np.int64)), dtype="int64")
        h = self.embed_tokens(idx) * cfg.scale_emb          # muP: x12
        new_caches = []
        run = self.layers if max_layers is None else self.layers[:max_layers]
        for i, layer in enumerate(run):
            cache = kv_caches[i] if kv_caches is not None else None
            h, kv = layer(h, self.rope_cos, self.rope_sin, position_offset, cache)
            new_caches.append(kv)
        if max_layers is not None:
            # partial forward (router capture etc.) — no head, no logits
            return None, new_caches
        h = self.norm(h)
        cdt = BlockTC.COMPUTE_DTYPE
        h = h.half() if cdt == "float16" else h.astype(cdt)
        if last_token_only and h.shape[1] > 1:
            h = h.slice(1, h.shape[1] - 1, 1)
        h = h * (1.0 / (cfg.hidden_dim / cfg.dim_model_base))  # muP: /10 pre-head
        return self.lm_head(h), new_caches

    def load_weights(self, model_dir=None):
        import torch
        d = model_dir or _snap()
        # this repo ships pytorch_model.bin (no safetensors); mmap to avoid a
        # full 8GB RAM copy.
        W = torch.load(os.path.join(d, "pytorch_model.bin"),
                       map_location="cpu", weights_only=True, mmap=True)

        def g(name):
            return W.pop(name).to(torch.float32).numpy()

        # no biases expected (attention_bias defaults False); fail loud if present
        assert not any(k.endswith("q_a_proj.bias") for k in W), "unexpected attention bias"

        emb = g("model.embed_tokens.weight").astype(np.float32)
        self.embed_tokens.weight = tc.tensor(np.ascontiguousarray(emb)).astype("bfloat16")

        for i in range(self.config.num_layers):
            p = f"model.layers.{i}"
            L = self.layers[i]
            L.input_layernorm.weight = tc.tensor(
                np.ascontiguousarray(g(f"{p}.input_layernorm.weight")), dtype="float32")
            L.post_attention_layernorm.weight = tc.tensor(
                np.ascontiguousarray(g(f"{p}.post_attention_layernorm.weight")), dtype="float32")
            A = L.self_attn
            A.q_a = QuantLinearTC(np.ascontiguousarray(g(f"{p}.self_attn.q_a_proj.weight")))
            A.q_a_norm.weight = tc.tensor(
                np.ascontiguousarray(g(f"{p}.self_attn.q_a_layernorm.weight")), dtype="float32")
            A.q_b = QuantLinearTC(np.ascontiguousarray(g(f"{p}.self_attn.q_b_proj.weight")))
            A.kv_a = QuantLinearTC(np.ascontiguousarray(g(f"{p}.self_attn.kv_a_proj_with_mqa.weight")))
            A.kv_a_norm.weight = tc.tensor(
                np.ascontiguousarray(g(f"{p}.self_attn.kv_a_layernorm.weight")), dtype="float32")
            A.kv_b = QuantLinearTC(np.ascontiguousarray(g(f"{p}.self_attn.kv_b_proj.weight")))
            A.o_proj = QuantLinearTC(np.ascontiguousarray(g(f"{p}.self_attn.o_proj.weight")))
            L.mlp.gate_proj = QuantLinearTC(np.ascontiguousarray(g(f"{p}.mlp.gate_proj.weight")))
            L.mlp.up_proj = QuantLinearTC(np.ascontiguousarray(g(f"{p}.mlp.up_proj.weight")))
            L.mlp.down_proj = QuantLinearTC(np.ascontiguousarray(g(f"{p}.mlp.down_proj.weight")))
            import gc; gc.collect()
            if (i + 1) % 10 == 0:
                print(f"    layer {i+1}/{self.config.num_layers}", flush=True)

        self.norm.weight = tc.tensor(
            np.ascontiguousarray(g("model.norm.weight")), dtype="float32")
        self.lm_head = _TiedLMHead(self.embed_tokens.weight)  # tied, UNscaled weight
        W.clear()
        return {
            "loaded": f"INT{QuantLinearTC.WEIGHT_BITS} MLA",
            "framework": "tensor_cuda MiniCPM3-4B",
            "weight_bits": QuantLinearTC.WEIGHT_BITS,
        }

    @classmethod
    def from_pretrained(cls, model_dir=None):
        cfg = MiniCPM3Config()
        from core.mistral7b_tc import LinearTC
        BlockTC.COMPUTE_DTYPE = "bfloat16"
        LinearTC.DTYPE = "bfloat16"
        with tc.no_grad():
            m = cls(cfg)
            info = m.load_weights(model_dir)
        return m, info
