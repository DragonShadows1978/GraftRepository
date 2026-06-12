"""Qwen3.5-9B (hybrid Gated-DeltaNet + gated attention) on tensor_cuda, INT4.

Text-only port of Qwen/Qwen3.5-9B: 32 layers = 24 GatedDeltaNet +
8 full-attention (indices 3,7,...,31), GQA 16q/4kv head_dim 256 with
per-head qk-norm and an elementwise sigmoid output gate, partial RoPE
(first 64 of 256 dims, theta 1e7; mRoPE is a no-op for text), SwiGLU
12288, vocab 248320 untied. Ledger: docs/QWEN35_PORT_LEDGER.md.

Parity traps honored here (see ledger for sources):
  - standard RMSNorms multiply by (1 + w): we bake 1+w into the
    weights AT LOAD; the DeltaNet gated norm uses plain w.
  - q_proj is fused per head as [q(256) | gate(256)]; the gate
    bypasses q_norm and RoPE.
  - DeltaNet: l2norm(q,k) INSIDE the rule (eps 1e-6 on the SUM),
    q scaled 1/sqrt(128) after l2norm, state decays BEFORE the delta
    correction, state/g/beta in fp32 (mamba_ssm_dtype).
  - conv1d is depthwise k=4, NO bias, SiLU after; causal left-pad 3.

Cache protocol (per layer, by layer type):
  attention  -> (k, v): (B, 4, S, 256) bf16 each, cat on dim 2
  deltanet   -> (conv, S): conv (B, 3, 8192) fp32 raw pre-conv tail;
                S (B, 32, 128, 128) fp32 recurrent state
Both are Markovian -> lossless save/restore (the GRM entry point).
"""
import gc
import glob
import os

import numpy as np

from core.mistral7b_tc import (BlockTC, LinearTC, QuantLinearTC, RMSNormTC,
                               _repeat_kv, F, tc)


class Qwen35Config:
    vocab_size = 248320
    hidden_dim = 4096
    intermediate_dim = 12288
    num_layers = 32
    full_attention_interval = 4          # attention at i % 4 == 3
    # full attention
    num_heads = 16
    num_kv_heads = 4
    head_dim = 256
    rope_theta = 10000000.0
    partial_rotary_dim = 64              # 0.25 * 256
    # gated deltanet
    n_k_heads = 16
    n_v_heads = 32
    d_k = 128
    d_v = 128
    conv_kernel = 4
    rms_norm_eps = 1e-6
    eos_token_id = 248044

    @staticmethod
    def is_attention(i):
        return i % 4 == 3


_SNAP_GLOB = ("/home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/"
              "snapshots/*")


def _snap():
    hits = sorted(glob.glob(_SNAP_GLOB))
    if not hits:
        raise FileNotFoundError("Qwen3.5-9B snapshot not found in HF cache")
    return hits[-1]


def _cast(t):
    cdt = BlockTC.COMPUTE_DTYPE
    return t.half() if cdt == "float16" else t.astype(cdt)


class HostEmbedding:
    """248320 x 4096 is 2GB+ even in bf16 — too big to sit on the 8GB
    card next to the INT4 body. Rows are gathered on HOST and uploaded
    per call ((B,L,4096) fp32, tiny)."""

    def __init__(self):
        self.weight = None               # np fp32 (V, H)

    def __call__(self, ids_np):
        rows = self.weight[ids_np.reshape(-1)]
        out = np.ascontiguousarray(
            rows.reshape(ids_np.shape + (self.weight.shape[1],)))
        return _cast(tc.tensor(out))


class F32Linear:
    """Tiny exact projection (the DeltaNet a/b heads, 4096->32): kept
    fp32 and unquantized — these drive the decay/write gates and INT4
    noise there compounds through the recurrence."""

    def __init__(self, w_fp32):
        self.wT = tc.tensor(np.ascontiguousarray(w_fp32.T))   # (in, out) fp32

    def __call__(self, x):
        return tc.matmul(x.float(), self.wT)


def _softplus(x):
    # log(1+e^x) = relu(x) + log(1 + e^{-|x|}) — overflow-safe compose.
    return x.relu() + ((x.abs() * (-1.0)).exp() + 1.0).log()


def _per_head_rmsnorm(x, w, eps, B, L, H, D):
    """Qwen3.5 per-head qk-norm; w already has the (1+w) bake."""
    xf = x.reshape([B, L, H, D]).float()
    ms = (xf * xf).mean([-1], True)
    return _cast(xf * (ms + eps).pow(-0.5) * w)


class GatedDeltaNetTC:
    def __init__(self, cfg):
        self.cfg = cfg
        self.in_proj_qkv = self.in_proj_z = self.out_proj = None
        self.in_proj_a = self.in_proj_b = None
        self.conv_w = None               # list of 4 (1,1,8192) fp32
        self.neg_A = None                # (1,1,32) fp32: -exp(A_log)
        self.dt_bias = None              # (1,1,32) fp32
        self.norm_w = None               # (128,) fp32 PLAIN-w gated norm

    def __call__(self, x, state=None):
        """x: (B, L, hidden) compute-dtype. state: (conv, S) or None.
        Returns (out (B, L, hidden), new_state)."""
        cfg = self.cfg
        B, L, _ = x.shape
        H, Dk, Dv = cfg.n_v_heads, cfg.d_k, cfg.d_v
        K = cfg.n_k_heads * cfg.d_k                      # 2048

        qkv = self.in_proj_qkv(x).float()                # (B,L,8192)
        z = self.in_proj_z(x).float()                    # (B,L,4096)
        a = self.in_proj_a(x)                            # (B,L,32) fp32
        b = self.in_proj_b(x)

        # ---- causal depthwise conv (k=4, no bias) + SiLU, fp32.
        # conv state = last 3 RAW (pre-conv) token rows.
        if state is None:
            conv_prev = tc.tensor(np.zeros((B, 3, 2 * K + H * Dv), np.float32))
        else:
            conv_prev = state[0]
        x_cat = tc.cat([conv_prev, qkv], dim=1)          # (B, L+3, 8192)
        conv = (x_cat.slice(1, 0, L) * self.conv_w[0]
                + x_cat.slice(1, 1, L) * self.conv_w[1]
                + x_cat.slice(1, 2, L) * self.conv_w[2]
                + x_cat.slice(1, 3, L) * self.conv_w[3]).silu()
        new_conv = x_cat.slice(1, L, 3)                  # last 3 raw rows

        q = conv.slice(2, 0, K).reshape([B, L, cfg.n_k_heads, Dk])
        k = conv.slice(2, K, K).reshape([B, L, cfg.n_k_heads, Dk])
        v = conv.slice(2, 2 * K, H * Dv).reshape([B, L, H, Dv])

        # v-heads 2i and 2i+1 share k/q head i (repeat_interleave x2)
        rep = H // cfg.n_k_heads
        q = q.reshape([B, L, cfg.n_k_heads, 1, Dk]).expand(
            [B, L, cfg.n_k_heads, rep, Dk]).reshape([B, L, H, Dk])
        k = k.reshape([B, L, cfg.n_k_heads, 1, Dk]).expand(
            [B, L, cfg.n_k_heads, rep, Dk]).reshape([B, L, H, Dk])

        # l2norm (SUM + eps, not mean) then q scale — per the HF kernel
        q = q * ((q * q).sum([-1], True) + 1e-6).pow(-0.5) * (Dk ** -0.5)
        k = k * ((k * k).sum([-1], True) + 1e-6).pow(-0.5)

        beta = b.sigmoid()                               # (B,L,32) fp32
        g = self.neg_A * _softplus(a + self.dt_bias)     # (B,L,32) fp32

        S = (tc.tensor(np.zeros((B, H, Dk, Dv), np.float32))
             if state is None else state[1])
        outs = []
        for t in range(L):
            q_t = q.slice(1, t, 1).reshape([B, H, Dk, 1])
            k_t = k.slice(1, t, 1).reshape([B, H, Dk, 1])
            v_t = v.slice(1, t, 1).reshape([B, H, 1, Dv])
            g_t = g.slice(1, t, 1).reshape([B, H, 1, 1]).exp()
            beta_t = beta.slice(1, t, 1).reshape([B, H, 1, 1])
            S = S * g_t                                  # decay FIRST
            kv_mem = (S * k_t).sum([2], True)            # (B,H,1,Dv)
            delta = (v_t - kv_mem) * beta_t              # (B,H,1,Dv)
            S = S + k_t * delta                          # outer product
            outs.append((S * q_t).sum([2], True))        # (B,H,1,Dv)
        o = tc.cat(outs, dim=2)                          # (B,H,L,Dv)
        o = o.transpose(1, 2)                            # (B,L,H,Dv)

        # gated RMSNorm per head (PLAIN w), gate = silu(z), all fp32
        of = o.reshape([B, L, H, Dv])
        ms = (of * of).mean([-1], True)
        of = of * (ms + cfg.rms_norm_eps).pow(-0.5) * self.norm_w
        of = of * z.reshape([B, L, H, Dv]).silu()
        out = self.out_proj(_cast(of.reshape([B, L, H * Dv])))
        return out, (new_conv, S)


class Qwen35AttentionTC:
    def __init__(self, cfg):
        self.cfg = cfg
        self.q_proj = self.k_proj = self.v_proj = self.o_proj = None
        self.q_norm_w = self.k_norm_w = None             # (256,) fp32, 1+w baked
        # APA dials (qk-norm family -> bulk_bits 4 per the measured law).
        # cuBLAS blend path only — the fused kernel's compile-time head_dim
        # templates stop at 128 and this model is 256.
        self.attention_mode = "standard"
        self.refine_percentile = 0.15
        self.bulk_bits = 4
        self.attn_block = 1024

    def __call__(self, x, cos, sin, position_offset=0, kv_cache=None):
        cfg = self.cfg
        B, L, _ = x.shape
        H, KV, D, R = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, \
            cfg.partial_rotary_dim

        qg = self.q_proj(x).reshape([B, L, H, 2 * D])
        q = qg.slice(3, 0, D).reshape([B, L, H * D])
        gate = qg.slice(3, D, D).reshape([B, L, H * D])  # bypasses norm+rope

        q = _per_head_rmsnorm(q, self.q_norm_w, cfg.rms_norm_eps, B, L, H, D)
        k = _per_head_rmsnorm(self.k_proj(x), self.k_norm_w,
                              cfg.rms_norm_eps, B, L, KV, D)
        q = q.reshape([B, L, H, D]).transpose(1, 2)      # (B,H,L,D)
        k = k.reshape([B, L, KV, D]).transpose(1, 2)
        v = self.v_proj(x).reshape([B, L, KV, D]).transpose(1, 2)

        # partial RoPE: rotate dims [0:64], pass [64:256]
        cseg = cos.slice(0, position_offset, L)
        sseg = sin.slice(0, position_offset, L)
        q = tc.cat([F.apply_rotary(q.slice(3, 0, R), cseg, sseg),
                    q.slice(3, R, D - R)], dim=3)
        k = tc.cat([F.apply_rotary(k.slice(3, 0, R), cseg, sseg),
                    k.slice(3, R, D - R)], dim=3)

        if kv_cache is not None:
            k = tc.cat([kv_cache[0], k], dim=2)
            v = tc.cat([kv_cache[1], v], dim=2)
        new_kv = (k, v)

        if self.attention_mode == "apa_selective":
            from tensor_cuda.quant import _norm_ppf, _quantize_keys, _tables
            from core.mistral7b_tc import _cublas_blend_attention
            dev = q.device.split(":")[0]
            R_t, C_t, B_t = _tables(D, self.bulk_bits, KV, True, dev)
            kq = _quantize_keys(k, R_t, C_t, B_t)        # KV-head granularity
            z_ = _norm_ppf(1.0 - max(0.0, min(1.0, self.refine_percentile)))
            S_all = k.shape[2]
            blk = max(64, min(self.attn_block,
                              int(300 * 1024 * 1024 // (H * S_all * 2))))
            attn = _cublas_blend_attention(
                q, _repeat_kv(k, H // KV), kq, v, H // KV, D ** -0.5,
                float(z_), L > 1, blk)
        else:
            attn = F.scaled_dot_product_attention(
                q, _repeat_kv(k, H // KV), _repeat_kv(v, H // KV),
                is_causal=(L > 1))
        attn = attn.transpose(1, 2).reshape([B, L, H * D])
        attn = attn * gate.sigmoid()                     # elementwise out-gate
        return self.o_proj(_cast(attn)), new_kv


class SwiGLUTC:
    def __init__(self):
        self.gate_proj = self.up_proj = self.down_proj = None

    def __call__(self, x):
        return self.down_proj(self.gate_proj(x).silu() * self.up_proj(x))


class Qwen35BlockTC:
    def __init__(self, cfg, layer_idx):
        self.is_attn = Qwen35Config.is_attention(layer_idx)
        self.input_layernorm = RMSNormTC(cfg.hidden_dim, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNormTC(cfg.hidden_dim,
                                                  cfg.rms_norm_eps)
        self.mixer = (Qwen35AttentionTC(cfg) if self.is_attn
                      else GatedDeltaNetTC(cfg))
        self.mlp = SwiGLUTC()

    def __call__(self, x, cos, sin, position_offset=0, cache=None):
        h_in = _cast(self.input_layernorm(x))
        if self.is_attn:
            a, new_cache = self.mixer(h_in, cos, sin, position_offset, cache)
        else:
            a, new_cache = self.mixer(h_in, cache)
        h = x + a
        mo = self.mlp(_cast(self.post_attention_layernorm(h)))
        return h + mo, new_cache


class Qwen35_TC:
    def __init__(self, cfg=None):
        self.config = cfg or Qwen35Config()
        self.embed_tokens = HostEmbedding()
        self.layers = [Qwen35BlockTC(self.config, i)
                       for i in range(self.config.num_layers)]
        self.norm = RMSNormTC(self.config.hidden_dim, self.config.rms_norm_eps)
        self.lm_head = None
        self._rope_len = 0
        self.extend_rope(4096)

    def extend_rope(self, seq_len):
        if seq_len <= self._rope_len:
            return
        cfg = self.config
        d = cfg.partial_rotary_dim
        inv = 1.0 / (cfg.rope_theta ** (np.arange(0, d, 2, np.float32) / d))
        pos = np.arange(seq_len, dtype=np.float32)[:, None] * inv[None, :]
        emb = np.concatenate([pos, pos], axis=-1)
        self.rope_cos = _cast(tc.tensor(np.cos(emb).astype(np.float32)))
        self.rope_sin = _cast(tc.tensor(np.sin(emb).astype(np.float32)))
        self._rope_len = seq_len

    def __call__(self, input_ids_np, caches=None, position_offset=0,
                 last_token_only=False, max_layers=None):
        B, L = input_ids_np.shape
        self.extend_rope(position_offset + L)
        h = self.embed_tokens(input_ids_np)
        new_caches = []
        run = self.layers if max_layers is None else self.layers[:max_layers]
        for i, layer in enumerate(run):
            cache = caches[i] if caches is not None else None
            h, c = layer(h, self.rope_cos, self.rope_sin, position_offset,
                         cache)
            new_caches.append(c)
        if max_layers is not None:
            return None, new_caches, h
        h = _cast(self.norm(h))
        if last_token_only and h.shape[1] > 1:
            h = h.slice(1, h.shape[1] - 1, 1)
        return self.lm_head(h), new_caches

    # ---- weights ------------------------------------------------------
    def load_weights(self, model_dir=None, progress=True):
        from safetensors import safe_open
        d = model_dir or _snap()
        cfg = self.config
        shards = sorted(glob.glob(os.path.join(d, "*.safetensors")))
        # one pass: name -> shard map (layer tensors are spread across shards)
        where = {}
        for s in shards:
            with safe_open(s, framework="pt") as f:
                for name in f.keys():
                    where[name] = s

        def gf(name):
            """fetch as fp32 numpy regardless of stored dtype (bf16)."""
            import torch
            with safe_open(where[name], framework="pt") as f:
                return f.get_tensor(name).to(torch.float32).numpy()

        P = "model.language_model"
        emb = gf(f"{P}.embed_tokens.weight")
        self.embed_tokens.weight = np.ascontiguousarray(emb)
        del emb
        gc.collect()

        for i in range(cfg.num_layers):
            p = f"{P}.layers.{i}"
            Lr = self.layers[i]
            # (1+w) bake for the standard norms — Qwen3.5 stores deltas
            Lr.input_layernorm.weight = tc.tensor(np.ascontiguousarray(
                1.0 + gf(f"{p}.input_layernorm.weight")), dtype="float32")
            Lr.post_attention_layernorm.weight = tc.tensor(
                np.ascontiguousarray(
                    1.0 + gf(f"{p}.post_attention_layernorm.weight")),
                dtype="float32")
            mx = Lr.mixer
            if Lr.is_attn:
                a = f"{p}.self_attn"
                mx.q_proj = QuantLinearTC(np.ascontiguousarray(
                    gf(f"{a}.q_proj.weight")))
                mx.k_proj = QuantLinearTC(np.ascontiguousarray(
                    gf(f"{a}.k_proj.weight")))
                mx.v_proj = QuantLinearTC(np.ascontiguousarray(
                    gf(f"{a}.v_proj.weight")))
                mx.o_proj = QuantLinearTC(np.ascontiguousarray(
                    gf(f"{a}.o_proj.weight")))
                mx.q_norm_w = tc.tensor(np.ascontiguousarray(
                    1.0 + gf(f"{a}.q_norm.weight")), dtype="float32")
                mx.k_norm_w = tc.tensor(np.ascontiguousarray(
                    1.0 + gf(f"{a}.k_norm.weight")), dtype="float32")
            else:
                la = f"{p}.linear_attn"
                mx.in_proj_qkv = QuantLinearTC(np.ascontiguousarray(
                    gf(f"{la}.in_proj_qkv.weight")))
                mx.in_proj_z = QuantLinearTC(np.ascontiguousarray(
                    gf(f"{la}.in_proj_z.weight")))
                mx.in_proj_a = F32Linear(gf(f"{la}.in_proj_a.weight"))
                mx.in_proj_b = F32Linear(gf(f"{la}.in_proj_b.weight"))
                mx.out_proj = QuantLinearTC(np.ascontiguousarray(
                    gf(f"{la}.out_proj.weight")))
                cw = gf(f"{la}.conv1d.weight").reshape(-1, cfg.conv_kernel)
                mx.conv_w = [tc.tensor(np.ascontiguousarray(
                    cw[:, j].reshape(1, 1, -1))) for j in range(4)]
                mx.neg_A = tc.tensor(np.ascontiguousarray(
                    (-np.exp(gf(f"{la}.A_log"))).reshape(1, 1, -1)))
                mx.dt_bias = tc.tensor(np.ascontiguousarray(
                    gf(f"{la}.dt_bias").reshape(1, 1, -1)))
                mx.norm_w = tc.tensor(np.ascontiguousarray(
                    gf(f"{la}.norm.weight")), dtype="float32")   # plain w
            mx_g = f"{p}.mlp"
            Lr.mlp.gate_proj = QuantLinearTC(np.ascontiguousarray(
                gf(f"{mx_g}.gate_proj.weight")))
            Lr.mlp.up_proj = QuantLinearTC(np.ascontiguousarray(
                gf(f"{mx_g}.up_proj.weight")))
            Lr.mlp.down_proj = QuantLinearTC(np.ascontiguousarray(
                gf(f"{mx_g}.down_proj.weight")))
            gc.collect()
            if progress and i % 4 == 3:
                print(f"    layer {i + 1}/{cfg.num_layers}", flush=True)

        self.norm.weight = tc.tensor(np.ascontiguousarray(
            1.0 + gf(f"{P}.norm.weight")), dtype="float32")
        self.lm_head = QuantLinearTC(np.ascontiguousarray(
            gf("lm_head.weight")))
        gc.collect()
        return {"loaded": "INT4 hybrid", "framework":
                "tensor_cuda Qwen3.5-9B (24 GDN + 8 attn)"}

    @classmethod
    def from_pretrained(cls, model_dir=None):
        BlockTC.COMPUTE_DTYPE = "bfloat16"
        LinearTC.DTYPE = "bfloat16"
        with tc.no_grad():
            m = cls()
            info = m.load_weights(model_dir)
        return m, info


# ---- GRM surface: lossless hybrid-state save/restore -----------------
# Layout pinned per the ledger (trap #8): DeltaNet recurrent state in HF
# order (B, heads, d_k, d_v) fp32; conv tail (B, 3, 8192) fp32 raw rows;
# attention KV (B, kv_heads, S, head_dim) in compute dtype (round-tripped
# through fp32 in the npz — bf16 -> fp32 is exact, restore re-casts).

def save_caches(caches, path, position_offset):
    arrs = {"position_offset": np.array([position_offset], np.int64)}
    for i, c in enumerate(caches):
        a, b = c
        arrs[f"l{i}_a"] = a.float().numpy()
        arrs[f"l{i}_b"] = b.float().numpy()
    tmp = path + ".tmp"
    np.savez(tmp, **arrs)                 # numpy appends .npz to the name
    os.replace(tmp + ".npz", path)


def load_caches(path, cfg=None):
    cfg = cfg or Qwen35Config()
    z = np.load(path)
    caches = []
    for i in range(cfg.num_layers):
        a = z[f"l{i}_a"]
        b = z[f"l{i}_b"]
        if Qwen35Config.is_attention(i):
            caches.append((_cast(tc.tensor(a)), _cast(tc.tensor(b))))
        else:
            caches.append((tc.tensor(a), tc.tensor(b)))     # fp32 exact
    return caches, int(z["position_offset"][0])
