"""INT4-noise isolation: full 48-layer host simulation in fp32 with
quantize->dequantize (engine group-128 INT4) weights. No engine, no
bf16 — if this alone reproduces the parity collapse (late-layer cosine
~0.3, flip costs ~8-9), the divergence is INT4 quantization noise on
Gemma 4's weight distribution, full stop. (Context: Google ships an
official QAT release for this model — post-hoc INT4 being rough is
exactly why QAT releases exist.)

Streams weights layer-by-layer (peak RAM ~1GB + embeddings at the end).

  python3 tests/gemma4_int4_sim.py [--exact]   (--exact skips quant)
"""
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.mistral7b_tc import _quantize_int4, GROUP_SIZE   # noqa: E402

SNAP = "/mnt/ForgeRealm/models/gemma-4-12B-it/model.safetensors"
EXACT = "--exact" in sys.argv
EPS = 1e-6
H = 16

gt = np.load("/mnt/ForgeRealm/gemma4_gt/gt_0.npz")
ids = gt["ids"][0]
S = len(ids)


def qd(w):
    """quantize -> dequantize through the engine's own host quantizer.
    Layout per _quantize_int4: packed (N, K/2) even|odd<<4, w = q*s + z
    (zeros are the group MINS, fp16)."""
    if EXACT:
        return w.astype(np.float32)
    packed, scales, zeros = _quantize_int4(w.astype(np.float32), GROUP_SIZE)
    N, K = w.shape
    q = np.zeros((N, K), np.float32)
    q[:, 0::2] = (packed & 0x0F).astype(np.float32)
    q[:, 1::2] = (packed >> 4).astype(np.float32)
    G = K // GROUP_SIZE
    deq = (q.reshape(N, G, GROUP_SIZE)
           * scales.astype(np.float32)[:, :, None]
           + zeros.astype(np.float32)[:, :, None])
    return deq.reshape(N, K)


def rmsnorm(v, w=None, eps=EPS):
    ms = (v * v).mean(-1, keepdims=True)
    o = v * (ms + eps) ** -0.5
    return o * w if w is not None else o


def cos(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def rope_tables(D, theta, n_real):
    inv = 1.0 / (theta ** (np.arange(0, 2 * n_real, 2, dtype=np.float64) / D))
    inv = np.concatenate([inv, np.zeros(D // 2 - n_real)])
    ang = np.arange(S, dtype=np.float64)[:, None] * inv[None, :]
    emb = np.concatenate([ang, ang], -1)
    return np.cos(emb), np.sin(emb)


CS_L, SN_L = rope_tables(256, 1e4, 128)      # sliding: full rotary
CS_G, SN_G = rope_tables(512, 1e6, 64)       # global: p-RoPE


def rope(t, cs, sn):
    t1, t2 = t[..., :t.shape[-1] // 2], t[..., t.shape[-1] // 2:]
    return t * cs[:, None, :] + np.concatenate([-t2, t1], -1) * sn[:, None, :]


P = "model.language_model"
mask = np.triu(np.full((S, S), -np.inf), k=1)

with safe_open(SNAP, framework="pt") as f:
    def gf(name):
        return f.get_tensor(name).to(torch.float32).numpy()

    emb_w = gf(f"{P}.embed_tokens.weight")
    h = (emb_w[ids] * np.float32(3840.0 ** 0.5)).astype(np.float64)

    for li in range(48):
        p = f"{P}.layers.{li}"
        is_g = li % 6 == 5
        D = 512 if is_g else 256
        KV = 1 if is_g else 8
        cs, sn = (CS_G, SN_G) if is_g else (CS_L, SN_L)

        h_ln = rmsnorm(h, gf(f"{p}.input_layernorm.weight"))
        q = (h_ln @ qd(gf(f"{p}.self_attn.q_proj.weight")).T.astype(
            np.float64)).reshape(S, H, D)
        kraw = (h_ln @ qd(gf(f"{p}.self_attn.k_proj.weight")).T.astype(
            np.float64)).reshape(S, KV, D)
        q = rmsnorm(q, gf(f"{p}.self_attn.q_norm.weight"))
        k = rmsnorm(kraw, gf(f"{p}.self_attn.k_norm.weight"))
        if is_g:
            v = rmsnorm(kraw)
        else:
            v = rmsnorm((h_ln @ qd(gf(f"{p}.self_attn.v_proj.weight")).T
                         .astype(np.float64)).reshape(S, KV, D))
        q, k = rope(q, cs, sn), rope(k, cs, sn)
        rep = H // KV
        sc = np.einsum("shd,thd->hst", q, np.repeat(k, rep, 1)) + mask[None]
        pr = np.exp(sc - sc.max(-1, keepdims=True))
        pr /= pr.sum(-1, keepdims=True)
        a = np.einsum("hst,thd->shd", pr,
                      np.repeat(v, rep, 1)).reshape(S, H * D)
        a = a @ qd(gf(f"{p}.self_attn.o_proj.weight")).T.astype(np.float64)
        h = h + rmsnorm(a, gf(f"{p}.post_attention_layernorm.weight"))

        ff = rmsnorm(h, gf(f"{p}.pre_feedforward_layernorm.weight"))
        gate = ff @ qd(gf(f"{p}.mlp.gate_proj.weight")).T.astype(np.float64)
        up = ff @ qd(gf(f"{p}.mlp.up_proj.weight")).T.astype(np.float64)
        gelu = 0.5 * gate * (1 + np.tanh(
            0.7978845608028654 * (gate + 0.044715 * gate ** 3)))
        mo = (gelu * up) @ qd(gf(f"{p}.mlp.down_proj.weight")).T.astype(
            np.float64)
        h = h + rmsnorm(mo, gf(f"{p}.post_feedforward_layernorm.weight"))
        h = h * gf(f"{p}.layer_scalar").astype(np.float64)

        c = cos(h[-1], gt["hidden"][li + 1][-1])
        tag = " G" if is_g else ""
        print(f"L{li:2d}{tag}: cos {c:.4f}", flush=True)

    h = rmsnorm(h, gf(f"{P}.norm.weight"))
    print(f"final-norm: cos {cos(h[-1], gt['hidden'][48][-1]):.4f}",
          flush=True)
    lg = 30.0 * np.tanh((h @ emb_w.T.astype(np.float64)) / 30.0)

ref = gt["logits"]
t_s, t_r = lg.argmax(-1), ref.argmax(-1)
flips = np.nonzero(t_s != t_r)[0]
costs = [float(ref[t].max() - ref[t][t_s[t]]) for t in flips]
print(f"top1 {int((t_s == t_r).sum())}/{S} | worst flip "
      f"{max(costs, default=0.0):.3f} | last|d| "
      f"{float(np.abs(lg[-1] - ref[-1]).max()):.3f}", flush=True)
print("DONE", flush=True)
