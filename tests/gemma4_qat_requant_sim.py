"""Does post-hoc group-128 requantization damage the QAT weights?

Host fp32 simulation, two trajectories through the same 48 layers:
  REF — QAT q4_0 weights dequantized exactly (the QAT grid itself)
  RQ  — the same weights re-quantized through the engine's group-128
        min/max INT4 (what load-from-dequant would produce)
If RQ tracks REF (>=0.97 to L47, near-tie flips only), the engine can
serve QAT weights through the existing g128 path with no kernel work.
If it collapses like the bf16-origin trail did, the exact group-32
loader is required.

Also verifies GGUF norm-weight convention against safetensors stats
(llama.cpp historically stores Gemma norms shifted by +1).

  python3 tests/gemma4_qat_requant_sim.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.mistral7b_tc import _quantize_int4, GROUP_SIZE   # noqa: E402
from gguf import GGUFReader                                # noqa: E402
from gguf.quants import dequantize                         # noqa: E402

GGUF = ("/mnt/ForgeRealm/models/gemma-4-12B-it-qat/"
        "gemma-4-12b-it-qat-q4_0.gguf")
EPS = 1e-6
H = 16

gt = np.load("/mnt/ForgeRealm/gemma4_gt/gt_0.npz")
ids = gt["ids"][0]
S = len(ids)

r = GGUFReader(GGUF)
T = {t.name: t for t in r.tensors}


def gw(name):
    t = T[name]
    return dequantize(t.data, t.tensor_type).astype(np.float32)


# ---- norm convention check vs safetensors (different training, but a
# +1 shift would show in the mean)
import torch                                                # noqa: E402
from safetensors import safe_open                          # noqa: E402
with safe_open("/mnt/ForgeRealm/models/gemma-4-12B-it/model.safetensors",
               framework="pt") as f:
    st_norm = f.get_tensor(
        "model.language_model.layers.5.input_layernorm.weight").to(
        torch.float32).numpy()
gg_norm = gw("blk.5.attn_norm.weight")
print(f"norm means: gguf {gg_norm.mean():.4f} vs safetensors "
      f"{st_norm.mean():.4f} (a ~+1 gap = shifted convention)", flush=True)


def rq(w):
    packed, scales, zeros = _quantize_int4(w, GROUP_SIZE)
    N, K = w.shape
    q = np.zeros((N, K), np.float32)
    q[:, 0::2] = (packed & 0x0F).astype(np.float32)
    q[:, 1::2] = (packed >> 4).astype(np.float32)
    G = K // GROUP_SIZE
    return (q.reshape(N, G, GROUP_SIZE)
            * scales.astype(np.float32)[:, :, None]
            + zeros.astype(np.float32)[:, :, None]).reshape(N, K)


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


CS_L, SN_L = rope_tables(256, 1e4, 128)
CS_G, SN_G = rope_tables(512, 1e6, 64)


def rope(t, cs, sn):
    t1, t2 = t[..., :t.shape[-1] // 2], t[..., t.shape[-1] // 2:]
    return t * cs[:, None, :] + np.concatenate([-t2, t1], -1) * sn[:, None, :]


mask = np.triu(np.full((S, S), -np.inf), k=1)

print("dequantizing token_embd (Q6_K)...", flush=True)
emb_w = gw("token_embd.weight")
print("embed shape:", emb_w.shape, flush=True)
h_ref = (emb_w[ids] * np.float32(3840.0 ** 0.5)).astype(np.float64)
h_rq = h_ref.copy()


def block(h, p, is_g, W):
    D = 512 if is_g else 256
    KV = 1 if is_g else 8
    cs, sn = (CS_G, SN_G) if is_g else (CS_L, SN_L)
    h_ln = rmsnorm(h, W["attn_norm"])
    q = (h_ln @ W["attn_q"].T.astype(np.float64)).reshape(S, H, D)
    kraw = (h_ln @ W["attn_k"].T.astype(np.float64)).reshape(S, KV, D)
    q = rmsnorm(q, W["attn_q_norm"])
    k = rmsnorm(kraw, W["attn_k_norm"])
    if is_g:
        v = rmsnorm(kraw)
    else:
        v = rmsnorm((h_ln @ W["attn_v"].T.astype(np.float64))
                    .reshape(S, KV, D))
    q, k = rope(q, cs, sn), rope(k, cs, sn)
    rep = H // KV
    sc = np.einsum("shd,thd->hst", q, np.repeat(k, rep, 1)) + mask[None]
    pr = np.exp(sc - sc.max(-1, keepdims=True))
    pr /= pr.sum(-1, keepdims=True)
    a = np.einsum("hst,thd->shd", pr, np.repeat(v, rep, 1)).reshape(S, H * D)
    a = a @ W["attn_output"].T.astype(np.float64)
    h = h + rmsnorm(a, W["post_attention_norm"])
    ff = rmsnorm(h, W["ffn_norm"])
    gate = ff @ W["ffn_gate"].T.astype(np.float64)
    up = ff @ W["ffn_up"].T.astype(np.float64)
    gelu = 0.5 * gate * (1 + np.tanh(
        0.7978845608028654 * (gate + 0.044715 * gate ** 3)))
    mo = (gelu * up) @ W["ffn_down"].T.astype(np.float64)
    h = h + rmsnorm(mo, W["post_ffw_norm"])
    return h * W["layer_output_scale"].astype(np.float64)


LINEARS = ("attn_q", "attn_k", "attn_v", "attn_output",
           "ffn_gate", "ffn_up", "ffn_down")
for li in range(48):
    is_g = li % 6 == 5
    W_ref, W_rq = {}, {}
    for nm in ("attn_norm", "post_attention_norm", "ffn_norm",
               "post_ffw_norm", "attn_q_norm", "attn_k_norm",
               "layer_output_scale"):
        W_ref[nm] = W_rq[nm] = gw(f"blk.{li}.{nm}.weight")
    for nm in LINEARS:
        if is_g and nm == "attn_v":
            continue
        w = gw(f"blk.{li}.{nm}.weight")
        W_ref[nm] = w
        W_rq[nm] = rq(w)
    h_ref = block(h_ref, li, is_g, W_ref)
    h_rq = block(h_rq, li, is_g, W_rq)
    tag = " G" if is_g else ""
    print(f"L{li:2d}{tag}: cos(rq, ref) {cos(h_rq[-1], h_ref[-1]):.4f}",
          flush=True)

fn = gw("output_norm.weight")
o_ref = rmsnorm(h_ref, fn)
o_rq = rmsnorm(h_rq, fn)
lg_ref = 30.0 * np.tanh((o_ref @ emb_w.T.astype(np.float64)) / 30.0)
lg_rq = 30.0 * np.tanh((o_rq @ emb_w.T.astype(np.float64)) / 30.0)
t_r, t_q = lg_ref.argmax(-1), lg_rq.argmax(-1)
flips = np.nonzero(t_r != t_q)[0]
costs = [float(lg_ref[t].max() - lg_ref[t][t_q[t]]) for t in flips]
print(f"final: cos {cos(o_rq[-1], o_ref[-1]):.4f} | top1 "
      f"{int((t_r == t_q).sum())}/{S} | worst flip "
      f"{max(costs, default=0.0):.3f} | last|d| "
      f"{float(np.abs(lg_rq[-1] - lg_ref[-1]).max()):.3f}", flush=True)
print("DONE", flush=True)
