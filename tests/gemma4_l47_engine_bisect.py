"""Engine-side bisect of layer 47's attention: the graph definition is
proven exact (fp64 host probe, cos 1.000000), so the parity divergence
is in engine execution. Builds layer 47's projections standalone on the
engine (INT4), runs the attention stage-by-stage on GT's L46 output,
and compares every intermediate against an fp64 numpy reference run
with the SAME INT4-DEQUANTIZED weights (op bugs and bf16 effects show;
quantization noise is excluded by construction). A separate line
reports quantization noise alone (deq-ref vs exact-ref).

  python3 tests/gemma4_l47_engine_bisect.py
"""
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from tensor_cuda import functional as F                    # noqa: E402
from core.mistral7b_tc import (BlockTC, LinearTC,           # noqa: E402
                               QuantLinearTC, RMSNormTC, _repeat_kv)
from core.gemma4_tc import _cast, _head_rmsnorm            # noqa: E402

SNAP = "/mnt/ForgeRealm/models/gemma-4-12B-it/model.safetensors"
LI, H, D = 47, 16, 512
EPS = 1e-6
P = f"model.language_model.layers.{LI}"

BlockTC.COMPUTE_DTYPE = "bfloat16"
LinearTC.DTYPE = "bfloat16"
QuantLinearTC.FUSED_DECODE = True
RMSNormTC.USE_FUSED = True

gt = np.load("/mnt/ForgeRealm/gemma4_gt/gt_0.npz")
x64 = gt["hidden"][47].astype(np.float64)
S = x64.shape[0]


def cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def rmsnorm(v, w=None, eps=EPS):
    ms = (v * v).mean(-1, keepdims=True)
    o = v * (ms + eps) ** -0.5
    return o * w if w is not None else o


with safe_open(SNAP, framework="pt") as f:
    w = {k.split(f"{P}.")[1]:
         f.get_tensor(k).to(torch.float32).numpy()
         for k in f.keys() if f"{P}." in k}

tc.set_alloc_pooling(True)
with tc.no_grad():
    qp = QuantLinearTC(np.ascontiguousarray(w["self_attn.q_proj.weight"]))
    kp = QuantLinearTC(np.ascontiguousarray(w["self_attn.k_proj.weight"]))

    # int4_dequant returns the packed layout W^T (in, out) — transpose
    # back to (out, in) for the reference math
    wq_d = tc.int4_dequant(qp.packed, qp.scales, qp.zeros,
                           out_dtype="float16").float().numpy().T
    wk_d = tc.int4_dequant(kp.packed, kp.scales, kp.zeros,
                           out_dtype="float16").float().numpy().T
    print(f"deq shapes (as out,in): {wq_d.shape} {wk_d.shape}", flush=True)

    # ---- fp64 reference intermediates, parameterized by weights
    def ref_chain(wq, wk):
        r = {}
        h_ln = rmsnorm(x64, w["input_layernorm.weight"].astype(np.float64))
        r["h_ln"] = h_ln
        r["q_lin"] = h_ln @ wq.T.astype(np.float64)
        r["k_lin"] = h_ln @ wk.T.astype(np.float64)
        q = rmsnorm(r["q_lin"].reshape(S, H, D),
                    w["self_attn.q_norm.weight"].astype(np.float64))
        k = rmsnorm(r["k_lin"].reshape(S, 1, D),
                    w["self_attn.k_norm.weight"].astype(np.float64))
        v = rmsnorm(r["k_lin"].reshape(S, 1, D))
        r["q_n"], r["k_n"], r["v_n"] = q, k, v
        inv = 1.0 / (1e6 ** (np.arange(0, 128, dtype=np.float64)[::2] / 512.0))
        inv = np.concatenate([inv, np.zeros(192)])
        ang = np.arange(S, dtype=np.float64)[:, None] * inv[None, :]
        emb = np.concatenate([ang, ang], -1)
        cs, sn = np.cos(emb), np.sin(emb)

        def rope(t):
            t1, t2 = t[..., :256], t[..., 256:]
            return (t * cs[:, None, :]
                    + np.concatenate([-t2, t1], -1) * sn[:, None, :])
        r["q_r"], r["k_r"] = rope(q), rope(k)
        sc = np.einsum("shd,thd->hst", r["q_r"], np.repeat(r["k_r"], H, 1))
        r["scores"] = sc.copy()
        sc = sc + np.triu(np.full((S, S), -np.inf), k=1)[None]
        p = np.exp(sc - sc.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        r["probs"] = p
        r["attn"] = np.einsum("hst,thd->shd", p, np.repeat(v, H, 1))
        return r

    R_x = ref_chain(w["self_attn.q_proj.weight"], w["self_attn.k_proj.weight"])
    R_d = ref_chain(wq_d, wk_d)
    print("INT4 noise alone (deq-ref vs exact-ref): "
          f"q_lin {cos(R_d['q_lin'], R_x['q_lin']):.5f} "
          f"k_lin {cos(R_d['k_lin'], R_x['k_lin']):.5f} "
          f"scores {cos(R_d['scores'], R_x['scores']):.5f} "
          f"attn {cos(R_d['attn'], R_x['attn']):.5f}", flush=True)

    # ---- engine chain, stage by stage (vs deq reference R_d)
    x_e = _cast(tc.tensor(x64[None].astype(np.float32)))
    ln_w = tc.tensor(np.ascontiguousarray(w["input_layernorm.weight"]),
                     dtype="float32")
    h_e = tc.rms_norm(x_e, ln_w, EPS)
    print(f"[0] input_ln       : "
          f"{cos(h_e.float().numpy()[0], R_d['h_ln']):.5f}", flush=True)
    h_e = _cast(h_e)

    qe_l = qp(h_e)
    ke_l = kp(h_e)
    print(f"[1] q_proj GEMV    : "
          f"{cos(qe_l.float().numpy()[0], R_d['q_lin']):.5f} | k_proj "
          f"{cos(ke_l.float().numpy()[0], R_d['k_lin']):.5f}", flush=True)

    qn_w = tc.tensor(np.ascontiguousarray(w["self_attn.q_norm.weight"]),
                     dtype="float32")
    kn_w = tc.tensor(np.ascontiguousarray(w["self_attn.k_norm.weight"]),
                     dtype="float32")
    qe_n = _head_rmsnorm(qe_l, qn_w, EPS, 1, S, H, D)
    ke_n = _head_rmsnorm(ke_l, kn_w, EPS, 1, S, 1, D)
    ve_n = _head_rmsnorm(ke_l, None, EPS, 1, S, 1, D)
    print(f"[2] q_norm         : "
          f"{cos(qe_n.float().numpy()[0], R_d['q_n'].reshape(S, -1)):.5f} | "
          f"k_norm {cos(ke_n.float().numpy()[0], R_d['k_n'].reshape(S, -1)):.5f} | "
          f"v_norm {cos(ve_n.float().numpy()[0], R_d['v_n'].reshape(S, -1)):.5f}",
          flush=True)

    # rope tables exactly as the port builds them
    posn = np.arange(S, dtype=np.float32)[:, None]
    inv_g = 1.0 / (1e6 ** (np.arange(0, 128, 2, np.float32) / np.float32(D)))
    inv_g = np.concatenate([inv_g, np.zeros(D // 2 - 64, np.float32)])
    emb_g = np.concatenate([posn * inv_g, posn * inv_g], axis=-1)
    cos_t = _cast(tc.tensor(np.cos(emb_g).astype(np.float32)))
    sin_t = _cast(tc.tensor(np.sin(emb_g).astype(np.float32)))

    qe_t = qe_n.reshape([1, S, H, D]).transpose(1, 2)
    ke_t = ke_n.reshape([1, S, 1, D]).transpose(1, 2)
    ve_t = ve_n.reshape([1, S, 1, D]).transpose(1, 2)
    qe_r = F.apply_rotary(qe_t, cos_t, sin_t)
    ke_r = F.apply_rotary(ke_t, cos_t, sin_t)
    print(f"[3] rope q         : "
          f"{cos(qe_r.float().numpy()[0].transpose(1, 0, 2), R_d['q_r']):.5f}"
          f" | rope k "
          f"{cos(ke_r.float().numpy()[0].transpose(1, 0, 2), R_d['k_r']):.5f}",
          flush=True)

    k_rep = _repeat_kv(ke_r, H)
    v_rep = _repeat_kv(ve_t, H)
    sc_e = tc.matmul(qe_r, k_rep, alpha=1.0, trans_b=True)
    sc_np = sc_e.float().numpy()[0]
    print(f"[4] scores         : {cos(sc_np, R_d['scores']):.5f} | "
          f"engine max|s| {float(np.abs(sc_np).max()):.1f} | "
          f"ref max|s| {float(np.abs(R_d['scores']).max()):.1f}", flush=True)

    attn_e = F.scaled_dot_product_attention(qe_r, k_rep, v_rep,
                                            is_causal=True, scale=1.0)
    print(f"[5] attn out       : "
          f"{cos(attn_e.float().numpy()[0].transpose(1, 0, 2), R_d['attn']):.5f}",
          flush=True)
print("DONE", flush=True)
