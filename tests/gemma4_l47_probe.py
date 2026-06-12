"""Host-side fp64 probe of layer 47 (worst global layer).

Feeds GT's OWN L46 output (gt hidden[47]) through this port's L47 graph
implemented in numpy fp64 with exact bf16-checkpoint weights (no INT4,
no engine). final_norm(out) must reproduce gt hidden[48] (cos ~1) if
the GRAPH DEFINITION matches HF; otherwise the definition is wrong and
the residual error localizes to an op.

  python3 tests/gemma4_l47_probe.py
"""
import numpy as np
import torch
from safetensors import safe_open

SNAP = "/mnt/ForgeRealm/models/gemma-4-12B-it/model.safetensors"
gt = np.load("/mnt/ForgeRealm/gemma4_gt/gt_0.npz")
x = gt["hidden"][47].astype(np.float64)        # (S, 3840) input to L47
target = gt["hidden"][48].astype(np.float64)   # post-final-norm
S = x.shape[0]
LI = 47
P = f"model.language_model.layers.{LI}"


def gf(f, name):
    return f.get_tensor(name).to(torch.float64).numpy()


def cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def rmsnorm(v, w=None, eps=1e-6):
    ms = (v * v).mean(-1, keepdims=True)
    o = v * (ms + eps) ** -0.5
    return o * w if w is not None else o


with safe_open(SNAP, framework="pt") as f:
    w = {k.split(f"{P}.")[1]: gf(f, k) for k in f.keys() if f"{P}." in k}
    fn = gf(f, "model.language_model.norm.weight")

print("L47 tensors:", sorted(w.keys()))
print("layer_scalar:", w["layer_scalar"])

H, D = 16, 512
h_in = rmsnorm(x, w["input_layernorm.weight"])           # input_ln

q = (h_in @ w["self_attn.q_proj.weight"].T).reshape(S, H, D)
kraw = (h_in @ w["self_attn.k_proj.weight"].T).reshape(S, 1, D)
q = rmsnorm(q, w["self_attn.q_norm.weight"])
k = rmsnorm(kraw, w["self_attn.k_norm.weight"])
v = rmsnorm(kraw)                                        # scale-free v_norm

# p-RoPE over 512: 64 real freqs (full-dim parameterization), 192 zero
inv = 1.0 / (1e6 ** (np.arange(0, 128, 2, dtype=np.float64) / 512.0))
inv = np.concatenate([inv, np.zeros(192)])
ang = np.arange(S, dtype=np.float64)[:, None] * inv[None, :]
emb = np.concatenate([ang, ang], -1)                      # (S, 512)
cs, sn = np.cos(emb), np.sin(emb)


def rope(t):                                              # (S, h, 512)
    t1, t2 = t[..., :256], t[..., 256:]
    rot = np.concatenate([-t2, t1], -1)
    return t * cs[:, None, :] + rot * sn[:, None, :]


q, k = rope(q), rope(k)

scores = np.einsum("shd,thd->hst", q, np.repeat(k, H, 1))  # scale 1.0
mask = np.triu(np.full((S, S), -np.inf), k=1)
scores = scores + mask[None]
p = np.exp(scores - scores.max(-1, keepdims=True))
p /= p.sum(-1, keepdims=True)
attn = np.einsum("hst,thd->shd", p, np.repeat(v, H, 1)).reshape(S, H * D)
a_out = attn @ w["self_attn.o_proj.weight"].T

h = x + rmsnorm(a_out, w["post_attention_layernorm.weight"])
ff_in = rmsnorm(h, w["pre_feedforward_layernorm.weight"])
gate = ff_in @ w["mlp.gate_proj.weight"].T
up = ff_in @ w["mlp.up_proj.weight"].T
gelu = 0.5 * gate * (1 + np.tanh(0.7978845608028654 * (gate + 0.044715 * gate ** 3)))
mo = (gelu * up) @ w["mlp.down_proj.weight"].T
h = h + rmsnorm(mo, w["post_feedforward_layernorm.weight"])
h = h * w["layer_scalar"]

out = rmsnorm(h, fn)                                      # final norm
print(f"cos(final_norm(myL47(gt_in)), gt_hidden[48]): "
      f"last {cos(out[-1], target[-1]):.6f} | "
      f"mean {np.mean([cos(out[t], target[t]) for t in range(S)]):.6f}")
print("DONE")
