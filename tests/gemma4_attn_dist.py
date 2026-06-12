"""Attention-distribution study — the OTHER side of the equation
(Architect, 2026-06-12): the K/V program quantifies key storage; this
quantifies what softmax DOES with it on Gemma 4.

Per layer (sliding and global) on a ~1200-token natural prompt, at the
last 64 query positions: post-softmax entropy, and the fraction of
probability mass captured by the top 5% / 10% / 15% of keys — the
direct empirical test of APA's refine_percentile (refine covers the
top-|score| keys; mass coverage predicts how much the bulk-4bit
approximation can matter at r0.05 / r0.10 / r0.15).

  python3 tests/gemma4_attn_dist.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from tensor_cuda import functional as F                    # noqa: E402
from core.gemma4_tc import Gemma4_TC, _repeat_kv           # noqa: E402

gt = np.load("/mnt/ForgeRealm/gemma4_gt_qat/gt_long.npz")
ids = gt["ids"][:, :1200]

tc.set_alloc_pooling(True)
m, info = Gemma4_TC.from_pretrained()
print(f"loaded: {info}", flush=True)

WANT = {0, 5, 11, 17, 23, 29, 35, 41, 46, 47}
for i, L in enumerate(m.layers):
    L.mixer._li = i

with tc.no_grad():
    _, caches = m(ids, last_token_only=True)

# Decode-step score capture: patch attention to stash probabilities
# for the next token's query against the full 1200-token state
probs = {}
orig_call2 = type(m.layers[0].mixer).__call__


def prob_call(self, x, cos, sin, position_offset=0, kv_cache=None):
    cfg = self.cfg
    B, L, _ = x.shape
    if L != 1:
        return orig_call2(self, x, cos, sin, position_offset, kv_cache)
    out, new_kv = orig_call2(self, x, cos, sin, position_offset, kv_cache)
    li = getattr(self, "_li", None)
    if li in WANT and kv_cache is not None:
        H, KV = cfg.num_heads, self.kv_heads
        D = self.head_dim
        from core.gemma4_tc import _head_rmsnorm
        if self.qkv_proj is not None:
            qkv = self.qkv_proj(x)
            q_lin = qkv.slice(2, 0, H * D)
        else:
            q_lin = self.q_proj(x)
        q = _head_rmsnorm(q_lin, self.q_norm_w, cfg.rms_norm_eps, B, 1, H, D)
        q = q.reshape([B, 1, H, D]).transpose(1, 2)
        q = tc.rope_apply(q, cos, sin, position_offset) \
            if hasattr(tc, "rope_apply") else q
        k = new_kv[0]
        sc = tc.matmul(q.reshape([B, KV, H // KV, D]), k, alpha=1.0,
                       trans_b=True)
        s = sc.float().numpy()[0].reshape(H, -1)            # (H, S)
        e = np.exp(s - s.max(-1, keepdims=True))
        probs[li] = e / e.sum(-1, keepdims=True)
    return out, new_kv


type(m.layers[0].mixer).__call__ = prob_call
nxt = np.array([[5279]], np.int64)
with tc.no_grad():
    _, _ = m(nxt, caches=caches, position_offset=1200)
type(m.layers[0].mixer).__call__ = orig_call2

print(f"{'layer':>6s} {'S':>5s} {'entropy':>8s} {'eff_keys':>8s} "
      f"{'top5%':>7s} {'top10%':>7s} {'top15%':>7s}", flush=True)
for li in sorted(probs):
    p = probs[li]                                           # (H, S)
    S = p.shape[1]
    ent = float((-p * np.log(p + 1e-12)).sum(-1).mean())
    eff = float(np.exp(ent))
    srt = np.sort(p, -1)[:, ::-1]
    cum = srt.cumsum(-1)
    cov = [float(cum[:, max(1, int(S * f)) - 1].mean())
           for f in (0.05, 0.10, 0.15)]
    g = " G" if li % 6 == 5 else ""
    print(f"  L{li:2d}{g} {S:5d} {ent:8.3f} {eff:8.1f} "
          f"{cov[0]:7.3f} {cov[1]:7.3f} {cov[2]:7.3f}", flush=True)
print("DONE", flush=True)
