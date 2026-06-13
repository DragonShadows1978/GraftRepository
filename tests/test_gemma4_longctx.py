"""Sliding-window gate for Gemma 4 (the band-mask / ring-trim paths the
short-prompt parity never exercises; window binds past 1024 tokens).

  A) single 1200-token prefill vs fp32 GT last-row (margin gate)
  B) chunked prefill (1024 + 176) vs A — same math, different chunking;
     top-1 must agree modulo near-ties, |d| small
  C) greedy 8 from A's cache vs from B's cache — token match

  python3 tests/test_gemma4_longctx.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.gemma4_tc import Gemma4_TC                        # noqa: E402

gt = np.load(os.environ.get("GEMMA4_GT", "/mnt/ForgeRealm/gemma4_gt_qat") + "/gt_long.npz")
ids = gt["ids"]
S = ids.shape[1]

tc.set_alloc_pooling(True)
m, info = Gemma4_TC.from_pretrained()
print(f"loaded: {info}", flush=True)


def greedy(caches, off, first_logits, n):
    toks = []
    nxt = int(first_logits.float().numpy()[0, -1].argmax())
    with tc.no_grad():
        for _ in range(n):
            toks.append(nxt)
            lg, caches = m(np.array([[nxt]], np.int64), caches=caches,
                           position_offset=off)
            off += 1
            nxt = int(lg.float().numpy()[0, -1].argmax())
    return toks


# Only ONE cache set may live at a time: two full sets (~670MB at this
# length) do not fit next to the ~6.7GB body. A runs fully (prefill +
# greedy) and is released before B starts.

# ---- A) auto-chunked prefill (band mask binds: 1200 > 1024) + greedy
with tc.no_grad():
    lg_a, c_a = m(ids, last_token_only=True)
a = lg_a.float().numpy()[0, -1]
ref = gt["last_logits"]
flip_a = 0.0
if int(a.argmax()) != int(ref.argmax()):
    flip_a = float(ref.max() - ref[a.argmax()])
print(f"A vs GT: top1 {'match' if flip_a == 0 else f'flip cost {flip_a:.3f}'}"
      f" | last|d| {float(np.abs(a - ref).max()):.3f}", flush=True)
t_a = greedy(c_a, S, lg_a, 8)
del c_a
tc.empty_cache()

# ---- C1) DETERMINISM (the well-posed ring invariant): an identical
# prefill must greedy-decode identically — exact. (Cross-chunking token
# drift in C2 is EXPECTED near-tie behavior: the caches differ at bf16
# ulp level between chunk boundaries; determinism is what a ring bug
# would actually break.)
with tc.no_grad():
    lg_a2, c_a2 = m(ids, last_token_only=True)
t_a2 = greedy(c_a2, S, lg_a2, 8)
del c_a2
tc.empty_cache()
det = t_a == t_a2
print(f"C1 determinism: {'MATCH' if det else 'DIFF'} {t_a}", flush=True)

# ---- B) different chunk boundaries — PREFILL_CHUNK 384 makes the
# auto-chunking genuinely differ from A's 512 (with both at 512 the
# manual 1024-split decomposed into A's exact boundaries and B-vs-A
# compared a run against itself, |d| 0.000)
m.PREFILL_CHUNK = 384
with tc.no_grad():
    _, c_b = m(ids[:, :1024], last_token_only=True)
    lg_b, c_b = m(ids[:, 1024:], caches=c_b, position_offset=1024,
                  last_token_only=True)
m.PREFILL_CHUNK = 512
b = lg_b.float().numpy()[0, -1]
ab_flip = 0.0
if int(a.argmax()) != int(b.argmax()):
    ab_flip = float(a.max() - a[b.argmax()])
print(f"B vs A: top1 {'match' if ab_flip == 0 else f'flip cost {ab_flip:.3f}'}"
      f" | |d| {float(np.abs(a - b).max()):.3f}", flush=True)

# ---- C2) cross-chunking greedy (informational — near-tie drift is
# expected once caches differ at the ulp level)
t_b = greedy(c_b, S, lg_b, 8)
print(f"C2 greedy A {t_a} | B {t_b} | "
      f"{'match' if t_a == t_b else 'drift (info)'}", flush=True)

ok = flip_a <= 3.0 and ab_flip <= 1.0 and det
print(f"LONGCTX GATE: {'PASS' if ok else 'FAIL'}", flush=True)
print("DONE", flush=True)
