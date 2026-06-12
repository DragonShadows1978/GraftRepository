"""GRM gate: hybrid-state save/restore must be LOSSLESS — a restored
cache continues bit-identically to the uninterrupted run.

Two checks on the same load:
  A) post-prefill save: prefill -> save -> restore -> 8 greedy tokens
     must equal the continuous run's 8 tokens (and logits bit-equal at
     the first step).
  B) mid-decode save: prefill -> 4 greedy -> save -> restore -> 4 more
     must equal the continuous run's tokens 5..8.

  python3 tests/test_qwen35_state.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.qwen35_tc import (Qwen35_TC, save_caches,         # noqa: E402
                            load_caches)

STATE = "/tmp/qwen35_state_test.npz"
gt = np.load("/mnt/ForgeRealm/qwen35_gt/gt_2.npz")
ids = gt["ids"]

tc.set_alloc_pooling(True)
m, info = Qwen35_TC.from_pretrained()
print(f"loaded: {info}", flush=True)


def greedy(caches, off, first_logits, n):
    toks, lgs = [], []
    nxt = int(first_logits.float().numpy()[0, -1].argmax())
    with tc.no_grad():
        for _ in range(n):
            toks.append(nxt)
            lg, caches = m(np.array([[nxt]], np.int64), caches=caches,
                           position_offset=off)
            off += 1
            lgs.append(lg.float().numpy()[0, -1])
            nxt = int(lgs[-1].argmax())
    return toks, lgs, caches, off


# ---- continuous reference
with tc.no_grad():
    lg0, c0 = m(ids, last_token_only=True)
ref_toks, ref_lgs, c_mid, off_mid = greedy(c0, ids.shape[1], lg0, 8)
print(f"continuous: {ref_toks}", flush=True)

# ---- A) post-prefill save/restore
with tc.no_grad():
    lg1, c1 = m(ids, last_token_only=True)
save_caches(c1, STATE, ids.shape[1])
r_caches, r_off = load_caches(STATE)
a_toks, a_lgs, _, _ = greedy(r_caches, r_off, lg1, 8)
a_bit = all(np.array_equal(x, y) for x, y in zip(a_lgs, ref_lgs))
print(f"A post-prefill restore: toks {'MATCH' if a_toks == ref_toks else 'DIFF'}"
      f" | logits bit-identical: {a_bit}", flush=True)

# ---- B) mid-decode save/restore (4 + 4)
with tc.no_grad():
    lg2, c2 = m(ids, last_token_only=True)
b1_toks, b1_lgs, c_after4, off4 = greedy(c2, ids.shape[1], lg2, 4)
save_caches(c_after4, STATE, off4)
rb_caches, rb_off = load_caches(STATE)
# continue: next token comes from the last logits of the 4-step run
last_lg = tc.tensor(b1_lgs[-1][None, None].astype(np.float32))
b2_toks, b2_lgs, _, _ = greedy(rb_caches, rb_off, last_lg, 4)
b_ok = (b1_toks + b2_toks) == ref_toks
b_bit = all(np.array_equal(x, y) for x, y in zip(b2_lgs, ref_lgs[4:]))
print(f"B mid-decode restore: toks {'MATCH' if b_ok else 'DIFF'} | "
      f"logits bit-identical: {b_bit}", flush=True)

print(f"STATE GATE: {'PASS' if a_toks == ref_toks and a_bit and b_ok and b_bit else 'FAIL'}",
      flush=True)
print("DONE", flush=True)
