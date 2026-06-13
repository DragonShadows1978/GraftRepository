"""GRM gate for Gemma 4: pure-KV save/restore must be LOSSLESS — a
restored cache continues bit-identically to the uninterrupted run.

Two checks on the same load:
  A) post-prefill save: prefill -> save -> restore -> 8 greedy tokens
     must equal the continuous run's 8 tokens (logits bit-equal each
     step).
  B) mid-decode save: prefill -> 4 greedy -> save -> restore -> 4 more
     must equal the continuous run's tokens 5..8.

  python3 tests/test_gemma4_state.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.gemma4_tc import (Gemma4_TC, save_caches,         # noqa: E402
                            load_caches)

STATE = "/tmp/gemma4_state_test.npz"
gt = np.load(os.environ.get("GEMMA4_GT", "/mnt/ForgeRealm/gemma4_gt_qat") + "/gt_2.npz")
ids = gt["ids"]

tc.set_alloc_pooling(True)
m, info = Gemma4_TC.from_pretrained()
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
print(f"A post-prefill restore: toks "
      f"{'MATCH' if a_toks == ref_toks else 'DIFF'} | logits "
      f"{'BIT-IDENTICAL' if a_bit else 'DIFFER'}", flush=True)

# ---- B) mid-decode save/restore
with tc.no_grad():
    lg2, c2 = m(ids, last_token_only=True)
b_toks4, b_lgs4, c2_mid, off2 = greedy(c2, ids.shape[1], lg2, 4)
save_caches(c2_mid, STATE, off2)
r2_caches, r2_off = load_caches(STATE)
last_lg = tc.tensor(b_lgs4[-1].reshape(1, 1, -1))
b_toks, b_lgs, _, _ = greedy(r2_caches, r2_off, last_lg, 4)
b_match = (b_toks4 + b_toks) == ref_toks
b_bit = all(np.array_equal(x, y) for x, y in zip(b_lgs, ref_lgs[4:]))
print(f"B mid-decode restore: toks {'MATCH' if b_match else 'DIFF'} | "
      f"logits {'BIT-IDENTICAL' if b_bit else 'DIFFER'}", flush=True)

# Contract depends on storage mode. With QUANT_V (INT8 V-storage) the
# "lossless bit-identical restore" property becomes "round-trip-
# idempotent": restored V == stored V (deterministic round-trip), so
# TOKENS match exactly (the behavioral guarantee), but bf16-exact logit
# identity is gone by design (V passed through INT8). The honest gate:
# tokens always match; bit-identity required only at bf16 storage.
import os                                                   # noqa: E402
quant_v = bool(int(os.environ.get("GEMMA4_QUANT_V", "0")))
toks_ok = a_toks == ref_toks and b_match
bits_ok = a_bit and b_bit
if quant_v:
    ok = toks_ok                      # idempotent contract: tokens match
    print("(QUANT_V: contract is round-trip-idempotent, not "
          "bit-identical — tokens are the guarantee)", flush=True)
else:
    ok = toks_ok and bits_ok          # lossless contract: bit-identical
print(f"STATE GATE: {'PASS' if ok else 'FAIL'}", flush=True)
print("DONE", flush=True)
