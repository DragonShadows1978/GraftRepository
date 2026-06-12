"""Fused GDN step gate: composed recurrence vs tc.gated_delta_step on
the REAL model — 12 greedy decode tokens from a GT prompt must match
(token-identical) with per-step logit deltas at noise floor, the
DeltaNet states must track, and the step timing is recorded.

  python3 tests/qwen35_fusedstep_gate.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.qwen35_tc import Qwen35_TC, GatedDeltaNetTC       # noqa: E402

gt = np.load("/mnt/ForgeRealm/qwen35_gt/gt_2.npz")
ids = gt["ids"]

tc.set_alloc_pooling(True)
m, info = Qwen35_TC.from_pretrained()
print(f"loaded: {info} | kernel present: "
      f"{hasattr(tc, 'gated_delta_step')}", flush=True)


def decode12(fused):
    GatedDeltaNetTC.USE_FUSED_STEP = fused
    with tc.no_grad():
        lg, caches = m(ids, last_token_only=True)
        nxt = int(lg.float().numpy()[0, -1].argmax())
        off, toks, lgs = ids.shape[1], [], []
        t0 = time.perf_counter()
        for _ in range(12):
            toks.append(nxt)
            lg, caches = m(np.array([[nxt]], np.int64), caches=caches,
                           position_offset=off)
            off += 1
            lgs.append(lg.float().numpy()[0, -1])
            nxt = int(lgs[-1].argmax())
    tc.synchronize()
    ms = (time.perf_counter() - t0) / 12 * 1000
    s0 = caches[0][1].numpy()                 # layer-0 DeltaNet state
    return toks, lgs, s0, ms


toks_c, lgs_c, s0_c, ms_c = decode12(False)
toks_f, lgs_f, s0_f, ms_f = decode12(True)

dl = max(float(np.abs(a - b).max()) for a, b in zip(lgs_c, lgs_f))
ds = float(np.abs(s0_c - s0_f).max())
tok_match = toks_c == toks_f
print(f"composed {ms_c:.1f} ms/tok | fused {ms_f:.1f} ms/tok "
      f"({ms_c / ms_f:.2f}x)", flush=True)
print(f"tokens {'MATCH' if tok_match else 'DIFF'} | max|Δlogit| {dl:.5f} "
      f"| L0 state max|Δ| {ds:.2e}", flush=True)
print(f"FUSED-STEP GATE: "
      f"{'PASS' if tok_match and dl <= 0.5 and ds <= 1e-3 else 'FAIL'}",
      flush=True)
print("DONE", flush=True)
