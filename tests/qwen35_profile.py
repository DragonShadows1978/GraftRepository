"""Decode-speed forensics: where do 833 ms/token go?

Components timed at M=1 (median of 20 after 3 warmup):
  - full decode step (model __call__ with cache)
  - lm_head alone, one MLP, one DeltaNet mixer, one attention mixer
  - host argmax round-trip
Then QuantLinearTC.FUSED_DECODE = True (engine int4 GEMV at M==1,
parity-gated on MiniCPM3) and the same table again, plus a logits
max|Δ| between the two dispatch modes on the same token.

  python3 tests/qwen35_profile.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.mistral7b_tc import QuantLinearTC                 # noqa: E402
from core.qwen35_tc import Qwen35_TC                        # noqa: E402

gt = np.load("/mnt/ForgeRealm/qwen35_gt/gt_2.npz")
ids = gt["ids"]

tc.set_alloc_pooling(True)
m, info = Qwen35_TC.from_pretrained()
print(f"loaded: {info}", flush=True)


def med(fn, n=20, warm=3):
    for _ in range(warm):
        fn()
    tc.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        tc.synchronize()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[n // 2] * 1000.0


def table(tag):
    with tc.no_grad():
        lg, caches = m(ids, last_token_only=True)
        nxt = np.array([[int(lg.float().numpy()[0, -1].argmax())]], np.int64)
        off = ids.shape[1]

        # full step (fresh cache copy each call would mutate state; use a
        # fixed input against the SAME caches — state drift is irrelevant
        # for timing)
        t_step = med(lambda: m(nxt, caches=caches, position_offset=off))

        h1 = tc.tensor(np.random.randn(1, 1, 4096).astype(np.float32))
        h1 = h1.astype("bfloat16")
        t_lmhead = med(lambda: m.lm_head(h1))
        t_mlp = med(lambda: m.layers[0].mlp(h1))
        dn = m.layers[0].mixer
        dn_state = caches[0]
        t_deltanet = med(lambda: dn(h1, dn_state))
        att = m.layers[3].mixer
        t_attn = med(lambda: att(h1, m.rope_cos, m.rope_sin, off, caches[3]))

        lgt = m(nxt, caches=caches, position_offset=off)[0]
        t_host = med(lambda: lgt.float().numpy()[0, -1].argmax())

    est = 24 * t_deltanet + 8 * t_attn + 32 * t_mlp + t_lmhead + t_host
    print(f"[{tag}] step {t_step:.1f}ms | lm_head {t_lmhead:.1f} | "
          f"mlp {t_mlp:.2f} (x32={32 * t_mlp:.1f}) | "
          f"deltanet {t_deltanet:.2f} (x24={24 * t_deltanet:.1f}) | "
          f"attn {t_attn:.2f} (x8={8 * t_attn:.1f}) | host {t_host:.2f} | "
          f"sum-of-parts {est:.1f}ms", flush=True)
    with tc.no_grad():
        out = m(nxt, caches=caches, position_offset=off)[0]
    return out.float().numpy()


lg_two = table("two-stage (FUSED_DECODE=False)")
QuantLinearTC.FUSED_DECODE = True
lg_gemv = table("int4 GEMV   (FUSED_DECODE=True) ")
print(f"dispatch parity max|Δlogit| = {np.abs(lg_two - lg_gemv).max():.5f}",
      flush=True)
print("DONE", flush=True)
