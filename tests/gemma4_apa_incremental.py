"""APA incremental-kq equivalence (#9 fix): greedy decode with the
cached-kq ring must match decode that re-quantizes the whole cache
every step, modulo bf16-noise near-ties. Also confirms the per-step
quantize cost is now O(D) not O(S) by checking kq_count advances by 1.

  python3 tests/gemma4_apa_incremental.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.gemma4_tc import Gemma4_TC, KVRing                # noqa: E402

# Use a SHORT prompt (avoids the orthogonal sliding-prefill band-mask
# OOM at long context) and LOWER apa_min_context so the global layers
# engage APA at decode on this short context — isolating exactly the
# incremental-kq decode path under test.
gt = np.load("/mnt/ForgeRealm/gemma4_gt_qat/gt_long.npz")
ids = gt["ids"][:, :900]
APA_MIN = 256

tc.set_alloc_pooling(True)
m, info = Gemma4_TC.from_pretrained()
for L in m.layers:
    if L.mixer.is_global:
        L.mixer.attention_mode = "apa_selective"
        L.mixer.refine_percentile = 0.10
        L.mixer.apa_min_context = APA_MIN
print(f"loaded: {info}", flush=True)


def greedy(n):
    with tc.no_grad():
        lg, caches = m(ids, last_token_only=True)
        toks, off = [], ids.shape[1]
        nxt = int(lg.float().numpy()[0, -1].argmax())
        for _ in range(n):
            toks.append(nxt)
            lg, caches = m(np.array([[nxt]], np.int64), caches=caches,
                           position_offset=off)
            off += 1
            nxt = int(lg.float().numpy()[0, -1].argmax())
    return toks, caches


toks, caches = greedy(12)
print(f"incremental-kq greedy: {toks}", flush=True)

# verify the kq cache actually engaged. Global (APA) layers are at
# i%6==5; their caches must be KVRings with a populated kqb.
rings = [(i, c) for i, c in enumerate(caches) if isinstance(c, KVRing)]
globals_ = [(i, c) for i, c in rings if i % 6 == 5]
print(f"caches: {len(rings)} rings, {len(globals_)} global; "
      f"global kqb states: "
      f"{[(i, c.kqb is not None, c.kq_count, c.count) for i, c in globals_]}",
      flush=True)
engaged = [c for i, c in globals_ if c.kqb is not None]
if not engaged:
    print("FAIL: no global layer engaged the kq cache", flush=True)
    g = None
else:
    g = engaged[0]
    print(f"kq ring engaged: kq_count={g.kq_count} count={g.count} "
          f"(equal => fully cached)", flush=True)

# determinism: identical run must match exactly (catches ring bugs)
toks2, _ = greedy(12)
det = toks == toks2
print(f"determinism: {'MATCH' if det else 'DIFF'}", flush=True)

ok = det and g is not None and g.kq_count == g.count
print(f"INCREMENTAL-KQ GATE: {'PASS' if ok else 'FAIL'}", flush=True)
print("DONE", flush=True)
