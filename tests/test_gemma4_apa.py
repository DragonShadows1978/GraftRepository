"""APA gate for Gemma 4: apa_selective (bulk4 / refine 0.15, cuBLAS
blend) on the 8 GLOBAL layers vs the engine's own standard attention —
engine-vs-engine, both INT4, so disagreements must be near-tie flips
under the STANDARD run's logits. Sliding layers are window-bounded and
never take APA. Plus a templated coherence smoke in APA mode.

  python3 tests/test_gemma4_apa.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.gemma4_tc import Gemma4_TC                        # noqa: E402

GT_DIR = os.environ.get("GEMMA4_GT", "/mnt/ForgeRealm/gemma4_gt_qat")

tc.set_alloc_pooling(True)
m, info = Gemma4_TC.from_pretrained()
print(f"loaded: {info}", flush=True)


def set_mode(mode):
    for L in m.layers:
        if L.mixer.is_global:
            L.mixer.attention_mode = mode
            # force the blend at any context — this gate tests the APA
            # MACHINERY; the serving threshold would skip it at short S
            L.mixer.apa_min_context = 0


overall = True
for pi in range(4):
    gt = np.load(os.path.join(GT_DIR, f"gt_{pi}.npz"))
    ids = gt["ids"]
    set_mode("standard")
    with tc.no_grad():
        lg_std, _ = m(ids)
    std = lg_std.float().numpy()[0]
    set_mode("apa_selective")
    with tc.no_grad():
        lg_apa, _ = m(ids)
    apa = lg_apa.float().numpy()[0]
    dis = np.nonzero(std.argmax(-1) != apa.argmax(-1))[0]
    costs = [float(std[t].max() - std[t][apa[t].argmax()]) for t in dis]
    worst = max(costs, default=0.0)
    ok = worst <= 3.0
    overall &= ok
    print(f"prompt {pi}: {len(dis)}/{ids.shape[1]} flips vs standard | "
          f"worst cost {worst:.3f} | {'OK' if ok else 'MISMATCH'}",
          flush=True)

# coherence smoke: templated prompt (gt_3), greedy 8 in APA mode
gt = np.load(os.path.join(GT_DIR, "gt_3.npz"))
ids = gt["ids"]
set_mode("apa_selective")
with tc.no_grad():
    lg, caches = m(ids, last_token_only=True)
off, toks = ids.shape[1], []
nxt = int(lg.float().numpy()[0, -1].argmax())
with tc.no_grad():
    for _ in range(8):
        toks.append(nxt)
        lg, caches = m(np.array([[nxt]], np.int64), caches=caches,
                       position_offset=off)
        off += 1
        nxt = int(lg.float().numpy()[0, -1].argmax())
print(f"APA greedy tail (templated France prompt): {toks}", flush=True)
print(f"APA GATE: {'PASS' if overall else 'FAIL'}", flush=True)
print("DONE", flush=True)
