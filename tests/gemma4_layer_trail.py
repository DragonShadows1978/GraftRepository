"""Per-layer divergence trail for the Gemma 4 port (prompt 0, S=5).

GT hidden semantics (adjudicated from checkpoint math): hidden[0] =
scaled embeds; hidden[i+1] = layer-i output for i<=46; hidden[48] =
POST-final-norm (reproduces GT logits through the tied head with NO
extra norm — max|d| 0.0). The naive li=47 probe compares pre-norm vs
post-norm and collapses by construction.

  python3 tests/gemma4_layer_trail.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.gemma4_tc import Gemma4_TC, Gemma4Config, _cast  # noqa: E402

gt = np.load("/mnt/ForgeRealm/gemma4_gt/gt_0.npz")
ids = gt["ids"]

tc.set_alloc_pooling(True)
m, info = Gemma4_TC.from_pretrained()
print(f"loaded: {info}", flush=True)


def cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


prev = 1.0
for li in range(48):
    with tc.no_grad():
        _, _, h = m(ids, max_layers=li + 1)
    he = h.float().numpy()[0]
    hr = gt["hidden"][li + 1]
    c_last = cos(he[-1], hr[-1])
    c_all = np.mean([cos(he[t], hr[t]) for t in range(he.shape[0])])
    tag = " G" if Gemma4Config.is_global(li) else ""
    drop = " <-- DROP" if c_last < prev - 0.03 else ""
    print(f"L{li:2d}{tag}: last {c_last:.4f} | mean {c_all:.4f}{drop}",
          flush=True)
    prev = c_last

# final-normed output vs GT hidden[48] (the valid last-layer probe)
with tc.no_grad():
    _, _, h = m(ids, max_layers=48)
    hn = _cast(m.norm(h)).float().numpy()[0]
hr = gt["hidden"][48]
print(f"final-norm: last {cos(hn[-1], hr[-1]):.4f} | "
      f"mean {np.mean([cos(hn[t], hr[t]) for t in range(hn.shape[0])]):.4f}",
      flush=True)
print("DONE", flush=True)
