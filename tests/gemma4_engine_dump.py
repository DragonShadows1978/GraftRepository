"""Dump engine teacher-forced logits for all GT prompts to npz so that
GT-vs-engine comparisons run host-side without GPU reloads.

  python3 tests/gemma4_engine_dump.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.gemma4_tc import Gemma4_TC                        # noqa: E402

GT_DIR = "/mnt/ForgeRealm/gemma4_gt"

tc.set_alloc_pooling(True)
m, info = Gemma4_TC.from_pretrained()
print(f"loaded: {info}", flush=True)

for pi in range(4):
    gt = np.load(os.path.join(GT_DIR, f"gt_{pi}.npz"))
    ids = gt["ids"]
    with tc.no_grad():
        logits, _ = m(ids)
    np.savez(os.path.join(GT_DIR, f"engine_{pi}.npz"),
             ids=ids, logits=logits.float().numpy()[0])
    print(f"engine_{pi} saved ({ids.shape[1]} tokens)", flush=True)
print("DONE", flush=True)
