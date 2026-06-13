"""Prefill-ceiling probe (fused path): ONE context per process (the
one-process-per-context discipline — fragmentation fakes OOMs across
contexts). Finds where prefill ACTUALLY walls now that the fused O(L)
kernel + chunked quantize removed the blend transient. Optional 'qv'
arg adds INT8 V-storage (shrinks resident -> reaches further).

  python3 tests/gemma4_prefill_ceiling.py <S> [qv]
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402

S = int(sys.argv[1]) if len(sys.argv) > 1 else 12288
if len(sys.argv) > 2 and sys.argv[2] == "qv":
    from core.gemma4_tc import KVRing
    KVRing.QUANT_V = True
    QV = True
else:
    QV = False
from core.gemma4_tc import Gemma4_TC                         # noqa: E402


def vram():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return int(o.stdout.strip())


tc.set_alloc_pooling(True)
m, info = Gemma4_TC.from_pretrained()
for L in m.layers:
    if L.mixer.is_global:
        L.mixer.attention_mode = "apa_selective"
        L.mixer.refine_percentile = 0.10
        L.mixer.apa_min_context = 256
        L.mixer.fast_max_seq = 4096            # fused path above 4K
print(f"loaded | S={S} | QV={QV} | vram after load {vram()} MiB",
      flush=True)

rng = np.random.default_rng(0)
ids = rng.integers(1000, 200000, size=(1, S)).astype(np.int64)
try:
    with tc.no_grad():
        lg, _ = m(ids, last_token_only=True)
    a = lg.float().numpy()[0, -1]
    print(f"S={S} QV={QV}: SURVIVED | peak vram {vram()} MiB | "
          f"argmax {int(a.argmax())}", flush=True)
except RuntimeError as e:
    print(f"S={S} QV={QV}: OOM ({str(e)[:50]})", flush=True)
print("DONE", flush=True)
