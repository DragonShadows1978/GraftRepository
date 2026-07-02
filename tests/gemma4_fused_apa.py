"""Fused-path gate: the fused O(L) apa_selective_attention on Gemma's
global layers (S > fast_max_seq) vs the cuBLAS-blend. Run ONE PART PER
PROCESS — A's 1024-ctx run fragments the pool and falsely OOMs B's 12K
(a bare 12K run survives at 7805 MiB). The standard gates use short
prompts (<4096) so they only exercise the blend; this is the only test
of the fused path.

  python3 tests/gemma4_fused_apa.py a    # equivalence (blend vs fused)
  python3 tests/gemma4_fused_apa.py b    # 12K prefill survival
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.gemma4_tc import Gemma4_TC                        # noqa: E402

PART = sys.argv[1] if len(sys.argv) > 1 else "a"


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
        L.mixer.apa_min_context = 256          # engage APA on short ctx
print(f"loaded: {info} | vram {vram()} MiB", flush=True)

rng = np.random.default_rng(0)


def last_logits(ids, fast_max):
    for L in m.layers:
        if L.mixer.is_global:
            L.mixer.fast_max_seq = fast_max
    with tc.no_grad():
        lg, _ = m(ids, last_token_only=True)
    return lg.float().numpy()[0, -1]


if PART == "a":
    # EQUIVALENCE: 1024-token prefill, blend (fast_max=1e18) vs fused
    # (fast_max=0). Both must agree (top1 match / near-tie).
    ids = rng.integers(1000, 200000, size=(1, 1024)).astype(np.int64)
    blend = last_logits(ids, 10 ** 18)
    tc.empty_cache()
    fused = last_logits(ids, 0)
    d = float(np.abs(blend - fused).max())
    t_b, t_f = int(blend.argmax()), int(fused.argmax())
    flip = 0.0 if t_b == t_f else float(blend.max() - blend[t_f])
    ok = flip <= 3.0
    print(f"A equivalence (1024, blend vs fused): top1 "
          f"{'match' if t_b == t_f else f'flip cost {flip:.3f}'} | "
          f"max|d| {d:.3f} | {'PASS' if ok else 'FAIL'}", flush=True)
    print(f"FUSED-APA GATE A: {'PASS' if ok else 'FAIL'}", flush=True)

elif PART == "b":
    # 12K prefill on the fused path (fast_max=4096, serving config).
    ids = rng.integers(1000, 200000, size=(1, 12288)).astype(np.int64)
    for L in m.layers:
        if L.mixer.is_global:
            L.mixer.fast_max_seq = 4096
    with tc.no_grad():
        lg, _ = m(ids, last_token_only=True)
    a = lg.float().numpy()[0, -1]
    print(f"B 12K prefill (fused >4K): SURVIVED | vram {vram()} MiB | "
          f"argmax {int(a.argmax())}", flush=True)
    print("FUSED-APA GATE B: PASS", flush=True)

print("DONE", flush=True)
