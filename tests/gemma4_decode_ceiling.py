"""Decode-ceiling probe: the REAL test of resident KV-quant. The prefill
probe only exercises the cat/APA transient (which KV-storage quant does
NOT shrink — the cache materializes at first DECODE). To prove K4+V4 buys
context, prefill a moderate seed then DECODE many tokens, growing the
global KV cache, and watch peak VRAM + survival.

  python3 tests/gemma4_decode_ceiling.py <target_tokens> [bf16|kv4]

ONE context per process (the fragmentation discipline). Run bf16 and kv4
in SEPARATE invocations and compare peak VRAM at the same target.
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
MODE = sys.argv[2] if len(sys.argv) > 2 else "bf16"
SEED = 512                                   # prefill seed length

from core.gemma4_tc import KVRing                            # noqa: E402
if MODE == "kv4":
    KVRing.QUANT_KV4 = True
elif MODE == "qv":
    KVRing.QUANT_V = True
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
        L.mixer.fast_max_seq = 4096
load_vram = vram()
print(f"loaded | MODE={MODE} | target={TARGET} | vram after load "
      f"{load_vram} MiB", flush=True)

rng = np.random.default_rng(0)
ids = rng.integers(1000, 200000, size=(1, SEED)).astype(np.int64)
peak = load_vram
try:
    with tc.no_grad():
        lg, caches = m(ids, last_token_only=True)
        pos = SEED
        # decode one token at a time, growing the global KV cache to TARGET
        while pos < TARGET:
            nxt = int(lg.float().numpy()[0, -1].argmax())
            step = np.array([[nxt]], dtype=np.int64)
            lg, caches = m(step, caches=caches, position_offset=pos)
            pos += 1
            if pos % 1024 == 0:
                v = vram()
                peak = max(peak, v)
                tc.empty_cache()
                print(f"  decoded to {pos} | vram {v} MiB", flush=True)
    print(f"MODE={MODE} target={TARGET}: SURVIVED | peak vram {peak} MiB "
          f"| KV-growth cost {peak - load_vram} MiB", flush=True)
except RuntimeError as e:
    print(f"MODE={MODE} target={TARGET}: OOM at ~{pos} ({str(e)[:60]})",
          flush=True)
print("DONE", flush=True)
