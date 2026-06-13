"""Decode-era ceiling probe v2 (rings): ONE prefill to the target S,
then a long timed per-step decode — the shape real serving has. The
v1 ladder interleaved prefill stages with decode samples, cycling
caches through unwrap/reconvert and trimming pools between chunks;
its decode numbers (773ms) contradicted clean per-step measurement
(28.8ms) by 27x. One process per (mode, S): OOM episodes
fragmentation-pin the allocator.

  python3 tests/gemma4_decode_probe.py <standard|apa_selective> <S>
"""
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.gemma4_tc import Gemma4_TC                        # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "standard"
S = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
N_DECODE = 64


def vram_mb():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(out.stdout.strip())


tc.set_alloc_pooling(True)
m, info = Gemma4_TC.from_pretrained()
for L in m.layers:
    if L.mixer.is_global:
        L.mixer.attention_mode = MODE
        L.mixer.refine_percentile = 0.10
        L.mixer.apa_min_context = 2048          # serving config
print(f"loaded | mode {MODE} | target S {S}", flush=True)

rng = np.random.default_rng(0)
ids = rng.integers(1000, 200000, size=(1, S)).astype(np.int64)

t0 = time.perf_counter()
with tc.no_grad():
    lg, caches = m(ids, last_token_only=True)
tc.synchronize()
print(f"prefill {S} tok: {time.perf_counter() - t0:.1f}s | "
      f"vram {vram_mb()} MiB", flush=True)

tok = np.array([[5279]], np.int64)
times = []
with tc.no_grad():
    for i in range(N_DECODE):
        t0 = time.perf_counter()
        _, caches = m(tok, caches=caches, position_offset=S + i)
        tc.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
t = np.array(times[1:])                          # step 0 = ring conversion
print(f"decode x{N_DECODE}: convert {times[0]:.0f} ms | steady "
      f"min {t.min():.1f} / med {np.median(t):.1f} / max {t.max():.1f} "
      f"ms/tok ({1000 / np.median(t):.1f} tok/s) | vram {vram_mb()} MiB",
      flush=True)
print("SURVIVED", flush=True)
