"""OOM ceiling probe: how far does each attention mode run on THIS
card (Architect, 2026-06-12)? One load, two modes (standard vs APA
r0.10 as served, blend engaging past 2048), prefill ladder in 2K
stages reusing caches, 4 decode steps at each rung. Reports VRAM and
decode ms/tok per rung; an OOM ends that mode's ladder.

Predictions registered BEFORE measurement (ledger): the shared cache
ceiling is ~12-18K (global KV 16KB/tok in ~0.65GB slack); standard's
decode transients are lean (no repeat_kv); the blend path pays a 16x
MQA expansion on this architecture, so APA may OOM EARLIER at decode
— inverting the MiniCPM3-era law until the blend is MQA-reworked.

  python3 tests/gemma4_oom_ceiling.py
"""
import gc
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.gemma4_tc import Gemma4_TC                        # noqa: E402

STAGE = 1024
MAX_S = 26 * 1024
MODE = sys.argv[1] if len(sys.argv) > 1 else "standard"

tc.set_alloc_pooling(True)
m, info = Gemma4_TC.from_pretrained()
print(f"loaded: {info}", flush=True)

rng = np.random.default_rng(0)
ids_all = rng.integers(1000, 200000, size=(1, MAX_S)).astype(np.int64)


def vram_mb():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(out.stdout.strip())


def set_mode(mode):
    for L in m.layers:
        if L.mixer.is_global:
            L.mixer.attention_mode = mode
            L.mixer.refine_percentile = 0.10
            L.mixer.apa_min_context = 2048


m.PREFILL_CHUNK = 256   # shrink the prefill transient wall
for mode in (MODE,):
    set_mode(mode)
    print(f"=== mode {mode} ===", flush=True)
    caches, pos = None, 0
    tok_id = np.array([[5279]], np.int64)
    try:
        while pos < MAX_S:
            seg = ids_all[:, pos:pos + STAGE]
            with tc.no_grad():
                t0 = time.perf_counter()
                _, caches = m(seg, caches=caches, position_offset=pos,
                              last_token_only=True)
                t_pre = time.perf_counter() - t0
            pos += STAGE
            tc.empty_cache()
            with tc.no_grad():
                t0 = time.perf_counter()
                for _ in range(4):
                    _, caches = m(tok_id, caches=caches,
                                  position_offset=pos)
                tc.synchronize()
                ms = (time.perf_counter() - t0) / 4 * 1000
            print(f"  S={pos:6d}: prefill +{t_pre:5.1f}s | decode "
                  f"{ms:6.1f} ms/tok | vram {vram_mb()} MiB", flush=True)
    except RuntimeError as e:
        print(f"  S={pos}: OOM ({str(e)[:60]}) — ceiling for {mode}",
              flush=True)
    del caches
    # the caught OOM forms a reference CYCLE (exception -> traceback ->
    # frame -> cache locals -> exception): only the cyclic GC frees it,
    # and ~1GB of cache tensors hang off those frames (post-cleanup
    # vram measured 7801 MiB without this). Then drain both pools.
    gc.collect()
    tc.set_alloc_pooling(False)
    tc.set_alloc_pooling(True)
    tc.empty_cache()
    print(f"  post-cleanup vram: {vram_mb()} MiB", flush=True)
print("DONE", flush=True)
