"""Decode-step forensics for the Gemma 4 stack (the Qwen ladder
pattern: measure where the milliseconds live before touching anything).

Prefills a corpus-shaped ~700-token context, then times 32 decode
steps: (a) true end-to-end ms/tok, (b) per-section synced breakdown
(sliding-attn / global-attn / MLP / norms+glue / lm_head+softcap),
(c) the orchestration share = true time minus synced-kernel sum is NOT
separable that way — instead compare a fully-synced step (every layer
boundary) against the free-running step: the gap bounds launch-pipeline
overlap.

  python3 tests/gemma4_profile.py [apa|standard]
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.gemma4_tc import Gemma4_TC, _cast                # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "apa"

tc.set_alloc_pooling(True)
m, info = Gemma4_TC.from_pretrained()
if MODE == "apa":
    for L in m.layers:
        if L.mixer.is_global:
            L.mixer.attention_mode = "apa_selective"
print(f"loaded: {info} | mode {MODE}", flush=True)

rng = np.random.default_rng(0)
ids = rng.integers(1000, 200000, size=(1, 700)).astype(np.int64)
with tc.no_grad():
    lg, caches = m(ids, last_token_only=True)
tc.synchronize()
print("prefill done (700 tok)", flush=True)

tok_id = np.array([[5279]], np.int64)

# ---- (a) true end-to-end decode
with tc.no_grad():
    for _ in range(3):                                  # warm
        _, caches = m(tok_id, caches=caches, position_offset=700)
    tc.synchronize()
    t0 = time.perf_counter()
    N = 32
    for _ in range(N):
        lg, caches = m(tok_id, caches=caches, position_offset=700)
    tc.synchronize()
    dt = (time.perf_counter() - t0) / N * 1000
print(f"[a] end-to-end: {dt:.1f} ms/tok ({1000 / dt:.1f} tok/s)",
      flush=True)

# ---- (b) per-section synced breakdown (one representative step)
sec = {"embed": 0.0, "sliding": 0.0, "global": 0.0, "head": 0.0}
n_by = {"sliding": 0, "global": 0}
with tc.no_grad():
    for rep in range(8):
        t = time.perf_counter()
        h = m.embed_tokens(tok_id)
        tc.synchronize()
        sec["embed"] += time.perf_counter() - t
        new_caches = []
        for i, layer in enumerate(m.layers):
            t = time.perf_counter()
            h, c = layer(h, m.ropes, 700, caches[i])
            new_caches.append(c)
            tc.synchronize()
            key = "global" if layer.is_global else "sliding"
            sec[key] += time.perf_counter() - t
            if rep == 0:
                n_by[key] += 1
        caches = new_caches
        t = time.perf_counter()
        hn = _cast(m.norm(h))
        out = m.lm_head(hn)
        c0 = m.config.logit_softcap
        _ = ((out.float() * (1.0 / c0)).tanh() * c0).numpy()
        tc.synchronize()
        sec["head"] += time.perf_counter() - t
tot = sum(sec.values()) / 8 * 1000
print(f"[b] synced step total {tot:.1f} ms "
      f"(sync overhead inflates vs [a]):", flush=True)
for k, v in sec.items():
    ms = v / 8 * 1000
    per = f" ({ms / n_by[k]:.2f} ms/layer x{n_by[k]})" if k in n_by else ""
    print(f"    {k:8s} {ms:6.1f} ms{per}", flush=True)
print("DONE", flush=True)
