"""STORAGE x SCORING interaction: quantized KV storage WITH APA r0.10
scoring engaged (the real deployed config). Companion to
gemma4_kv_quant_ppl.py (storage-only, bf16 scoring).

Original: does storing
the global-layer KV at reduced precision hold perplexity? No resident-
buffer redesign — we patch the KVRing to store round-tripped (quant ->
dequant) k/v on the 8 global layers and measure ppl drift vs bf16. If
ppl holds, quantized STORAGE is justified; if it tanks, the direction
is dead before any buffer code is written.

  baseline : bf16 K,V
  int8_v   : V stored INT8 (scale-free v_norm => bulk-bits law predicts safe)
  int8_kv  : K and V both INT8 (the proven GQA-port path => 2x cache)
  int4_v   : V stored 4-bit group-32 symmetric (the bulk floor)

  python3 tests/gemma4_kv_quant_ppl.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
import core.gemma4_tc as g4                                 # noqa: E402
from core.gemma4_tc import Gemma4_TC                        # noqa: E402
from core.mistral7b_tc import (BlockTC, _kv_quant,          # noqa: E402
                               _kv_dequant)

WINDOW, SCORED, STEP = 2048, 1024, 64
N_WINDOWS = 4


def rt_int8(x):
    q, s = _kv_quant(x)
    return _kv_dequant(q, s)


def rt_int4(x):
    """group-32 symmetric round-trip: s = max|x_group|/8, w = s*round(x/s)
    clamped to [-8,7] (the engine q4_0 grid, the bulk floor)."""
    cdt = BlockTC.COMPUTE_DTYPE
    B, H, S, D = x.shape
    xg = x.float().reshape([B, H, S, D // 32, 32])
    scale = xg.abs().max([-1], True) * (1.0 / 8.0) + 1e-8
    q = (xg / scale + 8.5).astype("uint8").astype("float32") - 8.0
    deq = (q * scale).reshape([B, H, S, D])
    return deq.astype(cdt) if cdt != "float32" else deq


MODES = {
    "baseline": None,
    "int8_v": ("v", rt_int8),
    "int8_kv": ("kv", rt_int8),
    "int4_v": ("v", rt_int4),
}

gt = np.load("/mnt/ForgeRealm/gemma4_gt_qat/gt_long.npz")
base_ids = gt["ids"][0]
reps = (N_WINDOWS * WINDOW) // len(base_ids) + 1
ids_all = np.tile(base_ids, reps)

tc.set_alloc_pooling(True)
m, info = Gemma4_TC.from_pretrained()
# engage APA r0.10 on the 8 global layers across the whole scored region
# (apa_min_context low) — measures quantized STORAGE x APA SCORING, the
# real deployed combination the Architect flagged.
for L in m.layers:
    if L.mixer.is_global:
        L.mixer.attention_mode = "apa_selective"
        L.mixer.refine_percentile = 0.10
        L.mixer.apa_min_context = 256
print(f"loaded: {info} | APA r0.10 engaged", flush=True)

_orig_init = g4.KVRing.__init__
_orig_append = g4.KVRing.append
SPEC = {"v": None}            # (which, fn) or None


def patched_init(self, k, v, ring_cap=None):
    _orig_init(self, k, v, ring_cap=ring_cap)
    self._is_global = ring_cap is None
    if SPEC["v"] is not None and self._is_global:
        which, fn = SPEC["v"]
        with tc.no_grad():
            if which in ("k", "kv"):
                tc.write_rows(self.kb, fn(self.kb.slice(2, 0, self.count)), 0)
            if which in ("v", "kv"):
                tc.write_rows(self.vb, fn(self.vb.slice(2, 0, self.count)), 0)


def patched_append(self, k1, v1, zero_row):
    if SPEC["v"] is not None and getattr(self, "_is_global", False):
        which, fn = SPEC["v"]
        if which in ("k", "kv"):
            k1 = fn(k1)
        if which in ("v", "kv"):
            v1 = fn(v1)
    return _orig_append(self, k1, v1, zero_row)


g4.KVRing.__init__ = patched_init
g4.KVRing.append = patched_append


def window_nll(ids):
    nll, count = 0.0, 0
    with tc.no_grad():
        ctx = WINDOW - SCORED
        _, caches = m(ids[None, :ctx], last_token_only=True)
        pos = ctx
        while pos < len(ids) - 1:
            n = min(STEP, len(ids) - 1 - pos)
            lg, caches = m(ids[None, pos:pos + n], caches=caches,
                           position_offset=pos)
            lp = lg.float().numpy()[0]
            tgt = ids[pos + 1:pos + n + 1]
            mx = lp.max(-1, keepdims=True)
            lse = mx[:, 0] + np.log(np.exp(lp - mx).sum(-1))
            nll += float((lse - lp[np.arange(len(tgt)), tgt]).sum())
            count += len(tgt)
            pos += n
            tc.empty_cache()
    return nll, count


results = {}
for name, spec in MODES.items():
    SPEC["v"] = spec
    tot_nll, tot_cnt = 0.0, 0
    for w in range(N_WINDOWS):
        ids = ids_all[w * WINDOW:(w + 1) * WINDOW]
        nll, cnt = window_nll(ids)
        tot_nll += nll
        tot_cnt += cnt
    ppl = float(np.exp(tot_nll / tot_cnt))
    results[name] = ppl
    base = results.get("baseline", ppl)
    delta = (f"  (+{(ppl / base - 1) * 100:.3f}% vs baseline)"
             if name != "baseline" else f"  ({tot_cnt} tok)")
    print(f"{name:10s} ppl {ppl:9.4f}{delta}", flush=True)

g4.KVRing.__init__ = _orig_init
g4.KVRing.append = _orig_append
print("DONE", flush=True)
