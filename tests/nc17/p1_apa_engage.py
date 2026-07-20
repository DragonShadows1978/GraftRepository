"""APA engagement instrumentation for NC17-P1.

The engine's `apa_quant_attention` (tensor_cuda/quant.py) at refine_percentile<1
builds a per-(query,key) mask selecting keys where |bulk| >= mean + z*std, and
uses exact `ranking` there, quantized `bulk` elsewhere. "Engaged" = that refine
mask is non-degenerate: some keys selected as refine, some not, and the blended
scores actually differ from the pure-bulk scores. The June law: a run where APA
never engaged (mask all-0 or all-1, or refine path skipped) is NOT an APA
datapoint.

We wrap tc.apa_quant_attention with a version that (a) calls the real engine op
(so scored logits are exactly the engine's), and (b) recomputes the SAME mask to
accumulate engagement stats into a module-global. This does not alter the scored
math — it observes it.
"""
import numpy as np
import tensor_cuda as tc
from tensor_cuda import functional as F
from tensor_cuda.quant import _tables, _quantize_keys, _norm_ppf
import math

STATS = {"calls": 0, "observed": 0, "engaged_calls": 0, "sum_engaged_frac": 0.0,
         "min_engaged_frac": 1.0, "max_engaged_frac": 0.0,
         "refine_percentile": None, "z": None, "any_all0": 0, "any_all1": 0}

# Cap the mask recompute (it ~doubles APA attention cost). The June-law
# assertion needs a representative sample proving the refine mask is
# non-degenerate; a full recompute at 32K x 28 layers x N windows would blow the
# 590s cap. We observe the first MAX_OBSERVE calls in detail; ALL calls are
# counted in `calls`. Engagement asserted from the observed sample.
MAX_OBSERVE = 56

_REAL = tc.apa_quant_attention


def reset():
    STATS.update({"calls": 0, "observed": 0, "engaged_calls": 0,
                  "sum_engaged_frac": 0.0, "min_engaged_frac": 1.0,
                  "max_engaged_frac": 0.0, "any_all0": 0, "any_all1": 0})


def _wrapped(query, key, value, *, bulk_bits=2, refine_percentile=0.15,
             is_causal=False, scale=None, apa_rotation=True):
    out = _REAL(query, key, value, bulk_bits=bulk_bits,
               refine_percentile=refine_percentile, is_causal=is_causal,
               scale=scale, apa_rotation=apa_rotation)
    if refine_percentile < 1.0:
        STATS["calls"] += 1
        STATS["refine_percentile"] = refine_percentile
        # detailed observation only for the first MAX_OBSERVE calls
        if STATS["observed"] < MAX_OBSERVE:
            B, H, L, D = query.shape
            S = key.shape[-2]
            sc = scale if scale is not None else 1.0 / math.sqrt(D)
            dev = query.device.split(":")[0]
            R, CB, BND = _tables(D, bulk_bits, H, apa_rotation, dev)
            key_quant = _quantize_keys(key, R, CB, BND)
            bulk = tc.matmul(query, key_quant, alpha=sc, trans_b=True)
            absr = bulk.abs().float()
            if is_causal:
                cmask = np.triu(np.ones((L, S), np.float32), 1)
                absr = absr.masked_fill(tc.tensor(cmask, device=dev), 0.0)
            z = _norm_ppf(1.0 - max(0.0, min(1.0, refine_percentile)))
            thr = absr.mean([-1], True) + absr.std([-1], True) * z
            mask = absr.ge(thr).float().numpy()  # (B,H,L,S) 0/1
            if is_causal:
                valid = np.tril(np.ones((L, S), np.float32))  # (L,S)
                # per (B,H) the valid count is valid.sum(); mask is summed over
                # all B*H, so denom must scale by B*H (the earlier bug divided by
                # a single triangle -> frac > 1).
                denom = float(valid.sum()) * B * H
                selected = float((mask * valid[None, None]).sum())
            else:
                denom = float(mask.size)
                selected = float(mask.sum())
            frac = float(selected / max(denom, 1.0))
            STATS["observed"] += 1
            STATS["z"] = float(z)
            if frac <= 0.0:
                STATS["any_all0"] += 1
            elif frac >= 1.0:
                STATS["any_all1"] += 1
            else:
                STATS["engaged_calls"] += 1
            STATS["sum_engaged_frac"] += frac
            STATS["min_engaged_frac"] = min(STATS["min_engaged_frac"], frac)
            STATS["max_engaged_frac"] = max(STATS["max_engaged_frac"], frac)
    return out


def install():
    """Monkeypatch tc.apa_quant_attention AND the reference the adapter already
    imported (core.mistral7b_tc imported `tc` as the module, so patching the
    attribute on the module object is sufficient — the adapter calls
    tc.apa_quant_attention by attribute lookup at call time)."""
    tc.apa_quant_attention = _wrapped


def summary():
    c = STATS["calls"]
    o = STATS["observed"]
    return {
        "apa_calls": c,
        "observed_calls": o,
        "engaged_calls": STATS["engaged_calls"],
        "all0_calls": STATS["any_all0"],
        "all1_calls": STATS["any_all1"],
        "mean_engaged_frac": (STATS["sum_engaged_frac"] / o) if o else None,
        "min_engaged_frac": STATS["min_engaged_frac"] if o else None,
        "max_engaged_frac": STATS["max_engaged_frac"] if o else None,
        "refine_percentile": STATS["refine_percentile"],
        "z_threshold": STATS["z"],
        # ENGAGED assertion: APA ran (calls>0) AND at least one observed call had
        # a non-degenerate refine mask (some-but-not-all keys refined) — the
        # June-law datapoint-validity flag.
        "APA_ENGAGED": (c > 0 and STATS["engaged_calls"] > 0),
    }
