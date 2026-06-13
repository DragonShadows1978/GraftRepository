"""Tail-law ppl gate: does storing KV at reduced precision hold ppl?
Hooks Gemma4AttentionTC.KV_STORE_HOOK — every scored token's K/V passes
through it on EVERY path (prefill/decode/chunk), unlike the KVRing which
only constructs at first decode (the earlier version patched the ring
and was a NO-OP — all 7 modes returned bit-identical ppl, the tell).

A SANITY GUARD asserts each non-baseline mode actually perturbs the
logits (a no-op patch -> bit-identical ppl -> false "free" claim).

  baseline   : bf16
  int8_v/_kv : INT8 storage of V / both
  int4v      : 4-bit group-32 V storage (bulk floor; scale-free v_norm)
  on global / sliding / all layers

  python3 tests/gemma4_kv_quant_ppl.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.gemma4_tc import Gemma4_TC, Gemma4AttentionTC     # noqa: E402
from core.mistral7b_tc import (BlockTC, _kv_quant,          # noqa: E402
                               _kv_dequant)

WINDOW, SCORED, STEP = 2048, 1024, 64
N_WINDOWS = 6


def get_text():
    try:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return "\n".join(r["text"] for r in ds)
    except Exception as e:
        print(f"wikitext unavailable ({e}); local docs fallback", flush=True)
        import glob
        return "\n".join(
            open(p).read() for p in
            sorted(glob.glob("/mnt/ForgeRealm/GraftRepository/docs/*.md"))) * 4


def rt_int8(x):
    q, s = _kv_quant(x)
    return _kv_dequant(q, s)


def rt_int4(x):
    cdt = BlockTC.COMPUTE_DTYPE
    B, H, S, D = x.shape
    xg = x.float().reshape([B, H, S, D // 32, 32])
    scale = xg.abs().max([-1], True) * (1.0 / 8.0) + 1e-8
    q = (xg / scale + 8.5).astype("uint8").astype("float32") - 8.0
    deq = (q * scale).reshape([B, H, S, D])
    return deq.astype(cdt) if cdt != "float32" else deq


def make_hook(which, fn, layers):
    def hook(k, v, is_global):
        applies = (layers == "all"
                   or (layers == "global" and is_global)
                   or (layers == "sliding" and not is_global))
        if not applies:
            return k, v
        if which in ("k", "kv"):
            k = fn(k)
        if which in ("v", "kv"):
            v = fn(v)
        return k, v
    return hook


def make_split_hook(k_fn, v_fn, layers):
    """Per-tensor: different quant for K vs V (the asymmetric scheme)."""
    def hook(k, v, is_global):
        applies = (layers == "all"
                   or (layers == "global" and is_global)
                   or (layers == "sliding" and not is_global))
        if applies:
            if k_fn is not None:
                k = k_fn(k)
            if v_fn is not None:
                v = v_fn(v)
        return k, v
    return hook


MODES = {
    "baseline":    None,
    "int8_v":      ("v", rt_int8, "all"),
    "int8_kv":     ("kv", rt_int8, "all"),
    "int8_glob":   ("kv", rt_int8, "global"),
    "int8_slide":  ("kv", rt_int8, "sliding"),
    "int4v_glob":  ("v", rt_int4, "global"),
    "int4v_slide": ("v", rt_int4, "sliding"),
    "int4v_all":   ("v", rt_int4, "all"),
}
# the asymmetric combined mode the workflow flagged as MUST-MEASURE:
# K-INT8 + V-4bit on global (does combined cost add to ~+11% or stay
# sub-additive ~+7%? — decides if asymmetric is worth building).
SPLIT_MODES = {
    "int8k_int4v_glob": (rt_int8, rt_int4, "global"),
}

from transformers import AutoTokenizer                      # noqa: E402
tok = AutoTokenizer.from_pretrained("/mnt/ForgeRealm/models/gemma-4-12B-it")
ids_all = tok(get_text(), return_tensors="np").input_ids[0].astype(np.int64)
print(f"eval stream: {len(ids_all)} tokens", flush=True)

tc.set_alloc_pooling(True)
m, info = Gemma4_TC.from_pretrained()
print(f"loaded: {info}", flush=True)


def window_logits_and_nll(ids, capture_first=False):
    """Returns (nll, count, first_step_logits_or_None)."""
    nll, count, first = 0.0, 0, None
    with tc.no_grad():
        ctx = WINDOW - SCORED
        _, caches = m(ids[None, :ctx], last_token_only=True)
        pos = ctx
        while pos < len(ids) - 1:
            n = min(STEP, len(ids) - 1 - pos)
            lg, caches = m(ids[None, pos:pos + n], caches=caches,
                           position_offset=pos)
            lp = lg.float().numpy()[0]
            if capture_first and first is None:
                first = lp[0].copy()
            tgt = ids[pos + 1:pos + n + 1]
            mx = lp.max(-1, keepdims=True)
            lse = mx[:, 0] + np.log(np.exp(lp - mx).sum(-1))
            nll += float((lse - lp[np.arange(len(tgt)), tgt]).sum())
            count += len(tgt)
            pos += n
            tc.empty_cache()
    return nll, count, first


# baseline first, capturing logits for the sanity guard
results = {}
base_logits = None
ALL = list(MODES.items()) + [(n, ("SPLIT",) + s)
                             for n, s in SPLIT_MODES.items()]
for name, spec in ALL:
    if spec is None:
        Gemma4AttentionTC.KV_STORE_HOOK = None
    elif spec[0] == "SPLIT":
        Gemma4AttentionTC.KV_STORE_HOOK = make_split_hook(*spec[1:])
    else:
        Gemma4AttentionTC.KV_STORE_HOOK = make_hook(*spec)
    tot_nll, tot_cnt = 0.0, 0
    first_logits = None
    for w in range(N_WINDOWS):
        ids = ids_all[w * WINDOW:(w + 1) * WINDOW]
        if len(ids) < WINDOW:
            break
        nll, cnt, fl = window_logits_and_nll(ids, capture_first=(w == 0))
        if fl is not None and first_logits is None:
            first_logits = fl
        tot_nll += nll
        tot_cnt += cnt
    ppl = float(np.exp(tot_nll / tot_cnt))
    results[name] = ppl
    if name == "baseline":
        base_logits = first_logits
        print(f"{name:12s} ppl {ppl:9.4f}  ({tot_cnt} tok)", flush=True)
    else:
        # SANITY GUARD: a real quant must perturb the first-step logits
        d = float(np.abs(first_logits - base_logits).max())
        active = "ACTIVE" if d > 1e-4 else "*** NO-OP (patch dead) ***"
        base = results["baseline"]
        print(f"{name:12s} ppl {ppl:9.4f}  "
              f"(+{(ppl / base - 1) * 100:.3f}%) | logit max|d| "
              f"{d:.3e} [{active}]", flush=True)

Gemma4AttentionTC.KV_STORE_HOOK = None
print("DONE", flush=True)
