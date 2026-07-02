"""STORAGE x REFINE-PERCENTILE sweep: does widening the APA refine band
recover K4+V4's ppl cost?

The single-point r0.10 measurement showed int4_kv_glob (K4+V4) = +12.1%
vs +5.55% under standard scoring — K-4bit storage starves APA's refine
path, which at r0.10 leans hard on the top-10% keys being full precision.
HYPOTHESIS: a WIDER refine band (more keys refined => less reliance on
any one being pristine) shrinks K4's damage. This sweeps it.

For each refine r in {0.10, 0.15, 0.20, 0.30}, measure ppl for:
  baseline (bf16 storage), int4_v_glob, int4_kv_glob, int8k_int4v_glob
Each storage Δ is vs its OWN refine-r baseline (the baseline moves with
r), so the storage cost is isolated from the scoring change.

Same proven mechanism as gemma4_kv_quant_apa_ppl.py: hook KV_STORE_HOOK
(every scored token, every path) + a sanity guard. Real wikitext.

  python3 tests/gemma4_kv_quant_refine_sweep.py
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
REFINES = [0.10, 0.15, 0.20, 0.30]


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


def hook_kv(k_fn, v_fn):
    """global-only; k_fn/v_fn None to skip that tensor."""
    def hook(k, v, is_global):
        if not is_global:
            return k, v
        if k_fn is not None:
            k = k_fn(k)
        if v_fn is not None:
            v = v_fn(v)
        return k, v
    return hook


# (label, k_fn, v_fn) — None hook => baseline
CONFIGS = [
    ("baseline",          None),
    ("int4_v_glob",       hook_kv(None,    rt_int4)),
    ("int4_kv_glob",      hook_kv(rt_int4, rt_int4)),   # K4+V4
    ("int8k_int4v_glob",  hook_kv(rt_int8, rt_int4)),   # K8+V4 (the safer mix)
]

from transformers import AutoTokenizer                      # noqa: E402
tok = AutoTokenizer.from_pretrained("/mnt/ForgeRealm/models/gemma-4-12B-it")
ids_all = tok(get_text(), return_tensors="np").input_ids[0].astype(np.int64)
print(f"eval stream: {len(ids_all)} tokens", flush=True)

tc.set_alloc_pooling(True)
m, info = Gemma4_TC.from_pretrained()
for L in m.layers:
    if L.mixer.is_global:
        L.mixer.attention_mode = "apa_selective"
        L.mixer.apa_min_context = 256
print(f"loaded: {info}", flush=True)


def set_refine(r):
    for L in m.layers:
        if L.mixer.is_global:
            L.mixer.refine_percentile = r


def window_nll(ids, capture_first=False):
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


def measure(hook):
    Gemma4AttentionTC.KV_STORE_HOOK = hook
    tot_nll, tot_cnt, first = 0.0, 0, None
    for w in range(N_WINDOWS):
        ids = ids_all[w * WINDOW:(w + 1) * WINDOW]
        if len(ids) < WINDOW:
            break
        nll, cnt, fl = window_nll(ids, capture_first=(w == 0))
        if fl is not None and first is None:
            first = fl
        tot_nll += nll
        tot_cnt += cnt
    return float(np.exp(tot_nll / tot_cnt)), first


print("\n  refine | " + " | ".join(f"{c[0]:>16s}" for c in CONFIGS), flush=True)
print("  " + "-" * (9 + 19 * len(CONFIGS)), flush=True)
grid = {}
for r in REFINES:
    set_refine(r)
    base_ppl, base_logits = measure(None)
    cells = [f"{base_ppl:8.2f}        "]
    grid[(r, "baseline")] = base_ppl
    for label, hook in CONFIGS[1:]:
        ppl, fl = measure(hook)
        d = float(np.abs(fl - base_logits).max())
        tag = "" if d > 1e-4 else "!DEAD"
        delta = (ppl / base_ppl - 1) * 100
        grid[(r, label)] = ppl
        cells.append(f"{ppl:8.2f}(+{delta:5.2f}%){tag}")
    print(f"  r{r:.2f}  | " + " | ".join(cells), flush=True)

Gemma4AttentionTC.KV_STORE_HOOK = None
# the headline: does widening refine recover K4+V4?
print("\n=== K4+V4 cost vs refine band ===", flush=True)
for r in REFINES:
    b, kv = grid[(r, "baseline")], grid[(r, "int4_kv_glob")]
    print(f"  r{r:.2f}: K4+V4 = +{(kv/b-1)*100:.2f}%", flush=True)
print("DONE", flush=True)
