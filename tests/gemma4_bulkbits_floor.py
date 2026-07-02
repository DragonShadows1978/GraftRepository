"""Gemma-4 12B (MQA + sliding-window, head_dim 512 globals): cross-the-floor
perplexity sweep. Brand-new frontier architecture, run on the 4070S.

Same model, same eval, APA on the GLOBAL layers only (sliding stay standard),
apa_min_context forced 0 so APA engages everywhere. Only bulk_bits changes:

  EXACT (standard)              -> baseline (attention works)
  bulk-only (r0.00) @ 4-bit     -> qk-norm law floor, expect ~= EXACT
  bulk-only (r0.00) @ 2-bit     -> BELOW floor, expect collapse
  apa r0.15 @ 4-bit             -> normal operating point, expect ~= EXACT

  python3 tests/gemma4_bulkbits_floor.py
"""
import os, sys, numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.gemma4_tc import Gemma4_TC                        # noqa: E402
from transformers import AutoTokenizer                      # noqa: E402

WINDOW, SCORED, STEP, N_WINDOWS = 2048, 1024, 64, 4
tok = AutoTokenizer.from_pretrained("/mnt/ForgeRealm/models/gemma-4-12B-it")


def get_text():
    try:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return "\n".join(r["text"] for r in ds)
    except Exception:
        import glob
        return ("\n".join(open(p).read()
                for p in sorted(glob.glob("/mnt/ForgeRealm/GraftRepository/docs/*.md")))) * 8


ids_all = tok(get_text(), return_tensors="np").input_ids[0].astype(np.int64)
print(f"eval stream: {len(ids_all)} tokens", flush=True)
assert len(ids_all) >= WINDOW * N_WINDOWS, "not enough eval text"
tc.set_alloc_pooling(True)
m, info = Gemma4_TC.from_pretrained()
print(f"loaded: {info}", flush=True)


def set_mode(mode, refine=0.0, bulk_bits=4):
    for L in m.layers:
        if L.mixer.is_global:
            L.mixer.attention_mode = mode
            L.mixer.refine_percentile = refine
            L.mixer.bulk_bits = bulk_bits
            L.mixer.apa_min_context = 0


def window_nll(ids):
    nll, count = 0.0, 0
    with tc.no_grad():
        ctx = WINDOW - SCORED
        _, caches = m(ids[None, :ctx], last_token_only=True)
        pos = ctx
        while pos < len(ids) - 1:
            n = min(STEP, len(ids) - 1 - pos)
            lg, caches = m(ids[None, pos:pos + n], caches=caches, position_offset=pos)
            lp = lg.float().numpy()[0]
            tgt = ids[pos + 1:pos + n + 1]
            mx = lp.max(-1, keepdims=True)
            lse = mx[:, 0] + np.log(np.exp(lp - mx).sum(-1))
            nll += float((lse - lp[np.arange(len(tgt)), tgt]).sum())
            count += len(tgt); pos += n
            tc.empty_cache()
    return nll, count


# (label, mode, refine, bulk_bits)
SETTINGS = [
    ("EXACT (standard)",         "standard",      0.00, 4),
    ("bulk-only  4-bit (floor)", "apa_selective", 0.00, 4),
    ("bulk-only  2-bit (sub)",   "apa_selective", 0.00, 2),
    ("apa r0.15  4-bit",         "apa_selective", 0.15, 4),
]
for name, mode, refine, bb in SETTINGS:
    set_mode(mode, refine, bb); tc.empty_cache()
    tot_nll, tot_cnt = 0.0, 0
    for w in range(N_WINDOWS):
        ids = ids_all[w * WINDOW:(w + 1) * WINDOW]
        nll, cnt = window_nll(ids); tot_nll += nll; tot_cnt += cnt
    print(f"{name:26s} ppl {np.exp(tot_nll/tot_cnt):12.3f}", flush=True)
print("DONE", flush=True)
