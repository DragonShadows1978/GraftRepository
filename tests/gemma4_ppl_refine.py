"""APA refine-percentile perplexity sweep for Gemma 4 (Architect:
"typically we run r0.10 or r0.05 — the ppl loss is minuscule" — that
prior comes from pre-Gemma architectures; this model's globals are
1-KV-head MQA, so measure, don't assume).

One load, four settings: standard attention, apa r0.15 / r0.10 /
r0.05 (apa_min_context forced 0 so the blend is active everywhere).
Teacher-forced NLL over the last 1024 tokens of each 2048-token
window, fed in 256-token cached chunks (full-window logits don't fit).

  python3 tests/gemma4_ppl_refine.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.gemma4_tc import Gemma4_TC                        # noqa: E402

WINDOW, SCORED, STEP = 2048, 1024, 64
N_WINDOWS = 6


def get_text():
    try:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return "\n".join(r["text"] for r in ds)
    except Exception as e:
        print(f"wikitext unavailable ({e}); falling back to local docs",
              flush=True)
        import glob
        txt = []
        for p in sorted(glob.glob("/mnt/ForgeRealm/GraftRepository/docs/*.md")):
            txt.append(open(p).read())
        return "\n".join(txt) * 4


from transformers import AutoTokenizer                      # noqa: E402
tok = AutoTokenizer.from_pretrained("/mnt/ForgeRealm/models/gemma-4-12B-it")
ids_all = tok(get_text(), return_tensors="np").input_ids[0].astype(np.int64)
print(f"eval stream: {len(ids_all)} tokens", flush=True)
assert len(ids_all) >= WINDOW * N_WINDOWS, "not enough eval text"

tc.set_alloc_pooling(True)
m, info = Gemma4_TC.from_pretrained()
print(f"loaded: {info}", flush=True)


def set_mode(mode, refine=0.15):
    for L in m.layers:
        if L.mixer.is_global:
            L.mixer.attention_mode = mode
            L.mixer.refine_percentile = refine
            L.mixer.apa_min_context = 0


def window_nll(ids):
    """context = first WINDOW-SCORED tokens; NLL over the rest."""
    nll, count = 0.0, 0
    with tc.no_grad():
        ctx = WINDOW - SCORED
        _, caches = m(ids[None, :ctx], last_token_only=True)
        pos = ctx
        while pos < len(ids) - 1:
            n = min(STEP, len(ids) - 1 - pos)
            # logits for positions pos..pos+n-1 predict pos+1..pos+n
            lg, caches = m(ids[None, pos:pos + n], caches=caches,
                           position_offset=pos)
            lp = lg.float().numpy()[0]                      # (n, V) capped
            tgt = ids[pos + 1:pos + n + 1]
            # softcapped logits are still logits: NLL via logsumexp
            mx = lp.max(-1, keepdims=True)
            lse = mx[:, 0] + np.log(np.exp(lp - mx).sum(-1))
            nll += float((lse - lp[np.arange(len(tgt)), tgt]).sum())
            count += len(tgt)
            pos += n
            tc.empty_cache()
    return nll, count


SETTINGS = [("standard", None), ("apa r0.15", 0.15),
            ("apa r0.10", 0.10), ("apa r0.05", 0.05)]
results = {}
for name, refine in SETTINGS:
    if refine is None:
        set_mode("standard")
    else:
        set_mode("apa_selective", refine)
    tot_nll, tot_cnt = 0.0, 0
    for w in range(N_WINDOWS):
        ids = ids_all[w * WINDOW:(w + 1) * WINDOW]
        nll, cnt = window_nll(ids)
        tot_nll += nll
        tot_cnt += cnt
    ppl = float(np.exp(tot_nll / tot_cnt))
    results[name] = ppl
    base = results.get("standard", ppl)
    print(f"{name:12s} ppl {ppl:8.4f}"
          + (f"  (+{(ppl / base - 1) * 100:.3f}% vs standard)"
             if name != "standard" else f"  ({tot_cnt} tokens scored)"),
          flush=True)
print("DONE", flush=True)
