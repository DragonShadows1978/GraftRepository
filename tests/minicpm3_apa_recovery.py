"""APA recovery sweep on MiniCPM3-4B (MLA), ThriftAttention-style.

ThriftAttention reports: FP16 (exact) and FP4 (broken) anchor the gap,
then top-k promotion recovers some % of the FP4->FP16 gap. APA analog:
  - EXACT  = standard full-precision attention            -> 100% (top anchor)
  - BULK   = apa at refine 0.0 (no keys refined; all bulk-quantized) -> 0% (bottom anchor)
  - apa rX = recovery between them, by refine fraction X.

recovery% = (nll_X - nll_BULK) / (nll_EXACT - nll_BULK) * 100  (in nll, lower=better,
so recovery toward EXACT). One load, all settings. Teacher-forced NLL over the
last SCORED tokens of each WINDOW-token window, fed in cached chunks.

  python3 tests/minicpm3_apa_recovery.py
"""
import os, sys, numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.minicpm3_tc import MiniCPM3_TC, _snap             # noqa: E402
from transformers import AutoTokenizer                      # noqa: E402

WINDOW, SCORED, STEP, N_WINDOWS = 1024, 512, 64, 6


def get_text():
    try:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return "\n".join(r["text"] for r in ds)
    except Exception as e:
        print(f"wikitext unavailable ({e}); local docs fallback", flush=True)
        import glob
        return ("\n".join(open(p).read()
                for p in sorted(glob.glob("/mnt/ForgeRealm/GraftRepository/docs/*.md")))) * 6


tok = AutoTokenizer.from_pretrained(_snap())
ids_all = tok(get_text(), return_tensors="np").input_ids[0].astype(np.int64)
print(f"eval stream: {len(ids_all)} tokens", flush=True)
assert len(ids_all) >= WINDOW * N_WINDOWS, "not enough eval text"

tc.set_alloc_pooling(True)
m, info = MiniCPM3_TC.from_pretrained()
print(f"loaded: {info}", flush=True)


def set_mode(mode, refine=0.10):
    for L in m.layers:
        L.self_attn.attention_mode = mode
        L.self_attn.refine_percentile = refine


def window_nll(ids):
    nll, count = 0.0, 0
    with tc.no_grad():
        ctx = WINDOW - SCORED
        _, caches = m(ids[None, :ctx], last_token_only=True)
        pos = ctx
        while pos < len(ids) - 1:
            n = min(STEP, len(ids) - 1 - pos)
            lg, caches = m(ids[None, pos:pos + n], kv_caches=caches, position_offset=pos)
            lp = lg.float().numpy()[0]
            tgt = ids[pos + 1:pos + n + 1]
            mx = lp.max(-1, keepdims=True)
            lse = mx[:, 0] + np.log(np.exp(lp - mx).sum(-1))
            nll += float((lse - lp[np.arange(len(tgt)), tgt]).sum())
            count += len(tgt)
            pos += n
            tc.empty_cache()
    return nll, count


# (label, mode, refine)
SETTINGS = [
    ("EXACT (standard)", "standard",      None),
    ("apa r1.00",        "apa_selective", 1.00),
    ("apa r0.25",        "apa_selective", 0.25),
    ("apa r0.15",        "apa_selective", 0.15),
    ("apa r0.10",        "apa_selective", 0.10),
    ("apa r0.05",        "apa_selective", 0.05),
    ("BULK (r0.00)",     "apa_selective", 0.00),
]
nlls = {}
for name, mode, refine in SETTINGS:
    set_mode(mode, refine if refine is not None else 0.10)
    tot_nll, tot_cnt = 0.0, 0
    for w in range(N_WINDOWS):
        ids = ids_all[w * WINDOW:(w + 1) * WINDOW]
        nll, cnt = window_nll(ids)
        tot_nll += nll; tot_cnt += cnt
    nlls[name] = (tot_nll / tot_cnt)
    print(f"{name:18s} nll {tot_nll/tot_cnt:7.4f}  ppl {np.exp(tot_nll/tot_cnt):8.4f}", flush=True)

print("\n=== RECOVERY (ThriftAttention idiom: EXACT=100%, BULK=0%) ===", flush=True)
hi = nlls["EXACT (standard)"]; lo = nlls["BULK (r0.00)"]
gap = lo - hi
for name, *_ in SETTINGS:
    rec = (lo - nlls[name]) / gap * 100 if gap > 1e-9 else float("nan")
    print(f"{name:18s} ppl {np.exp(nlls[name]):8.4f}  recovery {rec:6.1f}%", flush=True)
print("DONE", flush=True)
