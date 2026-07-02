"""APA recovery sweep on Qwen2.5-1.5B (raw-key GQA), ThriftAttention-style.

This is the regime where the recovery framing MEANS something: the adapter's
own note says 2-bit TurboQuant bulk DESTROYS Qwen's outlier-heavy keys
(score corr 0.03 -> ppl ~555), while 8-bit recovers (corr 0.85 -> ppl ~8.13
== standard). So at 2-bit, BULK (refine 0) is a genuinely broken anchor and we
can measure how much the refine pass claws back.

Two bulk-bit regimes x refine sweep, one load each:
  EXACT (standard) = full-precision attention                       -> 100%
  apa rX           = bulk-quantized + refine top X at full precision
  BULK (r0.00)     = all keys bulk-quantized, none refined          -> 0%

recovery% = (nll_BULK - nll_X)/(nll_BULK - nll_EXACT)*100.

  python3 tests/qwen25_apa_recovery.py
"""
import os, sys, numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.qwen_tc import Qwen_TC, _SNAP                     # noqa: E402
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


tok = AutoTokenizer.from_pretrained(_SNAP)
ids_all = tok(get_text(), return_tensors="np").input_ids[0].astype(np.int64)
print(f"eval stream: {len(ids_all)} tokens", flush=True)

tc.set_alloc_pooling(True)
m, info = Qwen_TC.from_pretrained()       # defaults bulk_bits=8 on every layer
print(f"loaded: {info}", flush=True)


def set_mode(mode, refine=0.10, bulk_bits=8):
    for L in m.layers:
        L.self_attn.attention_mode = mode
        L.self_attn.refine_percentile = refine
        L.self_attn.bulk_bits = bulk_bits


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


def sweep(bulk_bits):
    print(f"\n========== BULK_BITS = {bulk_bits} ==========", flush=True)
    settings = [("EXACT (standard)", "standard", None),
                ("apa r0.25", "apa_selective", 0.25),
                ("apa r0.15", "apa_selective", 0.15),
                ("apa r0.10", "apa_selective", 0.10),
                ("apa r0.05", "apa_selective", 0.05),
                ("BULK (r0.00)", "apa_selective", 0.00)]
    nlls = {}
    for name, mode, refine in settings:
        set_mode(mode, refine if refine is not None else 0.10, bulk_bits)
        tot_nll, tot_cnt = 0.0, 0
        for w in range(N_WINDOWS):
            ids = ids_all[w * WINDOW:(w + 1) * WINDOW]
            nll, cnt = window_nll(ids); tot_nll += nll; tot_cnt += cnt
        nlls[name] = tot_nll / tot_cnt
        print(f"  {name:18s} nll {nlls[name]:7.4f}  ppl {np.exp(nlls[name]):10.3f}", flush=True)
    hi, lo = nlls["EXACT (standard)"], nlls["BULK (r0.00)"]
    gap = lo - hi
    print(f"  --- recovery (gap = {gap:.4f} nll) ---", flush=True)
    for name, *_ in settings:
        rec = (lo - nlls[name]) / gap * 100 if abs(gap) > 1e-6 else float("nan")
        print(f"  {name:18s} ppl {np.exp(nlls[name]):10.3f}  recovery {rec:7.1f}%", flush=True)


sweep(2)   # the broken regime — BULK should collapse, refine should claw back
sweep(8)   # the recovered regime — BULK already ~exact, refine near-free
print("\nDONE", flush=True)
