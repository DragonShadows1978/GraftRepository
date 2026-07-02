"""MiniCPM3-4B (MLA): cross-the-floor perplexity sweep. Same model, same eval,
ONLY bulk_bits changes. Demonstrates the bulk-bits law on ONE architecture:
at/above floor -> faithful; below floor -> collapse.

  EXACT (standard)             -> the baseline (attention works)
  bulk-only (r0.00) @ 8-bit    -> above floor, expect ~= EXACT
  bulk-only (r0.00) @ 4-bit    -> at floor (MLA's law value), expect ~= EXACT
  bulk-only (r0.00) @ 2-bit    -> BELOW floor, expect COLLAPSE

  python3 tests/minicpm3_bulkbits_floor.py
"""
import os, sys, numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.minicpm3_tc import MiniCPM3_TC, _snap             # noqa: E402
from transformers import AutoTokenizer                      # noqa: E402

WINDOW, SCORED, STEP, N_WINDOWS = 1024, 512, 64, 6
tok = AutoTokenizer.from_pretrained(_snap())


def get_text():
    try:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return "\n".join(r["text"] for r in ds)
    except Exception:
        import glob
        return ("\n".join(open(p).read()
                for p in sorted(glob.glob("/mnt/ForgeRealm/GraftRepository/docs/*.md")))) * 6


ids_all = tok(get_text(), return_tensors="np").input_ids[0].astype(np.int64)
print(f"eval stream: {len(ids_all)} tokens", flush=True)
tc.set_alloc_pooling(True)
m, info = MiniCPM3_TC.from_pretrained()
print(f"loaded: {info}", flush=True)


def set_mode(mode, refine=0.0, bulk_bits=4):
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
            count += len(tgt); pos += n
            tc.empty_cache()
    return nll, count


# (label, mode, refine, bulk_bits)
SETTINGS = [
    ("EXACT (standard)",       "standard",      0.0, 4),
    ("bulk-only  8-bit",       "apa_selective", 0.0, 8),
    ("bulk-only  4-bit (floor)", "apa_selective", 0.0, 4),
    ("bulk-only  2-bit (sub)", "apa_selective", 0.0, 2),
]
for name, mode, refine, bb in SETTINGS:
    set_mode(mode, refine, bb); tc.empty_cache()
    tot_nll, tot_cnt = 0.0, 0
    for w in range(N_WINDOWS):
        ids = ids_all[w * WINDOW:(w + 1) * WINDOW]
        nll, cnt = window_nll(ids); tot_nll += nll; tot_cnt += cnt
    print(f"{name:26s} ppl {np.exp(tot_nll/tot_cnt):12.3f}", flush=True)
print("DONE", flush=True)
