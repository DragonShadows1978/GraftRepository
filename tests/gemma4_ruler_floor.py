"""Gemma-4 12B (MQA+SWA, head_dim 512 globals): RULER-style needle retrieval,
cross-the-floor. ACCURACY metric (not perplexity) — wikitext ppl is the wrong
instrument for an instruct model (it generated 8k+ audit-passing GRAPA corpus
samples; raw-text continuation ppl just measures OOD-ness, not capability).
Retrieval accuracy is modality-agnostic: it finds the needle or it doesn't.

APA on GLOBAL layers only (apa_min_context forced 0). One load, 4 settings:
  EXACT (standard) / APA r0.15 4-bit / bulk-only 4-bit (floor) / bulk-only 2-bit (sub)

  python3 tests/gemma4_ruler_floor.py
"""
import os, sys, numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.gemma4_tc import Gemma4_TC                        # noqa: E402
from transformers import AutoTokenizer                      # noqa: E402

tok = AutoTokenizer.from_pretrained("/mnt/ForgeRealm/models/gemma-4-12B-it")
RNG = np.random.default_rng(20260624)
WORDS = ("the quick brown fox jumps over lazy dogs while distant thunder rolls "
         "across quiet valleys and rivers wind through ancient stone forests "
         "where merchants trade silk spice and copper beneath pale autumn skies").split()
KEYS = ["azure-heron", "crimson-falcon", "silver-marmot", "golden-lynx", "violet-stag"]
CTX_LENS, DEPTHS, N_PER_CELL = [2048, 4096], [0.1, 0.5, 0.9], 3


def make_value(): return f"{RNG.integers(10,99)}-{RNG.integers(1000,9999)}"
def filler(n):
    out = []
    while len(tok(" ".join(out)).input_ids) < n:
        out.append(" ".join(RNG.choice(WORDS, size=12)))
    return " ".join(out)
def build_item(cl, d):
    key, val = RNG.choice(KEYS), make_value()
    needle = f" The secret code for {key} is {val}. "
    prompt = (f"{filler(int(cl*d))}{needle}{filler(int(cl*(1-d)))}\n"
              f"Question: What is the secret code for {key}?\nAnswer: The code is")
    return prompt, val


m, info = Gemma4_TC.from_pretrained()
print(f"loaded: {info}", flush=True)


def set_mode(mode, refine=0.15, bulk_bits=4):
    for L in m.layers:
        if L.mixer.is_global:
            L.mixer.attention_mode = mode
            L.mixer.refine_percentile = refine
            L.mixer.bulk_bits = bulk_bits
            L.mixer.apa_min_context = 0


def generate(prompt, max_new=12):
    ids = tok(prompt).input_ids
    with tc.no_grad():
        lg, caches = m(np.array([ids], dtype=np.int64), last_token_only=True)
        pos = len(ids); out = [int(lg.float().numpy()[0, -1].argmax())]
        for _ in range(max_new - 1):
            lg, caches = m(np.array([[out[-1]]], dtype=np.int64), caches=caches,
                           position_offset=pos)
            pos += 1; out.append(int(lg.float().numpy()[0, -1].argmax()))
        tc.empty_cache()
    return tok.decode(out)


ITEMS = []
for cl in CTX_LENS:
    for d in DEPTHS:
        for _ in range(N_PER_CELL):
            ITEMS.append((cl, *build_item(cl, d)))
print(f"grid: {len(CTX_LENS)}x{len(DEPTHS)}x{N_PER_CELL} = {len(ITEMS)} needles", flush=True)

SETTINGS = [("EXACT",            "standard",      None, 4),
            ("APA r0.15 4-bit",  "apa_selective", 0.15, 4),
            ("bulk-only 4-bit",  "apa_selective", 0.00, 4),
            ("bulk-only 2-bit",  "apa_selective", 0.00, 2)]
res = {}
for name, mode, refine, bb in SETTINGS:
    set_mode(mode, refine if refine is not None else 0.15, bb); tc.empty_cache()
    hits, per = 0, {cl: [0, 0] for cl in CTX_LENS}
    for cl, prompt, val in ITEMS:
        try: ans = generate(prompt)
        except RuntimeError as e: ans = f"<OOM:{e}>"
        ok = val in ans; hits += ok; per[cl][0] += ok; per[cl][1] += 1
    res[name] = hits / len(ITEMS)
    bylen = "  ".join(f"{cl//1024}k:{h}/{n}" for cl, (h, n) in per.items())
    print(f"{name:17s} acc {hits}/{len(ITEMS)} = {res[name]*100:5.1f}%   [{bylen}]", flush=True)

print("\n=== Gemma-4 12B RULER floor (accuracy) ===", flush=True)
ex = res["EXACT"]
for name, *_ in SETTINGS:
    rel = "" if name == "EXACT" else f"  ({(res[name]-ex)*100:+.1f} pts)"
    print(f"{name:17s} {res[name]*100:5.1f}%{rel}", flush=True)
print("DONE", flush=True)
