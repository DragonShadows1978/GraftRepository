"""RULER-style needle-in-haystack retrieval on MiniCPM3-4B (MLA), APA parity.

ThriftAttention reports RULER task accuracy (retrieval), not perplexity. This
is the generate-and-score analog: inject a key->value 'needle' at a known depth
into a long filler 'haystack', ask for the value, greedy-decode the answer,
exact-match against ground truth. Three attention settings per item:
  EXACT (standard) / APA r0.15 / BULK r0.00 (no refine).

Multiple context lengths x needle depths => a real retrieval grid, not one probe.
Accuracy is the metric (RULER idiom). If APA == EXACT, parity holds; BULK is the
broken-baseline control.

  python3 tests/minicpm3_ruler_parity.py
"""
import os, sys, numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.minicpm3_tc import MiniCPM3_TC, _snap             # noqa: E402
from transformers import AutoTokenizer                      # noqa: E402

tok = AutoTokenizer.from_pretrained(_snap())

# Deterministic synthetic data (no Date/random at module import — fixed seed).
RNG = np.random.default_rng(20260624)
WORDS = ("the quick brown fox jumps over lazy dogs while distant thunder rolls "
         "across quiet valleys and rivers wind through ancient stone forests "
         "where merchants trade silk spice and copper beneath pale autumn skies").split()
# needle keys/values: memorable but not in the filler vocab
KEYS = ["azure-heron", "crimson-falcon", "silver-marmot", "golden-lynx", "violet-stag"]
def make_value():
    return f"{RNG.integers(10,99)}-{RNG.integers(1000,9999)}"

CTX_LENS = [2048, 3072, 4096]          # token budgets for the haystack
DEPTHS   = [0.1, 0.5, 0.9]             # needle position as fraction of context
N_PER_CELL = 3                          # needles per (ctxlen, depth) cell
PREFILL_CHUNK = 512                     # chunk prefill so dense SDPA never builds
                                        # the full (L,S) score matrix at once


def filler(n_tokens):
    """A haystack of ~n_tokens of innocuous filler."""
    out = []
    while True:
        out.append(" ".join(RNG.choice(WORDS, size=12)))
        if len(tok(" ".join(out)).input_ids) >= n_tokens:
            break
    return " ".join(out)


def build_item(ctx_len, depth):
    key, val = RNG.choice(KEYS), make_value()
    needle = f" The secret code for {key} is {val}. "
    pre = filler(int(ctx_len * depth))
    post = filler(int(ctx_len * (1 - depth)))
    prompt = (f"{pre}{needle}{post}\n"
              f"Question: What is the secret code for {key}?\nAnswer: The code is")
    return prompt, val


m, info = MiniCPM3_TC.from_pretrained()
print(f"loaded: {info}", flush=True)
m.extend_rope(8704)


def set_mode(mode, refine=0.15):
    for L in m.layers:
        L.self_attn.attention_mode = mode
        L.self_attn.refine_percentile = refine


def generate(prompt, max_new=12):
    ids = tok(prompt).input_ids
    with tc.no_grad():
        # CHUNKED prefill: feed the haystack in PREFILL_CHUNK-token pieces so the
        # dense (EXACT) path never materializes the full (L,S) score matrix.
        caches, pos = None, 0
        for s in range(0, len(ids), PREFILL_CHUNK):
            chunk = ids[s:s + PREFILL_CHUNK]
            lg, caches = m(np.array([chunk], dtype=np.int64), kv_caches=caches,
                           position_offset=pos, last_token_only=True)
            pos += len(chunk)
        out = [int(lg.float().numpy()[0, -1].argmax())]
        for _ in range(max_new - 1):
            lg, caches = m(np.array([[out[-1]]], dtype=np.int64),
                           kv_caches=caches, position_offset=pos)
            pos += 1
            out.append(int(lg.float().numpy()[0, -1].argmax()))
        tc.empty_cache()
    return tok.decode(out)


# Pre-build the item grid ONCE so every setting sees identical needles.
ITEMS = []
for cl in CTX_LENS:
    for d in DEPTHS:
        for _ in range(N_PER_CELL):
            ITEMS.append((cl, d, *build_item(cl, d)))
print(f"grid: {len(CTX_LENS)}x{len(DEPTHS)} cells x {N_PER_CELL} = {len(ITEMS)} needles", flush=True)

SETTINGS = [("EXACT", "standard", None),
            ("APA r0.15", "apa_selective", 0.15),
            ("BULK r0.00", "apa_selective", 0.00)]

results = {}
for name, mode, refine in SETTINGS:
    set_mode(mode, refine if refine is not None else 0.15)
    tc.empty_cache()
    hits, per_len = 0, {cl: [0, 0] for cl in CTX_LENS}
    for cl, d, prompt, val in ITEMS:
        try:
            ans = generate(prompt)
        except RuntimeError as e:
            ans = f"<OOM:{e}>"
        ok = val in ans
        hits += ok
        per_len[cl][0] += ok; per_len[cl][1] += 1
    results[name] = hits / len(ITEMS)
    bylen = "  ".join(f"{cl//1024}k:{h}/{n}" for cl, (h, n) in per_len.items())
    print(f"{name:11s} acc {hits}/{len(ITEMS)} = {results[name]*100:5.1f}%   [{bylen}]", flush=True)

print("\n=== RULER-style retrieval parity (MiniCPM3-4B MLA) ===", flush=True)
ex = results["EXACT"]
for name, *_ in SETTINGS:
    rel = "" if name == "EXACT" else f"  ({(results[name]-ex)*100:+.1f} pts vs EXACT)"
    print(f"{name:11s} {results[name]*100:5.1f}%{rel}", flush=True)
print("DONE", flush=True)
