"""E2 — digest fidelity / the consolidation decay curve (MiniCPM3, MLA).

Per chunk (E1-MLA confirmation set, alien needles):
  D0   direct mount -> probe                      (denominator)
  G1   digest generated WITH the chunk mounted    (gen fidelity: needle in text?)
  D1   mount ONLY the digest graft -> probe       (retrieval through digest)
  G2   digest-of-digest (D1 graft mounted)        (chained gen fidelity)
  D2   mount ONLY the 2nd-gen digest -> probe     (era-graft decay point)

Decomposition matters: E2_gen (needle survives INTO digest text) vs E2_ret
(needle retrievable FROM a mounted digest graft). Selective-amnesia already
proved digests must spell facts in their own tokens; this measures whether
they DO (prompted) and whether the spelled fact reads back. Protocol fixed
in advance; greedy; fast stack (parity-gated 2026-06-10); digests harvested
standalone (clean keys, property measurement).
"""
import os, sys, numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc
from core.minicpm3_tc import MiniCPM3_TC, _snap
from core.mistral7b_tc import QuantLinearTC, RMSNormTC
from core import kv_graft
from tokenizers import Tokenizer as HFTok
import ast, re
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_graft_e1_mla.py")).read()
CASES = ast.literal_eval(re.search(r"CASES = (\[.*?\n\])\n", _src, re.S).group(1))

QuantLinearTC.FUSED_DECODE = True
RMSNormTC.USE_FUSED = True
tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
m, info = MiniCPM3_TC.from_pretrained()
tc.set_alloc_pooling(True)
for L in m.layers:
    L.self_attn.absorbed_decode = True
m.extend_rope(4096)
print(f"loaded: {info} (fast stack)", flush=True)

DIGEST_PROMPT = ("User: Write a one-sentence archive note for the document "
                 "above, preserving every name, code, number, and time "
                 "verbatim.\nAssistant:")

def last_logits(idlist, caches=None, pos=0):
    with tc.no_grad():
        lg, c = m(np.array([idlist], dtype=np.int64), kv_caches=caches,
                  position_offset=pos, last_token_only=True)
    return lg.numpy()[0, -1].astype(np.float32), c

def gen(prompt_text, mount, ngen):
    kv_graft.clear_injection(m)
    if mount is not None:
        kv_graft.set_injection_mla(m, mount)
    ids = tok.encode(prompt_text).ids
    row, caches = last_logits(ids)
    pos = len(ids)
    out = [int(row.argmax())]
    for _ in range(ngen - 1):
        row, caches = last_logits([out[-1]], caches, pos)
        pos += 1
        out.append(int(row.argmax()))
    kv_graft.clear_injection(m)
    txt = tok.decode(out)
    for stop in ("\nUser:", "User:", "\n\n"):
        if stop in txt:
            txt = txt.split(stop)[0]
    return txt.strip()

def hit(text, accepts):
    t = text.lower()
    return any(a in t for a in accepts)

d0 = g1 = d1 = g2 = d2 = 0
d1_given_g1 = [0, 0]   # retrieval success / attempts where digest had needle
d2_given_g2 = [0, 0]
for i, (chunk, q, acc) in enumerate(CASES):
    raw = kv_graft.harvest_kv_mla(m, tok.encode(chunk).ids)
    probe = f"User: {q}\nAssistant:"

    a0 = gen(probe, raw, 48)
    h0 = hit(a0, acc)
    d0 += h0

    dig1 = gen(DIGEST_PROMPT, raw, 80)
    hg1 = hit(dig1, acc)
    g1 += hg1
    note1 = f"ARCHIVE NOTE. {dig1}\n"
    harv1 = kv_graft.harvest_kv_mla(m, tok.encode(note1).ids)
    a1 = gen(probe, harv1, 48)
    h1 = hit(a1, acc)
    d1 += h1
    if hg1:
        d1_given_g1[1] += 1
        d1_given_g1[0] += h1

    dig2 = gen(DIGEST_PROMPT, harv1, 80)
    hg2 = hit(dig2, acc)
    g2 += hg2
    note2 = f"ARCHIVE NOTE. {dig2}\n"
    harv2 = kv_graft.harvest_kv_mla(m, tok.encode(note2).ids)
    a2 = gen(probe, harv2, 48)
    h2 = hit(a2, acc)
    d2 += h2
    if hg2:
        d2_given_g2[1] += 1
        d2_given_g2[0] += h2

    print(f"case {i}: D0 {'HIT ' if h0 else 'MISS'} | G1 {'in' if hg1 else 'LOST'} "
          f"D1 {'HIT ' if h1 else 'MISS'} | G2 {'in' if hg2 else 'LOST'} "
          f"D2 {'HIT ' if h2 else 'MISS'}", flush=True)
    print(f"  digest1: {dig1[:90]!r}", flush=True)
    print(f"  digest2: {dig2[:90]!r}", flush=True)

N = len(CASES)
print(f"\nE2: D0 direct {d0}/{N} | G1 gen {g1}/{N} -> D1 through-digest {d1}/{N} "
      f"| G2 chained-gen {g2}/{N} -> D2 {d2}/{N}", flush=True)
print(f"retrieval-given-spelled: D1 {d1_given_g1[0]}/{d1_given_g1[1]}, "
      f"D2 {d2_given_g2[0]}/{d2_given_g2[1]}", flush=True)
print("DONE", flush=True)
