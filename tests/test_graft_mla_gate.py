"""MLA latent-graft gates on MiniCPM3-4B, in house order:

G1 EQUIVALENCE (the falsifiable prediction): logits of [BRIEFING + PROMPT]
   in-context vs harvest(BRIEFING) mounted + PROMPT alone, compared at the
   SAME absolute seats, teacher-forced (no generation). Must match at the
   engine's numeric noise floor with top-1 identical, or the MLA port is wrong.
G2 CACHED STREAM (teacher-forced): fixed chunk stream fed through the latent
   cache with the graft mounted at chunk-0 prefill vs one full prefill.
   PLAIN run gives this model's own cache-vs-prefill noise floor; GRAFT must
   match it. graft_seats persistence is what's under test.
G3 RECALL sanity: needle questions answered via cached greedy decode.
"""
import os, sys, numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc
from core.minicpm3_tc import MiniCPM3_TC, _snap
from core import kv_graft
from tokenizers import Tokenizer as HFTok

tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
m, info = MiniCPM3_TC.from_pretrained()
print(f"loaded: {info}", flush=True)
m.extend_rope(4096)

BRIEFING = ("CLASSIFIED BRIEFING.\nProject codename: AZURE HERON.\n"
            "Access code: 73-4412.\nContact: Dr. Selena Vask.\nEND BRIEFING.\n")
PROMPT = ("User: What is the project codename?\nAssistant: The codename is")

b_ids = tok.encode(BRIEFING).ids
p_ids = tok.encode(PROMPT).ids
Sg = len(b_ids)
print(f"briefing {Sg} tokens, prompt {len(p_ids)} tokens", flush=True)

def all_logits(idlist, caches=None, pos=0):
    lg, c = m(np.array([idlist], dtype=np.int64), kv_caches=caches,
              position_offset=pos, last_token_only=False)
    return lg.numpy()[0].astype(np.float32), c

# ---- G1: equivalence at the prompt's seats
kv_graft.clear_injection(m)
ctx, _ = all_logits(b_ids + p_ids)
ctx_p = ctx[Sg:]                                  # logits at prompt positions

harv = kv_graft.harvest_kv_mla(m, b_ids)
sz = sum(h["c"].nbytes + h["kpe"].nbytes for h in harv if h)
print(f"harvested latent graft: {sz/1e6:.1f} MB host ({Sg} tokens, 62 layers)", flush=True)
kv_graft.set_injection_mla(m, harv)
gra, _ = all_logits(p_ids)                        # prompt alone, graft mounted
kv_graft.clear_injection(m)

d = float(np.max(np.abs(gra - ctx_p)))
t1 = bool((gra.argmax(-1) == ctx_p.argmax(-1)).all())
nt1 = int((gra.argmax(-1) != ctx_p.argmax(-1)).sum())
print(f"\nG1 graft-vs-incontext: max|logit diff|={d:.4f}  "
      f"top1 {'IDENTICAL' if t1 else f'{nt1} DIFF'}", flush=True)

# ---- G2: teacher-forced cached stream, PLAIN then GRAFT
CHUNKS = [tok.encode("User: Hello, are you ready?\nAssistant: Yes, ready.").ids,
          tok.encode("\nUser: Name a river in Europe.\nAssistant: The Rhine.").ids,
          tok.encode("\nUser: What is 17+25?\nAssistant: 42.").ids,
          tok.encode("\nUser: What is the access code?\nAssistant: 73-4412.").ids]
full_ids = [t for c in CHUNKS for t in c]

def run_stream(with_graft):
    kv_graft.clear_injection(m)
    if with_graft:
        kv_graft.set_injection_mla(m, harv)
    A, _ = all_logits(full_ids)                   # ground truth, one prefill
    kv_graft.clear_injection(m)
    if with_graft:
        kv_graft.set_injection_mla(m, harv)
    caches, pos, idx, out = None, 0, 0, []
    for ci, ch in enumerate(CHUNKS):
        Lg, caches = all_logits(ch, caches, pos)
        dd = float(np.max(np.abs(Lg - A[idx:idx + len(ch)])))
        tt = bool((Lg.argmax(-1) == A[idx:idx + len(ch)].argmax(-1)).all())
        out.append((ci, dd, tt))
        pos += len(ch); idx += len(ch)
    kv_graft.clear_injection(m)
    return out

for wg in (False, True):
    tag = f"GRAFT(Sg={Sg})" if wg else "PLAIN"
    print(f"\nG2 {tag} teacher-forced cache-vs-prefill:", flush=True)
    for ci, dd, tt in run_stream(wg):
        print(f"  chunk {ci}: max|logit diff|={dd:.4f}  all-top1 "
              f"{'SAME' if tt else 'DIFF'}", flush=True)

# ---- G3: recall through cached greedy decode
def last_logits(idlist, caches=None, pos=0):
    lg, c = m(np.array([idlist], dtype=np.int64), kv_caches=caches,
              position_offset=pos, last_token_only=True)
    return lg.numpy()[0, -1].astype(np.float32), c

QS = [("What is the project codename?", ["azure heron"]),
      ("What is the access code?", ["73-4412"]),
      ("Who is the contact?", ["vask"])]
print("\nG3 recall (graft mounted, cached decode):", flush=True)
hits = 0
for q, acc in QS:
    kv_graft.clear_injection(m)
    kv_graft.set_injection_mla(m, harv)
    ids = tok.encode(f"User: {q}\nAssistant:").ids
    row, caches = last_logits(ids)
    pos = len(ids)
    out = [int(row.argmax())]
    for _ in range(31):
        row, caches = last_logits([out[-1]], caches, pos)
        pos += 1
        out.append(int(row.argmax()))
    g = tok.decode(out)
    ok = any(a in g.lower() for a in acc)
    hits += ok
    print(f"  {'HIT ' if ok else 'MISS'} | {g.strip()[:70]!r}", flush=True)
kv_graft.clear_injection(m)
print(f"G3: {hits}/{len(QS)}", flush=True)
print("DONE", flush=True)
