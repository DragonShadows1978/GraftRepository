"""E4-C — consolidated conversation memory (Phase-2 gate).

E4-arena flow, but after the 14 scripted turns the librarian consolidates
turns 1-4 and 5-8 (all six facts) into TWO digest grafts and RETIRES the
sources from routing. Probes 15-20 can only recover needles through the
digest nodes — this gates, at once: digest generation QC, routing to
multi-topic digest nodes via hierarchical descent (child centroids), and
recall through mounted digests inside the persistent arena.

Pass: recall >= 5/6 (E2 ceiling: one-time ~0.89 digest coefficient; the
Vesper extraction failure may also recur) with digest mounts doing the work.
"""
import os, sys, time, numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc
from core.minicpm3_tc import MiniCPM3_TC, _snap
from core.mistral7b_tc import QuantLinearTC, RMSNormTC
from core.graft_arena import ArenaCache
from tokenizers import Tokenizer as HFTok
import ast, re
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_graft_e4_conversation.py")).read()
SCRIPTED = ast.literal_eval(re.search(r"SCRIPTED = (\[.*?\n\])\n", _src, re.S).group(1))
PROBES = ast.literal_eval(re.search(r"PROBES = (\[.*?\n\])\n", _src, re.S).group(1))

QuantLinearTC.FUSED_DECODE = True
RMSNormTC.USE_FUSED = True
tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
m, info = MiniCPM3_TC.from_pretrained()
tc.set_alloc_pooling(True)
for L in m.layers:
    L.self_attn.absorbed_decode = True
print(f"loaded: {info} (fast stack)", flush=True)

arena = ArenaCache(m,
                   encode=lambda t: tok.encode(t).ids,
                   decode=lambda ids: tok.decode(ids),
                   sink_text="<conversation>\n",
                   arena_width=256, route_layer=44, topk=3, live_turns=2)

for i, (u, a) in enumerate(SCRIPTED):
    arena.feed(f"User: {u}\nAssistant: {a}\n")
print(f"fed 14 turns; repository = {len(arena.grafts)} turn grafts", flush=True)

print("\n=== librarian: consolidate turns 1-4 and 5-8 ===", flush=True)
for span in ([0, 1, 2, 3], [4, 5, 6, 7]):
    t0 = time.perf_counter()
    didx, text = arena.consolidate(span)
    qc = ArenaCache._digest_qc(text)
    print(f"digest g{didx + 1} <- turns {[s + 1 for s in span]} "
          f"({time.perf_counter() - t0:.1f}s, QC {'ok' if qc else 'DEGENERATE'})",
          flush=True)
    print(f"  {text[:110]!r}", flush=True)
active = [i for i, g in enumerate(arena.grafts) if not g.get("retired")]
print(f"routing pool after consolidation: {len(active)} nodes "
      f"(was {len(arena.grafts) - 2 + 2})", flush=True)

print("\n=== probes (needles live ONLY in digest nodes) ===", flush=True)
hits = 0
for q, acc in PROBES:
    t0 = time.perf_counter()
    ans, info_ = arena.step(q, ngen=48, deposit=False)
    dt = time.perf_counter() - t0
    ok = any(s in ans.lower() for s in acc)
    hits += ok
    kinds = ["D" if arena.grafts[i - 1].get("kind") == "digest" else "t"
             for i in info_["mounts"]]
    print(f"  [res={info_['resident']:3d} {dt:4.1f}s] "
          f"mounts={info_['mounts']}({''.join(kinds)}) "
          f"{'HIT ' if ok else 'MISS'} | {ans[:56]!r}", flush=True)
print(f"\nE4-C: {hits}/{len(PROBES)} through consolidated digests "
      f"(E4-arena direct was 6/6)", flush=True)
print("DONE", flush=True)
