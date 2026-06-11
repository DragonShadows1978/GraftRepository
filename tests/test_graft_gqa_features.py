"""GQA feature gates (Qwen3-4B, unified GQAArenaCache) — one process:

  A) HARVEST-ON-GENERATE + TRIPS: the E4-trips protocol with the default
     hybrid deposit (payload sliced from cache + un-RoPE'd FULL keys;
     routing key standalone). Starved topk=1 arena, max_trips 0 vs 2.
     References: MLA topk=1 5/6 -> +trips 6/6.
  B) E4-C CONSOLIDATION: turns 1-4 and 5-8 folded into two digests
     (fidelity-gated), sources retired, probes through digests alone.
     Reference: MLA 6/6. Open question under test: do the E4-C primed
     prefixes hold on a reasoning-tuned model, or does Qwen3 leak
     meta-text into digests (QC should catch degenerates)?
"""
import os, sys, time, numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc
from core.qwen3_tc import Qwen3_TC, _snap
from core.graft_arena import GQAArenaCache, ArenaCache
from tokenizers import Tokenizer as HFTok
import ast, re
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "test_graft_e4_conversation.py")).read()
SCRIPTED = ast.literal_eval(re.search(r"SCRIPTED = (\[.*?\n\])\n", _src, re.S).group(1))
PROBES = ast.literal_eval(re.search(r"PROBES = (\[.*?\n\])\n", _src, re.S).group(1))

tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
m, info = Qwen3_TC.from_pretrained()
print(f"loaded: {info}", flush=True)

ENC = lambda t: tok.encode(t).ids
DEC = lambda ids: tok.decode(ids)

# ---- A: hybrid deposits + starved arena + trips --------------------------
for trips in (0, 2):
    arena = GQAArenaCache(m, encode=ENC, decode=DEC,
                          sink_text="<conversation>\n",
                          arena_width=256, route_layer=0, topk=1,
                          live_turns=2)
    for u, a in SCRIPTED:
        arena.feed(f"User: {u}\nAssistant: {a}\n")
    print(f"\n=== A: topk=1, max_trips={trips} (hybrid deposits) ===",
          flush=True)
    hits = 0
    for q, acc in PROBES:
        t0 = time.perf_counter()
        ans, info_ = arena.step(q, ngen=48, deposit=False, max_trips=trips)
        dt = time.perf_counter() - t0
        ok = any(s in ans.lower() for s in acc)
        hits += ok
        print(f"  [trip {info_['trip']} {dt:4.1f}s] mounts={info_['mounts']} "
              f"{'HIT ' if ok else 'MISS'} | {ans[:56]!r}", flush=True)
    print(f"A TRIPS topk=1 trips={trips}: {hits}/{len(PROBES)} "
          f"(MLA: 5/6 -> 6/6)", flush=True)

# ---- B: consolidation through digests ------------------------------------
arena = GQAArenaCache(m, encode=ENC, decode=DEC,
                      sink_text="<conversation>\n",
                      arena_width=256, route_layer=0, topk=3, live_turns=2)
for u, a in SCRIPTED:
    arena.feed(f"User: {u}\nAssistant: {a}\n")
print("\n=== B: consolidate turns 1-4 and 5-8 ===", flush=True)
for span in ([0, 1, 2, 3], [4, 5, 6, 7]):
    t0 = time.perf_counter()
    didx, text = arena.consolidate(span)
    if didx is None:
        print(f"FOLD ABORTED for {span} (fidelity gate) — sources stay",
              flush=True)
        continue
    arena.grafts[didx]["kind"] = "digest"
    qc = ArenaCache._digest_qc(text)
    print(f"digest g{didx} <- turns {[s + 1 for s in span]} "
          f"({time.perf_counter() - t0:.1f}s, QC {'ok' if qc else 'DEGENERATE'})",
          flush=True)
    print(f"  {text[:110]!r}", flush=True)
active = [i for i, g in enumerate(arena.grafts) if not g.get("retired")]
print(f"routing pool after consolidation: {len(active)} nodes", flush=True)

print("\n=== B probes (needles only in digests if both folds held) ===",
      flush=True)
hits = 0
for q, acc in PROBES:
    t0 = time.perf_counter()
    ans, info_ = arena.step(q, ngen=48, deposit=False, max_trips=2)
    dt = time.perf_counter() - t0
    ok = any(s in ans.lower() for s in acc)
    hits += ok
    mk = [(i, arena.grafts[i - 1].get("kind", "t")[0]) for i in info_["mounts"]]
    print(f"  [trip {info_.get('trip', 0)} {dt:4.1f}s] mounts={mk} "
          f"{'HIT ' if ok else 'MISS'} | {ans[:56]!r}", flush=True)
print(f"B E4-C: {hits}/{len(PROBES)} (MLA: 6/6)", flush=True)
print("DONE", flush=True)
