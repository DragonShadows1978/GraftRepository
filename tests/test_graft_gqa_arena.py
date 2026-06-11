"""GQA ARENA (Qwen3-4B) — round 2: the UNIFIED dialect class.

Round 1 (6/6) proved the surgery with a minimal in-test arena. This round
re-runs the same protocol through core.graft_arena.GQAArenaCache — the
full ArenaCache (ladder, grounding, consolidation machinery) with the GQA
dialect surface: (k, v) payload both seq dim=2, full-key re-RoPE at arena
seats, layer-0 unit-normalized |q.k| router (route_layer=0 is part of the
dialect). cache_deposits=False matches round 1's standalone deposits —
harvest-on-generate on GQA is a separate increment.

Target: 6/6 = round 1 = the MLA arena gate.
"""
import os, sys, time, numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc
from core.qwen3_tc import Qwen3_TC, _snap
from core.graft_arena import GQAArenaCache
from tokenizers import Tokenizer as HFTok
import ast, re
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "test_graft_e4_conversation.py")).read()
SCRIPTED = ast.literal_eval(re.search(r"SCRIPTED = (\[.*?\n\])\n", _src, re.S).group(1))
PROBES = ast.literal_eval(re.search(r"PROBES = (\[.*?\n\])\n", _src, re.S).group(1))

tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
m, info = Qwen3_TC.from_pretrained()
for L in m.layers:
    L.self_attn.quant_kv_cache = False
print(f"loaded: {info}", flush=True)

arena = GQAArenaCache(m,
                      encode=lambda t: tok.encode(t).ids,
                      decode=lambda ids: tok.decode(ids),
                      sink_text="<conversation>\n",
                      arena_width=256, route_layer=0, topk=3, live_turns=2,
                      cache_deposits=False)
print(f"arena: sink={arena.n_sink} seats, width={arena.width}, "
      f"live_shift={arena.live_shift}", flush=True)

for i, (u, a) in enumerate(SCRIPTED):
    t0 = time.perf_counter()
    arena.feed(f"User: {u}\nAssistant: {a}\n")
    S = arena._cache_len()
    print(f"turn {i+1:2d} fed+deposited  resident={S:3d}  "
          f"({time.perf_counter()-t0:.2f}s)", flush=True)

print("\n=== probes (unified GQAArenaCache, one persistent cache) ===",
      flush=True)
hits = 0
for q, acc in PROBES:
    t0 = time.perf_counter()
    ans, info_ = arena.step(q, ngen=48, deposit=False)
    dt = time.perf_counter() - t0
    ok = any(s in ans.lower() for s in acc)
    hits += ok
    print(f"  [res={info_['resident']:3d} live={info_['live_tokens']:3d} "
          f"evict={info_['evicted']:3d} {dt:4.1f}s] mounts={info_['mounts']} "
          f"{'HIT ' if ok else 'MISS'} | {ans[:58]!r}", flush=True)
print(f"\nGQA-ARENA-UNIFIED: {hits}/{len(PROBES)} (round 1 in-test arena "
      f"was 6/6; MLA arena gate 6/6)", flush=True)
print("DONE", flush=True)
