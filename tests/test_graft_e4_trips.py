"""Shuttling gate (Phase-2 exit): overflow handled by TRIPS, not a bigger
arena. E4-arena conversation STARVED to topk=1 (one mount per attempt — the
right graft often loses rank-1), then the same starved arena with
max_trips=2 (grounding check between trips; failed attempts rolled back).

PASS: trips recover most starved misses — small arena + trips ~= topk=3.
Reference points: topk=3 arena = 6/6; baseline 6/6.
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

for trips in (0, 2):
    arena = ArenaCache(m,
                       encode=lambda t: tok.encode(t).ids,
                       decode=lambda ids: tok.decode(ids),
                       sink_text="<conversation>\n",
                       arena_width=256, route_layer=44, topk=1, live_turns=2)
    for u, a in SCRIPTED:
        arena.feed(f"User: {u}\nAssistant: {a}\n")
    print(f"\n=== topk=1, max_trips={trips} ===", flush=True)
    hits = 0
    for q, acc in PROBES:
        t0 = time.perf_counter()
        ans, info_ = arena.step(q, ngen=48, deposit=False, max_trips=trips)
        dt = time.perf_counter() - t0
        ok = any(s in ans.lower() for s in acc)
        hits += ok
        print(f"  [trip {info_['trip']} {dt:4.1f}s] mounts={info_['mounts']} "
              f"{'HIT ' if ok else 'MISS'} | {ans[:56]!r}", flush=True)
    print(f"topk=1 trips={trips}: {hits}/{len(PROBES)}", flush=True)
print("DONE", flush=True)
