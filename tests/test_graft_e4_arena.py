"""E4-ARENA: the E4 conversation run through a PERSISTENT cache with the
Graft Arena — no per-turn re-prefill anywhere. Seating [SINK | ARENA | LIVE]:
scripted turns feed through the live region and are deposited as latent
grafts; per probe the arena swaps to the routed top-3 by cache surgery; old
live turns are EVICTED once outside the 2-turn recency window.

Protocol matches E4 v3 (L44 latent router, bare-question routing inside
ArenaCache.route via step(), retrieval turns not deposited). Targets:
  recall parity with the re-prefill harness (6/6), bounded residency,
  coherent generation across 6 arena swaps + ~20 evictions on ONE cache.
"""
import os, sys, time, numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc
from core.minicpm3_tc import MiniCPM3_TC, _snap
from core.graft_arena import ArenaCache
from tokenizers import Tokenizer as HFTok
import ast, re
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_graft_e4_conversation.py")).read()
SCRIPTED = ast.literal_eval(re.search(r"SCRIPTED = (\[.*?\n\])\n", _src, re.S).group(1))
PROBES = ast.literal_eval(re.search(r"PROBES = (\[.*?\n\])\n", _src, re.S).group(1))

from core.mistral7b_tc import QuantLinearTC, RMSNormTC
QuantLinearTC.FUSED_DECODE = True     # INT4 GEMV at decode shapes (gated)
RMSNormTC.USE_FUSED = True            # fused rms_norm (gated)
tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
m, info = MiniCPM3_TC.from_pretrained()
tc.set_alloc_pooling(True)            # transients pool (gated)
for L in m.layers:
    L.self_attn.absorbed_decode = True  # latent-space decode (gated)
print(f"loaded: {info} (fast stack on)", flush=True)

arena = ArenaCache(m,
                   encode=lambda t: tok.encode(t).ids,
                   decode=lambda ids: tok.decode(ids),
                   sink_text="<conversation>\n",
                   arena_width=256, route_layer=44, topk=3, live_turns=2)
print(f"arena: sink={arena.n_sink} seats, width={arena.width}, "
      f"live_shift={arena.live_shift}", flush=True)

for i, (u, a) in enumerate(SCRIPTED):
    t0 = time.perf_counter()
    arena.feed(f"User: {u}\nAssistant: {a}\n")
    S = arena.caches[0][0].shape[1]
    print(f"turn {i+1:2d} fed+deposited  resident={S:3d}  "
          f"({time.perf_counter()-t0:.2f}s)", flush=True)

print("\n=== probes (turns 15-20, one persistent cache) ===", flush=True)
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
print(f"\nE4-ARENA: {hits}/{len(PROBES)} (re-prefill harness was 6/6; "
      f"baseline 6/6 at 670-858 ctx)", flush=True)
print("DONE", flush=True)
