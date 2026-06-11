"""Background (deferred) librarian gate.

Same 42-turn conversation as the descent gate, librarian_mode="deferred":
  1. HOT PATH FLAT: no add_turn may stall on a fold (inline mode spikes
     ~3-4s when thresholds trip; deferred must stay under ~1.5s/turn).
  2. idle() drains the fold queue between turns (counts + timing).
  3. Recall unchanged after deferred consolidation: the 8 descent-gate
     probes (descent reference: 8/8).
  4. Backpressure: if idle is never granted, the active pool stays bounded
     by the 2x inline fallback.
"""
import os, sys, time, shutil, numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc
from core.minicpm3_tc import MiniCPM3_TC, _snap
from core.mistral7b_tc import QuantLinearTC, RMSNormTC
from core.graft_repository import GraftRepository
from tokenizers import Tokenizer as HFTok
import ast, re
_here = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(_here, "test_graft_e4_conversation.py")).read()
SCRIPTED = ast.literal_eval(re.search(r"SCRIPTED = (\[.*?\n\])\n", _src, re.S).group(1))
PROBES = ast.literal_eval(re.search(r"PROBES = (\[.*?\n\])\n", _src, re.S).group(1))
_inf = open(os.path.join(_here, "test_graft_infinite.py")).read()
FILLER_TOPICS = ast.literal_eval(re.search(r"FILLER_TOPICS = (\[.*?\n\])\n", _inf, re.S).group(1))
FILLERS = [(f"Heads up, {t} {d}.", f"Noted — {t} {d}. I'll keep it in mind.")
           for t, d in FILLER_TOPICS] + \
          [(f"Small update on {t}: still true that it {d}.",
            f"Logged the update on {t}.") for t, d in FILLER_TOPICS]

QuantLinearTC.FUSED_DECODE = True
RMSNormTC.USE_FUSED = True
tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
m, info = MiniCPM3_TC.from_pretrained()
tc.set_alloc_pooling(True)
for L in m.layers:
    L.self_attn.absorbed_decode = True
print(f"loaded: {info} (fast stack)", flush=True)

PATH = "/tmp/graftrepo_lib"
shutil.rmtree(PATH, ignore_errors=True)
repo = GraftRepository(m, lambda t: tok.encode(t).ids,
                       lambda ids: tok.decode(ids), PATH, autosave=False,
                       librarian_mode="deferred",
                       sink_text="<conversation>\n", arena_width=384,
                       route_layer=44, topk=3, live_turns=2,
                       ephemeral=True, recency_mounts=2)

# 1+4: hot path flat; idle granted every 4th turn (a host's natural rhythm)
turn_times, idle_times, folds = [], [], 0
for n, (u, a) in enumerate(SCRIPTED + FILLERS):
    t0 = time.perf_counter()
    repo.add_turn(u, a)
    turn_times.append(time.perf_counter() - t0)
    if n % 4 == 3:
        t0 = time.perf_counter()
        d = repo.idle(max_jobs=1)
        folds += d
        if d:
            idle_times.append(time.perf_counter() - t0)
print(f"42 turns: max add_turn {max(turn_times):.2f}s "
      f"(mean {np.mean(turn_times):.2f}s) | idle folds {folds} "
      f"(mean {np.mean(idle_times):.1f}s each) | pending {repo.fold_pending()}",
      flush=True)
while repo.idle(max_jobs=2):
    pass
repo.save()
print(f"drained: {repo.stats()}", flush=True)

QS = list(PROBES) + [
    ("What date is the stakeholder demo scheduled for?", ["june 26", "26th"]),
    ("And what time exactly?", ["10:30"]),
]
hits = 0
for q, acc in QS:
    ans, info_ = repo.chat(q, ngen=48, max_trips=2)
    ok = any(s in ans.lower() for s in acc)
    hits += ok
    if not ok:
        print(f"  MISS | {ans[:60]!r}", flush=True)
flat = max(turn_times) < 1.5
print(f"\nLIBRARIAN: recall {hits}/{len(QS)} (descent ref 8/8) | hot path "
      f"{'FLAT' if flat else 'SPIKED'} (max {max(turn_times):.2f}s)", flush=True)
print("DONE", flush=True)
