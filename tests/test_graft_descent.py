"""DESCENT gate: era folding ON + descent trips. The 42-turn infinite
scenario where era texts are UNRELIABLE readers (lists strip relations,
prose invents them — both measured): the router may land on an era node,
but when the read fails grounding the next trip re-mounts the era's CHILD
DIGESTS (identifier-filtered when the probe names codes; cold-storage
reload via node_loader if their VRAM was reclaimed).

PASS: recall through era-folded history recovers to ~E4-C levels (the
infinite gate scored 4/8 list-era and 3/8 prose-era WITHOUT descent).
"""
import os, sys, time, numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc
from core.minicpm3_tc import MiniCPM3_TC, _snap
from core.mistral7b_tc import QuantLinearTC, RMSNormTC
from core.graft_repository import GraftRepository
from tokenizers import Tokenizer as HFTok
import ast, re, shutil
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

PATH = "/tmp/graftrepo_descent"
shutil.rmtree(PATH, ignore_errors=True)
repo = GraftRepository(m, lambda t: tok.encode(t).ids,
                       lambda ids: tok.decode(ids), PATH, autosave=False,
                       sink_text="<conversation>\n", arena_width=384,
                       route_layer=44, topk=3, live_turns=2,
                       ephemeral=True, recency_mounts=2)
repo.DIGESTS_HIGH, repo.DIGESTS_FOLD = 6, 3      # era folding ON for this gate

t0 = time.perf_counter()
for u, a in SCRIPTED + FILLERS:
    repo.add_turn(u, a)
repo.save()      # cold storage in place so descent can reload freed nodes
print(f"42 turns in {time.perf_counter()-t0:.0f}s; stats: {repo.stats()}", flush=True)
for g in repo.arena.grafts:
    if g.get("kind") == "era":
        print(f"  ERA: {g['text'][:100]!r}", flush=True)

QS = list(PROBES) + [
    ("What date is the stakeholder demo scheduled for?", ["june 26", "26th"]),
    ("And what time exactly?", ["10:30"]),
]
print("\n=== probes (era-folded history, descent trips) ===", flush=True)
hits = 0
for q, acc in QS:
    t0 = time.perf_counter()
    ans, info_ = repo.chat(q, ngen=48, max_trips=2)
    ok = any(s in ans.lower() for s in acc)
    hits += ok
    print(f"  [res={info_['resident']:3d} trip {info_['trip']} "
          f"{time.perf_counter()-t0:4.1f}s] {'HIT ' if ok else 'MISS'} "
          f"| {ans[:54]!r}", flush=True)
print(f"\nDESCENT: {hits}/{len(QS)} (without descent: 4/8 list-era, "
      f"3/8 prose-era) | {repo.stats()}", flush=True)
print("DONE", flush=True)
