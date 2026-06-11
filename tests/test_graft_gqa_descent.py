"""GQA DESCENT gate (Qwen3-4B): the 42-turn ephemeral ("infinite context")
scenario with the full librarian — turn folds, fidelity gating, era folding
ON, descent trips — through the unified GQAArenaCache + GraftRepository.

This is the flagship end-to-end behavior on the second dialect: ephemeral
boat (constant residency), auto-consolidation with the fidelity gate,
hierarchical descent on grounding failure, anaphora through recency mounts.

Reference: MLA descent gate 8/8 (4/8 list-era / 3/8 prose-era without
descent); MLA infinite gate 8/8.
"""
import os, sys, time, numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc
from core.qwen3_tc import Qwen3_TC, _snap
from core.graft_arena import GQAArenaCache
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

tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
m, info = Qwen3_TC.from_pretrained()
print(f"loaded: {info}", flush=True)

PATH = "/tmp/graftrepo_gqa_descent"
shutil.rmtree(PATH, ignore_errors=True)
repo = GraftRepository(m, lambda t: tok.encode(t).ids,
                       lambda ids: tok.decode(ids), PATH, autosave=False,
                       arena_cls=GQAArenaCache,
                       sink_text="<conversation>\n", arena_width=384,
                       route_layer=0, topk=3, live_turns=2,
                       ephemeral=True, recency_mounts=2)

t0 = time.perf_counter()
resmax = 0
for u, a in SCRIPTED + FILLERS:
    repo.add_turn(u, a)
repo.save()      # cold storage in place so descent can reload freed nodes
print(f"42 turns in {time.perf_counter()-t0:.0f}s; stats: {repo.stats()}",
      flush=True)
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
    resmax = max(resmax, info_["resident"])
    print(f"  [res={info_['resident']:3d} trip {info_['trip']} "
          f"{time.perf_counter()-t0:4.1f}s] {'HIT ' if ok else 'MISS'} "
          f"| {ans[:54]!r}", flush=True)
print(f"\nGQA-DESCENT: {hits}/{len(QS)} (MLA descent gate: 8/8) | "
      f"max resident {resmax} | {repo.stats()}", flush=True)
print("DONE", flush=True)
