"""INFINITE-CONTEXT gate: ephemeral boat ("clear the memory window at the
start of each turn"). Every turn runs on [sink | mounts | turn] alone —
resident seats CONSTANT for any conversation length; history exists only as
repository nodes; recency is a MOUNT (last 2 turn-grafts) for anaphora.

42-turn conversation (E4's 14 + 28 fillers) through GraftRepository with
the auto-librarian chaining turn -> digest -> ERA (first depth exercise:
the early facts end up under an era node — recall through DOUBLE
consolidation). Probes: the six E4 needles + an anaphora pair. Measures:
recall, resident seats per probe (bounded), node census incl. eras.
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
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_graft_e4_conversation.py")).read()
SCRIPTED = ast.literal_eval(re.search(r"SCRIPTED = (\[.*?\n\])\n", _src, re.S).group(1))
PROBES = ast.literal_eval(re.search(r"PROBES = (\[.*?\n\])\n", _src, re.S).group(1))

FILLER_TOPICS = [
    ("the elevator inspection", "passed with one note about door timing"),
    ("the quarterly fire drill", "ran four minutes faster than last time"),
    ("the lobby plants", "got replaced with something less thirsty"),
    ("the team trivia night", "ended in a tie nobody wanted to break"),
    ("the new badge readers", "beep twice now for visitors"),
    ("the cafeteria soup rotation", "added a decent lentil option"),
    ("the bike rack", "moved closer to the south entrance"),
    ("the printer on four", "finally got its drum replaced"),
    ("the window washers", "are scheduled for Thursday morning"),
    ("the standing desks", "arrived for the third-floor pod"),
    ("the guest wifi", "got a captive portal nobody likes"),
    ("the recycling bins", "were re-labeled for the audit"),
    ("the air conditioning", "is being rebalanced zone by zone"),
    ("the parking validation", "switched to QR codes"),
]
FILLERS = [(f"Heads up, {t} {d}.",
            f"Noted — {t} {d}. I'll keep it in mind.")
           for t, d in FILLER_TOPICS] + \
          [(f"Small update on {t}: still true that it {d}.",
            f"Logged the update on {t}.")
           for t, d in FILLER_TOPICS]

QuantLinearTC.FUSED_DECODE = True
RMSNormTC.USE_FUSED = True
tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
m, info = MiniCPM3_TC.from_pretrained()
tc.set_alloc_pooling(True)
for L in m.layers:
    L.self_attn.absorbed_decode = True
print(f"loaded: {info} (fast stack)", flush=True)

PATH = "/tmp/graftrepo_inf"
shutil.rmtree(PATH, ignore_errors=True)
repo = GraftRepository(m, lambda t: tok.encode(t).ids,
                       lambda ids: tok.decode(ids), PATH, autosave=False,
                       sink_text="<conversation>\n", arena_width=384,
                       route_layer=44, topk=3, live_turns=2,
                       ephemeral=True, recency_mounts=2)

t0 = time.perf_counter()
for u, a in SCRIPTED + FILLERS:
    repo.add_turn(u, a)
print(f"42 turns deposited+consolidated in {time.perf_counter()-t0:.0f}s; "
      f"stats: {repo.stats()}", flush=True)
for g in repo.arena.grafts:
    if g.get("kind") == "era":
        print(f"  ERA: {g['text'][:110]!r}", flush=True)

QS = list(PROBES) + [
    ("What date is the stakeholder demo scheduled for?", ["june 26", "26th"]),
    ("And what time exactly?", ["10:30"]),          # anaphora -> recency mount
]
print("\n=== probes (ephemeral boat, constant residency) ===", flush=True)
hits, max_res = 0, 0
for q, acc in QS:
    t0 = time.perf_counter()
    ans, info_ = repo.chat(q, ngen=48, max_trips=2)
    ok = any(s in ans.lower() for s in acc)
    hits += ok
    max_res = max(max_res, info_["resident"])
    print(f"  [res={info_['resident']:3d} trip {info_['trip']} "
          f"{time.perf_counter()-t0:4.1f}s] {'HIT ' if ok else 'MISS'} "
          f"| {ans[:54]!r}", flush=True)
repo.save()
sz = sum(os.path.getsize(os.path.join(dp, f))
         for dp, _, fs in os.walk(PATH) for f in fs)
print(f"\nINFINITE: {hits}/{len(QS)} | max resident {max_res} seats "
      f"(42-turn history; baseline transcript would be ~2,300+ tokens and "
      f"growing) | repo {sz/1e6:.1f}MB | {repo.stats()}", flush=True)
print("DONE", flush=True)
