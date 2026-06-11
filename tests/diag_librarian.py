"""Diagnose the deferred-librarian recall dip: resume /tmp/graftrepo_lib,
print the two era texts + their child-digest texts, then re-run the three
failing probes with per-attempt mounts/answers/grounding.
"""
import os, sys
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc
from core.minicpm3_tc import MiniCPM3_TC, _snap
from core.mistral7b_tc import QuantLinearTC, RMSNormTC
from core.graft_arena import ArenaCache
from core.graft_repository import GraftRepository
from tokenizers import Tokenizer as HFTok

QuantLinearTC.FUSED_DECODE = True
RMSNormTC.USE_FUSED = True
tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
m, info = MiniCPM3_TC.from_pretrained()
tc.set_alloc_pooling(True)
for L in m.layers:
    L.self_attn.absorbed_decode = True
print(f"loaded: {info}", flush=True)

repo = GraftRepository(m, lambda t: tok.encode(t).ids,
                       lambda ids: tok.decode(ids), "/tmp/graftrepo_lib",
                       autosave=False, librarian_mode="deferred",
                       sink_text="<conversation>\n", arena_width=384,
                       route_layer=44, topk=3, live_turns=2,
                       ephemeral=True, recency_mounts=2)
print(f"resumed: {repo.stats()}", flush=True)
G = repo.arena.grafts
for i, g in enumerate(G):
    if g.get("kind") in ("era", "digest") and not g.get("retired"):
        kids = g.get("sources", [])
        print(f"\n[{i}] {g['kind']} <- {kids}: {g['text'][:130]!r}", flush=True)
        for k in kids:
            if G[k].get("kind") == "digest":
                print(f"    child[{k}] digest: {G[k]['text'][:110]!r}", flush=True)

_att = ArenaCache._attempt
def att(self, user_text, picks, ngen, deposit, stops):
    txt, info = _att(self, user_text, picks, ngen, deposit, stops)
    print(f"    {[(i, self.grafts[i].get('kind','turn')) for i in picks]} "
          f"-> {txt[:60]!r}", flush=True)
    return txt, info
ArenaCache._attempt = att

for q, acc in [("Which rack unit is the build server in now?", "bx-44"),
               ("What's our hardware budget for the quarter?", "7,400"),
               ("Where is the team offsite happening?", "arrowhead")]:
    print(f"\nprobe: {q}", flush=True)
    ans, info_ = repo.chat(q, ngen=48, max_trips=2)
    print(f"  => {'HIT' if acc in ans.lower() else 'MISS'} | {ans[:60]!r}",
          flush=True)
print("DONE", flush=True)
