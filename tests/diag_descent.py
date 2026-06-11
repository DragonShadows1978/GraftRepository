"""Instrumented descent diagnosis: RESUME the saved descent-gate repository
(/tmp/graftrepo_descent) and re-run the failing probes with per-attempt
prints: mount kinds, raw answer, grounding verdict + the content/have sets.
"""
import os, sys, numpy as np
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
                       lambda ids: tok.decode(ids), "/tmp/graftrepo_descent",
                       autosave=False, sink_text="<conversation>\n",
                       arena_width=384, route_layer=44, topk=3, live_turns=2,
                       ephemeral=True, recency_mounts=2)
print(f"resumed: {repo.stats()}", flush=True)

_att = ArenaCache._attempt
def att(self, user_text, picks, ngen, deposit, stops):
    txt, info = _att(self, user_text, picks, ngen, deposit, stops)
    desc = [f"{i}:{self.grafts[i].get('kind','turn')}" for i in picks]
    print(f"    attempt {desc} -> {txt[:64]!r}", flush=True)
    return txt, info
_grd = ArenaCache._grounded
def grd(self, ans, mount_idxs, question):
    v = _grd(self, ans, mount_idxs, question)
    content = ((self._rare_tokens(ans) | self._caps_tokens(ans))
               - self._rare_tokens(question)
               - self._caps_tokens(question, skip_sentence_initial=False))
    print(f"      grounded={v} content={sorted(content)}", flush=True)
    return v
ArenaCache._attempt = att
ArenaCache._grounded = grd

QS = [("What was the name of the backend hire?", ["priya", "raghunathan"]),
      ("Which rack unit is the build server in now?", ["bx-44", "bx44"]),
      ("What's our hardware budget for the quarter?", ["7,400", "7400"]),
      ("Where is the team offsite happening?", ["arrowhead"])]
for q, acc in QS:
    print(f"\nprobe: {q}", flush=True)
    ans, info_ = repo.chat(q, ngen=48, max_trips=2)
    ok = any(s in ans.lower() for s in acc)
    print(f"  => trip {info_['trip']} {'HIT' if ok else 'MISS'} | {ans[:60]!r}",
          flush=True)
print("DONE", flush=True)
