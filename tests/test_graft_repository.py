"""GraftRepository gate: build session -> save -> FRESH PROCESS -> resume ->
recall. Usage: repo_gate.py build | resume

build:  feed the E4 conversation through add_turn (auto-librarian fires at
        the 8-active-turn threshold -> digests appear mid-session), add two
        reference documents, save. Print stats + digest texts.
resume: new process loads the repository (dialect-guarded), probes the six
        E4 needles + one document fact via chat() with trips. Memory must
        come entirely from reloaded disk artifacts.
"""
import os, sys, time, numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc
from core.minicpm3_tc import MiniCPM3_TC, _snap
from core.mistral7b_tc import QuantLinearTC, RMSNormTC
from core.graft_repository import GraftRepository
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
print(f"loaded: {info}", flush=True)

PATH = "/tmp/graftrepo_test"
DOCS = [("SECURITY ADVISORY SA-2031-07. The legacy ingest daemon must be "
         "disabled before the July audit; its replacement listens on port "
         "7443 with mutual TLS.", ("security",)),
        ("RUNBOOK NOTE. The analytics warehouse rebuild takes 90 minutes "
         "and must start after the 02:00 backup completes.", ("ops",))]

repo = GraftRepository(m, lambda t: tok.encode(t).ids,
                       lambda ids: tok.decode(ids), PATH,
                       sink_text="<conversation>\n", arena_width=384,
                       route_layer=44, topk=3, live_turns=2)

if sys.argv[1] == "build":
    for u, a in SCRIPTED:
        repo.add_turn(u, a)
    for text, tags in DOCS:
        repo.add_document(text, tags)
    repo.save()
    print(f"stats: {repo.stats()}", flush=True)
    for g in repo.arena.grafts:
        if g.get("kind") in ("digest", "era"):
            print(f"  {g['kind']}: {g['text'][:100]!r}", flush=True)
    sz = sum(os.path.getsize(os.path.join(dp, f))
             for dp, _, fs in os.walk(PATH) for f in fs)
    print(f"repository on disk: {sz / 1e6:.1f} MB", flush=True)
else:
    print(f"resumed: {repo.stats()}", flush=True)
    qs = list(PROBES) + [("What port does the ingest daemon replacement "
                          "listen on?", ["7443"])]
    hits = 0
    for q, acc in qs:
        t0 = time.perf_counter()
        ans, info_ = repo.chat(q, ngen=48, max_trips=2)
        ok = any(s in ans.lower() for s in acc)
        hits += ok
        print(f"  [trip {info_['trip']} {time.perf_counter()-t0:4.1f}s] "
              f"{'HIT ' if ok else 'MISS'} | {ans[:58]!r}", flush=True)
    print(f"\nRESUME RECALL: {hits}/{len(qs)}", flush=True)
print("DONE", flush=True)
