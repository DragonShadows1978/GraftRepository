"""GQA GraftRepository gate (Qwen3-4B): build a repository in one process,
resume it in a FRESH process — the cross-session persistence proof on the
GQA dialect (nodes = (L,H,S,D) fp16 k/v; index = per-node rkey arrays;
dialect string Qwen3_TC:36x2560:g8x128; route_layer 0).

  python3 test_graft_gqa_repository.py build
  python3 test_graft_gqa_repository.py resume

Reference: the MLA repository gate resumes 7/7 from disk artifacts alone.
"""
import os, sys, time, shutil
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc
from core.qwen3_tc import Qwen3_TC, _snap
from core.graft_arena import GQAArenaCache
from core.graft_repository import GraftRepository
from tokenizers import Tokenizer as HFTok
import ast, re
_here = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(_here, "test_graft_e4_conversation.py")).read()
SCRIPTED = ast.literal_eval(re.search(r"SCRIPTED = (\[.*?\n\])\n", _src, re.S).group(1))
PROBES = ast.literal_eval(re.search(r"PROBES = (\[.*?\n\])\n", _src, re.S).group(1))

tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
m, info = Qwen3_TC.from_pretrained()
print(f"loaded: {info}", flush=True)

PATH = "/tmp/graftrepo_gqa"
DOCS = [("SECURITY ADVISORY SA-2031-07. The legacy ingest daemon must be "
         "disabled before the July audit; its replacement listens on port "
         "7443 with mutual TLS.", ("security",)),
        ("RUNBOOK NOTE. The analytics warehouse rebuild takes 90 minutes "
         "and must start after the 02:00 backup completes.", ("ops",))]

repo = GraftRepository(m, lambda t: tok.encode(t).ids,
                       lambda ids: tok.decode(ids), PATH,
                       arena_cls=GQAArenaCache,
                       sink_text="<conversation>\n", arena_width=384,
                       route_layer=0, topk=3, live_turns=2)

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
    QS = list(PROBES) + [("What port does the ingest replacement listen on?",
                          ["7443"])]
    hits = 0
    for q, acc in QS:
        t0 = time.perf_counter()
        ans, info_ = repo.chat(q, ngen=48, max_trips=2)
        ok = any(s in ans.lower() for s in acc)
        hits += ok
        print(f"  [trip {info_.get('trip', 0)} "
              f"{time.perf_counter() - t0:4.1f}s] "
              f"{'HIT ' if ok else 'MISS'} | {ans[:56]!r}", flush=True)
    print(f"\nGQA RESUME RECALL: {hits}/{len(QS)} (MLA resume gate: 7/7)",
          flush=True)
print("DONE", flush=True)
