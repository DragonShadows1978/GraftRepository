"""CROSS-MODEL MIGRATION gate: a repository built on MiniCPM3 (MLA) is
rebuilt on Qwen3-4B (GQA) from its TEXTS — the dialect wall holds for K/V,
text crosses it.

  python3 test_graft_migrate.py build     # MiniCPM3: 14 turns + 2 docs
  python3 test_graft_migrate.py migrate   # Qwen3: wall check, migrate, probe

The migrate phase first verifies the DIALECT WALL still refuses a direct
load of the MLA artifacts, then migrates (every node re-harvested under
Qwen3's weights, digest texts carried verbatim, lineage preserved) and
runs the resume-gate probes on the migrated repository.

Reference: MLA resume gate 7/7 on the same content.
"""
import os, sys, time, shutil
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc
from core.graft_repository import GraftRepository
from tokenizers import Tokenizer as HFTok
import ast, re
_here = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(_here, "test_graft_e4_conversation.py")).read()
SCRIPTED = ast.literal_eval(re.search(r"SCRIPTED = (\[.*?\n\])\n", _src, re.S).group(1))
PROBES = ast.literal_eval(re.search(r"PROBES = (\[.*?\n\])\n", _src, re.S).group(1))

SRC, DST = "/tmp/graftrepo_mig_src", "/tmp/graftrepo_mig_dst"
DOCS = [("SECURITY ADVISORY SA-2031-07. The legacy ingest daemon must be "
         "disabled before the July audit; its replacement listens on port "
         "7443 with mutual TLS.", ("security",)),
        ("RUNBOOK NOTE. The analytics warehouse rebuild takes 90 minutes "
         "and must start after the 02:00 backup completes.", ("ops",))]

if sys.argv[1] == "build":
    from core.minicpm3_tc import MiniCPM3_TC, _snap
    from core.mistral7b_tc import QuantLinearTC, RMSNormTC
    QuantLinearTC.FUSED_DECODE = True
    RMSNormTC.USE_FUSED = True
    tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
    m, info = MiniCPM3_TC.from_pretrained()
    tc.set_alloc_pooling(True)
    for L in m.layers:
        L.self_attn.absorbed_decode = True
    print(f"loaded: {info}", flush=True)
    shutil.rmtree(SRC, ignore_errors=True)
    repo = GraftRepository(m, lambda t: tok.encode(t).ids,
                           lambda ids: tok.decode(ids), SRC,
                           sink_text="<conversation>\n", arena_width=384,
                           route_layer=44, topk=3, live_turns=2)
    for u, a in SCRIPTED:
        repo.add_turn(u, a)
    for text, tags in DOCS:
        repo.add_document(text, tags)
    repo.save()
    print(f"built (MLA): {repo.stats()}", flush=True)
else:
    from core.qwen3_tc import Qwen3_TC, _snap
    from core.graft_arena import GQAArenaCache
    tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
    m, info = Qwen3_TC.from_pretrained()
    print(f"loaded: {info}", flush=True)
    enc = lambda t: tok.encode(t).ids
    dec = lambda ids: tok.decode(ids)

    # 1) the wall: a direct open of MLA artifacts on Qwen3 must refuse
    try:
        GraftRepository(m, enc, dec, SRC, arena_cls=GQAArenaCache,
                        sink_text="<conversation>\n", arena_width=384,
                        route_layer=0, topk=3, live_turns=2)
        print("WALL BREACHED — direct load of MLA artifacts succeeded",
              flush=True)
        sys.exit(1)
    except RuntimeError as e:
        print(f"dialect wall holds: {str(e)[:90]!r}", flush=True)

    # 2) migrate: rebuild from texts under Qwen3's weights
    shutil.rmtree(DST, ignore_errors=True)
    repo = GraftRepository(m, enc, dec, DST, arena_cls=GQAArenaCache,
                           sink_text="<conversation>\n", arena_width=384,
                           route_layer=0, topk=3, live_turns=2)
    t0 = time.perf_counter()
    n = repo.migrate(SRC)
    print(f"migrated {n} nodes in {time.perf_counter()-t0:.0f}s: "
          f"{repo.stats()}", flush=True)

    # 3) the resume-gate probes on the MIGRATED repository
    QS = list(PROBES) + [("What port does the ingest replacement listen on?",
                          ["7443"])]
    hits = 0
    for q, acc in QS:
        t0 = time.perf_counter()
        ans, info_ = repo.chat(q, ngen=48, max_trips=2)
        ok = any(s in ans.lower() for s in acc)
        hits += ok
        print(f"  [trip {info_.get('trip', 0)} "
              f"{time.perf_counter()-t0:4.1f}s] {'HIT ' if ok else 'MISS'} "
              f"| {ans[:56]!r}", flush=True)
    print(f"\nMIGRATE MLA->GQA: {hits}/{len(QS)} (MLA resume gate: 7/7)",
          flush=True)
print("DONE", flush=True)
