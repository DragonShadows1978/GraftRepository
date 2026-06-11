"""VRAM paging gate: 100 corpus documents through GraftRepository with a
deliberately tiny device budget (64MB ~= 17 nodes' worth) — the pager must
spill least-recently-mounted nodes to cold storage and the loader must
page them back on mount, with recall unchanged (corpus-100 reference:
20/20) and bounded device residency.
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
_src = open(os.path.join(_here, "test_graft_corpus100.py")).read()
exec(re.search(r"(FAMILIES = \[.*?\n\])\n", _src, re.S).group(1))

QuantLinearTC.FUSED_DECODE = True
RMSNormTC.USE_FUSED = True
tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
m, info = MiniCPM3_TC.from_pretrained()
tc.set_alloc_pooling(True)
for L in m.layers:
    L.self_attn.absorbed_decode = True
print(f"loaded: {info} (fast stack)", flush=True)

PATH = "/tmp/graftrepo_paging"
shutil.rmtree(PATH, ignore_errors=True)
repo = GraftRepository(m, lambda t: tok.encode(t).ids,
                       lambda ids: tok.decode(ids), PATH, autosave=False,
                       vram_budget_mb=64,
                       sink_text="<conversation>\n", arena_width=384,
                       route_layer=44, topk=3, live_turns=2,
                       ephemeral=True, recency_mounts=1)

t0 = time.perf_counter()
probes = []
for f, (fam, mk_text, mk_probe, mk_vals) in enumerate(FAMILIES):
    for i in range(10):
        code, val = mk_vals(i)
        repo.add_document(mk_text(code, val), tags=(fam,))
    for i in ((3 * f + 1) % 10, (7 * f + 4) % 10):
        code, val = mk_vals(i)
        probes.append((mk_probe(code), val.lower()))
repo.save()
repo._page()
print(f"100 docs in {time.perf_counter()-t0:.0f}s; {repo.stats()}", flush=True)

print("\n=== probes (64MB device budget) ===", flush=True)
hits = 0
times = []
for q, acc in probes:
    t0 = time.perf_counter()
    ans, info_ = repo.chat(q, ngen=40, max_trips=2)
    dt = time.perf_counter() - t0
    times.append(dt)
    ok = acc in ans.lower()
    hits += ok
    if not ok:
        print(f"  MISS | {ans[:60]!r}", flush=True)
print(f"\nPAGING: {hits}/20 (corpus-100 unbudgeted: 20/20) | "
      f"median {sorted(times)[len(times)//2]:.1f}s/probe | {repo.stats()}",
      flush=True)
print("DONE", flush=True)
