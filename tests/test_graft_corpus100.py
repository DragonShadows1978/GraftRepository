"""Corpus-scale routing stress (the style-guide-corpus regime).

100 chunks = 10 template families x 10 SIBLING instances (same topic, ~same
wording, different identifier codes + attribute values) — confusability is
the point: the router must find THE instance named in the probe among 9
near-duplicates, in a 100-node pool. 20 code-keyed probes (2/family).

Measures: router rank of correct chunk (recall@1/@3 over the full pool),
end recall with topk=3 + max_trips=2 grounded shuttling, per-probe wall
time, device-graft VRAM. All deterministic (index-derived values).
"""
import os, sys, time, subprocess, numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc
from core.minicpm3_tc import MiniCPM3_TC, _snap
from core.mistral7b_tc import QuantLinearTC, RMSNormTC
from core.graft_arena import ArenaCache
from tokenizers import Tokenizer as HFTok

QuantLinearTC.FUSED_DECODE = True
RMSNormTC.USE_FUSED = True
tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
m, info = MiniCPM3_TC.from_pretrained()
tc.set_alloc_pooling(True)
for L in m.layers:
    L.self_attn.absorbed_decode = True
print(f"loaded: {info} (fast stack)", flush=True)

def vram():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits", "-i", "0"],
                       capture_output=True, text=True)
    return int(o.stdout.strip().splitlines()[0])

# ---- 10 families x 10 instances, deterministic values
FAMILIES = [
    ("apiary", lambda c, v: (f"APIARY INSPECTION. The new queen in colony "
        f"{c} is marked and laying well across {v} frames. Mite counts low; "
        f"re-inspect in fourteen days."),
     lambda c: f"How many frames is the queen in colony {c} laying across?",
     lambda i: (f"QB-{113 + 7 * i}", str(4 + i))),
    ("sonde", lambda c, v: (f"UPPER-AIR SOUNDING. Radiosonde callsign {c} "
        f"launched on schedule; burst altitude {v} km, strong shear near "
        f"the tropopause. Recovery team dispatched."),
     lambda c: f"What was the burst altitude of radiosonde {c}?",
     lambda i: (f"WHISKEY-{21 + 3 * i}", f"{26 + i}.{(2 * i + 1) % 10}")),
    ("geothermal", lambda c, v: (f"GEOTHERMAL SHIFT REPORT. Well {c} is "
        f"delivering {v} megawatts after the scaling washout. Brine "
        f"pressure steady; reinjection pump online."),
     lambda c: f"How many megawatts is well {c} delivering?",
     lambda i: (f"GT-{3 + 2 * i}", f"{2 + i}.{(3 * i + 2) % 10}")),
    ("ticket", lambda c, v: (f"BUG TRIAGE NOTE. Filed as ticket {c}: crash "
        f"in the {v} parser when the includes array is empty. Severity "
        f"high, assigned to the runtime team."),
     lambda c: f"Which parser is crashing according to ticket {c}?",
     lambda i: (f"JIRA-{8810 + 17 * i}",
                ["manifest", "yaml", "config", "schema", "lexer", "header",
                 "macro", "route", "query", "token"][i])),
    ("vessel", lambda c, v: (f"SHIPPING MANIFEST. The cargo vessel {c} "
        f"departed Valparaiso carrying {v} crates of citrus for Rotterdam. "
        f"Panama transit booked."),
     lambda c: f"How many crates is the vessel {c} carrying?",
     lambda i: (f"SEALARK-{5 + i}", f"{3150 + 220 * i}")),
    ("depot", lambda c, v: (f"ANTARCTIC TRAVERSE PLAN. Depot cache {c} was "
        f"laid at {v} degrees south, flagged at double height. Contains "
        f"fuel and rations."),
     lambda c: f"At what latitude south was depot cache {c} laid?",
     lambda i: (f"GAMMA-{11 + 2 * i}", str(72 + i))),
    ("yeast", lambda c, v: (f"BREWHOUSE BATCH SHEET. Pitched yeast strain "
        f"{c} at {v} degrees, free rise planned. Gravity on schedule at "
        f"knockout."),
     lambda c: f"At what temperature was yeast strain {c} pitched?",
     lambda i: (f"WLP-{540 + 13 * i}", str(16 + i))),
    ("loco", lambda c, v: (f"FREIGHT DESPATCH WIRE. Locomotive {c} departed "
        f"the junction hauling {v} wagons of bauxite for the smelter spur. "
        f"Crew change at the river yard."),
     lambda c: f"How many wagons is locomotive {c} hauling?",
     lambda i: (str(4400 + 31 * i), str(48 + 3 * i))),
    ("psalter", lambda c, v: (f"LIBRARY ARCHIVE MEMO. The psalter under "
        f"shelfmark {c} moved to climate storage; {v} leaves show tide "
        f"damage. Digitization queued."),
     lambda c: f"How many damaged leaves does the psalter at shelfmark {c} have?",
     lambda i: (f"MS Vellum {200 + 9 * i}", str(2 + i))),
    ("reef", lambda c, v: (f"REEF SURVEY NOTE. Transect {c} resurveyed at "
        f"low tide: bleaching at {v} percent of colonies. Crown-of-thorns "
        f"count zero."),
     lambda c: f"What percentage of colonies are bleached on transect {c}?",
     lambda i: (f"T-{4 + 5 * i}", str(11 + 2 * i))),
]

arena = ArenaCache(m,
                   encode=lambda t: tok.encode(t).ids,
                   decode=lambda ids: tok.decode(ids),
                   sink_text="<conversation>\n",
                   arena_width=384, route_layer=44, topk=3, live_turns=2)

v0 = vram()
t0 = time.perf_counter()
meta = []                      # (family, code, value, graft_idx)
for fam, mk_text, mk_probe, mk_vals in FAMILIES:
    for i in range(10):
        code, val = mk_vals(i)
        gi = arena.deposit(mk_text(code, val))
        meta.append((fam, code, val, gi))
t_h = time.perf_counter() - t0
print(f"deposited 100 chunks in {t_h:.1f}s; device grafts +{vram() - v0}MB; "
      f"index = {len(arena.grafts) * 256 * 2} bytes", flush=True)

# ---- routing accuracy over the full pool (2 probes per family)
print("\n=== routing (100-node pool, sibling confusability) ===", flush=True)
r1 = r3 = 0
probes = []
for f, (fam, mk_text, mk_probe, mk_vals) in enumerate(FAMILIES):
    for i in ((3 * f + 1) % 10, (7 * f + 4) % 10):
        code, val = mk_vals(i)
        gi = next(g for ff, cc, vv, g in meta if ff == fam and cc == code)
        order = arena.route(mk_probe(code), exclude=set())
        rank = order.index(gi) + 1
        r1 += rank == 1
        r3 += rank <= 3
        probes.append((mk_probe(code), [val.lower()], gi, rank))
print(f"router: recall@1 {r1}/20  recall@3 {r3}/20", flush=True)

# ---- end-to-end with trips
print("\n=== probes (topk=3, max_trips=2) ===", flush=True)
hits = 0
times = []
for q, acc, gi, rank in probes:
    t0 = time.perf_counter()
    ans, info_ = arena.step(q, ngen=40, deposit=False, max_trips=2)
    dt = time.perf_counter() - t0
    times.append(dt)
    ok = any(s in ans.lower() for s in acc)
    hits += ok
    flag = "" if rank <= 3 else f" (router rank {rank})"
    print(f"  [trip {info_['trip']} {dt:4.1f}s] {'HIT ' if ok else 'MISS'} "
          f"| {ans[:48]!r}{flag}", flush=True)
print(f"\nCORPUS-100: router @1 {r1}/20 @3 {r3}/20 | end recall {hits}/20 "
      f"| median {sorted(times)[len(times)//2]:.1f}s/probe", flush=True)
print("DONE", flush=True)
