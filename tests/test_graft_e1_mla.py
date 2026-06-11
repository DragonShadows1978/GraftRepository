"""E1-MLA round 2 — CONFIRMATION of the LATENT router on MiniCPM3.

Round 1: pre-registered key-space router FAILED (probe-independent outlier-key
scores; no qk-norm on this model). Diagnostic sweep found latent-space routing:
cos(mean_t c_n_probe, mean_t c_n_graft) routes 10/10 @1 (L44/L51), 10/10 @3
from L8. Those picks were post-hoc -> this round fixes protocol IN ADVANCE on
10 FRESH never-used cases:

  PRIMARY ROUTER: latent cosine mean/mean at LAYER 10 (cost-optimal pick).
  Secondary (recorded only, no arms): same score at LAYER 44.
  Arms: 0 (none), A (all 10 mounted), B (L10 latent router top-3).
  TOPK=3, NGEN=64 greedy, substring hit rule. MiniCPM3-4B INT4 MLA.
"""
import os, sys, numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc
from core.minicpm3_tc import MiniCPM3_TC, _snap
from core import kv_graft
from tokenizers import Tokenizer as HFTok

tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
m, info = MiniCPM3_TC.from_pretrained()
print(f"loaded: {info}", flush=True)
m.extend_rope(4096)

CASES = [
    ("LIGHTHOUSE KEEPER'S LOG. Fog until noon. The lamp rotation motor, "
     "serial LR-2208, began grinding at dusk; greased the bearing race and "
     "noise cleared. Mercury bath level nominal. Relief boat due Friday.",
     "What is the serial number of the lighthouse lamp rotation motor?",
     ["lr-2208", "2208"]),
    ("FALCONRY MEWS RECORD. The white gyrfalcon named Vesper, ring number "
     "F-77, flew at 412 grams this morning and took the lure cleanly twice. "
     "New tail feather imped on the left side. Free flying by the weekend.",
     "What is the name of the gyrfalcon in the mews record?",
     ["vesper"]),
    ("GEOTHERMAL PLANT SHIFT REPORT. Well GT-9 is delivering 4.2 megawatts "
     "after yesterday's scaling washout, best output this quarter. Brine "
     "pressure steady. Reinjection pump two back online at 0300.",
     "How many megawatts is the geothermal well delivering?",
     ["4.2"]),
    ("ANTARCTIC TRAVERSE PLAN. Resupply confirmed: depot cache GAMMA-12 was "
     "laid at 81 degrees south, flagged at double height for drift. Contains "
     "fuel, rations for twelve days, and a spare alternator.",
     "What is the designation of the depot cache on the traverse?",
     ["gamma-12", "gamma 12"]),
    ("LUTHIER'S BENCH NOTES. The violin's interior label is dated 1741. "
     "Retouch areas logged under varnish code V-318; ground coat intact. "
     "Bass bar replacement deferred pending the owner's decision.",
     "What is the varnish code logged for the violin retouch areas?",
     ["v-318", "318"]),
    ("BREWHOUSE BATCH SHEET. Saison batch 41: pitched yeast strain WLP-566 "
     "at 19 degrees, free rise to 27 planned. Gravity 1.052 at knockout. "
     "Dry hop with Saaz on day five.",
     "Which yeast strain was pitched for the saison batch?",
     ["wlp-566", "566"]),
    ("FREIGHT DESPATCH WIRE. Locomotive 4471 departed the junction at 05:50 "
     "hauling 62 wagons of bauxite for the smelter spur. Axle counter at "
     "milepost 19 reported clean. Crew change at the river yard.",
     "What is the number of the locomotive hauling the bauxite?",
     ["4471"]),
    ("REEF SURVEY DATA NOTE. Transect T-9 resurveyed at low tide: bleaching "
     "now at 14 percent of colonies, down from 22 in March. Crown-of-thorns "
     "count zero. Water at 26.1 degrees.",
     "What percentage of colonies show bleaching on the resurveyed transect?",
     ["14"]),
    ("HOROLOGY WORKSHOP TICKET. The longcase clock takes escapement part "
     "ES-905, sourced from the Prague workshop's spring stock. Pallet faces "
     "repolished; beat set even. Case wax on collection.",
     "What is the part number of the escapement for the longcase clock?",
     ["es-905", "905"]),
    ("MYCOLOGY FORAY LIST. Under the beeches we tagged a Cortinarius "
     "specimen as CF-33 — violet cap, rusty spore print. Spore slides "
     "prepared; microscopy on Tuesday. No edibles collected.",
     "What tag was given to the Cortinarius specimen found under the beeches?",
     ["cf-33", "cf33"]),
]
N = len(CASES)
TOPK = 3
NGEN = 64
RL_PRIMARY = 10
RL_SECONDARY = 44

harvs = []
for i, (text, _, _) in enumerate(CASES):
    ids = tok.encode(text).ids
    harvs.append(kv_graft.harvest_kv_mla(m, ids))
    print(f"harvested chunk {i}: {len(ids)} tokens", flush=True)

def unit(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)

# graft summary keys: mean latent per layer (the 512-byte routing index)
cent = {li: [unit(harvs[g][li]["c"][0].astype(np.float32).mean(0)) for g in range(N)]
        for li in (RL_PRIMARY, RL_SECONDARY)}

def route(probe_text, li):
    plat = kv_graft.harvest_kv_mla(m, tok.encode(probe_text).ids,
                                   layer_filter={li})
    p = unit(plat[li]["c"][0].astype(np.float32).mean(0))
    return np.array([float(np.dot(p, cent[li][g])) for g in range(N)])

def concat_harvests(idxs):
    out = []
    for li in range(len(m.layers)):
        c = np.concatenate([harvs[g][li]["c"] for g in idxs], axis=1)
        kpe = np.concatenate([harvs[g][li]["kpe"] for g in idxs], axis=2)
        out.append({"c": np.ascontiguousarray(c), "kpe": np.ascontiguousarray(kpe)})
    return out

def last_logits(idlist, caches=None, pos=0):
    lg, c = m(np.array([idlist], dtype=np.int64), kv_caches=caches,
              position_offset=pos, last_token_only=True)
    return lg.numpy()[0, -1].astype(np.float32), c

def probe(question, mount_idxs):
    kv_graft.clear_injection(m)
    if mount_idxs:
        kv_graft.set_injection_mla(m, concat_harvests(mount_idxs))
    ids = tok.encode(f"User: {question}\nAssistant:").ids
    row, caches = last_logits(ids)
    pos = len(ids)
    out = [int(row.argmax())]
    for _ in range(NGEN - 1):
        row, caches = last_logits([out[-1]], caches, pos)
        pos += 1
        out.append(int(row.argmax()))
    kv_graft.clear_injection(m)
    return tok.decode(out)

def hit(text, accepts):
    t = text.lower()
    return any(a in t for a in accepts)

print(f"\n=== routing (PRIMARY = latent cos, layer {RL_PRIMARY}) ===", flush=True)
top3, ranks, ranks2 = [], [], []
for i, (_, q, _) in enumerate(CASES):
    p = f"User: {q}\nAssistant:"
    s = route(p, RL_PRIMARY)
    order = list(np.argsort(-s))
    rank = order.index(i) + 1
    ranks.append(rank)
    top3.append(order[:TOPK])
    s2 = route(p, RL_SECONDARY)
    ranks2.append(list(np.argsort(-s2)).index(i) + 1)
    print(f"probe {i}: rank-of-correct={rank}  top3={order[:TOPK]}  "
          f"margin={s[order[0]]-s[order[1]]:+.4f}", flush=True)
print(f"\nPRIMARY L{RL_PRIMARY}: recall@1 {sum(r == 1 for r in ranks)}/{N}  "
      f"recall@3 {sum(r <= 3 for r in ranks)}/{N}", flush=True)
print(f"diag L{RL_SECONDARY} ranks: {ranks2} "
      f"(@1 {sum(r == 1 for r in ranks2)}/{N}, @3 {sum(r <= 3 for r in ranks2)}/{N})", flush=True)

for arm, mounts in (("0 (none)", lambda i: []),
                    ("A (all 10)", lambda i: list(range(N))),
                    (f"B (L{RL_PRIMARY} latent top-3)", lambda i: sorted(top3[i]))):
    print(f"\n=== Arm {arm} ===", flush=True)
    hits = 0
    for i, (_, q, acc) in enumerate(CASES):
        g = probe(q, mounts(i))
        ok = hit(g, acc)
        hits += ok
        print(f"  probe {i}: {'HIT ' if ok else 'MISS'} | {g.strip()[:80]!r}", flush=True)
    print(f"Arm {arm}: {hits}/{N}", flush=True)
print("DONE", flush=True)
