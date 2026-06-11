"""E1 round 2 — CONFIRMATION SET for the layer-0 router.

Round 1 (e1_router_recall.py): pre-registered all-layer router passed at
parity with mount-all (6/10 vs 6/10); the per-layer diagnostic showed layer 0
alone routes 9/10 recall@1 and its top-3 arm scored 9/10 recall — but that
was post-hoc. This round fixes the protocol IN ADVANCE on 10 FRESH cases:

  ROUTER (primary, fixed): score(graft) = mean over q-heads of
      max over (probe q pos, graft k pos) of |q . k| / sqrt(Dh), LAYER 0 ONLY.
  Arms: 0 (none), A (all 10 mounted), B (layer-0 router top-3).
  Diagnostic only: all-layer-mean router ranks.

Needles are alien strings (no common-knowledge fallbacks like round 1's
ceftriaxone). Recall = accept-substring in 64 greedy tokens, same as round 1.
Model: Qwen3-4B, fp16 cache.
"""
import os, sys, numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc
from core.qwen3_tc import Qwen3_TC, _snap
from core import kv_graft
from tokenizers import Tokenizer as HFTok

tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
m, info = Qwen3_TC.from_pretrained()
print(f"loaded: {info}", flush=True)
for L in m.layers:
    L.self_attn.quant_kv_cache = False
m.extend_rope(4096)

CASES = [
    ("APIARY INSPECTION REPORT. Hive row C checked at dawn. The new queen in "
     "colony QB-117 is marked with a teal dot and laying well across eight "
     "frames. Mite counts low. Re-inspect in fourteen days.",
     "What is the colony number of the hive with the newly marked queen?",
     ["qb-117", "qb117"]),
    ("SUBMARINE MAINTENANCE LOG. Drydock day 12. Replaced the worn periscope "
     "seal, part number PS-4419-K, sourced from the Kiel depot. Pressure test "
     "passed at 110 percent. Sonar dome recoat scheduled tomorrow.",
     "What is the part number of the replaced periscope seal?",
     ["ps-4419", "4419"]),
    ("MUSEUM ACCESSION RECORD. The red-figure amphora recovered from the "
     "Aegean wreck enters the collection under catalog number AMP-0773. "
     "Attic workshop, fifth century BCE. Conservation: salt removal bath, "
     "six weeks.",
     "Under what catalog number was the amphora accessioned?",
     ["amp-0773", "0773"]),
    ("UPPER-AIR SOUNDING NOTE. The 06Z weather balloon launched on schedule. "
     "Radiosonde callsign WHISKEY-31, burst altitude 31.8 km, strong shear "
     "noted near the tropopause. Recovery team dispatched east of the ridge.",
     "What is the callsign of the radiosonde on the morning balloon?",
     ["whiskey-31", "whiskey 31"]),
    ("VINEYARD CELLAR LOG. Harvest week four. The pinot lot is fermenting in "
     "tank 14, held at 17.5 degrees Celsius after the cap warmed overnight. "
     "Punch-downs every eight hours. Brix falling on schedule.",
     "At what temperature is the pinot fermentation tank being held?",
     ["17.5"]),
    ("BUG TRIAGE NOTE. Crash on malformed config reproduced. Filed as ticket "
     "JIRA-8852: null pointer in the manifest parser when the includes array "
     "is empty. Severity high, assigned to the runtime team for the next "
     "patch train.",
     "What is the ticket number for the parser crash?",
     ["8852"]),
    ("EXPEDITION DISPATCH. Weather window holding. The team established camp "
     "three at 6,940 meters on the west ridge, two hundred meters below the "
     "serac band. Oxygen cached. Summit push planned for Thursday pre-dawn.",
     "At what altitude was camp three established?",
     ["6,940", "6940"]),
    ("LIBRARY ARCHIVE MEMO. The thirteenth-century psalter moved to climate "
     "storage today under shelfmark MS Vellum 212. Two leaves show tide "
     "damage; the binding is original. Digitization queued for spring.",
     "What is the shelfmark of the psalter moved to climate storage?",
     ["vellum 212"]),
    ("RADIO DRAMA SYNOPSIS. Episode nine of the serial introduces the "
     "villain's alias: Mr. Quillfeather, a forger posing as a stamp dealer "
     "in the harbor district. The detective suspects the alias by the cliff-"
     "hanger.",
     "What alias does the villain use in the radio serial?",
     ["quillfeather"]),
    ("AQUARIUM NIGHT REPORT. At 02:10 the giant Pacific octopus named "
     "Brindle escaped tank 7 through an unlatched feeding hatch and was "
     "found in the crab exhibit. Returned unharmed; hatch latch replaced "
     "at shift end.",
     "What is the name of the octopus that escaped overnight?",
     ["brindle"]),
]
N = len(CASES)
TOPK = 3
NGEN = 64
ROUTE_LAYER = 0   # fixed in advance — THE protocol under test

harvs = []
for i, (text, _, _) in enumerate(CASES):
    ids = tok.encode(text).ids
    harvs.append(kv_graft.harvest_kv(m, ids))
    print(f"harvested chunk {i}: {len(ids)} tokens", flush=True)

def concat_harvests(idxs):
    out = []
    for li in range(len(m.layers)):
        k = np.concatenate([harvs[g][li]["k"] for g in idxs], axis=2)
        v = np.concatenate([harvs[g][li]["v"] for g in idxs], axis=2)
        out.append({"k": np.ascontiguousarray(k), "v": np.ascontiguousarray(v)})
    return out

def route(probe_text):
    qcap = kv_graft.capture_queries(m, tok.encode(probe_text).ids)
    nl = len(qcap)
    s = np.zeros((nl, N), dtype=np.float64)
    for li in range(nl):
        q = qcap[li][0].astype(np.float32)
        H, Lq, Dh = q.shape
        for g in range(N):
            k = harvs[g][li]["k"][0].astype(np.float32)
            kk = np.repeat(k, H // k.shape[0], axis=0)
            sc = np.einsum("hqd,hkd->hqk", q, kk) / np.sqrt(Dh)
            s[li, g] = np.abs(sc).max(axis=(1, 2)).mean()
    return s

def last_logits(idlist, caches=None, pos=0):
    lg, c = m(np.array([idlist], dtype=np.int64), kv_caches=caches,
              position_offset=pos, last_token_only=True)
    return lg.numpy()[0, -1].astype(np.float32), c

def probe(question, mount_idxs):
    kv_graft.clear_injection(m)
    if mount_idxs:
        kv_graft.set_injection(m, concat_harvests(mount_idxs))
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

print("\n=== routing (PRIMARY = layer 0) ===", flush=True)
top3 = []
ranks = []
diag_all = []
for i, (_, q, _) in enumerate(CASES):
    s = route(f"User: {q}\nAssistant:")
    order = list(np.argsort(-s[ROUTE_LAYER]))
    rank = order.index(i) + 1
    ranks.append(rank)
    top3.append(order[:TOPK])
    diag_all.append(list(np.argsort(-s.mean(axis=0))).index(i) + 1)
    print(f"probe {i}: rank-of-correct={rank}  top3={order[:TOPK]}  "
          f"margin={s[ROUTE_LAYER][order[0]]-s[ROUTE_LAYER][order[1]]:+.3f}", flush=True)
r1 = sum(r == 1 for r in ranks)
r3 = sum(r <= 3 for r in ranks)
print(f"\nLAYER-0 router: recall@1 {r1}/{N}  recall@3 {r3}/{N}", flush=True)
print(f"diag all-layer-mean ranks: {diag_all} "
      f"(recall@3 {sum(r <= 3 for r in diag_all)}/{N})", flush=True)

for arm, mounts in (("0 (none)", lambda i: []),
                    ("A (all 10)", lambda i: list(range(N))),
                    ("B (layer-0 top-3)", lambda i: sorted(top3[i]))):
    print(f"\n=== Arm {arm} ===", flush=True)
    hits = 0
    for i, (_, q, acc) in enumerate(CASES):
        g = probe(q, mounts(i))
        ok = hit(g, acc)
        hits += ok
        print(f"  probe {i}: {'HIT ' if ok else 'MISS'} | {g.strip()[:80]!r}", flush=True)
    print(f"Arm {arm}: {hits}/{N}", flush=True)
print("DONE", flush=True)
