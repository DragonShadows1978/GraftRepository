"""E1 — Router recall (the Graft-Repository keystone number).

Harvest 10 distinct-topic chunks, each with a planted needle. Three arms:
  Arm 0: nothing mounted (control — needles must NOT be guessable)
  Arm A: all 10 grafts mounted (concatenated), probe each needle
  Arm B: per probe, mount ONLY the router's top-3 picks

ROUTER (protocol fixed in advance, before any results):
  score(graft) = mean over ALL layers of
                   mean over q-heads of
                     max over (probe q pos, graft k pos) of |q . k| / sqrt(Dh)
  q = probe's pre-RoPE post-norm queries (capture_queries), k = harvested
  pre-RoPE keys (GQA: kv-head repeated to q-heads). |.| matches APA bulk
  selection. Diagnostics (NOT used for arm B): signed variant, per-layer
  recall table, prefix-k layer policies.

PASS: Arm B recall ~= Arm A recall. Recall = needle substring (case-insens.)
in 24 greedy tokens. Model: Qwen3-4B INT4 (the graft testbed), fp16 cache.
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

# 10 chunks, distinct topics, one needle each. (chunk_text, probe, accept_substrings)
CASES = [
    ("VOLCANO MONITORING BULLETIN. The new seismic station on the north flank "
     "of Mount Rainier went live on Tuesday. Station codename: EMBERFALL-7. "
     "It reports tremor counts every six minutes via the USGS uplink and "
     "replaces the aging analog vault destroyed in the 2029 lahar.",
     "What is the codename of the new volcano seismic station?",
     ["emberfall"]),
    ("FAMILY RECIPE CARD. Grandmother's lamb stew feeds eight. Brown the lamb "
     "shanks, add onions, carrots, and stock. The secret ingredient is ground "
     "cardamom: exactly 11 grams, stirred in during the final ten minutes. "
     "Serve over barley with flatbread.",
     "How many grams of the secret spice does the stew recipe call for?",
     ["11 gram", "11g", "eleven gram"]),
    ("SHIPPING MANIFEST. The cargo vessel MV Stellar Tern departed Valparaiso "
     "on the 14th carrying 4,250 crates of citrus bound for Rotterdam. Master: "
     "Capt. H. Okafor. Draft 11.2 m, ETA twenty-two days, Panama transit booked.",
     "What is the name of the cargo vessel in the shipping manifest?",
     ["stellar tern"]),
    ("OPS RUNBOOK ENTRY. Staging cluster access rotated this sprint. The new "
     "admin passphrase for the staging server is velvet-octopus-29. Do not "
     "reuse it on production; production remains on hardware tokens only.",
     "What is the admin passphrase for the staging server?",
     ["velvet-octopus-29", "velvet octopus"]),
    ("PATIENT INTAKE SUMMARY. Marta Ellison, 54, presented with cellulitis of "
     "the left forearm. CRITICAL: patient is allergic to ceftriaxone — "
     "documented anaphylaxis in 2027. Started on clindamycin instead. Vitals "
     "stable, discharge expected Friday.",
     "Which antibiotic is the patient allergic to?",
     ["ceftriaxone"]),
    ("OBSERVATORY LOG. Clear skies, seeing 0.9 arcsec. Confirmed recovery of "
     "the long-period comet, official designation C/2031 X4 (Veld), at "
     "magnitude 14.2 in Cetus. Tail PA 280 degrees. Next window in nine days.",
     "What is the official designation of the comet in the observatory log?",
     ["x4", "veld"]),
    ("RAIL NOTICE. From Monday the morning express to Brindlemoor departs at "
     "07:42 from platform 9, four minutes earlier than the winter timetable. "
     "Seat reservations compulsory north of Harrowgate junction.",
     "At what time does the morning express to Brindlemoor depart?",
     ["07:42", "7:42"]),
    ("BOTANY FIELD NOTE. Ridge transect day 6. Collected a flowering orchid "
     "matching no published key; provisional name Dendrobium calyptra-novae. "
     "Found at elevation 2,340 m on the cloud-forest ridge, growing epiphytic "
     "on Weinmannia. Two vouchers pressed.",
     "At what elevation was the new orchid species found?",
     ["2,340", "2340"]),
    ("TREASURY MEMO. The quarterly settlement to the Lisbon counterparty "
     "cleared this morning. Wire transfer reference FRX-88412, value date "
     "the 9th, amount as contracted. Reconciliation closes Thursday.",
     "What is the wire transfer reference number in the treasury memo?",
     ["88412"]),
    ("CHESS ANNOTATION. Round 5 featured a sharp novelty: the opening now "
     "being called the Karpathy Gambit, where White sacrifices on move 12 "
     "with Nxf7 for a lasting initiative. Black declined and drifted into a "
     "worse endgame; 1-0 in 41.",
     "What is the name of the chess opening novelty in the annotation?",
     ["karpathy"]),
]
N = len(CASES)
TOPK = 3
NGEN = 64

# ---------------------------------------------------------------- harvest all
harvs = []
for i, (text, _, _) in enumerate(CASES):
    ids = tok.encode(text).ids
    h = kv_graft.harvest_kv(m, ids)
    harvs.append(h)
    print(f"harvested chunk {i}: {len(ids)} tokens", flush=True)

def concat_harvests(idxs):
    out = []
    for li in range(len(m.layers)):
        k = np.concatenate([harvs[g][li]["k"] for g in idxs], axis=2)
        v = np.concatenate([harvs[g][li]["v"] for g in idxs], axis=2)
        out.append({"k": np.ascontiguousarray(k), "v": np.ascontiguousarray(v)})
    return out

# ---------------------------------------------------------------- router
def route(probe_text):
    """Returns (NLAYERS, N) per-layer scores, abs and signed variants."""
    qcap = kv_graft.capture_queries(m, tok.encode(probe_text).ids)
    nl = len(qcap)
    s_abs = np.zeros((nl, N), dtype=np.float64)
    s_sgn = np.zeros((nl, N), dtype=np.float64)
    for li in range(nl):
        q = qcap[li][0].astype(np.float32)            # (H, Lq, Dh)
        H, Lq, Dh = q.shape
        for g in range(N):
            k = harvs[g][li]["k"][0].astype(np.float32)   # (KVH, S, Dh)
            kk = np.repeat(k, H // k.shape[0], axis=0)    # (H, S, Dh)
            sc = np.einsum("hqd,hkd->hqk", q, kk) / np.sqrt(Dh)
            s_abs[li, g] = np.abs(sc).max(axis=(1, 2)).mean()
            s_sgn[li, g] = sc.max(axis=(1, 2)).mean()
    return s_abs, s_sgn

# ---------------------------------------------------------------- generation
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

# ---------------------------------------------------------------- route all
print("\n=== routing ===", flush=True)
ranks_primary = []           # rank of correct graft under the PRIMARY policy
top3_primary = []
all_abs, all_sgn = [], []
for i, (_, q, _) in enumerate(CASES):
    s_abs, s_sgn = route(f"User: {q}\nAssistant:")
    all_abs.append(s_abs); all_sgn.append(s_sgn)
    score = s_abs.mean(axis=0)                       # PRIMARY: mean over layers
    order = list(np.argsort(-score))
    rank = order.index(i) + 1
    ranks_primary.append(rank)
    top3_primary.append(order[:TOPK])
    print(f"probe {i}: rank-of-correct={rank}  top3={order[:TOPK]}  "
          f"margin={score[order[0]]-score[order[1]]:+.3f}", flush=True)
top3_layer0 = [list(np.argsort(-all_abs[i][0]))[:TOPK] for i in range(N)]

r1 = sum(r == 1 for r in ranks_primary)
r3 = sum(r <= 3 for r in ranks_primary)
print(f"\nPRIMARY router (abs, all-layer mean): recall@1 {r1}/{N}  recall@3 {r3}/{N}", flush=True)

# diagnostics: signed variant + per-layer + prefix-k (printed, not used for arm B)
sgn_r3 = sum(list(np.argsort(-all_sgn[i].mean(axis=0))).index(i) < TOPK for i in range(N))
print(f"diag signed all-layer mean: recall@3 {sgn_r3}/{N}", flush=True)
nl = all_abs[0].shape[0]
per_layer_r1 = [sum(int(np.argmax(all_abs[i][li])) == i for i in range(N)) for li in range(nl)]
print("diag per-layer recall@1:", " ".join(f"{li}:{c}" for li, c in enumerate(per_layer_r1)), flush=True)
for kp in (4, 8, 18, nl):
    rk = sum(list(np.argsort(-np.mean(all_abs[i][:kp], axis=0))).index(i) < TOPK for i in range(N))
    print(f"diag prefix layers 0..{kp-1}: recall@3 {rk}/{N}", flush=True)

# ---------------------------------------------------------------- arms
for arm, mounts in (("0 (none)", lambda i: []),
                    ("A (all 10)", lambda i: list(range(N))),
                    ("B (routed top-3)", lambda i: sorted(top3_primary[i])),
                    ("B0 (diag: layer-0 router top-3)", lambda i: sorted(top3_layer0[i]))):
    print(f"\n=== Arm {arm} ===", flush=True)
    hits = 0
    for i, (_, q, acc) in enumerate(CASES):
        g = probe(q, mounts(i))
        ok = hit(g, acc)
        hits += ok
        print(f"  probe {i}: {'HIT ' if ok else 'MISS'} | {g.strip()[:80]!r}", flush=True)
    print(f"Arm {arm}: {hits}/{N}", flush=True)
print("DONE", flush=True)
