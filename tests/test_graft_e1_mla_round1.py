"""E1 on MiniCPM3 (MLA latent grafts) — router recall, pre-registered.

PROTOCOL (fixed in advance, transferred from the Qwen3 E1 winner):
  Router = LAYER 0 only: score(graft) = mean over 40 heads of
    max over (probe-q pos, graft-k pos) of |q . k| / sqrt(96),
  composite pre-RoPE q [q_nope|q_pe] vs composite pre-RoPE graft keys
  [kv_b(latent).k_nope | k_pe] — the keys the attention itself would see,
  positions stripped. Arms: 0 (none), A (all 10), B (layer-0 top-3).
  Diagnostic only: per-layer recall@1 over all 62 layers (does the
  layer-0 law transfer to MLA?). Cases = the Qwen3 round-2 fresh set.
"""
import os, sys, numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc
from core.minicpm3_tc import MiniCPM3_TC, _snap
from core.mistral7b_tc import BlockTC
from core import kv_graft
from tokenizers import Tokenizer as HFTok

tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
m, info = MiniCPM3_TC.from_pretrained()
print(f"loaded: {info}", flush=True)
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
ROUTE_LAYER = 0
cfg = m.layers[0].self_attn.cfg
H, NOPE, ROPE, DQK = cfg.num_heads, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.q_head_dim
dt = BlockTC.COMPUTE_DTYPE

harvs = []
for i, (text, _, _) in enumerate(CASES):
    ids = tok.encode(text).ids
    harvs.append(kv_graft.harvest_kv_mla(m, ids))
    print(f"harvested chunk {i}: {len(ids)} tokens", flush=True)

# Pre-expand each graft's composite router keys per layer (40, Sg, 96), host.
def router_keys(g, li):
    att = m.layers[li].self_attn
    c = tc.tensor(np.ascontiguousarray(harvs[g][li]["c"])).astype(dt)
    Sgg = c.shape[1]
    kv = att.kv_b(c).reshape([1, Sgg, H, NOPE + cfg.v_head_dim]).transpose(1, 2)
    k_nope = kv.slice(3, 0, NOPE)
    kpe = tc.tensor(np.ascontiguousarray(harvs[g][li]["kpe"])).astype(dt)
    k_full = tc.cat([k_nope, kpe.expand([1, H, Sgg, ROPE])], dim=3)
    return k_full.numpy()[0].astype(np.float32)

def concat_harvests(idxs):
    out = []
    for li in range(len(m.layers)):
        c = np.concatenate([harvs[g][li]["c"] for g in idxs], axis=1)
        kpe = np.concatenate([harvs[g][li]["kpe"] for g in idxs], axis=2)
        out.append({"c": np.ascontiguousarray(c), "kpe": np.ascontiguousarray(kpe)})
    return out

def route(probe_text):
    qcap = kv_graft.capture_queries(m, tok.encode(probe_text).ids)
    nl = len(qcap)
    s = np.zeros((nl, N), dtype=np.float64)
    for li in range(nl):
        q = qcap[li][0].astype(np.float32)            # (40, Lq, 96)
        for g in range(N):
            k = router_keys(g, li)                    # (40, Sg, 96)
            sc = np.einsum("hqd,hkd->hqk", q, k) / np.sqrt(DQK)
            s[li, g] = np.abs(sc).max(axis=(1, 2)).mean()
    return s

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

print("\n=== routing (PRIMARY = layer 0) ===", flush=True)
top3, ranks, all_s = [], [], []
for i, (_, q, _) in enumerate(CASES):
    s = route(f"User: {q}\nAssistant:")
    all_s.append(s)
    order = list(np.argsort(-s[ROUTE_LAYER]))
    rank = order.index(i) + 1
    ranks.append(rank)
    top3.append(order[:TOPK])
    print(f"probe {i}: rank-of-correct={rank}  top3={order[:TOPK]}  "
          f"margin={s[ROUTE_LAYER][order[0]]-s[ROUTE_LAYER][order[1]]:+.3f}", flush=True)
r1 = sum(r == 1 for r in ranks)
r3 = sum(r <= 3 for r in ranks)
print(f"\nLAYER-0 router: recall@1 {r1}/{N}  recall@3 {r3}/{N}", flush=True)
nl = all_s[0].shape[0]
pl = [sum(int(np.argmax(all_s[i][li])) == i for i in range(N)) for li in range(nl)]
print("diag per-layer recall@1:", " ".join(f"{li}:{c}" for li, c in enumerate(pl)), flush=True)
best = int(np.argmax(pl))
print(f"diag best layer: {best} ({pl[best]}/10)", flush=True)

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
