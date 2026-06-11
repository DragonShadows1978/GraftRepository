"""SCRIBE G2-G4 for a trained student (protocol §4; thresholds in
scribe/THRESHOLDS.md, registered 2026-06-11 BEFORE this file ever ran).

  G2 logit fidelity   — held-out chunks per domain (incl. the held-out
                        math domain): teacher-forced probe span, logits
                        under EXACT vs PREDICTED mounts. Report top-1
                        agreement (2-ULP tie allowance) + mean KL.
  G3 needle readback  — 10 FRESH needle docs (held-out generators):
                        greedy answers in-context vs exact vs predicted.
  G4 router recall    — 20 fresh identifier docs: latent-centroid router
                        over PREDICTED node keys vs the same repo EXACT.

Usage: python3 tests/test_scribe_floor_gates.py <mint_root> <ckpt> <ArmL|ArmS>
"""
import os, pickle, sys
import numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc
from core.minicpm3_tc import MiniCPM3_TC, _snap
from core import kv_graft
from scribe.mint import Minter
from scribe.student import ArmL, ArmS, StudentConfig, target_embeddings
from tokenizers import Tokenizer as HFTok

ROOT, CKPT, ARM = sys.argv[1], sys.argv[2], sys.argv[3]
tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
m, info = MiniCPM3_TC.from_pretrained()
m.extend_rope(4096)
print(f"loaded: {info}", flush=True)

np.random.seed(0)
st = (ArmL if ARM == "ArmL" else ArmS)(StudentConfig())
with open(CKPT, "rb") as fh:
    st.load_state_dict(pickle.load(fh)["model"])
print(f"student {ARM} loaded from {CKPT}", flush=True)
minter = Minter(m, lambda t: tok.encode(t).ids, ROOT)


def predict_graft(ids):
    """Student forward -> per-layer mountable dicts (the product)."""
    with tc.no_grad():
        pred = st(target_embeddings(m, list(ids))).numpy()[0]  # (62,S,288)
    return [{"c": pred[li, :, :256][None].astype(np.float32),
             "kpe": pred[li, :, 256:][None, None].astype(np.float32)}
            for li in range(pred.shape[0])]


def exact_graft(ids):
    return kv_graft.harvest_kv_mla(m, list(ids))


def logits_for(p_ids, graft=None):
    kv_graft.clear_injection(m)
    if graft is not None:
        kv_graft.set_injection_mla(m, graft)
    lg, _ = m(np.array([p_ids], dtype=np.int64), last_token_only=False)
    kv_graft.clear_injection(m)
    return lg.numpy()[0].astype(np.float32)


def kl(p_log, q_log):
    p = np.exp(p_log - p_log.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    q = np.exp(q_log - q_log.max(-1, keepdims=True))
    q /= q.sum(-1, keepdims=True)
    return float((p * (np.log(p + 1e-9) - np.log(q + 1e-9))).sum(-1).mean())


# ---------------- G2: logit fidelity on held-out chunks, per domain
PROBE = "User: Summarize the key facts above in one sentence.\nAssistant:"
p_ids = tok.encode(PROBE).ids
print("\n=== G2 logit fidelity (per domain) ===", flush=True)
g2 = {}
rows_by_dom = {}
for split in ("heldout", "heldout_domain"):
    for r in minter.rows(split=split):
        rows_by_dom.setdefault(r["domain"], []).append(r)
sampled = [r for rs in rows_by_dom.values() for r in rs[:4]]
if True:
    for r in sampled:
        ids = minter.load_pair(r["id"])[0][:384]
        ex = logits_for(p_ids, exact_graft(ids))
        pr = logits_for(p_ids, predict_graft(ids))
        agree = ex.argmax(-1) == pr.argmax(-1)
        # tie allowance: disagreements where exact's margin <= 0.125
        for i in np.where(~agree)[0]:
            row = np.sort(ex[i])[::-1]
            if row[0] - row[1] <= 0.125:
                agree[i] = True
        d = g2.setdefault(r["domain"], {"n": 0, "agree": 0, "kl": []})
        d["n"] += agree.size
        d["agree"] += int(agree.sum())
        d["kl"].append(kl(ex, pr))
g2_pass = True
for dom, d in sorted(g2.items()):
    a = d["agree"] / d["n"]
    K = float(np.mean(d["kl"]))
    ok = a >= 0.90 and K <= 0.5
    g2_pass &= ok
    print(f"  {dom:10s} top1 {a*100:5.1f}%  KL {K:.3f}  "
          f"{'ok' if ok else 'BELOW THRESHOLD'}", flush=True)

# ---------------- G3: needle readback (fresh generators)
rng = np.random.default_rng(99)
SYL = ["vor", "mek", "tal", "rin", "sub", "kez", "pla", "dro", "fen", "gul"]
def fresh_code():
    return ("".join(rng.choice(SYL, 2)).upper() + "-"
            + str(rng.integers(100, 9900)))
print("\n=== G3 needle readback (10 fresh docs) ===", flush=True)
g3 = {"ctx": 0, "exact": 0, "pred": 0}
for i in range(10):
    code = fresh_code()
    doc = (f"DISPATCH {i}. The relay station logged beacon {code} at "
           f"moonrise; signal strength steady; operator initialed the "
           f"ledger and sealed the cabinet.")
    q = "User: What beacon code did the relay station log?\nAssistant: Beacon"
    d_ids, q_ids = tok.encode(doc).ids, tok.encode(q).ids
    def greedy(graft, prefix_ids, n=16):
        kv_graft.clear_injection(m)
        if graft is not None:
            kv_graft.set_injection_mla(m, graft)
        ids2 = list(prefix_ids)
        lg, caches = m(np.array([ids2], np.int64), last_token_only=True)
        pos = len(ids2)
        out = [int(lg.numpy()[0, -1].argmax())]
        for _ in range(n - 1):
            lg, caches = m(np.array([[out[-1]]], np.int64), kv_caches=caches,
                           position_offset=pos, last_token_only=True)
            pos += 1
            out.append(int(lg.numpy()[0, -1].argmax()))
        kv_graft.clear_injection(m)
        return tok.decode(out)
    acc = code.lower()
    g3["ctx"] += acc in greedy(None, d_ids + q_ids).lower()
    g3["exact"] += acc in greedy(exact_graft(d_ids), q_ids).lower()
    g3["pred"] += acc in greedy(predict_graft(d_ids), q_ids).lower()
print(f"  in-context {g3['ctx']}/10 | exact {g3['exact']}/10 | "
      f"predicted {g3['pred']}/10", flush=True)
g3_pass = g3["pred"] >= g3["exact"] - 1

# ---------------- G4: router recall over predicted node keys
print("\n=== G4 router recall (20 fresh identifier docs) ===", flush=True)
docs, codes = [], []
for i in range(20):
    code = fresh_code()
    codes.append(code)
    TOPICS = ["greenhouse irrigation pump", "observatory dome motor",
              "harbor crane gearbox", "archive microfilm scanner",
              "bakery proofing cabinet", "ski lift brake assembly",
              "aquifer monitoring well", "tram pantograph arm",
              "vineyard frost fan", "foundry ladle preheater",
              "planetarium projector", "lighthouse rotation bearing",
              "cannery seam welder", "stadium turf heater",
              "mine ventilation fan", "ferry bow thruster",
              "brewery mash tun", "quarry conveyor belt",
              "hatchery water chiller", "windmill yaw drive"]
    docs.append(f"SERVICE RECORD. The {TOPICS[i]} (unit {code}) received "
                f"its overhaul; technician replaced the worn part and "
                f"logged {int(rng.integers(2, 60))} hours runtime since.")
keys_ex, keys_pr = [], []
for doc in docs:
    ids = tok.encode(doc).ids
    h = exact_graft(ids)
    keys_ex.append(kv_graft.latent_centroid(h, 44))
    pred = predict_graft(ids)
    v = pred[44]["c"][0].astype(np.float32).mean(0)
    keys_pr.append(v / (np.linalg.norm(v) + 1e-8))
hits = {"exact": [0, 0], "pred": [0, 0]}
for i, code in enumerate(codes):
    q = f"What unit code does the {TOPICS[i]} carry?"
    pl = kv_graft.harvest_kv_mla(m, tok.encode(q).ids, layer_filter={44},
                                 max_layers=45)
    p = pl[44]["c"][0].astype(np.float32).mean(0)
    p /= (np.linalg.norm(p) + 1e-8)
    for tag, keys in (("exact", keys_ex), ("pred", keys_pr)):
        scores = [float(np.dot(p, k)) for k in keys]
        order = np.argsort(scores)[::-1]
        hits[tag][0] += order[0] == i
        hits[tag][1] += i in order[:3]
print(f"  exact  recall@1 {hits['exact'][0]}/20  @3 {hits['exact'][1]}/20",
      flush=True)
print(f"  pred   recall@1 {hits['pred'][0]}/20  @3 {hits['pred'][1]}/20",
      flush=True)
g4_pass = (hits["exact"][0] - hits["pred"][0] <= 2
           and hits["exact"][1] - hits["pred"][1] <= 2)

print(f"\nFLOOR GATES ({ARM}): G2 {'PASS' if g2_pass else 'FAIL'} | "
      f"G3 {'PASS' if g3_pass else 'FAIL'} | "
      f"G4 {'PASS' if g4_pass else 'FAIL'}", flush=True)
print("DONE", flush=True)
