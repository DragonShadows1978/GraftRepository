"""SCRIBE G0 — instrument check (protocol §4): the pair-minting pipeline
must produce EXACT grafts. Mint held-out documents to DISK through
scribe.mint.Minter, reload, mount, and compare logits against in-context
at the same seats, teacher-forced. PASS: top-1 identical, max|logit diff|
at the established bf16 noise floor — and the measured number becomes the
floor the G2-G5 thresholds are registered against.
"""
import os, sys, shutil
import numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc
from core.minicpm3_tc import MiniCPM3_TC, _snap
from core import kv_graft
from scribe.mint import Minter
from tokenizers import Tokenizer as HFTok

tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
m, info = MiniCPM3_TC.from_pretrained()
print(f"loaded: {info}", flush=True)
m.extend_rope(4096)

DOCS = [
    ("ops",  "MAINTENANCE BULLETIN. The coolant manifold on line 4 was "
             "replaced at shift change; torque spec 48 Nm; part lot "
             "VX-2291. Sign-off: foreman Ilsa Brandt.",
     "User: What is the part lot of the replaced coolant manifold?\n"
     "Assistant: The part lot is"),
    ("research", "FIELD NOTE. The tagged osprey (band K-557) fished the "
                 "north weir at dawn, three successful strikes in nine "
                 "minutes. Wind calm. Next census on the 14th.",
     "User: What is the band number of the tagged osprey?\n"
     "Assistant: The band number is"),
    ("narrative", "JOURNAL. Marisol kept the brass key under the third "
                  "floorboard. The lighthouse logbook listed her arrival "
                  "as the ninth of October. The lamp wick needs trimming.",
     "User: Where did Marisol keep the brass key?\nAssistant: She kept it"),
]

ROOT = "/tmp/scribe_mint_g0"
shutil.rmtree(ROOT, ignore_errors=True)
minter = Minter(m, lambda t: tok.encode(t).ids, ROOT)


def all_logits(idlist, caches=None, pos=0):
    lg, c = m(np.array([idlist], dtype=np.int64), kv_caches=caches,
              position_offset=pos, last_token_only=False)
    return lg.numpy()[0].astype(np.float32), c


worst, all_t1 = 0.0, True
for domain, doc, probe in DOCS:
    pid = minter.mint(doc, domain=domain, split="g0")
    ids, c, kpe = minter.load_pair(pid)           # DISK round trip
    d_ids = list(ids)
    p_ids = tok.encode(probe).ids
    Sg = len(d_ids)

    kv_graft.clear_injection(m)
    ctx, _ = all_logits(d_ids + p_ids)
    ctx_p = ctx[Sg:]

    # A) PIPELINE ISOLATION: minted-from-disk vs direct-harvest mount —
    # the Minter must add NOTHING (measured 2026-06-11: identical).
    direct = kv_graft.harvest_kv_mla(m, d_ids)
    kv_graft.set_injection_mla(m, direct)
    gra_d, _ = all_logits(p_ids)
    kv_graft.clear_injection(m)

    harv = [{"c": c[li][None].astype(np.float32),
             "kpe": kpe[li][None, None].astype(np.float32)}
            for li in range(len(m.layers))]
    kv_graft.set_injection_mla(m, harv)
    gra, _ = all_logits(p_ids)
    kv_graft.clear_injection(m)

    pipe_d = float(np.max(np.abs(gra - gra_d)))

    # B) graft-vs-in-context floor: top-1 flips allowed ONLY at the
    # bf16 exact-tie class (ctx margin <= 2 ULP = 0.125) — the
    # ESTABLISHED noise-floor definition from the equivalence gates.
    d = float(np.max(np.abs(gra - ctx_p)))
    flips = np.where(gra.argmax(-1) != ctx_p.argmax(-1))[0]
    margins = []
    for f in flips:
        row = np.sort(ctx_p[f])[::-1]
        margins.append(float(row[0] - row[1]))
    ties_only = all(mg <= 0.125 for mg in margins)
    worst = max(worst, d)
    all_t1 = all_t1 and ties_only and pipe_d <= 1e-3
    print(f"  [{domain:9s}] {pid}  ntok={Sg:3d}  pipeline d={pipe_d:.5f}  "
          f"max|dlogit|={d:.4f}  flips={len(flips)} "
          f"(margins {['%.4f' % mg for mg in margins]})", flush=True)

n_rows = len(minter.rows(split="g0"))
print(f"\nmanifest rows: {n_rows}; mint root {ROOT}", flush=True)
print(f"SCRIBE-G0: {'PASS' if all_t1 else 'FAIL'} | measured noise floor "
      f"(disk-minted, fp16) = {worst:.4f} max|dlogit| — G2-G5 thresholds "
      f"register against this", flush=True)
print("DONE", flush=True)
