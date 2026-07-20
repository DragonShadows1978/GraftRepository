#!/usr/bin/env python3
"""NC17-P2 parity + quant-noise analysis (CPU; consumes captured chains).

Inputs (all on disk):
  logs/nc17/p0_gt.npz            (GT; sha asserted)
  logs/nc17/p2_chain_int4.npz    (tc-INT4 cached-chain per-position logits)
  logs/nc17/p2_chain_bf16.npz    (tc-bf16  cached-chain per-position logits, same engine)

Produces:
  (A) PARITY vs GT for INT4 — the SAME protocol/table as P1 (p1_parity.py):
      top-1 agreement over final-prefill + 64 teacher-forced decode positions;
      a flip is acceptable ONLY at a GT near-tie margin (< 0.25); every flip
      reported with GT margin + near-tie/HARD tag. GATE: PASS iff 0 HARD flips.
  (B) INT4-vs-P1-bf16 SAME-ENGINE cached-chain max|Delta logit| distribution —
      both chains drift identically, so this isolates quantization noise from
      port/drift noise (registered spec note). Reported as the clean measure.
  (C) Flip attribution for the P1b hand-off: for each INT4 HARD flip vs GT, note
      whether tc-bf16 (P1) ALSO flipped there — a flip present in bf16 is a
      drift/port flip carried through, NOT introduced by INT4; a flip present
      ONLY in INT4 is a candidate quant-noise flip that p2_adjudicate.py
      fresh-refeeds.

Matched-reference law: names the exact GT file + sha256.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
GT_PATH = REPO / "logs" / "nc17" / "p0_gt.npz"
INT4 = REPO / "logs" / "nc17" / "p2_chain_int4.npz"
BF16 = REPO / "logs" / "nc17" / "p2_chain_bf16.npz"
OUT = REPO / "logs" / "nc17" / "p2_parity.json"
NEAR_TIE_MARGIN = 0.25
EXPECTED_GT_SHA = "0fc4099b3537083ec99478c9cdb969a5afca038609636499d64344a3303575fb"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def top2_margin(logits):
    idx = np.argpartition(logits, -2)[-2:]
    a, b = logits[idx[0]], logits[idx[1]]
    hi, lo = (a, b) if a >= b else (b, a)
    return float(hi - lo)


def main():
    gt_sha = sha256_file(GT_PATH)
    assert gt_sha == EXPECTED_GT_SHA, f"GT sha mismatch {gt_sha}"
    gt = np.load(GT_PATH, allow_pickle=False)
    ci = np.load(INT4, allow_pickle=False)
    cb = np.load(BF16, allow_pickle=False)
    n_prompts = sum(1 for k in gt.files if k.startswith("prompt_ids_"))
    print(f"[parity] GT sha256: {gt_sha}", flush=True)
    print(f"[parity] n_prompts: {n_prompts} near_tie_margin: {NEAR_TIE_MARGIN}", flush=True)

    all_abs = []
    n_pos = 0
    n_agree = 0
    flips = []
    q_abs = []           # per-position max|int4 - bf16|
    q_all = []           # full-vector |int4 - bf16| for percentiles
    bf16_flip_set = set()
    per_prompt = []

    for i in range(n_prompts):
        pagree = 0; ppos = 0; pmax = 0.0
        nd = gt[f"decode_tokens_{i}"].shape[0]
        keys = ["final"] + [f"dec{s}" for s in range(nd)]
        for k in keys:
            if k == "final":
                gt_v = gt[f"final_logits_{i}"].astype(np.float32)
                human = "final"
            else:
                s = int(k[3:])
                gt_v = gt[f"decode_logits_{i}"].astype(np.float32)[s]
                human = f"decode{s}"
            iv = ci[f"logits_{i}_{k}"].astype(np.float32)
            bv = cb[f"logits_{i}_{k}"].astype(np.float32)

            # (A) parity vs GT
            d = np.abs(iv - gt_v)
            all_abs.append(d); n_pos += 1; ppos += 1
            pmax = max(pmax, float(d.max()))
            g1 = int(gt_v.argmax()); i1 = int(iv.argmax())
            mg = top2_margin(gt_v)
            if i1 == g1:
                n_agree += 1; pagree += 1
            else:
                flips.append({"prompt": i, "pos": human, "gt_top1": g1,
                              "tc_top1": i1, "gt_margin": mg,
                              "near_tie": mg < NEAR_TIE_MARGIN})
            # (B) same-engine int4 vs bf16
            qd = np.abs(iv - bv)
            q_abs.append(float(qd.max())); q_all.append(qd)
            # (C) did bf16 also flip vs GT here?
            b1 = int(bv.argmax())
            if b1 != g1:
                bf16_flip_set.add((i, human))
        per_prompt.append({"prompt": i, "n_positions": ppos, "top1_agree": pagree,
                           "agree_frac": pagree / ppos, "max_abs_dlogit": pmax})
        print(f"[parity] prompt {i}: INT4-vs-GT agree {pagree}/{ppos} "
              f"max|dlogit|={pmax:.4f}", flush=True)

    allabs = np.concatenate(all_abs)
    qall = np.concatenate(q_all)
    qabs = np.array(q_abs)
    n_flips = len(flips)
    n_nt = sum(1 for f in flips if f["near_tie"])
    n_hard = n_flips - n_nt

    for f in flips:
        f["bf16_also_flipped"] = (f["prompt"], f["pos"]) in bf16_flip_set
    hard_int4 = [f for f in flips if not f["near_tie"]]
    hard_int4_only = [f for f in hard_int4 if not f["bf16_also_flipped"]]
    hard_int4_shared = [f for f in hard_int4 if f["bf16_also_flipped"]]

    summary = {
        "stage": "NC17-P2",
        "gt_file": str(GT_PATH), "gt_sha256": gt_sha,
        "near_tie_margin": NEAR_TIE_MARGIN,
        "int4_parity_vs_gt": {
            "n_positions": n_pos, "n_top1_agree": n_agree,
            "top1_agreement": n_agree / n_pos,
            "n_flips": n_flips, "n_near_tie_flips": n_nt, "n_hard_flips": n_hard,
            "max_abs_dlogit": float(allabs.max()),
            "mean_abs_dlogit": float(allabs.mean()),
            "p50_abs_dlogit": float(np.percentile(allabs, 50)),
            "p99_abs_dlogit": float(np.percentile(allabs, 99)),
            "p999_abs_dlogit": float(np.percentile(allabs, 99.9)),
            "flips": flips,
            "verdict": "PASS" if n_hard == 0 else "FAIL",
        },
        "int4_vs_p1bf16_same_engine": {
            "note": "cached-chain, same engine, both chains drift identically -> "
                    "isolates INT4 quantization noise (registered spec note)",
            "per_position_maxabs_delta": {
                "max": float(qabs.max()), "mean": float(qabs.mean()),
                "p50": float(np.percentile(qabs, 50)),
                "p99": float(np.percentile(qabs, 99)),
            },
            "full_vector_absdelta": {
                "max": float(qall.max()), "mean": float(qall.mean()),
                "p50": float(np.percentile(qall, 50)),
                "p99": float(np.percentile(qall, 99)),
                "p999": float(np.percentile(qall, 99.9)),
            },
        },
        "hard_flip_attribution": {
            "n_hard_int4_vs_gt": len(hard_int4),
            "n_hard_shared_with_bf16": len(hard_int4_shared),
            "n_hard_int4_only": len(hard_int4_only),
            "hard_int4_only_positions": [{"prompt": f["prompt"], "pos": f["pos"],
                                          "gt_margin": f["gt_margin"]}
                                         for f in hard_int4_only],
            "note": "shared = flip already present in P1 bf16 (drift/port carried "
                    "through, not INT4-introduced); int4_only = candidate "
                    "quant-noise flips -> p2_adjudicate fresh-refeeds them",
        },
        "per_prompt": per_prompt,
    }
    OUT.write_text(json.dumps(summary, indent=2))

    print("\n=== NC17-P2 PARITY (tc-INT4 cached-chain vs P0 GT) ===", flush=True)
    print(f"top-1 agreement: {n_agree}/{n_pos} = {n_agree/n_pos:.4f}", flush=True)
    print(f"flips: {n_flips} total | {n_nt} near-tie (<{NEAR_TIE_MARGIN}) | {n_hard} HARD", flush=True)
    for f in flips:
        tag = "near-tie" if f["near_tie"] else "*** HARD FLIP ***"
        shr = " [bf16-shared]" if f["bf16_also_flipped"] else " [INT4-only]"
        print(f"  flip p{f['prompt']} {f['pos']}: GT {f['gt_top1']} -> int4 "
              f"{f['tc_top1']} | GT margin {f['gt_margin']:.4f} [{tag}]{shr}", flush=True)
    print(f"INT4-vs-GT max|dlogit|={allabs.max():.4f} mean={allabs.mean():.4f} "
          f"p99={np.percentile(allabs,99):.4f}", flush=True)
    print(f"\n=== INT4-vs-P1bf16 SAME-ENGINE quant-noise (clean) ===", flush=True)
    print(f"per-position max|delta|: max={qabs.max():.4f} mean={qabs.mean():.4f} "
          f"p50={np.percentile(qabs,50):.4f} p99={np.percentile(qabs,99):.4f}", flush=True)
    print(f"full-vector |delta|: mean={qall.mean():.5f} p99={np.percentile(qall,99):.4f} "
          f"p99.9={np.percentile(qall,99.9):.4f} max={qall.max():.4f}", flush=True)
    print(f"\n=== HARD-FLIP ATTRIBUTION ===", flush=True)
    print(f"INT4 HARD flips vs GT: {len(hard_int4)} | shared-with-bf16(drift): "
          f"{len(hard_int4_shared)} | INT4-ONLY(quant candidates): {len(hard_int4_only)}",
          flush=True)
    verdict = "PASS" if n_hard == 0 else "FAIL"
    print(f"\n[parity] INT4-vs-GT cached-chain GATE VERDICT: {verdict}", flush=True)
    print(f"[parity] wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
