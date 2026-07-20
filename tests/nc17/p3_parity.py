#!/usr/bin/env python3
"""NC17-P3 parity + quant-noise analysis (CPU; consumes captured chains).

Inputs (all on disk):
  logs/nc17/p0_gt.npz            (GT; sha asserted)
  logs/nc17/p3_chain_int6.npz    (tc-INT6 fork cached-chain per-position logits)
  logs/nc17/p3_chain_bf16.npz    (tc-bf16 fork cached-chain per-position logits, same engine)

Produces:
  (A) PARITY vs GT for INT6 — the SAME protocol/table as P1/P2: top-1 agreement
      over final-prefill + 64 teacher-forced decode positions; a flip is
      acceptable ONLY at a GT near-tie margin (<0.25); every flip reported with
      GT margin + near-tie/HARD tag. GATE: PASS iff 0 HARD flips (cached-chain
      is the drift-tolerance protocol, expected RED class per P1b spec note —
      the port-correctness question is the fresh-refeed adjudication).
  (B) INT6-vs-P1bf16 SAME-ENGINE (fork) cached-chain max|Delta logit| distribution
      — both chains drift identically on the SAME fork engine, isolating INT6
      quantization noise from port/drift noise (registered spec note).
  (C) Flip attribution for the adjudication hand-off: for each INT6 HARD flip vs
      GT, note whether tc-bf16(fork) ALSO flipped there — shared = drift carried
      through; int6_only = candidate quant-noise flip that p3_adjudicate
      fresh-refeeds.

Matched-reference law: names the exact GT file + sha256 and both chain files.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
GT_PATH = REPO / "logs" / "nc17" / "p0_gt.npz"
INT6 = REPO / "logs" / "nc17" / "p3_chain_int6.npz"
BF16 = REPO / "logs" / "nc17" / "p3_chain_bf16.npz"
OUT = REPO / "logs" / "nc17" / "p3_parity.json"
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


def _meta(npz):
    try:
        return json.loads(bytes(npz["_meta"]).decode())
    except Exception:
        return {}


def main():
    gt_sha = sha256_file(GT_PATH)
    assert gt_sha == EXPECTED_GT_SHA, f"GT sha mismatch {gt_sha}"
    gt = np.load(GT_PATH, allow_pickle=False)
    ci = np.load(INT6, allow_pickle=False)
    cb = np.load(BF16, allow_pickle=False)
    n_prompts = sum(1 for k in gt.files if k.startswith("prompt_ids_"))
    mi, mb = _meta(ci), _meta(cb)
    print(f"[parity] GT sha256: {gt_sha}", flush=True)
    print(f"[parity] int6 chain engine: {mi.get('engine_build')}", flush=True)
    print(f"[parity] bf16 chain engine: {mb.get('engine_build')}", flush=True)
    print(f"[parity] n_prompts: {n_prompts} near_tie_margin: {NEAR_TIE_MARGIN}", flush=True)

    all_abs = []
    n_pos = 0
    n_agree = 0
    flips = []
    q_abs = []
    q_all = []
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
            qd = np.abs(iv - bv)
            q_abs.append(float(qd.max())); q_all.append(qd)
            b1 = int(bv.argmax())
            if b1 != g1:
                bf16_flip_set.add((i, human))
        per_prompt.append({"prompt": i, "n_positions": ppos, "top1_agree": pagree,
                           "agree_frac": pagree / ppos, "max_abs_dlogit": pmax})
        print(f"[parity] prompt {i}: INT6-vs-GT agree {pagree}/{ppos} "
              f"max|dlogit|={pmax:.4f}", flush=True)

    allabs = np.concatenate(all_abs)
    qall = np.concatenate(q_all)
    qabs = np.array(q_abs)
    n_flips = len(flips)
    n_nt = sum(1 for f in flips if f["near_tie"])
    n_hard = n_flips - n_nt

    for f in flips:
        f["bf16_also_flipped"] = (f["prompt"], f["pos"]) in bf16_flip_set
    hard_int6 = [f for f in flips if not f["near_tie"]]
    hard_int6_only = [f for f in hard_int6 if not f["bf16_also_flipped"]]
    hard_int6_shared = [f for f in hard_int6 if f["bf16_also_flipped"]]

    summary = {
        "stage": "NC17-P3",
        "engine_build_int6_chain": mi.get("engine_build"),
        "engine_build_bf16_chain": mb.get("engine_build"),
        "gt_file": str(GT_PATH), "gt_sha256": gt_sha,
        "near_tie_margin": NEAR_TIE_MARGIN,
        "int6_parity_vs_gt": {
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
        "int6_vs_p1bf16_same_engine": {
            "note": "cached-chain, same FORK engine, both chains drift "
                    "identically -> isolates INT6 quantization noise (spec note)",
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
            "n_hard_int6_vs_gt": len(hard_int6),
            "n_hard_shared_with_bf16": len(hard_int6_shared),
            "n_hard_int6_only": len(hard_int6_only),
            "hard_int6_only_positions": [{"prompt": f["prompt"], "pos": f["pos"],
                                          "gt_margin": f["gt_margin"]}
                                         for f in hard_int6_only],
            "note": "shared = flip already present in P1/fork bf16 (drift/port "
                    "carried through, not INT6-introduced); int6_only = candidate "
                    "quant-noise flips -> p3_adjudicate fresh-refeeds them",
        },
        "per_prompt": per_prompt,
    }
    OUT.write_text(json.dumps(summary, indent=2))

    print("\n=== NC17-P3 PARITY (tc-INT6 fork cached-chain vs P0 GT) ===", flush=True)
    print(f"top-1 agreement: {n_agree}/{n_pos} = {n_agree/n_pos:.4f}", flush=True)
    print(f"flips: {n_flips} total | {n_nt} near-tie (<{NEAR_TIE_MARGIN}) | {n_hard} HARD", flush=True)
    for f in flips:
        tag = "near-tie" if f["near_tie"] else "*** HARD FLIP ***"
        shr = " [bf16-shared]" if f["bf16_also_flipped"] else " [INT6-only]"
        print(f"  flip p{f['prompt']} {f['pos']}: GT {f['gt_top1']} -> int6 "
              f"{f['tc_top1']} | GT margin {f['gt_margin']:.4f} [{tag}]{shr}", flush=True)
    print(f"INT6-vs-GT max|dlogit|={allabs.max():.4f} mean={allabs.mean():.4f} "
          f"p99={np.percentile(allabs,99):.4f}", flush=True)
    print(f"\n=== INT6-vs-P1bf16 SAME-ENGINE(fork) quant-noise (clean) ===", flush=True)
    print(f"per-position max|delta|: max={qabs.max():.4f} mean={qabs.mean():.4f} "
          f"p50={np.percentile(qabs,50):.4f} p99={np.percentile(qabs,99):.4f}", flush=True)
    print(f"full-vector |delta|: mean={qall.mean():.5f} p99={np.percentile(qall,99):.4f} "
          f"p99.9={np.percentile(qall,99.9):.4f} max={qall.max():.4f}", flush=True)
    print(f"\n=== HARD-FLIP ATTRIBUTION ===", flush=True)
    print(f"INT6 HARD flips vs GT: {len(hard_int6)} | shared-with-bf16(drift): "
          f"{len(hard_int6_shared)} | INT6-ONLY(quant candidates): {len(hard_int6_only)}",
          flush=True)
    verdict = "PASS" if n_hard == 0 else "FAIL"
    print(f"\n[parity] INT6-vs-GT cached-chain GATE VERDICT: {verdict}", flush=True)
    print(f"[parity] wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
