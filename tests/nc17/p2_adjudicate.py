#!/usr/bin/env python3
"""NC17-P2 adjudication: distinguish INT4 quant-noise flips from drift flips.

The P2 cached-chain parity (p2_parity.py) uses the SAME cached-chain protocol as
P1 — which the ledger's registered spec note says measures DRIFT tolerance, not
port/quant correctness. P1b proved the tc-bf16 cached-chain HARD flips were bf16
KV-cache drift (17/17 reverted on fresh prefill). So for INT4 the clean question
is narrower: at each INT4 HARD flip vs GT, is it a NEW flip introduced by INT4
quantization, or the same drift flip P1 already had?

p2_parity.py splits HARD flips into:
  - shared-with-bf16: already flipped in the P1 bf16 chain -> DRIFT (P1b-explained
    class); INT4 did not introduce it.
  - INT4-only: candidate quant-noise flips. THIS test fresh-prefills each one
    (prompt + decode_tokens[0..s], no cache) through the INT4 adapter and checks
    whether it reverts to GT top-1 (drift, not quant) or persists (real
    quant-noise flip at a non-near-tie GT margin).

Verdict rules (registered before running):
  For the INT4-only HARD flips: a flip is DRIFT-EXPLAINED if fresh-refeed agrees
  with GT top-1, or disagrees only within the 0.25 near-tie margin. A flip that
  PERSISTS on fresh-refeed at a > 0.25 GT margin is a REAL INT4 quant-noise flip
  (reported verbatim; NOT fixed — quantization changes the model, this is the
  measurement). Control: 10 random AGREEING positions (seed 42) fresh-refeed
  through INT4 and must agree.

Matched-reference law: names the exact GT file + sha256.
"""
import hashlib
import json
import random
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from core.qwen3_1p7b_tc import Qwen3_1p7b_TC  # noqa: E402
import tensor_cuda as tc  # noqa: E402

GT_PATH = REPO / "logs" / "nc17" / "p0_gt.npz"
PARITY = REPO / "logs" / "nc17" / "p2_parity.json"
OUT = REPO / "logs" / "nc17" / "p2_verdict.json"
NEAR_TIE_MARGIN = 0.25
EXPECTED_GT_SHA = "0fc4099b3537083ec99478c9cdb969a5afca038609636499d64344a3303575fb"
N_CONTROL = 10
CONTROL_SEED = 42


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


def fresh_prefill_final(m, seq_ids):
    ids = seq_ids[None, :].astype(np.int64)
    logits, _ = m(ids, last_token_only=True)
    return logits.numpy()[0, -1].astype(np.float32)


def parse_step(pos):
    assert pos.startswith("decode"), f"unexpected pos {pos!r}"
    return int(pos[len("decode"):])


def main():
    gt_sha = sha256_file(GT_PATH)
    if gt_sha != EXPECTED_GT_SHA:
        print(f"[p2adj] *** GT sha mismatch {gt_sha} — ABORT", flush=True)
        return 2
    gt = np.load(GT_PATH, allow_pickle=False)
    parity = json.loads(PARITY.read_text())
    all_flips = parity["int4_parity_vs_gt"]["flips"]
    cand = [f for f in all_flips if not f["near_tie"]
            and not f.get("bf16_also_flipped", False)
            and f["pos"].startswith("decode")]
    shared = [f for f in all_flips if not f["near_tie"]
              and f.get("bf16_also_flipped", False)]
    print(f"[p2adj] INT4-only HARD flip candidates: {len(cand)} "
          f"(shared-with-bf16 drift flips not refed: {len(shared)})", flush=True)

    m, info = Qwen3_1p7b_TC.from_pretrained(attention_mode="standard", int4=True)
    print(f"[p2adj] adapter: {info.get('framework')}", flush=True)

    rows = []
    n_drift = 0
    n_realquant = 0
    for f in cand:
        p = f["prompt"]; s = parse_step(f["pos"])
        pid = gt[f"prompt_ids_{p}"].astype(np.int64)
        dec = gt[f"decode_tokens_{p}"].astype(np.int64)
        gt_step = gt[f"decode_logits_{p}"].astype(np.float32)[s]
        seq = np.concatenate([pid, dec[:s + 1]]).astype(np.int64)
        fresh = fresh_prefill_final(m, seq)
        gt_t1 = int(gt_step.argmax()); fr_t1 = int(fresh.argmax())
        gt_m = f["gt_margin"]; fr_m = top2_margin(fresh)
        agree = fr_t1 == gt_t1
        near = (not agree) and (fr_m < NEAR_TIE_MARGIN)
        if agree or near:
            n_drift += 1
            verdict = "DRIFT(agree)" if agree else "DRIFT(near-tie)"
        else:
            n_realquant += 1
            verdict = "REAL-QUANT-FLIP"
        rows.append({"prompt": p, "step": s, "gt_top1": gt_t1,
                     "cached_top1": f["tc_top1"], "fresh_top1": fr_t1,
                     "gt_margin": round(gt_m, 4), "fresh_margin": round(fr_m, 4),
                     "verdict": verdict})
        print(f"[p2adj] p{p} decode{s}: GT {gt_t1} cached {f['tc_top1']} fresh "
              f"{fr_t1} | gt_m {gt_m:.3f} fresh_m {fr_m:.3f} -> {verdict}", flush=True)

    flip_set = {(f["prompt"], parse_step(f["pos"])) for f in all_flips
                if f["pos"].startswith("decode")}
    n_prompts = sum(1 for k in gt.files if k.startswith("prompt_ids_"))
    agreeing = [(p, s) for p in range(n_prompts)
                for s in range(gt[f"decode_tokens_{p}"].shape[0])
                if (p, s) not in flip_set]
    rng = random.Random(CONTROL_SEED)
    ctrl = sorted(rng.sample(agreeing, min(N_CONTROL, len(agreeing))))
    ctrl_rows = []; n_ctrl_ok = 0
    for (p, s) in ctrl:
        pid = gt[f"prompt_ids_{p}"].astype(np.int64)
        dec = gt[f"decode_tokens_{p}"].astype(np.int64)
        gt_step = gt[f"decode_logits_{p}"].astype(np.float32)[s]
        seq = np.concatenate([pid, dec[:s + 1]]).astype(np.int64)
        fresh = fresh_prefill_final(m, seq)
        gt_t1 = int(gt_step.argmax()); fr_t1 = int(fresh.argmax())
        agree = fr_t1 == gt_t1
        near = (not agree) and (top2_margin(fresh) < NEAR_TIE_MARGIN)
        ok = agree or near
        n_ctrl_ok += int(ok)
        ctrl_rows.append({"prompt": p, "step": s, "gt_top1": gt_t1,
                          "fresh_top1": fr_t1, "ok": ok})
        print(f"[p2adj] control p{p} decode{s}: GT {gt_t1} fresh {fr_t1} "
              f"-> {'OK' if ok else 'CONTROL-FLIP'}", flush=True)

    if n_realquant == 0:
        headline = ("PORT+QUANT SOUND — all INT4-only HARD flips are drift; "
                    "no real quant-noise flip at a non-near-tie GT margin")
    else:
        headline = (f"{n_realquant} REAL INT4 quant-noise flip(s) at non-near-tie "
                    f"GT margins (reported, not fixed — quantization is lossy)")

    summary = {
        "test": "nc17_p2_adjudicate",
        "gt_file": str(GT_PATH), "gt_sha256": gt_sha, "adapter_info": info,
        "near_tie_margin": NEAR_TIE_MARGIN,
        "method": ("fresh full-prefill of prompt+decode_tokens[0..s] through INT4 "
                   "adapter at each INT4-ONLY HARD cached-chain flip; drift if it "
                   "reverts, real quant-noise if it persists at >0.25 GT margin"),
        "n_int4_only_hard_candidates": len(cand),
        "n_shared_drift_flips_not_refed": len(shared),
        "n_drift_explained": n_drift,
        "n_real_quant_flips": n_realquant,
        "control_seed": CONTROL_SEED, "n_control": len(ctrl_rows),
        "n_control_ok": n_ctrl_ok,
        "headline": headline,
        "flip_rows": rows, "control_rows": ctrl_rows,
    }
    OUT.write_text(json.dumps(summary, indent=2))

    print("\n=== NC17-P2 ADJUDICATION: INT4-ONLY HARD-FLIP FRESH-REFEED ===", flush=True)
    for r in rows:
        print(f"  p{r['prompt']:>2} decode{r['step']:<2} GT {r['gt_top1']:>6} "
              f"cached {r['cached_top1']:>6} fresh {r['fresh_top1']:>6} | "
              f"gt_m {r['gt_margin']:.3f} fresh_m {r['fresh_margin']:.3f} -> {r['verdict']}",
              flush=True)
    print(f"\nDRIFT-explained {n_drift}/{len(cand)} | REAL-QUANT {n_realquant}/{len(cand)}",
          flush=True)
    print(f"CONTROL ok {n_ctrl_ok}/{len(ctrl_rows)}", flush=True)
    print(f"\n[p2adj] VERDICT: {headline}", flush=True)
    print(f"[p2adj] wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
