#!/usr/bin/env python3
"""NC17-P1b adjudication: cache-chain drift vs port defect for the P1 parity RED.

The P1 gate (tests/nc17/p1_parity.py) teacher-forces the 64 decode steps through
the tc adapter as a CACHED chain (feed one token at a time, reuse `caches` +
position_offset). It failed with 17 HARD flips (GT margin > NEAR_TIE) concentrated
late in decode. Hypothesis under test: those flips are accumulated bf16 KV-cache-
chain drift, not a port defect.

This test isolates cache reuse from the port. For EVERY one of the 17 HARD flips
(prompt p, decode step s), we rebuild the byte-identical teacher-forced context
    seq = prompt_ids[p] + decode_tokens[p][0..s]   (inclusive of step s)
and run it through the SAME adapter as ONE FRESH PREFILL (last_token_only, no cache
carried across steps, fresh forward per sequence). The final-position logits from
that fresh prefill are compared to the GT's decode_logits[p][s].

  GT decode indexing (from p0_gt_mint.py):
    decode_tokens[s] = token FED at step s (s=0 -> argmax of final_logits)
    decode_logits[s] = logits AFTER feeding decode_tokens[s]
  => the input context that produces decode_logits[s] is
     prompt_ids + decode_tokens[0..s] inclusive.

Verdict rules (registered before running):
  A flip is EXPLAINED if the fresh-refeed top-1 agrees with GT top-1 at that
  position, OR disagrees only within the 0.25 near-tie margin (fresh_margin < 0.25).
  >=15/17 EXPLAINED  -> cache-chain drift CONFIRMED; "PORT SOUND, gate-fail explained".
  >=3/17 UNEXPLAINED  -> "PORT DEFECT SUSPECTED"; emit per-flip table, DO NOT fix.

Control: 10 randomly chosen AGREEING positions (seed 42) get the same fresh-refeed
check; fresh-refeed should agree there too.

Matched-reference law: names the exact GT file + sha256 consumed.
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
PARITY = REPO / "logs" / "nc17" / "p1_parity.json"
OUT = REPO / "logs" / "nc17" / "p1b_verdict.json"
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
    """Run seq_ids (1D int64 array) as ONE fresh prefill; return final-pos logits
    as float32. No cache carried in or out. Fresh forward per call."""
    ids = seq_ids[None, :].astype(np.int64)
    logits, _caches = m(ids, last_token_only=True)
    return logits.numpy()[0, -1].astype(np.float32)


def parse_step(pos):
    assert pos.startswith("decode"), f"unexpected pos {pos!r}"
    return int(pos[len("decode"):])


def main():
    gt_sha = sha256_file(GT_PATH)
    print(f"[p1b] GT file: {GT_PATH}", flush=True)
    print(f"[p1b] GT sha256: {gt_sha}", flush=True)
    if gt_sha != EXPECTED_GT_SHA:
        print(f"[p1b] *** GT sha mismatch vs P1 ({EXPECTED_GT_SHA}) — ABORT", flush=True)
        return 2

    gt = np.load(GT_PATH, allow_pickle=False)
    parity = json.loads(PARITY.read_text())
    hard = [f for f in parity["flips"] if not f["near_tie"]]
    print(f"[p1b] hard flips from p1_parity.json: {len(hard)}", flush=True)

    m, info = Qwen3_1p7b_TC.from_pretrained(attention_mode="standard")
    print(f"[p1b] adapter: {info}", flush=True)

    # ---- adjudicate the 17 HARD flips ----
    rows = []
    n_explained = 0
    for f in hard:
        p = f["prompt"]
        s = parse_step(f["pos"])
        pid = gt[f"prompt_ids_{p}"].astype(np.int64)
        dec_tokens = gt[f"decode_tokens_{p}"].astype(np.int64)
        gt_dec_logits = gt[f"decode_logits_{p}"].astype(np.float32)

        # Context producing decode_logits[s] = prompt + decode_tokens[0..s] inclusive.
        seq = np.concatenate([pid, dec_tokens[: s + 1]]).astype(np.int64)
        fresh = fresh_prefill_final(m, seq)

        gt_step = gt_dec_logits[s]
        gt_t1 = int(gt_step.argmax())
        fresh_t1 = int(fresh.argmax())
        cached_t1 = f["tc_top1"]
        gt_margin = f["gt_margin"]
        fresh_margin = top2_margin(fresh)
        # sanity: gt_t1 from npz must match the recorded gt_top1 in the parity json
        assert gt_t1 == f["gt_top1"], (
            f"gt_top1 mismatch p{p} {f['pos']}: npz {gt_t1} vs json {f['gt_top1']}")

        agree = fresh_t1 == gt_t1
        # near-tie: fresh disagrees but the fresh distribution itself is a near tie
        near_tie = (not agree) and (fresh_margin < NEAR_TIE_MARGIN)
        explained = agree or near_tie
        if explained:
            n_explained += 1
        verdict = ("EXPLAINED(agree)" if agree else
                   "EXPLAINED(near-tie)" if near_tie else "UNEXPLAINED")
        rows.append({
            "prompt": p, "step": s,
            "gt_top1": gt_t1, "cached_top1": cached_t1, "fresh_top1": fresh_t1,
            "gt_margin": round(gt_margin, 4), "fresh_margin": round(fresh_margin, 4),
            "fresh_agrees": agree, "verdict": verdict,
        })
        print(f"[p1b] flip p{p} decode{s}: GT {gt_t1} cached {cached_t1} "
              f"fresh {fresh_t1} | gt_m {gt_margin:.3f} fresh_m {fresh_margin:.3f} "
              f"-> {verdict}", flush=True)

    n_unexplained = len(hard) - n_explained

    # ---- control: 10 random AGREEING positions (seed 42) ----
    # Build the full set of agreeing (prompt, step) decode positions: every decode
    # step that is NOT in the flips list (flips include both near-tie and hard).
    flip_set = {(f["prompt"], parse_step(f["pos"]))
                for f in parity["flips"] if f["pos"].startswith("decode")}
    n_prompts = sum(1 for k in gt.files if k.startswith("prompt_ids_"))
    agreeing = []
    for p in range(n_prompts):
        n_steps = gt[f"decode_tokens_{p}"].shape[0]
        for s in range(n_steps):
            if (p, s) not in flip_set:
                agreeing.append((p, s))
    rng = random.Random(CONTROL_SEED)
    control_sample = rng.sample(agreeing, min(N_CONTROL, len(agreeing)))
    control_sample.sort()

    control_rows = []
    n_control_agree = 0
    for (p, s) in control_sample:
        pid = gt[f"prompt_ids_{p}"].astype(np.int64)
        dec_tokens = gt[f"decode_tokens_{p}"].astype(np.int64)
        gt_step = gt[f"decode_logits_{p}"].astype(np.float32)[s]
        seq = np.concatenate([pid, dec_tokens[: s + 1]]).astype(np.int64)
        fresh = fresh_prefill_final(m, seq)
        gt_t1 = int(gt_step.argmax())
        fresh_t1 = int(fresh.argmax())
        gt_margin = top2_margin(gt_step)
        fresh_margin = top2_margin(fresh)
        agree = fresh_t1 == gt_t1
        near_tie = (not agree) and (fresh_margin < NEAR_TIE_MARGIN)
        ok = agree or near_tie
        if ok:
            n_control_agree += 1
        verdict = ("OK(agree)" if agree else
                   "OK(near-tie)" if near_tie else "CONTROL-FLIP")
        control_rows.append({
            "prompt": p, "step": s, "gt_top1": gt_t1, "fresh_top1": fresh_t1,
            "gt_margin": round(gt_margin, 4), "fresh_margin": round(fresh_margin, 4),
            "fresh_agrees": agree, "verdict": verdict,
        })
        print(f"[p1b] control p{p} decode{s}: GT {gt_t1} fresh {fresh_t1} "
              f"| gt_m {gt_margin:.3f} fresh_m {fresh_margin:.3f} -> {verdict}",
              flush=True)

    # ---- verdict ----
    if n_explained >= 15:
        headline = "PORT SOUND, gate-fail explained"
        mechanism = "cache-chain drift CONFIRMED as mechanism"
    elif n_unexplained >= 3:
        headline = "PORT DEFECT SUSPECTED"
        mechanism = "fresh-refeed still hard-flips >=3/17"
    else:
        # 13 or 14 explained: neither registered threshold met.
        headline = "INDETERMINATE"
        mechanism = (f"{n_explained}/17 explained, {n_unexplained} unexplained — "
                     "neither the >=15 nor the >=3 threshold met")

    summary = {
        "test": "nc17_p1b_adjudicate",
        "gt_file": str(GT_PATH),
        "gt_sha256": gt_sha,
        "adapter_info": info,
        "near_tie_margin": NEAR_TIE_MARGIN,
        "method": ("fresh full-prefill of prompt+decode_tokens[0..s] per HARD flip; "
                   "no cache reuse across steps; compare final-pos top1/margin vs "
                   "GT decode_logits[s]"),
        "n_hard_flips": len(hard),
        "n_explained": n_explained,
        "n_unexplained": n_unexplained,
        "control_seed": CONTROL_SEED,
        "n_control": len(control_rows),
        "n_control_agree": n_control_agree,
        "headline": headline,
        "mechanism": mechanism,
        "flip_rows": rows,
        "control_rows": control_rows,
    }
    OUT.write_text(json.dumps(summary, indent=2))

    # ---- printed tables ----
    print("\n=== NC17-P1b ADJUDICATION: 17 HARD-FLIP FRESH-REFEED TABLE ===", flush=True)
    hdr = (f"{'prompt':>6} {'step':>4} {'GT_top1':>8} {'cached_top1':>11} "
           f"{'fresh_top1':>10} {'GT_margin':>9} {'fresh_margin':>12}  verdict")
    print(hdr, flush=True)
    for r in rows:
        print(f"{r['prompt']:>6} {r['step']:>4} {r['gt_top1']:>8} "
              f"{r['cached_top1']:>11} {r['fresh_top1']:>10} "
              f"{r['gt_margin']:>9.3f} {r['fresh_margin']:>12.3f}  {r['verdict']}",
              flush=True)
    print(f"\nEXPLAINED {n_explained}/17 | UNEXPLAINED {n_unexplained}/17", flush=True)

    print("\n=== NC17-P1b CONTROL: 10 AGREEING POSITIONS (seed 42) FRESH-REFEED ===",
          flush=True)
    chdr = (f"{'prompt':>6} {'step':>4} {'GT_top1':>8} {'fresh_top1':>10} "
            f"{'GT_margin':>9} {'fresh_margin':>12}  verdict")
    print(chdr, flush=True)
    for r in control_rows:
        print(f"{r['prompt']:>6} {r['step']:>4} {r['gt_top1']:>8} "
              f"{r['fresh_top1']:>10} {r['gt_margin']:>9.3f} "
              f"{r['fresh_margin']:>12.3f}  {r['verdict']}", flush=True)
    print(f"\nCONTROL agree {n_control_agree}/{len(control_rows)}", flush=True)

    print(f"\n[p1b] VERDICT: {headline} — {mechanism}", flush=True)
    print(f"[p1b] wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
