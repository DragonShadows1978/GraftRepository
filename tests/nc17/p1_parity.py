#!/usr/bin/env python3
"""NC17-P1 parity gate: tensor_cuda Qwen3-1.7B bf16 (standard attn) vs P0 GT.

Matched-reference law: names the exact GT file + sha256 used.
Margin protocol (plan): top-1 agreement on the GT prompt set; a top-1 FLIP is
acceptable ONLY at a near-tie GT margin (GT logit gap between its top-1 and
top-2 small). Every flip is reported with its GT margin. max|Δlogit|
distribution reported across all compared positions.

Positions compared per prompt:
  - final prefill position (GT final_logits_i)
  - the 64 decode positions, TEACHER-FORCED on GT decode_tokens_i so the tc
    model sees byte-identical context to the GT capture at every step (this is
    the only way per-step logits are comparable across stacks; free-running
    would diverge after the first flip and stop being a matched comparison).
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from core.qwen3_1p7b_tc import Qwen3_1p7b_TC  # noqa: E402
import tensor_cuda as tc  # noqa: E402

GT_PATH = REPO / "logs" / "nc17" / "p0_gt.npz"
OUT = REPO / "logs" / "nc17" / "p1_parity.json"
# Near-tie threshold: a GT margin (top1 - top2 logit) below this makes a top-1
# flip acceptable (registered here, before running the gate).
NEAR_TIE_MARGIN = 0.25


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
    gt = np.load(GT_PATH, allow_pickle=False)
    n_prompts = sum(1 for k in gt.files if k.startswith("prompt_ids_"))
    print(f"[parity] GT file: {GT_PATH}", flush=True)
    print(f"[parity] GT sha256: {gt_sha}", flush=True)
    print(f"[parity] n_prompts: {n_prompts}  near_tie_margin: {NEAR_TIE_MARGIN}", flush=True)

    m, info = Qwen3_1p7b_TC.from_pretrained(attention_mode="standard")
    print(f"[parity] adapter: {info}", flush=True)

    all_abs_dlogit = []
    n_positions = 0
    n_top1_agree = 0
    flips = []          # every top-1 flip, with GT margin + whether near-tie
    per_prompt = []

    for i in range(n_prompts):
        pid = gt[f"prompt_ids_{i}"].astype(np.int64)
        gt_final = gt[f"final_logits_{i}"].astype(np.float32)
        gt_dec_logits = gt[f"decode_logits_{i}"].astype(np.float32)
        gt_dec_tokens = gt[f"decode_tokens_{i}"].astype(np.int64)

        # Full prefill, capture ALL positions' logits (last_token_only=False) so
        # we get the final-position logits and can seed the decode cache.
        ids = pid[None, :]
        logits, caches = m(ids, last_token_only=False)
        tc_final = logits.numpy()[0, -1].astype(np.float32)

        # ---- final prefill position ----
        d = tc_final - gt_final
        all_abs_dlogit.append(np.abs(d))
        n_positions += 1
        gt_t1 = int(gt_final.argmax()); tc_t1 = int(tc_final.argmax())
        margin = top2_margin(gt_final)
        if tc_t1 == gt_t1:
            n_top1_agree += 1
        else:
            flips.append({"prompt": i, "pos": "final", "gt_top1": gt_t1,
                          "tc_top1": tc_t1, "gt_margin": margin,
                          "near_tie": margin < NEAR_TIE_MARGIN})

        # ---- teacher-forced decode: feed GT decode tokens one at a time ----
        pos = pid.shape[0]
        prompt_agree = 1 if tc_t1 == gt_t1 else 0
        prompt_positions = 1
        prompt_maxabs = float(np.abs(d).max())
        for step in range(gt_dec_tokens.shape[0]):
            cur = np.array([[int(gt_dec_tokens[step])]], dtype=np.int64)
            slog, caches = m(cur, kv_caches=caches, position_offset=pos,
                             last_token_only=True)
            pos += 1
            tc_step = slog.numpy()[0, -1].astype(np.float32)
            gt_step = gt_dec_logits[step]
            dd = tc_step - gt_step
            all_abs_dlogit.append(np.abs(dd))
            n_positions += 1
            prompt_positions += 1
            prompt_maxabs = max(prompt_maxabs, float(np.abs(dd).max()))
            g1 = int(gt_step.argmax()); t1 = int(tc_step.argmax())
            mg = top2_margin(gt_step)
            if t1 == g1:
                n_top1_agree += 1
                prompt_agree += 1
            else:
                flips.append({"prompt": i, "pos": f"decode{step}", "gt_top1": g1,
                              "tc_top1": t1, "gt_margin": mg,
                              "near_tie": mg < NEAR_TIE_MARGIN})

        per_prompt.append({
            "prompt": i, "n_positions": prompt_positions,
            "top1_agree": prompt_agree,
            "agree_frac": prompt_agree / prompt_positions,
            "max_abs_dlogit": prompt_maxabs,
        })
        print(f"[parity] prompt {i}: agree {prompt_agree}/{prompt_positions} "
              f"max|dlogit|={prompt_maxabs:.4f}", flush=True)

    allabs = np.concatenate(all_abs_dlogit)
    agree_frac = n_top1_agree / n_positions
    n_flips = len(flips)
    n_neartie_flips = sum(1 for f in flips if f["near_tie"])
    n_hard_flips = n_flips - n_neartie_flips

    summary = {
        "gt_file": str(GT_PATH),
        "gt_sha256": gt_sha,
        "adapter_info": info,
        "near_tie_margin": NEAR_TIE_MARGIN,
        "n_positions": n_positions,
        "n_top1_agree": n_top1_agree,
        "top1_agreement": agree_frac,
        "n_flips": n_flips,
        "n_near_tie_flips": n_neartie_flips,
        "n_hard_flips": n_hard_flips,
        "flips": flips,
        "max_abs_dlogit": float(allabs.max()),
        "mean_abs_dlogit": float(allabs.mean()),
        "p50_abs_dlogit": float(np.percentile(allabs, 50)),
        "p99_abs_dlogit": float(np.percentile(allabs, 99)),
        "p999_abs_dlogit": float(np.percentile(allabs, 99.9)),
        "per_prompt": per_prompt,
    }
    OUT.write_text(json.dumps(summary, indent=2))

    print("\n=== NC17-P1 PARITY (tc-bf16 standard vs P0 GT) ===", flush=True)
    print(f"top-1 agreement: {n_top1_agree}/{n_positions} = {agree_frac:.4f}", flush=True)
    print(f"flips: {n_flips} total | {n_neartie_flips} near-tie (<{NEAR_TIE_MARGIN}) "
          f"| {n_hard_flips} HARD", flush=True)
    for f in flips:
        tag = "near-tie" if f["near_tie"] else "*** HARD FLIP ***"
        print(f"  flip p{f['prompt']} {f['pos']}: GT {f['gt_top1']} -> tc "
              f"{f['tc_top1']} | GT margin {f['gt_margin']:.4f} [{tag}]", flush=True)
    print(f"max|dlogit|={summary['max_abs_dlogit']:.4f} "
          f"mean={summary['mean_abs_dlogit']:.4f} "
          f"p50={summary['p50_abs_dlogit']:.4f} "
          f"p99={summary['p99_abs_dlogit']:.4f} "
          f"p99.9={summary['p999_abs_dlogit']:.4f}", flush=True)
    print(f"[parity] wrote {OUT}", flush=True)
    # Gate verdict: PASS if no HARD flips (near-tie flips allowed by protocol).
    verdict = "PASS" if n_hard_flips == 0 else "FAIL"
    print(f"[parity] VERDICT: {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
