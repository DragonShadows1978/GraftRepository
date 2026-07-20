#!/usr/bin/env python3
"""NC17-P2 cached-chain logit capture (ONE process per weight-format).

Runs the EXACT P1 parity protocol (full prefill final-position + 64 teacher-forced
decode steps as a CACHED chain, feeding GT decode_tokens) through the tc adapter
in a chosen weight format, and dumps every compared position's full logit vector
to an npz. Two captures (--fmt int4 / --fmt bf16) let the CPU differ compute:
  (a) parity vs GT (INT4 flips + margins) — same as P1's table, and
  (b) INT4-vs-P1-bf16 SAME-ENGINE cached-chain max|Delta logit| distribution (both
      chains drift identically, so the delta isolates quantization noise).

--fmt {int4,bf16}. Writes logs/nc17/p2_chain_<fmt>.npz with keys:
  _meta (fmt, gt_sha), and per (prompt i, position) the logit vector, indexed as
  logits_<i>_final and logits_<i>_dec<step>.
Bounded by the 590s cap; both formats fit the 11 GT prompts x 65 positions.
"""
import argparse
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
EXPECTED_GT_SHA = "0fc4099b3537083ec99478c9cdb969a5afca038609636499d64344a3303575fb"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", required=True, choices=["int4", "bf16"])
    args = ap.parse_args()

    gt_sha = sha256_file(GT_PATH)
    if gt_sha != EXPECTED_GT_SHA:
        print(f"[capture] *** GT sha mismatch {gt_sha} — ABORT", flush=True)
        return 2
    gt = np.load(GT_PATH, allow_pickle=False)
    n_prompts = sum(1 for k in gt.files if k.startswith("prompt_ids_"))

    m, info = Qwen3_1p7b_TC.from_pretrained(attention_mode="standard",
                                            int4=(args.fmt == "int4"))
    print(f"[capture] fmt={args.fmt} adapter={info.get('framework')}", flush=True)

    out = {}
    for i in range(n_prompts):
        pid = gt[f"prompt_ids_{i}"].astype(np.int64)
        gt_dec_tokens = gt[f"decode_tokens_{i}"].astype(np.int64)
        ids = pid[None, :]
        logits, caches = m(ids, last_token_only=False)
        out[f"logits_{i}_final"] = logits.numpy()[0, -1].astype(np.float32)
        pos = pid.shape[0]
        for step in range(gt_dec_tokens.shape[0]):
            cur = np.array([[int(gt_dec_tokens[step])]], dtype=np.int64)
            slog, caches = m(cur, kv_caches=caches, position_offset=pos,
                             last_token_only=True)
            pos += 1
            out[f"logits_{i}_dec{step}"] = slog.numpy()[0, -1].astype(np.float32)
        print(f"[capture] prompt {i}: {gt_dec_tokens.shape[0]} decode steps", flush=True)

    out["_meta"] = np.frombuffer(
        json.dumps({"fmt": args.fmt, "gt_sha256": gt_sha,
                    "framework": info.get("framework")}).encode(), dtype=np.uint8)
    OUT = REPO / "logs" / "nc17" / f"p2_chain_{args.fmt}.npz"
    np.savez(OUT, **out)
    print(f"[capture] wrote {OUT} ({n_prompts} prompts)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
