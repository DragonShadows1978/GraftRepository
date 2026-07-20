#!/usr/bin/env python3
"""NC17-P3 cached-chain logit capture (ONE process per weight-format), FORK engine.

Runs the EXACT P1/P2 parity protocol (full prefill final-position + 64
teacher-forced decode steps as a CACHED chain, feeding GT decode_tokens) through
the tc adapter in a chosen weight format on the INT6 fork engine, and dumps every
compared position's full logit vector to an npz.

Two captures let the CPU differ compute:
  (a) parity vs GT (INT6 flips + margins) — same table as P1/P2, and
  (b) INT6-vs-P1-bf16 SAME-ENGINE cached-chain max|Delta logit| distribution
      (both chains drift identically on the SAME fork engine, so the delta
      isolates INT6 quantization noise from port/drift noise).

IMPORTANT — matched-engine law for the clean delta: BOTH the int6 and bf16
captures run on the FORK engine here (--fmt int6 / --fmt bf16), so the
same-engine delta compares int6-fork vs bf16-fork. (P2's clean delta compared
int4-canonical vs bf16-canonical. The bf16 numbers should be ~identical across
builds — the fork touched only the int6 surface — but we capture bf16 on the fork
so the P3 same-engine isolation is genuinely same-engine.)

--fmt {int6,bf16}. Writes logs/nc17/p3_chain_<fmt>.npz.
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
import tensor_cuda as tc  # noqa: E402
from core.qwen3_1p7b_tc import Qwen3_1p7b_TC  # noqa: E402

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
    ap.add_argument("--fmt", required=True, choices=["int6", "bf16"])
    args = ap.parse_args()
    print(f"[capture] engine tc: {tc.__file__}", flush=True)
    assert "Project-Tensor-int6" in tc.__file__, (
        f"REFUSING: tc is not the fork int6 build: {tc.__file__}")

    gt_sha = sha256_file(GT_PATH)
    if gt_sha != EXPECTED_GT_SHA:
        print(f"[capture] *** GT sha mismatch {gt_sha} — ABORT", flush=True)
        return 2
    gt = np.load(GT_PATH, allow_pickle=False)
    n_prompts = sum(1 for k in gt.files if k.startswith("prompt_ids_"))

    m, info = Qwen3_1p7b_TC.from_pretrained(attention_mode="standard",
                                            int6=(args.fmt == "int6"))
    print(f"[capture] ENGINE=int6-fork fmt={args.fmt} adapter={info.get('framework')}",
          flush=True)

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
                    "engine_build": tc.__file__,
                    "framework": info.get("framework")}).encode(), dtype=np.uint8)
    OUT = REPO / "logs" / "nc17" / f"p3_chain_{args.fmt}.npz"
    np.savez(OUT, **out)
    print(f"[capture] wrote {OUT} ({n_prompts} prompts)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
