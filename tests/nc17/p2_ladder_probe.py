#!/usr/bin/env python3
"""NC17-P2 context-ceiling probe — INT4 weights. ONE PROCESS PER PROBE (Gemma
law). Identical protocol to p1_ladder_probe.py; the only change is int4=True in
from_pretrained (projections are QuantLinearTC INT4, residual/KV bf16).

Args: --mode {standard,apa} --kind {prefill,decode} --ctx L [--refine R]
prefill: one forward of an L-token block (last_token_only=True).
decode : chunked-prefill (L-1) then ONE cached decode step at pos L-1.
On CUDA OOM -> status=OOM (measurement). Always exits 0.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests" / "nc17"))
from core.qwen3_1p7b_tc import Qwen3_1p7b_TC  # noqa: E402
import tensor_cuda as tc  # noqa: E402
import p1_apa_engage as engage  # noqa: E402


def smi_used_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=10).decode().strip().splitlines()
        return float(out[0])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["standard", "apa"])
    ap.add_argument("--kind", required=True, choices=["prefill", "decode"])
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--refine", type=float, default=0.15)
    args = ap.parse_args()
    L = args.ctx
    if args.mode == "apa":
        engage.install(); engage.reset()

    t0 = time.time()
    status = "OK"; err = ""; smi_after = None
    try:
        m, info = Qwen3_1p7b_TC.from_pretrained(attention_mode=args.mode, int4=True)
        if args.mode == "apa":
            m.set_attention_mode("apa", refine_percentile=args.refine)
        m.extend_rope(L + 8)
        rng = np.random.default_rng(0)
        if args.kind == "prefill":
            block = rng.integers(0, 151936, size=(1, L), dtype=np.int64)
            logits, _ = m(block, last_token_only=True)
            _ = logits.numpy()
        else:  # chunked KV build then ONE cached decode step (matches p1 probe)
            CHUNK = 1024
            caches = None; pos = 0; remaining = L - 1
            while remaining > 0:
                c = min(CHUNK, remaining)
                blk = rng.integers(0, 151936, size=(1, c), dtype=np.int64)
                logits, caches = m(blk, kv_caches=caches, position_offset=pos,
                                   last_token_only=True)
                _ = logits.numpy(); pos += c; remaining -= c
            cur = rng.integers(0, 151936, size=(1, 1), dtype=np.int64)
            slog, _ = m(cur, kv_caches=caches, position_offset=pos,
                        last_token_only=True)
            _ = slog.numpy()
        smi_after = smi_used_mb()
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        low = msg.lower()
        if any(s in low for s in ("out of memory", "oom", "cuda_error_out",
                                  "alloc", "cublas_status_alloc",
                                  "cuda error", "cudamalloc")):
            status = "OOM"
        else:
            status = "ERR"
        err = msg

    eng = engage.summary() if args.mode == "apa" else {"APA_ENGAGED": None}
    rec = {"mode": args.mode, "kind": args.kind, "ctx": L, "status": status,
           "smi_used_mb_after": smi_after, "elapsed_s": time.time() - t0,
           "engagement": eng, "error": err}
    print("RESULT " + json.dumps(rec), flush=True)
    print(f"RESULT-LINE mode={args.mode} kind={args.kind} ctx={L} "
          f"status={status} smi_used_mb={smi_after}", flush=True)
    if args.mode == "apa" and status == "OK":
        print(f"APA-ENGAGE engaged={eng['APA_ENGAGED']} calls={eng['apa_calls']} "
              f"engaged_calls={eng['engaged_calls']} mean_frac={eng['mean_engaged_frac']}",
              flush=True)
    if err:
        print("ERRTEXT " + err[:500], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
