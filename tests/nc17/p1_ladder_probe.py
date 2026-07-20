#!/usr/bin/env python3
"""NC17-P1 context-ceiling probe — ONE PROCESS PER PROBE (Gemma law:
fragmentation fakes OOMs across contexts within a process, so each length gets a
fresh CUDA context).

Args: --mode {standard,apa} --kind {prefill,decode} --ctx L [--refine R]
Loads the model, attempts a single forward at context length L:
  prefill: one forward of an L-token block (last_token_only=True).
  decode : prefill an (L-1)-token block, then ONE cached decode step at pos L-1.

The engine exposes NO framework allocator-peak accessor (only set_alloc_pooling),
so peak fill is the run_gpu.sh 1s nvidia-smi poller (authoritative, A0 law). This
process also snapshots nvidia-smi used-MiB right after the forward as a secondary.
On CUDA OOM -> status=OOM (measurement, not failure). The driver binary-searches
L across processes. Always exits 0 (OOM is a measurement).
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
    status = "OK"
    err = ""
    smi_after = None
    try:
        m, info = Qwen3_1p7b_TC.from_pretrained(attention_mode=args.mode)
        if args.mode == "apa":
            m.set_attention_mode("apa", refine_percentile=args.refine)
        m.extend_rope(L + 8)
        rng = np.random.default_rng(0)
        if args.kind == "prefill":
            block = rng.integers(0, 151936, size=(1, L), dtype=np.int64)
            logits, _ = m(block, last_token_only=True)
            _ = logits.numpy()
        else:  # decode ceiling: build the KV cache in CHUNKS to reach contexts
            # the single-shot L^2 prefill can't hold, then do ONE cached decode
            # step attending over the whole span. This isolates the KV-cache-bound
            # decode wall from the prefill (score-matrix) wall.
            CHUNK = 1024
            caches = None
            pos = 0
            remaining = L - 1
            while remaining > 0:
                c = min(CHUNK, remaining)
                blk = rng.integers(0, 151936, size=(1, c), dtype=np.int64)
                logits, caches = m(blk, kv_caches=caches, position_offset=pos,
                                   last_token_only=True)
                _ = logits.numpy()
                pos += c
                remaining -= c
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
