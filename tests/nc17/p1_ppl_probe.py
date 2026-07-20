#!/usr/bin/env python3
"""NC17-P1 single perplexity probe (ONE process per mode x window).

Args: --mode {standard,apa} --window W --stride S [--max-windows N] [--refine R]
Sliding-window ppl over logs/nc17/p0_ppl_tokens.npz (the P0 GT tokenization,
identical across stages). For each window: prefill the W-token block, take
per-position logits (last_token_only=False), compute NLL of the next token;
count each target token exactly once (non-overlapping scoring across the stride)
— standard sliding protocol, stride 512.

APA mode installs the engagement instrument and ASSERTS engagement (June law):
the printed summary carries APA_ENGAGED and the engaged-fraction stats. A run
whose APA mask never engages is flagged APA_ENGAGED=false and is NOT a valid
APA datapoint.

Bounded <=590s by --max-windows / a soft 540s wall. Writes
logs/nc17/p1_ppl_<mode>_w<W>.json.
"""
import argparse
import json
import math
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


def per_pos_nll(logits_np, targets_np):
    """fp32 NLL (nats) of targets under logits. logits (T,V), targets (T,)."""
    xf = logits_np.astype(np.float64)
    m = xf.max(axis=1, keepdims=True)
    lse = np.log(np.exp(xf - m).sum(axis=1)) + m[:, 0]
    chosen = xf[np.arange(xf.shape[0]), targets_np]
    return lse - chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["standard", "apa"])
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--max-windows", type=int, default=8)
    ap.add_argument("--refine", type=float, default=0.15)
    args = ap.parse_args()

    toks = np.load(REPO / "logs" / "nc17" / "p0_ppl_tokens.npz")["token_ids"].astype(np.int64)
    N = toks.shape[0]
    W, stride = args.window, args.stride

    if args.mode == "apa":
        engage.install()
        engage.reset()

    t0 = time.time()
    m, info = Qwen3_1p7b_TC.from_pretrained(attention_mode=args.mode)
    if args.mode == "apa":
        m.set_attention_mode("apa", refine_percentile=args.refine)
    m.extend_rope(W + 8)
    print(f"[ppl] mode={args.mode} W={W} stride={stride} refine={args.refine} "
          f"load={time.time()-t0:.1f}s info={info.get('framework')}", flush=True)

    total_nll = 0.0
    total_tokens = 0
    n_windows = 0
    prev_end = 0
    status = "OK"
    err = ""
    try:
        for start in range(0, N - 1, stride):
            end = min(start + W, N)
            if end - start < 2:
                break
            block = toks[start:end][None, :]           # (1, w)
            logits, _ = m(block, last_token_only=False)
            lg = logits.numpy()[0].astype(np.float32)  # (w, V)
            w = end - start
            pred_logits = lg[:w - 1]                    # predict pos t -> token t+1
            targets = toks[start + 1:start + w]
            target_abs_pos = np.arange(start + 1, start + w)
            keep = target_abs_pos >= prev_end          # count each target once
            pl = pred_logits[keep]
            tg = targets[keep]
            if pl.shape[0] > 0:
                nll = per_pos_nll(pl, tg)
                total_nll += float(nll.sum())
                total_tokens += int(pl.shape[0])
            prev_end = start + w
            n_windows += 1
            if n_windows >= args.max_windows or end >= N:
                break
            if time.time() - t0 > 540:
                print("[ppl] time budget hit, stopping windows", flush=True)
                break
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        low = msg.lower()
        status = "OOM" if any(s in low for s in (
            "out of memory", "oom", "cudamalloc", "cuda error",
            "alloc", "cublas_status_alloc")) else "ERR"
        err = msg
        print(f"[ppl] mode={args.mode} W={W} caught {status}: {msg[:300]}", flush=True)

    mean_nll = total_nll / total_tokens if total_tokens else float("nan")
    ppl = math.exp(mean_nll) if total_tokens else float("nan")
    eng = engage.summary() if args.mode == "apa" else {"APA_ENGAGED": None}
    out = {
        "mode": args.mode, "window": W, "stride": stride,
        "status": status, "error": err,
        "refine_percentile": args.refine if args.mode == "apa" else None,
        "n_windows": n_windows, "scored_tokens": total_tokens,
        "mean_nll": mean_nll, "ppl": ppl,
        "elapsed_s": time.time() - t0,
        "engagement": eng,
    }
    OUT = REPO / "logs" / "nc17" / f"p1_ppl_{args.mode}_w{W}.json"
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[ppl] mode={args.mode} W={W}: ppl={ppl:.4f} nll={mean_nll:.4f} "
          f"tokens={total_tokens} windows={n_windows}", flush=True)
    if args.mode == "apa":
        print(f"[ppl] APA engagement: ENGAGED={eng['APA_ENGAGED']} "
              f"calls={eng['apa_calls']} engaged_calls={eng['engaged_calls']} "
              f"mean_frac={eng['mean_engaged_frac']} z={eng['z_threshold']}", flush=True)
        if not eng["APA_ENGAGED"]:
            print("[ppl] *** APA NEVER ENGAGED — not a valid APA datapoint "
                  "(June law) ***", flush=True)
    print(f"[ppl] wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
