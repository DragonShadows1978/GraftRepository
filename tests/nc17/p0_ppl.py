#!/usr/bin/env python3
"""NC17-P0 step 4: perplexity, ONE WINDOW PER SUBPROCESS.

wikitext-2-raw test split, sliding-window, stride 512. Uses the EXACT
tokenization saved by the GT mint step (logs/nc17/p0_ppl_tokens.npz) — no
re-tokenization, identical across all stages (matched-reference law).

Sliding-window NLL protocol (standard HF ppl): for each window start (stride
512), the model scores a window of `W` tokens but only the last `stride` tokens
(those not seen in the previous window) contribute to the loss; earlier tokens
are context (target = -100). ppl = exp(sum(nll) / n_scored_tokens).

Usage: p0_ppl.py <window>
Prints RESULT <json> with ppl, n_windows, n_scored_tokens, wall_s.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ID = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
STRIDE = 512
REPO_ROOT = Path(__file__).resolve().parents[2]
PPL_TOKENS = REPO_ROOT / "logs" / "nc17" / "p0_ppl_tokens.npz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("window", type=int)
    # Optional slicing so an expensive window can be split across several
    # flock subprocesses (rails: split the rung, don't raise the timeout).
    # A slice processes window-starts [start_win, start_win+n_win) of the
    # stride grid and reports partial nll_sum / n_scored to aggregate.
    ap.add_argument("--start-win", type=int, default=0)
    ap.add_argument("--n-win", type=int, default=0, help="0 = to end of corpus")
    args = ap.parse_args()
    W = args.window
    SLICED = (args.start_win != 0 or args.n_win != 0)

    import torch
    from transformers import AutoModelForCausalLM

    dev = "cuda"
    torch.cuda.reset_peak_memory_stats()

    def _oom_report(err):
        rec = {"window": W, "stride": STRIDE, "oom": True,
               "error": "CUDA OutOfMemoryError: " + str(err).split("\n")[0]}
        try:
            rec["max_mem_alloc_MiB"] = round(torch.cuda.max_memory_allocated() / (1024**2), 1)
        except Exception:
            pass
        print("RESULT " + json.dumps(rec), flush=True)

    z = np.load(PPL_TOKENS)
    all_ids = torch.tensor(z["token_ids"], dtype=torch.long)
    n_tok = all_ids.numel()

    model = AutoModelForCausalLM.from_pretrained(
        REPO_ID, revision=REVISION,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(dev)
    model.eval()

    # NOTE (memory-shape finding, recorded in ledger): the HF `labels=` loss
    # path materializes fp32 logits for the WHOLE window [1, W, vocab] plus a
    # cross_entropy intermediate, which OOMs at W>=4096 on the 12GB card even
    # though the forward pass itself fits. The plan's sliding-window protocol
    # only scores the last `trg_len` positions, so we request logits, slice to
    # exactly those scored positions BEFORE the fp32 cast + cross_entropy, and
    # compute the NLL ourselves. This is numerically the same statistic as the
    # HF path (mean token NLL over the scored tail), just without the
    # full-window logit blow-up. Tokenization is unchanged (matched-reference).
    # Enumerate the full stride grid of window-starts (identical for every
    # slice); slicing only selects which window indices this process runs.
    begins = list(range(0, n_tok, STRIDE))
    # Truncate at the last window that reaches the corpus end (avoid empty
    # trailing windows once end==n_tok is hit).
    grid = []
    for begin in begins:
        end = min(begin + W, n_tok)
        grid.append((begin, end))
        if end == n_tok:
            break
    total_windows = len(grid)
    lo_i = args.start_win
    hi_i = total_windows if args.n_win == 0 else min(args.start_win + args.n_win, total_windows)

    ce = torch.nn.CrossEntropyLoss(reduction="sum")
    t0 = time.time()
    nll_sum = 0.0
    n_scored = 0
    n_windows = 0
    try:
        with torch.no_grad():
            base = model.model  # base transformer, returns last_hidden_state
            lm_head = model.lm_head
            for i in range(lo_i, hi_i):
                begin, end = grid[i]
                # prev_end = end of the window one stride earlier (0 for the
                # very first window). Reconstructed so slices are independent.
                prev_end = 0 if i == 0 else grid[i - 1][1]
                trg_len = end - prev_end  # tokens newly scored in this window
                ids = all_ids[begin:end].unsqueeze(0).to(dev)

                # Frugal scoring: run only the base transformer to get hidden
                # states, then apply lm_head to ONLY the scored tail positions.
                # This avoids materializing full [W, vocab] logits (2.4 GiB at
                # W=8192), the dominant ppl-time allocation. Numerically
                # identical statistic to the full-logit path (verified at W=2048
                # against the HF labels= path: ppl 14.96245 both ways).
                hs = base(input_ids=ids, use_cache=False).last_hidden_state[0]  # [seq, hidden]

                # Standard causal shift: hidden[t] predicts token[t+1]. Score
                # only the newly-revealed tail: predictions for the last
                # `trg_len` targets are at hidden positions [seq-trg_len-1 ..
                # seq-2]; their targets are ids[seq-trg_len .. seq-1].
                seq = ids.shape[1]
                pred_lo = seq - trg_len - 1
                if pred_lo < 0:  # first window: no token before position 0
                    pred_lo = 0
                tail_hs = hs[pred_lo:seq - 1]                       # [n, hidden]
                pred_logits = lm_head(tail_hs).float()             # [n, vocab]
                tgt = ids[0, pred_lo + 1:seq]                      # [n]
                loss_sum = ce(pred_logits, tgt)
                n_this = tgt.numel()
                nll_sum += float(loss_sum)
                n_scored += n_this
                n_windows += 1
                del hs, tail_hs, pred_logits
    except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
        _oom_report(e)
        return 42
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            _oom_report(e)
            return 42
        raise

    torch.cuda.synchronize()
    wall = time.time() - t0
    result = {
        "window": W,
        "stride": STRIDE,
        "sliced": SLICED,
        "start_win": lo_i,
        "end_win": hi_i,
        "total_windows": total_windows,
        "n_windows": n_windows,
        "nll_sum": nll_sum,
        "n_scored_tokens": n_scored,
        "n_corpus_tokens": int(n_tok),
        "wall_s": round(wall, 2),
        "max_mem_alloc_MiB": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
        "max_mem_reserved_MiB": round(torch.cuda.max_memory_reserved() / (1024**2), 1),
    }
    if not SLICED:
        result["ppl"] = round(math.exp(nll_sum / n_scored), 6)
    print("RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
