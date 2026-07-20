#!/usr/bin/env python3
"""NC17-P0 step 3: OOM / context ladder — ONE PROBE PER SUBPROCESS.

Stock transformers, bf16, default SDPA attention, no quantization, no offload.
Gemma law: fragmentation fakes OOMs across contexts, so every probe is a fresh
process. This script runs exactly ONE probe and exits.

Modes:
  prefill S   -> forward a synthetic length-S prompt (token repeat), no decode.
  decode  S   -> prefill S then greedy-decode DECODE_STEPS with cache.

Exit codes:
  0   probe SOLID (fit); prints a JSON result line.
  42  CUDA OOM (the measurement, not a failure); prints JSON with oom=true.
  1   real error (import/other) -> RED, verbatim traceback to stderr.

Prints one line:  RESULT <json>
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

REPO_ID = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DECODE_STEPS = 64
# A real vocab token id to repeat for the synthetic prompt (arbitrary common id).
REPEAT_TOKEN = 785  # "The"-ish region; any valid id < vocab works


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prefill", "decode"])
    ap.add_argument("S", type=int, help="context length (tokens)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    dev = "cuda"
    torch.cuda.reset_peak_memory_stats()
    result = {"mode": args.mode, "S": args.S, "oom": False, "solid": False}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            REPO_ID, revision=REVISION,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to(dev)
        model.eval()

        ids = torch.full((1, args.S), REPEAT_TOKEN, dtype=torch.long, device=dev)
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=(args.mode == "decode"))
            if args.mode == "decode":
                past = out.past_key_values
                nxt = int(out.logits[0, -1, :].argmax())
                cur = torch.tensor([[nxt]], device=dev, dtype=torch.long)
                for _ in range(DECODE_STEPS):
                    step = model(input_ids=cur, past_key_values=past, use_cache=True)
                    past = step.past_key_values
                    cur = torch.tensor([[int(step.logits[0, -1, :].argmax())]],
                                       device=dev, dtype=torch.long)
            torch.cuda.synchronize()

        result["solid"] = True
        result["max_mem_alloc_MiB"] = round(torch.cuda.max_memory_allocated() / (1024**2), 1)
        result["max_mem_reserved_MiB"] = round(torch.cuda.max_memory_reserved() / (1024**2), 1)
        print("RESULT " + json.dumps(result), flush=True)
        return 0
    except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
        result["oom"] = True
        result["error"] = "CUDA OutOfMemoryError: " + str(e).split("\n")[0]
        try:
            result["max_mem_alloc_MiB"] = round(torch.cuda.max_memory_allocated() / (1024**2), 1)
            result["max_mem_reserved_MiB"] = round(torch.cuda.max_memory_reserved() / (1024**2), 1)
        except Exception:
            pass
        print("RESULT " + json.dumps(result), flush=True)
        return 42
    except RuntimeError as e:
        # Some OOMs surface as generic RuntimeError with "out of memory".
        if "out of memory" in str(e).lower():
            result["oom"] = True
            result["error"] = "RuntimeError OOM: " + str(e).split("\n")[0]
            print("RESULT " + json.dumps(result), flush=True)
            return 42
        traceback.print_exc()
        return 1
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
