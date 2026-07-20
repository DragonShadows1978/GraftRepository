#!/usr/bin/env python3
"""NC17-P0 step 2: GT mint (FIRST, before the long ladder work).

On GPU, bf16, greedy. Captures logits for a fixed, deterministic prompt set
(committed in this script) at the final prompt position + the first 64 decode
steps. Saves:
  - logs/nc17/p0_gt.npz           (per-prompt final-position + decode logits, token ids)
  - logs/nc17/p0_ppl_tokens.npz   (exact tokenized wikitext-2-raw test corpus token ids)
Then writes the sentinel logs/nc17/P0_GT_READY.

Determinism: greedy (do_sample=False), fixed prompts, torch manual seed, bf16.
Evidence class: this is a reference capture (memory/parity substrate), not a
capability claim.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ID = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
REPO_ROOT = Path(__file__).resolve().parents[2]
LOGDIR = REPO_ROOT / "logs" / "nc17"

DECODE_STEPS = 64

# Fixed, deterministic prompt set. >=8 prompts covering:
#   short factual, multilingual (8 coverage languages), code-ish,
#   and 2 name-verdict-shaped prompts.
# Coverage languages (8): English, Spanish, French, German, Chinese,
#   Japanese, Russian, Arabic.
PROMPTS = [
    # short factual (English)
    "The capital of France is",
    # multilingual coverage languages (non-ASCII given as \u escapes so the
    # source stays byte-unambiguous)
    "La capital de España es",                   # Spanish
    "La capitale de la France est",                  # French
    "Die Hauptstadt von Deutschland ist",            # German
    "中国的首都是",          # Chinese: "China's capital is"
    "日本の首都は",          # Japanese: "Japan's capital is"
    "Столица России —",  # Russian: "Capital of Russia -"
    "عاصمة مصر هي",  # Arabic: "The capital of Egypt is"
    # code-ish
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    # name-verdict-shaped (2)
    "Is the username 'Th3_D4rk_L0rd_666' appropriate for a family game? Answer:",
    "Player name: 'HelloKitty42'. Verdict (ALLOW or BLOCK):",
]


def sha256_of_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def build_ppl_tokens(tokenizer):
    """Tokenize the full wikitext-2-raw test split as one concatenated stream.

    Records source + sha of the raw text. If datasets is unavailable offline,
    this raises verbatim (RED honesty).
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw_text = "\n\n".join(ds["text"])
    raw_sha = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    enc = tokenizer(raw_text, return_tensors="pt")
    ids = enc["input_ids"][0].to(torch.int64).cpu().numpy()
    return ids, raw_sha, len(ds)


def main() -> int:
    torch.manual_seed(0)
    LOGDIR.mkdir(parents=True, exist_ok=True)
    dev = "cuda"

    tok = AutoTokenizer.from_pretrained(REPO_ID, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        REPO_ID,
        revision=REVISION,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(dev)
    model.eval()

    gt = {}
    meta = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "dtype": "bfloat16",
        "attn": "sdpa",
        "decode_steps": DECODE_STEPS,
        "prompts": PROMPTS,
        "vocab_size": int(model.config.vocab_size),
        "prompt_records": [],
    }

    with torch.no_grad():
        for i, prompt in enumerate(PROMPTS):
            enc = tok(prompt, return_tensors="pt").to(dev)
            prompt_ids = enc["input_ids"][0].to(torch.int64).cpu().numpy()

            # Final-position logits from the prefill forward pass.
            out = model(**enc, use_cache=True)
            final_logits = out.logits[0, -1, :].to(torch.float32).cpu().numpy()

            # Greedy decode 64 steps, capturing per-step logits + chosen token.
            past = out.past_key_values
            next_tok = int(final_logits.argmax())
            decode_logits = np.zeros((DECODE_STEPS, meta["vocab_size"]), dtype=np.float32)
            decode_tokens = np.zeros((DECODE_STEPS,), dtype=np.int64)
            cur = torch.tensor([[next_tok]], device=dev, dtype=torch.long)
            for step in range(DECODE_STEPS):
                decode_tokens[step] = int(cur.item())
                step_out = model(input_ids=cur, past_key_values=past, use_cache=True)
                logits = step_out.logits[0, -1, :].to(torch.float32).cpu().numpy()
                decode_logits[step] = logits
                past = step_out.past_key_values
                nxt = int(logits.argmax())
                cur = torch.tensor([[nxt]], device=dev, dtype=torch.long)

            gt[f"prompt_ids_{i}"] = prompt_ids
            gt[f"final_logits_{i}"] = final_logits
            gt[f"decode_logits_{i}"] = decode_logits
            gt[f"decode_tokens_{i}"] = decode_tokens

            gen_text = tok.decode(decode_tokens.tolist())
            meta["prompt_records"].append({
                "index": i,
                "prompt": prompt,
                "prompt_n_tokens": int(prompt_ids.shape[0]),
                "final_logits_sha256": sha256_of_array(final_logits),
                "decode_logits_sha256": sha256_of_array(decode_logits),
                "first_decode_token": int(next_tok),
                "generated_text": gen_text,
            })
            print(f"[GT] prompt {i}: {prompt_ids.shape[0]} tok -> gen: {gen_text!r}", flush=True)

    gt["_meta_json"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    np.savez(LOGDIR / "p0_gt.npz", **gt)
    print(f"[GT] wrote {LOGDIR / 'p0_gt.npz'}", flush=True)

    # Free GT graph before building ppl tokens (CPU-only tokenization anyway).
    del model
    torch.cuda.empty_cache()

    ppl_ids, raw_sha, n_docs = build_ppl_tokens(tok)
    np.savez(
        LOGDIR / "p0_ppl_tokens.npz",
        token_ids=ppl_ids,
        meta=np.frombuffer(json.dumps({
            "source": "wikitext / wikitext-2-raw-v1 / split=test (datasets lib)",
            "raw_text_sha256": raw_sha,
            "n_docs": n_docs,
            "n_tokens": int(ppl_ids.shape[0]),
            "join": "\\n\\n between docs",
            "tokenizer_repo": REPO_ID,
            "tokenizer_revision": REVISION,
        }).encode("utf-8"), dtype=np.uint8),
    )
    print(f"[GT] wrote {LOGDIR / 'p0_ppl_tokens.npz'} ({ppl_ids.shape[0]} tokens, raw_sha={raw_sha[:16]}...)", flush=True)

    # Sentinel LAST, only after both npz exist.
    (LOGDIR / "P0_GT_READY").write_text(
        f"revision={REVISION}\n"
        f"gt_npz={LOGDIR / 'p0_gt.npz'}\n"
        f"ppl_tokens_npz={LOGDIR / 'p0_ppl_tokens.npz'}\n"
        f"ppl_n_tokens={int(ppl_ids.shape[0])}\n"
        f"ppl_raw_sha256={raw_sha}\n"
    )
    print(f"[GT] wrote sentinel {LOGDIR / 'P0_GT_READY'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
