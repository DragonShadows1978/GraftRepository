#!/usr/bin/env python3
"""Greedy usability gate for Qwen3.5 INTN weight quantization.

This is not a PPL runner. It asks whether a low-bit model can still perform a
basic deterministic task that matters operationally: extract planted facts from
real prompt text under greedy decoding. Results are compared across INT4 and
INT3 using the same TensorCUDA loader and attention modes.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensor_cuda as tc  # noqa: E402
from core.mistral7b_tc import QuantLinearTC  # noqa: E402
from core.qwen35_tc import Qwen35_TC  # noqa: E402


QWEN35_9B = (
    "/home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/"
    "snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)

CASES = [
    {
        "name": "vault_code",
        "question": "What is the vault access code?",
        "expected": "LUMEN-482",
    },
    {
        "name": "backup_pilot",
        "question": "Who is the backup pilot?",
        "expected": "Mara Voss",
    },
    {
        "name": "coolant_color",
        "question": "What color is the coolant marker?",
        "expected": "amber",
    },
    {
        "name": "checkpoint_city",
        "question": "Which city is checkpoint delta?",
        "expected": "Kenora",
    },
    {
        "name": "numeric_handshake",
        "question": "What is the numeric handshake?",
        "expected": "73194",
    },
]

DOCUMENT = """\
ASTER FIELD RECORD

The vault access code is LUMEN-482.
The backup pilot is Mara Voss.
The coolant marker color is amber.
Checkpoint delta is located in Kenora.
The numeric handshake is 73194.

Answer only from the ASTER FIELD RECORD.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run greedy INT4/INT3 usability checks on Qwen3.5."
    )
    parser.add_argument("--model-dir", default=QWEN35_9B)
    parser.add_argument("--bits", default="4,3")
    parser.add_argument("--modes", default="standard,apa_r0.15")
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def parse_bits(raw: str) -> list[int]:
    bits = [int(x.strip()) for x in raw.split(",") if x.strip()]
    bad = [b for b in bits if b not in (2, 3, 4)]
    if bad:
        raise ValueError(f"bits must be 2, 3, or 4; got {bad}")
    return bits


def parse_modes(raw: str) -> list[tuple[str, str, float | None]]:
    modes: list[tuple[str, str, float | None]] = []
    for item in [x.strip() for x in raw.split(",") if x.strip()]:
        if item == "standard":
            modes.append(("standard", "standard", None))
        elif item.startswith("apa_r"):
            refine = float(item[5:])
            modes.append((item, "apa_selective", refine))
        else:
            raise ValueError(f"unknown mode {item!r}; use standard or apa_r0.15")
    return modes


def output_path(path: Path | None) -> Path:
    if path is not None:
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("artifacts") / "intn_usability" / f"qwen35_intn_usability_{stamp}.json"


def write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def gpu_memory_mib() -> int | None:
    try:
        tc.synchronize()
    except Exception:
        pass
    try:
        import subprocess

        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
        vals = [int(x.strip()) for x in out.splitlines() if x.strip()]
        return vals[0] if vals else None
    except Exception:
        return None


def set_attention(model: Qwen35_TC, mode: str, refine: float | None) -> None:
    for layer in model.layers:
        if not layer.is_attn:
            continue
        layer.mixer.attention_mode = mode
        if refine is not None:
            layer.mixer.refine_percentile = float(refine)


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def contains_expected(answer: str, expected: str) -> bool:
    return normalize(expected) in normalize(answer)


def degenerate(answer: str) -> bool:
    stripped = answer.strip()
    if not stripped:
        return True
    words = normalize(stripped).split()
    if len(words) >= 8:
        pairs = list(zip(words, words[1:]))
        if pairs and max(pairs.count(p) for p in set(pairs)) >= 4:
            return True
    return False


def make_prompt(question: str) -> str:
    return (
        DOCUMENT
        + "\nQuestion: "
        + question
        + "\nAnswer with only the requested value.\nAnswer:"
    )


def generate(model: Qwen35_TC, tokenizer, prompt: str, max_new: int) -> dict:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer(text, return_tensors="np").input_ids.astype(np.int64)
    start = time.perf_counter()
    with tc.no_grad():
        logits, caches = model(ids, last_token_only=True)
        nxt = int(logits.float().numpy()[0, -1].argmax())
        off = ids.shape[1]
        out: list[int] = []
        for _ in range(max_new):
            if nxt == model.config.eos_token_id:
                break
            out.append(nxt)
            logits, caches = model(
                np.array([[nxt]], dtype=np.int64),
                caches=caches,
                position_offset=off,
            )
            off += 1
            nxt = int(logits.float().numpy()[0, -1].argmax())
    try:
        tc.synchronize()
    except Exception:
        pass
    answer = tokenizer.decode(out, skip_special_tokens=True)
    return {
        "answer": answer,
        "generated_tokens": len(out),
        "wall_seconds": time.perf_counter() - start,
    }


def run_bits(bits: int, args: argparse.Namespace, tokenizer) -> dict:
    QuantLinearTC.set_weight_bits(bits)
    result = {
        "bits": bits,
        "status": "started",
        "memory_before_load_mib": gpu_memory_mib(),
        "settings": [],
    }
    model = None
    start = time.perf_counter()
    try:
        model, info = Qwen35_TC.from_pretrained(args.model_dir)
        result.update(
            {
                "status": "loaded",
                "load_info": info,
                "load_wall_seconds": time.perf_counter() - start,
                "memory_after_load_mib": gpu_memory_mib(),
            }
        )
        for name, mode, refine in parse_modes(args.modes):
            set_attention(model, mode, refine)
            cases = []
            hits = 0
            failures = 0
            max_mem = gpu_memory_mib()
            for case in CASES:
                prompt = make_prompt(case["question"])
                try:
                    generated = generate(model, tokenizer, prompt, args.max_new)
                    hit = contains_expected(generated["answer"], case["expected"])
                    bad = degenerate(generated["answer"])
                    cases.append(
                        {
                            "name": case["name"],
                            "question": case["question"],
                            "expected": case["expected"],
                            "answer": generated["answer"],
                            "hit": hit,
                            "degenerate": bad,
                            "generated_tokens": generated["generated_tokens"],
                            "wall_seconds": generated["wall_seconds"],
                        }
                    )
                    hits += int(hit and not bad)
                    failures += int((not hit) or bad)
                    max_mem = max(max_mem or 0, gpu_memory_mib() or 0)
                except Exception as exc:
                    cases.append(
                        {
                            "name": case["name"],
                            "question": case["question"],
                            "expected": case["expected"],
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    failures += 1
                    tc.empty_cache()
            result["settings"].append(
                {
                    "name": name,
                    "mode": mode,
                    "refine": refine,
                    "status": "ok" if failures == 0 else "failed_cases",
                    "hits": hits,
                    "total": len(CASES),
                    "cases": cases,
                    "max_observed_memory_mib": max_mem,
                }
            )
        result["status"] = "ok"
        return result
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "memory_at_error_mib": gpu_memory_mib(),
                "load_wall_seconds": time.perf_counter() - start,
            }
        )
        return result
    finally:
        del model
        gc.collect()
        tc.empty_cache()


def main() -> int:
    args = parse_args()
    out = output_path(args.output)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": os.path.abspath(args.model_dir),
        "bits": parse_bits(args.bits),
        "modes": args.modes,
        "max_new": args.max_new,
        "cases": CASES,
        "results": [],
    }
    write_payload(out, payload)

    tc.set_alloc_pooling(True)
    for bits in parse_bits(args.bits):
        print(f"loading INT{bits}: {args.model_dir}", flush=True)
        result = run_bits(bits, args, tokenizer)
        payload["results"].append(result)
        write_payload(out, payload)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)

    print(f"artifact={out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
