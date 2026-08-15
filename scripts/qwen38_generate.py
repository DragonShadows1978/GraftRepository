#!/usr/bin/env python3
"""Greedy text-only generation for Qwen3.8-27B INT3 on tensor_cuda."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.qwen38_tc import (DEFAULT_CACHE_DIR, DEFAULT_MODEL_DIR, INT3PackCache,
                            DEFAULT_LM_HEAD_CHUNK_ROWS, Qwen38Config, Qwen38_TC,
                            compute_qwen38_memory_budget)


DEMO_PROMPTS = {
    "code": "Write a Python function that returns the first n Fibonacci numbers, with a short explanation.",
    "factual": "What is the capital of Michigan, and what river runs through Detroit? Answer concisely.",
    "chat": "I have twenty minutes and feel stuck. Help me choose one small useful task and start it.",
}


def chat_ids(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    kwargs = dict(tokenize=True, add_generation_prompt=True,
                  return_tensors="np")
    try:
        ids = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    return np.ascontiguousarray(np.asarray(ids, dtype=np.int64).reshape(1, -1))


class VramSampler:
    """Phase-tagged, process-adjacent nvidia-smi sampler for gate receipts."""

    def __init__(self, interval_seconds=0.25):
        self.interval_seconds = float(interval_seconds)
        self.phase = "load"
        self.samples = []
        self.errors = []
        self._stop = threading.Event()
        self._thread = None

    def set_phase(self, phase):
        self.phase = str(phase)
        self.sample_now("boundary")

    def sample_now(self, kind="explicit"):
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=False)
            if proc.returncode:
                raise RuntimeError(
                    f"nvidia-smi rc={proc.returncode}: "
                    f"{(proc.stderr or proc.stdout).strip()}")
            used, total = [int(v.strip())
                           for v in proc.stdout.strip().splitlines()[0].split(",")]
            self.samples.append({"phase": self.phase, "kind": kind,
                                 "used_mib": used, "total_mib": total,
                                 "monotonic_seconds": time.monotonic()})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if not self.errors or self.errors[-1] != error:
                self.errors.append(error)

    def _run(self):
        while not self._stop.is_set():
            self.sample_now("poll")
            self._stop.wait(self.interval_seconds)

    def start(self):
        self._thread = threading.Thread(target=self._run,
                                        name="qwen38-vram", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)

    def receipt(self):
        peaks = {}
        detail = {}
        for sample in self.samples:
            phase = sample["phase"]
            family = phase.split(":", 1)[0]
            peaks[family] = max(peaks.get(family, 0), sample["used_mib"])
            detail[phase] = max(detail.get(phase, 0), sample["used_mib"])
        total = self.samples[0]["total_mib"] if self.samples else None
        return {
            "evidence": "sampled nvidia-smi whole-device memory.used",
            "interval_seconds": self.interval_seconds,
            "samples": len(self.samples),
            "total_mib": total,
            "peak_mib": peaks,
            "phase_peak_mib": detail,
            "errors": self.errors,
        }


def greedy(model, tokenizer, input_ids, max_new_tokens, sampler=None,
           phase_label="prompt"):
    from core.qwen38_tc import tc

    if sampler is not None:
        sampler.set_phase(f"prefill:{phase_label}")
    start = time.perf_counter()
    logits, caches = model(input_ids, caches=None, position_offset=0,
                           last_token_only=True)
    tc.synchronize()
    prefill = time.perf_counter() - start
    if sampler is not None:
        sampler.sample_now("after_prefill")
    generated = []
    decode_steps = 0
    if sampler is not None:
        sampler.set_phase(f"decode:{phase_label}")
    decode_start = time.perf_counter()
    for step in range(max_new_tokens):
        logits_np = logits.float().numpy().reshape(-1)
        if not np.isfinite(logits_np).all():
            raise FloatingPointError(
                f"non-finite logits prompt={phase_label} generation_step={step}")
        token = int(np.argmax(logits_np))
        generated.append(token)
        if token == model.config.eos_token_id or step + 1 == max_new_tokens:
            break
        one = np.array([[token]], dtype=np.int64)
        logits, caches = model(one, caches=caches,
                               position_offset=input_ids.shape[1] + step,
                               last_token_only=True)
        tc.synchronize()
        decode_steps += 1
        if sampler is not None:
            sampler.sample_now("decode_step")
    decode = time.perf_counter() - decode_start
    text = tokenizer.decode(generated, skip_special_tokens=False)
    return {
        "text": text,
        "token_ids": generated,
        "prompt_tokens": int(input_ids.shape[1]),
        "new_tokens": len(generated),
        "prefill_seconds": prefill,
        "decode_seconds": decode,
        "decode_steps": decode_steps,
        "decode_tok_s": decode_steps / decode if decode else float("inf"),
    }


def print_budget(model_dir):
    budget = compute_qwen38_memory_budget(model_dir)
    print("PER-COMPONENT MEMORY MAP (GiB; computed from checkpoint shapes)")
    for name, value in budget.items():
        location = "HOST" if name.startswith("host_") else "VRAM"
        print(f"  {name:42s} {value / 2**30:9.4f} GiB  {location}")
    return budget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--prompt")
    ap.add_argument("--demo", action="store_true",
                    help="run the registered code/factual/chat prompts")
    ap.add_argument("--benchmark", action="store_true",
                    help="use an exactly 128-token prompt")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--lm-head-chunk-rows", type=int,
                    default=int(os.environ.get("QWEN38_LM_HEAD_CHUNK_ROWS",
                                               DEFAULT_LM_HEAD_CHUNK_ROWS)))
    ap.add_argument("--output-gate-type", choices=("swish", "silu", "sigmoid"),
                    default=None,
                    help="diagnostic override; 27B attention defaults to sigmoid")
    ap.add_argument("--build-cache-only", action="store_true")
    ap.add_argument("--budget-only", action="store_true")
    ap.add_argument("--load-only", action="store_true")
    ap.add_argument("--skip-fused-gate", action="store_true",
                    help="deprecated compatibility flag; fused INT3 is disabled by default")
    ap.add_argument("--run-fused-gate", action="store_true",
                    help="rerun the known-failing fused INT3 diagnostic")
    args = ap.parse_args()

    cfg = Qwen38Config.from_model_dir(args.model_dir)
    if args.output_gate_type is not None:
        cfg.output_gate_type = args.output_gate_type
        cfg._validate_qwen38()
    print("QWEN38_CONFIG " + json.dumps(cfg.as_printable_dict(), sort_keys=True))
    if args.budget_only:
        print_budget(args.model_dir)
        return
    if args.build_cache_only:
        manifest = INT3PackCache(args.model_dir, args.cache_dir).build()
        print("CACHE_RESULT " + json.dumps({
            "status": "PASS", "cache_dir": str(Path(args.cache_dir).resolve()),
            "entries": len(manifest["entries"]),
            "elapsed_seconds": manifest["elapsed_seconds"],
        }, sort_keys=True))
        return

    if not (args.prompt or args.demo or args.benchmark or args.load_only):
        ap.error("provide --prompt, --demo, --benchmark, --load-only, "
                 "--build-cache-only, or --budget-only")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if args.skip_fused_gate and args.run_fused_gate:
        ap.error("--skip-fused-gate and --run-fused-gate are mutually exclusive")
    sampler = VramSampler()
    sampler.start()
    try:
        sampler.set_phase("load")
        model, info = Qwen38_TC.from_pretrained(
            args.model_dir, args.cache_dir, run_fused_gate=args.run_fused_gate,
            lm_head_chunk_rows=args.lm_head_chunk_rows,
            output_gate_type=args.output_gate_type)
        sampler.sample_now("after_load")
        print("LOAD_RESULT " + json.dumps(info, sort_keys=True))
        if args.load_only:
            return

        if args.benchmark:
            seed = chat_ids(tokenizer, "Explain why careful measurements matter in engineering.")
            while seed.shape[1] < 128:
                seed = np.concatenate([seed, seed], axis=1)
            prompts = {"benchmark_128": np.ascontiguousarray(seed[:, :128])}
        elif args.demo:
            prompts = {name: chat_ids(tokenizer, text)
                       for name, text in DEMO_PROMPTS.items()}
        else:
            prompts = {"prompt": chat_ids(tokenizer, args.prompt)}

        for name, ids in prompts.items():
            result = greedy(model, tokenizer, ids, args.max_new_tokens,
                            sampler=sampler, phase_label=name)
            print(f"===== RAW OUTPUT {name} =====")
            print(result.pop("text"))
            print(f"===== END RAW OUTPUT {name} =====")
            print("GENERATION_METRICS " + json.dumps({"prompt": name, **result},
                                                       sort_keys=True))
    finally:
        sampler.stop()
        print("VRAM_RECEIPT " + json.dumps(sampler.receipt(), sort_keys=True),
              flush=True)


if __name__ == "__main__":
    main()
