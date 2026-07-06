#!/usr/bin/env python3
"""Real-model Qwen3.5 INT4/INT3/INT2 weight PPL and memory sweep.

This is intentionally not a kernel benchmark. It loads a real Qwen3.5
TensorCUDA model with the selected weight bit width, runs standard attention
and APA selective attention at requested refine percentiles, and records PPL
plus observed GPU memory. Failures/OOMs are written as results.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
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


QWEN35_2B = (
    "/home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/"
    "snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real Qwen3.5 model PPL/OOM sweeps for INT4/INT3/INT2."
    )
    parser.add_argument("--model-dir", default=QWEN35_2B)
    parser.add_argument("--bits", default="4,3,2", help="Comma list from 4,3,2.")
    parser.add_argument("--refines", default="0.15,0.10,0.05")
    parser.add_argument("--no-standard", action="store_true")
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--scored", type=int, default=512)
    parser.add_argument("--step", type=int, default=64)
    parser.add_argument("--n-windows", type=int, default=2)
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def parse_bits(raw: str) -> list[int]:
    bits = [int(x.strip()) for x in raw.split(",") if x.strip()]
    bad = [b for b in bits if b not in (2, 3, 4)]
    if bad:
        raise ValueError(f"bits must be 2, 3, or 4; got {bad}")
    return bits


def parse_refines(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def output_path(path: Path | None) -> Path:
    if path is not None:
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("artifacts") / "intn_weight_ppl" / f"qwen35_intn_ppl_{stamp}.json"


def write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def gpu_memory_mib() -> int | None:
    try:
        tc.synchronize()
    except Exception:
        pass
    try:
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


def get_text() -> str:
    try:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join(r["text"] for r in ds)
        if text.strip():
            return text
    except Exception as exc:
        print(f"wikitext unavailable ({exc}); local docs fallback", flush=True)

    import glob

    docs = []
    for path in sorted(glob.glob("/mnt/ForgeRealm/GraftRepository/docs/*.md")):
        with open(path, encoding="utf-8", errors="ignore") as fh:
            docs.append(fh.read())
    return "\n".join(docs) * 4


def set_attention(model: Qwen35_TC, mode: str, refine: float | None) -> None:
    for layer in model.layers:
        if not layer.is_attn:
            continue
        layer.mixer.attention_mode = mode
        if refine is not None:
            layer.mixer.refine_percentile = float(refine)


def window_nll(model: Qwen35_TC, ids: np.ndarray, scored: int, step: int) -> tuple[float, int, int | None]:
    ctx = len(ids) - scored
    nll, count = 0.0, 0
    max_mem = gpu_memory_mib()
    with tc.no_grad():
        _, caches = model(ids[None, :ctx], last_token_only=True)
        max_mem = max_observed(max_mem, gpu_memory_mib())
        pos = ctx
        while pos < len(ids) - 1:
            n = min(step, len(ids) - 1 - pos)
            logits, caches = model(ids[None, pos:pos + n], caches=caches,
                                   position_offset=pos)
            lp = logits.float().numpy()[0]
            tgt = ids[pos + 1:pos + n + 1]
            mx = lp.max(-1, keepdims=True)
            lse = mx[:, 0] + np.log(np.exp(lp - mx).sum(-1))
            nll += float((lse - lp[np.arange(len(tgt)), tgt]).sum())
            count += len(tgt)
            pos += n
            max_mem = max_observed(max_mem, gpu_memory_mib())
            tc.empty_cache()
    return nll, count, max_mem


def max_observed(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def run_setting(
    model: Qwen35_TC,
    ids_all: np.ndarray,
    *,
    name: str,
    mode: str,
    refine: float | None,
    window: int,
    scored: int,
    step: int,
    n_windows: int,
) -> dict:
    set_attention(model, mode, refine)
    start = time.perf_counter()
    tot_nll, tot_cnt = 0.0, 0
    max_mem = gpu_memory_mib()
    for w in range(n_windows):
        lo = w * window
        ids = ids_all[lo:lo + window]
        if len(ids) < window:
            break
        nll, count, mem = window_nll(model, ids, scored, step)
        tot_nll += nll
        tot_cnt += count
        max_mem = max_observed(max_mem, mem)
    ppl = float(np.exp(tot_nll / tot_cnt)) if tot_cnt else float("nan")
    return {
        "name": name,
        "mode": mode,
        "refine": refine,
        "status": "ok",
        "ppl": ppl,
        "scored_tokens": tot_cnt,
        "wall_seconds": time.perf_counter() - start,
        "max_observed_memory_mib": max_mem,
    }


def run_bits(bits: int, args: argparse.Namespace, ids_all: np.ndarray) -> dict:
    QuantLinearTC.set_weight_bits(bits)
    result = {
        "bits": bits,
        "status": "started",
        "settings": [],
        "memory_before_load_mib": gpu_memory_mib(),
    }
    model = None
    start = time.perf_counter()
    try:
        model, info = Qwen35_TC.from_pretrained(args.model_dir)
        result.update({
            "status": "loaded",
            "load_info": info,
            "load_wall_seconds": time.perf_counter() - start,
            "memory_after_load_mib": gpu_memory_mib(),
        })
        if args.load_only:
            result["status"] = "load_only_ok"
            return result

        settings = []
        if not args.no_standard:
            settings.append(("standard", "standard", None))
        for refine in parse_refines(args.refines):
            settings.append((f"apa_r{refine:.2f}", "apa_selective", refine))

        for name, mode, refine in settings:
            try:
                print(f"INT{bits} {name}: scoring", flush=True)
                setting = run_setting(
                    model, ids_all, name=name, mode=mode, refine=refine,
                    window=args.window, scored=args.scored, step=args.step,
                    n_windows=args.n_windows,
                )
            except Exception as exc:
                setting = {
                    "name": name,
                    "mode": mode,
                    "refine": refine,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "max_observed_memory_mib": gpu_memory_mib(),
                }
                tc.empty_cache()
            result["settings"].append(setting)
        result["status"] = "ok"
        return result
    except Exception as exc:
        result.update({
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "memory_at_error_mib": gpu_memory_mib(),
            "load_wall_seconds": time.perf_counter() - start,
        })
        return result
    finally:
        del model
        gc.collect()
        tc.empty_cache()


def main() -> int:
    args = parse_args()
    if args.window <= args.scored:
        raise ValueError("--window must be > --scored")
    if args.step <= 0:
        raise ValueError("--step must be > 0")
    np.random.seed(args.seed)
    out = output_path(args.output)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    ids_all = tokenizer(get_text(), return_tensors="np").input_ids[0].astype(np.int64)
    needed = args.window * args.n_windows
    if len(ids_all) < needed:
        raise RuntimeError(f"need {needed} eval tokens, got {len(ids_all)}")

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": os.path.abspath(args.model_dir),
        "bits": parse_bits(args.bits),
        "refines": parse_refines(args.refines),
        "window": args.window,
        "scored": args.scored,
        "step": args.step,
        "n_windows": args.n_windows,
        "eval_tokens_available": int(len(ids_all)),
        "results": [],
    }
    write_payload(out, payload)

    tc.set_alloc_pooling(True)
    for bits in parse_bits(args.bits):
        print(f"loading INT{bits}: {args.model_dir}", flush=True)
        result = run_bits(bits, args, ids_all)
        payload["results"].append(result)
        write_payload(out, payload)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)

    print(f"artifact={out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
