#!/usr/bin/env python3
"""Deterministic long-context gates for Qwen3.8-27B INT3.

No corpus or network access is used: filler is generated from a fixed local
word table and a NumPy seed. Every execution emits one machine-readable result
line plus the phase-tagged whole-device nvidia-smi peak receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.qwen38_tc import (DEFAULT_CACHE_DIR, DEFAULT_KV_BLOCK,
                            DEFAULT_LM_HEAD_CHUNK_ROWS, DEFAULT_MODEL_DIR,
                            DEFAULT_MAX_CONTEXT, DEFAULT_PREFILL_CHUNK,
                            Qwen38_TC, tc)
from scripts.qwen38_generate import (DEMO_PROMPTS, VramSampler, chat_ids,
                                     greedy, print_budget,
                                     validate_budget_for_load)


# G-LC0 registration point. The lead must replace every string with the exact
# greedy token-id list from the preserved 2026-08-19 receipt before invoking
# --baseline-check. The harness intentionally refuses a placeholder baseline.
BASELINE_TOKEN_IDS = {
    "code": [71093, 12305, 198, 727, 1118, 1054, 736, 564, 38044, 1393, 1590, 198, 262, 4071, 198, 262, 5019, 279, 1118, 307, 76938, 4947, 13, 198, 1031, 262, 561, 76938, 8240, 8211, 440, 220, 15, 321, 220, 16, 11, 321, 1754, 16905, 198, 262, 1324, 369, 279, 2542, 314, 279, 1330, 36497, 5981, 13, 198, 1031, 262, 1690, 3010, 11, 413, 307, 28, 20, 11, 279],
    "factual": [760, 6511, 314, 14209, 369, 2972, 43, 581, 286, 93868, 321, 279, 2972, 93938, 10629, 332, 8213, 1472, 19158, 13, 248046, 198, 248044],
    "chat": [9764, 725, 1387, 279, 82104, 13, 8439, 488, 599, 6681, 220, 17, 15, 4276, 11, 567, 1144, 264, 3274, 421, 369, 2972, 435, 9075, 93868, 2972, 9997, 48683, 2399, 93868, 321, 682, 264, 2972, 7187, 6052, 1500, 159034, 271, 8160, 369, 678, 799, 2526, 11, 5222, 3274, 25, 271, 13962, 10838, 241, 251, 561, 328, 5666, 220, 18, 38540, 1, 1697, 271, 332, 9930],
}

FILLER_WORDS = (
    "amber atlas birch brook cedar circle cloud copper dawn delta ember field "
    "flint garden harbor hazel island juniper lantern maple meadow north oak "
    "orchard paper quiet river silver stone summit timber valley willow winter"
).split()


def vault_code(seed: int) -> str:
    digest = hashlib.sha256(f"qwen38-lc1:{int(seed)}".encode()).hexdigest()
    return f"VAULT-{digest[:10].upper()}"


def _encode(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [int(x) for x in ids]


def seeded_filler_ids(tokenizer, needed: int, seed: int):
    rng = np.random.default_rng(int(seed))
    out = []
    while len(out) < needed:
        words = rng.choice(FILLER_WORDS, size=max(256, needed - len(out)))
        out.extend(_encode(tokenizer, " ".join(words.tolist()) + ". "))
    return out[:needed]


def build_needle_ids(tokenizer, length: int, depth: float, seed: int):
    """Build an exactly ``length``-token, deterministic plain-text prompt."""
    if length < 64:
        raise ValueError("needle prompt length must be at least 64 tokens")
    if not 0.0 < depth < 1.0:
        raise ValueError("depth must be strictly between zero and one")
    code = vault_code(seed)
    header = _encode(tokenizer,
        "Read the following document carefully and retain every stated fact. Document:\n")
    fact = _encode(tokenizer, f" The vault access code is {code}. ")
    footer = _encode(tokenizer,
        "\nEnd of document. What is the vault access code? Reply with the exact code only.\nAnswer:")
    filler_n = length - len(header) - len(fact) - len(footer)
    if filler_n < 1:
        raise ValueError(
            f"length={length} too small for protocol overhead "
            f"{len(header) + len(fact) + len(footer)}")
    filler = seeded_filler_ids(tokenizer, filler_n, seed)
    # Depth is defined over the complete input-token sequence, not merely the
    # filler subsection. Protocol overhead causes only a clamp at tiny lengths.
    split = min(filler_n, max(0, int(round(depth * length)) - len(header)))
    ids = header + filler[:split] + fact + filler[split:] + footer
    assert len(ids) == length
    return np.ascontiguousarray(np.asarray(ids, np.int64).reshape(1, -1)), {
        "expected": code,
        "needle_token_start": len(header) + split,
        "actual_depth": (len(header) + split) / length,
        "prompt_sha256": hashlib.sha256(
            np.asarray(ids, dtype=np.int64).tobytes()).hexdigest(),
    }


def build_coherence_ids(tokenizer, length: int, seed: int):
    footer = _encode(tokenizer,
        "\nContinue this document in coherent prose without mentioning these instructions:\n")
    header = _encode(tokenizer, "A field journal records a long journey.\n")
    filler_n = length - len(header) - len(footer)
    if filler_n < 1:
        raise ValueError("coherence length too small")
    filler = seeded_filler_ids(tokenizer, filler_n, seed)
    ids = header + filler + footer
    return np.ascontiguousarray(np.asarray(ids, np.int64).reshape(1, -1))


def load_model(args, sampler):
    sampler.set_phase("load")
    model, info = Qwen38_TC.from_pretrained(
        args.model_dir, args.cache_dir,
        lm_head_chunk_rows=args.lm_head_chunk_rows,
        max_context=args.max_context, kv_int8=args.kv_int8,
        kv_host=args.kv_host, prefill_chunk=args.prefill_chunk,
        kv_block_rows=args.kv_block_rows,
        force_tiled_attention=getattr(args, "tf_margin", False),
        cache_read_only=True)
    sampler.sample_now("after_load")
    print("LOAD_RESULT " + json.dumps(info, sort_keys=True), flush=True)
    return model


def _kv_mode(args):
    return ("host_bf16" if args.kv_host else
            ("device_int8" if args.kv_int8 else "device_bf16"))


def validate_budget_for_run(args, budget):
    """Keep fail-loud default; only the explicit descent flag may override."""
    return validate_budget_for_load(
        budget, force_alloc=args.force_alloc,
        max_context=args.max_context, kv_mode=_kv_mode(args))


def is_tensor_cuda_oom(exc):
    """Recognize TensorCUDA's pybind-translated std::runtime_error OOMs."""
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    allocation_prefix = ("cudamalloc failed:", "cudamallocasync failed:")
    return (message.startswith(allocation_prefix)
            or ("cuda" in message and "out of memory" in message))


def _sampler_phase(sampler):
    phase = str(getattr(sampler, "phase", "load")).split(":", 1)[0]
    return phase if phase in {"load", "prefill", "decode"} else "load"


def int8_teacher_forced_agreement(model, tokenizer, ids, steps, sampler):
    """Compare INT8 top-1 on the BF16 greedy trajectory for exactly N steps."""
    model.kv_host = False
    model.kv_int8 = False
    ref = greedy(model, tokenizer, ids, steps, sampler, "kv_bf16_reference")
    reference = ref["token_ids"]
    tc.empty_cache()
    model.kv_int8 = True
    caches = model.new_caches(ids.shape[0])
    sampler.set_phase("prefill:kv_int8_teacher_forced")
    logits, caches = model(ids, caches=caches, position_offset=0,
                           last_token_only=True)
    matches = []
    observed = []
    for step, reference_token in enumerate(reference):
        top1 = int(np.argmax(logits.float().numpy().reshape(-1)))
        observed.append(top1)
        matches.append(top1 == reference_token)
        if step + 1 == len(reference):
            break
        one = np.asarray([[reference_token]], dtype=np.int64)
        sampler.set_phase("decode:kv_int8_teacher_forced")
        logits, caches = model(one, caches=caches,
                               position_offset=ids.shape[1] + step,
                               last_token_only=True)
        tc.synchronize()
    return {
        "steps": len(matches), "matches": int(sum(matches)),
        "agreement": float(sum(matches) / len(matches)) if matches else 0.0,
        "bf16_token_ids": reference, "int8_teacher_forced_top1": observed,
        "kv_key_stats_first_prefill_chunk": [
            {"attention_layer": i, **cache.key_stats}
            for i, cache in enumerate(caches)
            if getattr(cache, "key_stats", None) is not None
        ],
        "status": "PASS" if matches and sum(matches) / len(matches) >= 0.99 else "RED",
    }


def teacher_forced_tiled_margins(model, tokenizer, sampler):
    """Report tiled top-1 agreement on every registered baseline token."""
    if getattr(model, "attn_path", None) != "tiled":
        raise RuntimeError("--tf-margin requires the tiled attention path")
    summaries = {}
    for name, prompt in DEMO_PROMPTS.items():
        ids = chat_ids(tokenizer, prompt)
        reference = BASELINE_TOKEN_IDS[name]
        if ids.shape[1] + len(reference) > model.max_context:
            raise ValueError(
                f"teacher-forced {name} needs {ids.shape[1] + len(reference)} "
                f"tokens, configured --max-context is {model.max_context}")
        caches = model.take_preallocated_caches(ids.shape[0])
        sampler.set_phase(f"prefill:tf_margin_{name}")
        logits, caches = model(ids, caches=caches, position_offset=0,
                               last_token_only=True)
        matches = 0
        disagreements = []
        for step, baseline_token in enumerate(reference):
            values = logits.float().numpy().reshape(-1)
            if values.size < 2 or not np.isfinite(values).all():
                raise FloatingPointError(
                    f"invalid logits case={name} teacher_forced_step={step}")
            top1 = int(np.argmax(values))
            top_two = np.partition(values, -2)[-2:]
            margin = float(top_two.max() - top_two.min())
            agrees = top1 == baseline_token
            matches += int(agrees)
            step_result = {
                "status": "REPORT_ONLY", "case": name, "step": step,
                "baseline_token": int(baseline_token), "tiled_top1": top1,
                "top1_agreement": agrees,
            }
            if not agrees:
                step_result["logit_margin_top1_minus_top2"] = margin
                disagreements.append(step_result.copy())
            print("TF_MARGIN_STEP " + json.dumps(step_result, sort_keys=True),
                  flush=True)
            if step + 1 == len(reference):
                break
            one = np.asarray([[baseline_token]], dtype=np.int64)
            sampler.set_phase(f"decode:tf_margin_{name}")
            logits, caches = model(
                one, caches=caches, position_offset=ids.shape[1] + step,
                last_token_only=True)
            tc.synchronize()
        summaries[name] = {
            "steps": len(reference), "matches": matches,
            "agreement": matches / len(reference) if reference else 0.0,
            "disagreements": disagreements,
        }
        del logits, caches
        tc.empty_cache()
    return {"status": "REPORT_ONLY", "attn_path": model.attn_path,
            "cases": summaries}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--length", type=int, default=8192,
                    help="exact input-token length")
    ap.add_argument("--depth", type=float, default=0.5,
                    choices=(0.25, 0.5, 0.75))
    ap.add_argument("--seed", type=int, default=38001)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-context", type=int, default=None,
                    help="KV/RoPE capacity; defaults to length+decode tokens")
    ap.add_argument("--prefill-chunk", type=int, default=DEFAULT_PREFILL_CHUNK)
    ap.add_argument("--kv-block-rows", type=int, default=DEFAULT_KV_BLOCK)
    ap.add_argument("--lm-head-chunk-rows", type=int,
                    default=DEFAULT_LM_HEAD_CHUNK_ROWS)
    ap.add_argument("--kv-int8", action="store_true")
    ap.add_argument("--kv-host", action="store_true")
    ap.add_argument("--force-alloc", action="store_true",
                    help="override the predicted VRAM wall and attempt allocation")
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--coherence-smoke", action="store_true")
    modes.add_argument("--baseline-check", action="store_true")
    modes.add_argument("--kv-agreement", action="store_true")
    modes.add_argument("--tf-margin", action="store_true",
                       help="teacher-force tiled attention on G-LC0 baselines")
    ap.add_argument("--dry-run", action="store_true",
                    help="construct/validate prompt only; do not import GPU runtime")
    args = ap.parse_args()
    if args.kv_int8 and args.kv_host:
        ap.error("--kv-int8 and --kv-host are mutually exclusive")
    if args.max_context is None:
        args.max_context = (DEFAULT_MAX_CONTEXT
                            if args.baseline_check or args.tf_margin else
                            args.length + args.max_new_tokens)
    return args


def main():
    args = parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)

    if args.baseline_check or args.tf_margin:
        ids, meta = None, {}
    elif args.coherence_smoke:
        ids = build_coherence_ids(tokenizer, args.length, args.seed)
        meta = {"prompt_sha256": hashlib.sha256(ids.tobytes()).hexdigest()}
    else:
        ids, meta = build_needle_ids(
            tokenizer, args.length, args.depth, args.seed)
    if args.dry_run:
        print("DRY_RUN_RESULT " + json.dumps({
            "status": "PASS", "length": (int(ids.shape[1]) if ids is not None
                                             else None),
            "depth": args.depth, "seed": args.seed, **meta,
        }, sort_keys=True))
        return

    if (args.baseline_check or args.tf_margin) and any(
            v == "REPLACE-ME" for v in BASELINE_TOKEN_IDS.values()):
        raise RuntimeError(
            "G-LC0 baseline is unregistered: replace BASELINE_TOKEN_IDS "
            "REPLACE-ME values from the preserved 2026-08-19 receipt")

    budget = print_budget(
        args.model_dir, args.max_context, args.kv_int8, args.kv_host,
        args.lm_head_chunk_rows, args.prefill_chunk, args.kv_block_rows)
    sampler = VramSampler()
    sampler.start()
    try:
        validate_budget_for_run(args, budget)
    except MemoryError as exc:
        sampler.stop()
        print("RECALL_RESULT " + json.dumps({
            "status": "RED", "length": args.length, "depth": args.depth,
            "seed": args.seed, "error": f"MemoryError: {exc}",
            "vram": sampler.receipt(),
        }, sort_keys=True), flush=True)
        raise SystemExit(1)
    result = None
    error = None
    forced_oom = False
    try:
        model = load_model(args, sampler)
        if args.baseline_check:
            cases = {}
            for name, prompt in DEMO_PROMPTS.items():
                got = greedy(model, tokenizer, chat_ids(tokenizer, prompt),
                             args.max_new_tokens, sampler, f"baseline_{name}")
                expected = BASELINE_TOKEN_IDS[name]
                cases[name] = {"match": got["token_ids"] == expected,
                               "expected": expected, "got": got["token_ids"]}
            ok = all(case["match"] for case in cases.values())
            result = {"status": "PASS" if ok else "RED", "cases": cases}
            tag = "BASELINE_RESULT"
        elif args.tf_margin:
            result = teacher_forced_tiled_margins(model, tokenizer, sampler)
            tag = "TF_MARGIN_RESULT"
        elif args.kv_agreement:
            result = int8_teacher_forced_agreement(
                model, tokenizer, ids, args.max_new_tokens, sampler)
            result.update({"length": args.length, "seed": args.seed})
            tag = "KV_AGREEMENT_RESULT"
        elif args.coherence_smoke:
            got = greedy(model, tokenizer, ids, 64, sampler, "coherence")
            print("===== RAW COHERENCE OUTPUT =====")
            print(got["text"])
            print("===== END RAW COHERENCE OUTPUT =====")
            result = {"status": "REPORT_ONLY", "length": args.length,
                      "seed": args.seed, "token_ids": got["token_ids"],
                      "raw": got["text"]}
            tag = "COHERENCE_RESULT"
        else:
            got = greedy(model, tokenizer, ids, args.max_new_tokens,
                         sampler, f"needle_{args.length}_{args.depth}_{args.seed}")
            recalled = meta["expected"] in got["text"]
            result = {
                "status": "PASS" if recalled else "RED",
                "recalled": recalled, "length": args.length,
                "depth": args.depth, "seed": args.seed,
                "kv_mode": "host_bf16" if args.kv_host else
                           ("device_int8" if args.kv_int8 else "device_bf16"),
                "expected": meta["expected"], "raw": got["text"],
                "token_ids": got["token_ids"],
                "prefill_seconds": got["prefill_seconds"],
                "decode_seconds": got["decode_seconds"],
                "prompt_sha256": meta["prompt_sha256"],
                "needle_token_start": meta["needle_token_start"],
                "actual_depth": meta["actual_depth"],
            }
            if "kv_key_stats_first_prefill_chunk" in got:
                result["kv_key_stats_first_prefill_chunk"] = got[
                    "kv_key_stats_first_prefill_chunk"]
            tag = "RECALL_RESULT"
    except Exception as exc:
        if args.force_alloc and is_tensor_cuda_oom(exc):
            forced_oom = True
            tag = "RECALL_RESULT"
            result = {
                "status": "RED", "error": f"CUDA_OOM: {exc}",
                "phase": _sampler_phase(sampler),
            }
        else:
            error = exc
            tag = ("TF_MARGIN_RESULT" if args.tf_margin else "RECALL_RESULT")
            result = {"status": "RED", "length": args.length,
                      "depth": args.depth, "seed": args.seed,
                      "error": f"{type(exc).__name__}: {exc}"}
    finally:
        sampler.stop()
    result["vram"] = sampler.receipt()
    print(tag + " " + json.dumps(result, sort_keys=True), flush=True)
    if error is not None:
        raise error
    if forced_oom or result["status"] == "RED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
