#!/usr/bin/env python3
"""APAMQ-FB-D1 blend-vs-fused decode diagnostics.

Every invocation writes one self-contained JSON receipt:

  python3 scripts/apamq_fbd1_diag.py --leg <name> \
      --output artifacts/apamq_fbd1/<name>.json

GPU legs are ``bits4``, ``bits6``, ``bits8``, ``force_all``, and ``int4``.
``summary`` consumes the four non-INT4 receipts without loading a model;
``cpu-check`` exercises argument/root resolution and the reduction replays.

Selection observation is deliberately stronger than a cuBLAS-fp32 proxy:

* blend records the actual bf16 ``bulk`` matrix received by
  ``apa_blend_softmax`` and replays that kernel's 256-thread population-stat
  reduction;
* fused obtains the threshold actually saved by ``apa_selective_fwd_train``
  and exactly replays the D=512 warp-cooperative dot on the captured bf16
  q/kq values (bf16 products are exactly representable in fp32).

The normal fused output still comes from ``apa_selective_attention``.  The
training-forward output is retained only as a diagnostic cross-check.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
from typing import Any, Callable

import numpy as np


REPO = Path(__file__).resolve().parents[1]
CANONICAL_TC_ROOT = Path("/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
FA_TC_ROOT = Path("/mnt/ForgeRealm/wt/apamq-fa/tensor_cuda")


def requested_leg(argv: list[str]) -> str | None:
    try:
        return argv[argv.index("--leg") + 1]
    except (ValueError, IndexError):
        return None


TC_ROOT = Path(os.environ.get(
    "TENSOR_CUDA_ROOT",
    str(FA_TC_ROOT if requested_leg(sys.argv) == "int4" else CANONICAL_TC_ROOT),
)).expanduser().resolve()
PROMPT_LEN = 8192
DECODE_STEPS = 64
GLOBAL_LAYERS = tuple(range(5, 48, 6))
MAX_ABS_LOGIT_DELTA = 0.5
SEED = 20260813
LEG_CONFIG = {
    "bits4": {"bulk_bits": 4, "refine_percentile": 0.15},
    "bits6": {"bulk_bits": 6, "refine_percentile": 0.15},
    "bits8": {"bulk_bits": 8, "refine_percentile": 0.15},
    "force_all": {"bulk_bits": 4, "refine_percentile": 1.0},
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def tensor_float_numpy(t: Any) -> np.ndarray:
    return np.ascontiguousarray(t.float().numpy(), dtype=np.float32)


def json_scalar(value: float) -> float | str:
    if math.isinf(value):
        return "-inf" if value < 0 else "+inf"
    if math.isnan(value):
        return "nan"
    return float(value)


def safe_abs_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    same_inf = np.isinf(a) & np.isinf(b) & (np.signbit(a) == np.signbit(b))
    with np.errstate(invalid="ignore"):
        out = np.abs(a - b)
    out[same_inf] = 0.0
    return out


def _f32_add_inplace(dst: np.ndarray, src: np.ndarray) -> None:
    np.add(dst, src, out=dst, casting="unsafe")


def fma_f32(a: Any, b: Any, c: Any) -> np.ndarray:
    """Software float32 FMA using extended precision then one f32 rounding."""
    return np.asarray(
        np.asarray(a, dtype=np.longdouble) * np.asarray(b, dtype=np.longdouble)
        + np.asarray(c, dtype=np.longdouble), dtype=np.float32)


def fused_wcoop_bulk(q: np.ndarray, kq: np.ndarray,
                     scale: float) -> np.ndarray:
    """Replay apa_selective D=512 WCOOP dots in CUDA operation order."""
    if q.ndim != 4 or kq.ndim != 4 or q.shape[0] != 1:
        raise AssertionError((q.shape, kq.shape))
    _, heads, length, dim = q.shape
    _, kv_heads, seq, kdim = kq.shape
    if length != 1 or dim != 512 or kdim != dim or heads % kv_heads:
        raise AssertionError((q.shape, kq.shape))
    qv = q[0, :, 0, :]
    kv = kq[0]
    group = heads // kv_heads
    acc = np.zeros((heads, seq, 32), dtype=np.float32)
    for dl in range(16):
        cols = slice(dl * 32, (dl + 1) * 32)
        prod = np.empty_like(acc)
        for h in range(heads):
            np.multiply(qv[h, cols], kv[h // group, :, cols],
                        out=prod[h], casting="unsafe")
        _f32_add_inplace(acc, prod)
    for off in (16, 8, 4, 2, 1):
        np.add(acc[:, :, :off], acc[:, :, off:2 * off],
               out=acc[:, :, :off], casting="unsafe")
    return np.multiply(acc[:, :, 0], np.float32(scale), dtype=np.float32)


def fused_wcoop_threshold(bulk: np.ndarray, zthr: float) -> np.ndarray:
    """Replay the four-warp stats accumulation/tree for D=512 decode."""
    vals = np.abs(np.asarray(bulk, dtype=np.float32))
    heads, seq = vals.shape
    sums = np.zeros((heads, 4), dtype=np.float32)
    sumsqs = np.zeros((heads, 4), dtype=np.float32)
    for start in range(0, seq, 4):
        width = min(4, seq - start)
        v = vals[:, start:start + width]
        np.add(sums[:, :width], v, out=sums[:, :width])
        sumsqs[:, :width] = fma_f32(v, v, sumsqs[:, :width])

    # red[0]+=red[64], red[32]+=red[96], then red[0]+=red[32].
    total = np.add(sums[:, 0], sums[:, 2], dtype=np.float32)
    other = np.add(sums[:, 1], sums[:, 3], dtype=np.float32)
    total = np.add(total, other, dtype=np.float32)
    total_sq = np.add(sumsqs[:, 0], sumsqs[:, 2], dtype=np.float32)
    other_sq = np.add(sumsqs[:, 1], sumsqs[:, 3], dtype=np.float32)
    total_sq = np.add(total_sq, other_sq, dtype=np.float32)
    cnt = np.float32(seq)
    mean = np.divide(total, cnt, dtype=np.float32)
    var = fma_f32(-mean, mean, np.divide(total_sq, cnt, dtype=np.float32))
    np.maximum(var, np.float32(0.0), out=var)
    std = np.sqrt(var, dtype=np.float32)
    if math.isinf(zthr) and zthr < 0:
        return np.full(heads, -np.inf, dtype=np.float32)
    return fma_f32(std, np.float32(zthr), mean)


def blend_kernel_threshold(bulk: np.ndarray, zthr: float) -> np.ndarray:
    """Replay apa_blend_softmax_kernel2's 256-thread stats tree."""
    vals = np.abs(np.asarray(bulk, dtype=np.float32))
    heads, seq = vals.shape
    sums = np.zeros((heads, 256), dtype=np.float32)
    sumsqs = np.zeros((heads, 256), dtype=np.float32)
    for start in range(0, seq, 256):
        width = min(256, seq - start)
        v = vals[:, start:start + width]
        np.add(sums[:, :width], v, out=sums[:, :width])
        sumsqs[:, :width] = fma_f32(v, v, sumsqs[:, :width])
    for off in (128, 64, 32, 16, 8, 4, 2, 1):
        np.add(sums[:, :off], sums[:, off:2 * off], out=sums[:, :off])
        np.add(sumsqs[:, :off], sumsqs[:, off:2 * off],
               out=sumsqs[:, :off])
    cnt = np.float32(seq)
    mean = np.divide(sums[:, 0], cnt, dtype=np.float32)
    var = fma_f32(-mean, mean,
                  np.divide(sumsqs[:, 0], cnt, dtype=np.float32))
    np.maximum(var, np.float32(0.0), out=var)
    std = np.sqrt(var, dtype=np.float32)
    if math.isinf(zthr) and zthr < 0:
        return np.full(heads, -np.inf, dtype=np.float32)
    return fma_f32(std, np.float32(zthr), mean)


def coord(runtime: dict[str, Any], path: str) -> tuple[int, int]:
    n = runtime["call_counts"][path]
    runtime["call_counts"][path] = n + 1
    step, ordinal = divmod(n, len(GLOBAL_LAYERS))
    if step != runtime["decode_step"]:
        raise AssertionError(
            f"{path} call coordinate drift: derived step={step}, runtime="
            f"{runtime['decode_step']}")
    return step, GLOBAL_LAYERS[ordinal]


def configure_model(model: Any, *, bits: int, refine: float) -> None:
    for layer in model.layers:
        if layer.mixer.is_global:
            layer.mixer.attention_mode = "apa_selective"
            layer.mixer.refine_percentile = refine
            layer.mixer.bulk_bits = bits
            layer.mixer.apa_min_context = 2048
            layer.mixer.fast_max_seq = 4096


def ring_receipt(caches: list[Any], gemma: Any) -> dict[str, Any]:
    rows = []
    for layer in GLOBAL_LAYERS:
        cache = caches[layer]
        rows.append({
            "layer": layer,
            "is_kv_ring": isinstance(cache, gemma.KVRing),
            "count": int(cache.count),
            "kq_count": int(cache.kq_count),
            "kqb_is_none": cache.kqb is None,
        })
    return {"layers": rows}


def run_model_path(model: Any, tc: Any, gemma: Any, ids: np.ndarray,
                   runtime: dict[str, Any], *, path: str,
                   forced_tokens: list[int] | None = None,
                   int4: bool = False) -> dict[str, Any]:
    os.environ["GEMMA4_APA_DECODE_FUSED"] = "0" if path == "blend" else "1"
    # Prefill is common and non-INT4 so both comparisons begin with the same
    # algorithm/state. Enable the F-A ABI only at the L=1 boundary.
    os.environ["GEMMA4_APA_INT4"] = "0"
    runtime["phase"] = "prefill"
    runtime["path"] = path
    runtime["call_counts"][path] = 0
    started = time.perf_counter()
    with tc.no_grad():
        logits, caches = model(ids, last_token_only=True)
        next_token = int(logits.float().numpy()[0, -1].argmax())
        os.environ["GEMMA4_APA_INT4"] = "1" if int4 else "0"
        inputs: list[int] = []
        predictions: list[int] = []
        rows: list[np.ndarray] = []
        runtime["phase"] = "decode"
        for step in range(DECODE_STEPS):
            runtime["decode_step"] = step
            if forced_tokens is not None:
                next_token = int(forced_tokens[step])
            inputs.append(next_token)
            logits, caches = model(
                np.array([[next_token]], dtype=np.int64), caches=caches,
                position_offset=PROMPT_LEN + step, last_token_only=True)
            row = tensor_float_numpy(logits)[0, -1]
            rows.append(row)
            next_token = int(row.argmax())
            predictions.append(next_token)
    tc.synchronize()
    elapsed = time.perf_counter() - started
    rings = ring_receipt(caches, gemma)
    del caches
    tc.empty_cache()
    runtime["phase"] = "idle"
    return {
        "input_tokens": inputs,
        "predictions": predictions,
        "logits": rows,
        "wall_s": elapsed,
        "rings": rings,
    }


def logit_metrics(a: list[np.ndarray], b: list[np.ndarray]) -> dict[str, Any]:
    max_abs = 0.0
    total_abs = 0.0
    count = 0
    per_step = []
    for step, (x, y) in enumerate(zip(a, b)):
        d = np.abs(x - y)
        mx = float(d.max())
        mean = float(d.mean())
        max_abs = max(max_abs, mx)
        total_abs += float(d.sum(dtype=np.float64))
        count += d.size
        per_step.append({"step": step, "max_abs": mx, "mean_abs": mean})
    return {
        "max_abs": max_abs,
        "mean_abs": total_abs / count,
        "per_step": per_step,
    }


def install_selection_instrumentation(
        tc: Any, mistral: Any, runtime: dict[str, Any]
        ) -> tuple[list[dict[str, Any]], Callable[[], None]]:
    orig_blend = mistral._cublas_blend_attention
    orig_softmax = tc.apa_blend_softmax
    orig_fused = tc.apa_selective_attention
    paired: list[dict[str, Any]] = []

    def blend_call(q: Any, k: Any, kq: Any, v: Any, group: int,
                   scale: float, zthr: float, causal: bool, blk: int) -> Any:
        if runtime["phase"] != "decode" or int(q.shape[2]) != 1:
            return orig_blend(q, k, kq, v, group, scale, zthr, causal, blk)
        key = coord(runtime, "blend")
        rec: dict[str, Any] = {
            "step": key[0], "layer": key[1], "scale": float(scale),
            "zthr": float(zthr), "causal": bool(causal),
            "valid_count": int(k.shape[2]), "softmax_calls": 0,
        }
        runtime["current_blend"] = rec
        try:
            blend_out = orig_blend(
                q, k, kq, v, group, scale, zthr, causal, blk)
        finally:
            runtime["current_blend"] = None

        # D1 is a shadow comparison on the exact same q/k/kq/v objects.  Only
        # blend_out is returned to the model, so the probe cannot perturb the
        # state whose next layer/step is being observed.
        shadow_key = coord(runtime, "fused_shadow")
        if shadow_key != key:
            raise AssertionError((key, shadow_key))
        q_np = tensor_float_numpy(q)
        kq_np = tensor_float_numpy(kq)
        diag_out, _, engine_thr = tc.apa_selective_fwd_train(
            q, k, kq, v, scale, zthr, causal)
        threshold = tensor_float_numpy(engine_thr).reshape(-1)
        scores = fused_wcoop_bulk(q_np, kq_np, scale)
        replay_thr = fused_wcoop_threshold(scores, zthr)
        selection = np.abs(scores) >= threshold[:, None]
        fused_out = orig_fused(q, k, kq, v, scale, zthr, causal)
        blend_np = tensor_float_numpy(blend_out)
        fused_np = tensor_float_numpy(fused_out)
        diag_np = tensor_float_numpy(diag_out)

        bsel = rec["selection"]
        if np.isfinite(rec["threshold"]).all():
            blend_ulp = np.abs(np.spacing(rec["threshold"]))
            blend_boundary = int((
                np.abs(np.abs(rec["bulk_scores"]) - rec["threshold"][:, None])
                <= 4.0 * blend_ulp[:, None]).sum())
        else:
            blend_boundary = 0
        intersection = int(np.logical_and(bsel, selection).sum())
        union = int(np.logical_or(bsel, selection).sum())
        flips = int(np.logical_xor(bsel, selection).sum())
        attn_delta = np.abs(blend_np - fused_np)
        score_delta = np.abs(rec["bulk_scores"] - scores)
        thr_delta = safe_abs_delta(rec["threshold"], threshold)
        replay_delta = safe_abs_delta(replay_thr, threshold)
        diag_delta = np.abs(diag_np - fused_np)
        paired.append({
            "step": key[0], "layer": key[1],
            "same_state_tensor_inputs": True,
            "valid_count_blend": rec["valid_count"],
            "valid_count_fused": int(k.shape[2]),
            "scale_blend": rec["scale"], "scale_fused": float(scale),
            "zthr_blend": json_scalar(rec["zthr"]),
            "zthr_fused": json_scalar(float(zthr)),
            "refined_count_blend": int(bsel.sum()),
            "refined_count_fused": int(selection.sum()),
            "candidate_count": int(selection.size),
            "intersection_count": intersection, "union_count": union,
            "flip_count": flips,
            "jaccard": 1.0 if union == 0 else intersection / union,
            "attention_output_max_abs_delta": float(attn_delta.max()),
            "attention_output_mean_abs_delta": float(attn_delta.mean()),
            "bulk_score_max_abs_delta": float(score_delta.max()),
            "bulk_score_mean_abs_delta": float(score_delta.mean()),
            "threshold_max_abs_delta": float(thr_delta.max()),
            "threshold_mean_abs_delta": float(thr_delta.mean()),
            "fused_threshold_replay_max_abs_delta": float(replay_delta.max()),
            "fused_train_vs_infer_output_max_abs_delta": float(diag_delta.max()),
            "blend_softmax_calls": rec["softmax_calls"],
            "blend_threshold_boundary_keys_4ulp": blend_boundary,
        })
        return blend_out

    def softmax_call(bulk: Any, rank: Any, zthr: float, Lq: int = 0,
                     row0: int = 0, window: int = 0) -> Any:
        rec = runtime.get("current_blend")
        if rec is not None:
            if int(Lq) != 0 or int(row0) != 0 or int(window) != 0:
                raise AssertionError((Lq, row0, window))
            scores = tensor_float_numpy(bulk).reshape(-1, int(bulk.shape[-1]))
            threshold = blend_kernel_threshold(scores, float(zthr))
            rec["bulk_scores"] = scores
            rec["threshold"] = threshold
            rec["selection"] = np.abs(scores) >= threshold[:, None]
            rec["softmax_calls"] += 1
        return orig_softmax(bulk, rank, zthr, Lq, row0, window)

    def fused_call(q: Any, k: Any, kq: Any, v: Any, scale: float,
                   zthr: float, causal: bool) -> Any:
        if runtime["phase"] != "decode" or int(q.shape[2]) != 1:
            return orig_fused(q, k, kq, v, scale, zthr, causal)
        coord(runtime, "fused")
        return orig_fused(q, k, kq, v, scale, zthr, causal)

    mistral._cublas_blend_attention = blend_call
    tc.apa_blend_softmax = softmax_call
    tc.apa_selective_attention = fused_call

    def restore() -> None:
        mistral._cublas_blend_attention = orig_blend
        tc.apa_blend_softmax = orig_softmax
        tc.apa_selective_attention = orig_fused

    return paired, restore


def correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    value = float(np.corrcoef(np.asarray(xs), np.asarray(ys))[0, 1])
    return value if math.isfinite(value) else None


def aggregate_selection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = DECODE_STEPS * len(GLOBAL_LAYERS)
    if len(rows) != expected:
        raise AssertionError(f"selection rows {len(rows)} != {expected}")
    candidate = sum(r["candidate_count"] for r in rows)
    intersection = sum(r["intersection_count"] for r in rows)
    union = sum(r["union_count"] for r in rows)
    identical = [r for r in rows if r["flip_count"] == 0]
    return {
        "calls": len(rows),
        "refined_fraction_blend": sum(r["refined_count_blend"] for r in rows) / candidate,
        "refined_fraction_fused": sum(r["refined_count_fused"] for r in rows) / candidate,
        "flip_count": sum(r["flip_count"] for r in rows),
        "flip_fraction": sum(r["flip_count"] for r in rows) / candidate,
        "jaccard_global": 1.0 if union == 0 else intersection / union,
        "jaccard_per_call_mean": statistics.mean(r["jaccard"] for r in rows),
        "jaccard_per_call_min": min(r["jaccard"] for r in rows),
        "attention_output_max_abs_delta": max(
            r["attention_output_max_abs_delta"] for r in rows),
        "attention_output_mean_of_call_means": statistics.mean(
            r["attention_output_mean_abs_delta"] for r in rows),
        "flip_vs_attention_max_pearson": correlation(
            [float(r["flip_count"]) for r in rows],
            [r["attention_output_max_abs_delta"] for r in rows]),
        "identical_selection_calls": len(identical),
        "identical_selection_attention_max_abs_delta": (
            max(r["attention_output_max_abs_delta"] for r in identical)
            if identical else None),
        "bulk_score_max_abs_delta": max(r["bulk_score_max_abs_delta"] for r in rows),
        "bulk_score_mean_of_call_means": statistics.mean(
            r["bulk_score_mean_abs_delta"] for r in rows),
        "threshold_max_abs_delta": max(r["threshold_max_abs_delta"] for r in rows),
        "fused_threshold_replay_max_abs_delta": max(
            r["fused_threshold_replay_max_abs_delta"] for r in rows),
        "fused_train_vs_infer_output_max_abs_delta": max(
            r["fused_train_vs_infer_output_max_abs_delta"] for r in rows),
        "blend_threshold_boundary_keys_4ulp": sum(
            r["blend_threshold_boundary_keys_4ulp"] for r in rows),
        "semantic_controls": {
            "valid_count_equal_all_calls": all(
                r["valid_count_blend"] == r["valid_count_fused"] for r in rows),
            "scale_equal_all_calls": all(
                r["scale_blend"] == r["scale_fused"] for r in rows),
            "zthr_equal_all_calls": all(
                r["zthr_blend"] == r["zthr_fused"] for r in rows),
            "one_blend_softmax_per_call": all(
                r["blend_softmax_calls"] == 1 for r in rows),
            "same_state_tensor_inputs_all_calls": all(
                r["same_state_tensor_inputs"] for r in rows),
        },
    }


def run_bits_leg(leg: str, receipt: dict[str, Any]) -> None:
    cfg = LEG_CONFIG[leg]
    sys.path.insert(0, str(TC_ROOT))
    sys.path.insert(0, str(REPO))
    import tensor_cuda as tc
    import core.gemma4_tc as gemma
    import core.mistral7b_tc as mistral

    required = ("apa_selective_attention", "apa_selective_fwd_train",
                "apa_blend_softmax")
    missing = [name for name in required if not hasattr(tc, name)]
    if missing:
        raise RuntimeError(f"engine missing diagnostic ABI: {missing}")
    tc.set_alloc_pooling(True)
    model, info = gemma.Gemma4_TC.from_pretrained()
    configure_model(model, bits=cfg["bulk_bits"],
                    refine=cfg["refine_percentile"])
    ids = np.random.default_rng(SEED).integers(
        1000, 200000, size=(1, PROMPT_LEN), dtype=np.int64)
    runtime: dict[str, Any] = {
        "phase": "idle", "path": None, "decode_step": None,
        "current_blend": None,
        "call_counts": {"blend": 0, "fused_shadow": 0, "fused": 0},
    }
    rows, restore = install_selection_instrumentation(tc, mistral, runtime)
    try:
        blend = run_model_path(model, tc, gemma, ids, runtime, path="blend")
        fused = run_model_path(
            model, tc, gemma, ids, runtime, path="fused",
            forced_tokens=blend["input_tokens"])
    finally:
        restore()
        os.environ.pop("GEMMA4_APA_DECODE_FUSED", None)
        os.environ.pop("GEMMA4_APA_INT4", None)

    logit = logit_metrics(blend["logits"], fused["logits"])
    selection = aggregate_selection(rows)
    expected_calls = DECODE_STEPS * len(GLOBAL_LAYERS)
    expected_census = {
        "blend": expected_calls, "fused_shadow": expected_calls,
        "fused": expected_calls,
    }
    if runtime["call_counts"] != expected_census:
        raise AssertionError((runtime["call_counts"], expected_census))
    if leg == "force_all" and not all(
            r["refined_count_blend"] == r["candidate_count"]
            and r["refined_count_fused"] == r["candidate_count"]
            for r in rows):
        raise AssertionError("force_all did not refine every valid key")
    token_match = blend["predictions"] == fused["predictions"]
    receipt.update({
        "status": "complete", "model_info": {k: str(v) for k, v in info.items()},
        "bulk_bits": cfg["bulk_bits"],
        "refine_percentile": cfg["refine_percentile"],
        "branch_census": dict(runtime["call_counts"]),
        "common_input_stream": True,
        "greedy_prediction_match": token_match,
        "logits": logit,
        "selection": selection,
        "per_layer_step": rows,
        "rings": {"blend": blend["rings"], "fused": fused["rings"]},
        "wall_s": {"blend": blend["wall_s"], "fused": fused["wall_s"]},
        "registered_raw_logit_gate": {
            "tolerance": MAX_ABS_LOGIT_DELTA,
            "pass": token_match and logit["max_abs"] <= MAX_ABS_LOGIT_DELTA,
        },
    })
    print(f"MEASUREMENT {leg}: PASS calls={len(rows)}", flush=True)
    print(f"TOKENS: {'MATCH' if token_match else 'DIFF'} | "
          f"max|delta logit|={logit['max_abs']:.6f} | "
          f"mean|delta logit|={logit['mean_abs']:.6f}", flush=True)
    print(f"SELECTION: jaccard={selection['jaccard_global']:.9f} "
          f"flips={selection['flip_count']} "
          f"refined(blend/fused)="
          f"{selection['refined_fraction_blend']:.9f}/"
          f"{selection['refined_fraction_fused']:.9f}", flush=True)
    print("RAW-LOGIT GATE: " + (
        "PASS" if receipt["registered_raw_logit_gate"]["pass"] else "FAIL"),
        flush=True)


def install_int4_instrumentation(
        tc: Any, mistral: Any, runtime: dict[str, Any]
        ) -> tuple[list[dict[str, Any]], Callable[[], None]]:
    orig_blend = mistral._cublas_blend_attention
    orig_int4 = tc.apa_selective_attention_int4
    rows: list[dict[str, Any]] = []

    def blend_call(q: Any, k: Any, kq: Any, v: Any, group: int,
                   scale: float, zthr: float, causal: bool, blk: int) -> Any:
        if runtime["phase"] != "decode" or int(q.shape[2]) != 1:
            return orig_blend(q, k, kq, v, group, scale, zthr, causal, blk)
        key = coord(runtime, "blend")
        blend_out = orig_blend(q, k, kq, v, group, scale, zthr, causal, blk)
        shadow_key = coord(runtime, "int4_shadow")
        if shadow_key != key:
            raise AssertionError((key, shadow_key))
        int4_out = orig_int4(q, k, v, scale, zthr, causal)
        delta = np.abs(
            tensor_float_numpy(blend_out) - tensor_float_numpy(int4_out))
        rows.append({
            "step": key[0], "layer": key[1],
            "same_state_tensor_inputs": True,
            "attention_output_max_abs_delta": float(delta.max()),
            "attention_output_mean_abs_delta": float(delta.mean()),
            "valid_count": int(k.shape[2]), "scale": float(scale),
            "zthr": float(zthr),
        })
        return blend_out

    def int4_call(q: Any, k: Any, v: Any, scale: float,
                  zthr: float, causal: bool) -> Any:
        if runtime["phase"] == "decode" and int(q.shape[2]) == 1:
            coord(runtime, "int4")
        return orig_int4(q, k, v, scale, zthr, causal)

    mistral._cublas_blend_attention = blend_call
    tc.apa_selective_attention_int4 = int4_call

    def restore() -> None:
        mistral._cublas_blend_attention = orig_blend
        tc.apa_selective_attention_int4 = orig_int4

    return rows, restore


def run_int4_leg(receipt: dict[str, Any]) -> None:
    sys.path.insert(0, str(TC_ROOT))
    sys.path.insert(0, str(REPO))
    import tensor_cuda as tc
    import core.gemma4_tc as gemma
    import core.mistral7b_tc as mistral

    if not hasattr(tc, "apa_selective_attention_int4"):
        receipt.update({
            "status": "skipped", "skip_reason": "engine entry absent",
            "engine_has_apa_selective_attention_int4": False,
        })
        print(f"INT4 LEG: SKIP engine entry absent (root={TC_ROOT})", flush=True)
        return
    tc.set_alloc_pooling(True)
    model, info = gemma.Gemma4_TC.from_pretrained()
    configure_model(model, bits=4, refine=0.15)
    ids = np.random.default_rng(SEED).integers(
        1000, 200000, size=(1, PROMPT_LEN), dtype=np.int64)
    runtime: dict[str, Any] = {
        "phase": "idle", "path": None, "decode_step": None,
        "call_counts": {"blend": 0, "int4_shadow": 0, "int4": 0},
    }
    rows, restore = install_int4_instrumentation(tc, mistral, runtime)
    try:
        blend = run_model_path(model, tc, gemma, ids, runtime, path="blend")
        target = run_model_path(
            model, tc, gemma, ids, runtime, path="int4",
            forced_tokens=blend["input_tokens"], int4=True)
    finally:
        restore()
        os.environ.pop("GEMMA4_APA_DECODE_FUSED", None)
        os.environ.pop("GEMMA4_APA_INT4", None)
    logit = logit_metrics(blend["logits"], target["logits"])
    expected_calls = DECODE_STEPS * len(GLOBAL_LAYERS)
    expected_census = {
        "blend": expected_calls, "int4_shadow": expected_calls,
        "int4": expected_calls,
    }
    if len(rows) != expected_calls or runtime["call_counts"] != expected_census:
        raise AssertionError((len(rows), runtime["call_counts"], expected_census))
    token_match = blend["predictions"] == target["predictions"]
    ring_drop = all(
        r["is_kv_ring"] and r["kqb_is_none"] and r["kq_count"] == 0
        for r in target["rings"]["layers"])
    receipt.update({
        "status": "complete", "model_info": {k: str(v) for k, v in info.items()},
        "engine_has_apa_selective_attention_int4": True,
        "common_non_int4_prefill": True, "common_input_stream": True,
        "branch_census": dict(runtime["call_counts"]),
        "greedy_prediction_match": token_match, "logits": logit,
        "attention": {
            "calls": len(rows),
            "max_abs_delta": max(r["attention_output_max_abs_delta"] for r in rows),
            "mean_of_call_means": statistics.mean(
                r["attention_output_mean_abs_delta"] for r in rows),
            "per_layer_step": rows,
        },
        "rings": {"blend": blend["rings"], "int4": target["rings"]},
        "ring_drop_pass": ring_drop,
        "registered_raw_logit_gate": {
            "tolerance": MAX_ABS_LOGIT_DELTA,
            "pass": token_match and logit["max_abs"] <= MAX_ABS_LOGIT_DELTA,
        },
        "selection_overlap": {
            "status": "not_exposed_by_int4_abi",
            "reason": "FA entry accepts raw K and packs transient codes internally",
        },
    })
    print(f"MEASUREMENT int4: PASS calls={len(rows)}", flush=True)
    print(f"TOKENS: {'MATCH' if token_match else 'DIFF'} | "
          f"max|delta logit|={logit['max_abs']:.6f} | "
          f"mean|delta logit|={logit['mean_abs']:.6f}", flush=True)
    print(f"INT4 RING-DROP: {'PASS' if ring_drop else 'FAIL'}", flush=True)
    print("RAW-LOGIT GATE: " + (
        "PASS" if receipt["registered_raw_logit_gate"]["pass"] else "FAIL"),
        flush=True)


def run_summary(output: Path, receipt: dict[str, Any]) -> None:
    inputs = {}
    for leg in ("bits4", "bits6", "bits8", "force_all"):
        path = output.parent / f"{leg}.json"
        if not path.exists():
            raise FileNotFoundError(f"required receipt absent: {path}")
        inputs[leg] = json.loads(path.read_text())
        if inputs[leg].get("status") != "complete":
            raise RuntimeError(f"{leg} receipt is not complete")
    ladder = []
    for leg in ("bits4", "bits6", "bits8"):
        r = inputs[leg]
        ladder.append({
            "leg": leg, "bulk_bits": r["bulk_bits"],
            "max_abs_logit_delta": r["logits"]["max_abs"],
            "mean_abs_logit_delta": r["logits"]["mean_abs"],
            "jaccard": r["selection"]["jaccard_global"],
            "flip_fraction": r["selection"]["flip_fraction"],
            "refined_fraction_blend": r["selection"]["refined_fraction_blend"],
            "refined_fraction_fused": r["selection"]["refined_fraction_fused"],
        })
    maxes = [r["max_abs_logit_delta"] for r in ladder]
    means = [r["mean_abs_logit_delta"] for r in ladder]
    jaccards = [r["jaccard"] for r in ladder]
    flips = [r["flip_fraction"] for r in ladder]
    strict_monotone = (
        maxes[0] > maxes[1] > maxes[2]
        and means[0] > means[1] > means[2]
        and jaccards[0] <= jaccards[1] <= jaccards[2]
        and flips[0] >= flips[1] >= flips[2]
    )
    strong = strict_monotone and maxes[2] <= 0.5 * maxes[0]
    flat = (max(maxes) - min(maxes)) <= 0.1 * max(maxes[0], 1e-30)
    hypothesis = "supported_strong_monotone" if strong else (
        "refuted_flat" if flat else "mixed_or_weak")
    bits4 = inputs["bits4"]["selection"]
    controls = bits4["semantic_controls"]
    if not controls["valid_count_equal_all_calls"]:
        divergent_term = "mask range or valid-count semantics"
    elif not controls["scale_equal_all_calls"]:
        divergent_term = "scale application"
    elif not controls["zthr_equal_all_calls"]:
        divergent_term = "threshold parameter"
    elif not controls["same_state_tensor_inputs_all_calls"]:
        divergent_term = "q/k/kq slice alignment or state identity"
    elif bits4["fused_threshold_replay_max_abs_delta"] > 1e-4:
        divergent_term = "fused statistics population or reduction replay"
    elif bits4["flip_count"] > 0:
        divergent_term = "bf16 bulk-score materialization versus fused fp32 score"
    elif bits4["identical_selection_attention_max_abs_delta"] not in (None, 0.0):
        divergent_term = "non-selected score rounding or attention reduction"
    else:
        divergent_term = "not localized by registered semantic controls"
    receipt.update({
        "status": "complete", "input_receipts": {
            k: str(output.parent / f"{k}.json") for k in inputs},
        "bits_ladder": ladder,
        "hypothesis_classification": hypothesis,
        "strict_monotone": strict_monotone,
        "strong_improvement_rule": "strict max/mean improvement, overlap nondecrease, flip nonincrease, bits8 max <= 0.5*bits4 max",
        "flat_rule": "max-logit range <= 10% of bits4 max",
        "semantic_bisect_named_term": divergent_term,
        "force_all_control": {
            "logits": inputs["force_all"]["logits"],
            "selection": inputs["force_all"]["selection"],
        },
    })
    print("SUMMARY: PASS", flush=True)
    print(f"H-DIV: {hypothesis}", flush=True)
    print(f"SEMANTIC TERM: {divergent_term}", flush=True)


def run_cpu_check(receipt: dict[str, Any]) -> None:
    rng = np.random.default_rng(7)
    q = rng.standard_normal((1, 2, 1, 512), dtype=np.float32)
    kq = rng.standard_normal((1, 1, 17, 512), dtype=np.float32)
    bulk = fused_wcoop_bulk(q, kq, 1.0)
    fthr = fused_wcoop_threshold(bulk, 1.0)
    bthr = blend_kernel_threshold(bulk, 1.0)
    all_thr_f = fused_wcoop_threshold(bulk, float("-inf"))
    all_thr_b = blend_kernel_threshold(bulk, float("-inf"))
    assert bulk.shape == (2, 17)
    assert fthr.shape == bthr.shape == (2,)
    assert np.isneginf(all_thr_f).all() and np.isneginf(all_thr_b).all()
    assert (np.abs(bulk) >= all_thr_f[:, None]).all()
    default_root = CANONICAL_TC_ROOT.resolve()
    receipt.update({
        "status": "complete", "checks": {
            "fused_wcoop_replay_shape": "PASS",
            "blend_reduction_replay_shape": "PASS",
            "force_all_selects_all": "PASS",
            "root_exists": TC_ROOT.exists(),
        },
        "root_resolution": {
            "environment_value": os.environ.get("TENSOR_CUDA_ROOT"),
            "resolved": str(TC_ROOT), "default": str(default_root),
            "using_default": TC_ROOT == default_root,
        },
    })
    print("CPU REDUCTION REPLAYS: PASS", flush=True)
    print("CPU FORCE-ALL CONTROL: PASS", flush=True)
    print(f"TENSOR_CUDA_ROOT: {TC_ROOT} "
          f"({'DEFAULT' if TC_ROOT == default_root else 'OVERRIDE'})", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--leg", required=True,
                   choices=tuple(LEG_CONFIG) + ("int4", "summary", "cpu-check"))
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    receipt: dict[str, Any] = {
        "schema": "apamq_fbd1_v1", "leg": args.leg,
        "started_utc": utc_now(), "status": "running",
        "argv": sys.argv, "cwd": os.getcwd(),
        "tensor_cuda_root": str(TC_ROOT),
        "prompt_len": PROMPT_LEN, "decode_steps": DECODE_STEPS,
        "global_layers": list(GLOBAL_LAYERS), "seed": SEED,
        "evidence_class": "INSTRUMENTED PORT MEASUREMENT",
    }
    rc = 0
    try:
        if args.leg in LEG_CONFIG:
            run_bits_leg(args.leg, receipt)
        elif args.leg == "int4":
            run_int4_leg(receipt)
        elif args.leg == "summary":
            run_summary(args.output, receipt)
        else:
            run_cpu_check(receipt)
    except Exception as exc:
        rc = 1
        receipt.update({
            "status": "failed",
            "failure": {"type": type(exc).__name__, "message": str(exc),
                        "traceback": traceback.format_exc()},
        })
        print(f"{args.leg} MEASUREMENT: FAIL {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
    finally:
        receipt["finished_utc"] = utc_now()
        atomic_json(args.output, receipt)
        print(f"RECEIPT: {args.output} status={receipt['status']}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
