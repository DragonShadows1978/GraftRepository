#!/usr/bin/env python3
"""Gemma-4 APA fused-decode, INT4 ring-drop, and GEMM flag gates.

GPU gate (lead-run, one process):

  python3 tests/gemma4_apa_decode_fused.py

The gate compares the legacy cuBLAS blend with fused decode for 64 greedy
steps after an 8192-token prompt, asserts that the decode-only fused site is
actually called, and registers max |delta logit| <= 0.5 with exact greedy-token
agreement. The absolute tolerance admits fused online-softmax/reduction
reordering while remaining well below the existing 3.0 near-tie cost gate.

With ``--int4``, the gate compares the blend against INT4-fused decode using
the blend token stream, and asserts that every global KVRing has kqb=None and
kq_count=0.  That leg is a capability SKIP on engines without the F-A entry.

CPU/capability check (safe without a GPU):

  python3 tests/gemma4_apa_decode_fused.py --cpu-check
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
TC_ROOT = Path(os.environ.get(
    "TENSOR_CUDA_ROOT", "/mnt/ForgeRealm/Project-Tensor/tensor_cuda"
)).expanduser().resolve()
PROMPT_LEN = 8192
DECODE_STEPS = 64
MAX_ABS_LOGIT_DELTA = 0.5

sys.path.insert(0, str(TC_ROOT))
sys.path.insert(0, str(REPO))

import tensor_cuda as tc  # noqa: E402
import core.gemma4_tc as gemma  # noqa: E402
import core.mistral7b_tc as mistral  # noqa: E402


def cpu_check() -> int:
    """Exercise the env/capability matrix without allocating a tensor."""
    old_decode = os.environ.get("GEMMA4_APA_DECODE_FUSED")
    old_int4 = os.environ.get("GEMMA4_APA_INT4")
    old_gemm = os.environ.get("GEMMA4_APA_GEMM")
    marker_added = False
    try:
        expected_root = Path(os.environ.get(
            "TENSOR_CUDA_ROOT", "/mnt/ForgeRealm/Project-Tensor/tensor_cuda"
        )).expanduser().resolve()
        assert TC_ROOT == expected_root
        assert Path(tc.__file__).resolve().is_relative_to(TC_ROOT)
        os.environ.pop("GEMMA4_APA_DECODE_FUSED", None)
        assert gemma._apa_decode_fused_enabled()
        os.environ["GEMMA4_APA_DECODE_FUSED"] = "0"
        assert not gemma._apa_decode_fused_enabled()

        os.environ["GEMMA4_APA_INT4"] = "0"
        assert not gemma._apa_int4_enabled()
        os.environ["GEMMA4_APA_INT4"] = "1"
        has_int4 = hasattr(tc, "apa_selective_attention_int4")
        assert gemma._apa_int4_enabled() is has_int4
        if not has_int4:
            tc.apa_selective_attention_int4 = object()
            marker_added = True
            assert gemma._apa_int4_enabled()

        os.environ.pop("GEMMA4_APA_GEMM", None)
        assert not gemma._apa_gemm_enabled()
        os.environ["GEMMA4_APA_GEMM"] = "0"
        assert not gemma._apa_gemm_enabled()
        os.environ["GEMMA4_APA_GEMM"] = "1"
        has_gemm = hasattr(tc, "apa_gemm_selective_attention")
        assert gemma._apa_gemm_enabled() is has_gemm
        print("CPU FLAG MATRIX: PASS", flush=True)
        print("CPU ENGINE ROOT RESOLUTION: PASS", flush=True)
        print(f"TENSOR_CUDA_ROOT: {TC_ROOT}", flush=True)
        print(f"INT4 ENGINE ENTRY: {'PRESENT' if has_int4 else 'ABSENT (GPU ring leg SKIP)'}",
              flush=True)
        print(f"GEMM ENGINE ENTRY: {'PRESENT' if has_gemm else 'ABSENT (GEMM leg SKIP)'}",
              flush=True)
        return 0
    finally:
        if marker_added:
            delattr(tc, "apa_selective_attention_int4")
        if old_decode is None:
            os.environ.pop("GEMMA4_APA_DECODE_FUSED", None)
        else:
            os.environ["GEMMA4_APA_DECODE_FUSED"] = old_decode
        if old_int4 is None:
            os.environ.pop("GEMMA4_APA_INT4", None)
        else:
            os.environ["GEMMA4_APA_INT4"] = old_int4
        if old_gemm is None:
            os.environ.pop("GEMMA4_APA_GEMM", None)
        else:
            os.environ["GEMMA4_APA_GEMM"] = old_gemm


def configure(model: Any) -> None:
    for layer in model.layers:
        if layer.mixer.is_global:
            layer.mixer.attention_mode = "apa_selective"
            layer.mixer.refine_percentile = 0.15
            layer.mixer.apa_min_context = 2048
            layer.mixer.fast_max_seq = 4096


def global_rings(caches: list[Any]) -> list[tuple[int, Any]]:
    return [(i, caches[i]) for i in range(5, 48, 6)]


def run_decode(model: Any, ids: np.ndarray, steps: int,
               *, decode_fused: bool, int4: bool,
               forced_tokens: list[int] | None = None) -> tuple[
                   list[int], list[int], list[np.ndarray], list[Any]]:
    os.environ["GEMMA4_APA_DECODE_FUSED"] = "1" if decode_fused else "0"
    # Keep prefill common between the compared paths.  The F-A capability is
    # a decode leg here, so enable it only after the shared 8192-token prefill.
    os.environ["GEMMA4_APA_INT4"] = "0"
    with tc.no_grad():
        logits, caches = model(ids, last_token_only=True)
        os.environ["GEMMA4_APA_INT4"] = "1" if int4 else "0"
        nxt = int(logits.float().numpy()[0, -1].argmax())
        tokens: list[int] = []
        rows: list[np.ndarray] = []
        predictions: list[int] = []
        offset = ids.shape[1]
        for step in range(steps):
            if forced_tokens is not None:
                nxt = int(forced_tokens[step])
            tokens.append(nxt)
            logits, caches = model(
                np.array([[nxt]], dtype=np.int64), caches=caches,
                position_offset=offset, last_token_only=True)
            offset += 1
            row = logits.float().numpy()[0, -1]
            rows.append(row)
            nxt = int(row.argmax())
            predictions.append(nxt)
    return tokens, predictions, rows, caches


def gpu_gate(*, int4: bool = False) -> int:
    if not hasattr(tc, "apa_selective_attention"):
        print("FUSED-DECODE GATE: FAIL (engine fused entry absent)", flush=True)
        return 1
    if int4 and not hasattr(tc, "apa_selective_attention_int4"):
        print(f"INT4-FUSED PARITY GATE: SKIP (engine entry absent; root={TC_ROOT})",
              flush=True)
        return 0

    tc.set_alloc_pooling(True)
    model, info = gemma.Gemma4_TC.from_pretrained()
    configure(model)
    print(f"loaded: {info}", flush=True)
    ids = np.random.default_rng(20260813).integers(
        1000, 200000, size=(1, PROMPT_LEN), dtype=np.int64)

    census = {"blend_decode": 0, "fused_decode": 0, "int4_decode": 0}
    orig_blend = mistral._cublas_blend_attention
    orig_fused = tc.apa_selective_attention
    orig_int4 = getattr(tc, "apa_selective_attention_int4", None)

    def blend_count(q: Any, k: Any, kq: Any, v: Any, group: int,
                    scale: float, zthr: float, causal: bool, blk: int) -> Any:
        if int(q.shape[2]) == 1:
            census["blend_decode"] += 1
        return orig_blend(q, k, kq, v, group, scale, zthr, causal, blk)

    def fused_count(q: Any, k: Any, kq: Any, v: Any, scale: float,
                    zthr: float, causal: bool) -> Any:
        if int(q.shape[2]) == 1:
            census["fused_decode"] += 1
        return orig_fused(q, k, kq, v, scale, zthr, causal)

    def int4_count(q: Any, k: Any, v: Any, scale: float,
                   zthr: float, causal: bool) -> Any:
        if int(q.shape[2]) == 1:
            census["int4_decode"] += 1
        return orig_int4(q, k, v, scale, zthr, causal)

    mistral._cublas_blend_attention = blend_count
    tc.apa_selective_attention = fused_count
    if orig_int4 is not None:
        tc.apa_selective_attention_int4 = int4_count
    try:
        blend_start = dict(census)
        toks_blend, preds_blend, logits_blend, caches_blend = run_decode(
            model, ids, DECODE_STEPS, decode_fused=False, int4=False)
        blend_calls = census["blend_decode"] - blend_start["blend_decode"]
        assert blend_calls == 8 * DECODE_STEPS, blend_calls
        assert all(c.kqb is not None for _, c in global_rings(caches_blend))
        del caches_blend
        tc.empty_cache()

        target_start = dict(census)
        _, preds_target, logits_target, caches_target = run_decode(
            model, ids, DECODE_STEPS, decode_fused=True, int4=int4,
            forced_tokens=toks_blend)
        target_key = "int4_decode" if int4 else "fused_decode"
        target_calls = census[target_key] - target_start[target_key]
        assert target_calls == 8 * DECODE_STEPS, target_calls
        if int4:
            assert all(c.kqb is None and c.kq_count == 0
                       for _, c in global_rings(caches_target))
        else:
            assert all(c.kqb is not None for _, c in global_rings(caches_target))

        token_match = preds_blend == preds_target
        max_delta = max(float(np.abs(a - b).max())
                        for a, b in zip(logits_blend, logits_target))
        mean_delta = float(np.mean([
            np.abs(a - b).mean() for a, b in zip(logits_blend, logits_target)
        ]))
        parity_ok = token_match and max_delta <= MAX_ABS_LOGIT_DELTA
        print(f"decode branch census: blend={blend_calls}, {target_key}={target_calls}",
              flush=True)
        print(f"64-step greedy tokens: {'MATCH' if token_match else 'DIFF'} | "
              f"max|delta logit|={max_delta:.6f} "
              f"mean|delta logit|={mean_delta:.6f} "
              f"(tolerance {MAX_ABS_LOGIT_DELTA})", flush=True)
        label = "INT4-FUSED" if int4 else "FUSED-DECODE"
        print(f"{label} PARITY GATE: {'PASS' if parity_ok else 'FAIL'}",
              flush=True)
        if int4:
            print("INT4 RING-DROP GATE: PASS", flush=True)
        del caches_target
        tc.empty_cache()
        return 0 if parity_ok else 1
    finally:
        mistral._cublas_blend_attention = orig_blend
        tc.apa_selective_attention = orig_fused
        if orig_int4 is not None:
            tc.apa_selective_attention_int4 = orig_int4
        os.environ.pop("GEMMA4_APA_DECODE_FUSED", None)
        os.environ.pop("GEMMA4_APA_INT4", None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-check", action="store_true")
    parser.add_argument("--int4", action="store_true",
                        help="compare FA-engine INT4 fused decode vs blend")
    args = parser.parse_args()
    return cpu_check() if args.cpu_check else gpu_gate(int4=args.int4)


def test_cpu_flag_matrix() -> None:
    assert cpu_check() == 0


def test_int4_ring_drop_gpu_leg_capability() -> None:
    """Make the unavailable F-A engine dependency an explicit pytest skip."""
    if not hasattr(tc, "apa_selective_attention_int4"):
        import pytest
        pytest.skip("apa_selective_attention_int4 engine entry absent")


def test_gemm_gpu_leg_capability() -> None:
    """Make the canonical GEMM engine dependency an explicit pytest skip."""
    if not hasattr(tc, "apa_gemm_selective_attention"):
        import pytest
        pytest.skip("apa_gemm_selective_attention engine entry absent")


if __name__ == "__main__":
    raise SystemExit(main())
