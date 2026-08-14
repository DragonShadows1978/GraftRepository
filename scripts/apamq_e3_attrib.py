#!/usr/bin/env python3
"""APAMQ-E3 Gemma-4 peak-memory attribution probe.

Evidence class: INSTRUMENTED PORT MEASUREMENT.  This script records memory
and timing only; it does not make quality or perplexity claims.

One invocation runs exactly one attention mode and one context length.  The
caller is responsible for taking /tmp/forge-gpu.lock around the process.
The script never modifies core/ or the TensorCUDA engine.

Use --mode apa with --decode-fused for G-B1, or --int4 for G-B2. The INT4
toggle implies fused decode and remains inert when the engine ABI is absent.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Callable

import numpy as np


REPO = Path(__file__).resolve().parents[1]
TC_ROOT = Path(os.environ.get(
    "TENSOR_CUDA_ROOT", "/mnt/ForgeRealm/Project-Tensor/tensor_cuda"
)).expanduser().resolve()
CORE_GEMMA = REPO / "core/gemma4_tc.py"
CORE_MISTRAL = REPO / "core/mistral7b_tc.py"
KERNELS = TC_ROOT / "src/kernels.cu"
MIB = 1024 * 1024
DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "uint8": 1,
    "int8": 1,
    "int32": 4,
    "int64": 8,
    "bool": 1,
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_nbytes(t: Any) -> int:
    if t is None:
        return 0
    shape = [int(x) for x in t.shape]
    dtype = str(t.dtype)
    if dtype not in DTYPE_BYTES:
        raise AssertionError(f"unknown tensor dtype for byte accounting: {dtype}")
    return math.prod(shape) * DTYPE_BYTES[dtype]


def tensor_desc(t: Any) -> dict[str, Any] | None:
    if t is None:
        return None
    return {
        "shape": [int(x) for x in t.shape],
        "dtype": str(t.dtype),
        "bytes": tensor_nbytes(t),
    }


def prefill_chunks(target: int, chunk: int = 512, *,
                   int4: bool = False) -> list[dict[str, Any]]:
    """Mirror Gemma4_TC.__call__ adaptive chunking at core lines 789-829."""
    out: list[dict[str, Any]] = []
    off = 0
    s0 = 0
    index = 0
    while s0 < target:
        s_ctx = off + chunk
        step = min(chunk, max(64, (32 * MIB) // (32 * s_ctx)))
        step = max(64, (step // 64) * 64)
        length = min(step, target - s0)
        s_all_global = off + length
        s_all_sliding = min(off, 1023) + length
        out.append({
            "index": index,
            "offset": off,
            "length": length,
            "global_s_all": s_all_global,
            "sliding_s_all": s_all_sliding,
            "apa_global_branch": (
                "standard_below_apa_min_context"
                if s_all_global <= 2048 else
                "apa_cublas_blend"
                if s_all_global <= 4096 else
                "apa_fused_int4" if int4 else "apa_fused_selective"
            ),
            "standard_global_branch": "standard_mqa_deexpanded",
            "sliding_branch": (
                "sdpa_causal" if s_all_sliding <= 1024 else "sdpa_band_mask"
            ),
        })
        off += length
        s0 += length
        index += 1
    return out


def branch_audit(*, decode_fused: bool = False,
                 int4: bool = False) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for target in (4096, 8192, 16384, 32768):
        chunks = prefill_chunks(target, int4=int4)
        counts: dict[str, int] = {}
        for c in chunks:
            for key in ("apa_global_branch", "standard_global_branch", "sliding_branch"):
                tag = f"{key}:{c[key]}"
                counts[tag] = counts.get(tag, 0) + 1
        rows[str(target)] = {
            "prefill_chunk_count": len(chunks),
            "prefill_chunks": chunks,
            "branch_counts": counts,
            "decode_standard": "L==1 standard ring matmul/causal_softmax",
            "decode_apa": (
                "L==1 APA INT4 fused when engine-capable; no kqb ring"
                if int4 else
                "L==1 APA fused selective above fast_max_seq"
                if decode_fused else
                "L==1 APA cuBLAS blend attribution baseline"
            ),
        }
    return {
        "evidence_class": "static live-source path audit",
        "source_refs": {
            "global_layers": "core/gemma4_tc.py:61-63",
            "apa_min_context_definition": "core/gemma4_tc.py:476-488",
            "decode_gate_and_unconditional_blend": "core/gemma4_tc.py:545-579",
            "prefill_apa_gate": "core/gemma4_tc.py:616-662",
            "standard_global_prefill": "core/gemma4_tc.py:663-683",
            "sliding_prefill": "core/gemma4_tc.py:684-701",
            "adaptive_prefill_chunker": "core/gemma4_tc.py:783-831",
            "mqa_blend": "core/mistral7b_tc.py:249-287",
            "kernel_cap": "/mnt/ForgeRealm/Project-Tensor/tensor_cuda/src/kernels.cu:1053",
            "kernel_dispatch": "/mnt/ForgeRealm/Project-Tensor/tensor_cuda/src/kernels.cu:1847-1850,1868-1869",
        },
        "constants": {
            "apa_min_context": 2048,
            "fast_max_seq": 4096,
            "prefill_chunk": 512,
            "global_head_dim": 512,
            "TC_APA_MAXD": 512,
            "fused_decode_legal": True,
            "fused_decode_wired": True,
            "decode_fused_requested": decode_fused,
            "int4_requested": int4,
        },
        "targets": rows,
    }


class NvmlPoller:
    """One-second NVML poller via nvidia-smi, running in a side thread."""

    def __init__(self, interval_s: float, phase_getter: Callable[[], str]):
        self.interval_s = interval_s
        self.phase_getter = phase_getter
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="apamq-nvml", daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        try:
            p = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            if p.returncode != 0:
                msg = (p.stderr or p.stdout).strip()
                raise RuntimeError(f"nvidia-smi rc={p.returncode}: {msg}")
            first = p.stdout.strip().splitlines()[0]
            used, total, util = [int(x.strip()) for x in first.split(",")]
            self.samples.append({
                "monotonic_s": time.monotonic(),
                "utc": utc_now(),
                "phase": self.phase_getter(),
                "memory_used_mib": used,
                "memory_total_mib": total,
                "utilization_gpu_percent": util,
            })
        except Exception as exc:  # a failed NVML sample is part of the receipt
            msg = f"{type(exc).__name__}: {exc}"
            if not self.errors or self.errors[-1] != msg:
                self.errors.append(msg)

    def sample_now(self) -> None:
        self._sample()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
        self._sample()

    def summary(self) -> dict[str, Any]:
        phases: dict[str, Any] = {}
        for s in self.samples:
            ph = s["phase"]
            d = phases.setdefault(ph, {"count": 0, "peak_memory_used_mib": None,
                                       "min_memory_used_mib": None})
            d["count"] += 1
            v = s["memory_used_mib"]
            d["peak_memory_used_mib"] = v if d["peak_memory_used_mib"] is None else max(d["peak_memory_used_mib"], v)
            d["min_memory_used_mib"] = v if d["min_memory_used_mib"] is None else min(d["min_memory_used_mib"], v)
        return {"interval_s": self.interval_s, "phases": phases,
                "sample_count": len(self.samples), "errors": self.errors,
                "samples": self.samples}


class CudaPool:
    """Read/reset CUDA default-mempool counters through libcudart."""

    RESERVED_CURRENT = 0x5
    RESERVED_HIGH = 0x6
    USED_CURRENT = 0x7
    USED_HIGH = 0x8

    def __init__(self):
        self.lib = ctypes.CDLL("libcudart.so")
        self.lib.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.lib.cudaGetDevice.restype = ctypes.c_int
        self.lib.cudaDeviceGetDefaultMemPool.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        self.lib.cudaDeviceGetDefaultMemPool.restype = ctypes.c_int
        self.lib.cudaMemPoolGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.lib.cudaMemPoolGetAttribute.restype = ctypes.c_int
        self.lib.cudaMemPoolSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.lib.cudaMemPoolSetAttribute.restype = ctypes.c_int
        dev = ctypes.c_int()
        self._check(self.lib.cudaGetDevice(ctypes.byref(dev)), "cudaGetDevice")
        self.device = dev.value
        pool = ctypes.c_void_p()
        self._check(self.lib.cudaDeviceGetDefaultMemPool(ctypes.byref(pool), self.device),
                    "cudaDeviceGetDefaultMemPool")
        self.pool = pool

    @staticmethod
    def _check(rc: int, name: str) -> None:
        if rc != 0:
            raise RuntimeError(f"{name} failed with CUDA error {rc}")

    def _get(self, attr: int) -> int:
        out = ctypes.c_uint64()
        self._check(self.lib.cudaMemPoolGetAttribute(self.pool, attr, ctypes.byref(out)),
                    f"cudaMemPoolGetAttribute({attr})")
        return int(out.value)

    def stats(self) -> dict[str, int]:
        return {
            "reserved_current_bytes": self._get(self.RESERVED_CURRENT),
            "reserved_high_bytes": self._get(self.RESERVED_HIGH),
            "used_current_bytes": self._get(self.USED_CURRENT),
            "used_high_bytes": self._get(self.USED_HIGH),
        }

    def reset_high(self) -> None:
        zero = ctypes.c_uint64(0)
        for attr in (self.RESERVED_HIGH, self.USED_HIGH):
            self._check(self.lib.cudaMemPoolSetAttribute(self.pool, attr, ctypes.byref(zero)),
                        f"cudaMemPoolSetAttribute({attr})")


class TraceManager:
    """Nested exact pool envelopes while preserving a true outer phase peak."""

    def __init__(self, tc: Any, pool: CudaPool, runtime: dict[str, Any]):
        self.tc = tc
        self.pool = pool
        self.runtime = runtime
        self.stack: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

    def checkpoint(self) -> dict[str, int]:
        self.tc.synchronize()
        s = self.pool.stats()
        for c in self.stack:
            c["max_used_abs"] = max(c["max_used_abs"], s["used_high_bytes"], s["used_current_bytes"])
            c["max_reserved_abs"] = max(c["max_reserved_abs"], s["reserved_high_bytes"], s["reserved_current_bytes"])
        return s

    def enter(self, label: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        self.checkpoint()
        self.pool.reset_high()
        s = self.pool.stats()
        ctx = {
            "label": label,
            "phase": self.runtime["phase"],
            "prefill_chunk_index": self.runtime.get("prefill_chunk_index"),
            "decode_step": self.runtime.get("decode_step"),
            "metadata": metadata or {},
            "start_monotonic_s": time.monotonic(),
            "baseline_used_bytes": s["used_current_bytes"],
            "baseline_reserved_bytes": s["reserved_current_bytes"],
            "max_used_abs": s["used_current_bytes"],
            "max_reserved_abs": s["reserved_current_bytes"],
        }
        self.stack.append(ctx)
        return ctx

    def exit(self, ctx: dict[str, Any], status: str = "ok", error: str | None = None) -> dict[str, Any]:
        end = self.checkpoint()
        assert self.stack and self.stack[-1] is ctx, "trace stack corruption"
        self.stack.pop()
        event = {
            **{k: v for k, v in ctx.items() if k not in ("start_monotonic_s",)},
            "status": status,
            "error": error,
            "wall_s": time.monotonic() - ctx["start_monotonic_s"],
            "end_used_bytes": end["used_current_bytes"],
            "end_reserved_bytes": end["reserved_current_bytes"],
            "used_peak_delta_bytes": max(0, ctx["max_used_abs"] - ctx["baseline_used_bytes"]),
            "reserved_peak_delta_bytes": max(0, ctx["max_reserved_abs"] - ctx["baseline_reserved_bytes"]),
        }
        self.events.append(event)
        self.pool.reset_high()
        return event

    def call(self, label: str, fn: Callable[[], Any], metadata: dict[str, Any] | None = None) -> Any:
        ctx = self.enter(label, metadata)
        try:
            out = fn()
        except Exception as exc:
            self.exit(ctx, "error", f"{type(exc).__name__}: {exc}")
            raise
        self.exit(ctx)
        return out


def aggregate_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for e in events:
        key = f"{e['phase']}|{e['label']}"
        groups.setdefault(key, []).append(e)
    out: dict[str, Any] = {}
    for key, es in groups.items():
        used = [int(e["used_peak_delta_bytes"]) for e in es]
        reserved = [int(e["reserved_peak_delta_bytes"]) for e in es]
        out[key] = {
            "count": len(es),
            "used_peak_delta_max_bytes": max(used),
            "used_peak_delta_sum_bytes": sum(used),
            "used_peak_delta_mean_bytes": int(statistics.mean(used)),
            "reserved_peak_delta_max_bytes": max(reserved),
            "max_absolute_pool_used_bytes": max(int(e["max_used_abs"]) for e in es),
            "status_counts": {s: sum(1 for e in es if e["status"] == s)
                              for s in sorted(set(e["status"] for e in es))},
        }
    return out


def cache_inventory(caches: list[Any], KVRing: type) -> dict[str, Any]:
    totals = {"all_bytes": 0, "k_bytes": 0, "v_bytes": 0, "bias_bytes": 0,
              "kqb_bytes": 0, "apa_extra_bytes": 0}
    layers: list[dict[str, Any]] = []
    for i, c in enumerate(caches):
        row: dict[str, Any] = {"layer": i, "class": "global" if i % 6 == 5 else "sliding"}
        if isinstance(c, KVRing):
            row.update({"kind": "KVRing", "count": int(c.count), "cap": int(c.cap),
                        "ring": bool(c.ring), "full": bool(c.full)})
            fields: dict[str, Any] = {}
            for name in ("kb", "vb", "vsb", "ksb", "bias", "kqb"):
                value = getattr(c, name)
                fields[name] = tensor_desc(value)
                n = tensor_nbytes(value)
                totals["all_bytes"] += n
                if name in ("kb", "ksb"):
                    totals["k_bytes"] += n
                elif name in ("vb", "vsb"):
                    totals["v_bytes"] += n
                elif name == "bias":
                    totals["bias_bytes"] += n
                elif name == "kqb":
                    totals["kqb_bytes"] += n
                    totals["apa_extra_bytes"] += n
            row["fields"] = fields
        else:
            k, v = c
            row.update({"kind": "tuple", "k": tensor_desc(k), "v": tensor_desc(v)})
            kb, vb = tensor_nbytes(k), tensor_nbytes(v)
            totals["all_bytes"] += kb + vb
            totals["k_bytes"] += kb
            totals["v_bytes"] += vb
        layers.append(row)
    return {"totals": totals, "layers": layers}


def mask_inventory(gemma: Any, F: Any) -> dict[str, Any]:
    band = [tensor_desc(v) for v in gemma._band_cache.values()]
    causal = [tensor_desc(v) for v in F._causal_cache.values()]
    return {
        "band_cache_entries": len(band),
        "band_cache_bytes": sum(int(x["bytes"]) for x in band if x),
        "causal_cache_entries": len(causal),
        "causal_cache_bytes": sum(int(x["bytes"]) for x in causal if x),
        "total_bytes": sum(int(x["bytes"]) for x in band + causal if x),
        "band_entries": band,
        "causal_entries": causal,
    }


def table_inventory(quant: Any) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for key, tensors in quant._TABLE_TENSORS.items():
        desc = [tensor_desc(t) for t in tensors]
        entries.append({"key": repr(key), "tensors": desc,
                        "bytes": sum(int(x["bytes"]) for x in desc if x)})
    return {"entries": entries, "total_bytes": sum(e["bytes"] for e in entries)}


def install_instrumentation(tc: Any, gemma: Any, mistral: Any, quant: Any,
                            F: Any, tracer: TraceManager, runtime: dict[str, Any]) -> Callable[[], None]:
    Attn = gemma.Gemma4AttentionTC
    Mlp = gemma.GegluTC
    orig_attn = Attn.__call__
    orig_mlp = Mlp.__call__
    orig_quant = quant._quantize_keys
    orig_blend = mistral._cublas_blend_attention
    orig_fused = tc.apa_selective_attention
    orig_int4 = getattr(tc, "apa_selective_attention_int4", None)
    orig_blend_softmax = tc.apa_blend_softmax
    orig_causal_softmax = tc.causal_softmax

    def attn_call(self: Any, x: Any, cos: Any, sin: Any,
                  position_offset: int = 0, kv_cache: Any = None) -> Any:
        cls = "global" if self.is_global else "sliding"
        prior = runtime.get("attention_class")
        runtime["attention_class"] = cls
        meta = {"class": cls, "input_L": int(x.shape[1]),
                "position_offset": int(position_offset),
                "layer_index": getattr(self, "_apamq_layer_index", None)}
        try:
            return tracer.call(f"attention.{cls}",
                               lambda: orig_attn(self, x, cos, sin, position_offset, kv_cache), meta)
        finally:
            runtime["attention_class"] = prior

    def mlp_call(self: Any, x: Any) -> Any:
        return tracer.call("mlp", lambda: orig_mlp(self, x),
                           {"input_L": int(x.shape[1])})

    def quant_call(k: Any, R: Any, CB: Any, BND: Any) -> Any:
        return tracer.call("component.quantize_keys",
                           lambda: orig_quant(k, R, CB, BND),
                           {"input": tensor_desc(k), "attention_class": runtime.get("attention_class")})

    def blend_call(q: Any, kH: Any, kq: Any, v: Any, group: int,
                   scale: float, z: float, causal: bool, blk: int) -> Any:
        return tracer.call("component.apa_blend",
                           lambda: orig_blend(q, kH, kq, v, group, scale, z, causal, blk),
                           {"q": tensor_desc(q), "k": tensor_desc(kH),
                            "kq": tensor_desc(kq), "v": tensor_desc(v),
                            "group": int(group), "causal": bool(causal), "blk": int(blk)})

    def fused_call(q: Any, k: Any, kq: Any, v: Any, scale: float,
                   z: float, causal: bool) -> Any:
        return tracer.call("component.apa_fused",
                           lambda: orig_fused(q, k, kq, v, scale, z, causal),
                           {"q": tensor_desc(q), "k": tensor_desc(k),
                            "kq": tensor_desc(kq), "v": tensor_desc(v),
                            "causal": bool(causal)})

    def int4_call(q: Any, k: Any, v: Any, scale: float,
                  z: float, causal: bool) -> Any:
        return tracer.call("component.apa_int4",
                           lambda: orig_int4(q, k, v, scale, z, causal),
                           {"q": tensor_desc(q), "k": tensor_desc(k),
                            "v": tensor_desc(v), "causal": bool(causal)})

    def blend_softmax_call(bulk: Any, rank: Any, z: float, Lq: int = 0,
                           row0: int = 0, window: int = 0) -> Any:
        return tracer.call("component.apa_refine_softmax",
                           lambda: orig_blend_softmax(bulk, rank, z, Lq, row0, window),
                           {"bulk": tensor_desc(bulk), "rank": tensor_desc(rank),
                            "Lq": int(Lq), "row0": int(row0), "window": int(window)})

    def causal_softmax_call(x: Any) -> Any:
        return tracer.call("component.causal_softmax",
                           lambda: orig_causal_softmax(x),
                           {"input": tensor_desc(x), "attention_class": runtime.get("attention_class")})

    Attn.__call__ = attn_call
    Mlp.__call__ = mlp_call
    quant._quantize_keys = quant_call
    mistral._cublas_blend_attention = blend_call
    tc.apa_selective_attention = fused_call
    if orig_int4 is not None:
        tc.apa_selective_attention_int4 = int4_call
    tc.apa_blend_softmax = blend_softmax_call
    tc.causal_softmax = causal_softmax_call

    def restore() -> None:
        Attn.__call__ = orig_attn
        Mlp.__call__ = orig_mlp
        quant._quantize_keys = orig_quant
        mistral._cublas_blend_attention = orig_blend
        tc.apa_selective_attention = orig_fused
        if orig_int4 is not None:
            tc.apa_selective_attention_int4 = orig_int4
        tc.apa_blend_softmax = orig_blend_softmax
        tc.causal_softmax = orig_causal_softmax

    return restore


def assert_apa_kqb(caches: list[Any], KVRing: type) -> dict[str, Any]:
    rows = []
    for i in range(5, 48, 6):
        c = caches[i]
        assert isinstance(c, KVRing), f"global layer {i}: expected KVRing, got {type(c)}"
        assert c.kqb is not None, f"global layer {i}: kqb was not allocated after L=1 APA decode"
        assert str(c.kqb.dtype) == "bfloat16", f"global layer {i}: kqb dtype {c.kqb.dtype}, expected bfloat16"
        assert int(c.kq_count) == int(c.count), (i, c.kq_count, c.count)
        rows.append({"layer": i, "dtype": str(c.kqb.dtype), "shape": [int(x) for x in c.kqb.shape],
                     "bytes": tensor_nbytes(c.kqb), "count": int(c.count),
                     "kq_count": int(c.kq_count), "assertion": "PASS"})
    return {"assertion": "PASS", "layers": rows,
            "total_bytes": sum(r["bytes"] for r in rows)}


def assert_apa_no_kqb(caches: list[Any], KVRing: type) -> dict[str, Any]:
    rows = []
    for i in range(5, 48, 6):
        c = caches[i]
        assert isinstance(c, KVRing), (
            f"global layer {i}: expected KVRing, got {type(c)}")
        assert c.kqb is None, (
            f"global layer {i}: INT4 mode allocated forbidden kqb")
        assert int(c.kq_count) == 0, (i, c.kq_count, c.count)
        rows.append({"layer": i, "kqb": None, "count": int(c.count),
                     "kq_count": int(c.kq_count), "assertion": "PASS"})
    return {"assertion": "PASS", "layers": rows, "total_bytes": 0}


def run_gpu(args: argparse.Namespace, receipt: dict[str, Any]) -> None:
    # The harness makes each attribution leg explicit even though the serving
    # port defaults fused decode on. --int4 implies the fused-decode leg.
    os.environ["GEMMA4_APA_DECODE_FUSED"] = (
        "1" if args.decode_fused or args.int4 else "0")
    os.environ["GEMMA4_APA_INT4"] = "1" if args.int4 else "0"
    sys.path.insert(0, str(TC_ROOT))
    sys.path.insert(0, str(REPO))
    import tensor_cuda as tc
    from tensor_cuda import functional as F
    import tensor_cuda.quant as quant
    import core.gemma4_tc as gemma
    import core.mistral7b_tc as mistral

    int4_capable = hasattr(tc, "apa_selective_attention_int4")
    int4_active = bool(args.int4 and int4_capable)
    receipt["feature_resolution"] = {
        "decode_fused_env": os.environ["GEMMA4_APA_DECODE_FUSED"],
        "int4_env": os.environ["GEMMA4_APA_INT4"],
        "engine_has_apa_selective_attention_int4": int4_capable,
        "int4_active": int4_active,
        "int4_inert_reason": (None if not args.int4 or int4_capable else
                               "engine entry point absent"),
    }

    runtime: dict[str, Any] = {"phase": "startup", "prefill_chunk_index": None,
                               "decode_step": None, "attention_class": None}
    poller = NvmlPoller(args.nvml_interval, lambda: runtime["phase"])
    receipt["nvml"] = {"started": True}
    poller.start()
    poller.sample_now()
    restore: Callable[[], None] | None = None
    try:
        runtime["phase"] = "load"
        # Match the current Gemma serving/test configuration: pooled engine
        # allocations are enabled before exact-QAT weight load.
        tc.set_alloc_pooling(True)
        load_t0 = time.perf_counter()
        model, info = gemma.Gemma4_TC.from_pretrained(qat=True)
        tc.synchronize()
        load_s = time.perf_counter() - load_t0
        for i, layer in enumerate(model.layers):
            layer.mixer._apamq_layer_index = i
            if layer.mixer.is_global:
                layer.mixer.attention_mode = "standard" if args.mode == "standard" else "apa_selective"
                layer.mixer.refine_percentile = args.refine
                layer.mixer.apa_min_context = 2048
                layer.mixer.fast_max_seq = 4096
        pool = CudaPool()
        load_pool = pool.stats()
        receipt["load"] = {"wall_s": load_s, "info": {k: str(v) for k, v in info.items()},
                           "pool": load_pool}
        runtime["phase"] = "post_load"
        poller.sample_now()

        tracer = TraceManager(tc, pool, runtime)
        restore = install_instrumentation(tc, gemma, mistral, quant, F, tracer, runtime)
        chunks = prefill_chunks(args.seq_len, int4=int4_active)
        receipt["actual_prefill_chunks"] = chunks
        rng = np.random.default_rng(args.seed)
        ids = rng.integers(1000, 200000, size=(1, args.seq_len), dtype=np.int64)

        # Monkeypatch the top-level adaptive loop only to expose its current
        # chunk index to nested event records. Its arithmetic and forward calls
        # remain the port's own implementation.
        orig_forward = model._forward
        prefill_counter = {"i": 0}

        def marked_forward(*a: Any, **kw: Any) -> Any:
            if runtime["phase"] == "prefill":
                runtime["prefill_chunk_index"] = prefill_counter["i"]
                prefill_counter["i"] += 1
            return orig_forward(*a, **kw)

        model._forward = marked_forward
        runtime["phase"] = "prefill"
        poller.sample_now()
        t0 = time.perf_counter()
        phase_ctx = tracer.enter("phase.prefill", {"target_S": args.seq_len})
        with tc.no_grad():
            logits, caches = model(ids, last_token_only=True)
        prefill_event = tracer.exit(phase_ctx)
        tc.synchronize()
        prefill_s = time.perf_counter() - t0
        receipt["prefill"] = {
            "status": "ok", "wall_s": prefill_s,
            "instrumented_ms_per_input_token": 1000.0 * prefill_s / args.seq_len,
            "pool_phase_event": prefill_event,
            "cache_inventory_before_decode": cache_inventory(caches, gemma.KVRing),
            "mask_inventory": mask_inventory(gemma, F),
        }
        runtime["phase"] = "post_prefill"
        poller.sample_now()

        runtime["phase"] = "decode"
        runtime["prefill_chunk_index"] = None
        poller.sample_now()
        fixed_token = np.array([[5279]], dtype=np.int64)
        times_ms: list[float] = []
        phase_ctx = tracer.enter("phase.decode", {"steps": args.decode_steps})
        kqb_assertion = None
        with tc.no_grad():
            for step in range(args.decode_steps):
                runtime["decode_step"] = step
                t0 = time.perf_counter()
                logits, caches = model(fixed_token, caches=caches,
                                       position_offset=args.seq_len + step,
                                       last_token_only=True)
                tc.synchronize()
                times_ms.append(1000.0 * (time.perf_counter() - t0))
                if step == 0 and args.mode == "apa":
                    kqb_assertion = (
                        assert_apa_no_kqb(caches, gemma.KVRing)
                        if int4_active else
                        assert_apa_kqb(caches, gemma.KVRing))
        decode_event = tracer.exit(phase_ctx)
        receipt["decode"] = {
            "status": "ok", "steps": args.decode_steps,
            "times_ms": times_ms,
            "conversion_step_ms": times_ms[0] if times_ms else None,
            "steady_ms_per_token_median": statistics.median(times_ms[1:]) if len(times_ms) > 1 else None,
            "steady_ms_per_token_min": min(times_ms[1:]) if len(times_ms) > 1 else None,
            "steady_ms_per_token_max": max(times_ms[1:]) if len(times_ms) > 1 else None,
            "pool_phase_event": decode_event,
            "kqb_assertion": kqb_assertion,
            "cache_inventory_after_decode": cache_inventory(caches, gemma.KVRing),
            "mask_inventory": mask_inventory(gemma, F),
            "apa_table_inventory": table_inventory(quant),
        }
        runtime["phase"] = "post_decode"
        poller.sample_now()
        receipt["pool_events"] = tracer.events
        receipt["pool_event_aggregates"] = aggregate_events(tracer.events)
        receipt["status"] = "complete"
    finally:
        if restore is not None:
            restore()
        runtime["phase"] = "shutdown"
        poller.stop()
        receipt["nvml"] = poller.summary()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("standard", "apa"), default="standard")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--decode-steps", type=int, default=64)
    p.add_argument("--decode-fused", action="store_true",
                   help="measure the fused APA decode leg (otherwise blend)")
    p.add_argument("--int4", action="store_true",
                   help="request engine-packed INT4 APA; implies fused decode")
    p.add_argument("--refine", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=20260813)
    p.add_argument("--nvml-interval", type=float, default=1.0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audit-only", action="store_true")
    args = p.parse_args()
    if (args.decode_fused or args.int4) and args.mode != "apa":
        p.error("--decode-fused/--int4 require --mode apa")
    return args


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    protected_before = {str(p): sha256(p) for p in (CORE_GEMMA, CORE_MISTRAL, KERNELS)}
    receipt: dict[str, Any] = {
        "schema": "apamq_e3_attrib_v1",
        "evidence_class": "INSTRUMENTED PORT MEASUREMENT - memory and speed attribution only",
        "quality_claims": "NONE",
        "started_utc": utc_now(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "mode": args.mode,
        "seq_len": args.seq_len,
        "decode_steps": args.decode_steps,
        "refine_percentile": args.refine,
        "decode_fused_requested": args.decode_fused,
        "int4_requested": args.int4,
        "one_mode_per_process": True,
        "branch_audit": branch_audit(
            decode_fused=args.decode_fused or args.int4, int4=args.int4),
        "protected_sha256_before": protected_before,
        "status": "running",
    }
    rc = 0
    try:
        if args.audit_only:
            receipt["status"] = "audit_only_complete"
        else:
            run_gpu(args, receipt)
    except Exception as exc:
        rc = 1
        receipt["status"] = "failed"
        receipt["failure"] = {
            "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(f"APAMQ-E3 FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        protected_after = {str(p): sha256(p) for p in (CORE_GEMMA, CORE_MISTRAL, KERNELS)}
        receipt["protected_sha256_after"] = protected_after
        receipt["protected_files_unchanged"] = protected_before == protected_after
        receipt["finished_utc"] = utc_now()
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, args.output)
        print(json.dumps({"status": receipt["status"], "output": str(args.output),
                          "protected_files_unchanged": receipt["protected_files_unchanged"]}),
              flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
