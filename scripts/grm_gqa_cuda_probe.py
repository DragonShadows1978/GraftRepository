#!/usr/bin/env python3
"""Probe a CUDA/cuBLAS GQA raw q.k router over real capture banks."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_gqa_router_benchmark as bench  # noqa: E402


CUDA_SOURCE = r"""
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static int cuda_code(cudaError_t rc) {
  return rc == cudaSuccess ? 0 : 1000 + static_cast<int>(rc);
}

static int cublas_code(cublasStatus_t rc) {
  return rc == CUBLAS_STATUS_SUCCESS ? 0 : 2000 + static_cast<int>(rc);
}

__global__ void pack_k_col_kernel(
    const float* __restrict__ keys,
    float* __restrict__ k_col,
    int nodes,
    int kv_heads,
    int key_tokens,
    int head_dim,
    int kh) {
  const int64_t total = static_cast<int64_t>(nodes) * key_tokens * head_dim;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int d = static_cast<int>(idx % head_dim);
    const int64_t token_row = idx / head_dim;
    const int node = static_cast<int>(token_row / key_tokens);
    const int kt = static_cast<int>(token_row % key_tokens);
    const int64_t src =
        (((static_cast<int64_t>(node) * kv_heads + kh) * key_tokens + kt) *
         head_dim) + d;
    k_col[d + token_row * head_dim] = keys[src];
  }
}

__global__ void pack_q_col_kernel(
    const float* __restrict__ queries,
    float* __restrict__ q_col,
    int query_idx,
    int query_heads,
    int query_tokens,
    int head_dim,
    int kv_heads,
    int kh) {
  const int repeat = query_heads / kv_heads;
  const int rows = repeat * query_tokens;
  const int total = rows * head_dim;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += blockDim.x * gridDim.x) {
    const int d = idx % head_dim;
    const int row = idx / head_dim;
    const int local_h = row / query_tokens;
    const int qi = row % query_tokens;
    const int h = kh * repeat + local_h;
    const int64_t src =
        (((static_cast<int64_t>(query_idx) * query_heads + h) * query_tokens + qi) *
         head_dim) + d;
    q_col[d + row * head_dim] = queries[src];
  }
}

__global__ void zero_kernel(float* __restrict__ values, int n) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < n;
       idx += blockDim.x * gridDim.x) {
    values[idx] = 0.0f;
  }
}

__global__ void reduce_scores_kernel(
    const float* __restrict__ scores,
    float* __restrict__ route,
    int nodes,
    int key_tokens,
    int repeat,
    int query_tokens,
    int matrix_rows,
    float inv_sqrt_dim) {
  extern __shared__ float scratch[];
  const int pair = blockIdx.x;
  const int node = pair / repeat;
  const int local_h = pair - node * repeat;
  float best = 0.0f;
  const int total = key_tokens * query_tokens;
  for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
    const int kt = idx / query_tokens;
    const int qi = idx - kt * query_tokens;
    const int row = node * key_tokens + kt;
    const int col = local_h * query_tokens + qi;
    const float v = fabsf(scores[row + col * matrix_rows]);
    best = fmaxf(best, v);
  }
  scratch[threadIdx.x] = best;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(route + node, scratch[0] * inv_sqrt_dim);
  }
}

__global__ void normalize_kernel(float* __restrict__ route, int nodes, int query_heads) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < nodes;
       idx += blockDim.x * gridDim.x) {
    route[idx] /= static_cast<float>(query_heads);
  }
}

extern "C" int grm_gqa_cublas_route(
    const float* keys_host,
    const float* queries_host,
    int nodes,
    int kv_heads,
    int key_tokens,
    int query_count,
    int query_heads,
    int query_tokens,
    int head_dim,
    int warmup,
    float* out_scores_host,
    float* elapsed_ms_host) {
  if (!keys_host || !queries_host || !out_scores_host || !elapsed_ms_host ||
      nodes <= 0 || kv_heads <= 0 || key_tokens <= 0 || query_count <= 0 ||
      query_heads <= 0 || query_tokens <= 0 || head_dim <= 0 ||
      query_heads % kv_heads != 0 || warmup < 0 || warmup >= query_count) {
    return -1;
  }

  const int repeat = query_heads / kv_heads;
  const int matrix_rows = nodes * key_tokens;
  const int query_rows = repeat * query_tokens;
  const int threads = 256;
  const int grid = 256;
  const size_t key_values =
      static_cast<size_t>(nodes) * kv_heads * key_tokens * head_dim;
  const size_t k_col_values =
      static_cast<size_t>(kv_heads) * matrix_rows * head_dim;
  const size_t q_col_values =
      static_cast<size_t>(query_rows) * head_dim;
  const size_t score_values =
      static_cast<size_t>(matrix_rows) * query_rows;
  const size_t route_values = static_cast<size_t>(nodes);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const float inv_sqrt_dim = 1.0f / sqrtf(static_cast<float>(head_dim));

  float* d_keys = nullptr;
  float* d_queries = nullptr;
  float* d_k_cols = nullptr;
  float* d_q_col = nullptr;
  float* d_scores = nullptr;
  float* d_route = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cublasHandle_t handle = nullptr;

  int rc = cuda_code(cudaMalloc(&d_keys, key_values * sizeof(float)));
  if (rc != 0) goto cleanup;
  rc = cuda_code(cudaMalloc(&d_queries, static_cast<size_t>(query_count) * query_heads *
                                      query_tokens * head_dim * sizeof(float)));
  if (rc != 0) goto cleanup;
  rc = cuda_code(cudaMalloc(&d_k_cols, k_col_values * sizeof(float)));
  if (rc != 0) goto cleanup;
  rc = cuda_code(cudaMalloc(&d_q_col, q_col_values * sizeof(float)));
  if (rc != 0) goto cleanup;
  rc = cuda_code(cudaMalloc(&d_scores, score_values * sizeof(float)));
  if (rc != 0) goto cleanup;
  rc = cuda_code(cudaMalloc(&d_route, route_values * sizeof(float)));
  if (rc != 0) goto cleanup;
  rc = cuda_code(cudaMemcpy(
      d_keys, keys_host, key_values * sizeof(float), cudaMemcpyHostToDevice));
  if (rc != 0) goto cleanup;
  rc = cuda_code(cudaMemcpy(
      d_queries,
      queries_host,
      static_cast<size_t>(query_count) * query_heads * query_tokens * head_dim *
          sizeof(float),
      cudaMemcpyHostToDevice));
  if (rc != 0) goto cleanup;

  for (int kh = 0; kh < kv_heads; ++kh) {
    float* k_col = d_k_cols + static_cast<size_t>(kh) * matrix_rows * head_dim;
    pack_k_col_kernel<<<grid, threads>>>(
        d_keys, k_col, nodes, kv_heads, key_tokens, head_dim, kh);
  }
  rc = cuda_code(cudaGetLastError());
  if (rc != 0) goto cleanup;
  rc = cuda_code(cudaFree(d_keys));
  d_keys = nullptr;
  if (rc != 0) goto cleanup;

  rc = cublas_code(cublasCreate(&handle));
  if (rc != 0) goto cleanup;
  rc = cublas_code(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  if (rc != 0) goto cleanup;
  rc = cuda_code(cudaEventCreate(&start));
  if (rc != 0) goto cleanup;
  rc = cuda_code(cudaEventCreate(&stop));
  if (rc != 0) goto cleanup;

  for (int qi = 0; qi < query_count; ++qi) {
    if (qi == warmup) {
      rc = cuda_code(cudaEventRecord(start, 0));
      if (rc != 0) goto cleanup;
    }
    zero_kernel<<<grid, threads>>>(d_route, nodes);
    for (int kh = 0; kh < kv_heads; ++kh) {
      pack_q_col_kernel<<<grid, threads>>>(
          d_queries, d_q_col, qi, query_heads, query_tokens, head_dim, kv_heads, kh);
      rc = cuda_code(cudaGetLastError());
      if (rc != 0) goto cleanup;
      const float* k_col =
          d_k_cols + static_cast<size_t>(kh) * matrix_rows * head_dim;
      rc = cublas_code(cublasSgemm(
          handle,
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          matrix_rows,
          query_rows,
          head_dim,
          &alpha,
          k_col,
          head_dim,
          d_q_col,
          head_dim,
          &beta,
          d_scores,
          matrix_rows));
      if (rc != 0) goto cleanup;
      reduce_scores_kernel<<<nodes * repeat, threads, threads * sizeof(float)>>>(
          d_scores, d_route, nodes, key_tokens, repeat, query_tokens,
          matrix_rows, inv_sqrt_dim);
      rc = cuda_code(cudaGetLastError());
      if (rc != 0) goto cleanup;
    }
    normalize_kernel<<<grid, threads>>>(d_route, nodes, query_heads);
    rc = cuda_code(cudaGetLastError());
    if (rc != 0) goto cleanup;
    rc = cuda_code(cudaMemcpy(
        out_scores_host + static_cast<size_t>(qi) * nodes,
        d_route,
        route_values * sizeof(float),
        cudaMemcpyDeviceToHost));
    if (rc != 0) goto cleanup;
  }
  rc = cuda_code(cudaEventRecord(stop, 0));
  if (rc != 0) goto cleanup;
  rc = cuda_code(cudaEventSynchronize(stop));
  if (rc != 0) goto cleanup;
  rc = cuda_code(cudaEventElapsedTime(elapsed_ms_host, start, stop));

cleanup:
  if (handle) cublasDestroy(handle);
  if (start) cudaEventDestroy(start);
  if (stop) cudaEventDestroy(stop);
  if (d_keys) cudaFree(d_keys);
  if (d_queries) cudaFree(d_queries);
  if (d_k_cols) cudaFree(d_k_cols);
  if (d_q_col) cudaFree(d_q_col);
  if (d_scores) cudaFree(d_scores);
  if (d_route) cudaFree(d_route);
  return rc;
}
"""


def build_probe(build_dir: Path, nvcc: str) -> Path:
    source = build_dir / "grm_gqa_cuda_probe.cu"
    lib = build_dir / "libgrm_gqa_cuda_probe.so"
    source.write_text(CUDA_SOURCE, encoding="utf-8")
    subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-O3",
            "--shared",
            "-Xcompiler",
            "-fPIC",
            str(source),
            "-lcublas",
            "-o",
            str(lib),
        ],
        check=True,
    )
    return lib


class CudaProbe:
    def __init__(self, lib_path: Path):
        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.grm_gqa_cublas_route.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.grm_gqa_cublas_route.restype = ctypes.c_int

    def route_scores(
        self,
        keys: np.ndarray,
        queries: np.ndarray,
        *,
        warmup: int,
    ) -> tuple[np.ndarray, float, float]:
        keys = np.ascontiguousarray(keys, dtype=np.float32)
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        nodes, kv_heads, key_tokens, head_dim = keys.shape
        query_count, query_heads, query_tokens, query_dim = queries.shape
        if query_dim != head_dim:
            raise ValueError("query head_dim does not match keys")
        out = np.empty((query_count, nodes), dtype=np.float32)
        elapsed = ctypes.c_float(0.0)
        t0 = time.perf_counter_ns()
        rc = self.lib.grm_gqa_cublas_route(
            keys.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            queries.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            nodes,
            kv_heads,
            key_tokens,
            query_count,
            query_heads,
            query_tokens,
            head_dim,
            warmup,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(elapsed),
        )
        wall_ms = (time.perf_counter_ns() - t0) / 1.0e6
        if rc != 0:
            raise RuntimeError(f"CUDA/cuBLAS probe failed with code {rc}")
        return out, float(elapsed.value), wall_ms


def route_topk(scores: np.ndarray, topk: int) -> list[int]:
    rows = [
        (float(score), int(node_id))
        for node_id, score in enumerate(scores)
        if math.isfinite(float(score))
    ]
    rows.sort(key=lambda item: (-item[0], item[1]))
    return [node_id for _score, node_id in rows[:topk]]


def write_markdown(result: dict, path: Path) -> None:
    lines = [
        "# GRM GQA CUDA/cuBLAS Probe",
        "",
        "Standalone CUDA/cuBLAS raw q.k route probe over capture banks.",
        "",
        "## Result",
        "",
        f"- nodes: {result['nodes']}",
        f"- layer: {result['capture']['layer']}",
        f"- key_tokens: {result['shape']['key_tokens']}",
        f"- query_tokens: {result['shape']['query_tokens']}",
        f"- parity: {str(result['parity']).lower()}",
        f"- device measured p50 ms: {result['device_route_ms_p50']:.4f}",
        f"- device measured p95 ms: {result['device_route_ms_p95']:.4f}",
        f"- wall total ms: {result['wall_ms']:.4f}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=sorted(bench.PRESETS), default="qwen35-2b-attn")
    p.add_argument("--capture-dir", type=Path, required=True)
    p.add_argument("--capture-role", default="source")
    p.add_argument("--capture-layer", type=int, default=3)
    p.add_argument("--capture-limit", type=int, default=768)
    p.add_argument("--node-count", type=int, default=512)
    p.add_argument("--queries", type=int, default=6)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--query-tokens", type=int, default=4)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--nvcc", default="nvcc")
    p.add_argument("--keep-build", type=Path)
    p.add_argument("--out", type=Path, default=Path("/tmp/grm_gqa_cuda_probe.json"))
    p.add_argument("--markdown-out", type=Path)
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    nvcc = shutil.which(args.nvcc)
    if not nvcc:
        raise RuntimeError(f"nvcc not found: {args.nvcc}")
    preset = bench.PRESETS[args.preset]
    routes, capture_stats = bench.load_capture_routes(
        args.capture_dir,
        role=args.capture_role,
        layer=args.capture_layer,
        limit=args.capture_limit,
        token_limit=0,
        kv_heads=int(preset["kv_heads"]),
        head_dim=int(preset["head_dim"]),
    )
    if routes.shape[1] != 1:
        raise RuntimeError("CUDA probe currently expects one route key per node")
    if routes.shape[0] < args.node_count:
        raise RuntimeError(
            f"requested {args.node_count} nodes but only loaded {routes.shape[0]}")
    routes = np.ascontiguousarray(routes[:args.node_count, 0], dtype=np.float32)
    query_count = args.queries
    if query_count <= args.warmup:
        query_count = args.warmup + 1
    queries = bench.make_capture_queries(
        routes[:, None, :, :, :],
        query_count=query_count,
        query_heads=int(preset["query_heads"]),
        query_tokens=args.query_tokens,
    )

    build_parent: tempfile.TemporaryDirectory[str] | None = None
    if args.keep_build:
        build_dir = args.keep_build
        build_dir.mkdir(parents=True, exist_ok=True)
    else:
        build_parent = tempfile.TemporaryDirectory(prefix="grm-gqa-cuda-")
        build_dir = Path(build_parent.name)
    try:
        lib = build_probe(build_dir, nvcc)
        probe = CudaProbe(lib)
        scores, device_ms, wall_ms = probe.route_scores(
            routes,
            queries,
            warmup=args.warmup,
        )
    finally:
        if build_parent is not None:
            build_parent.cleanup()

    measured = max(1, query_count - args.warmup)
    per_query_ms = [device_ms / measured for _ in range(measured)]
    mismatches: list[dict] = []
    for i, query in enumerate(queries):
        expected = bench.python_gqa_reference_route(
            routes[:, None, :, :, :],
            query,
            (),
            args.topk,
            "batched",
        )
        got = route_topk(scores[i], args.topk)
        if got != expected:
            mismatches.append({"query": i, "python": expected, "cuda": got})
    result = {
        "schema": "grm-gqa-cuda-cublas-probe-v1",
        "repo": str(ROOT),
        "preset": args.preset,
        "nodes": int(routes.shape[0]),
        "topk": int(args.topk),
        "warmup": int(args.warmup),
        "queries": int(query_count),
        "measured_queries": int(measured),
        "device_route_ms_total": float(device_ms),
        "device_route_ms_p50": float(np.median(per_query_ms)),
        "device_route_ms_p95": float(np.percentile(per_query_ms, 95)),
        "wall_ms": float(wall_ms),
        "parity": not mismatches,
        "mismatches": mismatches[:5],
        "capture": capture_stats,
        "shape": {
            "query_heads": int(queries.shape[1]),
            "kv_heads": int(routes.shape[1]),
            "head_dim": int(routes.shape[3]),
            "query_tokens": int(queries.shape[2]),
            "key_tokens": int(routes.shape[2]),
        },
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    if args.markdown_out:
        write_markdown(result, args.markdown_out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
