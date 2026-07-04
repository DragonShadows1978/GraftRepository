#!/usr/bin/env python3
"""Probe a CUDA/cuBLAS GQA raw q.k router over real capture banks."""

from __future__ import annotations

import argparse
import ctypes
import json
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

__device__ bool better_score(float lhs_score, uint64_t lhs_id,
                             float rhs_score, uint64_t rhs_id) {
  return lhs_score > rhs_score ||
         (lhs_score == rhs_score && lhs_id < rhs_id);
}

__device__ void insert_topk(float score, uint64_t node_id,
                            float* scores, uint64_t* ids, int topk) {
  if (topk <= 0 || !isfinite(score)) {
    return;
  }
  if (!better_score(score, node_id, scores[topk - 1], ids[topk - 1])) {
    return;
  }
  int pos = topk - 1;
  while (pos > 0 && better_score(score, node_id, scores[pos - 1], ids[pos - 1])) {
    scores[pos] = scores[pos - 1];
    ids[pos] = ids[pos - 1];
    --pos;
  }
  scores[pos] = score;
  ids[pos] = node_id;
}

__global__ void topk_kernel(
    const float* __restrict__ route,
    uint64_t* __restrict__ topk_ids,
    int nodes,
    int topk) {
  constexpr int kMaxTopK = 16;
  extern __shared__ unsigned char scratch_raw[];
  float* shared_scores = reinterpret_cast<float*>(scratch_raw);
  uint64_t* shared_ids = reinterpret_cast<uint64_t*>(
      shared_scores + blockDim.x * kMaxTopK);
  float local_scores[kMaxTopK];
  uint64_t local_ids[kMaxTopK];
  if (topk > kMaxTopK) {
    return;
  }
  for (int i = 0; i < kMaxTopK; ++i) {
    local_scores[i] = -INFINITY;
    local_ids[i] = UINT64_MAX;
  }
  for (int node = threadIdx.x; node < nodes; node += blockDim.x) {
    insert_topk(route[node], static_cast<uint64_t>(node),
                local_scores, local_ids, topk);
  }
  const int base = threadIdx.x * kMaxTopK;
  for (int i = 0; i < kMaxTopK; ++i) {
    shared_scores[base + i] = local_scores[i];
    shared_ids[base + i] = local_ids[i];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float best_scores[kMaxTopK];
    uint64_t best_ids[kMaxTopK];
    for (int i = 0; i < kMaxTopK; ++i) {
      best_scores[i] = -INFINITY;
      best_ids[i] = UINT64_MAX;
    }
    for (int t = 0; t < blockDim.x; ++t) {
      const int t_base = t * kMaxTopK;
      for (int i = 0; i < topk; ++i) {
        insert_topk(shared_scores[t_base + i], shared_ids[t_base + i],
                    best_scores, best_ids, topk);
      }
    }
    for (int i = 0; i < topk; ++i) {
      topk_ids[i] = best_ids[i];
    }
  }
}

struct GqaDeviceArena {
  int nodes = 0;
  int kv_heads = 0;
  int key_tokens = 0;
  int head_dim = 0;
  int matrix_rows = 0;
  int threads = 256;
  int grid = 256;
  float* d_k_cols = nullptr;
  float* d_queries = nullptr;
  float* d_q_col = nullptr;
  float* d_scores = nullptr;
  float* d_route = nullptr;
  uint64_t* d_topk = nullptr;
  size_t query_values_cap = 0;
  size_t q_col_values_cap = 0;
  size_t score_values_cap = 0;
  size_t route_values_cap = 0;
  size_t topk_values_cap = 0;
  cublasHandle_t handle = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
};

static void destroy_device_arena(GqaDeviceArena* arena) {
  if (!arena) {
    return;
  }
  if (arena->handle) cublasDestroy(arena->handle);
  if (arena->start) cudaEventDestroy(arena->start);
  if (arena->stop) cudaEventDestroy(arena->stop);
  if (arena->d_k_cols) cudaFree(arena->d_k_cols);
  if (arena->d_queries) cudaFree(arena->d_queries);
  if (arena->d_q_col) cudaFree(arena->d_q_col);
  if (arena->d_scores) cudaFree(arena->d_scores);
  if (arena->d_route) cudaFree(arena->d_route);
  if (arena->d_topk) cudaFree(arena->d_topk);
  delete arena;
}

static int reserve_float_buffer(float** ptr, size_t* capacity, size_t values) {
  if (values <= *capacity) {
    return 0;
  }
  if (*ptr) {
    const int rc = cuda_code(cudaFree(*ptr));
    *ptr = nullptr;
    *capacity = 0;
    if (rc != 0) {
      return rc;
    }
  }
  const int rc = cuda_code(
      cudaMalloc(reinterpret_cast<void**>(ptr), values * sizeof(float)));
  if (rc == 0) {
    *capacity = values;
  }
  return rc;
}

static int reserve_uint64_buffer(uint64_t** ptr, size_t* capacity, size_t values) {
  if (values <= *capacity) {
    return 0;
  }
  if (*ptr) {
    const int rc = cuda_code(cudaFree(*ptr));
    *ptr = nullptr;
    *capacity = 0;
    if (rc != 0) {
      return rc;
    }
  }
  const int rc = cuda_code(
      cudaMalloc(reinterpret_cast<void**>(ptr), values * sizeof(uint64_t)));
  if (rc == 0) {
    *capacity = values;
  }
  return rc;
}

extern "C" int grm_gqa_cuda_arena_create(
    const float* keys_host,
    int nodes,
    int kv_heads,
    int key_tokens,
    int head_dim,
    GqaDeviceArena** out_arena) {
  if (!keys_host || !out_arena || nodes <= 0 || kv_heads <= 0 ||
      key_tokens <= 0 || head_dim <= 0) {
    return -1;
  }
  *out_arena = nullptr;
  const int threads = 256;
  const int grid = 256;
  const int matrix_rows = nodes * key_tokens;
  const size_t key_values =
      static_cast<size_t>(nodes) * kv_heads * key_tokens * head_dim;
  const size_t k_col_values =
      static_cast<size_t>(kv_heads) * matrix_rows * head_dim;

  GqaDeviceArena* arena = new GqaDeviceArena();
  arena->nodes = nodes;
  arena->kv_heads = kv_heads;
  arena->key_tokens = key_tokens;
  arena->head_dim = head_dim;
  arena->matrix_rows = matrix_rows;
  arena->threads = threads;
  arena->grid = grid;

  float* d_keys = nullptr;
  int rc = cuda_code(cudaMalloc(&d_keys, key_values * sizeof(float)));
  if (rc != 0) goto cleanup;
  rc = cuda_code(cudaMalloc(&arena->d_k_cols, k_col_values * sizeof(float)));
  if (rc != 0) goto cleanup;
  rc = cuda_code(cudaMemcpy(
      d_keys, keys_host, key_values * sizeof(float), cudaMemcpyHostToDevice));
  if (rc != 0) goto cleanup;
  for (int kh = 0; kh < kv_heads; ++kh) {
    float* k_col =
        arena->d_k_cols + static_cast<size_t>(kh) * matrix_rows * head_dim;
    pack_k_col_kernel<<<grid, threads>>>(
        d_keys, k_col, nodes, kv_heads, key_tokens, head_dim, kh);
  }
  rc = cuda_code(cudaGetLastError());
  if (rc != 0) goto cleanup;
  rc = cuda_code(cudaDeviceSynchronize());
  if (rc != 0) goto cleanup;
  rc = cuda_code(cudaFree(d_keys));
  d_keys = nullptr;
  if (rc != 0) goto cleanup;
  rc = cublas_code(cublasCreate(&arena->handle));
  if (rc != 0) goto cleanup;
  rc = cublas_code(cublasSetMathMode(arena->handle, CUBLAS_DEFAULT_MATH));
  if (rc != 0) goto cleanup;
  rc = cuda_code(cudaEventCreate(&arena->start));
  if (rc != 0) goto cleanup;
  rc = cuda_code(cudaEventCreate(&arena->stop));
  if (rc != 0) goto cleanup;
  *out_arena = arena;
  return 0;

cleanup:
  if (d_keys) cudaFree(d_keys);
  destroy_device_arena(arena);
  return rc;
}

extern "C" int grm_gqa_cuda_arena_route(
    GqaDeviceArena* arena,
    const float* queries_host,
    int query_count,
    int query_heads,
    int query_tokens,
    int warmup,
    int topk,
    uint64_t* out_topk_host,
    float* elapsed_ms_host) {
  if (!arena || !queries_host || !out_topk_host || !elapsed_ms_host ||
      query_count <= 0 || query_heads <= 0 || query_tokens <= 0 ||
      query_heads % arena->kv_heads != 0 || warmup < 0 ||
      warmup >= query_count || topk <= 0 || topk > 16 ||
      topk > arena->nodes) {
    return -1;
  }

  const int repeat = query_heads / arena->kv_heads;
  const int query_rows = repeat * query_tokens;
  const int threads = arena->threads;
  const int grid = arena->grid;
  const int matrix_rows = arena->matrix_rows;
  const size_t query_values =
      static_cast<size_t>(query_count) * query_heads * query_tokens *
      arena->head_dim;
  const size_t q_col_values =
      static_cast<size_t>(query_rows) * arena->head_dim;
  const size_t score_values =
      static_cast<size_t>(matrix_rows) * query_rows;
  const size_t route_values = static_cast<size_t>(arena->nodes);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const float inv_sqrt_dim = 1.0f / sqrtf(static_cast<float>(arena->head_dim));

  int rc = reserve_float_buffer(
      &arena->d_queries, &arena->query_values_cap, query_values);
  if (rc != 0) goto cleanup;
  rc = reserve_float_buffer(
      &arena->d_q_col, &arena->q_col_values_cap, q_col_values);
  if (rc != 0) goto cleanup;
  rc = reserve_float_buffer(
      &arena->d_scores, &arena->score_values_cap, score_values);
  if (rc != 0) goto cleanup;
  rc = reserve_float_buffer(
      &arena->d_route, &arena->route_values_cap, route_values);
  if (rc != 0) goto cleanup;
  rc = reserve_uint64_buffer(
      &arena->d_topk, &arena->topk_values_cap,
      static_cast<size_t>(query_count) * topk);
  if (rc != 0) goto cleanup;
  rc = cuda_code(cudaMemcpy(
      arena->d_queries,
      queries_host,
      query_values * sizeof(float),
      cudaMemcpyHostToDevice));
  if (rc != 0) goto cleanup;

  for (int qi = 0; qi < query_count; ++qi) {
    if (qi == warmup) {
      rc = cuda_code(cudaEventRecord(arena->start, 0));
      if (rc != 0) goto cleanup;
    }
    zero_kernel<<<grid, threads>>>(arena->d_route, arena->nodes);
    for (int kh = 0; kh < arena->kv_heads; ++kh) {
      pack_q_col_kernel<<<grid, threads>>>(
          arena->d_queries, arena->d_q_col, qi, query_heads, query_tokens,
          arena->head_dim, arena->kv_heads, kh);
      rc = cuda_code(cudaGetLastError());
      if (rc != 0) goto cleanup;
      const float* k_col =
          arena->d_k_cols +
          static_cast<size_t>(kh) * matrix_rows * arena->head_dim;
      rc = cublas_code(cublasSgemm(
          arena->handle,
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          matrix_rows,
          query_rows,
          arena->head_dim,
          &alpha,
          k_col,
          arena->head_dim,
          arena->d_q_col,
          arena->head_dim,
          &beta,
          arena->d_scores,
          matrix_rows));
      if (rc != 0) goto cleanup;
      reduce_scores_kernel<<<arena->nodes * repeat, threads, threads * sizeof(float)>>>(
          arena->d_scores, arena->d_route, arena->nodes, arena->key_tokens, repeat,
          query_tokens, matrix_rows, inv_sqrt_dim);
      rc = cuda_code(cudaGetLastError());
      if (rc != 0) goto cleanup;
    }
    normalize_kernel<<<grid, threads>>>(arena->d_route, arena->nodes, query_heads);
    rc = cuda_code(cudaGetLastError());
    if (rc != 0) goto cleanup;
    topk_kernel<<<1, threads, threads * 16 * (sizeof(float) + sizeof(uint64_t))>>>(
        arena->d_route, arena->d_topk + static_cast<size_t>(qi) * topk,
        arena->nodes, topk);
    rc = cuda_code(cudaGetLastError());
    if (rc != 0) goto cleanup;
  }
  rc = cuda_code(cudaEventRecord(arena->stop, 0));
  if (rc != 0) goto cleanup;
  rc = cuda_code(cudaEventSynchronize(arena->stop));
  if (rc != 0) goto cleanup;
  rc = cuda_code(cudaEventElapsedTime(elapsed_ms_host, arena->start, arena->stop));
  if (rc != 0) goto cleanup;
  rc = cuda_code(cudaMemcpy(
      out_topk_host,
      arena->d_topk,
      static_cast<size_t>(query_count) * topk * sizeof(uint64_t),
      cudaMemcpyDeviceToHost));

cleanup:
  return rc;
}

extern "C" int grm_gqa_cuda_arena_destroy(GqaDeviceArena* arena) {
  destroy_device_arena(arena);
  return 0;
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
    int topk,
    uint64_t* out_topk_host,
    float* elapsed_ms_host) {
  GqaDeviceArena* arena = nullptr;
  int rc = grm_gqa_cuda_arena_create(
      keys_host, nodes, kv_heads, key_tokens, head_dim, &arena);
  if (rc != 0) {
    return rc;
  }
  rc = grm_gqa_cuda_arena_route(
      arena, queries_host, query_count, query_heads, query_tokens, warmup, topk,
      out_topk_host, elapsed_ms_host);
  const int destroy_rc = grm_gqa_cuda_arena_destroy(arena);
  return rc != 0 ? rc : destroy_rc;
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
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.grm_gqa_cublas_route.restype = ctypes.c_int
        self.lib.grm_gqa_cuda_arena_create.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.grm_gqa_cuda_arena_create.restype = ctypes.c_int
        self.lib.grm_gqa_cuda_arena_route.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.grm_gqa_cuda_arena_route.restype = ctypes.c_int
        self.lib.grm_gqa_cuda_arena_destroy.argtypes = [ctypes.c_void_p]
        self.lib.grm_gqa_cuda_arena_destroy.restype = ctypes.c_int

    def route_topk(
        self,
        keys: np.ndarray,
        queries: np.ndarray,
        *,
        warmup: int,
        topk: int,
    ) -> tuple[np.ndarray, float, float]:
        keys = np.ascontiguousarray(keys, dtype=np.float32)
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        nodes, kv_heads, key_tokens, head_dim = keys.shape
        query_count, query_heads, query_tokens, query_dim = queries.shape
        if query_dim != head_dim:
            raise ValueError("query head_dim does not match keys")
        out = np.empty((query_count, topk), dtype=np.uint64)
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
            topk,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.byref(elapsed),
        )
        wall_ms = (time.perf_counter_ns() - t0) / 1.0e6
        if rc != 0:
            raise RuntimeError(f"CUDA/cuBLAS probe failed with code {rc}")
        return out, float(elapsed.value), wall_ms

    def create_arena(self, keys: np.ndarray) -> "CudaDeviceArena":
        keys = np.ascontiguousarray(keys, dtype=np.float32)
        nodes, kv_heads, key_tokens, head_dim = keys.shape
        handle = ctypes.c_void_p()
        t0 = time.perf_counter_ns()
        rc = self.lib.grm_gqa_cuda_arena_create(
            keys.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            nodes,
            kv_heads,
            key_tokens,
            head_dim,
            ctypes.byref(handle),
        )
        setup_wall_ms = (time.perf_counter_ns() - t0) / 1.0e6
        if rc != 0:
            raise RuntimeError(f"CUDA arena create failed with code {rc}")
        return CudaDeviceArena(
            self,
            handle,
            nodes=nodes,
            kv_heads=kv_heads,
            key_tokens=key_tokens,
            head_dim=head_dim,
            setup_wall_ms=setup_wall_ms,
        )


class CudaDeviceArena:
    def __init__(
        self,
        probe: CudaProbe,
        handle: ctypes.c_void_p,
        *,
        nodes: int,
        kv_heads: int,
        key_tokens: int,
        head_dim: int,
        setup_wall_ms: float,
    ):
        self.probe = probe
        self.handle = handle
        self.nodes = nodes
        self.kv_heads = kv_heads
        self.key_tokens = key_tokens
        self.head_dim = head_dim
        self.setup_wall_ms = setup_wall_ms

    def close(self) -> None:
        if not self.handle:
            return
        rc = self.probe.lib.grm_gqa_cuda_arena_destroy(self.handle)
        self.handle = ctypes.c_void_p()
        if rc != 0:
            raise RuntimeError(f"CUDA arena destroy failed with code {rc}")

    def __del__(self) -> None:
        if getattr(self, "handle", None):
            self.probe.lib.grm_gqa_cuda_arena_destroy(self.handle)
            self.handle = ctypes.c_void_p()

    def route_topk(
        self,
        queries: np.ndarray,
        *,
        warmup: int,
        topk: int,
    ) -> tuple[np.ndarray, float, float]:
        if not self.handle:
            raise RuntimeError("CUDA arena is closed")
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        query_count, query_heads, query_tokens, query_dim = queries.shape
        if query_dim != self.head_dim:
            raise ValueError("query head_dim does not match keys")
        if topk > self.nodes:
            raise ValueError("topk cannot exceed node count")
        out = np.empty((query_count, topk), dtype=np.uint64)
        elapsed = ctypes.c_float(0.0)
        t0 = time.perf_counter_ns()
        rc = self.probe.lib.grm_gqa_cuda_arena_route(
            self.handle,
            queries.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            query_count,
            query_heads,
            query_tokens,
            warmup,
            topk,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.byref(elapsed),
        )
        wall_ms = (time.perf_counter_ns() - t0) / 1.0e6
        if rc != 0:
            raise RuntimeError(f"CUDA/cuBLAS arena route failed with code {rc}")
        return out, float(elapsed.value), wall_ms


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
        f"- output_mode: `{result['output_mode']}`",
        f"- arena_mode: `{result['arena_mode']}`",
        f"- scratch_mode: `{result['scratch_mode']}`",
        f"- route_repeats: {result['route_repeats']}",
        f"- parity: {str(result['parity']).lower()}",
        f"- device measured p50 ms: {result['device_route_ms_p50']:.4f}",
        f"- device measured p95 ms: {result['device_route_ms_p95']:.4f}",
        f"- setup wall ms: {result['setup_wall_ms']:.4f}",
        f"- route wall ms: {result['route_wall_ms']:.4f}",
        f"- total wall ms: {result['total_wall_ms']:.4f}",
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
    p.add_argument("--route-repeats", type=int, default=1,
                   help="Route the same queries repeatedly after arena setup; "
                        "the final repeat is reported as the reused-scratch run.")
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
    arena: CudaDeviceArena | None = None
    setup_wall_ms = 0.0
    try:
        lib = build_probe(build_dir, nvcc)
        probe = CudaProbe(lib)
        arena = probe.create_arena(routes)
        setup_wall_ms = arena.setup_wall_ms
        route_repeats = max(1, args.route_repeats)
        route_walls: list[float] = []
        device_times: list[float] = []
        topk_ids = None
        for _ in range(route_repeats):
            topk_ids, device_ms, route_wall_ms = arena.route_topk(
                queries,
                warmup=args.warmup,
                topk=args.topk,
            )
            route_walls.append(float(route_wall_ms))
            device_times.append(float(device_ms))
    finally:
        if arena is not None:
            arena.close()
        if build_parent is not None:
            build_parent.cleanup()
    if topk_ids is None:
        raise RuntimeError("CUDA route did not run")

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
        got = [int(x) for x in topk_ids[i].tolist()]
        if got != expected:
            mismatches.append({"query": i, "python": expected, "cuda": got})
    result = {
        "schema": "grm-gqa-cuda-cublas-probe-v1",
        "output_mode": "gpu_topk",
        "repo": str(ROOT),
        "preset": args.preset,
        "nodes": int(routes.shape[0]),
        "topk": int(args.topk),
        "warmup": int(args.warmup),
        "queries": int(query_count),
        "measured_queries": int(measured),
        "route_repeats": int(max(1, args.route_repeats)),
        "device_route_ms_runs": device_times,
        "device_route_ms_total": float(device_ms),
        "device_route_ms_p50": float(np.median(per_query_ms)),
        "device_route_ms_p95": float(np.percentile(per_query_ms, 95)),
        "wall_ms": float(route_wall_ms),
        "route_wall_ms": float(route_wall_ms),
        "route_wall_ms_first": float(route_walls[0]),
        "route_wall_ms_min": float(np.min(route_walls)),
        "route_wall_ms_runs": route_walls,
        "setup_wall_ms": float(setup_wall_ms),
        "total_wall_ms": float(setup_wall_ms + route_wall_ms),
        "arena_mode": "persistent_gpu_k",
        "scratch_mode": "persistent_route_buffers",
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
