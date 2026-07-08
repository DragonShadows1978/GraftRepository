#!/usr/bin/env python3
"""Persistent CUDA/cuBLAS MLA centroid route sidecar (P2,
docs/GRM_MLA_CUDA_ROUTE_PLAN.md).

Mirrors scripts/grm_gqa_cuda_probe.py's conventions exactly (persistent
device arena struct, GPU top-k with the same tie-break, host-pointer AND
device-pointer query entries, C ABI returning uint64 node-row ids) but
scores the MLA dialect's shape: ONE fp32 centroid vector per node (no
per-head/per-token reduction -- MLA's routing key is a single latent
centroid, `ArenaCache._node_key`/`_probe_key` in core/graft_arena.py), and
the exact scoring law is plain cosine similarity, replicated from
`RouterIndex::exact_mla_entry_score` (cpp/grm_runtime.cpp:3518-3545):

    score = dot(query, row) / (||query|| * ||row|| + 1e-8)

accumulated in fp64 on the CPU reference; this device path accumulates the
dot product in fp32 via cuBLAS Sgemv (memory-bound GEMV-class kernel per
the kernel-opt vocabulary -- Sgemv first, hand-roll only if receipts
demand) and divides by norms computed once at arena-build time (row
norms) and once per route call (query norm). Multi-row entries
(`route_keys.size() > 1`, e.g. hierarchical digest nodes with child
centroids) are OUT OF SCOPE for the CUDA path -- the P2 eligibility
contract (mirroring GQA) only ever builds the device arena from the
dense, single-centroid-per-node eligible base; multi-row entries fall
through to the CPU path exactly like GQA's `child_cents` exclusion.

Tie-order and NaN law replicate `score_node_better` (grm_runtime.cpp:75-82)
and the M6 house law: score DESC, tie-break node_id ASC, non-finite scores
dropped before ranking (never sorted).
"""

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

// Row L2 norms of the resident centroid matrix C[N x D] (row-major on the
// host). One norm per node, computed once at attach.
__global__ void row_norms_kernel(
    const float* __restrict__ rows,
    float* __restrict__ norms,
    int nodes,
    int dim) {
  for (int node = blockIdx.x * blockDim.x + threadIdx.x;
       node < nodes;
       node += blockDim.x * gridDim.x) {
    const float* row = rows + static_cast<int64_t>(node) * dim;
    double acc = 0.0;
    for (int d = 0; d < dim; ++d) {
      const double v = static_cast<double>(row[d]);
      acc += v * v;
    }
    norms[node] = static_cast<float>(sqrt(acc));
  }
}

// score[node] = dot[node] / (qnorm * row_norm[node] + 1e-8). NaN law: a
// non-finite dot or norm propagates NaN, filtered by the top-k inserter
// (M6: dropped, never sorted).
__global__ void cosine_score_kernel(
    const float* __restrict__ dots,
    const float* __restrict__ row_norms,
    float* __restrict__ scores,
    int nodes,
    float qnorm_eps,
    const float* __restrict__ qnorm) {
  for (int node = blockIdx.x * blockDim.x + threadIdx.x;
       node < nodes;
       node += blockDim.x * gridDim.x) {
    const float d = qnorm[0] * row_norms[node] + qnorm_eps;
    scores[node] = dots[node] / d;
  }
}

__global__ void query_norm_kernel(
    const float* __restrict__ query,
    float* __restrict__ qnorm_out,
    int dim) {
  extern __shared__ double scratch[];
  double acc = 0.0;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    const double v = static_cast<double>(query[d]);
    acc += v * v;
  }
  scratch[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    qnorm_out[0] = static_cast<float>(sqrt(scratch[0]));
  }
}

__device__ bool better_score(float lhs_score, uint64_t lhs_id,
                             float rhs_score, uint64_t rhs_id) {
  // score_node_better (grm_runtime.cpp:75-82): score DESC, node_id ASC.
  return lhs_score > rhs_score ||
         (lhs_score == rhs_score && lhs_id < rhs_id);
}

__device__ void insert_topk(float score, uint64_t node_id,
                            float* scores, uint64_t* ids, int topk) {
  if (topk <= 0 || !isfinite(score)) {
    return;  // M6: non-finite scores never enter the ranking.
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
    const float* __restrict__ scores_in,
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
    insert_topk(scores_in[node], static_cast<uint64_t>(node),
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

struct MlaDeviceArena {
  int nodes = 0;
  int dim = 0;
  int threads = 256;
  int grid = 256;
  float* d_rows = nullptr;       // C[N x D], row-major, resident
  float* d_row_norms = nullptr;  // ||C[i]||, resident, built once at attach
  float* d_queries = nullptr;    // scratch: uploaded host queries
  float* d_qnorm = nullptr;      // scratch: current query's ||q|| (scalar)
  float* d_dots = nullptr;       // scratch: dot(q, C), length N
  float* d_scores = nullptr;     // scratch: cosine scores, length N
  uint64_t* d_topk = nullptr;    // scratch: topk ids, Q x topk
  size_t query_values_cap = 0;
  size_t dots_cap = 0;
  size_t scores_cap = 0;
  size_t topk_cap = 0;
  cublasHandle_t handle = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
};

static void destroy_device_arena(MlaDeviceArena* arena) {
  if (!arena) {
    return;
  }
  if (arena->handle) cublasDestroy(arena->handle);
  if (arena->start) cudaEventDestroy(arena->start);
  if (arena->stop) cudaEventDestroy(arena->stop);
  if (arena->d_rows) cudaFree(arena->d_rows);
  if (arena->d_row_norms) cudaFree(arena->d_row_norms);
  if (arena->d_queries) cudaFree(arena->d_queries);
  if (arena->d_qnorm) cudaFree(arena->d_qnorm);
  if (arena->d_dots) cudaFree(arena->d_dots);
  if (arena->d_scores) cudaFree(arena->d_scores);
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

// Arena create: upload the dense eligible centroid bank (rows are exactly
// the P1b epoch-cached cand-base rows + their node ids -- built by the
// Python side; this ABI just takes the flat matrix) and precompute row
// norms once (memory-bound GEMV amortizes; norms are O(N*D) but paid only
// at attach, not per route call).
extern "C" int grm_mla_cuda_arena_create(
    const float* rows_host,
    int nodes,
    int dim,
    MlaDeviceArena** out_arena) {
  if (!rows_host || !out_arena || nodes <= 0 || dim <= 0) {
    return -1;
  }
  *out_arena = nullptr;
  const int threads = 256;
  const int grid = 256;
  const size_t row_values = static_cast<size_t>(nodes) * dim;

  MlaDeviceArena* arena = new MlaDeviceArena();
  arena->nodes = nodes;
  arena->dim = dim;
  arena->threads = threads;
  arena->grid = grid;

  int rc = cuda_code(cudaMalloc(&arena->d_rows, row_values * sizeof(float)));
  if (rc != 0) goto cleanup;
  rc = cuda_code(cudaMalloc(&arena->d_row_norms,
                            static_cast<size_t>(nodes) * sizeof(float)));
  if (rc != 0) goto cleanup;
  rc = cuda_code(cudaMalloc(&arena->d_qnorm, sizeof(float)));
  if (rc != 0) goto cleanup;
  rc = cuda_code(cudaMemcpy(
      arena->d_rows, rows_host, row_values * sizeof(float),
      cudaMemcpyHostToDevice));
  if (rc != 0) goto cleanup;
  row_norms_kernel<<<grid, threads>>>(
      arena->d_rows, arena->d_row_norms, nodes, dim);
  rc = cuda_code(cudaGetLastError());
  if (rc != 0) goto cleanup;
  rc = cuda_code(cudaDeviceSynchronize());
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
  destroy_device_arena(arena);
  return rc;
}

// Core routing loop over a DEVICE-resident query buffer (d_queries_in
// already holds (query_count, dim) floats on-device, row-major). Shared
// by both the host-pointer and device-pointer public entries below,
// mirroring the GQA bridge's route_device_queries split exactly.
//
// GEMV-class kernel-opt note (ledger opening, David): this is memory-
// bound -- prefer cuBLAS first (Sgemv per query; a query_count>1 batch
// still dispatches one Sgemv per query rather than a hand-rolled batched
// kernel, since receipts have not yet demanded more).
static int route_device_queries(
    MlaDeviceArena* arena,
    const float* d_queries_in,
    int query_count,
    int warmup,
    int topk,
    uint64_t* out_topk_host,
    float* elapsed_ms_host) {
  if (!arena || !d_queries_in || !out_topk_host || !elapsed_ms_host ||
      query_count <= 0 || warmup < 0 || warmup >= query_count ||
      topk <= 0 || topk > 16 || topk > arena->nodes) {
    return -1;
  }

  const int threads = arena->threads;
  const int grid = arena->grid;
  const int nodes = arena->nodes;
  const int dim = arena->dim;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  constexpr float kEps = 1.0e-8f;

  int rc = reserve_float_buffer(&arena->d_dots, &arena->dots_cap,
                                static_cast<size_t>(nodes));
  if (rc != 0) goto cleanup;
  rc = reserve_float_buffer(&arena->d_scores, &arena->scores_cap,
                            static_cast<size_t>(nodes));
  if (rc != 0) goto cleanup;
  rc = reserve_uint64_buffer(
      &arena->d_topk, &arena->topk_cap,
      static_cast<size_t>(query_count) * topk);
  if (rc != 0) goto cleanup;

  for (int qi = 0; qi < query_count; ++qi) {
    if (qi == warmup) {
      rc = cuda_code(cudaEventRecord(arena->start, 0));
      if (rc != 0) goto cleanup;
    }
    const float* q = d_queries_in + static_cast<int64_t>(qi) * dim;
    // dots[node] = sum_d rows[node, d] * q[d]. d_rows is row-major
    // (nodes x dim); cuBLAS is column-major, so the row-major (nodes x
    // dim) buffer IS the column-major (dim x nodes) transpose of itself
    // in memory -- CUBLAS_OP_T on that column-major view recovers
    // "nodes x dim times dim x 1" exactly (the same trick
    // grm_gqa_cuda_probe.py's k_col/CUBLAS_OP_T usage relies on).
    rc = cublas_code(cublasSgemv(
        arena->handle,
        CUBLAS_OP_T,
        dim,
        nodes,
        &alpha,
        arena->d_rows,
        dim,
        q,
        1,
        &beta,
        arena->d_dots,
        1));
    if (rc != 0) goto cleanup;
    query_norm_kernel<<<1, 256, 256 * sizeof(double)>>>(
        q, arena->d_qnorm, dim);
    rc = cuda_code(cudaGetLastError());
    if (rc != 0) goto cleanup;
    cosine_score_kernel<<<grid, threads>>>(
        arena->d_dots, arena->d_row_norms, arena->d_scores, nodes, kEps,
        arena->d_qnorm);
    rc = cuda_code(cudaGetLastError());
    if (rc != 0) goto cleanup;
    topk_kernel<<<1, threads, threads * 16 * (sizeof(float) + sizeof(uint64_t))>>>(
        arena->d_scores, arena->d_topk + static_cast<int64_t>(qi) * topk,
        nodes, topk);
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

extern "C" int grm_mla_cuda_arena_route(
    MlaDeviceArena* arena,
    const float* queries_host,
    int query_count,
    int dim,
    int warmup,
    int topk,
    uint64_t* out_topk_host,
    float* elapsed_ms_host) {
  if (!arena || !queries_host || query_count <= 0 || dim <= 0 ||
      dim != arena->dim) {
    return -1;
  }
  const size_t query_values = static_cast<size_t>(query_count) * dim;
  int rc = reserve_float_buffer(
      &arena->d_queries, &arena->query_values_cap, query_values);
  if (rc != 0) {
    return rc;
  }
  rc = cuda_code(cudaMemcpy(
      arena->d_queries,
      queries_host,
      query_values * sizeof(float),
      cudaMemcpyHostToDevice));
  if (rc != 0) {
    return rc;
  }
  return route_device_queries(
      arena, arena->d_queries, query_count, warmup, topk, out_topk_host,
      elapsed_ms_host);
}

// Device-pointer route entry: caller already holds (query_count, dim)
// float32 queries on GPU (contract: same layout the host entry would have
// H2D-uploaded) and passes the device pointer directly -- no H2D round
// trip. Mirrors grm_gqa_cuda_arena_route_device exactly.
extern "C" int grm_mla_cuda_arena_route_device(
    MlaDeviceArena* arena,
    const float* d_queries,
    int query_count,
    int dim,
    int warmup,
    int topk,
    uint64_t* out_topk_host,
    float* elapsed_ms_host) {
  if (!arena || dim != arena->dim) {
    return -1;
  }
  return route_device_queries(
      arena, d_queries, query_count, warmup, topk, out_topk_host,
      elapsed_ms_host);
}

// Test-only self-contained harness helpers (device-entry validation):
// upload a host float buffer to a fresh device allocation. NOT part of
// the production hot path. Mirrors grm_gqa_cuda_upload_queries exactly.
extern "C" int grm_mla_cuda_upload_queries(
    const float* host_buf, size_t num_floats, float** out_device_ptr) {
  if (!host_buf || !out_device_ptr || num_floats == 0) {
    return -1;
  }
  *out_device_ptr = nullptr;
  float* d_ptr = nullptr;
  int rc = cuda_code(cudaMalloc(
      reinterpret_cast<void**>(&d_ptr), num_floats * sizeof(float)));
  if (rc != 0) {
    return rc;
  }
  rc = cuda_code(cudaMemcpy(
      d_ptr, host_buf, num_floats * sizeof(float), cudaMemcpyHostToDevice));
  if (rc != 0) {
    cudaFree(d_ptr);
    return rc;
  }
  *out_device_ptr = d_ptr;
  return 0;
}

extern "C" int grm_mla_cuda_free_device_ptr(float* device_ptr) {
  if (!device_ptr) {
    return 0;
  }
  return cuda_code(cudaFree(device_ptr));
}

extern "C" int grm_mla_cuda_arena_destroy(MlaDeviceArena* arena) {
  destroy_device_arena(arena);
  return 0;
}

extern "C" int grm_mla_cublas_route(
    const float* rows_host,
    const float* queries_host,
    int nodes,
    int dim,
    int query_count,
    int warmup,
    int topk,
    uint64_t* out_topk_host,
    float* elapsed_ms_host) {
  MlaDeviceArena* arena = nullptr;
  int rc = grm_mla_cuda_arena_create(rows_host, nodes, dim, &arena);
  if (rc != 0) {
    return rc;
  }
  rc = grm_mla_cuda_arena_route(
      arena, queries_host, query_count, dim, warmup, topk, out_topk_host,
      elapsed_ms_host);
  const int destroy_rc = grm_mla_cuda_arena_destroy(arena);
  return rc != 0 ? rc : destroy_rc;
}
"""


def build_probe(build_dir: Path, nvcc: str) -> Path:
    source = build_dir / "grm_mla_cuda_probe.cu"
    lib = build_dir / "libgrm_mla_cuda_probe.so"
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
        self.lib.grm_mla_cublas_route.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.grm_mla_cublas_route.restype = ctypes.c_int
        self.lib.grm_mla_cuda_arena_create.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.grm_mla_cuda_arena_create.restype = ctypes.c_int
        self.lib.grm_mla_cuda_arena_route.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.grm_mla_cuda_arena_route.restype = ctypes.c_int
        self.lib.grm_mla_cuda_arena_destroy.argtypes = [ctypes.c_void_p]
        self.lib.grm_mla_cuda_arena_destroy.restype = ctypes.c_int
        self.lib.grm_mla_cuda_arena_route_device.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,   # device float* -- opaque on the Python side
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.grm_mla_cuda_arena_route_device.restype = ctypes.c_int
        self.lib.grm_mla_cuda_upload_queries.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.grm_mla_cuda_upload_queries.restype = ctypes.c_int
        self.lib.grm_mla_cuda_free_device_ptr.argtypes = [ctypes.c_void_p]
        self.lib.grm_mla_cuda_free_device_ptr.restype = ctypes.c_int

    def upload_queries(self, queries: np.ndarray) -> "CudaDeviceQueryPtr":
        """Test-only harness: upload a host float32 array to a fresh device
        allocation, returning an owning handle. Mirrors
        grm_gqa_cuda_probe.CudaProbe.upload_queries exactly."""
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        handle = ctypes.c_void_p()
        rc = self.lib.grm_mla_cuda_upload_queries(
            queries.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(queries.size),
            ctypes.byref(handle),
        )
        if rc != 0:
            raise RuntimeError(f"CUDA query upload failed with code {rc}")
        return CudaDeviceQueryPtr(self, handle, shape=queries.shape)

    def route_topk(
        self,
        rows: np.ndarray,
        queries: np.ndarray,
        *,
        warmup: int,
        topk: int,
    ) -> tuple[np.ndarray, float, float]:
        rows = np.ascontiguousarray(rows, dtype=np.float32)
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        nodes, dim = rows.shape
        query_count, query_dim = queries.shape
        if query_dim != dim:
            raise ValueError("query dim does not match rows")
        out = np.empty((query_count, topk), dtype=np.uint64)
        elapsed = ctypes.c_float(0.0)
        t0 = time.perf_counter_ns()
        rc = self.lib.grm_mla_cublas_route(
            rows.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            queries.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            nodes,
            dim,
            query_count,
            warmup,
            topk,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.byref(elapsed),
        )
        wall_ms = (time.perf_counter_ns() - t0) / 1.0e6
        if rc != 0:
            raise RuntimeError(f"CUDA/cuBLAS MLA probe failed with code {rc}")
        return out, float(elapsed.value), wall_ms

    def create_arena(self, rows: np.ndarray) -> "CudaDeviceArena":
        rows = np.ascontiguousarray(rows, dtype=np.float32)
        nodes, dim = rows.shape
        handle = ctypes.c_void_p()
        t0 = time.perf_counter_ns()
        rc = self.lib.grm_mla_cuda_arena_create(
            rows.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            nodes,
            dim,
            ctypes.byref(handle),
        )
        setup_wall_ms = (time.perf_counter_ns() - t0) / 1.0e6
        if rc != 0:
            raise RuntimeError(f"CUDA MLA arena create failed with code {rc}")
        return CudaDeviceArena(
            self, handle, nodes=nodes, dim=dim, setup_wall_ms=setup_wall_ms)


class CudaDeviceArena:
    def __init__(
        self,
        probe: CudaProbe,
        handle: ctypes.c_void_p,
        *,
        nodes: int,
        dim: int,
        setup_wall_ms: float,
    ):
        self.probe = probe
        self.handle = handle
        self.nodes = nodes
        self.dim = dim
        self.setup_wall_ms = setup_wall_ms

    def close(self) -> None:
        if not self.handle:
            return
        rc = self.probe.lib.grm_mla_cuda_arena_destroy(self.handle)
        self.handle = ctypes.c_void_p()
        if rc != 0:
            raise RuntimeError(f"CUDA MLA arena destroy failed with code {rc}")

    def __del__(self) -> None:
        if getattr(self, "handle", None):
            self.probe.lib.grm_mla_cuda_arena_destroy(self.handle)
            self.handle = ctypes.c_void_p()

    def route_topk(
        self,
        queries: np.ndarray,
        *,
        warmup: int,
        topk: int,
    ) -> tuple[np.ndarray, float, float]:
        if not self.handle:
            raise RuntimeError("CUDA MLA arena is closed")
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        query_count, query_dim = queries.shape
        if query_dim != self.dim:
            raise ValueError("query dim does not match rows")
        if topk > self.nodes:
            raise ValueError("topk cannot exceed node count")
        out = np.empty((query_count, topk), dtype=np.uint64)
        elapsed = ctypes.c_float(0.0)
        t0 = time.perf_counter_ns()
        rc = self.probe.lib.grm_mla_cuda_arena_route(
            self.handle,
            queries.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            query_count,
            self.dim,
            warmup,
            topk,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.byref(elapsed),
        )
        wall_ms = (time.perf_counter_ns() - t0) / 1.0e6
        if rc != 0:
            raise RuntimeError(f"CUDA/cuBLAS MLA arena route failed with code {rc}")
        return out, float(elapsed.value), wall_ms

    def route_topk_device(
        self,
        device_queries: "CudaDeviceQueryPtr",
        *,
        warmup: int,
        topk: int,
    ) -> tuple[np.ndarray, float, float]:
        if not self.handle:
            raise RuntimeError("CUDA MLA arena is closed")
        query_count, query_dim = device_queries.shape
        if query_dim != self.dim:
            raise ValueError("query dim does not match rows")
        if topk > self.nodes:
            raise ValueError("topk cannot exceed node count")
        out = np.empty((query_count, topk), dtype=np.uint64)
        elapsed = ctypes.c_float(0.0)
        t0 = time.perf_counter_ns()
        rc = self.probe.lib.grm_mla_cuda_arena_route_device(
            self.handle,
            device_queries.handle,
            query_count,
            self.dim,
            warmup,
            topk,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.byref(elapsed),
        )
        wall_ms = (time.perf_counter_ns() - t0) / 1.0e6
        if rc != 0:
            raise RuntimeError(
                f"CUDA/cuBLAS MLA arena device route failed with code {rc}")
        return out, float(elapsed.value), wall_ms


class CudaDeviceQueryPtr:
    """Owning handle for a device-resident query buffer (test harness).

    Mirrors grm_gqa_cuda_probe.CudaDeviceQueryPtr exactly: not part of the
    production hot path -- real device-resident callers hold their own
    pointer and pass it directly to grm_mla_cuda_arena_route_device."""

    def __init__(self, probe: "CudaProbe", handle: ctypes.c_void_p, *,
                shape: tuple[int, ...]):
        self.probe = probe
        self.handle = handle
        self.shape = tuple(int(x) for x in shape)

    def close(self) -> None:
        if self.handle:
            self.probe.lib.grm_mla_cuda_free_device_ptr(self.handle)
            self.handle = ctypes.c_void_p()

    def __del__(self) -> None:
        if getattr(self, "handle", None):
            self.probe.lib.grm_mla_cuda_free_device_ptr(self.handle)
            self.handle = ctypes.c_void_p()


def python_mla_reference_route(
    rows: np.ndarray,
    query: np.ndarray,
    topk: int,
) -> list[int]:
    """Ground-truth reference matching RouterIndex::exact_mla_entry_score
    exactly (fp64 accumulation, node_id ASC tie-break, non-finite dropped)."""
    q = np.asarray(query, dtype=np.float64).reshape(-1)
    qnorm = float(np.sqrt(np.sum(q * q)))
    scored: list[tuple[float, int]] = []
    for node_id in range(rows.shape[0]):
        row = np.asarray(rows[node_id], dtype=np.float64).reshape(-1)
        dot = float(np.dot(q, row))
        rnorm = float(np.sqrt(np.sum(row * row)))
        score = dot / (qnorm * rnorm + 1.0e-8)
        if np.isfinite(score):
            scored.append((score, node_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [node_id for _, node_id in scored[:topk]]


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--node-count", type=int, default=512)
    p.add_argument("--queries", type=int, default=6)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nvcc", default="nvcc")
    p.add_argument("--keep-build", type=Path)
    p.add_argument("--out", type=Path, default=Path("/tmp/grm_mla_cuda_probe.json"))
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    nvcc = shutil.which(args.nvcc)
    if not nvcc:
        raise RuntimeError(f"nvcc not found: {args.nvcc}")

    rng = np.random.default_rng(args.seed)
    rows = rng.standard_normal((args.node_count, args.dim)).astype(np.float32)
    query_count = max(args.queries, args.warmup + 1)
    queries = rng.standard_normal((query_count, args.dim)).astype(np.float32)

    build_parent: tempfile.TemporaryDirectory[str] | None = None
    if args.keep_build:
        build_dir = args.keep_build
        build_dir.mkdir(parents=True, exist_ok=True)
    else:
        build_parent = tempfile.TemporaryDirectory(prefix="grm-mla-cuda-")
        build_dir = Path(build_parent.name)
    arena: CudaDeviceArena | None = None
    try:
        lib = build_probe(build_dir, nvcc)
        probe = CudaProbe(lib)
        arena = probe.create_arena(rows)
        topk_ids, device_ms, route_wall_ms = arena.route_topk(
            queries, warmup=args.warmup, topk=args.topk)
    finally:
        if arena is not None:
            arena.close()
        if build_parent is not None:
            build_parent.cleanup()

    mismatches = []
    for i, q in enumerate(queries):
        expected = python_mla_reference_route(rows, q, args.topk)
        got = [int(x) for x in topk_ids[i].tolist()]
        if got != expected:
            mismatches.append({"query": i, "python": expected, "cuda": got})

    result = {
        "schema": "grm-mla-cuda-cublas-probe-v1",
        "nodes": int(rows.shape[0]),
        "dim": int(args.dim),
        "topk": int(args.topk),
        "queries": int(query_count),
        "device_route_ms_total": float(device_ms),
        "route_wall_ms": float(route_wall_ms),
        "setup_wall_ms": float(arena.setup_wall_ms if arena else 0.0),
        "parity": not mismatches,
        "mismatches": mismatches[:5],
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
