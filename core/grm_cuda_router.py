"""Optional CUDA/cuBLAS sidecar for GQA router banks.

The dependency-free C++ runtime remains the default router. This module wraps
the CUDA probe as an explicit sidecar so callers can opt into device-resident
GQA route banks without changing the CPU C ABI path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile

import numpy as np


class CudaRouterUnavailable(RuntimeError):
    """Raised when the optional CUDA route sidecar cannot be built or loaded."""


@dataclass(frozen=True)
class CudaRouteReceipt:
    device_route_ms_total: float
    device_route_ms_per_query: float
    route_wall_ms: float
    route_wall_ms_first: float
    route_wall_ms_min: float
    route_wall_ms_runs: tuple[float, ...]
    device_route_ms_runs: tuple[float, ...]
    measured_queries: int
    route_repeats: int


def validate_gqa_route_bank(
    keys: np.ndarray,
    node_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize and validate a dense GQA route bank.

    Shape is `(nodes, kv_heads, key_tokens, head_dim)`. Node IDs map CUDA top-k
    row indices back to GRM node IDs; default IDs are dense row indices.
    """
    keys_np = np.ascontiguousarray(keys, dtype=np.float32)
    if keys_np.ndim != 4:
        raise ValueError(
            "GQA CUDA route bank must have shape "
            "(nodes, kv_heads, key_tokens, head_dim)")
    if any(int(dim) <= 0 for dim in keys_np.shape):
        raise ValueError("GQA CUDA route bank dimensions must be nonzero")
    if node_ids is None:
        ids_np = np.arange(keys_np.shape[0], dtype=np.uint64)
    else:
        ids_np = np.ascontiguousarray(node_ids, dtype=np.uint64).reshape(-1)
    if ids_np.shape[0] != keys_np.shape[0]:
        raise ValueError("node_ids length must match GQA route-bank nodes")
    return keys_np, ids_np


def _load_probe_backend():
    # Kept as a lazy import so importing this runtime-facing helper never
    # requires nvcc/CUDA. The probe remains the source of truth until the sidecar
    # source is promoted into a compiled runtime artifact.
    from scripts.grm_gqa_cuda_probe import CudaProbe, build_probe

    return CudaProbe, build_probe


class CudaGQARouteSidecar:
    """Build/load the optional CUDA GQA router sidecar library."""

    def __init__(
        self,
        lib_path: Path,
        *,
        build_temp: tempfile.TemporaryDirectory[str] | None = None,
    ):
        CudaProbe, _ = _load_probe_backend()
        self._build_temp = build_temp
        self.lib_path = Path(lib_path)
        self._probe = CudaProbe(self.lib_path)

    @classmethod
    def build(
        cls,
        build_dir: Path | None = None,
        *,
        nvcc: str = "nvcc",
    ) -> "CudaGQARouteSidecar":
        nvcc_path = shutil.which(nvcc)
        if not nvcc_path:
            raise CudaRouterUnavailable(f"nvcc not found: {nvcc}")
        _, build_probe = _load_probe_backend()
        build_temp: tempfile.TemporaryDirectory[str] | None = None
        if build_dir is None:
            build_temp = tempfile.TemporaryDirectory(prefix="grm-gqa-cuda-sidecar-")
            build_dir = Path(build_temp.name)
        else:
            build_dir = Path(build_dir)
            build_dir.mkdir(parents=True, exist_ok=True)
        try:
            lib_path = build_probe(build_dir, nvcc_path)
        except Exception as exc:  # pragma: no cover - exact nvcc errors vary.
            if build_temp is not None:
                build_temp.cleanup()
            raise CudaRouterUnavailable(str(exc)) from exc
        return cls(lib_path, build_temp=build_temp)

    def create_bank(
        self,
        keys: np.ndarray,
        node_ids: np.ndarray | None = None,
    ) -> "CudaGQARouteBank":
        keys_np, ids_np = validate_gqa_route_bank(keys, node_ids)
        arena = self._probe.create_arena(keys_np)
        return CudaGQARouteBank(
            arena,
            ids_np,
            setup_wall_ms=float(arena.setup_wall_ms),
        )

    def close(self) -> None:
        if self._build_temp is not None:
            self._build_temp.cleanup()
            self._build_temp = None

    def __del__(self) -> None:
        self.close()


class CudaGQARouteBank:
    """Device-resident CUDA route bank with persistent route scratch buffers."""

    def __init__(self, arena, node_ids: np.ndarray, *, setup_wall_ms: float):
        self._arena = arena
        self.node_ids = np.ascontiguousarray(node_ids, dtype=np.uint64)
        self.setup_wall_ms = float(setup_wall_ms)

    def close(self) -> None:
        if self._arena is not None:
            self._arena.close()
            self._arena = None

    def __del__(self) -> None:
        self.close()

    def route_topk(
        self,
        queries: np.ndarray,
        *,
        topk: int,
        warmup: int = 0,
        route_repeats: int = 1,
    ) -> tuple[np.ndarray, CudaRouteReceipt]:
        if self._arena is None:
            raise RuntimeError("CUDA GQA route bank is closed")
        queries_np = np.ascontiguousarray(queries, dtype=np.float32)
        if queries_np.ndim == 3:
            queries_np = queries_np[None, ...]
        if queries_np.ndim != 4:
            raise ValueError(
                "GQA CUDA queries must have shape "
                "(queries, heads, tokens, dim) or (heads, tokens, dim)")
        if int(topk) <= 0:
            raise ValueError("topk must be positive")
        repeats = max(1, int(route_repeats))
        route_walls: list[float] = []
        device_times: list[float] = []
        topk_rows = None
        for _ in range(repeats):
            topk_rows, device_ms, route_wall_ms = self._arena.route_topk(
                queries_np,
                warmup=int(warmup),
                topk=int(topk),
            )
            route_walls.append(float(route_wall_ms))
            device_times.append(float(device_ms))
        if topk_rows is None:
            raise RuntimeError("CUDA GQA route did not run")
        if np.any(topk_rows >= self.node_ids.shape[0]):
            raise RuntimeError("CUDA GQA route returned an out-of-range row ID")
        mapped = self.node_ids[topk_rows]
        measured = max(1, int(queries_np.shape[0]) - int(warmup))
        receipt = CudaRouteReceipt(
            device_route_ms_total=device_times[-1],
            device_route_ms_per_query=device_times[-1] / measured,
            route_wall_ms=route_walls[-1],
            route_wall_ms_first=route_walls[0],
            route_wall_ms_min=float(np.min(route_walls)),
            route_wall_ms_runs=tuple(route_walls),
            device_route_ms_runs=tuple(device_times),
            measured_queries=measured,
            route_repeats=repeats,
        )
        return mapped, receipt
