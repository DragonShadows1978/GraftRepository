#!/usr/bin/env python3
"""Baseline the current GRM router scan before GEMV replacement work.

The benchmark intentionally measures the current native RouterIndex route path
and a Python per-node fallback with the same scoring law. Centroid samples are
harvested from the repository text by feature hashing real source/doc lines,
then repeated deterministically to larger node counts.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIRS = ("docs", "core", "cpp", "tests")
TEXT_SUFFIXES = {".md", ".py", ".cpp", ".hpp", ".h", ".txt"}


def _tokenize_ascii(text: str) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    for ch in text.lower():
        if "a" <= ch <= "z" or "0" <= ch <= "9" or ch == "_":
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur.clear()
    if cur:
        out.append("".join(cur))
    return out


def _stable_u64(text: str, *, person: bytes = b"grmroute") -> int:
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=8, person=person)
    return int.from_bytes(h.digest(), "little", signed=False)


def hashed_centroid(text: str, dim: int) -> np.ndarray | None:
    """Feature-hash one text row into a normalized fp32 route vector."""
    tokens = _tokenize_ascii(text)
    if not tokens:
        return None
    v = np.zeros(dim, dtype=np.float32)
    for tok in tokens:
        hv = _stable_u64(tok)
        idx = hv % dim
        sign = 1.0 if ((hv >> 63) & 1) == 0 else -1.0
        weight = 1.0 + min(len(tok), 16) / 16.0
        v[idx] += np.float32(sign * weight)
    norm = float(np.linalg.norm(v))
    if not math.isfinite(norm) or norm <= 0.0:
        return None
    v /= np.float32(norm)
    return v


def _iter_source_lines(root: Path, max_files: int) -> Iterable[tuple[Path, str]]:
    seen = 0
    for rel in DEFAULT_SOURCE_DIRS:
        base = root / rel
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if seen >= max_files:
                return
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            seen += 1
            for line in text.splitlines():
                line = line.strip()
                if len(line) >= 24:
                    yield path, line


def harvest_centroid_bank(
    root: Path,
    *,
    dim: int,
    max_vectors: int,
    max_files: int,
) -> tuple[np.ndarray, dict]:
    rows: list[np.ndarray] = []
    files: set[str] = set()
    token_counts: list[int] = []
    for path, line in _iter_source_lines(root, max_files=max_files):
        v = hashed_centroid(line, dim)
        if v is None:
            continue
        rows.append(v)
        files.add(str(path.relative_to(root)))
        token_counts.append(len(_tokenize_ascii(line)))
        if len(rows) >= max_vectors:
            break
    if not rows:
        raise RuntimeError(f"no source text vectors harvested from {root}")
    bank = np.stack(rows).astype(np.float32, copy=False)
    stats = {
        "source_root": str(root),
        "source_files": len(files),
        "centroid_bank_rows": int(bank.shape[0]),
        "dim": int(dim),
        "mean_tokens_per_row": float(statistics.fmean(token_counts)),
        "mean_l1": float(np.mean(np.sum(np.abs(bank), axis=1))),
        "mean_active_dims": float(np.mean(np.count_nonzero(bank, axis=1))),
    }
    return bank, stats


def make_route_matrix(bank: np.ndarray, node_count: int) -> np.ndarray:
    """Repeat and lightly permute harvested rows without gaussian sampling."""
    dim = int(bank.shape[1])
    out = np.empty((node_count, dim), dtype=np.float32)
    for i in range(node_count):
        src = bank[(i * 1315423911 + i // max(1, bank.shape[0])) % bank.shape[0]]
        shift = (i * 17) % dim
        row = np.roll(src, shift)
        if i % 11 == 0:
            row = (row + bank[(i * 31 + 7) % bank.shape[0]]) * np.float32(0.5)
        row = np.array(row, dtype=np.float32, copy=True)
        row[(i * 37 + 3) % dim] += np.float32(1.0e-3 * ((i % 13) + 1) / 13.0)
        row[(i * 53 + 11) % dim] -= np.float32(5.0e-4 * ((i % 7) + 1) / 7.0)
        norm = float(np.linalg.norm(row))
        out[i] = row if norm <= 0.0 else row / np.float32(norm)
    return out


def make_queries(bank: np.ndarray, query_count: int) -> np.ndarray:
    qs = []
    for i in range(query_count):
        q = np.array(bank[(i * 97 + 13) % bank.shape[0]], dtype=np.float32)
        if i % 3 == 0:
            q = np.roll(q, (i * 5) % bank.shape[1])
        norm = float(np.linalg.norm(q))
        qs.append(q if norm <= 0.0 else q / np.float32(norm))
    return np.stack(qs).astype(np.float32, copy=False)


def lexical_keys_for_node(i: int) -> tuple[str, ...]:
    keys = [f"node_{i:08x}", f"family_{i % 127:03d}"]
    if i % 17 == 0:
        keys.append(f"rare_{i % 4096:04x}")
    return tuple(keys)


def lexical_keys_for_query(i: int, node_count: int | None = None) -> tuple[str, ...]:
    if node_count is None or node_count <= 0:
        target = i
    else:
        target = (i * 37 + 11) % node_count
    return (
        f"node_{target:08x}",
        f"family_{target % 127:03d}",
        f"rare_{target % 4096:04x}",
    )


def lexical_blob(keys: Iterable[str]) -> bytes:
    return "\n".join(keys).encode("ascii")


def python_route_scan(
    routes: np.ndarray,
    query: np.ndarray,
    lexical: tuple[str, ...],
    topk: int,
) -> list[int]:
    scored: list[tuple[float, int]] = []
    for node_id, key in enumerate(routes):
        if key.shape != query.shape:
            score = 0.0
        else:
            dot = 0.0
            qnorm = 0.0
            knorm = 0.0
            for qv, kv in zip(query, key):
                qf = float(qv)
                kf = float(kv)
                dot += qf * kf
                qnorm += qf * qf
                knorm += kf * kf
            score = dot / ((math.sqrt(qnorm) * math.sqrt(knorm)) + 1.0e-8)
        if lexical:
            node_keys = lexical_keys_for_node(node_id)
            hits = sum(1 for q in lexical if q in node_keys)
            score += hits / len(lexical)
        if math.isfinite(score):
            scored.append((score, node_id))
    scored.sort(key=lambda item: -item[0])
    return [node_id for _, node_id in scored[:topk]]


def build_native_lib(out_dir: Path, *, cxx: str) -> Path:
    lib = out_dir / "libgrm_runtime.so"
    subprocess.run(
        [
            cxx,
            "-std=c++17",
            "-O3",
            "-shared",
            "-fPIC",
            "-I",
            str(ROOT / "cpp"),
            str(ROOT / "cpp" / "grm_runtime.cpp"),
            "-o",
            str(lib),
        ],
        check=True,
    )
    return lib


class NativeRouter:
    def __init__(self, lib_path: Path):
        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.grm_store_create_mla.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.grm_store_create_mla.restype = ctypes.c_void_p
        self.lib.grm_store_destroy.argtypes = [ctypes.c_void_p]
        self.lib.grm_store_destroy.restype = None
        self.lib.grm_store_add_node.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self.lib.grm_store_add_node.restype = ctypes.c_int
        self.lib.grm_store_set_route.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint64,
            ctypes.c_char_p,
        ]
        self.lib.grm_store_set_route.restype = ctypes.c_int
        self.lib.grm_store_route.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint64,
            ctypes.c_char_p,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self.lib.grm_store_route.restype = ctypes.c_int
        self.handle = self.lib.grm_store_create_mla(
            b"RouterBaseline", 1, 2048, 576, 0, 512, 64)
        if not self.handle:
            raise RuntimeError("failed to create native router store")

    def close(self) -> None:
        if self.handle:
            self.lib.grm_store_destroy(self.handle)
            self.handle = None

    def __enter__(self) -> "NativeRouter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def add_routes(self, routes: np.ndarray) -> None:
        empty_payload = ctypes.POINTER(ctypes.c_uint8)()
        for node_id, key in enumerate(routes):
            out = ctypes.c_uint64()
            rc = self.lib.grm_store_add_node(
                self.handle,
                f"baseline node {node_id}".encode("ascii"),
                1,
                empty_payload,
                0,
                ctypes.byref(out),
            )
            if rc != 0 or out.value != node_id:
                raise RuntimeError(f"native add_node failed at {node_id}")
            arr = np.ascontiguousarray(key, dtype=np.float32)
            rc = self.lib.grm_store_set_route(
                self.handle,
                ctypes.c_uint64(node_id),
                arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_uint64(arr.size),
                lexical_blob(lexical_keys_for_node(node_id)),
            )
            if rc != 0:
                raise RuntimeError(f"native set_route failed at {node_id}")

    def route(self, query: np.ndarray, lexical: tuple[str, ...], topk: int) -> list[int]:
        q = np.ascontiguousarray(query, dtype=np.float32)
        out = (ctypes.c_uint64 * topk)()
        count = ctypes.c_uint64()
        rc = self.lib.grm_store_route(
            self.handle,
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint64(q.size),
            lexical_blob(lexical),
            ctypes.c_uint64(topk),
            out,
            ctypes.c_uint64(topk),
            ctypes.byref(count),
        )
        if rc != 0:
            raise RuntimeError("native route failed")
        return [int(out[i]) for i in range(int(count.value))]


def _median_ms(values_ns: list[int]) -> float:
    return statistics.median(values_ns) / 1.0e6


def _p95_ms(values_ns: list[int]) -> float:
    if len(values_ns) < 2:
        return values_ns[0] / 1.0e6
    return statistics.quantiles(values_ns, n=20, method="inclusive")[18] / 1.0e6


def benchmark_count(
    *,
    routes: np.ndarray,
    queries: np.ndarray,
    lib_path: Path,
    topk: int,
    warmup: int,
) -> dict:
    py_ns: list[int] = []
    native_ns: list[int] = []
    mismatches: list[dict] = []
    with NativeRouter(lib_path) as native:
        build_t0 = time.perf_counter_ns()
        native.add_routes(routes)
        build_ms = (time.perf_counter_ns() - build_t0) / 1.0e6
        for i, query in enumerate(queries):
            lexical = lexical_keys_for_query(i, routes.shape[0])
            py = python_route_scan(routes, query, lexical, topk)
            got = native.route(query, lexical, topk)
            if got != py:
                mismatches.append({"query": i, "python": py, "native": got})
        for i, query in enumerate(queries):
            lexical = lexical_keys_for_query(i, routes.shape[0])
            if i < warmup:
                python_route_scan(routes, query, lexical, topk)
                native.route(query, lexical, topk)
                continue
            t0 = time.perf_counter_ns()
            python_route_scan(routes, query, lexical, topk)
            py_ns.append(time.perf_counter_ns() - t0)
            t0 = time.perf_counter_ns()
            native.route(query, lexical, topk)
            native_ns.append(time.perf_counter_ns() - t0)
    return {
        "nodes": int(routes.shape[0]),
        "dim": int(routes.shape[1]),
        "topk": int(topk),
        "queries": int(max(0, len(queries) - warmup)),
        "native_build_ms": build_ms,
        "native_route_ms_p50": _median_ms(native_ns),
        "native_route_ms_p95": _p95_ms(native_ns),
        "python_route_ms_p50": _median_ms(py_ns),
        "python_route_ms_p95": _p95_ms(py_ns),
        "parity": not mismatches,
        "mismatches": mismatches[:5],
    }


def write_markdown(result: dict, path: Path) -> None:
    lines = [
        "# GRM Router Baseline",
        "",
        "Current native `RouterIndex` per-node scan and Python fallback baseline.",
        "",
        "## Harvest",
        "",
        f"- source_root: `{result['harvest']['source_root']}`",
        f"- source_files: {result['harvest']['source_files']}",
        f"- centroid_bank_rows: {result['harvest']['centroid_bank_rows']}",
        f"- dim: {result['harvest']['dim']}",
        f"- mean_tokens_per_row: {result['harvest']['mean_tokens_per_row']:.2f}",
        f"- mean_active_dims: {result['harvest']['mean_active_dims']:.2f}",
        "",
        "## Results",
        "",
        "| nodes | dim | parity | native p50 ms | native p95 ms | python p50 ms | python p95 ms | native build ms |",
        "| ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["results"]:
        lines.append(
            f"| {row['nodes']} | {row['dim']} | {row['parity']} | "
            f"{row['native_route_ms_p50']:.4f} | "
            f"{row['native_route_ms_p95']:.4f} | "
            f"{row['python_route_ms_p50']:.4f} | "
            f"{row['python_route_ms_p95']:.4f} | "
            f"{row['native_build_ms']:.2f} |"
        )
    lines.extend([
        "",
        "## Full Curve Command",
        "",
        "```bash",
        "python3 scripts/grm_router_baseline.py "
        "--node-counts 1000 10000 100000 1000000 "
        "--queries 24 --warmup 4 --dim 512 "
        "--out /tmp/grm_router_baseline_full.json "
        "--markdown-out docs/ROUTER_BASELINE.md",
        "```",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, default=ROOT)
    p.add_argument("--node-counts", nargs="+", type=int,
                   default=[1000, 10000])
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--queries", type=int, default=16)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--max-vectors", type=int, default=8192)
    p.add_argument("--max-files", type=int, default=512)
    p.add_argument("--lib", type=Path)
    p.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    p.add_argument("--out", type=Path, default=Path("/tmp/grm_router_baseline.json"))
    p.add_argument("--markdown-out", type=Path)
    p.add_argument("--smoke", action="store_true",
                   help="small deterministic run for CI/sanity checks")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.smoke:
        args.node_counts = [32, 96]
        args.dim = min(args.dim, 32)
        args.queries = min(args.queries, 6)
        args.warmup = min(args.warmup, 1)
        args.max_vectors = min(args.max_vectors, 128)
        args.max_files = min(args.max_files, 64)
    if args.warmup >= args.queries:
        raise SystemExit("--warmup must be lower than --queries")
    bank, harvest = harvest_centroid_bank(
        args.source_root,
        dim=args.dim,
        max_vectors=args.max_vectors,
        max_files=args.max_files,
    )
    queries = make_queries(bank, args.queries)
    with tempfile.TemporaryDirectory(prefix="grm_router_baseline_") as td:
        lib_path = args.lib or build_native_lib(Path(td), cxx=args.cxx)
        results = []
        for node_count in args.node_counts:
            routes = make_route_matrix(bank, int(node_count))
            results.append(benchmark_count(
                routes=routes,
                queries=queries,
                lib_path=lib_path,
                topk=args.topk,
                warmup=args.warmup,
            ))
    output = {
        "schema": "grm-router-baseline-v1",
        "repo": str(ROOT),
        "harvest": harvest,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(output, args.markdown_out)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
