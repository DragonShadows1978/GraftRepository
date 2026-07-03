#!/usr/bin/env python3
"""Benchmark native GRM GQA raw q.k routing.

By default the input banks are deterministic feature-hash vectors from real
repository source/doc lines, reshaped into GQA key/query tensors. The benchmark
can also read Qwen3.5 capture shards in-place and route their stored K banks
without moving or modifying generated graft artifacts.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
from pathlib import Path
import sys
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_router_baseline as baseline  # noqa: E402


PRESETS = {
    "qwen35-2b-attn": {
        "model_type": "Qwen35_2B_TC",
        "num_layers": 24,
        "hidden_dim": 2048,
        "vals_per_tok_layer": 1024,
        "query_heads": 8,
        "kv_heads": 2,
        "head_dim": 256,
    },
    "qwen3-4b": {
        "model_type": "Qwen3_TC",
        "num_layers": 36,
        "hidden_dim": 2560,
        "vals_per_tok_layer": 2048,
        "query_heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
    },
    "qwen35-9b-attn": {
        "model_type": "Qwen35_TC",
        "num_layers": 32,
        "hidden_dim": 4096,
        "vals_per_tok_layer": 2048,
        "query_heads": 16,
        "kv_heads": 4,
        "head_dim": 256,
    },
}


def _normalize_last_dim(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32).copy()
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    np.divide(out, norms, out=out, where=norms > 0)
    return out.astype(np.float32, copy=False)


def make_gqa_routes(
    bank: np.ndarray,
    *,
    node_count: int,
    keys_per_node: int,
    kv_heads: int,
    key_tokens: int,
    head_dim: int,
) -> np.ndarray:
    flat = baseline.make_route_matrix(bank, node_count * keys_per_node)
    shaped = flat.reshape(
        node_count, keys_per_node, kv_heads, key_tokens, head_dim)
    return _normalize_last_dim(shaped)


def make_gqa_queries(
    bank: np.ndarray,
    *,
    query_count: int,
    query_heads: int,
    query_tokens: int,
    head_dim: int,
) -> np.ndarray:
    flat = baseline.make_route_matrix(bank, query_count)
    shaped = flat.reshape(query_count, query_heads, query_tokens, head_dim)
    return _normalize_last_dim(shaped)


def _capture_paths(capture_dir: Path, role: str, limit: int) -> list[Path]:
    paths = sorted(capture_dir.glob(f"{role}_*.npz"))
    return paths[:limit] if limit > 0 else paths


def load_capture_routes(
    capture_dir: Path,
    *,
    role: str,
    layer: int,
    limit: int,
    token_limit: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[np.ndarray, dict]:
    routes: list[np.ndarray] = []
    used: list[str] = []
    skipped_shape = 0
    target_tokens: int | None = None
    key_name = f"l{int(layer)}_k"
    for path in _capture_paths(capture_dir, role, limit):
        with np.load(path, allow_pickle=False) as shard:
            if key_name not in shard:
                continue
            arr = np.asarray(shard[key_name], dtype=np.float32)
        if arr.ndim != 4 or arr.shape[0] != 1:
            continue
        key = arr[0]
        if key.shape[0] != kv_heads or key.shape[2] != head_dim:
            raise RuntimeError(
                f"{path} {key_name} shape {key.shape} does not match "
                f"kv_heads={kv_heads}, head_dim={head_dim}")
        if token_limit > 0:
            if key.shape[1] < token_limit:
                skipped_shape += 1
                continue
            key = key[:, :token_limit, :]
        elif target_tokens is None:
            target_tokens = int(key.shape[1])
        elif key.shape[1] != target_tokens:
            skipped_shape += 1
            continue
        if key.shape[1] == 0:
            continue
        routes.append(_normalize_last_dim(key)[None, :, :, :])
        used.append(str(path))
    if not routes:
        raise RuntimeError(
            f"no {role} capture K banks for layer {layer} in {capture_dir}")
    stacked = np.stack(routes).astype(np.float32, copy=False)
    stats = {
        "capture_dir": str(capture_dir),
        "role": role,
        "layer": int(layer),
        "shards": len(used),
        "key_name": key_name,
        "key_tokens": int(stacked.shape[3]),
        "kv_heads": int(stacked.shape[2]),
        "head_dim": int(stacked.shape[4]),
        "skipped_shape": int(skipped_shape),
        "first_shards": used[:5],
    }
    return stacked, stats


def make_capture_queries(
    routes: np.ndarray,
    *,
    query_count: int,
    query_heads: int,
    query_tokens: int,
) -> np.ndarray:
    kv_heads = int(routes.shape[2])
    key_tokens = int(routes.shape[3])
    if query_heads % kv_heads != 0:
        raise RuntimeError(
            f"query_heads={query_heads} must be divisible by kv_heads={kv_heads}")
    out: list[np.ndarray] = []
    repeat = query_heads // kv_heads
    for i in range(query_count):
        key = routes[(i * 37 + 11) % routes.shape[0], 0]
        start = (i * 17) % max(1, key_tokens)
        idx = (np.arange(query_tokens) + start) % key_tokens
        q = np.repeat(key[:, idx, :], repeat, axis=0)
        out.append(q)
    return _normalize_last_dim(np.stack(out))


def gqa_raw_score(query: np.ndarray, key: np.ndarray) -> float:
    query_heads, _, head_dim = query.shape
    kv_heads = key.shape[0]
    if query_heads == 0 or kv_heads == 0 or query_heads % kv_heads != 0:
        return 0.0
    repeated = np.repeat(key, query_heads // kv_heads, axis=0)
    scores = np.einsum("hqd,hkd->hqk", query, repeated) / math.sqrt(head_dim)
    return float(np.abs(scores).max(axis=(1, 2)).mean())


def python_gqa_route_scan(
    routes: np.ndarray,
    query: np.ndarray,
    lexical: tuple[str, ...],
    topk: int,
) -> list[int]:
    rows: list[tuple[float, int, int]] = []
    max_abs = 0.0
    for node_id, keys in enumerate(routes):
        raw = max(gqa_raw_score(query, key) for key in keys)
        if not math.isfinite(raw):
            continue
        node_keys = baseline.lexical_keys_for_node(node_id)
        hits = sum(1 for q in lexical if q in node_keys)
        max_abs = max(max_abs, abs(raw))
        rows.append((raw, node_id, hits))
    norm = max_abs + 1.0e-8
    scored: list[tuple[float, int]] = []
    for raw, node_id, hits in rows:
        lex = 0.0 if not lexical else hits / len(lexical)
        scored.append(((raw / norm) + lex, node_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [node_id for _, node_id in scored[:topk]]


class NativeGqaRouter:
    def __init__(
        self,
        lib_path: Path,
        *,
        model_type: str,
        num_layers: int,
        hidden_dim: int,
        vals_per_tok_layer: int,
        route_layer: int,
        kv_heads: int,
        head_dim: int,
    ):
        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.grm_store_create_gqa.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.grm_store_create_gqa.restype = ctypes.c_void_p
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
        self.lib.grm_store_set_route_list.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint64,
            ctypes.c_char_p,
        ]
        self.lib.grm_store_set_route_list.restype = ctypes.c_int
        self.lib.grm_store_route_gqa.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self.lib.grm_store_route_gqa.restype = ctypes.c_int
        self.handle = self.lib.grm_store_create_gqa(
            model_type.encode("ascii"),
            int(num_layers),
            int(hidden_dim),
            int(vals_per_tok_layer),
            int(route_layer),
            int(kv_heads),
            int(head_dim),
        )
        if not self.handle:
            raise RuntimeError("failed to create native GQA router store")

    def close(self) -> None:
        if self.handle:
            self.lib.grm_store_destroy(self.handle)
            self.handle = None

    def __enter__(self) -> "NativeGqaRouter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def add_routes(self, routes: np.ndarray) -> None:
        empty_payload = ctypes.POINTER(ctypes.c_uint8)()
        for node_id, keys in enumerate(routes):
            out = ctypes.c_uint64()
            rc = self.lib.grm_store_add_node(
                self.handle,
                f"gqa route node {node_id}".encode("ascii"),
                1,
                empty_payload,
                0,
                ctypes.byref(out),
            )
            if rc != 0 or out.value != node_id:
                raise RuntimeError(f"native add_node failed at {node_id}")
            flat = np.ascontiguousarray(keys.reshape(-1), dtype=np.float32)
            values = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            key_size = int(flat.size // keys.shape[0])
            offsets = [i * key_size for i in range(keys.shape[0] + 1)]
            offsets_t = ctypes.c_uint64 * len(offsets)
            offsets_arr = offsets_t(*offsets)
            rc = self.lib.grm_store_set_route_list(
                self.handle,
                ctypes.c_uint64(node_id),
                values,
                ctypes.c_uint64(flat.size),
                offsets_arr,
                ctypes.c_uint64(keys.shape[0]),
                baseline.lexical_blob(baseline.lexical_keys_for_node(node_id)),
            )
            if rc != 0:
                raise RuntimeError(f"native set_route_list failed at {node_id}")

    def route(
        self,
        query: np.ndarray,
        lexical: tuple[str, ...],
        topk: int,
    ) -> list[int]:
        q = np.ascontiguousarray(query, dtype=np.float32)
        out = (ctypes.c_uint64 * max(0, topk))()
        count = ctypes.c_uint64()
        rc = self.lib.grm_store_route_gqa(
            self.handle,
            q.reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint64(q.shape[0]),
            ctypes.c_uint64(q.shape[1]),
            ctypes.c_uint64(q.shape[2]),
            baseline.lexical_blob(lexical),
            b"",
            b"",
            b"",
            b"",
            ctypes.c_uint64(topk),
            out,
            ctypes.c_uint64(topk),
            ctypes.byref(count),
        )
        if rc != 0:
            raise RuntimeError("native route_gqa failed")
        return [int(out[i]) for i in range(int(count.value))]


def benchmark_count(
    *,
    routes: np.ndarray,
    queries: np.ndarray,
    lib_path: Path,
    preset: dict,
    route_layer: int,
    topk: int,
    warmup: int,
    native_only: bool,
    use_lexical: bool,
) -> dict:
    py_ns: list[int] = []
    native_ns: list[int] = []
    mismatches: list[dict] = []
    with NativeGqaRouter(
        lib_path,
        model_type=preset["model_type"],
        num_layers=preset["num_layers"],
        hidden_dim=preset["hidden_dim"],
        vals_per_tok_layer=preset["vals_per_tok_layer"],
        route_layer=route_layer,
        kv_heads=preset["kv_heads"],
        head_dim=preset["head_dim"],
    ) as native:
        build_t0 = time.perf_counter_ns()
        native.add_routes(routes)
        build_ms = (time.perf_counter_ns() - build_t0) / 1.0e6
        if not native_only:
            for i, query in enumerate(queries):
                lexical = (baseline.lexical_keys_for_query(i, routes.shape[0])
                           if use_lexical else ())
                py = python_gqa_route_scan(routes, query, lexical, topk)
                got = native.route(query, lexical, topk)
                if got != py:
                    mismatches.append({"query": i, "python": py, "native": got})
        for i, query in enumerate(queries):
            lexical = (baseline.lexical_keys_for_query(i, routes.shape[0])
                       if use_lexical else ())
            if i < warmup:
                if not native_only:
                    python_gqa_route_scan(routes, query, lexical, topk)
                native.route(query, lexical, topk)
                continue
            if not native_only:
                t0 = time.perf_counter_ns()
                python_gqa_route_scan(routes, query, lexical, topk)
                py_ns.append(time.perf_counter_ns() - t0)
            t0 = time.perf_counter_ns()
            native.route(query, lexical, topk)
            native_ns.append(time.perf_counter_ns() - t0)
    return {
        "nodes": int(routes.shape[0]),
        "keys_per_node": int(routes.shape[1]),
        "kv_heads": int(routes.shape[2]),
        "key_tokens": int(routes.shape[3]),
        "query_heads": int(queries.shape[1]),
        "query_tokens": int(queries.shape[2]),
        "head_dim": int(routes.shape[4]),
        "topk": int(topk),
        "queries": int(max(0, len(queries) - warmup)),
        "native_build_ms": build_ms,
        "native_route_ms_p50": baseline._median_ms(native_ns),
        "native_route_ms_p95": baseline._p95_ms(native_ns),
        "python_route_ms_p50": None if native_only else baseline._median_ms(py_ns),
        "python_route_ms_p95": None if native_only else baseline._p95_ms(py_ns),
        "parity": None if native_only else not mismatches,
        "parity_reference": None if native_only else "python_gqa_raw",
        "lexical": bool(use_lexical),
        "mismatches": mismatches[:5],
    }


def progress_record(
    *,
    result: dict,
    completed: int,
    total: int,
    preset_name: str,
    route_source: str,
    openmp: bool,
    native_only: bool,
    route_stats: dict,
    query_stats: dict,
) -> dict:
    return {
        "schema": "grm-gqa-router-benchmark-progress-v1",
        "repo": str(ROOT),
        "preset": preset_name,
        "route_source": route_source,
        "openmp": bool(openmp),
        "native_only": bool(native_only),
        "completed": int(completed),
        "total": int(total),
        "harvest": {
            "route": route_stats,
            "query": query_stats,
        },
        "shape": {
            "query_heads": int(result["query_heads"]),
            "kv_heads": int(result["kv_heads"]),
            "head_dim": int(result["head_dim"]),
            "query_tokens": int(result["query_tokens"]),
            "key_tokens": int(result["key_tokens"]),
            "keys_per_node": int(result["keys_per_node"]),
        },
        "result": result,
    }


def append_progress(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
        fh.flush()


def print_progress(record: dict) -> None:
    row = record["result"]
    parity = "n/a" if row["parity"] is None else str(row["parity"]).lower()
    print(
        "[grm_gqa] "
        f"{record['completed']}/{record['total']} "
        f"nodes={row['nodes']} "
        f"native_p50_ms={row['native_route_ms_p50']:.4f} "
        f"python_p50_ms={row['python_route_ms_p50']} "
        f"parity={parity}",
        file=sys.stderr,
        flush=True,
    )


def write_markdown(result: dict, path: Path) -> None:
    lines = [
        "# GRM GQA Router Benchmark",
        "",
        "Native GQA raw q.k route benchmark over harvested route banks.",
        "",
        "## Shape",
        "",
        f"- preset: `{result['preset']}`",
        f"- query_heads: {result['shape']['query_heads']}",
        f"- kv_heads: {result['shape']['kv_heads']}",
        f"- head_dim: {result['shape']['head_dim']}",
        f"- query_tokens: {result['shape']['query_tokens']}",
        f"- key_tokens: {result['shape']['key_tokens']}",
        f"- keys_per_node: {result['shape']['keys_per_node']}",
        f"- route_source: `{result['route_source']}`",
        "",
        "## Results",
        "",
        "| nodes | parity | native p50 ms | native p95 ms | "
        "python p50 ms | python p95 ms | native build ms |",
        "| ---: | :---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["results"]:
        parity = "n/a" if row["parity"] is None else str(row["parity"]).lower()
        py50 = (
            "n/a" if row["python_route_ms_p50"] is None
            else f"{row['python_route_ms_p50']:.4f}")
        py95 = (
            "n/a" if row["python_route_ms_p95"] is None
            else f"{row['python_route_ms_p95']:.4f}")
        lines.append(
            f"| {row['nodes']} | {parity} | "
            f"{row['native_route_ms_p50']:.4f} | "
            f"{row['native_route_ms_p95']:.4f} | {py50} | {py95} | "
            f"{row['native_build_ms']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, default=ROOT)
    p.add_argument("--preset", choices=sorted(PRESETS), default="qwen35-2b-attn")
    p.add_argument("--node-counts", nargs="+", type=int)
    p.add_argument("--queries", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--query-tokens", type=int, default=4)
    p.add_argument("--key-tokens", type=int, default=1)
    p.add_argument("--keys-per-node", type=int, default=1)
    p.add_argument("--route-layer", type=int, default=0)
    p.add_argument("--max-vectors", type=int, default=8192)
    p.add_argument("--max-files", type=int, default=512)
    p.add_argument("--cxx", default="g++")
    p.add_argument("--openmp", action="store_true")
    p.add_argument("--native-only", action="store_true")
    p.add_argument("--no-lexical", action="store_true")
    p.add_argument("--capture-dir", type=Path,
                   help="read Qwen capture shard K banks from this directory")
    p.add_argument("--capture-role", default="source")
    p.add_argument("--capture-layer", type=int, default=3)
    p.add_argument("--capture-limit", type=int, default=128)
    p.add_argument("--capture-token-limit", type=int, default=0,
                   help="limit K tokens per captured shard; 0 keeps all tokens")
    p.add_argument("--out", type=Path,
                   default=Path("/tmp/grm_gqa_router_benchmark.json"))
    p.add_argument("--markdown-out", type=Path)
    p.add_argument("--progress-out", type=Path,
                   help="write one JSONL checkpoint row after each node count")
    p.add_argument("--progress", action="store_true",
                   help="print one-line progress updates to stderr")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    preset = dict(PRESETS[args.preset])
    if args.smoke:
        preset.update({
            "model_type": "GqaRouterSmoke",
            "num_layers": 2,
            "hidden_dim": 64,
            "vals_per_tok_layer": 64,
            "query_heads": 4,
            "kv_heads": 1,
            "head_dim": 16,
        })
        if args.node_counts is None:
            args.node_counts = [32, 96]
        args.queries = min(args.queries, 6)
        args.warmup = min(args.warmup, 1)
        args.query_tokens = min(args.query_tokens, 2)
        args.key_tokens = min(args.key_tokens, 2)
        args.keys_per_node = min(args.keys_per_node, 2)
        args.max_vectors = min(args.max_vectors, 128)
        args.max_files = min(args.max_files, 64)
    if args.warmup >= args.queries:
        raise SystemExit("--warmup must be lower than --queries")
    if args.key_tokens <= 0 or args.query_tokens <= 0 or args.keys_per_node <= 0:
        raise SystemExit("token and key counts must be positive")

    capture_routes = None
    capture_stats = None
    if args.capture_dir is not None:
        capture_routes, capture_stats = load_capture_routes(
            args.capture_dir,
            role=args.capture_role,
            layer=args.capture_layer,
            limit=args.capture_limit,
            token_limit=args.capture_token_limit,
            kv_heads=preset["kv_heads"],
            head_dim=preset["head_dim"],
        )
        if args.node_counts is None:
            args.node_counts = [
                min(32, capture_routes.shape[0]),
                min(96, capture_routes.shape[0]),
            ]
        queries = make_capture_queries(
            capture_routes,
            query_count=args.queries,
            query_heads=preset["query_heads"],
            query_tokens=args.query_tokens,
        )
        route_stats = capture_stats
        query_stats = {
            "source": "capture_key_derived",
            "query_count": int(queries.shape[0]),
        }
        route_source = "capture"
    else:
        if args.node_counts is None:
            args.node_counts = [1000, 10000]
        route_dim = preset["kv_heads"] * args.key_tokens * preset["head_dim"]
        query_dim = preset["query_heads"] * args.query_tokens * preset["head_dim"]
        route_bank, route_stats = baseline.harvest_centroid_bank(
            args.source_root, dim=route_dim, max_vectors=args.max_vectors,
            max_files=args.max_files)
        query_bank, query_stats = baseline.harvest_centroid_bank(
            args.source_root, dim=query_dim, max_vectors=args.max_vectors,
            max_files=args.max_files)
        queries = make_gqa_queries(
            query_bank,
            query_count=args.queries,
            query_heads=preset["query_heads"],
            query_tokens=args.query_tokens,
            head_dim=preset["head_dim"],
        )
        route_source = "harvested"

    results = []
    if args.progress_out is not None:
        args.progress_out.parent.mkdir(parents=True, exist_ok=True)
        args.progress_out.write_text("", encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="grm_gqa_router_") as td:
        lib = baseline.build_native_lib(Path(td), cxx=args.cxx, openmp=args.openmp)
        for idx, node_count in enumerate(args.node_counts, start=1):
            if capture_routes is not None:
                if node_count > capture_routes.shape[0]:
                    raise SystemExit(
                        f"node count {node_count} exceeds loaded captures "
                        f"{capture_routes.shape[0]}")
                routes = capture_routes[:node_count]
            else:
                routes = make_gqa_routes(
                    route_bank,
                    node_count=node_count,
                    keys_per_node=args.keys_per_node,
                    kv_heads=preset["kv_heads"],
                    key_tokens=args.key_tokens,
                    head_dim=preset["head_dim"],
                )
            row = benchmark_count(
                routes=routes,
                queries=queries,
                lib_path=lib,
                preset=preset,
                route_layer=args.route_layer,
                topk=args.topk,
                warmup=args.warmup,
                native_only=args.native_only,
                use_lexical=not args.no_lexical,
            )
            results.append(row)
            if args.progress_out is not None or args.progress:
                record = progress_record(
                    result=row,
                    completed=idx,
                    total=len(args.node_counts),
                    preset_name=args.preset,
                    route_source=route_source,
                    openmp=args.openmp,
                    native_only=args.native_only,
                    route_stats=route_stats,
                    query_stats=query_stats,
                )
                if args.progress_out is not None:
                    append_progress(args.progress_out, record)
                if args.progress:
                    print_progress(record)

    shape_row = results[0] if results else {}
    out = {
        "schema": "grm-gqa-router-benchmark-v1",
        "repo": str(ROOT),
        "preset": args.preset,
        "route_source": route_source,
        "openmp": bool(args.openmp),
        "native_only": bool(args.native_only),
        "harvest": {
            "route": route_stats,
            "query": query_stats,
        },
        "shape": {
            "query_heads": int(shape_row.get("query_heads", preset["query_heads"])),
            "kv_heads": int(shape_row.get("kv_heads", preset["kv_heads"])),
            "head_dim": int(shape_row.get("head_dim", preset["head_dim"])),
            "query_tokens": int(shape_row.get("query_tokens", args.query_tokens)),
            "key_tokens": int(shape_row.get("key_tokens", args.key_tokens)),
            "keys_per_node": int(shape_row.get("keys_per_node", args.keys_per_node)),
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(out, args.markdown_out)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
