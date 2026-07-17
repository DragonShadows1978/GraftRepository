#!/usr/bin/env python3
"""Development gate for exact mixed-length GQA CUDA routing.

The gate uses stored Qwen3.5 target Q/K captures, retains every K vector in
each registered row prefix, and compares four executions of the same raw
absolute-dot routing law:

1. the original ragged NumPy rows;
2. the native CPU GQA router over those ragged rows;
3. the direct CUDA router over the exact zero-padded bank; and
4. the normal opt-in GQAArenaCache bridge.

No model inference runs here and no capture is modified.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.graft_arena import GQAArenaCache  # noqa: E402
from core.grm_cuda_router import CudaGQARouteSidecar  # noqa: E402
from core.grm_native import NativeGraftStore  # noqa: E402
from scripts import grm_gqa_router_benchmark as bench  # noqa: E402
from scripts import grm_router_baseline as baseline  # noqa: E402


DEFAULT_CAPTURE_DIR = Path(
    "/mnt/ForgeRealm/qwen35_graft_translation_poc/captures")
DEFAULT_LENGTHS = (32, 48, 64, 96, 128, 192, 256)
DEFAULT_NODE_COUNTS = (32, 128, 512)
DEFAULT_QUERY_COUNTS = (25, 50, 100)
TOPKS = (1, 3, 5, 16)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path,
                        default=DEFAULT_CAPTURE_DIR)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--node-counts", type=int, nargs="+",
                        default=list(DEFAULT_NODE_COUNTS))
    parser.add_argument("--query-counts", type=int, nargs="+",
                        default=list(DEFAULT_QUERY_COUNTS))
    parser.add_argument("--lengths", type=int, nargs="+",
                        default=list(DEFAULT_LENGTHS))
    parser.add_argument("--query-tokens", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--route-repeats", type=int, default=3)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument(
        "--out", type=Path,
        default=Path(
            "artifacts/grm_gqa_exact_ragged_cuda/"
            "dev_gate_seed20260712_v3.json"),
    )
    return parser.parse_args(argv)


def _eligible_capture_paths(capture_dir: Path, minimum_tokens: int) -> list[Path]:
    paths = []
    for metadata_path in sorted(capture_dir.glob("target_*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not metadata.get("has_queries"):
            continue
        if int(metadata.get("token_count", 0)) < minimum_tokens:
            continue
        npz_path = Path(metadata.get("npz", metadata_path.with_suffix(".npz")))
        if npz_path.exists():
            paths.append(npz_path)
    return paths


def _select_paths(
    capture_dir: Path,
    *,
    count: int,
    minimum_tokens: int,
    seed: int,
) -> list[Path]:
    eligible = _eligible_capture_paths(capture_dir, minimum_tokens)
    if len(eligible) < count:
        raise RuntimeError(
            f"need {count} target Q/K captures with >= {minimum_tokens} "
            f"tokens; found {len(eligible)}")
    rng = np.random.default_rng(seed)
    # Preserve the RNG's sampled order: every prefix used by the 32/128/512
    # ladder is itself an unbiased without-replacement sample. Sorting the
    # chosen indices would bias the smaller prefixes toward early shards.
    chosen = rng.choice(len(eligible), size=count, replace=False)
    return [eligible[int(idx)] for idx in chosen]


def _load_rows(
    paths: list[Path], *, layer: int, lengths: tuple[int, ...]
) -> list[np.ndarray]:
    key_name = f"l{int(layer)}_k"
    rows = []
    for idx, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as shard:
            key = np.asarray(shard[key_name][0], dtype=np.float32)
        token_count = int(lengths[idx % len(lengths)])
        if key.ndim != 3 or key.shape[1] < token_count:
            raise RuntimeError(
                f"{path} {key_name} shape {key.shape} cannot supply "
                f"{token_count} tokens")
        rows.append(np.ascontiguousarray(key[:, :token_count, :]))
    return rows


def _load_queries(
    paths: list[Path],
    *,
    layer: int,
    count: int,
    query_tokens: int,
    seed: int,
) -> tuple[list[np.ndarray], list[dict]]:
    if count > len(paths):
        raise ValueError("query count cannot exceed node count")
    rng = np.random.default_rng(seed)
    query_nodes = np.sort(rng.choice(len(paths), size=count, replace=False))
    query_name = f"l{int(layer)}_q"
    queries = []
    provenance = []
    for ordinal, node_idx in enumerate(query_nodes):
        path = paths[int(node_idx)]
        with np.load(path, allow_pickle=False) as shard:
            stored = np.asarray(shard[query_name][0], dtype=np.float32)
        width = min(int(query_tokens), int(stored.shape[1]))
        start = ((ordinal * 17) + (int(node_idx) * 7)) % (
            int(stored.shape[1]) - width + 1)
        queries.append(np.ascontiguousarray(stored[:, start:start + width]))
        provenance.append({
            "node_index": int(node_idx),
            "token_start": int(start),
            "token_count": int(width),
            "path": str(path),
        })
    return queries, provenance


def _ragged_scores(query: np.ndarray, rows: list[np.ndarray]) -> np.ndarray:
    """Call the production ragged scorer verbatim, including fp32 order.

    An earlier harness-only grouped-head einsum was algebraically equivalent
    but changed accumulation order enough to reverse a ~4e-7 cutoff near-tie.
    This gate is an implementation-parity instrument, so `_key_score` itself
    is the only valid NumPy authority.
    """
    arena = GQAArenaCache.__new__(GQAArenaCache)
    return np.asarray(
        [arena._key_score(query, key) for key in rows], dtype=np.float64)


def _stable_order(scores: np.ndarray, topk: int) -> list[int]:
    return sorted(
        range(int(scores.shape[0])),
        key=lambda idx: (-float(scores[idx]), idx),
    )[:int(topk)]


def _classify_mismatch(
    expected: list[int], got: list[int], scores: np.ndarray
) -> str | None:
    if got == expected:
        return None
    width = min(len(expected), len(got))
    for pos in range(width):
        if expected[pos] == got[pos]:
            continue
        if scores[expected[pos]] != scores[got[pos]]:
            return "non_tie"
    if len(expected) != len(got):
        return "non_tie"
    return "exact_tie_order"


def _current_process_vram_mib() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    total = 0
    found = False
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            if int(fields[0]) == os.getpid():
                total += int(fields[1])
                found = True
        except ValueError:
            continue
    return total if found else 0


def _make_store(lib_path: Path, layer: int) -> NativeGraftStore:
    preset = bench.PRESETS["qwen35-9b-attn"]
    return NativeGraftStore(
        str(lib_path),
        model_type=preset["model_type"],
        num_layers=int(preset["num_layers"]),
        hidden_dim=int(preset["hidden_dim"]),
        vals_per_tok_layer=int(preset["vals_per_tok_layer"]),
        route_layer=int(layer),
        payload_kind="gqa",
        num_kv_heads=int(preset["kv_heads"]),
        head_dim=int(preset["head_dim"]),
    )


def _append_node(
    store: NativeGraftStore,
    arena: GQAArenaCache,
    row: np.ndarray,
    idx: int,
) -> int:
    node_id = int(store.add_node(f"ragged CUDA gate node {idx}", b"", ntok=1))
    store.set_route_key_list(node_id, [row], [])
    arena.grafts.append({
        "native_node_id": node_id,
        "cent": row,
        "rare": set(),
        "text": f"ragged CUDA gate node {idx}",
        "kind": "doc",
        "retired": False,
    })
    arena._bump_cuda_gqa_epoch()
    return node_id


def _compare_orders(
    *,
    expected: list[int],
    scores: np.ndarray,
    orders: dict[str, list[int]],
    query_index: int,
) -> list[dict]:
    mismatches = []
    for topk in TOPKS:
        expected_k = expected[:topk]
        for backend, order in orders.items():
            got = order[:topk]
            classification = _classify_mismatch(expected_k, got, scores)
            if classification is not None:
                mismatches.append({
                    "query": int(query_index),
                    "topk": int(topk),
                    "backend": backend,
                    "classification": classification,
                    "expected": expected_k,
                    "got": got,
                })
    return mismatches


def _run_size(
    *,
    store: NativeGraftStore,
    arena: GQAArenaCache,
    sidecar: CudaGQARouteSidecar,
    rows: list[np.ndarray],
    paths: list[Path],
    node_ids: list[int],
    node_count: int,
    query_count: int,
    args: argparse.Namespace,
) -> dict:
    stage_rows = rows[:node_count]
    stage_paths = paths[:node_count]
    queries, query_provenance = _load_queries(
        stage_paths,
        layer=int(args.layer),
        count=int(query_count),
        query_tokens=int(args.query_tokens),
        seed=int(args.seed) + int(node_count),
    )

    vram_samples = [{"point": "before_bank", "mib": _current_process_vram_mib()}]
    started = time.perf_counter_ns()
    bank_inputs = arena._cuda_route_bank_inputs()
    host_build_ms = (time.perf_counter_ns() - started) / 1.0e6
    if bank_inputs is None:
        raise RuntimeError("mixed-length leaf rows were not CUDA eligible")
    route_bank, bank_node_ids, signature = bank_inputs

    started = time.perf_counter_ns()
    store.configure_cuda_gqa_route_bank(
        route_bank, bank_node_ids, sidecar=sidecar, signature=signature)
    store._cuda_gqa_bank_signature = signature
    attach_ms = (time.perf_counter_ns() - started) / 1.0e6
    device_setup_ms = float(store._cuda_gqa_bank.setup_wall_ms)
    attach_host_overhead_ms = max(0.0, attach_ms - device_setup_ms)
    cold_build_ms = host_build_ms + attach_ms
    vram_samples.append({"point": "after_attach", "mib": _current_process_vram_mib()})

    query_batch = np.ascontiguousarray(np.stack(queries), dtype=np.float32)
    direct_ids, direct_receipt = store.route_gqa_cuda(
        query_batch,
        topk=max(TOPKS),
        warmup=min(int(args.warmup), max(0, len(queries) - 1)),
        route_repeats=int(args.route_repeats),
        return_receipt=True,
    )
    vram_samples.append({"point": "after_direct", "mib": _current_process_vram_mib()})

    id_to_idx = {int(node_id): idx for idx, node_id in enumerate(node_ids[:node_count])}
    direct_orders = [
        [id_to_idx[int(node_id)] for node_id in order]
        for order in direct_ids
    ]

    cpu_orders = []
    cpu_wall_ms = []
    for query in queries:
        started = time.perf_counter_ns()
        order = store.route_gqa(query, (), topk=max(TOPKS))
        cpu_wall_ms.append((time.perf_counter_ns() - started) / 1.0e6)
        cpu_orders.append([id_to_idx[int(node_id)] for node_id in order])

    # Warm the exact production bridge after manual cold attachment, then
    # measure one visible semantic route per stored query.
    arena._probe_key = lambda _text, q=queries[0]: q
    for warm_idx in range(min(int(args.warmup), len(queries))):
        arena._probe_key = lambda _text, q=queries[warm_idx]: q
        arena.route("plain probe", exclude=set(), limit=max(TOPKS))

    bridge_orders = []
    bridge_wall_ms = []
    backend_misses = []
    sample_stride = max(1, len(queries) // 5)
    for query_idx, query in enumerate(queries):
        arena._probe_key = lambda _text, q=query: q
        started = time.perf_counter_ns()
        order = arena.route("plain probe", exclude=set(), limit=max(TOPKS))
        bridge_wall_ms.append((time.perf_counter_ns() - started) / 1.0e6)
        bridge_orders.append([int(idx) for idx in order])
        if arena.last_route_backend != "cuda":
            backend_misses.append({
                "query": int(query_idx),
                "backend": arena.last_route_backend,
            })
        if (query_idx + 1) % sample_stride == 0 or query_idx + 1 == len(queries):
            vram_samples.append({
                "point": f"bridge_query_{query_idx + 1}",
                "mib": _current_process_vram_mib(),
            })

    mismatches = []
    python_wall_ms = []
    exact_tie_queries = []
    for query_idx, query in enumerate(queries):
        started = time.perf_counter_ns()
        scores = _ragged_scores(query, stage_rows)
        python_wall_ms.append((time.perf_counter_ns() - started) / 1.0e6)
        expected = _stable_order(scores, max(TOPKS))
        if any(scores[expected[pos]] == scores[expected[pos + 1]]
               for pos in range(len(expected) - 1)):
            exact_tie_queries.append(int(query_idx))
        mismatches.extend(_compare_orders(
            expected=expected,
            scores=scores,
            orders={
                "native_cpu": cpu_orders[query_idx],
                "direct_cuda": direct_orders[query_idx],
                "arena_bridge": bridge_orders[query_idx],
            },
            query_index=query_idx,
        ))

    padding_receipt = dict(arena._cuda_gqa_padding_receipt)
    padding_receipt["row_token_counts"] = list(
        padding_receipt["row_token_counts"])
    finite_vram = [
        int(sample["mib"]) for sample in vram_samples
        if sample["mib"] is not None
    ]
    steady_vram = [
        int(sample["mib"]) for sample in vram_samples
        if sample["mib"] is not None and sample["point"].startswith("bridge_query_")
    ]
    monotonic_growth = (
        len(steady_vram) >= 2
        and all(right > left for left, right in zip(steady_vram, steady_vram[1:]))
    )

    parity = not mismatches
    backend_engaged = not backend_misses
    latency_p50 = float(np.percentile(bridge_wall_ms, 50))
    latency_p95 = float(np.percentile(bridge_wall_ms, 95))
    result = {
        "node_count": int(node_count),
        "query_count": len(queries),
        "query_tokens": int(args.query_tokens),
        "topks": list(TOPKS),
        "parity": parity,
        "backend_engaged": backend_engaged,
        "backend_misses": backend_misses,
        "mismatch_count": len(mismatches),
        "non_tie_mismatch_count": sum(
            item["classification"] == "non_tie" for item in mismatches),
        "exact_tie_order_mismatch_count": sum(
            item["classification"] == "exact_tie_order" for item in mismatches),
        "mismatches": mismatches[:20],
        "exact_tie_queries": exact_tie_queries,
        "padding": padding_receipt,
        "cold": {
            "host_build_ms": float(host_build_ms),
            "attach_and_upload_ms": float(attach_ms),
            "device_create_ms": device_setup_ms,
            "attach_host_overhead_ms": attach_host_overhead_ms,
            "total_ms": float(cold_build_ms),
            "rail_ms": 250.0,
            "passes_rail": cold_build_ms <= 250.0,
        },
        "bridge_latency_ms": {
            "p50": latency_p50,
            "p95": latency_p95,
            "min": float(np.min(bridge_wall_ms)),
            "max": float(np.max(bridge_wall_ms)),
            "runs": [float(value) for value in bridge_wall_ms],
        },
        "native_cpu_latency_ms": {
            "p50": float(np.percentile(cpu_wall_ms, 50)),
            "p95": float(np.percentile(cpu_wall_ms, 95)),
        },
        "ragged_numpy_latency_ms": {
            "p50": float(np.percentile(python_wall_ms, 50)),
            "p95": float(np.percentile(python_wall_ms, 95)),
        },
        "direct_cuda_receipt": asdict(direct_receipt),
        "vram_samples_mib": vram_samples,
        "vram_sample_range_mib": (
            max(finite_vram) - min(finite_vram) if finite_vram else None),
        "steady_samples_strictly_monotonic": monotonic_growth,
        "query_provenance": query_provenance,
        "gates": {
            "parity": parity,
            "cuda_backend_every_query": backend_engaged,
            "padding_ratio_le_2_5": padding_receipt["padding_ratio"] <= 2.5,
            "resident_latency": (
                True if node_count != 512
                else latency_p50 <= 5.0 and latency_p95 <= 8.0
            ),
            "no_monotonic_vram_growth": not monotonic_growth,
        },
    }
    result["gates"]["quality_green"] = all(result["gates"].values())
    return result


def run(args: argparse.Namespace) -> dict:
    node_counts = tuple(int(value) for value in args.node_counts)
    query_counts = tuple(int(value) for value in args.query_counts)
    lengths = tuple(int(value) for value in args.lengths)
    if len(node_counts) != len(query_counts):
        raise ValueError("--node-counts and --query-counts must have equal length")
    if tuple(sorted(node_counts)) != node_counts or len(set(node_counts)) != len(node_counts):
        raise ValueError("node counts must be unique and ascending")
    if any(value <= 0 for value in (*node_counts, *query_counts, *lengths)):
        raise ValueError("node, query, and length values must be positive")
    if max(TOPKS) > min(node_counts):
        raise ValueError("smallest node count must cover topk=16")

    paths = _select_paths(
        args.capture_dir,
        count=max(node_counts),
        minimum_tokens=max(lengths),
        seed=int(args.seed),
    )
    rows = _load_rows(paths, layer=int(args.layer), lengths=lengths)

    old_cuda = os.environ.get("GRM_GQA_CUDA_ROUTE")
    old_query_lex = os.environ.get("GRM_ROUTE_QUERY_LEX")
    os.environ["GRM_GQA_CUDA_ROUTE"] = "1"
    os.environ["GRM_ROUTE_QUERY_LEX"] = "0"
    results = []
    try:
        with tempfile.TemporaryDirectory(prefix="grm-gqa-ragged-gate-") as td:
            build_dir = Path(td)
            native_build_dir = build_dir / "native"
            native_build_dir.mkdir(parents=True, exist_ok=True)
            lib_path = baseline.build_native_lib(
                native_build_dir, cxx=args.cxx, openmp=True)
            sidecar = CudaGQARouteSidecar.build(
                build_dir=build_dir / "cuda", nvcc=args.nvcc)
            try:
                with _make_store(lib_path, int(args.layer)) as store:
                    arena = GQAArenaCache.__new__(GQAArenaCache)
                    arena.native_store = store
                    arena.last_route_backend = "python"
                    arena._cuda_gqa_route_unavailable = False
                    arena._cuda_gqa_epoch = 0
                    arena.grafts = []
                    node_ids = []
                    added = 0
                    for node_count, query_count in zip(node_counts, query_counts):
                        for idx in range(added, node_count):
                            node_ids.append(_append_node(
                                store, arena, rows[idx], idx))
                        added = node_count
                        results.append(_run_size(
                            store=store,
                            arena=arena,
                            sidecar=sidecar,
                            rows=rows,
                            paths=paths,
                            node_ids=node_ids,
                            node_count=node_count,
                            query_count=query_count,
                            args=args,
                        ))
            finally:
                sidecar.close()
    finally:
        if old_cuda is None:
            os.environ.pop("GRM_GQA_CUDA_ROUTE", None)
        else:
            os.environ["GRM_GQA_CUDA_ROUTE"] = old_cuda
        if old_query_lex is None:
            os.environ.pop("GRM_ROUTE_QUERY_LEX", None)
        else:
            os.environ["GRM_ROUTE_QUERY_LEX"] = old_query_lex

    required = {
        "parity": all(row["gates"]["parity"] for row in results),
        "cuda_backend_every_query": all(
            row["gates"]["cuda_backend_every_query"] for row in results),
        "padding_ratio_le_2_5": all(
            row["gates"]["padding_ratio_le_2_5"] for row in results),
        "resident_latency_512": all(
            row["gates"]["resident_latency"] for row in results),
        "no_monotonic_vram_growth": all(
            row["gates"]["no_monotonic_vram_growth"] for row in results),
        "cold_rebuild_le_250_ms": all(
            row["cold"]["passes_rail"] for row in results),
    }
    stop_green = required["parity"] and required["cuda_backend_every_query"]
    compute_green = all(
        value for name, value in required.items()
        if name not in {"parity", "cuda_backend_every_query"}
    )
    receipt = {
        "schema": "grm_gqa_exact_ragged_cuda_dev_gate_v3",
        "evidence_tier": "development-visible",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(ROOT),
        "seed": int(args.seed),
        "capture_dir": str(args.capture_dir),
        "capture_role": "target",
        "layer": int(args.layer),
        "model_geometry": "Qwen3.5-9B attention (16 Q heads, 4 KV heads, D=256)",
        "node_counts": list(node_counts),
        "query_counts": list(query_counts),
        "length_schedule": list(lengths),
        "topks": list(TOPKS),
        "ragged_numpy_reference": (
            "core.graft_arena.GQAArenaCache._key_score verbatim"),
        "cuda_attach": (
            "epoch-signature reuse plus direct pitched host-to-final-head copy"),
        "selected_capture_paths": [str(path) for path in paths],
        "results": results,
        "required_gates": required,
        "verdict": (
            "QUALITY-GREEN" if stop_green and compute_green
            else "STOP-CORRECTNESS" if not stop_green
            else "QUALITY-GREEN / COMPUTE-RED"
        ),
    }
    return receipt


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    receipt = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "out": str(args.out),
        "verdict": receipt["verdict"],
        "required_gates": receipt["required_gates"],
        "results": [
            {
                "nodes": row["node_count"],
                "queries": row["query_count"],
                "parity": row["parity"],
                "padding_ratio": row["padding"]["padding_ratio"],
                "cold_ms": row["cold"]["total_ms"],
                "bridge_p50_ms": row["bridge_latency_ms"]["p50"],
                "bridge_p95_ms": row["bridge_latency_ms"]["p95"],
            }
            for row in receipt["results"]
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if receipt["verdict"] == "QUALITY-GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
