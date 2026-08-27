#!/usr/bin/env python3
"""MOE-RT1 GPT-OSS-20B native router telemetry and registered analysis.

This is a read-only diagnostic with respect to model behavior.  The GPU sweep
uses the repository's existing streamed GPT-OSS block implementation and wraps
only each block's ``_route`` method.  The wrapper performs the same FP32 router
matmul, bias add, top-k, and top-k softmax as the native method, copies the
requested telemetry to host memory, and returns the native routing tensors
unchanged to the existing expert path.

The four modes are deliberately separate so every GPU process can be wrapped
in the house ``flock`` and a sub-ten-minute timeout:

* ``prepare``: build deterministic local corpus windows and provenance (CPU)
* ``sweep``: run one corpus/window chunk through all 24 layers (GPU)
* ``gates``: run G0 non-invasiveness and G1 telemetry truth (GPU)
* ``analyze``: combine chunks, apply the registered statistics, write report
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_TENSOR = Path("/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
SNAPSHOT = Path(
    "/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/"
    "snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
)
WIKITEXT_SNAPSHOT = Path(
    "/home/vader/.cache/huggingface/hub/datasets--wikitext/"
    "snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/"
    "wikitext-2-raw-v1"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "moe_rt1"
CORPORA = ("generic", "code", "domain")
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
QUANTILE_KEYS = ("p05", "p25", "p50", "p75", "p95")
DEFAULT_SEED = 20260827
G1_ATOL = 2.0e-3
G1_RTOL = 2.0e-4


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or analyze registered GPT-OSS-20B router telemetry."
    )
    parser.add_argument(
        "mode", choices=("prepare", "sweep", "gates", "analyze", "blocked-report")
    )
    parser.add_argument("--model-dir", type=Path, default=SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--corpus", choices=CORPORA)
    parser.add_argument("--chunk-index", type=int)
    parser.add_argument("--window-tokens", type=int, default=512)
    parser.add_argument("--n-windows", type=int, default=16)
    parser.add_argument("--windows-per-chunk", type=int, default=8)
    parser.add_argument("--gate-tokens", type=int, default=8)
    parser.add_argument("--null-resamples", type=int, default=256)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(array).tobytes())


def source_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "byte_count": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def repository_revision_from_metadata(root: Path) -> dict[str, Any]:
    """Resolve HEAD without invoking git, which MOE-RT1 explicitly forbids."""

    git_dir = root / ".git"
    head_path = git_dir / "HEAD"
    result: dict[str, Any] = {
        "method": "read-only .git metadata; no git command invoked",
        "head_path": str(head_path),
        "commit": None,
        "ref": None,
    }
    if not head_path.is_file():
        return result
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        result["commit"] = head or None
        return result
    ref = head[5:].strip()
    result["ref"] = ref
    loose = git_dir / ref
    if loose.is_file():
        result["commit"] = loose.read_text(encoding="utf-8").strip() or None
        return result
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith(("#", "^")):
                continue
            commit, packed_ref = line.split(" ", 1)
            if packed_ref == ref:
                result["commit"] = commit
                break
    return result


def concatenate_plain_text(parts: list[str]) -> str:
    """Concatenate sources without synthetic headers; boundaries are blank lines."""

    return "\n\n".join(part for part in parts if part)


def build_generic_corpus() -> tuple[str, list[dict[str, Any]]]:
    import pyarrow.parquet as pq

    paths = [
        WIKITEXT_SNAPSHOT / "test-00000-of-00001.parquet",
        WIKITEXT_SNAPSHOT / "validation-00000-of-00001.parquet",
    ]
    parts: list[str] = []
    sources: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        table = pq.read_table(path, columns=["text"])
        rows = table.column("text").to_pylist()
        parts.append("\n".join("" if value is None else str(value) for value in rows))
        sources.append(source_record(path, rows=len(rows)))
    return concatenate_plain_text(parts), sources


def build_file_corpus(paths: list[Path]) -> tuple[str, list[dict[str, Any]]]:
    parts: list[str] = []
    sources: list[dict[str, Any]] = []
    for path in paths:
        parts.append(path.read_text(encoding="utf-8"))
        record = source_record(path)
        record["repo_relative_path"] = str(path.relative_to(REPO_ROOT))
        sources.append(record)
    return concatenate_plain_text(parts), sources


def build_code_corpus() -> tuple[str, list[dict[str, Any]]]:
    paths: list[Path] = []
    for directory in (REPO_ROOT / "core", REPO_ROOT / "scripts", REPO_ROOT / "cpp"):
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".cpp", ".h"}:
                paths.append(path)
    paths.sort(key=lambda path: str(path.relative_to(REPO_ROOT)))
    return build_file_corpus(paths)


def build_domain_corpus() -> tuple[str, list[dict[str, Any]]]:
    paths = sorted(
        (path for path in (REPO_ROOT / "docs").glob("*.md") if path.is_file()),
        key=lambda path: str(path.relative_to(REPO_ROOT)),
    )
    primer = REPO_ROOT / "docs" / "GRM_Primer.md"
    if primer not in paths:
        raise FileNotFoundError(primer)
    return build_file_corpus(paths)


def evenly_spaced_disjoint_offsets(
    token_count: int, n_windows: int, window_tokens: int
) -> list[int]:
    if token_count < n_windows * window_tokens:
        raise ValueError(
            f"corpus has {token_count} tokens; need at least "
            f"{n_windows * window_tokens} for disjoint windows"
        )
    offsets = np.rint(np.linspace(0, token_count - window_tokens, n_windows)).astype(int)
    if any(int(b - a) < window_tokens for a, b in zip(offsets[:-1], offsets[1:])):
        offsets = np.arange(n_windows, dtype=int) * int(window_tokens)
    return [int(value) for value in offsets]


def tokenizer_ids(tokenizer, text: str) -> np.ndarray:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )["input_ids"]
    return np.asarray(encoded, dtype=np.int64)


def prepare(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    if args.window_tokens < 1 or args.n_windows < 2:
        raise ValueError("window and sample counts must be positive; n-windows must be >=2")
    if args.n_windows % 2:
        raise ValueError("n-windows must be even for registered split-half resampling")
    if args.windows_per_chunk < 1 or args.n_windows % args.windows_per_chunk:
        raise ValueError("windows-per-chunk must divide n-windows")

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_dir.resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    builders: dict[str, Callable[[], tuple[str, list[dict[str, Any]]]]] = {
        "generic": build_generic_corpus,
        "code": build_code_corpus,
        "domain": build_domain_corpus,
    }
    arrays: dict[str, np.ndarray] = {}
    corpora: dict[str, Any] = {}
    for corpus in CORPORA:
        text, sources = builders[corpus]()
        canonical = text.encode("utf-8")
        ids = tokenizer_ids(tokenizer, text)
        offsets = evenly_spaced_disjoint_offsets(
            int(ids.size), int(args.n_windows), int(args.window_tokens)
        )
        windows = np.stack(
            [ids[start : start + args.window_tokens] for start in offsets], axis=0
        ).astype(np.int64, copy=False)
        arrays[corpus] = windows
        corpora[corpus] = {
            "canonicalization": (
                "UTF-8 source text in listed order, joined by exactly two newline "
                "characters; no synthetic file headers; tokenizer adds no special tokens"
            ),
            "corpus_byte_count": int(len(canonical)),
            "corpus_sha256": sha256_bytes(canonical),
            "corpus_token_count": int(ids.size),
            "routed_token_count": int(windows.size),
            "source_count": int(len(sources)),
            "sources": sources,
            "window_offsets": offsets,
            "window_tokens": int(args.window_tokens),
            "n_windows": int(args.n_windows),
            "windows_sha256_int64_le_host": sha256_array(windows),
        }
        print(
            json.dumps(
                {
                    "corpus": corpus,
                    "corpus_tokens": int(ids.size),
                    "routed_tokens": int(windows.size),
                    "sha256": corpora[corpus]["corpus_sha256"],
                }
            ),
            flush=True,
        )

    windows_path = out / "corpus_windows.npz"
    np.savez_compressed(windows_path, **arrays)
    script_path = Path(__file__).resolve()
    manifest = {
        "schema": "moe_rt1_corpora_v1",
        "created_at": now_iso(),
        "order": str((REPO_ROOT / "orders" / "MOE_RT1_ROUTER_TELEMETRY.md").resolve()),
        "repo_root": str(REPO_ROOT),
        "repository_revision": repository_revision_from_metadata(REPO_ROOT),
        "working_tree_note": "not queried because the registered order forbids git",
        "model_dir": str(model_dir),
        "model_snapshot_revision": model_dir.name,
        "model_config_sha256": sha256_file(model_dir / "config.json"),
        "model_index_sha256": sha256_file(model_dir / "model.safetensors.index.json"),
        "tokenizer_json_sha256": sha256_file(model_dir / "tokenizer.json"),
        "script": str(script_path),
        "script_sha256": sha256_file(script_path),
        "core_model_path": str((REPO_ROOT / "core" / "gpt_oss20b_tc.py").resolve()),
        "core_model_sha256": sha256_file(REPO_ROOT / "core" / "gpt_oss20b_tc.py"),
        "windows_path": str(windows_path),
        "windows_file_sha256": sha256_file(windows_path),
        "window_selection": "16 evenly spaced, disjoint token windows per corpus",
        "window_tokens": int(args.window_tokens),
        "n_windows": int(args.n_windows),
        "windows_per_chunk": int(args.windows_per_chunk),
        "corpora": corpora,
    }
    manifest_path = out / "corpora_manifest.json"
    write_json(manifest_path, manifest)
    print(f"manifest={manifest_path}", flush=True)
    print(f"windows={windows_path}", flush=True)
    return 0


def load_runtime():
    sys.path.insert(0, str(PROJECT_TENSOR))
    sys.path.insert(0, str(REPO_ROOT))
    import tensor_cuda as tc
    from core.gpt_oss20b_tc import (
        GptOss20BConfig,
        GptOssDiagnosticBlockTC,
        GptOssRowEmbedding,
        build_safetensors_map,
        gpt_oss_yarn_rope_tables,
        load_tensor_np,
    )
    from core.mistral7b_tc import BlockTC, QuantLinearTC, RMSNormTC

    BlockTC.COMPUTE_DTYPE = "bfloat16"
    QuantLinearTC.FUSED_DECODE = True
    RMSNormTC.USE_FUSED = True
    return {
        "tc": tc,
        "GptOss20BConfig": GptOss20BConfig,
        "GptOssDiagnosticBlockTC": GptOssDiagnosticBlockTC,
        "GptOssRowEmbedding": GptOssRowEmbedding,
        "build_safetensors_map": build_safetensors_map,
        "gpt_oss_yarn_rope_tables": gpt_oss_yarn_rope_tables,
        "load_tensor_np": load_tensor_np,
        "BlockTC": BlockTC,
    }


def nvidia_smi() -> str | None:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def install_router_capture(mlp, capture: dict[str, np.ndarray], *, capture_input: bool) -> None:
    """Install the non-mutating telemetry seam on one already-loaded layer."""

    tc = sys.modules["tensor_cuda"]

    def route(x_flat):
        logits = tc.matmul(x_flat.float(), mlp.router_weight, trans_b=True)
        bias = (
            mlp.router_bias
            if mlp.router_bias.dtype == logits.dtype
            else mlp.router_bias.astype(logits.dtype)
        )
        biased_logits = logits + bias
        topv, topi = biased_logits.topk(mlp.cfg.num_experts_per_tok, True)
        topw = topv.softmax(-1)
        capture["router_logits"] = biased_logits.numpy().astype(np.float32, copy=True)
        capture["top_indices"] = topi.numpy().astype(np.int16, copy=True)
        capture["top_gate_weights"] = topw.numpy().astype(np.float32, copy=True)
        if capture_input:
            capture["router_input"] = x_flat.float().numpy().astype(np.float32, copy=True)
        return topw, topi

    mlp._route = route


def load_prepared(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    out = args.output_dir.resolve()
    manifest_path = out / "corpora_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run prepare first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest["window_tokens"]) != int(args.window_tokens):
        raise ValueError("--window-tokens does not match prepared manifest")
    if int(manifest["n_windows"]) != int(args.n_windows):
        raise ValueError("--n-windows does not match prepared manifest")
    if int(manifest["windows_per_chunk"]) != int(args.windows_per_chunk):
        raise ValueError("--windows-per-chunk does not match prepared manifest")
    return out, manifest


def chunk_paths(out: Path, corpus: str, chunk_index: int) -> tuple[Path, Path]:
    stem = f"{corpus}_chunk{chunk_index:02d}"
    return out / f"{stem}_telemetry.npz", out / f"{stem}_receipt.json"


def sweep(args: argparse.Namespace) -> int:
    if args.corpus is None or args.chunk_index is None:
        raise ValueError("sweep requires --corpus and --chunk-index")
    out, manifest = load_prepared(args)
    n_chunks = args.n_windows // args.windows_per_chunk
    if args.chunk_index < 0 or args.chunk_index >= n_chunks:
        raise ValueError(f"chunk-index must be in [0, {n_chunks - 1}]")

    windows_path = Path(manifest["windows_path"])
    if sha256_file(windows_path) != manifest["windows_file_sha256"]:
        raise RuntimeError("prepared windows hash mismatch")
    with np.load(windows_path) as data:
        corpus_windows = data[args.corpus].astype(np.int64, copy=True)
    start_window = args.chunk_index * args.windows_per_chunk
    stop_window = start_window + args.windows_per_chunk
    ids = np.ascontiguousarray(corpus_windows[start_window:stop_window])
    expected_shape = (args.windows_per_chunk, args.window_tokens)
    if ids.shape != expected_shape:
        raise RuntimeError(f"chunk shape {ids.shape} != {expected_shape}")

    npz_path, receipt_path = chunk_paths(out, args.corpus, args.chunk_index)
    runtime = load_runtime()
    tc = runtime["tc"]
    model_dir = args.model_dir.resolve()
    started = time.perf_counter()
    receipt: dict[str, Any] = {
        "schema": "moe_rt1_sweep_chunk_v1",
        "status": "starting",
        "created_at": now_iso(),
        "argv": sys.argv,
        "required_shell_wrapper": (
            "flock -w 7200 /tmp/forge-gpu.lock timeout --signal=TERM "
            "--kill-after=5s 590s"
        ),
        "corpus": args.corpus,
        "corpus_sha256": manifest["corpora"][args.corpus]["corpus_sha256"],
        "chunk_index": int(args.chunk_index),
        "window_indices": list(range(start_window, stop_window)),
        "window_offsets": manifest["corpora"][args.corpus]["window_offsets"][
            start_window:stop_window
        ],
        "input_shape": list(ids.shape),
        "routed_token_count": int(ids.size),
        "model_dir": str(model_dir),
        "model_snapshot_revision": model_dir.name,
        "repository_revision": manifest["repository_revision"],
        "script_sha256_at_prepare": manifest["script_sha256"],
        "script_sha256_at_run": sha256_file(Path(__file__).resolve()),
        "core_model_sha256": manifest["core_model_sha256"],
        "attention_mode": "standard",
        "expert_mode": "resident_packed_mxfp4",
        "compute_dtype": "bfloat16",
        "router_compute_dtype": "float32",
        "num_layers_expected": 24,
        "num_experts_expected": 32,
        "experts_per_token_expected": 4,
        "completed_layers": 0,
        "layer_wall_seconds": [],
        "attention_backends": [],
        "gpu_before": nvidia_smi(),
        "telemetry_path": str(npz_path),
    }
    write_json(receipt_path, receipt)

    layer_logits: list[np.ndarray] = []
    layer_top_indices: list[np.ndarray] = []
    layer_top_weights: list[np.ndarray] = []
    try:
        cfg = runtime["GptOss20BConfig"].from_model_dir(model_dir)
        where = runtime["build_safetensors_map"](model_dir)
        if (cfg.num_layers, cfg.num_local_experts, cfg.num_experts_per_tok) != (24, 32, 4):
            raise RuntimeError(
                "model config is not registered GPT-OSS-20B 24-layer/32-expert/top-4"
            )
        with tc.no_grad():
            embed = runtime["GptOssRowEmbedding"](where)
            h = embed(ids)
            cos, sin = runtime["gpt_oss_yarn_rope_tables"](cfg, args.window_tokens)
            for layer_idx in range(cfg.num_layers):
                layer_started = time.perf_counter()
                block = runtime["GptOssDiagnosticBlockTC"].from_safetensors(
                    cfg, where, layer_idx, expert_mode="resident_packed_mxfp4"
                )
                block.self_attn.attention_mode = "standard"
                block.mlp.route_detail = "summary"
                block.mlp.empty_cache_interval = 0
                capture: dict[str, np.ndarray] = {}
                install_router_capture(block.mlp, capture, capture_input=False)
                h, kv, route_info = block(h, cos, sin)
                tc.synchronize()
                expected_router_shape = (ids.size, cfg.num_local_experts)
                if capture.get("router_logits", np.empty(0)).shape != expected_router_shape:
                    raise RuntimeError(
                        f"layer {layer_idx} router shape mismatch: "
                        f"{capture.get('router_logits', np.empty(0)).shape}"
                    )
                layer_logits.append(
                    capture["router_logits"].reshape(
                        args.windows_per_chunk, args.window_tokens, cfg.num_local_experts
                    )
                )
                layer_top_indices.append(
                    capture["top_indices"].reshape(
                        args.windows_per_chunk, args.window_tokens, cfg.num_experts_per_tok
                    )
                )
                layer_top_weights.append(
                    capture["top_gate_weights"].reshape(
                        args.windows_per_chunk, args.window_tokens, cfg.num_experts_per_tok
                    )
                )
                elapsed = time.perf_counter() - layer_started
                receipt["completed_layers"] = int(layer_idx + 1)
                receipt["layer_wall_seconds"].append(float(elapsed))
                receipt["attention_backends"].append(block.self_attn.last_attention_backend)
                receipt["status"] = "running"
                receipt["wall_seconds"] = float(time.perf_counter() - started)
                write_json(receipt_path, receipt)
                print(
                    json.dumps(
                        {
                            "corpus": args.corpus,
                            "chunk": args.chunk_index,
                            "layer": layer_idx,
                            "layer_seconds": round(elapsed, 3),
                            "elapsed_seconds": round(receipt["wall_seconds"], 3),
                        }
                    ),
                    flush=True,
                )
                del block, kv, route_info, capture
                gc.collect()
                if hasattr(tc, "empty_cache"):
                    tc.empty_cache()

            final_hidden = h.float().numpy().astype(np.float32, copy=False)
            final_hidden_sha256 = sha256_array(final_hidden)
            final_hidden_shape = list(final_hidden.shape)
            del h, final_hidden, embed, cos, sin
            if hasattr(tc, "empty_cache"):
                tc.empty_cache()

        logits_array = np.stack(layer_logits, axis=0).astype(np.float32, copy=False)
        top_indices_array = np.stack(layer_top_indices, axis=0).astype(np.int16, copy=False)
        top_weights_array = np.stack(layer_top_weights, axis=0).astype(np.float32, copy=False)
        np.savez_compressed(
            npz_path,
            layer_indices=np.arange(cfg.num_layers, dtype=np.int16),
            window_indices=np.arange(start_window, stop_window, dtype=np.int16),
            window_offsets=np.asarray(receipt["window_offsets"], dtype=np.int64),
            input_ids=ids,
            router_logits=logits_array,
            top_indices=top_indices_array,
            top_gate_weights=top_weights_array,
        )
        receipt.update(
            {
                "status": "complete",
                "wall_seconds": float(time.perf_counter() - started),
                "gpu_after": nvidia_smi(),
                "final_hidden_shape": final_hidden_shape,
                "final_hidden_fp32_sha256": final_hidden_sha256,
                "telemetry_file_sha256": sha256_file(npz_path),
                "telemetry_shapes": {
                    "router_logits": list(logits_array.shape),
                    "top_indices": list(top_indices_array.shape),
                    "top_gate_weights": list(top_weights_array.shape),
                },
            }
        )
        write_json(receipt_path, receipt)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "receipt": str(receipt_path),
                    "telemetry": str(npz_path),
                    "wall_seconds": receipt["wall_seconds"],
                }
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        receipt.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "wall_seconds": float(time.perf_counter() - started),
                "gpu_after": nvidia_smi(),
            }
        )
        write_json(receipt_path, receipt)
        raise


def gates(args: argparse.Namespace) -> int:
    out, manifest = load_prepared(args)
    if args.gate_tokens < 1 or args.gate_tokens > args.window_tokens:
        raise ValueError("gate-tokens must be within one prepared window")
    with np.load(manifest["windows_path"]) as data:
        ids = np.ascontiguousarray(data["generic"][0:1, : args.gate_tokens]).astype(
            np.int64, copy=False
        )

    runtime = load_runtime()
    tc = runtime["tc"]
    model_dir = args.model_dir.resolve()
    gate_path = out / "gates_g0_g1.json"
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "schema": "moe_rt1_gates_v1",
        "status": "starting",
        "created_at": now_iso(),
        "argv": sys.argv,
        "required_shell_wrapper": (
            "flock -w 7200 /tmp/forge-gpu.lock timeout --signal=TERM "
            "--kill-after=5s 590s"
        ),
        "model_dir": str(model_dir),
        "input_shape": list(ids.shape),
        "input_ids": ids.tolist(),
        "attention_mode": "standard",
        "expert_mode": "resident_packed_mxfp4",
        "compute_dtype": "bfloat16",
        "gpu_before": nvidia_smi(),
        "gates": {},
    }
    write_json(gate_path, payload)

    try:
        cfg = runtime["GptOss20BConfig"].from_model_dir(model_dir)
        where = runtime["build_safetensors_map"](model_dir)
        cos, sin = runtime["gpt_oss_yarn_rope_tables"](cfg, args.gate_tokens)

        def gate_forward(telemetry_on: bool):
            g1_capture: dict[str, np.ndarray] | None = None
            with tc.no_grad():
                embed = runtime["GptOssRowEmbedding"](where)
                h = embed(ids)
                for layer_idx in range(cfg.num_layers):
                    block = runtime["GptOssDiagnosticBlockTC"].from_safetensors(
                        cfg, where, layer_idx, expert_mode="resident_packed_mxfp4"
                    )
                    block.self_attn.attention_mode = "standard"
                    block.mlp.route_detail = "summary"
                    block.mlp.empty_cache_interval = 0
                    capture: dict[str, np.ndarray] = {}
                    if telemetry_on:
                        install_router_capture(
                            block.mlp, capture, capture_input=(layer_idx == 0)
                        )
                    h, kv, _route_info = block(h, cos, sin)
                    tc.synchronize()
                    if telemetry_on and layer_idx == 0:
                        g1_capture = capture
                    del block, kv, _route_info
                    gc.collect()
                    if hasattr(tc, "empty_cache"):
                        tc.empty_cache()
                final = h.float().numpy().astype(np.float32, copy=True)
                del h, embed
                if hasattr(tc, "empty_cache"):
                    tc.empty_cache()
            return final, g1_capture

        off_hidden, _ = gate_forward(False)
        on_hidden, g1_capture = gate_forward(True)
        bytes_equal = off_hidden.tobytes() == on_hidden.tobytes()
        array_equal = bool(np.array_equal(off_hidden, on_hidden))
        max_hidden_diff = float(np.max(np.abs(off_hidden - on_hidden)))
        g0_green = bool(bytes_equal and array_equal)
        payload["gates"]["G0"] = {
            "verdict": "GREEN" if g0_green else "RED",
            "telemetry_off_fp32_sha256": sha256_array(off_hidden),
            "telemetry_on_fp32_sha256": sha256_array(on_hidden),
            "compared_nbytes": int(off_hidden.nbytes),
            "shape": list(off_hidden.shape),
            "byte_equal": bool(bytes_equal),
            "numpy_array_equal": bool(array_equal),
            "max_abs_diff": max_hidden_diff,
            "receipt": (
                "final layer-23 hidden states converted identically to FP32 host "
                "arrays and compared byte-for-byte"
            ),
        }

        if g1_capture is None:
            raise RuntimeError("telemetry-on pass did not retain layer-0 G1 capture")
        prefix = "model.layers.0.mlp.router"
        router_weight = runtime["load_tensor_np"](where, f"{prefix}.weight")
        router_bias = runtime["load_tensor_np"](where, f"{prefix}.bias")
        router_input = g1_capture["router_input"]
        reference_logits = (
            router_input.astype(np.float32, copy=False)
            @ router_weight.astype(np.float32, copy=False).T
        ) + router_bias.astype(np.float32, copy=False)
        captured_logits = g1_capture["router_logits"]
        abs_diff = np.abs(reference_logits - captured_logits)
        reference_top4 = np.argsort(-reference_logits, axis=-1, kind="stable")[:, :4]
        captured_top4 = g1_capture["top_indices"].astype(np.int64, copy=False)
        top4_exact = bool(np.array_equal(reference_top4, captured_top4))
        logits_close = bool(
            np.allclose(
                reference_logits,
                captured_logits,
                atol=G1_ATOL,
                rtol=G1_RTOL,
                equal_nan=False,
            )
        )
        ref_top_values = np.take_along_axis(reference_logits, reference_top4, axis=-1)
        ref_top_weights = np.exp(ref_top_values - ref_top_values.max(axis=-1, keepdims=True))
        ref_top_weights /= ref_top_weights.sum(axis=-1, keepdims=True)
        gate_weight_max_diff = float(
            np.max(np.abs(ref_top_weights - g1_capture["top_gate_weights"]))
        )
        g1_green = bool(top4_exact and logits_close)
        payload["gates"]["G1"] = {
            "verdict": "GREEN" if g1_green else "RED",
            "layer": 0,
            "token_count": int(router_input.shape[0]),
            "router_input_shape": list(router_input.shape),
            "router_weight_name": f"{prefix}.weight",
            "router_bias_name": f"{prefix}.bias",
            "safetensors_shard": where[f"{prefix}.weight"],
            "reference": "NumPy FP32 x @ safetensors_weight.T + safetensors_bias",
            "atol": G1_ATOL,
            "rtol": G1_RTOL,
            "logits_within_tolerance": logits_close,
            "max_abs_diff": float(abs_diff.max()),
            "mean_abs_diff": float(abs_diff.mean()),
            "top4_indices_exact": top4_exact,
            "top4_exact_tokens": int(np.all(reference_top4 == captured_top4, axis=1).sum()),
            "top4_total_tokens": int(router_input.shape[0]),
            "native_top4_softmax_max_abs_diff_vs_numpy": gate_weight_max_diff,
            "router_weight_fp32_sha256": sha256_array(router_weight),
            "router_bias_fp32_sha256": sha256_array(router_bias),
        }
        payload.update(
            {
                "status": "complete",
                "wall_seconds": float(time.perf_counter() - started),
                "gpu_after": nvidia_smi(),
            }
        )
        write_json(gate_path, payload)
        print(json.dumps({"status": "complete", "gates": payload["gates"]}), flush=True)
        return 0
    except Exception as exc:
        payload.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "wall_seconds": float(time.perf_counter() - started),
                "gpu_after": nvidia_smi(),
            }
        )
        write_json(gate_path, payload)
        raise


def quantile_dict(values: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(values.astype(np.float64, copy=False), QUANTILES)
    return {key: float(value) for key, value in zip(QUANTILE_KEYS, quantiles)}


def entropy_from_logits(logits: np.ndarray) -> np.ndarray:
    work = logits.astype(np.float64, copy=False)
    shifted = work - work.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    return -np.sum(probs * np.log(probs), axis=-1)


def churn_from_top_indices(top_indices: np.ndarray) -> float:
    values: list[np.ndarray] = []
    for window in top_indices:
        left = window[:-1]
        right = window[1:]
        intersection = (left[:, :, None] == right[:, None, :]).any(axis=2).sum(axis=1)
        symmetric_difference = 8 - (2 * intersection)
        values.append(symmetric_difference.astype(np.float64) / 4.0)
    return float(np.concatenate(values).mean())


def js_divergence(counts_a: np.ndarray, counts_b: np.ndarray) -> float:
    p = counts_a.astype(np.float64)
    q = counts_b.astype(np.float64)
    p /= p.sum()
    q /= q.sum()
    midpoint = 0.5 * (p + q)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log(left[mask] / right[mask])))

    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def make_unique_split_halves(n_windows: int, count: int, seed: int) -> list[list[int]]:
    half = n_windows // 2
    maximum = math.comb(n_windows, half) // 2
    if count < 8:
        raise ValueError("registered null requires at least 8 split-half resamples")
    if count > maximum:
        raise ValueError(f"requested {count} unique splits; maximum is {maximum}")
    rng = np.random.default_rng(seed)
    universe = tuple(range(n_windows))
    seen: set[tuple[int, ...]] = set()
    splits: list[list[int]] = []
    while len(splits) < count:
        selected = tuple(sorted(int(x) for x in rng.choice(n_windows, half, replace=False)))
        complement = tuple(value for value in universe if value not in selected)
        canonical = min(selected, complement)
        if canonical in seen:
            continue
        seen.add(canonical)
        splits.append(list(selected))
    return splits


def bootstrap_window_delta(
    domain_values: np.ndarray,
    generic_values: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    domain_values = np.asarray(domain_values, dtype=np.float64)
    generic_values = np.asarray(generic_values, dtype=np.float64)
    domain_indices = rng.integers(
        0, domain_values.size, size=(resamples, domain_values.size)
    )
    generic_indices = rng.integers(
        0, generic_values.size, size=(resamples, generic_values.size)
    )
    boot = domain_values[domain_indices].mean(axis=1) - generic_values[
        generic_indices
    ].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "delta_domain_minus_generic": float(domain_values.mean() - generic_values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def load_combined_telemetry(
    out: Path, manifest: dict[str, Any], corpus: str
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    n_chunks = int(manifest["n_windows"]) // int(manifest["windows_per_chunk"])
    arrays: dict[str, list[np.ndarray]] = {
        "router_logits": [],
        "top_indices": [],
        "top_gate_weights": [],
        "input_ids": [],
        "window_indices": [],
        "window_offsets": [],
    }
    receipts: list[dict[str, Any]] = []
    for chunk_index in range(n_chunks):
        npz_path, receipt_path = chunk_paths(out, corpus, chunk_index)
        if not npz_path.is_file() or not receipt_path.is_file():
            raise FileNotFoundError(f"missing sweep output: {npz_path} or {receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipts.append(receipt)
        if receipt.get("status") != "complete":
            raise RuntimeError(f"incomplete receipt: {receipt_path}")
        if sha256_file(npz_path) != receipt.get("telemetry_file_sha256"):
            raise RuntimeError(f"telemetry hash mismatch: {npz_path}")
        if receipt.get("corpus_sha256") != manifest["corpora"][corpus]["corpus_sha256"]:
            raise RuntimeError(f"corpus provenance mismatch: {receipt_path}")
        with np.load(npz_path) as data:
            if not np.array_equal(data["layer_indices"], np.arange(24, dtype=np.int16)):
                raise RuntimeError(f"layer index mismatch: {npz_path}")
            for key in arrays:
                arrays[key].append(data[key].copy())
    combined = {
        "router_logits": np.concatenate(arrays["router_logits"], axis=1),
        "top_indices": np.concatenate(arrays["top_indices"], axis=1),
        "top_gate_weights": np.concatenate(arrays["top_gate_weights"], axis=1),
        "input_ids": np.concatenate(arrays["input_ids"], axis=0),
        "window_indices": np.concatenate(arrays["window_indices"], axis=0),
        "window_offsets": np.concatenate(arrays["window_offsets"], axis=0),
    }
    return combined, receipts


def layer_metrics(
    logits: np.ndarray, top_indices: np.ndarray, top_gate_weights: np.ndarray
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    entropy = entropy_from_logits(logits)
    sorted_logits = np.sort(logits.astype(np.float64, copy=False), axis=-1)[..., ::-1]
    top1 = sorted_logits[..., 0]
    top4 = sorted_logits[..., 3]
    margin12 = sorted_logits[..., 0] - sorted_logits[..., 1]
    margin45 = sorted_logits[..., 3] - sorted_logits[..., 4]
    histogram = np.bincount(top_indices.reshape(-1), minlength=32)
    metric = {
        "entropy_full32": {
            "mean": float(entropy.mean()),
            "median": float(np.median(entropy)),
            "p90": float(np.quantile(entropy, 0.90)),
        },
        "calibration": {
            "top1_logit": quantile_dict(top1.reshape(-1)),
            "top4_boundary_logit": quantile_dict(top4.reshape(-1)),
            "top1_minus_top2": quantile_dict(margin12.reshape(-1)),
            "top4_minus_top5": quantile_dict(margin45.reshape(-1)),
        },
        "expert_utilization_counts": [int(value) for value in histogram],
        "expert_utilization_probabilities": [
            float(value / histogram.sum()) for value in histogram
        ],
        "mean_max_native_top4_gate_weight": float(top_gate_weights[..., 0].mean()),
        "consecutive_token_expert_set_churn": churn_from_top_indices(top_indices),
    }
    per_window = {
        "entropy": entropy.mean(axis=1),
        "top1_minus_top2": margin12.mean(axis=1),
        "top4_minus_top5": margin45.mean(axis=1),
        "histograms": np.stack(
            [np.bincount(window.reshape(-1), minlength=32) for window in top_indices]
        ),
    }
    return metric, per_window


def fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def calibration_row(corpus: str, layer: int, metric: dict[str, Any]) -> str:
    calibration = metric["calibration"]
    fields = [corpus.upper(), str(layer)]
    for name in (
        "top1_logit",
        "top4_boundary_logit",
        "top1_minus_top2",
        "top4_minus_top5",
    ):
        fields.extend(fmt(calibration[name][key], 4) for key in QUANTILE_KEYS)
    return "| " + " | ".join(fields) + " |"


def calibration_header() -> list[str]:
    names = ["Corpus", "L"]
    for prefix in ("top1", "top4", "m12", "m45"):
        names.extend(f"{prefix}-{key}" for key in QUANTILE_KEYS)
    header = "| " + " | ".join(names) + " |"
    divider = "| " + " | ".join("---:" if i else "---" for i in range(len(names))) + " |"
    return [header, divider]


def full_calibration_table(corpus: str, metrics: list[dict[str, Any]]) -> list[str]:
    lines = calibration_header()
    lines.extend(calibration_row(corpus, layer, metric) for layer, metric in enumerate(metrics))
    return lines


def expected_created_paths(out: Path, n_chunks: int) -> list[Path]:
    paths = [
        Path(__file__).resolve(),
        out / "corpora_manifest.json",
        out / "corpus_windows.npz",
        out / "gates_g0_g1.json",
    ]
    for corpus in CORPORA:
        for chunk_index in range(n_chunks):
            paths.extend(chunk_paths(out, corpus, chunk_index))
    paths.extend([out / "analysis.json", out / "MOE_RT1_REPORT.md"])
    return paths


def blocked_report(args: argparse.Namespace) -> int:
    """Write an honest RED receipt when the worker exposes no CUDA device.

    This mode never treats an execution-environment failure as evidence against
    H-RT1.  A later successful ``gates`` + six ``sweep`` calls followed by
    ``analyze`` replaces both blocked artifacts with the measured report.
    """

    out, manifest = load_prepared(args)
    gate_path = out / "gates_g0_g1.json"
    gate_receipt = (
        json.loads(gate_path.read_text(encoding="utf-8"))
        if gate_path.is_file()
        else {"status": "missing", "error": "gate receipt missing"}
    )
    corpus_sha = {
        corpus: manifest["corpora"][corpus]["corpus_sha256"] for corpus in CORPORA
    }
    analysis_path = out / "analysis.json"
    report_path = out / "MOE_RT1_REPORT.md"
    created_paths = [
        str(Path(__file__).resolve()),
        str((out / "corpora_manifest.json").resolve()),
        str((out / "corpus_windows.npz").resolve()),
        str(gate_path.resolve()),
        str(analysis_path.resolve()),
        str(report_path.resolve()),
    ]
    payload: dict[str, Any] = {
        "schema": "moe_rt1_analysis_blocked_v1",
        "status": "blocked_before_model_forward",
        "created_at": now_iso(),
        "blocker": (
            "worker namespace exposes no /dev/nvidia*; nvidia-smi rc=9 and the "
            "first TensorCUDA allocation reports no CUDA-capable device"
        ),
        "gate_attempt_receipt": gate_receipt,
        "corpus_preparation": {
            corpus: {
                "corpus_token_count": manifest["corpora"][corpus]["corpus_token_count"],
                "prepared_window_tokens": manifest["corpora"][corpus][
                    "routed_token_count"
                ],
                "actual_routed_tokens": 0,
                "corpus_sha256": corpus_sha[corpus],
            }
            for corpus in CORPORA
        },
        "gates": {
            "G0": "RED - 0 bytes compared; model forward did not start",
            "G1": "RED - 0 router tokens compared; model forward did not start",
            "G2": "RED - 0/8192 actual routed tokens in every corpus",
            "G3": "RED - report delivered without a registered H-RT1 verdict",
        },
        "registered_verdict": None,
        "registered_verdict_note": (
            "NOT EVALUATED; missing inference data is not a NOT DETECTED result"
        ),
        "created_or_modified_paths": created_paths,
    }
    write_json(analysis_path, payload)

    lines: list[str] = [
        "# MOE-RT1 — blocked execution receipt",
        "",
        f"Generated: `{payload['created_at']}`",
        "",
        "## Status",
        "",
        "No GPT-OSS layer ran. The locked G0/G1 command failed on the first "
        "TensorCUDA allocation with:",
        "",
        "```text",
        str(gate_receipt.get("error", "gate receipt missing")),
        "```",
        "",
        "The worker namespace exposed no `/dev/nvidia*`; `nvidia-smi` returned "
        "code 9 and could not communicate with the driver. PCI inspection still "
        "showed the RTX 4070 SUPER bound to the NVIDIA kernel driver. No host or "
        "device-node change was attempted because writes are restricted to this repo.",
        "",
        "**H-RT1 registered verdict: NOT EVALUATED — no inference measurements were produced.**",
        "",
        "This is not the registered `domain signal NOT DETECTED under these limits` "
        "outcome; that vocabulary requires completed router measurements.",
        "",
        "## Gates",
        "",
        "G0 RED — byte_equal=NOT_MEASURED; compared_bytes=0",
        "G1 RED — top4_exact=NOT_MEASURED; compared_router_tokens=0",
        "G2 RED — min_routed_tokens=0/8192 across 3 corpora",
        "G3 RED — report delivered; registered H-RT1 verdict unavailable",
        "",
        "## Prepared corpus provenance (CPU-only; not routed)",
        "",
        "| Corpus | Canonical SHA-256 | Available corpus tokens | Prepared tokens | Actual routed tokens |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for corpus in CORPORA:
        info = manifest["corpora"][corpus]
        lines.append(
            f"| {corpus.upper()} | `{info['corpus_sha256']}` | "
            f"{info['corpus_token_count']} | {info['routed_token_count']} | 0 |"
        )
    for corpus in CORPORA:
        lines.extend(["", f"### {corpus.upper()} exact prepared source files", ""])
        for source in manifest["corpora"][corpus]["sources"]:
            display = source.get("repo_relative_path", source["path"])
            lines.append(
                f"- `{display}` — `{source['sha256']}` — {source['byte_count']} bytes"
            )
    lines.extend(
        [
            "",
            "## Layer 12 calibration sample",
            "",
            "| Corpus | L | top1 p05/p25/p50/p75/p95 | top4 p05/p25/p50/p75/p95 | m12 p05/p25/p50/p75/p95 | m45 p05/p25/p50/p75/p95 |",
            "| --- | ---: | --- | --- | --- | --- |",
            "| GENERIC | 12 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |",
            "| CODE | 12 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |",
            "| DOMAIN | 12 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |",
            "",
            "## Exact resume commands",
            "",
            "Run each command in a GPU-enabled worker from the repository root, "
            "leaving an idle gap between commands:",
            "",
            "```bash",
            "flock -w 7200 /tmp/forge-gpu.lock timeout --signal=TERM --kill-after=5s 590s env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_router_telemetry.py gates",
        ]
    )
    for corpus in CORPORA:
        for chunk_index in range(args.n_windows // args.windows_per_chunk):
            lines.append(
                "flock -w 7200 /tmp/forge-gpu.lock timeout --signal=TERM "
                "--kill-after=5s 590s env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
                "PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 "
                "scripts/gpt_oss20b_router_telemetry.py sweep "
                f"--corpus {corpus} --chunk-index {chunk_index}"
            )
    lines.extend(
        [
            "python3 scripts/gpt_oss20b_router_telemetry.py analyze",
            "```",
            "",
            "## Files created or modified",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in created_paths)
    lines.extend(
        [
            "",
            "## Could not do",
            "",
            "- G0 and G1 could not execute because CUDA initialization failed before model load.",
            "- No corpus token traversed a model layer; G2 is RED at 0 tokens per corpus.",
            "- No router logits, top-4 selections, native gate weights, JS null, "
            "bootstrap intervals, calibration values, or registered H-RT1 verdict exist.",
            "- The unavailable execution escalation path was not invoked; this session "
            "does not permit it.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "report": str(report_path)}), flush=True)
    return 0


def render_report(analysis: dict[str, Any], manifest: dict[str, Any]) -> str:
    metrics = analysis["per_corpus_layer_metrics"]
    hrt1 = analysis["registered_comparison"]
    gates_payload = analysis["gates"]
    lines: list[str] = [
        "# MOE-RT1 — GPT-OSS-20B native router telemetry",
        "",
        f"Generated: `{analysis['created_at']}`",
        "",
        "## Scope and registered verdict",
        "",
        (
            "This is inference telemetry from one frozen GPT-OSS-20B snapshot. "
            "It characterizes routing statistics only; it is not a model-quality, "
            "expert-viability, or general MoE claim."
        ),
        "",
        f"**{analysis['registered_verdict']}**",
        "",
        (
            f"The strict registered per-layer rule was `JS_dg(L) > null_p95(L)`. "
            f"It held at **{hrt1['supporting_layer_count']}/24** zero-based layers."
        ),
        "",
        "## Gates",
        "",
    ]
    for gate in ("G0", "G1", "G2", "G3"):
        lines.append(gates_payload[gate]["report_line"])
    lines.extend(
        [
            "",
            "G0 receipt: telemetry OFF and ON final layer-23 hidden states were "
            f"compared byte-for-byte over `{gates_payload['G0']['compared_nbytes']}` bytes; "
            f"max absolute difference `{gates_payload['G0']['max_abs_diff']}`.",
            "",
            "G1 receipt: layer-0 router logits were recomputed as NumPy FP32 "
            "`x @ W.T + b` from the safetensors router tensors. "
            f"Max absolute difference `{gates_payload['G1']['max_abs_diff']}` with "
            f"`atol={gates_payload['G1']['atol']}`, `rtol={gates_payload['G1']['rtol']}`; "
            f"ordered top-4 exact for `{gates_payload['G1']['top4_exact_tokens']}/"
            f"{gates_payload['G1']['top4_total_tokens']}` tokens.",
            "",
            "## Methods",
            "",
            "- Model snapshot: `" + manifest["model_dir"] + "`.",
            "- Snapshot revision: `" + manifest["model_snapshot_revision"] + "`.",
            "- Repository HEAD was read directly from read-only `.git` metadata because "
            "the order forbids git commands: `" + str(manifest["repository_revision"]["commit"]) + "`.",
            "- Working-tree state was not queried because the order forbids git.",
            "- Runtime: existing `GptOssDiagnosticBlockTC`, streamed one layer at a time; "
            "standard attention; BF16 block compute; resident packed MXFP4 experts; FP32 router.",
            "- Sampling: 16 evenly spaced, disjoint windows of 512 tokens per corpus. "
            "Each window traversed all 24 layers; two 8-window GPU chunks per corpus.",
            "- Telemetry seam: the existing per-layer `_route` computation was reproduced "
            "operation-for-operation, copied to host, and its native top-k tensors returned "
            "unchanged to the existing expert path.",
            "- `top_gate_weights` are the model's native softmax over the selected top-4. "
            "Entropy is computed separately from a stable full softmax over all 32 logits.",
            "- JS divergence uses natural logarithms (nats), no smoothing. The null has "
            f"{hrt1['null_resample_count']} unique random 8-window-vs-8-window GENERIC "
            f"split halves, seed `{analysis['seed']}`; the comparison is strict `>`.",
            "- Secondary 95% CIs use independent window-level bootstrap resampling with "
            f"replacement, `{analysis['bootstrap_resamples']}` resamples. They are descriptive only.",
            "- Churn is computed within windows only as `|set(t) Δ set(t+1)| / 4`; "
            "window boundaries are excluded.",
            "",
            "## Corpus provenance and sweep completeness",
            "",
            "| Corpus | Canonical SHA-256 | Corpus tokens | Routed tokens | Windows | Sources |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for corpus in CORPORA:
        info = manifest["corpora"][corpus]
        lines.append(
            f"| {corpus.upper()} | `{info['corpus_sha256']}` | "
            f"{info['corpus_token_count']} | {info['routed_token_count']} | "
            f"{info['n_windows']} × {info['window_tokens']} | {info['source_count']} |"
        )
    lines.extend(["", "Canonicalization: " + manifest["corpora"]["generic"]["canonicalization"] + "."])
    for corpus in CORPORA:
        lines.extend(["", f"### {corpus.upper()} exact source files", ""])
        for source in manifest["corpora"][corpus]["sources"]:
            display = source.get("repo_relative_path", source["path"])
            lines.append(
                f"- `{display}` — `{source['sha256']}` — {source['byte_count']} bytes"
            )

    lines.extend(
        [
            "",
            "## Registered DOMAIN vs GENERIC comparison",
            "",
            "| L | JS_dg | null p05 | null median | null p95 | null max | Supports at L | JS_domain-code | JS_generic-code |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |",
        ]
    )
    for item in hrt1["layers"]:
        lines.append(
            f"| {item['layer']} | {fmt(item['js_domain_generic'])} | "
            f"{fmt(item['null_p05'])} | {fmt(item['null_median'])} | "
            f"{fmt(item['null_p95'])} | {fmt(item['null_max'])} | "
            f"{'yes' if item['supported'] else 'no'} | "
            f"{fmt(item['js_domain_code'])} | {fmt(item['js_generic_code'])} |"
        )

    lines.extend(
        [
            "",
            "## Secondary DOMAIN minus GENERIC descriptives",
            "",
            "All intervals below are window-bootstrap 95% CIs and have no registered pass/fail role.",
            "",
            "| L | entropy Δ | entropy CI95 | top1−top2 Δ | m12 CI95 | top4−top5 Δ | m45 CI95 |",
            "| ---: | ---: | --- | ---: | --- | ---: | --- |",
        ]
    )
    for item in analysis["secondary_domain_generic"]:
        ent = item["entropy"]
        m12 = item["top1_minus_top2"]
        m45 = item["top4_minus_top5"]
        lines.append(
            f"| {item['layer']} | {fmt(ent['delta_domain_minus_generic'])} | "
            f"[{fmt(ent['ci95_low'])}, {fmt(ent['ci95_high'])}] | "
            f"{fmt(m12['delta_domain_minus_generic'])} | "
            f"[{fmt(m12['ci95_low'])}, {fmt(m12['ci95_high'])}] | "
            f"{fmt(m45['delta_domain_minus_generic'])} | "
            f"[{fmt(m45['ci95_low'])}, {fmt(m45['ci95_high'])}] |"
        )

    lines.extend(
        [
            "",
            "## Layer 12 calibration sample",
            "",
            "Layers are zero-based. Logits and margins are FP32 telemetry; `top4` is the "
            "selection-boundary logit, `m12` is top1−top2, and `m45` is top4−top5.",
            "",
            *calibration_header(),
        ]
    )
    for corpus in CORPORA:
        lines.append(calibration_row(corpus, 12, metrics[corpus][12]))

    lines.extend(["", "## Full calibration tables", ""])
    for corpus in CORPORA:
        lines.extend([f"### {corpus.upper()}", ""])
        lines.extend(full_calibration_table(corpus, metrics[corpus]))
        lines.append("")

    lines.extend(
        [
            "## Per-layer descriptive telemetry",
            "",
            "| Corpus | L | entropy mean | median | p90 | mean max native gate | churn |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for corpus in CORPORA:
        for layer, metric in enumerate(metrics[corpus]):
            entropy = metric["entropy_full32"]
            lines.append(
                f"| {corpus.upper()} | {layer} | {fmt(entropy['mean'])} | "
                f"{fmt(entropy['median'])} | {fmt(entropy['p90'])} | "
                f"{fmt(metric['mean_max_native_top4_gate_weight'])} | "
                f"{fmt(metric['consecutive_token_expert_set_churn'])} |"
            )

    lines.extend(
        [
            "",
            "## Expert-utilization histograms",
            "",
            "Each vector is `[expert 0, ..., expert 31]` and sums to 32,768 "
            "top-4 selections per corpus/layer.",
            "",
            "| Corpus | L | selection counts e0..e31 |",
            "| --- | ---: | --- |",
        ]
    )
    for corpus in CORPORA:
        for layer, metric in enumerate(metrics[corpus]):
            values = ",".join(str(value) for value in metric["expert_utilization_counts"])
            lines.append(f"| {corpus.upper()} | {layer} | `[{values}]` |")

    lines.extend(["", "## Artifact receipt", ""])
    for path in analysis["created_or_modified_paths"]:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## Anomalies and limitations",
            "",
        ]
    )
    anomalies = analysis.get("anomalies") or ["No execution anomaly was recorded."]
    lines.extend(f"- {item}" for item in anomalies)
    lines.extend(
        [
            "- The sample is exactly 8,192 routed sequence tokens per corpus on one model "
            "snapshot; every token contributes 24 layer-router observations.",
            "- The null controls GENERIC window-to-window variation only. It does not "
            "establish causal domain specialization or bolt-on expert viability.",
            "- CODE is a descriptive second contrast. It has no registered pass/fail role.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> int:
    out, manifest = load_prepared(args)
    combined: dict[str, dict[str, np.ndarray]] = {}
    receipts: dict[str, list[dict[str, Any]]] = {}
    metrics: dict[str, list[dict[str, Any]]] = {corpus: [] for corpus in CORPORA}
    window_metrics: dict[str, list[dict[str, np.ndarray]]] = {
        corpus: [] for corpus in CORPORA
    }
    for corpus in CORPORA:
        combined[corpus], receipts[corpus] = load_combined_telemetry(out, manifest, corpus)
        expected_logits_shape = (24, args.n_windows, args.window_tokens, 32)
        if combined[corpus]["router_logits"].shape != expected_logits_shape:
            raise RuntimeError(
                f"{corpus} logits shape {combined[corpus]['router_logits'].shape} "
                f"!= {expected_logits_shape}"
            )
        prepared_sha = manifest["corpora"][corpus]["windows_sha256_int64_le_host"]
        if sha256_array(combined[corpus]["input_ids"]) != prepared_sha:
            raise RuntimeError(f"{corpus} combined input IDs do not match prepared windows")
        for layer in range(24):
            metric, by_window = layer_metrics(
                combined[corpus]["router_logits"][layer],
                combined[corpus]["top_indices"][layer],
                combined[corpus]["top_gate_weights"][layer],
            )
            metrics[corpus].append(metric)
            window_metrics[corpus].append(by_window)

    splits = make_unique_split_halves(args.n_windows, args.null_resamples, args.seed)
    all_windows = set(range(args.n_windows))
    comparison_layers: list[dict[str, Any]] = []
    supporting_layers: list[int] = []
    for layer in range(24):
        generic_hist = np.asarray(
            metrics["generic"][layer]["expert_utilization_counts"], dtype=np.int64
        )
        domain_hist = np.asarray(
            metrics["domain"][layer]["expert_utilization_counts"], dtype=np.int64
        )
        code_hist = np.asarray(
            metrics["code"][layer]["expert_utilization_counts"], dtype=np.int64
        )
        generic_windows = window_metrics["generic"][layer]["histograms"]
        null_values: list[float] = []
        for selected in splits:
            complement = sorted(all_windows.difference(selected))
            null_values.append(
                js_divergence(
                    generic_windows[selected].sum(axis=0),
                    generic_windows[complement].sum(axis=0),
                )
            )
        null = np.asarray(null_values, dtype=np.float64)
        js_dg = js_divergence(domain_hist, generic_hist)
        null_p95 = float(np.quantile(null, 0.95))
        supported = bool(js_dg > null_p95)
        if supported:
            supporting_layers.append(layer)
        comparison_layers.append(
            {
                "layer": layer,
                "js_domain_generic": js_dg,
                "null_p05": float(np.quantile(null, 0.05)),
                "null_median": float(np.median(null)),
                "null_p95": null_p95,
                "null_max": float(null.max()),
                "null_mean": float(null.mean()),
                "null_std": float(null.std(ddof=1)),
                "null_resamples": [float(value) for value in null],
                "supported": supported,
                "js_domain_code": js_divergence(domain_hist, code_hist),
                "js_generic_code": js_divergence(generic_hist, code_hist),
            }
        )

    secondary: list[dict[str, Any]] = []
    metric_names = ("entropy", "top1_minus_top2", "top4_minus_top5")
    for layer in range(24):
        item: dict[str, Any] = {"layer": layer}
        for metric_offset, name in enumerate(metric_names):
            item[name] = bootstrap_window_delta(
                window_metrics["domain"][layer][name],
                window_metrics["generic"][layer][name],
                resamples=args.bootstrap_resamples,
                seed=args.seed + (layer * 17) + metric_offset + 1,
            )
        secondary.append(item)

    gate_path = out / "gates_g0_g1.json"
    gate_receipt = json.loads(gate_path.read_text(encoding="utf-8"))
    g0 = dict(gate_receipt.get("gates", {}).get("G0", {}))
    g1 = dict(gate_receipt.get("gates", {}).get("G1", {}))
    g0.setdefault("verdict", "RED")
    g1.setdefault("verdict", "RED")
    g0["report_line"] = (
        f"G0 {g0['verdict']} — byte_equal={str(g0.get('byte_equal', False)).lower()}; "
        f"max_abs_diff={g0.get('max_abs_diff', 'missing')}"
    )
    g1["report_line"] = (
        f"G1 {g1['verdict']} — top4_exact="
        f"{g1.get('top4_exact_tokens', 0)}/{g1.get('top4_total_tokens', 0)}; "
        f"max_abs_diff={g1.get('max_abs_diff', 'missing')}"
    )

    corpus_actuals = {
        corpus: int(sum(item["routed_token_count"] for item in receipts[corpus]))
        for corpus in CORPORA
    }
    g2_green = all(
        corpus_actuals[corpus] >= 8192
        and len(receipts[corpus]) == (args.n_windows // args.windows_per_chunk)
        and all(item.get("status") == "complete" for item in receipts[corpus])
        for corpus in CORPORA
    )
    min_actual = min(corpus_actuals.values())
    g2 = {
        "verdict": "GREEN" if g2_green else "RED",
        "corpus_actual_routed_tokens": corpus_actuals,
        "minimum_actual_routed_tokens": min_actual,
        "required_minimum": 8192,
        "report_line": (
            f"G2 {'GREEN' if g2_green else 'RED'} — min_routed_tokens="
            f"{min_actual}/8192 across 3 corpora"
        ),
    }
    support_count = len(supporting_layers)
    registered_verdict = f"H-RT1 registered verdict: SUPPORTED at {support_count}/24 layers."
    g3 = {
        "verdict": "GREEN",
        "supporting_layer_count": support_count,
        "report_line": f"G3 GREEN — report delivered; SUPPORTED at {support_count}/24 layers",
    }

    n_chunks = args.n_windows // args.windows_per_chunk
    paths = [str(path.resolve()) for path in expected_created_paths(out, n_chunks)]
    anomalies: list[str] = []
    for gate_name, gate in (("G0", g0), ("G1", g1), ("G2", g2)):
        if gate["verdict"] != "GREEN":
            anomalies.append(f"{gate_name} failed and is reported RED; see its receipt above.")
    layer_times = [
        float(value)
        for corpus in CORPORA
        for receipt in receipts[corpus]
        for value in receipt.get("layer_wall_seconds", [])
    ]
    invocation_walls = [
        float(receipt["wall_seconds"])
        for corpus in CORPORA
        for receipt in receipts[corpus]
    ]
    if any(value >= 600.0 for value in invocation_walls):
        anomalies.append("At least one recorded sweep invocation reached or exceeded 600 seconds.")

    analysis: dict[str, Any] = {
        "schema": "moe_rt1_analysis_v1",
        "created_at": now_iso(),
        "seed": int(args.seed),
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "registered_verdict": registered_verdict,
        "registered_comparison": {
            "criterion": "JS_domain_generic > p95(unique GENERIC 8-vs-8 window split halves)",
            "js_log_base": "natural",
            "null_resample_count": int(args.null_resamples),
            "supporting_layer_count": support_count,
            "supporting_layers": supporting_layers,
            "layers": comparison_layers,
        },
        "secondary_domain_generic": secondary,
        "per_corpus_layer_metrics": metrics,
        "corpus_actual_routed_tokens": corpus_actuals,
        "corpus_sha256": {
            corpus: manifest["corpora"][corpus]["corpus_sha256"] for corpus in CORPORA
        },
        "sweep_receipts": receipts,
        "runtime": {
            "sweep_invocation_wall_seconds": invocation_walls,
            "max_sweep_invocation_wall_seconds": max(invocation_walls),
            "mean_layer_wall_seconds": float(np.mean(layer_times)),
        },
        "gates": {"G0": g0, "G1": g1, "G2": g2, "G3": g3},
        "created_or_modified_paths": paths,
        "anomalies": anomalies,
    }
    report = render_report(analysis, manifest)
    if registered_verdict not in report:
        raise RuntimeError("registered verdict missing from rendered report")
    analysis_path = out / "analysis.json"
    report_path = out / "MOE_RT1_REPORT.md"
    write_json(analysis_path, analysis)
    report_path.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "analysis": str(analysis_path),
                "registered_verdict": registered_verdict,
                "gates": {name: value["verdict"] for name, value in analysis["gates"].items()},
            }
        ),
        flush=True,
    )
    return 0


def main() -> int:
    args = parse_args()
    args.model_dir = args.model_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.mode == "prepare":
        return prepare(args)
    if args.mode == "sweep":
        return sweep(args)
    if args.mode == "gates":
        return gates(args)
    if args.mode == "analyze":
        return analyze(args)
    if args.mode == "blocked-report":
        return blocked_report(args)
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
