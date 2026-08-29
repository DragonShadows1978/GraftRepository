#!/usr/bin/env python3
"""MOE-E3: consolidate one novel episode into a detachable OLMoE expert.

The frozen base sees a target window either alone (student) or after its true
2,048 preceding tokens (teacher).  A rank-64 threshold-gated residual adapter
is trained to reproduce the teacher-side layer delta without carrying context.

All model and corpus access is local-only.  The script is resumable by stage,
uses strict content provenance, and imports the battle-tested E2 model loader,
writer, scoring, ABI mount, and numerical primitives without modifying E2.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import platform
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("PYTHONPYCACHEPREFIX", "/tmp/olmoe_e3_pycache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

import olmoe_e2_experiment as e2


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
ORDER_PATH = REPO_ROOT / "orders" / "MOE_E3_EPISODIC_CONSOLIDATION.md"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "moe_e3"
STORIES_ROOT = Path("/mnt/Shared/01 - Narrative Stories")
E2_OUTPUT = REPO_ROOT / "artifacts" / "moe_e2"
E2_LEGACY_PREPARED = (
    E2_OUTPUT
    / "provenance_validation"
    / "legacy_prepared_windows_794e4604e81f94496b3702649953fb866fd89ad47d4cf2800d1925192c98ebf5.npz"
)
E2_LEGACY_CONTENT = (
    E2_OUTPUT
    / "provenance_validation"
    / "legacy_prepared_content_794e4604e81f94496b3702649953fb866fd89ad47d4cf2800d1925192c98ebf5.json"
)
E2_FIT_VALIDATION = E2_OUTPUT / "fit_key.provenance_validation.json"
E2_LEGACY_CONTENT_SHA256 = "794e4604e81f94496b3702649953fb866fd89ad47d4cf2800d1925192c98ebf5"
E2_LEGACY_ARCHIVE_RECORDED_SHA256 = "c4f61cdf9dbb34a49817e89e70c3f5dae46092a922fcfe113ad38a9282300de3"

MODEL_ID = e2.MODEL_ID
MODEL_REVISION = e2.MODEL_REVISION
DEFAULT_MODEL_DIR = e2.DEFAULT_MODEL_DIR
N_LAYERS = e2.N_LAYERS
HIDDEN_DIM = e2.HIDDEN_DIM
WINDOW_TOKENS = e2.WINDOW_TOKENS
PREFIX_TOKENS = e2.PREFIX_TOKENS
MAX_CONTEXT = e2.MAX_CONTEXT
N_KEY_WINDOWS = 16
N_PAIR_WINDOWS = 64
N_PAIR_TRAIN = 48
N_PAIR_VALIDATION = 16
N_BEHAVIORAL_WINDOWS = 16
PAIR_DOC_WINDOWS = tuple(range(4, 68))
DOC_A_KEY_WINDOWS = tuple(range(68, 84))
FIT_INDICES = tuple(range(0, N_KEY_WINDOWS, 2))
EVAL_INDICES = tuple(range(1, N_KEY_WINDOWS, 2))
INNER_TRAIN_INDICES = (0, 4, 8, 12)
INNER_VALIDATION_INDICES = (2, 6, 10, 14)
CAPTURE_CORPORA = ("doc_a", "doc_b")
NEGATIVE_CORPORA = ("wikitext", "code", "grm", "guides")
FIT_CORPORA = ("doc_a",) + NEGATIVE_CORPORA
TAU_GRID = e2.TAU_GRID
K4_ALPHA_GRID = e2.K4_ALPHA_GRID
FPR_CAPS = {"code": 0.05, "grm": 0.05, "guides": 0.10}
RECALL_FLOOR = 0.50
G3_IMPROVEMENT_FLOOR = 0.10
TEACHER_GAP_FLOOR_PPL = 0.5
G4_RECOVERY_FLOOR = 0.25
G5_WIKITEXT_PPL_DELTA_CAP_PERCENT = 0.5
G5_CODE_FIRE_CAP = 0.05
GENERIC_FIT_RANK_REFERENCE = 0.02
DEFAULT_SEED = 20260829
DEFAULT_BOOTSTRAP_RESAMPLES = 2000
EXPERTPACK_NAME = "expertpack_docA_olmoe_v0"
FORBIDDEN_PATH_TERMS = ("rcft", "consciousness", "thesis")
PREPARED_CONTENT_ALGORITHM = (
    "moe_e3_prepared_arrays_v1:sha256(canonical-json(name,shape,dtype,"
    "sha256(C-contiguous-little-endian-values)))"
)
PREPARED_ARRAY_SHAPES = {
    "doc_a_key_ids": (16, 512),
    "doc_b_ids": (16, 512),
    "pair_ids": (64, 512),
    "pair_prefix_ids": (64, 2048),
    "heldout_ids": (16, 512),
    "heldout_prefix_ids": (16, 2048),
    "wikitext_ids": (16, 512),
    "code_ids": (16, 512),
    "grm_ids": (16, 512),
    "guides_ids": (16, 512),
    "bringup_ids": (2048,),
}
GPU_PREAMBLE = (
    "flock -w 7200 /tmp/forge-gpu.lock "
    "timeout --signal=TERM --kill-after=5s 590s "
    "env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
    "HF_DEACTIVATE_ASYNC_LOAD=1 TOKENIZERS_PARALLELISM=false "
    "PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/olmoe_e3_pycache "
    "PYTHONUNBUFFERED=1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "prepare",
            "self-test",
            "bringup",
            "capture-keys",
            "fit-key",
            "capture-pairs",
            "train",
            "eval-gates",
            "precondition",
            "analyze",
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--max-gpu-memory", default="11GiB")
    parser.add_argument("--cpu-max-memory", default="46GiB")
    parser.add_argument("--cpu-threads", type=int, default=min(os.cpu_count() or 1, 12))
    parser.add_argument("--threads", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--corpus", choices=CAPTURE_CORPORA)
    parser.add_argument("--start-window", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument(
        "--eval-kind",
        choices=("abi", "doc-a-teacher", "doc-a-expert", "wikitext", "code", "doc-b"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--cg-rtol", type=float, default=1.0e-5)
    parser.add_argument("--cg-maxiter", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--token-batch-size", type=int, default=256)
    parser.add_argument("--train-tokens-per-window", type=int, default=128)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_output(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def prepared_content_record(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    records = {name: e2.array_content_record(value) for name, value in sorted(arrays.items())}
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "algorithm": PREPARED_CONTENT_ALGORITHM,
        "sha256": e2.sha256_bytes(canonical),
        "arrays": records,
    }


def content_fields(manifest: dict[str, Any]) -> dict[str, Any]:
    prepared = manifest["prepared_windows"]
    return {
        "prepared_windows_content_digest_algorithm": prepared["content_digest_algorithm"],
        "prepared_windows_content_sha256": prepared["content_sha256"],
    }


def validate_arrays(arrays: dict[str, np.ndarray]) -> None:
    if set(arrays) != set(PREPARED_ARRAY_SHAPES):
        raise RuntimeError(
            f"prepared arrays differ: observed={sorted(arrays)}, expected={sorted(PREPARED_ARRAY_SHAPES)}"
        )
    for name, shape in PREPARED_ARRAY_SHAPES.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != np.int64:
            raise RuntimeError(f"{name} contract {value.shape}/{value.dtype} != {shape}/int64")


def discover_documents() -> list[Path]:
    if not STORIES_ROOT.is_dir():
        raise FileNotFoundError(STORIES_ROOT)
    eligible: list[Path] = []
    for path in STORIES_ROOT.rglob("*_COMPLETE.txt"):
        lowered = str(path).lower()
        if any(term in lowered for term in FORBIDDEN_PATH_TERMS):
            continue
        if not path.is_file():
            continue
        eligible.append(path.resolve())
    eligible.sort(key=lambda path: (-int(path.stat().st_size), str(path)))
    if len(eligible) < 2:
        raise RuntimeError("fewer than two eligible *_COMPLETE.txt novels")
    return eligible


def document_record(path: Path, token_ids: np.ndarray, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "byte_count": int(path.stat().st_size),
        "sha256": e2.sha256_file(path),
        "token_count": int(token_ids.size),
        "full_512_window_count": int(token_ids.size // WINDOW_TOKENS),
        "trailing_tokens_dropped_from_windows": int(token_ids.size % WINDOW_TOKENS),
    }


def window_record(
    *,
    document: str,
    window_index: int,
    ids: np.ndarray,
    prefix: np.ndarray | None = None,
) -> dict[str, Any]:
    record = {
        "document": document,
        "window_index": int(window_index),
        "token_start": int(window_index * WINDOW_TOKENS),
        "token_stop_exclusive": int((window_index + 1) * WINDOW_TOKENS),
        "token_ids_sha256": e2.sha256_array(ids),
    }
    if prefix is not None:
        record.update(
            {
                "teacher_prefix_token_start": int(window_index * WINDOW_TOKENS - PREFIX_TOKENS),
                "teacher_prefix_token_stop_exclusive": int(window_index * WINDOW_TOKENS),
                "teacher_prefix_token_count": int(prefix.size),
                "teacher_prefix_ids_sha256": e2.sha256_array(prefix),
            }
        )
    return record


def load_e2_negative_arrays() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not E2_LEGACY_PREPARED.is_file() or not E2_LEGACY_CONTENT.is_file():
        raise FileNotFoundError("registered E2 legacy prepared lineage is missing")
    selected_source_names = (
        "narrative_key_ids",
        "wikitext_ids",
        "code_ids",
        "grm_ids",
        "bringup_ids",
    )
    with np.load(E2_LEGACY_PREPARED, allow_pickle=False) as archive:
        legacy = {name: archive[name].copy() for name in selected_source_names}
    validation = e2.read_json(E2_FIT_VALIDATION)
    binding = validation.get("validation", {}).get("binding", {})
    if (
        binding.get("content_sha256") != E2_LEGACY_CONTENT_SHA256
        or binding.get("content_digest_algorithm") != e2.PREPARED_CONTENT_ALGORITHM
    ):
        raise RuntimeError("E2 fit-key validation no longer binds the registered legacy lineage")
    for source_name in selected_source_names:
        observed = e2.array_content_record(legacy[source_name])
        if binding.get("arrays", {}).get(source_name) != observed:
            raise RuntimeError(f"E2 legacy array binding changed: {source_name}")
    arrays = {
        "guides_ids": np.ascontiguousarray(legacy["narrative_key_ids"], dtype=np.int64),
        "wikitext_ids": np.ascontiguousarray(legacy["wikitext_ids"], dtype=np.int64),
        "code_ids": np.ascontiguousarray(legacy["code_ids"], dtype=np.int64),
        "grm_ids": np.ascontiguousarray(legacy["grm_ids"], dtype=np.int64),
        "bringup_ids": np.ascontiguousarray(legacy["bringup_ids"], dtype=np.int64),
    }
    provenance = {
        "lineage": "E2 immutable key-capture input lineage",
        "legacy_prepared_path": str(E2_LEGACY_PREPARED),
        "legacy_prepared_file_sha256": e2.sha256_file(E2_LEGACY_PREPARED),
        "legacy_archive_sha256_recorded_by_receipts": E2_LEGACY_ARCHIVE_RECORDED_SHA256,
        "legacy_content_path": str(E2_LEGACY_CONTENT),
        "legacy_content_file_sha256": e2.sha256_file(E2_LEGACY_CONTENT),
        "legacy_content_digest_algorithm": binding["content_digest_algorithm"],
        "legacy_content_sha256": binding["content_sha256"],
        "fit_key_validation_path": str(E2_FIT_VALIDATION),
        "fit_key_validation_sha256": e2.sha256_file(E2_FIT_VALIDATION),
        "selected_arrays": {
            target: {
                "source_array": source,
                **e2.array_content_record(arrays[target]),
            }
            for target, source in {
                "guides_ids": "narrative_key_ids",
                "wikitext_ids": "wikitext_ids",
                "code_ids": "code_ids",
                "grm_ids": "grm_ids",
                "bringup_ids": "bringup_ids",
            }.items()
        },
    }
    return arrays, provenance


def initialize_expertpack(output: Path, manifest: dict[str, Any]) -> None:
    pack_dir = output / EXPERTPACK_NAME
    pack_dir.mkdir(parents=True, exist_ok=True)
    pack_path = pack_dir / "manifest.json"
    if pack_path.is_file():
        return
    pack = {
        "schema": EXPERTPACK_NAME,
        "status": "prepared_pending_address",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "document": manifest["documents"]["doc_a"],
        "layer": None,
        "rank": 64,
        "tau": None,
        "abi": "B @ silu(A @ h), added to native MLP output iff h @ key >= tau",
        "components": {
            "key": {"status": "pending"},
            "A": {"status": "pending", "shape": [64, HIDDEN_DIM], "storage_dtype": "float16"},
            "B": {
                "status": "pending_zero_init",
                "shape": [HIDDEN_DIM, 64],
                "storage_dtype": "float16",
            },
        },
        "provenance": {
            "order": str(ORDER_PATH),
            "order_sha256": e2.sha256_file(ORDER_PATH),
            "corpus_manifest": str(output / "corpus_manifest.json"),
            "corpus_manifest_sha256": e2.sha256_file(output / "corpus_manifest.json"),
            "prepared_windows_content": {
                "content_digest_algorithm": manifest["prepared_windows"]["content_digest_algorithm"],
                "content_sha256": manifest["prepared_windows"]["content_sha256"],
                "arrays": manifest["prepared_windows"]["arrays"],
            },
        },
        "registered_gates": {
            "E3-G2": {
                "doc_a_eval_recall_gte": RECALL_FLOOR,
                "code_eval_fpr_lte": FPR_CAPS["code"],
                "grm_eval_fpr_lte": FPR_CAPS["grm"],
                "guides_eval_fpr_lte": FPR_CAPS["guides"],
                "wikitext": "descriptive",
                "doc_b": "descriptive",
            },
            "E3-G3": {"validation_mse_improvement_gte": G3_IMPROVEMENT_FLOOR},
            "behavioral_precondition": {"teacher_gap_ppl_gte": TEACHER_GAP_FLOOR_PPL},
            "E3-G4": {
                "recovery_fraction_gte": G4_RECOVERY_FLOOR,
                "bootstrap_ci95_low_gt": 0.0,
            },
            "E3-G5": {
                "wikitext_ppl_delta_percent_lte": G5_WIKITEXT_PPL_DELTA_CAP_PERCENT,
                "code_fire_rate_lte": G5_CODE_FIRE_CAP,
            },
        },
        "limitations": ["address, adapter, ABI, and behavioral measurements pending"],
    }
    e2.write_json(pack_path, pack)


def prepare(args: argparse.Namespace) -> int:
    output = ensure_output(args.output_dir)
    manifest_path = output / "corpus_manifest.json"
    prepared_path = output / "prepared_windows.npz"
    if manifest_path.is_file() or prepared_path.is_file():
        if not (manifest_path.is_file() and prepared_path.is_file()):
            raise RuntimeError("partial prepare state: manifest and prepared archive must both exist")
        manifest, _arrays = load_prepared(output)
        initialize_expertpack(output, manifest)
        write_lead_script(output)
        print(
            json.dumps(
                {
                    "status": "existing",
                    "content_sha256": manifest["prepared_windows"]["content_sha256"],
                    "doc_a": manifest["documents"]["doc_a"],
                    "doc_b": manifest["documents"]["doc_b"],
                }
            ),
            flush=True,
        )
        return 0

    model_dir = e2.validate_model_dir(args.model_dir)
    documents = discover_documents()
    doc_a_path, doc_b_path = documents[:2]
    tokenizer = e2.load_tokenizer(model_dir)
    doc_a_tokens = e2.tokenizer_ids(tokenizer, doc_a_path.read_text(encoding="utf-8"))
    doc_b_tokens = e2.tokenizer_ids(tokenizer, doc_b_path.read_text(encoding="utf-8"))
    doc_a_record = document_record(doc_a_path, doc_a_tokens, "DOC-A")
    doc_b_record = document_record(doc_b_path, doc_b_tokens, "DOC-B")
    doc_a_full_windows = int(doc_a_tokens.size // WINDOW_TOKENS)
    doc_b_full_windows = int(doc_b_tokens.size // WINDOW_TOKENS)
    doc_a_heldout_windows = tuple(range(doc_a_full_windows - 16, doc_a_full_windows))
    doc_b_probe_windows = tuple(range(doc_b_full_windows - 16, doc_b_full_windows))
    if doc_a_full_windows < 84 or doc_b_full_windows < 20:
        raise RuntimeError("selected documents are too short for registered E3 regions")
    if set(PAIR_DOC_WINDOWS) & set(doc_a_heldout_windows):
        raise RuntimeError("DOC-A PAIR and HELDOUT regions overlap")
    if min(doc_a_heldout_windows) < 4:
        raise RuntimeError("DOC-A HELDOUT begins before a full teacher prefix is available")

    def window(tokens: np.ndarray, index: int) -> np.ndarray:
        start = index * WINDOW_TOKENS
        return np.ascontiguousarray(tokens[start : start + WINDOW_TOKENS], dtype=np.int64)

    def prefix(tokens: np.ndarray, index: int) -> np.ndarray:
        stop = index * WINDOW_TOKENS
        return np.ascontiguousarray(tokens[stop - PREFIX_TOKENS : stop], dtype=np.int64)

    arrays, e2_negative_provenance = load_e2_negative_arrays()
    arrays.update(
        {
            "doc_a_key_ids": np.stack([window(doc_a_tokens, index) for index in DOC_A_KEY_WINDOWS]),
            "doc_b_ids": np.stack([window(doc_b_tokens, index) for index in doc_b_probe_windows]),
            "pair_ids": np.stack([window(doc_a_tokens, index) for index in PAIR_DOC_WINDOWS]),
            "pair_prefix_ids": np.stack([prefix(doc_a_tokens, index) for index in PAIR_DOC_WINDOWS]),
            "heldout_ids": np.stack([window(doc_a_tokens, index) for index in doc_a_heldout_windows]),
            "heldout_prefix_ids": np.stack([prefix(doc_a_tokens, index) for index in doc_a_heldout_windows]),
        }
    )
    arrays = {name: np.ascontiguousarray(value, dtype=np.int64) for name, value in arrays.items()}
    validate_arrays(arrays)
    for local, source_index in enumerate(PAIR_DOC_WINDOWS):
        expected = doc_a_tokens[source_index * WINDOW_TOKENS - PREFIX_TOKENS : source_index * WINDOW_TOKENS]
        if not np.array_equal(arrays["pair_prefix_ids"][local], expected):
            raise RuntimeError("PAIR true-prefix contract failed")
    for local, source_index in enumerate(doc_a_heldout_windows):
        expected = doc_a_tokens[source_index * WINDOW_TOKENS - PREFIX_TOKENS : source_index * WINDOW_TOKENS]
        if not np.array_equal(arrays["heldout_prefix_ids"][local], expected):
            raise RuntimeError("HELDOUT true-prefix contract failed")

    content = prepared_content_record(arrays)
    e2.save_npz(prepared_path, compressed=True, **arrays)
    manifest = {
        "schema": "moe_e3_corpus_manifest_v1",
        "status": "complete",
        "created_at": now_iso(),
        "order": str(ORDER_PATH),
        "order_sha256": e2.sha256_file(ORDER_PATH),
        "script": str(SCRIPT_PATH),
        "script_sha256": e2.sha256_file(SCRIPT_PATH),
        "model_id": MODEL_ID,
        "model_dir": str(model_dir),
        "model_revision": MODEL_REVISION,
        "model_config_sha256": e2.sha256_file(model_dir / "config.json"),
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "add_special_tokens": False,
            "local_only": True,
        },
        "seed": args.seed,
        "eligibility": {
            "root": str(STORIES_ROOT),
            "filename_glob": "*_COMPLETE.txt",
            "forbidden_case_insensitive_path_terms": list(FORBIDDEN_PATH_TERMS),
            "eligible_file_count": len(documents),
            "selection": "descending byte_count, then ascending absolute path; take first two",
            "selected_byte_counts": [int(doc_a_path.stat().st_size), int(doc_b_path.stat().st_size)],
        },
        "documents": {"doc_a": doc_a_record, "doc_b": doc_b_record},
        "regions": {
            "window_tokens": WINDOW_TOKENS,
            "teacher_prefix_tokens": PREFIX_TOKENS,
            "doc_a_pair_source_windows": list(PAIR_DOC_WINDOWS),
            "doc_a_pair_local_split": {"train": [0, 47], "validation": [48, 63]},
            "doc_a_key_source_windows": list(DOC_A_KEY_WINDOWS),
            "doc_a_key_choice": (
                "first 16 complete windows after PAIR, disjoint from PAIR and HELDOUT; "
                "local even indices FIT and odd indices EVAL"
            ),
            "doc_a_heldout_source_windows": list(doc_a_heldout_windows),
            "doc_b_cross_probe_source_windows": list(doc_b_probe_windows),
        },
        "windows": {
            "doc_a_key": [
                window_record(document="DOC-A", window_index=index, ids=arrays["doc_a_key_ids"][local])
                for local, index in enumerate(DOC_A_KEY_WINDOWS)
            ],
            "pair": [
                window_record(
                    document="DOC-A",
                    window_index=index,
                    ids=arrays["pair_ids"][local],
                    prefix=arrays["pair_prefix_ids"][local],
                )
                for local, index in enumerate(PAIR_DOC_WINDOWS)
            ],
            "heldout": [
                window_record(
                    document="DOC-A",
                    window_index=index,
                    ids=arrays["heldout_ids"][local],
                    prefix=arrays["heldout_prefix_ids"][local],
                )
                for local, index in enumerate(doc_a_heldout_windows)
            ],
            "doc_b_cross_probe": [
                window_record(document="DOC-B", window_index=index, ids=arrays["doc_b_ids"][local])
                for local, index in enumerate(doc_b_probe_windows)
            ],
        },
        "key_split": {
            "fit_local_indices": list(FIT_INDICES),
            "eval_local_indices": list(EVAL_INDICES),
            "inner_train_local_indices": list(INNER_TRAIN_INDICES),
            "inner_validation_local_indices": list(INNER_VALIDATION_INDICES),
            "fit_eval_overlap": [],
        },
        "e2_negative_pool": e2_negative_provenance,
        "prepared_windows": {
            "path": str(prepared_path),
            "file_sha256": e2.sha256_file(prepared_path),
            "content_digest_algorithm": content["algorithm"],
            "content_sha256": content["sha256"],
            "arrays": content["arrays"],
        },
    }
    e2.write_json(manifest_path, manifest)
    initialize_expertpack(output, manifest)
    write_lead_script(output)
    print(
        json.dumps(
            {
                "status": "complete",
                "content_sha256": content["sha256"],
                "doc_a": doc_a_record,
                "doc_b": doc_b_record,
            }
        ),
        flush=True,
    )
    return 0


def load_prepared(output: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = output / "corpus_manifest.json"
    prepared_path = output / "prepared_windows.npz"
    manifest = e2.read_json(manifest_path)
    if (
        manifest.get("schema") != "moe_e3_corpus_manifest_v1"
        or manifest.get("status") != "complete"
        or manifest.get("model_revision") != MODEL_REVISION
    ):
        raise RuntimeError("E3 corpus manifest contract failed")
    declared = manifest.get("prepared_windows", {})
    if Path(declared.get("path", "")).resolve() != prepared_path.resolve():
        raise RuntimeError("prepared archive path linkage failed")
    if declared.get("file_sha256") != e2.sha256_file(prepared_path):
        raise RuntimeError("prepared archive file SHA changed")
    with np.load(prepared_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    validate_arrays(arrays)
    content = prepared_content_record(arrays)
    if (
        declared.get("content_digest_algorithm") != content["algorithm"]
        or declared.get("content_sha256") != content["sha256"]
        or declared.get("arrays") != content["arrays"]
    ):
        raise RuntimeError("prepared content-addressed provenance failed")
    selected = discover_documents()[:2]
    for role, path in zip(("doc_a", "doc_b"), selected):
        record = manifest["documents"][role]
        if (
            record.get("path") != str(path)
            or record.get("byte_count") != int(path.stat().st_size)
            or record.get("sha256") != e2.sha256_file(path)
        ):
            raise RuntimeError(f"{role} identity or deterministic selection changed")
    e2_arrays, e2_provenance = load_e2_negative_arrays()
    for name, value in e2_arrays.items():
        if not np.array_equal(arrays[name], value):
            raise RuntimeError(f"E2 inherited negative input changed: {name}")
    if manifest.get("e2_negative_pool", {}).get("legacy_content_sha256") != e2_provenance["legacy_content_sha256"]:
        raise RuntimeError("E2 negative lineage declaration changed")
    return manifest, arrays


def bringup(args: argparse.Namespace) -> int:
    output = ensure_output(args.output_dir)
    manifest, arrays = load_prepared(output)
    receipt_path = output / "bringup.json"
    if receipt_path.is_file():
        receipt = e2.read_json(receipt_path)
        if (
            receipt.get("schema") != "moe_e3_bringup_v1"
            or receipt.get("prepared_windows_content_sha256")
            != manifest["prepared_windows"]["content_sha256"]
        ):
            raise RuntimeError("existing E3 bringup receipt contract failed")
        print(json.dumps({"status": "existing", "gate": "E3-G-1", "verdict": receipt["gate"]["verdict"]}))
        return 0 if receipt["gate"]["verdict"] == "GREEN" else 3

    inheritance_error: str | None = None
    inherited: dict[str, Any] | None = None
    try:
        e2_manifest, e2_prepared = e2.load_prepared(E2_OUTPUT)
        validation = e2.validate_consolidated_bringup_receipt(E2_OUTPUT, e2_manifest, e2_prepared)
        source = validation["receipt"]
        if source.get("gate", {}).get("verdict") != "GREEN":
            raise RuntimeError("E2-G-1 source receipt is not GREEN")
        if not np.array_equal(arrays["bringup_ids"], e2_prepared["bringup_ids"]):
            raise RuntimeError("E3 bringup IDs differ from validated E2 bringup IDs")
        inherited = {
            "mode": "inherited_content_valid_E2_G1",
            "source_receipt": str(E2_OUTPUT / "bringup.json"),
            "source_receipt_sha256": e2.sha256_file(E2_OUTPUT / "bringup.json"),
            "source_validation_binding": {
                key: validation["binding"].get(key)
                for key in ("status", "mode", "content_label", "content_sha256", "dependency_kind")
            },
            "source_short_cell": source["cell_receipts"]["short"],
            "source_long_cell": source["cell_receipts"]["long"],
            "short_512": source["short_512"],
            "long_2048": source["long_2048"],
            "long_minus_short_mean_nll": source["long_minus_short_mean_nll"],
            "gate": source["gate"],
        }
    except Exception as exc:
        inheritance_error = f"{type(exc).__name__}: {exc}"

    if inherited is None:
        torch, model, runtime = e2.load_model(args)
        short = e2.score_target_window(torch, model, arrays["bringup_ids"][:512], arrays["bringup_ids"][:512])
        long = e2.score_target_window(torch, model, arrays["bringup_ids"][:2048], arrays["bringup_ids"][:2048])
        delta = float(long["mean_nll"] - short["mean_nll"])
        plausible = e2.BRINGUP_PPL_MIN <= short["ppl"] <= e2.BRINGUP_PPL_MAX
        context_ok = delta <= e2.BRINGUP_LONG_MINUS_SHORT_NLL_CAP
        inherited = {
            "mode": "rerun_after_invalid_E2_inheritance",
            "inheritance_error": inheritance_error,
            "runtime": runtime,
            "short_512": short,
            "long_2048": long,
            "long_minus_short_mean_nll": delta,
            "gate": {
                "verdict": "GREEN" if plausible and context_ok else "RED",
                "short_ppl_plausible": plausible,
                "long_context_delta_ok": context_ok,
            },
        }
        del model
        gc.collect()

    receipt = {
        "schema": "moe_e3_bringup_v1",
        "status": "complete",
        "created_at": now_iso(),
        "gate_name": "E3-G-1",
        "gate": inherited["gate"],
        "inheritance": {key: value for key, value in inherited.items() if key not in {"gate", "short_512", "long_2048", "long_minus_short_mean_nll"}},
        "short_512": inherited["short_512"],
        "long_2048": inherited["long_2048"],
        "long_minus_short_mean_nll": inherited["long_minus_short_mean_nll"],
        "registered_gate": {
            "short_ppl_plausible_range_inclusive": [e2.BRINGUP_PPL_MIN, e2.BRINGUP_PPL_MAX],
            "long_minus_short_mean_nll_lte": e2.BRINGUP_LONG_MINUS_SHORT_NLL_CAP,
        },
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        **content_fields(manifest),
        "script": str(SCRIPT_PATH),
        "script_sha256": e2.sha256_file(SCRIPT_PATH),
    }
    e2.write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "gate": "E3-G-1",
                "verdict": receipt["gate"]["verdict"],
                "mode": inherited["mode"],
                "ppl512": receipt["short_512"]["ppl"],
                "nll_delta": receipt["long_minus_short_mean_nll"],
            }
        ),
        flush=True,
    )
    return 0 if receipt["gate"]["verdict"] == "GREEN" else 3


def require_bringup_green(output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    receipt = e2.read_json(output / "bringup.json")
    if (
        receipt.get("schema") != "moe_e3_bringup_v1"
        or receipt.get("gate", {}).get("verdict") != "GREEN"
        or receipt.get("prepared_windows_content_sha256")
        != manifest["prepared_windows"]["content_sha256"]
    ):
        raise RuntimeError("E3-G-1 is not content-valid GREEN; STOP")
    return receipt


def key_capture_paths(output: Path, corpus: str, window_index: int) -> tuple[Path, Path]:
    root = output / "key_captures"
    stem = f"{corpus}_{window_index:02d}"
    return root / f"{stem}_router_inputs_fp16.npy", root / f"{stem}_receipt.json"


def validate_key_capture(
    output: Path,
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    corpus: str,
    window_index: int,
) -> dict[str, Any]:
    data_path, receipt_path = key_capture_paths(output, corpus, window_index)
    if not data_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"missing E3 key capture {corpus}/{window_index}")
    receipt = e2.read_json(receipt_path)
    array_name = "doc_a_key_ids" if corpus == "doc_a" else "doc_b_ids"
    if (
        receipt.get("schema") != "moe_e3_router_input_capture_v1"
        or receipt.get("status") != "complete"
        or receipt.get("corpus") != corpus
        or receipt.get("window_index") != window_index
        or receipt.get("model_revision") != MODEL_REVISION
        or receipt.get("input_ids_sha256") != e2.sha256_array(arrays[array_name][window_index])
        or receipt.get("prepared_windows_content_sha256")
        != manifest["prepared_windows"]["content_sha256"]
        or Path(receipt.get("capture_path", "")).resolve() != data_path.resolve()
        or receipt.get("capture_file_sha256") != e2.sha256_file(data_path)
    ):
        raise RuntimeError(f"E3 key capture receipt contract failed: {receipt_path}")
    payload = np.load(data_path, mmap_mode="r", allow_pickle=False)
    if payload.shape != (N_LAYERS, WINDOW_TOKENS, HIDDEN_DIM) or payload.dtype != np.float16:
        raise RuntimeError(f"E3 key capture payload contract failed: {data_path}")
    return {
        "corpus": corpus,
        "window_index": window_index,
        "capture_path": str(data_path),
        "capture_sha256": receipt["capture_file_sha256"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": e2.sha256_file(receipt_path),
        "source": "E3 fresh DOC capture",
    }


def e2_negative_capture_paths(corpus: str, window_index: int) -> tuple[Path, Path, Path]:
    e2_corpus = "narrative" if corpus == "guides" else corpus
    root = E2_OUTPUT / "key_captures"
    stem = f"{e2_corpus}_{window_index:02d}"
    receipt = root / f"{stem}_receipt.json"
    return (
        root / f"{stem}_router_inputs_fp16.npy",
        receipt,
        receipt.with_name(receipt.stem + ".provenance_validation.json"),
    )


def validate_e2_negative_capture(
    arrays: dict[str, np.ndarray], corpus: str, window_index: int
) -> dict[str, Any]:
    data_path, receipt_path, sidecar_path = e2_negative_capture_paths(corpus, window_index)
    if not data_path.is_file() or not receipt_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"missing E2 negative capture lineage {corpus}/{window_index}")
    receipt = e2.read_json(receipt_path)
    sidecar = e2.read_json(sidecar_path)
    validation = sidecar.get("validation", {})
    binding = validation.get("binding", {})
    expected_e2_corpus = "narrative" if corpus == "guides" else corpus
    if (
        receipt.get("status") != "complete"
        or receipt.get("corpus") != expected_e2_corpus
        or receipt.get("window_index") != window_index
        or receipt.get("model_revision") != MODEL_REVISION
        or receipt.get("input_ids_sha256") != e2.sha256_array(arrays[f"{corpus}_ids"][window_index])
        or receipt.get("capture_file_sha256") != e2.sha256_file(data_path)
        or sidecar.get("status") != "passed"
        or binding.get("status") != "passed"
        or binding.get("content_sha256") != E2_LEGACY_CONTENT_SHA256
        or validation.get("payload_sha256") != receipt.get("capture_file_sha256")
        or sidecar.get("original_receipt", {}).get("sha256") != e2.sha256_file(receipt_path)
    ):
        raise RuntimeError(f"E2 negative capture provenance failed: {receipt_path}")
    payload = np.load(data_path, mmap_mode="r", allow_pickle=False)
    if payload.shape != (N_LAYERS, WINDOW_TOKENS, HIDDEN_DIM) or payload.dtype != np.float16:
        raise RuntimeError(f"E2 negative capture payload contract failed: {data_path}")
    return {
        "corpus": corpus,
        "e2_corpus": expected_e2_corpus,
        "window_index": window_index,
        "capture_path": str(data_path),
        "capture_sha256": receipt["capture_file_sha256"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": e2.sha256_file(receipt_path),
        "provenance_validation_path": str(sidecar_path),
        "provenance_validation_sha256": e2.sha256_file(sidecar_path),
        "content_sha256": binding["content_sha256"],
        "source": "E2 immutable validated negative capture",
    }


def capture_keys(args: argparse.Namespace) -> int:
    if args.corpus is None:
        raise ValueError("capture-keys requires --corpus doc_a|doc_b")
    if args.start_window < 0 or args.count < 1 or args.start_window + args.count > N_KEY_WINDOWS:
        raise ValueError("capture-keys range must stay within local windows 0..15")
    output = ensure_output(args.output_dir)
    manifest, arrays = load_prepared(output)
    require_bringup_green(output, manifest)
    requested = list(range(args.start_window, args.start_window + args.count))
    missing: list[int] = []
    for index in requested:
        _data_path, receipt_path = key_capture_paths(output, args.corpus, index)
        if receipt_path.is_file():
            validate_key_capture(output, manifest, arrays, args.corpus, index)
        else:
            missing.append(index)
    if not missing:
        print(json.dumps({"status": "existing", "corpus": args.corpus, "windows": requested}))
        return 0

    started = time.perf_counter()
    torch, model, runtime = e2.load_model(args)
    captured: list[np.ndarray | None] = [None] * N_LAYERS
    handles: list[Any] = []

    def make_hook(layer_index: int):
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            captured[layer_index] = np.ascontiguousarray(
                inputs[0].detach().to(device="cpu", dtype=torch.float16).numpy()
            )

        return hook

    for layer_index, layer in enumerate(model.model.layers):
        handles.append(layer.mlp.register_forward_pre_hook(make_hook(layer_index)))
    array_name = "doc_a_key_ids" if args.corpus == "doc_a" else "doc_b_ids"
    ids_np = np.ascontiguousarray(arrays[array_name][missing], dtype=np.int64)
    ids = torch.from_numpy(ids_np).to(e2.input_device(model))
    try:
        with torch.inference_mode():
            output_object = model(input_ids=ids, use_cache=False, logits_to_keep=1)
        del output_object, ids
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in captured):
        raise RuntimeError("not every OLMoE layer executed its router-input hook")
    stacked = np.stack([value for value in captured if value is not None], axis=0)
    expected = (N_LAYERS, len(missing), WINDOW_TOKENS, HIDDEN_DIM)
    if stacked.shape != expected or stacked.dtype != np.float16 or not np.isfinite(stacked).all():
        raise RuntimeError(f"router capture contract {stacked.shape}/{stacked.dtype} != {expected}/float16")
    invocation = {
        "started_at": now_iso(),
        "window_indices": missing,
        "wall_seconds": time.perf_counter() - started,
        "runtime": runtime,
        "peak_rss_bytes": e2.peak_rss_bytes(),
    }
    for local_index, window_index in enumerate(missing):
        payload = np.ascontiguousarray(stacked[:, local_index], dtype=np.float16)
        data_path, receipt_path = key_capture_paths(output, args.corpus, window_index)
        e2.save_npy(data_path, payload)
        source_key = "doc_a_key" if args.corpus == "doc_a" else "doc_b_cross_probe"
        receipt = {
            "schema": "moe_e3_router_input_capture_v1",
            "status": "complete",
            "created_at": now_iso(),
            "corpus": args.corpus,
            "window_index": window_index,
            "source_window": manifest["windows"][source_key][window_index],
            "input_ids_sha256": e2.sha256_array(arrays[array_name][window_index]),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            **content_fields(manifest),
            "hook": "OlmoeDecoderLayer.mlp forward_pre_hook; exact native router input",
            "capture_shape": list(payload.shape),
            "capture_dtype": str(payload.dtype),
            "capture_payload_nbytes": int(payload.nbytes),
            "capture_path": str(data_path),
            "capture_file_sha256": e2.sha256_file(data_path),
            "invocation": invocation,
            "script": str(SCRIPT_PATH),
            "script_sha256": e2.sha256_file(SCRIPT_PATH),
        }
        e2.write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "status": "complete",
                "corpus": args.corpus,
                "windows": missing,
                "shape": list(stacked.shape),
                "wall_seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )
    del model, stacked, captured
    gc.collect()
    return 0


def collect_capture_provenance(
    output: Path, manifest: dict[str, Any], arrays: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    provenance: list[dict[str, Any]] = []
    for corpus in CAPTURE_CORPORA:
        for index in range(N_KEY_WINDOWS):
            provenance.append(validate_key_capture(output, manifest, arrays, corpus, index))
    for corpus in NEGATIVE_CORPORA:
        for index in range(N_KEY_WINDOWS):
            provenance.append(validate_e2_negative_capture(arrays, corpus, index))
    return provenance


def capture_payload_path(output: Path, corpus: str, index: int) -> Path:
    if corpus in CAPTURE_CORPORA:
        return key_capture_paths(output, corpus, index)[0]
    return e2_negative_capture_paths(corpus, index)[0]


def materialize_layer(output: Path, corpus: str, layer_index: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    for index in range(N_KEY_WINDOWS):
        payload = np.load(capture_payload_path(output, corpus, index), mmap_mode="r", allow_pickle=False)
        value = np.ascontiguousarray(payload[layer_index], dtype=np.float16)
        if value.shape != (WINDOW_TOKENS, HIDDEN_DIM) or not np.isfinite(value).all():
            raise RuntimeError(f"non-finite or malformed capture {corpus}/{index}/L{layer_index}")
        rows.append(value)
    return np.stack(rows)


def selected_tokens(array: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    dimension = int(np.asarray(array).shape[-1])
    return np.ascontiguousarray(
        np.asarray(array)[np.asarray(indices, dtype=np.int64)].reshape(-1, dimension),
        dtype=np.float32,
    )


def centered_five_corpora(
    hidden: dict[str, np.ndarray], indices: Sequence[int]
) -> tuple[np.ndarray, int, np.ndarray, float]:
    blocks = {corpus: selected_tokens(hidden[corpus], indices) for corpus in FIT_CORPORA}
    counts = {corpus: int(value.shape[0]) for corpus, value in blocks.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"K4 requires equal corpus token counts: {counts}")
    means = {corpus: value.mean(axis=0, dtype=np.float64) for corpus, value in blocks.items()}
    centered = np.concatenate(
        [
            np.ascontiguousarray(blocks[corpus] - means[corpus].astype(np.float32), dtype=np.float32)
            for corpus in FIT_CORPORA
        ],
        axis=0,
    )
    dof = int(centered.shape[0] - len(FIT_CORPORA))
    diagonal = np.sum(centered.astype(np.float64) ** 2, axis=0) / float(dof)
    negative_mean = np.mean(np.stack([means[corpus] for corpus in NEGATIVE_CORPORA]), axis=0)
    return centered, dof, means["doc_a"] - negative_mean, float(diagonal.mean())


def validation_auc_five(hidden: dict[str, np.ndarray], key: np.ndarray) -> dict[str, Any]:
    positive = selected_tokens(hidden["doc_a"], INNER_VALIDATION_INDICES) @ key
    aucs = {
        corpus: e2.exact_auc(
            positive,
            selected_tokens(hidden[corpus], INNER_VALIDATION_INDICES) @ key,
        )
        for corpus in NEGATIVE_CORPORA
    }
    return {
        **{f"auc_doc_a_vs_{corpus}": value for corpus, value in aucs.items()},
        "minimum_auc": min(aucs.values()),
        "mean_auc": float(np.mean(list(aucs.values()))),
    }


def fit_k4_direction_five(
    hidden: dict[str, np.ndarray], *, cg_rtol: float, cg_maxiter: int
) -> tuple[np.ndarray, dict[str, Any]]:
    centered, dof, delta, mean_variance = centered_five_corpora(hidden, INNER_TRAIN_INDICES)
    candidates_by_alpha: dict[float, dict[str, Any]] = {}
    warm: np.ndarray | None = None
    for alpha in reversed(K4_ALPHA_GRID):
        regularization = float(alpha * mean_variance)
        solution, solver = e2.cg_fisher_solve(
            centered, dof, delta, regularization, rtol=cg_rtol, maxiter=cg_maxiter, x0=warm
        )
        if not solver["converged"]:
            first = solver
            solution, solver = e2.cg_fisher_solve(
                centered,
                dof,
                delta,
                regularization,
                rtol=cg_rtol,
                maxiter=cg_maxiter * 4,
                x0=solution,
            )
            solver["initial_attempt"] = first
        candidate: dict[str, Any] = {
            "alpha": alpha,
            "lambda": regularization,
            "solver": solver,
            "fit_converged": bool(solver["converged"]),
        }
        if solver["converged"]:
            key, norm = e2.unit_vector(solution, name=f"E3 K4 alpha={alpha}")
            candidate["direction_source_norm"] = norm
            candidate["validation"] = validation_auc_five(hidden, key)
            warm = solution
        else:
            warm = None
        candidates_by_alpha[alpha] = candidate
    del centered
    gc.collect()
    candidates = [candidates_by_alpha[alpha] for alpha in K4_ALPHA_GRID]
    converged = [item for item in candidates if item["fit_converged"]]
    if not converged:
        raise RuntimeError("no converged E3 K4 shrinkage candidate")
    selected = max(
        converged,
        key=lambda item: (
            item["validation"]["minimum_auc"],
            item["validation"]["mean_auc"],
            item["alpha"],
        ),
    )
    selected_alpha = float(selected["alpha"])
    centered_full, dof_full, delta_full, variance_full = centered_five_corpora(hidden, FIT_INDICES)
    selected_lambda = float(selected_alpha * variance_full)
    solution, final_solver = e2.cg_fisher_solve(
        centered_full,
        dof_full,
        delta_full,
        selected_lambda,
        rtol=cg_rtol,
        maxiter=cg_maxiter,
    )
    if not final_solver["converged"]:
        first = final_solver
        solution, final_solver = e2.cg_fisher_solve(
            centered_full,
            dof_full,
            delta_full,
            selected_lambda,
            rtol=cg_rtol,
            maxiter=cg_maxiter * 4,
            x0=solution,
        )
        final_solver["initial_attempt"] = first
    del centered_full
    gc.collect()
    if not final_solver["converged"]:
        raise RuntimeError("final E3 full-FIT K4 solve did not converge")
    key, source_norm = e2.unit_vector(solution, name="E3 K4 final direction")
    return key, {
        "construction": (
            "unit_norm((pooled within-five-corpus FIT covariance + lambda I)^-1 "
            "(mean(DOC-A_FIT) - equal_mean(wikitext,code,grm,guides)_FIT))"
        ),
        "negative_pool_weighting": "equal tokens and one-fourth mean weight per negative corpus",
        "alpha_grid": list(K4_ALPHA_GRID),
        "inner_train_window_indices": list(INNER_TRAIN_INDICES),
        "inner_validation_window_indices": list(INNER_VALIDATION_INDICES),
        "selection_rule": (
            "highest minimum validation AUC over four negatives, then mean AUC, then stronger shrinkage"
        ),
        "validation_candidates": candidates,
        "selected_alpha": selected_alpha,
        "selected_full_fit_lambda": selected_lambda,
        "full_fit_mean_within_variance": variance_full,
        "full_fit_dof": dof_full,
        "final_solver": final_solver,
        "source_norm": source_norm,
        "stored_norm": float(np.linalg.norm(key.astype(np.float64))),
        "key_sha256": e2.sha256_array(key),
        "selected_validation": selected["validation"],
    }


def score_hidden(hidden: dict[str, np.ndarray], key: np.ndarray) -> dict[str, np.ndarray]:
    return {
        corpus: np.ascontiguousarray(
            (value.astype(np.float32).reshape(-1, key.size) @ key).reshape(
                value.shape[0], value.shape[1]
            ),
            dtype=np.float32,
        )
        for corpus, value in hidden.items()
    }


def select_tau_five(scores: dict[str, np.ndarray]) -> dict[str, Any]:
    fit = np.asarray(FIT_INDICES, dtype=np.int64)
    pooled = np.concatenate([scores[corpus][fit].reshape(-1) for corpus in NEGATIVE_CORPORA])
    curve: list[dict[str, Any]] = []
    for label, quantile in TAU_GRID:
        tau = float(np.quantile(pooled, quantile, method="linear"))
        rates = {
            "doc_a_recall": float(np.mean(scores["doc_a"][fit] >= tau)),
            **{
                f"{corpus}_fpr": float(np.mean(scores[corpus][fit] >= tau))
                for corpus in NEGATIVE_CORPORA
            },
        }
        feasible = all(rates[f"{corpus}_fpr"] <= cap for corpus, cap in FPR_CAPS.items())
        curve.append(
            {
                "quantile_label": label,
                "quantile": quantile,
                "tau": tau,
                "fit": rates,
                "fit_code_grm_guides_feasible": feasible,
            }
        )
    feasible_rows = [row for row in curve if row["fit_code_grm_guides_feasible"]]
    pool = feasible_rows if feasible_rows else curve
    selected = max(
        pool,
        key=lambda row: (
            row["fit"]["doc_a_recall"]
            if feasible_rows
            else -max(row["fit"][f"{corpus}_fpr"] / FPR_CAPS[corpus] for corpus in FPR_CAPS),
            -row["fit"]["guides_fpr"],
            -row["fit"]["grm_fpr"],
            -row["fit"]["code_fpr"],
            -row["fit"]["wikitext_fpr"],
            row["quantile"],
        ),
    )
    return {
        "tau_definition": (
            "linear p90/p95/p99/p99.5 over pooled wikitext+code+grm+guides FIT scores; "
            "maximize DOC-A FIT recall subject to code+grm+guides FIT caps; WikiText descriptive"
        ),
        "fit_constraint_corpora": list(FPR_CAPS),
        "wikitext_fire_rate": "descriptive_no_pass_fail_bound",
        "fire_comparison": "score >= tau",
        "fit_negative_token_count": int(pooled.size),
        "fit_fpr_feasible": bool(feasible_rows),
        "selected_quantile_label": selected["quantile_label"],
        "selected_quantile": selected["quantile"],
        "selected_tau": selected["tau"],
        "selected_fit_rates": selected["fit"],
        "operating_curve": curve,
    }


def analyze_key_scores_five(
    scores: dict[str, np.ndarray], doc_b_scores: np.ndarray, *, resamples: int, seed: int
) -> dict[str, Any]:
    threshold = select_tau_five(scores)
    tau = float(threshold["selected_tau"])
    evaluation = np.asarray(EVAL_INDICES, dtype=np.int64)
    fired = {corpus: scores[corpus][evaluation] >= tau for corpus in FIT_CORPORA}
    eval_metrics: dict[str, Any] = {
        "doc_a_recall": e2.bootstrap_rate(
            fired["doc_a"], resamples=resamples, seed=seed + 1, window_indices=EVAL_INDICES
        ),
        **{
            f"{corpus}_fpr": e2.bootstrap_rate(
                fired[corpus],
                resamples=resamples,
                seed=seed + 10 + index,
                window_indices=EVAL_INDICES,
            )
            for index, corpus in enumerate(NEGATIVE_CORPORA)
        },
        "auc": {
            corpus: e2.exact_auc(scores["doc_a"][evaluation], scores[corpus][evaluation])
            for corpus in NEGATIVE_CORPORA
        },
    }
    eval_metrics["generic_fire_rate"] = dict(eval_metrics["wikitext_fpr"])
    doc_b_fired = doc_b_scores >= tau
    doc_b = e2.bootstrap_rate(
        doc_b_fired,
        resamples=resamples,
        seed=seed + 90,
        window_indices=range(N_KEY_WINDOWS),
    )
    qualifies = bool(
        threshold["fit_fpr_feasible"]
        and eval_metrics["doc_a_recall"]["value"] >= RECALL_FLOOR
        and all(eval_metrics[f"{corpus}_fpr"]["value"] <= cap for corpus, cap in FPR_CAPS.items())
    )
    return {
        **threshold,
        "eval_window_indices": list(EVAL_INDICES),
        "eval": eval_metrics,
        "doc_b_cross_episode_fire_descriptive": doc_b,
        "qualifies": qualifies,
        "registered_criteria": {
            "eval_doc_a_recall_gte": RECALL_FLOOR,
            "eval_code_fpr_lte": FPR_CAPS["code"],
            "eval_grm_fpr_lte": FPR_CAPS["grm"],
            "eval_guides_fpr_lte": FPR_CAPS["guides"],
            "wikitext_fire_rate": "descriptive",
            "doc_b_fire_rate": "descriptive_not_used_for_selection",
            "fit_threshold_frozen_before_eval": True,
        },
    }


def fit_side_layer_rank(row: dict[str, Any]) -> tuple[float, ...]:
    fit = row["metrics"]["selected_fit_rates"]
    validation = row["key_metadata"]["selected_validation"]
    worst = max(
        fit["wikitext_fpr"] / GENERIC_FIT_RANK_REFERENCE,
        *(fit[f"{corpus}_fpr"] / FPR_CAPS[corpus] for corpus in FPR_CAPS),
    )
    return (
        float(row["metrics"]["fit_fpr_feasible"]),
        float(validation["minimum_auc"]),
        float(validation["mean_auc"]),
        float(fit["doc_a_recall"]),
        -float(worst),
        -float(row["layer"]),
    )


def fit_key(args: argparse.Namespace) -> int:
    if args.bootstrap_resamples < 2000:
        raise ValueError("fit-key requires at least 2,000 window bootstrap resamples")
    if args.rank != 64:
        raise ValueError("E3 adapter rank is frozen at r=64")
    output = ensure_output(args.output_dir)
    manifest, arrays = load_prepared(output)
    require_bringup_green(output, manifest)
    fit_path = output / "fit_key.json"
    if fit_path.is_file():
        receipt = validate_fit_key_inputs(output, manifest, arrays)
        print(json.dumps({"status": "existing", "gate": "E3-G2", "verdict": receipt["decision"]["verdict"]}))
        return 0 if receipt["decision"]["verdict"] == "GREEN" else 3
    started = time.perf_counter()
    capture_provenance = collect_capture_provenance(output, manifest, arrays)
    rows: list[dict[str, Any]] = []
    keys: list[np.ndarray] = []
    taus: list[float] = []
    alphas: list[float] = []
    lambdas: list[float] = []
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        @contextlib.contextmanager
        def threadpool_limits(**_kwargs: Any):
            yield
    with threadpool_limits(limits=args.threads):
        for layer_index in range(N_LAYERS):
            layer_started = time.perf_counter()
            hidden = {corpus: materialize_layer(output, corpus, layer_index) for corpus in FIT_CORPORA}
            doc_b_hidden = materialize_layer(output, "doc_b", layer_index)
            key, key_metadata = fit_k4_direction_five(
                hidden, cg_rtol=args.cg_rtol, cg_maxiter=args.cg_maxiter
            )
            scores = score_hidden(hidden, key)
            doc_b_scores = score_hidden({"doc_b": doc_b_hidden}, key)["doc_b"]
            metrics = analyze_key_scores_five(
                scores,
                doc_b_scores,
                resamples=args.bootstrap_resamples,
                seed=args.seed + 1000 * layer_index,
            )
            row = {
                "layer": layer_index,
                "key_metadata": key_metadata,
                "metrics": metrics,
                "wall_seconds": time.perf_counter() - layer_started,
            }
            row["fit_side_rank"] = list(fit_side_layer_rank(row))
            rows.append(row)
            keys.append(key)
            taus.append(float(metrics["selected_tau"]))
            alphas.append(float(key_metadata["selected_alpha"]))
            lambdas.append(float(key_metadata["selected_full_fit_lambda"]))
            print(
                json.dumps(
                    {
                        "layer": layer_index,
                        "qualifies": metrics["qualifies"],
                        "recall": metrics["eval"]["doc_a_recall"]["value"],
                        "wikitext_fire": metrics["eval"]["wikitext_fpr"]["value"],
                        "code_fpr": metrics["eval"]["code_fpr"]["value"],
                        "grm_fpr": metrics["eval"]["grm_fpr"]["value"],
                        "guides_fpr": metrics["eval"]["guides_fpr"]["value"],
                        "doc_b_fire": metrics["doc_b_cross_episode_fire_descriptive"]["value"],
                    }
                ),
                flush=True,
            )
            del hidden, doc_b_hidden, scores, doc_b_scores
            gc.collect()
    keys_path = output / "all_layer_k4_keys.npz"
    e2.save_npz(
        keys_path,
        K4=np.stack(keys).astype(np.float32),
        tau=np.asarray(taus, dtype=np.float64),
        selected_alpha=np.asarray(alphas, dtype=np.float64),
        selected_lambda=np.asarray(lambdas, dtype=np.float64),
    )
    qualifiers = sorted(
        [row for row in rows if row["metrics"]["qualifies"]],
        key=fit_side_layer_rank,
        reverse=True,
    )
    selected = qualifiers[0] if qualifiers else None
    decision = {
        "verdict": "GREEN" if selected is not None else "RED",
        "qualifying_layer_count": len(qualifiers),
        "qualifying_layers_fit_ranked": [int(row["layer"]) for row in qualifiers],
        "install_layer": None if selected is None else int(selected["layer"]),
        "selection_uses_eval_only_as_registered_qualification_filter": True,
        "doc_b_not_used_for_threshold_layer_or_gate": True,
        "within_qualifiers_selection_rule": (
            "FIT-only carried rank: constrained FIT feasibility, inner-validation minimum/mean "
            "AUC, FIT recall, lower worst normalized FIT rate, lower layer"
        ),
    }
    receipt = {
        "schema": "moe_e3_fit_key_v1",
        "status": "complete_g2_green" if selected else "complete_g2_red_stop",
        "created_at": now_iso(),
        "gate": "E3-G2",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        **content_fields(manifest),
        "seed": args.seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "input_integrity": {
            "capture_count": len(capture_provenance),
            "expected_capture_count": (len(CAPTURE_CORPORA) + len(NEGATIVE_CORPORA)) * N_KEY_WINDOWS,
            "captures": capture_provenance,
            "e2_negative_content_sha256": E2_LEGACY_CONTENT_SHA256,
        },
        "split_provenance": {
            "fit_window_indices": list(FIT_INDICES),
            "eval_window_indices": list(EVAL_INDICES),
            "inner_train_window_indices": list(INNER_TRAIN_INDICES),
            "inner_validation_window_indices": list(INNER_VALIDATION_INDICES),
            "eval_used_for_key_tau_or_hyperparameter_selection": False,
            "doc_b_used_for_selection": False,
            "doc_a_behavioral_heldout_used": False,
        },
        "registered_rule": {
            "eval_doc_a_recall_gte": RECALL_FLOOR,
            "eval_code_fpr_lte": FPR_CAPS["code"],
            "eval_grm_fpr_lte": FPR_CAPS["grm"],
            "eval_guides_fpr_lte": FPR_CAPS["guides"],
            "wikitext_fire": "descriptive",
            "doc_b_fire": "descriptive",
            "tau_candidate_pool": list(NEGATIVE_CORPORA),
            "frozen_fit_side_tau": True,
        },
        "rows": rows,
        "decision": decision,
        "keys_artifact": {"path": str(keys_path), "sha256": e2.sha256_file(keys_path)},
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "threads": args.threads,
            "cg_rtol": args.cg_rtol,
            "cg_maxiter": args.cg_maxiter,
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_bytes": e2.peak_rss_bytes(),
        },
        "script": str(SCRIPT_PATH),
        "script_sha256": e2.sha256_file(SCRIPT_PATH),
    }
    e2.write_json(fit_path, receipt)
    pack_path = output / EXPERTPACK_NAME / "manifest.json"
    pack = e2.read_json(pack_path)
    pack["updated_at"] = now_iso()
    pack["provenance"]["fit_key"] = str(fit_path)
    pack["provenance"]["fit_key_sha256"] = e2.sha256_file(fit_path)
    pack["provenance"]["all_layer_keys"] = str(keys_path)
    pack["provenance"]["all_layer_keys_sha256"] = e2.sha256_file(keys_path)
    if selected is None:
        pack["status"] = "g2_red_no_installable_address"
        pack["limitations"] = ["E3-G2 RED: no qualifying DOC-A address; STOP"]
    else:
        layer_index = int(selected["layer"])
        selected_key = np.ascontiguousarray(keys[layer_index], dtype=np.float32)
        key_path = output / EXPERTPACK_NAME / "key_fp32.npy"
        e2.save_npy(key_path, selected_key)
        evaluation = selected["metrics"]["eval"]
        pack.update(
            {
                "status": "addressed_pending_pair_capture",
                "layer": layer_index,
                "rank": args.rank,
                "tau": float(selected["metrics"]["selected_tau"]),
                "key_metrics": {
                    "eval_doc_a_recall": evaluation["doc_a_recall"]["value"],
                    "eval_wikitext_fire_descriptive": evaluation["wikitext_fpr"]["value"],
                    "eval_code_fpr": evaluation["code_fpr"]["value"],
                    "eval_grm_fpr": evaluation["grm_fpr"]["value"],
                    "eval_guides_fpr": evaluation["guides_fpr"]["value"],
                    "doc_b_fire_descriptive": selected["metrics"]["doc_b_cross_episode_fire_descriptive"]["value"],
                    "selected_alpha": selected["key_metadata"]["selected_alpha"],
                    "selected_quantile": selected["metrics"]["selected_quantile_label"],
                },
            }
        )
        pack["components"]["key"] = {
            "status": "complete",
            "path": str(key_path),
            "shape": [HIDDEN_DIM],
            "storage_dtype": "float32",
            "sha256": e2.sha256_file(key_path),
        }
        pack["limitations"] = ["adapter A/B pending capture-pairs and train"]
    e2.write_json(pack_path, pack)
    print(
        json.dumps(
            {
                "gate": "E3-G2",
                "verdict": decision["verdict"],
                "qualifying_layers": decision["qualifying_layers_fit_ranked"],
                "install_layer": decision["install_layer"],
            }
        ),
        flush=True,
    )
    return 0 if selected is not None else 3


def load_addressed_pack(output: Path, *, require_adapter: bool = False) -> dict[str, Any]:
    pack = e2.read_json(output / EXPERTPACK_NAME / "manifest.json")
    if (
        pack.get("schema") != EXPERTPACK_NAME
        or pack.get("model_revision") != MODEL_REVISION
        or pack.get("layer") is None
        or pack.get("tau") is None
    ):
        raise RuntimeError("E3-G2 did not produce an installable address; STOP")
    key_record = pack.get("components", {}).get("key", {})
    key_path = Path(key_record.get("path", ""))
    if not key_path.is_file() or key_record.get("sha256") != e2.sha256_file(key_path):
        raise RuntimeError("ExpertPack key linkage failed")
    if require_adapter:
        if pack.get("g3", {}).get("verdict") != "GREEN":
            raise RuntimeError("E3-G3 is not GREEN; behavioral evaluation must STOP")
        for name in ("A", "B"):
            record = pack.get("components", {}).get(name, {})
            path = Path(record.get("path", ""))
            if not path.is_file() or record.get("sha256") != e2.sha256_file(path):
                raise RuntimeError(f"ExpertPack {name} linkage failed")
    return pack


def validate_fit_key(output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    path = output / "fit_key.json"
    receipt = e2.read_json(path)
    if (
        receipt.get("schema") != "moe_e3_fit_key_v1"
        or receipt.get("status") not in {"complete_g2_green", "complete_g2_red_stop"}
        or receipt.get("model_revision") != MODEL_REVISION
        or receipt.get("prepared_windows_content_sha256")
        != manifest["prepared_windows"]["content_sha256"]
        or receipt.get("keys_artifact", {}).get("sha256")
        != e2.sha256_file(Path(receipt.get("keys_artifact", {}).get("path", "")))
    ):
        raise RuntimeError("fit-key receipt or artifact contract failed")
    decision = receipt.get("decision", {})
    if decision.get("verdict") == "GREEN":
        pack = e2.read_json(output / EXPERTPACK_NAME / "manifest.json")
        layer = int(decision["install_layer"])
        key_record = pack.get("components", {}).get("key", {})
        key_path = Path(key_record.get("path", ""))
        if (
            pack.get("schema") != EXPERTPACK_NAME
            or pack.get("layer") != layer
            or pack.get("tau") != receipt["rows"][layer]["metrics"]["selected_tau"]
            or pack.get("provenance", {}).get("fit_key") != str(path)
            or pack.get("provenance", {}).get("fit_key_sha256") != e2.sha256_file(path)
            or not key_path.is_file()
            or key_record.get("sha256") != e2.sha256_file(key_path)
        ):
            raise RuntimeError("ExpertPack differs from fit-key decision")
        with np.load(receipt["keys_artifact"]["path"], allow_pickle=False) as archive:
            selected_key = archive["K4"][layer]
        if not np.array_equal(np.load(key_path, allow_pickle=False), selected_key):
            raise RuntimeError("ExpertPack selected key differs from all-layer K4 artifact")
    return receipt


def validate_fit_key_inputs(
    output: Path, manifest: dict[str, Any], arrays: dict[str, np.ndarray]
) -> dict[str, Any]:
    receipt = validate_fit_key(output, manifest)
    observed = collect_capture_provenance(output, manifest, arrays)
    recorded = receipt.get("input_integrity", {}).get("captures", [])
    if len(recorded) != len(observed):
        raise RuntimeError("fit-key capture provenance count changed")
    fields = (
        "corpus",
        "window_index",
        "capture_path",
        "capture_sha256",
        "receipt_path",
        "receipt_sha256",
        "source",
    )
    for expected, actual in zip(recorded, observed):
        if any(expected.get(field) != actual.get(field) for field in fields):
            raise RuntimeError("fit-key capture provenance linkage changed")
        if "provenance_validation_sha256" in actual and (
            expected.get("provenance_validation_path")
            != actual.get("provenance_validation_path")
            or expected.get("provenance_validation_sha256")
            != actual.get("provenance_validation_sha256")
            or expected.get("content_sha256") != actual.get("content_sha256")
        ):
            raise RuntimeError("fit-key E2 negative validation linkage changed")
    return receipt


def pair_paths(output: Path, pair_index: int) -> tuple[Path, Path]:
    root = output / "pairs"
    return root / f"pair_{pair_index:03d}.npz", root / f"pair_{pair_index:03d}_receipt.json"


def validate_pair(
    output: Path,
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    pair_index: int,
) -> dict[str, Any]:
    data_path, receipt_path = pair_paths(output, pair_index)
    if not data_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"missing pair {pair_index}")
    receipt = e2.read_json(receipt_path)
    expected_split = "TRAIN" if pair_index < N_PAIR_TRAIN else "VALIDATION"
    if (
        receipt.get("schema") != "moe_e3_teacher_student_pair_v1"
        or receipt.get("status") != "complete"
        or receipt.get("pair_index") != pair_index
        or receipt.get("split") != expected_split
        or receipt.get("model_revision") != MODEL_REVISION
        or not receipt.get("shared_window_token_ids_exact")
        or receipt.get("student_ids_sha256") != e2.sha256_array(arrays["pair_ids"][pair_index])
        or receipt.get("teacher_shared_ids_sha256") != e2.sha256_array(arrays["pair_ids"][pair_index])
        or receipt.get("teacher_prefix_ids_sha256") != e2.sha256_array(arrays["pair_prefix_ids"][pair_index])
        or receipt.get("prepared_windows_content_sha256")
        != manifest["prepared_windows"]["content_sha256"]
        or Path(receipt.get("pair_path", "")).resolve() != data_path.resolve()
        or receipt.get("pair_file_sha256") != e2.sha256_file(data_path)
    ):
        raise RuntimeError(f"pair receipt contract failed: {receipt_path}")
    with np.load(data_path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    expected_names = {"h_student_fp16", "out_student_fp16", "out_teacher_fp16"}
    if set(data) != expected_names:
        raise RuntimeError(f"pair payload keys failed: {data_path}")
    hash_fields = {
        "h_student_fp16": "student_router_input_sha256",
        "out_student_fp16": "student_block_output_sha256",
        "out_teacher_fp16": "teacher_block_output_sha256",
    }
    for name, field in hash_fields.items():
        value = data[name]
        if (
            value.shape != (WINDOW_TOKENS, HIDDEN_DIM)
            or value.dtype != np.float16
            or not np.isfinite(value).all()
            or receipt.get(field) != e2.sha256_array(value)
        ):
            raise RuntimeError(f"pair payload array contract failed: {data_path}/{name}")
    fit = validate_fit_key(output, manifest)
    if (
        receipt.get("fit_key_receipt_sha256") != e2.sha256_file(output / "fit_key.json")
        or receipt.get("install_layer") != fit["decision"]["install_layer"]
    ):
        raise RuntimeError("pair fit-key linkage failed")
    return {
        "pair_index": pair_index,
        "split": expected_split,
        "receipt_path": str(receipt_path),
        "receipt_sha256": e2.sha256_file(receipt_path),
        "payload_path": str(data_path),
        "payload_sha256": receipt["pair_file_sha256"],
        "data": data,
        "receipt": receipt,
    }


def capture_pairs(args: argparse.Namespace) -> int:
    if args.start_window < 0 or args.count < 1 or args.start_window + args.count > N_PAIR_WINDOWS:
        raise ValueError("capture-pairs range must stay within local pair indices 0..63")
    output = ensure_output(args.output_dir)
    manifest, arrays = load_prepared(output)
    require_bringup_green(output, manifest)
    fit = validate_fit_key(output, manifest)
    if fit["decision"]["verdict"] != "GREEN":
        raise RuntimeError("E3-G2 is RED; pair capture must STOP")
    pack = load_addressed_pack(output)
    requested = list(range(args.start_window, args.start_window + args.count))
    missing: list[int] = []
    for index in requested:
        _data_path, receipt_path = pair_paths(output, index)
        if receipt_path.is_file():
            validate_pair(output, manifest, arrays, index)
        else:
            missing.append(index)
    if not missing:
        print(json.dumps({"status": "existing", "pairs": requested}))
        return 0

    started = time.perf_counter()
    torch, model, runtime = e2.load_model(args)
    layer_index = int(pack["layer"])
    layer = model.model.layers[layer_index]
    slot: dict[str, np.ndarray] = {}

    def pre_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
        slot["h"] = np.ascontiguousarray(
            inputs[0].detach().to(device="cpu", dtype=torch.float16).numpy()
        )

    def block_hook(_module: Any, _inputs: tuple[Any, ...], output_value: Any) -> None:
        slot["out"] = np.ascontiguousarray(
            output_value.detach().to(device="cpu", dtype=torch.float16).numpy()
        )

    pre_handle = layer.mlp.register_forward_pre_hook(pre_hook)
    block_handle = layer.register_forward_hook(block_hook)

    def one_forward(ids_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        slot.clear()
        ids = torch.from_numpy(np.ascontiguousarray(ids_np[None, :], dtype=np.int64)).to(
            e2.input_device(model)
        )
        with torch.inference_mode():
            output_object = model(input_ids=ids, use_cache=False, logits_to_keep=1)
        del output_object, ids
        if set(slot) != {"h", "out"}:
            raise RuntimeError(f"pair hook did not execute exactly: {slot.keys()}")
        return slot["h"][0].copy(), slot["out"][0].copy()

    try:
        for pair_index in missing:
            window_started = time.perf_counter()
            student_ids = arrays["pair_ids"][pair_index]
            prefix_ids = arrays["pair_prefix_ids"][pair_index]
            teacher_ids = np.concatenate([prefix_ids, student_ids])
            if (
                teacher_ids.size != PREFIX_TOKENS + WINDOW_TOKENS
                or teacher_ids.size > MAX_CONTEXT
                or not np.array_equal(teacher_ids[-WINDOW_TOKENS:], student_ids)
            ):
                raise RuntimeError("teacher context/shared-target contract failed")
            h_student, out_student = one_forward(student_ids)
            h_teacher_full, out_teacher_full = one_forward(teacher_ids)
            h_teacher = np.ascontiguousarray(h_teacher_full[-WINDOW_TOKENS:])
            out_teacher = np.ascontiguousarray(out_teacher_full[-WINDOW_TOKENS:])
            if (
                h_student.shape != (WINDOW_TOKENS, HIDDEN_DIM)
                or out_student.shape != h_student.shape
                or out_teacher.shape != h_student.shape
            ):
                raise RuntimeError("pair activation shapes failed")
            delta = out_teacher.astype(np.float32) - out_student.astype(np.float32)
            hidden_difference = h_teacher.astype(np.float32) - h_student.astype(np.float32)
            if not np.isfinite(delta).all() or not np.isfinite(hidden_difference).all():
                raise RuntimeError("pair capture contains non-finite values")
            data_path, receipt_path = pair_paths(output, pair_index)
            e2.save_npz(
                data_path,
                h_student_fp16=h_student,
                out_student_fp16=out_student,
                out_teacher_fp16=out_teacher,
            )
            split = "TRAIN" if pair_index < N_PAIR_TRAIN else "VALIDATION"
            receipt = {
                "schema": "moe_e3_teacher_student_pair_v1",
                "status": "complete",
                "created_at": now_iso(),
                "pair_index": pair_index,
                "doc_a_source_window": manifest["windows"]["pair"][pair_index],
                "split": split,
                "install_layer": layer_index,
                "teacher_prefix_arm": "true preceding DOC-A tokens",
                "student_tokens": WINDOW_TOKENS,
                "teacher_prefix_tokens": PREFIX_TOKENS,
                "teacher_total_tokens": int(teacher_ids.size),
                "shared_window_token_ids_exact": True,
                "student_ids_sha256": e2.sha256_array(student_ids),
                "teacher_shared_ids_sha256": e2.sha256_array(teacher_ids[-WINDOW_TOKENS:]),
                "teacher_prefix_ids_sha256": e2.sha256_array(prefix_ids),
                "student_router_input_sha256": e2.sha256_array(h_student),
                "teacher_router_input_sha256": e2.sha256_array(h_teacher),
                "student_block_output_sha256": e2.sha256_array(out_student),
                "teacher_block_output_sha256": e2.sha256_array(out_teacher),
                "router_input_hidden_states_differ": bool(np.any(hidden_difference != 0)),
                "router_input_difference_mean_abs": float(np.abs(hidden_difference).mean()),
                "target_delta_zero_predictor_mse": float(np.mean(delta.astype(np.float64) ** 2)),
                "target_delta_mean_abs": float(np.abs(delta.astype(np.float64)).mean()),
                "target_delta_max_abs": float(np.abs(delta.astype(np.float64)).max()),
                "pair_path": str(data_path),
                "pair_file_sha256": e2.sha256_file(data_path),
                "fit_key_receipt": str(output / "fit_key.json"),
                "fit_key_receipt_sha256": e2.sha256_file(output / "fit_key.json"),
                **content_fields(manifest),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "runtime": runtime,
                "window_wall_seconds": time.perf_counter() - window_started,
                "script": str(SCRIPT_PATH),
                "script_sha256": e2.sha256_file(SCRIPT_PATH),
            }
            e2.write_json(receipt_path, receipt)
            print(
                json.dumps(
                    {
                        "pair": pair_index,
                        "source_window": PAIR_DOC_WINDOWS[pair_index],
                        "split": split,
                        "alignment": True,
                        "zero_mse": receipt["target_delta_zero_predictor_mse"],
                    }
                ),
                flush=True,
            )
            del h_student, h_teacher, h_teacher_full, out_student, out_teacher, out_teacher_full
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        pre_handle.remove()
        block_handle.remove()
    print(
        json.dumps(
            {
                "status": "complete",
                "pairs": missing,
                "wall_seconds": time.perf_counter() - started,
                "peak_rss_bytes": e2.peak_rss_bytes(),
            }
        ),
        flush=True,
    )
    del model
    gc.collect()
    return 0


def bootstrap_window_ratio(
    numerator_by_window: np.ndarray,
    denominator_by_window: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    numerator = np.asarray(numerator_by_window, dtype=np.float64)
    denominator = np.asarray(denominator_by_window, dtype=np.float64)
    if numerator.shape != denominator.shape or numerator.ndim != 1 or numerator.size == 0:
        raise ValueError("bootstrap ratio requires equal nonempty one-dimensional arrays")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, numerator.size, size=(resamples, numerator.size))
    num = numerator[sampled].mean(axis=1)
    den = denominator[sampled].mean(axis=1)
    boot = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
    low, high = np.quantile(boot, [0.025, 0.975], method="linear")
    point = float(numerator.mean() / denominator.mean()) if denominator.mean() != 0 else 0.0
    return {
        "value": point,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_unit": "validation windows",
        "bootstrap_resamples": resamples,
    }


def validate_training(
    output: Path, manifest: dict[str, Any], arrays: dict[str, np.ndarray]
) -> dict[str, Any]:
    path = output / "train.json"
    receipt = e2.read_json(path)
    if (
        receipt.get("schema") != "moe_e3_adapter_training_v1"
        or receipt.get("status") not in {"complete_g3_green", "complete_g3_red_stop"}
        or receipt.get("model_revision") != MODEL_REVISION
        or receipt.get("prepared_windows_content_sha256")
        != manifest["prepared_windows"]["content_sha256"]
        or receipt.get("fit_key_receipt_sha256") != e2.sha256_file(output / "fit_key.json")
    ):
        raise RuntimeError("training receipt contract failed")
    recorded_pairs = receipt.get("data", {}).get("pair_receipts", [])
    if len(recorded_pairs) != N_PAIR_WINDOWS:
        raise RuntimeError("training pair receipt count changed")
    for index in range(N_PAIR_WINDOWS):
        observed = validate_pair(output, manifest, arrays, index)
        recorded = recorded_pairs[index]
        if (
            recorded.get("pair_index") != index
            or recorded.get("receipt_path") != observed["receipt_path"]
            or recorded.get("receipt_sha256") != observed["receipt_sha256"]
            or recorded.get("payload_sha256") != observed["payload_sha256"]
        ):
            raise RuntimeError(f"training pair linkage changed: {index}")
    for name in ("A", "B"):
        record = receipt.get("artifacts", {}).get(name, {})
        path_value = Path(record.get("path", ""))
        if not path_value.is_file() or record.get("sha256") != e2.sha256_file(path_value):
            raise RuntimeError(f"training {name} artifact linkage failed")
    return receipt


def train(args: argparse.Namespace) -> int:
    if args.rank != 64:
        raise ValueError("E3 rank is registered at r=64")
    if not 1 <= args.train_tokens_per_window <= WINDOW_TOKENS:
        raise ValueError("--train-tokens-per-window must be within 1..512")
    if (
        args.threads < 1
        or args.token_batch_size < 1
        or args.epochs < 1
        or args.patience < 1
        or args.bootstrap_resamples < 2000
    ):
        raise ValueError("training resources and bootstrap count are below registered minima")
    output = ensure_output(args.output_dir)
    manifest, arrays = load_prepared(output)
    require_bringup_green(output, manifest)
    fit = validate_fit_key(output, manifest)
    if fit["decision"]["verdict"] != "GREEN":
        raise RuntimeError("E3-G2 is RED; training must STOP")
    pack = load_addressed_pack(output)
    training_path = output / "train.json"
    if training_path.is_file():
        receipt = validate_training(output, manifest, arrays)
        print(json.dumps({"status": "existing", "gate": "E3-G3", "verdict": receipt["verdict"]}))
        return 0 if receipt["verdict"] == "GREEN" else 3
    started = time.perf_counter()
    torch = e2.configure_torch(args.threads, cuda_possible=False)
    import torch.nn as nn
    import torch.nn.functional as functional

    key = np.load(pack["components"]["key"]["path"], allow_pickle=False)
    if key.shape != (HIDDEN_DIM,) or key.dtype != np.float32:
        raise RuntimeError("ExpertPack key payload contract failed")
    key_tensor = torch.from_numpy(np.ascontiguousarray(key))
    tau = float(pack["tau"])

    class Adapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.A = nn.Parameter(torch.empty(args.rank, HIDDEN_DIM, dtype=torch.float32))
            self.B = nn.Parameter(torch.zeros(HIDDEN_DIM, args.rank, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        def forward(self, value: Any) -> Any:
            return functional.linear(functional.silu(functional.linear(value, self.A)), self.B)

    torch.manual_seed(args.seed)
    adapter = Adapter()
    initial_B = adapter.B.detach().numpy().copy()
    if not np.array_equal(initial_B, np.zeros_like(initial_B)):
        raise RuntimeError("B zero initialization failed")
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    validated_pairs = [validate_pair(output, manifest, arrays, index) for index in range(N_PAIR_WINDOWS)]

    def evaluate_validation(model: Any) -> tuple[float, float, int, int, list[dict[str, Any]]]:
        model.eval()
        squared = 0.0
        zero_squared = 0.0
        elements = 0
        fired_count = 0
        per_window: list[dict[str, Any]] = []
        with torch.no_grad():
            for index in range(N_PAIR_TRAIN, N_PAIR_WINDOWS):
                data = validated_pairs[index]["data"]
                hidden = torch.from_numpy(data["h_student_fp16"].astype(np.float32))
                target = torch.from_numpy(
                    data["out_teacher_fp16"].astype(np.float32)
                    - data["out_student_fp16"].astype(np.float32)
                )
                window_squared = 0.0
                window_zero = 0.0
                window_elements = 0
                window_fires = 0
                for start in range(0, WINDOW_TOKENS, args.token_batch_size):
                    h_batch = hidden[start : start + args.token_batch_size]
                    target_batch = target[start : start + args.token_batch_size]
                    fire = (h_batch @ key_tensor >= tau).to(torch.float32)[:, None]
                    prediction = model(h_batch) * fire
                    batch_squared = float(torch.sum((prediction - target_batch) ** 2).item())
                    batch_zero = float(torch.sum(target_batch**2).item())
                    batch_elements = int(target_batch.numel())
                    window_squared += batch_squared
                    window_zero += batch_zero
                    window_elements += batch_elements
                    window_fires += int(fire.sum().item())
                squared += window_squared
                zero_squared += window_zero
                elements += window_elements
                fired_count += window_fires
                per_window.append(
                    {
                        "pair_index": index,
                        "source_doc_window": PAIR_DOC_WINDOWS[index],
                        "adapter_mse": window_squared / window_elements,
                        "zero_predictor_mse": window_zero / window_elements,
                        "fire_count": window_fires,
                        "token_count": WINDOW_TOKENS,
                    }
                )
                del hidden, target
        return squared / elements, zero_squared / elements, fired_count, elements // HIDDEN_DIM, per_window

    zero_validation_mse, zero_again, _fires, validation_tokens, _rows = evaluate_validation(adapter)
    if zero_validation_mse != zero_again:
        raise RuntimeError("zero-initialized adapter did not equal zero predictor")
    best_mse = zero_validation_mse
    best_epoch = 0
    best_state = {name: tensor.detach().clone() for name, tensor in adapter.state_dict().items()}
    epochs_without_improvement = 0
    epoch_records: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        rng = np.random.default_rng(args.seed + epoch * 100003)
        order = rng.permutation(N_PAIR_TRAIN).tolist()
        epoch_squared = 0.0
        epoch_elements = 0
        epoch_fires = 0
        for index in order:
            data = validated_pairs[index]["data"]
            hidden_np = data["h_student_fp16"].astype(np.float32)
            target_np = data["out_teacher_fp16"].astype(np.float32) - data["out_student_fp16"].astype(np.float32)
            if args.train_tokens_per_window < WINDOW_TOKENS:
                token_indices = np.sort(
                    rng.choice(WINDOW_TOKENS, size=args.train_tokens_per_window, replace=False)
                )
                hidden_np = hidden_np[token_indices]
                target_np = target_np[token_indices]
            hidden = torch.from_numpy(np.ascontiguousarray(hidden_np))
            target = torch.from_numpy(np.ascontiguousarray(target_np))
            for start in range(0, hidden.shape[0], args.token_batch_size):
                h_batch = hidden[start : start + args.token_batch_size]
                target_batch = target[start : start + args.token_batch_size]
                fire = (h_batch @ key_tensor >= tau).to(torch.float32)[:, None]
                optimizer.zero_grad(set_to_none=True)
                prediction = adapter(h_batch) * fire
                loss = torch.mean((prediction - target_batch) ** 2)
                loss.backward()
                optimizer.step()
                epoch_squared += float(loss.item()) * int(target_batch.numel())
                epoch_elements += int(target_batch.numel())
                epoch_fires += int(fire.sum().item())
            del hidden, target
        validation_mse, _zero_mse, validation_fires, _tokens, _per_window = evaluate_validation(adapter)
        improved = validation_mse < best_mse - max(1.0e-12, abs(best_mse) * 1.0e-6)
        if improved:
            best_mse = validation_mse
            best_epoch = epoch
            best_state = {name: tensor.detach().clone() for name, tensor in adapter.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        record = {
            "epoch": epoch,
            "train_sampled_mse": epoch_squared / epoch_elements,
            "train_sampled_tokens": epoch_elements // HIDDEN_DIM,
            "train_sampled_fire_count": epoch_fires,
            "validation_mse": validation_mse,
            "validation_fire_count": validation_fires,
            "improved": improved,
            "epochs_without_improvement": epochs_without_improvement,
        }
        epoch_records.append(record)
        print(json.dumps(record), flush=True)
        if epochs_without_improvement >= args.patience:
            break
    adapter.load_state_dict(best_state)
    A_fp16 = adapter.A.detach().numpy().astype(np.float16)
    B_fp16 = adapter.B.detach().numpy().astype(np.float16)
    pack_dir = output / EXPERTPACK_NAME
    A_path = pack_dir / "A_fp16.npy"
    B_path = pack_dir / "B_fp16.npy"
    e2.save_npy(A_path, A_fp16)
    e2.save_npy(B_path, B_fp16)
    with torch.no_grad():
        adapter.A.copy_(torch.from_numpy(A_fp16.astype(np.float32)))
        adapter.B.copy_(torch.from_numpy(B_fp16.astype(np.float32)))
    stored_mse, stored_zero_mse, stored_fires, _tokens, per_window = evaluate_validation(adapter)
    improvement = (
        (stored_zero_mse - stored_mse) / stored_zero_mse if stored_zero_mse > 0 else None
    )
    improvement_ci = bootstrap_window_ratio(
        np.asarray([row["zero_predictor_mse"] - row["adapter_mse"] for row in per_window]),
        np.asarray([row["zero_predictor_mse"] for row in per_window]),
        resamples=args.bootstrap_resamples,
        seed=args.seed + 300,
    )
    green = bool(improvement is not None and improvement >= G3_IMPROVEMENT_FLOOR)
    pair_receipts = [
        {
            "pair_index": item["pair_index"],
            "receipt_path": item["receipt_path"],
            "receipt_sha256": item["receipt_sha256"],
            "payload_sha256": item["payload_sha256"],
        }
        for item in validated_pairs
    ]
    training = {
        "schema": "moe_e3_adapter_training_v1",
        "status": "complete_g3_green" if green else "complete_g3_red_stop",
        "created_at": now_iso(),
        "gate": "E3-G3",
        "verdict": "GREEN" if green else "RED",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        **content_fields(manifest),
        "install_layer": int(pack["layer"]),
        "rank": args.rank,
        "tau": tau,
        "abi": "B @ silu(A @ h), threshold gated by h @ key >= tau",
        "initialization": {
            "A": "torch kaiming_uniform, deterministic CPU seed",
            "B": "exact all-zero float32",
            "B_zero_exact": True,
            "B_initial_sha256": e2.sha256_array(initial_B),
        },
        "fit_key_receipt": str(output / "fit_key.json"),
        "fit_key_receipt_sha256": e2.sha256_file(output / "fit_key.json"),
        "data": {
            "train_pair_indices": list(range(N_PAIR_TRAIN)),
            "validation_pair_indices": list(range(N_PAIR_TRAIN, N_PAIR_WINDOWS)),
            "train_pair_count": N_PAIR_TRAIN,
            "validation_pair_count": N_PAIR_VALIDATION,
            "train_tokens_sampled_per_window_per_epoch": args.train_tokens_per_window,
            "validation_tokens": validation_tokens,
            "validation_fire_count": stored_fires,
            "pair_receipts": pair_receipts,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "token_batch_size": args.token_batch_size,
            "maximum_epochs": args.epochs,
            "patience": args.patience,
            "early_stopping_metric": "validation-pair full-token gated MSE",
        },
        "epochs": epoch_records,
        "best_epoch_float32": best_epoch,
        "best_validation_mse_float32": best_mse,
        "stored_fp16_validation": {
            "zero_predictor_mse": stored_zero_mse,
            "adapter_mse": stored_mse,
            "improvement_fraction": improvement,
            "improvement_percent": None if improvement is None else improvement * 100.0,
            "improvement_ci95_fraction": improvement_ci,
            "required_improvement_fraction": G3_IMPROVEMENT_FLOOR,
            "per_window": per_window,
        },
        "artifacts": {
            "A": {
                "path": str(A_path),
                "shape": list(A_fp16.shape),
                "dtype": "float16",
                "sha256": e2.sha256_file(A_path),
            },
            "B": {
                "path": str(B_path),
                "shape": list(B_fp16.shape),
                "dtype": "float16",
                "sha256": e2.sha256_file(B_path),
            },
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "threads": args.threads,
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_bytes": e2.peak_rss_bytes(),
        },
        "script": str(SCRIPT_PATH),
        "script_sha256": e2.sha256_file(SCRIPT_PATH),
    }
    e2.write_json(training_path, training)
    pack = e2.read_json(pack_dir / "manifest.json")
    pack["updated_at"] = now_iso()
    pack["components"]["A"] = {
        "status": "complete",
        "path": str(A_path),
        "shape": list(A_fp16.shape),
        "storage_dtype": "float16",
        "sha256": e2.sha256_file(A_path),
    }
    pack["components"]["B"] = {
        "status": "complete",
        "path": str(B_path),
        "shape": list(B_fp16.shape),
        "storage_dtype": "float16",
        "initialization": "exact zeros before training",
        "initial_zero_sha256": e2.sha256_array(initial_B),
        "sha256": e2.sha256_file(B_path),
    }
    pack["provenance"]["training"] = str(training_path)
    pack["provenance"]["training_sha256"] = e2.sha256_file(training_path)
    pack["g3"] = training["stored_fp16_validation"] | {"verdict": training["verdict"]}
    pack["status"] = "complete_g3_green" if green else "g3_red_not_installable"
    pack["limitations"] = (
        ["ABI and behavioral E3-G4/G5 evaluation pending"]
        if green
        else ["E3-G3 RED: stored adapter failed the 10 percent improvement gate; STOP"]
    )
    e2.write_json(pack_dir / "manifest.json", pack)
    print(
        json.dumps(
            {
                "gate": "E3-G3",
                "verdict": training["verdict"],
                "improvement_percent": training["stored_fp16_validation"]["improvement_percent"],
            }
        ),
        flush=True,
    )
    return 0 if green else 3


def load_expert_arrays(
    output: Path, manifest: dict[str, Any], arrays: dict[str, np.ndarray]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    training = validate_training(output, manifest, arrays)
    if training["verdict"] != "GREEN":
        raise RuntimeError("E3-G3 is RED; behavioral evaluation must STOP")
    pack = load_addressed_pack(output, require_adapter=True)
    if (
        pack.get("provenance", {}).get("training") != str(output / "train.json")
        or pack.get("provenance", {}).get("training_sha256")
        != e2.sha256_file(output / "train.json")
        or pack.get("g3", {}).get("verdict") != training["verdict"]
    ):
        raise RuntimeError("ExpertPack training receipt linkage failed")
    for name in ("A", "B"):
        if (
            pack["components"][name].get("path") != training["artifacts"][name]["path"]
            or pack["components"][name].get("sha256")
            != training["artifacts"][name]["sha256"]
        ):
            raise RuntimeError(f"ExpertPack {name} differs from training receipt")
    key = np.load(pack["components"]["key"]["path"], allow_pickle=False)
    A = np.load(pack["components"]["A"]["path"], allow_pickle=False)
    B = np.load(pack["components"]["B"]["path"], allow_pickle=False)
    rank = int(pack["rank"])
    if key.shape != (HIDDEN_DIM,) or key.dtype != np.float32:
        raise RuntimeError("key contract failed")
    if A.shape != (rank, HIDDEN_DIM) or A.dtype != np.float16:
        raise RuntimeError("A contract failed")
    if B.shape != (HIDDEN_DIM, rank) or B.dtype != np.float16:
        raise RuntimeError("B contract failed")
    return pack, np.ascontiguousarray(key), np.ascontiguousarray(A), np.ascontiguousarray(B)


def eval_root(output: Path) -> Path:
    root = output / "eval"
    root.mkdir(parents=True, exist_ok=True)
    return root


def eval_path(output: Path, kind: str, index: int | None = None) -> Path:
    root = eval_root(output)
    if kind == "abi":
        return root / "abi.json"
    if index is None:
        raise ValueError(f"{kind} requires an index")
    return root / f"{kind.replace('-', '_')}_{index:03d}.json"


def expert_binding(output: Path, pack: dict[str, Any]) -> dict[str, Any]:
    return {
        "training_receipt": str(output / "train.json"),
        "training_receipt_sha256": e2.sha256_file(output / "train.json"),
        "key_sha256": pack["components"]["key"]["sha256"],
        "A_sha256": pack["components"]["A"]["sha256"],
        "B_sha256": pack["components"]["B"]["sha256"],
    }


def validate_eval(
    output: Path,
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    kind: str,
    index: int | None = None,
) -> dict[str, Any]:
    path = eval_path(output, kind, index)
    receipt = e2.read_json(path)
    schemas = {
        "abi": "moe_e3_abi_identity_v1",
        "doc-a-teacher": "moe_e3_doc_a_teacher_eval_v1",
        "doc-a-expert": "moe_e3_doc_a_expert_eval_v1",
        "wikitext": "moe_e3_wikitext_noninterference_v1",
        "code": "moe_e3_code_fire_v1",
        "doc-b": "moe_e3_doc_b_selectivity_v1",
    }
    if (
        receipt.get("schema") != schemas[kind]
        or receipt.get("status") != "complete"
        or receipt.get("model_revision") != MODEL_REVISION
        or receipt.get("prepared_windows_content_sha256")
        != manifest["prepared_windows"]["content_sha256"]
    ):
        raise RuntimeError(f"eval receipt contract failed: {path}")
    if kind == "abi":
        expected = np.concatenate([arrays["doc_a_key_ids"][0, :64], arrays["code_ids"][0, :64]])
        if receipt.get("input_ids_sha256") != e2.sha256_array(expected):
            raise RuntimeError("ABI input binding failed")
        pack = load_addressed_pack(output, require_adapter=True)
        if receipt.get("expert_binding") != expert_binding(output, pack):
            raise RuntimeError("ABI ExpertPack binding failed")
        return receipt
    assert index is not None
    if receipt.get("window_index") != index:
        raise RuntimeError(f"eval window index failed: {path}")
    array_name = {
        "doc-a-teacher": "heldout_ids",
        "doc-a-expert": "heldout_ids",
        "wikitext": "wikitext_ids",
        "code": "code_ids",
        "doc-b": "doc_b_ids",
    }[kind]
    target = arrays[array_name][index]
    if receipt.get("input_ids_sha256") != e2.sha256_array(target):
        raise RuntimeError(f"eval input binding failed: {path}")
    if kind == "doc-a-teacher":
        if receipt.get("teacher_prefix_ids_sha256") != e2.sha256_array(arrays["heldout_prefix_ids"][index]):
            raise RuntimeError("teacher prefix binding failed")
        arms = ("student", "teacher")
    elif kind in {"wikitext", "doc-b"}:
        arms = ("base", "expert")
    elif kind == "doc-a-expert":
        arms = ("expert",)
    else:
        arms = ()
    for arm in arms:
        if receipt.get("arms", {}).get(arm, {}).get("target_ids_sha256") != e2.sha256_array(target[1:]):
            raise RuntimeError(f"eval scored target binding failed: {path}/{arm}")
    if kind != "doc-a-teacher":
        pack = load_addressed_pack(output, require_adapter=True)
        if receipt.get("expert_binding") != expert_binding(output, pack):
            raise RuntimeError(f"eval ExpertPack binding failed: {path}")
    return receipt


def eval_abi(args: argparse.Namespace) -> int:
    output = ensure_output(args.output_dir)
    manifest, arrays = load_prepared(output)
    require_bringup_green(output, manifest)
    pack, key, A, B = load_expert_arrays(output, manifest, arrays)
    receipt_path = eval_path(output, "abi")
    if receipt_path.is_file():
        receipt = validate_eval(output, manifest, arrays, "abi")
        print(json.dumps({"status": "existing", "kind": "abi", "verdict": receipt["verdict"]}))
        return 0 if receipt["verdict"] == "GREEN" else 3
    started = time.perf_counter()
    torch, model, runtime = e2.load_model(args)
    layer = model.model.layers[int(pack["layer"])]
    ids = np.concatenate([arrays["doc_a_key_ids"][0, :64], arrays["code_ids"][0, :64]])
    baseline = e2.forward_snapshot(torch, model, ids)
    repeat = e2.forward_snapshot(torch, model, ids)
    zero_install_equal = bool(
        torch.equal(baseline["hidden"], repeat["hidden"])
        and torch.equal(baseline["logits"], repeat["logits"])
    )
    with e2.ExpertMount(
        layer.mlp,
        torch_module=torch,
        key=key,
        tau=float(pack["tau"]),
        A=A,
        B=np.zeros_like(B),
        label="zero_B",
    ) as zero_mount:
        zero_snapshot = e2.forward_snapshot(torch, model, ids)
    zero_B_equal = bool(
        torch.equal(baseline["hidden"], zero_snapshot["hidden"])
        and torch.equal(baseline["logits"], zero_snapshot["logits"])
    )
    with e2.ExpertMount(
        layer.mlp,
        torch_module=torch,
        key=key,
        tau=float("inf"),
        A=A,
        B=B,
        label="forced_no_fire",
    ) as nofire_mount:
        nofire_snapshot = e2.forward_snapshot(torch, model, ids)
    nofire_equal = bool(
        torch.equal(baseline["hidden"], nofire_snapshot["hidden"])
        and torch.equal(baseline["logits"], nofire_snapshot["logits"])
    )
    with e2.ExpertMount(
        layer.mlp,
        torch_module=torch,
        key=key,
        tau=float(pack["tau"]),
        A=A,
        B=B,
        label="live",
    ) as live_mount:
        live_snapshot = e2.forward_snapshot(torch, model, ids)
    if len(zero_mount.calls) != 1 or len(nofire_mount.calls) != 1 or len(live_mount.calls) != 1:
        raise RuntimeError("ABI expected exactly one install-layer call per forward")
    live = live_mount.calls[0]
    live_nonfire_ok = bool(live["nonfire_count"] > 0 and live["nonfired_rows_bit_equal"])
    green = bool(zero_install_equal and zero_B_equal and nofire_equal and live_nonfire_ok)

    def public_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
        return {name: value for name, value in snapshot.items() if name not in {"hidden", "logits"}}

    receipt = {
        "schema": "moe_e3_abi_identity_v1",
        "status": "complete",
        "created_at": now_iso(),
        "gate": "ABI carry",
        "verdict": "GREEN" if green else "RED",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        **content_fields(manifest),
        "expert_binding": expert_binding(output, pack),
        "install_layer": int(pack["layer"]),
        "rank": int(pack["rank"]),
        "tau": float(pack["tau"]),
        "input_tokens": int(ids.size),
        "input_ids_sha256": e2.sha256_array(ids),
        "zero_experts_installed": {
            "bit_equal_final_hidden_and_logits": zero_install_equal,
            "baseline": public_snapshot(baseline),
            "repeat": public_snapshot(repeat),
        },
        "zero_initialized_B_mounted": {
            "bit_equal_final_hidden_and_logits": zero_B_equal,
            "mount": zero_mount.calls[0],
            "snapshot": public_snapshot(zero_snapshot),
        },
        "forced_no_fire_mounted": {
            "bit_equal_final_hidden_and_logits": nofire_equal,
            "mount": nofire_mount.calls[0],
            "snapshot": public_snapshot(nofire_snapshot),
        },
        "live_gate": {
            "nonfired_rows_bit_equal_at_native_mlp_side_path": live_nonfire_ok,
            "mount": live,
            "final_snapshot": public_snapshot(live_snapshot),
        },
        "identity_comparison": "torch.equal byte-exact BF16 final hidden states and logits",
        "runtime": runtime,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": e2.peak_rss_bytes(),
        "script": str(SCRIPT_PATH),
        "script_sha256": e2.sha256_file(SCRIPT_PATH),
    }
    e2.write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "kind": "abi",
                "verdict": receipt["verdict"],
                "zero_install_equal": zero_install_equal,
                "zero_B_equal": zero_B_equal,
                "forced_no_fire_equal": nofire_equal,
                "live_nonfire_count": live["nonfire_count"],
            }
        ),
        flush=True,
    )
    del model, baseline, repeat, zero_snapshot, nofire_snapshot, live_snapshot
    gc.collect()
    return 0 if green else 3


def requested_eval_indices(args: argparse.Namespace) -> list[int]:
    if args.start_window < 0 or args.count < 1 or args.start_window + args.count > N_BEHAVIORAL_WINDOWS:
        raise ValueError("eval range must stay within local windows 0..15")
    return list(range(args.start_window, args.start_window + args.count))


def eval_doc_a_teacher(args: argparse.Namespace) -> int:
    requested = requested_eval_indices(args)
    output = ensure_output(args.output_dir)
    manifest, arrays = load_prepared(output)
    require_bringup_green(output, manifest)
    load_expert_arrays(output, manifest, arrays)
    missing: list[int] = []
    for index in requested:
        path = eval_path(output, "doc-a-teacher", index)
        if path.is_file():
            validate_eval(output, manifest, arrays, "doc-a-teacher", index)
        else:
            missing.append(index)
    if not missing:
        print(json.dumps({"status": "existing", "kind": "doc-a-teacher", "windows": requested}))
        return 0
    started = time.perf_counter()
    torch, model, runtime = e2.load_model(args)
    for index in missing:
        target = arrays["heldout_ids"][index]
        prefix = arrays["heldout_prefix_ids"][index]
        teacher_full = np.concatenate([prefix, target])
        student = e2.score_target_window(torch, model, target, target)
        teacher = e2.score_target_window(torch, model, teacher_full, target)
        receipt = {
            "schema": "moe_e3_doc_a_teacher_eval_v1",
            "status": "complete",
            "created_at": now_iso(),
            "window_index": index,
            "source_window": manifest["windows"]["heldout"][index],
            "teacher_prefix_role": "true preceding 2,048 DOC-A tokens",
            "input_ids_sha256": e2.sha256_array(target),
            "teacher_prefix_ids_sha256": e2.sha256_array(prefix),
            "shared_scored_target_ids_exact": bool(
                student["target_ids_sha256"] == teacher["target_ids_sha256"]
            ),
            "teacher_scores_prefix_tokens": False,
            "arms": {"student": student, "teacher": teacher},
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            **content_fields(manifest),
            "runtime": runtime,
            "script": str(SCRIPT_PATH),
            "script_sha256": e2.sha256_file(SCRIPT_PATH),
        }
        e2.write_json(eval_path(output, "doc-a-teacher", index), receipt)
        print(
            json.dumps(
                {
                    "kind": "doc-a-teacher",
                    "window": index,
                    "source_window": manifest["windows"]["heldout"][index]["window_index"],
                    "ppl_student": student["ppl"],
                    "ppl_teacher": teacher["ppl"],
                }
            ),
            flush=True,
        )
    print(json.dumps({"status": "complete", "windows": missing, "wall_seconds": time.perf_counter() - started}))
    del model
    gc.collect()
    return 0


def load_eval_sequence(
    output: Path,
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    kind: str,
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for index in range(N_BEHAVIORAL_WINDOWS):
        path = eval_path(output, kind, index)
        if not path.is_file():
            return []
        receipts.append(validate_eval(output, manifest, arrays, kind, index))
    return receipts


def aggregate_nll(receipts: Sequence[dict[str, Any]], arm: str) -> dict[str, Any]:
    token_count = sum(int(receipt["arms"][arm]["token_count"]) for receipt in receipts)
    nll_sum = sum(float(receipt["arms"][arm]["nll_sum"]) for receipt in receipts)
    if token_count <= 0:
        raise ValueError("cannot aggregate empty NLL sequence")
    mean_nll = nll_sum / token_count
    return {"token_count": token_count, "nll_sum": nll_sum, "mean_nll": mean_nll, "ppl": math.exp(mean_nll)}


def bootstrap_ppl_triplet(
    first_nll: np.ndarray,
    second_nll: np.ndarray,
    third_nll: np.ndarray | None,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    first = np.asarray(first_nll, dtype=np.float64)
    second = np.asarray(second_nll, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1 or first.size == 0:
        raise ValueError("bootstrap PPL arrays must be equal nonempty vectors")
    third = None if third_nll is None else np.asarray(third_nll, dtype=np.float64)
    if third is not None and third.shape != first.shape:
        raise ValueError("bootstrap third arm shape mismatch")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, first.size, size=(resamples, first.size))
    first_ppl = np.exp(first[sampled].mean(axis=1))
    second_ppl = np.exp(second[sampled].mean(axis=1))
    gap = first_ppl - second_ppl
    gap_low, gap_high = np.quantile(gap, [0.025, 0.975], method="linear")
    result: dict[str, Any] = {
        "gap_ppl_ci95_low": float(gap_low),
        "gap_ppl_ci95_high": float(gap_high),
        "bootstrap_unit": "windows",
        "bootstrap_resamples": resamples,
    }
    if third is not None:
        third_ppl = np.exp(third[sampled].mean(axis=1))
        recovery = np.divide(
            first_ppl - third_ppl,
            gap,
            out=np.zeros_like(gap),
            where=gap != 0,
        )
        low, high = np.quantile(recovery, [0.025, 0.975], method="linear")
        result.update(
            {
                "recovery_ci95_low_fraction": float(low),
                "recovery_ci95_high_fraction": float(high),
                "recovery_ci95_low_percent": float(low * 100.0),
                "recovery_ci95_high_percent": float(high * 100.0),
            }
        )
    return result


def precondition(args: argparse.Namespace) -> int:
    if args.bootstrap_resamples < 2000:
        raise ValueError("precondition requires at least 2,000 bootstrap resamples")
    output = ensure_output(args.output_dir)
    manifest, arrays = load_prepared(output)
    receipts = load_eval_sequence(output, manifest, arrays, "doc-a-teacher")
    if len(receipts) != N_BEHAVIORAL_WINDOWS:
        raise RuntimeError("behavioral precondition requires all 16 DOC-A teacher receipts")
    student = aggregate_nll(receipts, "student")
    teacher = aggregate_nll(receipts, "teacher")
    gap = float(student["ppl"] - teacher["ppl"])
    bootstrap = bootstrap_ppl_triplet(
        np.asarray([receipt["arms"]["student"]["mean_nll"] for receipt in receipts]),
        np.asarray([receipt["arms"]["teacher"]["mean_nll"] for receipt in receipts]),
        None,
        resamples=args.bootstrap_resamples,
        seed=args.seed + 400,
    )
    passed = gap >= TEACHER_GAP_FLOOR_PPL
    receipt = {
        "schema": "moe_e3_behavioral_precondition_v1",
        "status": "pass" if passed else "stop",
        "created_at": now_iso(),
        "registered_floor_ppl": TEACHER_GAP_FLOOR_PPL,
        "comparison": "ppl_student - ppl_teacher",
        "ppl_student": student["ppl"],
        "ppl_teacher": teacher["ppl"],
        "teacher_gap_ppl": gap,
        "teacher_gap_bootstrap": bootstrap,
        "stop_reason": None if passed else "episodic teacher signal below registered floor",
        "teacher_receipts": [
            {"path": str(eval_path(output, "doc-a-teacher", index)), "sha256": e2.sha256_file(eval_path(output, "doc-a-teacher", index))}
            for index in range(N_BEHAVIORAL_WINDOWS)
        ],
        **content_fields(manifest),
        "script": str(SCRIPT_PATH),
        "script_sha256": e2.sha256_file(SCRIPT_PATH),
    }
    e2.write_json(output / "precondition.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "ppl_student": student["ppl"],
                "ppl_teacher": teacher["ppl"],
                "teacher_gap_ppl": gap,
                "stop_reason": receipt["stop_reason"],
            }
        ),
        flush=True,
    )
    return 0 if passed else 4


def require_precondition_pass(output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    receipt = e2.read_json(output / "precondition.json")
    links = receipt.get("teacher_receipts", [])
    links_valid = len(links) == N_BEHAVIORAL_WINDOWS and all(
        item.get("path") == str(eval_path(output, "doc-a-teacher", index))
        and item.get("sha256") == e2.sha256_file(eval_path(output, "doc-a-teacher", index))
        for index, item in enumerate(links)
    )
    if (
        receipt.get("schema") != "moe_e3_behavioral_precondition_v1"
        or receipt.get("status") != "pass"
        or receipt.get("teacher_gap_ppl", -math.inf) < TEACHER_GAP_FLOOR_PPL
        or not links_valid
        or receipt.get("prepared_windows_content_sha256")
        != manifest["prepared_windows"]["content_sha256"]
    ):
        reason = receipt.get("stop_reason") or "behavioral precondition not passed"
        raise RuntimeError(f"{reason}; STOP")
    return receipt


def eval_doc_a_expert(args: argparse.Namespace) -> int:
    requested = requested_eval_indices(args)
    output = ensure_output(args.output_dir)
    manifest, arrays = load_prepared(output)
    require_bringup_green(output, manifest)
    require_precondition_pass(output, manifest)
    pack, key, A, B = load_expert_arrays(output, manifest, arrays)
    missing: list[int] = []
    for index in requested:
        path = eval_path(output, "doc-a-expert", index)
        if path.is_file():
            validate_eval(output, manifest, arrays, "doc-a-expert", index)
        else:
            missing.append(index)
    if not missing:
        print(json.dumps({"status": "existing", "kind": "doc-a-expert", "windows": requested}))
        return 0
    torch, model, runtime = e2.load_model(args)
    layer = model.model.layers[int(pack["layer"])]
    for index in missing:
        target = arrays["heldout_ids"][index]
        with e2.ExpertMount(
            layer.mlp,
            torch_module=torch,
            key=key,
            tau=float(pack["tau"]),
            A=A,
            B=B,
            label=f"doc_a_expert_{index}",
        ) as mount:
            expert = e2.score_target_window(torch, model, target, target)
        if len(mount.calls) != 1:
            raise RuntimeError("DOC-A expert eval expected one install-layer call")
        receipt = {
            "schema": "moe_e3_doc_a_expert_eval_v1",
            "status": "complete",
            "created_at": now_iso(),
            "window_index": index,
            "source_window": manifest["windows"]["heldout"][index],
            "context_tokens": 0,
            "input_ids_sha256": e2.sha256_array(target),
            "arms": {"expert": expert},
            "fire": mount.calls[0],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            **content_fields(manifest),
            "expert_binding": expert_binding(output, pack),
            "install_layer": int(pack["layer"]),
            "runtime": runtime,
            "script": str(SCRIPT_PATH),
            "script_sha256": e2.sha256_file(SCRIPT_PATH),
        }
        e2.write_json(eval_path(output, "doc-a-expert", index), receipt)
        print(
            json.dumps(
                {
                    "kind": "doc-a-expert",
                    "window": index,
                    "ppl_expert": expert["ppl"],
                    "fire_rate": mount.calls[0]["fire_rate"],
                }
            ),
            flush=True,
        )
    del model
    gc.collect()
    return 0


def eval_noninterference(args: argparse.Namespace, kind: str) -> int:
    requested = requested_eval_indices(args)
    output = ensure_output(args.output_dir)
    manifest, arrays = load_prepared(output)
    require_bringup_green(output, manifest)
    require_precondition_pass(output, manifest)
    pack, key, A, B = load_expert_arrays(output, manifest, arrays)
    missing: list[int] = []
    for index in requested:
        path = eval_path(output, kind, index)
        if path.is_file():
            validate_eval(output, manifest, arrays, kind, index)
        else:
            missing.append(index)
    if not missing:
        print(json.dumps({"status": "existing", "kind": kind, "windows": requested}))
        return 0
    array_name = {"wikitext": "wikitext_ids", "code": "code_ids", "doc-b": "doc_b_ids"}[kind]
    torch, model, runtime = e2.load_model(args)
    layer = model.model.layers[int(pack["layer"])]
    for index in missing:
        target = arrays[array_name][index]
        if kind in {"wikitext", "doc-b"}:
            base = e2.score_target_window(torch, model, target, target)
            with e2.ExpertMount(
                layer.mlp,
                torch_module=torch,
                key=key,
                tau=float(pack["tau"]),
                A=A,
                B=B,
                label=f"{kind}_{index}",
            ) as mount:
                expert = e2.score_target_window(torch, model, target, target)
            arms: dict[str, Any] = {"base": base, "expert": expert}
            delta_percent = (expert["ppl"] - base["ppl"]) / base["ppl"] * 100.0
        else:
            ids = torch.from_numpy(np.ascontiguousarray(target[None, :], dtype=np.int64)).to(
                e2.input_device(model)
            )
            with e2.ExpertMount(
                layer.mlp,
                torch_module=torch,
                key=key,
                tau=float(pack["tau"]),
                A=A,
                B=B,
                label=f"code_{index}",
            ) as mount:
                with torch.inference_mode():
                    output_object = model(input_ids=ids, use_cache=False, logits_to_keep=1)
            del output_object, ids
            arms = {}
            delta_percent = None
        if len(mount.calls) != 1:
            raise RuntimeError(f"{kind} eval expected one install-layer call")
        schema = {
            "wikitext": "moe_e3_wikitext_noninterference_v1",
            "code": "moe_e3_code_fire_v1",
            "doc-b": "moe_e3_doc_b_selectivity_v1",
        }[kind]
        source_window = (
            manifest["windows"]["doc_b_cross_probe"][index] if kind == "doc-b" else None
        )
        receipt = {
            "schema": schema,
            "status": "complete",
            "created_at": now_iso(),
            "window_index": index,
            "source_window": source_window,
            "input_ids_sha256": e2.sha256_array(target),
            "arms": arms,
            "per_window_ppl_delta_percent": delta_percent,
            "fire": mount.calls[0],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            **content_fields(manifest),
            "expert_binding": expert_binding(output, pack),
            "install_layer": int(pack["layer"]),
            "runtime": runtime,
            "script": str(SCRIPT_PATH),
            "script_sha256": e2.sha256_file(SCRIPT_PATH),
        }
        e2.write_json(eval_path(output, kind, index), receipt)
        print(
            json.dumps(
                {
                    "kind": kind,
                    "window": index,
                    "ppl_delta_percent": delta_percent,
                    "fire_rate": mount.calls[0]["fire_rate"],
                }
            ),
            flush=True,
        )
    del model
    gc.collect()
    return 0


def eval_gates(args: argparse.Namespace) -> int:
    if args.eval_kind is None:
        raise ValueError("eval-gates requires --eval-kind")
    if args.eval_kind == "abi":
        return eval_abi(args)
    if args.eval_kind == "doc-a-teacher":
        return eval_doc_a_teacher(args)
    if args.eval_kind == "doc-a-expert":
        return eval_doc_a_expert(args)
    return eval_noninterference(args, args.eval_kind)


def bootstrap_mean(
    values: Sequence[float], *, resamples: int, seed: int, unit: str
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("bootstrap mean requires a finite nonempty vector")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, array.size, size=(resamples, array.size))
    boot = array[sampled].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975], method="linear")
    return {
        "value": float(array.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_unit": unit,
        "bootstrap_resamples": resamples,
        "per_window": [float(value) for value in array],
    }


def bootstrap_ppl_delta(
    base_nll: Sequence[float],
    expert_nll: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    base = np.asarray(base_nll, dtype=np.float64)
    expert = np.asarray(expert_nll, dtype=np.float64)
    if base.shape != expert.shape or base.ndim != 1 or base.size == 0:
        raise ValueError("PPL delta bootstrap requires equal nonempty vectors")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, base.size, size=(resamples, base.size))
    base_ppl = np.exp(base[sampled].mean(axis=1))
    expert_ppl = np.exp(expert[sampled].mean(axis=1))
    delta = (expert_ppl - base_ppl) / base_ppl * 100.0
    low, high = np.quantile(delta, [0.025, 0.975], method="linear")
    point_base = math.exp(float(base.mean()))
    point_expert = math.exp(float(expert.mean()))
    return {
        "ppl_base": point_base,
        "ppl_expert": point_expert,
        "ppl_delta_percent": (point_expert - point_base) / point_base * 100.0,
        "ci95_low_percent": float(low),
        "ci95_high_percent": float(high),
        "bootstrap_unit": "windows",
        "bootstrap_resamples": resamples,
    }


def gate_record(verdict: str, report_line: str, **details: Any) -> dict[str, Any]:
    return {"verdict": verdict, "report_line": report_line, **details}


def fmt(value: float | None, digits: int = 6) -> str:
    return "NOT_MEASURED" if value is None else f"{value:.{digits}g}"


def analyze(args: argparse.Namespace) -> int:
    if args.bootstrap_resamples < 2000:
        raise ValueError("analyze requires at least 2,000 bootstrap resamples")
    output = ensure_output(args.output_dir)
    manifest, arrays = load_prepared(output)
    doc_a = manifest["documents"]["doc_a"]
    doc_b = manifest["documents"]["doc_b"]
    lead_path = output / "GPU_E3_RESUME_COMMANDS.sh"

    bringup_receipt = e2.read_json(output / "bringup.json") if (output / "bringup.json").is_file() else None
    if bringup_receipt is None:
        g1 = gate_record("NOT_MEASURED", f"E3-G-1 NOT_MEASURED — lead script: {lead_path}")
    else:
        if (
            bringup_receipt.get("schema") != "moe_e3_bringup_v1"
            or bringup_receipt.get("model_revision") != MODEL_REVISION
            or bringup_receipt.get("prepared_windows_content_sha256")
            != manifest["prepared_windows"]["content_sha256"]
        ):
            raise RuntimeError("bringup receipt content contract changed")
        verdict = bringup_receipt["gate"]["verdict"]
        g1 = gate_record(
            verdict,
            (
                f"E3-G-1 {verdict} — WikiText ppl512={bringup_receipt['short_512']['ppl']:.6g}, "
                f"nll2048-nll512={bringup_receipt['long_minus_short_mean_nll']:+.6g} nats"
            ),
            ppl512=bringup_receipt["short_512"]["ppl"],
            long_minus_short_mean_nll=bringup_receipt["long_minus_short_mean_nll"],
            inheritance_mode=bringup_receipt["inheritance"]["mode"],
        )

    fit_receipt = (
        validate_fit_key_inputs(output, manifest, arrays)
        if (output / "fit_key.json").is_file()
        else None
    )
    selected: dict[str, Any] | None = None
    if fit_receipt is None:
        g2 = gate_record("NOT_MEASURED", f"E3-G2 NOT_MEASURED — lead script: {lead_path}")
        install_row = None
    else:
        decision = fit_receipt["decision"]
        if decision["install_layer"] is not None:
            selected = fit_receipt["rows"][int(decision["install_layer"])]
        verdict = decision["verdict"]
        if selected is None:
            g2 = gate_record(verdict, f"E3-G2 {verdict} — no qualifying install layer")
            install_row = None
        else:
            key_eval = selected["metrics"]["eval"]
            g2 = gate_record(
                verdict,
                (
                    f"E3-G2 {verdict} — L*={decision['install_layer']}, "
                    f"recall={key_eval['doc_a_recall']['value']:.6g}, "
                    f"code={key_eval['code_fpr']['value']:.6g}, "
                    f"GRM={key_eval['grm_fpr']['value']:.6g}, "
                    f"guides={key_eval['guides_fpr']['value']:.6g}, "
                    f"WikiText={key_eval['wikitext_fpr']['value']:.6g} descriptive, "
                    f"DOC-B={selected['metrics']['doc_b_cross_episode_fire_descriptive']['value']:.6g} descriptive"
                ),
                install_layer=decision["install_layer"],
                metrics=selected["metrics"],
            )
            install_row = {
                "layer": decision["install_layer"],
                "rank": 64,
                "tau": selected["metrics"]["selected_tau"],
                "recall": key_eval["doc_a_recall"],
                "wikitext_fire": key_eval["wikitext_fpr"],
                "code_fpr": key_eval["code_fpr"],
                "grm_fpr": key_eval["grm_fpr"],
                "guides_fpr": key_eval["guides_fpr"],
                "doc_b_fire": selected["metrics"]["doc_b_cross_episode_fire_descriptive"],
            }

    training = (
        validate_training(output, manifest, arrays)
        if (output / "train.json").is_file()
        else None
    )
    if training is None:
        g3 = gate_record("NOT_MEASURED", f"E3-G3 NOT_MEASURED — lead script: {lead_path}")
    else:
        metrics = training["stored_fp16_validation"]
        verdict = training["verdict"]
        g3 = gate_record(
            verdict,
            (
                f"E3-G3 {verdict} — zero MSE={metrics['zero_predictor_mse']:.6g}, "
                f"adapter MSE={metrics['adapter_mse']:.6g}, "
                f"improvement={fmt(metrics['improvement_percent'])}%"
            ),
            metrics=metrics,
        )

    abi = (
        validate_eval(output, manifest, arrays, "abi")
        if eval_path(output, "abi").is_file()
        else None
    )
    teacher_receipts = load_eval_sequence(output, manifest, arrays, "doc-a-teacher")
    pre = e2.read_json(output / "precondition.json") if (output / "precondition.json").is_file() else None
    if pre is not None and pre.get("prepared_windows_content_sha256") != manifest["prepared_windows"]["content_sha256"]:
        raise RuntimeError("precondition content binding changed")
    if pre is not None and len(teacher_receipts) == N_BEHAVIORAL_WINDOWS:
        pre_student = aggregate_nll(teacher_receipts, "student")["ppl"]
        pre_teacher = aggregate_nll(teacher_receipts, "teacher")["ppl"]
        if (
            pre.get("ppl_student") != pre_student
            or pre.get("ppl_teacher") != pre_teacher
            or pre.get("teacher_gap_ppl") != pre_student - pre_teacher
        ):
            raise RuntimeError("precondition metrics no longer match teacher receipts")
    expert_receipts = load_eval_sequence(output, manifest, arrays, "doc-a-expert")
    g4_measurement: dict[str, Any] | None = None
    if pre is not None and pre.get("status") == "stop":
        g4 = gate_record(
            "NOT_MEASURED",
            "E3-G4 NOT_MEASURED — STOP: episodic teacher signal below registered floor",
            precondition=pre,
        )
    elif len(teacher_receipts) == 16 and len(expert_receipts) == 16 and pre is not None and pre.get("status") == "pass":
        student = aggregate_nll(teacher_receipts, "student")
        teacher = aggregate_nll(teacher_receipts, "teacher")
        expert = aggregate_nll(expert_receipts, "expert")
        gap = student["ppl"] - teacher["ppl"]
        recovery = (student["ppl"] - expert["ppl"]) / gap
        bootstrap = bootstrap_ppl_triplet(
            np.asarray([receipt["arms"]["student"]["mean_nll"] for receipt in teacher_receipts]),
            np.asarray([receipt["arms"]["teacher"]["mean_nll"] for receipt in teacher_receipts]),
            np.asarray([receipt["arms"]["expert"]["mean_nll"] for receipt in expert_receipts]),
            resamples=args.bootstrap_resamples,
            seed=args.seed + 500,
        )
        g4_measurement = {
            "ppl_student": student["ppl"],
            "ppl_teacher": teacher["ppl"],
            "ppl_expert": expert["ppl"],
            "teacher_gap_ppl": gap,
            "recovery_fraction": recovery,
            "recovery_percent": recovery * 100.0,
            **bootstrap,
            "expert_fire": bootstrap_mean(
                [receipt["fire"]["fire_rate"] for receipt in expert_receipts],
                resamples=args.bootstrap_resamples,
                seed=args.seed + 501,
                unit="DOC-A heldout windows",
            ),
        }
        verdict = (
            "GREEN"
            if recovery >= G4_RECOVERY_FLOOR
            and bootstrap["recovery_ci95_low_fraction"] > 0.0
            else "RED"
        )
        g4 = gate_record(
            verdict,
            (
                f"E3-G4 {verdict} — recovery={recovery * 100.0:.6g}% "
                f"CI95=[{bootstrap['recovery_ci95_low_percent']:.6g}%, "
                f"{bootstrap['recovery_ci95_high_percent']:.6g}%]"
            ),
            measurement=g4_measurement,
        )
    else:
        g4 = gate_record("NOT_MEASURED", f"E3-G4 NOT_MEASURED — lead script: {lead_path}")

    wikitext_receipts = load_eval_sequence(output, manifest, arrays, "wikitext")
    code_receipts = load_eval_sequence(output, manifest, arrays, "code")
    doc_b_receipts = load_eval_sequence(output, manifest, arrays, "doc-b")
    wikitext_metrics: dict[str, Any] | None = None
    code_metrics: dict[str, Any] | None = None
    doc_b_metrics: dict[str, Any] | None = None
    if len(wikitext_receipts) == 16:
        wikitext_metrics = bootstrap_ppl_delta(
            [receipt["arms"]["base"]["mean_nll"] for receipt in wikitext_receipts],
            [receipt["arms"]["expert"]["mean_nll"] for receipt in wikitext_receipts],
            resamples=args.bootstrap_resamples,
            seed=args.seed + 600,
        )
        wikitext_metrics["fire"] = bootstrap_mean(
            [receipt["fire"]["fire_rate"] for receipt in wikitext_receipts],
            resamples=args.bootstrap_resamples,
            seed=args.seed + 601,
            unit="WikiText windows",
        )
    if len(code_receipts) == 16:
        code_metrics = bootstrap_mean(
            [receipt["fire"]["fire_rate"] for receipt in code_receipts],
            resamples=args.bootstrap_resamples,
            seed=args.seed + 610,
            unit="code windows",
        )
    if len(doc_b_receipts) == 16:
        doc_b_metrics = bootstrap_ppl_delta(
            [receipt["arms"]["base"]["mean_nll"] for receipt in doc_b_receipts],
            [receipt["arms"]["expert"]["mean_nll"] for receipt in doc_b_receipts],
            resamples=args.bootstrap_resamples,
            seed=args.seed + 620,
        )
        doc_b_metrics["fire"] = bootstrap_mean(
            [receipt["fire"]["fire_rate"] for receipt in doc_b_receipts],
            resamples=args.bootstrap_resamples,
            seed=args.seed + 621,
            unit="DOC-B cross-probe windows",
        )
    if pre is not None and pre.get("status") == "stop":
        g5 = gate_record(
            "NOT_MEASURED",
            "E3-G5 NOT_MEASURED — STOP: episodic teacher signal below registered floor",
        )
    elif wikitext_metrics is not None and code_metrics is not None:
        verdict = (
            "GREEN"
            if wikitext_metrics["ppl_delta_percent"] <= G5_WIKITEXT_PPL_DELTA_CAP_PERCENT
            and code_metrics["value"] <= G5_CODE_FIRE_CAP
            else "RED"
        )
        g5 = gate_record(
            verdict,
            (
                f"E3-G5 {verdict} — WikiText ppl delta={wikitext_metrics['ppl_delta_percent']:+.6g}% "
                f"CI95=[{wikitext_metrics['ci95_low_percent']:+.6g}%, "
                f"{wikitext_metrics['ci95_high_percent']:+.6g}%], "
                f"code fire={code_metrics['value']:.6g}"
            ),
            wikitext=wikitext_metrics,
            code=code_metrics,
        )
    else:
        g5 = gate_record("NOT_MEASURED", f"E3-G5 NOT_MEASURED — lead script: {lead_path}")

    gates = {"E3-G-1": g1, "E3-G2": g2, "E3-G3": g3, "E3-G4": g4, "E3-G5": g5}
    all_gate_green = all(value["verdict"] == "GREEN" for value in gates.values())
    abi_green = abi is not None and abi.get("verdict") == "GREEN"
    analysis = {
        "schema": "moe_e3_analysis_v1",
        "status": (
            "complete_all_green"
            if all_gate_green and abi_green
            else "complete_with_red_or_not_measured"
        ),
        "created_at": now_iso(),
        "documents": {"doc_a": doc_a, "doc_b": doc_b},
        **content_fields(manifest),
        "gates": gates,
        "abi_carry": (
            {"verdict": "NOT_MEASURED", "lead_script": str(lead_path)}
            if abi is None
            else {
                "verdict": abi["verdict"],
                "zero_install": abi["zero_experts_installed"]["bit_equal_final_hidden_and_logits"],
                "zero_B": abi["zero_initialized_B_mounted"]["bit_equal_final_hidden_and_logits"],
                "forced_no_fire": abi["forced_no_fire_mounted"]["bit_equal_final_hidden_and_logits"],
                "live_nonfired_exact": abi["live_gate"]["nonfired_rows_bit_equal_at_native_mlp_side_path"],
            }
        ),
        "install_row": install_row,
        "behavioral_precondition": pre,
        "g4_row": g4_measurement,
        "wikitext_noninterference": wikitext_metrics,
        "code_fire": code_metrics,
        "doc_b_selectivity": doc_b_metrics,
        "lead_script": str(lead_path),
        "expertpack": str(output / EXPERTPACK_NAME),
        "limitations": [
            "One frozen base model and one registered novel episode; behavioral inference evidence only.",
            "Finite measurements do not establish general episodic consolidation.",
            "Novel text is never excerpted; reports contain only provenance identities and aggregate metrics.",
        ],
        "script": str(SCRIPT_PATH),
        "script_sha256": e2.sha256_file(SCRIPT_PATH),
    }
    analysis_path = output / "analysis.json"
    e2.write_json(analysis_path, analysis)

    lines = [
        "# MOE-E3 episodic consolidation report",
        "",
        "## Gate status",
        "",
        *[record["report_line"] for record in gates.values()],
        "",
        "## Registered documents",
        "",
        "| Role | Path | Bytes | SHA-256 | Token count | Full 512-token windows |",
        "|---|---|---:|---|---:|---:|",
        f"| DOC-A | `{doc_a['path']}` | {doc_a['byte_count']} | `{doc_a['sha256']}` | {doc_a['token_count']} | {doc_a['full_512_window_count']} |",
        f"| DOC-B | `{doc_b['path']}` | {doc_b['byte_count']} | `{doc_b['sha256']}` | {doc_b['token_count']} | {doc_b['full_512_window_count']} |",
        "",
        "No corpus excerpt is included. Token counts use the frozen local OLMoE tokenizer with `add_special_tokens=false`.",
        "",
        "## Install row",
        "",
        "| L* | r | tau | DOC-A recall (95% CI) | WikiText fire (95% CI, descriptive) | code FPR (95% CI) | GRM FPR (95% CI) | guides FPR (95% CI) | DOC-B key fire (95% CI, descriptive) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if install_row is None:
        lines.append("| NOT_MEASURED | 64 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |")
    else:
        def rate_cell(record: dict[str, Any]) -> str:
            return f"{record['value']:.6g} [{record['ci95_low']:.6g}, {record['ci95_high']:.6g}]"

        lines.append(
            f"| {install_row['layer']} | 64 | {install_row['tau']:.10g} | "
            f"{rate_cell(install_row['recall'])} | {rate_cell(install_row['wikitext_fire'])} | "
            f"{rate_cell(install_row['code_fpr'])} | {rate_cell(install_row['grm_fpr'])} | "
            f"{rate_cell(install_row['guides_fpr'])} | {rate_cell(install_row['doc_b_fire'])} |"
        )
    lines.extend(
        [
            "",
            "Layer selection and tau use FIT-side data only after the registered eval qualification filter. DOC-B is descriptive and never enters fitting, threshold selection, layer selection, or a gate.",
            "",
            "## E3-G3 validation",
            "",
            "| Zero MSE | Adapter MSE | Improvement | Bootstrap 95% CI | Gate |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    if training is None:
        lines.append("| NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |")
    else:
        metrics = training["stored_fp16_validation"]
        ci = metrics["improvement_ci95_fraction"]
        lines.append(
            f"| {metrics['zero_predictor_mse']:.8g} | {metrics['adapter_mse']:.8g} | "
            f"{fmt(metrics['improvement_percent'])}% | [{ci['ci95_low'] * 100.0:.6g}%, {ci['ci95_high'] * 100.0:.6g}%] | {training['verdict']} |"
        )
    lines.extend(
        [
            "",
            "## Behavioral precondition and E3-G4",
            "",
            "Registered precondition: `ppl_student - ppl_teacher >= 0.5`. Failure stops expert and non-interference evaluation with the exact reason `episodic teacher signal below registered floor`.",
            "",
            "| ppl_student | ppl_teacher | teacher gap (95% CI) | ppl_expert | Recovery | Bootstrap 95% CI | Gate |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    if g4_measurement is None:
        student_value = None if pre is None else pre.get("ppl_student")
        teacher_value = None if pre is None else pre.get("ppl_teacher")
        gap_value = None if pre is None else pre.get("teacher_gap_ppl")
        gap_cell = fmt(gap_value)
        if pre is not None:
            gap_bootstrap = pre["teacher_gap_bootstrap"]
            gap_cell = (
                f"{gap_value:.6g} [{gap_bootstrap['gap_ppl_ci95_low']:.6g}, "
                f"{gap_bootstrap['gap_ppl_ci95_high']:.6g}]"
            )
        lines.append(
            f"| {fmt(student_value)} | {fmt(teacher_value)} | {gap_cell} | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | {g4['verdict']} |"
        )
    else:
        lines.append(
            f"| {g4_measurement['ppl_student']:.6g} | {g4_measurement['ppl_teacher']:.6g} | "
            f"{g4_measurement['teacher_gap_ppl']:.6g} [{g4_measurement['gap_ppl_ci95_low']:.6g}, {g4_measurement['gap_ppl_ci95_high']:.6g}] | "
            f"{g4_measurement['ppl_expert']:.6g} | "
            f"{g4_measurement['recovery_percent']:.6g}% | "
            f"[{g4_measurement['recovery_ci95_low_percent']:.6g}%, {g4_measurement['recovery_ci95_high_percent']:.6g}%] | {g4['verdict']} |"
        )
    lines.extend(
        [
            "",
            "## Non-interference and DOC-B selectivity",
            "",
            "| Probe | Base ppl | Expert ppl | Ppl delta (95% CI) | Fire rate (95% CI) | Role |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    if wikitext_metrics is None:
        lines.append("| WikiText | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | E3-G5 |")
    else:
        fire = wikitext_metrics["fire"]
        lines.append(
            f"| WikiText | {wikitext_metrics['ppl_base']:.6g} | {wikitext_metrics['ppl_expert']:.6g} | "
            f"{wikitext_metrics['ppl_delta_percent']:+.6g}% [{wikitext_metrics['ci95_low_percent']:+.6g}%, {wikitext_metrics['ci95_high_percent']:+.6g}%] | "
            f"{fire['value']:.6g} [{fire['ci95_low']:.6g}, {fire['ci95_high']:.6g}] | E3-G5 |"
        )
    if code_metrics is None:
        lines.append("| code | n/a | n/a | n/a | NOT_MEASURED | E3-G5 |")
    else:
        lines.append(
            f"| code | n/a | n/a | n/a | {code_metrics['value']:.6g} "
            f"[{code_metrics['ci95_low']:.6g}, {code_metrics['ci95_high']:.6g}] | E3-G5 |"
        )
    if doc_b_metrics is None:
        lines.append("| DOC-B | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | descriptive |")
    else:
        fire = doc_b_metrics["fire"]
        lines.append(
            f"| DOC-B | {doc_b_metrics['ppl_base']:.6g} | {doc_b_metrics['ppl_expert']:.6g} | "
            f"{doc_b_metrics['ppl_delta_percent']:+.6g}% [{doc_b_metrics['ci95_low_percent']:+.6g}%, {doc_b_metrics['ci95_high_percent']:+.6g}%] | "
            f"{fire['value']:.6g} [{fire['ci95_low']:.6g}, {fire['ci95_high']:.6g}] | descriptive |"
        )
    abi_line = (
        "ABI carry NOT_MEASURED."
        if abi is None
        else (
            f"ABI carry {abi['verdict']}: zero-install={abi['zero_experts_installed']['bit_equal_final_hidden_and_logits']}, "
            f"zero-B={abi['zero_initialized_B_mounted']['bit_equal_final_hidden_and_logits']}, "
            f"forced-no-fire={abi['forced_no_fire_mounted']['bit_equal_final_hidden_and_logits']}, "
            f"live non-fired exact={abi['live_gate']['nonfired_rows_bit_equal_at_native_mlp_side_path']}."
        )
    )
    lines.extend(
        [
            "",
            "## ABI and provenance",
            "",
            abi_line,
            "",
            f"Prepared content SHA-256: `{manifest['prepared_windows']['content_sha256']}`.",
            "",
            f"E2 negative lineage content SHA-256: `{E2_LEGACY_CONTENT_SHA256}`. The reused router captures are individually bound through immutable E2 provenance-validation sidecars.",
            "",
            f"ExpertPack: `{output / EXPERTPACK_NAME}`.",
            "",
            f"Lead script: `{lead_path}`.",
            "",
            "## Evidence limits",
            "",
            "This is behavioral inference evidence for one frozen base model, one consolidated document, and one cross-probe document. It is not a general capability claim. A RED gate is retained as a result; thresholds are not widened after measurement.",
            "",
        ]
    )
    report_path = output / "MOE_E3_REPORT.md"
    e2.write_text(report_path, "\n".join(lines))
    pack_path = output / EXPERTPACK_NAME / "manifest.json"
    pack = e2.read_json(pack_path)
    pack["updated_at"] = now_iso()
    pack["analysis"] = {
        "path": str(analysis_path),
        "sha256": e2.sha256_file(analysis_path),
        "report": str(report_path),
        "report_sha256": e2.sha256_file(report_path),
        "gates": {name: record["verdict"] for name, record in gates.items()},
        "abi_carry": analysis["abi_carry"]["verdict"],
    }
    if training is None:
        pack["status"] = pack.get("status", "prepared_pending_address")
    elif pre is not None and pre.get("status") == "stop":
        pack["status"] = "trained_stopped_teacher_signal_below_floor"
        pack["limitations"] = ["episodic teacher signal below registered floor"]
    elif g4["verdict"] != "NOT_MEASURED" and g5["verdict"] != "NOT_MEASURED":
        pack["status"] = "evaluated_all_gates_and_abi_green" if g4["verdict"] == g5["verdict"] == "GREEN" and abi_green else "evaluated_with_red_or_missing_abi"
        pack["limitations"] = analysis["limitations"]
    e2.write_json(pack_path, pack)
    print(
        json.dumps(
            {
                "status": analysis["status"],
                "gates": {name: record["verdict"] for name, record in gates.items()},
                "report": str(report_path),
            }
        ),
        flush=True,
    )
    if all_gate_green and abi_green:
        return 0
    if pre is not None and pre.get("status") == "stop":
        return 4
    return 3


def self_test(args: argparse.Namespace) -> int:
    output = ensure_output(args.output_dir)
    manifest, arrays = load_prepared(output)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    documents = discover_documents()
    check(
        "two_largest_eligible_documents_frozen",
        [str(path) for path in documents[:2]]
        == [manifest["documents"]["doc_a"]["path"], manifest["documents"]["doc_b"]["path"]],
        [str(path) for path in documents[:2]],
    )
    check(
        "selected_paths_obey_grant",
        all(
            path.name.endswith("_COMPLETE.txt")
            and not any(term in str(path).lower() for term in FORBIDDEN_PATH_TERMS)
            for path in documents[:2]
        ),
        "glob and exclusions",
    )
    heldout_source = [row["window_index"] for row in manifest["windows"]["heldout"]]
    check(
        "regions_disjoint_and_registered",
        [row["window_index"] for row in manifest["windows"]["pair"]] == list(PAIR_DOC_WINDOWS)
        and [row["window_index"] for row in manifest["windows"]["doc_a_key"]] == list(DOC_A_KEY_WINDOWS)
        and not set(PAIR_DOC_WINDOWS) & set(heldout_source),
        {"pair": [4, 67], "key": [68, 83], "heldout": [min(heldout_source), max(heldout_source)]},
    )
    prefix_hashes_ok = all(
        row["teacher_prefix_ids_sha256"] == e2.sha256_array(arrays["pair_prefix_ids"][index])
        for index, row in enumerate(manifest["windows"]["pair"])
    ) and all(
        row["teacher_prefix_ids_sha256"] == e2.sha256_array(arrays["heldout_prefix_ids"][index])
        for index, row in enumerate(manifest["windows"]["heldout"])
    )
    check("true_prefix_hashes_bound", prefix_hashes_ok, "64 pair and 16 heldout prefixes")
    check(
        "e2_negative_lineage_content_bound",
        manifest["e2_negative_pool"]["legacy_content_sha256"] == E2_LEGACY_CONTENT_SHA256,
        manifest["e2_negative_pool"]["legacy_content_sha256"],
    )
    negative_validation = [validate_e2_negative_capture(arrays, corpus, 0) for corpus in NEGATIVE_CORPORA]
    check(
        "e2_negative_capture_sidecars_validate",
        all(row["content_sha256"] == E2_LEGACY_CONTENT_SHA256 for row in negative_validation),
        [row["corpus"] for row in negative_validation],
    )

    rng = np.random.default_rng(args.seed)
    dimension = 16
    native = rng.standard_normal((2, 7, dimension)).astype(np.float32)
    hidden_small = rng.standard_normal((2, 7, dimension)).astype(np.float32)
    key_small, _norm = e2.unit_vector(rng.standard_normal(dimension), name="E3 synthetic key")
    A_small = rng.standard_normal((4, dimension)).astype(np.float32) * 0.1
    B_small = rng.standard_normal((dimension, 4)).astype(np.float32) * 0.1
    tau_small = float(np.median(hidden_small.reshape(-1, dimension) @ key_small))
    result, _scores, fired = e2.numpy_expert_add(native, hidden_small, key_small, tau_small, A_small, B_small)
    check("abi_has_mixed_gate", bool(fired.any() and (~fired).any()), fired.tolist())
    check(
        "abi_nonfired_bit_identity",
        np.array_equal(result.reshape(-1, dimension)[~fired], native.reshape(-1, dimension)[~fired]),
        "np.array_equal",
    )
    zero, _scores, _fired = e2.numpy_expert_add(native, hidden_small, key_small, tau_small, A_small, np.zeros_like(B_small))
    check("abi_zero_B_bit_identity", np.array_equal(zero, native), "np.array_equal")

    import torch

    mount_module = torch.nn.Identity()
    mount_hidden = torch.zeros((1, 4, dimension), dtype=torch.bfloat16)
    mount_hidden[0, :, 0] = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    mount_key = np.zeros(dimension, dtype=np.float32)
    mount_key[0] = 1.0
    mount_A = np.zeros((4, dimension), dtype=np.float16)
    mount_A[:, 0] = 1.0
    mount_B = np.zeros((dimension, 4), dtype=np.float16)
    mount_B[0, :] = 0.25
    baseline = mount_module(mount_hidden)
    with e2.ExpertMount(
        mount_module,
        torch_module=torch,
        key=mount_key,
        tau=0.0,
        A=mount_A,
        B=mount_B,
        label="E3 self-test live",
    ) as mount:
        mounted = mount_module(mount_hidden)
    check(
        "mount_live_mixed_and_nonfire_exact",
        mount.calls[0]["fire_count"] == 2
        and mount.calls[0]["nonfire_count"] == 2
        and mount.calls[0]["nonfired_rows_bit_equal"]
        and not torch.equal(mounted, baseline),
        mount.calls[0],
    )

    nonfinite = {"bad": float("nan")}
    with tempfile.TemporaryDirectory(prefix="olmoe-e3-self-test-", dir="/tmp") as temporary:
        probe = Path(temporary) / "nonfinite.json"
        failed_loud = False
        try:
            e2.write_json(probe, nonfinite)
        except ValueError as exc:
            failed_loud = "$.bad" in str(exc) and not probe.exists()
    check("writer_rejects_nonfinite_before_write", failed_loud, "$.bad named")
    content = prepared_content_record(arrays)
    check(
        "prepared_content_digest_revalidates",
        content["sha256"] == manifest["prepared_windows"]["content_sha256"],
        content["sha256"],
    )

    direction = e2.unit_vector(rng.standard_normal(dimension), name="E3 K4 fixture direction")[0]
    synthetic: dict[str, np.ndarray] = {}
    for corpus, shift in (
        ("doc_a", 1.2),
        ("wikitext", 0.0),
        ("code", -0.2),
        ("grm", -0.4),
        ("guides", 0.1),
    ):
        values = rng.standard_normal((16, 8, dimension)).astype(np.float32) * 0.4
        values += shift * direction
        synthetic[corpus] = values
    fitted, metadata = fit_k4_direction_five(synthetic, cg_rtol=1.0e-6, cg_maxiter=100)
    scores = score_hidden(synthetic, fitted)
    tau = select_tau_five(scores)["selected_tau"]
    mutated = {name: value.copy() for name, value in synthetic.items()}
    for value in mutated.values():
        value[np.asarray(EVAL_INDICES)] += rng.standard_normal(value[np.asarray(EVAL_INDICES)].shape).astype(np.float32) * 100.0
    fitted_mutated, metadata_mutated = fit_k4_direction_five(mutated, cg_rtol=1.0e-6, cg_maxiter=100)
    tau_mutated = select_tau_five(score_hidden(mutated, fitted_mutated))["selected_tau"]
    check("k4_unit_norm", abs(float(np.linalg.norm(fitted)) - 1.0) < 1.0e-5, float(np.linalg.norm(fitted)))
    check("k4_eval_mutation_key_exact", np.array_equal(fitted, fitted_mutated), e2.sha256_array(fitted_mutated))
    check(
        "k4_eval_mutation_alpha_exact",
        metadata["selected_alpha"] == metadata_mutated["selected_alpha"],
        [metadata["selected_alpha"], metadata_mutated["selected_alpha"]],
    )
    check("k4_eval_mutation_tau_exact", tau == tau_mutated, [tau, tau_mutated])
    recovery_fixture = bootstrap_ppl_triplet(
        np.full(16, math.log(20.0)),
        np.full(16, math.log(10.0)),
        np.full(16, math.log(15.0)),
        resamples=max(200, args.bootstrap_resamples),
        seed=args.seed,
    )
    check(
        "bootstrap_recovery_fixture_50pct",
        abs(recovery_fixture["recovery_ci95_low_percent"] - 50.0) < 1.0e-9
        and abs(recovery_fixture["recovery_ci95_high_percent"] - 50.0) < 1.0e-9,
        recovery_fixture,
    )
    passed = all(row["passed"] for row in checks)
    receipt = {
        "schema": "moe_e3_cpu_validation_v1",
        "status": "passed" if passed else "failed",
        "created_at": now_iso(),
        "synthetic_only_not_gate_evidence": True,
        **content_fields(manifest),
        "checks_passed": sum(row["passed"] for row in checks),
        "checks_total": len(checks),
        "checks": checks,
        "script": str(SCRIPT_PATH),
        "script_sha256": e2.sha256_file(SCRIPT_PATH),
    }
    e2.write_json(output / "cpu_validation.json", receipt)
    print(json.dumps({"status": receipt["status"], "checks": len(checks)}), flush=True)
    return 0 if passed else 3


def lead_script_text(output: Path) -> str:
    script = SCRIPT_PATH.relative_to(REPO_ROOT).as_posix()
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {REPO_ROOT}",
        "",
        "run_gpu_e3() {",
        f"  {GPU_PREAMBLE} \"$@\"",
        "  gpu_rc_e3=$?",
        "  sleep 30",
        "  return \"${gpu_rc_e3}\"",
        "}",
        "",
        "run_cpu_e3() {",
        "  timeout --signal=TERM --kill-after=5s 7200s env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DEACTIVATE_ASYNC_LOAD=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/olmoe_e3_pycache PYTHONUNBUFFERED=1 \"$@\"",
        "}",
        "",
        "run_or_analyze_e3() {",
        "  set +e",
        "  \"$@\"",
        "  stage_rc_e3=$?",
        "  set -e",
        "  if [ \"${stage_rc_e3}\" -ne 0 ]; then",
        "    set +e",
        f"    run_cpu_e3 python3 {script} analyze --bootstrap-resamples 2000",
        "    set -e",
        "    exit \"${stage_rc_e3}\"",
        "  fi",
        "}",
        "",
        f"run_or_analyze_e3 run_cpu_e3 python3 {script} prepare",
        f"run_or_analyze_e3 run_cpu_e3 python3 {script} self-test --bootstrap-resamples 2000",
        f"run_or_analyze_e3 run_gpu_e3 python3 {script} bringup --device auto --max-gpu-memory 11GiB",
        "",
        "# DOC-A address captures and DOC-B descriptive cross-episode captures.",
    ]
    for corpus in CAPTURE_CORPORA:
        for start in range(0, N_KEY_WINDOWS, 4):
            lines.append(
                f"run_or_analyze_e3 run_gpu_e3 python3 {script} capture-keys --device auto --max-gpu-memory 11GiB --corpus {corpus} --start-window {start} --count 4"
            )
    lines.extend(
        [
            "",
            "# K4 reuses the individually validated E2 negative captures read-only.",
            f"run_or_analyze_e3 run_cpu_e3 python3 {script} fit-key --threads 8 --bootstrap-resamples 2000",
            "",
            "# True preceding-token teacher/student pairs, four per bounded lease.",
        ]
    )
    for start in range(0, N_PAIR_WINDOWS, 4):
        lines.append(
            f"run_or_analyze_e3 run_gpu_e3 python3 {script} capture-pairs --device auto --max-gpu-memory 11GiB --start-window {start} --count 4"
        )
    lines.extend(
        [
            "",
            f"run_or_analyze_e3 run_cpu_e3 python3 {script} train --threads 8 --rank 64 --train-tokens-per-window 128 --bootstrap-resamples 2000",
            "",
            "# Exact ABI identity stays CPU BF16 to avoid CUDA reduction nondeterminism.",
            f"run_or_analyze_e3 timeout --signal=TERM --kill-after=5s 590s env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DEACTIVATE_ASYNC_LOAD=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/olmoe_e3_pycache PYTHONUNBUFFERED=1 python3 {script} eval-gates --device cpu --eval-kind abi",
            "",
            "# Measure only student and teacher first. The registered floor controls STOP.",
        ]
    )
    for start in range(0, N_BEHAVIORAL_WINDOWS, 2):
        lines.append(
            f"run_or_analyze_e3 run_gpu_e3 python3 {script} eval-gates --device auto --max-gpu-memory 11GiB --eval-kind doc-a-teacher --start-window {start} --count 2"
        )
    lines.append(
        f"run_or_analyze_e3 run_cpu_e3 python3 {script} precondition --bootstrap-resamples 2000"
    )
    lines.extend(["", "# The expert and G5 probes run only after the precondition passes."])
    for start in range(0, N_BEHAVIORAL_WINDOWS, 2):
        lines.append(
            f"run_or_analyze_e3 run_gpu_e3 python3 {script} eval-gates --device auto --max-gpu-memory 11GiB --eval-kind doc-a-expert --start-window {start} --count 2"
        )
    for kind, stride in (("wikitext", 4), ("code", 4), ("doc-b", 2)):
        lines.append("")
        for start in range(0, N_BEHAVIORAL_WINDOWS, stride):
            lines.append(
                f"run_or_analyze_e3 run_gpu_e3 python3 {script} eval-gates --device auto --max-gpu-memory 11GiB --eval-kind {kind} --start-window {start} --count {stride}"
            )
    lines.extend(
        [
            "",
            "set +e",
            f"run_cpu_e3 python3 {script} analyze --bootstrap-resamples 2000",
            "analysis_rc_e3=$?",
            "set -e",
            "exit \"${analysis_rc_e3}\"",
            "",
        ]
    )
    return "\n".join(lines)


def write_lead_script(output: Path) -> None:
    path = output / "GPU_E3_RESUME_COMMANDS.sh"
    e2.write_text(path, lead_script_text(output))
    path.chmod(0o755)


def main() -> int:
    args = parse_args()
    if args.seed != DEFAULT_SEED:
        raise ValueError(f"registered E3 seed is frozen at {DEFAULT_SEED}")
    if args.mode == "prepare":
        return prepare(args)
    if args.mode == "self-test":
        return self_test(args)
    if args.mode == "bringup":
        return bringup(args)
    if args.mode == "capture-keys":
        return capture_keys(args)
    if args.mode == "fit-key":
        return fit_key(args)
    if args.mode == "capture-pairs":
        return capture_pairs(args)
    if args.mode == "train":
        return train(args)
    if args.mode == "eval-gates":
        return eval_gates(args)
    if args.mode == "precondition":
        return precondition(args)
    return analyze(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(json.dumps({"status": "interrupted"}), file=sys.stderr, flush=True)
        raise
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            ),
            file=sys.stderr,
            flush=True,
        )
        raise
