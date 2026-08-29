#!/usr/bin/env python3
"""MOE-E2: grow frozen OLMoE-1B-7B with one gated narrative expert.

The script is intentionally self-contained and has no GPT-OSS imports.  It
implements the registered stages as resumable modes:

  prepare / self-test / validate-provenance / load-check / bringup /
  capture-keys / fit-key / capture-pairs / train / eval-gates /
  fallback-check / analyze

Model and dataset access is local-only.  CUDA runs first complete Transformers'
fused-expert conversion on CPU, hard-gate
the finalized checkpoint load report, and only then apply Accelerate placement
capped at 11 GiB on CUDA device 0.  No quantized model path is accepted.  Each
invocation writes an independent receipt so the lead-side shell scripts can keep
every GPU lease below 590 s.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import platform
import re
import resource
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("PYTHONPYCACHEPREFIX", "/tmp/olmoe_e2_pycache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
ORDER_PATH = REPO_ROOT / "orders" / "MOE_E2_OLMOE_TESTBED.md"
F4_ORDER_PATH = REPO_ROOT / "orders" / "MOE_E2_F4_CONTENT_PROVENANCE.md"
GLC_PATH = REPO_ROOT / "docs" / "GLC_LONG_CONTEXT_SYNTHESIS.md"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "moe_e2"
DEFAULT_MODEL_DIR = Path(
    "/home/vader/.cache/huggingface/hub/"
    "models--allenai--OLMoE-1B-7B-0924/"
    "snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5"
)
WIKITEXT_SNAPSHOT = Path(
    "/home/vader/.cache/huggingface/hub/datasets--wikitext/"
    "snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/"
    "wikitext-2-raw-v1"
)
CORPUS_ROOTS = (
    Path(
        "/mnt/Shared/01 - Narrative Project/01 - Narrative Research/"
        "Methodology/GUIDELINES/Active_Guides"
    ),
    Path(
        "/mnt/Shared/01 - Narrative Project/01 - Narrative Research/"
        "Methodology/GUIDELINES/Active_J_Guides"
    ),
    Path(
        "/mnt/Shared/01 - Narrative Project/01 - Narrative Research/"
        "Methodology/GUIDELINES/Active_O_Guides"
    ),
)
FORBIDDEN_DIRECTORY_TERMS = ("consciousness", "rcft", "thesis")

MODEL_REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
MODEL_ID = "allenai/OLMoE-1B-7B-0924"
TRANSFORMERS_LOAD_REPORT_VERSION = "5.12.0"
N_FILES = 35
N_TRAIN_FILES = 25
N_HELDOUT_FILES = 10
N_LAYERS = 16
HIDDEN_DIM = 2048
N_EXPERTS = 64
EXPERTS_PER_TOKEN = 8
MAX_CONTEXT = 4096
WINDOW_TOKENS = 512
PREFIX_TOKENS = 2048
BRINGUP_LONG_TOKENS = 2048
BRINGUP_SHORT_TOKENS = 512
N_KEY_WINDOWS = 16
N_PAIR_TRAIN = 48
N_PAIR_VALIDATION = 16
N_PAIR_WINDOWS = N_PAIR_TRAIN + N_PAIR_VALIDATION
N_BEHAVIORAL_WINDOWS = 16
FIT_INDICES = tuple(range(0, N_KEY_WINDOWS, 2))
EVAL_INDICES = tuple(range(1, N_KEY_WINDOWS, 2))
INNER_TRAIN_INDICES = (0, 4, 8, 12)
INNER_VALIDATION_INDICES = (2, 6, 10, 14)
KEY_CORPORA = ("narrative", "wikitext", "code", "grm")
NEGATIVE_CORPORA = ("wikitext", "code", "grm")

TAU_GRID = (
    ("p90", 0.90),
    ("p95", 0.95),
    ("p99", 0.99),
    ("p99.5", 0.995),
)
K4_ALPHA_GRID = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0)
RECALL_FLOOR = 0.50
FPR_CAPS = {"code": 0.05, "grm": 0.05}
G3_IMPROVEMENT_FLOOR = 0.10
G4_RECOVERY_FLOOR = 0.25
G5_WIKITEXT_PPL_DELTA_CAP_PERCENT = 0.5
G5_CODE_FIRE_CAP = 0.05
GENERIC_FIT_RANK_REFERENCE = 0.02
BRINGUP_PPL_MIN = 5.0
BRINGUP_PPL_MAX = 100.0
BRINGUP_LONG_MINUS_SHORT_NLL_CAP = 0.15
DEFAULT_SEED = 20260829
DEFAULT_BOOTSTRAP_RESAMPLES = 2000
EXPERTPACK_NAME = "expertpack_narrative_olmoe_v0"
PREPARED_CONTENT_ALGORITHM = (
    "moe_e2_prepared_arrays_v1:sha256(canonical-json(name,shape,dtype,sha256("
    "C-contiguous-little-endian-values)))"
)
PROVENANCE_VALIDATION_DIR = "provenance_validation"

PREPARED_ARRAY_SHAPES = {
    "narrative_key_ids": (16, 512),
    "pair_ids": (64, 512),
    "heldout_ids": (16, 512),
    "pair_p0_prefix_ids": (64, 2048),
    "heldout_p0_prefix_ids": (16, 2048),
    "heldout_p1_prefix_ids": (16, 2048),
    "wikitext_ids": (16, 512),
    "code_ids": (16, 512),
    "grm_ids": (16, 512),
    "bringup_ids": (2048,),
}

GPU_PREAMBLE = (
    "flock -w 7200 /tmp/forge-gpu.lock "
    "timeout --signal=TERM --kill-after=5s 590s "
    "env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
    "HF_DEACTIVATE_ASYNC_LOAD=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 "
    "PYTHONPYCACHEPREFIX=/tmp/olmoe_e2_pycache PYTHONUNBUFFERED=1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "prepare",
            "self-test",
            "validate-provenance",
            "load-check",
            "bringup",
            "capture-keys",
            "fit-key",
            "capture-pairs",
            "train",
            "eval-gates",
            "fallback-check",
            "analyze",
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", choices=("cpu", "auto"), default="cpu")
    parser.add_argument(
        "--bringup-cell", choices=("both", "short", "long"), default="both"
    )
    parser.add_argument("--max-gpu-memory", default="11GiB")
    parser.add_argument("--cpu-max-memory", default="46GiB")
    parser.add_argument("--cpu-threads", type=int, default=min(os.cpu_count() or 1, 12))
    parser.add_argument("--threads", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--corpus", choices=KEY_CORPORA)
    parser.add_argument("--start-window", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument(
        "--eval-kind", choices=("abi", "narrative", "wikitext", "code")
    )
    parser.add_argument("--prefix-arm", choices=("p0", "p1"), default="p0")
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
    parser.add_argument(
        "--legacy-script-snapshot",
        type=Path,
        help=(
            "validate-provenance only: exact historical harness used to rebuild "
            "a legacy prepared-content lineage"
        ),
    )
    parser.add_argument(
        "--legacy-archive-sha256",
        help=(
            "validate-provenance only: historical prepared_windows.npz byte SHA "
            "recorded by the immutable receipts"
        ),
    )
    parser.add_argument(
        "--legacy-source-commit",
        help="validate-provenance only: source commit label for the legacy snapshot",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_array(array: np.ndarray) -> np.ndarray:
    """Return content-hash bytes in C order with an explicit little-endian dtype."""

    observed = np.asarray(array)
    dtype = observed.dtype
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
        raise TypeError(f"unsupported array dtype for content provenance: {dtype}")
    normalized_dtype = dtype.newbyteorder("<")
    normalized = observed.astype(normalized_dtype, copy=False)
    return np.ascontiguousarray(normalized)


def sha256_array(array: np.ndarray) -> str:
    """SHA-256 of dtype-normalized logical values, independent of array layout."""

    return sha256_bytes(canonical_array(array).tobytes(order="C"))


def array_content_record(array: np.ndarray) -> dict[str, Any]:
    canonical = canonical_array(array)
    return {
        "shape": [int(value) for value in canonical.shape],
        "dtype": canonical.dtype.str,
        "sha256": sha256_bytes(canonical.tobytes(order="C")),
    }


def prepared_content_record(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    records = {
        name: array_content_record(arrays[name]) for name in sorted(arrays)
    }
    envelope = {
        "algorithm": PREPARED_CONTENT_ALGORITHM,
        "arrays": records,
    }
    canonical_json = json.dumps(
        envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return {
        "algorithm": PREPARED_CONTENT_ALGORITHM,
        "sha256": sha256_bytes(canonical_json),
        "arrays": records,
    }


def ensure_output(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to((REPO_ROOT / "artifacts" / "moe_e2").resolve())
    except ValueError as exc:
        if resolved != DEFAULT_OUTPUT_DIR.resolve():
            raise ValueError("E2 output must remain under artifacts/moe_e2") from exc
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def find_non_finite_json_values(
    value: Any, path: str = "$"
) -> list[tuple[str, str]]:
    """Return JSON paths and spellings for every nested non-finite float."""

    if isinstance(value, (float, np.floating)):
        number = float(value)
        return [] if math.isfinite(number) else [(path, repr(number))]
    if isinstance(value, dict):
        found: list[tuple[str, str]] = []
        for key, item in value.items():
            if isinstance(key, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                child_path = f"{path}.{key}"
            else:
                child_path = f"{path}[{json.dumps(str(key))}]"
            found.extend(find_non_finite_json_values(item, child_path))
        return found
    if isinstance(value, (list, tuple)):
        found = []
        for index, item in enumerate(value):
            found.extend(find_non_finite_json_values(item, f"{path}[{index}]"))
        return found
    return []


def require_finite_json_values(payload: Any) -> None:
    offenders = find_non_finite_json_values(payload)
    if offenders:
        named = ", ".join(f"{path}={spelling}" for path, spelling in offenders)
        raise ValueError(f"refusing to serialize non-finite JSON value(s): {named}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    require_finite_json_values(payload)
    atomic_write_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )


def write_json_if_changed(path: Path, payload: dict[str, Any]) -> bool:
    """Write canonical JSON only when its bytes differ; return whether it changed."""

    require_finite_json_values(payload)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if path.is_file() and path.read_bytes() == encoded:
        return False
    atomic_write_bytes(path, encoded)
    return True


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_npz(path: Path, *, compressed: bool = False, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        writer = np.savez_compressed if compressed else np.savez
        writer(handle, **{name: np.ascontiguousarray(value) for name, value in arrays.items()})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_prepared_array_contracts(arrays: dict[str, np.ndarray]) -> None:
    observed_names = set(arrays)
    expected_names = set(PREPARED_ARRAY_SHAPES)
    if observed_names != expected_names:
        raise RuntimeError(
            "prepared array names changed: "
            f"missing={sorted(expected_names - observed_names)}, "
            f"extra={sorted(observed_names - expected_names)}"
        )
    for name, expected_shape in PREPARED_ARRAY_SHAPES.items():
        value = arrays[name]
        if value.shape != expected_shape or value.dtype != np.int64:
            raise RuntimeError(
                f"prepared {name} contract {value.shape}/{value.dtype} "
                f"!= {expected_shape}/int64"
            )


def validate_declared_prepared_content(
    declared: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    require_combined: bool,
) -> dict[str, Any]:
    """Validate v1 or v2 manifest declarations using array content only."""

    validate_prepared_array_contracts(arrays)
    actual = prepared_content_record(arrays)
    declared_arrays = declared.get("arrays")
    if not isinstance(declared_arrays, dict):
        raise RuntimeError("prepared manifest has no per-array content records")
    if set(declared_arrays) != set(actual["arrays"]):
        raise RuntimeError("prepared manifest array-name set does not match archive")
    for name, observed in actual["arrays"].items():
        record = declared_arrays[name]
        if record.get("shape") != observed["shape"]:
            raise RuntimeError(f"prepared array shape changed: {name}")
        if require_combined and "dtype" not in record:
            raise RuntimeError(f"v2 prepared array dtype is missing: {name}")
        if "dtype" in record and record.get("dtype") != observed["dtype"]:
            raise RuntimeError(f"prepared array dtype changed: {name}")
        if record.get("sha256") != observed["sha256"]:
            raise RuntimeError(f"prepared array content hash mismatch: {name}")
    declared_digest = declared.get("content_sha256")
    declared_algorithm = declared.get("content_digest_algorithm")
    if (declared_digest is None) != (declared_algorithm is None):
        raise RuntimeError("prepared combined content provenance is incomplete")
    if require_combined and declared_digest is None:
        raise RuntimeError("v2 prepared manifest lacks combined content provenance")
    if declared_digest is not None and declared_digest != actual["sha256"]:
        raise RuntimeError("prepared combined content digest mismatch")
    if declared_algorithm is not None and declared_algorithm != actual["algorithm"]:
        raise RuntimeError("prepared content digest algorithm mismatch")
    return actual


def augment_manifest_content(
    manifest: dict[str, Any], content: dict[str, Any]
) -> dict[str, Any]:
    """Return an in-memory v2 view without rewriting a legacy manifest."""

    augmented = dict(manifest)
    prepared = dict(manifest["prepared_windows"])
    prepared["content_digest_algorithm"] = content["algorithm"]
    prepared["content_sha256"] = content["sha256"]
    prepared["arrays"] = content["arrays"]
    augmented["prepared_windows"] = prepared
    return augmented


def prepared_content_provenance(manifest: dict[str, Any]) -> dict[str, Any]:
    prepared = manifest["prepared_windows"]
    return {
        "path": prepared["path"],
        "content_digest_algorithm": prepared["content_digest_algorithm"],
        "content_sha256": prepared["content_sha256"],
        "arrays": prepared["arrays"],
    }


def prepared_content_fields(manifest: dict[str, Any]) -> dict[str, Any]:
    prepared = manifest["prepared_windows"]
    return {
        "prepared_windows_content_digest_algorithm": prepared[
            "content_digest_algorithm"
        ],
        "prepared_windows_content_sha256": prepared["content_sha256"],
    }


def content_fields_from_binding(binding: dict[str, Any]) -> dict[str, Any]:
    return {
        "prepared_windows_content_digest_algorithm": binding[
            "content_digest_algorithm"
        ],
        "prepared_windows_content_sha256": binding["content_sha256"],
    }


def content_provenance_from_binding(binding: dict[str, Any]) -> dict[str, Any]:
    return {
        "content_digest_algorithm": binding["content_digest_algorithm"],
        "content_sha256": binding["content_sha256"],
        "arrays": binding["arrays"],
        "content_label": binding["content_label"],
        "content_metadata_path": binding.get("content_metadata_path"),
    }


def source_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "byte_count": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def validate_model_dir(path: Path) -> Path:
    resolved = path.resolve()
    registered_root = Path(
        "/home/vader/.cache/huggingface/hub/models--allenai--OLMoE-1B-7B-0924"
    ).resolve()
    try:
        resolved.relative_to(registered_root)
    except ValueError as exc:
        raise ValueError(f"model path is outside the registered base cache: {resolved}") from exc
    if "instruct" in str(resolved).casefold():
        raise ValueError("the -Instruct model is forbidden; E2 requires the base release")
    required = (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    missing = [name for name in required if not (resolved / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete OLMoE snapshot {resolved}: {missing}")
    revision = resolved.name
    if revision != MODEL_REVISION:
        raise ValueError(f"snapshot revision {revision} != registered {MODEL_REVISION}")
    return resolved


def corpus_relative(path: Path) -> str:
    resolved = path.resolve()
    for root in CORPUS_ROOTS:
        try:
            relative = resolved.relative_to(root.resolve())
            return f"{root.name}/{relative.as_posix()}"
        except ValueError:
            continue
    raise ValueError(f"guide path escapes the three authorized roots: {resolved}")


def discover_guide_files() -> list[Path]:
    files: list[Path] = []
    for root in CORPUS_ROOTS:
        resolved_root = root.resolve()
        if any(
            term in component.casefold()
            for component in resolved_root.parts
            for term in FORBIDDEN_DIRECTORY_TERMS
        ):
            raise RuntimeError(f"forbidden directory component in guide root: {resolved_root}")
        if not resolved_root.is_dir():
            raise FileNotFoundError(resolved_root)
        for path in sorted(resolved_root.iterdir(), key=lambda item: item.name):
            if path.is_symlink():
                raise RuntimeError(f"guide symlink rejected: {path}")
            if path.is_dir():
                if any(
                    term in component.casefold()
                    for component in path.parts
                    for term in FORBIDDEN_DIRECTORY_TERMS
                ):
                    continue
                raise RuntimeError(f"unexpected guide subdirectory; not traversed: {path}")
            if path.is_file():
                files.append(path.resolve())
    if len(files) != N_FILES:
        raise RuntimeError(f"registered guide corpus requires 35 files, found {len(files)}")
    basenames = [path.name for path in files]
    if len(set(basenames)) != len(basenames):
        raise RuntimeError("sha256(filename) split is ambiguous: duplicate basenames")
    return files


def deterministic_file_split(files: Sequence[Path]) -> tuple[list[Path], list[Path]]:
    ranked = sorted(
        files,
        key=lambda path: (sha256_bytes(path.name.encode("utf-8")), path.name),
    )
    train = list(ranked[:N_TRAIN_FILES])
    heldout = list(ranked[N_TRAIN_FILES:])
    if len(train) != N_TRAIN_FILES or len(heldout) != N_HELDOUT_FILES:
        raise RuntimeError("registered 25/10 file split failed")
    return train, heldout


def tokenizer_ids(tokenizer: Any, text: str) -> np.ndarray:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )["input_ids"]
    return np.asarray(encoded, dtype=np.int64)


def tokenize_guide_files(files: Sequence[Path], tokenizer: Any) -> dict[Path, np.ndarray]:
    return {
        path: tokenizer_ids(tokenizer, path.read_text(encoding="utf-8", errors="strict"))
        for path in files
    }


def window_candidates(
    tokenized: dict[Path, np.ndarray], *, seed: int, label: str
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path, ids in tokenized.items():
        if ids.size < WINDOW_TOKENS:
            continue
        digest = hashlib.sha256(
            f"{seed}|{label}|{corpus_relative(path)}".encode("utf-8")
        ).digest()
        base = int.from_bytes(digest[:8], "big") % WINDOW_TOKENS
        if base + WINDOW_TOKENS > ids.size:
            base = 0
        for start in range(base, ids.size - WINDOW_TOKENS + 1, WINDOW_TOKENS):
            rank = sha256_bytes(
                f"{seed}|{label}|{corpus_relative(path)}|{start}".encode("utf-8")
            )
            candidates.append(
                {
                    "path": path,
                    "start": int(start),
                    "stop": int(start + WINDOW_TOKENS),
                    "rank": rank,
                }
            )
    return sorted(candidates, key=lambda item: (item["rank"], str(item["path"])))


def windows_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return bool(
        left["path"] == right["path"]
        and int(left["start"]) < int(right["stop"])
        and int(right["start"]) < int(left["stop"])
    )


def select_disjoint_windows(
    candidates: Sequence[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for candidate in candidates:
        if not any(windows_overlap(candidate, prior) for prior in chosen):
            chosen.append(dict(candidate))
            if len(chosen) == count:
                break
    if len(chosen) != count:
        raise RuntimeError(f"needed {count} disjoint guide windows, found {len(chosen)}")
    return chosen


def materialize_guide_windows(
    chosen: Sequence[dict[str, Any]], tokenized: dict[Path, np.ndarray]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    arrays: list[np.ndarray] = []
    provenance: list[dict[str, Any]] = []
    for index, item in enumerate(chosen):
        ids = np.ascontiguousarray(
            tokenized[item["path"]][int(item["start"]) : int(item["stop"])],
            dtype=np.int64,
        )
        if ids.shape != (WINDOW_TOKENS,):
            raise RuntimeError(f"guide window {index} shape {ids.shape}")
        arrays.append(ids)
        provenance.append(
            {
                "index": index,
                "source": corpus_relative(item["path"]),
                "source_path": str(item["path"]),
                "start_token": int(item["start"]),
                "stop_token_exclusive": int(item["stop"]),
                "token_ids_sha256": sha256_array(ids),
                "selection_rank_sha256": item["rank"],
            }
        )
    return np.stack(arrays), provenance


def build_raw_prefixes(
    targets: Sequence[dict[str, Any]],
    tokenized_train: dict[Path, np.ndarray],
    *,
    seed: int,
    label: str,
    avoid_same_source: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    eligible = sorted(
        [path for path, ids in tokenized_train.items() if ids.size >= PREFIX_TOKENS],
        key=lambda path: (sha256_bytes(path.name.encode("utf-8")), path.name),
    )
    if not eligible:
        raise RuntimeError("no TRAIN guide has a 2,048-token raw prefix excerpt")
    rotation_start = int.from_bytes(
        hashlib.sha256(f"{seed}|{label}|prefix-rotation".encode()).digest()[:8],
        "big",
    ) % len(eligible)
    arrays: list[np.ndarray] = []
    provenance: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        offset = 0
        while True:
            source = eligible[(rotation_start + index + offset) % len(eligible)]
            if not avoid_same_source or source != target["path"]:
                break
            offset += 1
            if offset >= len(eligible):
                raise RuntimeError("cannot choose a distinct TRAIN prefix source")
        ids = tokenized_train[source]
        span = int(ids.size - PREFIX_TOKENS + 1)
        start = int.from_bytes(
            hashlib.sha256(
                f"{seed}|{label}|{index}|{corpus_relative(source)}".encode()
            ).digest()[:8],
            "big",
        ) % span
        excerpt = np.ascontiguousarray(ids[start : start + PREFIX_TOKENS], dtype=np.int64)
        arrays.append(excerpt)
        provenance.append(
            {
                "index": index,
                "source": corpus_relative(source),
                "source_path": str(source),
                "start_token": start,
                "stop_token_exclusive": start + PREFIX_TOKENS,
                "token_ids_sha256": sha256_array(excerpt),
                "rotation_start": rotation_start,
                "rotation_offset": index + offset,
                "same_file_as_target": bool(source == target["path"]),
                "construction": "P0 raw 2,048-token TRAIN-file excerpt",
            }
        )
    return np.stack(arrays), provenance


STRUCTURAL_LINE = re.compile(r"^(?:#{1,6}\s|[-*+>]\s|\d+[.)]\s|```|\|)")
SENTENCE = re.compile(r"[^.!?]+[.!?]+", re.DOTALL)


def prose_only_text(text: str) -> tuple[str, dict[str, int]]:
    """Deterministic E2 P1 filter: prose sentences, no Markdown structures."""

    accepted_lines: list[str] = []
    rejected_structural = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if STRUCTURAL_LINE.match(line) or re.fullmatch(r"[:|\-\s]+", line):
            rejected_structural += 1
            continue
        if line.count("|") >= 2:
            rejected_structural += 1
            continue
        accepted_lines.append(line)
    joined = " ".join(accepted_lines)
    sentences: list[str] = []
    for match in SENTENCE.finditer(joined):
        sentence = re.sub(r"\s+", " ", match.group(0)).strip()
        words = re.findall(r"[A-Za-z][A-Za-z'’-]*", sentence)
        visible = sum(character.isalnum() for character in sentence)
        if len(words) < 6 or visible < max(1, len(sentence) // 3):
            continue
        sentences.append(sentence)
    return " ".join(sentences), {
        "input_lines": len(text.splitlines()),
        "accepted_nonempty_lines": len(accepted_lines),
        "rejected_structural_lines": rejected_structural,
        "accepted_sentences": len(sentences),
    }


def build_p1_prefixes(
    targets: Sequence[dict[str, Any]],
    train_files: Sequence[Path],
    tokenizer: Any,
    *,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    filtered: dict[Path, np.ndarray] = {}
    filter_receipts: list[dict[str, Any]] = []
    for path in train_files:
        prose, stats = prose_only_text(path.read_text(encoding="utf-8", errors="strict"))
        ids = tokenizer_ids(tokenizer, prose)
        filtered[path] = ids
        filter_receipts.append(
            {
                "source": corpus_relative(path),
                "source_path": str(path),
                "content_sha256": sha256_file(path),
                "filtered_text_sha256": sha256_bytes(prose.encode("utf-8")),
                "filtered_token_count": int(ids.size),
                **stats,
            }
        )
    eligible = sorted(
        [path for path, ids in filtered.items() if ids.size >= PREFIX_TOKENS],
        key=lambda path: (sha256_bytes(path.name.encode("utf-8")), path.name),
    )
    if not eligible:
        raise RuntimeError("P1 filter produced no TRAIN file with 2,048 prose tokens")
    rotation_start = int.from_bytes(
        hashlib.sha256(f"{seed}|e2-p1|prefix-rotation".encode()).digest()[:8],
        "big",
    ) % len(eligible)
    arrays: list[np.ndarray] = []
    provenance: list[dict[str, Any]] = []
    for index, _target in enumerate(targets):
        source = eligible[(rotation_start + index) % len(eligible)]
        ids = filtered[source]
        span = int(ids.size - PREFIX_TOKENS + 1)
        start = int.from_bytes(
            hashlib.sha256(
                f"{seed}|e2-p1|{index}|{corpus_relative(source)}".encode()
            ).digest()[:8],
            "big",
        ) % span
        excerpt = np.ascontiguousarray(ids[start : start + PREFIX_TOKENS], dtype=np.int64)
        arrays.append(excerpt)
        provenance.append(
            {
                "index": index,
                "source": corpus_relative(source),
                "source_path": str(source),
                "filtered_start_token": start,
                "filtered_stop_token_exclusive": start + PREFIX_TOKENS,
                "token_ids_sha256": sha256_array(excerpt),
                "rotation_start": rotation_start,
                "rotation_offset": index,
                "construction": (
                    "P1 sentences-only TRAIN-file excerpt; blank/header/table/list/"
                    "quote/fence lines removed; >=6 alphabetic-word terminal sentences"
                ),
            }
        )
    return np.stack(arrays), provenance, filter_receipts


def concatenate_plain_text(parts: Iterable[str]) -> str:
    return "\n\n".join(part for part in parts if part)


def build_wikitext_corpus() -> tuple[str, list[dict[str, Any]]]:
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


def build_file_corpus(
    paths: Sequence[Path], *, replacements: dict[Path, Path] | None = None
) -> tuple[str, list[dict[str, Any]]]:
    parts: list[str] = []
    sources: list[dict[str, Any]] = []
    normalized_replacements = {
        source.resolve(): replacement.resolve()
        for source, replacement in (replacements or {}).items()
    }
    for path in paths:
        content_path = normalized_replacements.get(path.resolve(), path)
        parts.append(content_path.read_text(encoding="utf-8", errors="strict"))
        record = source_record(content_path)
        record["path"] = str(path.resolve())
        record["repo_relative_path"] = path.relative_to(REPO_ROOT).as_posix()
        if content_path.resolve() != path.resolve():
            record["content_reconstructed_from"] = str(content_path.resolve())
        sources.append(record)
    return concatenate_plain_text(parts), sources


def build_code_corpus(
    *, script_snapshot: Path | None = None
) -> tuple[str, list[dict[str, Any]]]:
    paths: list[Path] = []
    for directory in (REPO_ROOT / "core", REPO_ROOT / "scripts", REPO_ROOT / "cpp"):
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".cpp", ".h"}:
                paths.append(path)
    paths.sort(key=lambda path: path.relative_to(REPO_ROOT).as_posix())
    replacements = None
    if script_snapshot is not None:
        snapshot = script_snapshot.resolve()
        if not snapshot.is_file():
            raise FileNotFoundError(snapshot)
        replacements = {SCRIPT_PATH: snapshot}
    return build_file_corpus(paths, replacements=replacements)


def build_grm_corpus() -> tuple[str, list[dict[str, Any]]]:
    # This intentionally preserves the RT2 "GRM-domain" recipe: all top-level
    # repository docs, with GRM_Primer.md required as the domain anchor.
    paths = sorted(
        (path for path in (REPO_ROOT / "docs").glob("*.md") if path.is_file()),
        key=lambda path: path.relative_to(REPO_ROOT).as_posix(),
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
            f"corpus has {token_count} tokens; need {n_windows * window_tokens}"
        )
    offsets = np.rint(np.linspace(0, token_count - window_tokens, n_windows)).astype(int)
    if any(int(right - left) < window_tokens for left, right in zip(offsets[:-1], offsets[1:])):
        offsets = np.arange(n_windows, dtype=int) * int(window_tokens)
    return [int(value) for value in offsets]


def file_record(path: Path, token_count: int, split: str) -> dict[str, Any]:
    return {
        "split": split,
        "filename": path.name,
        "filename_sha256": sha256_bytes(path.name.encode("utf-8")),
        "relative_path": corpus_relative(path),
        "path": str(path),
        "byte_count": int(path.stat().st_size),
        "content_sha256": sha256_file(path),
        "token_count": int(token_count),
    }


def load_tokenizer(model_dir: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, use_fast=True, trust_remote_code=False
    )


def prepare(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    output = ensure_output(args.output_dir)
    model_dir = validate_model_dir(args.model_dir)
    existing_manifest_path = output / "corpus_manifest.json"
    if existing_manifest_path.is_file():
        existing, _arrays = load_prepared(output)
        if existing.get("model_dir") != str(model_dir):
            raise RuntimeError("existing prepared corpus is bound to a different model path")
        print(
            json.dumps(
                {
                    "status": "existing",
                    "manifest": str(existing_manifest_path),
                    "windows": existing["prepared_windows"]["path"],
                    "content_sha256": existing["prepared_windows"][
                        "content_sha256"
                    ],
                    "npz_and_manifest_rewritten": False,
                }
            ),
            flush=True,
        )
        return 0
    tokenizer = load_tokenizer(model_dir)
    files = discover_guide_files()
    train_files, heldout_files = deterministic_file_split(files)
    tokenized = tokenize_guide_files(files, tokenizer)
    tokenized_train = {path: tokenized[path] for path in train_files}
    tokenized_heldout = {path: tokenized[path] for path in heldout_files}

    key_chosen = select_disjoint_windows(
        window_candidates(tokenized_train, seed=args.seed, label="e2-key"),
        N_KEY_WINDOWS,
    )
    pair_chosen = select_disjoint_windows(
        window_candidates(tokenized_train, seed=args.seed, label="e2-pairs"),
        N_PAIR_WINDOWS,
    )
    heldout_chosen = select_disjoint_windows(
        window_candidates(tokenized_heldout, seed=args.seed, label="e2-heldout"),
        N_BEHAVIORAL_WINDOWS,
    )
    narrative_key_ids, key_provenance = materialize_guide_windows(key_chosen, tokenized)
    pair_ids, pair_provenance = materialize_guide_windows(pair_chosen, tokenized)
    heldout_ids, heldout_provenance = materialize_guide_windows(heldout_chosen, tokenized)
    pair_p0_prefix_ids, pair_prefix_provenance = build_raw_prefixes(
        pair_chosen,
        tokenized_train,
        seed=args.seed,
        label="e2-pair-p0",
        avoid_same_source=True,
    )
    heldout_p0_prefix_ids, heldout_prefix_provenance = build_raw_prefixes(
        heldout_chosen,
        tokenized_train,
        seed=args.seed,
        label="e2-heldout-p0",
        avoid_same_source=False,
    )
    heldout_p1_prefix_ids, p1_prefix_provenance, p1_filters = build_p1_prefixes(
        heldout_chosen, train_files, tokenizer, seed=args.seed
    )

    corpus_builders = {
        "wikitext": build_wikitext_corpus,
        "code": build_code_corpus,
        "grm": build_grm_corpus,
    }
    negative_arrays: dict[str, np.ndarray] = {}
    negative_manifest: dict[str, Any] = {}
    wikitext_all_ids: np.ndarray | None = None
    for name, builder in corpus_builders.items():
        text, sources = builder()
        canonical = text.encode("utf-8")
        ids = tokenizer_ids(tokenizer, text)
        offsets = evenly_spaced_disjoint_offsets(ids.size, N_KEY_WINDOWS, WINDOW_TOKENS)
        windows = np.stack([ids[start : start + WINDOW_TOKENS] for start in offsets])
        negative_arrays[name] = np.ascontiguousarray(windows, dtype=np.int64)
        negative_manifest[name] = {
            "canonicalization": "plain source concatenation with two newlines between files",
            "corpus_byte_count": len(canonical),
            "corpus_sha256": sha256_bytes(canonical),
            "corpus_token_count": int(ids.size),
            "window_offsets": offsets,
            "window_ids_sha256": sha256_array(windows),
            "sources": sources,
        }
        if name == "wikitext":
            wikitext_all_ids = ids
    if wikitext_all_ids is None or wikitext_all_ids.size < BRINGUP_LONG_TOKENS:
        raise RuntimeError("WikiText corpus is too short for E2-G-1")
    bringup_span = int(wikitext_all_ids.size - BRINGUP_LONG_TOKENS + 1)
    bringup_offset = int.from_bytes(
        hashlib.sha256(f"{args.seed}|e2-bringup".encode()).digest()[:8], "big"
    ) % bringup_span
    bringup_ids = np.ascontiguousarray(
        wikitext_all_ids[bringup_offset : bringup_offset + BRINGUP_LONG_TOKENS],
        dtype=np.int64,
    )

    arrays = {
        "narrative_key_ids": narrative_key_ids,
        "pair_ids": pair_ids,
        "heldout_ids": heldout_ids,
        "pair_p0_prefix_ids": pair_p0_prefix_ids,
        "heldout_p0_prefix_ids": heldout_p0_prefix_ids,
        "heldout_p1_prefix_ids": heldout_p1_prefix_ids,
        "wikitext_ids": negative_arrays["wikitext"],
        "code_ids": negative_arrays["code"],
        "grm_ids": negative_arrays["grm"],
        "bringup_ids": bringup_ids,
    }
    validate_prepared_array_contracts(arrays)
    train_set = {str(path) for path in train_files}
    heldout_set = {str(path) for path in heldout_files}
    if train_set & heldout_set:
        raise RuntimeError("TRAIN/HELDOUT file leakage")
    for record in key_provenance + pair_provenance + pair_prefix_provenance + heldout_prefix_provenance + p1_prefix_provenance:
        if record["source_path"] not in train_set:
            raise RuntimeError(f"non-TRAIN source entered fit/prefix data: {record}")
    for record in heldout_provenance:
        if record["source_path"] not in heldout_set:
            raise RuntimeError(f"behavioral target is not HELDOUT: {record}")

    windows_path = output / "prepared_windows.npz"
    save_npz(windows_path, compressed=True, **arrays)
    content = prepared_content_record(arrays)
    manifest = {
        "schema": "moe_e2_corpus_manifest_v2",
        "status": "passed",
        "created_at": now_iso(),
        "order": str(ORDER_PATH),
        "order_sha256": sha256_file(ORDER_PATH),
        "glc_law": str(GLC_PATH),
        "glc_sha256": sha256_file(GLC_PATH),
        "script": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "model_id": MODEL_ID,
        "model_dir": str(model_dir),
        "model_revision": MODEL_REVISION,
        "config_sha256": sha256_file(model_dir / "config.json"),
        "tokenizer_class": type(tokenizer).__name__,
        "seed": args.seed,
        "authorized_roots": [str(path) for path in CORPUS_ROOTS],
        "scope_interpretation": (
            "only the three authorized directory roots were enumerated; forbidden terms "
            "are enforced on directory components, while filenames directly inside the "
            "registered roots remain in the 35-file corpus (the carried E1 rule)"
        ),
        "file_split": {
            "algorithm": (
                "sort ascending by sha256(UTF-8 basename), basename tie-break; "
                "first 25 TRAIN, final 10 HELDOUT"
            ),
            "train_count": len(train_files),
            "heldout_count": len(heldout_files),
            "train": [file_record(path, tokenized[path].size, "TRAIN") for path in train_files],
            "heldout": [file_record(path, tokenized[path].size, "HELDOUT") for path in heldout_files],
            "total_bytes": sum(path.stat().st_size for path in files),
        },
        "windows": {
            "narrative_key": key_provenance,
            "pair_train": [dict(item, split="TRAIN") for item in pair_provenance[:N_PAIR_TRAIN]],
            "pair_validation": [dict(item, split="VALIDATION") for item in pair_provenance[N_PAIR_TRAIN:]],
            "heldout_behavioral": heldout_provenance,
            "pair_p0_prefixes": pair_prefix_provenance,
            "heldout_p0_prefixes": heldout_prefix_provenance,
            "heldout_p1_prefixes": p1_prefix_provenance,
        },
        "p1_filter": {
            "registered_role": "single fallback arm only if P0 teacher gap <= 0",
            "recipe": (
                "strip blank lines and Markdown headers, tables, bullets, numbered lists, "
                "quotes, and fences; split terminal .!? sentences; retain sentences with "
                "at least six alphabetic words and at least one-third alphanumeric content"
            ),
            "per_train_file": p1_filters,
        },
        "negative_corpora": negative_manifest,
        "bringup": {
            "source": "same contiguous canonical WikiText-2 test+validation stream",
            "offset": bringup_offset,
            "long_tokens": BRINGUP_LONG_TOKENS,
            "short_tokens": BRINGUP_SHORT_TOKENS,
            "short_is_exact_long_prefix": bool(
                np.array_equal(
                    bringup_ids[:BRINGUP_SHORT_TOKENS],
                    wikitext_all_ids[
                        bringup_offset : bringup_offset + BRINGUP_SHORT_TOKENS
                    ],
                )
            ),
            "ids_sha256": sha256_array(bringup_ids),
        },
        "split_contract": {
            "key_fit_window_indices": list(FIT_INDICES),
            "key_eval_window_indices": list(EVAL_INDICES),
            "key_inner_train_window_indices": list(INNER_TRAIN_INDICES),
            "key_inner_validation_window_indices": list(INNER_VALIDATION_INDICES),
            "fit_eval_overlap": [],
            "heldout_text_in_key_pair_or_prefix": False,
            "pair_train_count": N_PAIR_TRAIN,
            "pair_validation_count": N_PAIR_VALIDATION,
            "behavioral_window_count": N_BEHAVIORAL_WINDOWS,
        },
        "prepared_windows": {
            "path": str(windows_path),
            "content_digest_algorithm": content["algorithm"],
            "content_sha256": content["sha256"],
            "arrays": content["arrays"],
        },
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(output / "corpus_manifest.json", manifest)
    initialize_pending_expertpack(output, manifest)
    write_gpu_scripts(output)
    print(
        json.dumps(
            {
                "status": "passed",
                "train_files": len(train_files),
                "heldout_files": len(heldout_files),
                "windows": str(windows_path),
                "manifest": str(output / "corpus_manifest.json"),
            }
        ),
        flush=True,
    )
    return 0


def load_prepared(output: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = output / "corpus_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run prepare first: {manifest_path}")
    manifest = read_json(manifest_path)
    manifest_schema = manifest.get("schema")
    if manifest_schema not in {
        "moe_e2_corpus_manifest_v1",
        "moe_e2_corpus_manifest_v2",
    }:
        raise RuntimeError("prepared corpus manifest schema is unsupported")
    if manifest.get("status") != "passed":
        raise RuntimeError("prepared corpus manifest is not passed")
    if manifest.get("model_revision") != MODEL_REVISION:
        raise RuntimeError("prepared corpus model revision changed")
    if int(manifest.get("seed", -1)) != DEFAULT_SEED:
        raise RuntimeError("prepared corpus seed changed")
    windows_path = Path(manifest["prepared_windows"]["path"])
    if windows_path.resolve() != (output / "prepared_windows.npz").resolve():
        raise RuntimeError("prepared window archive path escapes its output directory")
    if not windows_path.is_file():
        raise FileNotFoundError(windows_path)
    with np.load(windows_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    content = validate_declared_prepared_content(
        manifest["prepared_windows"],
        arrays,
        require_combined=manifest_schema == "moe_e2_corpus_manifest_v2",
    )
    return augment_manifest_content(manifest, content), arrays


def provenance_validation_root(output: Path) -> Path:
    return output / PROVENANCE_VALIDATION_DIR


def content_set_from_arrays(
    *,
    label: str,
    arrays: dict[str, np.ndarray],
    content: dict[str, Any],
    legacy_archive_sha256: str | None,
    metadata_path: Path | None,
) -> dict[str, Any]:
    return {
        "label": label,
        "arrays": arrays,
        "content": content,
        "legacy_archive_sha256": legacy_archive_sha256,
        "metadata_path": None if metadata_path is None else str(metadata_path),
    }


def validate_content_set_registry(content_sets: Sequence[dict[str, Any]]) -> None:
    for field in ("label",):
        values = [item[field] for item in content_sets]
        if len(set(values)) != len(values):
            raise RuntimeError(f"duplicate prepared-content registry {field}")
    digests = [item["content"]["sha256"] for item in content_sets]
    if len(set(digests)) != len(digests):
        raise RuntimeError("duplicate prepared-content lineages are registered")
    legacy_hashes = [
        item["legacy_archive_sha256"]
        for item in content_sets
        if item.get("legacy_archive_sha256") is not None
    ]
    if len(set(legacy_hashes)) != len(legacy_hashes):
        raise RuntimeError("one legacy archive SHA maps to multiple content lineages")


def load_prepared_content_sets(
    output: Path, manifest: dict[str, Any], prepared: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    current_record = manifest["prepared_windows"]
    sets = [
        content_set_from_arrays(
            label="current_manifest",
            arrays=prepared,
            content={
                "algorithm": current_record["content_digest_algorithm"],
                "sha256": current_record["content_sha256"],
                "arrays": current_record["arrays"],
            },
            legacy_archive_sha256=current_record.get("sha256"),
            metadata_path=output / "corpus_manifest.json",
        )
    ]
    root = provenance_validation_root(output)
    if not root.is_dir():
        return sets
    for metadata_path in sorted(root.glob("legacy_prepared_content_*.json")):
        metadata = read_json(metadata_path)
        if (
            metadata.get("schema") != "moe_e2_legacy_prepared_content_v1"
            or metadata.get("status") != "passed"
            or metadata.get("model_revision") != MODEL_REVISION
            or int(metadata.get("seed", -1)) != DEFAULT_SEED
        ):
            raise RuntimeError(f"invalid legacy prepared-content metadata: {metadata_path}")
        archive_path = Path(metadata["reconstructed_archive"]["path"])
        try:
            archive_path.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise RuntimeError(
                f"legacy prepared-content archive escapes validation root: {archive_path}"
            ) from exc
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        with np.load(archive_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        content = prepared_content_record(arrays)
        validate_prepared_array_contracts(arrays)
        if content != metadata.get("content"):
            raise RuntimeError(
                f"legacy prepared-content archive no longer matches metadata: {metadata_path}"
            )
        sets.append(
            content_set_from_arrays(
                label=str(metadata["label"]),
                arrays=arrays,
                content=content,
                legacy_archive_sha256=metadata.get("legacy_archive_sha256"),
                metadata_path=metadata_path,
            )
        )
    validate_content_set_registry(sets)
    return sets


def reconstruct_legacy_prepared_content(
    args: argparse.Namespace,
    output: Path,
    manifest: dict[str, Any],
    prepared: dict[str, np.ndarray],
) -> Path | None:
    snapshot = args.legacy_script_snapshot
    legacy_archive_sha256 = args.legacy_archive_sha256
    if snapshot is None and legacy_archive_sha256 is None:
        return None
    if snapshot is None or legacy_archive_sha256 is None:
        raise ValueError(
            "--legacy-script-snapshot and --legacy-archive-sha256 must be supplied together"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", legacy_archive_sha256):
        raise ValueError("--legacy-archive-sha256 must be a lowercase SHA-256 hex digest")
    snapshot = snapshot.resolve()
    if not snapshot.is_file():
        raise FileNotFoundError(snapshot)

    tokenizer = load_tokenizer(validate_model_dir(args.model_dir))
    code_text, code_sources = build_code_corpus(script_snapshot=snapshot)
    canonical = code_text.encode("utf-8")
    code_ids_all = tokenizer_ids(tokenizer, code_text)
    offsets = evenly_spaced_disjoint_offsets(
        code_ids_all.size, N_KEY_WINDOWS, WINDOW_TOKENS
    )
    code_ids = np.ascontiguousarray(
        np.stack(
            [code_ids_all[start : start + WINDOW_TOKENS] for start in offsets]
        ),
        dtype=np.int64,
    )
    legacy_arrays = {name: value.copy() for name, value in prepared.items()}
    legacy_arrays["code_ids"] = code_ids
    validate_prepared_array_contracts(legacy_arrays)
    content = prepared_content_record(legacy_arrays)
    if content["sha256"] == manifest["prepared_windows"]["content_sha256"]:
        raise RuntimeError("legacy reconstruction did not produce a distinct content lineage")

    root = provenance_validation_root(output)
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / f"legacy_prepared_windows_{content['sha256']}.npz"
    snapshot_sha256 = sha256_file(snapshot)
    snapshot_copy = root / f"legacy_harness_{snapshot_sha256}.py.txt"
    metadata_path = root / f"legacy_prepared_content_{content['sha256']}.json"
    existing = read_json(metadata_path) if metadata_path.is_file() else {}
    if existing:
        existing_source = existing.get("source_reconstruction", {})
        if (
            existing.get("legacy_archive_sha256") != legacy_archive_sha256
            or existing.get("legacy_source_commit") != args.legacy_source_commit
            or existing.get("content") != content
            or existing_source.get("historical_harness_sha256")
            != snapshot_sha256
        ):
            raise RuntimeError(
                "existing legacy content metadata conflicts with this registration: "
                f"{metadata_path}"
            )
    else:
        proposed = content_set_from_arrays(
            label=f"legacy_archive_{legacy_archive_sha256[:16]}",
            arrays=legacy_arrays,
            content=content,
            legacy_archive_sha256=legacy_archive_sha256,
            metadata_path=metadata_path,
        )
        validate_content_set_registry(
            load_prepared_content_sets(output, manifest, prepared) + [proposed]
        )

    if archive_path.is_file():
        with np.load(archive_path, allow_pickle=False) as archive:
            existing_arrays = {name: archive[name].copy() for name in archive.files}
        if prepared_content_record(existing_arrays) != content:
            raise RuntimeError(f"existing legacy content archive conflicts: {archive_path}")
    else:
        save_npz(archive_path, compressed=True, **legacy_arrays)

    snapshot_bytes = snapshot.read_bytes()
    if snapshot_copy.is_file() and snapshot_copy.read_bytes() != snapshot_bytes:
        raise RuntimeError(f"historical harness sidecar changed: {snapshot_copy}")
    if not snapshot_copy.is_file():
        atomic_write_bytes(snapshot_copy, snapshot_bytes)

    metadata = {
        "schema": "moe_e2_legacy_prepared_content_v1",
        "status": "passed",
        "created_at": existing.get("created_at", now_iso()),
        "label": f"legacy_archive_{legacy_archive_sha256[:16]}",
        "order": str(F4_ORDER_PATH),
        "order_sha256": sha256_file(F4_ORDER_PATH),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": DEFAULT_SEED,
        "legacy_archive_sha256": legacy_archive_sha256,
        "legacy_source_commit": args.legacy_source_commit,
        "source_reconstruction": {
            "method": (
                "current internally-sealed prepared arrays with code_ids rebuilt from "
                "the exact historical harness substituted at its original corpus path"
            ),
            "historical_harness_copy": str(snapshot_copy),
            "historical_harness_sha256": sha256_file(snapshot_copy),
            "code_corpus_source_count": len(code_sources),
            "code_corpus_byte_count": len(canonical),
            "code_corpus_sha256": sha256_bytes(canonical),
            "code_corpus_token_count": int(code_ids_all.size),
            "code_window_offsets": offsets,
            "code_ids_content_sha256": sha256_array(code_ids),
            "current_code_ids_content_sha256": sha256_array(prepared["code_ids"]),
            "only_code_ids_replaced": True,
        },
        "content": content,
        "reconstructed_archive": existing.get(
            "reconstructed_archive",
            {
                "path": str(archive_path),
                "sha256_at_registration": sha256_file(archive_path),
                "byte_count_at_registration": int(archive_path.stat().st_size),
                "byte_identical_to_recorded_legacy_archive_at_registration": (
                    sha256_file(archive_path) == legacy_archive_sha256
                ),
                "archive_bytes_are_diagnostic_not_provenance": True,
            },
        ),
        "original_receipts_mutated": False,
    }
    if existing and existing != metadata:
        raise RuntimeError(
            "existing legacy content metadata conflicts with this registration: "
            f"{metadata_path}"
        )
    write_json_if_changed(metadata_path, metadata)
    return metadata_path


def bind_receipt_to_prepared_content(
    receipt: dict[str, Any],
    content_sets: Sequence[dict[str, Any]],
    checks_for_arrays: Callable[[dict[str, np.ndarray]], list[dict[str, Any]]],
    *,
    allow_unregistered_legacy: bool = False,
    require_declared_content: bool = False,
) -> dict[str, Any]:
    """Bind a receipt to one content lineage without trusting NPZ container bytes."""

    declared_content = receipt.get("prepared_windows_content_sha256")
    declared_algorithm = receipt.get("prepared_windows_content_digest_algorithm")
    legacy_archive = receipt.get("prepared_windows_sha256")
    if (declared_content is None) != (declared_algorithm is None):
        raise RuntimeError("receipt prepared-content provenance is incomplete")
    if require_declared_content and declared_content is None:
        raise RuntimeError("modern receipt lacks combined content provenance")
    if (
        declared_algorithm is not None
        and declared_algorithm != PREPARED_CONTENT_ALGORITHM
    ):
        raise RuntimeError("receipt prepared-content algorithm is unknown")

    candidates: list[tuple[int, dict[str, Any], list[dict[str, Any]], str]] = []
    failures: list[dict[str, Any]] = []
    for content_set in content_sets:
        checks = checks_for_arrays(content_set["arrays"])
        arrays_match = bool(checks) and all(item.get("passed") for item in checks)
        failures.append(
            {
                "content_label": content_set["label"],
                "content_sha256": content_set["content"]["sha256"],
                "array_checks": checks,
            }
        )
        if not arrays_match:
            continue
        if declared_content is not None:
            if content_set["content"]["sha256"] != declared_content:
                continue
            candidates.append((40, content_set, checks, "native_content_digest"))
            continue
        if legacy_archive is not None:
            if content_set.get("legacy_archive_sha256") == legacy_archive:
                candidates.append(
                    (30, content_set, checks, "registered_legacy_archive_to_content")
                )
            elif allow_unregistered_legacy:
                score = 11 if content_set["label"] == "current_manifest" else 10
                candidates.append(
                    (score, content_set, checks, "recorded_arrays_only_migration")
                )
            continue
        score = 21 if content_set["label"] == "current_manifest" else 20
        candidates.append((score, content_set, checks, "recorded_arrays_only_migration"))

    if not candidates:
        if declared_content is not None:
            reason = "declared prepared-content digest did not validate"
        elif legacy_archive is not None and not allow_unregistered_legacy:
            reason = "legacy archive SHA has no registered content lineage"
        else:
            reason = "recorded prepared-array hashes did not validate"
        raise RuntimeError(f"{reason}: {failures}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    _score, selected, checks, mode = candidates[0]
    return {
        "status": "passed",
        "mode": mode,
        "content_label": selected["label"],
        "content_digest_algorithm": selected["content"]["algorithm"],
        "content_sha256": selected["content"]["sha256"],
        "arrays": selected["content"]["arrays"],
        "content_metadata_path": selected["metadata_path"],
        "legacy_archive_sha256_recorded": legacy_archive,
        "array_checks": checks,
        "original_receipt_mutated": False,
    }


def receipt_array_check(
    *,
    receipt_field: str,
    recorded_sha256: Any,
    array_name: str,
    selection: str,
    value: np.ndarray,
) -> dict[str, Any]:
    observed = array_content_record(np.asarray(value))
    return {
        "receipt_field": receipt_field,
        "array_name": array_name,
        "selection": selection,
        "recorded_sha256": recorded_sha256,
        "observed": observed,
        "passed": recorded_sha256 == observed["sha256"],
    }


def bind_transitive_receipt_content(
    receipt: dict[str, Any],
    dependency_bindings: Sequence[dict[str, Any]],
    *,
    dependency_kind: str,
    require_declared_content: bool = False,
) -> dict[str, Any]:
    if not dependency_bindings:
        raise RuntimeError(f"no {dependency_kind} bindings for transitive provenance")
    content_digests = {item["content_sha256"] for item in dependency_bindings}
    algorithms = {item["content_digest_algorithm"] for item in dependency_bindings}
    labels = {item["content_label"] for item in dependency_bindings}
    if len(content_digests) != 1 or len(algorithms) != 1 or len(labels) != 1:
        raise RuntimeError(
            f"{dependency_kind} dependencies span multiple prepared-content lineages"
        )
    content_sha256 = next(iter(content_digests))
    algorithm = next(iter(algorithms))
    arrays = dependency_bindings[0].get("arrays")
    if arrays is None or any(item.get("arrays") != arrays for item in dependency_bindings):
        raise RuntimeError(f"{dependency_kind} dependency array records changed")
    declared = receipt.get("prepared_windows_content_sha256")
    declared_algorithm = receipt.get("prepared_windows_content_digest_algorithm")
    if (declared is None) != (declared_algorithm is None):
        raise RuntimeError("transitive receipt prepared-content provenance is incomplete")
    if require_declared_content and declared is None:
        raise RuntimeError("modern transitive receipt lacks combined content provenance")
    if declared is not None and declared != content_sha256:
        raise RuntimeError("transitive receipt prepared-content digest mismatch")
    if declared_algorithm is not None and declared_algorithm != algorithm:
        raise RuntimeError("transitive receipt prepared-content algorithm mismatch")
    return {
        "status": "passed",
        "mode": (
            "native_content_digest_transitively_revalidated"
            if declared is not None
            else f"legacy_transitive_{dependency_kind}_migration"
        ),
        "content_label": next(iter(labels)),
        "content_digest_algorithm": algorithm,
        "content_sha256": content_sha256,
        "arrays": arrays,
        "content_metadata_path": dependency_bindings[0].get(
            "content_metadata_path"
        ),
        "dependency_kind": dependency_kind,
        "dependency_count": len(dependency_bindings),
        "original_receipt_mutated": False,
    }


def initialize_pending_expertpack(output: Path, manifest: dict[str, Any]) -> None:
    pack_dir = output / EXPERTPACK_NAME
    pack_dir.mkdir(parents=True, exist_ok=True)
    path = pack_dir / "manifest.json"
    existing = read_json(path) if path.is_file() else {}
    pack = {
        "schema": "expertpack_narrative_olmoe_v0",
        "status": existing.get("status", "pending_e2_g2_address"),
        "created_at": existing.get("created_at", now_iso()),
        "updated_at": now_iso(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "architecture": "OLMoE base, 16 layers, 64 native experts, top-8",
        "abi": "out += B @ silu(A @ h) iff h @ key >= tau at layer L* MLP side path",
        "layer": existing.get("layer"),
        "rank": 64,
        "tau": existing.get("tau"),
        "key_metrics": existing.get("key_metrics"),
        "components": existing.get(
            "components",
            {
                "key": {"status": "missing", "path": str(pack_dir / "key_fp32.npy")},
                "A": {"status": "missing", "path": str(pack_dir / "A_fp16.npy")},
                "B": {
                    "status": "missing",
                    "path": str(pack_dir / "B_fp16.npy"),
                    "initialization": "exact zeros before training",
                },
            },
        ),
        "provenance": {
            **existing.get("provenance", {}),
            "order": str(ORDER_PATH),
            "order_sha256": sha256_file(ORDER_PATH),
            "glc_law": str(GLC_PATH),
            "glc_sha256": sha256_file(GLC_PATH),
            "corpus_manifest": str(output / "corpus_manifest.json"),
            "corpus_manifest_sha256": sha256_file(output / "corpus_manifest.json"),
            "prepared_windows_content": prepared_content_provenance(manifest),
            "script": str(SCRIPT_PATH),
            "script_sha256": sha256_file(SCRIPT_PATH),
        },
        "limitations": existing.get(
            "limitations", ["E2-G2/G3/G4/G5 not yet measured"]
        ),
    }
    for key in ("g3", "behavioral"):
        if key in existing:
            pack[key] = existing[key]
    pack["provenance"].pop("prepared_windows", None)
    write_json(path, pack)


def require_bringup_green(
    output: Path,
    manifest: dict[str, Any] | None = None,
    prepared: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    path = output / "bringup.json"
    if not path.is_file():
        raise RuntimeError(f"E2-G-1 is NOT_MEASURED; run bringup first: {path}")
    receipt = read_json(path)
    if receipt.get("gate", {}).get("verdict") != "GREEN":
        raise RuntimeError(
            f"E2-G-1 is {receipt.get('gate', {}).get('verdict', 'NOT_MEASURED')}; STOP"
        )
    if manifest is None or prepared is None:
        manifest, prepared = load_prepared(output)
    validate_consolidated_bringup_receipt(
        output, manifest, prepared, receipt=receipt
    )
    return receipt


def configure_torch(cpu_threads: int, *, cuda_possible: bool) -> Any:
    import torch

    if cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(DEFAULT_SEED)
    torch.use_deterministic_algorithms(True, warn_only=cuda_possible)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return torch


LOAD_REPORT_FIELDS = (
    "missing_keys",
    "unexpected_keys",
    "mismatched_keys",
    "error_msgs",
)


def canonical_loading_info(loading_info: Any) -> dict[str, list[Any]]:
    if not isinstance(loading_info, dict):
        raise RuntimeError(
            "checkpoint load-report gate failed: Transformers did not return a report dictionary"
        )
    missing_fields = sorted(set(LOAD_REPORT_FIELDS) - set(loading_info))
    if missing_fields:
        raise RuntimeError(
            "checkpoint load-report gate failed: missing report fields "
            f"{missing_fields}"
        )
    canonical: dict[str, list[Any]] = {}
    for field in LOAD_REPORT_FIELDS:
        values = list(loading_info[field])
        if field == "mismatched_keys":
            canonical[field] = sorted(
                (
                    {
                        "key": str(item[0]),
                        "checkpoint_shape": [int(value) for value in item[1]],
                        "model_shape": [int(value) for value in item[2]],
                    }
                    for item in values
                ),
                key=lambda item: item["key"],
            )
        else:
            canonical[field] = sorted(str(value) for value in values)
    return canonical


def checkpoint_tensor_manifest(model_dir: Path) -> dict[str, Any]:
    from safetensors import safe_open

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError(f"checkpoint load-report gate failed: missing {index_path}")
    index = read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(
            "checkpoint load-report gate failed: safetensors index has no weight_map"
        )
    expected_by_shard: dict[str, set[str]] = {}
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not isinstance(shard_name, str):
            raise RuntimeError(
                "checkpoint load-report gate failed: malformed safetensors weight_map"
            )
        expected_by_shard.setdefault(shard_name, set()).add(tensor_name)
    actual_keys: set[str] = set()
    shard_records: list[dict[str, Any]] = []
    for shard_name in sorted(expected_by_shard):
        shard_path = model_dir / shard_name
        if not shard_path.is_file():
            raise RuntimeError(
                f"checkpoint load-report gate failed: missing shard {shard_path}"
            )
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            shard_keys = set(handle.keys())
        expected_keys = expected_by_shard[shard_name]
        if shard_keys != expected_keys:
            raise RuntimeError(
                "checkpoint load-report gate failed: index/shard key mismatch for "
                f"{shard_name}; absent={sorted(expected_keys - shard_keys)}, "
                f"unindexed={sorted(shard_keys - expected_keys)}"
            )
        if actual_keys & shard_keys:
            raise RuntimeError(
                "checkpoint load-report gate failed: duplicate tensors across shards"
            )
        actual_keys.update(shard_keys)
        shard_records.append(
            {
                "filename": shard_name,
                "tensor_count": len(shard_keys),
                "byte_count": int(shard_path.stat().st_size),
            }
        )
    total_size = index.get("metadata", {}).get("total_size")
    if not isinstance(total_size, int) or total_size <= 0:
        raise RuntimeError(
            "checkpoint load-report gate failed: missing safetensors metadata.total_size"
        )
    return {
        "index": str(index_path),
        "index_sha256": sha256_file(index_path),
        "tensor_count": len(actual_keys),
        "total_size_bytes": total_size,
        "shards": shard_records,
    }


def assert_checkpoint_load_complete(
    model: Any, loading_info: Any, model_dir: Path
) -> dict[str, Any]:
    report = canonical_loading_info(loading_info)
    failures = {field: values for field, values in report.items() if values}
    meta_parameters = sorted(
        name for name, value in model.named_parameters() if value.device.type == "meta"
    )
    meta_buffers = sorted(
        name for name, value in model.named_buffers() if value.device.type == "meta"
    )
    ignored_missing = list(getattr(model, "_keys_to_ignore_on_load_missing", None) or [])
    ignored_unexpected = list(
        getattr(model, "_keys_to_ignore_on_load_unexpected", None) or []
    )
    if failures or meta_parameters or meta_buffers or ignored_missing or ignored_unexpected:
        raise RuntimeError(
            "checkpoint load-report gate failed before forward: "
            f"report={failures}, meta_parameters={meta_parameters}, "
            f"meta_buffers={meta_buffers}, ignored_missing={ignored_missing}, "
            f"ignored_unexpected={ignored_unexpected}"
        )
    checkpoint = checkpoint_tensor_manifest(model_dir)
    parameter_count = int(sum(value.numel() for value in model.parameters()))
    parameter_bytes = int(
        sum(value.numel() * value.element_size() for value in model.parameters())
    )
    if parameter_bytes != checkpoint["total_size_bytes"]:
        raise RuntimeError(
            "checkpoint load-report gate failed before forward: converted model byte "
            f"ledger {parameter_bytes} != checkpoint {checkpoint['total_size_bytes']}"
        )
    return {
        "status": "passed",
        "report": report,
        "missing_count": 0,
        "unexpected_count": 0,
        "mismatched_count": 0,
        "error_count": 0,
        "conversion_count": 0,
        "conversion_semantics": (
            f"Transformers {TRANSFORMERS_LOAD_REPORT_VERSION} raises on any "
            "CONVERSION entry before "
            "output_loading_info can return"
        ),
        "meta_parameter_count": 0,
        "meta_buffer_count": 0,
        "ignored_missing_patterns": [],
        "ignored_unexpected_patterns": [],
        "checkpoint": checkpoint,
        "loaded_parameter_count": parameter_count,
        "loaded_parameter_bytes": parameter_bytes,
        "model_state_tensor_count": len(model.state_dict()),
        "every_checkpoint_tensor_consumed": True,
        "every_model_tensor_present": True,
    }


def device_label(device: Any) -> str:
    if device == 0:
        return "cuda:0"
    return str(device)


def dispatch_loaded_model(
    torch: Any, model: Any, args: argparse.Namespace
) -> tuple[Any, dict[str, Any]]:
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.utils import compute_module_sizes, get_balanced_memory

    requested_max_memory = {0: args.max_gpu_memory, "cpu": args.cpu_max_memory}
    no_split_modules = sorted(getattr(model, "_no_split_modules", None) or [])
    balanced_max_memory = get_balanced_memory(
        model,
        max_memory=requested_max_memory,
        no_split_module_classes=no_split_modules,
        dtype=torch.bfloat16,
        low_zero=False,
    )
    device_map = infer_auto_device_map(
        model,
        max_memory=balanced_max_memory,
        no_split_module_classes=no_split_modules,
        dtype=torch.bfloat16,
        clean_result=True,
        offload_buffers=False,
        fallback_allocation=False,
    )
    mapped_devices = {device_label(device) for device in device_map.values()}
    if "disk" in mapped_devices:
        raise RuntimeError(
            "post-load dispatch refused disk placement; raise --cpu-max-memory instead"
        )
    if "cuda:0" not in mapped_devices:
        raise RuntimeError("post-load dispatch produced no CUDA placement")
    if "cpu" not in mapped_devices:
        raise RuntimeError(
            "post-load dispatch unexpectedly mapped the full 13.8 GB model to CUDA"
        )
    module_sizes = compute_module_sizes(model, dtype=torch.bfloat16)
    planned_bytes: dict[str, int] = {}
    planned_root_sizes: dict[str, list[int]] = {}
    for module_name, device in device_map.items():
        label = device_label(device)
        module_bytes = int(module_sizes[module_name])
        planned_bytes[label] = planned_bytes.get(label, 0) + module_bytes
        planned_root_sizes.setdefault(label, []).append(module_bytes)
    cuda_budget_bytes = int(balanced_max_memory[0])
    if planned_bytes.get("cuda:0", 0) > cuda_budget_bytes:
        raise RuntimeError(
            "post-load dispatch plan exceeds CUDA budget: "
            f"{planned_bytes['cuda:0']} > {cuda_budget_bytes}"
        )
    largest_cpu_module_bytes = max(planned_root_sizes.get("cpu", [0]))
    swap_peak_bytes = (
        planned_bytes.get("cuda:0", 0) + largest_cpu_module_bytes
    )
    if swap_peak_bytes > cuda_budget_bytes:
        raise RuntimeError(
            "post-load dispatch plan lacks room to swap its largest CPU module: "
            f"{swap_peak_bytes} > {cuda_budget_bytes}"
        )
    torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    model = dispatch_model(
        model,
        device_map=device_map,
        main_device=0,
        offload_buffers=False,
        skip_keys=getattr(model, "_skip_keys_device_placement", None),
    )
    allocated_bytes = int(torch.cuda.memory_allocated(0))
    peak_allocated_bytes = int(torch.cuda.max_memory_allocated(0))
    if max(allocated_bytes, peak_allocated_bytes) > cuda_budget_bytes:
        raise RuntimeError(
            "post-load dispatch CUDA allocation exceeds budget: "
            f"resident={allocated_bytes}, peak={peak_allocated_bytes}, "
            f"budget={cuda_budget_bytes}"
        )
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    return model, {
        "strategy": "full_cpu_bf16_load_then_balanced_dispatch",
        "dispatch_seconds": time.perf_counter() - started,
        "requested_max_memory": {
            "cuda:0": args.max_gpu_memory,
            "cpu": args.cpu_max_memory,
        },
        "resolved_max_memory_bytes": {
            device_label(device): int(value)
            for device, value in balanced_max_memory.items()
        },
        "no_split_module_classes": no_split_modules,
        "device_map": {
            name: device_label(device) for name, device in device_map.items()
        },
        "planned_module_bytes_by_device": planned_bytes,
        "largest_cpu_module_bytes": largest_cpu_module_bytes,
        "planned_cuda_plus_largest_cpu_module_bytes": swap_peak_bytes,
        "planned_swap_headroom_bytes": cuda_budget_bytes - swap_peak_bytes,
        "cuda_allocated_bytes_after_dispatch": allocated_bytes,
        "cuda_peak_allocated_bytes_during_dispatch": peak_allocated_bytes,
        "cuda_free_bytes_after_dispatch": int(free_bytes),
        "cuda_total_bytes": int(total_bytes),
        "disk_offload": False,
    }


def load_model(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    import transformers
    from transformers import AutoModelForCausalLM

    model_dir = validate_model_dir(args.model_dir)
    if transformers.__version__ != TRANSFORMERS_LOAD_REPORT_VERSION:
        raise RuntimeError(
            "checkpoint load-report gate requires Transformers "
            f"{TRANSFORMERS_LOAD_REPORT_VERSION}; observed {transformers.__version__}"
        )
    cuda_possible = args.device == "auto"
    torch = configure_torch(args.cpu_threads, cuda_possible=cuda_possible)
    if args.device == "auto" and not torch.cuda.is_available():
        raise RuntimeError("--device auto requested but torch.cuda.is_available() is false")
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "output_loading_info": True,
    }
    started = time.perf_counter()
    cpu_load_started = time.perf_counter()
    try:
        model, loading_info = AutoModelForCausalLM.from_pretrained(
            str(model_dir), **kwargs
        )
    except Exception as exc:
        raise RuntimeError(
            "checkpoint load-report gate failed before forward; CPU fused-expert "
            f"conversion did not return cleanly ({type(exc).__name__}): {exc}"
        ) from exc
    cpu_load_seconds = time.perf_counter() - cpu_load_started
    load_report_gate = assert_checkpoint_load_complete(
        model, loading_info, model_dir
    )
    model.eval()
    model.config.use_cache = False
    validate_model_contract(model.config)
    if bool(getattr(model, "is_quantized", False)):
        raise RuntimeError("quantized backend detected; E2 requires BF16")
    dtype_counts: dict[str, int] = {}
    for parameter in model.parameters():
        name = str(parameter.dtype)
        dtype_counts[name] = dtype_counts.get(name, 0) + parameter.numel()
    if set(dtype_counts) != {"torch.bfloat16"}:
        raise RuntimeError(
            f"non-BF16 parameter dtypes detected after load: {sorted(dtype_counts)}"
        )
    dispatch = {
        "strategy": "cpu_only",
        "dispatch_seconds": 0.0,
        "device_map": None,
        "disk_offload": False,
    }
    if args.device == "auto":
        model, dispatch = dispatch_loaded_model(torch, model, args)
    first_parameter = next(model.parameters())
    input_device = model.model.embed_tokens.weight.device
    runtime = {
        "device_request": args.device,
        "load_strategy": dispatch["strategy"],
        "input_device": str(input_device),
        "first_parameter_device": str(first_parameter.device),
        "hf_device_map": dispatch["device_map"],
        "max_gpu_memory": args.max_gpu_memory if args.device == "auto" else None,
        "cpu_max_memory": args.cpu_max_memory if args.device == "auto" else None,
        "model_load_seconds": time.perf_counter() - started,
        "cpu_fused_conversion_load_seconds": cpu_load_seconds,
        "transformers_load_report_version": transformers.__version__,
        "load_report_gate": load_report_gate,
        "dispatch": dispatch,
        "parameter_dtype_counts": dtype_counts,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "is_quantized": bool(getattr(model, "is_quantized", False)),
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "eval_mode": not model.training,
        "use_cache": model.config.use_cache,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32": False,
        "cpu_threads": args.cpu_threads,
    }
    return torch, model, runtime


def validate_model_contract(config: Any) -> None:
    observed = {
        "architectures": list(config.architectures or []),
        "model_type": config.model_type,
        "num_hidden_layers": int(config.num_hidden_layers),
        "hidden_size": int(config.hidden_size),
        "num_experts": int(config.num_experts),
        "num_experts_per_tok": int(config.num_experts_per_tok),
        "max_position_embeddings": int(config.max_position_embeddings),
    }
    expected = {
        "architectures": ["OlmoeForCausalLM"],
        "model_type": "olmoe",
        "num_hidden_layers": N_LAYERS,
        "hidden_size": HIDDEN_DIM,
        "num_experts": N_EXPERTS,
        "num_experts_per_tok": EXPERTS_PER_TOKEN,
        "max_position_embeddings": MAX_CONTEXT,
    }
    if observed != expected:
        raise RuntimeError(f"OLMoE base contract mismatch: {observed} != {expected}")


def input_device(model: Any) -> Any:
    return model.model.embed_tokens.weight.device


def peak_rss_bytes() -> int:
    # Linux ru_maxrss is KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def score_target_window(
    torch: Any, model: Any, full_ids: np.ndarray, target_ids: np.ndarray
) -> dict[str, Any]:
    import torch.nn.functional as functional

    full = np.asarray(full_ids, dtype=np.int64).reshape(-1)
    target = np.asarray(target_ids, dtype=np.int64).reshape(-1)
    if full.size > MAX_CONTEXT or target.size < 2 or not np.array_equal(full[-target.size :], target):
        raise ValueError("causal scoring alignment/context contract failed")
    ids = torch.from_numpy(np.ascontiguousarray(full[None, :])).to(input_device(model))
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(input_ids=ids, use_cache=False, logits_to_keep=int(target.size))
    logits = output.logits[0, :-1]
    targets = torch.from_numpy(np.ascontiguousarray(target[1:])).to(logits.device)
    nll_parts: list[np.ndarray] = []
    for start in range(0, targets.numel(), 128):
        stop = min(start + 128, targets.numel())
        rows = logits[start:stop].float()
        losses = functional.cross_entropy(rows, targets[start:stop], reduction="none")
        nll_parts.append(losses.detach().cpu().numpy().astype(np.float64))
    nll = np.concatenate(nll_parts)
    elapsed = time.perf_counter() - started
    del output, logits, targets, ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mean_nll = float(nll.mean())
    return {
        "token_count": int(nll.size),
        "nll_sum": float(nll.sum(dtype=np.float64)),
        "mean_nll": mean_nll,
        "ppl": float(math.exp(mean_nll)),
        "target_ids_sha256": sha256_array(target[1:]),
        "per_token_nll_sha256": sha256_array(nll),
        "nll_quantiles": {
            label: float(np.quantile(nll, quantile, method="linear"))
            for label, quantile in (("p05", 0.05), ("p50", 0.50), ("p95", 0.95))
        },
        "forward_seconds": elapsed,
    }


def bringup_cell_path(output: Path, name: str) -> Path:
    if name not in {"short", "long"}:
        raise ValueError(name)
    length = BRINGUP_SHORT_TOKENS if name == "short" else BRINGUP_LONG_TOKENS
    return output / f"bringup_{name}_{length}.json"


def validate_bringup_cell_receipt(
    path: Path,
    manifest: dict[str, Any],
    prepared: dict[str, np.ndarray],
    *,
    name: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    receipt = read_json(path)
    receipt_schema = receipt.get("schema")
    length = BRINGUP_SHORT_TOKENS if name == "short" else BRINGUP_LONG_TOKENS
    if (
        receipt_schema
        not in {"moe_e2_bringup_cell_v1", "moe_e2_bringup_cell_v2"}
        or receipt.get("status") != "complete"
        or receipt.get("cell") != name
        or receipt.get("length") != length
        or receipt.get("model_revision") != MODEL_REVISION
    ):
        raise RuntimeError(f"bringup cell contract failed: {path}")
    content_sets = load_prepared_content_sets(path.parent, manifest, prepared)

    def checks(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
        ids = arrays["bringup_ids"][:length]
        return [
            receipt_array_check(
                receipt_field="input_ids_sha256",
                recorded_sha256=receipt.get("input_ids_sha256"),
                array_name="bringup_ids",
                selection=f"[:{length}]",
                value=ids,
            ),
            receipt_array_check(
                receipt_field="measurement.target_ids_sha256",
                recorded_sha256=receipt.get("measurement", {}).get(
                    "target_ids_sha256"
                ),
                array_name="bringup_ids",
                selection=f"[1:{length}]",
                value=ids[1:],
            ),
        ]

    binding = bind_receipt_to_prepared_content(
        receipt,
        content_sets,
        checks,
        require_declared_content=receipt_schema == "moe_e2_bringup_cell_v2",
    )
    return {"receipt": receipt, "binding": binding}


def valid_bringup_cell(
    path: Path,
    manifest: dict[str, Any],
    prepared: dict[str, np.ndarray],
    *,
    name: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    if read_json(path).get("status") == "running":
        return None
    return validate_bringup_cell_receipt(
        path, manifest, prepared, name=name
    )["receipt"]


def validate_consolidated_bringup_receipt(
    output: Path,
    manifest: dict[str, Any],
    prepared: dict[str, np.ndarray],
    *,
    receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = output / "bringup.json"
    receipt = read_json(path) if receipt is None else receipt
    receipt_schema = receipt.get("schema")
    if (
        receipt_schema not in {"moe_e2_bringup_v2", "moe_e2_bringup_v3"}
        or receipt.get("status") != "complete"
        or receipt.get("model_revision") != MODEL_REVISION
        or receipt.get("gate", {}).get("verdict") not in {"GREEN", "RED"}
    ):
        raise RuntimeError(f"consolidated bringup contract failed: {path}")
    cells = {
        name: validate_bringup_cell_receipt(
            bringup_cell_path(output, name), manifest, prepared, name=name
        )
        for name in ("short", "long")
    }
    for name in ("short", "long"):
        cell_path = bringup_cell_path(output, name)
        recorded = receipt.get("cell_receipts", {}).get(name, {})
        if (
            recorded.get("path") != str(cell_path)
            or recorded.get("sha256") != sha256_file(cell_path)
        ):
            raise RuntimeError(f"consolidated bringup cell linkage failed: {name}")
    content_sets = load_prepared_content_sets(output, manifest, prepared)

    def checks(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
        ids = arrays["bringup_ids"]
        return [
            receipt_array_check(
                receipt_field="short_512.target_ids_sha256",
                recorded_sha256=receipt.get("short_512", {}).get(
                    "target_ids_sha256"
                ),
                array_name="bringup_ids",
                selection="[1:512]",
                value=ids[1:BRINGUP_SHORT_TOKENS],
            ),
            receipt_array_check(
                receipt_field="long_2048.target_ids_sha256",
                recorded_sha256=receipt.get("long_2048", {}).get(
                    "target_ids_sha256"
                ),
                array_name="bringup_ids",
                selection="[1:2048]",
                value=ids[1:BRINGUP_LONG_TOKENS],
            ),
        ]

    binding = bind_receipt_to_prepared_content(
        receipt,
        content_sets,
        checks,
        require_declared_content=receipt_schema == "moe_e2_bringup_v3",
    )
    composite_binding = bind_transitive_receipt_content(
        {},
        [binding] + [cells[name]["binding"] for name in ("short", "long")],
        dependency_kind="consolidated_bringup_and_cells",
    )
    return {
        "receipt": receipt,
        "binding": composite_binding,
        "receipt_binding": binding,
        "cells": cells,
    }


def consolidate_bringup(
    output: Path, manifest: dict[str, Any], prepared: dict[str, np.ndarray]
) -> dict[str, Any] | None:
    existing_path = output / "bringup.json"
    if existing_path.is_file():
        existing = read_json(existing_path)
        if existing.get("status") != "running_cells":
            return validate_consolidated_bringup_receipt(
                output, manifest, prepared, receipt=existing
            )["receipt"]
    short_receipt = valid_bringup_cell(
        bringup_cell_path(output, "short"), manifest, prepared, name="short"
    )
    long_receipt = valid_bringup_cell(
        bringup_cell_path(output, "long"), manifest, prepared, name="long"
    )
    if short_receipt is None or long_receipt is None:
        return None
    cell_input_binding = bind_transitive_receipt_content(
        {},
        [
            validate_bringup_cell_receipt(
                bringup_cell_path(output, name),
                manifest,
                prepared,
                name=name,
            )["binding"]
            for name in ("short", "long")
        ],
        dependency_kind="bringup_cells",
    )
    short = short_receipt["measurement"]
    long = long_receipt["measurement"]
    delta = float(long["mean_nll"] - short["mean_nll"])
    plausible = BRINGUP_PPL_MIN <= short["ppl"] <= BRINGUP_PPL_MAX
    context_ok = delta <= BRINGUP_LONG_MINUS_SHORT_NLL_CAP
    green = bool(plausible and context_ok)
    receipt = {
        "schema": "moe_e2_bringup_v3",
        "status": "complete",
        "created_at": now_iso(),
        "order": str(ORDER_PATH),
        "script": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "model_id": MODEL_ID,
        "model_dir": str(DEFAULT_MODEL_DIR),
        "model_revision": MODEL_REVISION,
        **content_fields_from_binding(cell_input_binding),
        "registered_gate": {
            "short_ppl_plausible_range_inclusive": [BRINGUP_PPL_MIN, BRINGUP_PPL_MAX],
            "long_minus_short_mean_nll_lte": BRINGUP_LONG_MINUS_SHORT_NLL_CAP,
            "lengths": [BRINGUP_SHORT_TOKENS, BRINGUP_LONG_TOKENS],
            "delta_direction": "mean_nll_2048 - mean_nll_512 (negative improvement allowed)",
        },
        "deterministic_forward": {
            "model_dtype": "torch.bfloat16",
            "attention": "transformers SDPA",
            "quantization": "forbidden and absent",
            "model_eval": True,
            "dropout": 0.0,
            "use_cache": False,
            "add_special_tokens": False,
            "scoring": "raw contiguous WikiText; all next-token targets except first",
            "seed": DEFAULT_SEED,
            "offline": True,
            "cell_invocations_may_be_split": True,
        },
        "cell_receipts": {
            "short": {
                "path": str(bringup_cell_path(output, "short")),
                "sha256": sha256_file(bringup_cell_path(output, "short")),
            },
            "long": {
                "path": str(bringup_cell_path(output, "long")),
                "sha256": sha256_file(bringup_cell_path(output, "long")),
            },
        },
        "short_512": short,
        "long_2048": long,
        "long_minus_short_mean_nll": delta,
        "gate": {
            "verdict": "GREEN" if green else "RED",
            "short_ppl_plausible": plausible,
            "long_context_delta_ok": context_ok,
        },
        "peak_rss_bytes_max": max(
            int(short_receipt["peak_rss_bytes"]), int(long_receipt["peak_rss_bytes"])
        ),
    }
    write_json(output / "bringup.json", receipt)
    return receipt


def bringup(args: argparse.Namespace) -> int:
    output = ensure_output(args.output_dir)
    manifest, prepared = load_prepared(output)
    existing = consolidate_bringup(output, manifest, prepared)
    if existing is not None:
        print(json.dumps({"status": "existing", "receipt": str(output / "bringup.json")}))
        return 0 if existing["gate"]["verdict"] == "GREEN" else 3
    requested = (
        ["short", "long"] if args.bringup_cell == "both" else [args.bringup_cell]
    )
    missing = [
        name
        for name in requested
        if valid_bringup_cell(
            bringup_cell_path(output, name), manifest, prepared, name=name
        )
        is None
    ]
    if not missing:
        print(json.dumps({"status": "partial_existing", "cells": requested}), flush=True)
        return 0
    aggregate_running = {
        "schema": "moe_e2_bringup_v3",
        "status": "running_cells",
        "created_at": now_iso(),
        "requested_cells": requested,
        "missing_cells_at_start": missing,
        "script_sha256": sha256_file(SCRIPT_PATH),
        "model_revision": MODEL_REVISION,
        **prepared_content_fields(manifest),
    }
    write_json(output / "bringup.json", aggregate_running)
    started = time.perf_counter()
    torch, model, runtime = load_model(args)
    ids = prepared["bringup_ids"]
    try:
        for name in missing:
            length = BRINGUP_SHORT_TOKENS if name == "short" else BRINGUP_LONG_TOKENS
            cell_path = bringup_cell_path(output, name)
            cell: dict[str, Any] = {
                "schema": "moe_e2_bringup_cell_v2",
                "status": "running",
                "created_at": now_iso(),
                "cell": name,
                "length": length,
                "script": str(SCRIPT_PATH),
                "script_sha256": sha256_file(SCRIPT_PATH),
                "model_id": MODEL_ID,
                "model_dir": str(validate_model_dir(args.model_dir)),
                "model_revision": MODEL_REVISION,
                **prepared_content_fields(manifest),
                "input_ids_sha256": sha256_array(ids[:length]),
                "runtime": runtime,
            }
            write_json(cell_path, cell)
            cell_started = time.perf_counter()
            measurement = score_target_window(torch, model, ids[:length], ids[:length])
            cell.update(
                {
                    "status": "complete",
                    "completed_at": now_iso(),
                    "measurement": measurement,
                    "cell_wall_seconds": time.perf_counter() - cell_started,
                    "invocation_wall_seconds": time.perf_counter() - started,
                    "peak_rss_bytes": peak_rss_bytes(),
                }
            )
            write_json(cell_path, cell)
            print(
                json.dumps(
                    {
                        "cell": name,
                        "length": length,
                        "ppl": measurement["ppl"],
                        "mean_nll": measurement["mean_nll"],
                        "forward_seconds": measurement["forward_seconds"],
                    }
                ),
                flush=True,
            )
    except Exception as exc:
        if "cell" in locals():
            cell.update(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "peak_rss_bytes": peak_rss_bytes(),
                }
            )
            write_json(cell_path, cell)
        raise
    finally:
        del model
        gc.collect()
    consolidated = consolidate_bringup(output, manifest, prepared)
    if consolidated is None:
        write_json(
            output / "bringup.json",
            {
                **aggregate_running,
                "status": "partial_cells_complete",
                "completed_cells": requested,
            },
        )
        print(
            json.dumps(
                {
                    "status": "partial_cells_complete",
                    "next": "run the remaining bringup cell",
                }
            ),
            flush=True,
        )
        return 0
    print(
        json.dumps(
            {
                "gate": "E2-G-1",
                "verdict": consolidated["gate"]["verdict"],
                "ppl_512": consolidated["short_512"]["ppl"],
                "nll_delta_2048_minus_512": consolidated[
                    "long_minus_short_mean_nll"
                ],
                "receipt": str(output / "bringup.json"),
            }
        ),
        flush=True,
    )
    return 0 if consolidated["gate"]["verdict"] == "GREEN" else 3


def numpy_expert_add(
    native_output: np.ndarray,
    hidden: np.ndarray,
    key: np.ndarray,
    tau: float,
    A: np.ndarray,
    B: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference ABI: non-fired rows are copied and never enter an add."""

    native = np.asarray(native_output)
    h = np.asarray(hidden, dtype=np.float32)
    flat_native = native.reshape(-1, native.shape[-1])
    flat_h = h.reshape(-1, h.shape[-1])
    scores = flat_h @ np.asarray(key, dtype=np.float32)
    fired = scores >= float(tau)
    result = flat_native.copy()
    if fired.any() and np.count_nonzero(B):
        z = flat_h[fired] @ np.asarray(A, dtype=np.float32).T
        z = z / (1.0 + np.exp(-z))
        delta = z @ np.asarray(B, dtype=np.float32).T
        result[fired] = result[fired] + delta.astype(result.dtype)
    return result.reshape(native.shape), scores, fired


def unit_vector(vector: np.ndarray, *, name: str) -> tuple[np.ndarray, float]:
    work = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(work))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise RuntimeError(f"{name} has degenerate norm {norm}")
    return np.ascontiguousarray((work / norm).astype(np.float32)), norm


def selected_tokens(
    array: np.ndarray, indices: Sequence[int], hidden_dim: int
) -> np.ndarray:
    return np.ascontiguousarray(
        np.asarray(array)[np.asarray(indices, dtype=np.int64)].reshape(-1, hidden_dim),
        dtype=np.float32,
    )


def centered_four_corpora(
    hidden: dict[str, np.ndarray], indices: Sequence[int]
) -> tuple[np.ndarray, int, np.ndarray, float, dict[str, np.ndarray]]:
    dimension = int(hidden["narrative"].shape[-1])
    blocks = {
        corpus: selected_tokens(hidden[corpus], indices, dimension)
        for corpus in KEY_CORPORA
    }
    token_counts = {corpus: int(block.shape[0]) for corpus, block in blocks.items()}
    if len(set(token_counts.values())) != 1:
        raise RuntimeError(f"K4 requires equal corpus token counts: {token_counts}")
    means = {
        corpus: block.mean(axis=0, dtype=np.float64) for corpus, block in blocks.items()
    }
    centered = np.concatenate(
        [
            np.ascontiguousarray(
                blocks[corpus] - means[corpus].astype(np.float32), dtype=np.float32
            )
            for corpus in KEY_CORPORA
        ],
        axis=0,
    )
    dof = int(centered.shape[0] - len(KEY_CORPORA))
    diagonal = np.sum(centered.astype(np.float64) ** 2, axis=0) / float(dof)
    mean_variance = float(diagonal.mean())
    negative_mean = np.mean(
        np.stack([means[corpus] for corpus in NEGATIVE_CORPORA]), axis=0
    )
    delta = means["narrative"] - negative_mean
    return centered, dof, delta, mean_variance, means


def cg_fisher_solve(
    centered: np.ndarray,
    dof: int,
    delta: np.ndarray,
    regularization_lambda: float,
    *,
    rtol: float,
    maxiter: int,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy.sparse.linalg import LinearOperator, cg

    x = np.asarray(centered, dtype=np.float32)
    b = np.asarray(delta, dtype=np.float32)
    lam = np.float32(regularization_lambda)
    inv_dof = np.float32(1.0 / float(dof))
    diagonal = (
        np.sum(x.astype(np.float64) ** 2, axis=0) / float(dof)
    ).astype(np.float32)

    def matvec(vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float32)
        return np.asarray(x.T @ (x @ value) * inv_dof + lam * value, dtype=np.float32)

    def precondition(vector: np.ndarray) -> np.ndarray:
        return np.asarray(vector, dtype=np.float32) / (diagonal + lam)

    operator = LinearOperator((b.size, b.size), matvec=matvec, dtype=np.float32)
    preconditioner = LinearOperator(
        (b.size, b.size), matvec=precondition, dtype=np.float32
    )
    iterations = 0

    def callback(_value: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    started = time.perf_counter()
    solution, info = cg(
        operator,
        b,
        x0=None if x0 is None else np.asarray(x0, dtype=np.float32),
        rtol=rtol,
        atol=0.0,
        maxiter=maxiter,
        M=preconditioner,
        callback=callback,
    )
    denominator = max(float(np.linalg.norm(b.astype(np.float64))), 1.0e-30)
    relative_residual = float(
        np.linalg.norm(matvec(solution).astype(np.float64) - b.astype(np.float64))
        / denominator
    )
    return np.asarray(solution, dtype=np.float32), {
        "solver": "scipy CG over exact pooled within-corpus covariance matvec",
        "info": int(info),
        "converged": bool(info == 0),
        "iterations": iterations,
        "rtol": rtol,
        "maxiter": maxiter,
        "relative_residual": relative_residual,
        "wall_seconds": time.perf_counter() - started,
    }


def exact_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    from scipy.stats import rankdata

    pos = np.asarray(positive, dtype=np.float64).reshape(-1)
    neg = np.asarray(negative, dtype=np.float64).reshape(-1)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("AUC inputs must be nonempty")
    combined = np.concatenate([pos, neg])
    ranks = rankdata(combined, method="average")
    u = ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def validation_auc_four(hidden: dict[str, np.ndarray], key: np.ndarray) -> dict[str, Any]:
    positive = selected_tokens(
        hidden["narrative"], INNER_VALIDATION_INDICES, key.size
    ) @ key
    aucs = {
        corpus: exact_auc(
            positive,
            selected_tokens(hidden[corpus], INNER_VALIDATION_INDICES, key.size) @ key,
        )
        for corpus in NEGATIVE_CORPORA
    }
    return {
        **{f"auc_narrative_vs_{corpus}": value for corpus, value in aucs.items()},
        "minimum_auc": min(aucs.values()),
        "mean_auc": float(np.mean(list(aucs.values()))),
    }


def fit_k4_direction(
    hidden: dict[str, np.ndarray], *, cg_rtol: float, cg_maxiter: int
) -> tuple[np.ndarray, dict[str, Any]]:
    centered, dof, delta, mean_variance, _means = centered_four_corpora(
        hidden, INNER_TRAIN_INDICES
    )
    candidates_by_alpha: dict[float, dict[str, Any]] = {}
    warm: np.ndarray | None = None
    for alpha in reversed(K4_ALPHA_GRID):
        regularization = float(alpha * mean_variance)
        solution, solver = cg_fisher_solve(
            centered,
            dof,
            delta,
            regularization,
            rtol=cg_rtol,
            maxiter=cg_maxiter,
            x0=warm,
        )
        if not solver["converged"]:
            first = solver
            solution, solver = cg_fisher_solve(
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
            key, norm = unit_vector(solution, name=f"K4 alpha={alpha}")
            candidate["direction_source_norm"] = norm
            candidate["validation"] = validation_auc_four(hidden, key)
            warm = solution
        else:
            warm = None
        candidates_by_alpha[alpha] = candidate
    del centered
    gc.collect()
    candidates = [candidates_by_alpha[alpha] for alpha in K4_ALPHA_GRID]
    converged = [item for item in candidates if item["fit_converged"]]
    if not converged:
        raise RuntimeError("no converged K4 shrinkage candidate")
    selected = max(
        converged,
        key=lambda item: (
            item["validation"]["minimum_auc"],
            item["validation"]["mean_auc"],
            item["alpha"],
        ),
    )
    selected_alpha = float(selected["alpha"])
    centered_full, dof_full, delta_full, variance_full, _means_full = centered_four_corpora(
        hidden, FIT_INDICES
    )
    selected_lambda = float(selected_alpha * variance_full)
    solution, final_solver = cg_fisher_solve(
        centered_full,
        dof_full,
        delta_full,
        selected_lambda,
        rtol=cg_rtol,
        maxiter=cg_maxiter,
    )
    if not final_solver["converged"]:
        first = final_solver
        solution, final_solver = cg_fisher_solve(
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
        raise RuntimeError("final full-FIT K4 solve did not converge")
    key, source_norm = unit_vector(solution, name="K4 final direction")
    return key, {
        "construction": (
            "unit_norm((pooled within-four-corpus FIT covariance + lambda I)^-1 "
            "(mean(narrative_FIT) - equal_mean(wikitext,code,grm)_FIT))"
        ),
        "negative_pool_weighting": "equal tokens and one-third mean weight per negative corpus",
        "alpha_grid": list(K4_ALPHA_GRID),
        "inner_train_window_indices": list(INNER_TRAIN_INDICES),
        "inner_validation_window_indices": list(INNER_VALIDATION_INDICES),
        "selection_rule": (
            "highest minimum validation AUC over wikitext/code/grm, then mean AUC, "
            "then stronger shrinkage"
        ),
        "validation_candidates": candidates,
        "selected_alpha": selected_alpha,
        "selected_full_fit_lambda": selected_lambda,
        "full_fit_mean_within_variance": variance_full,
        "full_fit_dof": dof_full,
        "final_solver": final_solver,
        "source_norm": source_norm,
        "stored_norm": float(np.linalg.norm(key.astype(np.float64))),
        "key_sha256": sha256_array(key),
        "selected_validation": selected["validation"],
    }


def score_four(hidden: dict[str, np.ndarray], key: np.ndarray) -> dict[str, np.ndarray]:
    return {
        corpus: np.ascontiguousarray(
            (array.astype(np.float32).reshape(-1, key.size) @ key).reshape(
                array.shape[0], array.shape[1]
            ),
            dtype=np.float32,
        )
        for corpus, array in hidden.items()
    }


def bootstrap_rate(
    indicator: np.ndarray,
    *,
    resamples: int,
    seed: int,
    window_indices: Sequence[int],
) -> dict[str, Any]:
    per_window = np.asarray(indicator, dtype=np.float64).mean(axis=1)
    if len(window_indices) != per_window.size:
        raise ValueError("bootstrap window index mismatch")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, per_window.size, size=(resamples, per_window.size))
    boot = per_window[sampled].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975], method="linear")
    return {
        "value": float(per_window.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_unit": "windows",
        "bootstrap_resamples": resamples,
        "per_window": [
            {"window_index": int(index), "rate": float(rate)}
            for index, rate in zip(window_indices, per_window)
        ],
    }


def select_tau_four(scores: dict[str, np.ndarray]) -> dict[str, Any]:
    fit = np.asarray(FIT_INDICES, dtype=np.int64)
    pooled_negative = np.concatenate(
        [scores[corpus][fit].reshape(-1) for corpus in NEGATIVE_CORPORA]
    )
    curve: list[dict[str, Any]] = []
    for label, quantile in TAU_GRID:
        tau = float(np.quantile(pooled_negative, quantile, method="linear"))
        rates = {
            "narrative_recall": float(np.mean(scores["narrative"][fit] >= tau)),
            **{
                f"{corpus}_fpr": float(np.mean(scores[corpus][fit] >= tau))
                for corpus in NEGATIVE_CORPORA
            },
        }
        feasible = all(rates[f"{corpus}_fpr"] <= FPR_CAPS[corpus] for corpus in FPR_CAPS)
        curve.append(
            {
                "quantile_label": label,
                "quantile": quantile,
                "tau": tau,
                "fit": rates,
                "fit_code_grm_feasible": feasible,
            }
        )
    feasible = [item for item in curve if item["fit_code_grm_feasible"]]
    pool = feasible if feasible else curve
    selected = max(
        pool,
        key=lambda item: (
            item["fit"]["narrative_recall"]
            if feasible
            else -max(
                item["fit"][f"{corpus}_fpr"] / FPR_CAPS[corpus]
                for corpus in FPR_CAPS
            ),
            -item["fit"]["grm_fpr"],
            -item["fit"]["code_fpr"],
            -item["fit"]["wikitext_fpr"],
            item["quantile"],
        ),
    )
    return {
        "tau_definition": (
            "linear p90/p95/p99/p99.5 over pooled wikitext+code+grm FIT scores; "
            "maximize narrative FIT recall subject to code+grm FIT FPR caps; "
            "generic WikiText is descriptive; ties lower grm/code/wikitext rate"
        ),
        "fit_constraint_corpora": ["code", "grm"],
        "generic_wikitext_fire_rate": "descriptive_no_pass_fail_bound",
        "fire_comparison": "score >= tau",
        "fit_negative_token_count": int(pooled_negative.size),
        "fit_fpr_feasible": bool(feasible),
        "selected_quantile_label": selected["quantile_label"],
        "selected_quantile": selected["quantile"],
        "selected_tau": selected["tau"],
        "selected_fit_rates": selected["fit"],
        "operating_curve": curve,
    }


def analyze_key_scores(
    scores: dict[str, np.ndarray], *, resamples: int, seed: int
) -> dict[str, Any]:
    threshold = select_tau_four(scores)
    tau = float(threshold["selected_tau"])
    evaluation = np.asarray(EVAL_INDICES, dtype=np.int64)
    fired = {corpus: scores[corpus][evaluation] >= tau for corpus in KEY_CORPORA}
    metrics = {
        **threshold,
        "eval_window_indices": list(EVAL_INDICES),
        "eval": {
            "narrative_recall": bootstrap_rate(
                fired["narrative"],
                resamples=resamples,
                seed=seed + 1,
                window_indices=EVAL_INDICES,
            ),
            **{
                f"{corpus}_fpr": bootstrap_rate(
                    fired[corpus],
                    resamples=resamples,
                    seed=seed + 10 + index,
                    window_indices=EVAL_INDICES,
                )
                for index, corpus in enumerate(NEGATIVE_CORPORA)
            },
            "auc": {
                corpus: exact_auc(
                    scores["narrative"][evaluation], scores[corpus][evaluation]
                )
                for corpus in NEGATIVE_CORPORA
            },
        },
    }
    metrics["eval"]["generic_fire_rate"] = dict(metrics["eval"]["wikitext_fpr"])
    metrics["qualifies"] = bool(
        metrics["fit_fpr_feasible"]
        and metrics["eval"]["narrative_recall"]["value"] >= RECALL_FLOOR
        and metrics["eval"]["code_fpr"]["value"] <= FPR_CAPS["code"]
        and metrics["eval"]["grm_fpr"]["value"] <= FPR_CAPS["grm"]
    )
    metrics["registered_criteria"] = {
        "eval_narrative_recall_gte": RECALL_FLOOR,
        "eval_code_fpr_lte": FPR_CAPS["code"],
        "eval_grm_fpr_lte": FPR_CAPS["grm"],
        "generic_wikitext_fire_rate": "descriptive_no_pass_fail_bound",
        "fit_threshold_frozen_before_eval": True,
    }
    return metrics


def bootstrap_recovery(
    base_mean_nll: np.ndarray,
    teacher_mean_nll: np.ndarray,
    expert_mean_nll: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    base = np.asarray(base_mean_nll, dtype=np.float64)
    teacher = np.asarray(teacher_mean_nll, dtype=np.float64)
    expert = np.asarray(expert_mean_nll, dtype=np.float64)
    if not (base.shape == teacher.shape == expert.shape) or base.size == 0:
        raise ValueError("bootstrap recovery arms must have equal nonempty shape")

    def recovery_for(indices: np.ndarray) -> np.ndarray:
        base_ppl = np.exp(base[indices].mean(axis=1))
        teacher_ppl = np.exp(teacher[indices].mean(axis=1))
        expert_ppl = np.exp(expert[indices].mean(axis=1))
        denominator = base_ppl - teacher_ppl
        return np.divide(
            base_ppl - expert_ppl,
            denominator,
            out=np.full_like(denominator, np.nan),
            where=denominator != 0,
        )

    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, base.size, size=(resamples, base.size))
    boot = recovery_for(sampled)
    finite = boot[np.isfinite(boot)]
    if finite.size == 0:
        low = high = float("nan")
    else:
        low, high = np.quantile(finite, [0.025, 0.975], method="linear")
    base_ppl = math.exp(float(base.mean()))
    teacher_ppl = math.exp(float(teacher.mean()))
    expert_ppl = math.exp(float(expert.mean()))
    gap = base_ppl - teacher_ppl
    point = (base_ppl - expert_ppl) / gap if gap != 0 else float("nan")
    return {
        "ppl_base": base_ppl,
        "ppl_teacher": teacher_ppl,
        "ppl_expert": expert_ppl,
        "teacher_gap_ppl": gap,
        "recovery_fraction": point,
        "recovery_percent": point * 100.0,
        "ci95_low_fraction": float(low),
        "ci95_high_fraction": float(high),
        "ci95_low_percent": float(low * 100.0),
        "ci95_high_percent": float(high * 100.0),
        "bootstrap_unit": "behavioral windows",
        "bootstrap_resamples": resamples,
        "finite_bootstrap_replicates": int(finite.size),
    }


def load_check(args: argparse.Namespace) -> int:
    import accelerate
    import transformers

    output = ensure_output(args.output_dir)
    receipt_path = output / (
        "load_report_cpu_mimic.json"
        if args.device == "cpu"
        else "load_report_gpu_dispatch.json"
    )
    started = time.perf_counter()
    torch, model, runtime = load_model(args)
    try:
        gate = runtime["load_report_gate"]
        if gate["status"] != "passed":
            raise RuntimeError("load-check observed a non-passing load-report gate")
        receipt = {
            "schema": "moe_e2_f1_load_report_v1",
            "status": "passed",
            "created_at": now_iso(),
            "order": str(REPO_ROOT / "orders" / "MOE_E2_F1_GPU_LOAD_FIX.md"),
            "script": str(SCRIPT_PATH),
            "script_sha256": sha256_file(SCRIPT_PATH),
            "model_id": MODEL_ID,
            "model_dir": str(validate_model_dir(args.model_dir)),
            "model_revision": MODEL_REVISION,
            "device_request": args.device,
            "no_forward_executed": True,
            "runtime": runtime,
            "software": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "accelerate": accelerate.__version__,
            },
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_bytes": peak_rss_bytes(),
        }
        write_json(receipt_path, receipt)
        print(
            json.dumps(
                {
                    "status": "passed",
                    "receipt": str(receipt_path),
                    "missing": gate["missing_count"],
                    "unexpected": gate["unexpected_count"],
                    "mismatched": gate["mismatched_count"],
                    "conversion": gate["conversion_count"],
                    "checkpoint_tensors": gate["checkpoint"]["tensor_count"],
                    "loaded_parameter_bytes": gate["loaded_parameter_bytes"],
                    "no_forward_executed": True,
                }
            ),
            flush=True,
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


def self_test(args: argparse.Namespace) -> int:
    output = ensure_output(args.output_dir)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    rng = np.random.default_rng(args.seed)
    dimension = 16
    native = rng.standard_normal((2, 7, dimension)).astype(np.float32)
    hidden = rng.standard_normal((2, 7, dimension)).astype(np.float32)
    key, _norm = unit_vector(rng.standard_normal(dimension), name="synthetic key")
    A = rng.standard_normal((4, dimension)).astype(np.float32) * 0.1
    B = rng.standard_normal((dimension, 4)).astype(np.float32) * 0.1
    tau = float(np.median(hidden.reshape(-1, dimension) @ key))
    result, _scores, fired = numpy_expert_add(native, hidden, key, tau, A, B)
    check("abi_has_mixed_gate", bool(fired.any() and (~fired).any()), fired.tolist())
    check(
        "abi_nonfired_bit_identity",
        np.array_equal(result.reshape(-1, dimension)[~fired], native.reshape(-1, dimension)[~fired]),
        "np.array_equal",
    )
    zero, _scores, _fired = numpy_expert_add(native, hidden, key, tau, A, np.zeros_like(B))
    check("abi_zero_B_bit_identity", np.array_equal(zero, native), "np.array_equal")

    nonfinite_fixture = {
        "forced_no_fire_mounted": {"mount": {"tau": float("inf")}},
        "live_gate": {"mount": {"score_max": float("nan")}},
    }
    expected_nonfinite = [
        ("$.forced_no_fire_mounted.mount.tau", "inf"),
        ("$.live_gate.mount.score_max", "nan"),
    ]
    observed_nonfinite = find_non_finite_json_values(nonfinite_fixture)
    check(
        "json_nonfinite_walker_names_all_paths",
        observed_nonfinite == expected_nonfinite,
        observed_nonfinite,
    )
    named_failure = ""
    try:
        require_finite_json_values(nonfinite_fixture)
    except ValueError as exc:
        named_failure = str(exc)
    check(
        "json_nonfinite_validation_fails_loud_and_named",
        all(path in named_failure for path, _spelling in expected_nonfinite),
        named_failure,
    )
    rejection_probe = output / f"nonfinite_rejection_probe_{os.getpid()}.json"
    write_failure = ""
    try:
        write_json(rejection_probe, nonfinite_fixture)
    except ValueError as exc:
        write_failure = str(exc)
    check(
        "write_json_rejects_before_write_with_named_paths",
        not rejection_probe.exists()
        and all(path in write_failure for path, _spelling in expected_nonfinite),
        {"error": write_failure, "path_exists": rejection_probe.exists()},
    )

    # CPU-only structural mimic of eval_abi's zero-B, forced-no-fire, and live
    # mount snapshots. This exercises the same ExpertMount telemetry without
    # loading the 7B checkpoint.
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
    baseline_mount_output = mount_module(mount_hidden)
    with ExpertMount(
        mount_module,
        torch_module=torch,
        key=mount_key,
        tau=0.0,
        A=mount_A,
        B=np.zeros_like(mount_B),
        label="zero_B",
    ) as zero_mount:
        zero_mount_output = mount_module(mount_hidden)
    with ExpertMount(
        mount_module,
        torch_module=torch,
        key=mount_key,
        tau=float("inf"),
        A=mount_A,
        B=mount_B,
        label="forced_no_fire",
    ) as nofire_mount:
        nofire_mount_output = mount_module(mount_hidden)
    with ExpertMount(
        mount_module,
        torch_module=torch,
        key=mount_key,
        tau=0.0,
        A=mount_A,
        B=mount_B,
        label="live",
    ) as live_mount:
        live_mount_output = mount_module(mount_hidden)
    abi_mimic = {
        "zero_initialized_B_mounted": {"mount": zero_mount.calls[0]},
        "forced_no_fire_mounted": {"mount": nofire_mount.calls[0]},
        "live_gate": {"mount": live_mount.calls[0]},
    }
    check(
        "abi_mount_mimic_zero_and_nofire_bit_identity",
        torch.equal(zero_mount_output, baseline_mount_output)
        and torch.equal(nofire_mount_output, baseline_mount_output),
        {
            "zero_B": torch.equal(zero_mount_output, baseline_mount_output),
            "forced_no_fire": torch.equal(nofire_mount_output, baseline_mount_output),
        },
    )
    check(
        "abi_mount_mimic_forced_tau_uses_json_sentinel",
        nofire_mount.calls[0]["tau"] == "inf",
        nofire_mount.calls[0],
    )
    check(
        "abi_mount_mimic_live_gate_mixed_and_exact_nonfire",
        live_mount.calls[0]["fire_count"] == 2
        and live_mount.calls[0]["nonfire_count"] == 2
        and live_mount.calls[0]["nonfired_rows_bit_equal"]
        and not torch.equal(live_mount_output, baseline_mount_output),
        live_mount.calls[0],
    )
    check(
        "abi_mount_mimic_strict_json_finite",
        not find_non_finite_json_values(abi_mimic),
        find_non_finite_json_values(abi_mimic),
    )
    json.dumps(abi_mimic, allow_nan=False)

    digest_fixture = np.arange(24, dtype=np.int64).reshape(4, 6)
    layout_variants = (
        digest_fixture,
        np.asfortranarray(digest_fixture),
        digest_fixture[:, ::-1][:, ::-1],
        digest_fixture.astype(">i8"),
    )
    layout_records = [array_content_record(value) for value in layout_variants]
    check(
        "prepared_array_digest_normalizes_layout_and_endian",
        all(record == layout_records[0] for record in layout_records[1:]),
        layout_records,
    )
    check(
        "prepared_array_digest_binds_shape_and_dtype",
        array_content_record(digest_fixture.reshape(2, 12)) != layout_records[0]
        and array_content_record(digest_fixture.astype(np.int32)) != layout_records[0],
        {
            "base": layout_records[0],
            "reshaped": array_content_record(digest_fixture.reshape(2, 12)),
            "int32": array_content_record(digest_fixture.astype(np.int32)),
        },
    )
    digest_arrays = {
        "alpha": digest_fixture,
        "beta": digest_fixture + 100,
    }
    combined = prepared_content_record(digest_arrays)
    combined_reversed = prepared_content_record(
        {"beta": digest_arrays["beta"], "alpha": digest_arrays["alpha"]}
    )
    check(
        "prepared_combined_digest_is_name_order_independent",
        combined["sha256"] == combined_reversed["sha256"],
        [combined["sha256"], combined_reversed["sha256"]],
    )
    one_value_changed = digest_arrays["beta"].copy()
    one_value_changed[0, 0] += 1
    check(
        "prepared_combined_digest_changes_with_content_name_shape_or_dtype",
        len(
            {
                combined["sha256"],
                prepared_content_record(
                    {"alpha": digest_fixture, "beta": one_value_changed}
                )["sha256"],
                prepared_content_record(
                    {"renamed": digest_fixture, "beta": digest_arrays["beta"]}
                )["sha256"],
                prepared_content_record(
                    {
                        "alpha": digest_fixture.reshape(2, 12),
                        "beta": digest_arrays["beta"],
                    }
                )["sha256"],
                prepared_content_record(
                    {
                        "alpha": digest_fixture.astype(np.int32),
                        "beta": digest_arrays["beta"],
                    }
                )["sha256"],
            }
        )
        == 5,
        "five distinct combined content digests",
    )
    with tempfile.TemporaryDirectory(prefix="olmoe-e2-f4-content-") as temporary:
        temporary_root = Path(temporary)
        plain_path = temporary_root / "plain.npz"
        compressed_path = temporary_root / "compressed.npz"
        save_npz(plain_path, compressed=False, **digest_arrays)
        save_npz(compressed_path, compressed=True, **digest_arrays)
        with np.load(plain_path, allow_pickle=False) as archive:
            plain_arrays = {name: archive[name].copy() for name in archive.files}
        with np.load(compressed_path, allow_pickle=False) as archive:
            compressed_arrays = {
                name: archive[name].copy() for name in archive.files
            }
        check(
            "prepared_content_ignores_npz_container_bytes",
            sha256_file(plain_path) != sha256_file(compressed_path)
            and prepared_content_record(plain_arrays)
            == prepared_content_record(compressed_arrays),
            {
                "plain_archive_sha256": sha256_file(plain_path),
                "compressed_archive_sha256": sha256_file(compressed_path),
                "content_sha256": prepared_content_record(plain_arrays)["sha256"],
            },
        )

    prepared_fixture = {
        name: np.zeros(shape, dtype=np.int64)
        for name, shape in PREPARED_ARRAY_SHAPES.items()
    }
    prepared_fixture_content = prepared_content_record(prepared_fixture)
    legacy_declaration = {
        "sha256": "container-bytes-are-not-provenance",
        "arrays": {
            name: {
                "shape": record["shape"],
                "sha256": record["sha256"],
            }
            for name, record in prepared_fixture_content["arrays"].items()
        },
    }
    modern_declaration = {
        "content_digest_algorithm": prepared_fixture_content["algorithm"],
        "content_sha256": prepared_fixture_content["sha256"],
        "arrays": prepared_fixture_content["arrays"],
    }
    check(
        "legacy_manifest_ignores_container_sha_but_v2_requires_full_content",
        validate_declared_prepared_content(
            legacy_declaration, prepared_fixture, require_combined=False
        )
        == prepared_fixture_content
        and validate_declared_prepared_content(
            modern_declaration, prepared_fixture, require_combined=True
        )
        == prepared_fixture_content,
        prepared_fixture_content["sha256"],
    )
    stripped_v2_rejected = False
    missing_dtype_v2_rejected = False
    try:
        validate_declared_prepared_content(
            {"arrays": modern_declaration["arrays"]},
            prepared_fixture,
            require_combined=True,
        )
    except RuntimeError:
        stripped_v2_rejected = True
    try:
        dtype_stripped = json.loads(json.dumps(modern_declaration))
        dtype_stripped["arrays"]["code_ids"].pop("dtype")
        validate_declared_prepared_content(
            dtype_stripped, prepared_fixture, require_combined=True
        )
    except RuntimeError:
        missing_dtype_v2_rejected = True
    check(
        "v2_manifest_cannot_downgrade_to_legacy_array_records",
        stripped_v2_rejected and missing_dtype_v2_rejected,
        {
            "combined_fields_stripped_rejected": stripped_v2_rejected,
            "dtype_stripped_rejected": missing_dtype_v2_rejected,
        },
    )

    synthetic_set = content_set_from_arrays(
        label="synthetic_registered_legacy",
        arrays=digest_arrays,
        content=combined,
        legacy_archive_sha256="a" * 64,
        metadata_path=None,
    )

    def synthetic_checks(
        arrays: dict[str, np.ndarray], receipt: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return [
            receipt_array_check(
                receipt_field="input_ids_sha256",
                recorded_sha256=receipt.get("input_ids_sha256"),
                array_name="alpha",
                selection="[all]",
                value=arrays["alpha"],
            )
        ]

    legacy_receipt = {
        "prepared_windows_sha256": "a" * 64,
        "input_ids_sha256": sha256_array(digest_arrays["alpha"]),
    }
    legacy_binding = bind_receipt_to_prepared_content(
        legacy_receipt,
        [synthetic_set],
        lambda arrays: synthetic_checks(arrays, legacy_receipt),
    )
    check(
        "legacy_receipt_uses_registered_content_not_container_bytes",
        legacy_binding["mode"] == "registered_legacy_archive_to_content",
        legacy_binding,
    )
    unknown_legacy_rejected = False
    try:
        unknown = dict(legacy_receipt, prepared_windows_sha256="b" * 64)
        bind_receipt_to_prepared_content(
            unknown,
            [synthetic_set],
            lambda arrays: synthetic_checks(arrays, unknown),
        )
    except RuntimeError:
        unknown_legacy_rejected = True
    check(
        "unknown_legacy_archive_sha_is_rejected",
        unknown_legacy_rejected,
        "no unregistered byte-SHA fallback",
    )
    wrong_modern_digest_rejected = False
    try:
        wrong_modern = {
            **legacy_receipt,
            "prepared_windows_content_digest_algorithm": PREPARED_CONTENT_ALGORITHM,
            "prepared_windows_content_sha256": "c" * 64,
        }
        bind_receipt_to_prepared_content(
            wrong_modern,
            [synthetic_set],
            lambda arrays: synthetic_checks(arrays, wrong_modern),
        )
    except RuntimeError:
        wrong_modern_digest_rejected = True
    check(
        "wrong_modern_content_digest_cannot_downgrade_to_legacy",
        wrong_modern_digest_rejected,
        "present modern digest fails closed",
    )
    partial_modern_provenance_rejected = False
    try:
        partial_modern = {
            **legacy_receipt,
            "prepared_windows_content_digest_algorithm": PREPARED_CONTENT_ALGORITHM,
        }
        bind_receipt_to_prepared_content(
            partial_modern,
            [synthetic_set],
            lambda arrays: synthetic_checks(arrays, partial_modern),
        )
    except RuntimeError:
        partial_modern_provenance_rejected = True
    check(
        "partial_modern_content_provenance_cannot_downgrade_to_legacy",
        partial_modern_provenance_rejected,
        "algorithm and combined digest are an indivisible provenance pair",
    )
    stripped_modern_receipt_rejected = False
    try:
        stripped_modern = {
            "input_ids_sha256": legacy_receipt["input_ids_sha256"]
        }
        bind_receipt_to_prepared_content(
            stripped_modern,
            [synthetic_set],
            lambda arrays: synthetic_checks(arrays, stripped_modern),
            require_declared_content=True,
        )
    except RuntimeError:
        stripped_modern_receipt_rejected = True
    check(
        "modern_receipt_cannot_strip_both_content_fields",
        stripped_modern_receipt_rejected,
        "modern schema requires the combined digest and algorithm",
    )
    derived_binding = bind_transitive_receipt_content(
        {}, [legacy_binding], dependency_kind="synthetic_dependency"
    )
    modern_transitive_receipt = content_fields_from_binding(derived_binding)
    modern_transitive_binding = bind_transitive_receipt_content(
        modern_transitive_receipt,
        [legacy_binding],
        dependency_kind="synthetic_dependency",
        require_declared_content=True,
    )
    stripped_transitive_rejected = False
    try:
        bind_transitive_receipt_content(
            {},
            [legacy_binding],
            dependency_kind="synthetic_dependency",
            require_declared_content=True,
        )
    except RuntimeError:
        stripped_transitive_rejected = True
    check(
        "transitive_content_is_derived_and_modern_receipts_fail_closed",
        modern_transitive_binding["content_sha256"] == combined["sha256"]
        and stripped_transitive_rejected,
        modern_transitive_binding,
    )
    duplicate_registry_rejected = False
    try:
        validate_content_set_registry(
            [
                synthetic_set,
                {
                    **synthetic_set,
                    "label": "different_lineage",
                    "content": {
                        **synthetic_set["content"],
                        "sha256": "f" * 64,
                    },
                },
            ]
        )
    except RuntimeError:
        duplicate_registry_rejected = True
    check(
        "legacy_archive_sha_maps_to_only_one_content_lineage",
        duplicate_registry_rejected,
        "duplicate legacy archive mapping rejected",
    )
    pair_fixture = {
        "prepared_windows_sha256": "a" * 64,
        "student_ids_sha256": sha256_array(digest_arrays["alpha"]),
        "teacher_shared_ids_sha256": sha256_array(digest_arrays["alpha"]),
        "teacher_prefix_ids_sha256": "d" * 64,
    }

    def pair_fixture_checks(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
        return [
            receipt_array_check(
                receipt_field="student_ids_sha256",
                recorded_sha256=pair_fixture["student_ids_sha256"],
                array_name="alpha",
                selection="[all]",
                value=arrays["alpha"],
            ),
            receipt_array_check(
                receipt_field="teacher_shared_ids_sha256",
                recorded_sha256=pair_fixture["teacher_shared_ids_sha256"],
                array_name="alpha",
                selection="[all]",
                value=arrays["alpha"],
            ),
            receipt_array_check(
                receipt_field="teacher_prefix_ids_sha256",
                recorded_sha256=pair_fixture["teacher_prefix_ids_sha256"],
                array_name="beta",
                selection="[all]",
                value=arrays["beta"],
            ),
        ]

    pair_prefix_mismatch_rejected = False
    try:
        bind_receipt_to_prepared_content(
            pair_fixture, [synthetic_set], pair_fixture_checks
        )
    except RuntimeError:
        pair_prefix_mismatch_rejected = True
    check(
        "legacy_pair_requires_student_shared_and_prefix_hashes",
        pair_prefix_mismatch_rejected,
        "prefix-only mismatch rejected",
    )

    synthetic: dict[str, np.ndarray] = {}
    direction = unit_vector(rng.standard_normal(dimension), name="fixture direction")[0]
    for corpus, shift in (("narrative", 1.0), ("wikitext", 0.0), ("code", -0.3), ("grm", -0.5)):
        values = rng.standard_normal((16, 8, dimension)).astype(np.float32) * 0.5
        values += shift * direction
        synthetic[corpus] = values
    fitted, metadata = fit_k4_direction(synthetic, cg_rtol=1.0e-6, cg_maxiter=100)
    scores = score_four(synthetic, fitted)
    metrics = analyze_key_scores(scores, resamples=max(200, args.bootstrap_resamples), seed=args.seed)
    mutated = {name: value.copy() for name, value in synthetic.items()}
    for value in mutated.values():
        value[np.asarray(EVAL_INDICES)] += rng.standard_normal(
            value[np.asarray(EVAL_INDICES)].shape
        ).astype(np.float32) * 100.0
    fitted_mutated, metadata_mutated = fit_k4_direction(
        mutated, cg_rtol=1.0e-6, cg_maxiter=100
    )
    tau_mutated = select_tau_four(score_four(mutated, fitted_mutated))["selected_tau"]
    check("k4_unit_norm", abs(float(np.linalg.norm(fitted)) - 1.0) < 1.0e-5, float(np.linalg.norm(fitted)))
    check("k4_eval_mutation_key_exact", np.array_equal(fitted, fitted_mutated), sha256_array(fitted_mutated))
    check(
        "k4_eval_mutation_alpha_exact",
        metadata["selected_alpha"] == metadata_mutated["selected_alpha"],
        [metadata["selected_alpha"], metadata_mutated["selected_alpha"]],
    )
    check(
        "k4_eval_mutation_tau_exact",
        metrics["selected_tau"] == tau_mutated,
        [metrics["selected_tau"], tau_mutated],
    )
    recovery = bootstrap_recovery(
        np.full(16, math.log(20.0)),
        np.full(16, math.log(10.0)),
        np.full(16, math.log(15.0)),
        resamples=max(200, args.bootstrap_resamples),
        seed=args.seed,
    )
    check("bootstrap_recovery_50pct", abs(recovery["recovery_percent"] - 50.0) < 1.0e-9, recovery)
    prose, prose_stats = prose_only_text(
        "# Header\n- bullet sentence should vanish.\nThis is a complete prose sentence with enough useful words.\n| a | b |\n"
    )
    check("p1_filter_removes_structures", "bullet" not in prose and "Header" not in prose, {"text": prose, **prose_stats})

    passed = all(item["passed"] for item in checks)
    receipt = {
        "schema": "moe_e2_cpu_validation_v1",
        "status": "passed" if passed else "failed",
        "created_at": now_iso(),
        "script": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "synthetic_only_not_gate_evidence": True,
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
    }
    write_json(output / "cpu_validation.json", receipt)
    print(json.dumps({"status": receipt["status"], "checks": len(checks)}), flush=True)
    return 0 if passed else 3


def key_capture_paths(output: Path, corpus: str, window_index: int) -> tuple[Path, Path]:
    root = output / "key_captures"
    stem = f"{corpus}_{window_index:02d}"
    return root / f"{stem}_router_inputs_fp16.npy", root / f"{stem}_receipt.json"


def validate_key_capture_receipt(
    output: Path,
    manifest: dict[str, Any],
    prepared: dict[str, np.ndarray],
    corpus: str,
    window_index: int,
    *,
    data_path: Path | None = None,
    receipt_path: Path | None = None,
    classification: str = "active",
    content_sets: Sequence[dict[str, Any]] | None = None,
    allow_unregistered_legacy: bool = False,
) -> dict[str, Any]:
    expected_data_path, expected_receipt_path = key_capture_paths(
        output, corpus, window_index
    )
    data_path = expected_data_path if data_path is None else data_path
    receipt_path = expected_receipt_path if receipt_path is None else receipt_path
    if not data_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"missing key capture payload/receipt: {receipt_path}")
    receipt = read_json(receipt_path)
    receipt_schema = receipt.get("schema")
    if (
        receipt_schema
        not in {
            "moe_e2_router_input_capture_v1",
            "moe_e2_router_input_capture_v2",
        }
        or receipt.get("status") != "complete"
        or receipt.get("corpus") != corpus
        or int(receipt.get("window_index", -1)) != window_index
        or receipt.get("model_revision") != MODEL_REVISION
    ):
        raise RuntimeError(f"key capture receipt contract failed: {receipt_path}")
    observed_file_sha = sha256_file(data_path)
    if receipt.get("capture_file_sha256") != observed_file_sha:
        raise RuntimeError(f"key capture file hash failed: {data_path}")
    payload = np.load(data_path, mmap_mode="r", allow_pickle=False)
    if payload.shape != (N_LAYERS, WINDOW_TOKENS, HIDDEN_DIM) or payload.dtype != np.float16:
        raise RuntimeError(f"key capture payload contract failed: {data_path}")
    array_name = "narrative_key_ids" if corpus == "narrative" else f"{corpus}_ids"
    content_sets = list(content_sets or load_prepared_content_sets(output, manifest, prepared))

    def checks(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
        return [
            receipt_array_check(
                receipt_field="input_ids_sha256",
                recorded_sha256=receipt.get("input_ids_sha256"),
                array_name=array_name,
                selection=f"[{window_index}]",
                value=arrays[array_name][window_index],
            )
        ]

    binding = bind_receipt_to_prepared_content(
        receipt,
        content_sets,
        checks,
        allow_unregistered_legacy=allow_unregistered_legacy,
        require_declared_content=(
            receipt_schema == "moe_e2_router_input_capture_v2"
        ),
    )
    recorded_capture_path = Path(receipt.get("capture_path", ""))
    relocated = recorded_capture_path.resolve() != data_path.resolve()
    if classification == "active" and relocated:
        raise RuntimeError(f"active key capture path linkage failed: {receipt_path}")
    return {
        "kind": "key_capture",
        "classification": classification,
        "status": "passed",
        "corpus": corpus,
        "window_index": window_index,
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "payload_path": str(data_path),
        "payload_sha256": observed_file_sha,
        "payload_shape": list(payload.shape),
        "payload_dtype": str(payload.dtype),
        "recorded_payload_path_relocated": relocated,
        "binding": binding,
        "receipt": receipt,
        "payload": payload,
    }


def capture_keys(args: argparse.Namespace) -> int:
    if args.corpus is None:
        raise ValueError("capture-keys requires --corpus")
    if args.start_window < 0 or args.count < 1 or args.start_window + args.count > N_KEY_WINDOWS:
        raise ValueError("capture-keys window range must stay within 0..15")
    output = ensure_output(args.output_dir)
    manifest, prepared = load_prepared(output)
    require_bringup_green(output, manifest, prepared)
    content_sets = load_prepared_content_sets(output, manifest, prepared)
    array_name = "narrative_key_ids" if args.corpus == "narrative" else f"{args.corpus}_ids"
    requested = list(range(args.start_window, args.start_window + args.count))
    missing: list[int] = []
    for index in requested:
        _data_path, receipt_path = key_capture_paths(output, args.corpus, index)
        if receipt_path.exists():
            validate_key_capture_receipt(
                output,
                manifest,
                prepared,
                args.corpus,
                index,
                content_sets=content_sets,
            )
            continue
        # A payload is written before its receipt; payload-only is recoverable scratch.
        missing.append(index)
    if not missing:
        print(json.dumps({"status": "existing", "corpus": args.corpus, "windows": requested}))
        return 0
    started = time.perf_counter()
    torch, model, runtime = load_model(args)
    captured: list[np.ndarray | None] = [None] * N_LAYERS
    handles: list[Any] = []

    def make_hook(layer_index: int):
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            value = inputs[0].detach().to(device="cpu", dtype=torch.float16).contiguous().numpy()
            captured[layer_index] = np.ascontiguousarray(value)

        return hook

    for layer_index, layer in enumerate(model.model.layers):
        handles.append(layer.mlp.register_forward_pre_hook(make_hook(layer_index)))
    ids_np = np.ascontiguousarray(prepared[array_name][missing], dtype=np.int64)
    ids = torch.from_numpy(ids_np).to(input_device(model))
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
        "peak_rss_bytes": peak_rss_bytes(),
    }
    for local_index, window_index in enumerate(missing):
        payload = np.ascontiguousarray(stacked[:, local_index], dtype=np.float16)
        data_path, receipt_path = key_capture_paths(output, args.corpus, window_index)
        save_npy(data_path, payload)
        receipt = {
            "schema": "moe_e2_router_input_capture_v2",
            "status": "complete",
            "created_at": now_iso(),
            "corpus": args.corpus,
            "window_index": window_index,
            "input_ids_sha256": sha256_array(prepared[array_name][window_index]),
            **prepared_content_fields(manifest),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "hook": "OlmoeDecoderLayer.mlp forward_pre_hook; exact native router input",
            "capture_shape": list(payload.shape),
            "capture_dtype": str(payload.dtype),
            "capture_payload_nbytes": int(payload.nbytes),
            "capture_path": str(data_path),
            "capture_file_sha256": sha256_file(data_path),
            "script": str(SCRIPT_PATH),
            "script_sha256": sha256_file(SCRIPT_PATH),
            "invocation": invocation,
        }
        write_json(receipt_path, receipt)
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


def verified_key_captures(
    output: Path, manifest: dict[str, Any], prepared: dict[str, np.ndarray]
) -> tuple[dict[str, list[np.ndarray]], list[dict[str, Any]]]:
    arrays: dict[str, list[np.ndarray]] = {corpus: [] for corpus in KEY_CORPORA}
    provenance: list[dict[str, Any]] = []
    content_sets = load_prepared_content_sets(output, manifest, prepared)
    for corpus in KEY_CORPORA:
        array_name = "narrative_key_ids" if corpus == "narrative" else f"{corpus}_ids"
        for window_index in range(N_KEY_WINDOWS):
            data_path, receipt_path = key_capture_paths(output, corpus, window_index)
            if not data_path.is_file() or not receipt_path.is_file():
                raise FileNotFoundError(
                    f"missing {corpus} key capture window {window_index}: {data_path}"
                )
            validation = validate_key_capture_receipt(
                output,
                manifest,
                prepared,
                corpus,
                window_index,
                content_sets=content_sets,
            )
            receipt = validation["receipt"]
            observed_hash = validation["payload_sha256"]
            array = validation["payload"]
            arrays[corpus].append(array)
            provenance.append(
                {
                    "corpus": corpus,
                    "window_index": window_index,
                    "capture_path": str(data_path),
                    "capture_sha256": observed_hash,
                    "receipt_path": str(receipt_path),
                    "receipt_sha256": sha256_file(receipt_path),
                    "prepared_content_binding": validation["binding"],
                }
            )
    return arrays, provenance


def materialize_key_layer(
    captures: dict[str, list[np.ndarray]], layer_index: int
) -> dict[str, np.ndarray]:
    hidden: dict[str, np.ndarray] = {}
    for corpus in KEY_CORPORA:
        combined = np.stack(
            [np.asarray(array[layer_index]) for array in captures[corpus]], axis=0
        )
        if combined.shape != (N_KEY_WINDOWS, WINDOW_TOKENS, HIDDEN_DIM):
            raise RuntimeError(f"{corpus} L{layer_index} shape {combined.shape}")
        if combined.dtype != np.float16 or not np.isfinite(combined).all():
            raise RuntimeError(f"{corpus} L{layer_index} contains invalid capture values")
        hidden[corpus] = np.ascontiguousarray(combined)
    return hidden


def declared_arrays_match_content(
    declared_arrays: Any, content_arrays: dict[str, Any]
) -> bool:
    if not isinstance(declared_arrays, dict) or set(declared_arrays) != set(
        content_arrays
    ):
        return False
    for name, expected in content_arrays.items():
        declared = declared_arrays[name]
        if not isinstance(declared, dict):
            return False
        if (
            declared.get("shape") != expected["shape"]
            or declared.get("sha256") != expected["sha256"]
            or (
                "dtype" in declared
                and declared.get("dtype") != expected["dtype"]
            )
        ):
            return False
    return True


def validate_expertpack_content_binding(
    output: Path,
    pack: dict[str, Any],
    dependency_binding: dict[str, Any],
    *,
    dependency_kind: str,
    dependency_receipt_path: Path,
    content_sets: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Recover mutable ExpertPack provenance through an immutable stage receipt."""

    if (
        pack.get("schema") != EXPERTPACK_NAME
        or pack.get("model_revision") != MODEL_REVISION
    ):
        raise RuntimeError("ExpertPack schema/model contract failed")
    if dependency_kind not in {"fit_key", "training"}:
        raise ValueError(f"unsupported ExpertPack dependency: {dependency_kind}")
    expected_dependency_path = output / (
        "fit_key.json" if dependency_kind == "fit_key" else "train.json"
    )
    if dependency_receipt_path.resolve() != expected_dependency_path.resolve():
        raise RuntimeError("ExpertPack dependency receipt path is outside its output")
    provenance = pack.get("provenance", {})
    if not isinstance(provenance, dict):
        raise RuntimeError("ExpertPack provenance record is missing")
    path_field = "fit_key" if dependency_kind == "fit_key" else "training"
    sha_field = f"{path_field}_sha256"
    if (
        provenance.get(path_field) != str(dependency_receipt_path)
        or provenance.get(sha_field) != sha256_file(dependency_receipt_path)
    ):
        raise RuntimeError(
            f"ExpertPack {dependency_kind} immutable receipt linkage changed"
        )

    native = provenance.get("prepared_windows_content")
    legacy = provenance.get("prepared_windows")
    declared: dict[str, Any]
    if native is not None:
        if not isinstance(native, dict):
            raise RuntimeError("ExpertPack canonical prepared provenance is malformed")
        if (
            native.get("content_digest_algorithm")
            != dependency_binding["content_digest_algorithm"]
            or native.get("content_sha256")
            != dependency_binding["content_sha256"]
            or native.get("arrays") != dependency_binding["arrays"]
        ):
            raise RuntimeError(
                "ExpertPack canonical content differs from its immutable dependency"
            )
        mode = "native_content_digest_transitively_revalidated"
        declared = {
            "kind": "canonical_content",
            "content_sha256": native["content_sha256"],
            "content_label": dependency_binding["content_label"],
            "conflicts_with_component_lineage": False,
        }
    else:
        if not isinstance(legacy, dict):
            raise RuntimeError("ExpertPack has no prepared-content provenance")
        matching_sets = [
            item
            for item in content_sets
            if declared_arrays_match_content(
                legacy.get("arrays"), item["content"]["arrays"]
            )
        ]
        if len(matching_sets) != 1:
            raise RuntimeError(
                "ExpertPack legacy prepared arrays do not identify one content lineage"
            )
        legacy_set = matching_sets[0]
        mode = "legacy_mutable_expertpack_transitive_migration"
        declared = {
            "kind": "legacy_npz_byte_and_array_record",
            "legacy_archive_sha256": legacy.get("sha256"),
            "content_sha256": legacy_set["content"]["sha256"],
            "content_label": legacy_set["label"],
            "conflicts_with_component_lineage": (
                legacy_set["content"]["sha256"]
                != dependency_binding["content_sha256"]
            ),
        }

    return {
        "status": "passed",
        "mode": mode,
        "content_label": dependency_binding["content_label"],
        "content_digest_algorithm": dependency_binding[
            "content_digest_algorithm"
        ],
        "content_sha256": dependency_binding["content_sha256"],
        "arrays": dependency_binding["arrays"],
        "content_metadata_path": dependency_binding.get("content_metadata_path"),
        "dependency_kind": dependency_kind,
        "dependency_receipt_path": str(dependency_receipt_path),
        "dependency_receipt_sha256": sha256_file(dependency_receipt_path),
        "expertpack_declared_prepared_provenance": declared,
        "original_receipt_mutated": False,
    }


def validate_fit_key_artifacts(
    output: Path, manifest: dict[str, Any], prepared: dict[str, np.ndarray]
) -> dict[str, Any]:
    fit_path = output / "fit_key.json"
    if not fit_path.is_file():
        raise FileNotFoundError(fit_path)
    receipt = read_json(fit_path)
    receipt_schema = receipt.get("schema")
    if (
        receipt_schema not in {"moe_e2_fit_key_v1", "moe_e2_fit_key_v2"}
        or receipt.get("status")
        not in {"complete_g2_green", "complete_g2_red_stop"}
        or receipt.get("model_revision") != MODEL_REVISION
    ):
        raise RuntimeError(f"fit-key receipt contract failed: {fit_path}")
    captures, observed_provenance = verified_key_captures(
        output, manifest, prepared
    )
    recorded_provenance = receipt.get("input_integrity", {}).get("captures", [])
    if len(recorded_provenance) != len(observed_provenance):
        raise RuntimeError("fit-key capture provenance count changed")
    for recorded, observed in zip(recorded_provenance, observed_provenance):
        for key in (
            "corpus",
            "window_index",
            "capture_path",
            "capture_sha256",
            "receipt_path",
            "receipt_sha256",
        ):
            if recorded.get(key) != observed.get(key):
                raise RuntimeError(f"fit-key capture linkage changed: {key}")
    dependency_bindings = [
        item["prepared_content_binding"] for item in observed_provenance
    ]
    binding = bind_transitive_receipt_content(
        receipt,
        dependency_bindings,
        dependency_kind="key_captures",
        require_declared_content=receipt_schema == "moe_e2_fit_key_v2",
    )
    keys_record = receipt.get("keys_artifact", {})
    keys_path = Path(keys_record.get("path", ""))
    if not keys_path.is_file() or keys_record.get("sha256") != sha256_file(keys_path):
        raise RuntimeError("fit-key all-layer key artifact changed")
    with np.load(keys_path, allow_pickle=False) as archive:
        keys_arrays = {name: archive[name].copy() for name in archive.files}
    if (
        set(keys_arrays)
        != {"K4", "tau", "selected_alpha", "selected_lambda"}
        or keys_arrays["K4"].shape != (N_LAYERS, HIDDEN_DIM)
        or keys_arrays["K4"].dtype != np.float32
    ):
        raise RuntimeError("fit-key all-layer key payload contract failed")
    pack_path = output / EXPERTPACK_NAME / "manifest.json"
    pack = read_json(pack_path)
    decision = receipt.get("decision", {})
    install_layer = decision.get("install_layer")
    if install_layer is not None:
        if int(pack.get("layer", -1)) != int(install_layer):
            raise RuntimeError("fit-key install layer differs from ExpertPack")
        key_record = pack.get("components", {}).get("key", {})
        key_path = Path(key_record.get("path", ""))
        if not key_path.is_file() or key_record.get("sha256") != sha256_file(key_path):
            raise RuntimeError("fit-key selected key artifact changed")
    pack_binding = validate_expertpack_content_binding(
        output,
        pack,
        binding,
        dependency_kind="fit_key",
        dependency_receipt_path=fit_path,
        content_sets=load_prepared_content_sets(output, manifest, prepared),
    )
    del captures
    return {
        "kind": "fit_key",
        "status": "passed",
        "receipt_path": str(fit_path),
        "receipt_sha256": sha256_file(fit_path),
        "capture_count": len(observed_provenance),
        "keys_artifact_path": str(keys_path),
        "keys_artifact_sha256": sha256_file(keys_path),
        "verdict": receipt.get("decision", {}).get("verdict"),
        "binding": binding,
        "expertpack_binding": pack_binding,
        "receipt": receipt,
    }


def fit_side_layer_rank(row: dict[str, Any]) -> tuple[float, ...]:
    fit = row["metrics"]["selected_fit_rates"]
    validation = row["key_metadata"]["selected_validation"]
    # WikiText remains descriptive at E2-G2.  Its carried 0.02 scale appears
    # only in the already-registered FIT-side tie/rank, never qualification.
    worst_normalized_rate = max(
        fit["wikitext_fpr"] / GENERIC_FIT_RANK_REFERENCE,
        fit["code_fpr"] / FPR_CAPS["code"],
        fit["grm_fpr"] / FPR_CAPS["grm"],
    )
    return (
        float(row["metrics"]["fit_fpr_feasible"]),
        float(validation["minimum_auc"]),
        float(validation["mean_auc"]),
        float(fit["narrative_recall"]),
        -float(worst_normalized_rate),
        -float(row["layer"]),
    )


def fit_key(args: argparse.Namespace) -> int:
    if args.bootstrap_resamples < 2000:
        raise ValueError("fit-key requires at least 2,000 window-bootstrap resamples")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.rank != 64:
        raise ValueError("MOE-E2 primary rank is frozen at r=64")
    output = ensure_output(args.output_dir)
    manifest, prepared = load_prepared(output)
    require_bringup_green(output, manifest, prepared)
    existing_fit_path = output / "fit_key.json"
    if existing_fit_path.is_file():
        validation = validate_fit_key_artifacts(output, manifest, prepared)
        print(
            json.dumps(
                {
                    "status": "existing",
                    "gate": "E2-G2",
                    "verdict": validation["verdict"],
                    "receipt": str(existing_fit_path),
                    "content_sha256": validation["binding"]["content_sha256"],
                }
            ),
            flush=True,
        )
        return 0 if validation["verdict"] == "GREEN" else 3
    started = time.perf_counter()
    captures, capture_provenance = verified_key_captures(output, manifest, prepared)
    input_binding = bind_transitive_receipt_content(
        {},
        [item["prepared_content_binding"] for item in capture_provenance],
        dependency_kind="key_captures",
    )
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
            hidden = materialize_key_layer(captures, layer_index)
            key, key_metadata = fit_k4_direction(
                hidden, cg_rtol=args.cg_rtol, cg_maxiter=args.cg_maxiter
            )
            scores = score_four(hidden, key)
            metrics = analyze_key_scores(
                scores,
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
                        "recall": metrics["eval"]["narrative_recall"]["value"],
                        "generic_fire_rate": metrics["eval"]["wikitext_fpr"]["value"],
                        "code_fpr": metrics["eval"]["code_fpr"]["value"],
                        "grm_fpr": metrics["eval"]["grm_fpr"]["value"],
                    }
                ),
                flush=True,
            )
            del hidden, scores
            gc.collect()

    keys_path = output / "all_layer_k4_keys.npz"
    save_npz(
        keys_path,
        K4=np.stack(keys).astype(np.float32),
        tau=np.asarray(taus, dtype=np.float64),
        selected_alpha=np.asarray(alphas, dtype=np.float64),
        selected_lambda=np.asarray(lambdas, dtype=np.float64),
    )
    qualifiers = [row for row in rows if row["metrics"]["qualifies"]]
    qualifiers_ranked = sorted(qualifiers, key=fit_side_layer_rank, reverse=True)
    selected = qualifiers_ranked[0] if qualifiers_ranked else None
    decision = {
        "verdict": "GREEN" if selected is not None else "RED",
        "qualifying_layer_count": len(qualifiers_ranked),
        "qualifying_layers_fit_ranked": [int(row["layer"]) for row in qualifiers_ranked],
        "install_layer": None if selected is None else int(selected["layer"]),
        "selection_uses_eval_only_as_registered_qualification_filter": True,
        "within_qualifiers_selection_rule": (
            "FIT-only carried rank: code+GRM feasible threshold, inner-validation "
            "minimum AUC, mean AUC, FIT recall, lower worst normalized FIT rate, lower layer"
        ),
    }
    analysis = {
        "schema": "moe_e2_fit_key_v2",
        "status": "complete_g2_green" if selected else "complete_g2_red_stop",
        "created_at": now_iso(),
        "gate": "E2-G2",
        "order": str(ORDER_PATH),
        "script": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        **content_fields_from_binding(input_binding),
        "seed": args.seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "input_integrity": {
            "capture_count": len(capture_provenance),
            "expected_capture_count": len(KEY_CORPORA) * N_KEY_WINDOWS,
            "captures": capture_provenance,
        },
        "split_provenance": {
            "fit_window_indices": list(FIT_INDICES),
            "eval_window_indices": list(EVAL_INDICES),
            "inner_train_window_indices": list(INNER_TRAIN_INDICES),
            "inner_validation_window_indices": list(INNER_VALIDATION_INDICES),
            "fit_eval_overlap": [],
            "eval_used_for_key_tau_or_hyperparameter_selection": False,
            "heldout_file_text_used": False,
        },
        "registered_rule": {
            "eval_recall_gte": RECALL_FLOOR,
            "eval_code_fpr_lte": FPR_CAPS["code"],
            "eval_grm_fpr_lte": FPR_CAPS["grm"],
            "generic_wikitext_fire_rate": "descriptive_per_layer_and_eval_window",
            "fit_feasibility_constraints": ["code", "grm"],
            "tau_candidate_pool": list(NEGATIVE_CORPORA),
            "frozen_fit_side_tau": True,
        },
        "rows": rows,
        "decision": decision,
        "keys_artifact": {"path": str(keys_path), "sha256": sha256_file(keys_path)},
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "threads": args.threads,
            "cg_rtol": args.cg_rtol,
            "cg_maxiter": args.cg_maxiter,
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_bytes": peak_rss_bytes(),
        },
    }
    fit_path = output / "fit_key.json"
    write_json(fit_path, analysis)

    pack_path = output / EXPERTPACK_NAME / "manifest.json"
    pack = read_json(pack_path)
    pack["updated_at"] = now_iso()
    pack["provenance"]["fit_key"] = str(fit_path)
    pack["provenance"]["fit_key_sha256"] = sha256_file(fit_path)
    pack["provenance"]["all_layer_keys"] = str(keys_path)
    pack["provenance"]["all_layer_keys_sha256"] = sha256_file(keys_path)
    pack["provenance"]["prepared_windows_content"] = (
        content_provenance_from_binding(input_binding)
    )
    pack["provenance"].pop("prepared_windows", None)
    if selected is None:
        pack["status"] = "g2_red_no_installable_address"
        pack["limitations"] = ["E2-G2 RED: no qualifying K4 narrative address; STOP"]
    else:
        layer_index = int(selected["layer"])
        selected_key = np.ascontiguousarray(keys[layer_index], dtype=np.float32)
        key_path = output / EXPERTPACK_NAME / "key_fp32.npy"
        save_npy(key_path, selected_key)
        evaluation = selected["metrics"]["eval"]
        pack.update(
            {
                "status": "addressed_pending_pair_capture",
                "layer": layer_index,
                "rank": args.rank,
                "tau": float(selected["metrics"]["selected_tau"]),
                "key_metrics": {
                    "eval_recall": evaluation["narrative_recall"]["value"],
                    "eval_generic_fire_rate_descriptive": evaluation["wikitext_fpr"]["value"],
                    "eval_code_fpr": evaluation["code_fpr"]["value"],
                    "eval_grm_fpr": evaluation["grm_fpr"]["value"],
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
            "sha256": sha256_file(key_path),
        }
        pack["limitations"] = ["adapter A/B pending capture-pairs and train"]
    write_json(pack_path, pack)
    print(
        json.dumps(
            {
                "gate": "E2-G2",
                "verdict": decision["verdict"],
                "qualifying_layers": decision["qualifying_layers_fit_ranked"],
                "install_layer": decision["install_layer"],
                "fit_key": str(fit_path),
            }
        ),
        flush=True,
    )
    return 0 if selected is not None else 3


def load_addressed_pack(output: Path, *, require_adapter: bool = False) -> dict[str, Any]:
    path = output / EXPERTPACK_NAME / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing ExpertPack manifest: {path}")
    pack = read_json(path)
    if (
        pack.get("schema") != EXPERTPACK_NAME
        or pack.get("model_revision") != MODEL_REVISION
    ):
        raise RuntimeError("ExpertPack schema/model contract failed")
    if pack.get("layer") is None or pack.get("tau") is None:
        raise RuntimeError("E2-G2 did not produce an installable address; STOP")
    key_record = pack.get("components", {}).get("key", {})
    key_path = Path(key_record.get("path", ""))
    if not key_path.is_file() or key_record.get("sha256") != sha256_file(key_path):
        raise RuntimeError("ExpertPack key is missing or changed")
    if require_adapter:
        for name in ("A", "B"):
            record = pack.get("components", {}).get(name, {})
            component_path = Path(record.get("path", ""))
            if not component_path.is_file() or record.get("sha256") != sha256_file(component_path):
                raise RuntimeError(f"ExpertPack {name} is missing or changed")
    return pack


def pair_paths(output: Path, window_index: int) -> tuple[Path, Path]:
    root = output / "pairs"
    return root / f"pair_{window_index:03d}.npz", root / f"pair_{window_index:03d}_receipt.json"


def validate_pair_capture_receipt(
    output: Path,
    manifest: dict[str, Any],
    prepared: dict[str, np.ndarray],
    window_index: int,
    *,
    content_sets: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    data_path, receipt_path = pair_paths(output, window_index)
    if not data_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"missing pair capture {window_index}")
    receipt = read_json(receipt_path)
    receipt_schema = receipt.get("schema")
    expected_split = "TRAIN" if window_index < N_PAIR_TRAIN else "VALIDATION"
    if (
        receipt_schema
        not in {
            "moe_e2_teacher_student_pair_v1",
            "moe_e2_teacher_student_pair_v2",
        }
        or receipt.get("status") != "complete"
        or int(receipt.get("pair_index", -1)) != window_index
        or receipt.get("split") != expected_split
        or receipt.get("model_revision") != MODEL_REVISION
        or not receipt.get("shared_window_token_ids_exact")
    ):
        raise RuntimeError(f"pair receipt contract failed: {receipt_path}")
    if Path(receipt.get("pair_path", "")).resolve() != data_path.resolve():
        raise RuntimeError(f"pair payload path linkage failed: {receipt_path}")
    observed_file_sha = sha256_file(data_path)
    if receipt.get("pair_file_sha256") != observed_file_sha:
        raise RuntimeError(f"pair file hash failed: {data_path}")
    with np.load(data_path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    expected_names = {
        "h_student_fp16",
        "out_student_fp16",
        "out_teacher_fp16",
    }
    if set(data) != expected_names:
        raise RuntimeError(f"pair payload keys failed: {data_path}")
    for name, value in data.items():
        if value.shape != (WINDOW_TOKENS, HIDDEN_DIM) or value.dtype != np.float16:
            raise RuntimeError(f"pair {window_index} {name} contract failed")
    payload_hash_fields = {
        "h_student_fp16": "student_router_input_sha256",
        "out_student_fp16": "student_block_output_sha256",
        "out_teacher_fp16": "teacher_block_output_sha256",
    }
    payload_checks = [
        {
            "array": name,
            "receipt_field": field,
            "recorded_sha256": receipt.get(field),
            "observed_sha256": sha256_array(data[name]),
            "passed": receipt.get(field) == sha256_array(data[name]),
        }
        for name, field in payload_hash_fields.items()
    ]
    if not all(item["passed"] for item in payload_checks):
        raise RuntimeError(f"pair payload content hashes failed: {receipt_path}")
    content_sets = list(content_sets or load_prepared_content_sets(output, manifest, prepared))

    def checks(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
        student = arrays["pair_ids"][window_index]
        prefix = arrays["pair_p0_prefix_ids"][window_index]
        return [
            receipt_array_check(
                receipt_field="student_ids_sha256",
                recorded_sha256=receipt.get("student_ids_sha256"),
                array_name="pair_ids",
                selection=f"[{window_index}]",
                value=student,
            ),
            receipt_array_check(
                receipt_field="teacher_shared_ids_sha256",
                recorded_sha256=receipt.get("teacher_shared_ids_sha256"),
                array_name="pair_ids",
                selection=f"[{window_index}]",
                value=student,
            ),
            receipt_array_check(
                receipt_field="teacher_prefix_ids_sha256",
                recorded_sha256=receipt.get("teacher_prefix_ids_sha256"),
                array_name="pair_p0_prefix_ids",
                selection=f"[{window_index}]",
                value=prefix,
            ),
        ]

    binding = bind_receipt_to_prepared_content(
        receipt,
        content_sets,
        checks,
        require_declared_content=(
            receipt_schema == "moe_e2_teacher_student_pair_v2"
        ),
    )
    return {
        "kind": "pair_capture",
        "classification": "active",
        "status": "passed",
        "pair_index": window_index,
        "split": expected_split,
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "payload_path": str(data_path),
        "payload_sha256": observed_file_sha,
        "payload_checks": payload_checks,
        "binding": binding,
        "receipt": receipt,
        "data": data,
    }


def capture_pairs(args: argparse.Namespace) -> int:
    if args.start_window < 0 or args.count < 1 or args.start_window + args.count > N_PAIR_WINDOWS:
        raise ValueError("capture-pairs window range must stay within 0..63")
    output = ensure_output(args.output_dir)
    manifest, prepared = load_prepared(output)
    require_bringup_green(output, manifest, prepared)
    content_sets = load_prepared_content_sets(output, manifest, prepared)
    fit_validation = validate_fit_key_artifacts(output, manifest, prepared)
    fit_content_sets = [
        item
        for item in content_sets
        if item["content"]["sha256"]
        == fit_validation["binding"]["content_sha256"]
    ]
    if len(fit_content_sets) != 1:
        raise RuntimeError("fit-key content lineage is not uniquely registered")
    pair_prepared = fit_content_sets[0]["arrays"]
    pack = load_addressed_pack(output)
    requested = list(range(args.start_window, args.start_window + args.count))
    missing: list[int] = []
    for index in requested:
        _data_path, receipt_path = pair_paths(output, index)
        if receipt_path.exists():
            validation = validate_pair_capture_receipt(
                output,
                manifest,
                prepared,
                index,
                content_sets=content_sets,
            )
            bind_transitive_receipt_content(
                {},
                [fit_validation["binding"], validation["binding"]],
                dependency_kind="fit_key_and_pair_capture",
            )
            continue
        # A payload is written before its receipt; payload-only is recoverable scratch.
        missing.append(index)
    if not missing:
        print(json.dumps({"status": "existing", "pairs": requested}))
        return 0
    started = time.perf_counter()
    torch, model, runtime = load_model(args)
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
            input_device(model)
        )
        with torch.inference_mode():
            output_object = model(input_ids=ids, use_cache=False, logits_to_keep=1)
        del output_object, ids
        if set(slot) != {"h", "out"}:
            raise RuntimeError(f"pair hook did not execute exactly: {slot.keys()}")
        return slot["h"][0].copy(), slot["out"][0].copy()

    try:
        for window_index in missing:
            window_started = time.perf_counter()
            student_ids = pair_prepared["pair_ids"][window_index]
            prefix_ids = pair_prepared["pair_p0_prefix_ids"][window_index]
            teacher_ids = np.concatenate([prefix_ids, student_ids])
            if teacher_ids.size != PREFIX_TOKENS + WINDOW_TOKENS or teacher_ids.size > MAX_CONTEXT:
                raise RuntimeError("teacher context length contract failed")
            if not np.array_equal(teacher_ids[-WINDOW_TOKENS:], student_ids):
                raise RuntimeError("G1 shared-window token alignment failed")
            h_student, out_student = one_forward(student_ids)
            h_teacher_full, out_teacher_full = one_forward(teacher_ids)
            h_teacher = np.ascontiguousarray(h_teacher_full[-WINDOW_TOKENS:])
            out_teacher = np.ascontiguousarray(out_teacher_full[-WINDOW_TOKENS:])
            if h_student.shape != (WINDOW_TOKENS, HIDDEN_DIM) or out_student.shape != h_student.shape or out_teacher.shape != h_student.shape:
                raise RuntimeError("pair activation shapes failed")
            delta = out_teacher.astype(np.float32) - out_student.astype(np.float32)
            hidden_difference = h_teacher.astype(np.float32) - h_student.astype(np.float32)
            if not np.isfinite(delta).all() or not np.isfinite(hidden_difference).all():
                raise RuntimeError("pair capture contains non-finite values")
            data_path, receipt_path = pair_paths(output, window_index)
            save_npz(
                data_path,
                h_student_fp16=h_student,
                out_student_fp16=out_student,
                out_teacher_fp16=out_teacher,
            )
            split = "TRAIN" if window_index < N_PAIR_TRAIN else "VALIDATION"
            receipt = {
                "schema": "moe_e2_teacher_student_pair_v2",
                "status": "complete",
                "created_at": now_iso(),
                "pair_index": window_index,
                "split": split,
                "install_layer": layer_index,
                "teacher_prefix_arm": "P0 raw excerpt rotation",
                "student_tokens": WINDOW_TOKENS,
                "teacher_prefix_tokens": PREFIX_TOKENS,
                "teacher_total_tokens": int(teacher_ids.size),
                "shared_window_token_ids_exact": True,
                "student_ids_sha256": sha256_array(student_ids),
                "teacher_shared_ids_sha256": sha256_array(teacher_ids[-WINDOW_TOKENS:]),
                "teacher_prefix_ids_sha256": sha256_array(prefix_ids),
                "student_router_input_sha256": sha256_array(h_student),
                "teacher_router_input_sha256": sha256_array(h_teacher),
                "student_block_output_sha256": sha256_array(out_student),
                "teacher_block_output_sha256": sha256_array(out_teacher),
                "router_input_hidden_states_differ": bool(np.any(hidden_difference != 0)),
                "router_input_difference_mean_abs": float(np.abs(hidden_difference).mean()),
                "target_delta_zero_predictor_mse": float(np.mean(delta.astype(np.float64) ** 2)),
                "target_delta_mean_abs": float(np.abs(delta.astype(np.float64)).mean()),
                "target_delta_max_abs": float(np.abs(delta.astype(np.float64)).max()),
                "pair_path": str(data_path),
                "pair_file_sha256": sha256_file(data_path),
                **content_fields_from_binding(fit_validation["binding"]),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "runtime": runtime,
                "window_wall_seconds": time.perf_counter() - window_started,
                "script": str(SCRIPT_PATH),
                "script_sha256": sha256_file(SCRIPT_PATH),
            }
            write_json(receipt_path, receipt)
            print(
                json.dumps(
                    {
                        "pair": window_index,
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
                "peak_rss_bytes": peak_rss_bytes(),
            }
        ),
        flush=True,
    )
    del model
    gc.collect()
    return 0


def validate_training_artifacts(
    output: Path, manifest: dict[str, Any], prepared: dict[str, np.ndarray]
) -> dict[str, Any]:
    training_path = output / "train.json"
    if not training_path.is_file():
        raise FileNotFoundError(training_path)
    receipt = read_json(training_path)
    receipt_schema = receipt.get("schema")
    if (
        receipt_schema
        not in {
            "moe_e2_adapter_training_v1",
            "moe_e2_adapter_training_v2",
        }
        or receipt.get("status")
        not in {"complete_g3_green", "complete_g3_red_stop"}
        or receipt.get("model_revision") != MODEL_REVISION
        or receipt.get("verdict") not in {"GREEN", "RED"}
    ):
        raise RuntimeError(f"training receipt contract failed: {training_path}")
    content_sets = load_prepared_content_sets(output, manifest, prepared)
    fit_validation = validate_fit_key_artifacts(output, manifest, prepared)
    recorded_pairs = receipt.get("data", {}).get("pair_receipts", [])
    if len(recorded_pairs) != N_PAIR_WINDOWS:
        raise RuntimeError("training receipt does not bind all 64 pairs")
    pair_validations: list[dict[str, Any]] = []
    for index in range(N_PAIR_WINDOWS):
        validation = validate_pair_capture_receipt(
            output,
            manifest,
            prepared,
            index,
            content_sets=content_sets,
        )
        recorded = recorded_pairs[index]
        expected_receipt_path = pair_paths(output, index)[1]
        expected = {
            "pair_index": index,
            "receipt_path": str(expected_receipt_path),
            "receipt_sha256": sha256_file(expected_receipt_path),
            "pair_sha256": validation["payload_sha256"],
            "alignment": True,
        }
        if any(recorded.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"training pair linkage changed: {index}")
        pair_validations.append(validation)
    binding = bind_transitive_receipt_content(
        receipt,
        [fit_validation["binding"]]
        + [item["binding"] for item in pair_validations],
        dependency_kind="fit_key_and_pair_captures",
        require_declared_content=(
            receipt_schema == "moe_e2_adapter_training_v2"
        ),
    )
    artifact_results: dict[str, Any] = {}
    for name in ("A", "B"):
        record = receipt.get("artifacts", {}).get(name, {})
        path = Path(record.get("path", ""))
        if not path.is_file() or record.get("sha256") != sha256_file(path):
            raise RuntimeError(f"training {name} artifact changed")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_shape = (
            (int(receipt["rank"]), HIDDEN_DIM)
            if name == "A"
            else (HIDDEN_DIM, int(receipt["rank"]))
        )
        if array.shape != expected_shape or array.dtype != np.float16:
            raise RuntimeError(f"training {name} payload contract failed")
        artifact_results[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
    pack = load_addressed_pack(output, require_adapter=True)
    if (
        int(receipt.get("install_layer", -1)) != int(pack["layer"])
        or int(receipt.get("rank", -1)) != int(pack["rank"])
        or float(receipt.get("tau", math.nan)) != float(pack["tau"])
    ):
        raise RuntimeError("training key/layer/rank contract differs from ExpertPack")
    if receipt_schema == "moe_e2_adapter_training_v2":
        fit_link = receipt.get("data", {}).get("fit_key_receipt", {})
        fit_path = output / "fit_key.json"
        if (
            fit_link.get("path") != str(fit_path)
            or fit_link.get("sha256") != sha256_file(fit_path)
        ):
            raise RuntimeError("training fit-key receipt linkage changed")
    for name in ("A", "B"):
        if pack["components"][name].get("sha256") != artifact_results[name]["sha256"]:
            raise RuntimeError(f"ExpertPack {name} differs from training receipt")
    if pack.get("g3", {}).get("verdict") != receipt.get("verdict"):
        raise RuntimeError("ExpertPack G3 verdict differs from training receipt")
    pack_binding = validate_expertpack_content_binding(
        output,
        pack,
        binding,
        dependency_kind="training",
        dependency_receipt_path=training_path,
        content_sets=content_sets,
    )
    return {
        "kind": "train",
        "status": "passed",
        "receipt_path": str(training_path),
        "receipt_sha256": sha256_file(training_path),
        "verdict": receipt["verdict"],
        "pair_count": len(pair_validations),
        "pair_validations": pair_validations,
        "artifacts": artifact_results,
        "binding": binding,
        "expertpack_binding": pack_binding,
        "receipt": receipt,
    }


def train(args: argparse.Namespace) -> int:
    if args.rank != 64:
        raise ValueError("MOE-E2 rank is registered at r=64")
    if not 1 <= args.train_tokens_per_window <= WINDOW_TOKENS:
        raise ValueError("--train-tokens-per-window must be within 1..512")
    if args.threads < 1 or args.token_batch_size < 1 or args.epochs < 1 or args.patience < 1:
        raise ValueError("training resource/epoch arguments must be positive")
    output = ensure_output(args.output_dir)
    manifest, prepared = load_prepared(output)
    require_bringup_green(output, manifest, prepared)
    content_sets = load_prepared_content_sets(output, manifest, prepared)
    fit_validation = validate_fit_key_artifacts(output, manifest, prepared)
    pack = load_addressed_pack(output)
    existing_training_path = output / "train.json"
    if existing_training_path.is_file():
        validation = validate_training_artifacts(output, manifest, prepared)
        print(
            json.dumps(
                {
                    "status": "existing",
                    "gate": "E2-G3",
                    "verdict": validation["verdict"],
                    "receipt": str(existing_training_path),
                    "content_sha256": validation["binding"]["content_sha256"],
                }
            ),
            flush=True,
        )
        return 0 if validation["verdict"] == "GREEN" else 3
    started = time.perf_counter()
    torch = configure_torch(args.threads, cuda_possible=False)
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

    pair_receipts: list[dict[str, Any]] = []
    pair_bindings: list[dict[str, Any]] = []
    validated_pair_data: list[dict[str, np.ndarray]] = []
    for index in range(N_PAIR_WINDOWS):
        validation = validate_pair_capture_receipt(
            output,
            manifest,
            prepared,
            index,
            content_sets=content_sets,
        )
        receipt = validation["receipt"]
        pair_bindings.append(validation["binding"])
        validated_pair_data.append(validation["data"])
        pair_receipts.append(
            {
                "pair_index": index,
                "receipt_path": str(pair_paths(output, index)[1]),
                "receipt_sha256": sha256_file(pair_paths(output, index)[1]),
                "pair_sha256": receipt["pair_file_sha256"],
                "alignment": receipt["shared_window_token_ids_exact"],
            }
        )
        del validation
    input_binding = bind_transitive_receipt_content(
        {},
        [fit_validation["binding"]] + pair_bindings,
        dependency_kind="fit_key_and_pair_captures",
    )

    def evaluate_validation(model: Any) -> tuple[float, float, int, int]:
        model.eval()
        squared = 0.0
        zero_squared = 0.0
        elements = 0
        fired_count = 0
        with torch.no_grad():
            for index in range(N_PAIR_TRAIN, N_PAIR_WINDOWS):
                data = validated_pair_data[index]
                hidden = torch.from_numpy(data["h_student_fp16"].astype(np.float32))
                target = torch.from_numpy(
                    data["out_teacher_fp16"].astype(np.float32)
                    - data["out_student_fp16"].astype(np.float32)
                )
                for start in range(0, WINDOW_TOKENS, args.token_batch_size):
                    h_batch = hidden[start : start + args.token_batch_size]
                    target_batch = target[start : start + args.token_batch_size]
                    fire = (h_batch @ key_tensor >= tau).to(torch.float32)[:, None]
                    prediction = model(h_batch) * fire
                    squared += float(torch.sum((prediction - target_batch) ** 2).item())
                    zero_squared += float(torch.sum(target_batch**2).item())
                    elements += int(target_batch.numel())
                    fired_count += int(fire.sum().item())
                del hidden, target
        return squared / elements, zero_squared / elements, fired_count, elements // HIDDEN_DIM

    zero_validation_mse, zero_again, _initial_fires, validation_tokens = evaluate_validation(adapter)
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
            data = validated_pair_data[index]
            hidden_np = data["h_student_fp16"].astype(np.float32)
            target_np = (
                data["out_teacher_fp16"].astype(np.float32)
                - data["out_student_fp16"].astype(np.float32)
            )
            if args.train_tokens_per_window < WINDOW_TOKENS:
                token_indices = np.sort(
                    rng.choice(
                        WINDOW_TOKENS,
                        size=args.train_tokens_per_window,
                        replace=False,
                    )
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
        validation_mse, _zero_mse, validation_fires, _tokens = evaluate_validation(adapter)
        improved = validation_mse < best_mse - max(1.0e-12, abs(best_mse) * 1.0e-6)
        if improved:
            best_mse = validation_mse
            best_epoch = epoch
            best_state = {
                name: tensor.detach().clone() for name, tensor in adapter.state_dict().items()
            }
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
    save_npy(A_path, A_fp16)
    save_npy(B_path, B_fp16)
    with torch.no_grad():
        adapter.A.copy_(torch.from_numpy(A_fp16.astype(np.float32)))
        adapter.B.copy_(torch.from_numpy(B_fp16.astype(np.float32)))
    stored_mse, stored_zero_mse, stored_fires, _tokens = evaluate_validation(adapter)
    improvement = (
        (stored_zero_mse - stored_mse) / stored_zero_mse
        if stored_zero_mse > 0
        else float("-inf")
    )
    green = bool(np.isfinite(improvement) and improvement >= G3_IMPROVEMENT_FLOOR)
    training = {
        "schema": "moe_e2_adapter_training_v2",
        "status": "complete_g3_green" if green else "complete_g3_red_stop",
        "created_at": now_iso(),
        "gate": "E2-G3",
        "verdict": "GREEN" if green else "RED",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        **content_fields_from_binding(input_binding),
        "install_layer": int(pack["layer"]),
        "rank": args.rank,
        "tau": tau,
        "abi": "B @ silu(A @ h), threshold gated by h @ key >= tau",
        "initialization": {
            "A": "torch kaiming_uniform, deterministic CPU seed",
            "B": "exact all-zero float32",
            "B_zero_exact": True,
            "B_initial_sha256": sha256_array(initial_B),
        },
        "data": {
            "fit_key_receipt": {
                "path": str(output / "fit_key.json"),
                "sha256": sha256_file(output / "fit_key.json"),
            },
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
            "improvement_percent": improvement * 100.0,
            "required_improvement_fraction": G3_IMPROVEMENT_FLOOR,
        },
        "artifacts": {
            "A": {"path": str(A_path), "shape": list(A_fp16.shape), "dtype": "float16", "sha256": sha256_file(A_path)},
            "B": {"path": str(B_path), "shape": list(B_fp16.shape), "dtype": "float16", "sha256": sha256_file(B_path)},
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "threads": args.threads,
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_bytes": peak_rss_bytes(),
        },
        "script": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
    }
    training_path = output / "train.json"
    write_json(training_path, training)
    pack = read_json(pack_dir / "manifest.json")
    pack["updated_at"] = now_iso()
    pack["components"]["A"] = {
        "status": "complete",
        "path": str(A_path),
        "shape": list(A_fp16.shape),
        "storage_dtype": "float16",
        "sha256": sha256_file(A_path),
    }
    pack["components"]["B"] = {
        "status": "complete",
        "path": str(B_path),
        "shape": list(B_fp16.shape),
        "storage_dtype": "float16",
        "initialization": "exact zeros before training",
        "initial_zero_sha256": sha256_array(initial_B),
        "sha256": sha256_file(B_path),
    }
    pack["provenance"]["training"] = str(training_path)
    pack["provenance"]["training_sha256"] = sha256_file(training_path)
    pack["provenance"]["prepared_windows_content"] = (
        content_provenance_from_binding(input_binding)
    )
    pack["provenance"].pop("prepared_windows", None)
    pack["g3"] = training["stored_fp16_validation"] | {"verdict": training["verdict"]}
    pack["status"] = "complete_g3_green" if green else "g3_red_not_installable"
    pack["limitations"] = (
        ["behavioral E2-G0/G4/G5 evaluation pending"]
        if green
        else ["E2-G3 RED: stored adapter failed the 10 percent improvement gate; STOP"]
    )
    write_json(pack_dir / "manifest.json", pack)
    print(
        json.dumps(
            {
                "gate": "E2-G3",
                "verdict": training["verdict"],
                "improvement_percent": improvement * 100.0,
                "training": str(training_path),
            }
        ),
        flush=True,
    )
    return 0 if green else 3


def load_expert_arrays(
    output: Path,
    manifest: dict[str, Any],
    prepared: dict[str, np.ndarray],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    training_validation = validate_training_artifacts(output, manifest, prepared)
    pack = load_addressed_pack(output, require_adapter=True)
    key = np.load(pack["components"]["key"]["path"], allow_pickle=False)
    A = np.load(pack["components"]["A"]["path"], allow_pickle=False)
    B = np.load(pack["components"]["B"]["path"], allow_pickle=False)
    rank = int(pack["rank"])
    if key.shape != (HIDDEN_DIM,) or key.dtype != np.float32:
        raise RuntimeError(f"key contract failed: {key.shape}/{key.dtype}")
    if A.shape != (rank, HIDDEN_DIM) or A.dtype != np.float16:
        raise RuntimeError(f"A contract failed: {A.shape}/{A.dtype}")
    if B.shape != (HIDDEN_DIM, rank) or B.dtype != np.float16:
        raise RuntimeError(f"B contract failed: {B.shape}/{B.dtype}")
    return (
        pack,
        np.ascontiguousarray(key),
        np.ascontiguousarray(A),
        np.ascontiguousarray(B),
        training_validation["binding"],
    )


class ExpertMount:
    """Threshold-gated OLMoE MLP side path with exact non-fired copies."""

    def __init__(
        self,
        module: Any,
        *,
        torch_module: Any,
        key: np.ndarray,
        tau: float,
        A: np.ndarray,
        B: np.ndarray,
        label: str,
    ) -> None:
        self.module = module
        self.torch = torch_module
        self.key_np = np.ascontiguousarray(key, dtype=np.float32)
        self.tau = float(tau)
        self.A_np = np.ascontiguousarray(A.astype(np.float32))
        self.B_np = np.ascontiguousarray(B.astype(np.float32))
        self.B_has_signal = bool(np.count_nonzero(self.B_np))
        self.label = label
        self.handle: Any | None = None
        self.calls: list[dict[str, Any]] = []
        self._device_cache: dict[str, tuple[Any, Any, Any]] = {}

    def _tensors(self, device: Any) -> tuple[Any, Any, Any]:
        cache_key = str(device)
        if cache_key not in self._device_cache:
            self._device_cache[cache_key] = (
                self.torch.from_numpy(self.key_np).to(device=device, dtype=self.torch.float32),
                self.torch.from_numpy(self.A_np).to(device=device, dtype=self.torch.float32),
                self.torch.from_numpy(self.B_np).to(device=device, dtype=self.torch.float32),
            )
        return self._device_cache[cache_key]

    def _hook(self, _module: Any, inputs: tuple[Any, ...], native_output: Any) -> Any:
        import torch.nn.functional as functional

        hidden = inputs[0]
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_native = native_output.reshape(-1, native_output.shape[-1])
        key, A, B = self._tensors(hidden.device)
        scores = flat_hidden.float() @ key
        fired = scores >= self.tau
        fire_count = int(fired.sum().item())
        token_count = int(fired.numel())
        call: dict[str, Any] = {
            "label": self.label,
            # Positive infinity is the registered forced-no-fire threshold in
            # eval_abi. Keep it numeric for the comparison above, but spell
            # that intentional telemetry value as strict-JSON sentinel text.
            # Score/delta telemetry remains numeric so real overflow is caught
            # and named by write_json instead of being sanitized here.
            "tau": "inf" if self.tau == math.inf else self.tau,
            "token_count": token_count,
            "fire_count": fire_count,
            "fire_rate": fire_count / token_count,
            "nonfire_count": token_count - fire_count,
            "score_min": float(scores.min().item()),
            "score_mean": float(scores.double().mean().item()),
            "score_max": float(scores.max().item()),
            "B_has_signal": self.B_has_signal,
            "nonfired_rows_bit_equal": True,
            "fired_mask_sha256": sha256_array(
                fired.detach().to(device="cpu", dtype=self.torch.uint8).numpy()
            ),
        }
        if not self.B_has_signal or fire_count == 0:
            self.calls.append(call)
            return native_output
        selected = flat_hidden[fired].float()
        delta = functional.linear(functional.silu(functional.linear(selected, A)), B)
        result = flat_native.clone()
        result[fired] = result[fired] + delta.to(dtype=flat_native.dtype)
        if token_count > fire_count:
            call["nonfired_rows_bit_equal"] = bool(
                self.torch.equal(result[~fired], flat_native[~fired])
            )
        call["delta_mean_abs"] = float(delta.double().abs().mean().item())
        call["delta_max_abs"] = float(delta.double().abs().max().item())
        self.calls.append(call)
        return result.reshape_as(native_output)

    def __enter__(self) -> "ExpertMount":
        if self.handle is not None:
            raise RuntimeError("ExpertMount cannot be entered twice")
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self._device_cache.clear()


def tensor_sha256(torch: Any, value: Any) -> str:
    raw = (
        value.detach()
        .to(device="cpu")
        .contiguous()
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    return sha256_bytes(raw)


def forward_snapshot(torch: Any, model: Any, ids_np: np.ndarray) -> dict[str, Any]:
    ids = torch.from_numpy(np.ascontiguousarray(ids_np[None, :], dtype=np.int64)).to(
        input_device(model)
    )
    with torch.inference_mode():
        output = model(
            input_ids=ids,
            use_cache=False,
            logits_to_keep=int(ids_np.size),
            output_hidden_states=True,
        )
    if output.hidden_states is None:
        raise RuntimeError("Transformers did not return final hidden states for E2-G0")
    final_hidden = output.hidden_states[-1].detach().to(device="cpu").contiguous()
    logits = output.logits.detach().to(device="cpu").contiguous()
    result = {
        "hidden": final_hidden,
        "logits": logits,
        "hidden_sha256": tensor_sha256(torch, final_hidden),
        "logits_sha256": tensor_sha256(torch, logits),
        "hidden_bytes": int(final_hidden.numel() * final_hidden.element_size()),
        "logits_bytes": int(logits.numel() * logits.element_size()),
    }
    del output, ids
    return result


def eval_root(output: Path) -> Path:
    root = output / "eval"
    root.mkdir(parents=True, exist_ok=True)
    return root


def eval_receipt_path(
    output: Path, kind: str, window_index: int | None = None, prefix_arm: str = "p0"
) -> Path:
    root = eval_root(output)
    if kind == "abi":
        return root / "abi.json"
    if window_index is None:
        raise ValueError(f"{kind} receipt requires a window index")
    if kind == "narrative":
        return root / f"narrative_{prefix_arm}_{window_index:03d}.json"
    return root / f"{kind}_{window_index:03d}.json"


def validate_eval_receipt(
    output: Path,
    manifest: dict[str, Any],
    prepared: dict[str, np.ndarray],
    kind: str,
    *,
    window_index: int | None = None,
    prefix_arm: str = "p0",
    content_sets: Sequence[dict[str, Any]] | None = None,
    expert_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = eval_receipt_path(output, kind, window_index, prefix_arm)
    if not path.is_file():
        raise FileNotFoundError(path)
    receipt = read_json(path)
    expected_schemas = {
        "abi": {"moe_e2_abi_identity_v1", "moe_e2_abi_identity_v2"},
        "narrative": {"moe_e2_narrative_eval_v1", "moe_e2_narrative_eval_v2"},
        "wikitext": {
            "moe_e2_wikitext_noninterference_v1",
            "moe_e2_wikitext_noninterference_v2",
        },
        "code": {"moe_e2_code_fire_v1", "moe_e2_code_fire_v2"},
    }
    receipt_schema = receipt.get("schema")
    if (
        receipt_schema not in expected_schemas[kind]
        or receipt.get("status") != "complete"
        or receipt.get("model_revision") != MODEL_REVISION
    ):
        raise RuntimeError(f"eval receipt contract failed: {path}")
    if kind != "abi" and int(receipt.get("window_index", -1)) != window_index:
        raise RuntimeError(f"eval receipt window index failed: {path}")
    if kind == "narrative" and receipt.get("prefix_arm") != prefix_arm:
        raise RuntimeError(f"narrative eval prefix arm failed: {path}")
    content_sets = list(content_sets or load_prepared_content_sets(output, manifest, prepared))

    def checks(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
        if kind == "abi":
            ids = np.concatenate(
                [
                    arrays["narrative_key_ids"][0, :64],
                    arrays["code_ids"][0, :64],
                ]
            )
            return [
                receipt_array_check(
                    receipt_field="input_ids_sha256",
                    recorded_sha256=receipt.get("input_ids_sha256"),
                    array_name="narrative_key_ids+code_ids",
                    selection="narrative[0,:64] || code[0,:64]",
                    value=ids,
                )
            ]
        assert window_index is not None
        if kind == "narrative":
            target = arrays["heldout_ids"][window_index]
            prefix_name = (
                "heldout_p0_prefix_ids"
                if prefix_arm == "p0"
                else "heldout_p1_prefix_ids"
            )
            prefix = arrays[prefix_name][window_index]
            result = [
                receipt_array_check(
                    receipt_field="target_ids_sha256",
                    recorded_sha256=receipt.get("target_ids_sha256"),
                    array_name="heldout_ids",
                    selection=f"[{window_index}]",
                    value=target,
                ),
                receipt_array_check(
                    receipt_field="teacher_prefix_ids_sha256",
                    recorded_sha256=receipt.get("teacher_prefix_ids_sha256"),
                    array_name=prefix_name,
                    selection=f"[{window_index}]",
                    value=prefix,
                ),
            ]
            for arm in ("base", "teacher", "expert"):
                result.append(
                    receipt_array_check(
                        receipt_field=f"arms.{arm}.target_ids_sha256",
                        recorded_sha256=receipt.get("arms", {})
                        .get(arm, {})
                        .get("target_ids_sha256"),
                        array_name="heldout_ids",
                        selection=f"[{window_index},1:]",
                        value=target[1:],
                    )
                )
            return result
        array_name = "wikitext_ids" if kind == "wikitext" else "code_ids"
        target = arrays[array_name][window_index]
        result = [
            receipt_array_check(
                receipt_field="input_ids_sha256",
                recorded_sha256=receipt.get("input_ids_sha256"),
                array_name=array_name,
                selection=f"[{window_index}]",
                value=target,
            )
        ]
        if kind == "wikitext":
            for arm in ("base", "expert"):
                result.append(
                    receipt_array_check(
                        receipt_field=f"arms.{arm}.target_ids_sha256",
                        recorded_sha256=receipt.get("arms", {})
                        .get(arm, {})
                        .get("target_ids_sha256"),
                        array_name=array_name,
                        selection=f"[{window_index},1:]",
                        value=target[1:],
                    )
                )
        return result

    binding = bind_receipt_to_prepared_content(
        receipt,
        content_sets,
        checks,
        require_declared_content=str(receipt_schema).endswith("_v2"),
    )
    if str(receipt_schema).endswith("_v2"):
        if expert_binding is None:
            expert_binding = validate_training_artifacts(
                output, manifest, prepared
            )["binding"]
        expected_expert_provenance = content_provenance_from_binding(
            expert_binding
        )
        if receipt.get("expertpack_prepared_content") != expected_expert_provenance:
            raise RuntimeError(
                f"eval ExpertPack content provenance changed: {path}"
            )
    return {
        "kind": "eval",
        "eval_kind": kind,
        "window_index": window_index,
        "prefix_arm": prefix_arm if kind == "narrative" else None,
        "status": "passed",
        "receipt_path": str(path),
        "receipt_sha256": sha256_file(path),
        "binding": binding,
        "expertpack_binding": expert_binding,
        "receipt": receipt,
    }


def ensure_g3_green(pack: dict[str, Any]) -> None:
    if pack.get("g3", {}).get("verdict") != "GREEN":
        raise RuntimeError("E2-G3 is not GREEN; behavioral evaluation must STOP")


def eval_abi(args: argparse.Namespace) -> int:
    output = ensure_output(args.output_dir)
    manifest, prepared = load_prepared(output)
    require_bringup_green(output, manifest, prepared)
    content_sets = load_prepared_content_sets(output, manifest, prepared)
    pack, key, A, B, expert_content_binding = load_expert_arrays(
        output, manifest, prepared
    )
    ensure_g3_green(pack)
    receipt_path = eval_receipt_path(output, "abi")
    if receipt_path.is_file():
        validation = validate_eval_receipt(
            output,
            manifest,
            prepared,
            "abi",
            content_sets=content_sets,
            expert_binding=expert_content_binding,
        )
        print(json.dumps({"status": "existing", "receipt": str(receipt_path)}))
        return 0 if validation["receipt"].get("verdict") == "GREEN" else 3
    started = time.perf_counter()
    torch, model, runtime = load_model(args)
    layer = model.model.layers[int(pack["layer"])]
    ids = np.concatenate(
        [prepared["narrative_key_ids"][0, :64], prepared["code_ids"][0, :64]]
    )
    baseline = forward_snapshot(torch, model, ids)
    repeat = forward_snapshot(torch, model, ids)
    zero_install_equal = bool(
        torch.equal(baseline["hidden"], repeat["hidden"])
        and torch.equal(baseline["logits"], repeat["logits"])
    )
    with ExpertMount(
        layer.mlp,
        torch_module=torch,
        key=key,
        tau=float(pack["tau"]),
        A=A,
        B=np.zeros_like(B),
        label="zero_B",
    ) as zero_mount:
        zero_snapshot = forward_snapshot(torch, model, ids)
    zero_B_equal = bool(
        torch.equal(baseline["hidden"], zero_snapshot["hidden"])
        and torch.equal(baseline["logits"], zero_snapshot["logits"])
    )
    with ExpertMount(
        layer.mlp,
        torch_module=torch,
        key=key,
        tau=float("inf"),
        A=A,
        B=B,
        label="forced_no_fire",
    ) as nofire_mount:
        nofire_snapshot = forward_snapshot(torch, model, ids)
    nofire_equal = bool(
        torch.equal(baseline["hidden"], nofire_snapshot["hidden"])
        and torch.equal(baseline["logits"], nofire_snapshot["logits"])
    )
    with ExpertMount(
        layer.mlp,
        torch_module=torch,
        key=key,
        tau=float(pack["tau"]),
        A=A,
        B=B,
        label="live",
    ) as live_mount:
        live_snapshot = forward_snapshot(torch, model, ids)
    if len(zero_mount.calls) != 1 or len(nofire_mount.calls) != 1 or len(live_mount.calls) != 1:
        raise RuntimeError("E2-G0 expected exactly one install-layer call per forward")
    live = live_mount.calls[0]
    live_nonfire_ok = bool(live["nonfire_count"] > 0 and live["nonfired_rows_bit_equal"])
    green = bool(zero_install_equal and zero_B_equal and nofire_equal and live_nonfire_ok)
    receipt = {
        "schema": "moe_e2_abi_identity_v2",
        "status": "complete",
        "created_at": now_iso(),
        "gate": "E2-G0",
        "verdict": "GREEN" if green else "RED",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        **prepared_content_fields(manifest),
        "expertpack_prepared_content": content_provenance_from_binding(
            expert_content_binding
        ),
        "install_layer": int(pack["layer"]),
        "rank": int(pack["rank"]),
        "tau": float(pack["tau"]),
        "input_tokens": int(ids.size),
        "input_ids_sha256": sha256_array(ids),
        "zero_experts_installed": {
            "bit_equal_final_hidden_and_logits": zero_install_equal,
            "baseline": {key_name: value for key_name, value in baseline.items() if key_name not in {"hidden", "logits"}},
            "repeat": {key_name: value for key_name, value in repeat.items() if key_name not in {"hidden", "logits"}},
        },
        "zero_initialized_B_mounted": {
            "bit_equal_final_hidden_and_logits": zero_B_equal,
            "mount": zero_mount.calls[0],
            "snapshot": {key_name: value for key_name, value in zero_snapshot.items() if key_name not in {"hidden", "logits"}},
        },
        "forced_no_fire_mounted": {
            "bit_equal_final_hidden_and_logits": nofire_equal,
            "mount": nofire_mount.calls[0],
            "snapshot": {key_name: value for key_name, value in nofire_snapshot.items() if key_name not in {"hidden", "logits"}},
        },
        "live_gate": {
            "nonfired_rows_bit_equal_at_native_mlp_side_path": live_nonfire_ok,
            "mount": live,
            "final_snapshot": {key_name: value for key_name, value in live_snapshot.items() if key_name not in {"hidden", "logits"}},
        },
        "identity_comparison": "torch.equal byte-exact BF16 final hidden states and logits",
        "runtime": runtime,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss_bytes(),
        "script": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
    }
    write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "gate": "E2-G0",
                "verdict": receipt["verdict"],
                "zero_install_equal": zero_install_equal,
                "zero_B_equal": zero_B_equal,
                "nofire_equal": nofire_equal,
                "live_nonfire_count": live["nonfire_count"],
            }
        ),
        flush=True,
    )
    del model, baseline, repeat, zero_snapshot, nofire_snapshot, live_snapshot
    gc.collect()
    return 0 if green else 3


def eval_narrative(args: argparse.Namespace) -> int:
    if args.start_window < 0 or args.count < 1 or args.start_window + args.count > N_BEHAVIORAL_WINDOWS:
        raise ValueError("narrative eval range must stay within 0..15")
    output = ensure_output(args.output_dir)
    manifest, prepared = load_prepared(output)
    require_bringup_green(output, manifest, prepared)
    content_sets = load_prepared_content_sets(output, manifest, prepared)
    pack, key, A, B, expert_content_binding = load_expert_arrays(
        output, manifest, prepared
    )
    ensure_g3_green(pack)
    requested = list(range(args.start_window, args.start_window + args.count))
    missing: list[int] = []
    for index in requested:
        path = eval_receipt_path(output, "narrative", index, args.prefix_arm)
        if path.is_file():
            validate_eval_receipt(
                output,
                manifest,
                prepared,
                "narrative",
                window_index=index,
                prefix_arm=args.prefix_arm,
                content_sets=content_sets,
                expert_binding=expert_content_binding,
            )
        else:
            missing.append(index)
    if not missing:
        print(json.dumps({"status": "existing", "kind": "narrative", "arm": args.prefix_arm, "windows": requested}))
        return 0
    started = time.perf_counter()
    torch, model, runtime = load_model(args)
    layer = model.model.layers[int(pack["layer"])]
    prefix_array = (
        prepared["heldout_p0_prefix_ids"]
        if args.prefix_arm == "p0"
        else prepared["heldout_p1_prefix_ids"]
    )
    prefix_manifest_key = (
        "heldout_p0_prefixes" if args.prefix_arm == "p0" else "heldout_p1_prefixes"
    )
    for window_index in missing:
        window_started = time.perf_counter()
        target = prepared["heldout_ids"][window_index]
        prefix = prefix_array[window_index]
        teacher_full = np.concatenate([prefix, target])
        base = score_target_window(torch, model, target, target)
        teacher = score_target_window(torch, model, teacher_full, target)
        with ExpertMount(
            layer.mlp,
            torch_module=torch,
            key=key,
            tau=float(pack["tau"]),
            A=A,
            B=B,
            label=f"narrative_{args.prefix_arm}_{window_index}",
        ) as mount:
            expert = score_target_window(torch, model, target, target)
        if len(mount.calls) != 1:
            raise RuntimeError("narrative eval expected one install-layer gate call")
        receipt = {
            "schema": "moe_e2_narrative_eval_v2",
            "status": "complete",
            "created_at": now_iso(),
            "window_index": window_index,
            "prefix_arm": args.prefix_arm,
            "prefix_arm_role": "primary" if args.prefix_arm == "p0" else "registered_single_fallback",
            "target_source": manifest["windows"]["heldout_behavioral"][window_index],
            "teacher_prefix_source": manifest["windows"][prefix_manifest_key][window_index],
            "target_ids_sha256": sha256_array(target),
            "teacher_prefix_ids_sha256": sha256_array(prefix),
            "shared_scored_target_ids_exact": bool(
                base["target_ids_sha256"] == teacher["target_ids_sha256"] == expert["target_ids_sha256"]
            ),
            "teacher_scores_prefix_tokens": False,
            "arms": {"base": base, "teacher": teacher, "expert": expert},
            "fire": mount.calls[0],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            **prepared_content_fields(manifest),
            "expertpack_prepared_content": content_provenance_from_binding(
                expert_content_binding
            ),
            "install_layer": int(pack["layer"]),
            "runtime": runtime,
            "window_wall_seconds": time.perf_counter() - window_started,
            "script": str(SCRIPT_PATH),
            "script_sha256": sha256_file(SCRIPT_PATH),
        }
        path = eval_receipt_path(output, "narrative", window_index, args.prefix_arm)
        write_json(path, receipt)
        print(
            json.dumps(
                {
                    "kind": "narrative",
                    "arm": args.prefix_arm,
                    "window": window_index,
                    "ppl_base": base["ppl"],
                    "ppl_teacher": teacher["ppl"],
                    "ppl_expert": expert["ppl"],
                    "fire_rate": mount.calls[0]["fire_rate"],
                }
            ),
            flush=True,
        )
    print(json.dumps({"status": "complete", "windows": missing, "wall_seconds": time.perf_counter() - started}), flush=True)
    del model
    gc.collect()
    return 0


def eval_wikitext(args: argparse.Namespace) -> int:
    if args.start_window < 0 or args.count < 1 or args.start_window + args.count > N_KEY_WINDOWS:
        raise ValueError("WikiText eval range must stay within 0..15")
    output = ensure_output(args.output_dir)
    manifest, prepared = load_prepared(output)
    require_bringup_green(output, manifest, prepared)
    content_sets = load_prepared_content_sets(output, manifest, prepared)
    pack, key, A, B, expert_content_binding = load_expert_arrays(
        output, manifest, prepared
    )
    ensure_g3_green(pack)
    requested = list(range(args.start_window, args.start_window + args.count))
    missing: list[int] = []
    for index in requested:
        path = eval_receipt_path(output, "wikitext", index)
        if path.is_file():
            validate_eval_receipt(
                output,
                manifest,
                prepared,
                "wikitext",
                window_index=index,
                content_sets=content_sets,
                expert_binding=expert_content_binding,
            )
        else:
            missing.append(index)
    if not missing:
        print(json.dumps({"status": "existing", "kind": "wikitext", "windows": requested}))
        return 0
    torch, model, runtime = load_model(args)
    layer = model.model.layers[int(pack["layer"])]
    for window_index in missing:
        target = prepared["wikitext_ids"][window_index]
        base = score_target_window(torch, model, target, target)
        with ExpertMount(
            layer.mlp,
            torch_module=torch,
            key=key,
            tau=float(pack["tau"]),
            A=A,
            B=B,
            label=f"wikitext_{window_index}",
        ) as mount:
            expert = score_target_window(torch, model, target, target)
        if len(mount.calls) != 1:
            raise RuntimeError("WikiText eval expected one install-layer gate call")
        delta_percent = (expert["ppl"] - base["ppl"]) / base["ppl"] * 100.0
        receipt = {
            "schema": "moe_e2_wikitext_noninterference_v2",
            "status": "complete",
            "created_at": now_iso(),
            "window_index": window_index,
            "input_ids_sha256": sha256_array(target),
            "arms": {"base": base, "expert": expert},
            "per_window_ppl_delta_percent": delta_percent,
            "fire": mount.calls[0],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            **prepared_content_fields(manifest),
            "expertpack_prepared_content": content_provenance_from_binding(
                expert_content_binding
            ),
            "install_layer": int(pack["layer"]),
            "runtime": runtime,
            "script": str(SCRIPT_PATH),
            "script_sha256": sha256_file(SCRIPT_PATH),
        }
        write_json(eval_receipt_path(output, "wikitext", window_index), receipt)
        print(json.dumps({"kind": "wikitext", "window": window_index, "ppl_delta_percent": delta_percent, "fire_rate": mount.calls[0]["fire_rate"]}), flush=True)
    del model
    gc.collect()
    return 0


def eval_code(args: argparse.Namespace) -> int:
    if args.start_window < 0 or args.count < 1 or args.start_window + args.count > N_KEY_WINDOWS:
        raise ValueError("code eval range must stay within 0..15")
    output = ensure_output(args.output_dir)
    manifest, prepared = load_prepared(output)
    require_bringup_green(output, manifest, prepared)
    content_sets = load_prepared_content_sets(output, manifest, prepared)
    pack, key, A, B, expert_content_binding = load_expert_arrays(
        output, manifest, prepared
    )
    ensure_g3_green(pack)
    requested = list(range(args.start_window, args.start_window + args.count))
    missing: list[int] = []
    for index in requested:
        path = eval_receipt_path(output, "code", index)
        if path.is_file():
            validate_eval_receipt(
                output,
                manifest,
                prepared,
                "code",
                window_index=index,
                content_sets=content_sets,
                expert_binding=expert_content_binding,
            )
        else:
            missing.append(index)
    if not missing:
        print(json.dumps({"status": "existing", "kind": "code", "windows": requested}))
        return 0
    torch, model, runtime = load_model(args)
    layer = model.model.layers[int(pack["layer"])]
    for window_index in missing:
        target = prepared["code_ids"][window_index]
        ids = torch.from_numpy(np.ascontiguousarray(target[None, :], dtype=np.int64)).to(input_device(model))
        with ExpertMount(
            layer.mlp,
            torch_module=torch,
            key=key,
            tau=float(pack["tau"]),
            A=A,
            B=B,
            label=f"code_{window_index}",
        ) as mount:
            with torch.inference_mode():
                output_object = model(input_ids=ids, use_cache=False, logits_to_keep=1)
        del output_object, ids
        if len(mount.calls) != 1:
            raise RuntimeError("code eval expected one install-layer gate call")
        receipt = {
            "schema": "moe_e2_code_fire_v2",
            "status": "complete",
            "created_at": now_iso(),
            "window_index": window_index,
            "input_ids_sha256": sha256_array(target),
            "fire": mount.calls[0],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            **prepared_content_fields(manifest),
            "expertpack_prepared_content": content_provenance_from_binding(
                expert_content_binding
            ),
            "install_layer": int(pack["layer"]),
            "runtime": runtime,
            "script": str(SCRIPT_PATH),
            "script_sha256": sha256_file(SCRIPT_PATH),
        }
        write_json(eval_receipt_path(output, "code", window_index), receipt)
        print(json.dumps({"kind": "code", "window": window_index, "fire_rate": mount.calls[0]["fire_rate"]}), flush=True)
    del model
    gc.collect()
    return 0


def eval_gates(args: argparse.Namespace) -> int:
    if args.eval_kind is None:
        raise ValueError("eval-gates requires --eval-kind abi|narrative|wikitext|code")
    if args.eval_kind == "abi":
        return eval_abi(args)
    if args.eval_kind == "narrative":
        return eval_narrative(args)
    if args.eval_kind == "wikitext":
        return eval_wikitext(args)
    return eval_code(args)


def load_eval_validations(
    output: Path,
    kind: str,
    *,
    prefix_arm: str = "p0",
    count: int = 16,
    manifest: dict[str, Any] | None = None,
    prepared: dict[str, np.ndarray] | None = None,
    expert_binding: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if manifest is None or prepared is None:
        manifest, prepared = load_prepared(output)
    content_sets = load_prepared_content_sets(output, manifest, prepared)
    validations: list[dict[str, Any]] = []
    for index in range(count):
        path = eval_receipt_path(output, kind, index, prefix_arm)
        if not path.is_file():
            return []
        validation = validate_eval_receipt(
            output,
            manifest,
            prepared,
            kind,
            window_index=index,
            prefix_arm=prefix_arm,
            content_sets=content_sets,
            expert_binding=expert_binding,
        )
        if validation.get("expertpack_binding") is not None:
            expert_binding = validation["expertpack_binding"]
        validations.append(validation)
    return validations


def load_eval_sequence(
    output: Path,
    kind: str,
    *,
    prefix_arm: str = "p0",
    count: int = 16,
    manifest: dict[str, Any] | None = None,
    prepared: dict[str, np.ndarray] | None = None,
    expert_binding: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return [
        item["receipt"]
        for item in load_eval_validations(
            output,
            kind,
            prefix_arm=prefix_arm,
            count=count,
            manifest=manifest,
            prepared=prepared,
            expert_binding=expert_binding,
        )
    ]


def aggregate_nll(receipts: Sequence[dict[str, Any]], arm: str) -> dict[str, Any]:
    token_count = sum(int(receipt["arms"][arm]["token_count"]) for receipt in receipts)
    nll_sum = sum(float(receipt["arms"][arm]["nll_sum"]) for receipt in receipts)
    if token_count <= 0:
        raise ValueError("cannot aggregate empty NLL receipts")
    mean_nll = nll_sum / token_count
    return {
        "token_count": token_count,
        "nll_sum": nll_sum,
        "mean_nll": mean_nll,
        "ppl": math.exp(mean_nll),
    }


def fallback_check(args: argparse.Namespace) -> int:
    output = ensure_output(args.output_dir)
    manifest, prepared = load_prepared(output)
    receipts = load_eval_sequence(
        output,
        "narrative",
        prefix_arm="p0",
        manifest=manifest,
        prepared=prepared,
    )
    if len(receipts) != N_BEHAVIORAL_WINDOWS:
        raise RuntimeError("P0 fallback check requires all 16 primary narrative receipts")
    base = aggregate_nll(receipts, "base")
    teacher = aggregate_nll(receipts, "teacher")
    gap = base["ppl"] - teacher["ppl"]
    needed = gap <= 0.0
    print(
        json.dumps(
            {
                "fallback": "P1",
                "authorized": needed,
                "ppl_base": base["ppl"],
                "ppl_teacher_p0": teacher["ppl"],
                "teacher_gap_p0": gap,
            }
        ),
        flush=True,
    )
    return 0 if needed else 3


def status_gate(verdict: str, report_line: str, **details: Any) -> dict[str, Any]:
    return {"verdict": verdict, "report_line": report_line, **details}


def format_number(value: float | None, digits: int = 6) -> str:
    return "NOT_MEASURED" if value is None else f"{value:.{digits}g}"


def analyze(args: argparse.Namespace) -> int:
    if args.bootstrap_resamples < 2000:
        raise ValueError("analyze requires at least 2,000 bootstrap resamples")
    output = ensure_output(args.output_dir)
    manifest, prepared = load_prepared(output)
    content_sets = load_prepared_content_sets(output, manifest, prepared)
    gates: dict[str, dict[str, Any]] = {}

    bringup_path = output / "bringup.json"
    bringup_validation = (
        validate_consolidated_bringup_receipt(output, manifest, prepared)
        if bringup_path.is_file()
        else None
    )
    bringup = (
        None if bringup_validation is None else bringup_validation["receipt"]
    )
    if bringup and bringup.get("status") == "complete" and bringup.get("gate", {}).get("verdict") in {"GREEN", "RED"}:
        verdict = bringup["gate"]["verdict"]
        short_ppl = float(bringup["short_512"]["ppl"])
        delta = float(bringup["long_minus_short_mean_nll"])
        gates["E2-G-1"] = status_gate(
            verdict,
            f"E2-G-1 {verdict} — WikiText ppl512={short_ppl:.6g}, nll2048-nll512={delta:+.6f} nats (cap +0.15)",
            short_ppl=short_ppl,
            long_minus_short_nll=delta,
            receipt=str(bringup_path),
        )
    else:
        gates["E2-G-1"] = status_gate(
            "NOT_MEASURED",
            f"E2-G-1 NOT_MEASURED — run {output / 'GPU_E2_RESUME_COMMANDS.sh'}",
            receipt=str(bringup_path),
        )
    bringup_green = gates["E2-G-1"]["verdict"] == "GREEN"

    abi_path = eval_receipt_path(output, "abi")
    abi_validation = (
        validate_eval_receipt(
            output,
            manifest,
            prepared,
            "abi",
            content_sets=content_sets,
        )
        if abi_path.is_file()
        else None
    )
    abi = None if abi_validation is None else abi_validation["receipt"]
    if bringup_green and abi and abi.get("status") == "complete":
        verdict = str(abi.get("verdict", "RED"))
        live = abi["live_gate"]["mount"]
        gates["E2-G0"] = status_gate(
            verdict,
            (
                f"E2-G0 {verdict} — zero-install={abi['zero_experts_installed']['bit_equal_final_hidden_and_logits']}, "
                f"zero-B={abi['zero_initialized_B_mounted']['bit_equal_final_hidden_and_logits']}, "
                f"forced-no-fire={abi['forced_no_fire_mounted']['bit_equal_final_hidden_and_logits']}, "
                f"live nonfired exact={abi['live_gate']['nonfired_rows_bit_equal_at_native_mlp_side_path']} "
                f"({live['nonfire_count']}/{live['token_count']} tokens)"
            ),
            receipt=str(abi_path),
        )
    else:
        reason = "E2-G-1 STOP" if not bringup_green else "ABI receipt absent"
        gates["E2-G0"] = status_gate(
            "NOT_MEASURED",
            f"E2-G0 NOT_MEASURED — {reason}; lead script {output / 'GPU_E2_RESUME_COMMANDS.sh'}",
            receipt=str(abi_path),
        )

    fit_path = output / "fit_key.json"
    fit_validation = (
        validate_fit_key_artifacts(output, manifest, prepared)
        if fit_path.is_file()
        else None
    )
    fit = None if fit_validation is None else fit_validation["receipt"]
    install: dict[str, Any] = {
        "layer": None,
        "rank": 64,
        "recall": None,
        "generic_fire_rate": None,
        "code_fpr": None,
        "grm_fpr": None,
        "tau": None,
    }
    if bringup_green and fit and fit.get("status") in {"complete_g2_green", "complete_g2_red_stop"}:
        decision = fit["decision"]
        verdict = decision["verdict"]
        if decision["install_layer"] is not None:
            layer_index = int(decision["install_layer"])
            row = next(item for item in fit["rows"] if int(item["layer"]) == layer_index)
            evaluation = row["metrics"]["eval"]
            install.update(
                {
                    "layer": layer_index,
                    "recall": float(evaluation["narrative_recall"]["value"]),
                    "generic_fire_rate": float(evaluation["wikitext_fpr"]["value"]),
                    "code_fpr": float(evaluation["code_fpr"]["value"]),
                    "grm_fpr": float(evaluation["grm_fpr"]["value"]),
                    "tau": float(row["metrics"]["selected_tau"]),
                }
            )
            detail = (
                f"L*={layer_index}, recall={install['recall']:.6f}, "
                f"generic fire={install['generic_fire_rate']:.6f} descriptive, "
                f"code FPR={install['code_fpr']:.6f}, GRM FPR={install['grm_fpr']:.6f}, "
                f"tau={install['tau']:.9g}"
            )
        else:
            detail = "qualifying layers=0/16 under frozen FIT tau; STOP"
        gates["E2-G2"] = status_gate(
            verdict,
            f"E2-G2 {verdict} — {detail}",
            receipt=str(fit_path),
            qualifying_layers=decision["qualifying_layers_fit_ranked"],
        )
    else:
        reason = "E2-G-1 STOP" if not bringup_green else "64 all-layer corpus captures and K4 fit absent"
        gates["E2-G2"] = status_gate(
            "NOT_MEASURED",
            f"E2-G2 NOT_MEASURED — {reason}; lead script {output / 'GPU_E2_RESUME_COMMANDS.sh'}",
            receipt=str(fit_path),
            qualifying_layers=[],
        )

    train_path = output / "train.json"
    training_validation = (
        validate_training_artifacts(output, manifest, prepared)
        if train_path.is_file()
        else None
    )
    training = (
        None if training_validation is None else training_validation["receipt"]
    )
    if gates["E2-G2"]["verdict"] == "GREEN" and training and training.get("status") in {"complete_g3_green", "complete_g3_red_stop"}:
        verdict = training["verdict"]
        stored = training["stored_fp16_validation"]
        gates["E2-G3"] = status_gate(
            verdict,
            (
                f"E2-G3 {verdict} — zero MSE={stored['zero_predictor_mse']:.9g}, "
                f"adapter MSE={stored['adapter_mse']:.9g}, improvement={stored['improvement_percent']:.3f}% "
                "(floor 10%)"
            ),
            receipt=str(train_path),
        )
    else:
        reason = "E2-G2 STOP" if gates["E2-G2"]["verdict"] != "GREEN" else "64 pair captures/training absent"
        gates["E2-G3"] = status_gate(
            "NOT_MEASURED",
            f"E2-G3 NOT_MEASURED — {reason}; lead script {output / 'GPU_E2_RESUME_COMMANDS.sh'}",
            receipt=str(train_path),
        )

    pair_truth = {
        "complete_pairs": 0,
        "aligned_pairs": 0,
        "hidden_different_pairs": 0,
        "first_two_receipts": [],
        "content_lineages": {},
    }
    for index in range(N_PAIR_WINDOWS):
        data_path, receipt_path = pair_paths(output, index)
        if not data_path.exists() and not receipt_path.exists():
            continue
        validation = validate_pair_capture_receipt(
            output,
            manifest,
            prepared,
            index,
            content_sets=content_sets,
        )
        receipt = validation["receipt"]
        pair_truth["complete_pairs"] += 1
        pair_truth["aligned_pairs"] += int(bool(receipt.get("shared_window_token_ids_exact")))
        pair_truth["hidden_different_pairs"] += int(bool(receipt.get("router_input_hidden_states_differ")))
        content_label = validation["binding"]["content_label"]
        pair_truth["content_lineages"][content_label] = (
            pair_truth["content_lineages"].get(content_label, 0) + 1
        )
        if len(pair_truth["first_two_receipts"]) < 2:
            pair_truth["first_two_receipts"].append(str(receipt_path))

    g4_measurement: dict[str, Any] | None = None
    g4_row = (
        "G4 row: ppl_base=NOT_MEASURED, ppl_teacher=NOT_MEASURED, "
        "ppl_expert=NOT_MEASURED, recovery=NOT_MEASURED, CI95=NOT_MEASURED"
    )
    p0_validations = load_eval_validations(
        output,
        "narrative",
        prefix_arm="p0",
        manifest=manifest,
        prepared=prepared,
        expert_binding=(
            None if training_validation is None else training_validation["binding"]
        ),
    )
    p0_receipts = [item["receipt"] for item in p0_validations]
    p0_content_binding = (
        bind_transitive_receipt_content(
            {},
            [item["binding"] for item in p0_validations],
            dependency_kind="narrative_p0_eval_receipts",
        )
        if p0_validations
        else None
    )
    p1_validations = load_eval_validations(
        output,
        "narrative",
        prefix_arm="p1",
        manifest=manifest,
        prepared=prepared,
        expert_binding=(
            None if training_validation is None else training_validation["binding"]
        ),
    )
    p1_receipts = [item["receipt"] for item in p1_validations]
    p1_content_binding = (
        bind_transitive_receipt_content(
            {},
            [item["binding"] for item in p1_validations],
            dependency_kind="narrative_p1_eval_receipts",
        )
        if p1_validations
        else None
    )
    fallback_state = "not_evaluated"
    if gates["E2-G3"]["verdict"] == "GREEN" and len(p0_receipts) == N_BEHAVIORAL_WINDOWS:
        p0_base = aggregate_nll(p0_receipts, "base")
        p0_teacher = aggregate_nll(p0_receipts, "teacher")
        p0_gap = p0_base["ppl"] - p0_teacher["ppl"]
        active_receipts = p0_receipts
        active_arm = "p0"
        fallback_state = "not_needed" if p0_gap > 0 else "required"
        if p0_gap <= 0 and len(p1_receipts) == N_BEHAVIORAL_WINDOWS:
            active_receipts = p1_receipts
            active_arm = "p1"
            fallback_state = "completed_once"
        if p0_gap <= 0 and len(p1_receipts) != N_BEHAVIORAL_WINDOWS:
            gates["E2-G4"] = status_gate(
                "NOT_MEASURED",
                (
                    f"E2-G4 NOT_MEASURED — P0 precondition failed (ppl gap={p0_gap:+.6g}); "
                    f"registered one-time P1 fallback pending at {output / 'GPU_E2_P1_FALLBACK_COMMANDS.sh'}"
                ),
                prefix_arm="p0",
                fallback_state=fallback_state,
            )
        else:
            base = aggregate_nll(active_receipts, "base")
            teacher = aggregate_nll(active_receipts, "teacher")
            expert = aggregate_nll(active_receipts, "expert")
            teacher_gap = base["ppl"] - teacher["ppl"]
            if teacher_gap <= 0:
                gates["E2-G4"] = status_gate(
                    "RED",
                    (
                        f"E2-G4 RED — {active_arm.upper()} teacher precondition gap={teacher_gap:+.6g} <= 0; "
                        "P0 and the registered one-time P1 fallback do not establish the premise"
                    ),
                    prefix_arm=active_arm,
                    fallback_state=fallback_state,
                )
                g4_measurement = {
                    "prefix_arm": active_arm,
                    "ppl_base": base["ppl"],
                    "ppl_teacher": teacher["ppl"],
                    "ppl_expert": expert["ppl"],
                    "teacher_gap_ppl": teacher_gap,
                    "recovery_percent": None,
                    "ci95_low_percent": None,
                    "ci95_high_percent": None,
                }
                g4_row = (
                    f"G4 row: ppl_base={base['ppl']:.6g}, ppl_teacher={teacher['ppl']:.6g}, "
                    f"ppl_expert={expert['ppl']:.6g}, recovery=UNDEFINED, CI95=UNDEFINED"
                )
            else:
                recovery = bootstrap_recovery(
                    np.asarray([receipt["arms"]["base"]["mean_nll"] for receipt in active_receipts]),
                    np.asarray([receipt["arms"]["teacher"]["mean_nll"] for receipt in active_receipts]),
                    np.asarray([receipt["arms"]["expert"]["mean_nll"] for receipt in active_receipts]),
                    resamples=args.bootstrap_resamples,
                    seed=args.seed + 90000,
                )
                verdict = "GREEN" if recovery["recovery_fraction"] >= G4_RECOVERY_FLOOR else "RED"
                gates["E2-G4"] = status_gate(
                    verdict,
                    (
                        f"E2-G4 {verdict} — arm={active_arm.upper()}, recovery={recovery['recovery_percent']:.3f}% "
                        f"CI95=[{recovery['ci95_low_percent']:.3f}%, {recovery['ci95_high_percent']:.3f}%], floor=25%"
                    ),
                    prefix_arm=active_arm,
                    fallback_state=fallback_state,
                    measurement=recovery,
                )
                g4_measurement = {"prefix_arm": active_arm, **recovery}
                g4_row = (
                    f"G4 row: ppl_base={recovery['ppl_base']:.6g}, "
                    f"ppl_teacher={recovery['ppl_teacher']:.6g}, "
                    f"ppl_expert={recovery['ppl_expert']:.6g}, "
                    f"recovery={recovery['recovery_percent']:.3f}%, "
                    f"CI95=[{recovery['ci95_low_percent']:.3f}%, {recovery['ci95_high_percent']:.3f}%]"
                )
    else:
        reason = "E2-G3 STOP" if gates["E2-G3"]["verdict"] != "GREEN" else "16 P0 held-out evaluations absent"
        gates["E2-G4"] = status_gate(
            "NOT_MEASURED",
            f"E2-G4 NOT_MEASURED — {reason}; lead script {output / 'GPU_E2_RESUME_COMMANDS.sh'}",
            fallback_state=fallback_state,
        )

    wikitext_validations = load_eval_validations(
        output,
        "wikitext",
        manifest=manifest,
        prepared=prepared,
        expert_binding=(
            None if training_validation is None else training_validation["binding"]
        ),
    )
    wikitext_receipts = [item["receipt"] for item in wikitext_validations]
    wikitext_content_binding = (
        bind_transitive_receipt_content(
            {},
            [item["binding"] for item in wikitext_validations],
            dependency_kind="wikitext_eval_receipts",
        )
        if wikitext_validations
        else None
    )
    code_validations = load_eval_validations(
        output,
        "code",
        manifest=manifest,
        prepared=prepared,
        expert_binding=(
            None if training_validation is None else training_validation["binding"]
        ),
    )
    code_receipts = [item["receipt"] for item in code_validations]
    code_content_binding = (
        bind_transitive_receipt_content(
            {},
            [item["binding"] for item in code_validations],
            dependency_kind="code_eval_receipts",
        )
        if code_validations
        else None
    )
    g5_measurement: dict[str, Any] | None = None
    if gates["E2-G3"]["verdict"] == "GREEN" and len(wikitext_receipts) == 16 and len(code_receipts) == 16:
        wiki_base = aggregate_nll(wikitext_receipts, "base")
        wiki_expert = aggregate_nll(wikitext_receipts, "expert")
        wiki_delta_percent = (wiki_expert["ppl"] - wiki_base["ppl"]) / wiki_base["ppl"] * 100.0
        code_fires = sum(int(receipt["fire"]["fire_count"]) for receipt in code_receipts)
        code_tokens = sum(int(receipt["fire"]["token_count"]) for receipt in code_receipts)
        code_fire_rate = code_fires / code_tokens
        green = bool(
            wiki_delta_percent <= G5_WIKITEXT_PPL_DELTA_CAP_PERCENT
            and code_fire_rate <= G5_CODE_FIRE_CAP
        )
        verdict = "GREEN" if green else "RED"
        per_window = [
            {
                "window_index": int(receipt["window_index"]),
                "fire_rate": float(receipt["fire"]["fire_rate"]),
                "ppl_delta_percent": float(receipt["per_window_ppl_delta_percent"]),
            }
            for receipt in wikitext_receipts
        ]
        firing_windows = [row for row in per_window if row["fire_rate"] > 0]
        g5_measurement = {
            "wikitext_ppl_base": wiki_base["ppl"],
            "wikitext_ppl_expert": wiki_expert["ppl"],
            "wikitext_ppl_delta_percent": wiki_delta_percent,
            "code_fire_rate": code_fire_rate,
            "per_window_wikitext": per_window,
            "firing_wikitext_windows_descriptive": firing_windows,
        }
        gates["E2-G5"] = status_gate(
            verdict,
            (
                f"E2-G5 {verdict} — WikiText ppl delta={wiki_delta_percent:+.4f}% (cap +0.5%), "
                f"code fire={code_fire_rate:.4%} (cap 5%)"
            ),
            measurement=g5_measurement,
        )
    else:
        reason = "E2-G3 STOP" if gates["E2-G3"]["verdict"] != "GREEN" else "16 WikiText + 16 code receipts absent"
        gates["E2-G5"] = status_gate(
            "NOT_MEASURED",
            f"E2-G5 NOT_MEASURED — {reason}; lead script {output / 'GPU_E2_RESUME_COMMANDS.sh'}",
        )

    install_layer_text = (
        "NOT_MEASURED" if install["layer"] is None else str(int(install["layer"]))
    )
    install_row = (
        f"INSTALL row: layer L*={install_layer_text}, r=64, "
        f"recall={format_number(install['recall'])}, "
        f"generic fire={format_number(install['generic_fire_rate'])} descriptive, "
        f"code FPR={format_number(install['code_fpr'])}, "
        f"GRM FPR={format_number(install['grm_fpr'])}, tau={format_number(install['tau'], 9)}"
    )
    stage_bindings = {
        "bringup": (
            None if bringup_validation is None else bringup_validation["binding"]
        ),
        "abi_input": None if abi_validation is None else abi_validation["binding"],
        "fit_key": None if fit_validation is None else fit_validation["binding"],
        "training": (
            None if training_validation is None else training_validation["binding"]
        ),
        "expertpack": (
            training_validation["expertpack_binding"]
            if training_validation is not None
            else (
                fit_validation["expertpack_binding"]
                if fit_validation is not None
                else None
            )
        ),
        "narrative_p0_eval": p0_content_binding,
        "narrative_p1_eval": p1_content_binding,
        "wikitext_eval": wikitext_content_binding,
        "code_eval": code_content_binding,
    }
    validated_stage_content = {
        name: {
            **content_provenance_from_binding(binding),
            "validation_mode": binding["mode"],
        }
        for name, binding in stage_bindings.items()
        if binding is not None
    }
    analysis = {
        "schema": "moe_e2_analysis_v1",
        "created_at": now_iso(),
        "status": (
            "complete_all_green"
            if all(gates[name]["verdict"] == "GREEN" for name in ("E2-G-1", "E2-G0", "E2-G2", "E2-G3", "E2-G4", "E2-G5"))
            else "complete_with_red_or_not_measured"
        ),
        "evidence_class": "behavioral inference measurement, one base model, one guide corpus",
        "likelihood_instrument_valid_only_if_e2_g_1_green": bringup_green,
        "gates": gates,
        "install": install,
        "install_row": install_row,
        "g4": g4_measurement,
        "g4_row": g4_row,
        "g5": g5_measurement,
        "fallback_state": fallback_state,
        "pair_truth": pair_truth,
        "prepared_windows_content": prepared_content_provenance(manifest),
        "validated_stage_content": validated_stage_content,
        "corpus_split_receipt": {
            "manifest": str(output / "corpus_manifest.json"),
            "manifest_sha256": sha256_file(output / "corpus_manifest.json"),
            "algorithm": manifest["file_split"]["algorithm"],
            "train_count": manifest["file_split"]["train_count"],
            "heldout_count": manifest["file_split"]["heldout_count"],
            "train_files": [item["relative_path"] for item in manifest["file_split"]["train"]],
            "heldout_files": [item["relative_path"] for item in manifest["file_split"]["heldout"]],
            "heldout_text_in_key_pair_or_prefix": False,
        },
        "lead_scripts": {
            "main": str(output / "GPU_E2_RESUME_COMMANDS.sh"),
            "p1_fallback": str(output / "GPU_E2_P1_FALLBACK_COMMANDS.sh"),
        },
        "expertpack_manifest": str(output / EXPERTPACK_NAME / "manifest.json"),
        "bootstrap_resamples": args.bootstrap_resamples,
        "script": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
    }
    analysis_path = output / "analysis.json"
    write_json(analysis_path, analysis)

    report_path = output / "MOE_E2_REPORT.md"
    files = sorted(
        [SCRIPT_PATH, report_path]
        + [path for path in output.rglob("*") if path.is_file()],
        key=str,
    )
    files = list(dict.fromkeys(files))
    lines = [
        "# MOE-E2 OLMoE Narrative Expert Report",
        "",
        f"Generated: `{analysis['created_at']}`",
        "",
        "## Gate status",
        "",
    ]
    for name in ("E2-G-1", "E2-G0", "E2-G2", "E2-G3", "E2-G4", "E2-G5"):
        lines.append(gates[name]["report_line"])
    lines.extend(
        [
            "",
            install_row,
            g4_row,
            "",
            "## Pair truth (informational carried G1 contract)",
            "",
            (
                f"Complete pairs={pair_truth['complete_pairs']}/64; exact shared-token alignment="
                f"{pair_truth['aligned_pairs']}/64; differing teacher/student router inputs="
                f"{pair_truth['hidden_different_pairs']}/64."
            ),
            "",
            "## Corpus split receipt",
            "",
            f"Algorithm: {manifest['file_split']['algorithm']}",
            "",
            f"TRAIN files ({manifest['file_split']['train_count']}):",
            "",
        ]
    )
    lines.extend(f"- `{item['relative_path']}` — `{item['content_sha256']}`" for item in manifest["file_split"]["train"])
    lines.extend(["", f"HELDOUT files ({manifest['file_split']['heldout_count']}):", ""])
    lines.extend(f"- `{item['relative_path']}` — `{item['content_sha256']}`" for item in manifest["file_split"]["heldout"])
    lines.extend(
        [
            "",
            "HELDOUT file text is excluded from key fit/eval, pair train/validation, and every P0/P1 teacher prefix.",
            "",
            "## Prefix fallback",
            "",
            f"State: `{fallback_state}`. P0 is primary; P1 may run exactly once only when the aggregate P0 teacher gap is non-positive.",
            "",
        ]
    )
    if g5_measurement is not None:
        lines.extend(
            [
                "## E2-G5 firing-window descriptors",
                "",
                "| WikiText window | Fire rate | PPL delta |",
                "| ---: | ---: | ---: |",
            ]
        )
        for row in g5_measurement["firing_wikitext_windows_descriptive"]:
            lines.append(
                f"| {row['window_index']} | {row['fire_rate']:.4%} | {row['ppl_delta_percent']:+.4f}% |"
            )
        lines.append("")
    lines.extend(
        [
            "## Lead execution scripts",
            "",
            f"- Main chain: `{output / 'GPU_E2_RESUME_COMMANDS.sh'}`",
            f"- Conditional P1 fallback: `{output / 'GPU_E2_P1_FALLBACK_COMMANDS.sh'}`",
            "",
            "Both scripts enforce one visible GPU, `flock /tmp/forge-gpu.lock`, a 590 s hard wall per GPU invocation, and at least 30 s idle gaps.",
            "",
            "## ExpertPack",
            "",
            f"Manifest: `{output / EXPERTPACK_NAME / 'manifest.json'}`",
            "",
            "## Files created or modified",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in files)
    lines.extend(
        [
            "",
            "## Evidence limits",
            "",
            "This is behavioral inference evidence on one base model and one guide corpus. Finite address, MSE, identity, and PPL receipts do not establish general expert-growth capability.",
            "",
        ]
    )
    write_text(report_path, "\n".join(lines))

    pack_path = output / EXPERTPACK_NAME / "manifest.json"
    if pack_path.is_file():
        pack = read_json(pack_path)
        pack["updated_at"] = now_iso()
        pack["behavioral"] = {
            "analysis": str(analysis_path),
            "analysis_sha256": sha256_file(analysis_path),
            "report": str(report_path),
            "gates": {name: gate["verdict"] for name, gate in gates.items()},
        }
        if gates["E2-G3"]["verdict"] == "GREEN":
            pack["status"] = (
                "complete_all_gates_green"
                if all(gates[name]["verdict"] == "GREEN" for name in gates)
                else "trained_behavioral_gates_red_or_incomplete"
            )
        write_json(pack_path, pack)
    print(
        json.dumps(
            {
                "status": analysis["status"],
                "gates": {name: gate["verdict"] for name, gate in gates.items()},
                "report": str(report_path),
                "analysis": str(analysis_path),
            }
        ),
        flush=True,
    )
    return 0 if analysis["status"] == "complete_all_green" else 3


def receipt_validation_sidecar_path(receipt_path: Path) -> Path:
    return receipt_path.with_name(
        f"{receipt_path.stem}.provenance_validation.json"
    )


def public_validation_summary(validation: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "receipt",
        "payload",
        "data",
        "pair_validations",
        "cells",
    }
    return {key: value for key, value in validation.items() if key not in excluded}


def write_receipt_validation_sidecar(
    receipt_path: Path,
    validation: dict[str, Any] | None,
    *,
    error: Exception | None = None,
) -> dict[str, Any]:
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    original_sha = sha256_file(receipt_path)
    original_stat = receipt_path.stat()
    sidecar_path = receipt_validation_sidecar_path(receipt_path)
    existing = read_json(sidecar_path) if sidecar_path.is_file() else {}
    if existing:
        sealed_original = existing.get("original_receipt", {})
        if (
            sealed_original.get("sha256") != original_sha
            or int(sealed_original.get("byte_count", -1)) != original_stat.st_size
            or int(sealed_original.get("mtime_ns_before_validation", -1))
            != original_stat.st_mtime_ns
        ):
            raise RuntimeError(
                "original receipt differs from its immutable validation sidecar: "
                f"{receipt_path}"
            )
    binding = None if validation is None else validation.get("binding")
    try:
        original_receipt = read_json(receipt_path)
    except (json.JSONDecodeError, UnicodeDecodeError):
        original_receipt = {}
    legacy_field = original_receipt.get("prepared_windows_sha256")
    if legacy_field is None:
        provenance = original_receipt.get("provenance", {})
        prepared_record = (
            provenance.get("prepared_windows", {})
            if isinstance(provenance, dict)
            else {}
        )
        legacy_field = (
            prepared_record.get("sha256")
            if isinstance(prepared_record, dict)
            else None
        )
    if error is None:
        mode = None if binding is None else binding.get("mode")
        if mode and mode.startswith("native_content"):
            migration_note = (
                "Original receipt already records the canonical combined content digest."
            )
        elif mode and "transitive" in mode:
            migration_note = (
                "Original receipt is immutable. Prepared provenance was recovered "
                "transitively through its exact immutable dependency receipts."
            )
        elif legacy_field is not None:
            migration_note = (
                "Original receipt is immutable. Its legacy NPZ byte SHA was not used "
                "as content; every recorded consumed-array hash was revalidated "
                "against the selected canonical content lineage."
            )
        else:
            migration_note = (
                "Original receipt is immutable; all recorded prepared-array content "
                "was revalidated."
            )
        status = "passed"
        summary = public_validation_summary(validation or {})
        error_record = None
    else:
        status = "failed"
        migration_note = (
            "Original receipt was not mutated. Content migration was rejected because "
            "its recorded dependencies did not validate."
        )
        summary = None
        error_record = {"type": type(error).__name__, "message": str(error)}
    payload = {
        "schema": "moe_e2_receipt_provenance_validation_v1",
        "status": status,
        "created_at": existing.get("created_at", now_iso()),
        "order": str(F4_ORDER_PATH),
        "order_sha256": sha256_file(F4_ORDER_PATH),
        "validator_script": str(SCRIPT_PATH),
        "validator_script_sha256": sha256_file(SCRIPT_PATH),
        "original_receipt": {
            "path": str(receipt_path),
            "sha256": original_sha,
            "byte_count": int(original_stat.st_size),
            "mtime_ns_before_validation": int(original_stat.st_mtime_ns),
        },
        "migration": {
            "note": migration_note,
            "legacy_prepared_windows_byte_sha256": legacy_field,
            "original_receipt_mutated": False,
        },
        "validation": summary,
        "error": error_record,
    }
    write_json_if_changed(sidecar_path, payload)
    after_stat = receipt_path.stat()
    if (
        sha256_file(receipt_path) != original_sha
        or after_stat.st_mtime_ns != original_stat.st_mtime_ns
        or after_stat.st_size != original_stat.st_size
    ):
        raise RuntimeError(f"original receipt changed during sidecar write: {receipt_path}")
    return {
        "status": status,
        "original_receipt_path": str(receipt_path),
        "original_receipt_sha256": original_sha,
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": sha256_file(sidecar_path),
        "content_binding": binding,
        "error": error_record,
    }


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "byte_count": int(stat.st_size),
        "inode": int(stat.st_ino),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def validate_provenance(args: argparse.Namespace) -> int:
    output = ensure_output(args.output_dir)
    manifest_path = output / "corpus_manifest.json"
    windows_path = output / "prepared_windows.npz"
    manifest, prepared = load_prepared(output)
    reconstruct_legacy_prepared_content(args, output, manifest, prepared)
    content_sets = load_prepared_content_sets(output, manifest, prepared)

    root = provenance_validation_root(output)
    root.mkdir(parents=True, exist_ok=True)
    content_sidecar_path = root / "prepared_windows_content_validation.json"
    existing_content_sidecar = (
        read_json(content_sidecar_path) if content_sidecar_path.is_file() else {}
    )
    content_sidecar = {
        "schema": "moe_e2_prepared_content_validation_v1",
        "status": "passed",
        "created_at": existing_content_sidecar.get("created_at", now_iso()),
        "order": str(F4_ORDER_PATH),
        "order_sha256": sha256_file(F4_ORDER_PATH),
        "manifest": file_identity(manifest_path),
        "archive_bytes_informational_only": file_identity(windows_path),
        "current_content": {
            "algorithm": manifest["prepared_windows"]["content_digest_algorithm"],
            "sha256": manifest["prepared_windows"]["content_sha256"],
            "arrays": manifest["prepared_windows"]["arrays"],
        },
        "registered_content_lineages": [
            {
                "label": item["label"],
                "content_sha256": item["content"]["sha256"],
                "legacy_archive_sha256": item["legacy_archive_sha256"],
                "metadata_path": item["metadata_path"],
            }
            for item in content_sets
        ],
        "manifest_and_archive_mutated": False,
    }
    write_json_if_changed(content_sidecar_path, content_sidecar)

    validations: list[tuple[Path, dict[str, Any] | None, Exception | None]] = []

    def collect(receipt_path: Path, callback: Callable[[], dict[str, Any]]) -> None:
        try:
            validations.append((receipt_path, callback(), None))
        except Exception as exc:
            validations.append((receipt_path, None, exc))

    for name in ("short", "long"):
        path = bringup_cell_path(output, name)
        if path.is_file():
            collect(
                path,
                lambda name=name, path=path: {
                    "kind": "bringup_cell",
                    "status": "passed",
                    "cell": name,
                    **validate_bringup_cell_receipt(
                        path, manifest, prepared, name=name
                    ),
                },
            )
    consolidated_path = output / "bringup.json"
    if consolidated_path.is_file():
        collect(
            consolidated_path,
            lambda: {
                "kind": "bringup",
                "status": "passed",
                **validate_consolidated_bringup_receipt(
                    output, manifest, prepared
                ),
            },
        )

    active_key_paths: set[Path] = set()
    key_results: list[dict[str, Any]] = []
    for corpus in KEY_CORPORA:
        for index in range(N_KEY_WINDOWS):
            data_path, receipt_path = key_capture_paths(output, corpus, index)
            active_key_paths.add(receipt_path.resolve())

            def validate_active_key(
                corpus: str = corpus,
                index: int = index,
            ) -> dict[str, Any]:
                result = validate_key_capture_receipt(
                    output,
                    manifest,
                    prepared,
                    corpus,
                    index,
                    content_sets=content_sets,
                )
                key_results.append(result)
                return result

            collect(receipt_path, validate_active_key)

    key_root = output / "key_captures"
    if key_root.is_dir():
        for receipt_path in sorted(key_root.rglob("*_receipt.json")):
            if receipt_path.resolve() in active_key_paths:
                continue
            receipt = read_json(receipt_path)
            corpus = str(receipt.get("corpus"))
            index = int(receipt.get("window_index", -1))
            stem = receipt_path.name.removesuffix("_receipt.json")
            data_path = receipt_path.parent / f"{stem}_router_inputs_fp16.npy"
            collect(
                receipt_path,
                lambda corpus=corpus, index=index, data_path=data_path, receipt_path=receipt_path: validate_key_capture_receipt(
                    output,
                    manifest,
                    prepared,
                    corpus,
                    index,
                    data_path=data_path,
                    receipt_path=receipt_path,
                    classification="superseded_relocated",
                    content_sets=content_sets,
                    allow_unregistered_legacy=True,
                ),
            )

    fit_path = output / "fit_key.json"
    fit_result: dict[str, Any] | None = None
    if fit_path.is_file():
        def validate_fit() -> dict[str, Any]:
            nonlocal fit_result
            fit_result = validate_fit_key_artifacts(output, manifest, prepared)
            return fit_result

        collect(fit_path, validate_fit)

    pair_results: list[dict[str, Any]] = []
    for index in range(N_PAIR_WINDOWS):
        _data_path, receipt_path = pair_paths(output, index)

        def validate_pair(index: int = index) -> dict[str, Any]:
            result = validate_pair_capture_receipt(
                output,
                manifest,
                prepared,
                index,
                content_sets=content_sets,
            )
            pair_results.append(result)
            return result

        collect(receipt_path, validate_pair)

    training_path = output / "train.json"
    training_result: dict[str, Any] | None = None
    if training_path.is_file():
        def validate_training() -> dict[str, Any]:
            nonlocal training_result
            training_result = validate_training_artifacts(output, manifest, prepared)
            return training_result

        collect(training_path, validate_training)

    expertpack_path = output / EXPERTPACK_NAME / "manifest.json"
    expertpack_result: dict[str, Any] | None = None
    expertpack_validation_entry: dict[str, Any] | None = None
    if expertpack_path.is_file() and (
        training_result is not None or fit_result is not None
    ):
        dependency_result = (
            training_result if training_result is not None else fit_result
        )
        assert dependency_result is not None
        expertpack_result = {
            "kind": "expertpack",
            "status": "passed",
            "receipt_path": str(expertpack_path),
            "receipt_sha256": sha256_file(expertpack_path),
            "binding": dependency_result["expertpack_binding"],
        }
        expertpack_ledger_path = root / (
            f"expertpack_manifest_{sha256_file(expertpack_path)}."
            "provenance_validation.json"
        )
        existing_expertpack_ledger = (
            read_json(expertpack_ledger_path)
            if expertpack_ledger_path.is_file()
            else {}
        )
        expertpack_ledger = {
            "schema": "moe_e2_mutable_expertpack_provenance_validation_v1",
            "status": "passed",
            "created_at": existing_expertpack_ledger.get("created_at", now_iso()),
            "order": str(F4_ORDER_PATH),
            "order_sha256": sha256_file(F4_ORDER_PATH),
            "validator_script": str(SCRIPT_PATH),
            "validator_script_sha256": sha256_file(SCRIPT_PATH),
            "expertpack_manifest": file_identity(expertpack_path),
            "mutable_manifest_not_a_receipt": True,
            "validation": public_validation_summary(expertpack_result),
            "original_manifest_mutated": False,
        }
        write_json_if_changed(expertpack_ledger_path, expertpack_ledger)
        expertpack_validation_entry = {
            "path": str(expertpack_ledger_path),
            "sha256": sha256_file(expertpack_ledger_path),
        }

    eval_results: list[dict[str, Any]] = []
    eval_dir = output / "eval"
    if eval_dir.is_dir():
        for path in sorted(eval_dir.glob("*.json")):
            if path.name.endswith(".provenance_validation.json"):
                continue
            kind: str | None = None
            index: int | None = None
            arm = "p0"
            if path.name == "abi.json":
                kind = "abi"
            else:
                match = re.fullmatch(r"narrative_(p[01])_(\d{3})\.json", path.name)
                if match:
                    kind, arm, index = "narrative", match.group(1), int(match.group(2))
                else:
                    match = re.fullmatch(r"(wikitext|code)_(\d{3})\.json", path.name)
                    if match:
                        kind, index = match.group(1), int(match.group(2))
            if kind is None:
                continue

            def validate_one_eval(
                kind: str = kind, index: int | None = index, arm: str = arm
            ) -> dict[str, Any]:
                result = validate_eval_receipt(
                    output,
                    manifest,
                    prepared,
                    kind,
                    window_index=index,
                    prefix_arm=arm,
                    content_sets=content_sets,
                    expert_binding=(
                        None
                        if training_result is None
                        else training_result["binding"]
                    ),
                )
                eval_results.append(result)
                return result

            collect(path, validate_one_eval)

    sidecar_entries: list[dict[str, Any]] = []
    for path, result, error in validations:
        if path.is_file():
            sidecar_entries.append(
                write_receipt_validation_sidecar(path, result, error=error)
            )
        else:
            missing_error = error or FileNotFoundError(path)
            sidecar_entries.append(
                {
                    "status": "failed",
                    "original_receipt_path": str(path),
                    "original_receipt_sha256": None,
                    "sidecar_path": None,
                    "sidecar_sha256": None,
                    "content_binding": None,
                    "error": {
                        "type": type(missing_error).__name__,
                        "message": str(missing_error),
                    },
                }
            )

    before_prepare = {
        "manifest": file_identity(manifest_path),
        "prepared_windows": file_identity(windows_path),
    }
    prepare_result = prepare(args)
    after_prepare = {
        "manifest": file_identity(manifest_path),
        "prepared_windows": file_identity(windows_path),
    }
    prepare_idempotent = before_prepare == after_prepare and prepare_result == 0
    if not prepare_idempotent:
        raise RuntimeError("prepare rewrote the sealed manifest or archive")

    current_arrays = content_sets[0]["arrays"]
    current_compatible_keys: list[str] = []
    current_incompatible_keys: list[str] = []
    for corpus in KEY_CORPORA:
        array_name = "narrative_key_ids" if corpus == "narrative" else f"{corpus}_ids"
        for index in range(N_KEY_WINDOWS):
            label = f"{corpus}_{index:02d}"
            receipt_path = key_capture_paths(output, corpus, index)[1]
            if not receipt_path.is_file():
                current_incompatible_keys.append(label)
                continue
            receipt = read_json(receipt_path)
            if receipt.get("input_ids_sha256") == sha256_array(
                current_arrays[array_name][index]
            ):
                current_compatible_keys.append(label)
            else:
                current_incompatible_keys.append(label)

    failed = [item for item in sidecar_entries if item["status"] != "passed"]
    legacy_sets = [item for item in content_sets if item["label"] != "current_manifest"]
    legacy_array_differences: dict[str, list[str]] = {}
    for item in legacy_sets:
        legacy_array_differences[item["label"]] = [
            name
            for name in PREPARED_ARRAY_SHAPES
            if sha256_array(item["arrays"][name])
            != sha256_array(current_arrays[name])
        ]
    narrative_00 = next(
        (
            item
            for item in sidecar_entries
            if item["original_receipt_path"]
            == str(key_capture_paths(output, "narrative", 0)[1])
        ),
        None,
    )
    pair_000 = next(
        (
            item
            for item in sidecar_entries
            if item["original_receipt_path"] == str(pair_paths(output, 0)[1])
        ),
        None,
    )
    ledger_path = output / "f4_provenance_validation.json"
    existing_ledger = read_json(ledger_path) if ledger_path.is_file() else {}
    ledger = {
        "schema": "moe_e2_f4_provenance_validation_v1",
        "status": "passed" if not failed else "failed",
        "created_at": existing_ledger.get("created_at", now_iso()),
        "order": str(F4_ORDER_PATH),
        "order_sha256": sha256_file(F4_ORDER_PATH),
        "validator_script": str(SCRIPT_PATH),
        "validator_script_sha256": sha256_file(SCRIPT_PATH),
        "mechanism": {
            "npz_container_bytes_are_provenance": False,
            "content_digest_algorithm": PREPARED_CONTENT_ALGORITHM,
            "current_archive_sha256_informational": sha256_file(windows_path),
            "reported_timestamp_only_hypothesis_confirmed": False,
            "observed_root_cause": (
                "F3 edited scripts/olmoe_e2_experiment.py, which is itself part "
                "of build_code_corpus; 14 interior code windows changed while "
                "nine other prepared arrays stayed equal"
            ),
            "legacy_array_differences_from_current": legacy_array_differences,
        },
        "current_content": prepared_content_provenance(manifest),
        "registered_content_lineages": [
            {
                "label": item["label"],
                "content_sha256": item["content"]["sha256"],
                "legacy_archive_sha256": item["legacy_archive_sha256"],
                "metadata_path": item["metadata_path"],
            }
            for item in content_sets
        ],
        "prepare_idempotence_proof": {
            "passed": prepare_idempotent,
            "before": before_prepare,
            "after": after_prepare,
            "npz_and_manifest_rewritten": False,
            "chain_scripts_regenerated": False,
        },
        "salvage": {
            "active_key_receipts_valid": len(key_results),
            "active_key_receipts_expected": len(KEY_CORPORA) * N_KEY_WINDOWS,
            "active_keys_matching_current_content": len(current_compatible_keys),
            "active_keys_not_matching_current_content": current_incompatible_keys,
            "active_keys_valid_against_registered_historical_content": sum(
                1
                for item in key_results
                if item["binding"]["content_label"] != "current_manifest"
            ),
            "pair_receipts_valid": len(pair_results),
            "pair_receipts_expected": N_PAIR_WINDOWS,
            "fit_key_valid": fit_result is not None,
            "fit_key_verdict": None if fit_result is None else fit_result["verdict"],
            "training_valid": training_result is not None,
            "training_verdict": (
                None if training_result is None else training_result["verdict"]
            ),
            "expertpack_valid": expertpack_result is not None,
            "expertpack_validation_ledger": expertpack_validation_entry,
            "expertpack_legacy_provenance_conflicts_with_component_lineage": (
                None
                if expertpack_result is None
                else expertpack_result["binding"]
                ["expertpack_declared_prepared_provenance"]
                ["conflicts_with_component_lineage"]
            ),
            "eval_receipts_valid": len(eval_results),
            "eval_receipts": [
                {
                    "kind": item["eval_kind"],
                    "window_index": item["window_index"],
                    "receipt_path": item["receipt_path"],
                }
                for item in eval_results
            ],
            "behavioral_eval_receipts_absent": not any(
                item["eval_kind"] != "abi" for item in eval_results
            ),
            "narrative_00_proof": narrative_00,
            "pair_000_proof": pair_000,
        },
        "sidecar_count": len(sidecar_entries),
        "sidecars": sidecar_entries,
        "failed_validations": failed,
        "original_receipts_mutated": False,
        "anything_not_done": [
            "No GPU work was run.",
            "No missing narrative, WikiText, or code behavioral evaluations were fabricated.",
        ],
    }
    write_json_if_changed(ledger_path, ledger)
    print(
        json.dumps(
            {
                "status": ledger["status"],
                "ledger": str(ledger_path),
                "sidecars": len(sidecar_entries),
                "active_keys": len(key_results),
                "pairs": len(pair_results),
                "eval_receipts": len(eval_results),
                "current_compatible_keys": len(current_compatible_keys),
                "historical_lineage_keys": ledger["salvage"][
                    "active_keys_valid_against_registered_historical_content"
                ],
            }
        ),
        flush=True,
    )
    return 0 if ledger["status"] == "passed" else 3


def gpu_script_text(output: Path) -> str:
    script = SCRIPT_PATH.relative_to(REPO_ROOT).as_posix()
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {REPO_ROOT}",
        "",
        "run_gpu_e2() {",
        f"  {GPU_PREAMBLE} \"$@\"",
        "  sleep 30",
        "}",
        "",
        "run_cpu_e2() {",
        "  timeout --signal=TERM --kill-after=5s 7200s env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DEACTIVATE_ASYNC_LOAD=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/olmoe_e2_pycache PYTHONUNBUFFERED=1 \"$@\"",
        "}",
        "",
        "# --device auto: full BF16 CPU conversion + clean load-report gate, then 11 GiB dispatch.",
        f"run_cpu_e2 python3 {script} prepare",
        f"run_gpu_e2 python3 {script} bringup --device auto --max-gpu-memory 11GiB",
        "",
        "# Fresh OLMoE address-space captures: four windows per bounded lease.",
    ]
    for corpus in KEY_CORPORA:
        for start in range(0, N_KEY_WINDOWS, 4):
            lines.append(
                f"run_gpu_e2 python3 {script} capture-keys --device auto --max-gpu-memory 11GiB --corpus {corpus} --start-window {start} --count 4"
            )
    lines.extend(
        [
            "",
            f"run_cpu_e2 python3 {script} fit-key --threads 8 --bootstrap-resamples 2000",
            "",
            "# P0 teacher/student pairs: four pairs per bounded lease.",
        ]
    )
    for start in range(0, N_PAIR_WINDOWS, 4):
        lines.append(
            f"run_gpu_e2 python3 {script} capture-pairs --device auto --max-gpu-memory 11GiB --start-window {start} --count 4"
        )
    lines.extend(
        [
            "",
            f"run_cpu_e2 python3 {script} train --threads 8 --rank 64 --train-tokens-per-window 128",
            "",
            "# Exact identity is intentionally CPU BF16 to avoid CUDA reduction nondeterminism.",
            f"timeout --signal=TERM --kill-after=5s 590s env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DEACTIVATE_ASYNC_LOAD=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/olmoe_e2_pycache PYTHONUNBUFFERED=1 python3 {script} eval-gates --device cpu --eval-kind abi",
            "",
            "# Primary P0 held-out behavioral arm.",
        ]
    )
    for start in range(0, N_BEHAVIORAL_WINDOWS, 2):
        lines.append(
            f"run_gpu_e2 python3 {script} eval-gates --device auto --max-gpu-memory 11GiB --eval-kind narrative --prefix-arm p0 --start-window {start} --count 2"
        )
    lines.append("")
    for kind in ("wikitext", "code"):
        for start in range(0, N_KEY_WINDOWS, 4):
            lines.append(
                f"run_gpu_e2 python3 {script} eval-gates --device auto --max-gpu-memory 11GiB --eval-kind {kind} --start-window {start} --count 4"
            )
    lines.extend(
        [
            "",
            "set +e",
            f"run_cpu_e2 python3 {script} analyze --bootstrap-resamples 2000",
            "analysis_rc_e2=$?",
            "set -e",
            f"echo \"Main analysis exit ${'{'}analysis_rc_e2{'}'}; if P0 precondition failed, run {output / 'GPU_E2_P1_FALLBACK_COMMANDS.sh'} exactly once.\"",
            "exit \"${analysis_rc_e2}\"",
            "",
        ]
    )
    return "\n".join(lines)


def fallback_script_text(output: Path) -> str:
    script = SCRIPT_PATH.relative_to(REPO_ROOT).as_posix()
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {REPO_ROOT}",
        "",
        "run_gpu_e2() {",
        f"  {GPU_PREAMBLE} \"$@\"",
        "  sleep 30",
        "}",
        "",
        "# --device auto: full BF16 CPU conversion + clean load-report gate, then 11 GiB dispatch.",
        f"python3 {script} fallback-check",
        "# The check above refuses P1 unless the complete P0 aggregate gap is <= 0.",
    ]
    for start in range(0, N_BEHAVIORAL_WINDOWS, 2):
        lines.append(
            f"run_gpu_e2 python3 {script} eval-gates --device auto --max-gpu-memory 11GiB --eval-kind narrative --prefix-arm p1 --start-window {start} --count 2"
        )
    lines.extend(
        [
            "set +e",
            f"python3 {script} analyze --bootstrap-resamples 2000",
            "analysis_rc_e2=$?",
            "set -e",
            "exit \"${analysis_rc_e2}\"",
            "",
        ]
    )
    return "\n".join(lines)


def write_gpu_scripts(output: Path) -> None:
    main_path = output / "GPU_E2_RESUME_COMMANDS.sh"
    fallback_path = output / "GPU_E2_P1_FALLBACK_COMMANDS.sh"
    write_text(main_path, gpu_script_text(output))
    write_text(fallback_path, fallback_script_text(output))
    main_path.chmod(0o755)
    fallback_path.chmod(0o755)


def main() -> int:
    args = parse_args()
    if args.seed != DEFAULT_SEED:
        raise ValueError(f"registered E2 seed is frozen at {DEFAULT_SEED}")
    if args.mode == "prepare":
        return prepare(args)
    if args.mode == "self-test":
        return self_test(args)
    if args.mode == "validate-provenance":
        return validate_provenance(args)
    if args.mode == "load-check":
        return load_check(args)
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
    if args.mode == "fallback-check":
        return fallback_check(args)
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
