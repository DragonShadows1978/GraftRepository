#!/usr/bin/env python3
"""Shared, sealed inputs and append-only helpers for GLC Phase 0.

This module deliberately has no GPU imports.  The GPU-only port capture imports
TensorCUDA lazily after it has validated the Phase 0 inputs and device contract.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
PLAN_PATH = REPO_ROOT / "docs" / "GLC_LONG_CONTEXT_PLAN.md"
OUTPUT_ROOT = REPO_ROOT / "artifacts" / "glc_p0"
CONTROL_ROOT = REPO_ROOT / "artifacts" / "moe_e1_3b"
CONTROL_MANIFEST = CONTROL_ROOT / "control_manifest.json"
CONTROL_ARRAYS = CONTROL_ROOT / "controls.npz"
PORT_RUNS = CONTROL_ROOT / "runs"
MODEL_DIR = Path(
    "/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/"
    "snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
)
MODEL_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"

RAMP_LENGTHS = (1024, 1536, 2048, 2304, 2560)
SHORT_SEQUENCE_LENGTH = 512
LONG_SEQUENCE_LENGTH = 2560
N_TARGETS = 511
HG1_DELTA_NATS_MAX = 0.05
LONG_TO_SHORT_DEVIATION_RATIO = 10.0
SHORT_PROBE_POSITIONS = (100, 500)
LONG_PROBE_POSITIONS = (100, 500, 1100, 2100, 2500)

EXPECTED_CONTROL_ARRAYS_SHA256 = (
    "27fbcebf7d176cc33fd8143c3ac20cb19d32fb3a58a6ae5f71dd89ccec4e4210"
)
EXPECTED_WIKITEXT_SHA256 = (
    "2fde696c089735c20dd8869a31fa58374041ee5d38b81f6f2d13de656f8bdd15"
)


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    work = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(work).cast("B")).hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(raw)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"append-only artifact already exists: {path}")
    atomic_write_json(path, payload)


def write_new_npz(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"append-only artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def write_once_text(path: Path, content: str, *, executable: bool = False) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"append-only artifact differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    if executable:
        path.chmod(0o755)


def ensure_output_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != OUTPUT_ROOT.resolve():
        raise ValueError(f"GLC P0 artifacts are restricted to {OUTPUT_ROOT.resolve()}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_model_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != MODEL_DIR.resolve():
        raise ValueError(f"GLC P0 is sealed to snapshot {MODEL_DIR.resolve()}")
    if resolved.name != MODEL_REVISION or not (resolved / "config.json").is_file():
        raise FileNotFoundError(f"pinned snapshot is incomplete: {resolved}")
    return resolved


def validate_sealed_controls() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load only the existing E1.3b token payload; never regenerate tokens."""

    if not CONTROL_MANIFEST.is_file() or not CONTROL_ARRAYS.is_file():
        raise FileNotFoundError("sealed E1.3b controls are missing")
    manifest = read_json(CONTROL_MANIFEST)
    observed_file_hash = sha256_file(CONTROL_ARRAYS)
    expected_file_hash = manifest.get("control_arrays_sha256")
    if observed_file_hash != expected_file_hash:
        raise RuntimeError(
            f"controls.npz hash {observed_file_hash} != manifest {expected_file_hash}"
        )
    if observed_file_hash != EXPECTED_CONTROL_ARRAYS_SHA256:
        raise RuntimeError(
            f"controls.npz hash drifted from GLC registration: {observed_file_hash}"
        )
    if manifest.get("model_revision") != MODEL_REVISION:
        raise RuntimeError("control manifest model revision is not the GLC snapshot")
    with np.load(CONTROL_ARRAYS, allow_pickle=False) as stored:
        arrays = {
            name: np.ascontiguousarray(stored[name], dtype=np.int64)
            for name in stored.files
        }
    wiki = arrays.get("wikitext_0_2560")
    if wiki is None or wiki.shape != (LONG_SEQUENCE_LENGTH,):
        raise RuntimeError(f"bad sealed WikiText shape: {getattr(wiki, 'shape', None)}")
    wiki_hash = sha256_array(wiki)
    if wiki_hash != manifest["wikitext"]["token_ids_sha256"]:
        raise RuntimeError("WikiText array hash differs from control manifest")
    if wiki_hash != EXPECTED_WIKITEXT_SHA256:
        raise RuntimeError(f"WikiText GLC registration drifted: {wiki_hash}")
    return manifest, arrays


def port_ramp_receipt(length: int, view: str) -> dict[str, Any]:
    path = PORT_RUNS / f"ramp_{int(length)}_{view}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    receipt = read_json(path)
    if receipt.get("status") != "complete":
        raise RuntimeError(f"port receipt is not complete: {path}")
    return receipt


def ramp_arm(wikitext: np.ndarray, length: int, view: str) -> dict[str, Any]:
    length = int(length)
    if length not in RAMP_LENGTHS or view not in {"long", "reference"}:
        raise ValueError("ramp arm requires a registered length and long|reference")
    source_start = 0 if view == "long" else length - SHORT_SEQUENCE_LENGTH
    full = np.ascontiguousarray(wikitext[source_start:length], dtype=np.int64)
    inputs = np.ascontiguousarray(full[:-1][None, :], dtype=np.int64)
    targets = np.ascontiguousarray(full[-N_TARGETS:], dtype=np.int64)
    predictors = np.ascontiguousarray(full[-SHORT_SEQUENCE_LENGTH:-1], dtype=np.int64)
    if inputs.shape != (1, full.size - 1):
        raise AssertionError("causal input shift failed")
    if targets.shape != (N_TARGETS,) or predictors.shape != (N_TARGETS,):
        raise AssertionError("registered 511-token score alignment failed")

    receipt = port_ramp_receipt(length, view)
    observed = {
        "full_sequence_ids_sha256": sha256_array(full),
        "input_ids_sha256": sha256_array(inputs),
        "predictor_ids_sha256": sha256_array(predictors),
        "target_ids_sha256": sha256_array(targets),
    }
    for key, value in observed.items():
        if receipt.get(key) != value:
            raise RuntimeError(
                f"sealed {length}/{view} {key} {value} != port receipt {receipt.get(key)}"
            )
    return {
        "length": length,
        "view": view,
        "source_start": source_start,
        "source_stop_exclusive": length,
        "full_sequence": full,
        "inputs": inputs,
        "targets": targets,
        "predictors": predictors,
        **observed,
        "port_receipt": str(PORT_RUNS / f"ramp_{length}_{view}.json"),
        "port_mean_nll": float(receipt["score"]["mean_nll"]),
        "port_ppl": float(receipt["score"]["ppl"]),
    }


def divergence_sequence(wikitext: np.ndarray, sequence: str) -> dict[str, Any]:
    if sequence == "short":
        length = SHORT_SEQUENCE_LENGTH
        positions = SHORT_PROBE_POSITIONS
    elif sequence == "long":
        length = LONG_SEQUENCE_LENGTH
        positions = LONG_PROBE_POSITIONS
    else:
        raise ValueError("sequence must be short or long")
    ids = np.ascontiguousarray(wikitext[:length][None, :], dtype=np.int64)
    if any(position < 0 or position >= length for position in positions):
        raise AssertionError("probe position escapes sequence")
    return {
        "sequence": sequence,
        "length": length,
        "positions": np.asarray(positions, dtype=np.int64),
        "input_ids": ids,
        "input_ids_sha256": sha256_array(ids),
    }


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def runtime_facts() -> dict[str, Any]:
    facts: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "kernels_package_available": package_available("kernels"),
        "triton_available": package_available("triton"),
        "accelerate_available": package_available("accelerate"),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
    }
    try:
        import torch

        facts.update(
            {
                "torch_version": torch.__version__,
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
        )
        if torch.cuda.is_available():
            facts["cuda_device_name"] = torch.cuda.get_device_name(0)
            facts["cuda_compute_capability"] = list(torch.cuda.get_device_capability(0))
    except Exception as exc:  # pragma: no cover - diagnostic only
        facts["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import transformers

        facts["transformers_version"] = transformers.__version__
    except Exception as exc:  # pragma: no cover - diagnostic only
        facts["transformers_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import accelerate

        facts["accelerate_version"] = accelerate.__version__
    except Exception as exc:  # pragma: no cover - diagnostic only
        facts["accelerate_error"] = f"{type(exc).__name__}: {exc}"
    return facts


def load_hf_reference_model(
    *,
    model_dir: Path,
    cpu_memory: str,
    offload_dir: Path,
    for_causal_lm: bool,
) -> tuple[Any, dict[str, Any]]:
    """Load the pinned HF model dequantized to BF16 with CPU/disk offload.

    The local installation has Transformers MXFP4 support but no ``kernels``
    package, so native MXFP4 execution cannot be an evidence-bearing path.
    Explicit dequantization avoids an implicit fallback and records the route.
    """

    model_dir = ensure_model_dir(model_dir)
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE=1 is mandatory")
    if os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("TRANSFORMERS_OFFLINE=1 is mandatory")

    import torch
    from transformers import AutoModel, AutoModelForCausalLM
    from transformers.utils.quantization_config import Mxfp4Config

    raw_config = read_json(model_dir / "config.json")
    quantization = raw_config.get("quantization_config") or {}
    if quantization.get("quant_method") != "mxfp4":
        raise RuntimeError("pinned snapshot is not the registered MXFP4 checkpoint")
    qconfig = Mxfp4Config.from_dict(quantization, dequantize=True)
    if not qconfig.dequantize:
        raise AssertionError("explicit HF CPU dequantization was not enabled")

    offload_dir = offload_dir.expanduser().resolve()
    offload_dir.mkdir(parents=True, exist_ok=True)
    cls = AutoModelForCausalLM if for_causal_lm else AutoModel
    model = cls.from_pretrained(
        str(model_dir),
        local_files_only=True,
        quantization_config=qconfig,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={"cpu": cpu_memory},
        offload_folder=str(offload_dir),
        offload_state_dict=True,
        attn_implementation="eager",
    )
    model.eval()
    device_map = {
        str(key): str(value) for key, value in getattr(model, "hf_device_map", {}).items()
    }
    info = {
        "route": "hf_transformers_cpu_bf16_dequant_accelerate_offload",
        "checkpoint_quant_method": "mxfp4",
        "dequantize": True,
        "attention_implementation": "eager",
        "cpu_memory_limit": cpu_memory,
        "offload_dir": str(offload_dir),
        "model_class": type(model).__name__,
        "hf_device_map": device_map,
        "runtime": runtime_facts(),
    }
    return model, info


def latest_complete_receipt(directory: Path, stem: str) -> tuple[Path, dict[str, Any]] | None:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(directory.glob(f"{stem}_attempt*.json")):
        try:
            payload = read_json(path)
        except Exception:
            continue
        if payload.get("status") == "complete":
            candidates.append((path, payload))
    return candidates[-1] if candidates else None

