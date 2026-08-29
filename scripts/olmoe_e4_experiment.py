#!/usr/bin/env python3
"""MOE-E4: compare closed-form and SGD episodic experts on OLMoE.

ARM-CF solves the frozen-gate least-squares residual map in closed form,
selects ridge lambda on the registered pair-validation split, and factorizes
the selected map with an exact SVD.  ARM-SGD is the inherited E2/E3 rank-64
SiLU adapter trained on the exact same pair payloads.  Model and corpus access
is local-only; every stage is resumable and content-addressed.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import shlex
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("PYTHONPYCACHEPREFIX", "/tmp/olmoe_e4_pycache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

import olmoe_e2_experiment as e2
import olmoe_e3_experiment as e3


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
ORDER_PATH = REPO_ROOT / "orders" / "MOE_E4_CLOSED_FORM_CONSOLIDATION.md"
E3_ORDER_PATH = REPO_ROOT / "orders" / "MOE_E3_EPISODIC_CONSOLIDATION.md"
E31_ORDER_PATH = REPO_ROOT / "orders" / "MOE_E3_1_EPISODE_KEY.md"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "moe_e4"
E3_OUTPUT = REPO_ROOT / "artifacts" / "moe_e3"
E31_OUTPUT = REPO_ROOT / "artifacts" / "moe_e3_1"
E31_FIT_PATH = E31_OUTPUT / "fit_keys.json"
E31_KEYS_PATH = E31_OUTPUT / "all_layer_linear_keys.npz"
E31_MANIFEST_PATH = E31_OUTPUT / "artifact_manifest.json"

MODEL_ID = e2.MODEL_ID
MODEL_REVISION = e2.MODEL_REVISION
DEFAULT_MODEL_DIR = e2.DEFAULT_MODEL_DIR
N_LAYERS = e2.N_LAYERS
HIDDEN_DIM = e2.HIDDEN_DIM
WINDOW_TOKENS = e2.WINDOW_TOKENS
PREFIX_TOKENS = e2.PREFIX_TOKENS
MAX_CONTEXT = e2.MAX_CONTEXT
N_PAIR_WINDOWS = e3.N_PAIR_WINDOWS
N_PAIR_TRAIN = e3.N_PAIR_TRAIN
N_PAIR_VALIDATION = e3.N_PAIR_VALIDATION
N_BEHAVIORAL_WINDOWS = e3.N_BEHAVIORAL_WINDOWS
PAIR_DOC_WINDOWS = e3.PAIR_DOC_WINDOWS
RANK = 64
DEFAULT_SEED = 20260829
DEFAULT_BOOTSTRAP_RESAMPLES = 2000
RIDGE_ALPHA_GRID = (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0)
G3_IMPROVEMENT_FLOOR = 0.10
TEACHER_GAP_FLOOR_PPL = 0.5
G4_RECOVERY_FLOOR = 0.25
E4_H_MARGIN_POINTS = 5.0
G5_WIKITEXT_PPL_DELTA_CAP_PERCENT = 0.5
G5_CODE_FIRE_CAP = 0.05
ADDRESS_CAVEAT = "domain-grade address"
DOC_B_ROLE = "DESCRIPTIVE"
GUIDES_ROLE = "DESCRIPTIVE"
ARMS = ("cf", "sgd")
ARM_NAMES = {"cf": "ARM-CF", "sgd": "ARM-SGD"}
ARM_ACTIVATIONS = {"cf": "identity", "sgd": "silu"}
PACK_NAMES = {
    "cf": "expertpack_docA_olmoe_cf_r64_v0",
    "sgd": "expertpack_docA_olmoe_sgd_r64_v0",
}
TRAINING_FILES = {"cf": "cf_fit.json", "sgd": "sgd_train.json"}
EVAL_KINDS = ("abi", "doc-a-teacher", "doc-a-expert", "wikitext", "code", "doc-b", "guides")
PAIR_ARRAY_NAMES = {"h_student_fp16", "out_student_fp16", "out_teacher_fp16"}
CPU_WALL_SECONDS = 7200
GPU_WALL_SECONDS = 590
GPU_PREAMBLE = (
    "flock -w 7200 /tmp/forge-gpu.lock "
    "timeout --signal=TERM --kill-after=5s 590s "
    "env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
    "HF_DEACTIVATE_ASYNC_LOAD=1 TOKENIZERS_PARALLELISM=false "
    "PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/olmoe_e4_pycache "
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
            "capture-pairs",
            "fit-cf",
            "train-sgd",
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
    parser.add_argument("--start-window", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--eval-kind", choices=EVAL_KINDS)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--rank", type=int, default=RANK)
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
    resolved = path.resolve()
    root = (REPO_ROOT / "artifacts" / "moe_e4").resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        if resolved != root:
            raise ValueError("E4 output must remain under artifacts/moe_e4") from exc
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return e2.sha256_bytes(payload)


def caveat_fields() -> dict[str, Any]:
    return {
        "address_caveat": ADDRESS_CAVEAT,
        "doc_b_firing_role": DOC_B_ROLE,
        "guides_firing_role": GUIDES_ROLE,
    }


def require_caveat(receipt: dict[str, Any], label: str) -> None:
    if (
        receipt.get("address_caveat") != ADDRESS_CAVEAT
        or receipt.get("doc_b_firing_role") != DOC_B_ROLE
        or receipt.get("guides_firing_role") != GUIDES_ROLE
    ):
        raise RuntimeError(f"{label} is missing the registered fallback caveat/roles")


def content_fields(manifest: dict[str, Any]) -> dict[str, Any]:
    prepared = manifest["prepared_windows"]
    return {
        "prepared_windows_content_digest_algorithm": prepared["content_digest_algorithm"],
        "prepared_windows_content_sha256": prepared["content_sha256"],
    }


def script_fields() -> dict[str, Any]:
    return {"script": str(SCRIPT_PATH), "script_sha256": e2.sha256_file(SCRIPT_PATH)}


def require_finite_array(name: str, value: np.ndarray) -> None:
    if not np.isfinite(np.asarray(value)).all():
        raise ValueError(f"refusing to write non-finite array: {name}")


def source_install_decision() -> dict[str, Any]:
    for path in (E31_FIT_PATH, E31_KEYS_PATH, E31_MANIFEST_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"E3.1 dependency absent: {path}")
    source_manifest = e2.read_json(E31_MANIFEST_PATH)
    fit = e2.read_json(E31_FIT_PATH)
    if (
        source_manifest.get("schema") != "moe_e3_1_artifact_manifest_v1"
        or source_manifest.get("status") != "complete"
        or fit.get("schema") != "moe_e3_1_fit_keys_v1"
        or fit.get("status") != "complete"
        or fit.get("model_revision") != MODEL_REVISION
        or fit.get("prepared_windows_content_sha256")
        != e3.load_prepared(E3_OUTPUT)[0]["prepared_windows"]["content_sha256"]
        or fit.get("keys_artifact", {}).get("path") != str(E31_KEYS_PATH)
        or fit.get("keys_artifact", {}).get("sha256") != e2.sha256_file(E31_KEYS_PATH)
    ):
        raise RuntimeError("E3.1 dependency contract failed")
    listed = {row.get("path"): row.get("sha256") for row in source_manifest.get("files", [])}
    if listed.get(str(E31_FIT_PATH)) != e2.sha256_file(E31_FIT_PATH):
        raise RuntimeError("E3.1 artifact manifest no longer binds fit_keys.json")
    if listed.get(str(E31_KEYS_PATH)) != e2.sha256_file(E31_KEYS_PATH):
        raise RuntimeError("E3.1 artifact manifest no longer binds all-layer keys")

    decision = fit.get("decision", {})
    k4 = decision.get("K4", {})
    qualifying_count = int(k4.get("qualifying_layer_count", -1))
    if qualifying_count > 0:
        layer = int(k4["selected_qualifying_layer"])
        branch = "E3.1 K4 qualifying-layer branch"
        branch_code = "e3_1_k4_qualifying"
        row_summary = k4["selected_qualifying_row"]
        caveat = None
    elif qualifying_count == 0:
        layer = int(k4["carried_fit_ranked_layer"])
        branch = "registered fallback: best E3.1 K4 layer by fit-side rank"
        branch_code = "registered_fallback_best_e3_1_k4_fit_side_rank"
        row_summary = k4["carried_fit_ranked_row"]
        caveat = ADDRESS_CAVEAT
    else:
        raise RuntimeError("E3.1 K4 decision is incomplete")
    if layer < 0 or layer >= N_LAYERS or row_summary.get("layer") != layer:
        raise RuntimeError("E3.1 selected/fallback layer linkage failed")
    rows = [row for row in fit.get("rows", []) if row.get("arm") == "K4" and row.get("layer") == layer]
    if len(rows) != 1:
        raise RuntimeError("E3.1 full K4 install row is missing or duplicated")
    row = rows[0]
    tau = float(row["metrics"]["selected_tau"])
    if tau != float(row_summary["tau"]):
        raise RuntimeError("E3.1 summary/full-row tau mismatch")
    with np.load(E31_KEYS_PATH, allow_pickle=False) as archive:
        key = np.ascontiguousarray(archive["K4"][layer], dtype=np.float32)
    if key.shape != (HIDDEN_DIM,) or not np.isfinite(key).all():
        raise RuntimeError("E3.1 K4 key payload contract failed")
    if abs(float(np.linalg.norm(key.astype(np.float64))) - 1.0) > 1.0e-5:
        raise RuntimeError("E3.1 K4 key is not unit norm")
    if row.get("key_metadata", {}).get("key_sha256") != e2.sha256_array(key):
        raise RuntimeError("E3.1 row does not bind selected K4 key")
    metrics = row["metrics"]["eval"]
    install_row = {
        "layer": layer,
        "rank": RANK,
        "tau": tau,
        "tau_quantile": row["metrics"]["selected_quantile_label"],
        "qualifies": bool(row["metrics"]["qualifies"]),
        "doc_a_recall": metrics["doc_a_recall"]["value"],
        "doc_b_fire": metrics["doc_b_fpr"]["value"],
        "guides_fire": metrics["guides_fpr"]["value"],
        "code_fire": metrics["code_fpr"]["value"],
        "grm_fire": metrics["grm_fpr"]["value"],
        "wikitext_fire": metrics["wikitext_fpr"]["value"],
        "fit_side_rank": row["fit_side_rank"],
        "hyperparameter": row_summary["hyperparameter"],
    }
    return {
        "fit": fit,
        "source_manifest": source_manifest,
        "layer": layer,
        "tau": tau,
        "key": key,
        "row": row,
        "install_row": install_row,
        "branch": branch,
        "branch_code": branch_code,
        "caveat": caveat,
        "registered_sentence": decision.get("registered_sentence"),
    }


def pack_path(output: Path, arm: str) -> Path:
    return output / PACK_NAMES[arm] / "manifest.json"


def initialize_pack(output: Path, manifest: dict[str, Any], install: dict[str, Any], arm: str) -> None:
    path = pack_path(output, arm)
    if path.is_file():
        return
    pack_dir = path.parent
    pack_dir.mkdir(parents=True, exist_ok=True)
    pack = {
        "schema": PACK_NAMES[arm],
        "status": "prepared_pending_pairs",
        "created_at": now_iso(),
        "arm": ARM_NAMES[arm],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "document": manifest["documents"]["doc_a"],
        "layer": install["install_row"]["layer"],
        "rank": RANK,
        "tau": install["install_row"]["tau"],
        "activation": ARM_ACTIVATIONS[arm],
        "abi": (
            "B @ A @ h, added to native MLP output iff h @ key >= tau"
            if arm == "cf"
            else "B @ silu(A @ h), added to native MLP output iff h @ key >= tau"
        ),
        "components": {
            "key": install["key"],
            "A": {"status": "pending", "shape": [RANK, HIDDEN_DIM], "storage_dtype": "float16"},
            "B": {"status": "pending", "shape": [HIDDEN_DIM, RANK], "storage_dtype": "float16"},
        },
        "provenance": {
            "order": str(ORDER_PATH),
            "order_sha256": e2.sha256_file(ORDER_PATH),
            "corpus_manifest": str(output / "corpus_manifest.json"),
            "corpus_manifest_sha256": e2.sha256_file(output / "corpus_manifest.json"),
            "install_receipt": str(output / "install.json"),
            "install_receipt_sha256": e2.sha256_file(output / "install.json"),
        },
        "registered_gates": {
            "E4-G3": {"validation_mse_improvement_gte": G3_IMPROVEMENT_FLOOR},
            "E4-G4": {
                "recovery_fraction_gte": G4_RECOVERY_FLOOR,
                "bootstrap_ci95_low_gt": 0.0,
            },
            "E4-G5": {
                "wikitext_ppl_delta_percent_lte": G5_WIKITEXT_PPL_DELTA_CAP_PERCENT,
                "code_fire_rate_lte": G5_CODE_FIRE_CAP,
                "doc_b": DOC_B_ROLE,
                "guides": GUIDES_ROLE,
            },
        },
        "limitations": ["pair capture, construction, ABI, and behavioral measurements pending"],
        **caveat_fields(),
    }
    e2.write_json(path, pack)


def prepare(args: argparse.Namespace) -> int:
    output = ensure_output(args.output_dir)
    core_paths = (
        output / "corpus_manifest.json",
        output / "prepared_windows.npz",
        output / "install.json",
        output / "address" / "key_K4_fp32.npy",
    )
    existing = [path.is_file() for path in core_paths]
    if any(existing):
        if not all(existing):
            raise RuntimeError("partial E4 prepare state; refusing silent repair")
        manifest, _arrays = load_prepared(output)
        install = validate_install(output, manifest)
        for arm in ARMS:
            initialize_pack(output, manifest, install, arm)
        write_lead_script(output)
        print(
            json.dumps(
                {
                    "status": "existing",
                    "content_sha256": manifest["prepared_windows"]["content_sha256"],
                    "install_layer": install["install_row"]["layer"],
                    "install_branch": install["dependency_branch"],
                    "address_caveat": ADDRESS_CAVEAT,
                }
            ),
            flush=True,
        )
        return 0

    source_manifest, arrays = e3.load_prepared(E3_OUTPUT)
    decision = source_install_decision()
    if decision["branch_code"] != "registered_fallback_best_e3_1_k4_fit_side_rank":
        raise RuntimeError("live E3.1 dependency no longer selects the registered fallback branch")
    if decision["caveat"] != ADDRESS_CAVEAT:
        raise RuntimeError("registered fallback must carry the exact domain-grade caveat")
    prepared_path = output / "prepared_windows.npz"
    e2.save_npz(prepared_path, compressed=True, **arrays)
    content = e3.prepared_content_record(arrays)
    manifest = {
        "schema": "moe_e4_corpus_manifest_v1",
        "status": "complete",
        "created_at": now_iso(),
        "order": str(ORDER_PATH),
        "order_sha256": e2.sha256_file(ORDER_PATH),
        "parent_order": str(E3_ORDER_PATH),
        "parent_order_sha256": e2.sha256_file(E3_ORDER_PATH),
        **script_fields(),
        "model_id": MODEL_ID,
        "model_dir": str(e2.validate_model_dir(args.model_dir)),
        "model_revision": MODEL_REVISION,
        "seed": args.seed,
        "documents": source_manifest["documents"],
        "regions": source_manifest["regions"],
        "windows": source_manifest["windows"],
        "registered_design": {
            "arms": {
                "ARM-CF": "ridge closed form on frozen-gate FIT rows; exact SVD; rank 64 linear A/B",
                "ARM-SGD": "E2-style zero-B rank 64 SiLU adapter; identical pair payloads",
                "ARM-CF-FULL": "selected-lambda untruncated ridge; descriptive validation ceiling",
            },
            "pair_split": {"fit": [0, 47], "validation": [48, 63]},
            "rank": RANK,
            "ridge_alpha_grid": list(RIDGE_ALPHA_GRID),
            "ridge_lambda_definition": "alpha * trace(H_fit^T H_fit) / hidden_dim",
            "ridge_selection": (
                "minimum ARM-CF-FULL MSE on pair-validation rows, then stronger shrinkage; "
                "no HELDOUT behavioral rows"
            ),
            "ridge_formula": "W = Delta^T H (H^T H + lambda I)^-1",
            "frozen_gate_fit_semantics": (
                "H contains FIT-pair student rows with h@k>=tau; non-fired target rows are an "
                "irreducible zero-predictor term, matching the installed and SGD objectives"
            ),
            "behavioral_precondition": {"teacher_gap_ppl_gte": TEACHER_GAP_FLOOR_PPL},
            "gates": {
                "E4-G3": {"validation_mse_improvement_gte": G3_IMPROVEMENT_FLOOR},
                "E4-G4": {
                    "recovery_fraction_gte": G4_RECOVERY_FLOOR,
                    "bootstrap_ci95_low_gt": 0.0,
                },
                "E4-H": {"absolute_margin_points": E4_H_MARGIN_POINTS},
                "E4-G5": {
                    "wikitext_ppl_delta_percent_lte": G5_WIKITEXT_PPL_DELTA_CAP_PERCENT,
                    "code_fire_rate_lte": G5_CODE_FIRE_CAP,
                },
            },
        },
        "source_e3": {
            "manifest": str(E3_OUTPUT / "corpus_manifest.json"),
            "manifest_sha256": e2.sha256_file(E3_OUTPUT / "corpus_manifest.json"),
            "prepared": str(E3_OUTPUT / "prepared_windows.npz"),
            "prepared_file_sha256": e2.sha256_file(E3_OUTPUT / "prepared_windows.npz"),
            "prepared_content_sha256": source_manifest["prepared_windows"]["content_sha256"],
        },
        "source_e3_1": {
            "registered_sentence": decision["registered_sentence"],
            "fit_keys": str(E31_FIT_PATH),
            "fit_keys_sha256": e2.sha256_file(E31_FIT_PATH),
            "keys": str(E31_KEYS_PATH),
            "keys_sha256": e2.sha256_file(E31_KEYS_PATH),
            "artifact_manifest": str(E31_MANIFEST_PATH),
            "artifact_manifest_sha256": e2.sha256_file(E31_MANIFEST_PATH),
            "dependency_branch": decision["branch"],
        },
        "prepared_windows": {
            "path": str(prepared_path),
            "file_sha256": e2.sha256_file(prepared_path),
            "content_digest_algorithm": content["algorithm"],
            "content_sha256": content["sha256"],
            "arrays": content["arrays"],
        },
        **caveat_fields(),
    }
    e2.write_json(output / "corpus_manifest.json", manifest)
    key_path = output / "address" / "key_K4_fp32.npy"
    e2.save_npy(key_path, decision["key"])
    install = {
        "schema": "moe_e4_install_v1",
        "status": "complete_registered_fallback",
        "created_at": now_iso(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        **content_fields(manifest),
        "dependency_run": "20260829T230723Z-3971615",
        "e3_1_registered_sentence": decision["registered_sentence"],
        "dependency_branch": decision["branch"],
        "dependency_branch_code": decision["branch_code"],
        "install_row": decision["install_row"],
        "full_source_row_sha256": canonical_json_sha256(decision["row"]),
        "source": {
            "fit_keys": str(E31_FIT_PATH),
            "fit_keys_sha256": e2.sha256_file(E31_FIT_PATH),
            "keys": str(E31_KEYS_PATH),
            "keys_sha256": e2.sha256_file(E31_KEYS_PATH),
            "artifact_manifest": str(E31_MANIFEST_PATH),
            "artifact_manifest_sha256": e2.sha256_file(E31_MANIFEST_PATH),
        },
        "key": {
            "path": str(key_path),
            "shape": [HIDDEN_DIM],
            "dtype": "float32",
            "array_sha256": e2.sha256_array(decision["key"]),
            "file_sha256": e2.sha256_file(key_path),
        },
        "descriptive_policy": {
            "DOC-B firing": DOC_B_ROLE,
            "guides firing": GUIDES_ROLE,
            "reason": ADDRESS_CAVEAT,
        },
        **caveat_fields(),
        **script_fields(),
    }
    e2.write_json(output / "install.json", install)
    for arm in ARMS:
        initialize_pack(output, manifest, install, arm)
    write_lead_script(output)
    print(
        json.dumps(
            {
                "status": "complete",
                "content_sha256": content["sha256"],
                "install_layer": decision["layer"],
                "install_branch": decision["branch"],
                "address_caveat": ADDRESS_CAVEAT,
            }
        ),
        flush=True,
    )
    return 0


def load_prepared(output: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = output / "corpus_manifest.json"
    prepared_path = output / "prepared_windows.npz"
    manifest = e2.read_json(manifest_path)
    require_caveat(manifest, "E4 corpus manifest")
    if (
        manifest.get("schema") != "moe_e4_corpus_manifest_v1"
        or manifest.get("status") != "complete"
        or manifest.get("model_revision") != MODEL_REVISION
        or manifest.get("order_sha256") != e2.sha256_file(ORDER_PATH)
        or manifest.get("script_sha256") != e2.sha256_file(SCRIPT_PATH)
    ):
        raise RuntimeError("E4 corpus manifest contract failed")
    declared = manifest.get("prepared_windows", {})
    if Path(declared.get("path", "")).resolve() != prepared_path.resolve():
        raise RuntimeError("E4 prepared archive path linkage failed")
    if declared.get("file_sha256") != e2.sha256_file(prepared_path):
        raise RuntimeError("E4 prepared archive file SHA changed")
    with np.load(prepared_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    e3.validate_arrays(arrays)
    content = e3.prepared_content_record(arrays)
    if (
        declared.get("content_digest_algorithm") != content["algorithm"]
        or declared.get("content_sha256") != content["sha256"]
        or declared.get("arrays") != content["arrays"]
    ):
        raise RuntimeError("E4 prepared content provenance failed")
    source_manifest, source_arrays = e3.load_prepared(E3_OUTPUT)
    if (
        manifest.get("source_e3", {}).get("manifest_sha256")
        != e2.sha256_file(E3_OUTPUT / "corpus_manifest.json")
        or manifest.get("source_e3", {}).get("prepared_file_sha256")
        != e2.sha256_file(E3_OUTPUT / "prepared_windows.npz")
        or source_manifest["prepared_windows"]["content_sha256"] != content["sha256"]
    ):
        raise RuntimeError("E4 source E3 lineage changed")
    for name in arrays:
        if not np.array_equal(arrays[name], source_arrays[name]):
            raise RuntimeError(f"E4 prepared array differs from E3 source: {name}")
    return manifest, arrays


def validate_install(output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    path = output / "install.json"
    receipt = e2.read_json(path)
    require_caveat(receipt, "E4 install receipt")
    key_record = receipt.get("key", {})
    key_path = Path(key_record.get("path", ""))
    if (
        receipt.get("schema") != "moe_e4_install_v1"
        or receipt.get("status") != "complete_registered_fallback"
        or receipt.get("model_revision") != MODEL_REVISION
        or receipt.get("prepared_windows_content_sha256")
        != manifest["prepared_windows"]["content_sha256"]
        or receipt.get("dependency_branch_code")
        != "registered_fallback_best_e3_1_k4_fit_side_rank"
        or receipt.get("install_row", {}).get("layer") != 10
        or not key_path.is_file()
        or key_record.get("file_sha256") != e2.sha256_file(key_path)
        or receipt.get("script_sha256") != e2.sha256_file(SCRIPT_PATH)
    ):
        raise RuntimeError("E4 install receipt contract failed")
    key = np.load(key_path, allow_pickle=False)
    if (
        key.shape != (HIDDEN_DIM,)
        or key.dtype != np.float32
        or not np.isfinite(key).all()
        or key_record.get("array_sha256") != e2.sha256_array(key)
    ):
        raise RuntimeError("E4 installed key payload failed")
    source = source_install_decision()
    if (
        receipt["source"]["fit_keys_sha256"] != e2.sha256_file(E31_FIT_PATH)
        or receipt["source"]["keys_sha256"] != e2.sha256_file(E31_KEYS_PATH)
        or receipt["full_source_row_sha256"] != canonical_json_sha256(source["row"])
        or receipt["install_row"] != source["install_row"]
        or not np.array_equal(key, source["key"])
    ):
        raise RuntimeError("E4 install no longer matches the frozen E3.1 fallback")
    return receipt


def bringup(args: argparse.Namespace) -> int:
    output = ensure_output(args.output_dir)
    manifest, _arrays = load_prepared(output)
    validate_install(output, manifest)
    path = output / "bringup.json"
    if path.is_file():
        receipt = e2.read_json(path)
        require_caveat(receipt, "E4 bringup receipt")
        if (
            receipt.get("schema") != "moe_e4_bringup_v1"
            or receipt.get("gate", {}).get("verdict") != "GREEN"
            or receipt.get("prepared_windows_content_sha256")
            != manifest["prepared_windows"]["content_sha256"]
        ):
            raise RuntimeError("existing E4 bringup receipt failed")
        print(json.dumps({"status": "existing", "verdict": "GREEN"}), flush=True)
        return 0
    source_manifest, _source_arrays = e3.load_prepared(E3_OUTPUT)
    source = e3.require_bringup_green(E3_OUTPUT, source_manifest)
    source_path = E3_OUTPUT / "bringup.json"
    receipt = {
        "schema": "moe_e4_bringup_v1",
        "status": "complete",
        "created_at": now_iso(),
        "gate_name": "inherited E3-G-1 testbed",
        "gate": source["gate"],
        "short_512": source["short_512"],
        "long_2048": source["long_2048"],
        "long_minus_short_mean_nll": source["long_minus_short_mean_nll"],
        "inheritance": {
            "mode": "inherited_content_identical_E3_bringup",
            "source": str(source_path),
            "source_sha256": e2.sha256_file(source_path),
            "source_prepared_content_sha256": source_manifest["prepared_windows"]["content_sha256"],
        },
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        **content_fields(manifest),
        **caveat_fields(),
        **script_fields(),
    }
    e2.write_json(path, receipt)
    print(
        json.dumps(
            {
                "status": "complete",
                "verdict": receipt["gate"]["verdict"],
                "ppl512": receipt["short_512"]["ppl"],
                "nll_delta": receipt["long_minus_short_mean_nll"],
            }
        ),
        flush=True,
    )
    return 0


def require_bringup_green(output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    receipt = e2.read_json(output / "bringup.json")
    require_caveat(receipt, "E4 bringup receipt")
    if (
        receipt.get("schema") != "moe_e4_bringup_v1"
        or receipt.get("gate", {}).get("verdict") != "GREEN"
        or receipt.get("prepared_windows_content_sha256")
        != manifest["prepared_windows"]["content_sha256"]
        or receipt.get("inheritance", {}).get("source_sha256")
        != e2.sha256_file(E3_OUTPUT / "bringup.json")
    ):
        raise RuntimeError("content-valid E4 bringup is not GREEN; STOP")
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
        raise FileNotFoundError(f"missing E4 pair {pair_index}")
    receipt = e2.read_json(receipt_path)
    require_caveat(receipt, f"E4 pair {pair_index}")
    expected_split = "FIT" if pair_index < N_PAIR_TRAIN else "VALIDATION"
    if (
        receipt.get("schema") != "moe_e4_teacher_student_pair_v1"
        or receipt.get("status") != "complete"
        or receipt.get("pair_index") != pair_index
        or receipt.get("split") != expected_split
        or receipt.get("install_layer") != 10
        or receipt.get("model_revision") != MODEL_REVISION
        or not receipt.get("shared_window_token_ids_exact")
        or receipt.get("student_ids_sha256") != e2.sha256_array(arrays["pair_ids"][pair_index])
        or receipt.get("teacher_shared_ids_sha256") != e2.sha256_array(arrays["pair_ids"][pair_index])
        or receipt.get("teacher_prefix_ids_sha256")
        != e2.sha256_array(arrays["pair_prefix_ids"][pair_index])
        or receipt.get("prepared_windows_content_sha256")
        != manifest["prepared_windows"]["content_sha256"]
        or receipt.get("install_receipt_sha256") != e2.sha256_file(output / "install.json")
        or Path(receipt.get("pair_path", "")).resolve() != data_path.resolve()
        or receipt.get("pair_file_sha256") != e2.sha256_file(data_path)
        or receipt.get("script_sha256") != e2.sha256_file(SCRIPT_PATH)
    ):
        raise RuntimeError(f"E4 pair receipt contract failed: {receipt_path}")
    with np.load(data_path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    if set(data) != PAIR_ARRAY_NAMES:
        raise RuntimeError(f"E4 pair payload keys failed: {data_path}")
    for name, field in {
        "h_student_fp16": "student_router_input_sha256",
        "out_student_fp16": "student_block_output_sha256",
        "out_teacher_fp16": "teacher_block_output_sha256",
    }.items():
        value = data[name]
        if (
            value.shape != (WINDOW_TOKENS, HIDDEN_DIM)
            or value.dtype != np.float16
            or not np.isfinite(value).all()
            or receipt.get(field) != e2.sha256_array(value)
        ):
            raise RuntimeError(f"E4 pair payload array failed: {data_path}/{name}")
    return {
        "pair_index": pair_index,
        "split": expected_split,
        "receipt_path": str(receipt_path),
        "receipt_sha256": e2.sha256_file(receipt_path),
        "payload_path": str(data_path),
        "payload_sha256": e2.sha256_file(data_path),
        "receipt": receipt,
        "data": data,
    }


def capture_pairs(args: argparse.Namespace) -> int:
    if args.start_window < 0 or args.count < 1 or args.start_window + args.count > N_PAIR_WINDOWS:
        raise ValueError("capture-pairs range must stay within local pair indices 0..63")
    output = ensure_output(args.output_dir)
    manifest, arrays = load_prepared(output)
    require_bringup_green(output, manifest)
    install = validate_install(output, manifest)
    requested = list(range(args.start_window, args.start_window + args.count))
    missing: list[int] = []
    for index in requested:
        data_path, receipt_path = pair_paths(output, index)
        if data_path.is_file() or receipt_path.is_file():
            if not (data_path.is_file() and receipt_path.is_file()):
                raise RuntimeError(f"partial pair state for {index}")
            validate_pair(output, manifest, arrays, index)
        else:
            missing.append(index)
    if not missing:
        print(json.dumps({"status": "existing", "pairs": requested}), flush=True)
        return 0

    started = time.perf_counter()
    torch, model, runtime = e2.load_model(args)
    layer_index = int(install["install_row"]["layer"])
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
            raise RuntimeError(f"E4 pair hook did not execute exactly: {slot.keys()}")
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
                raise RuntimeError("E4 teacher context/shared-target contract failed")
            h_student, out_student = one_forward(student_ids)
            h_teacher_full, out_teacher_full = one_forward(teacher_ids)
            h_teacher = np.ascontiguousarray(h_teacher_full[-WINDOW_TOKENS:])
            out_teacher = np.ascontiguousarray(out_teacher_full[-WINDOW_TOKENS:])
            if (
                h_student.shape != (WINDOW_TOKENS, HIDDEN_DIM)
                or out_student.shape != h_student.shape
                or out_teacher.shape != h_student.shape
            ):
                raise RuntimeError("E4 pair activation shapes failed")
            delta = out_teacher.astype(np.float32) - out_student.astype(np.float32)
            hidden_difference = h_teacher.astype(np.float32) - h_student.astype(np.float32)
            if not np.isfinite(delta).all() or not np.isfinite(hidden_difference).all():
                raise RuntimeError("E4 pair capture contains non-finite values")
            data_path, receipt_path = pair_paths(output, pair_index)
            e2.save_npz(
                data_path,
                h_student_fp16=h_student,
                out_student_fp16=out_student,
                out_teacher_fp16=out_teacher,
            )
            split = "FIT" if pair_index < N_PAIR_TRAIN else "VALIDATION"
            receipt = {
                "schema": "moe_e4_teacher_student_pair_v1",
                "status": "complete",
                "created_at": now_iso(),
                "pair_index": pair_index,
                "doc_a_source_window": manifest["windows"]["pair"][pair_index],
                "split": split,
                "install_layer": layer_index,
                "install_branch": install["dependency_branch"],
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
                "install_receipt": str(output / "install.json"),
                "install_receipt_sha256": e2.sha256_file(output / "install.json"),
                **content_fields(manifest),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "runtime": runtime,
                "window_wall_seconds": time.perf_counter() - window_started,
                **caveat_fields(),
                **script_fields(),
            }
            e2.write_json(receipt_path, receipt)
            print(
                json.dumps(
                    {
                        "pair": pair_index,
                        "source_window": PAIR_DOC_WINDOWS[pair_index],
                        "split": split,
                        "zero_mse": receipt["target_delta_zero_predictor_mse"],
                        "wall_seconds": receipt["window_wall_seconds"],
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


def pair_provenance(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "pair_index": item["pair_index"],
            "split": item["split"],
            "receipt_path": item["receipt_path"],
            "receipt_sha256": item["receipt_sha256"],
            "payload_path": item["payload_path"],
            "payload_sha256": item["payload_sha256"],
        }
        for item in items
    ]


def fire_mask(hidden: np.ndarray, key: np.ndarray, tau: float) -> np.ndarray:
    scores = np.asarray(hidden, dtype=np.float32) @ np.asarray(key, dtype=np.float32)
    return np.ascontiguousarray(scores >= float(tau))


def ridge_trace_solve(
    gram_fit: np.ndarray,
    cross_fit: np.ndarray,
    gram_validation: np.ndarray,
    cross_validation: np.ndarray,
    validation_target_squared: float,
    validation_elements: int,
    alpha_grid: Sequence[float],
) -> dict[str, Any]:
    """Select ridge lambda by exact validation SSE trace identities."""

    gram_fit = np.asarray(gram_fit, dtype=np.float64)
    cross_fit = np.asarray(cross_fit, dtype=np.float64)
    gram_validation = np.asarray(gram_validation, dtype=np.float64)
    cross_validation = np.asarray(cross_validation, dtype=np.float64)
    dimension = int(gram_fit.shape[0])
    if (
        gram_fit.shape != (dimension, dimension)
        or cross_fit.shape != (dimension, dimension)
        or gram_validation.shape != (dimension, dimension)
        or cross_validation.shape != (dimension, dimension)
        or validation_elements <= 0
    ):
        raise ValueError("ridge trace inputs have incompatible contracts")
    started = time.perf_counter()
    gram_fit = (gram_fit + gram_fit.T) * 0.5
    gram_validation = (gram_validation + gram_validation.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(gram_fit)
    maximum = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(eigenvalues.min()) < -maximum * 1.0e-8:
        raise RuntimeError("fit Gram matrix has materially negative eigenvalues")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    ridge_scale = float(np.trace(gram_fit) / dimension)
    if not math.isfinite(ridge_scale) or ridge_scale <= 0.0:
        raise RuntimeError("ridge scale is non-positive")
    rotated_cross = eigenvectors.T @ cross_fit
    rotated_validation_cross = eigenvectors.T @ cross_validation
    rotated_validation_gram = eigenvectors.T @ gram_validation @ eigenvectors
    row_cross = np.sum(rotated_cross * rotated_validation_cross, axis=1, dtype=np.float64)
    row_gram = rotated_cross @ rotated_cross.T
    zero_mse = float(validation_target_squared / validation_elements)
    candidates: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        ridge_lambda = float(alpha * ridge_scale)
        inverse = 1.0 / (eigenvalues + ridge_lambda)
        cross_term = float(np.dot(inverse, row_cross))
        prediction_squared = float(
            np.sum(
                rotated_validation_gram
                * row_gram.T
                * inverse[:, None]
                * inverse[None, :],
                dtype=np.float64,
            )
        )
        squared_error = validation_target_squared - 2.0 * cross_term + prediction_squared
        tolerance = max(1.0, validation_target_squared) * 1.0e-10
        if squared_error < 0.0 and abs(squared_error) <= tolerance:
            squared_error = 0.0
        if squared_error < 0.0 or not math.isfinite(squared_error):
            raise RuntimeError(f"ridge trace produced invalid SSE for alpha={alpha}")
        mse = float(squared_error / validation_elements)
        improvement = (zero_mse - mse) / zero_mse if zero_mse > 0.0 else None
        candidates.append(
            {
                "alpha": float(alpha),
                "lambda": ridge_lambda,
                "validation_mse": mse,
                "zero_predictor_mse": zero_mse,
                "improvement_fraction": improvement,
                "improvement_percent": None if improvement is None else improvement * 100.0,
                "cross_term": cross_term,
                "prediction_squared": prediction_squared,
            }
        )
    selected = min(candidates, key=lambda row: (row["validation_mse"], -row["alpha"]))
    inverse = 1.0 / (eigenvalues + selected["lambda"])
    coefficient = eigenvectors @ (inverse[:, None] * rotated_cross)
    if coefficient.shape != (dimension, dimension) or not np.isfinite(coefficient).all():
        raise RuntimeError("selected ridge coefficient is invalid")
    return {
        "coefficient": np.ascontiguousarray(coefficient),
        "eigenvalues": np.ascontiguousarray(eigenvalues),
        "ridge_scale": ridge_scale,
        "candidates": candidates,
        "selected": selected,
        "solver": {
            "name": "symmetric eigendecomposition plus exact validation-SSE trace",
            "dimension": dimension,
            "fit_gram_min_eigenvalue_after_clip": float(eigenvalues.min()),
            "fit_gram_max_eigenvalue": float(eigenvalues.max()),
            "wall_seconds": time.perf_counter() - started,
        },
    }


def canonical_svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, singular, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64), full_matrices=False)
    anchors = np.argmax(np.abs(vt), axis=1)
    signs = np.sign(vt[np.arange(vt.shape[0]), anchors])
    signs[signs == 0.0] = 1.0
    vt = vt * signs[:, None]
    u = u * signs[None, :]
    return np.ascontiguousarray(u), np.ascontiguousarray(singular), np.ascontiguousarray(vt)


def factor_from_svd(
    u: np.ndarray, singular: np.ndarray, vt: np.ndarray, rank: int
) -> tuple[np.ndarray, np.ndarray]:
    roots = np.sqrt(np.asarray(singular[:rank], dtype=np.float64))
    a = roots[:, None] * np.asarray(vt[:rank], dtype=np.float64)
    b = np.asarray(u[:, :rank], dtype=np.float64) * roots[None, :]
    return np.ascontiguousarray(a), np.ascontiguousarray(b)


def bootstrap_window_ratio(
    numerator_by_window: Sequence[float],
    denominator_by_window: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    numerator = np.asarray(numerator_by_window, dtype=np.float64)
    denominator = np.asarray(denominator_by_window, dtype=np.float64)
    if numerator.shape != denominator.shape or numerator.ndim != 1 or numerator.size == 0:
        raise ValueError("bootstrap ratio requires equal nonempty vectors")
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
        "bootstrap_unit": "validation pairs",
        "bootstrap_resamples": resamples,
    }


def numpy_validation_metrics(
    validated_pairs: Sequence[dict[str, Any]],
    key: np.ndarray,
    tau: float,
    *,
    activation: str,
    A: np.ndarray | None = None,
    B: np.ndarray | None = None,
    W: np.ndarray | None = None,
    start_index: int = N_PAIR_TRAIN,
) -> dict[str, Any]:
    if activation not in {"identity", "silu", "direct"}:
        raise ValueError(f"unsupported validation activation {activation}")
    per_window: list[dict[str, Any]] = []
    total_squared = 0.0
    total_zero = 0.0
    total_elements = 0
    total_fires = 0
    for index in range(start_index, N_PAIR_WINDOWS):
        data = validated_pairs[index]["data"]
        hidden = data["h_student_fp16"].astype(np.float32)
        target = data["out_teacher_fp16"].astype(np.float32) - data["out_student_fp16"].astype(np.float32)
        fired = fire_mask(hidden, key, tau)
        prediction = np.zeros_like(target, dtype=np.float32)
        if fired.any():
            selected = hidden[fired]
            if activation == "direct":
                if W is None:
                    raise ValueError("direct validation requires W")
                prediction[fired] = selected @ np.asarray(W, dtype=np.float32).T
            else:
                if A is None or B is None:
                    raise ValueError("factor validation requires A/B")
                z = selected @ np.asarray(A, dtype=np.float32).T
                if activation == "silu":
                    z = z / (1.0 + np.exp(-z))
                prediction[fired] = z @ np.asarray(B, dtype=np.float32).T
        difference = prediction.astype(np.float64) - target.astype(np.float64)
        squared = float(np.sum(difference**2, dtype=np.float64))
        zero = float(np.sum(target.astype(np.float64) ** 2, dtype=np.float64))
        elements = int(target.size)
        total_squared += squared
        total_zero += zero
        total_elements += elements
        total_fires += int(fired.sum())
        per_window.append(
            {
                "pair_index": index,
                "source_doc_window": PAIR_DOC_WINDOWS[index],
                "adapter_mse": squared / elements,
                "zero_predictor_mse": zero / elements,
                "fire_count": int(fired.sum()),
                "token_count": WINDOW_TOKENS,
            }
        )
    mse = total_squared / total_elements
    zero_mse = total_zero / total_elements
    improvement = (zero_mse - mse) / zero_mse if zero_mse > 0.0 else None
    return {
        "zero_predictor_mse": zero_mse,
        "adapter_mse": mse,
        "improvement_fraction": improvement,
        "improvement_percent": None if improvement is None else improvement * 100.0,
        "validation_tokens": total_elements // HIDDEN_DIM,
        "validation_fire_count": total_fires,
        "per_window": per_window,
    }


def completed_pack(
    output: Path,
    manifest: dict[str, Any],
    install: dict[str, Any],
    arm: str,
    training_path: Path,
    artifacts: dict[str, dict[str, Any]],
    g3_metrics: dict[str, Any],
    extra_components: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = pack_path(output, arm)
    pack = e2.read_json(path)
    require_caveat(pack, f"{ARM_NAMES[arm]} ExpertPack")
    pack.update(
        {
            "status": "complete_g3_green" if g3_metrics["verdict"] == "GREEN" else "complete_g3_red",
            "completed_at": now_iso(),
            "components": {
                "key": install["key"],
                "A": {"status": "complete", **artifacts["A"]},
                "B": {"status": "complete", **artifacts["B"]},
                **(extra_components or {}),
            },
            "g3": g3_metrics,
            "limitations": ["ABI and behavioral E4-G4/G5 evaluation pending"],
        }
    )
    pack["provenance"]["construction_receipt"] = str(training_path)
    pack["provenance"]["construction_receipt_sha256"] = e2.sha256_file(training_path)
    e2.write_json(path, pack)
    return pack


def fit_cf(args: argparse.Namespace) -> int:
    if args.rank != RANK or args.bootstrap_resamples < DEFAULT_BOOTSTRAP_RESAMPLES or args.threads < 1:
        raise ValueError("ARM-CF requires registered rank 64, >=2000 bootstraps, and >=1 thread")
    output = ensure_output(args.output_dir)
    manifest, arrays = load_prepared(output)
    require_bringup_green(output, manifest)
    install = validate_install(output, manifest)
    receipt_path = output / TRAINING_FILES["cf"]
    if receipt_path.is_file():
        receipt = validate_arm_training(output, manifest, arrays, "cf")
        print(json.dumps({"status": "existing", "arm": "ARM-CF", "verdict": receipt["verdict"]}), flush=True)
        return 0
    started = time.perf_counter()
    validated = [validate_pair(output, manifest, arrays, index) for index in range(N_PAIR_WINDOWS)]
    key = np.load(install["key"]["path"], allow_pickle=False)
    tau = float(install["install_row"]["tau"])

    h_fit_blocks: list[np.ndarray] = []
    d_fit_blocks: list[np.ndarray] = []
    h_validation_blocks: list[np.ndarray] = []
    d_validation_blocks: list[np.ndarray] = []
    validation_target_squared = 0.0
    validation_elements = 0
    split_fire: dict[str, int] = {"FIT": 0, "VALIDATION": 0}
    split_tokens: dict[str, int] = {"FIT": 0, "VALIDATION": 0}
    for index, item in enumerate(validated):
        data = item["data"]
        hidden = data["h_student_fp16"].astype(np.float32)
        delta = data["out_teacher_fp16"].astype(np.float32) - data["out_student_fp16"].astype(np.float32)
        fired = fire_mask(hidden, key, tau)
        split = "FIT" if index < N_PAIR_TRAIN else "VALIDATION"
        split_fire[split] += int(fired.sum())
        split_tokens[split] += int(fired.size)
        if index < N_PAIR_TRAIN:
            h_fit_blocks.append(np.ascontiguousarray(hidden[fired]))
            d_fit_blocks.append(np.ascontiguousarray(delta[fired]))
        else:
            h_validation_blocks.append(np.ascontiguousarray(hidden[fired]))
            d_validation_blocks.append(np.ascontiguousarray(delta[fired]))
            validation_target_squared += float(np.sum(delta.astype(np.float64) ** 2, dtype=np.float64))
            validation_elements += int(delta.size)
    h_fit = np.ascontiguousarray(np.concatenate(h_fit_blocks, axis=0), dtype=np.float64)
    d_fit = np.ascontiguousarray(np.concatenate(d_fit_blocks, axis=0), dtype=np.float64)
    h_validation = np.ascontiguousarray(np.concatenate(h_validation_blocks, axis=0), dtype=np.float64)
    d_validation = np.ascontiguousarray(np.concatenate(d_validation_blocks, axis=0), dtype=np.float64)
    del h_fit_blocks, d_fit_blocks, h_validation_blocks, d_validation_blocks
    if h_fit.shape[0] == 0 or h_validation.shape[0] == 0:
        raise RuntimeError("frozen fallback gate has no fired FIT or validation rows")
    gram_fit = h_fit.T @ h_fit
    cross_fit = h_fit.T @ d_fit
    gram_validation = h_validation.T @ h_validation
    cross_validation = h_validation.T @ d_validation
    del h_fit, d_fit, h_validation, d_validation
    solve = ridge_trace_solve(
        gram_fit,
        cross_fit,
        gram_validation,
        cross_validation,
        validation_target_squared,
        validation_elements,
        RIDGE_ALPHA_GRID,
    )
    del gram_fit, cross_fit, gram_validation, cross_validation
    coefficient = solve.pop("coefficient")
    # coefficient maps row h to row delta; the order's W maps column h to column delta.
    W = np.ascontiguousarray(coefficient.T)
    del coefficient
    svd_started = time.perf_counter()
    u, singular, vt = canonical_svd(W)
    A64, B64 = factor_from_svd(u, singular, vt, RANK)
    reconstruction = B64 @ A64
    if reconstruction.shape != W.shape or not np.isfinite(reconstruction).all():
        raise RuntimeError("rank-64 SVD reconstruction failed")
    total_energy = float(np.sum(singular**2, dtype=np.float64))
    rank_energy = float(np.sum(singular[:RANK] ** 2, dtype=np.float64))
    energy_fraction = rank_energy / total_energy if total_energy > 0.0 else 0.0
    alignments = [
        {
            "direction": index + 1,
            "singular_value": float(singular[index]),
            "cosine_vs_gate_key": float(np.dot(vt[index], key.astype(np.float64))),
            "absolute_cosine_vs_gate_key": float(abs(np.dot(vt[index], key.astype(np.float64)))),
            "energy_fraction_individual": float(singular[index] ** 2 / total_energy) if total_energy > 0.0 else 0.0,
        }
        for index in range(RANK)
    ]
    A_fp16 = np.ascontiguousarray(A64.astype(np.float16))
    B_fp16 = np.ascontiguousarray(B64.astype(np.float16))
    W_fp16 = np.ascontiguousarray(W.astype(np.float16))
    V_fp16 = np.ascontiguousarray(vt[:RANK].astype(np.float16))
    for artifact_name, artifact_value in {
        "A_fp16": A_fp16,
        "B_fp16": B_fp16,
        "W_full_fp16": W_fp16,
        "singular_values_fp64": singular,
        "right_singular_vectors_r64_fp16": V_fp16,
    }.items():
        require_finite_array(artifact_name, artifact_value)
    pack_dir = pack_path(output, "cf").parent
    artifacts_paths = {
        "A": pack_dir / "A_fp16.npy",
        "B": pack_dir / "B_fp16.npy",
        "W_full": pack_dir / "W_full_fp16.npy",
        "singular_values": pack_dir / "singular_values_fp64.npy",
        "right_singular_vectors": pack_dir / "right_singular_vectors_r64_fp16.npy",
    }
    e2.save_npy(artifacts_paths["A"], A_fp16)
    e2.save_npy(artifacts_paths["B"], B_fp16)
    e2.save_npy(artifacts_paths["W_full"], W_fp16)
    e2.save_npy(artifacts_paths["singular_values"], singular)
    e2.save_npy(artifacts_paths["right_singular_vectors"], V_fp16)
    rank_metrics = numpy_validation_metrics(
        validated, key, tau, activation="identity", A=A_fp16, B=B_fp16
    )
    full_metrics = numpy_validation_metrics(
        validated, key, tau, activation="direct", W=W_fp16
    )
    rank_ci = bootstrap_window_ratio(
        [row["zero_predictor_mse"] - row["adapter_mse"] for row in rank_metrics["per_window"]],
        [row["zero_predictor_mse"] for row in rank_metrics["per_window"]],
        resamples=args.bootstrap_resamples,
        seed=args.seed + 300,
    )
    full_ci = bootstrap_window_ratio(
        [row["zero_predictor_mse"] - row["adapter_mse"] for row in full_metrics["per_window"]],
        [row["zero_predictor_mse"] for row in full_metrics["per_window"]],
        resamples=args.bootstrap_resamples,
        seed=args.seed + 301,
    )
    green = bool(
        rank_metrics["improvement_fraction"] is not None
        and rank_metrics["improvement_fraction"] >= G3_IMPROVEMENT_FLOOR
    )
    inspectability_path = output / "cf_inspectability.json"
    inspectability = {
        "schema": "moe_e4_cf_inspectability_v1",
        "status": "complete",
        "created_at": now_iso(),
        "arm": "ARM-CF",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        **content_fields(manifest),
        "install_layer": install["install_row"]["layer"],
        "rank": RANK,
        "selected_alpha": solve["selected"]["alpha"],
        "selected_lambda": solve["selected"]["lambda"],
        "top_8_singular_values": [float(value) for value in singular[:8]],
        "energy_fraction_at_rank_64": energy_fraction,
        "total_spectral_energy": total_energy,
        "rank_64_spectral_energy": rank_energy,
        "direction_key_alignments": alignments,
        "svd_sign_convention": "largest-absolute right-vector coordinate is positive",
        "artifacts": {
            name: {
                "path": str(path),
                "sha256": e2.sha256_file(path),
            }
            for name, path in artifacts_paths.items()
        },
        **caveat_fields(),
        **script_fields(),
    }
    e2.write_json(inspectability_path, inspectability)
    artifact_records = {
        "A": {
            "path": str(artifacts_paths["A"]),
            "shape": list(A_fp16.shape),
            "storage_dtype": "float16",
            "sha256": e2.sha256_file(artifacts_paths["A"]),
        },
        "B": {
            "path": str(artifacts_paths["B"]),
            "shape": list(B_fp16.shape),
            "storage_dtype": "float16",
            "sha256": e2.sha256_file(artifacts_paths["B"]),
        },
    }
    receipt = {
        "schema": "moe_e4_cf_fit_v1",
        "status": "complete_g3_green" if green else "complete_g3_red",
        "created_at": now_iso(),
        "arm": "ARM-CF",
        "gate": "E4-G3",
        "verdict": "GREEN" if green else "RED",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        **content_fields(manifest),
        "install_layer": install["install_row"]["layer"],
        "rank": RANK,
        "tau": tau,
        "activation": "identity",
        "formula": "W = Delta^T H (H^T H + lambda I)^-1; delta_row = h_row @ W^T",
        "fit_rows": "frozen-gate-active student hidden rows from pair indices 0..47",
        "lambda_grid": list(RIDGE_ALPHA_GRID),
        "lambda_selection": {
            "rule": "minimum untruncated validation MSE, then stronger shrinkage",
            "behavioral_heldout_used": False,
            "ridge_scale": solve["ridge_scale"],
            "candidates": solve["candidates"],
            "selected": solve["selected"],
        },
        "solver": solve["solver"],
        "gate_rows": {
            "fit_fire_count": split_fire["FIT"],
            "fit_token_count": split_tokens["FIT"],
            "fit_fire_rate": split_fire["FIT"] / split_tokens["FIT"],
            "validation_fire_count": split_fire["VALIDATION"],
            "validation_token_count": split_tokens["VALIDATION"],
            "validation_fire_rate": split_fire["VALIDATION"] / split_tokens["VALIDATION"],
        },
        "stored_fp16_rank64_validation": {
            **rank_metrics,
            "improvement_ci95_fraction": rank_ci,
            "required_improvement_fraction": G3_IMPROVEMENT_FLOOR,
        },
        "arm_cf_full_descriptive": {
            "role": "DESCRIPTIVE ceiling",
            "selected_float64_trace_validation": solve["selected"],
            "stored_fp16_validation": {**full_metrics, "improvement_ci95_fraction": full_ci},
        },
        "spectrum": {
            "top_8_singular_values": [float(value) for value in singular[:8]],
            "energy_fraction_at_rank_64": energy_fraction,
            "svd_wall_seconds": time.perf_counter() - svd_started,
        },
        "inspectability_receipt": str(inspectability_path),
        "inspectability_receipt_sha256": e2.sha256_file(inspectability_path),
        "data": {
            "fit_pair_indices": list(range(N_PAIR_TRAIN)),
            "validation_pair_indices": list(range(N_PAIR_TRAIN, N_PAIR_WINDOWS)),
            "pair_receipts": pair_provenance(validated),
        },
        "artifacts": {
            **artifact_records,
            "W_full": {
                "path": str(artifacts_paths["W_full"]),
                "shape": list(W_fp16.shape),
                "storage_dtype": "float16",
                "sha256": e2.sha256_file(artifacts_paths["W_full"]),
            },
            "singular_values": {
                "path": str(artifacts_paths["singular_values"]),
                "shape": list(singular.shape),
                "storage_dtype": "float64",
                "sha256": e2.sha256_file(artifacts_paths["singular_values"]),
            },
            "right_singular_vectors": {
                "path": str(artifacts_paths["right_singular_vectors"]),
                "shape": list(V_fp16.shape),
                "storage_dtype": "float16",
                "sha256": e2.sha256_file(artifacts_paths["right_singular_vectors"]),
            },
        },
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "threads_requested": args.threads,
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_bytes": e2.peak_rss_bytes(),
        },
        **caveat_fields(),
        **script_fields(),
    }
    e2.write_json(receipt_path, receipt)
    completed_pack(
        output,
        manifest,
        install,
        "cf",
        receipt_path,
        artifact_records,
        {"verdict": receipt["verdict"], **rank_metrics},
        extra_components={
            "W_full": {"status": "complete_descriptive", **receipt["artifacts"]["W_full"]},
            "singular_values": {"status": "complete", **receipt["artifacts"]["singular_values"]},
            "right_singular_vectors": {
                "status": "complete",
                **receipt["artifacts"]["right_singular_vectors"],
            },
            "inspectability": {
                "status": "complete",
                "path": str(inspectability_path),
                "sha256": e2.sha256_file(inspectability_path),
            },
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "arm": "ARM-CF",
                "gate": "E4-G3",
                "verdict": receipt["verdict"],
                "improvement_percent": rank_metrics["improvement_percent"],
                "selected_lambda": solve["selected"]["lambda"],
                "energy_fraction_at_rank_64": energy_fraction,
            }
        ),
        flush=True,
    )
    del W, u, singular, vt, A64, B64, reconstruction
    gc.collect()
    return 0


def train_sgd(args: argparse.Namespace) -> int:
    if args.rank != RANK:
        raise ValueError("ARM-SGD rank is registered at 64")
    if not 1 <= args.train_tokens_per_window <= WINDOW_TOKENS:
        raise ValueError("--train-tokens-per-window must be within 1..512")
    if (
        args.threads < 1
        or args.token_batch_size < 1
        or args.epochs < 1
        or args.patience < 1
        or args.bootstrap_resamples < DEFAULT_BOOTSTRAP_RESAMPLES
    ):
        raise ValueError("ARM-SGD resources/bootstraps are below registered minima")
    output = ensure_output(args.output_dir)
    manifest, arrays = load_prepared(output)
    require_bringup_green(output, manifest)
    install = validate_install(output, manifest)
    receipt_path = output / TRAINING_FILES["sgd"]
    if receipt_path.is_file():
        receipt = validate_arm_training(output, manifest, arrays, "sgd")
        print(json.dumps({"status": "existing", "arm": "ARM-SGD", "verdict": receipt["verdict"]}), flush=True)
        return 0
    started = time.perf_counter()
    torch = e2.configure_torch(args.threads, cuda_possible=False)
    import torch.nn as nn
    import torch.nn.functional as functional

    key = np.load(install["key"]["path"], allow_pickle=False)
    key_tensor = torch.from_numpy(np.ascontiguousarray(key))
    tau = float(install["install_row"]["tau"])

    class Adapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.A = nn.Parameter(torch.empty(RANK, HIDDEN_DIM, dtype=torch.float32))
            self.B = nn.Parameter(torch.zeros(HIDDEN_DIM, RANK, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        def forward(self, value: Any) -> Any:
            return functional.linear(functional.silu(functional.linear(value, self.A)), self.B)

    torch.manual_seed(args.seed)
    adapter = Adapter()
    initial_B = adapter.B.detach().numpy().copy()
    if not np.array_equal(initial_B, np.zeros_like(initial_B)):
        raise RuntimeError("ARM-SGD B zero initialization failed")
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    validated = [validate_pair(output, manifest, arrays, index) for index in range(N_PAIR_WINDOWS)]

    def evaluate_validation(model: Any) -> tuple[float, float, int, int, list[dict[str, Any]]]:
        model.eval()
        squared = 0.0
        zero_squared = 0.0
        elements = 0
        fired_count = 0
        per_window: list[dict[str, Any]] = []
        with torch.no_grad():
            for index in range(N_PAIR_TRAIN, N_PAIR_WINDOWS):
                data = validated[index]["data"]
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
        return squared / elements, zero_squared / elements, fired_count, elements // HIDDEN_DIM, per_window

    zero_validation_mse, zero_again, _fires, validation_tokens, _rows = evaluate_validation(adapter)
    if zero_validation_mse != zero_again:
        raise RuntimeError("zero-initialized ARM-SGD did not equal zero predictor")
    best_mse = zero_validation_mse
    best_epoch = 0
    best_state = {name: tensor.detach().clone() for name, tensor in adapter.state_dict().items()}
    epochs_without_improvement = 0
    epochs: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        rng = np.random.default_rng(args.seed + epoch * 100003)
        order = rng.permutation(N_PAIR_TRAIN).tolist()
        epoch_squared = 0.0
        epoch_elements = 0
        epoch_fires = 0
        for index in order:
            data = validated[index]["data"]
            hidden_np = data["h_student_fp16"].astype(np.float32)
            target_np = data["out_teacher_fp16"].astype(np.float32) - data["out_student_fp16"].astype(np.float32)
            if args.train_tokens_per_window < WINDOW_TOKENS:
                indices = np.sort(
                    rng.choice(WINDOW_TOKENS, size=args.train_tokens_per_window, replace=False)
                )
                hidden_np = hidden_np[indices]
                target_np = target_np[indices]
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
        validation_mse, _zero, validation_fires, _tokens, _per_window = evaluate_validation(adapter)
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
        epochs.append(record)
        print(json.dumps(record), flush=True)
        if epochs_without_improvement >= args.patience:
            break
    adapter.load_state_dict(best_state)
    A_fp16 = np.ascontiguousarray(adapter.A.detach().numpy().astype(np.float16))
    B_fp16 = np.ascontiguousarray(adapter.B.detach().numpy().astype(np.float16))
    require_finite_array("ARM-SGD A_fp16", A_fp16)
    require_finite_array("ARM-SGD B_fp16", B_fp16)
    pack_dir = pack_path(output, "sgd").parent
    A_path = pack_dir / "A_fp16.npy"
    B_path = pack_dir / "B_fp16.npy"
    e2.save_npy(A_path, A_fp16)
    e2.save_npy(B_path, B_fp16)
    with torch.no_grad():
        adapter.A.copy_(torch.from_numpy(A_fp16.astype(np.float32)))
        adapter.B.copy_(torch.from_numpy(B_fp16.astype(np.float32)))
    stored_mse, stored_zero, stored_fires, _tokens, per_window = evaluate_validation(adapter)
    improvement = (stored_zero - stored_mse) / stored_zero if stored_zero > 0.0 else None
    improvement_ci = bootstrap_window_ratio(
        [row["zero_predictor_mse"] - row["adapter_mse"] for row in per_window],
        [row["zero_predictor_mse"] for row in per_window],
        resamples=args.bootstrap_resamples,
        seed=args.seed + 310,
    )
    green = bool(improvement is not None and improvement >= G3_IMPROVEMENT_FLOOR)
    artifacts = {
        "A": {
            "path": str(A_path),
            "shape": list(A_fp16.shape),
            "storage_dtype": "float16",
            "sha256": e2.sha256_file(A_path),
        },
        "B": {
            "path": str(B_path),
            "shape": list(B_fp16.shape),
            "storage_dtype": "float16",
            "sha256": e2.sha256_file(B_path),
        },
    }
    stored = {
        "zero_predictor_mse": stored_zero,
        "adapter_mse": stored_mse,
        "improvement_fraction": improvement,
        "improvement_percent": None if improvement is None else improvement * 100.0,
        "improvement_ci95_fraction": improvement_ci,
        "required_improvement_fraction": G3_IMPROVEMENT_FLOOR,
        "validation_tokens": validation_tokens,
        "validation_fire_count": stored_fires,
        "per_window": per_window,
    }
    receipt = {
        "schema": "moe_e4_sgd_training_v1",
        "status": "complete_g3_green" if green else "complete_g3_red",
        "created_at": now_iso(),
        "arm": "ARM-SGD",
        "gate": "E4-G3",
        "verdict": "GREEN" if green else "RED",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        **content_fields(manifest),
        "install_layer": install["install_row"]["layer"],
        "rank": RANK,
        "tau": tau,
        "activation": "silu",
        "initialization": {
            "A": "torch kaiming_uniform, deterministic CPU seed",
            "B": "exact all-zero float32",
            "B_zero_exact": True,
            "B_initial_sha256": e2.sha256_array(initial_B),
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "token_batch_size": args.token_batch_size,
            "maximum_epochs": args.epochs,
            "patience": args.patience,
            "early_stopping_metric": "validation-pair full-token frozen-gate MSE",
            "train_tokens_sampled_per_window_per_epoch": args.train_tokens_per_window,
        },
        "epochs": epochs,
        "best_epoch_float32": best_epoch,
        "best_validation_mse_float32": best_mse,
        "stored_fp16_validation": stored,
        "data": {
            "fit_pair_indices": list(range(N_PAIR_TRAIN)),
            "validation_pair_indices": list(range(N_PAIR_TRAIN, N_PAIR_WINDOWS)),
            "pair_receipts": pair_provenance(validated),
        },
        "artifacts": artifacts,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "threads": args.threads,
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_bytes": e2.peak_rss_bytes(),
        },
        **caveat_fields(),
        **script_fields(),
    }
    e2.write_json(receipt_path, receipt)
    completed_pack(
        output,
        manifest,
        install,
        "sgd",
        receipt_path,
        artifacts,
        {"verdict": receipt["verdict"], **stored},
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "arm": "ARM-SGD",
                "gate": "E4-G3",
                "verdict": receipt["verdict"],
                "improvement_percent": stored["improvement_percent"],
                "best_epoch": best_epoch,
            }
        ),
        flush=True,
    )
    return 0


def validate_arm_training(
    output: Path,
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    arm: str,
) -> dict[str, Any]:
    receipt_path = output / TRAINING_FILES[arm]
    receipt = e2.read_json(receipt_path)
    require_caveat(receipt, f"{ARM_NAMES[arm]} construction receipt")
    expected_schema = "moe_e4_cf_fit_v1" if arm == "cf" else "moe_e4_sgd_training_v1"
    if (
        receipt.get("schema") != expected_schema
        or receipt.get("status") not in {"complete_g3_green", "complete_g3_red"}
        or receipt.get("arm") != ARM_NAMES[arm]
        or receipt.get("model_revision") != MODEL_REVISION
        or receipt.get("prepared_windows_content_sha256")
        != manifest["prepared_windows"]["content_sha256"]
        or receipt.get("install_layer") != 10
        or receipt.get("rank") != RANK
        or receipt.get("script_sha256") != e2.sha256_file(SCRIPT_PATH)
    ):
        raise RuntimeError(f"{ARM_NAMES[arm]} construction receipt failed")
    recorded_pairs = receipt.get("data", {}).get("pair_receipts", [])
    if len(recorded_pairs) != N_PAIR_WINDOWS:
        raise RuntimeError(f"{ARM_NAMES[arm]} does not bind all 64 pairs")
    for index in range(N_PAIR_WINDOWS):
        observed = validate_pair(output, manifest, arrays, index)
        recorded = recorded_pairs[index]
        for field in ("pair_index", "split", "receipt_path", "receipt_sha256", "payload_path", "payload_sha256"):
            if recorded.get(field) != observed.get(field):
                raise RuntimeError(f"{ARM_NAMES[arm]} pair linkage changed: {index}/{field}")
    for name in ("A", "B"):
        record = receipt.get("artifacts", {}).get(name, {})
        path = Path(record.get("path", ""))
        if not path.is_file() or record.get("sha256") != e2.sha256_file(path):
            raise RuntimeError(f"{ARM_NAMES[arm]} artifact linkage failed: {name}")
    if arm == "cf":
        for name in ("W_full", "singular_values", "right_singular_vectors"):
            record = receipt.get("artifacts", {}).get(name, {})
            path = Path(record.get("path", ""))
            if not path.is_file() or record.get("sha256") != e2.sha256_file(path):
                raise RuntimeError(f"ARM-CF artifact linkage failed: {name}")
        inspect_path = Path(receipt.get("inspectability_receipt", ""))
        if (
            not inspect_path.is_file()
            or receipt.get("inspectability_receipt_sha256") != e2.sha256_file(inspect_path)
        ):
            raise RuntimeError("ARM-CF inspectability linkage failed")
        inspect = e2.read_json(inspect_path)
        require_caveat(inspect, "ARM-CF inspectability receipt")
    pack = e2.read_json(pack_path(output, arm))
    require_caveat(pack, f"{ARM_NAMES[arm]} ExpertPack")
    if (
        pack.get("schema") != PACK_NAMES[arm]
        or pack.get("arm") != ARM_NAMES[arm]
        or pack.get("layer") != 10
        or pack.get("rank") != RANK
        or pack.get("activation") != ARM_ACTIVATIONS[arm]
        or pack.get("provenance", {}).get("construction_receipt") != str(receipt_path)
        or pack.get("provenance", {}).get("construction_receipt_sha256") != e2.sha256_file(receipt_path)
    ):
        raise RuntimeError(f"{ARM_NAMES[arm]} ExpertPack contract failed")
    for name in ("A", "B"):
        if pack["components"][name].get("sha256") != receipt["artifacts"][name]["sha256"]:
            raise RuntimeError(f"{ARM_NAMES[arm]} ExpertPack differs from receipt: {name}")
    return receipt


def load_arm(
    output: Path,
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    arm: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    validate_arm_training(output, manifest, arrays, arm)
    pack = e2.read_json(pack_path(output, arm))
    key = np.load(pack["components"]["key"]["path"], allow_pickle=False)
    A = np.load(pack["components"]["A"]["path"], allow_pickle=False)
    B = np.load(pack["components"]["B"]["path"], allow_pickle=False)
    if key.shape != (HIDDEN_DIM,) or key.dtype != np.float32:
        raise RuntimeError(f"{ARM_NAMES[arm]} key contract failed")
    if A.shape != (RANK, HIDDEN_DIM) or A.dtype != np.float16:
        raise RuntimeError(f"{ARM_NAMES[arm]} A contract failed")
    if B.shape != (HIDDEN_DIM, RANK) or B.dtype != np.float16:
        raise RuntimeError(f"{ARM_NAMES[arm]} B contract failed")
    return pack, np.ascontiguousarray(key), np.ascontiguousarray(A), np.ascontiguousarray(B)


def load_pack_binding(output: Path, arm: str) -> dict[str, Any]:
    """Validate immutable eval-facing pack links without re-reading all 64 pairs."""

    training_path = output / TRAINING_FILES[arm]
    pack = e2.read_json(pack_path(output, arm))
    receipt = e2.read_json(training_path)
    require_caveat(pack, f"{ARM_NAMES[arm]} ExpertPack")
    require_caveat(receipt, f"{ARM_NAMES[arm]} construction receipt")
    expected_schema = "moe_e4_cf_fit_v1" if arm == "cf" else "moe_e4_sgd_training_v1"
    if (
        pack.get("schema") != PACK_NAMES[arm]
        or pack.get("arm") != ARM_NAMES[arm]
        or pack.get("layer") != 10
        or pack.get("rank") != RANK
        or pack.get("activation") != ARM_ACTIVATIONS[arm]
        or receipt.get("schema") != expected_schema
        or receipt.get("status") not in {"complete_g3_green", "complete_g3_red"}
        or receipt.get("model_revision") != MODEL_REVISION
        or pack.get("provenance", {}).get("construction_receipt") != str(training_path)
        or pack.get("provenance", {}).get("construction_receipt_sha256")
        != e2.sha256_file(training_path)
    ):
        raise RuntimeError(f"{ARM_NAMES[arm]} eval-facing ExpertPack binding failed")
    for name in ("key", "A", "B"):
        record = pack.get("components", {}).get(name, {})
        path_key = "path"
        hash_key = "file_sha256" if name == "key" else "sha256"
        path = Path(record.get(path_key, ""))
        if not path.is_file() or record.get(hash_key) != e2.sha256_file(path):
            raise RuntimeError(f"{ARM_NAMES[arm]} eval-facing component failed: {name}")
    return pack


class E4ExpertMount:
    """Frozen-threshold side path supporting linear CF and SiLU SGD arms."""

    def __init__(
        self,
        module: Any,
        *,
        torch_module: Any,
        key: np.ndarray,
        tau: float,
        A: np.ndarray,
        B: np.ndarray,
        activation: str,
        label: str,
    ) -> None:
        if activation not in {"identity", "silu"}:
            raise ValueError(f"unsupported E4 activation: {activation}")
        self.module = module
        self.torch = torch_module
        self.key_np = np.ascontiguousarray(key, dtype=np.float32)
        self.tau = float(tau)
        self.A_np = np.ascontiguousarray(A.astype(np.float32))
        self.B_np = np.ascontiguousarray(B.astype(np.float32))
        self.B_has_signal = bool(np.count_nonzero(self.B_np))
        self.activation = activation
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
            "activation": self.activation,
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
            "fired_mask_sha256": e2.sha256_array(
                fired.detach().to(device="cpu", dtype=self.torch.uint8).numpy()
            ),
        }
        if not self.B_has_signal or fire_count == 0:
            self.calls.append(call)
            return native_output
        selected = flat_hidden[fired].float()
        z = functional.linear(selected, A)
        if self.activation == "silu":
            z = functional.silu(z)
        delta = functional.linear(z, B)
        result = flat_native.clone()
        result[fired] = result[fired] + delta.to(dtype=flat_native.dtype)
        if token_count > fire_count:
            call["nonfired_rows_bit_equal"] = bool(self.torch.equal(result[~fired], flat_native[~fired]))
        call["delta_mean_abs"] = float(delta.double().abs().mean().item())
        call["delta_max_abs"] = float(delta.double().abs().max().item())
        self.calls.append(call)
        return result.reshape_as(native_output)

    def __enter__(self) -> "E4ExpertMount":
        if self.handle is not None:
            raise RuntimeError("E4ExpertMount cannot be entered twice")
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self._device_cache.clear()


def expert_binding(output: Path, arm: str, pack: dict[str, Any]) -> dict[str, Any]:
    training_path = output / TRAINING_FILES[arm]
    return {
        "arm": ARM_NAMES[arm],
        "construction_receipt": str(training_path),
        "construction_receipt_sha256": e2.sha256_file(training_path),
        "expertpack_manifest": str(pack_path(output, arm)),
        "expertpack_manifest_sha256": e2.sha256_file(pack_path(output, arm)),
        "key_sha256": pack["components"]["key"]["file_sha256"],
        "A_sha256": pack["components"]["A"]["sha256"],
        "B_sha256": pack["components"]["B"]["sha256"],
        "activation": pack["activation"],
    }


def eval_path(output: Path, kind: str, arm: str | None = None, index: int | None = None) -> Path:
    if kind == "doc-a-teacher":
        root = output / "eval" / "common"
    else:
        if arm not in ARMS:
            raise ValueError(f"{kind} requires --arm cf|sgd")
        root = output / "eval" / arm
    root.mkdir(parents=True, exist_ok=True)
    if kind == "abi":
        return root / "abi.json"
    if index is None:
        raise ValueError(f"{kind} requires a window index")
    return root / f"{kind.replace('-', '_')}_{index:03d}.json"


def requested_eval_indices(args: argparse.Namespace) -> list[int]:
    if args.start_window < 0 or args.count < 1 or args.start_window + args.count > N_BEHAVIORAL_WINDOWS:
        raise ValueError("eval range must stay within local windows 0..15")
    return list(range(args.start_window, args.start_window + args.count))


def validate_eval(
    output: Path,
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    kind: str,
    arm: str | None = None,
    index: int | None = None,
) -> dict[str, Any]:
    path = eval_path(output, kind, arm, index)
    receipt = e2.read_json(path)
    require_caveat(receipt, f"E4 eval {kind}/{arm}/{index}")
    schemas = {
        "abi": "moe_e4_abi_identity_v1",
        "doc-a-teacher": "moe_e4_doc_a_teacher_eval_v1",
        "doc-a-expert": "moe_e4_doc_a_expert_eval_v1",
        "wikitext": "moe_e4_wikitext_noninterference_v1",
        "code": "moe_e4_code_fire_v1",
        "doc-b": "moe_e4_doc_b_selectivity_v1",
        "guides": "moe_e4_guides_fire_descriptive_v1",
    }
    if (
        receipt.get("schema") != schemas[kind]
        or receipt.get("status") != "complete"
        or receipt.get("model_revision") != MODEL_REVISION
        or receipt.get("prepared_windows_content_sha256")
        != manifest["prepared_windows"]["content_sha256"]
        or receipt.get("script_sha256") != e2.sha256_file(SCRIPT_PATH)
    ):
        raise RuntimeError(f"E4 eval receipt contract failed: {path}")
    if kind == "doc-a-teacher":
        assert index is not None
        if receipt.get("window_index") != index:
            raise RuntimeError("E4 teacher eval window mismatch")
        target = arrays["heldout_ids"][index]
        if (
            receipt.get("input_ids_sha256") != e2.sha256_array(target)
            or receipt.get("teacher_prefix_ids_sha256")
            != e2.sha256_array(arrays["heldout_prefix_ids"][index])
        ):
            raise RuntimeError("E4 teacher input binding failed")
        for name in ("student", "teacher"):
            if receipt["arms"][name]["target_ids_sha256"] != e2.sha256_array(target[1:]):
                raise RuntimeError("E4 teacher scored target binding failed")
        return receipt
    if arm not in ARMS or receipt.get("arm") != ARM_NAMES[arm]:
        raise RuntimeError(f"E4 eval arm mismatch: {path}")
    pack = load_pack_binding(output, arm)
    if receipt.get("expert_binding") != expert_binding(output, arm, pack):
        raise RuntimeError(f"E4 eval ExpertPack binding failed: {path}")
    if kind == "abi":
        expected = np.concatenate([arrays["doc_a_key_ids"][0, :64], arrays["code_ids"][0, :64]])
        if receipt.get("input_ids_sha256") != e2.sha256_array(expected):
            raise RuntimeError("E4 ABI input binding failed")
        return receipt
    assert index is not None
    if receipt.get("window_index") != index:
        raise RuntimeError(f"E4 eval window index failed: {path}")
    array_name = {
        "doc-a-expert": "heldout_ids",
        "wikitext": "wikitext_ids",
        "code": "code_ids",
        "doc-b": "doc_b_ids",
        "guides": "guides_ids",
    }[kind]
    target = arrays[array_name][index]
    if receipt.get("input_ids_sha256") != e2.sha256_array(target):
        raise RuntimeError(f"E4 eval input binding failed: {path}")
    score_arms = {
        "doc-a-expert": ("expert",),
        "wikitext": ("base", "expert"),
        "doc-b": ("base", "expert"),
        "code": (),
        "guides": (),
    }[kind]
    for score_arm in score_arms:
        if receipt["arms"][score_arm]["target_ids_sha256"] != e2.sha256_array(target[1:]):
            raise RuntimeError(f"E4 eval scored target binding failed: {path}/{score_arm}")
    return receipt


def eval_abi(args: argparse.Namespace) -> int:
    if args.arm not in ARMS:
        raise ValueError("ABI evaluation requires --arm")
    arm = args.arm
    output = ensure_output(args.output_dir)
    manifest, arrays = load_prepared(output)
    require_bringup_green(output, manifest)
    pack, key, A, B = load_arm(output, manifest, arrays, arm)
    path = eval_path(output, "abi", arm)
    if path.is_file():
        receipt = validate_eval(output, manifest, arrays, "abi", arm)
        print(json.dumps({"status": "existing", "arm": ARM_NAMES[arm], "verdict": receipt["verdict"]}), flush=True)
        return 0
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
    with E4ExpertMount(
        layer.mlp,
        torch_module=torch,
        key=key,
        tau=float(pack["tau"]),
        A=A,
        B=np.zeros_like(B),
        activation=pack["activation"],
        label=f"{arm}_zero_B",
    ) as zero_mount:
        zero_snapshot = e2.forward_snapshot(torch, model, ids)
    zero_B_equal = bool(
        torch.equal(baseline["hidden"], zero_snapshot["hidden"])
        and torch.equal(baseline["logits"], zero_snapshot["logits"])
    )
    with E4ExpertMount(
        layer.mlp,
        torch_module=torch,
        key=key,
        tau=math.inf,
        A=A,
        B=B,
        activation=pack["activation"],
        label=f"{arm}_forced_no_fire",
    ) as nofire_mount:
        nofire_snapshot = e2.forward_snapshot(torch, model, ids)
    nofire_equal = bool(
        torch.equal(baseline["hidden"], nofire_snapshot["hidden"])
        and torch.equal(baseline["logits"], nofire_snapshot["logits"])
    )
    with E4ExpertMount(
        layer.mlp,
        torch_module=torch,
        key=key,
        tau=float(pack["tau"]),
        A=A,
        B=B,
        activation=pack["activation"],
        label=f"{arm}_live",
    ) as live_mount:
        live_snapshot = e2.forward_snapshot(torch, model, ids)
    if len(zero_mount.calls) != 1 or len(nofire_mount.calls) != 1 or len(live_mount.calls) != 1:
        raise RuntimeError("E4 ABI expected one install-layer call per forward")
    live = live_mount.calls[0]
    live_nonfire_ok = bool(live["nonfire_count"] > 0 and live["nonfired_rows_bit_equal"])
    green = bool(zero_install_equal and zero_B_equal and nofire_equal and live_nonfire_ok)

    def public_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
        return {name: value for name, value in snapshot.items() if name not in {"hidden", "logits"}}

    receipt = {
        "schema": "moe_e4_abi_identity_v1",
        "status": "complete",
        "created_at": now_iso(),
        "arm": ARM_NAMES[arm],
        "gate": "ABI carry",
        "verdict": "GREEN" if green else "RED",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        **content_fields(manifest),
        "expert_binding": expert_binding(output, arm, pack),
        "install_layer": int(pack["layer"]),
        "rank": RANK,
        "tau": float(pack["tau"]),
        "activation": pack["activation"],
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
        **caveat_fields(),
        **script_fields(),
    }
    e2.write_json(path, receipt)
    print(json.dumps({"status": "complete", "arm": ARM_NAMES[arm], "verdict": receipt["verdict"]}), flush=True)
    del model, baseline, repeat, zero_snapshot, nofire_snapshot, live_snapshot
    gc.collect()
    return 0


def eval_doc_a_teacher(args: argparse.Namespace) -> int:
    requested = requested_eval_indices(args)
    output = ensure_output(args.output_dir)
    manifest, arrays = load_prepared(output)
    require_bringup_green(output, manifest)
    validate_install(output, manifest)
    missing: list[int] = []
    for index in requested:
        path = eval_path(output, "doc-a-teacher", index=index)
        if path.is_file():
            validate_eval(output, manifest, arrays, "doc-a-teacher", index=index)
        else:
            missing.append(index)
    if not missing:
        print(json.dumps({"status": "existing", "kind": "doc-a-teacher", "windows": requested}), flush=True)
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
            "schema": "moe_e4_doc_a_teacher_eval_v1",
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
            **caveat_fields(),
            **script_fields(),
        }
        e2.write_json(eval_path(output, "doc-a-teacher", index=index), receipt)
        print(
            json.dumps(
                {
                    "kind": "doc-a-teacher",
                    "window": index,
                    "ppl_student": student["ppl"],
                    "ppl_teacher": teacher["ppl"],
                }
            ),
            flush=True,
        )
    print(json.dumps({"status": "complete", "windows": missing, "wall_seconds": time.perf_counter() - started}), flush=True)
    del model
    gc.collect()
    return 0


def load_eval_sequence(
    output: Path,
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    kind: str,
    arm: str | None = None,
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for index in range(N_BEHAVIORAL_WINDOWS):
        path = eval_path(output, kind, arm, index)
        if not path.is_file():
            return []
        receipts.append(validate_eval(output, manifest, arrays, kind, arm, index))
    return receipts


def aggregate_nll(receipts: Sequence[dict[str, Any]], arm: str) -> dict[str, Any]:
    token_count = sum(int(receipt["arms"][arm]["token_count"]) for receipt in receipts)
    nll_sum = sum(float(receipt["arms"][arm]["nll_sum"]) for receipt in receipts)
    if token_count <= 0:
        raise ValueError("cannot aggregate empty NLL sequence")
    mean_nll = nll_sum / token_count
    return {"token_count": token_count, "nll_sum": nll_sum, "mean_nll": mean_nll, "ppl": math.exp(mean_nll)}


def precondition(args: argparse.Namespace) -> int:
    if args.bootstrap_resamples < DEFAULT_BOOTSTRAP_RESAMPLES:
        raise ValueError("precondition requires at least 2,000 bootstrap resamples")
    output = ensure_output(args.output_dir)
    manifest, arrays = load_prepared(output)
    path = output / "precondition.json"
    if path.is_file():
        receipt = e2.read_json(path)
        require_caveat(receipt, "E4 precondition receipt")
        validate_precondition(output, manifest, arrays, require_pass=False)
        print(json.dumps({"status": receipt["status"], "teacher_gap_ppl": receipt["teacher_gap_ppl"]}), flush=True)
        return 0 if receipt["status"] == "pass" else 4
    receipts = load_eval_sequence(output, manifest, arrays, "doc-a-teacher")
    if len(receipts) != N_BEHAVIORAL_WINDOWS:
        raise RuntimeError("E4 precondition requires all 16 DOC-A teacher receipts")
    student = aggregate_nll(receipts, "student")
    teacher = aggregate_nll(receipts, "teacher")
    gap = float(student["ppl"] - teacher["ppl"])
    bootstrap = e3.bootstrap_ppl_triplet(
        np.asarray([receipt["arms"]["student"]["mean_nll"] for receipt in receipts]),
        np.asarray([receipt["arms"]["teacher"]["mean_nll"] for receipt in receipts]),
        None,
        resamples=args.bootstrap_resamples,
        seed=args.seed + 400,
    )
    passed = gap >= TEACHER_GAP_FLOOR_PPL
    receipt = {
        "schema": "moe_e4_behavioral_precondition_v1",
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
            {
                "path": str(eval_path(output, "doc-a-teacher", index=index)),
                "sha256": e2.sha256_file(eval_path(output, "doc-a-teacher", index=index)),
            }
            for index in range(N_BEHAVIORAL_WINDOWS)
        ],
        **content_fields(manifest),
        **caveat_fields(),
        **script_fields(),
    }
    e2.write_json(path, receipt)
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


def validate_precondition(
    output: Path,
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    require_pass: bool,
) -> dict[str, Any]:
    receipt = e2.read_json(output / "precondition.json")
    require_caveat(receipt, "E4 precondition receipt")
    links = receipt.get("teacher_receipts", [])
    links_valid = len(links) == N_BEHAVIORAL_WINDOWS and all(
        item.get("path") == str(eval_path(output, "doc-a-teacher", index=index))
        and item.get("sha256") == e2.sha256_file(eval_path(output, "doc-a-teacher", index=index))
        for index, item in enumerate(links)
    )
    if (
        receipt.get("schema") != "moe_e4_behavioral_precondition_v1"
        or receipt.get("status") not in {"pass", "stop"}
        or receipt.get("prepared_windows_content_sha256")
        != manifest["prepared_windows"]["content_sha256"]
        or not links_valid
    ):
        raise RuntimeError("E4 behavioral precondition receipt failed")
    teacher_receipts = load_eval_sequence(output, manifest, arrays, "doc-a-teacher")
    student = aggregate_nll(teacher_receipts, "student")
    teacher = aggregate_nll(teacher_receipts, "teacher")
    if (
        receipt["ppl_student"] != student["ppl"]
        or receipt["ppl_teacher"] != teacher["ppl"]
        or receipt["teacher_gap_ppl"] != student["ppl"] - teacher["ppl"]
    ):
        raise RuntimeError("E4 precondition metrics changed")
    if require_pass and (
        receipt["status"] != "pass" or receipt["teacher_gap_ppl"] < TEACHER_GAP_FLOOR_PPL
    ):
        raise RuntimeError(f"{receipt.get('stop_reason') or 'behavioral precondition not passed'}; STOP")
    return receipt


def eval_doc_a_expert(args: argparse.Namespace) -> int:
    if args.arm not in ARMS:
        raise ValueError("DOC-A expert evaluation requires --arm")
    arm = args.arm
    requested = requested_eval_indices(args)
    output = ensure_output(args.output_dir)
    manifest, arrays = load_prepared(output)
    require_bringup_green(output, manifest)
    validate_precondition(output, manifest, arrays, require_pass=True)
    pack, key, A, B = load_arm(output, manifest, arrays, arm)
    missing: list[int] = []
    for index in requested:
        path = eval_path(output, "doc-a-expert", arm, index)
        if path.is_file():
            validate_eval(output, manifest, arrays, "doc-a-expert", arm, index)
        else:
            missing.append(index)
    if not missing:
        print(json.dumps({"status": "existing", "arm": ARM_NAMES[arm], "windows": requested}), flush=True)
        return 0
    torch, model, runtime = e2.load_model(args)
    layer = model.model.layers[int(pack["layer"])]
    for index in missing:
        target = arrays["heldout_ids"][index]
        with E4ExpertMount(
            layer.mlp,
            torch_module=torch,
            key=key,
            tau=float(pack["tau"]),
            A=A,
            B=B,
            activation=pack["activation"],
            label=f"{arm}_doc_a_expert_{index}",
        ) as mount:
            expert = e2.score_target_window(torch, model, target, target)
        if len(mount.calls) != 1:
            raise RuntimeError("E4 DOC-A expert eval expected one install-layer call")
        receipt = {
            "schema": "moe_e4_doc_a_expert_eval_v1",
            "status": "complete",
            "created_at": now_iso(),
            "arm": ARM_NAMES[arm],
            "window_index": index,
            "source_window": manifest["windows"]["heldout"][index],
            "context_tokens": 0,
            "input_ids_sha256": e2.sha256_array(target),
            "arms": {"expert": expert},
            "fire": mount.calls[0],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            **content_fields(manifest),
            "expert_binding": expert_binding(output, arm, pack),
            "install_layer": int(pack["layer"]),
            "runtime": runtime,
            **caveat_fields(),
            **script_fields(),
        }
        e2.write_json(eval_path(output, "doc-a-expert", arm, index), receipt)
        print(
            json.dumps(
                {
                    "arm": ARM_NAMES[arm],
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
    if args.arm not in ARMS:
        raise ValueError(f"{kind} evaluation requires --arm")
    arm = args.arm
    requested = requested_eval_indices(args)
    output = ensure_output(args.output_dir)
    manifest, arrays = load_prepared(output)
    require_bringup_green(output, manifest)
    validate_precondition(output, manifest, arrays, require_pass=True)
    pack, key, A, B = load_arm(output, manifest, arrays, arm)
    missing: list[int] = []
    for index in requested:
        path = eval_path(output, kind, arm, index)
        if path.is_file():
            validate_eval(output, manifest, arrays, kind, arm, index)
        else:
            missing.append(index)
    if not missing:
        print(json.dumps({"status": "existing", "arm": ARM_NAMES[arm], "kind": kind, "windows": requested}), flush=True)
        return 0
    array_name = {
        "wikitext": "wikitext_ids",
        "code": "code_ids",
        "doc-b": "doc_b_ids",
        "guides": "guides_ids",
    }[kind]
    torch, model, runtime = e2.load_model(args)
    layer = model.model.layers[int(pack["layer"])]
    for index in missing:
        target = arrays[array_name][index]
        if kind in {"wikitext", "doc-b"}:
            base = e2.score_target_window(torch, model, target, target)
            with E4ExpertMount(
                layer.mlp,
                torch_module=torch,
                key=key,
                tau=float(pack["tau"]),
                A=A,
                B=B,
                activation=pack["activation"],
                label=f"{arm}_{kind}_{index}",
            ) as mount:
                expert = e2.score_target_window(torch, model, target, target)
            arms: dict[str, Any] = {"base": base, "expert": expert}
            delta_percent = (expert["ppl"] - base["ppl"]) / base["ppl"] * 100.0
        else:
            ids = torch.from_numpy(np.ascontiguousarray(target[None, :], dtype=np.int64)).to(
                e2.input_device(model)
            )
            with E4ExpertMount(
                layer.mlp,
                torch_module=torch,
                key=key,
                tau=float(pack["tau"]),
                A=A,
                B=B,
                activation=pack["activation"],
                label=f"{arm}_{kind}_{index}",
            ) as mount:
                with torch.inference_mode():
                    output_object = model(input_ids=ids, use_cache=False, logits_to_keep=1)
            del output_object, ids
            arms = {}
            delta_percent = None
        if len(mount.calls) != 1:
            raise RuntimeError(f"E4 {kind} eval expected one install-layer call")
        schema = {
            "wikitext": "moe_e4_wikitext_noninterference_v1",
            "code": "moe_e4_code_fire_v1",
            "doc-b": "moe_e4_doc_b_selectivity_v1",
            "guides": "moe_e4_guides_fire_descriptive_v1",
        }[kind]
        source_window = manifest["windows"]["doc_b_cross_probe"][index] if kind == "doc-b" else None
        receipt = {
            "schema": schema,
            "status": "complete",
            "created_at": now_iso(),
            "arm": ARM_NAMES[arm],
            "window_index": index,
            "source_window": source_window,
            "input_ids_sha256": e2.sha256_array(target),
            "arms": arms,
            "per_window_ppl_delta_percent": delta_percent,
            "fire": mount.calls[0],
            "measurement_role": (
                DOC_B_ROLE if kind == "doc-b" else GUIDES_ROLE if kind == "guides" else "E4-G5"
            ),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            **content_fields(manifest),
            "expert_binding": expert_binding(output, arm, pack),
            "install_layer": int(pack["layer"]),
            "runtime": runtime,
            **caveat_fields(),
            **script_fields(),
        }
        e2.write_json(eval_path(output, kind, arm, index), receipt)
        print(
            json.dumps(
                {
                    "arm": ARM_NAMES[arm],
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
    if args.eval_kind == "doc-a-teacher":
        if args.arm is not None:
            raise ValueError("common DOC-A teacher eval must not specify --arm")
        return eval_doc_a_teacher(args)
    if args.eval_kind == "abi":
        return eval_abi(args)
    if args.eval_kind == "doc-a-expert":
        return eval_doc_a_expert(args)
    return eval_noninterference(args, args.eval_kind)


def classify_e4_h(cf_g4: str, sgd_g4: str, cf_recovery: float | None, sgd_recovery: float | None) -> str:
    if cf_recovery is None or sgd_recovery is None:
        return "NOT_MEASURED"
    cf_pass = cf_g4 == "SUPPORTED"
    sgd_pass = sgd_g4 == "SUPPORTED"
    if cf_pass and cf_recovery * 100.0 >= sgd_recovery * 100.0 - E4_H_MARGIN_POINTS:
        return "ALGEBRA-SUFFICES"
    if not cf_pass and sgd_pass:
        return "ALGEBRA-INSUFFICIENT"
    if not cf_pass and not sgd_pass:
        return "BOTH-FAIL"
    return "UNCLASSIFIED-BY-REGISTERED-E4-H"


def gate_record(verdict: str, report_line: str, **details: Any) -> dict[str, Any]:
    return {"verdict": verdict, "report_line": report_line, **details}


def fmt(value: float | None, digits: int = 6) -> str:
    return "NOT_MEASURED" if value is None else f"{value:.{digits}g}"


def analyze_arm(
    args: argparse.Namespace,
    output: Path,
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    arm: str,
    teacher_receipts: Sequence[dict[str, Any]],
    precondition_receipt: dict[str, Any] | None,
) -> dict[str, Any]:
    lead = output / "GPU_E4_RESUME_COMMANDS.sh"
    construction_path = output / TRAINING_FILES[arm]
    construction = (
        validate_arm_training(output, manifest, arrays, arm) if construction_path.is_file() else None
    )
    if construction is None:
        g3 = gate_record("NOT_MEASURED", f"E4-G3 {ARM_NAMES[arm]} NOT_MEASURED — lead script: {lead}")
    else:
        metrics = (
            construction["stored_fp16_rank64_validation"]
            if arm == "cf"
            else construction["stored_fp16_validation"]
        )
        g3 = gate_record(
            construction["verdict"],
            (
                f"E4-G3 {ARM_NAMES[arm]} {construction['verdict']} — zero MSE={metrics['zero_predictor_mse']:.6g}, "
                f"arm MSE={metrics['adapter_mse']:.6g}, improvement={metrics['improvement_percent']:.6g}%"
            ),
            metrics=metrics,
        )
    abi_path = eval_path(output, "abi", arm)
    abi = validate_eval(output, manifest, arrays, "abi", arm) if abi_path.is_file() else None
    expert_receipts = load_eval_sequence(output, manifest, arrays, "doc-a-expert", arm)
    g4_measurement: dict[str, Any] | None = None
    if precondition_receipt is not None and precondition_receipt["status"] == "stop":
        g4 = gate_record(
            "NOT_MEASURED",
            f"E4-G4 {ARM_NAMES[arm]} NOT_MEASURED — STOP: episodic teacher signal below registered floor",
        )
    elif (
        len(teacher_receipts) == N_BEHAVIORAL_WINDOWS
        and len(expert_receipts) == N_BEHAVIORAL_WINDOWS
        and precondition_receipt is not None
        and precondition_receipt["status"] == "pass"
    ):
        student = aggregate_nll(teacher_receipts, "student")
        teacher = aggregate_nll(teacher_receipts, "teacher")
        expert = aggregate_nll(expert_receipts, "expert")
        gap = student["ppl"] - teacher["ppl"]
        recovery = (student["ppl"] - expert["ppl"]) / gap
        bootstrap = e3.bootstrap_ppl_triplet(
            np.asarray([receipt["arms"]["student"]["mean_nll"] for receipt in teacher_receipts]),
            np.asarray([receipt["arms"]["teacher"]["mean_nll"] for receipt in teacher_receipts]),
            np.asarray([receipt["arms"]["expert"]["mean_nll"] for receipt in expert_receipts]),
            resamples=args.bootstrap_resamples,
            seed=args.seed + (500 if arm == "cf" else 510),
        )
        g4_measurement = {
            "ppl_student": student["ppl"],
            "ppl_teacher": teacher["ppl"],
            "ppl_expert": expert["ppl"],
            "teacher_gap_ppl": gap,
            "recovery_fraction": recovery,
            "recovery_percent": recovery * 100.0,
            **bootstrap,
            "expert_fire": e3.bootstrap_mean(
                [receipt["fire"]["fire_rate"] for receipt in expert_receipts],
                resamples=args.bootstrap_resamples,
                seed=args.seed + (501 if arm == "cf" else 511),
                unit=f"{ARM_NAMES[arm]} DOC-A heldout windows",
            ),
        }
        verdict = (
            "SUPPORTED"
            if recovery >= G4_RECOVERY_FLOOR
            and bootstrap["recovery_ci95_low_fraction"] > 0.0
            else "RED"
        )
        g4 = gate_record(
            verdict,
            (
                f"E4-G4 {ARM_NAMES[arm]} {verdict} — recovery={recovery * 100.0:.6g}% "
                f"CI95=[{bootstrap['recovery_ci95_low_percent']:.6g}%, "
                f"{bootstrap['recovery_ci95_high_percent']:.6g}%]"
            ),
            measurement=g4_measurement,
        )
    else:
        g4 = gate_record("NOT_MEASURED", f"E4-G4 {ARM_NAMES[arm]} NOT_MEASURED — lead script: {lead}")

    sequences = {
        kind: load_eval_sequence(output, manifest, arrays, kind, arm)
        for kind in ("wikitext", "code", "doc-b", "guides")
    }
    wikitext_metrics: dict[str, Any] | None = None
    code_metrics: dict[str, Any] | None = None
    doc_b_metrics: dict[str, Any] | None = None
    guides_metrics: dict[str, Any] | None = None
    if len(sequences["wikitext"]) == N_BEHAVIORAL_WINDOWS:
        wikitext_metrics = e3.bootstrap_ppl_delta(
            [receipt["arms"]["base"]["mean_nll"] for receipt in sequences["wikitext"]],
            [receipt["arms"]["expert"]["mean_nll"] for receipt in sequences["wikitext"]],
            resamples=args.bootstrap_resamples,
            seed=args.seed + (600 if arm == "cf" else 620),
        )
        wikitext_metrics["fire"] = e3.bootstrap_mean(
            [receipt["fire"]["fire_rate"] for receipt in sequences["wikitext"]],
            resamples=args.bootstrap_resamples,
            seed=args.seed + (601 if arm == "cf" else 621),
            unit="WikiText windows",
        )
    if len(sequences["code"]) == N_BEHAVIORAL_WINDOWS:
        code_metrics = e3.bootstrap_mean(
            [receipt["fire"]["fire_rate"] for receipt in sequences["code"]],
            resamples=args.bootstrap_resamples,
            seed=args.seed + (610 if arm == "cf" else 630),
            unit="code windows",
        )
    if len(sequences["doc-b"]) == N_BEHAVIORAL_WINDOWS:
        doc_b_metrics = e3.bootstrap_ppl_delta(
            [receipt["arms"]["base"]["mean_nll"] for receipt in sequences["doc-b"]],
            [receipt["arms"]["expert"]["mean_nll"] for receipt in sequences["doc-b"]],
            resamples=args.bootstrap_resamples,
            seed=args.seed + (640 if arm == "cf" else 650),
        )
        doc_b_metrics["fire"] = e3.bootstrap_mean(
            [receipt["fire"]["fire_rate"] for receipt in sequences["doc-b"]],
            resamples=args.bootstrap_resamples,
            seed=args.seed + (641 if arm == "cf" else 651),
            unit="DOC-B windows",
        )
    if len(sequences["guides"]) == N_BEHAVIORAL_WINDOWS:
        guides_metrics = e3.bootstrap_mean(
            [receipt["fire"]["fire_rate"] for receipt in sequences["guides"]],
            resamples=args.bootstrap_resamples,
            seed=args.seed + (660 if arm == "cf" else 670),
            unit="guides windows",
        )
    if wikitext_metrics is not None and code_metrics is not None:
        g5_verdict = (
            "SUPPORTED"
            if wikitext_metrics["ppl_delta_percent"] <= G5_WIKITEXT_PPL_DELTA_CAP_PERCENT
            and code_metrics["value"] <= G5_CODE_FIRE_CAP
            else "RED"
        )
        g5 = gate_record(
            g5_verdict,
            (
                f"E4-G5 {ARM_NAMES[arm]} {g5_verdict} — WikiText delta="
                f"{wikitext_metrics['ppl_delta_percent']:+.6g}%, code fire={code_metrics['value']:.6g}"
            ),
            wikitext=wikitext_metrics,
            code=code_metrics,
            doc_b_descriptive=doc_b_metrics,
            guides_descriptive=guides_metrics,
        )
    elif precondition_receipt is not None and precondition_receipt["status"] == "stop":
        g5 = gate_record(
            "NOT_MEASURED",
            f"E4-G5 {ARM_NAMES[arm]} NOT_MEASURED — STOP: episodic teacher signal below registered floor",
            doc_b_descriptive=doc_b_metrics,
            guides_descriptive=guides_metrics,
        )
    else:
        g5 = gate_record(
            "NOT_MEASURED",
            f"E4-G5 {ARM_NAMES[arm]} NOT_MEASURED — lead script: {lead}",
            doc_b_descriptive=doc_b_metrics,
            guides_descriptive=guides_metrics,
        )
    return {
        "construction": construction,
        "gates": {"E4-G3": g3, "E4-G4": g4, "E4-G5": g5},
        "abi": None if abi is None else {"verdict": abi["verdict"], "receipt": str(abi_path)},
        "g4": g4_measurement,
        "wikitext": wikitext_metrics,
        "code": code_metrics,
        "doc_b_descriptive": doc_b_metrics,
        "guides_descriptive": guides_metrics,
    }


def analyze(args: argparse.Namespace) -> int:
    if args.bootstrap_resamples < DEFAULT_BOOTSTRAP_RESAMPLES:
        raise ValueError("analyze requires at least 2,000 bootstrap resamples")
    output = ensure_output(args.output_dir)
    manifest, arrays = load_prepared(output)
    install = validate_install(output, manifest)
    lead = output / "GPU_E4_RESUME_COMMANDS.sh"
    teacher_receipts = load_eval_sequence(output, manifest, arrays, "doc-a-teacher")
    pre = None
    if (output / "precondition.json").is_file():
        pre = validate_precondition(output, manifest, arrays, require_pass=False)
    arms = {
        arm: analyze_arm(args, output, manifest, arrays, arm, teacher_receipts, pre)
        for arm in ARMS
    }
    cf_g4 = arms["cf"]["gates"]["E4-G4"]["verdict"]
    sgd_g4 = arms["sgd"]["gates"]["E4-G4"]["verdict"]
    cf_recovery = None if arms["cf"]["g4"] is None else arms["cf"]["g4"]["recovery_fraction"]
    sgd_recovery = None if arms["sgd"]["g4"] is None else arms["sgd"]["g4"]["recovery_fraction"]
    h_class = classify_e4_h(cf_g4, sgd_g4, cf_recovery, sgd_recovery)
    if h_class == "NOT_MEASURED":
        stop = pre is not None and pre["status"] == "stop"
        h_sentence = (
            "E4-H NOT_MEASURED — STOP: episodic teacher signal below registered floor"
            if stop
            else f"E4-H NOT_MEASURED — lead script: {lead}"
        )
    elif h_class == "ALGEBRA-SUFFICES":
        h_sentence = (
            f"E4-H ALGEBRA-SUFFICES — ARM-CF recovery {cf_recovery * 100.0:.6g}% is at least "
            f"ARM-SGD recovery {sgd_recovery * 100.0:.6g}% minus 5 absolute points, and ARM-CF passes E4-G4."
        )
    elif h_class == "ALGEBRA-INSUFFICIENT":
        h_sentence = (
            f"E4-H ALGEBRA-INSUFFICIENT — ARM-CF fails E4-G4 at {cf_recovery * 100.0:.6g}% "
            f"while ARM-SGD passes at {sgd_recovery * 100.0:.6g}%."
        )
    elif h_class == "BOTH-FAIL":
        h_sentence = (
            f"E4-H BOTH-FAIL — ARM-CF recovery {cf_recovery * 100.0:.6g}%; ARM-SGD recovery "
            f"{sgd_recovery * 100.0:.6g}%; the episodic-consolidation premise itself remains open."
        )
    else:
        h_sentence = (
            f"E4-H UNCLASSIFIED-BY-REGISTERED-E4-H — ARM-CF passes E4-G4 at "
            f"{cf_recovery * 100.0:.6g}% but trails ARM-SGD {sgd_recovery * 100.0:.6g}% by more than 5 points."
        )

    missing_pairs = [index for index in range(N_PAIR_WINDOWS) if not pair_paths(output, index)[1].is_file()]
    anything_not_done: list[str] = []
    if missing_pairs:
        anything_not_done.append(
            f"{len(missing_pairs)} of 64 L10 teacher/student pairs remain uncaptured; lead script {lead}"
        )
    for arm in ARMS:
        if arms[arm]["construction"] is None:
            anything_not_done.append(f"{ARM_NAMES[arm]} construction and E4-G3 are not measured")
        if arms[arm]["gates"]["E4-G4"]["verdict"] == "NOT_MEASURED":
            anything_not_done.append(f"{ARM_NAMES[arm]} E4-G4 is not measured")
        if arms[arm]["gates"]["E4-G5"]["verdict"] == "NOT_MEASURED":
            anything_not_done.append(f"{ARM_NAMES[arm]} E4-G5 is not measured")
    cf_construction = arms["cf"]["construction"]
    spectrum = None if cf_construction is None else cf_construction["spectrum"]
    analysis = {
        "schema": "moe_e4_analysis_v1",
        "status": "complete" if not anything_not_done else "complete_with_not_measured",
        "created_at": now_iso(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        **content_fields(manifest),
        "install": {
            "branch": install["dependency_branch"],
            "e3_1_registered_sentence": install["e3_1_registered_sentence"],
            "row": install["install_row"],
        },
        "arms": arms,
        "E4-H": {"verdict": h_class, "sentence": h_sentence},
        "spectrum": spectrum,
        "precondition": pre,
        "lead_script": str(lead),
        "anything_not_done": anything_not_done,
        **caveat_fields(),
        **script_fields(),
    }
    analysis_path = output / "analysis.json"
    e2.write_json(analysis_path, analysis)

    lines = [
        "# MOE-E4 closed-form consolidation report",
        "",
        "## E4-H verdict",
        "",
        h_sentence,
        "",
        f"Registered fallback caveat: **{ADDRESS_CAVEAT}**. DOC-B and guides firing remain **DESCRIPTIVE** in every arm.",
        "",
        "## Install row",
        "",
        f"E3.1 branch: `{install['dependency_branch']}`; E3.1 sentence: `{install['e3_1_registered_sentence']}`.",
        "",
        "| Layer | Key | Rank | tau | DOC-A recall | DOC-B fire | Guides fire | Code fire | GRM fire | WikiText fire | Qualifies |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| {install['install_row']['layer']} | K4 | {RANK} | {install['install_row']['tau']:.8g} | "
            f"{install['install_row']['doc_a_recall']:.6f} | {install['install_row']['doc_b_fire']:.6f} (DESCRIPTIVE) | "
            f"{install['install_row']['guides_fire']:.6f} (DESCRIPTIVE) | {install['install_row']['code_fire']:.6f} | "
            f"{install['install_row']['grm_fire']:.6f} | {install['install_row']['wikitext_fire']:.6f} | "
            f"{'YES' if install['install_row']['qualifies'] else 'NO'} |"
        ),
        "",
        "## Registered gates per arm",
        "",
        "| Arm | E4-G3 | E4-G4 | E4-G5 | ABI carry |",
        "|---|---|---|---|---|",
    ]
    for arm in ARMS:
        record = arms[arm]
        abi = "NOT_MEASURED" if record["abi"] is None else record["abi"]["verdict"]
        lines.append(
            f"| {ARM_NAMES[arm]} | {record['gates']['E4-G3']['verdict']} | "
            f"{record['gates']['E4-G4']['verdict']} | {record['gates']['E4-G5']['verdict']} | {abi} |"
        )
    lines.extend(["", arms["cf"]["gates"]["E4-G3"]["report_line"], "", arms["cf"]["gates"]["E4-G4"]["report_line"], "", arms["cf"]["gates"]["E4-G5"]["report_line"], "", arms["sgd"]["gates"]["E4-G3"]["report_line"], "", arms["sgd"]["gates"]["E4-G4"]["report_line"], "", arms["sgd"]["gates"]["E4-G5"]["report_line"]])
    lines.extend(["", "## Singular-spectrum inspectability", ""])
    if spectrum is None:
        lines.append(f"Top-8 singular values: NOT_MEASURED. Energy fraction at r=64: NOT_MEASURED. Lead script: `{lead}`.")
    else:
        top = ", ".join(f"{value:.8g}" for value in spectrum["top_8_singular_values"])
        lines.append(f"Top-8 singular values: `{top}`.")
        lines.append("")
        lines.append(f"Energy fraction at r=64: `{spectrum['energy_fraction_at_rank_64']:.8g}`.")
        lines.append("")
        lines.append(f"All 64 per-direction right-vector/key cosines: `{output / 'cf_inspectability.json'}`.")
    lines.extend(
        [
            "",
            "## E4-G4 rows",
            "",
            "| Arm | ppl_student | ppl_teacher | teacher gap | ppl_expert | Recovery | Bootstrap 95% CI | Gate |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for arm in ARMS:
        measurement = arms[arm]["g4"]
        if measurement is None:
            student = None if pre is None else pre.get("ppl_student")
            teacher = None if pre is None else pre.get("ppl_teacher")
            gap = None if pre is None else pre.get("teacher_gap_ppl")
            lines.append(
                f"| {ARM_NAMES[arm]} | {fmt(student)} | {fmt(teacher)} | {fmt(gap)} | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | {arms[arm]['gates']['E4-G4']['verdict']} |"
            )
        else:
            lines.append(
                f"| {ARM_NAMES[arm]} | {measurement['ppl_student']:.6g} | {measurement['ppl_teacher']:.6g} | "
                f"{measurement['teacher_gap_ppl']:.6g} | {measurement['ppl_expert']:.6g} | "
                f"{measurement['recovery_percent']:.6g}% | [{measurement['recovery_ci95_low_percent']:.6g}%, "
                f"{measurement['recovery_ci95_high_percent']:.6g}%] | {arms[arm]['gates']['E4-G4']['verdict']} |"
            )
    lines.extend(
        [
            "",
            "## G5 and descriptive selectivity",
            "",
            "| Arm | WikiText ppl delta | Code fire | DOC-B ppl delta / fire (DESCRIPTIVE) | Guides fire (DESCRIPTIVE) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        record = arms[arm]
        wiki = record["wikitext"]
        code = record["code"]
        doc_b = record["doc_b_descriptive"]
        guides = record["guides_descriptive"]
        wiki_cell = "NOT_MEASURED" if wiki is None else f"{wiki['ppl_delta_percent']:+.6g}%"
        code_cell = "NOT_MEASURED" if code is None else f"{code['value']:.6g}"
        doc_b_cell = "NOT_MEASURED" if doc_b is None else f"{doc_b['ppl_delta_percent']:+.6g}% / {doc_b['fire']['value']:.6g}"
        guides_cell = "NOT_MEASURED" if guides is None else f"{guides['value']:.6g}"
        lines.append(f"| {ARM_NAMES[arm]} | {wiki_cell} | {code_cell} | {doc_b_cell} | {guides_cell} |")
    lines.extend(
        [
            "",
            "## Provenance and execution",
            "",
            f"Prepared content SHA-256: `{manifest['prepared_windows']['content_sha256']}`.",
            "",
            f"Install receipt: `{output / 'install.json'}`.",
            "",
            f"Lead script: `{lead}`. GPU leases are bounded at {GPU_WALL_SECONDS}s under flock with 30s gaps; CPU construction is bounded at {CPU_WALL_SECONDS}s.",
            "",
            "No novel text is excerpted; identities are limited to paths, hashes, byte counts, and token counts inherited from the content-addressed E3 manifest.",
            "",
            "## Anything not done",
            "",
        ]
    )
    if anything_not_done:
        lines.extend([f"- {item}" for item in anything_not_done])
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Evidence limits",
            "",
            "This is finite evidence for one frozen model revision, one document pair, one registered layer/address, and one rank. ARM-CF-FULL is a descriptive validation ceiling. A RED or NOT_MEASURED gate is retained without threshold widening.",
            "",
        ]
    )
    report_path = output / "MOE_E4_REPORT.md"
    e2.write_text(report_path, "\n".join(lines))
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            files.append(
                {
                    "path": str(path),
                    "byte_count": int(path.stat().st_size),
                    "sha256": e2.sha256_file(path),
                }
            )
    artifact_manifest = {
        "schema": "moe_e4_artifact_manifest_v1",
        "status": analysis["status"],
        "created_at": now_iso(),
        "order": str(ORDER_PATH),
        "order_sha256": e2.sha256_file(ORDER_PATH),
        "files": files,
        "anything_not_done": anything_not_done,
        **caveat_fields(),
        **script_fields(),
    }
    e2.write_json(output / "artifact_manifest.json", artifact_manifest)
    print(
        json.dumps(
            {
                "status": analysis["status"],
                "E4-H": h_class,
                "gates": {
                    arm: {name: gate["verdict"] for name, gate in arms[arm]["gates"].items()}
                    for arm in ARMS
                },
                "report": str(report_path),
            }
        ),
        flush=True,
    )
    complete_supported = (
        h_class == "ALGEBRA-SUFFICES"
        and all(
            arms[arm]["gates"]["E4-G3"]["verdict"] == "GREEN"
            and arms[arm]["gates"]["E4-G4"]["verdict"] == "SUPPORTED"
            and arms[arm]["gates"]["E4-G5"]["verdict"] == "SUPPORTED"
            for arm in ARMS
        )
    )
    return 0 if complete_supported else 3


def self_test(args: argparse.Namespace) -> int:
    output = ensure_output(args.output_dir)
    manifest, arrays = load_prepared(output)
    install = validate_install(output, manifest)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check(
        "e3_prepared_content_exact",
        manifest["prepared_windows"]["content_sha256"]
        == e3.load_prepared(E3_OUTPUT)[0]["prepared_windows"]["content_sha256"],
        manifest["prepared_windows"]["content_sha256"],
    )
    check(
        "registered_fallback_layer_10",
        install["install_row"]["layer"] == 10
        and install["dependency_branch_code"] == "registered_fallback_best_e3_1_k4_fit_side_rank",
        install["dependency_branch"],
    )
    check("caveat_verbatim", install["address_caveat"] == ADDRESS_CAVEAT, install["address_caveat"])
    check(
        "doc_b_guides_descriptive",
        install["doc_b_firing_role"] == DOC_B_ROLE and install["guides_firing_role"] == GUIDES_ROLE,
        [install["doc_b_firing_role"], install["guides_firing_role"]],
    )
    check(
        "true_pair_prefix_contract",
        all(row["teacher_prefix_token_count"] == PREFIX_TOKENS for row in manifest["windows"]["pair"]),
        "64 true-context prefixes",
    )

    rng = np.random.default_rng(args.seed)
    n_fit, n_validation, dimension = 41, 17, 9
    H_fit = rng.standard_normal((n_fit, dimension))
    H_validation = rng.standard_normal((n_validation, dimension))
    true_w = rng.standard_normal((dimension, dimension)) * 0.1
    D_fit = H_fit @ true_w.T + rng.standard_normal((n_fit, dimension)) * 0.01
    D_validation = H_validation @ true_w.T + rng.standard_normal((n_validation, dimension)) * 0.01
    ridge = ridge_trace_solve(
        H_fit.T @ H_fit,
        H_fit.T @ D_fit,
        H_validation.T @ H_validation,
        H_validation.T @ D_validation,
        float(np.sum(D_validation**2)),
        int(D_validation.size),
        (1.0e-4, 1.0e-2, 1.0),
    )
    selected = ridge["selected"]
    coefficient = ridge["coefficient"]
    direct = np.linalg.solve(
        H_fit.T @ H_fit + np.eye(dimension) * selected["lambda"], H_fit.T @ D_fit
    )
    direct_mse = float(np.mean((H_validation @ direct - D_validation) ** 2))
    check("ridge_coefficient_matches_direct_solve", np.allclose(coefficient, direct, rtol=1e-9, atol=1e-10), float(np.max(np.abs(coefficient - direct))))
    check("ridge_trace_mse_matches_direct", abs(selected["validation_mse"] - direct_mse) < 1.0e-10, [selected["validation_mse"], direct_mse])
    u, singular, vt = canonical_svd(coefficient.T)
    A_full, B_full = factor_from_svd(u, singular, vt, dimension)
    check("full_svd_reconstructs_order_W", np.allclose(B_full @ A_full, coefficient.T, rtol=1e-9, atol=1e-10), float(np.max(np.abs(B_full @ A_full - coefficient.T))))

    import torch

    mount_module = torch.nn.Identity()
    hidden = torch.zeros((1, 4, dimension), dtype=torch.bfloat16)
    hidden[0, :, 0] = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    key = np.zeros(dimension, dtype=np.float32)
    key[0] = 1.0
    A = np.zeros((3, dimension), dtype=np.float16)
    A[:, 0] = 1.0
    B = np.zeros((dimension, 3), dtype=np.float16)
    B[0, :] = 0.25
    baseline = mount_module(hidden)
    with E4ExpertMount(
        mount_module,
        torch_module=torch,
        key=key,
        tau=0.0,
        A=A,
        B=B,
        activation="identity",
        label="self_test_cf",
    ) as mount:
        mounted = mount_module(hidden)
    check(
        "linear_mount_mixed_nonfire_exact",
        mount.calls[0]["fire_count"] == 2
        and mount.calls[0]["nonfire_count"] == 2
        and mount.calls[0]["nonfired_rows_bit_equal"]
        and not torch.equal(mounted, baseline),
        mount.calls[0],
    )
    zero = np.zeros_like(B)
    with E4ExpertMount(
        mount_module,
        torch_module=torch,
        key=key,
        tau=0.0,
        A=A,
        B=zero,
        activation="silu",
        label="self_test_zero",
    ) as zero_mount:
        zero_result = mount_module(hidden)
    check("zero_B_exact_identity", torch.equal(zero_result, baseline), zero_mount.calls[0])
    check(
        "e4_h_boundary_inclusive",
        classify_e4_h("SUPPORTED", "SUPPORTED", 0.25, 0.30) == "ALGEBRA-SUFFICES",
        classify_e4_h("SUPPORTED", "SUPPORTED", 0.25, 0.30),
    )
    check(
        "e4_h_insufficient_branch",
        classify_e4_h("RED", "SUPPORTED", 0.20, 0.30) == "ALGEBRA-INSUFFICIENT",
        classify_e4_h("RED", "SUPPORTED", 0.20, 0.30),
    )
    check(
        "e4_h_both_fail_branch",
        classify_e4_h("RED", "RED", 0.20, 0.20) == "BOTH-FAIL",
        classify_e4_h("RED", "RED", 0.20, 0.20),
    )
    with tempfile.TemporaryDirectory(prefix="olmoe-e4-self-test-", dir="/tmp") as temporary:
        probe = Path(temporary) / "nonfinite.json"
        failed_loud = False
        try:
            e2.write_json(probe, {"bad": float("nan")})
        except ValueError as exc:
            failed_loud = "$.bad" in str(exc) and not probe.exists()
    check("writer_rejects_nonfinite_before_write", failed_loud, "$.bad named")
    lead = (output / "GPU_E4_RESUME_COMMANDS.sh").read_text(encoding="utf-8")
    check(
        "lead_gpu_discipline",
        "flock -w 7200 /tmp/forge-gpu.lock" in lead
        and "590s" in lead
        and "sleep 30" in lead
        and "CUDA_VISIBLE_DEVICES=''" in lead,
        {"gpu_wall_seconds": GPU_WALL_SECONDS, "cpu_wall_seconds": CPU_WALL_SECONDS},
    )
    passed = all(row["passed"] for row in checks)
    receipt = {
        "schema": "moe_e4_cpu_validation_v1",
        "status": "passed" if passed else "failed",
        "created_at": now_iso(),
        "synthetic_only_not_gate_evidence": True,
        **content_fields(manifest),
        "checks_passed": sum(row["passed"] for row in checks),
        "checks_total": len(checks),
        "checks": checks,
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        **caveat_fields(),
        **script_fields(),
    }
    e2.write_json(output / "cpu_validation.json", receipt)
    print(json.dumps({"status": receipt["status"], "checks": len(checks)}), flush=True)
    return 0 if passed else 3


def lead_script_text(output: Path) -> str:
    script = shlex.quote(SCRIPT_PATH.relative_to(REPO_ROOT).as_posix())
    out = shlex.quote(str(output))
    common = f"python3 {script} --output-dir {out}"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        "",
        f"# E3.1 registered fallback caveat: {ADDRESS_CAVEAT}",
        "# DOC-B and guides firing remain DESCRIPTIVE.",
        "",
        "run_gpu_e4() {",
        "  set +e",
        f"  {GPU_PREAMBLE} \"$@\"",
        "  gpu_rc_e4=$?",
        "  set -e",
        "  sleep 30",
        "  return \"${gpu_rc_e4}\"",
        "}",
        "",
        "run_cpu_e4() {",
        "  timeout --signal=TERM --kill-after=5s 7200s env CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DEACTIVATE_ASYNC_LOAD=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/olmoe_e4_pycache PYTHONUNBUFFERED=1 \"$@\"",
        "}",
        "",
        "run_or_analyze_e4() {",
        "  set +e",
        "  \"$@\"",
        "  stage_rc_e4=$?",
        "  set -e",
        "  if [ \"${stage_rc_e4}\" -ne 0 ]; then",
        "    set +e",
        f"    run_cpu_e4 {common} analyze --bootstrap-resamples 2000",
        "    set -e",
        "    exit \"${stage_rc_e4}\"",
        "  fi",
        "}",
        "",
        f"run_or_analyze_e4 run_cpu_e4 {common} prepare",
        f"run_or_analyze_e4 run_cpu_e4 {common} self-test --bootstrap-resamples 2000",
        f"run_or_analyze_e4 run_cpu_e4 {common} bringup",
        "",
        "# Fresh L10 true-context pairs, four pairs per <=590s GPU lease.",
    ]
    for start in range(0, N_PAIR_WINDOWS, 4):
        lines.append(
            f"run_or_analyze_e4 run_gpu_e4 {common} capture-pairs --device auto --max-gpu-memory 11GiB --start-window {start} --count 4"
        )
    lines.extend(
        [
            "",
            "# CPU algebra/training wall: <=7200s per stage.",
            f"run_or_analyze_e4 run_cpu_e4 {common} fit-cf --threads 8 --rank 64 --bootstrap-resamples 2000",
            f"run_or_analyze_e4 run_cpu_e4 {common} train-sgd --threads 8 --rank 64 --train-tokens-per-window 128 --bootstrap-resamples 2000",
            "",
            "# Exact ABI checks stay CPU BF16 and get an additional strict 590s inner wall.",
        ]
    )
    for arm in ARMS:
        lines.append(
            f"run_or_analyze_e4 run_cpu_e4 timeout --signal=TERM --kill-after=5s 590s env CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DEACTIVATE_ASYNC_LOAD=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/olmoe_e4_pycache PYTHONUNBUFFERED=1 {common} eval-gates --device cpu --eval-kind abi --arm {arm}"
        )
    lines.extend(["", "# Common teacher precondition is measured before either expert/G5 arm."])
    for start in range(0, N_BEHAVIORAL_WINDOWS, 2):
        lines.append(
            f"run_or_analyze_e4 run_gpu_e4 {common} eval-gates --device auto --max-gpu-memory 11GiB --eval-kind doc-a-teacher --start-window {start} --count 2"
        )
    lines.append(
        f"run_or_analyze_e4 run_cpu_e4 {common} precondition --bootstrap-resamples 2000"
    )
    for arm in ARMS:
        lines.extend(["", f"# {ARM_NAMES[arm]} behavioral and non-interference cells."])
        for start in range(0, N_BEHAVIORAL_WINDOWS, 2):
            lines.append(
                f"run_or_analyze_e4 run_gpu_e4 {common} eval-gates --device auto --max-gpu-memory 11GiB --eval-kind doc-a-expert --arm {arm} --start-window {start} --count 2"
            )
        for kind, stride in (("wikitext", 4), ("code", 4), ("doc-b", 2), ("guides", 4)):
            lines.append("")
            for start in range(0, N_BEHAVIORAL_WINDOWS, stride):
                lines.append(
                    f"run_or_analyze_e4 run_gpu_e4 {common} eval-gates --device auto --max-gpu-memory 11GiB --eval-kind {kind} --arm {arm} --start-window {start} --count {stride}"
                )
    lines.extend(
        [
            "",
            "set +e",
            f"run_cpu_e4 {common} analyze --bootstrap-resamples 2000",
            "analysis_rc_e4=$?",
            "set -e",
            "exit \"${analysis_rc_e4}\"",
            "",
        ]
    )
    return "\n".join(lines)


def write_lead_script(output: Path) -> None:
    path = output / "GPU_E4_RESUME_COMMANDS.sh"
    e2.write_text(path, lead_script_text(output))
    path.chmod(0o755)


def main() -> int:
    args = parse_args()
    if args.seed != DEFAULT_SEED:
        raise ValueError(f"registered E4 seed is frozen at {DEFAULT_SEED}")
    if args.mode == "prepare":
        return prepare(args)
    if args.mode == "self-test":
        return self_test(args)
    if args.mode == "bringup":
        return bringup(args)
    if args.mode == "capture-pairs":
        return capture_pairs(args)
    if args.mode == "fit-cf":
        return fit_cf(args)
    if args.mode == "train-sgd":
        return train_sgd(args)
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
