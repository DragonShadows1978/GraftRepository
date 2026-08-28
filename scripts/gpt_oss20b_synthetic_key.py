#!/usr/bin/env python3
"""MOE-RT2 frozen-router synthetic-key capture and registered analysis.

The GPU path imports the frozen MOE-RT1 harness and uses its exact streamed
GPT-OSS-20B forward path plus its non-mutating router seam.  The seam receives
the flattened post-attention-norm tensor which the native router multiplies.
RT2 stores that tensor as fp16 and never changes model weights or routing.

Modes are intentionally separate so every GPU process can be wrapped by the
registered ``flock`` and 590-second timeout:

* ``preflight``: verify frozen RT1 corpus/window identity and registered splits
* ``self-test``: exercise closed-form keys, AUC/bootstrap, and leakage rails
* ``g1``: replay a small captured batch against safetensors router weights
* ``capture``: dump one 8-window corpus chunk across all 24 router inputs
* ``analyze``: construct K1/K2, apply the fixed decision rule, and report
* ``blocked-report``: emit the registered all-RED NOT_MEASURED GPU-blocked receipt

No mode trains, optimizes, or modifies the model.  FIT data construct keys and
calibrate thresholds; every RT2 outcome metric uses odd-index EVAL windows.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import stat
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


# Avoid adding or refreshing repository __pycache__ files while importing the
# frozen RT1 harness.  The order limits new code to this script (or an analyzer).
sys.dont_write_bytecode = True
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import gpt_oss20b_router_telemetry as rt1  # noqa: E402


REPO_ROOT = SCRIPT_PATH.parents[1]
ORDER_PATH = REPO_ROOT / "orders" / "MOE_RT2_SYNTHETIC_KEY.md"
DEFAULT_RT1_DIR = REPO_ROOT / "artifacts" / "moe_rt1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "moe_rt2"

CORPORA = ("generic", "code", "domain")
ARMS = ("K1", "K2")
N_LAYERS = 24
N_EXPERTS = 32
EXPERTS_PER_TOKEN = 4
N_WINDOWS = 16
WINDOW_TOKENS = 512
WINDOWS_PER_CHUNK = 8
HIDDEN_DIM = 2880
FIT_INDICES = tuple(range(0, N_WINDOWS, 2))
EVAL_INDICES = tuple(range(1, N_WINDOWS, 2))

SCORE_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
SCORE_QUANTILE_KEYS = ("p05", "p25", "p50", "p75", "p95")
DEFAULT_SEED = 20260827
DEFAULT_BOOTSTRAP_RESAMPLES = 2000

# FP32 replay uses RT1's registered tolerances.  The persisted fp16 state has a
# separately declared tolerance; ordered top-4 equality remains exact.
G1_FP32_ATOL = rt1.G1_ATOL
G1_FP32_RTOL = rt1.G1_RTOL
G1_FP16_ATOL = 1.0e-2
G1_FP16_RTOL = 1.0e-3

GPU_WRAPPER = (
    "flock -w 7200 /tmp/forge-gpu.lock "
    "timeout --signal=TERM --kill-after=5s 590s "
    "env CUDA_VISIBLE_DEVICES=0"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MOE-RT2 frozen-router synthetic-key capture/analysis."
    )
    parser.add_argument(
        "mode",
        choices=("preflight", "self-test", "g1", "capture", "analyze", "blocked-report"),
    )
    parser.add_argument("--rt1-dir", type=Path, default=DEFAULT_RT1_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=rt1.SNAPSHOT)
    parser.add_argument("--corpus", choices=CORPORA)
    parser.add_argument("--chunk-index", type=int)
    parser.add_argument("--gate-tokens", type=int, default=8)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an incomplete/stale capture artifact for the selected chunk",
    )
    return parser.parse_args()


def now_iso() -> str:
    return rt1.now_iso()


def sha256_file(path: Path) -> str:
    return rt1.sha256_file(path)


def sha256_array(array: np.ndarray) -> str:
    return rt1.sha256_array(array)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    rt1.write_json(path, payload)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def ensure_output_scope(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        raise ValueError(f"output-dir must remain inside writable repo: {REPO_ROOT}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_record(
    records: list[dict[str, Any]],
    name: str,
    expected: Any,
    observed: Any,
    *,
    critical: bool = True,
) -> bool:
    passed = observed == expected
    records.append(
        {
            "name": name,
            "critical": bool(critical),
            "passed": bool(passed),
            "expected": expected,
            "observed": observed,
        }
    )
    return bool(passed)


def split_provenance() -> dict[str, Any]:
    fit = list(FIT_INDICES)
    evaluation = list(EVAL_INDICES)
    provenance: dict[str, Any] = {
        "schema": "moe_rt2_split_provenance_v1",
        "window_index_basis": "zero-based indices into frozen RT1 corpus_windows.npz",
        "key_construction": {
            "K1": {"domain": fit, "generic": fit},
            "K2": {"domain": fit},
        },
        "tau_calibration": {"K1": {"generic": fit}, "K2": {"generic": fit}},
        "evaluation": {corpus: evaluation for corpus in CORPORA},
        "registered_shared_fit_use": (
            "GENERIC-FIT is intentionally shared by K1 mean subtraction and both "
            "arms' tau calibration; it is never an EVAL source"
        ),
        "tokens_per_corpus_fit": len(fit) * WINDOW_TOKENS,
        "tokens_per_corpus_eval": len(evaluation) * WINDOW_TOKENS,
    }

    overlap_checks: list[dict[str, Any]] = []
    eval_set = set(EVAL_INDICES)
    for arm, corpus_map in provenance["key_construction"].items():
        for corpus, indices in corpus_map.items():
            overlap = sorted(set(indices).intersection(eval_set))
            overlap_checks.append(
                {
                    "left": f"key_construction.{arm}.{corpus}",
                    "right": f"evaluation.{corpus}",
                    "overlap": overlap,
                    "overlap_count": len(overlap),
                }
            )
    for arm, corpus_map in provenance["tau_calibration"].items():
        indices = corpus_map["generic"]
        overlap = sorted(set(indices).intersection(eval_set))
        overlap_checks.append(
            {
                "left": f"tau_calibration.{arm}.generic",
                "right": "evaluation.generic",
                "overlap": overlap,
                "overlap_count": len(overlap),
            }
        )
    total_overlap = int(sum(item["overlap_count"] for item in overlap_checks))
    provenance["fit_eval_overlap_checks"] = overlap_checks
    provenance["fit_eval_overlap_count"] = total_overlap
    provenance["planned_split_integrity"] = bool(total_overlap == 0)
    return provenance


def rt1_chunk_paths(rt1_dir: Path, corpus: str, chunk_index: int) -> tuple[Path, Path]:
    stem = f"{corpus}_chunk{chunk_index:02d}"
    return rt1_dir / f"{stem}_telemetry.npz", rt1_dir / f"{stem}_receipt.json"


def verify_frozen_inputs(rt1_dir: Path, model_dir: Path) -> dict[str, Any]:
    """Verify G0 inputs without rebuilding today's mutable source corpora."""

    rt1_dir = rt1_dir.resolve()
    model_dir = model_dir.resolve()
    manifest_path = rt1_dir / "corpora_manifest.json"
    windows_path = rt1_dir / "corpus_windows.npz"
    rt1_analysis_path = rt1_dir / "analysis.json"
    required = (manifest_path, windows_path, rt1_analysis_path, rt1.SCRIPT_PATH if hasattr(rt1, "SCRIPT_PATH") else Path(rt1.__file__))
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        return {
            "schema": "moe_rt2_g0_preflight_v1",
            "status": "failed",
            "passed": False,
            "missing_paths": missing,
            "checks": [],
            "sha_comparison_lines": [],
        }

    manifest = json_load(manifest_path)
    rt1_analysis = json_load(rt1_analysis_path)
    checks: list[dict[str, Any]] = []
    sha_lines: list[str] = []

    check_record(checks, "manifest.schema", "moe_rt1_corpora_v1", manifest.get("schema"))
    check_record(checks, "manifest.n_windows", N_WINDOWS, int(manifest.get("n_windows", -1)))
    check_record(
        checks, "manifest.window_tokens", WINDOW_TOKENS, int(manifest.get("window_tokens", -1))
    )
    check_record(
        checks,
        "manifest.windows_per_chunk",
        WINDOWS_PER_CHUNK,
        int(manifest.get("windows_per_chunk", -1)),
    )
    check_record(checks, "rt1_analysis.schema", "moe_rt1_analysis_v1", rt1_analysis.get("schema"))

    actual_windows_file_sha = sha256_file(windows_path)
    expected_windows_file_sha = manifest.get("windows_file_sha256")
    check_record(
        checks,
        "corpus_windows.npz.sha256",
        expected_windows_file_sha,
        actual_windows_file_sha,
    )
    sha_lines.append(
        "corpus_windows.npz sha256 "
        f"manifest={expected_windows_file_sha} actual={actual_windows_file_sha} "
        f"match={str(expected_windows_file_sha == actual_windows_file_sha).lower()}"
    )

    manifest_windows_path = Path(str(manifest.get("windows_path", ""))).resolve()
    check_record(
        checks,
        "manifest.windows_path",
        str(windows_path.resolve()),
        str(manifest_windows_path),
    )

    with np.load(windows_path, allow_pickle=False) as frozen:
        check_record(checks, "windows.keys", sorted(CORPORA), sorted(frozen.files))
        frozen_arrays = {
            corpus: frozen[corpus].astype(np.int64, copy=True) for corpus in CORPORA
        }

    corpus_sha_receipts: dict[str, list[str]] = {corpus: [] for corpus in CORPORA}
    for corpus in CORPORA:
        info = manifest["corpora"][corpus]
        array = frozen_arrays[corpus]
        expected_array_sha = info["windows_sha256_int64_le_host"]
        actual_array_sha = sha256_array(array)
        check_record(checks, f"{corpus}.windows.shape", [N_WINDOWS, WINDOW_TOKENS], list(array.shape))
        check_record(checks, f"{corpus}.windows.dtype", "int64", str(array.dtype))
        check_record(checks, f"{corpus}.windows.sha256", expected_array_sha, actual_array_sha)
        check_record(
            checks,
            f"{corpus}.corpus_sha.rt1_analysis",
            info["corpus_sha256"],
            rt1_analysis.get("corpus_sha256", {}).get(corpus),
        )
        sha_lines.append(
            f"{corpus.upper()} corpus sha256 manifest={info['corpus_sha256']} "
            f"rt1_analysis={rt1_analysis.get('corpus_sha256', {}).get(corpus)} "
            f"match={str(info['corpus_sha256'] == rt1_analysis.get('corpus_sha256', {}).get(corpus)).lower()}"
        )
        sha_lines.append(
            f"{corpus.upper()} window tensor sha256 manifest={expected_array_sha} "
            f"actual={actual_array_sha} match={str(expected_array_sha == actual_array_sha).lower()}"
        )

        combined_inputs: list[np.ndarray] = []
        for chunk_index in range(N_WINDOWS // WINDOWS_PER_CHUNK):
            telemetry_path, receipt_path = rt1_chunk_paths(rt1_dir, corpus, chunk_index)
            check_record(checks, f"{corpus}.chunk{chunk_index}.telemetry_exists", True, telemetry_path.is_file())
            check_record(checks, f"{corpus}.chunk{chunk_index}.receipt_exists", True, receipt_path.is_file())
            if not telemetry_path.is_file() or not receipt_path.is_file():
                continue
            receipt = json_load(receipt_path)
            expected_indices = list(
                range(chunk_index * WINDOWS_PER_CHUNK, (chunk_index + 1) * WINDOWS_PER_CHUNK)
            )
            check_record(checks, f"{corpus}.chunk{chunk_index}.status", "complete", receipt.get("status"))
            check_record(
                checks,
                f"{corpus}.chunk{chunk_index}.corpus_sha256",
                info["corpus_sha256"],
                receipt.get("corpus_sha256"),
            )
            corpus_sha_receipts[corpus].append(str(receipt.get("corpus_sha256")))
            check_record(
                checks,
                f"{corpus}.chunk{chunk_index}.window_indices",
                expected_indices,
                receipt.get("window_indices"),
            )
            check_record(
                checks,
                f"{corpus}.chunk{chunk_index}.telemetry_sha256",
                receipt.get("telemetry_file_sha256"),
                sha256_file(telemetry_path),
            )
            with np.load(telemetry_path, allow_pickle=False) as telemetry:
                chunk_inputs = telemetry["input_ids"].astype(np.int64, copy=True)
                observed_indices = telemetry["window_indices"].astype(np.int64).tolist()
            check_record(
                checks,
                f"{corpus}.chunk{chunk_index}.npz_window_indices",
                expected_indices,
                observed_indices,
            )
            check_record(
                checks,
                f"{corpus}.chunk{chunk_index}.input_ids_exact",
                True,
                bool(np.array_equal(array[expected_indices], chunk_inputs)),
            )
            combined_inputs.append(chunk_inputs)

        if len(combined_inputs) == N_WINDOWS // WINDOWS_PER_CHUNK:
            combined = np.concatenate(combined_inputs, axis=0)
            check_record(
                checks,
                f"{corpus}.combined_rt1_input_ids_exact",
                True,
                bool(np.array_equal(array, combined)),
            )
            check_record(
                checks,
                f"{corpus}.combined_rt1_input_ids_sha256",
                expected_array_sha,
                sha256_array(combined),
            )

    # The frozen forward implementation and cached snapshot must still be the
    # exact files registered by RT1 before any RT2 GPU work is allowed.
    rt1_script_path = Path(rt1.__file__).resolve()
    core_model_path = Path(manifest["core_model_path"]).resolve()
    check_record(
        checks,
        "rt1_script.sha256",
        manifest.get("script_sha256"),
        sha256_file(rt1_script_path),
    )
    check_record(
        checks,
        "core_model.sha256",
        manifest.get("core_model_sha256"),
        sha256_file(core_model_path) if core_model_path.is_file() else None,
    )
    check_record(
        checks,
        "model.config.sha256",
        manifest.get("model_config_sha256"),
        sha256_file(model_dir / "config.json") if (model_dir / "config.json").is_file() else None,
    )
    check_record(
        checks,
        "model.index.sha256",
        manifest.get("model_index_sha256"),
        sha256_file(model_dir / "model.safetensors.index.json")
        if (model_dir / "model.safetensors.index.json").is_file()
        else None,
    )
    check_record(
        checks,
        "model.snapshot_revision",
        manifest.get("model_snapshot_revision"),
        model_dir.name,
    )

    for corpus in CORPORA:
        expected_sha = manifest["corpora"][corpus]["corpus_sha256"]
        check_record(
            checks,
            f"{corpus}.corpus_sha.all_rt1_chunks",
            [expected_sha, expected_sha],
            corpus_sha_receipts[corpus],
        )
        native_layers = rt1_analysis.get("per_corpus_layer_metrics", {}).get(corpus, [])
        check_record(checks, f"{corpus}.native_calibration.layer_count", N_LAYERS, len(native_layers))
        if len(native_layers) == N_LAYERS:
            calibration_keys_ok = all(
                set(item.get("calibration", {}).get("top1_logit", {}))
                == set(SCORE_QUANTILE_KEYS)
                for item in native_layers
            )
            check_record(
                checks,
                f"{corpus}.native_calibration.top1_quantile_keys",
                True,
                bool(calibration_keys_ok),
            )

    failed = [item for item in checks if item["critical"] and not item["passed"]]
    return {
        "schema": "moe_rt2_g0_preflight_v1",
        "status": "passed" if not failed else "failed",
        "passed": not failed,
        "created_at": now_iso(),
        "rt1_dir": str(rt1_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "windows_path": str(windows_path),
        "rt1_analysis_path": str(rt1_analysis_path),
        "rt1_analysis_sha256": sha256_file(rt1_analysis_path),
        "model_dir": str(model_dir),
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "failed_check_names": [item["name"] for item in failed],
        "checks": checks,
        "sha_comparison_lines": sha_lines,
        "manifest": manifest,
    }


def preflight(args: argparse.Namespace) -> int:
    out = ensure_output_scope(args.output_dir)
    verification = verify_frozen_inputs(args.rt1_dir, args.model_dir)
    splits = split_provenance()
    payload = {
        "schema": "moe_rt2_preflight_v1",
        "created_at": now_iso(),
        "order": str(ORDER_PATH.resolve()),
        "rt2_script": str(SCRIPT_PATH),
        "rt2_script_sha256": sha256_file(SCRIPT_PATH),
        "status": (
            "passed"
            if verification.get("passed") and splits["planned_split_integrity"]
            else "failed_stop"
        ),
        "g0_cpu_identity_verification": verification,
        "g2_registered_split_plan": splits,
        "official_blocked_protocol_note": (
            "CPU checks are preflight evidence. If CUDA is unavailable, the order "
            "requires all official gates RED/NOT_MEASURED despite CPU checks passing."
        ),
    }
    path = out / "preflight.json"
    write_json(path, payload)
    for line in verification.get("sha_comparison_lines", []):
        print(line, flush=True)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "preflight": str(path),
                "identity_checks": (
                    f"{verification.get('checks_passed', 0)}/"
                    f"{verification.get('checks_total', 0)}"
                ),
                "fit_eval_overlap_count": splits["fit_eval_overlap_count"],
            }
        ),
        flush=True,
    )
    return 0 if payload["status"] == "passed" else 2


def require_frozen_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    verification = verify_frozen_inputs(args.rt1_dir, args.model_dir)
    if not verification.get("passed"):
        failed = ", ".join(verification.get("failed_check_names", []))
        raise RuntimeError(f"G0 frozen-input mismatch; stop before model work: {failed}")
    splits = split_provenance()
    if not splits["planned_split_integrity"]:
        raise RuntimeError("G2 registered FIT/EVAL plan overlaps; stop")
    return verification, splits


def cuda_environment_probe() -> dict[str, Any]:
    device_entries = sorted(Path("/dev").glob("nvidia*"))
    device_nodes: list[str] = []
    for path in device_entries:
        try:
            if stat.S_ISCHR(path.stat().st_mode):
                device_nodes.append(str(path))
        except OSError:
            continue
    command = ["nvidia-smi", "-L"]
    try:
        proc = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        nvidia = {
            "argv": command,
            "returncode": int(proc.returncode),
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover - environment-specific
        nvidia = {
            "argv": command,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }
    torch_probe: dict[str, Any]
    try:
        import torch

        torch_probe = {
            "version": torch.__version__,
            "cuda_is_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        }
    except Exception as exc:  # pragma: no cover - environment-specific
        torch_probe = {"error_type": type(exc).__name__, "error": str(exc)}
    return {
        "device_entries": [str(path) for path in device_entries],
        "device_nodes": device_nodes,
        "device_node_count": len(device_nodes),
        "nvidia_smi": nvidia,
        "torch": torch_probe,
    }


def capture_paths(out: Path, corpus: str, chunk_index: int) -> tuple[Path, Path, Path]:
    stem = f"{corpus}_chunk{chunk_index:02d}_router_inputs_fp16"
    final_path = out / f"{stem}.npy"
    partial_path = out / f"{stem}.partial.npy"
    receipt_path = out / f"{corpus}_chunk{chunk_index:02d}_capture_receipt.json"
    return final_path, partial_path, receipt_path


def expected_capture_shape() -> tuple[int, int, int, int]:
    return (N_LAYERS, WINDOWS_PER_CHUNK, WINDOW_TOKENS, HIDDEN_DIM)


def expected_capture_payload_nbytes() -> int:
    return int(np.prod(expected_capture_shape(), dtype=np.int64) * np.dtype(np.float16).itemsize)


def capture(args: argparse.Namespace) -> int:
    if args.corpus is None or args.chunk_index is None:
        raise ValueError("capture requires --corpus and --chunk-index")
    if args.chunk_index not in range(N_WINDOWS // WINDOWS_PER_CHUNK):
        raise ValueError("chunk-index must be 0 or 1")

    out = ensure_output_scope(args.output_dir)
    verification, _splits = require_frozen_inputs(args)
    manifest = verification["manifest"]
    windows_path = Path(verification["windows_path"])
    start = args.chunk_index * WINDOWS_PER_CHUNK
    stop = start + WINDOWS_PER_CHUNK
    window_indices = list(range(start, stop))
    with np.load(windows_path, allow_pickle=False) as frozen:
        ids = np.ascontiguousarray(frozen[args.corpus][start:stop]).astype(
            np.int64, copy=False
        )
    if ids.shape != (WINDOWS_PER_CHUNK, WINDOW_TOKENS):
        raise RuntimeError(f"unexpected frozen chunk shape: {ids.shape}")

    final_path, partial_path, receipt_path = capture_paths(
        out, args.corpus, args.chunk_index
    )
    if final_path.is_file() and receipt_path.is_file() and not args.overwrite:
        prior = json_load(receipt_path)
        if (
            prior.get("status") == "complete"
            and prior.get("capture_file_sha256") == sha256_file(final_path)
        ):
            print(
                json.dumps(
                    {
                        "status": "already_complete",
                        "capture": str(final_path),
                        "receipt": str(receipt_path),
                    }
                ),
                flush=True,
            )
            return 0
        raise FileExistsError(
            f"stale/incomplete capture exists; inspect it or rerun with --overwrite: {final_path}"
        )
    if (final_path.exists() or partial_path.exists() or receipt_path.exists()) and not args.overwrite:
        raise FileExistsError(
            "capture output already exists; inspect it or use --overwrite for this exact chunk"
        )
    if args.overwrite:
        for path in (final_path, partial_path, receipt_path):
            if path.exists():
                path.unlink()

    started = time.perf_counter()
    receipt: dict[str, Any] = {
        "schema": "moe_rt2_router_input_capture_v1",
        "status": "starting",
        "created_at": now_iso(),
        "argv": sys.argv,
        "required_shell_wrapper": GPU_WRAPPER,
        "corpus": args.corpus,
        "chunk_index": int(args.chunk_index),
        "window_indices": window_indices,
        "window_offsets": manifest["corpora"][args.corpus]["window_offsets"][start:stop],
        "input_shape": list(ids.shape),
        "input_ids_sha256": sha256_array(ids),
        "frozen_windows_file_sha256": manifest["windows_file_sha256"],
        "corpus_sha256": manifest["corpora"][args.corpus]["corpus_sha256"],
        "model_dir": str(args.model_dir.resolve()),
        "model_snapshot_revision": args.model_dir.resolve().name,
        "rt2_script_path": str(SCRIPT_PATH),
        "rt2_script_sha256_at_run": sha256_file(SCRIPT_PATH),
        "rt1_script_path": str(Path(rt1.__file__).resolve()),
        "rt1_script_sha256": sha256_file(Path(rt1.__file__).resolve()),
        "core_model_sha256": manifest["core_model_sha256"],
        "attention_mode": "standard",
        "expert_mode": "resident_packed_mxfp4",
        "compute_dtype": "bfloat16",
        "router_input_source_dtype": "float32 host copy of native x_flat.float()",
        "capture_storage_dtype": "float16",
        "capture_shape_expected": list(expected_capture_shape()),
        "capture_payload_nbytes_expected": expected_capture_payload_nbytes(),
        "capture_path": str(final_path),
        "partial_path": str(partial_path),
        "completed_layers": 0,
        "layer_wall_seconds": [],
        "layer_router_input_fp16_sha256": [],
        "layer_fp16_roundtrip_max_abs": [],
        "attention_backends": [],
        "gpu_before": rt1.nvidia_smi(),
        "cuda_environment": cuda_environment_probe(),
    }
    write_json(receipt_path, receipt)

    try:
        runtime = rt1.load_runtime()
        tc = runtime["tc"]
        cfg = runtime["GptOss20BConfig"].from_model_dir(args.model_dir.resolve())
        where = runtime["build_safetensors_map"](args.model_dir.resolve())
        observed_contract = (
            cfg.num_layers,
            cfg.num_local_experts,
            cfg.num_experts_per_tok,
            cfg.hidden_dim,
        )
        expected_contract = (N_LAYERS, N_EXPERTS, EXPERTS_PER_TOKEN, HIDDEN_DIM)
        if observed_contract != expected_contract:
            raise RuntimeError(
                f"model contract {observed_contract} != registered {expected_contract}"
            )

        dump = np.lib.format.open_memmap(
            partial_path,
            mode="w+",
            dtype=np.float16,
            shape=expected_capture_shape(),
        )
        with tc.no_grad():
            embed = runtime["GptOssRowEmbedding"](where)
            h = embed(ids)
            cos, sin = runtime["gpt_oss_yarn_rope_tables"](cfg, WINDOW_TOKENS)
            for layer_idx in range(N_LAYERS):
                layer_started = time.perf_counter()
                block = runtime["GptOssDiagnosticBlockTC"].from_safetensors(
                    cfg,
                    where,
                    layer_idx,
                    expert_mode="resident_packed_mxfp4",
                )
                block.self_attn.attention_mode = "standard"
                block.mlp.route_detail = "summary"
                block.mlp.empty_cache_interval = 0
                capture_state: dict[str, np.ndarray] = {}
                rt1.install_router_capture(
                    block.mlp, capture_state, capture_input=True
                )
                h, kv, route_info = block(h, cos, sin)
                tc.synchronize()
                router_input = capture_state.get("router_input")
                expected_input_shape = (
                    WINDOWS_PER_CHUNK * WINDOW_TOKENS,
                    HIDDEN_DIM,
                )
                if router_input is None or router_input.shape != expected_input_shape:
                    observed = None if router_input is None else router_input.shape
                    raise RuntimeError(
                        f"layer {layer_idx} router input {observed} != {expected_input_shape}"
                    )
                stored = router_input.reshape(
                    WINDOWS_PER_CHUNK, WINDOW_TOKENS, HIDDEN_DIM
                ).astype(np.float16)
                if not np.isfinite(stored).all():
                    raise RuntimeError(f"layer {layer_idx} fp16 capture contains non-finite values")
                dump[layer_idx] = stored
                dump.flush()
                receipt["layer_router_input_fp16_sha256"].append(sha256_array(stored))
                receipt["layer_fp16_roundtrip_max_abs"].append(
                    float(
                        np.max(
                            np.abs(
                                router_input
                                - stored.reshape(-1, HIDDEN_DIM).astype(np.float32)
                            )
                        )
                    )
                )
                elapsed = time.perf_counter() - layer_started
                receipt["completed_layers"] = layer_idx + 1
                receipt["layer_wall_seconds"].append(float(elapsed))
                receipt["attention_backends"].append(
                    block.self_attn.last_attention_backend
                )
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
                del block, kv, route_info, capture_state, router_input, stored
                gc.collect()
                if hasattr(tc, "empty_cache"):
                    tc.empty_cache()

            final_hidden = h.float().numpy().astype(np.float32, copy=True)
            receipt["final_hidden_shape"] = list(final_hidden.shape)
            receipt["final_hidden_fp32_sha256"] = sha256_array(final_hidden)
            del final_hidden, h, embed, cos, sin
            if hasattr(tc, "empty_cache"):
                tc.empty_cache()

        dump.flush()
        del dump
        partial_path.replace(final_path)
        loaded = np.load(final_path, mmap_mode="r", allow_pickle=False)
        if loaded.shape != expected_capture_shape() or loaded.dtype != np.float16:
            raise RuntimeError(
                f"persisted capture contract mismatch: {loaded.shape} {loaded.dtype}"
            )
        receipt.update(
            {
                "status": "complete",
                "wall_seconds": float(time.perf_counter() - started),
                "gpu_after": rt1.nvidia_smi(),
                "capture_shape": list(loaded.shape),
                "capture_dtype": str(loaded.dtype),
                "capture_payload_nbytes": int(loaded.nbytes),
                "capture_file_bytes": int(final_path.stat().st_size),
                "capture_file_sha256": sha256_file(final_path),
            }
        )
        write_json(receipt_path, receipt)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "capture": str(final_path),
                    "receipt": str(receipt_path),
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
                "gpu_after": rt1.nvidia_smi(),
                "partial_file_exists": partial_path.exists(),
                "partial_file_bytes": (
                    int(partial_path.stat().st_size) if partial_path.exists() else 0
                ),
            }
        )
        write_json(receipt_path, receipt)
        raise


def ordered_top4(logits: np.ndarray) -> np.ndarray:
    return np.argsort(-logits, axis=-1, kind="stable")[:, :EXPERTS_PER_TOKEN]


def g1_capture_truth(args: argparse.Namespace) -> int:
    if args.gate_tokens < 1 or args.gate_tokens > WINDOW_TOKENS:
        raise ValueError("gate-tokens must be in [1, 512]")
    out = ensure_output_scope(args.output_dir)
    verification, _splits = require_frozen_inputs(args)
    manifest = verification["manifest"]
    with np.load(verification["windows_path"], allow_pickle=False) as frozen:
        ids = np.ascontiguousarray(frozen["generic"][0:1, : args.gate_tokens]).astype(
            np.int64, copy=False
        )

    receipt_path = out / "g1_capture_truth.json"
    started = time.perf_counter()
    receipt: dict[str, Any] = {
        "schema": "moe_rt2_g1_capture_truth_v1",
        "status": "starting",
        "created_at": now_iso(),
        "argv": sys.argv,
        "required_shell_wrapper": GPU_WRAPPER,
        "model_dir": str(args.model_dir.resolve()),
        "model_snapshot_revision": args.model_dir.resolve().name,
        "rt2_script_path": str(SCRIPT_PATH),
        "rt2_script_sha256_at_run": sha256_file(SCRIPT_PATH),
        "input_corpus": "generic",
        "input_window_index": 0,
        "input_shape": list(ids.shape),
        "input_ids": ids.tolist(),
        "input_ids_sha256": sha256_array(ids),
        "frozen_windows_file_sha256": manifest["windows_file_sha256"],
        "rt1_capture_helper": "gpt_oss20b_router_telemetry.install_router_capture",
        "reference": "NumPy FP32 captured_h @ safetensors_weight.T + safetensors_bias",
        "fp32_tolerance": {"atol": G1_FP32_ATOL, "rtol": G1_FP32_RTOL},
        "fp16_persisted_state_tolerance": {
            "atol": G1_FP16_ATOL,
            "rtol": G1_FP16_RTOL,
        },
        "completed_layers": 0,
        "layers": [],
        "gpu_before": rt1.nvidia_smi(),
        "cuda_environment": cuda_environment_probe(),
    }
    write_json(receipt_path, receipt)

    try:
        runtime = rt1.load_runtime()
        tc = runtime["tc"]
        cfg = runtime["GptOss20BConfig"].from_model_dir(args.model_dir.resolve())
        where = runtime["build_safetensors_map"](args.model_dir.resolve())
        if (
            cfg.num_layers,
            cfg.num_local_experts,
            cfg.num_experts_per_tok,
            cfg.hidden_dim,
        ) != (N_LAYERS, N_EXPERTS, EXPERTS_PER_TOKEN, HIDDEN_DIM):
            raise RuntimeError("model config is not the registered GPT-OSS-20B contract")
        cos, sin = runtime["gpt_oss_yarn_rope_tables"](cfg, args.gate_tokens)

        with tc.no_grad():
            embed = runtime["GptOssRowEmbedding"](where)
            h = embed(ids)
            for layer_idx in range(N_LAYERS):
                layer_started = time.perf_counter()
                block = runtime["GptOssDiagnosticBlockTC"].from_safetensors(
                    cfg,
                    where,
                    layer_idx,
                    expert_mode="resident_packed_mxfp4",
                )
                block.self_attn.attention_mode = "standard"
                block.mlp.route_detail = "summary"
                block.mlp.empty_cache_interval = 0
                captured: dict[str, np.ndarray] = {}
                rt1.install_router_capture(block.mlp, captured, capture_input=True)
                h, kv, route_info = block(h, cos, sin)
                tc.synchronize()

                prefix = f"model.layers.{layer_idx}.mlp.router"
                weight = runtime["load_tensor_np"](where, f"{prefix}.weight").astype(
                    np.float32, copy=False
                )
                bias = runtime["load_tensor_np"](where, f"{prefix}.bias").astype(
                    np.float32, copy=False
                )
                captured_h = captured["router_input"].astype(np.float32, copy=False)
                native_logits = captured["router_logits"].astype(np.float32, copy=False)
                native_top4 = captured["top_indices"].astype(np.int64, copy=False)

                fp32_logits = captured_h @ weight.T + bias
                fp32_diff = np.abs(fp32_logits - native_logits)
                fp32_top4 = ordered_top4(fp32_logits)
                fp32_close = bool(
                    np.allclose(
                        fp32_logits,
                        native_logits,
                        atol=G1_FP32_ATOL,
                        rtol=G1_FP32_RTOL,
                        equal_nan=False,
                    )
                )
                fp32_top4_exact = bool(np.array_equal(fp32_top4, native_top4))

                # Serialize in-memory through the exact .npy fp16 representation
                # used by capture mode, then replay the loaded state.
                buffer = io.BytesIO()
                np.save(buffer, captured_h.astype(np.float16), allow_pickle=False)
                buffer.seek(0)
                stored_h_fp16 = np.load(buffer, allow_pickle=False)
                stored_h = stored_h_fp16.astype(np.float32)
                fp16_logits = stored_h @ weight.T + bias
                fp16_diff = np.abs(fp16_logits - native_logits)
                fp16_top4 = ordered_top4(fp16_logits)
                fp16_close = bool(
                    np.allclose(
                        fp16_logits,
                        native_logits,
                        atol=G1_FP16_ATOL,
                        rtol=G1_FP16_RTOL,
                        equal_nan=False,
                    )
                )
                fp16_top4_exact = bool(np.array_equal(fp16_top4, native_top4))
                exact_fp32_tokens = int(
                    np.all(fp32_top4 == native_top4, axis=1).sum()
                )
                exact_fp16_tokens = int(
                    np.all(fp16_top4 == native_top4, axis=1).sum()
                )
                layer_green = bool(
                    fp32_close
                    and fp32_top4_exact
                    and fp16_close
                    and fp16_top4_exact
                )
                layer_receipt = {
                    "layer": layer_idx,
                    "token_count": int(captured_h.shape[0]),
                    "router_input_shape": list(captured_h.shape),
                    "router_weight_name": f"{prefix}.weight",
                    "router_bias_name": f"{prefix}.bias",
                    "safetensors_shard": where[f"{prefix}.weight"],
                    "router_weight_fp32_sha256": sha256_array(weight),
                    "router_bias_fp32_sha256": sha256_array(bias),
                    "captured_h_fp32_sha256": sha256_array(captured_h),
                    "persisted_h_fp16_sha256": sha256_array(stored_h_fp16),
                    "fp32_capture_replay": {
                        "logits_within_tolerance": fp32_close,
                        "max_abs_diff": float(fp32_diff.max()),
                        "mean_abs_diff": float(fp32_diff.mean()),
                        "top4_indices_exact": fp32_top4_exact,
                        "top4_exact_tokens": exact_fp32_tokens,
                        "top4_total_tokens": int(captured_h.shape[0]),
                    },
                    "fp16_persisted_replay": {
                        "state_roundtrip_max_abs_diff": float(
                            np.max(np.abs(captured_h - stored_h))
                        ),
                        "logits_within_tolerance": fp16_close,
                        "max_abs_diff": float(fp16_diff.max()),
                        "mean_abs_diff": float(fp16_diff.mean()),
                        "top4_indices_exact": fp16_top4_exact,
                        "top4_exact_tokens": exact_fp16_tokens,
                        "top4_total_tokens": int(captured_h.shape[0]),
                    },
                    "verdict": "GREEN" if layer_green else "RED",
                    "wall_seconds": float(time.perf_counter() - layer_started),
                }
                receipt["layers"].append(layer_receipt)
                receipt["completed_layers"] = layer_idx + 1
                receipt["status"] = "running"
                receipt["wall_seconds"] = float(time.perf_counter() - started)
                write_json(receipt_path, receipt)
                print(
                    json.dumps(
                        {
                            "layer": layer_idx,
                            "verdict": layer_receipt["verdict"],
                            "fp16_top4_exact": (
                                f"{exact_fp16_tokens}/{captured_h.shape[0]}"
                            ),
                            "fp16_max_abs_diff": float(fp16_diff.max()),
                        }
                    ),
                    flush=True,
                )
                del (
                    block,
                    kv,
                    route_info,
                    captured,
                    weight,
                    bias,
                    captured_h,
                    native_logits,
                    native_top4,
                    fp32_logits,
                    fp32_diff,
                    fp32_top4,
                    buffer,
                    stored_h_fp16,
                    stored_h,
                    fp16_logits,
                    fp16_diff,
                    fp16_top4,
                )
                gc.collect()
                if hasattr(tc, "empty_cache"):
                    tc.empty_cache()

            del h, embed, cos, sin
            if hasattr(tc, "empty_cache"):
                tc.empty_cache()

        all_layers_green = bool(
            len(receipt["layers"]) == N_LAYERS
            and all(item["verdict"] == "GREEN" for item in receipt["layers"])
        )
        fp32_exact = int(
            sum(
                item["fp32_capture_replay"]["top4_exact_tokens"]
                for item in receipt["layers"]
            )
        )
        fp16_exact = int(
            sum(
                item["fp16_persisted_replay"]["top4_exact_tokens"]
                for item in receipt["layers"]
            )
        )
        total_tokens = int(
            sum(
                item["fp16_persisted_replay"]["top4_total_tokens"]
                for item in receipt["layers"]
            )
        )
        receipt.update(
            {
                "status": "complete",
                "verdict": "GREEN" if all_layers_green else "RED",
                "wall_seconds": float(time.perf_counter() - started),
                "gpu_after": rt1.nvidia_smi(),
                "layers_green": int(
                    sum(item["verdict"] == "GREEN" for item in receipt["layers"])
                ),
                "layers_total": N_LAYERS,
                "fp32_top4_exact_tokens": fp32_exact,
                "fp16_top4_exact_tokens": fp16_exact,
                "top4_total_layer_tokens": total_tokens,
                "fp32_max_abs_diff": float(
                    max(
                        item["fp32_capture_replay"]["max_abs_diff"]
                        for item in receipt["layers"]
                    )
                ),
                "fp16_max_abs_diff": float(
                    max(
                        item["fp16_persisted_replay"]["max_abs_diff"]
                        for item in receipt["layers"]
                    )
                ),
            }
        )
        write_json(receipt_path, receipt)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "verdict": receipt["verdict"],
                    "layers_green": f"{receipt['layers_green']}/{N_LAYERS}",
                    "fp16_top4_exact": f"{fp16_exact}/{total_tokens}",
                    "receipt": str(receipt_path),
                }
            ),
            flush=True,
        )
        return 0 if all_layers_green else 3
    except Exception as exc:
        receipt.update(
            {
                "status": "error",
                "verdict": "RED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "wall_seconds": float(time.perf_counter() - started),
                "gpu_after": rt1.nvidia_smi(),
            }
        )
        write_json(receipt_path, receipt)
        raise


def exact_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Mann-Whitney token AUC with ties worth one half."""

    positive = np.asarray(positive, dtype=np.float64).reshape(-1)
    negative = np.asarray(negative, dtype=np.float64).reshape(-1)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("AUC requires both classes")
    if not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError("AUC inputs must be finite")
    sorted_negative = np.sort(negative)
    left = np.searchsorted(sorted_negative, positive, side="left")
    right = np.searchsorted(sorted_negative, positive, side="right")
    wins = left.astype(np.float64) + 0.5 * (right - left)
    return float(wins.sum() / (positive.size * negative.size))


def pairwise_window_auc_matrix(
    positive_windows: np.ndarray, negative_windows: np.ndarray
) -> np.ndarray:
    positive_windows = np.asarray(positive_windows)
    negative_windows = np.asarray(negative_windows)
    if positive_windows.ndim != 2 or negative_windows.ndim != 2:
        raise ValueError("window AUC inputs must be [windows, tokens]")
    if positive_windows.shape[1] != negative_windows.shape[1]:
        raise ValueError("positive and negative windows must have equal token counts")
    matrix = np.empty(
        (positive_windows.shape[0], negative_windows.shape[0]), dtype=np.float64
    )
    for positive_idx, positive in enumerate(positive_windows):
        for negative_idx, negative in enumerate(negative_windows):
            matrix[positive_idx, negative_idx] = exact_auc(positive, negative)
    return matrix


def bootstrap_window_auc(
    positive_windows: np.ndarray,
    negative_windows: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    if resamples < 1:
        raise ValueError("bootstrap-resamples must be positive")
    pairwise = pairwise_window_auc_matrix(positive_windows, negative_windows)
    n_positive, n_negative = pairwise.shape
    rng = np.random.default_rng(seed)
    positive_indices = rng.integers(
        0, n_positive, size=(resamples, n_positive)
    )
    negative_indices = rng.integers(
        0, n_negative, size=(resamples, n_negative)
    )
    boot = pairwise[
        positive_indices[:, :, None], negative_indices[:, None, :]
    ].mean(axis=(1, 2))
    low, high = np.quantile(boot, [0.025, 0.975], method="linear")
    return {
        "value": float(pairwise.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_unit": "windows",
        "bootstrap_resamples": int(resamples),
    }


def bootstrap_window_rate(
    token_indicator: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    indicator = np.asarray(token_indicator, dtype=np.float64)
    if indicator.ndim != 2:
        raise ValueError("rate input must be [windows, tokens]")
    per_window = indicator.mean(axis=1)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, per_window.size, size=(resamples, per_window.size)
    )
    boot = per_window[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975], method="linear")
    return {
        "value": float(per_window.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_unit": "windows",
        "bootstrap_resamples": int(resamples),
    }


def bootstrap_window_majority(
    token_indicator: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    indicator = np.asarray(token_indicator, dtype=np.float64)
    if indicator.ndim != 2:
        raise ValueError("majority input must be [windows, tokens]")
    majority = (indicator.mean(axis=1) > 0.5).astype(np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, majority.size, size=(resamples, majority.size))
    boot = majority[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975], method="linear")
    return {
        "value": float(majority.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "majority_window_count": int(majority.sum()),
        "eval_window_count": int(majority.size),
        "definition": "strictly more than 50% of tokens have score >= tau",
        "bootstrap_unit": "windows",
        "bootstrap_resamples": int(resamples),
    }


def bootstrap_score_quantiles(
    score_windows: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Pooled token quantiles with a nonparametric window-cluster bootstrap."""

    values = np.asarray(score_windows, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("score quantiles require [windows, tokens]")
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("score quantiles require non-empty finite values")
    if resamples < 1:
        raise ValueError("bootstrap-resamples must be positive")
    point = np.quantile(
        values.reshape(-1), SCORE_QUANTILES, method="linear"
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, values.shape[0], size=(resamples, values.shape[0])
    )
    sampled = values[indices].reshape(resamples, -1)
    boot = np.quantile(
        sampled, SCORE_QUANTILES, axis=1, method="linear"
    )
    low = np.quantile(boot, 0.025, axis=1, method="linear")
    high = np.quantile(boot, 0.975, axis=1, method="linear")
    return {
        key: {
            "value": float(value),
            "ci95_low": float(low_value),
            "ci95_high": float(high_value),
            "bootstrap_unit": "windows",
            "bootstrap_resamples": int(resamples),
        }
        for key, value, low_value, high_value in zip(
            SCORE_QUANTILE_KEYS, point, low, high
        )
    }


def unit_vector(vector: np.ndarray, *, name: str) -> tuple[np.ndarray, float]:
    work = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(work))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise RuntimeError(f"{name} has degenerate norm {norm}")
    key = np.ascontiguousarray((work / norm).astype(np.float32))
    return key, norm


def construct_keys(
    hidden_by_corpus: dict[str, np.ndarray],
    fit_indices: tuple[int, ...] = FIT_INDICES,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    domain = np.asarray(hidden_by_corpus["domain"])
    generic = np.asarray(hidden_by_corpus["generic"])
    if domain.ndim != 3 or generic.shape != domain.shape:
        raise ValueError("domain/generic hidden states must share [windows,tokens,hidden]")
    selected = np.asarray(fit_indices, dtype=np.int64)
    domain_mean = domain[selected].mean(axis=(0, 1), dtype=np.float64)
    generic_mean = generic[selected].mean(axis=(0, 1), dtype=np.float64)
    k1, k1_source_norm = unit_vector(
        domain_mean - generic_mean, name="K1 domain-minus-generic mean"
    )
    k2, k2_source_norm = unit_vector(domain_mean, name="K2 domain centroid")
    keys = {"K1": k1, "K2": k2}
    metadata = {
        "fit_window_indices": list(fit_indices),
        "fit_domain_token_count": int(len(fit_indices) * domain.shape[1]),
        "fit_generic_token_count_K1": int(len(fit_indices) * generic.shape[1]),
        "K1": {
            "construction": "unit_norm(mean(domain_FIT) - mean(generic_FIT))",
            "pre_normalization_norm": k1_source_norm,
            "stored_fp32_norm": float(np.linalg.norm(k1.astype(np.float64))),
            "stored_fp32_sha256": sha256_array(k1),
        },
        "K2": {
            "construction": "unit_norm(mean(domain_FIT))",
            "pre_normalization_norm": k2_source_norm,
            "stored_fp32_norm": float(np.linalg.norm(k2.astype(np.float64))),
            "stored_fp32_sha256": sha256_array(k2),
        },
    }
    return keys, metadata


def score_hidden(
    hidden_by_corpus: dict[str, np.ndarray], keys: dict[str, np.ndarray]
) -> dict[str, dict[str, np.ndarray]]:
    key_matrix = np.stack([keys[arm] for arm in ARMS], axis=1).astype(
        np.float32, copy=False
    )
    result: dict[str, dict[str, np.ndarray]] = {}
    for corpus in CORPORA:
        hidden = np.asarray(hidden_by_corpus[corpus])
        flat = hidden.astype(np.float32).reshape(-1, hidden.shape[-1])
        scores = (flat @ key_matrix).reshape(hidden.shape[0], hidden.shape[1], len(ARMS))
        result[corpus] = {
            arm: np.ascontiguousarray(scores[:, :, arm_idx])
            for arm_idx, arm in enumerate(ARMS)
        }
    return result


def analyze_arm_scores(
    scores: dict[str, np.ndarray],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    fit = np.asarray(FIT_INDICES, dtype=np.int64)
    evaluation = np.asarray(EVAL_INDICES, dtype=np.int64)
    generic_fit = scores["generic"][fit]
    tau = float(np.quantile(generic_fit, 0.99, method="linear"))
    eval_scores = {corpus: scores[corpus][evaluation] for corpus in CORPORA}
    fired = {corpus: eval_scores[corpus] >= tau for corpus in CORPORA}

    auc_domain_generic = bootstrap_window_auc(
        eval_scores["domain"],
        eval_scores["generic"],
        resamples=resamples,
        seed=seed + 1,
    )
    auc_domain_code = bootstrap_window_auc(
        eval_scores["domain"],
        eval_scores["code"],
        resamples=resamples,
        seed=seed + 2,
    )
    rates = {
        corpus: bootstrap_window_rate(
            fired[corpus], resamples=resamples, seed=seed + 10 + offset
        )
        for offset, corpus in enumerate(CORPORA)
    }
    majorities = {
        corpus: bootstrap_window_majority(
            fired[corpus], resamples=resamples, seed=seed + 20 + offset
        )
        for offset, corpus in enumerate(CORPORA)
    }
    return {
        "tau": tau,
        "tau_definition": "NumPy linear p99 over 4096 GENERIC-FIT token scores",
        "fire_comparison": "score >= tau",
        "auc_domain_vs_generic": auc_domain_generic,
        "auc_domain_vs_code": auc_domain_code,
        "domain_recall_at_tau": rates["domain"],
        "generic_fpr_at_tau": rates["generic"],
        "code_fpr_at_tau": rates["code"],
        "window_majority_firing": majorities,
        "score_quantiles_eval": {
            corpus: bootstrap_score_quantiles(
                eval_scores[corpus],
                resamples=resamples,
                seed=seed + 30 + offset,
            )
            for offset, corpus in enumerate(CORPORA)
        },
        "metric_window_indices": {
            corpus: list(EVAL_INDICES) for corpus in CORPORA
        },
        "tau_window_indices": {"generic": list(FIT_INDICES)},
    }


def run_self_test(args: argparse.Namespace) -> int:
    out = ensure_output_scope(args.output_dir)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    splits = split_provenance()
    check(
        "registered_split_has_zero_fit_eval_overlap",
        splits["fit_eval_overlap_count"] == 0,
        splits["fit_eval_overlap_checks"],
    )
    check(
        "registered_index_lists",
        FIT_INDICES == tuple(range(0, 16, 2))
        and EVAL_INDICES == tuple(range(1, 16, 2)),
        {"fit": list(FIT_INDICES), "eval": list(EVAL_INDICES)},
    )

    positive = np.asarray([[2.0, 3.0], [4.0, 5.0]])
    negative = np.asarray([[0.0, 1.0], [1.0, 2.0]])
    brute = float(
        np.mean(
            (positive.reshape(-1, 1) > negative.reshape(1, -1))
            + 0.5
            * (positive.reshape(-1, 1) == negative.reshape(1, -1))
        )
    )
    computed = exact_auc(positive, negative)
    check("exact_auc_matches_bruteforce", abs(computed - brute) < 1.0e-15, {"computed": computed, "brute": brute})
    tie_auc = exact_auc(np.ones(8), np.ones(8))
    check("auc_ties_are_half", tie_auc == 0.5, tie_auc)

    boot_a = bootstrap_window_auc(positive, negative, resamples=200, seed=args.seed)
    boot_b = bootstrap_window_auc(positive, negative, resamples=200, seed=args.seed)
    check("window_bootstrap_is_deterministic", boot_a == boot_b, boot_a)

    rng = np.random.default_rng(args.seed)
    shape = (N_WINDOWS, 32, 8)
    generic = rng.normal(0.0, 1.0, size=shape).astype(np.float16)
    code = rng.normal(0.0, 1.0, size=shape).astype(np.float16)
    domain = rng.normal(0.0, 1.0, size=shape).astype(np.float16)
    domain[:, :, 0] += np.float16(4.0)
    hidden = {"generic": generic, "code": code, "domain": domain}
    keys, key_meta = construct_keys(hidden)
    check(
        "closed_form_keys_are_unit_norm",
        all(abs(np.linalg.norm(keys[arm].astype(np.float64)) - 1.0) < 1.0e-6 for arm in ARMS),
        {arm: float(np.linalg.norm(keys[arm].astype(np.float64))) for arm in ARMS},
    )
    scores = score_hidden(hidden, keys)
    metrics = analyze_arm_scores(
        {corpus: scores[corpus]["K1"] for corpus in CORPORA},
        resamples=200,
        seed=args.seed + 100,
    )
    check(
        "synthetic_shift_has_high_K1_eval_auc",
        metrics["auc_domain_vs_generic"]["value"] > 0.99,
        metrics["auc_domain_vs_generic"],
    )
    score_quantile_contract = all(
        set(metrics["score_quantiles_eval"][corpus])
        == set(SCORE_QUANTILE_KEYS)
        and all(
            item["bootstrap_unit"] == "windows"
            and item["bootstrap_resamples"] == 200
            and np.isfinite(
                [item["value"], item["ci95_low"], item["ci95_high"]]
            ).all()
            for item in metrics["score_quantiles_eval"][corpus].values()
        )
        for corpus in CORPORA
    )
    check(
        "eval_score_quantiles_have_window_bootstrap_cis",
        score_quantile_contract,
        metrics["score_quantiles_eval"],
    )

    # Mutating EVAL data must not change a key or tau because both functions
    # select FIT indices explicitly.
    mutated_hidden = {corpus: array.copy() for corpus, array in hidden.items()}
    mutated_hidden["domain"][np.asarray(EVAL_INDICES)] += np.float16(20.0)
    mutated_keys, _ = construct_keys(mutated_hidden)
    check(
        "key_construction_ignores_eval_windows",
        all(np.array_equal(keys[arm], mutated_keys[arm]) for arm in ARMS),
        {arm: sha256_array(mutated_keys[arm]) for arm in ARMS},
    )
    generic_scores = scores["generic"]["K1"].copy()
    tau_before = float(
        np.quantile(generic_scores[np.asarray(FIT_INDICES)], 0.99, method="linear")
    )
    generic_scores[np.asarray(EVAL_INDICES)] += 1000.0
    tau_after = float(
        np.quantile(generic_scores[np.asarray(FIT_INDICES)], 0.99, method="linear")
    )
    check(
        "tau_calibration_ignores_generic_eval_windows",
        tau_before == tau_after,
        {"before": tau_before, "after": tau_after},
    )

    buffer = io.BytesIO()
    probe = rng.normal(size=(8, HIDDEN_DIM)).astype(np.float16)
    np.save(buffer, probe, allow_pickle=False)
    buffer.seek(0)
    roundtrip = np.load(buffer, allow_pickle=False)
    check(
        "fp16_npy_roundtrip_is_exact",
        np.array_equal(probe, roundtrip),
        {"sha_before": sha256_array(probe), "sha_after": sha256_array(roundtrip)},
    )
    expected_bytes = N_LAYERS * WINDOWS_PER_CHUNK * WINDOW_TOKENS * HIDDEN_DIM * 2
    check(
        "capture_payload_byte_count",
        expected_capture_payload_nbytes() == expected_bytes,
        expected_capture_payload_nbytes(),
    )

    failed = [item for item in checks if not item["passed"]]
    payload = {
        "schema": "moe_rt2_cpu_validation_v1",
        "created_at": now_iso(),
        "rt2_script": str(SCRIPT_PATH),
        "rt2_script_sha256": sha256_file(SCRIPT_PATH),
        "status": "passed" if not failed else "failed",
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "failed_check_names": [item["name"] for item in failed],
        "checks": checks,
        "synthetic_fixture": {
            "shape_per_corpus": list(shape),
            "key_metadata": key_meta,
            "K1_metrics": metrics,
        },
        "production_capture_geometry": {
            "shape_per_chunk": list(expected_capture_shape()),
            "payload_bytes_per_chunk": expected_capture_payload_nbytes(),
            "chunk_count": len(CORPORA) * (N_WINDOWS // WINDOWS_PER_CHUNK),
            "total_payload_bytes": expected_capture_payload_nbytes()
            * len(CORPORA)
            * (N_WINDOWS // WINDOWS_PER_CHUNK),
        },
    }
    path = out / "cpu_validation.json"
    write_json(path, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "checks": f"{payload['checks_passed']}/{payload['checks_total']}",
                "receipt": str(path),
            }
        ),
        flush=True,
    )
    return 0 if not failed else 2


def validate_capture_set(
    out: Path,
    verification: dict[str, Any],
) -> tuple[dict[str, list[tuple[list[int], np.ndarray]]], dict[str, Any]]:
    manifest = verification["manifest"]
    with np.load(verification["windows_path"], allow_pickle=False) as frozen:
        frozen_windows = {
            corpus: frozen[corpus].astype(np.int64, copy=True) for corpus in CORPORA
        }

    opened: dict[str, list[tuple[list[int], np.ndarray]]] = {
        corpus: [] for corpus in CORPORA
    }
    checks: list[dict[str, Any]] = []
    receipts: dict[str, list[dict[str, Any]]] = {corpus: [] for corpus in CORPORA}
    observed_indices: dict[str, list[int]] = {corpus: [] for corpus in CORPORA}
    for corpus in CORPORA:
        for chunk_index in range(N_WINDOWS // WINDOWS_PER_CHUNK):
            capture_path, _partial, receipt_path = capture_paths(out, corpus, chunk_index)
            if not capture_path.is_file() or not receipt_path.is_file():
                raise FileNotFoundError(
                    f"missing RT2 capture or receipt: {capture_path}, {receipt_path}"
                )
            receipt = json_load(receipt_path)
            receipts[corpus].append(receipt)
            expected_indices = list(
                range(chunk_index * WINDOWS_PER_CHUNK, (chunk_index + 1) * WINDOWS_PER_CHUNK)
            )
            check_record(checks, f"{corpus}.chunk{chunk_index}.status", "complete", receipt.get("status"))
            check_record(
                checks,
                f"{corpus}.chunk{chunk_index}.window_indices",
                expected_indices,
                receipt.get("window_indices"),
            )
            check_record(
                checks,
                f"{corpus}.chunk{chunk_index}.corpus_sha256",
                manifest["corpora"][corpus]["corpus_sha256"],
                receipt.get("corpus_sha256"),
            )
            expected_ids = frozen_windows[corpus][expected_indices]
            check_record(
                checks,
                f"{corpus}.chunk{chunk_index}.input_ids_sha256",
                sha256_array(expected_ids),
                receipt.get("input_ids_sha256"),
            )
            check_record(
                checks,
                f"{corpus}.chunk{chunk_index}.capture_file_sha256",
                receipt.get("capture_file_sha256"),
                sha256_file(capture_path),
            )
            array = np.load(capture_path, mmap_mode="r", allow_pickle=False)
            check_record(
                checks,
                f"{corpus}.chunk{chunk_index}.shape",
                list(expected_capture_shape()),
                list(array.shape),
            )
            check_record(
                checks,
                f"{corpus}.chunk{chunk_index}.dtype",
                "float16",
                str(array.dtype),
            )
            check_record(
                checks,
                f"{corpus}.chunk{chunk_index}.payload_nbytes",
                expected_capture_payload_nbytes(),
                int(array.nbytes),
            )
            observed_indices[corpus].extend(expected_indices)
            opened[corpus].append((expected_indices, array))

    for corpus in CORPORA:
        check_record(
            checks,
            f"{corpus}.captured_window_coverage",
            list(range(N_WINDOWS)),
            observed_indices[corpus],
        )
    failed = [item for item in checks if item["critical"] and not item["passed"]]
    receipt = {
        "schema": "moe_rt2_capture_set_validation_v1",
        "passed": not failed,
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "failed_check_names": [item["name"] for item in failed],
        "checks": checks,
        "window_indices_by_corpus": observed_indices,
        "capture_receipts": receipts,
    }
    if failed:
        raise RuntimeError(
            "capture-set validation failed: "
            + ", ".join(item["name"] for item in failed)
        )
    return opened, receipt


def materialize_layer(
    opened: dict[str, list[tuple[list[int], np.ndarray]]], layer: int
) -> dict[str, np.ndarray]:
    hidden: dict[str, np.ndarray] = {}
    for corpus in CORPORA:
        chunks = []
        indices: list[int] = []
        for chunk_indices, array in opened[corpus]:
            chunks.append(np.asarray(array[layer]))
            indices.extend(chunk_indices)
        combined = np.concatenate(chunks, axis=0)
        order = np.argsort(np.asarray(indices, dtype=np.int64))
        combined = np.ascontiguousarray(combined[order])
        if combined.shape != (N_WINDOWS, WINDOW_TOKENS, HIDDEN_DIM):
            raise RuntimeError(f"{corpus} layer {layer} shape mismatch: {combined.shape}")
        if not np.isfinite(combined).all():
            raise RuntimeError(f"{corpus} layer {layer} contains non-finite values")
        hidden[corpus] = combined
    return hidden


def native_top1_reference(rt1_analysis: dict[str, Any], layer: int) -> dict[str, Any]:
    return {
        corpus: rt1_analysis["per_corpus_layer_metrics"][corpus][layer]["calibration"][
            "top1_logit"
        ]
        for corpus in CORPORA
    }


def save_keys(
    path: Path,
    k1: np.ndarray,
    k2: np.ndarray,
    tau_k1: np.ndarray,
    tau_k2: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez(
            handle,
            layer_indices=np.arange(N_LAYERS, dtype=np.int16),
            K1=np.ascontiguousarray(k1, dtype=np.float32),
            K2=np.ascontiguousarray(k2, dtype=np.float32),
            tau_K1=np.ascontiguousarray(tau_k1, dtype=np.float32),
            tau_K2=np.ascontiguousarray(tau_k2, dtype=np.float32),
            fit_window_indices=np.asarray(FIT_INDICES, dtype=np.int16),
            eval_window_indices=np.asarray(EVAL_INDICES, dtype=np.int16),
        )


def metric_fmt(metric: dict[str, Any], digits: int = 4) -> str:
    return (
        f"{metric['value']:.{digits}f} "
        f"[{metric['ci95_low']:.{digits}f}, {metric['ci95_high']:.{digits}f}]"
    )


def q_fmt(quantiles: dict[str, Any]) -> str:
    return "/".join(f"{float(quantiles[key]):.4f}" for key in SCORE_QUANTILE_KEYS)


def q_metric_fmt(quantiles: dict[str, dict[str, Any]]) -> str:
    return "; ".join(
        f"{key} {metric_fmt(quantiles[key])}" for key in SCORE_QUANTILE_KEYS
    )


def measured_paths(out: Path) -> list[str]:
    paths: list[Path] = [
        SCRIPT_PATH,
        out / "preflight.json",
        out / "cpu_validation.json",
        out / "g1_capture_truth.json",
    ]
    for corpus in CORPORA:
        for chunk_index in range(N_WINDOWS // WINDOWS_PER_CHUNK):
            capture_path, _partial, receipt_path = capture_paths(out, corpus, chunk_index)
            paths.extend((capture_path, receipt_path))
    paths.extend((out / "synthetic_keys.npz", out / "analysis.json", out / "MOE_RT2_REPORT.md"))
    return [str(path.resolve()) for path in paths]


def render_measured_report(analysis: dict[str, Any]) -> str:
    lines: list[str] = [
        "# MOE-RT2 — synthetic-key addressability on frozen GPT-OSS-20B",
        "",
        f"Generated: `{analysis['created_at']}`",
        "",
        "## Scope and registered verdict",
        "",
        (
            "This is inference measurement plus closed-form linear analysis on one "
            "frozen model. It tests only the gate half of a bolt-on expert ABI. It "
            "does not establish expert viability or model quality."
        ),
        "",
        f"**{analysis['registered_verdict']}**",
        "",
        (
            f"Qualifying PRIMARY K1 layers: **{analysis['decision']['qualifying_layer_count']}/24**. "
            f"Best layer selection: {analysis['decision']['best_layer_selection_rule']}."
        ),
        "",
        "## Best-layer row",
        "",
        "| Layer | K1 AUC domain-vs-generic | K1 AUC domain-vs-code | recall@tau | generic FPR | code FPR |",
        "| ---: | --- | --- | --- | --- | --- |",
    ]
    best = analysis["decision"]["best_layer_row"]
    if best is None:
        lines.append("| NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |")
    else:
        lines.append(
            f"| {best['layer']} | {metric_fmt(best['auc_domain_vs_generic'])} | "
            f"{metric_fmt(best['auc_domain_vs_code'])} | "
            f"{metric_fmt(best['domain_recall_at_tau'])} | "
            f"{metric_fmt(best['generic_fpr_at_tau'])} | "
            f"{metric_fmt(best['code_fpr_at_tau'])} |"
        )

    lines.extend(["", "## Gates", ""])
    for gate in ("G0", "G1", "G2", "G3"):
        lines.append(analysis["gates"][gate]["report_line"])
    lines.extend(
        [
            "",
            "## Registered method",
            "",
            "- Frozen windows: exact RT1 `corpus_windows.npz`; 16 × 512 tokens per corpus.",
            "- FIT: even window indices `[0,2,4,6,8,10,12,14]`; EVAL: odd indices `[1,3,5,7,9,11,13,15]`.",
            "- K1: `unit_norm(mean(domain_FIT) - mean(generic_FIT))`; K2: `unit_norm(mean(domain_FIT))`.",
            "- Tau: NumPy linear p99 of GENERIC-FIT scores, independently per layer and arm; firing is `score >= tau`.",
            "- Every AUC, recall/FPR, window-majority, and RT2 score quantile below uses EVAL windows only.",
            (
                f"- 95% intervals use `{analysis['bootstrap_resamples']}` window bootstrap resamples. "
                "RT1 native top-1 values are the frozen all-window RT1 calibration reference, not RT2 outcomes."
            ),
            "- Token AUC is the exact Mann-Whitney statistic with ties worth one half; equal-sized window-pair AUCs make window resampling exact.",
            "",
            "## Leakage receipt",
            "",
            f"- Key FIT indices: `{analysis['split_provenance']['key_construction']}`",
            f"- Tau indices: `{analysis['split_provenance']['tau_calibration']}`",
            f"- EVAL indices: `{analysis['split_provenance']['evaluation']}`",
            f"- FIT/EVAL overlap count: `{analysis['split_provenance']['fit_eval_overlap_count']}`",
            "",
            "## PRIMARY K1 per-layer metrics",
            "",
            "Values are point estimate `[window-bootstrap 95% CI]`.",
            "",
            "| L | AUC domain-generic | AUC domain-code | recall@tau | generic FPR | code FPR | tau | Qualifies |",
            "| ---: | --- | --- | --- | --- | --- | ---: | :---: |",
        ]
    )
    for layer_item in analysis["layers"]:
        arm = layer_item["arms"]["K1"]
        lines.append(
            f"| {layer_item['layer']} | {metric_fmt(arm['auc_domain_vs_generic'])} | "
            f"{metric_fmt(arm['auc_domain_vs_code'])} | "
            f"{metric_fmt(arm['domain_recall_at_tau'])} | "
            f"{metric_fmt(arm['generic_fpr_at_tau'])} | "
            f"{metric_fmt(arm['code_fpr_at_tau'])} | {arm['tau']:.6f} | "
            f"{'yes' if layer_item['K1_qualifies'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Secondary K2 per-layer metrics",
            "",
            "K2 is descriptive and has no registered pass/fail role.",
            "",
            "| L | AUC domain-generic | AUC domain-code | recall@tau | generic FPR | code FPR | tau |",
            "| ---: | --- | --- | --- | --- | --- | ---: |",
        ]
    )
    for layer_item in analysis["layers"]:
        arm = layer_item["arms"]["K2"]
        lines.append(
            f"| {layer_item['layer']} | {metric_fmt(arm['auc_domain_vs_generic'])} | "
            f"{metric_fmt(arm['auc_domain_vs_code'])} | "
            f"{metric_fmt(arm['domain_recall_at_tau'])} | "
            f"{metric_fmt(arm['generic_fpr_at_tau'])} | "
            f"{metric_fmt(arm['code_fpr_at_tau'])} | {arm['tau']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Window-majority firing",
            "",
            "A window fires by majority only when strictly more than 50% of its EVAL tokens have `score >= tau`.",
            "",
            "| Arm | L | DOMAIN | GENERIC | CODE |",
            "| --- | ---: | --- | --- | --- |",
        ]
    )
    for arm_name in ARMS:
        for layer_item in analysis["layers"]:
            majority = layer_item["arms"][arm_name]["window_majority_firing"]
            lines.append(
                f"| {arm_name} | {layer_item['layer']} | {metric_fmt(majority['domain'])} | "
                f"{metric_fmt(majority['generic'])} | {metric_fmt(majority['code'])} |"
            )

    lines.extend(
        [
            "",
            "## Score scale versus RT1 native top-1 logit scale",
            "",
            "RT2 score cells list each p05/p25/p50/p75/p95 point estimate with its window-bootstrap 95% CI. Native top-1 cells list p05/p25/p50/p75/p95 copied from RT1's frozen calibration table and are shown only as a scale reference.",
            "",
            "| Arm | L | Corpus | RT2 score EVAL quantiles | RT1 native top-1 reference quantiles | tau |",
            "| --- | ---: | --- | --- | --- | ---: |",
        ]
    )
    for arm_name in ARMS:
        for layer_item in analysis["layers"]:
            arm = layer_item["arms"][arm_name]
            for corpus in CORPORA:
                lines.append(
                    f"| {arm_name} | {layer_item['layer']} | {corpus.upper()} | "
                    f"{q_metric_fmt(arm['score_quantiles_eval'][corpus])} | "
                    f"{q_fmt(layer_item['native_top1_logit_reference'][corpus])} | "
                    f"{arm['tau']:.6f} |"
                )

    lines.extend(
        [
            "",
            "## Key receipts",
            "",
            "| L | K1 source norm | K1 fp32 norm | K1 SHA-256 | K2 source norm | K2 fp32 norm | K2 SHA-256 |",
            "| ---: | ---: | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for layer_item in analysis["layers"]:
        meta = layer_item["key_metadata"]
        lines.append(
            f"| {layer_item['layer']} | {meta['K1']['pre_normalization_norm']:.8f} | "
            f"{meta['K1']['stored_fp32_norm']:.8f} | `{meta['K1']['stored_fp32_sha256']}` | "
            f"{meta['K2']['pre_normalization_norm']:.8f} | "
            f"{meta['K2']['stored_fp32_norm']:.8f} | `{meta['K2']['stored_fp32_sha256']}` |"
        )

    lines.extend(["", "## Frozen identity SHA comparison lines", ""])
    lines.extend(
        f"- `{line}`" for line in analysis["g0_verification"]["sha_comparison_lines"]
    )
    lines.extend(["", "## Files created or modified", ""])
    lines.extend(f"- `{path}`" for path in analysis["created_or_modified_paths"])
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Evidence class: inference measurement plus closed-form linear analysis on one frozen model.",
            "- The gate keys use eight FIT windows per corpus and no training; the result does not establish a useful external expert.",
            "- CODE and K2 are descriptive and cannot make the registered decision pass or fail.",
            "- RT1 native top-1 scale values use RT1's registered full 16-window calibration table and are context only.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> int:
    if args.bootstrap_resamples < 1:
        raise ValueError("bootstrap-resamples must be positive")
    out = ensure_output_scope(args.output_dir)
    verification, splits = require_frozen_inputs(args)
    opened, capture_validation = validate_capture_set(out, verification)
    g1_path = out / "g1_capture_truth.json"
    if not g1_path.is_file():
        raise FileNotFoundError(f"missing G1 receipt: {g1_path}")
    g1 = json_load(g1_path)

    rt1_analysis = json_load(Path(verification["rt1_analysis_path"]))
    layer_results: list[dict[str, Any]] = []
    k1_values = np.empty((N_LAYERS, HIDDEN_DIM), dtype=np.float32)
    k2_values = np.empty((N_LAYERS, HIDDEN_DIM), dtype=np.float32)
    tau_k1 = np.empty(N_LAYERS, dtype=np.float32)
    tau_k2 = np.empty(N_LAYERS, dtype=np.float32)
    qualifying_layers: list[int] = []

    started = time.perf_counter()
    for layer in range(N_LAYERS):
        hidden = materialize_layer(opened, layer)
        keys, key_metadata = construct_keys(hidden)
        scores = score_hidden(hidden, keys)
        arms: dict[str, Any] = {}
        for arm_offset, arm in enumerate(ARMS):
            arm_scores = {corpus: scores[corpus][arm] for corpus in CORPORA}
            arms[arm] = analyze_arm_scores(
                arm_scores,
                resamples=args.bootstrap_resamples,
                seed=args.seed + layer * 1000 + arm_offset * 100,
            )
        k1_values[layer] = keys["K1"]
        k2_values[layer] = keys["K2"]
        tau_k1[layer] = arms["K1"]["tau"]
        tau_k2[layer] = arms["K2"]["tau"]
        primary = arms["K1"]
        criteria = {
            "auc_domain_vs_generic_at_least_0_90": bool(
                primary["auc_domain_vs_generic"]["value"] >= 0.90
            ),
            "auc_ci95_lower_at_least_0_85": bool(
                primary["auc_domain_vs_generic"]["ci95_low"] >= 0.85
            ),
            "domain_recall_at_least_0_30": bool(
                primary["domain_recall_at_tau"]["value"] >= 0.30
            ),
            "generic_fpr_at_most_0_02": bool(
                primary["generic_fpr_at_tau"]["value"] <= 0.02
            ),
        }
        qualifies = bool(all(criteria.values()))
        if qualifies:
            qualifying_layers.append(layer)
        layer_results.append(
            {
                "layer": layer,
                "key_metadata": key_metadata,
                "arms": arms,
                "K1_registered_criteria": criteria,
                "K1_qualifies": qualifies,
                "native_top1_logit_reference": native_top1_reference(
                    rt1_analysis, layer
                ),
            }
        )
        print(
            json.dumps(
                {
                    "layer": layer,
                    "K1_auc_domain_generic": primary["auc_domain_vs_generic"]["value"],
                    "K1_auc_ci95_low": primary["auc_domain_vs_generic"]["ci95_low"],
                    "K1_recall": primary["domain_recall_at_tau"]["value"],
                    "K1_generic_fpr": primary["generic_fpr_at_tau"]["value"],
                    "qualifies": qualifies,
                }
            ),
            flush=True,
        )
        del hidden, keys, scores, arms

    keys_path = out / "synthetic_keys.npz"
    save_keys(keys_path, k1_values, k2_values, tau_k1, tau_k2)

    g0_green = bool(verification["passed"])
    g1_green = bool(g1.get("status") == "complete" and g1.get("verdict") == "GREEN")
    g2_green = bool(
        capture_validation["passed"]
        and splits["planned_split_integrity"]
        and splits["fit_eval_overlap_count"] == 0
    )
    integrity_green = bool(g0_green and g1_green and g2_green)

    best_pool = qualifying_layers if qualifying_layers else list(range(N_LAYERS))
    best_layer = max(
        best_pool,
        key=lambda layer: (
            layer_results[layer]["arms"]["K1"]["auc_domain_vs_generic"]["value"],
            layer_results[layer]["arms"]["K1"]["auc_domain_vs_generic"]["ci95_low"],
            layer_results[layer]["arms"]["K1"]["domain_recall_at_tau"]["value"],
            -layer_results[layer]["arms"]["K1"]["generic_fpr_at_tau"]["value"],
            -layer,
        ),
    )
    best_metrics = layer_results[best_layer]["arms"]["K1"]
    best_row = {
        "layer": best_layer,
        "auc_domain_vs_generic": best_metrics["auc_domain_vs_generic"],
        "auc_domain_vs_code": best_metrics["auc_domain_vs_code"],
        "domain_recall_at_tau": best_metrics["domain_recall_at_tau"],
        "generic_fpr_at_tau": best_metrics["generic_fpr_at_tau"],
        "code_fpr_at_tau": best_metrics["code_fpr_at_tau"],
    }

    if integrity_green and qualifying_layers:
        registered_verdict = "H-RT2 SUPPORTED."
    elif integrity_green:
        registered_verdict = (
            "synthetic-key addressability NOT DETECTED under these limits "
            "(closed-form keys, 8-window fits, token-level gating)"
        )
    else:
        registered_verdict = (
            "H-RT2 registered verdict: NOT_MEASURED — one or more evidence-integrity "
            "gates failed; descriptive rows are not an adjudication."
        )

    gates = {
        "G0": {
            "verdict": "GREEN" if g0_green else "RED",
            "identity_checks_passed": verification["checks_passed"],
            "identity_checks_total": verification["checks_total"],
            "report_line": (
                f"G0 {'GREEN' if g0_green else 'RED'} — frozen identity checks="
                f"{verification['checks_passed']}/{verification['checks_total']}"
            ),
        },
        "G1": {
            "verdict": "GREEN" if g1_green else "RED",
            "receipt": g1,
            "report_line": (
                f"G1 {'GREEN' if g1_green else 'RED'} — fp16 top4 exact="
                f"{g1.get('fp16_top4_exact_tokens', 0)}/"
                f"{g1.get('top4_total_layer_tokens', 0)}; max_abs_diff="
                f"{g1.get('fp16_max_abs_diff', 'NOT_MEASURED')}"
            ),
        },
        "G2": {
            "verdict": "GREEN" if g2_green else "RED",
            "capture_checks_passed": capture_validation["checks_passed"],
            "capture_checks_total": capture_validation["checks_total"],
            "fit_eval_overlap_count": splits["fit_eval_overlap_count"],
            "report_line": (
                f"G2 {'GREEN' if g2_green else 'RED'} — capture checks="
                f"{capture_validation['checks_passed']}/{capture_validation['checks_total']}; "
                f"FIT/EVAL overlap={splits['fit_eval_overlap_count']}"
            ),
        },
        "G3": {
            "verdict": "GREEN" if integrity_green else "RED",
            "measured_layer_arm_rows": N_LAYERS * len(ARMS),
            "scale_rows": N_LAYERS * len(ARMS) * len(CORPORA),
            "report_line": (
                f"G3 {'GREEN' if integrity_green else 'RED'} — per-layer arm rows="
                f"{N_LAYERS * len(ARMS)}/48; scale rows="
                f"{N_LAYERS * len(ARMS) * len(CORPORA)}/144"
            ),
        },
    }

    analysis: dict[str, Any] = {
        "schema": "moe_rt2_analysis_v1",
        "status": "complete" if integrity_green else "complete_with_failed_gate",
        "created_at": now_iso(),
        "order": str(ORDER_PATH.resolve()),
        "rt2_script": str(SCRIPT_PATH),
        "rt2_script_sha256": sha256_file(SCRIPT_PATH),
        "seed": int(args.seed),
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "registered_verdict": registered_verdict,
        "registered_decision_rule": {
            "arm": "K1",
            "criterion": (
                "at least one layer has AUC(domain,generic)>=0.90, AUC window-bootstrap "
                "CI95 lower>=0.85, domain recall@tau>=0.30, and generic EVAL FPR<=0.02"
            ),
            "code_fpr_role": "descriptive",
            "K2_role": "descriptive",
        },
        "decision": {
            "qualifying_layer_count": len(qualifying_layers),
            "qualifying_layers": qualifying_layers,
            "best_layer": best_layer,
            "best_layer_selection_rule": (
                "highest K1 domain-vs-generic AUC among qualifying layers when any "
                "qualify, otherwise among all layers; tie-break CI lower, recall, "
                "lower generic FPR, then lower layer index"
            ),
            "best_layer_row": best_row,
        },
        "split_provenance": splits,
        "g0_verification": verification,
        "g1_capture_truth": g1,
        "capture_set_validation": capture_validation,
        "gates": gates,
        "layers": layer_results,
        "keys_artifact": {
            "path": str(keys_path.resolve()),
            "sha256": sha256_file(keys_path),
            "K1_shape": list(k1_values.shape),
            "K2_shape": list(k2_values.shape),
            "tau_shape": list(tau_k1.shape),
        },
        "native_scale_reference": {
            "source": verification["rt1_analysis_path"],
            "source_sha256": verification["rt1_analysis_sha256"],
            "scope": "RT1 registered all-16-window calibration; context only",
        },
        "runtime": {
            "analysis_wall_seconds": float(time.perf_counter() - started),
            "numpy_version": np.__version__,
        },
        "created_or_modified_paths": measured_paths(out),
        "limitations": [
            "one frozen GPT-OSS-20B snapshot",
            "closed-form keys from eight FIT windows per corpus",
            "token-level gate only; external expert is stubbed out",
            "no model-quality or expert-viability claim",
        ],
    }
    report = render_measured_report(analysis)
    if registered_verdict not in report:
        raise RuntimeError("registered verdict missing from report")
    if report.count("| K1 |") < N_LAYERS * len(CORPORA):
        raise RuntimeError("K1 score-scale rows missing from report")
    if report.count("| K2 |") < N_LAYERS * len(CORPORA):
        raise RuntimeError("K2 score-scale rows missing from report")

    analysis_path = out / "analysis.json"
    report_path = out / "MOE_RT2_REPORT.md"
    write_json(analysis_path, analysis)
    write_text(report_path, report)
    analysis["gates"]["G3"]["report_path"] = str(report_path.resolve())
    analysis["gates"]["G3"]["report_sha256"] = sha256_file(report_path)
    write_json(analysis_path, analysis)
    print(
        json.dumps(
            {
                "status": analysis["status"],
                "registered_verdict": registered_verdict,
                "qualifying_layers": len(qualifying_layers),
                "best_layer": best_layer,
                "gates": {name: gate["verdict"] for name, gate in gates.items()},
                "analysis": str(analysis_path),
                "report": str(report_path),
            }
        ),
        flush=True,
    )
    return 0 if integrity_green else 4


def resume_commands() -> list[str]:
    script = "scripts/gpt_oss20b_synthetic_key.py"
    commands = [
        f"python3 {script} preflight",
        f"python3 {script} self-test",
        f"{GPU_WRAPPER} python3 {script} g1",
        "sleep 20",
    ]
    capture_specs = [
        (corpus, chunk_index)
        for corpus in CORPORA
        for chunk_index in range(N_WINDOWS // WINDOWS_PER_CHUNK)
    ]
    for offset, (corpus, chunk_index) in enumerate(capture_specs):
        commands.append(
            f"{GPU_WRAPPER} python3 {script} capture --corpus {corpus} "
            f"--chunk-index {chunk_index}"
        )
        if offset != len(capture_specs) - 1:
            commands.append("sleep 20")
    commands.extend(["sleep 20", f"python3 {script} analyze"])
    return commands


def blocked_existing_paths(out: Path) -> list[str]:
    candidates = [
        SCRIPT_PATH,
        out / "preflight.json",
        out / "cpu_validation.json",
        out / "g1_capture_truth.json",
        out / "analysis.json",
        out / "MOE_RT2_REPORT.md",
    ]
    # analysis/report are about to be written, so include them; include other
    # receipts only when they actually exist.
    result = [SCRIPT_PATH]
    result.extend(path for path in candidates[1:4] if path.exists())
    result.extend((out / "analysis.json", out / "MOE_RT2_REPORT.md"))
    return [str(path.resolve()) for path in result]


def not_measured_per_layer_tables() -> list[str]:
    lines = [
        "## Per-layer arm metrics",
        "",
        "| Arm | L | AUC domain-generic | AUC domain-code | recall@tau | generic FPR | code FPR | tau |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in ARMS:
        for layer in range(N_LAYERS):
            lines.append(
                f"| {arm} | {layer} | NOT_MEASURED | NOT_MEASURED | "
                "NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |"
            )
    lines.extend(
        [
            "",
            "## Window-majority firing",
            "",
            "| Arm | L | DOMAIN | GENERIC | CODE |",
            "| --- | ---: | --- | --- | --- |",
        ]
    )
    for arm in ARMS:
        for layer in range(N_LAYERS):
            lines.append(
                f"| {arm} | {layer} | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |"
            )
    lines.extend(
        [
            "",
            "## Score scale versus RT1 native top-1 logit scale",
            "",
            "No RT2 score exists. The native reference is intentionally not substituted for the missing RT2 measurement.",
            "",
            "| Arm | L | Corpus | RT2 score EVAL quantiles | RT1 native top-1 reference | tau |",
            "| --- | ---: | --- | --- | --- | --- |",
        ]
    )
    for arm in ARMS:
        for layer in range(N_LAYERS):
            for corpus in CORPORA:
                lines.append(
                    f"| {arm} | {layer} | {corpus.upper()} | NOT_MEASURED | "
                    "NOT_REPORTED_WITHOUT_RT2_MEASUREMENT | NOT_MEASURED |"
                )
    return lines


def render_blocked_report(payload: dict[str, Any]) -> str:
    verification = payload["cpu_preflight"]["g0_cpu_identity_verification"]
    splits = payload["split_provenance"]
    probe = payload["cuda_environment"]
    g1_attempt = payload["g1_attempt_receipt"]
    lines: list[str] = [
        "# MOE-RT2 — GPU-blocked execution receipt",
        "",
        f"Generated: `{payload['created_at']}`",
        "",
        "## Status",
        "",
        "No GPT-OSS router-input capture completed in this worker. Missing measurements are not replaced by zeros, RT1 logits, or synthetic-fixture values.",
        "",
        "**H-RT2 registered verdict: NOT_MEASURED — CUDA unavailable; no RT2 router-input capture or EVAL metric was produced.**",
        "",
        f"Device nodes exposed: `{probe['device_nodes']}`.",
        f"`nvidia-smi -L` return code: `{probe['nvidia_smi']['returncode']}`; output: `{probe['nvidia_smi']['stdout'] or probe['nvidia_smi']['stderr']}`.",
        f"PyTorch CUDA probe: `{probe['torch']}`.",
        f"Locked G1 attempt status: `{g1_attempt.get('status', 'missing')}`; error: `{g1_attempt.get('error', 'NOT_MEASURED')}`.",
        "",
        "## Gates",
        "",
    ]
    for gate in ("G0", "G1", "G2", "G3"):
        lines.append(payload["gates"][gate]["report_line"])
    lines.extend(
        [
            "",
            "The all-RED adjudication is the order's registered CUDA-blocked protocol. CPU checks below passed or failed only as preflight evidence; they are not silently promoted to official GREEN gates.",
            "",
            "## Best-layer row",
            "",
            "| Layer | K1 AUC domain-vs-generic | K1 AUC domain-vs-code | recall@tau | generic FPR | code FPR |",
            "| ---: | --- | --- | --- | --- | --- |",
            "| NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |",
            "",
            "## CPU preflight receipts",
            "",
            f"Frozen identity checks: `{verification.get('checks_passed', 0)}/{verification.get('checks_total', 0)}` passed.",
            f"Planned FIT/EVAL overlap count: `{splits['fit_eval_overlap_count']}`.",
            f"CPU self-test checks: `{payload['cpu_validation_summary']['checks_passed']}/{payload['cpu_validation_summary']['checks_total']}` passed.",
            "",
            "### Frozen SHA comparison lines",
            "",
        ]
    )
    lines.extend(f"- `{line}`" for line in verification.get("sha_comparison_lines", []))
    lines.extend(
        [
            "",
            "### Split index lists",
            "",
            f"- Key construction: `{splits['key_construction']}`",
            f"- Tau calibration: `{splits['tau_calibration']}`",
            f"- Evaluation: `{splits['evaluation']}`",
            f"- FIT/EVAL overlap checks: `{splits['fit_eval_overlap_checks']}`",
            "",
        ]
    )
    lines.extend(not_measured_per_layer_tables())
    lines.extend(
        [
            "",
            "## Exact GPU resume commands",
            "",
            "Run from `/mnt/ForgeRealm/GraftRepository` in a GPU-enabled worker. Every GPU process has its own lock and 590-second timeout; the 20-second lines are mandatory idle gaps.",
            "",
            "```bash",
            "set -euo pipefail",
            *payload["resume_commands"],
            "```",
            "",
            "## Files created or modified",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in payload["created_or_modified_paths"])
    lines.extend(
        [
            "",
            "## Could not do",
            "",
            "- G1 could not compare any captured router-input token because TensorCUDA could not start a model forward without a CUDA device.",
            "- No fp16 router-input capture exists for any of 24 layers or three corpora.",
            "- No K1/K2 key, tau, EVAL AUC, bootstrap interval, recall/FPR, window-majority rate, score quantile, qualifying-layer count, or best layer was measured.",
            "- The registered SUPPORTED/NOT DETECTED decision cannot be applied to missing data.",
            "- No expert-viability or model-quality claim is made.",
            "",
        ]
    )
    return "\n".join(lines)


def blocked_report(args: argparse.Namespace) -> int:
    out = ensure_output_scope(args.output_dir)
    verification = verify_frozen_inputs(args.rt1_dir, args.model_dir)
    splits = split_provenance()
    preflight_path = out / "preflight.json"
    if preflight_path.is_file():
        cpu_preflight = json_load(preflight_path)
    else:
        cpu_preflight = {
            "schema": "moe_rt2_preflight_v1",
            "status": "not_run",
            "g0_cpu_identity_verification": verification,
            "g2_registered_split_plan": splits,
        }
    validation_path = out / "cpu_validation.json"
    if validation_path.is_file():
        cpu_validation = json_load(validation_path)
        validation_summary = {
            "status": cpu_validation.get("status"),
            "checks_passed": int(cpu_validation.get("checks_passed", 0)),
            "checks_total": int(cpu_validation.get("checks_total", 0)),
            "receipt": str(validation_path.resolve()),
        }
    else:
        validation_summary = {
            "status": "not_run",
            "checks_passed": 0,
            "checks_total": 0,
            "receipt": str(validation_path.resolve()),
        }
    g1_path = out / "g1_capture_truth.json"
    g1_attempt = (
        json_load(g1_path)
        if g1_path.is_file()
        else {"status": "missing", "error": "locked G1 attempt receipt missing"}
    )
    probe = cuda_environment_probe()
    identity_passed = int(verification.get("checks_passed", 0))
    identity_total = int(verification.get("checks_total", 0))
    payload: dict[str, Any] = {
        "schema": "moe_rt2_analysis_blocked_v1",
        "status": "blocked_before_router_input_capture",
        "created_at": now_iso(),
        "order": str(ORDER_PATH.resolve()),
        "rt2_script": str(SCRIPT_PATH),
        "rt2_script_sha256": sha256_file(SCRIPT_PATH),
        "blocker": (
            "worker exposes no usable CUDA device; the locked G1 attempt produced "
            "zero captured router-input tokens"
        ),
        "cuda_environment": probe,
        "g1_attempt_receipt": g1_attempt,
        "cpu_preflight": cpu_preflight,
        "cpu_validation_summary": validation_summary,
        "split_provenance": splits,
        "gates": {
            "G0": {
                "verdict": "RED",
                "official_status": "NOT_MEASURED",
                "cpu_identity_checks_passed": identity_passed,
                "cpu_identity_checks_total": identity_total,
                "report_line": (
                    f"G0 RED — NOT_MEASURED; CPU identity preflight="
                    f"{identity_passed}/{identity_total}, but the registered blocked "
                    "protocol keeps all official gates RED"
                ),
            },
            "G1": {
                "verdict": "RED",
                "official_status": "NOT_MEASURED",
                "compared_router_input_tokens": 0,
                "report_line": (
                    "G1 RED — NOT_MEASURED; fp16 top4 exact=0/0; "
                    "compared router-input tokens=0"
                ),
            },
            "G2": {
                "verdict": "RED",
                "official_status": "NOT_MEASURED",
                "captured_layers": 0,
                "required_layers": N_LAYERS,
                "planned_fit_eval_overlap_count": splits["fit_eval_overlap_count"],
                "report_line": (
                    f"G2 RED — NOT_MEASURED; captured layers=0/{N_LAYERS}; "
                    f"planned FIT/EVAL overlap={splits['fit_eval_overlap_count']}"
                ),
            },
            "G3": {
                "verdict": "RED",
                "official_status": "NOT_MEASURED",
                "measured_layer_arm_rows": 0,
                "required_layer_arm_rows": N_LAYERS * len(ARMS),
                "report_line": (
                    "G3 RED — NOT_MEASURED; measured per-layer arm rows=0/48; "
                    "blocked report delivered without a registered measured verdict"
                ),
            },
        },
        "registered_verdict": None,
        "registered_verdict_note": (
            "H-RT2 registered verdict: NOT_MEASURED — CUDA unavailable; no RT2 "
            "router-input capture or EVAL metric was produced."
        ),
        "decision": {
            "qualifying_layer_count": None,
            "best_layer": None,
            "best_layer_row": None,
        },
        "resume_commands": resume_commands(),
        "created_or_modified_paths": blocked_existing_paths(out),
        "could_not_measure": [
            "G1 replay and top-4 comparison",
            "all 24-layer fp16 router-input captures",
            "K1 and K2 keys and thresholds",
            "all EVAL metrics and bootstrap intervals",
            "registered qualifying-layer count and best layer",
        ],
    }
    report = render_blocked_report(payload)
    analysis_path = out / "analysis.json"
    report_path = out / "MOE_RT2_REPORT.md"
    write_json(analysis_path, payload)
    write_text(report_path, report)
    payload["gates"]["G3"]["report_path"] = str(report_path.resolve())
    payload["gates"]["G3"]["report_sha256"] = sha256_file(report_path)
    write_json(analysis_path, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "verdict": payload["registered_verdict_note"],
                "gates": {name: gate["verdict"] for name, gate in payload["gates"].items()},
                "analysis": str(analysis_path),
                "report": str(report_path),
            }
        ),
        flush=True,
    )
    return 0


def main() -> int:
    args = parse_args()
    args.rt1_dir = args.rt1_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.model_dir = args.model_dir.expanduser().resolve()
    if args.mode == "preflight":
        return preflight(args)
    if args.mode == "self-test":
        return run_self_test(args)
    if args.mode == "g1":
        return g1_capture_truth(args)
    if args.mode == "capture":
        return capture(args)
    if args.mode == "analyze":
        return analyze(args)
    if args.mode == "blocked-report":
        return blocked_report(args)
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
