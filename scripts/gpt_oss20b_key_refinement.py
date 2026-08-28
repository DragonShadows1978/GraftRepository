#!/usr/bin/env python3
"""MOE-RT2.1 CPU-only key refinement over frozen RT2 captures.

This analyzer never imports the model runtime, torch, or tensor_cuda.  It opens
the six RT2 router-input arrays read-only with NumPy mmap, processes one layer
at a time, and writes only beneath artifacts/moe_rt2_1/.

Registered outer split:
  FIT  = even windows [0, 2, ..., 14]
  EVAL = odd windows  [1, 3, ..., 15]

Inner FIT-only hyperparameter split (declared before measurement):
  train      = [0, 4, 8, 12]
  validation = [2, 6, 10, 14]

K4 shrinkage and K5 C are selected by the largest minimum of validation
AUC(domain,generic) and AUC(domain,code), then their mean; exact ties prefer
stronger regularization.  The selected model is refit on all FIT windows.
Thresholds are then chosen from the registered pooled-negative FIT quantile
grid and frozen before any EVAL metric is computed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
import platform
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable

# A hard CPU-only rail.  Set this before importing numerical libraries.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy.sparse.linalg import LinearOperator, cg  # noqa: E402
import sklearn  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from threadpoolctl import threadpool_info, threadpool_limits  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
ORDER_PATH = REPO_ROOT / "orders" / "MOE_RT2_1_KEY_REFINEMENT.md"
RT2_DIR = REPO_ROOT / "artifacts" / "moe_rt2"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "moe_rt2_1"
RT2_ANALYSIS_PATH = RT2_DIR / "analysis.json"

CORPORA = ("generic", "code", "domain")
ARMS = ("K3", "K4", "K5")
N_LAYERS = 24
N_WINDOWS = 16
WINDOWS_PER_CHUNK = 8
TOKENS_PER_WINDOW = 512
HIDDEN_DIM = 2880
CAPTURE_SHAPE = (N_LAYERS, WINDOWS_PER_CHUNK, TOKENS_PER_WINDOW, HIDDEN_DIM)
CAPTURE_DTYPE = np.dtype("float16")
FIT_INDICES = tuple(range(0, N_WINDOWS, 2))
EVAL_INDICES = tuple(range(1, N_WINDOWS, 2))
INNER_TRAIN_INDICES = (0, 4, 8, 12)
INNER_VALIDATION_INDICES = (2, 6, 10, 14)

TAU_GRID = (
    ("p90", 0.90),
    ("p95", 0.95),
    ("p99", 0.99),
    ("p99.5", 0.995),
)
K4_ALPHA_GRID = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0)
K5_C_GRID = (1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0)

GENERIC_FPR_CAP = 0.02
CODE_FPR_CAP = 0.05
DOMAIN_RECALL_FLOOR = 0.50
G1_TOLERANCE = 1.0e-3
G1_REFERENCE = {
    "auc_domain_vs_generic": 0.9996,
    "domain_recall_at_tau": 0.995,
    "generic_fpr_at_tau": 0.011,
    "code_fpr_at_tau": 0.998,
}
DEFAULT_SEED = 20260827
DEFAULT_BOOTSTRAP_RESAMPLES = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the sealed MOE-RT2.1 CPU-only key-refinement analysis."
    )
    parser.add_argument("mode", choices=("preflight", "self-test", "analyze", "verify"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--cg-rtol", type=float, default=1.0e-5)
    parser.add_argument("--cg-maxiter", type=int, default=500)
    parser.add_argument("--logistic-maxiter", type=int, default=500)
    return parser.parse_args()


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    work = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(work).cast("B")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def ensure_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != DEFAULT_OUTPUT_DIR.resolve():
        raise ValueError(
            f"ORDER MOE-RT2.1 permits artifacts only at {DEFAULT_OUTPUT_DIR.resolve()}"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def capture_paths(corpus: str, chunk: int) -> tuple[Path, Path]:
    stem = f"{corpus}_chunk{chunk:02d}"
    return (
        RT2_DIR / f"{stem}_router_inputs_fp16.npy",
        RT2_DIR / f"{stem}_capture_receipt.json",
    )


def add_check(
    checks: list[dict[str, Any]], name: str, expected: Any, observed: Any
) -> bool:
    passed = expected == observed
    checks.append(
        {
            "name": name,
            "expected": expected,
            "observed": observed,
            "passed": bool(passed),
        }
    )
    return bool(passed)


def verify_inputs(*, hash_payloads: bool = True) -> tuple[
    dict[str, list[np.ndarray]], dict[str, Any]
]:
    """Open RT2 captures read-only and validate each against its receipt."""

    if not RT2_ANALYSIS_PATH.is_file():
        raise FileNotFoundError(RT2_ANALYSIS_PATH)
    rt2_analysis_sha = sha256_file(RT2_ANALYSIS_PATH)
    checks: list[dict[str, Any]] = []
    arrays: dict[str, list[np.ndarray]] = {corpus: [] for corpus in CORPORA}
    receipt_summaries: dict[str, list[dict[str, Any]]] = {
        corpus: [] for corpus in CORPORA
    }

    for corpus in CORPORA:
        observed_indices: list[int] = []
        for chunk in range(2):
            capture_path, receipt_path = capture_paths(corpus, chunk)
            if not capture_path.is_file() or not receipt_path.is_file():
                raise FileNotFoundError(
                    f"missing RT2 input: {capture_path} or {receipt_path}"
                )
            receipt = read_json(receipt_path)
            array = np.load(capture_path, mmap_mode="r", allow_pickle=False)
            expected_indices = list(
                range(chunk * WINDOWS_PER_CHUNK, (chunk + 1) * WINDOWS_PER_CHUNK)
            )
            add_check(checks, f"{corpus}.chunk{chunk}.receipt_status", "complete", receipt.get("status"))
            add_check(checks, f"{corpus}.chunk{chunk}.receipt_corpus", corpus, receipt.get("corpus"))
            add_check(checks, f"{corpus}.chunk{chunk}.receipt_chunk", chunk, receipt.get("chunk_index"))
            add_check(checks, f"{corpus}.chunk{chunk}.window_indices", expected_indices, receipt.get("window_indices"))
            add_check(checks, f"{corpus}.chunk{chunk}.receipt_shape", list(CAPTURE_SHAPE), receipt.get("capture_shape"))
            add_check(checks, f"{corpus}.chunk{chunk}.receipt_expected_shape", list(CAPTURE_SHAPE), receipt.get("capture_shape_expected"))
            add_check(checks, f"{corpus}.chunk{chunk}.array_shape", list(CAPTURE_SHAPE), list(array.shape))
            add_check(checks, f"{corpus}.chunk{chunk}.receipt_dtype", str(CAPTURE_DTYPE), receipt.get("capture_storage_dtype"))
            add_check(checks, f"{corpus}.chunk{chunk}.array_dtype", str(CAPTURE_DTYPE), str(array.dtype))
            add_check(checks, f"{corpus}.chunk{chunk}.payload_nbytes", int(array.nbytes), int(receipt.get("capture_payload_nbytes", -1)))
            add_check(checks, f"{corpus}.chunk{chunk}.file_bytes", int(capture_path.stat().st_size), int(receipt.get("capture_file_bytes", -1)))
            observed_sha = sha256_file(capture_path) if hash_payloads else None
            if hash_payloads:
                add_check(checks, f"{corpus}.chunk{chunk}.file_sha256", receipt.get("capture_file_sha256"), observed_sha)
            observed_indices.extend(expected_indices)
            arrays[corpus].append(array)
            receipt_summaries[corpus].append(
                {
                    "capture_path": str(capture_path),
                    "receipt_path": str(receipt_path),
                    "window_indices": expected_indices,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "payload_nbytes": int(array.nbytes),
                    "file_bytes": int(capture_path.stat().st_size),
                    "receipt_sha256": sha256_file(receipt_path),
                    "capture_sha256": observed_sha,
                    "mmap_mode": "r",
                    "writeable": bool(array.flags.writeable),
                }
            )
            add_check(checks, f"{corpus}.chunk{chunk}.mmap_not_writeable", False, bool(array.flags.writeable))
        add_check(checks, f"{corpus}.window_coverage", list(range(N_WINDOWS)), observed_indices)

    failed = [item for item in checks if not item["passed"]]
    receipt = {
        "schema": "moe_rt2_1_input_integrity_v1",
        "created_at": now_iso(),
        "passed": not failed,
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "failed_check_names": [item["name"] for item in failed],
        "checks": checks,
        "rt2_analysis_path": str(RT2_ANALYSIS_PATH),
        "rt2_analysis_sha256_at_open": rt2_analysis_sha,
        "capture_payload_hashes_checked": bool(hash_payloads),
        "captures": receipt_summaries,
    }
    if failed:
        raise RuntimeError(
            "G0 input verification failed: "
            + ", ".join(item["name"] for item in failed)
        )
    return arrays, receipt


def materialize_layer(
    arrays: dict[str, list[np.ndarray]], layer: int
) -> dict[str, np.ndarray]:
    """Copy exactly one layer from read-only mmaps into fp16 host arrays."""

    hidden: dict[str, np.ndarray] = {}
    for corpus in CORPORA:
        combined = np.concatenate(
            [np.asarray(chunk[layer]) for chunk in arrays[corpus]], axis=0
        )
        if combined.shape != (N_WINDOWS, TOKENS_PER_WINDOW, HIDDEN_DIM):
            raise RuntimeError(
                f"layer {layer} {corpus} materialized shape {combined.shape}"
            )
        if combined.dtype != CAPTURE_DTYPE:
            raise RuntimeError(
                f"layer {layer} {corpus} materialized dtype {combined.dtype}"
            )
        if not np.isfinite(combined).all():
            raise RuntimeError(f"layer {layer} {corpus} contains non-finite values")
        hidden[corpus] = np.ascontiguousarray(combined)
    return hidden


def selected_tokens(array: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    return np.ascontiguousarray(
        array[np.asarray(indices, dtype=np.int64)].reshape(-1, HIDDEN_DIM),
        dtype=np.float32,
    )


def unit_vector(vector: np.ndarray, *, name: str) -> tuple[np.ndarray, float]:
    work = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(work))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise RuntimeError(f"{name} has degenerate norm {norm}")
    return np.ascontiguousarray((work / norm).astype(np.float32)), norm


def exact_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Mann-Whitney token AUC with ties worth one half."""

    positive = np.asarray(positive, dtype=np.float64).reshape(-1)
    negative = np.asarray(negative, dtype=np.float64).reshape(-1)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("AUC requires non-empty classes")
    if not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError("AUC inputs must be finite")
    ordered = np.sort(negative)
    left = np.searchsorted(ordered, positive, side="left")
    right = np.searchsorted(ordered, positive, side="right")
    wins = left.astype(np.float64) + 0.5 * (right - left)
    return float(wins.sum() / (positive.size * negative.size))


def score_key(hidden: dict[str, np.ndarray], key: np.ndarray) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    for corpus in CORPORA:
        flat = hidden[corpus].astype(np.float32).reshape(-1, HIDDEN_DIM)
        scores[corpus] = np.ascontiguousarray(
            (flat @ key).reshape(N_WINDOWS, TOKENS_PER_WINDOW), dtype=np.float32
        )
    return scores


def score_keys(
    hidden: dict[str, np.ndarray], keys: dict[str, np.ndarray]
) -> dict[str, dict[str, np.ndarray]]:
    matrix = np.stack([keys[arm] for arm in ARMS], axis=1).astype(np.float32)
    result: dict[str, dict[str, np.ndarray]] = {}
    for corpus in CORPORA:
        flat = hidden[corpus].astype(np.float32).reshape(-1, HIDDEN_DIM)
        block = (flat @ matrix).reshape(N_WINDOWS, TOKENS_PER_WINDOW, len(ARMS))
        result[corpus] = {
            arm: np.ascontiguousarray(block[:, :, index])
            for index, arm in enumerate(ARMS)
        }
    return result


def k1_direction(hidden: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    fit = np.asarray(FIT_INDICES, dtype=np.int64)
    domain_mean = hidden["domain"][fit].mean(axis=(0, 1), dtype=np.float64)
    generic_mean = hidden["generic"][fit].mean(axis=(0, 1), dtype=np.float64)
    key, source_norm = unit_vector(
        domain_mean - generic_mean, name="K1 domain-minus-generic"
    )
    return key, {
        "construction": "unit_norm(mean(domain_FIT) - mean(generic_FIT))",
        "source_norm": source_norm,
        "key_sha256": sha256_array(key),
    }


def k3_direction(hidden: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    fit = np.asarray(FIT_INDICES, dtype=np.int64)
    means = {
        corpus: hidden[corpus][fit].mean(axis=(0, 1), dtype=np.float64)
        for corpus in CORPORA
    }
    negative_mean = 0.5 * (means["generic"] + means["code"])
    key, source_norm = unit_vector(
        means["domain"] - negative_mean,
        name="K3 domain-minus-pooled-hard-negatives",
    )
    return key, {
        "construction": (
            "unit_norm(mean(domain_FIT) - mean(generic_FIT union code_FIT))"
        ),
        "fit_tokens": {
            corpus: len(FIT_INDICES) * TOKENS_PER_WINDOW for corpus in CORPORA
        },
        "source_norm": source_norm,
        "stored_norm": float(np.linalg.norm(key.astype(np.float64))),
        "key_sha256": sha256_array(key),
    }


def centered_within_corpora(
    hidden: dict[str, np.ndarray], indices: tuple[int, ...]
) -> tuple[np.ndarray, dict[str, np.ndarray], int, np.ndarray, float]:
    blocks = {corpus: selected_tokens(hidden[corpus], indices) for corpus in CORPORA}
    means = {
        corpus: blocks[corpus].mean(axis=0, dtype=np.float64) for corpus in CORPORA
    }
    centered_blocks = [
        np.ascontiguousarray(
            blocks[corpus] - means[corpus].astype(np.float32), dtype=np.float32
        )
        for corpus in CORPORA
    ]
    centered = np.concatenate(centered_blocks, axis=0)
    dof = int(centered.shape[0] - len(CORPORA))
    diagonal = np.sum(
        centered.astype(np.float64) ** 2, axis=0, dtype=np.float64
    ) / float(dof)
    mean_variance = float(diagonal.mean())
    delta = means["domain"] - 0.5 * (means["generic"] + means["code"])
    del blocks
    return centered, means, dof, delta, mean_variance


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
    """Solve (pooled within-corpus covariance + lambda I) w = delta."""

    x = np.asarray(centered, dtype=np.float32)
    b = np.asarray(delta, dtype=np.float32)
    lam = np.float32(regularization_lambda)
    inv_dof = np.float32(1.0 / float(dof))
    diagonal = (
        np.sum(x.astype(np.float64) ** 2, axis=0, dtype=np.float64) / float(dof)
    ).astype(np.float32)

    def matvec(vector: np.ndarray) -> np.ndarray:
        v = np.asarray(vector, dtype=np.float32)
        return np.asarray(x.T @ (x @ v) * inv_dof + lam * v, dtype=np.float32)

    def precondition(vector: np.ndarray) -> np.ndarray:
        return np.asarray(vector, dtype=np.float32) / (diagonal + lam)

    operator = LinearOperator((HIDDEN_DIM, HIDDEN_DIM), matvec=matvec, dtype=np.float32)
    preconditioner = LinearOperator(
        (HIDDEN_DIM, HIDDEN_DIM), matvec=precondition, dtype=np.float32
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
    residual = float(
        np.linalg.norm(matvec(solution).astype(np.float64) - b.astype(np.float64))
        / np.linalg.norm(b.astype(np.float64))
    )
    metadata = {
        "solver": "scipy.sparse.linalg.cg over exact covariance matvec",
        "info": int(info),
        "converged": bool(info == 0),
        "iterations": int(iterations),
        "rtol": float(rtol),
        "atol": 0.0,
        "maxiter": int(maxiter),
        "relative_residual": residual,
        "wall_seconds": float(time.perf_counter() - started),
    }
    return np.asarray(solution, dtype=np.float32), metadata


def validation_auc(
    hidden: dict[str, np.ndarray], key: np.ndarray
) -> dict[str, float]:
    # Project only the declared FIT-validation windows.  Candidate selection
    # does not even materialize EVAL scores as an unused side effect.
    scores = {
        corpus: (
            selected_tokens(hidden[corpus], INNER_VALIDATION_INDICES) @ key
        ).reshape(len(INNER_VALIDATION_INDICES), TOKENS_PER_WINDOW)
        for corpus in CORPORA
    }
    domain = scores["domain"]
    auc_dg = exact_auc(domain, scores["generic"])
    auc_dc = exact_auc(domain, scores["code"])
    return {
        "auc_domain_vs_generic": auc_dg,
        "auc_domain_vs_code": auc_dc,
        "minimum_auc": min(auc_dg, auc_dc),
        "mean_auc": 0.5 * (auc_dg + auc_dc),
    }


def choose_validation_candidate(
    candidates: list[dict[str, Any]], *, stronger: Callable[[dict[str, Any]], float]
) -> dict[str, Any]:
    valid = [item for item in candidates if item.get("fit_converged", True)]
    if not valid:
        raise RuntimeError("no converged FIT-only hyperparameter candidate")
    # Higher minimum AUC, higher mean AUC, then stronger regularization.
    return max(
        valid,
        key=lambda item: (
            item["validation"]["minimum_auc"],
            item["validation"]["mean_auc"],
            stronger(item),
        ),
    )


def k4_direction(
    hidden: dict[str, np.ndarray], *, cg_rtol: float, cg_maxiter: int
) -> tuple[np.ndarray, dict[str, Any]]:
    centered, _means, dof, delta, mean_variance = centered_within_corpora(
        hidden, INNER_TRAIN_INDICES
    )
    candidates_by_alpha: dict[float, dict[str, Any]] = {}
    warm: np.ndarray | None = None
    # Solve strongest to weakest shrinkage so CG warm starts remain stable.
    for alpha in reversed(K4_ALPHA_GRID):
        regularization_lambda = float(alpha * mean_variance)
        solution, solve_meta = cg_fisher_solve(
            centered,
            dof,
            delta,
            regularization_lambda,
            rtol=cg_rtol,
            maxiter=cg_maxiter,
            x0=warm,
        )
        if not solve_meta["converged"]:
            initial_attempt = solve_meta
            solution, solve_meta = cg_fisher_solve(
                centered,
                dof,
                delta,
                regularization_lambda,
                rtol=cg_rtol,
                maxiter=cg_maxiter * 4,
                x0=solution,
            )
            solve_meta["initial_attempt"] = initial_attempt
        warm = solution if solve_meta["converged"] else None
        candidate: dict[str, Any] = {
            "alpha": float(alpha),
            "lambda": regularization_lambda,
            "mean_within_variance": mean_variance,
            "solver": solve_meta,
            "fit_converged": bool(solve_meta["converged"]),
        }
        if solve_meta["converged"]:
            key, source_norm = unit_vector(
                solution, name=f"K4 validation alpha={alpha}"
            )
            candidate["direction_source_norm"] = source_norm
            candidate["validation"] = validation_auc(hidden, key)
        candidates_by_alpha[alpha] = candidate
    del centered
    gc.collect()

    candidates = [candidates_by_alpha[alpha] for alpha in K4_ALPHA_GRID]
    selected = choose_validation_candidate(
        candidates, stronger=lambda item: float(item["alpha"])
    )
    selected_alpha = float(selected["alpha"])

    centered_full, _means_full, dof_full, delta_full, mean_variance_full = (
        centered_within_corpora(hidden, FIT_INDICES)
    )
    selected_lambda = float(selected_alpha * mean_variance_full)
    solution, final_solver = cg_fisher_solve(
        centered_full,
        dof_full,
        delta_full,
        selected_lambda,
        rtol=cg_rtol,
        maxiter=cg_maxiter,
    )
    if not final_solver["converged"]:
        solution, final_solver = cg_fisher_solve(
            centered_full,
            dof_full,
            delta_full,
            selected_lambda,
            rtol=cg_rtol,
            maxiter=cg_maxiter * 4,
        )
    del centered_full
    gc.collect()
    if not final_solver["converged"]:
        raise RuntimeError(
            f"K4 final CG did not converge at alpha={selected_alpha}: {final_solver}"
        )
    key, source_norm = unit_vector(solution, name="K4 final Fisher direction")
    metadata = {
        "construction": (
            "unit_norm((Sigma_within_three_FIT_corpora + lambda I)^-1 "
            "(mean(domain_FIT) - mean(generic_FIT union code_FIT)))"
        ),
        "covariance": (
            "pooled sample covariance of residuals centered separately within "
            "domain, generic, and code"
        ),
        "alpha_grid": list(K4_ALPHA_GRID),
        "lambda_definition": "alpha * mean diagonal within-corpus variance",
        "inner_train_window_indices": list(INNER_TRAIN_INDICES),
        "inner_validation_window_indices": list(INNER_VALIDATION_INDICES),
        "selection_rule": (
            "highest min(validation AUC_dg, validation AUC_dc), then mean AUC, "
            "then larger alpha"
        ),
        "validation_candidates": candidates,
        "selected_alpha": selected_alpha,
        "selected_inner_lambda": float(selected["lambda"]),
        "selected_full_fit_lambda": selected_lambda,
        "full_fit_mean_within_variance": mean_variance_full,
        "full_fit_dof": dof_full,
        "final_solver": final_solver,
        "source_norm": source_norm,
        "stored_norm": float(np.linalg.norm(key.astype(np.float64))),
        "key_sha256": sha256_array(key),
    }
    return key, metadata


def logistic_dataset(
    hidden: dict[str, np.ndarray], indices: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    domain = selected_tokens(hidden["domain"], indices)
    generic = selected_tokens(hidden["generic"], indices)
    code = selected_tokens(hidden["code"], indices)
    x = np.concatenate([domain, generic, code], axis=0)
    y = np.concatenate(
        [
            np.ones(domain.shape[0], dtype=np.int8),
            np.zeros(generic.shape[0] + code.shape[0], dtype=np.int8),
        ]
    )
    return np.ascontiguousarray(x, dtype=np.float32), y


def fit_logistic(
    x: np.ndarray, y: np.ndarray, *, c_value: float, maxiter: int
) -> tuple[np.ndarray, dict[str, Any]]:
    model = LogisticRegression(
        C=float(c_value),
        penalty="l2",
        solver="lbfgs",
        fit_intercept=True,
        class_weight="balanced",
        max_iter=int(maxiter),
        tol=1.0e-5,
        random_state=DEFAULT_SEED,
    )
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    convergence_messages = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    coefficient = np.asarray(model.coef_[0], dtype=np.float32)
    key, source_norm = unit_vector(
        coefficient, name=f"K5 logistic C={c_value} coefficient"
    )
    metadata = {
        "C": float(c_value),
        "solver": "sklearn LogisticRegression(lbfgs)",
        "penalty": "l2",
        "class_weight": "balanced",
        "fit_intercept": True,
        "raw_intercept": float(model.intercept_[0]),
        "normalized_intercept": float(model.intercept_[0] / source_norm),
        "score_used_for_gate": (
            "h @ unit_norm(coef); the constant intercept is omitted because FIT "
            "quantile calibration absorbs any additive constant"
        ),
        "source_norm": source_norm,
        "n_iter": int(model.n_iter_[0]),
        "max_iter": int(maxiter),
        "tol": 1.0e-5,
        "convergence_warning": bool(convergence_messages),
        "convergence_messages": convergence_messages,
        "converged": not convergence_messages,
        "wall_seconds": float(time.perf_counter() - started),
    }
    return key, metadata


def k5_direction(
    hidden: dict[str, np.ndarray], *, logistic_maxiter: int
) -> tuple[np.ndarray, dict[str, Any]]:
    train_x, train_y = logistic_dataset(hidden, INNER_TRAIN_INDICES)
    candidates: list[dict[str, Any]] = []
    for c_value in K5_C_GRID:
        key, fit_meta = fit_logistic(
            train_x, train_y, c_value=c_value, maxiter=logistic_maxiter
        )
        candidate: dict[str, Any] = {
            "C": float(c_value),
            "fit": fit_meta,
            "fit_converged": bool(fit_meta["converged"]),
        }
        if fit_meta["converged"]:
            candidate["validation"] = validation_auc(hidden, key)
        candidates.append(candidate)
    del train_x, train_y
    gc.collect()
    selected = choose_validation_candidate(
        candidates, stronger=lambda item: -float(item["C"])
    )
    selected_c = float(selected["C"])

    full_x, full_y = logistic_dataset(hidden, FIT_INDICES)
    key, final_fit = fit_logistic(
        full_x, full_y, c_value=selected_c, maxiter=logistic_maxiter
    )
    if not final_fit["converged"]:
        key, final_fit = fit_logistic(
            full_x, full_y, c_value=selected_c, maxiter=logistic_maxiter * 4
        )
    del full_x, full_y
    gc.collect()
    if not final_fit["converged"]:
        raise RuntimeError(
            f"K5 final logistic fit did not converge at C={selected_c}: {final_fit}"
        )
    metadata = {
        "label": "trained linear logistic probe",
        "interpretation": "bounds what a linear gate can do; not a closed-form key",
        "construction": (
            "L2 logistic regression on raw router inputs, domain=1 and pooled "
            "generic+code=0; coefficient unit-normalized after fitting"
        ),
        "feature_standardization": "none (router inputs are post-normalization states)",
        "C_grid": list(K5_C_GRID),
        "inner_train_window_indices": list(INNER_TRAIN_INDICES),
        "inner_validation_window_indices": list(INNER_VALIDATION_INDICES),
        "selection_rule": (
            "highest min(validation AUC_dg, validation AUC_dc), then mean AUC, "
            "then smaller C"
        ),
        "validation_candidates": candidates,
        "selected_C": selected_c,
        "final_fit": final_fit,
        "stored_norm": float(np.linalg.norm(key.astype(np.float64))),
        "key_sha256": sha256_array(key),
    }
    return key, metadata


def pairwise_window_auc_matrix(
    positive: np.ndarray, negative: np.ndarray
) -> np.ndarray:
    positive = np.asarray(positive)
    negative = np.asarray(negative)
    if positive.ndim != 2 or negative.ndim != 2:
        raise ValueError("window AUC arrays must be [windows,tokens]")
    matrix = np.empty((positive.shape[0], negative.shape[0]), dtype=np.float64)
    for i, positive_window in enumerate(positive):
        for j, negative_window in enumerate(negative):
            matrix[i, j] = exact_auc(positive_window, negative_window)
    return matrix


def bootstrap_auc(
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    pairwise = pairwise_window_auc_matrix(positive, negative)
    rng = np.random.default_rng(seed)
    positive_indices = rng.integers(
        0, pairwise.shape[0], size=(resamples, pairwise.shape[0])
    )
    negative_indices = rng.integers(
        0, pairwise.shape[1], size=(resamples, pairwise.shape[1])
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


def bootstrap_rate(
    indicator: np.ndarray, *, resamples: int, seed: int
) -> dict[str, Any]:
    indicator = np.asarray(indicator, dtype=np.float64)
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


def bootstrap_majority(
    indicator: np.ndarray, *, resamples: int, seed: int
) -> dict[str, Any]:
    indicator = np.asarray(indicator, dtype=np.float64)
    majority = (indicator.mean(axis=1) > 0.5).astype(np.float64)
    metric = bootstrap_rate(
        majority[:, None], resamples=resamples, seed=seed
    )
    metric.update(
        {
            "majority_window_count": int(majority.sum()),
            "eval_window_count": int(majority.size),
            "definition": "strictly more than 50% of tokens have score >= tau",
        }
    )
    return metric


def select_tau(scores: dict[str, np.ndarray]) -> dict[str, Any]:
    fit = np.asarray(FIT_INDICES, dtype=np.int64)
    evaluation = np.asarray(EVAL_INDICES, dtype=np.int64)
    pooled_negative_fit = np.concatenate(
        [scores["generic"][fit].reshape(-1), scores["code"][fit].reshape(-1)]
    )
    curve: list[dict[str, Any]] = []
    for label, quantile in TAU_GRID:
        tau = float(np.quantile(pooled_negative_fit, quantile, method="linear"))
        fit_rates = {
            "domain_recall": float(np.mean(scores["domain"][fit] >= tau)),
            "generic_fpr": float(np.mean(scores["generic"][fit] >= tau)),
            "code_fpr": float(np.mean(scores["code"][fit] >= tau)),
        }
        eval_rates = {
            "domain_recall": float(np.mean(scores["domain"][evaluation] >= tau)),
            "generic_fpr": float(np.mean(scores["generic"][evaluation] >= tau)),
            "code_fpr": float(np.mean(scores["code"][evaluation] >= tau)),
        }
        fit_feasible = bool(
            fit_rates["generic_fpr"] <= GENERIC_FPR_CAP
            and fit_rates["code_fpr"] <= CODE_FPR_CAP
        )
        eval_majority = {
            corpus: float(
                np.mean(
                    np.mean(scores[corpus][evaluation] >= tau, axis=1) > 0.5
                )
            )
            for corpus in CORPORA
        }
        curve.append(
            {
                "quantile_label": label,
                "quantile": float(quantile),
                "tau": tau,
                "fit": {**fit_rates, "fpr_constraints_met": fit_feasible},
                "eval": eval_rates,
                "eval_window_majority_firing": eval_majority,
            }
        )
    feasible = [row for row in curve if row["fit"]["fpr_constraints_met"]]
    if not feasible:
        raise RuntimeError("registered FIT tau grid has no FPR-feasible threshold")
    # Primary law: maximum FIT recall.  Exact ties are resolved conservatively.
    selected = max(
        feasible,
        key=lambda row: (
            row["fit"]["domain_recall"],
            -row["fit"]["code_fpr"],
            -row["fit"]["generic_fpr"],
            row["quantile"],
        ),
    )
    return {
        "policy": (
            "NumPy linear p90/p95/p99/p99.5 of pooled generic+code FIT scores; "
            "maximize FIT domain recall subject to generic FPR <= 0.02 and code "
            "FPR <= 0.05; ties prefer lower code FPR, lower generic FPR, then "
            "higher quantile"
        ),
        "fire_comparison": "score >= tau",
        "fit_negative_token_count": int(pooled_negative_fit.size),
        "selected_quantile_label": selected["quantile_label"],
        "selected_quantile": selected["quantile"],
        "selected_tau": selected["tau"],
        "selected_fit_rates": selected["fit"],
        "operating_curve": curve,
    }


def analyze_scores(
    scores: dict[str, np.ndarray], *, resamples: int, seed: int
) -> dict[str, Any]:
    tau_result = select_tau(scores)
    tau = float(tau_result["selected_tau"])
    evaluation = np.asarray(EVAL_INDICES, dtype=np.int64)
    evaluated = {corpus: scores[corpus][evaluation] for corpus in CORPORA}
    fired = {corpus: evaluated[corpus] >= tau for corpus in CORPORA}
    metrics = {
        "auc_domain_vs_generic": bootstrap_auc(
            evaluated["domain"], evaluated["generic"],
            resamples=resamples, seed=seed + 1,
        ),
        "auc_domain_vs_code": bootstrap_auc(
            evaluated["domain"], evaluated["code"],
            resamples=resamples, seed=seed + 2,
        ),
        "domain_recall_at_tau": bootstrap_rate(
            fired["domain"], resamples=resamples, seed=seed + 10,
        ),
        "generic_fpr_at_tau": bootstrap_rate(
            fired["generic"], resamples=resamples, seed=seed + 11,
        ),
        "code_fpr_at_tau": bootstrap_rate(
            fired["code"], resamples=resamples, seed=seed + 12,
        ),
        "window_majority_firing": {
            corpus: bootstrap_majority(
                fired[corpus], resamples=resamples, seed=seed + 20 + index
            )
            for index, corpus in enumerate(CORPORA)
        },
        "eval_score_means": {
            corpus: float(evaluated[corpus].mean(dtype=np.float64))
            for corpus in CORPORA
        },
        "eval_window_indices": {corpus: list(EVAL_INDICES) for corpus in CORPORA},
        "tau_calibration_window_indices": {
            "generic": list(FIT_INDICES),
            "code": list(FIT_INDICES),
        },
        **tau_result,
    }
    metrics["qualifies"] = bool(
        metrics["domain_recall_at_tau"]["value"] >= DOMAIN_RECALL_FLOOR
        and metrics["generic_fpr_at_tau"]["value"] <= GENERIC_FPR_CAP
        and metrics["code_fpr_at_tau"]["value"] <= CODE_FPR_CAP
    )
    metrics["registered_criteria"] = {
        "domain_recall_gte": DOMAIN_RECALL_FLOOR,
        "generic_fpr_lte": GENERIC_FPR_CAP,
        "code_fpr_lte": CODE_FPR_CAP,
    }
    return metrics


def split_provenance() -> dict[str, Any]:
    sources: list[tuple[str, list[int]]] = []
    key_fit: dict[str, Any] = {
        "K3": {corpus: list(FIT_INDICES) for corpus in CORPORA},
        "K4": {
            "hyperparameter_train": {corpus: list(INNER_TRAIN_INDICES) for corpus in CORPORA},
            "hyperparameter_validation": {corpus: list(INNER_VALIDATION_INDICES) for corpus in CORPORA},
            "final_refit": {corpus: list(FIT_INDICES) for corpus in CORPORA},
        },
        "K5": {
            "hyperparameter_train": {corpus: list(INNER_TRAIN_INDICES) for corpus in CORPORA},
            "hyperparameter_validation": {corpus: list(INNER_VALIDATION_INDICES) for corpus in CORPORA},
            "final_refit": {corpus: list(FIT_INDICES) for corpus in CORPORA},
        },
    }
    for corpus in CORPORA:
        sources.append((f"key_fit.K3.{corpus}", list(FIT_INDICES)))
        for arm in ("K4", "K5"):
            for phase, indices in (
                ("hyperparameter_train", INNER_TRAIN_INDICES),
                ("hyperparameter_validation", INNER_VALIDATION_INDICES),
                ("final_refit", FIT_INDICES),
            ):
                sources.append((f"key_fit.{arm}.{phase}.{corpus}", list(indices)))
    for arm in ARMS:
        for corpus in ("generic", "code", "domain"):
            sources.append((f"tau_fit.{arm}.{corpus}", list(FIT_INDICES)))
    evaluation_set = set(EVAL_INDICES)
    checks = []
    for name, indices in sources:
        overlap = sorted(set(indices).intersection(evaluation_set))
        checks.append(
            {
                "fit_source": name,
                "evaluation_windows": list(EVAL_INDICES),
                "overlap": overlap,
                "overlap_count": len(overlap),
            }
        )
    inner_overlap = sorted(
        set(INNER_TRAIN_INDICES).intersection(INNER_VALIDATION_INDICES)
    )
    return {
        "schema": "moe_rt2_1_split_provenance_v1",
        "window_index_basis": "zero-based RT2 capture window indices",
        "fit_window_indices": list(FIT_INDICES),
        "eval_window_indices": list(EVAL_INDICES),
        "inner_fit_train_window_indices": list(INNER_TRAIN_INDICES),
        "inner_fit_validation_window_indices": list(INNER_VALIDATION_INDICES),
        "inner_train_validation_overlap": inner_overlap,
        "key_construction_and_hyperparameter_selection": key_fit,
        "tau_calibration": {
            arm: {
                "domain_scores_for_recall": list(FIT_INDICES),
                "generic_negative_scores": list(FIT_INDICES),
                "code_negative_scores": list(FIT_INDICES),
            }
            for arm in ARMS
        },
        "evaluation": {corpus: list(EVAL_INDICES) for corpus in CORPORA},
        "fit_eval_overlap_checks": checks,
        "fit_eval_overlap_count": int(sum(item["overlap_count"] for item in checks)),
        "tokens_per_corpus_fit": len(FIT_INDICES) * TOKENS_PER_WINDOW,
        "tokens_per_corpus_eval": len(EVAL_INDICES) * TOKENS_PER_WINDOW,
        "eval_used_for_key_tau_or_hyperparameter_selection": False,
    }


def reproduce_g1(
    arrays: dict[str, list[np.ndarray]], *, output_dir: Path
) -> dict[str, Any]:
    """Reproduce the registered K1 L12 row directly from capture arrays."""

    started = time.perf_counter()
    hidden = materialize_layer(arrays, 12)
    key, key_meta = k1_direction(hidden)
    scores = score_key(hidden, key)
    fit = np.asarray(FIT_INDICES, dtype=np.int64)
    evaluation = np.asarray(EVAL_INDICES, dtype=np.int64)
    tau = float(np.quantile(scores["generic"][fit], 0.99, method="linear"))
    observed = {
        "auc_domain_vs_generic": exact_auc(
            scores["domain"][evaluation], scores["generic"][evaluation]
        ),
        "domain_recall_at_tau": float(np.mean(scores["domain"][evaluation] >= tau)),
        "generic_fpr_at_tau": float(np.mean(scores["generic"][evaluation] >= tau)),
        "code_fpr_at_tau": float(np.mean(scores["code"][evaluation] >= tau)),
    }
    errors = {
        name: abs(observed[name] - reference)
        for name, reference in G1_REFERENCE.items()
    }
    passed = all(value <= G1_TOLERANCE for value in errors.values())
    receipt = {
        "schema": "moe_rt2_1_g1_math_truth_v1",
        "created_at": now_iso(),
        "passed": bool(passed),
        "verdict": "GREEN" if passed else "RED",
        "layer": 12,
        "source": "raw RT2 router-input fp16 npy captures",
        "pipeline_independence": (
            "implemented in gpt_oss20b_key_refinement.py; no RT2 analyzer helper imported"
        ),
        "key": key_meta,
        "tau": tau,
        "tau_definition": "NumPy linear p99 of GENERIC-FIT K1 scores",
        "fit_window_indices": list(FIT_INDICES),
        "eval_window_indices": list(EVAL_INDICES),
        "reference_registered_rounded_row": G1_REFERENCE,
        "observed_raw_capture_row": observed,
        "absolute_errors": errors,
        "maximum_absolute_error": max(errors.values()),
        "tolerance": G1_TOLERANCE,
        "wall_seconds": float(time.perf_counter() - started),
    }
    write_json(output_dir / "g1_math_truth.json", receipt)
    del hidden, scores
    gc.collect()
    if not passed:
        raise RuntimeError(f"G1 RED: {receipt}")
    return receipt


def run_preflight(args: argparse.Namespace) -> int:
    output = ensure_output_dir(args.output_dir)
    started = time.perf_counter()
    arrays, input_receipt = verify_inputs(hash_payloads=True)
    g1 = reproduce_g1(arrays, output_dir=output)
    payload = {
        "schema": "moe_rt2_1_preflight_v1",
        "created_at": now_iso(),
        "status": "passed",
        "cpu_only": True,
        "cuda_visible_devices_effective": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "script_path": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "order_path": str(ORDER_PATH),
        "input_integrity": input_receipt,
        "g1_math_truth": g1,
        "split_provenance": split_provenance(),
        "wall_seconds": float(time.perf_counter() - started),
    }
    write_json(output / "preflight.json", payload)
    print(
        json.dumps(
            {
                "status": "passed",
                "G0_checks": f"{input_receipt['checks_passed']}/{input_receipt['checks_total']}",
                "G1_max_abs_error": g1["maximum_absolute_error"],
                "preflight": str(output / "preflight.json"),
            }
        ),
        flush=True,
    )
    return 0


def run_self_test(args: argparse.Namespace) -> int:
    output = ensure_output_dir(args.output_dir)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    splits = split_provenance()
    check("fit_eval_overlap_is_zero", splits["fit_eval_overlap_count"] == 0, splits["fit_eval_overlap_count"])
    check("inner_train_validation_overlap_is_zero", not splits["inner_train_validation_overlap"], splits["inner_train_validation_overlap"])
    positives = np.asarray([[2.0, 3.0], [4.0, 5.0]])
    negatives = np.asarray([[0.0, 1.0], [1.0, 2.0]])
    brute = float(
        np.mean(
            (positives.reshape(-1, 1) > negatives.reshape(1, -1))
            + 0.5 * (positives.reshape(-1, 1) == negatives.reshape(1, -1))
        )
    )
    check("exact_auc_matches_bruteforce", abs(exact_auc(positives, negatives) - brute) < 1.0e-15, brute)
    check("auc_ties_equal_half", exact_auc(np.ones(8), np.ones(8)) == 0.5, exact_auc(np.ones(8), np.ones(8)))
    metric_a = bootstrap_auc(positives, negatives, resamples=200, seed=args.seed)
    metric_b = bootstrap_auc(positives, negatives, resamples=200, seed=args.seed)
    check("bootstrap_is_deterministic", metric_a == metric_b, metric_a)

    rng = np.random.default_rng(args.seed)
    synthetic_scores = {
        "domain": rng.normal(3.0, 1.0, size=(N_WINDOWS, 32)).astype(np.float32),
        "generic": rng.normal(0.0, 1.0, size=(N_WINDOWS, 32)).astype(np.float32),
        "code": rng.normal(0.5, 1.0, size=(N_WINDOWS, 32)).astype(np.float32),
    }
    tau = select_tau(synthetic_scores)
    check(
        "selected_tau_satisfies_fit_fpr_caps",
        bool(tau["selected_fit_rates"]["fpr_constraints_met"]),
        tau["selected_fit_rates"],
    )
    check("tau_curve_has_four_rows", len(tau["operating_curve"]) == 4, len(tau["operating_curve"]))
    failed = [item for item in checks if not item["passed"]]
    payload = {
        "schema": "moe_rt2_1_cpu_validation_v1",
        "created_at": now_iso(),
        "status": "passed" if not failed else "failed",
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "failed_check_names": [item["name"] for item in failed],
        "checks": checks,
    }
    write_json(output / "cpu_validation.json", payload)
    print(json.dumps({"status": payload["status"], "checks": f"{payload['checks_passed']}/{payload['checks_total']}"}), flush=True)
    return 0 if not failed else 2


def metric_value(row: dict[str, Any], name: str) -> float:
    return float(row["metrics"][name]["value"])


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Prefer qualifiers, then FPR-feasible rows, recall, robust AUC, and layer."""

    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        metrics = row["metrics"]
        fpr_feasible = bool(
            metrics["generic_fpr_at_tau"]["value"] <= GENERIC_FPR_CAP
            and metrics["code_fpr_at_tau"]["value"] <= CODE_FPR_CAP
        )
        return (
            bool(metrics["qualifies"]),
            fpr_feasible,
            metrics["domain_recall_at_tau"]["value"],
            min(
                metrics["auc_domain_vs_generic"]["value"],
                metrics["auc_domain_vs_code"]["value"],
            ),
            -metrics["code_fpr_at_tau"]["value"],
            -metrics["generic_fpr_at_tau"]["value"],
            -row["layer"],
        )

    return max(rows, key=key)


def save_keys(
    path: Path,
    key_store: dict[str, list[np.ndarray]],
    tau_store: dict[str, list[float]],
    k4_alphas: list[float],
    k4_lambdas: list[float],
    k5_cs: list[float],
) -> None:
    with path.open("wb") as handle:
        np.savez(
            handle,
            layer_indices=np.arange(N_LAYERS, dtype=np.int16),
            K3=np.stack(key_store["K3"]).astype(np.float32),
            K4=np.stack(key_store["K4"]).astype(np.float32),
            K5=np.stack(key_store["K5"]).astype(np.float32),
            tau_K3=np.asarray(tau_store["K3"], dtype=np.float32),
            tau_K4=np.asarray(tau_store["K4"], dtype=np.float32),
            tau_K5=np.asarray(tau_store["K5"], dtype=np.float32),
            K4_selected_alpha=np.asarray(k4_alphas, dtype=np.float64),
            K4_selected_lambda=np.asarray(k4_lambdas, dtype=np.float64),
            K5_selected_C=np.asarray(k5_cs, dtype=np.float64),
            fit_window_indices=np.asarray(FIT_INDICES, dtype=np.int16),
            eval_window_indices=np.asarray(EVAL_INDICES, dtype=np.int16),
            inner_train_window_indices=np.asarray(INNER_TRAIN_INDICES, dtype=np.int16),
            inner_validation_window_indices=np.asarray(INNER_VALIDATION_INDICES, dtype=np.int16),
        )


def fmt_metric(metric: dict[str, Any], digits: int = 4) -> str:
    return (
        f"{metric['value']:.{digits}f} "
        f"[{metric['ci95_low']:.{digits}f}, {metric['ci95_high']:.{digits}f}]"
    )


def report_row(row: dict[str, Any]) -> str:
    metrics = row["metrics"]
    majority = metrics["window_majority_firing"]
    return (
        f"| {row['layer']} | {fmt_metric(metrics['auc_domain_vs_generic'])} | "
        f"{fmt_metric(metrics['auc_domain_vs_code'])} | "
        f"{metrics['selected_quantile_label']} | {metrics['selected_tau']:.6f} | "
        f"{fmt_metric(metrics['domain_recall_at_tau'])} | "
        f"{fmt_metric(metrics['generic_fpr_at_tau'])} | "
        f"{fmt_metric(metrics['code_fpr_at_tau'])} | "
        f"{majority['domain']['value']:.3f}/{majority['generic']['value']:.3f}/{majority['code']['value']:.3f} | "
        f"{'yes' if metrics['qualifies'] else 'no'} |"
    )


def render_report(analysis: dict[str, Any]) -> str:
    lines: list[str] = [
        "# MOE-RT2.1 Key Refinement Report",
        "",
        f"Generated: `{analysis['created_at']}`",
        "",
        "## Evidence boundary",
        "",
        "This is linear analysis over captured router-input activations from one model. It is not a model-quality result, an expert-viability result, or evidence from a model run. K5 is a trained linear probe; K3 and K4 are the closed-form key arms.",
        "",
        "## Registered verdict",
        "",
        analysis["registered_verdict_sentence"],
        "",
        f"Qualifying layer-arm pairs: **{analysis['decision']['qualifying_pair_count']}**.",
        "",
        "The fixed EVAL rule is domain recall >= 0.50, generic FPR <= 0.02, and code FPR <= 0.05 at the FIT-frozen threshold.",
        "",
        "## Gates",
        "",
    ]
    for gate in ("G0", "G1", "G2", "G3"):
        lines.append(f"- {analysis['gates'][gate]['report_line']}")
    lines.extend(
        [
            "",
            "## Split and selection law",
            "",
            f"- FIT windows: `{list(FIT_INDICES)}`; EVAL windows: `{list(EVAL_INDICES)}`.",
            f"- Inner FIT train windows: `{list(INNER_TRAIN_INDICES)}`; inner FIT validation windows: `{list(INNER_VALIDATION_INDICES)}`.",
            "- K4/K5 hyperparameters maximize the smaller validation AUC across generic and code, then mean AUC; exact ties prefer stronger regularization. Selected arms are refit on all FIT windows.",
            "- Tau is selected only from p90/p95/p99/p99.5 of pooled generic+code FIT scores, maximizing FIT recall under both FIT FPR caps. It is then frozen.",
            "- Every AUC, selected-threshold rate, and operating-curve value below is from EVAL windows only.",
            f"- Recorded FIT/EVAL overlap count: `{analysis['split_provenance']['fit_eval_overlap_count']}`.",
            "",
            "## Best row per arm",
            "",
            "Best-row rule: prefer registered qualifiers, then rows meeting both EVAL FPR caps, then higher recall, higher worst-case AUC, lower FPRs, and lower layer index.",
            "",
            "| Arm | Layer | AUC_dg (95% CI) | AUC_dc (95% CI) | Recall@tau (95% CI) | Generic FPR (95% CI) | Code FPR (95% CI) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        row = analysis["decision"]["best_row_per_arm"][arm]
        metrics = row["metrics"]
        lines.append(
            f"| {arm} | {row['layer']} | {fmt_metric(metrics['auc_domain_vs_generic'])} | "
            f"{fmt_metric(metrics['auc_domain_vs_code'])} | {fmt_metric(metrics['domain_recall_at_tau'])} | "
            f"{fmt_metric(metrics['generic_fpr_at_tau'])} | {fmt_metric(metrics['code_fpr_at_tau'])} |"
        )
    lines.extend(["", "## Every qualifying pair", ""])
    qualifying = analysis["decision"]["qualifying_pairs"]
    if qualifying:
        lines.extend(
            [
                "| Arm | Layer | AUC_dg | AUC_dc | Recall@tau | Generic FPR | Code FPR | Frozen quantile |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in qualifying:
            metrics = row["metrics"]
            lines.append(
                f"| {row['arm']} | {row['layer']} | {fmt_metric(metrics['auc_domain_vs_generic'])} | "
                f"{fmt_metric(metrics['auc_domain_vs_code'])} | {fmt_metric(metrics['domain_recall_at_tau'])} | "
                f"{fmt_metric(metrics['generic_fpr_at_tau'])} | {fmt_metric(metrics['code_fpr_at_tau'])} | "
                f"{metrics['selected_quantile_label']} |"
            )
    else:
        lines.append("No layer-arm pair qualified.")

    for arm in ARMS:
        lines.extend(
            [
                "",
                f"## {arm} per-layer EVAL results",
                "",
                "| L | AUC_dg (95% CI) | AUC_dc (95% CI) | q | tau | Recall (95% CI) | Generic FPR (95% CI) | Code FPR (95% CI) | Window-majority D/G/C | Qualifies |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
            ]
        )
        for row in analysis["rows"]:
            if row["arm"] == arm:
                lines.append(report_row(row))

    lines.extend(["", "## FIT-only hyperparameter selections", ""])
    lines.extend(
        [
            "| L | K4 alpha | K4 lambda (full FIT) | K4 min validation AUC | K5 C | K5 min validation AUC |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    by_layer = {row["layer"]: row for row in analysis["layers"]}
    for layer in range(N_LAYERS):
        item = by_layer[layer]
        k4 = item["key_metadata"]["K4"]
        k5 = item["key_metadata"]["K5"]
        k4_selected = next(
            candidate for candidate in k4["validation_candidates"]
            if candidate["alpha"] == k4["selected_alpha"]
        )
        k5_selected = next(
            candidate for candidate in k5["validation_candidates"]
            if candidate["C"] == k5["selected_C"]
        )
        lines.append(
            f"| {layer} | {k4['selected_alpha']:.4g} | {k4['selected_full_fit_lambda']:.6g} | "
            f"{k4_selected['validation']['minimum_auc']:.4f} | {k5['selected_C']:.4g} | "
            f"{k5_selected['validation']['minimum_auc']:.4f} |"
        )

    for arm in ARMS:
        lines.extend(
            [
                "",
                f"## {arm} full frozen-threshold EVAL operating curves",
                "",
                "Each tau in this descriptive table was constructed from pooled FIT negatives; no EVAL-side threshold tuning occurred.",
                "",
                "| L | FIT quantile | tau | EVAL recall | EVAL generic FPR | EVAL code FPR | EVAL majority D/G/C | Selected |",
                "|---:|---:|---:|---:|---:|---:|---:|:---:|",
            ]
        )
        for row in analysis["rows"]:
            if row["arm"] != arm:
                continue
            metrics = row["metrics"]
            for curve in metrics["operating_curve"]:
                rates = curve["eval"]
                majority = curve["eval_window_majority_firing"]
                lines.append(
                    f"| {row['layer']} | {curve['quantile_label']} | {curve['tau']:.6f} | "
                    f"{rates['domain_recall']:.4f} | {rates['generic_fpr']:.4f} | {rates['code_fpr']:.4f} | "
                    f"{majority['domain']:.3f}/{majority['generic']:.3f}/{majority['code']:.3f} | "
                    f"{'yes' if curve['quantile_label'] == metrics['selected_quantile_label'] else 'no'} |"
                )

    cosine_summary = analysis["mechanism"]["cosine_to_K1_summary"]
    lines.extend(
        [
            "",
            "## Mechanism note",
            "",
            "K1 subtracts only the generic centroid, so code remains on the positive side. K3 changes the negative centroid to the equal-token union of generic and code, explicitly rotating the key toward features that distinguish domain from both. K4 applies the same hard-negative mean contrast after inverse within-corpus covariance weighting, suppressing high-variance directions. K5 is the trained linear upper-bound arm and optimizes a balanced logistic loss rather than a closed-form centroid rule.",
            "",
            f"Across layers, cosine-to-K1 medians were K3 `{cosine_summary['K3']['median']:.4f}`, K4 `{cosine_summary['K4']['median']:.4f}`, and K5 `{cosine_summary['K5']['median']:.4f}`. These rotations and the observed EVAL code FPRs describe addressability only; they do not establish expert usefulness.",
            "",
            "## Reproducibility and artifacts",
            "",
            f"- CPU command: `{analysis['runtime']['command']}`",
            f"- Bootstrap: `{analysis['bootstrap_resamples']}` window-cluster resamples.",
            f"- RT2 analysis SHA-256 before/after: `{analysis['g0']['rt2_analysis_sha256_before']}` / `{analysis['g0']['rt2_analysis_sha256_after']}`.",
            f"- Refined keys: `{analysis['keys_artifact']['path']}` (`{analysis['keys_artifact']['sha256']}`).",
            "- Exact created/modified paths:",
        ]
    )
    lines.extend(f"  - `{path}`" for path in analysis["created_or_modified_paths"])
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- One model and three frozen 16-window corpora; only eight windows per corpus fit each final key.",
            "- Token-level linear gates only; no nonlinear, sequence-aware, or model-in-the-loop expert test.",
            "- Window bootstrap quantifies window sampling variability for this captured set; it does not cover model, corpus, or hyperparameter-selection uncertainty.",
            "- No model-quality or expert-viability claim is made.",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(args: argparse.Namespace) -> int:
    if args.bootstrap_resamples < 2000:
        raise ValueError("ORDER requires at least 2000 bootstrap resamples")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    output = ensure_output_dir(args.output_dir)
    overall_started = time.perf_counter()

    # G0 and G1 are intentionally completed and persisted before any K3/K4/K5 fit.
    arrays, input_receipt = verify_inputs(hash_payloads=True)
    g1 = reproduce_g1(arrays, output_dir=output)
    preflight = {
        "schema": "moe_rt2_1_preflight_v1",
        "created_at": now_iso(),
        "status": "passed",
        "cpu_only": True,
        "cuda_visible_devices_effective": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "script_path": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "input_integrity": input_receipt,
        "g1_math_truth": g1,
        "split_provenance": split_provenance(),
    }
    write_json(output / "preflight.json", preflight)

    layer_receipts: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    key_store: dict[str, list[np.ndarray]] = {arm: [] for arm in ARMS}
    tau_store: dict[str, list[float]] = {arm: [] for arm in ARMS}
    k4_alphas: list[float] = []
    k4_lambdas: list[float] = []
    k5_cs: list[float] = []

    with threadpool_limits(limits=args.threads):
        for layer in range(N_LAYERS):
            layer_started = time.perf_counter()
            hidden = materialize_layer(arrays, layer)
            k1, k1_meta = k1_direction(hidden)
            k3, k3_meta = k3_direction(hidden)
            k4, k4_meta = k4_direction(
                hidden, cg_rtol=args.cg_rtol, cg_maxiter=args.cg_maxiter
            )
            k5, k5_meta = k5_direction(
                hidden, logistic_maxiter=args.logistic_maxiter
            )
            keys = {"K3": k3, "K4": k4, "K5": k5}
            metadata = {"K1_reference": k1_meta, "K3": k3_meta, "K4": k4_meta, "K5": k5_meta}
            for arm in ARMS:
                metadata[arm]["cosine_to_K1"] = float(
                    np.dot(keys[arm].astype(np.float64), k1.astype(np.float64))
                )
                metadata[arm]["cosine_to_K3"] = float(
                    np.dot(keys[arm].astype(np.float64), k3.astype(np.float64))
                )

            scored = score_keys(hidden, keys)
            arm_results: dict[str, Any] = {}
            for arm_index, arm in enumerate(ARMS):
                arm_scores = {corpus: scored[corpus][arm] for corpus in CORPORA}
                metrics = analyze_scores(
                    arm_scores,
                    resamples=args.bootstrap_resamples,
                    seed=args.seed + layer * 1000 + arm_index * 100,
                )
                arm_results[arm] = metrics
                row = {"layer": layer, "arm": arm, "metrics": metrics}
                rows.append(row)
                key_store[arm].append(keys[arm].copy())
                tau_store[arm].append(float(metrics["selected_tau"]))
            k4_alphas.append(float(k4_meta["selected_alpha"]))
            k4_lambdas.append(float(k4_meta["selected_full_fit_lambda"]))
            k5_cs.append(float(k5_meta["selected_C"]))
            layer_receipt = {
                "layer": layer,
                "key_metadata": metadata,
                "arms": arm_results,
                "wall_seconds": float(time.perf_counter() - layer_started),
            }
            layer_receipts.append(layer_receipt)
            print(
                json.dumps(
                    {
                        "layer": layer,
                        "seconds": layer_receipt["wall_seconds"],
                        "K4_alpha": k4_meta["selected_alpha"],
                        "K5_C": k5_meta["selected_C"],
                        "qualified": [arm for arm in ARMS if arm_results[arm]["qualifies"]],
                    }
                ),
                flush=True,
            )
            del hidden, scored, keys, k1, k3, k4, k5
            gc.collect()

    keys_path = output / "refined_keys.npz"
    save_keys(keys_path, key_store, tau_store, k4_alphas, k4_lambdas, k5_cs)
    split = split_provenance()
    if split["fit_eval_overlap_count"] != 0:
        raise RuntimeError("G2 RED: nonzero FIT/EVAL overlap")

    qualifying = [row for row in rows if row["metrics"]["qualifies"]]
    best = {
        arm: best_row([row for row in rows if row["arm"] == arm]) for arm in ARMS
    }
    supported = bool(qualifying)
    if supported:
        verdict_sentence = "H-RT2.1 SUPPORTED."
    else:
        verdict_sentence = (
            "fine-grained linear addressability NOT DETECTED under these limits "
            "(linear keys, 8-window fits, token-level gating, frozen-τ policy)"
        )

    rt2_after = sha256_file(RT2_ANALYSIS_PATH)
    g0_passed = bool(
        input_receipt["passed"]
        and input_receipt["rt2_analysis_sha256_at_open"] == rt2_after
    )
    created_paths = [
        str(SCRIPT_PATH),
        str(output / "preflight.json"),
        str(output / "g1_math_truth.json"),
        str(output / "cpu_validation.json"),
        str(keys_path),
        str(output / "analysis.json"),
        str(output / "MOE_RT2_1_REPORT.md"),
    ]
    cosine_summary: dict[str, Any] = {}
    for arm in ARMS:
        values = np.asarray(
            [layer["key_metadata"][arm]["cosine_to_K1"] for layer in layer_receipts],
            dtype=np.float64,
        )
        cosine_summary[arm] = {
            "minimum": float(values.min()),
            "median": float(np.median(values)),
            "maximum": float(values.max()),
        }

    gates = {
        "G0": {
            "verdict": "GREEN" if g0_passed else "RED",
            "report_line": (
                f"G0 {'GREEN' if g0_passed else 'RED'} — capture/receipt checks="
                f"{input_receipt['checks_passed']}/{input_receipt['checks_total']}; "
                f"RT2 analysis SHA before=after={rt2_after}"
            ),
            "checks_passed": input_receipt["checks_passed"],
            "checks_total": input_receipt["checks_total"],
        },
        "G1": {
            "verdict": g1["verdict"],
            "report_line": (
                f"G1 {g1['verdict']} — L12 K1 AUC_dg="
                f"{g1['observed_raw_capture_row']['auc_domain_vs_generic']:.7f}, "
                f"recall={g1['observed_raw_capture_row']['domain_recall_at_tau']:.7f}, "
                f"generic FPR={g1['observed_raw_capture_row']['generic_fpr_at_tau']:.7f}, "
                f"code FPR={g1['observed_raw_capture_row']['code_fpr_at_tau']:.7f}; "
                f"max error={g1['maximum_absolute_error']:.3g} <= {G1_TOLERANCE}"
            ),
        },
        "G2": {
            "verdict": "GREEN" if split["fit_eval_overlap_count"] == 0 else "RED",
            "report_line": (
                f"G2 {'GREEN' if split['fit_eval_overlap_count'] == 0 else 'RED'} — "
                f"FIT={list(FIT_INDICES)}, EVAL={list(EVAL_INDICES)}, "
                f"recorded overlap={split['fit_eval_overlap_count']}"
            ),
        },
        "G3": {
            "verdict": "GREEN",
            "report_line": (
                f"G3 GREEN — report has {len(rows)}/72 layer-arm rows and "
                f"{sum(len(row['metrics']['operating_curve']) for row in rows)}/288 operating points"
            ),
            "report_path": str(output / "MOE_RT2_1_REPORT.md"),
            "layer_arm_rows": len(rows),
            "operating_curve_rows": sum(
                len(row["metrics"]["operating_curve"]) for row in rows
            ),
        },
    }
    analysis: dict[str, Any] = {
        "schema": "moe_rt2_1_analysis_v1",
        "created_at": now_iso(),
        "status": "complete",
        "order": str(ORDER_PATH),
        "script": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "cpu_only": True,
        "evidence_class": "linear analysis over captured activations of one model",
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "seed": int(args.seed),
        "registered_verdict_sentence": verdict_sentence,
        "registered_decision_rule": {
            "supported_iff_at_least_one_pair_qualifies": True,
            "domain_recall_gte": DOMAIN_RECALL_FLOOR,
            "generic_fpr_lte": GENERIC_FPR_CAP,
            "code_fpr_lte": CODE_FPR_CAP,
        },
        "g0": {
            "input_integrity": input_receipt,
            "rt2_analysis_sha256_before": input_receipt["rt2_analysis_sha256_at_open"],
            "rt2_analysis_sha256_after": rt2_after,
            "rt2_artifacts_untouched": g0_passed,
        },
        "g1": g1,
        "split_provenance": split,
        "gates": gates,
        "layers": layer_receipts,
        "rows": rows,
        "decision": {
            "supported": supported,
            "qualifying_pair_count": len(qualifying),
            "qualifying_pairs": qualifying,
            "best_row_selection_rule": (
                "prefer qualifiers, then both EVAL FPR caps, then recall, worst-case "
                "AUC, lower code FPR, lower generic FPR, lower layer"
            ),
            "best_row_per_arm": best,
        },
        "mechanism": {"cosine_to_K1_summary": cosine_summary},
        "keys_artifact": {
            "path": str(keys_path),
            "sha256": sha256_file(keys_path),
            "arrays": [
                "K3", "K4", "K5", "tau_K3", "tau_K4", "tau_K5",
                "K4_selected_alpha", "K4_selected_lambda", "K5_selected_C",
            ],
        },
        "runtime": {
            "command": (
                f"CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "
                f"python3 scripts/gpt_oss20b_key_refinement.py analyze "
                f"--bootstrap-resamples {args.bootstrap_resamples} --seed {args.seed} "
                f"--threads {args.threads} --cg-rtol {args.cg_rtol} "
                f"--cg-maxiter {args.cg_maxiter} "
                f"--logistic-maxiter {args.logistic_maxiter}"
            ),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "threadpool_info": threadpool_info(),
            "threads_limit": int(args.threads),
            "cg_rtol": float(args.cg_rtol),
            "cg_maxiter": int(args.cg_maxiter),
            "logistic_maxiter": int(args.logistic_maxiter),
            "wall_seconds": float(time.perf_counter() - overall_started),
        },
        "created_or_modified_paths": created_paths,
        "limitations": [
            "one model and three frozen 16-window corpora",
            "eight FIT windows per corpus",
            "token-level linear gates only",
            "no model-quality or expert-viability claim",
        ],
    }
    report_path = output / "MOE_RT2_1_REPORT.md"
    write_text(report_path, render_report(analysis))
    gates["G3"]["report_sha256"] = sha256_file(report_path)
    gates["G3"]["report_bytes"] = int(report_path.stat().st_size)
    write_json(output / "analysis.json", analysis)
    print(
        json.dumps(
            {
                "status": "complete",
                "verdict": verdict_sentence,
                "qualifying_pairs": len(qualifying),
                "analysis": str(output / "analysis.json"),
                "report": str(report_path),
                "wall_seconds": analysis["runtime"]["wall_seconds"],
            }
        ),
        flush=True,
    )
    return 0 if all(gate["verdict"] == "GREEN" for gate in gates.values()) else 3


def run_verify(args: argparse.Namespace) -> int:
    output = ensure_output_dir(args.output_dir)
    analysis_path = output / "analysis.json"
    report_path = output / "MOE_RT2_1_REPORT.md"
    keys_path = output / "refined_keys.npz"
    required = [
        output / "preflight.json",
        output / "g1_math_truth.json",
        output / "cpu_validation.json",
        analysis_path,
        report_path,
        keys_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing output artifacts: {missing}")
    analysis = read_json(analysis_path)
    checks: list[dict[str, Any]] = []
    add_check(checks, "schema", "moe_rt2_1_analysis_v1", analysis.get("schema"))
    add_check(checks, "status", "complete", analysis.get("status"))
    add_check(checks, "layer_count", N_LAYERS, len(analysis.get("layers", [])))
    add_check(checks, "row_count", N_LAYERS * len(ARMS), len(analysis.get("rows", [])))
    add_check(
        checks,
        "operating_curve_count",
        N_LAYERS * len(ARMS) * len(TAU_GRID),
        sum(len(row["metrics"]["operating_curve"]) for row in analysis.get("rows", [])),
    )
    add_check(checks, "fit_eval_overlap", 0, analysis["split_provenance"]["fit_eval_overlap_count"])
    add_check(checks, "bootstrap_minimum", True, analysis["bootstrap_resamples"] >= 2000)
    add_check(checks, "rt2_analysis_hash_stable", analysis["g0"]["rt2_analysis_sha256_before"], sha256_file(RT2_ANALYSIS_PATH))
    add_check(checks, "keys_hash", analysis["keys_artifact"]["sha256"], sha256_file(keys_path))
    add_check(checks, "report_hash", analysis["gates"]["G3"]["report_sha256"], sha256_file(report_path))
    add_check(checks, "all_gates_green", True, all(analysis["gates"][gate]["verdict"] == "GREEN" for gate in ("G0", "G1", "G2", "G3")))
    with np.load(keys_path, allow_pickle=False) as keys:
        for arm in ARMS:
            add_check(checks, f"keys.{arm}.shape", [N_LAYERS, HIDDEN_DIM], list(keys[arm].shape))
            norms = np.linalg.norm(keys[arm].astype(np.float64), axis=1)
            add_check(checks, f"keys.{arm}.unit_norm", True, bool(np.all(np.abs(norms - 1.0) < 1.0e-5)))
    report = report_path.read_text(encoding="utf-8")
    add_check(checks, "report_contains_verdict", True, analysis["registered_verdict_sentence"] in report)
    add_check(checks, "report_contains_operating_curves", True, all(f"## {arm} full frozen-threshold EVAL operating curves" in report for arm in ARMS))
    failed = [item for item in checks if not item["passed"]]
    payload = {
        "schema": "moe_rt2_1_final_verification_v1",
        "created_at": now_iso(),
        "passed": not failed,
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "failed_check_names": [item["name"] for item in failed],
        "checks": checks,
    }
    # Verification is printed only so the promised artifact set remains sealed.
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0 if not failed else 4


def main() -> int:
    args = parse_args()
    try:
        if args.mode == "preflight":
            return run_preflight(args)
        if args.mode == "self-test":
            return run_self_test(args)
        if args.mode == "analyze":
            return run_analysis(args)
        if args.mode == "verify":
            return run_verify(args)
        raise AssertionError(args.mode)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
