#!/usr/bin/env python3
"""MOE-E3.1: test episode addressability with DOC-B as a hard negative.

This is a CPU-only analysis over the content-validated OLMoE router-input
captures produced by MOE-E3.  Per layer, it fits the registered closed-form K4
Fisher/LDA key and a trained-linear K5 logistic-probe ceiling.  DOC-B joins
WikiText, code, GRM, and guides in the negative pool for fitting, FIT-side
threshold calibration, and held-out evaluation.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import platform
import sys
import tempfile
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence


# This order is analysis-only and must not discover or initialize CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("PYTHONPYCACHEPREFIX", "/tmp/olmoe_e3_1_pycache")
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from threadpoolctl import threadpool_info, threadpool_limits  # noqa: E402

import olmoe_e3_experiment as e3  # noqa: E402


e2 = e3.e2
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
ORDER_PATH = REPO_ROOT / "orders" / "MOE_E3_1_EPISODE_KEY.md"
PARENT_ORDER_PATH = REPO_ROOT / "orders" / "MOE_E3_EPISODIC_CONSOLIDATION.md"
E3_OUTPUT = REPO_ROOT / "artifacts" / "moe_e3"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "moe_e3_1"

N_LAYERS = e3.N_LAYERS
N_WINDOWS = e3.N_KEY_WINDOWS
WINDOW_TOKENS = e3.WINDOW_TOKENS
HIDDEN_DIM = e3.HIDDEN_DIM
FIT_INDICES = e3.FIT_INDICES
EVAL_INDICES = e3.EVAL_INDICES
INNER_TRAIN_INDICES = e3.INNER_TRAIN_INDICES
INNER_VALIDATION_INDICES = e3.INNER_VALIDATION_INDICES

POSITIVE_CORPUS = "doc_a"
NEGATIVE_CORPORA = ("doc_b", "wikitext", "code", "grm", "guides")
ALL_CORPORA = (POSITIVE_CORPUS,) + NEGATIVE_CORPORA
ARMS = ("K4", "K5")
K4_ALPHA_GRID = e3.K4_ALPHA_GRID
K5_C_GRID = (1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0)
TAU_GRID = e3.TAU_GRID

RECALL_FLOOR = 0.50
FPR_CAPS = {"doc_b": 0.10, "guides": 0.10, "code": 0.05, "grm": 0.05}
DEFAULT_SEED = 20260829
DEFAULT_BOOTSTRAP_RESAMPLES = 2000

SENTENCE_EPISODE = "EPISODE-ADDRESSABLE"
SENTENCE_PROBE = "PROBE-ONLY"
SENTENCE_NOT_LINEAR = (
    "NOT LINEARLY EPISODE-ADDRESSABLE under these limits — router-geometry keys "
    "are domain-grade; episode selection belongs to the mount stage (GRM semantic routing)."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("validate-inputs", "self-test", "fit", "analyze", "run")
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--cg-rtol", type=float, default=1.0e-5)
    parser.add_argument("--cg-maxiter", type=int, default=200)
    parser.add_argument("--logistic-maxiter", type=int, default=500)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_output(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def input_binding(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": e3.MODEL_ID,
        "model_revision": e3.MODEL_REVISION,
        "prepared_windows_content_digest_algorithm": manifest["prepared_windows"][
            "content_digest_algorithm"
        ],
        "prepared_windows_content_sha256": manifest["prepared_windows"]["content_sha256"],
        "e2_negative_content_sha256": e3.E2_LEGACY_CONTENT_SHA256,
    }


def validate_inputs(args: argparse.Namespace) -> int:
    output = ensure_output(args.output_dir)
    manifest, arrays = e3.load_prepared(E3_OUTPUT)
    bringup = e3.require_bringup_green(E3_OUTPUT, manifest)
    provenance = e3.collect_capture_provenance(E3_OUTPUT, manifest, arrays)
    expected_count = len(ALL_CORPORA) * N_WINDOWS
    if len(provenance) != expected_count:
        raise RuntimeError(f"capture count {len(provenance)} != {expected_count}")

    e3_doc_receipts: list[dict[str, Any]] = []
    for corpus in ("doc_a", "doc_b"):
        for index in range(N_WINDOWS):
            _data_path, receipt_path = e3.key_capture_paths(E3_OUTPUT, corpus, index)
            receipt = e2.read_json(receipt_path)
            e3_doc_receipts.append(receipt)
    hooks = sorted({receipt.get("hook") for receipt in e3_doc_receipts})
    shapes = sorted({tuple(receipt.get("capture_shape", [])) for receipt in e3_doc_receipts})
    dtypes = sorted({receipt.get("capture_dtype") for receipt in e3_doc_receipts})
    device_requests = sorted(
        {
            str(receipt.get("invocation", {}).get("runtime", {}).get("device_request"))
            for receipt in e3_doc_receipts
        }
    )
    if hooks != ["OlmoeDecoderLayer.mlp forward_pre_hook; exact native router input"]:
        raise RuntimeError(f"DOC-A/DOC-B hook mismatch: {hooks}")
    if shapes != [(N_LAYERS, WINDOW_TOKENS, HIDDEN_DIM)] or dtypes != ["float16"]:
        raise RuntimeError(f"DOC-A/DOC-B capture machinery mismatch: {shapes}/{dtypes}")

    doc_b_rows = [row for row in provenance if row["corpus"] == "doc_b"]
    doc_a_rows = [row for row in provenance if row["corpus"] == "doc_a"]
    if len(doc_a_rows) != N_WINDOWS or len(doc_b_rows) != N_WINDOWS:
        raise RuntimeError("DOC-A/DOC-B does not contain exactly 16 validated captures")

    receipt = {
        "schema": "moe_e3_1_capture_validation_v1",
        "status": "passed",
        "created_at": now_iso(),
        "execution_scope": "CPU-only validation; no model inference performed",
        "reuse_mode": (
            "content-valid MOE-E3 DOC-A and DOC-B router-input payloads reused read-only; "
            "the parent capture command is idempotent"
        ),
        "same_machinery_and_provenance": True,
        "capture_contract": {
            "hook": hooks[0],
            "shape_per_window": list(shapes[0]),
            "dtype": dtypes[0],
            "doc_a_window_count": len(doc_a_rows),
            "doc_b_window_count": len(doc_b_rows),
            "doc_b_layer_count": N_LAYERS,
            "prior_capture_runtime_device_requests": device_requests,
            "e3_1_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "doc_b_source_windows": manifest["regions"]["doc_b_cross_probe_source_windows"],
        "doc_b_captures": doc_b_rows,
        "all_capture_count": len(provenance),
        "expected_capture_count": expected_count,
        "all_captures": provenance,
        "bringup": {
            "path": str(E3_OUTPUT / "bringup.json"),
            "sha256": e2.sha256_file(E3_OUTPUT / "bringup.json"),
            "verdict": bringup["gate"]["verdict"],
        },
        **input_binding(manifest),
        "source_manifest": {
            "path": str(E3_OUTPUT / "corpus_manifest.json"),
            "sha256": e2.sha256_file(E3_OUTPUT / "corpus_manifest.json"),
        },
        "source_parent_fit": {
            "path": str(E3_OUTPUT / "fit_key.json"),
            "sha256": e2.sha256_file(E3_OUTPUT / "fit_key.json"),
        },
        "script": str(SCRIPT_PATH),
        "script_sha256": e2.sha256_file(SCRIPT_PATH),
        "order": str(ORDER_PATH),
        "order_sha256": e2.sha256_file(ORDER_PATH),
    }
    path = output / "capture_validation.json"
    if path.is_file():
        existing = e2.read_json(path)
        stable_fields = (
            "schema",
            "status",
            "same_machinery_and_provenance",
            "capture_contract",
            "doc_b_source_windows",
            "doc_b_captures",
            "all_capture_count",
            "expected_capture_count",
            "all_captures",
            "bringup",
            "model_id",
            "model_revision",
            "prepared_windows_content_digest_algorithm",
            "prepared_windows_content_sha256",
            "e2_negative_content_sha256",
            "source_manifest",
            "source_parent_fit",
            "script",
            "script_sha256",
            "order",
            "order_sha256",
        )
        if all(existing.get(field) == receipt.get(field) for field in stable_fields):
            print(
                json.dumps(
                    {
                        "status": "existing_valid",
                        "doc_b_windows": len(doc_b_rows),
                        "layers": N_LAYERS,
                        "receipt": str(path),
                    }
                ),
                flush=True,
            )
            return 0
    e2.write_json(path, receipt)
    print(
        json.dumps(
            {
                "status": "passed",
                "doc_b_windows": len(doc_b_rows),
                "layers": N_LAYERS,
                "receipt": str(path),
            }
        ),
        flush=True,
    )
    return 0


def selected_tokens(array: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    return e3.selected_tokens(array, indices)


def centered_six_corpora(
    hidden: dict[str, np.ndarray], indices: Sequence[int]
) -> tuple[np.ndarray, int, np.ndarray, float]:
    blocks = {corpus: selected_tokens(hidden[corpus], indices) for corpus in ALL_CORPORA}
    counts = {corpus: int(block.shape[0]) for corpus, block in blocks.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"K4 requires equal corpus token counts: {counts}")
    means = {
        corpus: block.mean(axis=0, dtype=np.float64) for corpus, block in blocks.items()
    }
    centered = np.concatenate(
        [
            np.ascontiguousarray(
                blocks[corpus] - means[corpus].astype(np.float32), dtype=np.float32
            )
            for corpus in ALL_CORPORA
        ],
        axis=0,
    )
    dof = int(centered.shape[0] - len(ALL_CORPORA))
    diagonal = np.sum(centered.astype(np.float64) ** 2, axis=0) / float(dof)
    negative_mean = np.mean(
        np.stack([means[corpus] for corpus in NEGATIVE_CORPORA]), axis=0
    )
    delta = means[POSITIVE_CORPUS] - negative_mean
    return centered, dof, delta, float(diagonal.mean())


def validation_auc(hidden: dict[str, np.ndarray], key: np.ndarray) -> dict[str, Any]:
    positive = selected_tokens(hidden[POSITIVE_CORPUS], INNER_VALIDATION_INDICES) @ key
    aucs = {
        corpus: e2.exact_auc(
            positive, selected_tokens(hidden[corpus], INNER_VALIDATION_INDICES) @ key
        )
        for corpus in NEGATIVE_CORPORA
    }
    return {
        **{f"auc_doc_a_vs_{corpus}": value for corpus, value in aucs.items()},
        "minimum_auc": min(aucs.values()),
        "mean_auc": float(np.mean(list(aucs.values()))),
    }


def choose_validation_candidate(
    candidates: list[dict[str, Any]], stronger: Callable[[dict[str, Any]], float]
) -> dict[str, Any]:
    converged = [item for item in candidates if item["fit_converged"]]
    if not converged:
        raise RuntimeError("no converged hyperparameter candidate")
    return max(
        converged,
        key=lambda item: (
            item["validation"]["minimum_auc"],
            item["validation"]["mean_auc"],
            stronger(item),
        ),
    )


def fit_k4_direction(
    hidden: dict[str, np.ndarray], *, cg_rtol: float, cg_maxiter: int
) -> tuple[np.ndarray, dict[str, Any]]:
    centered, dof, delta, mean_variance = centered_six_corpora(
        hidden, INNER_TRAIN_INDICES
    )
    candidates_by_alpha: dict[float, dict[str, Any]] = {}
    warm: np.ndarray | None = None
    for alpha in reversed(K4_ALPHA_GRID):
        regularization = float(alpha * mean_variance)
        solution, solver = e2.cg_fisher_solve(
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
            "alpha": float(alpha),
            "lambda": regularization,
            "fit": solver,
            "fit_converged": bool(solver["converged"]),
        }
        if solver["converged"]:
            key, norm = e2.unit_vector(solution, name=f"E3.1 K4 alpha={alpha}")
            candidate["direction_source_norm"] = norm
            candidate["validation"] = validation_auc(hidden, key)
            warm = solution
        else:
            warm = None
        candidates_by_alpha[alpha] = candidate
    del centered
    gc.collect()
    candidates = [candidates_by_alpha[alpha] for alpha in K4_ALPHA_GRID]
    selected = choose_validation_candidate(candidates, stronger=lambda item: float(item["alpha"]))
    selected_alpha = float(selected["alpha"])

    centered_full, dof_full, delta_full, variance_full = centered_six_corpora(
        hidden, FIT_INDICES
    )
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
        raise RuntimeError("E3.1 K4 full-FIT solve did not converge")
    key, source_norm = e2.unit_vector(solution, name="E3.1 K4 final direction")
    return key, {
        "label": "closed-form K4 Fisher/LDA key",
        "construction": (
            "unit_norm((pooled within-six-corpus FIT covariance + lambda I)^-1 "
            "(mean(DOC-A_FIT) - equal_mean(DOC-B,wikitext,code,grm,guides)_FIT))"
        ),
        "negative_pool_weighting": "equal tokens and one-fifth mean weight per negative corpus",
        "alpha_grid": list(K4_ALPHA_GRID),
        "inner_train_window_indices": list(INNER_TRAIN_INDICES),
        "inner_validation_window_indices": list(INNER_VALIDATION_INDICES),
        "selection_rule": (
            "highest minimum validation AUC over five negatives, then mean AUC, "
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
        "key_sha256": e2.sha256_array(key),
        "selected_validation": selected["validation"],
    }


def logistic_dataset(
    hidden: dict[str, np.ndarray], indices: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    positive = selected_tokens(hidden[POSITIVE_CORPUS], indices)
    negatives = [selected_tokens(hidden[corpus], indices) for corpus in NEGATIVE_CORPORA]
    x = np.concatenate([positive, *negatives], axis=0)
    y = np.concatenate(
        [
            np.ones(positive.shape[0], dtype=np.int8),
            np.zeros(sum(block.shape[0] for block in negatives), dtype=np.int8),
        ]
    )
    return np.ascontiguousarray(x, dtype=np.float32), y


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    c_value: float,
    maxiter: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    model = LogisticRegression(
        C=float(c_value),
        penalty="l2",
        solver="lbfgs",
        fit_intercept=True,
        class_weight="balanced",
        max_iter=int(maxiter),
        tol=1.0e-5,
        random_state=int(seed),
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
    key, source_norm = e2.unit_vector(
        coefficient, name=f"E3.1 K5 logistic C={c_value} coefficient"
    )
    return key, {
        "C": float(c_value),
        "solver": "sklearn LogisticRegression(lbfgs)",
        "penalty": "l2",
        "class_weight": "balanced",
        "fit_intercept": True,
        "raw_intercept": float(model.intercept_[0]),
        "normalized_intercept": float(model.intercept_[0] / source_norm),
        "score_used_for_gate": (
            "h @ unit_norm(coef); FIT quantile calibration absorbs the omitted additive intercept"
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


def fit_k5_direction(
    hidden: dict[str, np.ndarray], *, logistic_maxiter: int, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    train_x, train_y = logistic_dataset(hidden, INNER_TRAIN_INDICES)
    candidates: list[dict[str, Any]] = []
    for c_value in K5_C_GRID:
        key, fit_metadata = fit_logistic(
            train_x,
            train_y,
            c_value=c_value,
            maxiter=logistic_maxiter,
            seed=seed,
        )
        candidate: dict[str, Any] = {
            "C": float(c_value),
            "fit": fit_metadata,
            "fit_converged": bool(fit_metadata["converged"]),
        }
        if fit_metadata["converged"]:
            candidate["validation"] = validation_auc(hidden, key)
        candidates.append(candidate)
    del train_x, train_y
    gc.collect()
    selected = choose_validation_candidate(candidates, stronger=lambda item: -float(item["C"]))
    selected_c = float(selected["C"])

    full_x, full_y = logistic_dataset(hidden, FIT_INDICES)
    key, final_fit = fit_logistic(
        full_x,
        full_y,
        c_value=selected_c,
        maxiter=logistic_maxiter,
        seed=seed,
    )
    if not final_fit["converged"]:
        key, final_fit = fit_logistic(
            full_x,
            full_y,
            c_value=selected_c,
            maxiter=logistic_maxiter * 4,
            seed=seed,
        )
    del full_x, full_y
    gc.collect()
    if not final_fit["converged"]:
        raise RuntimeError(f"E3.1 K5 full-FIT fit did not converge at C={selected_c}")
    return key, {
        "label": "trained linear logistic probe",
        "interpretation": "linear ceiling arm; not a closed-form router-geometry key",
        "construction": (
            "balanced L2 logistic regression on raw router inputs; DOC-A=1 and equal-token "
            "pooled DOC-B+wikitext+code+grm+guides=0; coefficient unit-normalized"
        ),
        "feature_standardization": "none (router inputs are post-normalization states)",
        "C_grid": list(K5_C_GRID),
        "inner_train_window_indices": list(INNER_TRAIN_INDICES),
        "inner_validation_window_indices": list(INNER_VALIDATION_INDICES),
        "selection_rule": (
            "highest minimum validation AUC over five negatives, then mean AUC, then smaller C"
        ),
        "validation_candidates": candidates,
        "selected_C": selected_c,
        "final_fit": final_fit,
        "stored_norm": float(np.linalg.norm(key.astype(np.float64))),
        "key_sha256": e2.sha256_array(key),
        "selected_validation": selected["validation"],
    }


def score_hidden(hidden: dict[str, np.ndarray], key: np.ndarray) -> dict[str, np.ndarray]:
    return e3.score_hidden(hidden, key)


def select_tau(scores: dict[str, np.ndarray]) -> dict[str, Any]:
    fit = np.asarray(FIT_INDICES, dtype=np.int64)
    pooled = np.concatenate(
        [scores[corpus][fit].reshape(-1) for corpus in NEGATIVE_CORPORA]
    )
    curve: list[dict[str, Any]] = []
    for label, quantile in TAU_GRID:
        tau = float(np.quantile(pooled, quantile, method="linear"))
        rates = {
            "doc_a_recall": float(np.mean(scores[POSITIVE_CORPUS][fit] >= tau)),
            **{
                f"{corpus}_fpr": float(np.mean(scores[corpus][fit] >= tau))
                for corpus in NEGATIVE_CORPORA
            },
        }
        feasible = all(rates[f"{corpus}_fpr"] <= cap for corpus, cap in FPR_CAPS.items())
        curve.append(
            {
                "quantile_label": label,
                "quantile": float(quantile),
                "tau": tau,
                "fit": rates,
                "fit_registered_caps_feasible": bool(feasible),
            }
        )
    feasible_rows = [row for row in curve if row["fit_registered_caps_feasible"]]
    pool = feasible_rows if feasible_rows else curve

    def selection_rank(row: dict[str, Any]) -> tuple[float, ...]:
        fit_rates = row["fit"]
        if feasible_rows:
            primary = fit_rates["doc_a_recall"]
        else:
            primary = -max(
                fit_rates[f"{corpus}_fpr"] / cap for corpus, cap in FPR_CAPS.items()
            )
        return (
            float(primary),
            -float(fit_rates["doc_b_fpr"]),
            -float(fit_rates["guides_fpr"]),
            -float(fit_rates["code_fpr"]),
            -float(fit_rates["grm_fpr"]),
            -float(fit_rates["wikitext_fpr"]),
            float(row["quantile"]),
        )

    selected = max(pool, key=selection_rank)
    return {
        "tau_definition": (
            "linear p90/p95/p99/p99.5 over pooled equal-token DOC-B+wikitext+code+grm+guides "
            "FIT scores; maximize DOC-A FIT recall subject to registered capped-negative FIT "
            "rates; if none is feasible minimize worst normalized FIT violation"
        ),
        "fit_constraint_corpora": list(FPR_CAPS),
        "wikitext_fire_rate": "descriptive_no_pass_fail_bound",
        "fire_comparison": "score >= tau",
        "fit_negative_token_count": int(pooled.size),
        "fit_registered_caps_feasible": bool(feasible_rows),
        "selected_quantile_label": selected["quantile_label"],
        "selected_quantile": selected["quantile"],
        "selected_tau": selected["tau"],
        "selected_fit_rates": selected["fit"],
        "operating_curve": curve,
    }


def registered_qualifies(values: dict[str, float]) -> bool:
    return bool(
        values["doc_a_recall"] >= RECALL_FLOOR
        and values["doc_b_fpr"] <= FPR_CAPS["doc_b"]
        and values["guides_fpr"] <= FPR_CAPS["guides"]
        and values["code_fpr"] <= FPR_CAPS["code"]
        and values["grm_fpr"] <= FPR_CAPS["grm"]
    )


def analyze_scores(
    scores: dict[str, np.ndarray], *, resamples: int, seed: int
) -> dict[str, Any]:
    threshold = select_tau(scores)
    tau = float(threshold["selected_tau"])
    evaluation = np.asarray(EVAL_INDICES, dtype=np.int64)
    fired = {corpus: scores[corpus][evaluation] >= tau for corpus in ALL_CORPORA}
    eval_metrics: dict[str, Any] = {
        "doc_a_recall": e2.bootstrap_rate(
            fired["doc_a"], resamples=resamples, seed=seed + 1, window_indices=EVAL_INDICES
        )
    }
    for index, corpus in enumerate(NEGATIVE_CORPORA):
        eval_metrics[f"{corpus}_fpr"] = e2.bootstrap_rate(
            fired[corpus],
            resamples=resamples,
            seed=seed + 10 + index,
            window_indices=EVAL_INDICES,
        )
    eval_metrics["auc"] = {
        corpus: e2.exact_auc(
            scores["doc_a"][evaluation], scores[corpus][evaluation]
        )
        for corpus in NEGATIVE_CORPORA
    }
    values = {
        "doc_a_recall": eval_metrics["doc_a_recall"]["value"],
        **{
            f"{corpus}_fpr": eval_metrics[f"{corpus}_fpr"]["value"]
            for corpus in NEGATIVE_CORPORA
        },
    }
    qualifies = registered_qualifies(values)
    return {
        **threshold,
        "eval_window_indices": {corpus: list(EVAL_INDICES) for corpus in ALL_CORPORA},
        "eval": eval_metrics,
        "qualifies": qualifies,
        "registered_criteria": {
            "eval_doc_a_recall_gte": RECALL_FLOOR,
            "eval_doc_b_fire_lte": FPR_CAPS["doc_b"],
            "eval_guides_fpr_lte": FPR_CAPS["guides"],
            "eval_code_fpr_lte": FPR_CAPS["code"],
            "eval_grm_fpr_lte": FPR_CAPS["grm"],
            "eval_wikitext_fire": "descriptive",
            "qualification_is_exactly_the_five_registered_eval_inequalities": True,
            "fit_threshold_frozen_before_eval": True,
        },
    }


def fit_side_layer_rank(row: dict[str, Any]) -> tuple[float, ...]:
    fit = row["metrics"]["selected_fit_rates"]
    validation = row["key_metadata"]["selected_validation"]
    worst = max(fit[f"{corpus}_fpr"] / cap for corpus, cap in FPR_CAPS.items())
    return (
        float(row["metrics"]["fit_registered_caps_feasible"]),
        float(validation["minimum_auc"]),
        float(validation["mean_auc"]),
        float(fit["doc_a_recall"]),
        -float(worst),
        -float(row["layer"]),
    )


def split_provenance() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    eval_set = set(EVAL_INDICES)
    for arm in ARMS:
        for corpus in ALL_CORPORA:
            for phase, indices in (
                ("hyperparameter_train", INNER_TRAIN_INDICES),
                ("hyperparameter_validation", INNER_VALIDATION_INDICES),
                ("final_refit", FIT_INDICES),
                ("tau_fit", FIT_INDICES),
            ):
                overlap = sorted(set(indices).intersection(eval_set))
                checks.append(
                    {
                        "arm": arm,
                        "corpus": corpus,
                        "phase": phase,
                        "fit_indices": list(indices),
                        "eval_indices": list(EVAL_INDICES),
                        "overlap": overlap,
                    }
                )
    return {
        "schema": "moe_e3_1_split_provenance_v1",
        "window_index_basis": "zero-based local index within each 16-window corpus capture",
        "all_corpora": list(ALL_CORPORA),
        "negative_corpora": list(NEGATIVE_CORPORA),
        "fit_indices": list(FIT_INDICES),
        "eval_indices": list(EVAL_INDICES),
        "inner_train_indices": list(INNER_TRAIN_INDICES),
        "inner_validation_indices": list(INNER_VALIDATION_INDICES),
        "fit_eval_parity": {corpus: {"fit": 8, "eval": 8} for corpus in ALL_CORPORA},
        "tokens_per_corpus_fit": len(FIT_INDICES) * WINDOW_TOKENS,
        "tokens_per_corpus_eval": len(EVAL_INDICES) * WINDOW_TOKENS,
        "fit_eval_overlap_checks": checks,
        "fit_eval_overlap_count": sum(len(item["overlap"]) for item in checks),
        "eval_used_for_key_tau_or_hyperparameter_selection": False,
    }


def summarize_row(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row["metrics"]
    evaluated = metrics["eval"]
    metadata = row["key_metadata"]
    hyperparameter = (
        {"alpha": metadata["selected_alpha"], "lambda": metadata["selected_full_fit_lambda"]}
        if row["arm"] == "K4"
        else {"C": metadata["selected_C"]}
    )
    return {
        "arm": row["arm"],
        "layer": row["layer"],
        "hyperparameter": hyperparameter,
        "tau": metrics["selected_tau"],
        "tau_quantile": metrics["selected_quantile_label"],
        "fit_caps_feasible": metrics["fit_registered_caps_feasible"],
        "doc_a_recall": evaluated["doc_a_recall"]["value"],
        "doc_b_fire": evaluated["doc_b_fpr"]["value"],
        "guides_fpr": evaluated["guides_fpr"]["value"],
        "code_fpr": evaluated["code_fpr"]["value"],
        "grm_fpr": evaluated["grm_fpr"]["value"],
        "wikitext_fire_descriptive": evaluated["wikitext_fpr"]["value"],
        "qualifies": metrics["qualifies"],
        "fit_side_rank": row["fit_side_rank"],
    }


def validate_fit_receipt(output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    fit_path = output / "fit_keys.json"
    keys_path = output / "all_layer_linear_keys.npz"
    capture_path = output / "capture_validation.json"
    receipt = e2.read_json(fit_path)
    if (
        receipt.get("schema") != "moe_e3_1_fit_keys_v1"
        or receipt.get("status") != "complete"
        or receipt.get("prepared_windows_content_sha256")
        != manifest["prepared_windows"]["content_sha256"]
        or receipt.get("script_sha256") != e2.sha256_file(SCRIPT_PATH)
        or receipt.get("order_sha256") != e2.sha256_file(ORDER_PATH)
        or receipt.get("input_integrity", {}).get("capture_validation_sha256")
        != e2.sha256_file(capture_path)
        or receipt.get("keys_artifact", {}).get("sha256") != e2.sha256_file(keys_path)
        or len(receipt.get("rows", [])) != N_LAYERS * len(ARMS)
    ):
        raise RuntimeError("E3.1 fit receipt contract failed")
    with np.load(keys_path, allow_pickle=False) as archive:
        required = {
            "K4",
            "K5",
            "tau_K4",
            "tau_K5",
            "K4_selected_alpha",
            "K4_selected_lambda",
            "K5_selected_C",
        }
        if set(archive.files) != required:
            raise RuntimeError(f"E3.1 keys fields differ: {sorted(archive.files)}")
        if archive["K4"].shape != (N_LAYERS, HIDDEN_DIM) or archive["K5"].shape != (
            N_LAYERS,
            HIDDEN_DIM,
        ):
            raise RuntimeError("E3.1 key shape contract failed")
        if not all(np.isfinite(archive[name]).all() for name in archive.files):
            raise RuntimeError("E3.1 keys artifact contains non-finite values")
    return receipt


def fit(args: argparse.Namespace) -> int:
    if args.bootstrap_resamples < 2000:
        raise ValueError("E3.1 requires at least 2,000 window bootstrap resamples")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.logistic_maxiter < 1 or args.cg_maxiter < 1:
        raise ValueError("solver iteration limits must be positive")
    output = ensure_output(args.output_dir)
    manifest, _arrays = e3.load_prepared(E3_OUTPUT)
    capture_path = output / "capture_validation.json"
    if not capture_path.is_file():
        raise RuntimeError("run validate-inputs before fit")
    capture = e2.read_json(capture_path)
    if (
        capture.get("status") != "passed"
        or capture.get("prepared_windows_content_sha256")
        != manifest["prepared_windows"]["content_sha256"]
    ):
        raise RuntimeError("capture validation is not content-valid")
    fit_path = output / "fit_keys.json"
    if fit_path.is_file():
        existing = validate_fit_receipt(output, manifest)
        print(
            json.dumps(
                {
                    "status": "existing_valid",
                    "sentence": existing["decision"]["registered_sentence"],
                    "fit": str(fit_path),
                }
            ),
            flush=True,
        )
        return 0

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    key_store: dict[str, list[np.ndarray]] = {arm: [] for arm in ARMS}
    tau_store: dict[str, list[float]] = {arm: [] for arm in ARMS}
    alphas: list[float] = []
    lambdas: list[float] = []
    c_values: list[float] = []
    with threadpool_limits(limits=args.threads):
        for layer_index in range(N_LAYERS):
            layer_started = time.perf_counter()
            hidden = {
                corpus: e3.materialize_layer(E3_OUTPUT, corpus, layer_index)
                for corpus in ALL_CORPORA
            }
            k4_started = time.perf_counter()
            k4, k4_metadata = fit_k4_direction(
                hidden, cg_rtol=args.cg_rtol, cg_maxiter=args.cg_maxiter
            )
            k4_metrics = analyze_scores(
                score_hidden(hidden, k4),
                resamples=args.bootstrap_resamples,
                seed=args.seed + 10000 * layer_index + 100,
            )
            k4_row = {
                "arm": "K4",
                "layer": layer_index,
                "key_metadata": k4_metadata,
                "metrics": k4_metrics,
                "wall_seconds": time.perf_counter() - k4_started,
            }
            k4_row["fit_side_rank"] = list(fit_side_layer_rank(k4_row))
            rows.append(k4_row)
            key_store["K4"].append(k4)
            tau_store["K4"].append(float(k4_metrics["selected_tau"]))
            alphas.append(float(k4_metadata["selected_alpha"]))
            lambdas.append(float(k4_metadata["selected_full_fit_lambda"]))
            print(json.dumps({"layer": layer_index, **summarize_row(k4_row)}), flush=True)

            k5_started = time.perf_counter()
            k5, k5_metadata = fit_k5_direction(
                hidden,
                logistic_maxiter=args.logistic_maxiter,
                seed=args.seed + layer_index,
            )
            k5_metrics = analyze_scores(
                score_hidden(hidden, k5),
                resamples=args.bootstrap_resamples,
                seed=args.seed + 10000 * layer_index + 500,
            )
            k5_row = {
                "arm": "K5",
                "layer": layer_index,
                "key_metadata": k5_metadata,
                "metrics": k5_metrics,
                "wall_seconds": time.perf_counter() - k5_started,
            }
            k5_row["fit_side_rank"] = list(fit_side_layer_rank(k5_row))
            rows.append(k5_row)
            key_store["K5"].append(k5)
            tau_store["K5"].append(float(k5_metrics["selected_tau"]))
            c_values.append(float(k5_metadata["selected_C"]))
            print(json.dumps({"layer": layer_index, **summarize_row(k5_row)}), flush=True)
            print(
                json.dumps(
                    {
                        "layer_complete": layer_index,
                        "wall_seconds": time.perf_counter() - layer_started,
                    }
                ),
                flush=True,
            )
            del hidden, k4, k5
            gc.collect()

    keys_path = output / "all_layer_linear_keys.npz"
    e2.save_npz(
        keys_path,
        K4=np.stack(key_store["K4"]).astype(np.float32),
        K5=np.stack(key_store["K5"]).astype(np.float32),
        tau_K4=np.asarray(tau_store["K4"], dtype=np.float64),
        tau_K5=np.asarray(tau_store["K5"], dtype=np.float64),
        K4_selected_alpha=np.asarray(alphas, dtype=np.float64),
        K4_selected_lambda=np.asarray(lambdas, dtype=np.float64),
        K5_selected_C=np.asarray(c_values, dtype=np.float64),
    )

    arm_decisions: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        fit_ranked = sorted(arm_rows, key=fit_side_layer_rank, reverse=True)
        qualifiers = sorted(
            [row for row in arm_rows if row["metrics"]["qualifies"]],
            key=fit_side_layer_rank,
            reverse=True,
        )
        arm_decisions[arm] = {
            "qualifying_layer_count": len(qualifiers),
            "qualifying_layers_fit_ranked": [row["layer"] for row in qualifiers],
            "selected_qualifying_layer": qualifiers[0]["layer"] if qualifiers else None,
            "selected_qualifying_row": summarize_row(qualifiers[0]) if qualifiers else None,
            "carried_fit_ranked_layer": fit_ranked[0]["layer"],
            "carried_fit_ranked_row": summarize_row(fit_ranked[0]),
            "within_qualifiers_selection_rule": (
                "FIT-only rank: FIT cap feasibility, inner-validation minimum/mean AUC, "
                "FIT recall, lower worst normalized capped-negative FIT rate, lower layer"
            ),
        }
    if arm_decisions["K4"]["qualifying_layer_count"]:
        sentence = SENTENCE_EPISODE
    elif arm_decisions["K5"]["qualifying_layer_count"]:
        sentence = SENTENCE_PROBE
    else:
        sentence = SENTENCE_NOT_LINEAR
    decision = {
        "registered_sentence": sentence,
        "registered_sentence_class": (
            "EPISODE-ADDRESSABLE"
            if sentence == SENTENCE_EPISODE
            else "PROBE-ONLY"
            if sentence == SENTENCE_PROBE
            else "NOT LINEARLY EPISODE-ADDRESSABLE under these limits"
        ),
        "K4": arm_decisions["K4"],
        "K5": arm_decisions["K5"],
    }
    split = split_provenance()
    if split["fit_eval_overlap_count"] != 0:
        raise RuntimeError("FIT/EVAL split leakage detected")
    receipt = {
        "schema": "moe_e3_1_fit_keys_v1",
        "status": "complete",
        "created_at": now_iso(),
        "gate": "E3.1-G2",
        **input_binding(manifest),
        "seed": args.seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "cpu_only": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "model_inference": False,
            "analysis_device": "CPU",
        },
        "input_integrity": {
            "capture_validation": str(capture_path),
            "capture_validation_sha256": e2.sha256_file(capture_path),
            "capture_count": capture["all_capture_count"],
            "expected_capture_count": capture["expected_capture_count"],
        },
        "split_provenance": split,
        "registered_rule": {
            "eval_doc_a_recall_gte": RECALL_FLOOR,
            "eval_doc_b_fire_lte": FPR_CAPS["doc_b"],
            "eval_guides_fpr_lte": FPR_CAPS["guides"],
            "eval_code_fpr_lte": FPR_CAPS["code"],
            "eval_grm_fpr_lte": FPR_CAPS["grm"],
            "wikitext_fire": "descriptive",
            "qualification_is_exactly_the_registered_eval_rule": True,
            "tau_candidate_pool": list(NEGATIVE_CORPORA),
            "tau_grid": [{"label": label, "quantile": value} for label, value in TAU_GRID],
            "frozen_fit_side_tau": True,
        },
        "arms": {
            "K4": "closed-form Fisher/LDA router-geometry key",
            "K5": "trained linear logistic probe ceiling",
        },
        "rows": rows,
        "decision": decision,
        "keys_artifact": {"path": str(keys_path), "sha256": e2.sha256_file(keys_path)},
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "threads": args.threads,
            "threadpools": threadpool_info(),
            "cg_rtol": args.cg_rtol,
            "cg_maxiter": args.cg_maxiter,
            "logistic_maxiter": args.logistic_maxiter,
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_bytes": e2.peak_rss_bytes(),
        },
        "script": str(SCRIPT_PATH),
        "script_sha256": e2.sha256_file(SCRIPT_PATH),
        "order": str(ORDER_PATH),
        "order_sha256": e2.sha256_file(ORDER_PATH),
        "parent_order": str(PARENT_ORDER_PATH),
        "parent_order_sha256": e2.sha256_file(PARENT_ORDER_PATH),
    }
    e2.write_json(fit_path, receipt)
    print(
        json.dumps(
            {
                "status": "complete",
                "sentence": sentence,
                "K4_qualifiers": arm_decisions["K4"]["qualifying_layers_fit_ranked"],
                "K5_qualifiers": arm_decisions["K5"]["qualifying_layers_fit_ranked"],
                "fit": str(fit_path),
            }
        ),
        flush=True,
    )
    return 0


def rate_cell(metric: dict[str, Any]) -> str:
    return (
        f"{metric['value']:.6f} [{metric['ci95_low']:.6f}, {metric['ci95_high']:.6f}]"
    )


def row_table_line(row: dict[str, Any]) -> str:
    metrics = row["metrics"]
    evaluated = metrics["eval"]
    metadata = row["key_metadata"]
    hyperparameter = (
        f"alpha={metadata['selected_alpha']:.4g}"
        if row["arm"] == "K4"
        else f"C={metadata['selected_C']:.4g}"
    )
    return (
        f"| {row['layer']} | {row['arm']} | {hyperparameter} | "
        f"{metrics['selected_quantile_label']} / {metrics['selected_tau']:.8g} | "
        f"{rate_cell(evaluated['doc_a_recall'])} | "
        f"{rate_cell(evaluated['doc_b_fpr'])} | "
        f"{rate_cell(evaluated['guides_fpr'])} | "
        f"{rate_cell(evaluated['code_fpr'])} | "
        f"{rate_cell(evaluated['grm_fpr'])} | "
        f"{rate_cell(evaluated['wikitext_fpr'])} | "
        f"{'YES' if metrics['qualifies'] else 'NO'} |"
    )


def best_table_line(label: str, row: dict[str, Any] | None) -> str:
    if row is None:
        return f"| {label} | NONE | — | — | — | — | — | — | — | — |"
    return (
        f"| {label} | {row['layer']} | {row['doc_a_recall']:.6f} | "
        f"{row['doc_b_fire']:.6f} | {row['guides_fpr']:.6f} | {row['code_fpr']:.6f} | "
        f"{row['grm_fpr']:.6f} | {row['wikitext_fire_descriptive']:.6f} | "
        f"{row['tau_quantile']} / {row['tau']:.8g} | {'YES' if row['qualifies'] else 'NO'} |"
    )


def analyze(args: argparse.Namespace) -> int:
    output = ensure_output(args.output_dir)
    manifest, _arrays = e3.load_prepared(E3_OUTPUT)
    fit_receipt = validate_fit_receipt(output, manifest)
    decision = fit_receipt["decision"]
    rows = fit_receipt["rows"]
    capture = e2.read_json(output / "capture_validation.json")
    cpu_validation_path = output / "cpu_validation.json"
    cpu_validation = e2.read_json(cpu_validation_path)
    if cpu_validation.get("status") != "passed":
        raise RuntimeError("E3.1 CPU validation is not passed")

    best_rows = {
        arm: {
            "selected_qualifying": decision[arm]["selected_qualifying_row"],
            "carried_fit_ranked": decision[arm]["carried_fit_ranked_row"],
        }
        for arm in ARMS
    }
    analysis_path = output / "analysis.json"
    analysis = {
        "schema": "moe_e3_1_analysis_v1",
        "status": "complete",
        "created_at": now_iso(),
        "gate": "E3.1-G2",
        "registered_sentence": decision["registered_sentence"],
        "registered_sentence_class": decision["registered_sentence_class"],
        "qualifying_layers": {
            arm: decision[arm]["qualifying_layers_fit_ranked"] for arm in ARMS
        },
        "best_rows": best_rows,
        "documents": manifest["documents"],
        "capture": {
            "receipt": str(output / "capture_validation.json"),
            "receipt_sha256": e2.sha256_file(output / "capture_validation.json"),
            "doc_b_windows": capture["capture_contract"]["doc_b_window_count"],
            "doc_b_layers": capture["capture_contract"]["doc_b_layer_count"],
            "reuse_mode": capture["reuse_mode"],
        },
        "fit": {"path": str(output / "fit_keys.json"), "sha256": e2.sha256_file(output / "fit_keys.json")},
        "keys": fit_receipt["keys_artifact"],
        "cpu_validation": {
            "path": str(cpu_validation_path),
            "sha256": e2.sha256_file(cpu_validation_path),
            "checks_passed": cpu_validation["checks_passed"],
            "checks_total": cpu_validation["checks_total"],
        },
        "anything_not_done": [],
        "execution_notes": [
            (
                "Fresh DOC-B inference was not repeated: MOE-E3 already held all 16 exact router-input "
                "payloads used for its descriptive cross-probe, and E3.1 revalidated every payload and "
                "receipt before read-only reuse."
            ),
            "All E3.1 fitting, bootstrap evaluation, analysis, and reporting ran CPU-only.",
        ],
        "limitations": [
            "One frozen OLMoE revision, one registered DOC-A/DOC-B pair, and 8 held-out windows per corpus.",
            "K5 is a trained linear ceiling, not a deployable closed-form router-geometry key.",
            "Finite held-out rates do not establish a universal law; the registered sentence is scoped to these limits.",
            "Novel text is not excerpted; only provenance identities and aggregate metrics are reported.",
        ],
        **input_binding(manifest),
        "script": str(SCRIPT_PATH),
        "script_sha256": e2.sha256_file(SCRIPT_PATH),
        "order": str(ORDER_PATH),
        "order_sha256": e2.sha256_file(ORDER_PATH),
    }
    e2.write_json(analysis_path, analysis)

    doc_a = manifest["documents"]["doc_a"]
    doc_b = manifest["documents"]["doc_b"]
    lines = [
        "# MOE-E3.1 episode-addressability report",
        "",
        "## Registered sentence",
        "",
        decision["registered_sentence"],
        "",
        f"K4 qualifying layers: `{decision['K4']['qualifying_layers_fit_ranked']}`.  ",
        f"K5 qualifying layers: `{decision['K5']['qualifying_layers_fit_ranked']}`.",
        "",
        "K4 is the registered closed-form Fisher/LDA router-geometry key. K5 is explicitly a trained linear logistic-probe ceiling.",
        "",
        "## Registered E3.1-G2 rule",
        "",
        "A layer qualifies iff DOC-A EVAL recall is at least 0.50, DOC-B fire is at most 0.10, guides FPR is at most 0.10, code FPR is at most 0.05, and GRM FPR is at most 0.05. WikiText fire is descriptive.",
        "",
        "All corpora use local even windows for FIT and local odd windows for EVAL (8 windows / 4,096 tokens on each side). K4 alpha and K5 C use FIT-internal windows 0,4,8,12 for training and 2,6,10,14 for validation. Tau is selected on FIT scores only and frozen before EVAL.",
        "",
        "## Registered documents",
        "",
        "| Role | Path | Bytes | SHA-256 | Token count | Full 512-token windows |",
        "|---|---|---:|---|---:|---:|",
        f"| DOC-A | `{doc_a['path']}` | {doc_a['byte_count']} | `{doc_a['sha256']}` | {doc_a['token_count']} | {doc_a['full_512_window_count']} |",
        f"| DOC-B | `{doc_b['path']}` | {doc_b['byte_count']} | `{doc_b['sha256']}` | {doc_b['token_count']} | {doc_b['full_512_window_count']} |",
        "",
        "No novel excerpt is included. Token counts are from the frozen local OLMoE tokenizer with `add_special_tokens=false`.",
        "",
        "## DOC-B capture receipt",
        "",
        f"Validated `{capture['capture_contract']['doc_b_window_count']}` DOC-B window payloads, each carrying all `{capture['capture_contract']['doc_b_layer_count']}` layers with shape `{capture['capture_contract']['shape_per_window']}` and dtype `{capture['capture_contract']['dtype']}`. DOC-A and DOC-B use the same `{capture['capture_contract']['hook']}` hook and content-addressed model/input provenance.",
        "",
        "The payloads already existed from MOE-E3's descriptive DOC-B cross-probe. The CPU-only E3.1 command revalidated and reused them; it did not repeat identical model inference. The original capture receipts record their original runtime placement, while every new E3.1 fit and report stage records `CUDA_VISIBLE_DEVICES=\"\"`.",
        "",
        "## Best rows",
        "",
        "A selected qualifying row is chosen only among registered qualifiers using the carried FIT-only rank. When an arm has no qualifier, the separate fit-ranked carry is reported without relabeling it as passing.",
        "",
        "| Row | Layer | DOC-A recall | DOC-B fire | Guides FPR | Code FPR | GRM FPR | WikiText fire (desc.) | tau | Qualifies |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for arm in ARMS:
        lines.append(best_table_line(f"{arm} selected qualifying", best_rows[arm]["selected_qualifying"]))
        lines.append(best_table_line(f"{arm} fit-ranked carry", best_rows[arm]["carried_fit_ranked"]))
    lines.extend(
        [
            "",
            "## All 16 layers, both arms",
            "",
            "Rates include window-bootstrap 95% confidence intervals. The exact aggregate value, not a confidence bound, controls the registered gate.",
            "",
            "| Layer | Arm | Hyperparameter | FIT tau | DOC-A recall | DOC-B fire | Guides FPR | Code FPR | GRM FPR | WikiText fire (desc.) | Qualifies |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for layer_index in range(N_LAYERS):
        for arm in ARMS:
            row = next(row for row in rows if row["layer"] == layer_index and row["arm"] == arm)
            lines.append(row_table_line(row))
    lines.extend(
        [
            "",
            "## Receipts and reproducibility",
            "",
            f"- Capture validation: `{output / 'capture_validation.json'}` (`{e2.sha256_file(output / 'capture_validation.json')}`)",
            f"- CPU self-test: `{cpu_validation_path}` (`{e2.sha256_file(cpu_validation_path)}`; {cpu_validation['checks_passed']}/{cpu_validation['checks_total']} passed)",
            f"- Full fit receipt: `{output / 'fit_keys.json'}` (`{e2.sha256_file(output / 'fit_keys.json')}`)",
            f"- Key arrays: `{fit_receipt['keys_artifact']['path']}` (`{fit_receipt['keys_artifact']['sha256']}`)",
            f"- Analysis: `{analysis_path}` (`{e2.sha256_file(analysis_path)}`)",
            f"- Script: `{SCRIPT_PATH}` (`{e2.sha256_file(SCRIPT_PATH)}`)",
            f"- Order: `{ORDER_PATH}` (`{e2.sha256_file(ORDER_PATH)}`)",
            "",
            "Reproduction command:",
            "",
            "```bash",
            f"CUDA_VISIBLE_DEVICES='' python3 scripts/{SCRIPT_PATH.name} run --threads {fit_receipt['runtime']['threads']} --bootstrap-resamples {fit_receipt['bootstrap_resamples']} --cg-rtol {fit_receipt['runtime']['cg_rtol']} --cg-maxiter {fit_receipt['runtime']['cg_maxiter']} --logistic-maxiter {fit_receipt['runtime']['logistic_maxiter']}",
            "```",
            "",
            "## Anything not done",
            "",
            "None. Fresh DOC-B inference was intentionally not duplicated because the exact all-layer payloads already existed and passed the same-machine/provenance validation; all requested E3.1 fits, both arms, all layers, receipts, and the registered decision were completed.",
            "",
            "## Evidence limits",
            "",
            "This is finite linear-addressability evidence for one frozen model revision and one pre-registered novel pair. It is not a theorem or a universal model-family claim. K5 is a trained linear probe, and no result here upgrades it into a closed-form key or an expert-quality result.",
            "",
        ]
    )
    report_path = output / "MOE_E3_1_REPORT.md"
    e2.write_text(report_path, "\n".join(lines))
    manifest_receipt = {
        "schema": "moe_e3_1_artifact_manifest_v1",
        "status": "complete",
        "created_at": now_iso(),
        "registered_sentence": decision["registered_sentence"],
        "files": [
            {"path": str(path), "sha256": e2.sha256_file(path), "byte_count": path.stat().st_size}
            for path in (
                output / "capture_validation.json",
                cpu_validation_path,
                output / "all_layer_linear_keys.npz",
                output / "fit_keys.json",
                analysis_path,
                report_path,
            )
        ],
        "anything_not_done": [],
        "script": str(SCRIPT_PATH),
        "script_sha256": e2.sha256_file(SCRIPT_PATH),
        "order": str(ORDER_PATH),
        "order_sha256": e2.sha256_file(ORDER_PATH),
    }
    e2.write_json(output / "artifact_manifest.json", manifest_receipt)
    print(
        json.dumps(
            {
                "status": "complete",
                "sentence": decision["registered_sentence"],
                "report": str(report_path),
                "artifact_manifest": str(output / "artifact_manifest.json"),
            }
        ),
        flush=True,
    )
    return 0


def self_test(args: argparse.Namespace) -> int:
    output = ensure_output(args.output_dir)
    manifest, _arrays = e3.load_prepared(E3_OUTPUT)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    split = split_provenance()
    check("fit_eval_parity_all_six_corpora", all(value == {"fit": 8, "eval": 8} for value in split["fit_eval_parity"].values()), split["fit_eval_parity"])
    check("fit_eval_overlap_zero", split["fit_eval_overlap_count"] == 0, split["fit_eval_overlap_count"])
    check("doc_b_is_fifth_negative", NEGATIVE_CORPORA == ("doc_b", "wikitext", "code", "grm", "guides"), NEGATIVE_CORPORA)
    check("wikitext_not_a_gate", "wikitext" not in FPR_CAPS, FPR_CAPS)

    passing = {
        "doc_a_recall": 0.50,
        "doc_b_fpr": 0.10,
        "guides_fpr": 0.10,
        "code_fpr": 0.05,
        "grm_fpr": 0.05,
        "wikitext_fpr": 1.0,
    }
    check("registered_rule_inclusive_boundaries", registered_qualifies(passing), passing)
    for field in ("doc_a_recall", "doc_b_fpr", "guides_fpr", "code_fpr", "grm_fpr"):
        failing = dict(passing)
        failing[field] += -1.0e-6 if field == "doc_a_recall" else 1.0e-6
        check(f"registered_rule_rejects_{field}", not registered_qualifies(failing), failing)

    rng = np.random.default_rng(args.seed)
    dimension = 12
    direction = e2.unit_vector(rng.standard_normal(dimension), name="E3.1 fixture direction")[0]
    synthetic: dict[str, np.ndarray] = {}
    shifts = {
        "doc_a": 1.2,
        "doc_b": 0.2,
        "wikitext": -0.1,
        "code": -0.3,
        "grm": -0.5,
        "guides": 0.0,
    }
    for corpus in ALL_CORPORA:
        values = rng.standard_normal((N_WINDOWS, 8, dimension)).astype(np.float32) * 0.35
        values += shifts[corpus] * direction
        synthetic[corpus] = values
    k4, k4_meta = fit_k4_direction(synthetic, cg_rtol=1.0e-7, cg_maxiter=100)
    k5, k5_meta = fit_k5_direction(synthetic, logistic_maxiter=200, seed=args.seed)
    tau_k4 = select_tau(score_hidden(synthetic, k4))["selected_tau"]
    tau_k5 = select_tau(score_hidden(synthetic, k5))["selected_tau"]
    mutated = {name: value.copy() for name, value in synthetic.items()}
    for value in mutated.values():
        value[np.asarray(EVAL_INDICES)] += (
            rng.standard_normal(value[np.asarray(EVAL_INDICES)].shape).astype(np.float32) * 100.0
        )
    k4_mutated, k4_meta_mutated = fit_k4_direction(
        mutated, cg_rtol=1.0e-7, cg_maxiter=100
    )
    k5_mutated, k5_meta_mutated = fit_k5_direction(
        mutated, logistic_maxiter=200, seed=args.seed
    )
    tau_k4_mutated = select_tau(score_hidden(mutated, k4_mutated))["selected_tau"]
    tau_k5_mutated = select_tau(score_hidden(mutated, k5_mutated))["selected_tau"]
    check("k4_unit_norm", abs(float(np.linalg.norm(k4)) - 1.0) < 1.0e-5, float(np.linalg.norm(k4)))
    check("k5_unit_norm", abs(float(np.linalg.norm(k5)) - 1.0) < 1.0e-5, float(np.linalg.norm(k5)))
    check("k5_labeled_trained_linear", k5_meta["label"] == "trained linear logistic probe", k5_meta["label"])
    check("k4_eval_mutation_key_exact", np.array_equal(k4, k4_mutated), e2.sha256_array(k4_mutated))
    check("k5_eval_mutation_key_exact", np.array_equal(k5, k5_mutated), e2.sha256_array(k5_mutated))
    check("k4_eval_mutation_hyperparameter_exact", k4_meta["selected_alpha"] == k4_meta_mutated["selected_alpha"], [k4_meta["selected_alpha"], k4_meta_mutated["selected_alpha"]])
    check("k5_eval_mutation_hyperparameter_exact", k5_meta["selected_C"] == k5_meta_mutated["selected_C"], [k5_meta["selected_C"], k5_meta_mutated["selected_C"]])
    check("k4_eval_mutation_tau_exact", tau_k4 == tau_k4_mutated, [tau_k4, tau_k4_mutated])
    check("k5_eval_mutation_tau_exact", tau_k5 == tau_k5_mutated, [tau_k5, tau_k5_mutated])

    with tempfile.TemporaryDirectory(prefix="olmoe-e3-1-self-test-", dir="/tmp") as temporary:
        bad_path = Path(temporary) / "nonfinite.json"
        failed_loud = False
        try:
            e2.write_json(bad_path, {"bad": float("nan")})
        except ValueError as exc:
            failed_loud = "$.bad" in str(exc) and not bad_path.exists()
    check("writer_rejects_nonfinite", failed_loud, "$.bad named before write")

    passed = all(row["passed"] for row in checks)
    receipt = {
        "schema": "moe_e3_1_cpu_validation_v1",
        "status": "passed" if passed else "failed",
        "created_at": now_iso(),
        "synthetic_only_not_gate_evidence": True,
        **input_binding(manifest),
        "checks_passed": sum(row["passed"] for row in checks),
        "checks_total": len(checks),
        "checks": checks,
        "runtime": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "script": str(SCRIPT_PATH),
        "script_sha256": e2.sha256_file(SCRIPT_PATH),
        "order": str(ORDER_PATH),
        "order_sha256": e2.sha256_file(ORDER_PATH),
    }
    e2.write_json(output / "cpu_validation.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "checks_passed": receipt["checks_passed"],
                "checks_total": receipt["checks_total"],
            }
        ),
        flush=True,
    )
    return 0 if passed else 3


def run(args: argparse.Namespace) -> int:
    for stage in (validate_inputs, self_test, fit, analyze):
        code = stage(args)
        if code != 0:
            return code
    return 0


def main() -> int:
    args = parse_args()
    dispatch = {
        "validate-inputs": validate_inputs,
        "self-test": self_test,
        "fit": fit,
        "analyze": analyze,
        "run": run,
    }
    return dispatch[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
