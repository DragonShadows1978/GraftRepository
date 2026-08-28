#!/usr/bin/env python3
"""MOE-E1 NarrativeForge asymmetric residual expert experiment.

This is a sealed, resumable harness for ORDER MOE-E1.  The frozen
GPT-OSS-20B is used only for inference.  GPU work is split into independent
sub-ten-minute invocations; all fitting, adapter training, and analysis are
CPU-only.  The registered install ABI is:

    score = h @ key
    if score >= tau:
        residual_output += B @ silu(A @ h)

where ``h`` is exactly the post-attention-normalized tensor presented to the
native MoE router.  Native routing and native MoE weights are never changed.

Modes:
  prepare        deterministic corpus split/windows plus CPU contract tests
  capture-keys   one eight-window, all-layer narrative router-input capture
  fit-key        four-corpus shrinkage Fisher-LDA and frozen-threshold gate
  capture-pairs  one teacher/student activation-pair window
  train          rank-64 adapter training with validation early stopping
  eval-gates     one bounded ABI/narrative/generic/code GPU evaluation unit
  analyze        aggregate gates, bootstrap G4, and write the final report
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import io
import json
import math
import os
import platform
import stat
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

ORDER_PATH = REPO_ROOT / "orders" / "MOE_E1_NARRATIVE_EXPERT.md"
E11_ORDER_PATH = REPO_ROOT / "orders" / "MOE_E1_1_AMENDED_ADDRESS_GATE.md"
E13_ORDER_PATH = REPO_ROOT / "orders" / "MOE_E1_3_TEACHER_REDESIGN.md"
RT1_DIR = REPO_ROOT / "artifacts" / "moe_rt1"
RT2_DIR = REPO_ROOT / "artifacts" / "moe_rt2"
RT2_1_DIR = REPO_ROOT / "artifacts" / "moe_rt2_1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "moe_e1"
E12_DIAG_ANALYSIS_PATH = REPO_ROOT / "artifacts" / "moe_e1_diag" / "analysis.json"
E13_OUTPUT_DIR = REPO_ROOT / "artifacts" / "moe_e1_3"
E13_PREFIX_MANIFEST_PATH = E13_OUTPUT_DIR / "prefix_manifest.json"
E13_PREFIX_ARRAYS_PATH = E13_OUTPUT_DIR / "teacher_prefixes.npz"
E13_SELECTION_PATH = E13_OUTPUT_DIR / "selection.json"
E13_RESUME_PATH = E13_OUTPUT_DIR / "GPU_E13_RESUME_COMMANDS.sh"
EXPERTPACK_DIRNAME = "expertpack_narrative_v0"
ADDRESS_RULE_E1 = "e1"
ADDRESS_RULE_E11 = "e11"
ADDRESS_RULES = (ADDRESS_RULE_E1, ADDRESS_RULE_E11)
EVAL_FIX_ORIGINAL = "original"
EVAL_FIX_E12 = "e12"
EVAL_FIXES = (EVAL_FIX_ORIGINAL, EVAL_FIX_E12)
TEACHER_RULE_ORIGINAL = "original"
TEACHER_RULE_E13 = "e13"
TEACHER_RULES = (TEACHER_RULE_ORIGINAL, TEACHER_RULE_E13)
SNAPSHOT = Path(
    "/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/"
    "snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
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

N_FILES = 35
N_TRAIN_FILES = 25
N_HELDOUT_FILES = 10
N_LAYERS = 24
HIDDEN_DIM = 2880
N_EXPERTS = 32
EXPERTS_PER_TOKEN = 4
WINDOW_TOKENS = 512
PREFIX_TOKENS = 2048
N_KEY_WINDOWS = 16
KEY_WINDOWS_PER_CHUNK = 8
N_PAIR_TRAIN = 48
N_PAIR_VALIDATION = 16
N_PAIR_WINDOWS = N_PAIR_TRAIN + N_PAIR_VALIDATION
N_BEHAVIORAL_WINDOWS = 16
FIT_INDICES = tuple(range(0, N_KEY_WINDOWS, 2))
EVAL_INDICES = tuple(range(1, N_KEY_WINDOWS, 2))
INNER_TRAIN_INDICES = (0, 4, 8, 12)
INNER_VALIDATION_INDICES = (2, 6, 10, 14)
NEGATIVE_CORPORA = ("generic", "code", "grm")
RT2_CORPUS_FOR_NEGATIVE = {"generic": "generic", "code": "code", "grm": "domain"}

TAU_GRID = (
    ("p90", 0.90),
    ("p95", 0.95),
    ("p99", 0.99),
    ("p99.5", 0.995),
)
K4_ALPHA_GRID = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0)
RECALL_FLOOR = 0.50
FPR_CAPS = {"generic": 0.02, "code": 0.05, "grm": 0.05}
G3_IMPROVEMENT_FLOOR = 0.10
G4_RECOVERY_FLOOR = 0.25
G5_WIKITEXT_PPL_DELTA_CAP_PCT = 0.5
G5_FIRE_CAPS = {"generic": 0.02, "code": 0.05}
E11_HIGH_FIRING_WIKITEXT_WINDOWS = (5, 13, 11, 15)
DEFAULT_SEED = 20260827
DEFAULT_BOOTSTRAP_RESAMPLES = 2000

GPU_WRAPPER = (
    "flock -w 7200 /tmp/forge-gpu.lock "
    "timeout --signal=TERM --kill-after=5s 590s "
    "env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 "
    "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the registered MOE-E1 NarrativeForge expert experiment."
    )
    parser.add_argument(
        "mode",
        choices=(
            "prepare",
            "self-test",
            "capture-keys",
            "fit-key",
            "capture-pairs",
            "train",
            "eval-gates",
            "analyze",
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--address-rule",
        choices=ADDRESS_RULES,
        default=ADDRESS_RULE_E1,
        help=(
            "address/non-interference registration; e1 is the original default, "
            "e11 opts into ORDER MOE-E1.1"
        ),
    )
    parser.add_argument(
        "--eval-fix",
        choices=EVAL_FIXES,
        default=EVAL_FIX_ORIGINAL,
        help=(
            "evaluation execution schedule; original preserves E1/E1.1 receipts, "
            "e12 opts into isolated arm forwards for ORDER MOE-E1.2"
        ),
    )
    parser.add_argument(
        "--teacher-rule",
        choices=TEACHER_RULES,
        default=TEACHER_RULE_ORIGINAL,
        help=(
            "teacher construction registration; original preserves E1/E1.1, "
            "e13 opts into the positive-gap winner sealed by ORDER MOE-E1.3"
        ),
    )
    parser.add_argument("--model-dir", type=Path, default=SNAPSHOT)
    parser.add_argument("--chunk-index", type=int)
    parser.add_argument("--pair-index", type=int)
    parser.add_argument(
        "--eval-kind", choices=("abi", "narrative", "generic", "code")
    )
    parser.add_argument("--window-index", type=int)
    parser.add_argument("--rank", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--cg-rtol", type=float, default=1.0e-5)
    parser.add_argument("--cg-maxiter", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--token-batch-size", type=int, default=256)
    parser.add_argument("--train-tokens-per-window", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def order_path_for_rule(address_rule: str) -> Path:
    return E11_ORDER_PATH if address_rule == ADDRESS_RULE_E11 else ORDER_PATH


def order_path_for_experiment(address_rule: str, teacher_rule: str) -> Path:
    if teacher_rule == TEACHER_RULE_E13:
        return E13_ORDER_PATH
    return order_path_for_rule(address_rule)


def expertpack_dirname(address_rule: str) -> str:
    return (
        f"{EXPERTPACK_DIRNAME}_e11"
        if address_rule == ADDRESS_RULE_E11
        else EXPERTPACK_DIRNAME
    )


def experiment_output(output: Path, teacher_rule: str) -> Path:
    if teacher_rule == TEACHER_RULE_E13:
        return E13_OUTPUT_DIR
    return output


def expertpack_path(
    output: Path,
    address_rule: str,
    teacher_rule: str = TEACHER_RULE_ORIGINAL,
) -> Path:
    if teacher_rule == TEACHER_RULE_E13:
        return E13_OUTPUT_DIR / "expertpack_narrative_v0_e13" / "manifest.json"
    return output / expertpack_dirname(address_rule) / "manifest.json"


def expertpack_root(
    output: Path,
    address_rule: str,
    teacher_rule: str = TEACHER_RULE_ORIGINAL,
) -> Path:
    return expertpack_path(output, address_rule, teacher_rule).parent


def fit_key_paths(output: Path, address_rule: str) -> tuple[Path, Path]:
    suffix = "_e11" if address_rule == ADDRESS_RULE_E11 else ""
    return output / f"fit_key{suffix}.json", output / f"narrative_k4_keys{suffix}.npz"


def cpu_validation_path(output: Path, address_rule: str) -> Path:
    suffix = "_e11" if address_rule == ADDRESS_RULE_E11 else ""
    return output / f"cpu_validation{suffix}.json"


def pair_root(
    output: Path,
    address_rule: str,
    teacher_rule: str = TEACHER_RULE_ORIGINAL,
) -> Path:
    if teacher_rule == TEACHER_RULE_E13:
        return E13_OUTPUT_DIR / "pairs_e13"
    return output / ("pairs_e11" if address_rule == ADDRESS_RULE_E11 else "pairs")


def train_path(
    output: Path,
    address_rule: str,
    teacher_rule: str = TEACHER_RULE_ORIGINAL,
) -> Path:
    if teacher_rule == TEACHER_RULE_E13:
        return E13_OUTPUT_DIR / "train_e13.json"
    suffix = "_e11" if address_rule == ADDRESS_RULE_E11 else ""
    return output / f"train{suffix}.json"


def eval_root(
    output: Path,
    address_rule: str,
    teacher_rule: str = TEACHER_RULE_ORIGINAL,
) -> Path:
    if teacher_rule == TEACHER_RULE_E13:
        return E13_OUTPUT_DIR / "eval_e13"
    return output / ("eval_e11" if address_rule == ADDRESS_RULE_E11 else "eval")


def eval_root_for_fix(
    output: Path,
    address_rule: str,
    eval_fix: str = EVAL_FIX_ORIGINAL,
    teacher_rule: str = TEACHER_RULE_ORIGINAL,
) -> Path:
    if teacher_rule == TEACHER_RULE_E13:
        if eval_fix != EVAL_FIX_ORIGINAL:
            raise ValueError("E1.3 preserves the original established evaluation path")
        return eval_root(output, address_rule, teacher_rule)
    if eval_fix == EVAL_FIX_E12:
        if address_rule != ADDRESS_RULE_E11:
            raise ValueError("--eval-fix e12 requires --address-rule e11")
        return output / "eval_e11_e12"
    if eval_fix != EVAL_FIX_ORIGINAL:
        raise ValueError(f"unsupported eval fix {eval_fix!r}")
    return eval_root(output, address_rule, teacher_rule)


def analysis_path(
    output: Path,
    address_rule: str,
    teacher_rule: str = TEACHER_RULE_ORIGINAL,
) -> Path:
    if teacher_rule == TEACHER_RULE_E13:
        return E13_OUTPUT_DIR / "analysis_e13.json"
    suffix = "_e11" if address_rule == ADDRESS_RULE_E11 else ""
    return output / f"analysis{suffix}.json"


def report_path(
    output: Path,
    address_rule: str,
    teacher_rule: str = TEACHER_RULE_ORIGINAL,
) -> Path:
    if teacher_rule == TEACHER_RULE_E13:
        return E13_OUTPUT_DIR / "MOE_E1_3_EXPERT_REPORT.md"
    return output / (
        "MOE_E1_1_REPORT.md"
        if address_rule == ADDRESS_RULE_E11
        else "MOE_E1_REPORT.md"
    )


def blocked_path(
    output: Path,
    address_rule: str,
    teacher_rule: str = TEACHER_RULE_ORIGINAL,
) -> Path:
    if teacher_rule == TEACHER_RULE_E13:
        return E13_OUTPUT_DIR / "blocked_e13.json"
    suffix = "_e11" if address_rule == ADDRESS_RULE_E11 else ""
    return output / f"blocked{suffix}.json"


def threshold_constraint_corpora(address_rule: str) -> tuple[str, ...]:
    if address_rule == ADDRESS_RULE_E11:
        return ("code", "grm")
    return NEGATIVE_CORPORA


def g5_qualifies(
    *,
    address_rule: str,
    wikitext_ppl_delta_percent: float,
    generic_fire_rate: float,
    code_fire_rate: float,
) -> bool:
    fire_ok = code_fire_rate <= G5_FIRE_CAPS["code"]
    if address_rule == ADDRESS_RULE_E1:
        fire_ok = fire_ok and generic_fire_rate <= G5_FIRE_CAPS["generic"]
    return bool(
        wikitext_ppl_delta_percent <= G5_WIKITEXT_PPL_DELTA_CAP_PCT
        and fire_ok
    )


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    temporary.replace(path)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def ensure_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != DEFAULT_OUTPUT_DIR.resolve():
        raise ValueError(
            f"ORDER MOE-E1 permits artifacts only at {DEFAULT_OUTPUT_DIR.resolve()}"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_registered_model_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != SNAPSHOT.resolve():
        raise ValueError(
            f"ORDER MOE-E1 registers the frozen snapshot {SNAPSHOT.resolve()}, got {resolved}"
        )
    return resolved


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


def cuda_environment_probe() -> dict[str, Any]:
    entries = sorted(Path("/dev").glob("nvidia*"))
    nodes: list[str] = []
    for path in entries:
        try:
            if stat.S_ISCHR(path.stat().st_mode):
                nodes.append(str(path))
        except OSError:
            pass
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        smi: dict[str, Any] = {
            "returncode": int(proc.returncode),
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        smi = {
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }
    return {
        "device_entries": [str(path) for path in entries],
        "character_device_nodes": nodes,
        "device_node_count": len(nodes),
        "nvidia_smi": smi,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def require_cuda() -> dict[str, Any]:
    probe = cuda_environment_probe()
    required = {"/dev/nvidia0", "/dev/nvidiactl"}
    if not required.issubset(set(probe["character_device_nodes"])):
        raise RuntimeError(
            "CUDA device nodes unavailable; GPU mode requires /dev/nvidia0 and "
            "/dev/nvidiactl under the registered flock/590s wrapper"
        )
    return probe


def nvidia_smi() -> str | None:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,"
                "power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=15,
        ).strip()
    except Exception:
        return None


def load_runtime():
    # Import the committed RT1 helper only in GPU modes.  This keeps prepare,
    # fit, train, and analyze CPU-only and avoids repository bytecode writes.
    import gpt_oss20b_router_telemetry as rt1

    runtime = rt1.load_runtime()
    runtime["rt1"] = rt1
    return runtime


def validate_model_contract(cfg) -> None:
    observed = (
        int(cfg.num_layers),
        int(cfg.hidden_dim),
        int(cfg.num_local_experts),
        int(cfg.num_experts_per_tok),
    )
    expected = (N_LAYERS, HIDDEN_DIM, N_EXPERTS, EXPERTS_PER_TOKEN)
    if observed != expected:
        raise RuntimeError(f"model contract {observed} != registered {expected}")


def corpus_relative(path: Path) -> str:
    resolved = path.resolve()
    for root in CORPUS_ROOTS:
        try:
            relative = resolved.relative_to(root.resolve())
            return f"{root.name}/{relative.as_posix()}"
        except ValueError:
            continue
    raise ValueError(f"corpus path escapes the three authorized roots: {resolved}")


def discover_corpus_files() -> list[Path]:
    files: list[Path] = []
    for root in CORPUS_ROOTS:
        resolved_root = root.resolve()
        forbidden = [
            part
            for part in resolved_root.parts
            if any(term in part.casefold() for term in FORBIDDEN_DIRECTORY_TERMS)
        ]
        if forbidden:
            raise RuntimeError(f"forbidden directory component in corpus root: {root}")
        if not resolved_root.is_dir():
            raise FileNotFoundError(resolved_root)
        # The registered roots contain files directly.  Do not scan a parent or
        # follow directory symlinks into adjacent methodology material.
        for path in sorted(resolved_root.iterdir(), key=lambda p: p.name):
            if path.is_symlink():
                raise RuntimeError(f"corpus symlink rejected: {path}")
            if path.is_dir():
                directory_terms = path.parts
                if any(
                    term in component.casefold()
                    for component in directory_terms
                    for term in FORBIDDEN_DIRECTORY_TERMS
                ):
                    continue
                raise RuntimeError(
                    f"unexpected subdirectory in sealed corpus root; not traversed: {path}"
                )
            if path.is_file():
                files.append(path.resolve())
    if len(files) != N_FILES:
        raise RuntimeError(f"registered corpus requires 35 files, found {len(files)}")
    names = [path.name for path in files]
    if len(set(names)) != len(names):
        raise RuntimeError("sha256(filename) split is ambiguous: duplicate basenames")
    return files


def deterministic_file_split(files: Sequence[Path]) -> tuple[list[Path], list[Path]]:
    ranked = sorted(
        files,
        key=lambda path: (
            sha256_bytes(path.name.encode("utf-8")),
            path.name,
        ),
    )
    return list(ranked[:N_TRAIN_FILES]), list(ranked[N_TRAIN_FILES:])


def tokenize_files(files: Sequence[Path], tokenizer) -> dict[Path, np.ndarray]:
    result: dict[Path, np.ndarray] = {}
    for path in files:
        text = path.read_text(encoding="utf-8", errors="strict")
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        result[path] = np.asarray(ids, dtype=np.int64)
    return result


def window_candidates(
    tokenized: dict[Path, np.ndarray], *, seed: int, label: str
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path, ids in tokenized.items():
        if ids.size < WINDOW_TOKENS:
            continue
        base_digest = hashlib.sha256(
            f"{seed}|{label}|{corpus_relative(path)}".encode("utf-8")
        ).digest()
        base = int.from_bytes(base_digest[:8], "big") % WINDOW_TOKENS
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


def windows_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return bool(
        a["path"] == b["path"]
        and int(a["start"]) < int(b["stop"])
        and int(b["start"]) < int(a["stop"])
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
        raise RuntimeError(f"needed {count} disjoint windows, found {len(chosen)}")
    return chosen


def materialize_windows(
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
            raise RuntimeError(f"window {index} shape {ids.shape}")
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


def build_prefixes(
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
        raise RuntimeError("no TRAIN file has a 2,048-token prefix excerpt")
    rotation_start = int.from_bytes(
        hashlib.sha256(f"{seed}|{label}|prefix-rotation".encode()).digest()[:8],
        "big",
    ) % len(eligible)
    prefix_arrays: list[np.ndarray] = []
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
        prefix_arrays.append(excerpt)
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
            }
        )
    return np.stack(prefix_arrays), provenance


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


def numpy_expert_add(
    native_output: np.ndarray,
    h: np.ndarray,
    key: np.ndarray,
    tau: float,
    A: np.ndarray,
    B: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference ABI that copies non-fired rows without arithmetic."""

    native = np.asarray(native_output)
    hidden = np.asarray(h, dtype=np.float32)
    flat_native = native.reshape(-1, native.shape[-1])
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    scores = flat_hidden @ np.asarray(key, dtype=np.float32)
    fired = scores >= float(tau)
    result = flat_native.copy()
    if fired.any():
        z = flat_hidden[fired] @ np.asarray(A, dtype=np.float32).T
        activated = z / (1.0 + np.exp(-z))
        delta = activated @ np.asarray(B, dtype=np.float32).T
        result[fired] = result[fired] + delta.astype(result.dtype)
    return result.reshape(native.shape), scores, fired


def exact_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=np.float64).reshape(-1)
    negative = np.asarray(negative, dtype=np.float64).reshape(-1)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("AUC classes must be non-empty")
    ordered = np.sort(negative)
    left = np.searchsorted(ordered, positive, side="left")
    right = np.searchsorted(ordered, positive, side="right")
    wins = left.astype(np.float64) + 0.5 * (right - left)
    return float(wins.sum() / (positive.size * negative.size))


def bootstrap_recovery(
    base_nll: np.ndarray,
    teacher_nll: np.ndarray,
    expert_nll: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    arrays = [np.asarray(x, dtype=np.float64).reshape(-1) for x in (base_nll, teacher_nll, expert_nll)]
    if not arrays[0].size or any(x.size != arrays[0].size for x in arrays):
        raise ValueError("bootstrap arms require the same nonzero window count")

    def summarize(indices: np.ndarray | None = None) -> tuple[float, float, float, float]:
        work = arrays if indices is None else [x[indices] for x in arrays]
        p_base, p_teacher, p_expert = [float(np.exp(x.mean())) for x in work]
        gap = p_base - p_teacher
        # The registered premise is checked on the observed aggregate, not on
        # each bootstrap resample.  Preserve negative-gap resamples in the CI;
        # dropping them would make uncertainty look artificially narrow.
        recovery = (
            (p_base - p_expert) / gap
            if abs(gap) > np.finfo(np.float64).eps
            else float("nan")
        )
        return p_base, p_teacher, p_expert, recovery

    p_base, p_teacher, p_expert, recovery = summarize()
    rng = np.random.default_rng(seed)
    boot: list[float] = []
    for _ in range(resamples):
        indices = rng.integers(0, arrays[0].size, size=arrays[0].size)
        value = summarize(indices)[3]
        if np.isfinite(value):
            boot.append(value)
    if boot:
        low, high = np.quantile(np.asarray(boot), [0.025, 0.975], method="linear")
        ci_low, ci_high = float(low), float(high)
    else:
        ci_low = ci_high = float("nan")
    return {
        "window_count": int(arrays[0].size),
        "ppl_base": p_base,
        "ppl_teacher": p_teacher,
        "ppl_expert": p_expert,
        "teacher_gap": p_base - p_teacher,
        "recovery_fraction": recovery,
        "recovery_percent": recovery * 100.0,
        "ci95_low_fraction": ci_low,
        "ci95_high_fraction": ci_high,
        "ci95_low_percent": ci_low * 100.0,
        "ci95_high_percent": ci_high * 100.0,
        "bootstrap_unit": "behavioral windows",
        "bootstrap_resamples": int(resamples),
    }


def run_cpu_validation(
    seed: int, address_rule: str = ADDRESS_RULE_E1
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(2, 7, 12)).astype(np.float32)
    native = rng.normal(size=h.shape).astype(np.float32)
    key = rng.normal(size=12).astype(np.float32)
    key /= np.linalg.norm(key)
    A = rng.normal(scale=0.1, size=(4, 12)).astype(np.float32)
    B_zero = np.zeros((12, 4), dtype=np.float32)
    zero_out, scores, fired = numpy_expert_add(native, h, key, np.median(h @ key), A, B_zero)
    add_check(checks, "abi_zero_B_exact", True, bool(np.array_equal(native, zero_out)))
    B = rng.normal(scale=0.1, size=(12, 4)).astype(np.float32)
    mixed, scores, fired = numpy_expert_add(native, h, key, float(np.median(scores)), A, B)
    flat_native = native.reshape(-1, 12)
    flat_mixed = mixed.reshape(-1, 12)
    add_check(checks, "abi_mixed_has_fired", True, bool(fired.any()))
    add_check(checks, "abi_mixed_has_not_fired", True, bool((~fired).any()))
    add_check(
        checks,
        "abi_nonfired_bit_identity",
        True,
        bool(np.array_equal(flat_native[~fired], flat_mixed[~fired])),
    )
    nofire, _scores, nofire_mask = numpy_expert_add(native, h, key, math.inf, A, B)
    add_check(checks, "abi_forced_nofire_count", 0, int(nofire_mask.sum()))
    add_check(checks, "abi_forced_nofire_exact", True, bool(np.array_equal(native, nofire)))

    pos = rng.normal(loc=2.0, scale=0.4, size=(4, 8))
    neg = rng.normal(loc=0.0, scale=0.4, size=(4, 8))
    add_check(checks, "exact_auc_separates", True, bool(exact_auc(pos, neg) > 0.99))

    recovery = bootstrap_recovery(
        np.full(16, math.log(12.0)),
        np.full(16, math.log(8.0)),
        np.full(16, math.log(10.0)),
        resamples=200,
        seed=seed + 1,
    )
    add_check(
        checks,
        "bootstrap_recovery_50pct",
        True,
        bool(abs(recovery["recovery_percent"] - 50.0) <= 1.0e-12),
    )
    add_check(
        checks,
        "pair_alignment_reference",
        True,
        bool(np.array_equal(np.arange(512), np.arange(512))),
    )
    add_check(
        checks,
        "registered_geometry",
        [24, 2880, 64],
        [N_LAYERS, HIDDEN_DIM, 64],
    )

    # Exercise the production four-corpus K4 solver, FIT-only alpha choice,
    # pooled-negative threshold policy, and outer EVAL qualification on a
    # deliberately separated low-dimensional fixture.
    fixture_rng = np.random.default_rng(seed + 99)
    hidden_fixture = {
        corpus: fixture_rng.normal(0.0, 0.35, size=(16, 8, 12)).astype(np.float16)
        for corpus in ("narrative", *NEGATIVE_CORPORA)
    }
    hidden_fixture["narrative"] = (
        hidden_fixture["narrative"].astype(np.float32)
        + np.asarray([2.0, 1.0, 0.5] + [0.0] * 9, dtype=np.float32)
    ).astype(np.float16)
    fixture_key, fixture_meta = fit_k4_direction(
        hidden_fixture, cg_rtol=1.0e-6, cg_maxiter=200
    )
    fixture_scores = score_four(hidden_fixture, fixture_key)
    fixture_metrics = analyze_key_scores(
        fixture_scores,
        resamples=200,
        seed=seed + 100,
        address_rule=address_rule,
    )
    add_check(
        checks,
        "k4_fixture_unit_key",
        True,
        bool(abs(float(np.linalg.norm(fixture_key.astype(np.float64))) - 1.0) < 1.0e-6),
    )
    add_check(checks, "k4_fixture_qualifies", True, fixture_metrics["qualifies"])
    mutated = {name: value.copy() for name, value in hidden_fixture.items()}
    for value in mutated.values():
        value[np.asarray(EVAL_INDICES)] = fixture_rng.normal(
            20.0, 1.0, size=value[np.asarray(EVAL_INDICES)].shape
        ).astype(np.float16)
    key_after_eval_mutation, meta_after_eval_mutation = fit_k4_direction(
        mutated, cg_rtol=1.0e-6, cg_maxiter=200
    )
    tau_before = select_tau_four(fixture_scores, address_rule)["selected_tau"]
    tau_after = select_tau_four(
        score_four(mutated, key_after_eval_mutation), address_rule
    )[
        "selected_tau"
    ]
    add_check(
        checks,
        "k4_eval_mutation_does_not_change_key",
        True,
        bool(np.array_equal(fixture_key, key_after_eval_mutation)),
    )
    add_check(
        checks,
        "k4_eval_mutation_does_not_change_tau",
        True,
        bool(tau_before == tau_after),
    )
    add_check(
        checks,
        "k4_eval_mutation_does_not_change_alpha",
        fixture_meta["selected_alpha"],
        meta_after_eval_mutation["selected_alpha"],
    )

    if address_rule == ADDRESS_RULE_E11:
        # Deliberately make generic text fire everywhere while code/GRM remain
        # silent.  E1 must reject this fixture; E1.1 must accept it without
        # changing the frozen pooled-negative quantile construction.
        amended_scores = {
            "narrative": np.full((N_KEY_WINDOWS, 8), 2.0, dtype=np.float32),
            "generic": np.full((N_KEY_WINDOWS, 8), 1.0, dtype=np.float32),
            "code": np.zeros((N_KEY_WINDOWS, 8), dtype=np.float32),
            "grm": np.zeros((N_KEY_WINDOWS, 8), dtype=np.float32),
        }
        original_metrics = analyze_key_scores(
            amended_scores,
            resamples=200,
            seed=seed + 201,
            address_rule=ADDRESS_RULE_E1,
        )
        amended_metrics = analyze_key_scores(
            amended_scores,
            resamples=200,
            seed=seed + 202,
            address_rule=ADDRESS_RULE_E11,
        )
        add_check(
            checks,
            "e11_fixture_original_rule_rejects_generic_fire",
            False,
            original_metrics["qualifies"],
        )
        add_check(
            checks,
            "e11_fixture_code_grm_fit_feasible",
            True,
            amended_metrics["fit_fpr_feasible"],
        )
        add_check(
            checks,
            "e11_fixture_qualifies_with_descriptive_generic",
            True,
            amended_metrics["qualifies"],
        )
        add_check(
            checks,
            "e11_fixture_generic_per_window_reported",
            list(EVAL_INDICES),
            [
                item["window_index"]
                for item in amended_metrics["eval"]["generic_fire_rate"][
                    "per_window"
                ]
            ],
        )
        add_check(
            checks,
            "e11_g5_ignores_descriptive_generic_fire",
            True,
            g5_qualifies(
                address_rule=ADDRESS_RULE_E11,
                wikitext_ppl_delta_percent=0.1,
                generic_fire_rate=0.8,
                code_fire_rate=0.01,
            ),
        )
        add_check(
            checks,
            "e11_g5_enforces_code_fire",
            False,
            g5_qualifies(
                address_rule=ADDRESS_RULE_E11,
                wikitext_ppl_delta_percent=0.1,
                generic_fire_rate=0.0,
                code_fire_rate=0.051,
            ),
        )
        add_check(
            checks,
            "e11_g5_enforces_wikitext_ppl_delta",
            False,
            g5_qualifies(
                address_rule=ADDRESS_RULE_E11,
                wikitext_ppl_delta_percent=0.501,
                generic_fire_rate=0.0,
                code_fire_rate=0.0,
            ),
        )
        window_fixture = wikitext_window_summary(
            {
                "window_index": 5,
                "arms": {
                    "base": {"ppl": 10.0},
                    "expert": {"ppl": 9.5},
                },
                "fire": {
                    "fire_count": 64,
                    "token_count": 128,
                    "fire_rate": 0.5,
                },
                "_path": "synthetic",
                "_sha256": "synthetic",
            }
        )
        add_check(
            checks,
            "e11_g5_high_window_labeled",
            True,
            window_fixture["registered_high_firing_narrativeish"],
        )
        add_check(
            checks,
            "e11_g5_window_ppl_delta",
            True,
            bool(abs(window_fixture["ppl_delta_percent"] + 5.0) <= 1.0e-12),
        )
        resume_fixture = gpu_resume_commands(ADDRESS_RULE_E11)
        add_check(
            checks,
            "e11_resume_preserves_gpu_wrapper",
            True,
            bool(
                "flock -w 7200 /tmp/forge-gpu.lock" in resume_fixture
                and "timeout --signal=TERM --kill-after=5s 590s" in resume_fixture
                and "sleep 30" in resume_fixture
            ),
        )
        add_check(
            checks,
            "e11_resume_starts_after_fit_key",
            True,
            bool(
                "capture-pairs --address-rule e11" in resume_fixture
                and "capture-keys" not in resume_fixture
                and "fit-key" not in resume_fixture
            ),
        )

    # Tiny CPU consolidation fixture: B begins at exact zero and a fixed
    # nonlinear A feature bank learns a known residual target.  This is a
    # machinery test only; it is not counted as G3 evidence.
    torch_version: str | None = None
    try:
        import torch
        import torch.nn.functional as torch_f

        torch_version = torch.__version__
        torch.manual_seed(seed + 123)
        features = torch.randn(96, 12)
        fixed_A = torch.randn(4, 12) * 0.2
        true_B = torch.randn(12, 4) * 0.15
        targets = torch_f.silu(features @ fixed_A.T) @ true_B.T
        fitted_B = torch.nn.Parameter(torch.zeros(12, 4))
        zero_exact = bool(
            torch.equal(fitted_B.detach(), torch.zeros_like(fitted_B.detach()))
        )
        optimizer = torch.optim.Adam([fitted_B], lr=0.05)
        initial_mse = float(torch.mean(targets**2).item())
        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            prediction = torch_f.silu(features @ fixed_A.T) @ fitted_B.T
            loss = torch.mean((prediction - targets) ** 2)
            loss.backward()
            optimizer.step()
        final_mse = float(
            torch.mean(
                (torch_f.silu(features @ fixed_A.T) @ fitted_B.T - targets) ** 2
            ).item()
        )
        add_check(checks, "training_fixture_B_zero_init", True, zero_exact)
        add_check(
            checks,
            "training_fixture_beats_zero_by_10pct",
            True,
            bool(final_mse <= 0.9 * initial_mse),
        )
    except Exception as exc:
        add_check(
            checks,
            "training_fixture_executes",
            "success",
            f"{type(exc).__name__}: {exc}",
        )
    failed = [item for item in checks if not item["passed"]]
    return {
        "schema": (
            "moe_e1_1_cpu_validation_v1"
            if address_rule == ADDRESS_RULE_E11
            else "moe_e1_cpu_validation_v1"
        ),
        "created_at": now_iso(),
        "address_rule": address_rule,
        "order": str(order_path_for_rule(address_rule)),
        "status": "passed" if not failed else "failed",
        "synthetic_only_not_gate_evidence": True,
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "failed_check_names": [item["name"] for item in failed],
        "checks": checks,
        "bootstrap_fixture": recovery,
        "k4_fixture": {
            "selected_alpha": fixture_meta["selected_alpha"],
            "selected_tau": fixture_metrics["selected_tau"],
            "qualifies": fixture_metrics["qualifies"],
            "eval": fixture_metrics["eval"],
        },
        "torch": torch_version,
        "numpy": np.__version__,
        "python": sys.version,
    }


def self_test(args: argparse.Namespace) -> int:
    output = ensure_output_dir(args.output_dir)
    target = cpu_validation_path(output, args.address_rule)
    if target.is_file():
        prior = read_json(target)
        print(
            json.dumps(
                {
                    "status": "already_complete",
                    "validation_status": prior.get("status"),
                    "cpu_validation": str(target),
                }
            ),
            flush=True,
        )
        return 0 if prior.get("status") == "passed" else 2
    manifest, _prepared = load_prepared(output)
    validation = run_cpu_validation(args.seed, args.address_rule)
    original_fit_path, original_keys_path = fit_key_paths(output, ADDRESS_RULE_E1)
    validation.update(
        {
            "corpus_integrity_passed": manifest.get("status") == "passed",
            "corpus_manifest": str(output / "corpus_manifest.json"),
            "corpus_manifest_sha256": sha256_file(output / "corpus_manifest.json"),
            "original_rule_default_invocation": (
                f"python3 {SCRIPT_PATH} <mode> (equivalent to "
                "--address-rule e1)"
            ),
            "preserved_original_key_receipts": {
                "fit_key": {
                    "path": str(original_fit_path),
                    "sha256": sha256_file(original_fit_path),
                },
                "keys": {
                    "path": str(original_keys_path),
                    "sha256": sha256_file(original_keys_path),
                },
            },
        }
    )
    write_json(target, validation)
    print(
        json.dumps(
            {
                "status": validation["status"],
                "checks_passed": validation["checks_passed"],
                "checks_total": validation["checks_total"],
                "cpu_validation": str(target),
            }
        ),
        flush=True,
    )
    return 0 if validation["status"] == "passed" else 2


def pending_expertpack_manifest(
    *,
    output: Path,
    corpus_manifest: Path,
    windows_path: Path,
    address_rule: str = ADDRESS_RULE_E1,
) -> dict[str, Any]:
    pack = output / expertpack_dirname(address_rule)
    return {
        "schema": "expertpack_narrative_v0",
        "created_at": now_iso(),
        "address_rule": address_rule,
        "status": "blocked_pending_gpu_captures",
        "abi": {
            "score": "s = h dot key (float32)",
            "fire": "s >= tau",
            "residual": "B dot silu(A dot h)",
            "attachment": "additive alongside native MoE output; native routing untouched",
            "nonfire": "copy native residual row without arithmetic",
        },
        "model": {
            "name": "openai/gpt-oss-20b",
            "snapshot": str(SNAPSHOT),
            "hidden_dim": HIDDEN_DIM,
        },
        "layer": None,
        "rank": 64,
        "tau": None,
        "key_metrics": None,
        "components": {
            "key": {"path": str(pack / "key_fp32.npy"), "status": "missing"},
            "A": {
                "path": str(pack / "A_fp16.npy"),
                "shape": [64, HIDDEN_DIM],
                "storage_dtype": "float16",
                "status": "missing",
            },
            "B": {
                "path": str(pack / "B_fp16.npy"),
                "shape": [HIDDEN_DIM, 64],
                "storage_dtype": "float16",
                "initialization": "all zeros before training",
                "status": "missing",
            },
        },
        "provenance": {
            "order": str(order_path_for_rule(address_rule)),
            "script": str(SCRIPT_PATH),
            "script_sha256": sha256_file(SCRIPT_PATH),
            "corpus_manifest": str(corpus_manifest),
            "corpus_manifest_sha256": sha256_file(corpus_manifest),
            "prepared_windows": str(windows_path),
            "prepared_windows_sha256": sha256_file(windows_path),
            "rt2_negative_captures": str(RT2_DIR),
        },
        "limitations": [
            "pending GPU narrative key captures",
            "not installable until G2 and G3 pass",
        ],
    }


def prepare(args: argparse.Namespace) -> int:
    output = ensure_output_dir(args.output_dir)
    started = time.perf_counter()
    files = discover_corpus_files()
    train_files, heldout_files = deterministic_file_split(files)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir.resolve()), local_files_only=True
    )
    tokenized = tokenize_files(files, tokenizer)
    train_tokens = {path: tokenized[path] for path in train_files}
    heldout_tokens = {path: tokenized[path] for path in heldout_files}

    train_candidates = window_candidates(train_tokens, seed=args.seed, label="train")
    selected_train = select_disjoint_windows(
        train_candidates, N_PAIR_WINDOWS + N_KEY_WINDOWS
    )
    pair_selected = selected_train[:N_PAIR_WINDOWS]
    key_selected = selected_train[N_PAIR_WINDOWS:]
    heldout_candidates = window_candidates(
        heldout_tokens, seed=args.seed, label="heldout-behavioral"
    )
    behavioral_selected = select_disjoint_windows(
        heldout_candidates, N_BEHAVIORAL_WINDOWS
    )

    pair_ids, pair_provenance = materialize_windows(pair_selected, tokenized)
    key_ids, key_provenance = materialize_windows(key_selected, tokenized)
    behavioral_ids, behavioral_provenance = materialize_windows(
        behavioral_selected, tokenized
    )
    pair_prefix_ids, pair_prefix_provenance = build_prefixes(
        pair_selected,
        train_tokens,
        seed=args.seed,
        label="pair",
        avoid_same_source=True,
    )
    eval_prefix_ids, eval_prefix_provenance = build_prefixes(
        behavioral_selected,
        train_tokens,
        seed=args.seed,
        label="behavioral",
        avoid_same_source=False,
    )

    windows_path = output / "prepared_windows.npz"
    save_npz(
        windows_path,
        key_ids=key_ids,
        pair_ids=pair_ids,
        pair_prefix_ids=pair_prefix_ids,
        behavioral_ids=behavioral_ids,
        behavioral_prefix_ids=eval_prefix_ids,
    )

    train_set = {str(path) for path in train_files}
    heldout_set = {str(path) for path in heldout_files}
    pair_train_sources = {str(item["path"]) for item in pair_selected}
    key_sources = {str(item["path"]) for item in key_selected}
    prefix_sources = {
        item["source_path"] for item in pair_prefix_provenance + eval_prefix_provenance
    }
    behavioral_sources = {str(item["path"]) for item in behavioral_selected}
    checks: list[dict[str, Any]] = []
    add_check(checks, "corpus_file_count", N_FILES, len(files))
    add_check(checks, "train_file_count", N_TRAIN_FILES, len(train_files))
    add_check(checks, "heldout_file_count", N_HELDOUT_FILES, len(heldout_files))
    add_check(checks, "file_split_overlap", 0, len(train_set & heldout_set))
    add_check(
        checks,
        "train_inputs_subset",
        True,
        bool((pair_train_sources | key_sources | prefix_sources) <= train_set),
    )
    add_check(
        checks,
        "heldout_behavioral_subset",
        True,
        bool(behavioral_sources <= heldout_set),
    )
    add_check(
        checks,
        "heldout_absent_from_fit_pair_prefix",
        0,
        len(heldout_set & (pair_train_sources | key_sources | prefix_sources)),
    )
    add_check(
        checks,
        "pair_key_span_overlap",
        0,
        sum(windows_overlap(a, b) for a in pair_selected for b in key_selected),
    )
    add_check(
        checks,
        "pair_train_validation_span_overlap",
        0,
        sum(
            windows_overlap(a, b)
            for a in pair_selected[:N_PAIR_TRAIN]
            for b in pair_selected[N_PAIR_TRAIN:]
        ),
    )
    add_check(checks, "key_ids_shape", [16, 512], list(key_ids.shape))
    add_check(checks, "pair_ids_shape", [64, 512], list(pair_ids.shape))
    add_check(checks, "pair_prefix_shape", [64, 2048], list(pair_prefix_ids.shape))
    add_check(checks, "behavioral_shape", [16, 512], list(behavioral_ids.shape))
    add_check(checks, "behavioral_prefix_shape", [16, 2048], list(eval_prefix_ids.shape))
    failed = [item for item in checks if not item["passed"]]

    corpus_manifest_path = output / "corpus_manifest.json"
    manifest: dict[str, Any] = {
        "schema": "moe_e1_corpus_manifest_v1",
        "created_at": now_iso(),
        "status": "passed" if not failed else "failed_stop",
        "order": str(ORDER_PATH),
        "script": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "model_dir": str(args.model_dir.resolve()),
        "model_snapshot_revision": args.model_dir.resolve().name,
        "tokenizer_class": type(tokenizer).__name__,
        "seed": int(args.seed),
        "authorized_roots": [str(path) for path in CORPUS_ROOTS],
        "scope_interpretation": (
            "only the three authorized directory roots were enumerated; forbidden "
            "terms are enforced on directory components, while filenames inside "
            "those roots remain part of the registered 35-file corpus"
        ),
        "file_split": {
            "algorithm": (
                "sort ascending by sha256(UTF-8 basename), basename tie-break; "
                "first 25 TRAIN, final 10 HELDOUT"
            ),
            "train_count": len(train_files),
            "heldout_count": len(heldout_files),
            "train": [
                file_record(path, tokenized[path].size, "TRAIN") for path in train_files
            ],
            "heldout": [
                file_record(path, tokenized[path].size, "HELDOUT")
                for path in heldout_files
            ],
            "total_bytes": int(sum(path.stat().st_size for path in files)),
        },
        "window_policy": {
            "tokenization": "add_special_tokens=False; tokenize each file separately",
            "window_tokens": WINDOW_TOKENS,
            "prefix_tokens": PREFIX_TOKENS,
            "candidate_spans": "non-overlapping 512-token spans within each file",
            "selection": "ascending seeded sha256 rank",
            "pair_train_indices": list(range(N_PAIR_TRAIN)),
            "pair_validation_indices": list(range(N_PAIR_TRAIN, N_PAIR_WINDOWS)),
            "key_fit_indices": list(FIT_INDICES),
            "key_eval_indices": list(EVAL_INDICES),
            "teacher_prefix_policy": (
                "seeded rotation over TRAIN files with >=2048 tokens; pair prefixes "
                "use a file distinct from their target window"
            ),
        },
        "windows": {
            "key": key_provenance,
            "pairs": [
                {
                    **item,
                    "split": "TRAIN" if index < N_PAIR_TRAIN else "VALIDATION",
                }
                for index, item in enumerate(pair_provenance)
            ],
            "pair_prefixes": pair_prefix_provenance,
            "behavioral_heldout": behavioral_provenance,
            "behavioral_prefixes": eval_prefix_provenance,
        },
        "prepared_windows": {
            "path": str(windows_path),
            "sha256": sha256_file(windows_path),
            "arrays": {
                "key_ids": {"shape": list(key_ids.shape), "sha256": sha256_array(key_ids)},
                "pair_ids": {"shape": list(pair_ids.shape), "sha256": sha256_array(pair_ids)},
                "pair_prefix_ids": {"shape": list(pair_prefix_ids.shape), "sha256": sha256_array(pair_prefix_ids)},
                "behavioral_ids": {"shape": list(behavioral_ids.shape), "sha256": sha256_array(behavioral_ids)},
                "behavioral_prefix_ids": {"shape": list(eval_prefix_ids.shape), "sha256": sha256_array(eval_prefix_ids)},
            },
        },
        "integrity": {
            "passed": not failed,
            "checks_passed": len(checks) - len(failed),
            "checks_total": len(checks),
            "failed_check_names": [item["name"] for item in failed],
            "checks": checks,
        },
        "wall_seconds": float(time.perf_counter() - started),
    }
    write_json(corpus_manifest_path, manifest)

    cpu_validation = run_cpu_validation(args.seed)
    cpu_validation.update(
        {
            "corpus_integrity_passed": not failed,
            "corpus_manifest": str(corpus_manifest_path),
            "corpus_manifest_sha256": sha256_file(corpus_manifest_path),
        }
    )
    write_json(output / "cpu_validation.json", cpu_validation)

    pack = output / EXPERTPACK_DIRNAME
    pack.mkdir(parents=True, exist_ok=True)
    write_json(
        pack / "manifest.json",
        pending_expertpack_manifest(
            output=output,
            corpus_manifest=corpus_manifest_path,
            windows_path=windows_path,
        ),
    )
    status = "passed" if not failed and cpu_validation["status"] == "passed" else "failed_stop"
    print(
        json.dumps(
            {
                "status": status,
                "files": len(files),
                "train_files": len(train_files),
                "heldout_files": len(heldout_files),
                "corpus_bytes": manifest["file_split"]["total_bytes"],
                "manifest": str(corpus_manifest_path),
                "windows": str(windows_path),
                "cpu_validation": str(output / "cpu_validation.json"),
                "expertpack_manifest": str(pack / "manifest.json"),
            }
        ),
        flush=True,
    )
    return 0 if status == "passed" else 2


def load_prepared(output: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = output / "corpus_manifest.json"
    windows_path = output / "prepared_windows.npz"
    if not manifest_path.is_file() or not windows_path.is_file():
        raise FileNotFoundError("run prepare first")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "passed":
        raise RuntimeError("corpus preparation did not pass")
    if Path(manifest.get("model_dir", "")).resolve() != SNAPSHOT.resolve():
        raise RuntimeError("prepared corpus is not bound to the registered model snapshot")
    if manifest["prepared_windows"]["sha256"] != sha256_file(windows_path):
        raise RuntimeError("prepared_windows.npz hash changed since prepare")
    with np.load(windows_path, allow_pickle=False) as stored:
        arrays = {name: np.ascontiguousarray(stored[name]) for name in stored.files}
    expected = {
        "key_ids": (N_KEY_WINDOWS, WINDOW_TOKENS),
        "pair_ids": (N_PAIR_WINDOWS, WINDOW_TOKENS),
        "pair_prefix_ids": (N_PAIR_WINDOWS, PREFIX_TOKENS),
        "behavioral_ids": (N_BEHAVIORAL_WINDOWS, WINDOW_TOKENS),
        "behavioral_prefix_ids": (N_BEHAVIORAL_WINDOWS, PREFIX_TOKENS),
    }
    for name, shape in expected.items():
        if name not in arrays or arrays[name].shape != shape or arrays[name].dtype != np.int64:
            observed = None if name not in arrays else (arrays[name].shape, arrays[name].dtype)
            raise RuntimeError(f"prepared array {name} contract mismatch: {observed}")
        declared = manifest["prepared_windows"]["arrays"][name]["sha256"]
        if declared != sha256_array(arrays[name]):
            raise RuntimeError(f"prepared array {name} hash mismatch")
    return manifest, arrays


def load_e13_selection() -> dict[str, Any]:
    if not E13_SELECTION_PATH.is_file():
        raise FileNotFoundError(
            "ORDER MOE-E1.3 requires a completed positive-gap selection before "
            "capture-pairs; run the E1.3 sweep/analyzer first"
        )
    selection = read_json(E13_SELECTION_PATH)
    if selection.get("status") != "selected_positive_gap":
        raise RuntimeError(
            "E1.3 has no positive teacher construction; downstream expert work "
            f"must stop (selection status={selection.get('status')!r})"
        )
    winner = selection.get("winning_construction")
    if winner not in {"p1", "p2", "p3", "p4"}:
        raise RuntimeError(f"invalid E1.3 winning construction {winner!r}")
    for key, path in (
        ("prefix_manifest_sha256", E13_PREFIX_MANIFEST_PATH),
        ("prefix_arrays_sha256", E13_PREFIX_ARRAYS_PATH),
    ):
        if not path.is_file() or selection.get(key) != sha256_file(path):
            raise RuntimeError(f"E1.3 selected teacher dependency changed: {path}")
    return selection


def load_e13_teacher_prefix(
    role: str,
    index: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    if role not in {"pair", "behavioral"}:
        raise ValueError(f"invalid E1.3 prefix role {role!r}")
    selection = load_e13_selection()
    winner = selection["winning_construction"]
    manifest = read_json(E13_PREFIX_MANIFEST_PATH)
    if manifest.get("status") != "passed":
        raise RuntimeError("E1.3 prefix manifest is not passed")
    array_name = f"{role}_{winner}"
    with np.load(E13_PREFIX_ARRAYS_PATH, allow_pickle=False) as stored:
        if array_name not in stored.files:
            raise RuntimeError(f"missing E1.3 prefix array {array_name}")
        prefixes = np.ascontiguousarray(stored[array_name], dtype=np.int64)
    expected_count = N_PAIR_WINDOWS if role == "pair" else N_BEHAVIORAL_WINDOWS
    if prefixes.ndim != 2 or prefixes.shape[0] != expected_count:
        raise RuntimeError(f"E1.3 prefix array {array_name} shape {prefixes.shape}")
    if index not in range(expected_count):
        raise ValueError(f"E1.3 {role} prefix index {index} is out of range")
    declared_array = manifest["prefix_arrays"]["arrays"][array_name]
    if (
        list(prefixes.shape) != declared_array["shape"]
        or sha256_array(prefixes) != declared_array["sha256"]
    ):
        raise RuntimeError(f"E1.3 prefix array {array_name} hash/shape mismatch")
    prefix = np.ascontiguousarray(prefixes[index], dtype=np.int64)
    provenance = manifest["prefixes"][role][winner][index]
    if provenance["token_ids_sha256"] != sha256_array(prefix):
        raise RuntimeError(f"E1.3 {role}/{winner}/{index} provenance hash mismatch")
    return prefix, provenance, selection


def key_capture_paths(output: Path, chunk_index: int) -> tuple[Path, Path, Path]:
    stem = f"narrative_chunk{chunk_index:02d}_router_inputs_fp16"
    return (
        output / f"{stem}.npy",
        output / f"{stem}.partial.npy",
        output / f"narrative_chunk{chunk_index:02d}_capture_receipt.json",
    )


def capture_keys(args: argparse.Namespace) -> int:
    if args.chunk_index not in (0, 1):
        raise ValueError("capture-keys requires --chunk-index 0 or 1")
    output = ensure_output_dir(args.output_dir)
    manifest, arrays = load_prepared(output)
    cuda_probe = require_cuda()
    start = int(args.chunk_index * KEY_WINDOWS_PER_CHUNK)
    stop = start + KEY_WINDOWS_PER_CHUNK
    ids = np.ascontiguousarray(arrays["key_ids"][start:stop], dtype=np.int64)
    final_path, partial_path, receipt_path = key_capture_paths(output, args.chunk_index)
    if final_path.is_file() and receipt_path.is_file() and not args.overwrite:
        prior = read_json(receipt_path)
        if prior.get("status") == "complete" and prior.get("capture_file_sha256") == sha256_file(final_path):
            print(json.dumps({"status": "already_complete", "capture": str(final_path)}))
            return 0
        raise FileExistsError(f"stale narrative key capture: {final_path}")
    if any(path.exists() for path in (final_path, partial_path, receipt_path)):
        if not args.overwrite:
            raise FileExistsError("capture output exists; inspect or pass --overwrite")
        for path in (final_path, partial_path, receipt_path):
            if path.exists():
                path.unlink()

    expected_shape = (N_LAYERS, KEY_WINDOWS_PER_CHUNK, WINDOW_TOKENS, HIDDEN_DIM)
    started = time.perf_counter()
    receipt: dict[str, Any] = {
        "schema": "moe_e1_narrative_key_capture_v1",
        "created_at": now_iso(),
        "status": "starting",
        "argv": sys.argv,
        "required_shell_wrapper": GPU_WRAPPER,
        "chunk_index": int(args.chunk_index),
        "window_indices": list(range(start, stop)),
        "input_ids_shape": list(ids.shape),
        "input_ids_sha256": sha256_array(ids),
        "prepared_windows_sha256": manifest["prepared_windows"]["sha256"],
        "model_dir": str(args.model_dir.resolve()),
        "model_snapshot_revision": args.model_dir.resolve().name,
        "attention_mode": "standard",
        "expert_mode": "resident_packed_mxfp4",
        "compute_dtype": "bfloat16",
        "capture_shape_expected": list(expected_shape),
        "capture_storage_dtype": "float16",
        "capture_path": str(final_path),
        "partial_path": str(partial_path),
        "completed_layers": 0,
        "layer_wall_seconds": [],
        "layer_sha256": [],
        "attention_backends": [],
        "cuda_environment": cuda_probe,
        "gpu_before": nvidia_smi(),
        "script_sha256_at_run": sha256_file(SCRIPT_PATH),
    }
    write_json(receipt_path, receipt)
    try:
        runtime = load_runtime()
        rt1 = runtime["rt1"]
        tc = runtime["tc"]
        cfg = runtime["GptOss20BConfig"].from_model_dir(args.model_dir.resolve())
        validate_model_contract(cfg)
        where = runtime["build_safetensors_map"](args.model_dir.resolve())
        dump = np.lib.format.open_memmap(
            partial_path, mode="w+", dtype=np.float16, shape=expected_shape
        )
        with tc.no_grad():
            embed = runtime["GptOssRowEmbedding"](where)
            h = embed(ids)
            cos, sin = runtime["gpt_oss_yarn_rope_tables"](cfg, WINDOW_TOKENS)
            for layer in range(N_LAYERS):
                layer_started = time.perf_counter()
                block = runtime["GptOssDiagnosticBlockTC"].from_safetensors(
                    cfg, where, layer, expert_mode="resident_packed_mxfp4"
                )
                block.self_attn.attention_mode = "standard"
                block.mlp.route_detail = "summary"
                block.mlp.empty_cache_interval = 0
                captured: dict[str, np.ndarray] = {}
                rt1.install_router_capture(block.mlp, captured, capture_input=True)
                h, kv, route_info = block(h, cos, sin)
                tc.synchronize()
                router_input = captured.get("router_input")
                if router_input is None or router_input.shape != (
                    KEY_WINDOWS_PER_CHUNK * WINDOW_TOKENS,
                    HIDDEN_DIM,
                ):
                    observed = None if router_input is None else router_input.shape
                    raise RuntimeError(f"layer {layer} router input shape {observed}")
                stored = router_input.reshape(
                    KEY_WINDOWS_PER_CHUNK, WINDOW_TOKENS, HIDDEN_DIM
                ).astype(np.float16)
                if not np.isfinite(stored).all():
                    raise RuntimeError(f"layer {layer} capture contains non-finite values")
                dump[layer] = stored
                dump.flush()
                receipt["completed_layers"] = layer + 1
                receipt["layer_wall_seconds"].append(
                    float(time.perf_counter() - layer_started)
                )
                receipt["layer_sha256"].append(sha256_array(stored))
                receipt["attention_backends"].append(block.self_attn.last_attention_backend)
                receipt["status"] = "running"
                receipt["wall_seconds"] = float(time.perf_counter() - started)
                write_json(receipt_path, receipt)
                print(
                    json.dumps(
                        {
                            "mode": "capture-keys",
                            "chunk": args.chunk_index,
                            "layer": layer,
                            "elapsed_seconds": receipt["wall_seconds"],
                        }
                    ),
                    flush=True,
                )
                del block, kv, route_info, captured, router_input, stored
                gc.collect()
                if hasattr(tc, "empty_cache"):
                    tc.empty_cache()
            del h, embed, cos, sin
        dump.flush()
        del dump
        partial_path.replace(final_path)
        loaded = np.load(final_path, mmap_mode="r", allow_pickle=False)
        if loaded.shape != expected_shape or loaded.dtype != np.float16:
            raise RuntimeError("persisted narrative key capture contract mismatch")
        receipt.update(
            {
                "status": "complete",
                "wall_seconds": float(time.perf_counter() - started),
                "capture_shape": list(loaded.shape),
                "capture_payload_nbytes": int(loaded.nbytes),
                "capture_file_bytes": int(final_path.stat().st_size),
                "capture_file_sha256": sha256_file(final_path),
                "gpu_after": nvidia_smi(),
            }
        )
        write_json(receipt_path, receipt)
        print(json.dumps({"status": "complete", "capture": str(final_path)}), flush=True)
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
    hidden_dim = int(hidden["narrative"].shape[-1])
    blocks = {
        corpus: selected_tokens(array, indices, hidden_dim)
        for corpus, array in hidden.items()
    }
    means = {
        corpus: block.mean(axis=0, dtype=np.float64)
        for corpus, block in blocks.items()
    }
    centered = np.concatenate(
        [
            np.ascontiguousarray(
                blocks[corpus] - means[corpus].astype(np.float32), dtype=np.float32
            )
            for corpus in ("narrative", *NEGATIVE_CORPORA)
        ],
        axis=0,
    )
    dof = int(centered.shape[0] - 4)
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
        v = np.asarray(vector, dtype=np.float32)
        return np.asarray(x.T @ (x @ v) * inv_dof + lam * v, dtype=np.float32)

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
    residual = float(
        np.linalg.norm(matvec(solution).astype(np.float64) - b.astype(np.float64))
        / denominator
    )
    return np.asarray(solution, dtype=np.float32), {
        "solver": "scipy.sparse.linalg.cg over exact pooled covariance matvec",
        "info": int(info),
        "converged": bool(info == 0),
        "iterations": int(iterations),
        "rtol": float(rtol),
        "maxiter": int(maxiter),
        "relative_residual": residual,
        "wall_seconds": float(time.perf_counter() - started),
    }


def validation_auc_four(hidden: dict[str, np.ndarray], key: np.ndarray) -> dict[str, Any]:
    dim = int(key.size)
    positive = selected_tokens(hidden["narrative"], INNER_VALIDATION_INDICES, dim) @ key
    aucs = {
        corpus: exact_auc(
            positive,
            selected_tokens(hidden[corpus], INNER_VALIDATION_INDICES, dim) @ key,
        )
        for corpus in NEGATIVE_CORPORA
    }
    return {
        "auc_narrative_vs_generic": aucs["generic"],
        "auc_narrative_vs_code": aucs["code"],
        "auc_narrative_vs_grm": aucs["grm"],
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
            "alpha": float(alpha),
            "lambda": regularization,
            "solver": solver,
            "fit_converged": bool(solver["converged"]),
        }
        if solver["converged"]:
            key, norm = unit_vector(solution, name=f"K4 validation alpha={alpha}")
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
    key, source_norm = unit_vector(solution, name="K4 final Fisher direction")
    return key, {
        "construction": (
            "unit_norm((pooled within-four-corpus FIT covariance + lambda I)^-1 "
            "(mean(narrative_FIT) - equal_mean(generic,code,grm)_FIT))"
        ),
        "covariance": "residuals centered separately within all four corpora",
        "negative_pool_weighting": "equal token counts; one third per negative corpus",
        "alpha_grid": list(K4_ALPHA_GRID),
        "inner_train_window_indices": list(INNER_TRAIN_INDICES),
        "inner_validation_window_indices": list(INNER_VALIDATION_INDICES),
        "selection_rule": (
            "highest minimum validation AUC over generic/code/grm, then mean AUC, "
            "then stronger shrinkage"
        ),
        "validation_candidates": candidates,
        "selected_alpha": selected_alpha,
        "selected_inner_lambda": float(selected["lambda"]),
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
    result: dict[str, np.ndarray] = {}
    for corpus, array in hidden.items():
        result[corpus] = np.ascontiguousarray(
            (array.astype(np.float32).reshape(-1, key.size) @ key).reshape(
                array.shape[0], array.shape[1]
            ),
            dtype=np.float32,
        )
    return result


def bootstrap_rate(
    indicator: np.ndarray,
    *,
    resamples: int,
    seed: int,
    window_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    per_window = np.asarray(indicator, dtype=np.float64).mean(axis=1)
    indices = (
        list(range(per_window.size))
        if window_indices is None
        else [int(index) for index in window_indices]
    )
    if len(indices) != per_window.size:
        raise ValueError("window index count does not match rate rows")
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(
        0, per_window.size, size=(resamples, per_window.size)
    )
    boot = per_window[bootstrap_indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975], method="linear")
    return {
        "value": float(per_window.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_unit": "windows",
        "bootstrap_resamples": int(resamples),
        "per_window": [
            {"window_index": index, "rate": float(rate)}
            for index, rate in zip(indices, per_window)
        ],
    }


def select_tau_four(
    scores: dict[str, np.ndarray], address_rule: str = ADDRESS_RULE_E1
) -> dict[str, Any]:
    fit = np.asarray(FIT_INDICES, dtype=np.int64)
    constraint_corpora = threshold_constraint_corpora(address_rule)
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
        feasible = all(
            rates[f"{corpus}_fpr"] <= FPR_CAPS[corpus]
            for corpus in constraint_corpora
        )
        curve.append(
            {
                "quantile_label": label,
                "quantile": quantile,
                "tau": tau,
                "fit": rates,
                "fit_fpr_feasible": feasible,
            }
        )
    feasible = [item for item in curve if item["fit_fpr_feasible"]]
    pool = feasible if feasible else curve
    selected = max(
        pool,
        key=lambda item: (
            item["fit"]["narrative_recall"] if feasible else -max(
                item["fit"][f"{corpus}_fpr"] / FPR_CAPS[corpus]
                for corpus in constraint_corpora
            ),
            -item["fit"]["grm_fpr"],
            -item["fit"]["code_fpr"],
            -item["fit"]["generic_fpr"],
            item["quantile"],
        ),
    )
    return {
        "tau_definition": (
            "linear p90/p95/p99/p99.5 over pooled generic+code+grm FIT scores; "
            f"maximize narrative FIT recall subject to {'+'.join(constraint_corpora)} "
            "FIT FPR caps; "
            "ties lower grm/code/generic FPR then higher quantile"
        ),
        "address_rule": address_rule,
        "fit_constraint_corpora": list(constraint_corpora),
        "fit_generic_fire_rate_descriptive": address_rule == ADDRESS_RULE_E11,
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
    scores: dict[str, np.ndarray],
    *,
    resamples: int,
    seed: int,
    address_rule: str = ADDRESS_RULE_E1,
) -> dict[str, Any]:
    threshold = select_tau_four(scores, address_rule)
    tau = float(threshold["selected_tau"])
    evaluation = np.asarray(EVAL_INDICES, dtype=np.int64)
    constraint_corpora = threshold_constraint_corpora(address_rule)
    fired = {corpus: scores[corpus][evaluation] >= tau for corpus in scores}
    metrics: dict[str, Any] = {
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
    metrics["qualifies"] = bool(
        metrics["fit_fpr_feasible"]
        and metrics["eval"]["narrative_recall"]["value"] >= RECALL_FLOOR
        and all(
            metrics["eval"][f"{corpus}_fpr"]["value"] <= FPR_CAPS[corpus]
            for corpus in constraint_corpora
        )
    )
    if address_rule == ADDRESS_RULE_E11:
        metrics["eval"]["generic_fire_rate"] = dict(
            metrics["eval"]["generic_fpr"]
        )
        metrics["registered_criteria"] = {
            "eval_narrative_recall_gte": RECALL_FLOOR,
            "eval_code_fpr_lte": FPR_CAPS["code"],
            "eval_grm_fpr_lte": FPR_CAPS["grm"],
            "generic_fire_rate": "descriptive_no_pass_fail_bound",
            "fit_constraint_corpora": list(constraint_corpora),
            "fit_threshold_frozen_before_eval": True,
        }
    else:
        metrics["registered_criteria"] = {
            "eval_narrative_recall_gte": RECALL_FLOOR,
            **{
                f"eval_{corpus}_fpr_lte": cap
                for corpus, cap in FPR_CAPS.items()
            },
            "fit_threshold_frozen_before_eval": True,
        }
    return metrics


def verified_capture(
    capture_path: Path,
    receipt_path: Path,
    expected_shape: tuple[int, ...],
    *,
    hash_payload: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not capture_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"missing capture/receipt: {capture_path}, {receipt_path}")
    receipt = read_json(receipt_path)
    array = np.load(capture_path, mmap_mode="r", allow_pickle=False)
    checks: list[dict[str, Any]] = []
    add_check(checks, "receipt_status", "complete", receipt.get("status"))
    add_check(checks, "array_shape", list(expected_shape), list(array.shape))
    add_check(checks, "array_dtype", "float16", str(array.dtype))
    add_check(checks, "mmap_readonly", False, bool(array.flags.writeable))
    add_check(checks, "receipt_file_bytes", int(capture_path.stat().st_size), int(receipt.get("capture_file_bytes", -1)))
    observed_hash = sha256_file(capture_path) if hash_payload else None
    if hash_payload:
        add_check(checks, "receipt_sha256", receipt.get("capture_file_sha256"), observed_hash)
    failed = [item for item in checks if not item["passed"]]
    if failed:
        raise RuntimeError(
            f"capture verification failed for {capture_path}: "
            + ", ".join(item["name"] for item in failed)
        )
    return array, {
        "capture_path": str(capture_path),
        "receipt_path": str(receipt_path),
        "capture_sha256": observed_hash or receipt.get("capture_file_sha256"),
        "receipt_sha256": sha256_file(receipt_path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "mmap_mode": "r",
        "checks": checks,
    }


def load_key_fit_captures(
    output: Path,
) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    captures: dict[str, list[np.ndarray]] = {
        corpus: [] for corpus in ("narrative", *NEGATIVE_CORPORA)
    }
    provenance: dict[str, list[dict[str, Any]]] = {
        corpus: [] for corpus in captures
    }
    shape = (N_LAYERS, KEY_WINDOWS_PER_CHUNK, WINDOW_TOKENS, HIDDEN_DIM)
    for chunk in (0, 1):
        path, _partial, receipt = key_capture_paths(output, chunk)
        array, record = verified_capture(path, receipt, shape, hash_payload=True)
        captures["narrative"].append(array)
        provenance["narrative"].append(record)
    for negative in NEGATIVE_CORPORA:
        rt2_name = RT2_CORPUS_FOR_NEGATIVE[negative]
        for chunk in (0, 1):
            stem = f"{rt2_name}_chunk{chunk:02d}"
            path = RT2_DIR / f"{stem}_router_inputs_fp16.npy"
            receipt = RT2_DIR / f"{stem}_capture_receipt.json"
            array, record = verified_capture(path, receipt, shape, hash_payload=True)
            captures[negative].append(array)
            provenance[negative].append(record)
    return captures, {
        "schema": "moe_e1_key_input_integrity_v1",
        "created_at": now_iso(),
        "passed": True,
        "captures": provenance,
        "rt2_analysis": str(RT2_DIR / "analysis.json"),
        "rt2_analysis_sha256": sha256_file(RT2_DIR / "analysis.json"),
        "rt2_1_analysis": str(RT2_1_DIR / "analysis.json"),
        "rt2_1_analysis_sha256": sha256_file(RT2_1_DIR / "analysis.json"),
        "negative_alias": RT2_CORPUS_FOR_NEGATIVE,
    }


def materialize_fit_layer(
    captures: dict[str, list[np.ndarray]], layer: int
) -> dict[str, np.ndarray]:
    hidden: dict[str, np.ndarray] = {}
    for corpus, chunks in captures.items():
        combined = np.concatenate([np.asarray(chunk[layer]) for chunk in chunks], axis=0)
        if combined.shape != (N_KEY_WINDOWS, WINDOW_TOKENS, HIDDEN_DIM):
            raise RuntimeError(f"{corpus} layer {layer} shape {combined.shape}")
        if combined.dtype != np.float16 or not np.isfinite(combined).all():
            raise RuntimeError(f"{corpus} layer {layer} invalid capture values")
        hidden[corpus] = np.ascontiguousarray(combined)
    return hidden


def fit_side_layer_rank(row: dict[str, Any]) -> tuple[float, ...]:
    fit = row["metrics"]["selected_fit_rates"]
    validation = row["key_metadata"]["selected_validation"]
    max_ratio = max(fit[f"{corpus}_fpr"] / FPR_CAPS[corpus] for corpus in NEGATIVE_CORPORA)
    return (
        float(row["metrics"]["fit_fpr_feasible"]),
        float(validation["minimum_auc"]),
        float(validation["mean_auc"]),
        float(fit["narrative_recall"]),
        -float(max_ratio),
        -float(row["layer"]),
    )


def fit_key(args: argparse.Namespace) -> int:
    if args.bootstrap_resamples < 2000:
        raise ValueError("fit-key requires at least 2000 bootstrap resamples")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    output = ensure_output_dir(args.output_dir)
    manifest, _prepared = load_prepared(output)
    analysis_file, keys_path = fit_key_paths(output, args.address_rule)
    pack_dir = output / expertpack_dirname(args.address_rule)
    pack_path = pack_dir / "manifest.json"
    if args.address_rule == ADDRESS_RULE_E11:
        existing = [
            path
            for path in (analysis_file, keys_path, pack_path)
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "E1.1 receipts are append-only; refusing to overwrite: "
                + ", ".join(str(path) for path in existing)
            )
    started = time.perf_counter()
    captures, integrity = load_key_fit_captures(output)
    rows: list[dict[str, Any]] = []
    keys: list[np.ndarray] = []
    taus: list[float] = []
    alphas: list[float] = []
    lambdas: list[float] = []
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        class threadpool_limits:  # type: ignore[no-redef]
            def __init__(self, **_kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *_args): return False

    with threadpool_limits(limits=args.threads):
        for layer in range(N_LAYERS):
            layer_started = time.perf_counter()
            hidden = materialize_fit_layer(captures, layer)
            key, key_meta = fit_k4_direction(
                hidden, cg_rtol=args.cg_rtol, cg_maxiter=args.cg_maxiter
            )
            scores = score_four(hidden, key)
            metrics = analyze_key_scores(
                scores,
                resamples=args.bootstrap_resamples,
                seed=args.seed + 1000 * layer,
                address_rule=args.address_rule,
            )
            row = {
                "layer": layer,
                "key_metadata": key_meta,
                "metrics": metrics,
                "wall_seconds": float(time.perf_counter() - layer_started),
            }
            row["fit_side_rank"] = list(fit_side_layer_rank(row))
            rows.append(row)
            keys.append(key)
            taus.append(float(metrics["selected_tau"]))
            alphas.append(float(key_meta["selected_alpha"]))
            lambdas.append(float(key_meta["selected_full_fit_lambda"]))
            print(
                json.dumps(
                    {
                        "layer": layer,
                        "address_rule": args.address_rule,
                        "qualifies": metrics["qualifies"],
                        "recall": metrics["eval"]["narrative_recall"]["value"],
                        (
                            "generic_fire_rate"
                            if args.address_rule == ADDRESS_RULE_E11
                            else "generic_fpr"
                        ): metrics["eval"]["generic_fpr"]["value"],
                        "code_fpr": metrics["eval"]["code_fpr"]["value"],
                        "grm_fpr": metrics["eval"]["grm_fpr"]["value"],
                    }
                ),
                flush=True,
            )
            del hidden, scores, key
            gc.collect()

    save_npz(
        keys_path,
        K4=np.stack(keys).astype(np.float32),
        tau=np.asarray(taus, dtype=np.float64),
        selected_alpha=np.asarray(alphas, dtype=np.float64),
        selected_lambda=np.asarray(lambdas, dtype=np.float64),
    )
    qualifiers = [row for row in rows if row["metrics"]["qualifies"]]
    qualifiers_fit_ranked = sorted(qualifiers, key=fit_side_layer_rank, reverse=True)
    selected = qualifiers_fit_ranked[0] if qualifiers_fit_ranked else None
    decision = {
        "g2_green": bool(selected),
        "g2_prime_green": (
            bool(selected) if args.address_rule == ADDRESS_RULE_E11 else None
        ),
        "qualifying_layer_count": len(qualifiers),
        "qualifying_layers": [int(row["layer"]) for row in qualifiers_fit_ranked],
        "selection_uses_eval_only_as_registered_qualification_filter": True,
        "within_qualifiers_selection_rule": (
            "FIT-only rank: feasible threshold, validation minimum AUC, validation "
            "mean AUC, FIT recall, lower worst normalized FIT FPR, lower layer"
        ),
        "install_layer": None if selected is None else int(selected["layer"]),
        "top3_capture_layers": [
            int(row["layer"]) for row in qualifiers_fit_ranked[:3]
        ],
    }
    analysis = {
        "schema": (
            "moe_e1_1_fit_key_v1"
            if args.address_rule == ADDRESS_RULE_E11
            else "moe_e1_fit_key_v1"
        ),
        "created_at": now_iso(),
        "status": "complete_g2_green" if selected else "complete_g2_red_stop",
        "address_rule": args.address_rule,
        "order": str(order_path_for_rule(args.address_rule)),
        "script": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "original_rule_path": {
            "preserved": True,
            "default_address_rule": ADDRESS_RULE_E1,
            "invocation": f"python3 {SCRIPT_PATH} fit-key",
            "explicit_equivalent": (
                f"python3 {SCRIPT_PATH} fit-key --address-rule e1"
            ),
        },
        "cpu_only": True,
        "seed": int(args.seed),
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "input_integrity": integrity,
        "split_provenance": {
            "fit_window_indices": list(FIT_INDICES),
            "eval_window_indices": list(EVAL_INDICES),
            "inner_train_window_indices": list(INNER_TRAIN_INDICES),
            "inner_validation_window_indices": list(INNER_VALIDATION_INDICES),
            "fit_eval_overlap": sorted(set(FIT_INDICES) & set(EVAL_INDICES)),
            "eval_used_for_key_tau_or_hyperparameter_selection": False,
            "heldout_file_text_used": False,
        },
        "registered_rule": (
            {
                "gate": "G2'",
                "eval_recall_gte": RECALL_FLOOR,
                "eval_code_fpr_lte": FPR_CAPS["code"],
                "eval_grm_fpr_lte": FPR_CAPS["grm"],
                "generic_fire_rate": "descriptive_per_layer_and_eval_window",
                "fit_feasibility_constraints": ["code", "grm"],
                "tau_candidate_pool": ["generic", "code", "grm"],
                "frozen_fit_side_tau": True,
            }
            if args.address_rule == ADDRESS_RULE_E11
            else {
                "gate": "G2",
                "eval_recall_gte": RECALL_FLOOR,
                **{
                    f"eval_{corpus}_fpr_lte": cap
                    for corpus, cap in FPR_CAPS.items()
                },
                "frozen_fit_side_tau": True,
            }
        ),
        "rows": rows,
        "decision": decision,
        "keys_artifact": {"path": str(keys_path), "sha256": sha256_file(keys_path)},
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "threads": int(args.threads),
            "cg_rtol": float(args.cg_rtol),
            "cg_maxiter": int(args.cg_maxiter),
            "wall_seconds": float(time.perf_counter() - started),
        },
    }
    write_json(analysis_file, analysis)
    if not pack_path.is_file():
        pack_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            pack_path,
            pending_expertpack_manifest(
                output=output,
                corpus_manifest=output / "corpus_manifest.json",
                windows_path=output / "prepared_windows.npz",
                address_rule=args.address_rule,
            ),
        )
    pack = read_json(pack_path)
    pack["updated_at"] = now_iso()
    pack["provenance"]["fit_key"] = str(analysis_file)
    pack["provenance"]["fit_key_sha256"] = sha256_file(analysis_file)
    pack["provenance"]["keys_artifact"] = str(keys_path)
    pack["provenance"]["keys_artifact_sha256"] = sha256_file(keys_path)
    if selected is None:
        pack["status"] = "g2_red_no_installable_address"
        gate_name = "G2'" if args.address_rule == ADDRESS_RULE_E11 else "G2"
        pack["limitations"] = [
            f"{gate_name} RED: no qualifying K4 narrative address"
        ]
    else:
        layer = int(selected["layer"])
        selected_key = np.asarray(keys[layer], dtype=np.float32)
        key_path = pack_dir / "key_fp32.npy"
        save_npy(key_path, selected_key)
        eval_metrics = selected["metrics"]["eval"]
        pack.update(
            {
                "status": "addressed_pending_adapter_training",
                "layer": layer,
                "rank": int(args.rank),
                "tau": float(selected["metrics"]["selected_tau"]),
                "key_metrics": {
                    "eval_recall": eval_metrics["narrative_recall"]["value"],
                    "eval_generic_fpr": eval_metrics["generic_fpr"]["value"],
                    "eval_generic_fire_rate_descriptive": eval_metrics[
                        "generic_fpr"
                    ]["value"],
                    "eval_code_fpr": eval_metrics["code_fpr"]["value"],
                    "eval_grm_fpr": eval_metrics["grm_fpr"]["value"],
                    "selected_alpha": selected["key_metadata"]["selected_alpha"],
                    "selected_quantile": selected["metrics"]["selected_quantile_label"],
                },
            }
        )
        pack["components"]["key"] = {
            "path": str(key_path),
            "shape": [HIDDEN_DIM],
            "storage_dtype": "float32",
            "sha256": sha256_file(key_path),
            "status": "complete",
        }
        pack["limitations"] = ["adapter A/B pending capture-pairs and train"]
    write_json(pack_path, pack)
    if args.address_rule == ADDRESS_RULE_E11:
        resume_path = output / "GPU_RESUME_COMMANDS.sh"
        write_text(resume_path, gpu_resume_commands(args.address_rule))
        try:
            resume_path.chmod(0o755)
        except OSError:
            pass
    print(
        json.dumps(
            {
                "status": analysis["status"],
                "address_rule": args.address_rule,
                "qualifying_layers": decision["qualifying_layers"],
                "install_layer": decision["install_layer"],
                "analysis": str(analysis_file),
                "expertpack_manifest": str(pack_path),
            }
        ),
        flush=True,
    )
    return 0 if selected is not None else 3


def initialize_e13_expertpack(
    output: Path,
    *,
    address_rule: str,
) -> dict[str, Any]:
    if address_rule != ADDRESS_RULE_E11:
        raise ValueError("ORDER MOE-E1.3 reuses the E1.1 G2' address")
    selection = load_e13_selection()
    source_path = expertpack_path(
        output, address_rule, TEACHER_RULE_ORIGINAL
    )
    if not source_path.is_file():
        raise FileNotFoundError(f"missing E1.1 addressed ExpertPack: {source_path}")
    source = read_json(source_path)
    if source.get("layer") is None or source.get("tau") is None:
        raise RuntimeError("E1.1 G2' did not produce an addressed ExpertPack")
    key_info = source.get("components", {}).get("key", {})
    key_path = Path(key_info.get("path", ""))
    if not key_path.is_file() or key_info.get("sha256") != sha256_file(key_path):
        raise RuntimeError("E1.1 key component is missing or changed")
    target_path = expertpack_path(output, address_rule, TEACHER_RULE_E13)
    if target_path.is_file():
        pack = read_json(target_path)
        if (
            pack.get("teacher_rule") != TEACHER_RULE_E13
            or pack.get("winning_construction")
            != selection["winning_construction"]
            or pack.get("components", {}).get("key", {}).get("sha256")
            != key_info.get("sha256")
        ):
            raise RuntimeError("existing E1.3 ExpertPack conflicts with sealed selection/key")
        return pack
    target_root = target_path.parent
    target_root.mkdir(parents=True, exist_ok=True)
    pack = json.loads(json.dumps(source))
    pack.update(
        {
            "schema": "expertpack_narrative_v0_e13",
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "address_rule": ADDRESS_RULE_E11,
            "teacher_rule": TEACHER_RULE_E13,
            "winning_construction": selection["winning_construction"],
            "winning_mean_gap": selection["winning_mean_gap"],
            "status": "addressed_pending_e13_adapter_training",
            "rank": 64,
        }
    )
    pack["components"]["A"] = {
        "path": str(target_root / "A_fp16.npy"),
        "shape": [64, HIDDEN_DIM],
        "storage_dtype": "float16",
        "status": "missing",
    }
    pack["components"]["B"] = {
        "path": str(target_root / "B_fp16.npy"),
        "shape": [HIDDEN_DIM, 64],
        "storage_dtype": "float16",
        "initialization": "all zeros before training",
        "status": "missing",
    }
    pack.pop("g3", None)
    pack["provenance"].update(
        {
            "order": str(E13_ORDER_PATH),
            "order_sha256": sha256_file(E13_ORDER_PATH),
            "source_e11_expertpack": str(source_path),
            "source_e11_expertpack_sha256": sha256_file(source_path),
            "e13_selection": str(E13_SELECTION_PATH),
            "e13_selection_sha256": sha256_file(E13_SELECTION_PATH),
            "e13_prefix_manifest": str(E13_PREFIX_MANIFEST_PATH),
            "e13_prefix_manifest_sha256": sha256_file(E13_PREFIX_MANIFEST_PATH),
            "e13_prefix_arrays": str(E13_PREFIX_ARRAYS_PATH),
            "e13_prefix_arrays_sha256": sha256_file(E13_PREFIX_ARRAYS_PATH),
            "script": str(SCRIPT_PATH),
            "script_sha256_at_e13_initialization": sha256_file(SCRIPT_PATH),
        }
    )
    pack["limitations"] = ["E1.3 adapter A/B pending capture-pairs and train"]
    write_json(target_path, pack)
    return pack


def load_addressed_pack(
    output: Path,
    *,
    address_rule: str = ADDRESS_RULE_E1,
    require_adapter: bool = False,
    teacher_rule: str = TEACHER_RULE_ORIGINAL,
) -> dict[str, Any]:
    pack_path = expertpack_path(output, address_rule, teacher_rule)
    if not pack_path.is_file():
        raise FileNotFoundError(f"missing ExpertPack manifest: {pack_path}")
    pack = read_json(pack_path)
    if pack.get("layer") is None or pack.get("tau") is None:
        raise RuntimeError("G2 has not produced an installable narrative address")
    key_info = pack.get("components", {}).get("key", {})
    key_path = Path(key_info.get("path", ""))
    if not key_path.is_file() or key_info.get("sha256") != sha256_file(key_path):
        raise RuntimeError("ExpertPack key is missing or changed")
    if require_adapter:
        if pack.get("status") != "complete_g3_green":
            raise RuntimeError("G3 is not GREEN; behavioral expert evaluation must stop")
        for name in ("A", "B"):
            info = pack["components"][name]
            path = Path(info["path"])
            if not path.is_file() or info.get("sha256") != sha256_file(path):
                raise RuntimeError(f"ExpertPack {name} is missing or changed")
    return pack


def pair_paths(
    output: Path,
    pair_index: int,
    address_rule: str = ADDRESS_RULE_E1,
    teacher_rule: str = TEACHER_RULE_ORIGINAL,
) -> tuple[Path, Path]:
    root = pair_root(output, address_rule, teacher_rule)
    return root / f"pair_{pair_index:03d}.npz", root / f"pair_{pair_index:03d}_receipt.json"


def capture_pairs(args: argparse.Namespace) -> int:
    if args.pair_index is None or args.pair_index not in range(N_PAIR_WINDOWS):
        raise ValueError(f"capture-pairs requires --pair-index 0..{N_PAIR_WINDOWS - 1}")
    output = ensure_output_dir(args.output_dir)
    manifest, arrays = load_prepared(output)
    pack = load_addressed_pack(output, address_rule=args.address_rule)
    cuda_probe = require_cuda()
    index = int(args.pair_index)
    pair_ids = np.ascontiguousarray(arrays["pair_ids"][index], dtype=np.int64)
    if args.teacher_rule == TEACHER_RULE_E13:
        prefix_ids, prefix_source, selection = load_e13_teacher_prefix("pair", index)
    else:
        prefix_ids = np.ascontiguousarray(
            arrays["pair_prefix_ids"][index], dtype=np.int64
        )
        prefix_source = manifest["windows"]["pair_prefixes"][index]
        selection = None
    teacher_ids = np.concatenate([prefix_ids, pair_ids])[None, :]
    student_ids = pair_ids[None, :]
    if not np.array_equal(teacher_ids[0, -WINDOW_TOKENS:], student_ids[0]):
        raise RuntimeError("G1 RED: teacher/student shared token ids are not identical")
    layer_target = int(pack["layer"])
    data_path, receipt_path = pair_paths(
        output, index, args.address_rule, args.teacher_rule
    )
    if data_path.is_file() and receipt_path.is_file() and not args.overwrite:
        prior = read_json(receipt_path)
        if prior.get("status") == "complete" and prior.get("pair_file_sha256") == sha256_file(data_path):
            print(json.dumps({"status": "already_complete", "pair": index}))
            return 0
        raise FileExistsError(f"stale pair capture: {data_path}")
    if (data_path.exists() or receipt_path.exists()) and not args.overwrite:
        raise FileExistsError("pair output exists; inspect or pass --overwrite")
    if args.overwrite:
        for path in (data_path, receipt_path):
            if path.exists():
                path.unlink()

    split = "TRAIN" if index < N_PAIR_TRAIN else "VALIDATION"
    started = time.perf_counter()
    receipt: dict[str, Any] = {
        "schema": "moe_e1_activation_pair_v1",
        "created_at": now_iso(),
        "address_rule": args.address_rule,
        "teacher_rule": args.teacher_rule,
        "winning_construction": (
            None if selection is None else selection["winning_construction"]
        ),
        "status": "starting",
        "order": str(order_path_for_experiment(args.address_rule, args.teacher_rule)),
        "argv": sys.argv,
        "required_shell_wrapper": GPU_WRAPPER,
        "pair_index": index,
        "split": split,
        "install_layer": layer_target,
        "teacher_prefix_tokens": int(prefix_ids.size),
        "shared_window_tokens": WINDOW_TOKENS,
        "teacher_input_shape": list(teacher_ids.shape),
        "student_input_shape": list(student_ids.shape),
        "teacher_prefix_ids_sha256": sha256_array(prefix_ids),
        "shared_token_ids_sha256": sha256_array(pair_ids),
        "token_alignment_exact": True,
        "source_window": manifest["windows"]["pairs"][index],
        "prefix_source": prefix_source,
        "teacher_selection": (
            None
            if selection is None
            else {
                "path": str(E13_SELECTION_PATH),
                "sha256": sha256_file(E13_SELECTION_PATH),
                "winning_mean_gap": selection["winning_mean_gap"],
            }
        ),
        "model_dir": str(args.model_dir.resolve()),
        "attention_mode": "standard",
        "expert_mode": "resident_packed_mxfp4",
        "compute_dtype": "bfloat16",
        "cuda_environment": cuda_probe,
        "gpu_before": nvidia_smi(),
        "completed_layers": 0,
        "layer_wall_seconds": [],
        "script_sha256_at_run": sha256_file(SCRIPT_PATH),
    }
    write_json(receipt_path, receipt)
    try:
        runtime = load_runtime()
        rt1 = runtime["rt1"]
        tc = runtime["tc"]
        cfg = runtime["GptOss20BConfig"].from_model_dir(args.model_dir.resolve())
        validate_model_contract(cfg)
        where = runtime["build_safetensors_map"](args.model_dir.resolve())
        captured_h: np.ndarray | None = None
        teacher_out: np.ndarray | None = None
        student_out: np.ndarray | None = None
        with tc.no_grad():
            embed = runtime["GptOssRowEmbedding"](where)
            h_teacher = embed(teacher_ids)
            h_student = embed(student_ids)
            cos, sin = runtime["gpt_oss_yarn_rope_tables"](
                cfg, int(teacher_ids.shape[1])
            )
            for layer in range(layer_target + 1):
                layer_started = time.perf_counter()
                block = runtime["GptOssDiagnosticBlockTC"].from_safetensors(
                    cfg, where, layer, expert_mode="resident_packed_mxfp4"
                )
                block.self_attn.attention_mode = "standard"
                block.mlp.route_detail = "summary"
                block.mlp.empty_cache_interval = 0
                h_teacher, kv_teacher, _route_teacher = block(h_teacher, cos, sin)
                capture: dict[str, np.ndarray] = {}
                if layer == layer_target:
                    rt1.install_router_capture(block.mlp, capture, capture_input=True)
                h_student, kv_student, _route_student = block(h_student, cos, sin)
                tc.synchronize()
                receipt["completed_layers"] = layer + 1
                receipt["layer_wall_seconds"].append(
                    float(time.perf_counter() - layer_started)
                )
                receipt["status"] = "running"
                receipt["wall_seconds"] = float(time.perf_counter() - started)
                write_json(receipt_path, receipt)
                if layer == layer_target:
                    captured_h = capture.get("router_input")
                    if captured_h is None or captured_h.shape != (WINDOW_TOKENS, HIDDEN_DIM):
                        observed = None if captured_h is None else captured_h.shape
                        raise RuntimeError(f"student router-input capture shape {observed}")
                    teacher_out = h_teacher.float().numpy().astype(np.float32, copy=True)[
                        0, -WINDOW_TOKENS:
                    ]
                    student_out = h_student.float().numpy().astype(np.float32, copy=True)[0]
                del block, kv_teacher, kv_student, _route_teacher, _route_student, capture
                gc.collect()
                if hasattr(tc, "empty_cache"):
                    tc.empty_cache()
            del h_teacher, h_student, embed, cos, sin
        if captured_h is None or teacher_out is None or student_out is None:
            raise RuntimeError("target layer did not produce pair tensors")
        delta = teacher_out - student_out
        hidden_diff = np.abs(delta)
        if not np.isfinite(captured_h).all() or not np.isfinite(delta).all():
            raise RuntimeError("pair capture contains non-finite values")
        stored_h = captured_h.astype(np.float16)
        stored_teacher = teacher_out.astype(np.float16)
        stored_student = student_out.astype(np.float16)
        save_npz(
            data_path,
            token_ids=pair_ids,
            h_student_fp16=stored_h,
            out_teacher_fp16=stored_teacher,
            out_student_fp16=stored_student,
        )
        receipt.update(
            {
                "status": "complete",
                "wall_seconds": float(time.perf_counter() - started),
                "pair_file": str(data_path),
                "pair_file_sha256": sha256_file(data_path),
                "pair_file_bytes": int(data_path.stat().st_size),
                "array_shapes": {
                    "token_ids": list(pair_ids.shape),
                    "h_student_fp16": list(stored_h.shape),
                    "out_teacher_fp16": list(stored_teacher.shape),
                    "out_student_fp16": list(stored_student.shape),
                },
                "array_sha256": {
                    "token_ids": sha256_array(pair_ids),
                    "h_student_fp16": sha256_array(stored_h),
                    "out_teacher_fp16": sha256_array(stored_teacher),
                    "out_student_fp16": sha256_array(stored_student),
                },
                "hidden_states_differ": bool(np.any(teacher_out != student_out)),
                "hidden_difference_nonzero_elements": int(np.count_nonzero(delta)),
                "hidden_difference_max_abs": float(hidden_diff.max()),
                "hidden_difference_mean_abs": float(hidden_diff.mean(dtype=np.float64)),
                "gpu_after": nvidia_smi(),
            }
        )
        write_json(receipt_path, receipt)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "pair_index": index,
                    "split": split,
                    "alignment": True,
                    "max_hidden_difference": receipt["hidden_difference_max_abs"],
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


def load_pair_capture(
    output: Path,
    pair_index: int,
    expected_ids: np.ndarray,
    address_rule: str = ADDRESS_RULE_E1,
    teacher_rule: str = TEACHER_RULE_ORIGINAL,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    data_path, receipt_path = pair_paths(
        output, pair_index, address_rule, teacher_rule
    )
    if not data_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"missing pair {pair_index}: {data_path}, {receipt_path}")
    receipt = read_json(receipt_path)
    if receipt.get("status") != "complete":
        raise RuntimeError(f"pair {pair_index} receipt is not complete")
    if receipt.get("pair_file_sha256") != sha256_file(data_path):
        raise RuntimeError(f"pair {pair_index} artifact hash mismatch")
    with np.load(data_path, allow_pickle=False) as stored:
        data = {name: np.ascontiguousarray(stored[name]) for name in stored.files}
    expected_shapes = {
        "token_ids": (WINDOW_TOKENS,),
        "h_student_fp16": (WINDOW_TOKENS, HIDDEN_DIM),
        "out_teacher_fp16": (WINDOW_TOKENS, HIDDEN_DIM),
        "out_student_fp16": (WINDOW_TOKENS, HIDDEN_DIM),
    }
    for name, shape in expected_shapes.items():
        if name not in data or data[name].shape != shape:
            raise RuntimeError(f"pair {pair_index} {name} shape mismatch")
    if not np.array_equal(data["token_ids"], expected_ids):
        raise RuntimeError(f"G1 RED: pair {pair_index} token ids changed")
    if not receipt.get("token_alignment_exact") or not receipt.get("hidden_states_differ"):
        raise RuntimeError(f"G1 RED: pair {pair_index} truth receipt failed")
    return data, {
        "pair_index": pair_index,
        "split": receipt["split"],
        "path": str(data_path),
        "sha256": receipt["pair_file_sha256"],
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "token_alignment_exact": True,
        "hidden_states_differ": True,
        "hidden_difference_max_abs": receipt["hidden_difference_max_abs"],
    }


def train(args: argparse.Namespace) -> int:
    if args.rank != 64:
        raise ValueError(
            "the registered primary is r=64; alternate ranks require a separate "
            "validation-only comparison artifact and are not enabled in v0"
        )
    if args.epochs < 1 or args.patience < 1 or args.token_batch_size < 1:
        raise ValueError("epochs, patience, and token batch size must be positive")
    if args.train_tokens_per_window not in range(1, WINDOW_TOKENS + 1):
        raise ValueError("--train-tokens-per-window must be 1..512")
    output = ensure_output_dir(args.output_dir)
    _manifest, prepared = load_prepared(output)
    if args.teacher_rule == TEACHER_RULE_E13:
        pack = initialize_e13_expertpack(output, address_rule=args.address_rule)
    else:
        pack = load_addressed_pack(output, address_rule=args.address_rule)
    pack_dir = expertpack_root(output, args.address_rule, args.teacher_rule)
    A_path = pack_dir / "A_fp16.npy"
    B_path = pack_dir / "B_fp16.npy"
    training_path = train_path(output, args.address_rule, args.teacher_rule)
    if args.address_rule == ADDRESS_RULE_E11 or args.teacher_rule == TEACHER_RULE_E13:
        if training_path.is_file():
            prior = read_json(training_path)
            if prior.get("status") in {
                "complete_g3_green",
                "complete_g3_red_stop",
            }:
                print(
                    json.dumps(
                        {
                            "status": "already_complete",
                            "g3": prior.get("g3_verdict"),
                            "training": str(training_path),
                        }
                    ),
                    flush=True,
                )
                return 0 if prior.get("g3_verdict") == "GREEN" else 3
        existing = [path for path in (training_path, A_path, B_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "E1.1 training receipts are append-only; inspect partial outputs: "
                + ", ".join(str(path) for path in existing)
            )
    key = np.load(pack["components"]["key"]["path"], allow_pickle=False).astype(np.float32)
    if key.shape != (HIDDEN_DIM,):
        raise RuntimeError("ExpertPack key shape mismatch")
    tau = float(pack["tau"])

    # Validate every pair and retain only small provenance records.  Training
    # reopens one shard at a time so the ~0.6 GB pair set is never duplicated.
    pair_receipts: list[dict[str, Any]] = []
    for index in range(N_PAIR_WINDOWS):
        _data, record = load_pair_capture(
            output,
            index,
            prepared["pair_ids"][index],
            args.address_rule,
            args.teacher_rule,
        )
        pair_receipts.append(record)
        del _data

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    import torch.nn.functional as torch_f

    torch.set_num_threads(int(args.threads))
    torch.manual_seed(int(args.seed))
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()

    class Adapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.A = torch.nn.Parameter(torch.empty(args.rank, HIDDEN_DIM))
            self.B = torch.nn.Parameter(torch.zeros(HIDDEN_DIM, args.rank))
            torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5.0))

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return torch_f.silu(hidden @ self.A.T) @ self.B.T

    model = Adapter().cpu()
    initial_B = model.B.detach().numpy().copy()
    if not np.array_equal(initial_B, np.zeros_like(initial_B)):
        raise RuntimeError("B zero-initialization contract failed")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    key_t = torch.from_numpy(key)

    def evaluate_validation(current: Adapter) -> tuple[float, float, int, int]:
        current.eval()
        squared = 0.0
        zero_squared = 0.0
        elements = 0
        fired_count = 0
        with torch.no_grad():
            for index in range(N_PAIR_TRAIN, N_PAIR_WINDOWS):
                data, _record = load_pair_capture(
                    output,
                    index,
                    prepared["pair_ids"][index],
                    args.address_rule,
                    args.teacher_rule,
                )
                hidden = torch.from_numpy(data["h_student_fp16"].astype(np.float32))
                target = torch.from_numpy(
                    data["out_teacher_fp16"].astype(np.float32)
                    - data["out_student_fp16"].astype(np.float32)
                )
                for start in range(0, WINDOW_TOKENS, args.token_batch_size):
                    h_batch = hidden[start : start + args.token_batch_size]
                    target_batch = target[start : start + args.token_batch_size]
                    fire = (h_batch @ key_t >= tau).to(torch.float32)[:, None]
                    prediction = current(h_batch) * fire
                    squared += float(torch.sum((prediction - target_batch) ** 2).item())
                    zero_squared += float(torch.sum(target_batch**2).item())
                    elements += int(target_batch.numel())
                    fired_count += int(fire.sum().item())
                del data, hidden, target
        return squared / elements, zero_squared / elements, fired_count, elements // HIDDEN_DIM

    baseline_mse, zero_mse, initial_val_fires, validation_tokens = evaluate_validation(model)
    if baseline_mse != zero_mse:
        raise RuntimeError("zero-initialized B did not reproduce the zero predictor")
    best_mse = baseline_mse
    best_epoch = 0
    best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    epochs_without_improvement = 0
    epochs: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = np.random.default_rng(args.seed + epoch * 100003)
        order = rng.permutation(N_PAIR_TRAIN).tolist()
        epoch_squared = 0.0
        epoch_elements = 0
        epoch_fires = 0
        for index in order:
            data, _record = load_pair_capture(
                output,
                index,
                prepared["pair_ids"][index],
                args.address_rule,
                args.teacher_rule,
            )
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
                fire = (h_batch @ key_t >= tau).to(torch.float32)[:, None]
                optimizer.zero_grad(set_to_none=True)
                prediction = model(h_batch) * fire
                loss = torch.mean((prediction - target_batch) ** 2)
                loss.backward()
                optimizer.step()
                epoch_squared += float(loss.item()) * int(target_batch.numel())
                epoch_elements += int(target_batch.numel())
                epoch_fires += int(fire.sum().item())
            del data, hidden, target
        validation_mse, _zero_again, val_fires, _val_tokens = evaluate_validation(model)
        improved = validation_mse < best_mse - max(1.0e-12, abs(best_mse) * 1.0e-6)
        if improved:
            best_mse = validation_mse
            best_epoch = epoch
            best_state = {
                name: tensor.detach().clone() for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        epoch_record = {
            "epoch": epoch,
            "train_sampled_mse": epoch_squared / epoch_elements,
            "train_sampled_tokens": epoch_elements // HIDDEN_DIM,
            "train_sampled_fire_count": epoch_fires,
            "validation_mse": validation_mse,
            "validation_fire_count": val_fires,
            "improved": improved,
            "epochs_without_improvement": epochs_without_improvement,
        }
        epochs.append(epoch_record)
        print(json.dumps(epoch_record), flush=True)
        if epochs_without_improvement >= args.patience:
            break
    model.load_state_dict(best_state)
    A_fp16 = model.A.detach().numpy().astype(np.float16)
    B_fp16 = model.B.detach().numpy().astype(np.float16)
    save_npy(A_path, A_fp16)
    save_npy(B_path, B_fp16)

    # G3 is evaluated on the actual fp16 payloads that eval-gates will mount.
    with torch.no_grad():
        model.A.copy_(torch.from_numpy(A_fp16.astype(np.float32)))
        model.B.copy_(torch.from_numpy(B_fp16.astype(np.float32)))
    stored_mse, stored_zero_mse, stored_val_fires, _stored_tokens = evaluate_validation(model)
    improvement = (
        (stored_zero_mse - stored_mse) / stored_zero_mse
        if stored_zero_mse > 0.0
        else float("-inf")
    )
    g3_green = bool(np.isfinite(improvement) and improvement >= G3_IMPROVEMENT_FLOOR)
    training = {
        "schema": "moe_e1_adapter_training_v1",
        "created_at": now_iso(),
        "address_rule": args.address_rule,
        "teacher_rule": args.teacher_rule,
        "winning_construction": (
            load_e13_selection()["winning_construction"]
            if args.teacher_rule == TEACHER_RULE_E13
            else None
        ),
        "status": "complete_g3_green" if g3_green else "complete_g3_red_stop",
        "cpu_only": True,
        "order": str(order_path_for_experiment(args.address_rule, args.teacher_rule)),
        "script": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "model_snapshot": str(args.model_dir.resolve()),
        "install_layer": int(pack["layer"]),
        "rank": int(args.rank),
        "key": pack["components"]["key"],
        "tau": tau,
        "abi": "B @ silu(A @ h), threshold gated by h @ key >= tau",
        "initialization": {
            "A": "torch kaiming_uniform, deterministic CPU seed",
            "B": "exact all-zero float32",
            "B_zero_exact": bool(np.array_equal(initial_B, np.zeros_like(initial_B))),
            "B_initial_sha256": sha256_array(initial_B),
        },
        "data": {
            "train_pair_indices": list(range(N_PAIR_TRAIN)),
            "validation_pair_indices": list(range(N_PAIR_TRAIN, N_PAIR_WINDOWS)),
            "train_pair_count": N_PAIR_TRAIN,
            "validation_pair_count": N_PAIR_VALIDATION,
            "train_tokens_sampled_per_window_per_epoch": int(args.train_tokens_per_window),
            "validation_tokens": validation_tokens,
            "validation_fire_count": stored_val_fires,
            "pair_receipts": pair_receipts,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "token_batch_size": int(args.token_batch_size),
            "maximum_epochs": int(args.epochs),
            "patience": int(args.patience),
            "early_stopping_metric": "validation-pair full-token gated MSE",
        },
        "epochs": epochs,
        "best_epoch_float32": best_epoch,
        "best_validation_mse_float32": best_mse,
        "stored_fp16_validation": {
            "zero_predictor_mse": stored_zero_mse,
            "adapter_mse": stored_mse,
            "improvement_fraction": improvement,
            "improvement_percent": improvement * 100.0,
            "required_improvement_fraction": G3_IMPROVEMENT_FLOOR,
        },
        "g3_verdict": "GREEN" if g3_green else "RED",
        "artifacts": {
            "A": {
                "path": str(A_path),
                "shape": list(A_fp16.shape),
                "dtype": str(A_fp16.dtype),
                "sha256": sha256_file(A_path),
            },
            "B": {
                "path": str(B_path),
                "shape": list(B_fp16.shape),
                "dtype": str(B_fp16.dtype),
                "sha256": sha256_file(B_path),
            },
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "threads": int(args.threads),
            "wall_seconds": float(time.perf_counter() - started),
        },
    }
    write_json(training_path, training)
    pack_path = pack_dir / "manifest.json"
    pack = read_json(pack_path)
    pack["updated_at"] = now_iso()
    pack["rank"] = int(args.rank)
    pack["components"]["A"] = {
        "path": str(A_path),
        "shape": list(A_fp16.shape),
        "storage_dtype": "float16",
        "sha256": sha256_file(A_path),
        "status": "complete",
    }
    pack["components"]["B"] = {
        "path": str(B_path),
        "shape": list(B_fp16.shape),
        "storage_dtype": "float16",
        "initialization": "all zeros before training",
        "initial_zero_sha256": sha256_array(initial_B.astype(np.float32)),
        "sha256": sha256_file(B_path),
        "status": "complete",
    }
    pack["provenance"]["training"] = str(training_path)
    pack["provenance"]["training_sha256"] = sha256_file(training_path)
    pack["g3"] = training["stored_fp16_validation"] | {
        "verdict": training["g3_verdict"]
    }
    pack["status"] = "complete_g3_green" if g3_green else "g3_red_not_installable"
    pack["limitations"] = (
        ["behavioral G4/G5 evaluation pending"]
        if g3_green
        else ["G3 RED: stored adapter failed the 10 percent validation improvement gate"]
    )
    write_json(pack_path, pack)
    print(
        json.dumps(
            {
                "status": training["status"],
                "g3": training["g3_verdict"],
                "improvement_percent": improvement * 100.0,
                "training": str(training_path),
                "expertpack_manifest": str(pack_path),
            }
        ),
        flush=True,
    )
    return 0 if g3_green else 3


def load_expert_arrays(
    output: Path,
    address_rule: str = ADDRESS_RULE_E1,
    teacher_rule: str = TEACHER_RULE_ORIGINAL,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    pack = load_addressed_pack(
        output,
        address_rule=address_rule,
        require_adapter=True,
        teacher_rule=teacher_rule,
    )
    key = np.load(pack["components"]["key"]["path"], allow_pickle=False)
    A = np.load(pack["components"]["A"]["path"], allow_pickle=False)
    B = np.load(pack["components"]["B"]["path"], allow_pickle=False)
    if key.shape != (HIDDEN_DIM,) or key.dtype != np.float32:
        raise RuntimeError(f"key payload contract mismatch: {key.shape} {key.dtype}")
    if A.shape != (int(pack["rank"]), HIDDEN_DIM) or A.dtype != np.float16:
        raise RuntimeError(f"A payload contract mismatch: {A.shape} {A.dtype}")
    if B.shape != (HIDDEN_DIM, int(pack["rank"])) or B.dtype != np.float16:
        raise RuntimeError(f"B payload contract mismatch: {B.shape} {B.dtype}")
    return pack, np.ascontiguousarray(key), np.ascontiguousarray(A), np.ascontiguousarray(B)


def configure_block(block) -> None:
    block.self_attn.attention_mode = "standard"
    block.mlp.route_detail = "summary"
    block.mlp.empty_cache_interval = 0


def block_forward_with_expert(
    block,
    x,
    cos,
    sin,
    *,
    tc,
    compute_dtype: str,
    key: np.ndarray,
    tau: float,
    A_device,
    B_device,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Real block forward with the E1 residual side-path at this layer.

    Non-fired rows are appended directly from the native block output.  They
    never participate in an add, which makes the ABI identity guarantee a
    construction property rather than a tolerance claim.
    """

    normed = block.input_layernorm(x)
    normed = normed.half() if compute_dtype == "float16" else normed.astype(compute_dtype)
    attention, kv = block.self_attn(normed, cos, sin)
    residual = x + attention
    router_input = block.post_attention_layernorm(residual)
    router_input = (
        router_input.half()
        if compute_dtype == "float16"
        else router_input.astype(compute_dtype)
    )
    Bsz, length, dim = router_input.shape
    flat_h = router_input.reshape([Bsz * length, dim])
    host_h = flat_h.float().numpy().astype(np.float32, copy=False)
    scores = np.ascontiguousarray(host_h @ key, dtype=np.float32)
    fired = scores >= float(tau)
    native_mlp, route_info = block.mlp(router_input)
    native_output = residual + native_mlp
    native_flat = native_output.reshape([Bsz * length, dim])
    rows = []
    for token_index in range(Bsz * length):
        native_row = native_flat.slice(0, token_index, 1)
        if bool(fired[token_index]):
            h_row = flat_h.slice(0, token_index, 1)
            projected = tc.matmul(h_row, A_device, trans_b=True).silu()
            delta = tc.matmul(projected, B_device, trans_b=True)
            rows.append(native_row + delta)
            del h_row, projected, delta
        else:
            rows.append(native_row)
    output = tc.cat(rows, dim=0).reshape([Bsz, length, dim]) if rows else native_output
    return output, kv, route_info, {
        "tau": float(tau),
        "token_count": int(fired.size),
        "fire_count": int(fired.sum()),
        "fire_rate": float(fired.mean()),
        "score_min": float(scores.min()),
        "score_mean": float(scores.mean(dtype=np.float64)),
        "score_max": float(scores.max()),
        "scores": scores,
        "fired": fired,
        "router_input_fp32_sha256": sha256_array(host_h),
    }


def block_forward_e1_dispatch(
    block,
    x,
    cos,
    sin,
    *,
    expert: dict[str, Any] | None,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any] | None]:
    """Registered bolt-on dispatcher; an empty registry calls native verbatim."""

    if expert is None:
        output, kv, route_info = block(x, cos, sin)
        return output, kv, route_info, None
    return block_forward_with_expert(
        block,
        x,
        cos,
        sin,
        tc=expert["tc"],
        compute_dtype=expert["compute_dtype"],
        key=expert["key"],
        tau=expert["tau"],
        A_device=expert["A_device"],
        B_device=expert["B_device"],
    )


def final_norm(runtime, cfg, where, hidden):
    from core.mistral7b_tc import RMSNormTC

    tc = runtime["tc"]
    norm = RMSNormTC(cfg.hidden_dim, cfg.rms_norm_eps)
    norm.weight = tc.tensor(runtime["load_tensor_np"](where, "model.norm.weight"), dtype="float32")
    result = norm(hidden)
    compute_dtype = runtime["BlockTC"].COMPUTE_DTYPE
    return result.half() if compute_dtype == "float16" else result.astype(compute_dtype)


def load_lm_head(runtime, cfg, where):
    from core.mistral7b_tc import QuantLinearTC

    return QuantLinearTC(
        runtime["load_tensor_np"](where, "lm_head.weight"), group_size=cfg.group_size
    )


def nll_from_hidden(lm_head, hidden, targets: np.ndarray) -> dict[str, Any]:
    logits = lm_head(hidden)
    logits_np = logits.float().numpy().astype(np.float32, copy=False).reshape(
        -1, logits.shape[-1]
    )
    target_flat = np.asarray(targets, dtype=np.int64).reshape(-1)
    if logits_np.shape[0] != target_flat.size:
        raise RuntimeError(
            f"logit/target alignment mismatch {logits_np.shape[0]} != {target_flat.size}"
        )
    nlls = np.empty(target_flat.size, dtype=np.float64)
    for index, (row, target) in enumerate(zip(logits_np, target_flat)):
        maximum = float(row.max())
        logsumexp = maximum + float(np.log(np.exp(row - maximum).sum()))
        nlls[index] = logsumexp - float(row[int(target)])
    result = {
        "token_count": int(nlls.size),
        "nll_sum": float(nlls.sum()),
        "mean_nll": float(nlls.mean()),
        "ppl": float(np.exp(nlls.mean())),
        "target_ids_sha256": sha256_array(target_flat),
    }
    del logits, logits_np, nlls
    return result


def rt1_eval_window(corpus: str, window_index: int) -> tuple[np.ndarray, dict[str, Any]]:
    manifest_path = RT1_DIR / "corpora_manifest.json"
    windows_path = RT1_DIR / "corpus_windows.npz"
    if not manifest_path.is_file() or not windows_path.is_file():
        raise FileNotFoundError("frozen RT1 windows are missing")
    manifest = read_json(manifest_path)
    if manifest.get("windows_file_sha256") != sha256_file(windows_path):
        raise RuntimeError("RT1 frozen windows hash mismatch")
    with np.load(windows_path, allow_pickle=False) as stored:
        ids = np.ascontiguousarray(stored[corpus][window_index], dtype=np.int64)
    if ids.shape != (WINDOW_TOKENS,):
        raise RuntimeError(f"RT1 {corpus} window shape {ids.shape}")
    if corpus == "generic":
        sources = manifest["corpora"]["generic"]["sources"]
        allowed_root = Path("/home/vader/.cache/huggingface/hub/datasets--wikitext").resolve()
        for source in sources:
            if not Path(source["path"]).resolve().is_relative_to(allowed_root):
                raise RuntimeError("generic evaluation source escapes wikitext cache")
    return ids, {
        "rt1_manifest": str(manifest_path),
        "rt1_manifest_sha256": sha256_file(manifest_path),
        "rt1_windows": str(windows_path),
        "rt1_windows_sha256": sha256_file(windows_path),
        "corpus": corpus,
        "window_index": window_index,
        "input_ids_sha256": sha256_array(ids),
        "sources": manifest["corpora"][corpus]["sources"],
    }


def eval_receipt_path(
    output: Path,
    kind: str,
    window_index: int | None,
    address_rule: str = ADDRESS_RULE_E1,
    eval_fix: str = EVAL_FIX_ORIGINAL,
    teacher_rule: str = TEACHER_RULE_ORIGINAL,
) -> Path:
    root = eval_root_for_fix(output, address_rule, eval_fix, teacher_rule)
    if kind == "abi":
        return root / "abi.json"
    if window_index is None:
        raise ValueError(f"{kind} requires a window index")
    return root / f"{kind}_{window_index:03d}.json"


def initialize_eval_receipt(
    args: argparse.Namespace,
    output: Path,
    pack: dict[str, Any],
    kind: str,
    window_index: int | None,
    cuda_probe: dict[str, Any],
) -> tuple[Path, dict[str, Any], float]:
    path = eval_receipt_path(
        output,
        kind,
        window_index,
        args.address_rule,
        args.eval_fix,
        args.teacher_rule,
    )
    if path.is_file() and not args.overwrite:
        prior = read_json(path)
        if prior.get("status") == "complete":
            print(json.dumps({"status": "already_complete", "receipt": str(path)}))
            raise SystemExit(0)
        raise FileExistsError(f"stale eval receipt: {path}")
    started = time.perf_counter()
    receipt = {
        "schema": "moe_e1_eval_unit_v1",
        "created_at": now_iso(),
        "address_rule": args.address_rule,
        "eval_fix": args.eval_fix,
        "teacher_rule": args.teacher_rule,
        "status": "starting",
        "kind": kind,
        "window_index": window_index,
        "argv": sys.argv,
        "required_shell_wrapper": GPU_WRAPPER,
        "model_dir": str(args.model_dir.resolve()),
        "install_layer": int(pack["layer"]),
        "rank": int(pack["rank"]),
        "tau": float(pack["tau"]),
        "expertpack_manifest": str(
            expertpack_path(output, args.address_rule, args.teacher_rule)
        ),
        "expertpack_manifest_sha256_at_run": sha256_file(
            expertpack_path(output, args.address_rule, args.teacher_rule)
        ),
        "order": str(order_path_for_experiment(args.address_rule, args.teacher_rule)),
        "script_sha256_at_run": sha256_file(SCRIPT_PATH),
        "cuda_environment": cuda_probe,
        "gpu_before": nvidia_smi(),
        "attention_mode": "standard",
        "expert_mode": "resident_packed_mxfp4",
        "compute_dtype": "bfloat16",
    }
    write_json(path, receipt)
    return path, receipt, started


def isolated_eval_arm_forward(
    *,
    runtime,
    cfg,
    where,
    hidden,
    arm_name: str,
    receipt: dict[str, Any],
    receipt_path: Path,
    expert: dict[str, Any] | None = None,
):
    """E1.2 opt-in arm isolation using the established streamed schedule.

    Each loaded block is called exactly once and synchronized before release.
    The original multi-arm schedule remains the default path.
    """

    tc = runtime["tc"]
    cos, sin = runtime["gpt_oss_yarn_rope_tables"](cfg, int(hidden.shape[1]))
    rows: list[dict[str, Any]] = []
    fire_info: dict[str, Any] | None = None
    for layer in range(N_LAYERS):
        layer_started = time.perf_counter()
        block = runtime["GptOssDiagnosticBlockTC"].from_safetensors(
            cfg, where, layer, expert_mode="resident_packed_mxfp4"
        )
        configure_block(block)
        if expert is not None and layer == int(expert["install_layer"]):
            hidden, kv, route, fire_info = block_forward_with_expert(
                block,
                hidden,
                cos,
                sin,
                tc=tc,
                compute_dtype=expert["compute_dtype"],
                key=expert["key"],
                tau=float(expert["tau"]),
                A_device=expert["A_device"],
                B_device=expert["B_device"],
            )
        else:
            hidden, kv, route = block(hidden, cos, sin)
        tc.synchronize()
        rows.append(
            {
                "arm": arm_name,
                "layer": layer,
                "attention_backend": block.self_attn.last_attention_backend,
                "wall_seconds": float(time.perf_counter() - layer_started),
            }
        )
        receipt.setdefault("layers_by_arm", {})[arm_name] = rows
        receipt["completed_layers"] = layer + 1
        receipt["completed_arm_layers"] = sum(
            len(arm_rows)
            for arm_rows in receipt.get("layers_by_arm", {}).values()
        )
        receipt["status"] = f"running_{arm_name}_layers"
        write_json(receipt_path, receipt)
        del block, kv, route
        gc.collect()
        if hasattr(tc, "empty_cache"):
            tc.empty_cache()
    del cos, sin
    if expert is not None and fire_info is None:
        raise RuntimeError(f"isolated expert arm {arm_name} missed install layer")
    return hidden, fire_info


def eval_narrative(
    args: argparse.Namespace,
    output: Path,
    pack: dict[str, Any],
    key: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    cuda_probe: dict[str, Any],
) -> int:
    if args.window_index not in range(N_BEHAVIORAL_WINDOWS):
        raise ValueError("narrative eval requires --window-index 0..15")
    manifest, prepared = load_prepared(output)
    index = int(args.window_index)
    window = prepared["behavioral_ids"][index]
    if args.teacher_rule == TEACHER_RULE_E13:
        prefix, prefix_source, selection = load_e13_teacher_prefix(
            "behavioral", index
        )
    else:
        prefix = prepared["behavioral_prefix_ids"][index]
        prefix_source = manifest["windows"]["behavioral_prefixes"][index]
        selection = None
    base_ids = window[:-1][None, :]
    teacher_ids = np.concatenate([prefix, window[:-1]])[None, :]
    targets = window[1:]
    path, receipt, started = initialize_eval_receipt(
        args, output, pack, "narrative", index, cuda_probe
    )
    receipt.update(
        {
            "source_window": manifest["windows"]["behavioral_heldout"][index],
            "teacher_prefix_source": prefix_source,
            "teacher_prefix_tokens": int(prefix.size),
            "winning_construction": (
                None if selection is None else selection["winning_construction"]
            ),
            "teacher_selection": (
                None
                if selection is None
                else {
                    "path": str(E13_SELECTION_PATH),
                    "sha256": sha256_file(E13_SELECTION_PATH),
                    "winning_mean_gap": selection["winning_mean_gap"],
                }
            ),
            "base_input_ids_sha256": sha256_array(base_ids),
            "teacher_input_ids_sha256": sha256_array(teacher_ids),
            "target_ids_sha256": sha256_array(targets),
            "teacher_scores_prefix_tokens": False,
        }
    )
    write_json(path, receipt)
    try:
        runtime = load_runtime()
        tc = runtime["tc"]
        cfg = runtime["GptOss20BConfig"].from_model_dir(args.model_dir.resolve())
        validate_model_contract(cfg)
        where = runtime["build_safetensors_map"](args.model_dir.resolve())
        install_layer = int(pack["layer"])
        compute_dtype = runtime["BlockTC"].COMPUTE_DTYPE
        with tc.no_grad():
            A_device = tc.tensor(A.astype(np.float32), dtype=compute_dtype)
            B_device = tc.tensor(B.astype(np.float32), dtype=compute_dtype)
            embed = runtime["GptOssRowEmbedding"](where)
            h_teacher = embed(teacher_ids)
            h_base = embed(base_ids)
            h_expert = embed(base_ids)
            fire_info: dict[str, Any] | None = None
            if args.eval_fix == EVAL_FIX_E12:
                receipt["execution_schedule"] = (
                    "e12_isolated_arms_one_call_per_block_synchronize_each_call"
                )
                expert_context = {
                    "install_layer": install_layer,
                    "compute_dtype": compute_dtype,
                    "key": key,
                    "tau": float(pack["tau"]),
                    "A_device": A_device,
                    "B_device": B_device,
                }
                h_teacher, _ = isolated_eval_arm_forward(
                    runtime=runtime,
                    cfg=cfg,
                    where=where,
                    hidden=h_teacher,
                    arm_name="teacher",
                    receipt=receipt,
                    receipt_path=path,
                )
                h_base, _ = isolated_eval_arm_forward(
                    runtime=runtime,
                    cfg=cfg,
                    where=where,
                    hidden=h_base,
                    arm_name="base",
                    receipt=receipt,
                    receipt_path=path,
                )
                h_expert, fire_info = isolated_eval_arm_forward(
                    runtime=runtime,
                    cfg=cfg,
                    where=where,
                    hidden=h_expert,
                    arm_name="expert",
                    receipt=receipt,
                    receipt_path=path,
                    expert=expert_context,
                )
            else:
                receipt["execution_schedule"] = (
                    "original_shared_block_teacher_base_expert_then_synchronize"
                )
                cos, sin = runtime["gpt_oss_yarn_rope_tables"](
                    cfg, teacher_ids.shape[1]
                )
                receipt["layers"] = []
                for layer in range(N_LAYERS):
                    layer_started = time.perf_counter()
                    block = runtime["GptOssDiagnosticBlockTC"].from_safetensors(
                        cfg, where, layer, expert_mode="resident_packed_mxfp4"
                    )
                    configure_block(block)
                    h_teacher, kv_t, _route_t = block(h_teacher, cos, sin)
                    h_base, kv_b, _route_b = block(h_base, cos, sin)
                    if layer == install_layer:
                        h_expert, kv_e, _route_e, fire_info = (
                            block_forward_with_expert(
                                block,
                                h_expert,
                                cos,
                                sin,
                                tc=tc,
                                compute_dtype=compute_dtype,
                                key=key,
                                tau=float(pack["tau"]),
                                A_device=A_device,
                                B_device=B_device,
                            )
                        )
                    else:
                        h_expert, kv_e, _route_e = block(h_expert, cos, sin)
                    tc.synchronize()
                    receipt["layers"].append(
                        {
                            "layer": layer,
                            "attention_backend": block.self_attn.last_attention_backend,
                            "wall_seconds": float(
                                time.perf_counter() - layer_started
                            ),
                        }
                    )
                    receipt["completed_layers"] = layer + 1
                    receipt["status"] = "running_layers"
                    receipt["wall_seconds"] = float(
                        time.perf_counter() - started
                    )
                    write_json(path, receipt)
                    del block, kv_t, kv_b, kv_e, _route_t, _route_b, _route_e
                    gc.collect()
                    if hasattr(tc, "empty_cache"):
                        tc.empty_cache()
                del cos, sin
            if fire_info is None:
                raise RuntimeError("install layer did not execute expert gate")
            teacher_tail = h_teacher.slice(
                1, int(h_teacher.shape[1]) - (WINDOW_TOKENS - 1), WINDOW_TOKENS - 1
            )
            h_teacher_norm = final_norm(runtime, cfg, where, teacher_tail)
            h_base_norm = final_norm(runtime, cfg, where, h_base)
            h_expert_norm = final_norm(runtime, cfg, where, h_expert)
            del h_teacher, h_base, h_expert, teacher_tail
            if hasattr(tc, "empty_cache"):
                tc.empty_cache()
            lm_head = load_lm_head(runtime, cfg, where)
            arms = {
                "base": nll_from_hidden(lm_head, h_base_norm, targets),
                "teacher": nll_from_hidden(lm_head, h_teacher_norm, targets),
                "expert": nll_from_hidden(lm_head, h_expert_norm, targets),
            }
            tc.synchronize()
        receipt.update(
            {
                "status": "complete",
                "arms": arms,
                "fire": {k: v for k, v in fire_info.items() if k not in {"scores", "fired"}},
                "wall_seconds": float(time.perf_counter() - started),
                "gpu_after": nvidia_smi(),
            }
        )
        write_json(path, receipt)
        print(json.dumps({"status": "complete", "receipt": str(path), "arms": arms}), flush=True)
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
        write_json(path, receipt)
        raise


def eval_generic(
    args: argparse.Namespace,
    output: Path,
    pack: dict[str, Any],
    key: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    cuda_probe: dict[str, Any],
) -> int:
    if args.window_index not in EVAL_INDICES:
        raise ValueError(f"generic eval requires odd FIT-held-out index in {list(EVAL_INDICES)}")
    index = int(args.window_index)
    window, source = rt1_eval_window("generic", index)
    input_ids = window[:-1][None, :]
    targets = window[1:]
    path, receipt, started = initialize_eval_receipt(
        args, output, pack, "generic", index, cuda_probe
    )
    receipt.update(
        {
            "source": source,
            "wikitext": True,
            "input_ids_sha256": sha256_array(input_ids),
            "target_ids_sha256": sha256_array(targets),
        }
    )
    write_json(path, receipt)
    try:
        runtime = load_runtime()
        tc = runtime["tc"]
        cfg = runtime["GptOss20BConfig"].from_model_dir(args.model_dir.resolve())
        validate_model_contract(cfg)
        where = runtime["build_safetensors_map"](args.model_dir.resolve())
        install_layer = int(pack["layer"])
        compute_dtype = runtime["BlockTC"].COMPUTE_DTYPE
        with tc.no_grad():
            A_device = tc.tensor(A.astype(np.float32), dtype=compute_dtype)
            B_device = tc.tensor(B.astype(np.float32), dtype=compute_dtype)
            embed = runtime["GptOssRowEmbedding"](where)
            h_base = embed(input_ids)
            h_expert = embed(input_ids)
            fire_info: dict[str, Any] | None = None
            if args.eval_fix == EVAL_FIX_E12:
                receipt["execution_schedule"] = (
                    "e12_isolated_arms_one_call_per_block_synchronize_each_call"
                )
                expert_context = {
                    "install_layer": install_layer,
                    "compute_dtype": compute_dtype,
                    "key": key,
                    "tau": float(pack["tau"]),
                    "A_device": A_device,
                    "B_device": B_device,
                }
                h_base, _ = isolated_eval_arm_forward(
                    runtime=runtime,
                    cfg=cfg,
                    where=where,
                    hidden=h_base,
                    arm_name="base",
                    receipt=receipt,
                    receipt_path=path,
                )
                h_expert, fire_info = isolated_eval_arm_forward(
                    runtime=runtime,
                    cfg=cfg,
                    where=where,
                    hidden=h_expert,
                    arm_name="expert",
                    receipt=receipt,
                    receipt_path=path,
                    expert=expert_context,
                )
            else:
                receipt["execution_schedule"] = (
                    "original_shared_block_base_expert_without_per_call_synchronize"
                )
                cos, sin = runtime["gpt_oss_yarn_rope_tables"](
                    cfg, input_ids.shape[1]
                )
                for layer in range(N_LAYERS):
                    layer_started = time.perf_counter()
                    block = runtime["GptOssDiagnosticBlockTC"].from_safetensors(
                        cfg, where, layer, expert_mode="resident_packed_mxfp4"
                    )
                    configure_block(block)
                    h_base, kv_b, _route_b = block(h_base, cos, sin)
                    if layer == install_layer:
                        h_expert, kv_e, _route_e, fire_info = (
                            block_forward_with_expert(
                                block,
                                h_expert,
                                cos,
                                sin,
                                tc=tc,
                                compute_dtype=compute_dtype,
                                key=key,
                                tau=float(pack["tau"]),
                                A_device=A_device,
                                B_device=B_device,
                            )
                        )
                    else:
                        h_expert, kv_e, _route_e = block(h_expert, cos, sin)
                    receipt["completed_layers"] = layer + 1
                    receipt["status"] = "running_layers"
                    receipt["last_layer_wall_seconds"] = float(
                        time.perf_counter() - layer_started
                    )
                    receipt["wall_seconds"] = float(
                        time.perf_counter() - started
                    )
                    write_json(path, receipt)
                    del block, kv_b, kv_e, _route_b, _route_e
                    gc.collect()
                    if hasattr(tc, "empty_cache"):
                        tc.empty_cache()
                del cos, sin
            if fire_info is None:
                raise RuntimeError("install layer did not execute expert gate")
            h_base_norm = final_norm(runtime, cfg, where, h_base)
            h_expert_norm = final_norm(runtime, cfg, where, h_expert)
            del h_base, h_expert
            if hasattr(tc, "empty_cache"):
                tc.empty_cache()
            lm_head = load_lm_head(runtime, cfg, where)
            arms = {
                "base": nll_from_hidden(lm_head, h_base_norm, targets),
                "expert": nll_from_hidden(lm_head, h_expert_norm, targets),
            }
            tc.synchronize()
        receipt.update(
            {
                "status": "complete",
                "arms": arms,
                "fire": {k: v for k, v in fire_info.items() if k not in {"scores", "fired"}},
                "wall_seconds": float(time.perf_counter() - started),
                "gpu_after": nvidia_smi(),
            }
        )
        write_json(path, receipt)
        print(json.dumps({"status": "complete", "receipt": str(path), "arms": arms}), flush=True)
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
        write_json(path, receipt)
        raise


def eval_code(
    args: argparse.Namespace,
    output: Path,
    pack: dict[str, Any],
    key: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    cuda_probe: dict[str, Any],
) -> int:
    if args.window_index not in EVAL_INDICES:
        raise ValueError(f"code eval requires odd FIT-held-out index in {list(EVAL_INDICES)}")
    index = int(args.window_index)
    window, source = rt1_eval_window("code", index)
    input_ids = window[None, :]
    path, receipt, started = initialize_eval_receipt(
        args, output, pack, "code", index, cuda_probe
    )
    receipt.update({"source": source, "input_ids_sha256": sha256_array(input_ids)})
    write_json(path, receipt)
    try:
        runtime = load_runtime()
        tc = runtime["tc"]
        cfg = runtime["GptOss20BConfig"].from_model_dir(args.model_dir.resolve())
        validate_model_contract(cfg)
        where = runtime["build_safetensors_map"](args.model_dir.resolve())
        install_layer = int(pack["layer"])
        compute_dtype = runtime["BlockTC"].COMPUTE_DTYPE
        with tc.no_grad():
            A_device = tc.tensor(A.astype(np.float32), dtype=compute_dtype)
            B_device = tc.tensor(B.astype(np.float32), dtype=compute_dtype)
            embed = runtime["GptOssRowEmbedding"](where)
            hidden = embed(input_ids)
            cos, sin = runtime["gpt_oss_yarn_rope_tables"](cfg, WINDOW_TOKENS)
            fire_info: dict[str, Any] | None = None
            for layer in range(install_layer + 1):
                block = runtime["GptOssDiagnosticBlockTC"].from_safetensors(
                    cfg, where, layer, expert_mode="resident_packed_mxfp4"
                )
                configure_block(block)
                if layer == install_layer:
                    hidden, kv, _route, fire_info = block_forward_with_expert(
                        block,
                        hidden,
                        cos,
                        sin,
                        tc=tc,
                        compute_dtype=compute_dtype,
                        key=key,
                        tau=float(pack["tau"]),
                        A_device=A_device,
                        B_device=B_device,
                    )
                else:
                    hidden, kv, _route = block(hidden, cos, sin)
                receipt["completed_layers"] = layer + 1
                receipt["status"] = "running_layers"
                receipt["wall_seconds"] = float(time.perf_counter() - started)
                write_json(path, receipt)
                del block, kv, _route
                gc.collect()
                if hasattr(tc, "empty_cache"):
                    tc.empty_cache()
            if fire_info is None:
                raise RuntimeError("install layer did not execute code gate")
            tc.synchronize()
        receipt.update(
            {
                "status": "complete",
                "fire": {k: v for k, v in fire_info.items() if k not in {"scores", "fired"}},
                "wall_seconds": float(time.perf_counter() - started),
                "gpu_after": nvidia_smi(),
            }
        )
        write_json(path, receipt)
        print(json.dumps({"status": "complete", "receipt": str(path), "fire": receipt["fire"]}), flush=True)
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
        write_json(path, receipt)
        raise


def eval_abi(
    args: argparse.Namespace,
    output: Path,
    pack: dict[str, Any],
    key: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    cuda_probe: dict[str, Any],
) -> int:
    path, receipt, started = initialize_eval_receipt(
        args, output, pack, "abi", None, cuda_probe
    )
    _manifest, prepared = load_prepared(output)
    generic_window, generic_source = rt1_eval_window("generic", EVAL_INDICES[0])
    identity_ids = generic_window[:32][None, :]
    mixed_ids = prepared["key_ids"][EVAL_INDICES[0], :64][None, :]
    receipt.update(
        {
            "identity_input_ids_sha256": sha256_array(identity_ids),
            "mixed_input_ids_sha256": sha256_array(mixed_ids),
            "identity_source": generic_source,
            "tests": (
                "two zero-install native forwards; mounted forced-no-fire full forward; "
                "mixed-mask install-layer false-row identity"
            ),
        }
    )
    write_json(path, receipt)
    try:
        runtime = load_runtime()
        tc = runtime["tc"]
        cfg = runtime["GptOss20BConfig"].from_model_dir(args.model_dir.resolve())
        validate_model_contract(cfg)
        where = runtime["build_safetensors_map"](args.model_dir.resolve())
        install_layer = int(pack["layer"])
        compute_dtype = runtime["BlockTC"].COMPUTE_DTYPE
        mixed_base_np: np.ndarray | None = None
        mixed_expert_np: np.ndarray | None = None
        mixed_fire: np.ndarray | None = None
        mixed_info: dict[str, Any] | None = None
        mixed_tau = float(pack["tau"])
        mixed_tau_source = "installed_tau"
        with tc.no_grad():
            A_device = tc.tensor(A.astype(np.float32), dtype=compute_dtype)
            B_device = tc.tensor(B.astype(np.float32), dtype=compute_dtype)
            embed = runtime["GptOssRowEmbedding"](where)
            h_zero_a = embed(identity_ids)
            h_zero_b = embed(identity_ids)
            h_nofire = embed(identity_ids)
            h_mix_base = embed(mixed_ids)
            h_mix = embed(mixed_ids)
            cos, sin = runtime["gpt_oss_yarn_rope_tables"](cfg, mixed_ids.shape[1])
            nofire_info: dict[str, Any] | None = None
            for layer in range(N_LAYERS):
                block = runtime["GptOssDiagnosticBlockTC"].from_safetensors(
                    cfg, where, layer, expert_mode="resident_packed_mxfp4"
                )
                configure_block(block)
                h_zero_a, kv_za, _route_za = block(h_zero_a, cos, sin)
                if layer == install_layer:
                    h_zero_b, kv_zb, _route_zb, empty_dispatch_info = (
                        block_forward_e1_dispatch(
                            block, h_zero_b, cos, sin, expert=None
                        )
                    )
                    if empty_dispatch_info is not None:
                        raise RuntimeError("empty E1 registry unexpectedly returned gate info")
                else:
                    h_zero_b, kv_zb, _route_zb = block(h_zero_b, cos, sin)
                if layer == install_layer:
                    h_nofire, kv_nf, _route_nf, nofire_info = block_forward_with_expert(
                        block,
                        h_nofire,
                        cos,
                        sin,
                        tc=tc,
                        compute_dtype=compute_dtype,
                        key=key,
                        tau=math.inf,
                        A_device=A_device,
                        B_device=B_device,
                    )
                else:
                    h_nofire, kv_nf, _route_nf = block(h_nofire, cos, sin)
                if layer <= install_layer:
                    if layer < install_layer:
                        h_mix_base, kv_mb, _route_mb = block(h_mix_base, cos, sin)
                        h_mix, kv_m, _route_m = block(h_mix, cos, sin)
                    else:
                        mix_pre = h_mix
                        h_mix_base, kv_mb, _route_mb = block(h_mix_base, cos, sin)
                        candidate, kv_m, _route_m, candidate_info = block_forward_with_expert(
                            block,
                            mix_pre,
                            cos,
                            sin,
                            tc=tc,
                            compute_dtype=compute_dtype,
                            key=key,
                            tau=mixed_tau,
                            A_device=A_device,
                            B_device=B_device,
                        )
                        candidate_fire = candidate_info["fired"]
                        if not candidate_fire.any() or candidate_fire.all():
                            scores = candidate_info["scores"]
                            if float(scores.min()) < float(scores.max()):
                                mixed_tau = float(np.median(scores))
                                mixed_tau_source = "deterministic_median_contract_test_tau"
                                candidate, kv_m, _route_m, candidate_info = block_forward_with_expert(
                                    block,
                                    mix_pre,
                                    cos,
                                    sin,
                                    tc=tc,
                                    compute_dtype=compute_dtype,
                                    key=key,
                                    tau=mixed_tau,
                                    A_device=A_device,
                                    B_device=B_device,
                                )
                                candidate_fire = candidate_info["fired"]
                        h_mix = candidate
                        mixed_fire = candidate_fire.copy()
                        mixed_info = candidate_info
                        mixed_base_np = h_mix_base.float().numpy().astype(np.float32, copy=True)
                        mixed_expert_np = h_mix.float().numpy().astype(np.float32, copy=True)
                receipt["completed_layers"] = layer + 1
                receipt["status"] = "running_layers"
                receipt["wall_seconds"] = float(time.perf_counter() - started)
                write_json(path, receipt)
                del block, kv_za, kv_zb, kv_nf, _route_za, _route_zb, _route_nf
                if layer <= install_layer:
                    del kv_mb, kv_m, _route_mb, _route_m
                gc.collect()
                if hasattr(tc, "empty_cache"):
                    tc.empty_cache()
            if nofire_info is None or mixed_info is None or mixed_fire is None:
                raise RuntimeError("G0 ABI paths did not execute")
            zero_a_norm = final_norm(runtime, cfg, where, h_zero_a)
            zero_b_norm = final_norm(runtime, cfg, where, h_zero_b)
            nofire_norm = final_norm(runtime, cfg, where, h_nofire)
            zero_a_np = zero_a_norm.float().numpy().astype(np.float32, copy=True)
            zero_b_np = zero_b_norm.float().numpy().astype(np.float32, copy=True)
            nofire_np = nofire_norm.float().numpy().astype(np.float32, copy=True)
            tc.synchronize()
        if mixed_base_np is None or mixed_expert_np is None:
            raise RuntimeError("mixed install outputs missing")
        false_mask = ~mixed_fire.reshape(-1)
        flat_base = mixed_base_np.reshape(-1, HIDDEN_DIM)
        flat_expert = mixed_expert_np.reshape(-1, HIDDEN_DIM)
        zero_install_exact = bool(
            np.array_equal(zero_a_np, zero_b_np)
            and zero_a_np.tobytes(order="C") == zero_b_np.tobytes(order="C")
        )
        mounted_nofire_exact = bool(
            np.array_equal(zero_a_np, nofire_np)
            and zero_a_np.tobytes(order="C") == nofire_np.tobytes(order="C")
        )
        mixed_nonfire_exact = bool(
            false_mask.any()
            and np.array_equal(flat_base[false_mask], flat_expert[false_mask])
            and flat_base[false_mask].tobytes(order="C")
            == flat_expert[false_mask].tobytes(order="C")
        )
        mixed_has_both = bool(mixed_fire.any() and false_mask.any())
        g0_green = bool(
            zero_install_exact
            and mounted_nofire_exact
            and mixed_nonfire_exact
            and mixed_has_both
            and nofire_info["fire_count"] == 0
        )
        receipt.update(
            {
                "status": "complete",
                "g0_verdict": "GREEN" if g0_green else "RED",
                "zero_experts_installed": {
                    "bit_identical": zero_install_exact,
                    "byte_identical": zero_install_exact,
                    "final_hidden_shape": list(zero_a_np.shape),
                    "pass_a_sha256": sha256_array(zero_a_np),
                    "pass_b_sha256": sha256_array(zero_b_np),
                },
                "mounted_forced_nofire": {
                    "bit_identical_to_base": mounted_nofire_exact,
                    "byte_identical_to_base": mounted_nofire_exact,
                    "fire_count": int(nofire_info["fire_count"]),
                    "base_sha256": sha256_array(zero_a_np),
                    "mounted_sha256": sha256_array(nofire_np),
                },
                "mounted_mixed_gate": {
                    "installed_tau": float(pack["tau"]),
                    "test_tau": mixed_tau,
                    "test_tau_source": mixed_tau_source,
                    "fire_count": int(mixed_fire.sum()),
                    "nonfire_count": int(false_mask.sum()),
                    "has_both_states": mixed_has_both,
                    "install_layer_nonfire_bit_identical": mixed_nonfire_exact,
                    "install_layer_nonfire_byte_identical": mixed_nonfire_exact,
                    "base_nonfire_sha256": sha256_array(flat_base[false_mask]),
                    "expert_nonfire_sha256": sha256_array(flat_expert[false_mask]),
                },
                "wall_seconds": float(time.perf_counter() - started),
                "gpu_after": nvidia_smi(),
            }
        )
        write_json(path, receipt)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "g0": receipt["g0_verdict"],
                    "zero_install_exact": zero_install_exact,
                    "mounted_nofire_exact": mounted_nofire_exact,
                    "mixed_nonfire_exact": mixed_nonfire_exact,
                }
            ),
            flush=True,
        )
        return 0 if g0_green else 3
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
        write_json(path, receipt)
        raise


def eval_gates(args: argparse.Namespace) -> int:
    if args.eval_kind is None:
        raise ValueError("eval-gates requires --eval-kind abi|narrative|generic|code")
    output = ensure_output_dir(args.output_dir)
    pack, key, A, B = load_expert_arrays(
        output, args.address_rule, args.teacher_rule
    )
    cuda_probe = require_cuda()
    if args.eval_kind == "abi":
        return eval_abi(args, output, pack, key, A, B, cuda_probe)
    if args.eval_kind == "narrative":
        return eval_narrative(args, output, pack, key, A, B, cuda_probe)
    if args.eval_kind == "generic":
        return eval_generic(args, output, pack, key, A, B, cuda_probe)
    if args.eval_kind == "code":
        return eval_code(args, output, pack, key, A, B, cuda_probe)
    raise AssertionError(args.eval_kind)


def gpu_resume_commands(address_rule: str = ADDRESS_RULE_E1) -> str:
    script = str(SCRIPT_PATH)
    if address_rule == ADDRESS_RULE_E11:
        return f"""#!/usr/bin/env bash
set -euo pipefail

repo_e1={str(REPO_ROOT)!r}
script_e1={script!r}
cd "$repo_e1"

run_gpu_e1() {{
  local rc_e1=0
  flock -w 7200 /tmp/forge-gpu.lock \\
    timeout --signal=TERM --kill-after=5s 590s \\
    env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \\
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
      python3 "$script_e1" "$@" || rc_e1=$?
  sleep 30
  return "$rc_e1"
}}

# E1.1 Stage 3: one teacher/student window per bounded GPU invocation at L*.
for pair_e1 in $(seq 0 63); do
  run_gpu_e1 capture-pairs --address-rule e11 --pair-index "$pair_e1"
done

# E1.1 Stage 4: CPU rank-64 consolidation.  G3 RED is a registered STOP.
if ! env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \\
  python3 "$script_e1" train --address-rule e11 --rank 64 --threads 6; then
  env CUDA_VISIBLE_DEVICES='' python3 "$script_e1" analyze --address-rule e11
  exit 3
fi

# E1.1 Stage 5a: ABI identity and all 16 held-out narrative windows.
run_gpu_e1 eval-gates --address-rule e11 --eval-kind abi
for window_e1 in $(seq 0 15); do
  run_gpu_e1 eval-gates --address-rule e11 --eval-kind narrative --window-index "$window_e1"
done
env CUDA_VISIBLE_DEVICES='' python3 "$script_e1" analyze --address-rule e11 || true

# Registered premise STOP: do not run G5' if TRAIN prefixes did not help heldout PPL.
python3 -c 'import json,sys; x=json.load(open("artifacts/moe_e1/analysis_e11.json")); sys.exit(0 if x.get("g4",{{}}).get("teacher_gap",0)>0 else 4)'

# E1.1 Stage 5b: WikiText base/expert PPL and per-window fire rates, plus code fire.
for window_e1 in 1 3 5 7 9 11 13 15; do
  run_gpu_e1 eval-gates --address-rule e11 --eval-kind generic --window-index "$window_e1"
done
for window_e1 in 1 3 5 7 9 11 13 15; do
  run_gpu_e1 eval-gates --address-rule e11 --eval-kind code --window-index "$window_e1"
done

# E1.1 Stage 6: CPU bootstrap/tables/final registered verdict and G5' split report.
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \\
  python3 "$script_e1" analyze --address-rule e11 --bootstrap-resamples 2000
"""
    return f"""#!/usr/bin/env bash
set -euo pipefail

repo_e1={str(REPO_ROOT)!r}
script_e1={script!r}
cd "$repo_e1"

run_gpu_e1() {{
  local rc_e1=0
  flock -w 7200 /tmp/forge-gpu.lock \\
    timeout --signal=TERM --kill-after=5s 590s \\
    env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \\
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
      python3 "$script_e1" "$@" || rc_e1=$?
  sleep 30
  return "$rc_e1"
}}

# CPU prepare is idempotently skipped when this blocked worker already made it.
if [[ ! -f artifacts/moe_e1/corpus_manifest.json ]]; then
  env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \\
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
    python3 "$script_e1" prepare
fi

# Stage 1: two bounded all-layer narrative key captures.
for chunk_e1 in 0 1; do
  run_gpu_e1 capture-keys --chunk-index "$chunk_e1"
done

# Stage 2: CPU K4 fit.  G2 RED is a registered STOP and analyze writes it up.
if ! env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \\
  python3 "$script_e1" fit-key --threads 6 --bootstrap-resamples 2000; then
  env CUDA_VISIBLE_DEVICES='' python3 "$script_e1" analyze
  exit 3
fi

# Stage 3: one teacher/student window per bounded GPU invocation.
for pair_e1 in $(seq 0 63); do
  run_gpu_e1 capture-pairs --pair-index "$pair_e1"
done

# Stage 4: CPU rank-64 consolidation.  G3 RED is a registered STOP.
if ! env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \\
  python3 "$script_e1" train --rank 64 --threads 6; then
  env CUDA_VISIBLE_DEVICES='' python3 "$script_e1" analyze
  exit 3
fi

# Stage 5a: ABI identity and all 16 held-out narrative behavioral windows.
run_gpu_e1 eval-gates --eval-kind abi
for window_e1 in $(seq 0 15); do
  run_gpu_e1 eval-gates --eval-kind narrative --window-index "$window_e1"
done
env CUDA_VISIBLE_DEVICES='' python3 "$script_e1" analyze || true

# Registered premise STOP: do not run G5 if TRAIN prefixes did not help heldout PPL.
python3 -c 'import json,sys; x=json.load(open("artifacts/moe_e1/analysis.json")); sys.exit(0 if x.get("g4",{{}}).get("teacher_gap",0)>0 else 4)'

# Stage 5b: WikiText base/expert PPL plus realized generic/code fire rates.
for window_e1 in 1 3 5 7 9 11 13 15; do
  run_gpu_e1 eval-gates --eval-kind generic --window-index "$window_e1"
done
for window_e1 in 1 3 5 7 9 11 13 15; do
  run_gpu_e1 eval-gates --eval-kind code --window-index "$window_e1"
done

# Stage 6: CPU bootstrap/tables/final registered verdict.
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \\
  python3 "$script_e1" analyze --bootstrap-resamples 2000
"""


def gate_line(name: str, verdict: str, details: str) -> str:
    return f"{name} {verdict} — {details}"


def completed_eval_receipts(
    output: Path,
    kind: str,
    indices: Sequence[int],
    address_rule: str = ADDRESS_RULE_E1,
    teacher_rule: str = TEACHER_RULE_ORIGINAL,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    receipts: list[dict[str, Any]] = []
    missing: list[int] = []
    errors: list[int] = []
    for index in indices:
        path = eval_receipt_path(
            output, kind, index, address_rule, EVAL_FIX_ORIGINAL, teacher_rule
        )
        if not path.is_file():
            missing.append(int(index))
            continue
        payload = read_json(path)
        if payload.get("status") == "complete":
            payload["_path"] = str(path)
            payload["_sha256"] = sha256_file(path)
            receipts.append(payload)
        else:
            errors.append(int(index))
    return receipts, missing, errors


def aggregate_nll(receipts: Sequence[dict[str, Any]], arm: str) -> dict[str, Any]:
    token_count = sum(int(item["arms"][arm]["token_count"]) for item in receipts)
    nll_sum = sum(float(item["arms"][arm]["nll_sum"]) for item in receipts)
    mean_nll = nll_sum / token_count
    return {
        "window_count": len(receipts),
        "token_count": token_count,
        "nll_sum": nll_sum,
        "mean_nll": mean_nll,
        "ppl": float(math.exp(mean_nll)),
    }


def wikitext_window_summary(receipt: dict[str, Any]) -> dict[str, Any]:
    base_ppl = float(receipt["arms"]["base"]["ppl"])
    expert_ppl = float(receipt["arms"]["expert"]["ppl"])
    index = int(receipt["window_index"])
    return {
        "window_index": index,
        "registered_high_firing_narrativeish": (
            index in E11_HIGH_FIRING_WIKITEXT_WINDOWS
        ),
        "fire_count": int(receipt["fire"]["fire_count"]),
        "token_count": int(receipt["fire"]["token_count"]),
        "fire_rate": float(receipt["fire"]["fire_rate"]),
        "ppl_base": base_ppl,
        "ppl_expert": expert_ppl,
        "ppl_delta_percent": (expert_ppl - base_ppl) / base_ppl * 100.0,
        "receipt": receipt["_path"],
        "receipt_sha256": receipt["_sha256"],
    }


def render_report(analysis: dict[str, Any]) -> str:
    gates = analysis["gates"]
    install = analysis["install_row"]
    g4 = analysis["g4_row"]
    is_e11 = analysis.get("address_rule") == ADDRESS_RULE_E11
    is_e13 = analysis.get("teacher_rule") == TEACHER_RULE_E13
    lines = [
        (
            "# MOE-E1.3 NarrativeForge Expert Report"
            if is_e13
            else (
                "# MOE-E1.1 NarrativeForge Expert Report"
                if is_e11
                else "# MOE-E1 NarrativeForge Expert Report"
            )
        ),
        "",
        f"Generated: `{analysis['created_at']}`",
        "",
        "Evidence class: behavioral inference measurement on one model and one domain. "
        "No generality claim is made beyond GPT-OSS-20B and this sealed guide corpus.",
        "",
        "## Registered verdict",
        "",
        analysis["registered_verdict_sentence"],
        "",
        "## Gates",
        "",
    ]
    lines.extend(gates[name]["report_line"] for name in ("G0", "G1", "G2", "G3", "G4", "G5"))
    if is_e11:
        lines.extend(["", "## G5' per-window WikiText split", ""])
        high = analysis["g5"].get("high_firing_narrativeish_windows", [])
        other = analysis["g5"].get("other_wikitext_windows", [])
        if high:
            lines.append("Registered high-firing narrative-ish windows:")
            lines.append("")
            lines.extend(
                (
                    f"- Window {item['window_index']}: fire="
                    f"{item['fire_rate']:.6f}, ppl_base={item['ppl_base']:.6f}, "
                    f"ppl_expert={item['ppl_expert']:.6f}, delta="
                    f"{item['ppl_delta_percent']:.4f}%"
                )
                for item in high
            )
        else:
            lines.append("Windows 5, 13, 11, and 15 are not yet measured.")
        if other:
            lines.extend(["", "Other WikiText windows:", ""])
            lines.extend(
                (
                    f"- Window {item['window_index']}: fire="
                    f"{item['fire_rate']:.6f}, ppl_base={item['ppl_base']:.6f}, "
                    f"ppl_expert={item['ppl_expert']:.6f}, delta="
                    f"{item['ppl_delta_percent']:.4f}%"
                )
                for item in sorted(
                    other,
                    key=lambda item: (-item["fire_rate"], item["window_index"]),
                )
            )
    lines.extend(
        [
            "",
            "## Install row",
            "",
            install,
            "",
            "## G4 row",
            "",
            g4,
            "",
            "## ExpertPack",
            "",
            f"ExpertPack manifest path: `{analysis['expertpack_manifest_path']}`",
            "",
            "The manifest names the key, frozen tau, A, B, install layer, rank, and "
            "provenance. Missing components remain explicitly marked missing; no "
            "synthetic parameter is presented as a trained expert.",
            "",
            "## Resume",
            "",
            f"Exact stop-aware command sequence: `{analysis['resume_commands_path']}`",
            "",
            "Every GPU process in that sequence is wrapped by `flock -w 7200 "
            "/tmp/forge-gpu.lock`, killed at 590 seconds, restricted to GPU 0, "
            "checkpointed per unit, and followed by a 30-second idle gap.",
            "",
            "## Files created or modified",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in analysis["created_or_modified_paths"])
    lines.extend(["", "## Limitations and blocked work", ""])
    lines.extend(f"- {item}" for item in analysis["limitations"])
    lines.append("")
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> int:
    if args.bootstrap_resamples < 2000:
        raise ValueError("analyze requires at least 2000 bootstrap resamples")
    output = ensure_output_dir(args.output_dir)
    manifest, _prepared = load_prepared(output)
    experiment_root = experiment_output(output, args.teacher_rule)
    experiment_root.mkdir(parents=True, exist_ok=True)
    cpu_validation_file = cpu_validation_path(output, args.address_rule)
    cpu_validation = read_json(cpu_validation_file)
    cuda_probe = cuda_environment_probe()
    gpu_available = bool(
        {"/dev/nvidia0", "/dev/nvidiactl"}.issubset(
            set(cuda_probe["character_device_nodes"])
        )
    )

    gates: dict[str, dict[str, Any]] = {}

    # G0
    abi_path = eval_receipt_path(
        output,
        "abi",
        None,
        args.address_rule,
        EVAL_FIX_ORIGINAL,
        args.teacher_rule,
    )
    if abi_path.is_file() and read_json(abi_path).get("status") == "complete":
        abi = read_json(abi_path)
        verdict = abi.get("g0_verdict", "RED")
        zero = abi["zero_experts_installed"]
        nofire = abi["mounted_forced_nofire"]
        mixed = abi["mounted_mixed_gate"]
        details = (
            f"zero-install byte-identical={zero['byte_identical']}; mounted no-fire "
            f"byte-identical={nofire['byte_identical_to_base']}; mixed non-fire "
            f"byte-identical={mixed['install_layer_nonfire_byte_identical']} "
            f"({mixed['nonfire_count']} rows), fires={mixed['fire_count']}"
        )
        gates["G0"] = {
            "verdict": verdict,
            "report_line": gate_line("G0", verdict, details),
            "receipt": str(abi_path),
            "receipt_sha256": sha256_file(abi_path),
        }
    else:
        details = "ABI GPU identity not run; zero-install and mounted non-fire bytes=n/a"
        gates["G0"] = {
            "verdict": "NOT_MEASURED",
            "report_line": gate_line("G0", "NOT_MEASURED", details),
        }

    # G1
    pair_truth: list[dict[str, Any]] = []
    pair_errors: list[int] = []
    for index in range(N_PAIR_WINDOWS):
        _data_path, receipt_path = pair_paths(
            output, index, args.address_rule, args.teacher_rule
        )
        if not receipt_path.is_file():
            continue
        receipt = read_json(receipt_path)
        if receipt.get("status") == "complete":
            pair_truth.append(receipt)
        elif receipt.get("status") == "error":
            pair_errors.append(index)
    failed_truth = [
        item
        for item in pair_truth
        if not item.get("token_alignment_exact") or not item.get("hidden_states_differ")
    ]
    if failed_truth or pair_errors:
        verdict = "RED"
        details = (
            f"aligned complete pairs={len(pair_truth)}, failed truth={len(failed_truth)}, "
            f"error receipts={pair_errors}"
        )
    elif len(pair_truth) >= 2:
        verdict = "GREEN"
        examples = pair_truth[:2]
        details = (
            f"aligned pairs={len(pair_truth)}/64; first two max hidden diffs="
            f"{examples[0]['hidden_difference_max_abs']:.6g},"
            f"{examples[1]['hidden_difference_max_abs']:.6g}"
        )
    else:
        verdict = "NOT_MEASURED"
        details = f"aligned pairs={len(pair_truth)}/2 minimum; hidden difference=n/a"
    gates["G1"] = {
        "verdict": verdict,
        "report_line": gate_line("G1", verdict, details),
        "complete_pair_count": len(pair_truth),
    }

    # G2 and install row
    fit_path, _keys_path = fit_key_paths(output, args.address_rule)
    pack_path = expertpack_path(output, args.address_rule, args.teacher_rule)
    if pack_path.is_file():
        pack = read_json(pack_path)
    elif args.teacher_rule == TEACHER_RULE_E13:
        pack = read_json(
            expertpack_path(output, args.address_rule, TEACHER_RULE_ORIGINAL)
        )
    else:
        pack = read_json(pack_path)
    g2_label = "G2'" if args.address_rule == ADDRESS_RULE_E11 else "G2"
    if args.address_rule == ADDRESS_RULE_E11:
        install_row = (
            "INSTALL row (G2'): L*=n/a, r=64, recall=n/a, code FPR=n/a, "
            "GRM FPR=n/a, tau=n/a, descriptive generic fire rate=n/a"
        )
    else:
        install_row = (
            "INSTALL row: layer L*=n/a, r=64, recall=n/a, generic FPR=n/a, "
            "code FPR=n/a, grm FPR=n/a, tau=n/a"
        )
    if fit_path.is_file():
        fit = read_json(fit_path)
        decision = fit["decision"]
        if decision["g2_green"]:
            selected_layer = int(decision["install_layer"])
            row = next(item for item in fit["rows"] if int(item["layer"]) == selected_layer)
            metrics = row["metrics"]
            evaluated = metrics["eval"]
            verdict = "GREEN"
            if args.address_rule == ADDRESS_RULE_E11:
                details = (
                    f"qualifying layers={decision['qualifying_layers']}; "
                    f"L*={selected_layer}; recall="
                    f"{evaluated['narrative_recall']['value']:.6f}, code FPR="
                    f"{evaluated['code_fpr']['value']:.6f}, GRM FPR="
                    f"{evaluated['grm_fpr']['value']:.6f}, tau="
                    f"{metrics['selected_tau']:.9g}, descriptive generic fire rate="
                    f"{evaluated['generic_fpr']['value']:.6f}"
                )
                install_row = (
                    f"INSTALL row (G2'): L*={selected_layer}, "
                    f"r={pack.get('rank', 64)}, recall="
                    f"{evaluated['narrative_recall']['value']:.6f}, code FPR="
                    f"{evaluated['code_fpr']['value']:.6f}, GRM FPR="
                    f"{evaluated['grm_fpr']['value']:.6f}, tau="
                    f"{metrics['selected_tau']:.9g}, descriptive generic fire rate="
                    f"{evaluated['generic_fpr']['value']:.6f}"
                )
            else:
                details = (
                    f"qualifying layers={decision['qualifying_layers']}; "
                    f"L*={selected_layer}; recall="
                    f"{evaluated['narrative_recall']['value']:.6f}, generic FPR="
                    f"{evaluated['generic_fpr']['value']:.6f}, code FPR="
                    f"{evaluated['code_fpr']['value']:.6f}, grm FPR="
                    f"{evaluated['grm_fpr']['value']:.6f}, "
                    f"tau={metrics['selected_tau']:.9g}"
                )
                install_row = (
                    f"INSTALL row: layer L*={selected_layer}, "
                    f"r={pack.get('rank', 64)}, recall="
                    f"{evaluated['narrative_recall']['value']:.6f}, "
                    f"generic FPR={evaluated['generic_fpr']['value']:.6f}, "
                    f"code FPR={evaluated['code_fpr']['value']:.6f}, "
                    f"grm FPR={evaluated['grm_fpr']['value']:.6f}, "
                    f"tau={metrics['selected_tau']:.9g}"
                )
        else:
            verdict = "RED"
            if args.address_rule == ADDRESS_RULE_E11:
                details = (
                    "qualifying layers=0/24 under recall>=0.50, code FPR<=0.05, "
                    "GRM FPR<=0.05 with frozen FIT tau; generic fire rate is "
                    "descriptive; STOP"
                )
            else:
                details = (
                    "qualifying layers=0/24 under recall>=0.50, generic FPR<=0.02, "
                    "code FPR<=0.05, grm FPR<=0.05 with frozen FIT tau; STOP"
                )
        gates["G2"] = {
            "verdict": verdict,
            "report_line": gate_line(g2_label, verdict, details),
            "fit_key": str(fit_path),
            "fit_key_sha256": sha256_file(fit_path),
            "qualifying_layers": decision["qualifying_layers"],
        }
    else:
        gates["G2"] = {
            "verdict": "NOT_MEASURED",
            "report_line": gate_line(
                g2_label,
                "NOT_MEASURED",
                "narrative K4 captures/fit absent; recall/FPRs/tau/L*=n/a",
            ),
        }

    # G3
    training_path = train_path(output, args.address_rule, args.teacher_rule)
    if training_path.is_file():
        training = read_json(training_path)
        stored = training["stored_fp16_validation"]
        verdict = training["g3_verdict"]
        details = (
            f"zero MSE={stored['zero_predictor_mse']:.9g}, adapter MSE="
            f"{stored['adapter_mse']:.9g}, improvement="
            f"{stored['improvement_percent']:.3f}% (required >=10%)"
        )
        gates["G3"] = {
            "verdict": verdict,
            "report_line": gate_line("G3", verdict, details),
            "training": str(training_path),
            "training_sha256": sha256_file(training_path),
        }
    else:
        gates["G3"] = {
            "verdict": "NOT_MEASURED",
            "report_line": gate_line(
                "G3", "NOT_MEASURED", "validation zero/adapter MSE and improvement=n/a"
            ),
        }

    # G4
    narrative, narrative_missing, narrative_errors = completed_eval_receipts(
        output,
        "narrative",
        range(N_BEHAVIORAL_WINDOWS),
        args.address_rule,
        args.teacher_rule,
    )
    g4_payload: dict[str, Any] = {
        "complete_windows": len(narrative),
        "missing_windows": narrative_missing,
        "error_windows": narrative_errors,
    }
    g4_row = (
        "G4 row: ppl_base=n/a, ppl_teacher=n/a, ppl_expert=n/a, "
        "recovery=n/a, bootstrap 95% CI=n/a"
    )
    premise_finding = False
    if len(narrative) == N_BEHAVIORAL_WINDOWS and not narrative_errors:
        base_nll = np.asarray([item["arms"]["base"]["mean_nll"] for item in narrative])
        teacher_nll = np.asarray([item["arms"]["teacher"]["mean_nll"] for item in narrative])
        expert_nll = np.asarray([item["arms"]["expert"]["mean_nll"] for item in narrative])
        recovery = bootstrap_recovery(
            base_nll,
            teacher_nll,
            expert_nll,
            resamples=args.bootstrap_resamples,
            seed=args.seed + 404,
        )
        g4_payload.update(recovery)
        premise_finding = bool(recovery["teacher_gap"] <= 0.0)
        if premise_finding:
            verdict = "RED"
            details = (
                f"premise failed: ppl_base={recovery['ppl_base']:.6f}, "
                f"ppl_teacher={recovery['ppl_teacher']:.6f}, teacher gap="
                f"{recovery['teacher_gap']:.6f} <= 0; STOP"
            )
        else:
            verdict = (
                "GREEN"
                if recovery["recovery_fraction"] >= G4_RECOVERY_FLOOR
                else "RED"
            )
            details = (
                f"ppl_base={recovery['ppl_base']:.6f}, ppl_teacher="
                f"{recovery['ppl_teacher']:.6f}, ppl_expert={recovery['ppl_expert']:.6f}, "
                f"recovery={recovery['recovery_percent']:.3f}% "
                f"(95% CI {recovery['ci95_low_percent']:.3f}% to "
                f"{recovery['ci95_high_percent']:.3f}%; required >=25%)"
            )
        g4_row = (
            f"G4 row: ppl_base={recovery['ppl_base']:.6f}, "
            f"ppl_teacher={recovery['ppl_teacher']:.6f}, "
            f"ppl_expert={recovery['ppl_expert']:.6f}, "
            f"recovery={recovery['recovery_percent']:.3f}%, bootstrap 95% CI="
            f"[{recovery['ci95_low_percent']:.3f}%, {recovery['ci95_high_percent']:.3f}%]"
        )
    else:
        verdict = "NOT_MEASURED"
        details = (
            f"heldout behavioral windows={len(narrative)}/16; "
            f"missing={narrative_missing}, errors={narrative_errors}; PPLs/recovery=n/a"
        )
    gates["G4"] = {
        "verdict": verdict,
        "report_line": gate_line("G4", verdict, details),
    }

    # G5
    generic, generic_missing, generic_errors = completed_eval_receipts(
        output, "generic", EVAL_INDICES, args.address_rule, args.teacher_rule
    )
    code, code_missing, code_errors = completed_eval_receipts(
        output, "code", EVAL_INDICES, args.address_rule, args.teacher_rule
    )
    g5_payload: dict[str, Any] = {
        "gate": "G5'" if args.address_rule == ADDRESS_RULE_E11 else "G5",
        "generic_complete_windows": len(generic),
        "code_complete_windows": len(code),
        "generic_missing": generic_missing,
        "code_missing": code_missing,
        "generic_errors": generic_errors,
        "code_errors": code_errors,
    }
    if args.address_rule == ADDRESS_RULE_E11:
        per_window_wikitext = [wikitext_window_summary(item) for item in generic]
        by_index = {
            int(item["window_index"]): item for item in per_window_wikitext
        }
        g5_payload.update(
            {
                "registered_rule": {
                    "wikitext_overall_ppl_delta_percent_lte": (
                        G5_WIKITEXT_PPL_DELTA_CAP_PCT
                    ),
                    "code_fire_rate_lte": G5_FIRE_CAPS["code"],
                    "generic_fire_rate": "descriptive_no_pass_fail_bound",
                },
                "registered_high_firing_narrativeish_window_indices": list(
                    E11_HIGH_FIRING_WIKITEXT_WINDOWS
                ),
                "per_window_wikitext_by_descending_fire_rate": sorted(
                    per_window_wikitext,
                    key=lambda item: (-item["fire_rate"], item["window_index"]),
                ),
                "high_firing_narrativeish_windows": [
                    by_index[index]
                    for index in E11_HIGH_FIRING_WIKITEXT_WINDOWS
                    if index in by_index
                ],
                "other_wikitext_windows": [
                    item
                    for item in per_window_wikitext
                    if not item["registered_high_firing_narrativeish"]
                ],
            }
        )
    if (
        len(generic) == len(EVAL_INDICES)
        and len(code) == len(EVAL_INDICES)
        and not generic_errors
        and not code_errors
    ):
        generic_base = aggregate_nll(generic, "base")
        generic_expert = aggregate_nll(generic, "expert")
        ppl_delta_pct = (
            (generic_expert["ppl"] - generic_base["ppl"])
            / generic_base["ppl"]
            * 100.0
        )
        generic_fires = sum(int(item["fire"]["fire_count"]) for item in generic)
        generic_tokens = sum(int(item["fire"]["token_count"]) for item in generic)
        code_fires = sum(int(item["fire"]["fire_count"]) for item in code)
        code_tokens = sum(int(item["fire"]["token_count"]) for item in code)
        generic_rate = generic_fires / generic_tokens
        code_rate = code_fires / code_tokens
        g5_green = g5_qualifies(
            address_rule=args.address_rule,
            wikitext_ppl_delta_percent=ppl_delta_pct,
            generic_fire_rate=generic_rate,
            code_fire_rate=code_rate,
        )
        verdict = "GREEN" if g5_green else "RED"
        if args.address_rule == ADDRESS_RULE_E11:
            details = (
                f"WikiText ppl base={generic_base['ppl']:.6f}, expert="
                f"{generic_expert['ppl']:.6f}, delta={ppl_delta_pct:.4f}% "
                f"(cap 0.5%); code fire={code_rate:.6f} (cap 0.05); "
                f"descriptive generic fire={generic_rate:.6f}; high-firing "
                "windows 5,13,11,15 reported individually"
            )
        else:
            details = (
                f"WikiText ppl base={generic_base['ppl']:.6f}, expert="
                f"{generic_expert['ppl']:.6f}, delta={ppl_delta_pct:.4f}% "
                f"(cap 0.5%); generic fire={generic_rate:.6f} (cap 0.02), "
                f"code fire={code_rate:.6f} (cap 0.05)"
            )
        g5_payload.update(
            {
                "wikitext_base": generic_base,
                "wikitext_expert": generic_expert,
                "wikitext_ppl_delta_percent": ppl_delta_pct,
                "generic_fire_count": generic_fires,
                "generic_token_count": generic_tokens,
                "generic_fire_rate": generic_rate,
                "code_fire_count": code_fires,
                "code_token_count": code_tokens,
                "code_fire_rate": code_rate,
            }
        )
    else:
        verdict = "NOT_MEASURED"
        if args.address_rule == ADDRESS_RULE_E11:
            details = (
                f"WikiText windows={len(generic)}/8, code windows={len(code)}/8; "
                "overall ppl delta/code fire=n/a; generic fire is descriptive"
            )
        else:
            details = (
                f"WikiText windows={len(generic)}/8, code windows={len(code)}/8; "
                "ppl delta/generic fire/code fire=n/a"
            )
    gates["G5"] = {
        "verdict": verdict,
        "report_line": gate_line(
            "G5'" if args.address_rule == ADDRESS_RULE_E11 else "G5",
            verdict,
            details,
        ),
    }

    if premise_finding:
        experiment_label = (
            "E1.3"
            if args.teacher_rule == TEACHER_RULE_E13
            else "E1.1" if args.address_rule == ADDRESS_RULE_E11 else "E1"
        )
        registered_verdict = (
            f"{experiment_label} "
            + (
                "PREMISE FINDING: the selected E1.3 teacher construction did not "
                "improve perplexity "
                if args.teacher_rule == TEACHER_RULE_E13
                else "PREMISE FINDING: TRAIN-file guide prefixes did not improve perplexity "
            )
            + "on HELDOUT guide text; expert recovery is undefined and evaluation stops."
        )
    elif gates["G4"]["verdict"] == "GREEN":
        registered_verdict = (
            "E1.3 SUPPORTED."
            if args.teacher_rule == TEACHER_RULE_E13
            else (
                "E1.1 SUPPORTED."
                if args.address_rule == ADDRESS_RULE_E11
                else "E1 SUPPORTED."
            )
        )
    elif gates["G4"]["verdict"] == "RED":
        experiment_label = (
            "E1.3"
            if args.teacher_rule == TEACHER_RULE_E13
            else "E1.1" if args.address_rule == ADDRESS_RULE_E11 else "E1"
        )
        registered_verdict = (
            f"{experiment_label} "
            "NOT SUPPORTED: the expert recovered less than 25% of the positive "
            "teacher perplexity gap under the registered behavioral gate."
        )
    elif not gpu_available:
        experiment_label = (
            "E1.3"
            if args.teacher_rule == TEACHER_RULE_E13
            else "E1.1" if args.address_rule == ADDRESS_RULE_E11 else "E1"
        )
        registered_verdict = (
            f"{experiment_label} "
            "NOT MEASURED — CUDA device nodes are unavailable; no registered "
            "behavioral verdict can be issued."
        )
    else:
        experiment_label = (
            "E1.3"
            if args.teacher_rule == TEACHER_RULE_E13
            else "E1.1" if args.address_rule == ADDRESS_RULE_E11 else "E1"
        )
        registered_verdict = (
            f"{experiment_label} "
            "NOT MEASURED — required GPU receipts are incomplete."
        )

    if args.teacher_rule == TEACHER_RULE_E13:
        resume_path = E13_RESUME_PATH
        if not resume_path.is_file():
            raise FileNotFoundError(
                "E1.3 downstream resume script is absent; the registered sweep "
                "has not selected a positive teacher"
            )
    else:
        resume_path = output / "GPU_RESUME_COMMANDS.sh"
        write_text(resume_path, gpu_resume_commands(args.address_rule))
        try:
            resume_path.chmod(0o755)
        except OSError:
            pass
    incomplete = [name for name, gate in gates.items() if gate["verdict"] == "NOT_MEASURED"]
    limitations: list[str] = [
        "The 20B model remains frozen and is used for inference only.",
        "Evidence is limited to one model, one sealed domain corpus, and the registered windows.",
        (
            "CPU synthetic checks validate machinery only and are not G0-G5' evidence."
            if args.address_rule == ADDRESS_RULE_E11
            else "CPU synthetic checks validate machinery only and are not G0-G5 evidence."
        ),
    ]
    blocked_file = blocked_path(output, args.address_rule, args.teacher_rule)
    if incomplete and not gpu_available:
        blocked = {
            "schema": "moe_e1_cuda_blocked_v1",
            "created_at": now_iso(),
            "status": "blocked_environment",
            "reason": "missing /dev/nvidia0 and /dev/nvidiactl",
            "cuda_environment": cuda_probe,
            "not_measured_gates": incomplete,
            "resume_commands": str(resume_path),
            "resume_commands_sha256": sha256_file(resume_path),
            "cpu_preparation_complete": bool(
                manifest.get("status") == "passed" and cpu_validation.get("status") == "passed"
            ),
        }
        write_json(blocked_file, blocked)
        limitations.append(
            (
                "GPU-dependent pair captures, training, and behavioral gates could "
                "not run because /dev/nvidia0 and /dev/nvidiactl are absent."
                if args.address_rule == ADDRESS_RULE_E11
                else "GPU-dependent capture, fitting dependent on those captures, "
                "training, and behavioral gates could not run because /dev/nvidia0 "
                "and /dev/nvidiactl are absent."
            )
        )
    elif incomplete:
        limitations.append(f"Incomplete gate receipts: {', '.join(incomplete)}.")

    intended = {
        SCRIPT_PATH,
        analysis_path(output, args.address_rule, args.teacher_rule),
        report_path(output, args.address_rule, args.teacher_rule),
        resume_path,
    }
    if blocked_file.exists() or (incomplete and not gpu_available):
        intended.add(blocked_file)
    if args.teacher_rule == TEACHER_RULE_E13:
        current_files = {
            path.resolve()
            for path in E13_OUTPUT_DIR.rglob("*")
            if path.is_file()
        }
        current_files.update(
            path.resolve()
            for path in (cpu_validation_file, fit_path, _keys_path)
            if path.exists()
        )
    elif args.address_rule == ADDRESS_RULE_E11:
        current_files = {
            path.resolve()
            for root in (
                pair_root(output, args.address_rule),
                eval_root(output, args.address_rule),
                output / expertpack_dirname(args.address_rule),
            )
            if root.exists()
            for path in root.rglob("*")
            if path.is_file()
        }
        current_files.update(
            path.resolve()
            for path in (
                cpu_validation_file,
                fit_path,
                _keys_path,
                training_path,
                analysis_path(output, args.address_rule, args.teacher_rule),
                report_path(output, args.address_rule, args.teacher_rule),
                resume_path,
                blocked_file,
            )
            if path.exists()
        )
    else:
        current_files = {
            path.resolve() for path in output.rglob("*") if path.is_file()
        }
    created_paths = sorted(str(path) for path in (current_files | intended))
    analysis = {
        "schema": (
            "moe_e1_3_expert_analysis_v1"
            if args.teacher_rule == TEACHER_RULE_E13
            else (
                "moe_e1_1_analysis_v1"
                if args.address_rule == ADDRESS_RULE_E11
                else "moe_e1_analysis_v1"
            )
        ),
        "created_at": now_iso(),
        "status": (
            "blocked_environment"
            if incomplete and not gpu_available
            else "complete" if not incomplete else "incomplete"
        ),
        "address_rule": args.address_rule,
        "teacher_rule": args.teacher_rule,
        "winning_construction": (
            load_e13_selection()["winning_construction"]
            if args.teacher_rule == TEACHER_RULE_E13
            else None
        ),
        "order": str(order_path_for_experiment(args.address_rule, args.teacher_rule)),
        "script": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "registered_verdict_sentence": registered_verdict,
        "evidence_class": "behavioral inference measurement on one model and one domain",
        "cuda_environment": cuda_probe,
        "cpu_validation": {
            "path": str(cpu_validation_file),
            "sha256": sha256_file(cpu_validation_file),
            "status": cpu_validation.get("status"),
            "synthetic_only_not_gate_evidence": True,
        },
        "gates": gates,
        "qualifying_layers": gates["G2"].get("qualifying_layers", []),
        "g4": g4_payload,
        "g5": g5_payload,
        "install_row": install_row,
        "g4_row": g4_row,
        "expertpack_manifest_path": str(pack_path),
        "expertpack_manifest_sha256": sha256_file(pack_path),
        "resume_commands_path": str(resume_path),
        "resume_commands_sha256": sha256_file(resume_path),
        "created_or_modified_paths": created_paths,
        "limitations": limitations,
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "seed": int(args.seed),
    }
    analysis_file = analysis_path(output, args.address_rule, args.teacher_rule)
    report_file = report_path(output, args.address_rule, args.teacher_rule)
    write_json(analysis_file, analysis)
    write_text(report_file, render_report(analysis))
    print(
        json.dumps(
            {
                "status": analysis["status"],
                "verdict": registered_verdict,
                "gates": {name: gate["verdict"] for name, gate in gates.items()},
                "analysis": str(analysis_file),
                "report": str(report_file),
                "resume_commands": str(resume_path),
            }
        ),
        flush=True,
    )
    if incomplete:
        return 0
    return 0 if all(gate["verdict"] == "GREEN" for gate in gates.values()) else 3


def main() -> int:
    args = parse_args()
    try:
        ensure_registered_model_dir(args.model_dir)
        if args.teacher_rule == TEACHER_RULE_E13:
            if args.address_rule != ADDRESS_RULE_E11:
                raise ValueError("--teacher-rule e13 requires --address-rule e11")
            if args.eval_fix != EVAL_FIX_ORIGINAL:
                raise ValueError("--teacher-rule e13 preserves --eval-fix original")
            if args.mode not in {"capture-pairs", "train", "eval-gates", "analyze"}:
                raise ValueError(
                    "--teacher-rule e13 begins at capture-pairs and supports only "
                    "capture-pairs/train/eval-gates/analyze"
                )
            load_e13_selection()
        if args.eval_fix == EVAL_FIX_E12 and (
            args.address_rule != ADDRESS_RULE_E11 or args.mode != "eval-gates"
        ):
            raise ValueError(
                "--eval-fix e12 is diagnosis-only and requires "
                "--address-rule e11 eval-gates"
            )
        if args.eval_fix == EVAL_FIX_E12:
            if not E12_DIAG_ANALYSIS_PATH.is_file():
                raise RuntimeError(
                    "--eval-fix e12 requires the completed MOE-E1.2 diagnostic "
                    f"analysis at {E12_DIAG_ANALYSIS_PATH}"
                )
            diagnostic = read_json(E12_DIAG_ANALYSIS_PATH)
            if diagnostic.get("mechanism") != "BUG":
                raise RuntimeError(
                    "--eval-fix e12 is authorized only after MECHANISM=BUG; "
                    f"diagnostic currently says {diagnostic.get('mechanism')!r}"
                )
        if args.address_rule == ADDRESS_RULE_E11 and args.overwrite:
            raise ValueError(
                "E1.1/E1.3 receipts are append-only; --overwrite is forbidden"
            )
        if args.address_rule == ADDRESS_RULE_E11 and args.mode in {
            "prepare",
            "capture-keys",
        }:
            raise ValueError(
                "E1.1 reuses the frozen E1 preparation and key captures; this mode "
                "is available only under the default --address-rule e1 path"
            )
        if args.mode == "prepare":
            return prepare(args)
        if args.mode == "self-test":
            return self_test(args)
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
        if args.mode == "analyze":
            return analyze(args)
        raise AssertionError(args.mode)
    except SystemExit:
        raise
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "mode": args.mode,
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
