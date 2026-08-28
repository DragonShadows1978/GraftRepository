#!/usr/bin/env python3
"""MOE-E1.4 fixed-backend sanity, teacher re-sweep, and handoff.

The GPU modes are deliberately one-forward units.  Every full-attention layer
must report the fused sink-aware APA backend used by the GPT-OSS context ladder
and 96K lineage; bounded sliding layers remain on their standard chunked path.
CPU modes prepare append-only ``_e14`` inputs, validate orchestration, select
the registered P0-P4 teacher, and render receipt-backed FG verdicts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

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

import gpt_oss20b_e13b_position_controls as e13b
import gpt_oss20b_e1_diag as diag
import gpt_oss20b_expert_e1 as e1


ORDER_PATH = REPO_ROOT / "orders" / "MOE_E1_4_BACKEND_FIX.md"
OUTPUT = e1.E14_OUTPUT_DIR
PREFIX_MANIFEST = e1.E14_PREFIX_MANIFEST_PATH
PREFIX_ARRAYS = e1.E14_PREFIX_ARRAYS_PATH
SELECTION = e1.E14_SELECTION_PATH
RESWEEP_SCRIPT = OUTPUT / "GPU_E14_RESWEEP_COMMANDS.sh"
CHAIN_SCRIPT = e1.E14_CHAIN_PATH
MASTER_SCRIPT = OUTPUT / "GPU_E14_COMMANDS.sh"
REPORT = OUTPUT / "MOE_E1_4_REPORT.md"
OPEN_DEFECT_NOTE = OUTPUT / "PROJECT_TENSOR_OPEN_DEFECT_E14.md"

FG0_LENGTHS = (2048, 2560)
TEACHER_WINDOWS = (11, 13, 15, 1)
CONSTRUCTIONS = ("p0", "p1", "p2", "p3", "p4")
N_TARGETS = e1.WINDOW_TOKENS - 1
FG0_MAX_MEAN_NLL_DELTA = 0.15
C0_EXPLOSION_PPL_RATIO = diag.E13_C0_EXPLOSION_PPL_RATIO
EVIDENCE_CLASS = "behavioral inference measurement, one model, one corpus"
GPU_WRAPPER = (
    "flock -w 7200 /tmp/forge-gpu.lock "
    "timeout --signal=TERM --kill-after=5s 590s "
    "env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 "
    "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ORDER MOE-E1.4 fixed-backend gates and teacher re-sweep."
    )
    parser.add_argument(
        "mode",
        choices=("prepare", "self-test", "sanity", "teacher-score", "analyze"),
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--model-dir", type=Path, default=e1.SNAPSHOT)
    parser.add_argument(
        "--backend",
        choices=e1.BACKENDS,
        default=e1.BACKEND_HISTORICAL,
        help=(
            "historical-standard remains the default; GPU E1.4 score modes "
            "require the explicit --backend e14-apa opt-in"
        ),
    )
    parser.add_argument("--length", type=int, choices=FG0_LENGTHS)
    parser.add_argument("--view", choices=("long", "reference"))
    parser.add_argument(
        "--score-kind", choices=("reference", "c0", "sweep")
    )
    parser.add_argument("--window-index", type=int, choices=TEACHER_WINDOWS)
    parser.add_argument("--construction", choices=CONSTRUCTIONS)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--require-fg0", action="store_true")
    parser.add_argument("--require-c0", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


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


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"append-only E1.4 artifact already exists: {path}")
    atomic_write_json(path, payload)


def write_once_text(path: Path, text: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"append-only E1.4 artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_text_once(path: Path, marker: str, text: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if marker in existing:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    separator = "" if not existing or existing.endswith("\n") else "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(separator + marker + "\n" + text.rstrip() + "\n")


def save_new_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"append-only E1.4 artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def ensure_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != OUTPUT.resolve():
        raise ValueError(f"E1.4 artifacts are restricted to {OUTPUT.resolve()}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_model(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != e1.SNAPSHOT.resolve():
        raise ValueError(f"E1.4 model must be the frozen snapshot {e1.SNAPSHOT}")
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return resolved


def cuda_probe() -> dict[str, Any]:
    entries = sorted(Path("/dev").glob("nvidia*"))
    nodes: list[str] = []
    for path in entries:
        try:
            if stat.S_ISCHR(path.stat().st_mode):
                nodes.append(str(path))
        except OSError:
            pass
    return {
        "device_entries": [str(path) for path in entries],
        "character_device_nodes": nodes,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def require_cuda() -> dict[str, Any]:
    probe = cuda_probe()
    required = {"/dev/nvidia0", "/dev/nvidiactl"}
    if not required.issubset(set(probe["character_device_nodes"])):
        raise RuntimeError(
            "CUDA device nodes unavailable; the lead must run GPU E1.4 score "
            "units under the emitted flock/590s wrapper"
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


def resweep_commands() -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

repo_e14={str(REPO_ROOT)!r}
runner_e14={str(SCRIPT_PATH)!r}
cd "$repo_e14"

run_gpu_e14() {{
  local rc_e14=0
  flock -w 7200 /tmp/forge-gpu.lock \\
    timeout --signal=TERM --kill-after=5s 590s \\
    env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \\
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
      python3 "$runner_e14" "$@" || rc_e14=$?
  sleep 30
  return "$rc_e14"
}}

# CPU preflight. Historical attention remains the parser default.
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \\
  python3 "$runner_e14" prepare
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \\
  python3 "$runner_e14" self-test

# FG0 RUNS FIRST: exact BG2 2048/2560 long/reference cells, now on e14-apa.
for length_e14 in 2048 2560; do
  run_gpu_e14 sanity --backend e14-apa --length "$length_e14" --view long
  run_gpu_e14 sanity --backend e14-apa --length "$length_e14" --view reference
done
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \\
  python3 "$runner_e14" analyze --require-fg0

# Re-run C0 under the fixed backend; its same-target reference is window 11.
run_gpu_e14 teacher-score --backend e14-apa --score-kind reference --window-index 11
run_gpu_e14 teacher-score --backend e14-apa --score-kind c0
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \\
  python3 "$runner_e14" analyze --require-c0

# Remaining references are reused by all five construction arms.
for window_e14 in 13 15 1; do
  run_gpu_e14 teacher-score --backend e14-apa --score-kind reference \\
    --window-index "$window_e14"
done

# E1.3's 16 P1-P4 cells plus four P0 cells (one per registered window).
for construction_e14 in p0 p1 p2 p3 p4; do
  for window_e14 in 11 13 15 1; do
    run_gpu_e14 teacher-score --backend e14-apa --score-kind sweep \\
      --construction "$construction_e14" --window-index "$window_e14"
  done
done

# FG1 selection is sealed only after 20/20 cells; no-positive is a STOP.
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \\
  python3 "$runner_e14" analyze --require-complete
"""


def chain_commands() -> str:
    expert = str(e1.SCRIPT_PATH)
    return f"""#!/usr/bin/env bash
set -euo pipefail

repo_e14={str(REPO_ROOT)!r}
runner_e14={str(SCRIPT_PATH)!r}
expert_e14={expert!r}
cd "$repo_e14"

run_gpu_e14() {{
  local rc_e14=0
  flock -w 7200 /tmp/forge-gpu.lock \\
    timeout --signal=TERM --kill-after=5s 590s \\
    env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \\
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
      python3 "$expert_e14" "$@" || rc_e14=$?
  sleep 30
  return "$rc_e14"
}}

# Hard precondition: FG0 PASS, sane C0, complete positive P0-P4 selection.
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \\
  python3 "$runner_e14" analyze --require-complete

# Append-only _e14 pair captures under the selected teacher and proven backend.
for pair_e14 in $(seq 0 63); do
  run_gpu_e14 capture-pairs --address-rule e11 --teacher-rule e14 \\
    --backend e14-apa --pair-index "$pair_e14"
done

# Unchanged G3 rank-64 CPU consolidation; RED is a registered STOP.
if ! env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \\
  python3 "$expert_e14" train --address-rule e11 --teacher-rule e14 \\
    --backend e14-apa --rank 64 --threads 6; then
  env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \\
    python3 "$expert_e14" analyze --address-rule e11 --teacher-rule e14 \\
      --backend e14-apa || true
  exit 3
fi

# G0 identity is re-registered here, then unchanged G4 narrative windows.
run_gpu_e14 eval-gates --address-rule e11 --teacher-rule e14 \\
  --backend e14-apa --eval-kind abi
for window_e14 in $(seq 0 15); do
  run_gpu_e14 eval-gates --address-rule e11 --teacher-rule e14 \\
    --backend e14-apa --eval-kind narrative --window-index "$window_e14"
done
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \\
  python3 "$expert_e14" analyze --address-rule e11 --teacher-rule e14 \\
    --backend e14-apa || true

# Unchanged registered premise STOP before G5'.
python3 -c 'import json,sys; x=json.load(open("artifacts/moe_e1_4/analysis_chain_e14.json")); sys.exit(0 if x.get("g4",{{}}).get("teacher_gap",0)>0 else 4)'

# E1.1 amended G5': WikiText overall PPL/code fire, generic fire descriptive.
for window_e14 in 1 3 5 7 9 11 13 15; do
  run_gpu_e14 eval-gates --address-rule e11 --teacher-rule e14 \\
    --backend e14-apa --eval-kind generic --window-index "$window_e14"
done
for window_e14 in 1 3 5 7 9 11 13 15; do
  run_gpu_e14 eval-gates --address-rule e11 --teacher-rule e14 \\
    --backend e14-apa --eval-kind code --window-index "$window_e14"
done

# Final unchanged G0/G1/G2'/G3/G4/G5' analysis.
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \\
  python3 "$expert_e14" analyze --address-rule e11 --teacher-rule e14 \\
    --backend e14-apa --bootstrap-resamples 2000
"""


def master_commands() -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

bash {str(RESWEEP_SCRIPT)!r}
bash {str(CHAIN_SCRIPT)!r}
"""


def _with_construction(
    rows: Sequence[dict[str, Any]], construction: str
) -> list[dict[str, Any]]:
    return [{**dict(row), "construction": construction} for row in rows]


def prepare(args: argparse.Namespace) -> int:
    output = ensure_output(args.output_dir)
    ensure_model(args.model_dir)
    if PREFIX_MANIFEST.is_file() or PREFIX_ARRAYS.is_file():
        if not (PREFIX_MANIFEST.is_file() and PREFIX_ARRAYS.is_file()):
            raise FileExistsError("partial append-only E1.4 preparation exists")
        manifest = read_json(PREFIX_MANIFEST)
        if (
            manifest.get("status") != "passed"
            or manifest.get("prefix_arrays", {}).get("sha256")
            != sha256_file(PREFIX_ARRAYS)
        ):
            raise RuntimeError("existing E1.4 preparation failed its sealed hash")
        for path, text in (
            (RESWEEP_SCRIPT, resweep_commands()),
            (CHAIN_SCRIPT, chain_commands()),
            (MASTER_SCRIPT, master_commands()),
        ):
            write_once_text(path, text)
            path.chmod(0o755)
        write_once_text(OPEN_DEFECT_NOTE, open_defect_note())
        print(
            json.dumps(
                {
                    "status": "already_complete",
                    "manifest": str(PREFIX_MANIFEST),
                    "prefix_arrays": str(PREFIX_ARRAYS),
                    "resweep_script": str(RESWEEP_SCRIPT),
                    "chain_script": str(CHAIN_SCRIPT),
                }
            ),
            flush=True,
        )
        return 0

    started = time.perf_counter()
    base_manifest, base_arrays = e1.load_prepared(e1.DEFAULT_OUTPUT_DIR)
    e13_manifest, e13_arrays = diag.load_e13_prepared(diag.E13_OUTPUT)
    e13b_manifest, _controls = e13b.validate_prepared()
    arrays: dict[str, np.ndarray] = {
        "pair_p0": np.ascontiguousarray(base_arrays["pair_prefix_ids"]),
        "behavioral_p0": np.ascontiguousarray(
            base_arrays["behavioral_prefix_ids"]
        ),
        "c0_stream": np.ascontiguousarray(e13_arrays["c0_stream"]),
    }
    for construction in ("p1", "p2", "p3", "p4"):
        arrays[f"pair_{construction}"] = np.ascontiguousarray(
            e13_arrays[f"pair_{construction}"]
        )
        arrays[f"behavioral_{construction}"] = np.ascontiguousarray(
            e13_arrays[f"behavioral_{construction}"]
        )
    for name, array in arrays.items():
        if array.dtype != np.int64:
            raise RuntimeError(f"E1.4 prefix array {name} dtype {array.dtype}")
    save_new_npz(PREFIX_ARRAYS, arrays)

    for path, text in (
        (RESWEEP_SCRIPT, resweep_commands()),
        (CHAIN_SCRIPT, chain_commands()),
        (MASTER_SCRIPT, master_commands()),
    ):
        write_once_text(path, text)
        path.chmod(0o755)

    p0_pair = _with_construction(
        base_manifest["windows"]["pair_prefixes"], "p0"
    )
    p0_behavioral = _with_construction(
        base_manifest["windows"]["behavioral_prefixes"], "p0"
    )
    prefix_records = {
        "pair": {
            "p0": p0_pair,
            **{
                construction: e13_manifest["prefixes"]["pair"][construction]
                for construction in ("p1", "p2", "p3", "p4")
            },
        },
        "behavioral": {
            "p0": p0_behavioral,
            **{
                construction: e13_manifest["prefixes"]["behavioral"][construction]
                for construction in ("p1", "p2", "p3", "p4")
            },
        },
    }
    array_records = {
        name: {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": sha256_array(array),
        }
        for name, array in arrays.items()
    }
    recipes = {
        "p0": (
            "original registered E1 raw-excerpt rotation: select TRAIN files with "
            ">=2,048 tokens in SHA-256(filename) order; derive the rotation start "
            "from SHA-256(seed|role|prefix-rotation), rotate one source per target, "
            "and derive the exact 2,048-token excerpt start from "
            "SHA-256(seed|role|index|relative-source); pair targets avoid their own source"
        ),
        **e13_manifest["recipes"],
    }
    manifest = {
        "schema": "moe_e1_4_teacher_prefix_manifest_v1",
        "created_at": now_iso(),
        "status": "passed",
        "order": str(ORDER_PATH),
        "order_sha256": sha256_file(ORDER_PATH),
        "script": str(SCRIPT_PATH),
        "script_sha256_at_prepare": sha256_file(SCRIPT_PATH),
        "expert_script": str(e1.SCRIPT_PATH),
        "expert_script_sha256_at_prepare": sha256_file(e1.SCRIPT_PATH),
        "model_dir": str(args.model_dir.resolve()),
        "seed": e1.DEFAULT_SEED,
        "backend_registration": e1.backend_receipt_contract(e1.BACKEND_E14_APA),
        "registered_windows_in_order": list(TEACHER_WINDOWS),
        "registered_gpu_units": {
            "fg0": 4,
            "c0": 1,
            "teacher_references": 4,
            "p1_p4_cells_from_e13": 16,
            "p0_cells_added_by_e14": 4,
            "teacher_selection_cells_total": 20,
        },
        "selection_rule": (
            "highest arithmetic mean of (reference mean_nll - teacher mean_nll) "
            "over windows [11,13,15,1]; require mean > 0; exact ties break by "
            "lower P number; P0 winning means no redesign was needed and the "
            "E1.1 teacher stands"
        ),
        "recipes": recipes,
        "prose_filter": e13_manifest["prose_filter"],
        "frame": e13_manifest["frame"],
        "prefixes": prefix_records,
        "c0": e13_manifest["c0"],
        "prefix_arrays": {
            "path": str(PREFIX_ARRAYS),
            "sha256": sha256_file(PREFIX_ARRAYS),
            "arrays": array_records,
        },
        "dependencies": {
            "e1_manifest": {
                "path": str(e1.DEFAULT_OUTPUT_DIR / "corpus_manifest.json"),
                "sha256": sha256_file(e1.DEFAULT_OUTPUT_DIR / "corpus_manifest.json"),
            },
            "e1_prepared_windows": {
                "path": str(e1.DEFAULT_OUTPUT_DIR / "prepared_windows.npz"),
                "sha256": sha256_file(e1.DEFAULT_OUTPUT_DIR / "prepared_windows.npz"),
            },
            "e13_prefix_manifest": {
                "path": str(diag.E13_PREFIX_MANIFEST),
                "sha256": sha256_file(diag.E13_PREFIX_MANIFEST),
            },
            "e13_prefix_arrays": {
                "path": str(diag.E13_PREFIX_ARRAYS),
                "sha256": sha256_file(diag.E13_PREFIX_ARRAYS),
            },
            "e13b_control_manifest": {
                "path": str(e13b.CONTROL_MANIFEST),
                "sha256": sha256_file(e13b.CONTROL_MANIFEST),
                "status": e13b_manifest["status"],
            },
        },
        "scripts": {
            "resweep": {"path": str(RESWEEP_SCRIPT), "sha256": sha256_file(RESWEEP_SCRIPT)},
            "chain": {"path": str(CHAIN_SCRIPT), "sha256": sha256_file(CHAIN_SCRIPT)},
            "master": {"path": str(MASTER_SCRIPT), "sha256": sha256_file(MASTER_SCRIPT)},
        },
        "wall_seconds": float(time.perf_counter() - started),
    }
    write_new_json(PREFIX_MANIFEST, manifest)
    write_once_text(OPEN_DEFECT_NOTE, open_defect_note())
    print(
        json.dumps(
            {
                "status": "passed",
                "manifest": str(PREFIX_MANIFEST),
                "prefix_arrays": str(PREFIX_ARRAYS),
                "resweep_script": str(RESWEEP_SCRIPT),
                "chain_script": str(CHAIN_SCRIPT),
                "master_script": str(MASTER_SCRIPT),
            }
        ),
        flush=True,
    )
    return 0


def open_defect_note() -> str:
    rows: list[str] = []
    for length in FG0_LENGTHS:
        long_path = e13b.OUTPUT / "runs" / f"ramp_{length}_long.json"
        reference_path = e13b.OUTPUT / "runs" / f"ramp_{length}_reference.json"
        long_receipt = read_json(long_path)
        reference_receipt = read_json(reference_path)
        long_score = long_receipt["score"]
        reference_score = reference_receipt["score"]
        delta = float(long_score["mean_nll"]) - float(reference_score["mean_nll"])
        rows.append(
            f"| {length} | {long_score['mean_nll']:.9f} | "
            f"{reference_score['mean_nll']:.9f} | {delta:+.9f} | "
            f"{float(long_score['ppl']) / float(reference_score['ppl']):.9f}x | "
            f"`{long_path}` (`{sha256_file(long_path)}`) | "
            f"`{reference_path}` (`{sha256_file(reference_path)}`) |"
        )
    return "\n".join(
        [
            "# Project-Tensor-adjacent open defect — GPT-OSS standard full attention",
            "",
            "Status: OPEN; report only under ORDER MOE-E1.4. No engine fix was attempted.",
            "",
            "The standard full-attention mask/sink path in `core/gpt_oss20b_tc.py` "
            "degrades as sequence length grows on identical WikiText targets. The "
            "interleaved bounded sliding layers remain on their healthy chunked backend. "
            "This is a behavioral inference on one frozen model and one corpus, not a "
            "localized kernel proof.",
            "",
            "| Length | long mean NLL | reference mean NLL | delta | PPL ratio | long receipt (SHA-256) | reference receipt (SHA-256) |",
            "|---:|---:|---:|---:|---:|---|---|",
            *rows,
            "",
            "Successor scope: diagnose and repair the standard full-attention "
            "mask/sink backend in Project-Tensor/core integration. E1.4 instead moves "
            "long E1 forwards to the already-proven fused `apa_selective_attention_sink` "
            "lineage and preserves the historical standard path as the default.",
            "",
        ]
    )


def load_prepared() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not PREFIX_MANIFEST.is_file() or not PREFIX_ARRAYS.is_file():
        raise FileNotFoundError("run E1.4 prepare first")
    manifest = read_json(PREFIX_MANIFEST)
    if manifest.get("status") != "passed":
        raise RuntimeError("E1.4 prefix preparation is not passed")
    if manifest["prefix_arrays"]["sha256"] != sha256_file(PREFIX_ARRAYS):
        raise RuntimeError("E1.4 prefix array artifact hash changed")
    with np.load(PREFIX_ARRAYS, allow_pickle=False) as stored:
        arrays = {name: np.ascontiguousarray(stored[name]) for name in stored.files}
    expected = {
        *(f"pair_{construction}" for construction in CONSTRUCTIONS),
        *(f"behavioral_{construction}" for construction in CONSTRUCTIONS),
        "c0_stream",
    }
    if set(arrays) != expected:
        raise RuntimeError(
            f"E1.4 prefix array names {sorted(arrays)} != {sorted(expected)}"
        )
    for name, array in arrays.items():
        declared = manifest["prefix_arrays"]["arrays"][name]
        if (
            list(array.shape) != declared["shape"]
            or str(array.dtype) != declared["dtype"]
            or sha256_array(array) != declared["sha256"]
        ):
            raise RuntimeError(f"E1.4 prefix array contract mismatch: {name}")
    return manifest, arrays


def sanity_inputs(length: int, view: str) -> dict[str, Any]:
    _manifest, controls = e13b.validate_prepared()
    return e13b.case_inputs(
        controls["wikitext_0_2560"],
        controls["guide_w11"],
        case_name="ramp",
        length=length,
        view=view,
    )


def teacher_inputs(
    score_kind: str,
    *,
    construction: str | None,
    window_index: int | None,
) -> dict[str, Any]:
    prefix_manifest, prefix_arrays = load_prepared()
    base_manifest, base_arrays = e1.load_prepared(e1.DEFAULT_OUTPUT_DIR)
    if score_kind == "reference":
        if window_index not in TEACHER_WINDOWS or construction is not None:
            raise ValueError("reference scoring requires a registered window only")
        index = int(window_index)
        window = np.ascontiguousarray(
            base_arrays["behavioral_ids"][index], dtype=np.int64
        )
        prefix = np.empty((0,), dtype=np.int64)
        stream = window
        source = base_manifest["windows"]["behavioral_heldout"][index]
        construction_name = "reference"
    elif score_kind == "c0":
        if construction is not None or window_index is not None:
            raise ValueError("C0 accepts no construction/window selector")
        index = int(prefix_manifest["c0"]["window_index"])
        stream = np.ascontiguousarray(prefix_arrays["c0_stream"], dtype=np.int64)
        window = np.ascontiguousarray(stream[-e1.WINDOW_TOKENS :])
        if not np.array_equal(window, base_arrays["behavioral_ids"][index]):
            raise RuntimeError("E1.4 C0 tail differs from sealed behavioral window")
        prefix = np.ascontiguousarray(stream[: -e1.WINDOW_TOKENS])
        source = prefix_manifest["c0"]["construction_source"]
        construction_name = "c0_contiguous"
    elif score_kind == "sweep":
        if construction not in CONSTRUCTIONS or window_index not in TEACHER_WINDOWS:
            raise ValueError("sweep scoring requires a registered construction/window")
        index = int(window_index)
        prefix = np.ascontiguousarray(
            prefix_arrays[f"behavioral_{construction}"][index], dtype=np.int64
        )
        window = np.ascontiguousarray(
            base_arrays["behavioral_ids"][index], dtype=np.int64
        )
        stream = np.concatenate([prefix, window]).astype(np.int64, copy=False)
        source = prefix_manifest["prefixes"]["behavioral"][construction][index]
        construction_name = construction
    else:
        raise AssertionError(score_kind)
    inputs = np.ascontiguousarray(stream[:-1][None, :], dtype=np.int64)
    targets = np.ascontiguousarray(window[1:], dtype=np.int64)
    if targets.shape != (N_TARGETS,):
        raise RuntimeError(f"E1.4 target shape {targets.shape}")
    return {
        "score_kind": score_kind,
        "construction": construction_name,
        "window_index": index,
        "prefix": prefix,
        "window": window,
        "full_sequence": np.ascontiguousarray(stream),
        "inputs": inputs,
        "targets": targets,
        "prefix_source": source,
        "source_window": base_manifest["windows"]["behavioral_heldout"][index],
    }


def sanity_receipt_path(length: int, view: str, attempt: int = 0) -> Path:
    suffix = "" if attempt == 0 else f"_attempt{attempt:02d}"
    return OUTPUT / "runs_e14" / "fg0" / f"ramp_{length}_{view}_e14{suffix}.json"


def teacher_receipt_path(
    score_kind: str,
    *,
    construction: str | None = None,
    window_index: int | None = None,
    attempt: int = 0,
) -> Path:
    suffix = "" if attempt == 0 else f"_attempt{attempt:02d}"
    root = OUTPUT / "runs_e14" / "teacher"
    if score_kind == "reference" and window_index is not None:
        return root / f"reference_window_{window_index:03d}_e14{suffix}.json"
    if score_kind == "c0":
        return root / f"c0_contiguous_e14{suffix}.json"
    if score_kind == "sweep" and construction and window_index is not None:
        return (
            root
            / f"window_{window_index:03d}"
            / f"{construction}_e14{suffix}.json"
        )
    raise ValueError("invalid E1.4 teacher receipt selector")


def forward_once(
    *,
    args: argparse.Namespace,
    data: dict[str, Any],
    receipt: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    if args.backend != e1.BACKEND_E14_APA:
        raise ValueError("E1.4 GPU score modes require --backend e14-apa")
    runtime = e1.load_runtime()
    tc = runtime["tc"]
    if not hasattr(tc, "apa_selective_attention_sink"):
        raise RuntimeError(
            "TensorCUDA lacks apa_selective_attention_sink; E1.4 refuses a blend fallback"
        )
    cfg = runtime["GptOss20BConfig"].from_model_dir(args.model_dir.resolve())
    e1.validate_model_contract(cfg)
    where = runtime["build_safetensors_map"](args.model_dir.resolve())
    with tc.no_grad():
        embed = runtime["GptOssRowEmbedding"](where)
        hidden = embed(data["inputs"])
        rope_len = int(data["inputs"].shape[1])
        cos, sin = runtime["gpt_oss_yarn_rope_tables"](cfg, rope_len)
        receipt["rope_table_length"] = rope_len
        receipt["layers"] = []
        for layer in range(e1.N_LAYERS):
            layer_started = time.perf_counter()
            block = runtime["GptOssDiagnosticBlockTC"].from_safetensors(
                cfg,
                where,
                layer,
                expert_mode="resident_packed_mxfp4",
            )
            e1.configure_block(block, args.backend)
            if (
                block.self_attn.inject_kv is not None
                or int(block.self_attn.graft_seats) != 0
            ):
                raise RuntimeError("E1.4 base/teacher scoring unexpectedly mounted a graft")
            hidden, kv, route = block(hidden, cos, sin)
            tc.synchronize()
            observed_backend = e1.verify_block_attention_backend(block, args.backend)
            receipt["layers"].append(
                {
                    "layer": int(layer),
                    "layer_type": cfg.layer_types[layer],
                    "effective_attention_mode": block.self_attn.attention_mode,
                    "attention_backend": observed_backend,
                    "sliding_window": block.self_attn.sliding_window,
                    "refine_percentile": float(block.self_attn.refine_percentile),
                    "bulk_bits": int(block.self_attn.bulk_bits),
                    "wall_seconds": float(time.perf_counter() - layer_started),
                }
            )
            receipt["completed_layers"] = layer + 1
            receipt["status"] = "running_layers"
            atomic_write_json(path, receipt)
            del block, kv, route
            gc.collect()
            if hasattr(tc, "empty_cache"):
                tc.empty_cache()
        tail = hidden.slice(1, int(hidden.shape[1]) - N_TARGETS, N_TARGETS)
        normalized = e1.final_norm(runtime, cfg, where, tail)
        del hidden, tail, cos, sin
        if hasattr(tc, "empty_cache"):
            tc.empty_cache()
        lm_head = e1.load_lm_head(runtime, cfg, where)
        score = e1.nll_from_hidden(lm_head, normalized, data["targets"])
        del lm_head, normalized, embed
        tc.synchronize()
    if score["target_ids_sha256"] != sha256_array(data["targets"]):
        raise RuntimeError("E1.4 scored target hash mismatch")
    counts = Counter(row["attention_backend"] for row in receipt["layers"])
    expected = {
        "apa_selective_sink_fused": 12,
        "standard_sink_sliding_chunked": 12,
    }
    if dict(counts) != expected:
        raise RuntimeError(f"E1.4 attention backend counts {dict(counts)} != {expected}")
    receipt["attention_backend_counts"] = dict(counts)
    return score


def _begin_receipt(
    *,
    args: argparse.Namespace,
    path: Path,
    schema: str,
    data: dict[str, Any],
    fields: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    if path.is_file():
        prior = read_json(path)
        if prior.get("status") == "complete":
            print(json.dumps({"status": "already_complete", "receipt": str(path)}))
            raise SystemExit(0)
        raise FileExistsError(
            f"append-only E1.4 attempt exists at {path}; select a new --attempt"
        )
    probe = require_cuda()
    started = time.perf_counter()
    receipt: dict[str, Any] = {
        "schema": schema,
        "created_at": now_iso(),
        "status": "starting",
        "argv": sys.argv,
        "required_shell_wrapper": GPU_WRAPPER,
        "order": str(ORDER_PATH),
        "order_sha256": sha256_file(ORDER_PATH),
        "backend": args.backend,
        "attention_registration": e1.backend_receipt_contract(args.backend),
        "model_dir": str(args.model_dir.resolve()),
        "compute_dtype": "bfloat16",
        "expert_mode": "resident_packed_mxfp4",
        "execution_schedule": "one_sequence_call_per_loaded_block",
        "input_tokens": int(data["inputs"].shape[1]),
        "full_sequence_tokens": int(data["full_sequence"].size),
        "scored_target_tokens": int(data["targets"].size),
        "input_ids_sha256": sha256_array(data["inputs"]),
        "full_sequence_ids_sha256": sha256_array(data["full_sequence"]),
        "target_ids_sha256": sha256_array(data["targets"]),
        "cuda_environment": probe,
        "gpu_before": nvidia_smi(),
        "script_sha256_at_run": sha256_file(SCRIPT_PATH),
        "expert_script_sha256_at_run": sha256_file(e1.SCRIPT_PATH),
        "core_script_sha256_at_run": sha256_file(e13b.CORE_PATH),
        "completed_layers": 0,
        **fields,
    }
    atomic_write_json(path, receipt)
    return receipt, started


def _finish_score(
    *,
    args: argparse.Namespace,
    data: dict[str, Any],
    receipt: dict[str, Any],
    path: Path,
    started: float,
) -> int:
    try:
        result = forward_once(args=args, data=data, receipt=receipt, path=path)
        receipt.update(
            {
                "status": "complete",
                "score": result,
                "wall_seconds": float(time.perf_counter() - started),
                "gpu_after": nvidia_smi(),
            }
        )
        atomic_write_json(path, receipt)
        print(
            json.dumps(
                {"status": "complete", "receipt": str(path), "score": result}
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
        atomic_write_json(path, receipt)
        raise


def sanity(args: argparse.Namespace) -> int:
    if args.backend != e1.BACKEND_E14_APA:
        raise ValueError("sanity requires explicit --backend e14-apa")
    if args.length not in FG0_LENGTHS or args.view not in {"long", "reference"}:
        raise ValueError("sanity requires --length 2048|2560 and --view long|reference")
    if args.attempt < 0:
        raise ValueError("--attempt must be nonnegative")
    ensure_output(args.output_dir)
    ensure_model(args.model_dir)
    load_prepared()
    data = sanity_inputs(int(args.length), str(args.view))
    path = sanity_receipt_path(int(args.length), str(args.view), args.attempt)
    receipt, started = _begin_receipt(
        args=args,
        path=path,
        schema="moe_e1_4_fg0_score_v1",
        data=data,
        fields={
            "gate": "FG0",
            "length": int(args.length),
            "view": str(args.view),
            "source": data["source"],
            "target_positions": data["target_positions"],
            "e13b_control_manifest": str(e13b.CONTROL_MANIFEST),
            "e13b_control_manifest_sha256": sha256_file(e13b.CONTROL_MANIFEST),
            "e13b_controls": str(e13b.CONTROL_ARRAYS),
            "e13b_controls_sha256": sha256_file(e13b.CONTROL_ARRAYS),
            "historical_bg2_receipt": str(
                e13b.OUTPUT
                / "runs"
                / f"ramp_{int(args.length)}_{str(args.view)}.json"
            ),
        },
    )
    return _finish_score(
        args=args, data=data, receipt=receipt, path=path, started=started
    )


def teacher_score(args: argparse.Namespace) -> int:
    if args.backend != e1.BACKEND_E14_APA:
        raise ValueError("teacher-score requires explicit --backend e14-apa")
    if args.score_kind is None:
        raise ValueError("teacher-score requires --score-kind reference|c0|sweep")
    if args.attempt < 0:
        raise ValueError("--attempt must be nonnegative")
    if args.score_kind == "reference":
        if args.window_index is None or args.construction is not None:
            raise ValueError("reference requires --window-index and no construction")
    elif args.score_kind == "c0":
        if args.window_index is not None or args.construction is not None:
            raise ValueError("C0 accepts no window/construction selector")
    elif args.window_index is None or args.construction is None:
        raise ValueError("sweep requires --window-index and --construction")
    ensure_output(args.output_dir)
    ensure_model(args.model_dir)
    data = teacher_inputs(
        args.score_kind,
        construction=args.construction,
        window_index=args.window_index,
    )
    path = teacher_receipt_path(
        args.score_kind,
        construction=args.construction,
        window_index=args.window_index,
        attempt=args.attempt,
    )
    receipt, started = _begin_receipt(
        args=args,
        path=path,
        schema="moe_e1_4_teacher_score_v1",
        data=data,
        fields={
            "gate": "FG1",
            "score_kind": args.score_kind,
            "construction": data["construction"],
            "window_index": int(data["window_index"]),
            "prefix_tokens": int(data["prefix"].size),
            "prefix_ids_sha256": sha256_array(data["prefix"]),
            "source_window": data["source_window"],
            "prefix_source": data["prefix_source"],
            "teacher_scores_prefix_tokens": False,
            "prefix_manifest": str(PREFIX_MANIFEST),
            "prefix_manifest_sha256": sha256_file(PREFIX_MANIFEST),
            "prefix_arrays": str(PREFIX_ARRAYS),
            "prefix_arrays_sha256": sha256_file(PREFIX_ARRAYS),
        },
    )
    return _finish_score(
        args=args, data=data, receipt=receipt, path=path, started=started
    )


def choose_teacher(means: dict[str, float | None]) -> tuple[str | None, float | None]:
    if any(means.get(name) is None for name in CONSTRUCTIONS):
        return None, None
    ranked = sorted(
        CONSTRUCTIONS,
        key=lambda name: (-float(means[name]), CONSTRUCTIONS.index(name)),
    )
    best = ranked[0]
    best_gap = float(means[best])
    return (best, best_gap) if best_gap > 0.0 else (None, best_gap)


class _DummyAttention:
    def __init__(self, sliding_window: int | None) -> None:
        self.sliding_window = sliding_window
        self.attention_mode = "unset"
        self.refine_percentile = -1.0
        self.bulk_bits = -1
        self.last_attention_backend = ""


class _DummyMlp:
    def __init__(self) -> None:
        self.route_detail = "unset"
        self.empty_cache_interval = -1


class _DummyBlock:
    def __init__(self, sliding_window: int | None) -> None:
        self.self_attn = _DummyAttention(sliding_window)
        self.mlp = _DummyMlp()


def _add_check(
    checks: list[dict[str, Any]], name: str, passed: bool, observed: Any
) -> None:
    checks.append({"name": name, "passed": bool(passed), "observed": observed})


def self_test(args: argparse.Namespace) -> int:
    ensure_output(args.output_dir)
    ensure_model(args.model_dir)
    manifest, arrays = load_prepared()
    checks: list[dict[str, Any]] = []

    full = _DummyBlock(None)
    sliding = _DummyBlock(128)
    e1.configure_block(full, e1.BACKEND_E14_APA)
    e1.configure_block(sliding, e1.BACKEND_E14_APA)
    _add_check(
        checks,
        "e14_full_requests_apa_selective",
        full.self_attn.attention_mode == "apa_selective"
        and full.self_attn.refine_percentile == 0.15
        and full.self_attn.bulk_bits == 8,
        vars(full.self_attn),
    )
    _add_check(
        checks,
        "e14_sliding_preserves_standard",
        sliding.self_attn.attention_mode == "standard"
        and sliding.self_attn.refine_percentile == 0.15
        and sliding.self_attn.bulk_bits == 8,
        vars(sliding.self_attn),
    )
    full.self_attn.last_attention_backend = "apa_selective_sink_fused"
    sliding.self_attn.last_attention_backend = "standard_sink_sliding_chunked"
    _add_check(
        checks,
        "e14_exact_backend_verifier",
        e1.verify_block_attention_backend(full, e1.BACKEND_E14_APA)
        == "apa_selective_sink_fused"
        and e1.verify_block_attention_backend(sliding, e1.BACKEND_E14_APA)
        == "standard_sink_sliding_chunked",
        [full.self_attn.last_attention_backend, sliding.self_attn.last_attention_backend],
    )

    historical = _DummyBlock(None)
    e1.configure_block(historical)
    _add_check(
        checks,
        "historical_configure_default_standard",
        historical.self_attn.attention_mode == "standard",
        historical.self_attn.attention_mode,
    )
    saved_argv = sys.argv
    try:
        sys.argv = [str(e1.SCRIPT_PATH), "analyze"]
        defaults = e1.parse_args()
    finally:
        sys.argv = saved_argv
    _add_check(
        checks,
        "expert_parser_historical_default",
        defaults.backend == e1.BACKEND_HISTORICAL
        and defaults.teacher_rule == e1.TEACHER_RULE_ORIGINAL,
        {"backend": defaults.backend, "teacher_rule": defaults.teacher_rule},
    )
    _add_check(
        checks,
        "historical_paths_unchanged_by_default",
        e1.pair_root(e1.DEFAULT_OUTPUT_DIR, e1.ADDRESS_RULE_E11)
        == e1.DEFAULT_OUTPUT_DIR / "pairs_e11"
        and e1.eval_root(e1.DEFAULT_OUTPUT_DIR, e1.ADDRESS_RULE_E11)
        == e1.DEFAULT_OUTPUT_DIR / "eval_e11",
        {
            "pairs": str(e1.pair_root(e1.DEFAULT_OUTPUT_DIR, e1.ADDRESS_RULE_E11)),
            "eval": str(e1.eval_root(e1.DEFAULT_OUTPUT_DIR, e1.ADDRESS_RULE_E11)),
        },
    )

    for length in FG0_LENGTHS:
        long = sanity_inputs(length, "long")
        reference = sanity_inputs(length, "reference")
        _add_check(
            checks,
            f"fg0_{length}_same_targets",
            np.array_equal(long["targets"], reference["targets"]),
            {
                "long": sha256_array(long["targets"]),
                "reference": sha256_array(reference["targets"]),
            },
        )
    for construction in CONSTRUCTIONS:
        array = arrays[f"behavioral_{construction}"]
        _add_check(
            checks,
            f"{construction}_behavioral_count",
            array.shape[0] == e1.N_BEHAVIORAL_WINDOWS,
            list(array.shape),
        )
        for window in TEACHER_WINDOWS:
            cell = teacher_inputs(
                "sweep", construction=construction, window_index=window
            )
            reference = teacher_inputs(
                "reference", construction=None, window_index=window
            )
            _add_check(
                checks,
                f"{construction}_w{window}_same_targets",
                np.array_equal(cell["targets"], reference["targets"]),
                sha256_array(cell["targets"]),
            )
    _add_check(
        checks,
        "selection_prefers_highest_positive_and_p0_tie_order",
        choose_teacher(
            {"p0": 0.2, "p1": 0.2, "p2": 0.1, "p3": -0.1, "p4": 0.0}
        )
        == ("p0", 0.2),
        choose_teacher(
            {"p0": 0.2, "p1": 0.2, "p2": 0.1, "p3": -0.1, "p4": 0.0}
        ),
    )
    _add_check(
        checks,
        "selection_requires_positive_mean",
        choose_teacher(
            {"p0": 0.0, "p1": -0.1, "p2": -0.2, "p3": -0.3, "p4": -0.4}
        )[0]
        is None,
        "no winner",
    )

    rng = np.random.default_rng(e1.DEFAULT_SEED + 14)
    native = rng.normal(size=(2, 5, 8)).astype(np.float32)
    hidden = rng.normal(size=native.shape).astype(np.float32)
    key = rng.normal(size=8).astype(np.float32)
    A = rng.normal(size=(4, 8)).astype(np.float32)
    B = rng.normal(size=(8, 4)).astype(np.float32)
    nofire, _scores, fired = e1.numpy_expert_add(
        native, hidden, key, math.inf, A, B
    )
    _add_check(
        checks,
        "g0_nofire_identity_backend_independent",
        not fired.any() and np.array_equal(native, nofire),
        {"fire_count": int(fired.sum()), "bit_identical": np.array_equal(native, nofire)},
    )

    scripts = {
        "resweep": resweep_commands(),
        "chain": chain_commands(),
        "master": master_commands(),
    }
    for label, shell in scripts.items():
        for law, needle in (
            ("flock", "flock -w 7200 /tmp/forge-gpu.lock"),
            ("timeout", "timeout --signal=TERM --kill-after=5s 590s"),
            ("single_gpu", "CUDA_VISIBLE_DEVICES=0"),
            ("gap", "sleep 30"),
        ):
            if label == "master":
                continue
            _add_check(checks, f"{label}_{law}", needle in shell, needle)
    resweep = scripts["resweep"]
    fg0_pos = resweep.index("# FG0 RUNS FIRST")
    teacher_pos = resweep.index("# Re-run C0")
    _add_check(
        checks,
        "fg0_precedes_teacher_work",
        fg0_pos < teacher_pos and "analyze --require-fg0" in resweep,
        {"fg0_position": fg0_pos, "teacher_position": teacher_pos},
    )
    chain = scripts["chain"]
    _add_check(
        checks,
        "chain_is_all_e14_and_carries_registered_gates",
        all(
            needle in chain
            for needle in (
                "--teacher-rule e14",
                "--backend e14-apa",
                "--eval-kind abi",
                "--eval-kind narrative",
                "--eval-kind generic",
                "--eval-kind code",
                "--bootstrap-resamples 2000",
            )
        ),
        "pairs/train/G0/G4/G5'/analyze present",
    )
    _add_check(
        checks,
        "prefix_manifest_backend_and_cell_registration",
        manifest["backend_registration"]
        == e1.backend_receipt_contract(e1.BACKEND_E14_APA)
        and manifest["registered_gpu_units"]["teacher_selection_cells_total"] == 20,
        manifest["registered_gpu_units"],
    )

    passed = all(item["passed"] for item in checks)
    script_hash = sha256_file(SCRIPT_PATH)
    payload = {
        "schema": "moe_e1_4_cpu_self_test_v1",
        "created_at": now_iso(),
        "status": "pass" if passed else "fail",
        "evidence_class": "synthetic CPU machinery check; not model behavior",
        "script": str(SCRIPT_PATH),
        "script_sha256": script_hash,
        "expert_script": str(e1.SCRIPT_PATH),
        "expert_script_sha256": sha256_file(e1.SCRIPT_PATH),
        "checks": checks,
        "passed": sum(1 for item in checks if item["passed"]),
        "total": len(checks),
        "gpu_used": False,
    }
    path = OUTPUT / "self_test_e14" / f"cpu_self_test_e14_{script_hash[:16]}.json"
    if path.is_file():
        prior = read_json(path)
        if prior.get("status") != payload["status"]:
            raise RuntimeError(f"existing E1.4 self-test status differs: {path}")
        status = "already_complete"
    else:
        write_new_json(path, payload)
        status = payload["status"]
    print(
        json.dumps(
            {
                "status": status,
                "self_test": str(path),
                "passed": payload["passed"],
                "total": payload["total"],
            }
        ),
        flush=True,
    )
    return 0 if passed else 2


def completed_receipt(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = read_json(path)
    if payload.get("status") != "complete":
        return None
    if payload.get("backend") != e1.BACKEND_E14_APA:
        raise RuntimeError(f"non-e14 receipt found in E1.4 path: {path}")
    expected = {
        "apa_selective_sink_fused": 12,
        "standard_sink_sliding_chunked": 12,
    }
    if payload.get("attention_backend_counts") != expected:
        raise RuntimeError(f"E1.4 receipt backend counts changed: {path}")
    payload["_path"] = str(path)
    payload["_sha256"] = sha256_file(path)
    return payload


def score_summary(receipt: dict[str, Any]) -> dict[str, Any]:
    score = receipt["score"]
    return {
        "mean_nll": float(score["mean_nll"]),
        "ppl": float(score["ppl"]),
        "nll_sum": float(score["nll_sum"]),
        "token_count": int(score["token_count"]),
        "target_ids_sha256": score["target_ids_sha256"],
        "receipt": receipt["_path"],
        "receipt_sha256": receipt["_sha256"],
    }


def fg0_table(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| Length | long mean_nll | reference mean_nll | delta (long - reference) | long ppl | reference ppl | verdict |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if not row["complete"]:
            lines.append(f"| {row['length']} | n/a | n/a | n/a | n/a | n/a | NOT_MEASURED |")
        else:
            lines.append(
                f"| {row['length']} | {row['long']['mean_nll']:.9f} | "
                f"{row['reference']['mean_nll']:.9f} | {row['mean_nll_delta']:+.9f} | "
                f"{row['long']['ppl']:.6f} | {row['reference']['ppl']:.6f} | "
                f"{'PASS' if row['passes'] else 'FAIL'} |"
            )
    return "\n".join(lines)


def teacher_table(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| Construction | Window | mean_nll_base | mean_nll_teacher | gap (base - teacher) | ppl_base | ppl_teacher |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["status"] != "complete":
            lines.append(
                f"| {row['construction'].upper()} | {row['window_index']} | n/a | n/a | n/a | n/a | n/a |"
            )
        else:
            lines.append(
                f"| {row['construction'].upper()} | {row['window_index']} | "
                f"{row['mean_nll_base']:.9f} | {row['mean_nll_teacher']:.9f} | "
                f"{row['gap']:+.9f} | {row['ppl_base']:.6f} | "
                f"{row['ppl_teacher']:.6f} |"
            )
    return "\n".join(lines)


def selection_sentence_for(
    winner: str | None,
    best_gap: float | None,
    *,
    complete: bool,
) -> str:
    if not complete:
        return "Registered P0-P4 selection awaits all 20 fixed-backend cells."
    if winner == "p0":
        return (
            f"E1.4 registered selection: P0 has the highest mean gap "
            f"({float(best_gap):+.9f} > 0); no redesign was needed and the "
            "E1.1 teacher stands."
        )
    if winner is not None:
        return (
            f"E1.4 registered selection: {winner.upper()} has the highest mean "
            f"gap ({float(best_gap):+.9f} > 0) and becomes the fixed-backend teacher."
        )
    return "The E1.1 premise finding STANDS for this model+corpus."


def render_report(analysis: dict[str, Any]) -> str:
    gates = analysis["gates"]
    c0 = analysis["c0"]
    lines = [
        "# MOE-E1.4 Backend Re-specification Report",
        "",
        f"Snapshot generated: `{analysis['created_at']}`",
        "",
        f"Evidence class: {EVIDENCE_CLASS}. CPU checks are synthetic machinery checks, not behavioral evidence.",
        "",
        "## FG verdicts",
        "",
        gates["FG0"]["line"],
        gates["FG1"]["line"],
        gates["FG2"]["line"],
        "",
        "## FG0 backend sanity",
        "",
        analysis["fg0_table_markdown"],
        "",
        f"Registered PASS rail: long/reference mean-NLL delta <= {FG0_MAX_MEAN_NLL_DELTA:.2f} nats at both lengths.",
        "",
        "## C0 fixed-backend contiguity control",
        "",
    ]
    if c0["complete"]:
        lines.extend(
            [
                f"- Window: {c0['window_index']}",
                f"- Reference: mean_nll={c0['reference']['mean_nll']:.9f}, ppl={c0['reference']['ppl']:.6f}",
                f"- Contiguous 2,560: mean_nll={c0['long']['mean_nll']:.9f}, ppl={c0['long']['ppl']:.6f}",
                f"- Delta={c0['mean_nll_delta']:+.9f}; PPL ratio={c0['ppl_ratio']:.9f}x; classification={c0['classification']}",
            ]
        )
    else:
        lines.append("C0 and/or its fixed-backend reference are NOT_MEASURED; numbers are n/a.")
    lines.extend(
        [
            "",
            "## FG1 P0-vs-P1..P4 table",
            "",
            "E1.3 registered C0 plus 16 P1-P4 cells; E1.4 adds four P0 cells, one per registered window. C0 is reported separately above.",
            "",
            analysis["teacher_table_markdown"],
            "",
            "Construction means:",
            "",
        ]
    )
    for construction in CONSTRUCTIONS:
        value = analysis["construction_mean_gaps"][construction]
        lines.append(
            f"- {construction.upper()}: mean gap="
            + ("n/a" if value is None else f"{value:+.9f}")
        )
    lines.extend(
        [
            "",
            analysis["selection_sentence"],
            "",
            "## FG2 scripts and CPU checks",
            "",
            f"- Resweep: `{RESWEEP_SCRIPT}`",
            f"- Chain: `{CHAIN_SCRIPT}`",
            f"- Master: `{MASTER_SCRIPT}`",
            f"- CPU self-test: `{analysis['fg2']['self_test_path']}`",
            "- Historical E1/E1.1/E1.3 behavior remains the default; `e14-apa` is explicit and writes only `_e14` paths.",
            "",
            "## Engine successor note",
            "",
            f"Open defect receipt: `{OPEN_DEFECT_NOTE}`. The standard full-attention mask/sink backend remains unfixed in this order.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in analysis["limitations"])
    lines.append("")
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> int:
    ensure_output(args.output_dir)
    ensure_model(args.model_dir)
    prefix_manifest, _arrays = load_prepared()

    fg0_rows: list[dict[str, Any]] = []
    receipt_inventory: dict[str, Any] = {}
    for length in FG0_LENGTHS:
        long_path = sanity_receipt_path(length, "long")
        reference_path = sanity_receipt_path(length, "reference")
        long_receipt = completed_receipt(long_path)
        reference_receipt = completed_receipt(reference_path)
        row: dict[str, Any] = {"length": length, "complete": False}
        for label, path, receipt in (
            ("long", long_path, long_receipt),
            ("reference", reference_path, reference_receipt),
        ):
            receipt_inventory[f"fg0_{length}_{label}"] = {
                "path": str(path),
                "state": "complete" if receipt is not None else (
                    read_json(path).get("status") if path.is_file() else "missing"
                ),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        if long_receipt is not None and reference_receipt is not None:
            long = score_summary(long_receipt)
            reference = score_summary(reference_receipt)
            if long["target_ids_sha256"] != reference["target_ids_sha256"]:
                raise RuntimeError(f"FG0 {length} long/reference target mismatch")
            delta = long["mean_nll"] - reference["mean_nll"]
            row.update(
                {
                    "complete": True,
                    "long": long,
                    "reference": reference,
                    "mean_nll_delta": delta,
                    "passes": bool(delta <= FG0_MAX_MEAN_NLL_DELTA),
                }
            )
        fg0_rows.append(row)
    fg0_complete = all(row["complete"] for row in fg0_rows)
    fg0_pass = fg0_complete and all(row["passes"] for row in fg0_rows)
    if not fg0_complete:
        fg0_verdict = "NOT_MEASURED"
        fg0_line = (
            f"FG0 NOT_MEASURED — {sum(row['complete'] for row in fg0_rows)}/2 "
            "long/reference length pairs are complete; deltas are n/a where absent."
        )
    elif fg0_pass:
        fg0_verdict = "PASS"
        fg0_line = (
            "FG0 PASS — fixed-backend long/reference mean-NLL deltas: "
            + ", ".join(
                f"{row['length']}={row['mean_nll_delta']:+.9f} nats"
                for row in fg0_rows
            )
            + f"; both <= {FG0_MAX_MEAN_NLL_DELTA:.2f}."
        )
    else:
        fg0_verdict = "FAIL"
        fg0_line = (
            "FG0 FAIL — fixed-backend long/reference mean-NLL deltas: "
            + ", ".join(
                f"{row['length']}={row['mean_nll_delta']:+.9f} nats"
                for row in fg0_rows
            )
            + f"; at least one exceeds {FG0_MAX_MEAN_NLL_DELTA:.2f}; STOP."
        )

    reference_receipts: dict[int, dict[str, Any] | None] = {}
    for window in TEACHER_WINDOWS:
        path = teacher_receipt_path("reference", window_index=window)
        receipt = completed_receipt(path)
        reference_receipts[window] = receipt
        receipt_inventory[f"teacher_reference_{window}"] = {
            "path": str(path),
            "state": "complete" if receipt is not None else (
                read_json(path).get("status") if path.is_file() else "missing"
            ),
            "sha256": sha256_file(path) if path.is_file() else None,
        }

    c0_path = teacher_receipt_path("c0")
    c0_receipt = completed_receipt(c0_path)
    receipt_inventory["c0"] = {
        "path": str(c0_path),
        "state": "complete" if c0_receipt is not None else (
            read_json(c0_path).get("status") if c0_path.is_file() else "missing"
        ),
        "sha256": sha256_file(c0_path) if c0_path.is_file() else None,
    }
    c0: dict[str, Any] = {
        "complete": False,
        "window_index": int(prefix_manifest["c0"]["window_index"]),
    }
    c0_sane = False
    c0_reference = reference_receipts[c0["window_index"]]
    if c0_receipt is not None and c0_reference is not None:
        long = score_summary(c0_receipt)
        reference = score_summary(c0_reference)
        if long["target_ids_sha256"] != reference["target_ids_sha256"]:
            raise RuntimeError("E1.4 C0/reference target mismatch")
        ppl_ratio = long["ppl"] / reference["ppl"]
        c0_sane = bool(ppl_ratio <= C0_EXPLOSION_PPL_RATIO)
        c0.update(
            {
                "complete": True,
                "long": long,
                "reference": reference,
                "mean_nll_delta": long["mean_nll"] - reference["mean_nll"],
                "ppl_ratio": ppl_ratio,
                "classification": (
                    "SANE" if c0_sane else "EXPLOSION — fixed-backend teacher sweep STOP"
                ),
            }
        )

    rows: list[dict[str, Any]] = []
    complete_cells = 0
    for construction in CONSTRUCTIONS:
        for window in TEACHER_WINDOWS:
            path = teacher_receipt_path(
                "sweep", construction=construction, window_index=window
            )
            teacher_receipt = completed_receipt(path)
            receipt_inventory[f"teacher_{construction}_{window}"] = {
                "path": str(path),
                "state": "complete" if teacher_receipt is not None else (
                    read_json(path).get("status") if path.is_file() else "missing"
                ),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
            row: dict[str, Any] = {
                "construction": construction,
                "window_index": window,
                "status": "not_measured",
            }
            reference_receipt = reference_receipts[window]
            if teacher_receipt is not None and reference_receipt is not None:
                teacher = score_summary(teacher_receipt)
                base = score_summary(reference_receipt)
                if teacher["target_ids_sha256"] != base["target_ids_sha256"]:
                    raise RuntimeError(
                        f"E1.4 {construction}/window {window} target mismatch"
                    )
                row.update(
                    {
                        "status": "complete",
                        "mean_nll_base": base["mean_nll"],
                        "mean_nll_teacher": teacher["mean_nll"],
                        "gap": base["mean_nll"] - teacher["mean_nll"],
                        "ppl_base": base["ppl"],
                        "ppl_teacher": teacher["ppl"],
                        "target_ids_sha256": base["target_ids_sha256"],
                        "base_receipt": base["receipt"],
                        "base_receipt_sha256": base["receipt_sha256"],
                        "teacher_receipt": teacher["receipt"],
                        "teacher_receipt_sha256": teacher["receipt_sha256"],
                    }
                )
                complete_cells += 1
            rows.append(row)
    sweep_complete = complete_cells == len(CONSTRUCTIONS) * len(TEACHER_WINDOWS)
    construction_means: dict[str, float | None] = {}
    for construction in CONSTRUCTIONS:
        construction_rows = [
            row
            for row in rows
            if row["construction"] == construction and row["status"] == "complete"
        ]
        construction_means[construction] = (
            float(np.mean([row["gap"] for row in construction_rows]))
            if len(construction_rows) == len(TEACHER_WINDOWS)
            else None
        )
    winner, best_gap = choose_teacher(construction_means)
    selection_ready = bool(fg0_pass and c0["complete"] and c0_sane and sweep_complete)
    selection_sentence = selection_sentence_for(
        winner, best_gap, complete=selection_ready
    )

    if selection_ready:
        selection_payload = {
            "schema": "moe_e1_4_teacher_selection_v1",
            "created_at": now_iso(),
            "status": (
                "selected_positive_gap"
                if winner is not None
                else "premise_stands_no_positive_construction"
            ),
            "order": str(ORDER_PATH),
            "backend": e1.BACKEND_E14_APA,
            "attention_registration": e1.backend_receipt_contract(
                e1.BACKEND_E14_APA
            ),
            "fg0_verdict": fg0_verdict,
            "fg0_rows": fg0_rows,
            "c0": c0,
            "seed": e1.DEFAULT_SEED,
            "registered_windows_in_order": list(TEACHER_WINDOWS),
            "construction_mean_gaps": construction_means,
            "winning_construction": winner,
            "winning_mean_gap": best_gap if winner is not None else None,
            "selection_sentence": selection_sentence,
            "winning_recipe": (
                prefix_manifest["recipes"][winner] if winner is not None else None
            ),
            "prefix_manifest": str(PREFIX_MANIFEST),
            "prefix_manifest_sha256": sha256_file(PREFIX_MANIFEST),
            "prefix_arrays": str(PREFIX_ARRAYS),
            "prefix_arrays_sha256": sha256_file(PREFIX_ARRAYS),
            "score_receipts": [
                {
                    "construction": row["construction"],
                    "window_index": row["window_index"],
                    "teacher_receipt": row["teacher_receipt"],
                    "teacher_receipt_sha256": row["teacher_receipt_sha256"],
                    "base_receipt": row["base_receipt"],
                    "base_receipt_sha256": row["base_receipt_sha256"],
                    "gap": row["gap"],
                }
                for row in rows
            ],
        }
        if SELECTION.is_file():
            prior = read_json(SELECTION)
            comparable = (
                "status",
                "backend",
                "fg0_verdict",
                "construction_mean_gaps",
                "winning_construction",
                "winning_mean_gap",
                "selection_sentence",
            )
            if any(prior.get(key) != selection_payload.get(key) for key in comparable):
                raise RuntimeError("append-only E1.4 selection conflicts with live receipts")
        else:
            write_new_json(SELECTION, selection_payload)

    if not fg0_pass:
        fg1_verdict = "NOT_MEASURED"
        fg1_line = (
            f"FG1 NOT_MEASURED — {complete_cells}/20 P0-P4 cells complete; "
            "FG0 must PASS before adjudication."
        )
    elif not c0["complete"]:
        fg1_verdict = "NOT_MEASURED"
        fg1_line = "FG1 NOT_MEASURED — fixed-backend C0/reference is incomplete."
    elif not c0_sane:
        fg1_verdict = "FAIL"
        fg1_line = (
            f"FG1 FAIL — C0 PPL ratio={c0['ppl_ratio']:.9f}x exceeds "
            f"{C0_EXPLOSION_PPL_RATIO:g}x; teacher sweep STOP."
        )
    elif not sweep_complete:
        fg1_verdict = "NOT_MEASURED"
        fg1_line = f"FG1 NOT_MEASURED — {complete_cells}/20 P0-P4 cells complete."
    else:
        fg1_verdict = "PASS"
        fg1_line = f"FG1 PASS — 20/20 P0-P4 cells complete; {selection_sentence}"

    script_hash = sha256_file(SCRIPT_PATH)
    self_test_path = (
        OUTPUT / "self_test_e14" / f"cpu_self_test_e14_{script_hash[:16]}.json"
    )
    self_test_payload = read_json(self_test_path) if self_test_path.is_file() else None
    scripts_match = all(
        path.is_file() and path.read_text(encoding="utf-8") == expected
        for path, expected in (
            (RESWEEP_SCRIPT, resweep_commands()),
            (CHAIN_SCRIPT, chain_commands()),
            (MASTER_SCRIPT, master_commands()),
        )
    )
    historical_default = (
        e1.BACKEND_HISTORICAL == "historical-standard"
        and e1.TEACHER_RULE_ORIGINAL == "original"
        and e1.pair_root(e1.DEFAULT_OUTPUT_DIR, e1.ADDRESS_RULE_E11)
        == e1.DEFAULT_OUTPUT_DIR / "pairs_e11"
    )
    fg2_pass = bool(
        self_test_payload is not None
        and self_test_payload.get("status") == "pass"
        and self_test_payload.get("script_sha256") == script_hash
        and scripts_match
        and historical_default
    )
    if fg2_pass:
        fg2_verdict = "PASS"
        fg2_line = (
            "FG2 PASS — resweep/chain/master scripts emitted; CPU self-tests pass; "
            "historical standard/original paths remain the defaults."
        )
    else:
        fg2_verdict = "NOT_MEASURED" if self_test_payload is None else "FAIL"
        fg2_line = (
            f"FG2 {fg2_verdict} — script/self-test/default checks are not all passing."
        )

    snapshot_material = {
        "script_sha256": script_hash,
        "expert_script_sha256": sha256_file(e1.SCRIPT_PATH),
        "prefix_manifest_sha256": sha256_file(PREFIX_MANIFEST),
        "receipt_inventory": receipt_inventory,
        "selection_sha256": sha256_file(SELECTION) if SELECTION.is_file() else None,
        "self_test_sha256": sha256_file(self_test_path) if self_test_path.is_file() else None,
    }
    snapshot_id = sha256_bytes(
        json.dumps(snapshot_material, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )[:16]
    limitations = [
        "Behavioral evidence is bounded to frozen GPT-OSS-20B and the sealed NarrativeForge/WikiText inputs.",
        "CPU self-tests validate data, backend registration, identity machinery, selection, and shell discipline only.",
        "The standard full-attention backend remains an open Project-Tensor-adjacent defect; E1.4 does not repair it.",
    ]
    if not fg0_complete:
        limitations.append("FG0 GPU measurements remain incomplete in this sandbox.")
    if not sweep_complete:
        limitations.append(f"Only {complete_cells}/20 fixed-backend teacher cells are complete.")
    analysis = {
        "schema": "moe_e1_4_analysis_snapshot_v1",
        "snapshot_id": snapshot_id,
        "created_at": now_iso(),
        "status": (
            "fg0_fail_stop"
            if fg0_complete and not fg0_pass
            else (
                "complete"
                if selection_ready and fg2_pass
                else "incomplete_gpu" if not selection_ready else "cpu_gate_fail"
            )
        ),
        "order": str(ORDER_PATH),
        "evidence_class": EVIDENCE_CLASS,
        "backend": e1.BACKEND_E14_APA,
        "attention_registration": e1.backend_receipt_contract(e1.BACKEND_E14_APA),
        "gates": {
            "FG0": {"verdict": fg0_verdict, "line": fg0_line},
            "FG1": {"verdict": fg1_verdict, "line": fg1_line},
            "FG2": {"verdict": fg2_verdict, "line": fg2_line},
        },
        "fg0": {"complete": fg0_complete, "pass": fg0_pass, "rows": fg0_rows},
        "fg0_table_markdown": fg0_table(fg0_rows),
        "c0": c0,
        "teacher_rows": rows,
        "teacher_table_markdown": teacher_table(rows),
        "construction_mean_gaps": construction_means,
        "selection_ready": selection_ready,
        "winner": winner if selection_ready else None,
        "winning_mean_gap": best_gap if selection_ready and winner else None,
        "selection_sentence": selection_sentence,
        "selection_path": str(SELECTION) if SELECTION.is_file() else None,
        "fg2": {
            "self_test_path": str(self_test_path),
            "self_test_status": (
                None if self_test_payload is None else self_test_payload.get("status")
            ),
            "scripts_match_emitted_content": scripts_match,
            "historical_default_preserved": historical_default,
            "resweep_script": str(RESWEEP_SCRIPT),
            "chain_script": str(CHAIN_SCRIPT),
            "master_script": str(MASTER_SCRIPT),
        },
        "receipt_inventory": receipt_inventory,
        "open_defect_note": str(OPEN_DEFECT_NOTE),
        "limitations": limitations,
    }
    snapshot_path = OUTPUT / "analysis_e14" / f"analysis_e14_{snapshot_id}.json"
    if snapshot_path.is_file():
        prior = read_json(snapshot_path)
        if prior.get("snapshot_id") != snapshot_id:
            raise RuntimeError(f"E1.4 analysis snapshot collision: {snapshot_path}")
    else:
        write_new_json(snapshot_path, analysis)
    marker = f"<!-- E14_SNAPSHOT:{snapshot_id} -->"
    append_text_once(REPORT, marker, render_report(analysis))
    print(
        json.dumps(
            {
                "status": analysis["status"],
                "snapshot": str(snapshot_path),
                "report": str(REPORT),
                "gates": {
                    name: gate["verdict"] for name, gate in analysis["gates"].items()
                },
                "fg0_deltas": {
                    str(row["length"]): (
                        row.get("mean_nll_delta") if row["complete"] else None
                    )
                    for row in fg0_rows
                },
                "complete_teacher_cells": complete_cells,
                "winner": analysis["winner"],
                "selection_sentence": selection_sentence,
                "chain_script": str(CHAIN_SCRIPT),
            }
        ),
        flush=True,
    )
    if args.require_fg0:
        return 0 if fg0_pass else (3 if fg0_complete else 4)
    if args.require_c0:
        if not fg0_pass:
            return 3 if fg0_complete else 4
        if not c0["complete"]:
            return 4
        return 0 if c0_sane else 3
    if args.require_complete:
        if not fg0_pass:
            return 3 if fg0_complete else 4
        if not c0["complete"] or not sweep_complete:
            return 4
        if not c0_sane or winner is None or not fg2_pass:
            return 3
        return 0
    return 0


def main() -> int:
    args = parse_args()
    try:
        selector_values = (
            args.length,
            args.view,
            args.score_kind,
            args.window_index,
            args.construction,
        )
        if args.mode == "sanity":
            if any(
                value is not None
                for value in (args.score_kind, args.window_index, args.construction)
            ):
                raise ValueError("teacher selectors are invalid for sanity")
        elif args.mode == "teacher-score":
            if args.length is not None or args.view is not None:
                raise ValueError("FG0 length/view selectors are invalid for teacher-score")
        elif any(value is not None for value in selector_values):
            raise ValueError("GPU score selectors are valid only for score modes")
        if args.mode != "analyze" and (
            args.require_fg0 or args.require_c0 or args.require_complete
        ):
            raise ValueError("--require-* selectors are valid only for analyze")
        if args.mode in {"prepare", "self-test", "analyze"} and (
            args.backend != e1.BACKEND_HISTORICAL
        ):
            raise ValueError(
                "CPU modes use the historical parser default; e14-apa is required "
                "only on explicit GPU score invocations"
            )
        if args.mode == "prepare":
            return prepare(args)
        if args.mode == "self-test":
            return self_test(args)
        if args.mode == "sanity":
            return sanity(args)
        if args.mode == "teacher-score":
            return teacher_score(args)
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
