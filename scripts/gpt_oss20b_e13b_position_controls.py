#!/usr/bin/env python3
"""MOE-E1.3b positional controls and long-context-path audit.

This runner is deliberately separate from the registered E1/E1.1/E1.2/E1.3
modes.  It never changes their defaults and writes only append-only E1.3b
artifacts under ``artifacts/moe_e1_3b``.

GPU scoring remains the frozen base model with standard sink-aware attention:
no teacher prefix, expert adapter, APA substitution, or eval fix is active.
The CPU-only modes prepare the sealed token controls, exercise synthetic
alignment/analysis machinery, and emit the A1 source audit.
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

import gpt_oss20b_expert_e1 as e1


ORDER_PATH = REPO_ROOT / "orders" / "MOE_E1_3B_POSITION_CONTROLS.md"
OUTPUT = REPO_ROOT / "artifacts" / "moe_e1_3b"
E1_OUTPUT = REPO_ROOT / "artifacts" / "moe_e1"
E1_BASE_W11 = E1_OUTPUT / "eval_e11" / "narrative_011.json"
E12_GUIDE256_W11 = (
    REPO_ROOT
    / "artifacts"
    / "moe_e1_diag"
    / "runs"
    / "window_011"
    / "e1_guide256.json"
)
RT1_MANIFEST = REPO_ROOT / "artifacts" / "moe_rt1" / "corpora_manifest.json"
RT1_WINDOWS = REPO_ROOT / "artifacts" / "moe_rt1" / "corpus_windows.npz"

CONTROL_MANIFEST = OUTPUT / "control_manifest.json"
CONTROL_ARRAYS = OUTPUT / "controls.npz"
GPU_SCRIPT = OUTPUT / "GPU_E13B_COMMANDS.sh"
REPORT = OUTPUT / "MOE_E1_3B_REPORT.md"

DIAG_PATH = SCRIPT_DIR / "gpt_oss20b_e1_diag.py"
E1_PATH = SCRIPT_DIR / "gpt_oss20b_expert_e1.py"
CONTEXT_LADDER_PATH = SCRIPT_DIR / "gpt_oss20b_context_ladder.py"
STREAM_PATH = SCRIPT_DIR / "gpt_oss20b_stream_forward_smoke.py"
CORE_PATH = REPO_ROOT / "core" / "gpt_oss20b_tc.py"
TC_INIT_PATH = Path(
    "/mnt/ForgeRealm/Project-Tensor/tensor_cuda/tensor_cuda/__init__.py"
)
TC_KERNEL_PATH = Path(
    "/mnt/ForgeRealm/Project-Tensor/tensor_cuda/src/kernels.cu"
)
H4_64K_PARENT = (
    REPO_ROOT / "artifacts" / "gpt_oss_20b" / "h4_context_ladder_apa_64k_sampled.json"
)
H4_64K_RUN = (
    REPO_ROOT
    / "artifacts"
    / "gpt_oss_20b"
    / "h4_context_ladder_apa_64k_sampled"
    / "apa_r0.15_65536.json"
)
H5_96K_PARENT = (
    REPO_ROOT
    / "artifacts"
    / "gpt_oss_20b"
    / "h5_bulk_graft_96k_candidate_gate_rerun.json"
)
H5_96K_CAPTURE = (
    REPO_ROOT
    / "artifacts"
    / "gpt_oss_20b"
    / "h5_bulk_graft_96k_candidate_gate_rerun"
    / "capture_forward.json"
)

RAMP_LENGTHS = (1024, 1536, 2048, 2304, 2560)
WIKITEXT_STREAM_TOKENS = 2560
SHORT_WINDOW_TOKENS = 512
N_TARGETS = 511
C2_LENGTH = 768
C4_WIKITEXT_PREFIX = 256
C4_WINDOW_INDEX = 11

GPU_WRAPPER = (
    "flock -w 7200 /tmp/forge-gpu.lock "
    "timeout --signal=TERM --kill-after=5s 590s "
    "env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 "
    "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
)

# These are descriptive synthesis rails, not replacements for the registered
# BG gates.  They make the post-run mechanism sentence reproducible while the
# report always prints the raw BG1-BG3 numbers beside the interpretation.
POSITION_HOT_MEAN_NLL_RATIO = 1.25
SUB2048_SPLICE_PPL_RATIO = 1.50
C2_REFERENCE_MEDIAN_MEAN_NLL_RATIO = 1.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run registered MOE-E1.3b position controls."
    )
    parser.add_argument(
        "mode", choices=("prepare", "self-test", "audit", "score", "analyze")
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--model-dir", type=Path, default=e1.SNAPSHOT)
    parser.add_argument("--case", dest="case_name", choices=("ramp", "c2", "c4"))
    parser.add_argument("--length", type=int, choices=RAMP_LENGTHS)
    parser.add_argument("--view", choices=("long", "reference"))
    parser.add_argument("--attempt", type=int, default=0)
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


def write_once_text(path: Path, text: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(
                f"append-only artifact differs from requested content: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def append_text_once(path: Path, marker: str, text: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and marker in path.read_text(encoding="utf-8"):
        return False
    with path.open("a", encoding="utf-8") as handle:
        if path.stat().st_size:
            handle.write("\n")
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")
    return True


def ensure_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != OUTPUT.resolve():
        raise ValueError(f"E1.3b artifacts are restricted to {OUTPUT.resolve()}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_model(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != e1.SNAPSHOT.resolve():
        raise ValueError(f"E1.3b is sealed to {e1.SNAPSHOT.resolve()}")
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return resolved


def registered_runs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for length in RAMP_LENGTHS:
        gates = ["BG2/C3"]
        if length == 2560:
            gates.append("BG1/C1-R1")
        for view in ("long", "reference"):
            rows.append(
                {
                    "run_id": f"ramp_{length}_{view}",
                    "case": "ramp",
                    "length": int(length),
                    "view": view,
                    "gates": gates,
                }
            )
    rows.extend(
        [
            {
                "run_id": "c2_wikitext_768",
                "case": "c2",
                "length": C2_LENGTH,
                "view": "long",
                "gates": ["BG1/C2"],
            },
            {
                "run_id": "c4_w11_wikitext256",
                "case": "c4",
                "length": C2_LENGTH,
                "view": "long",
                "gates": ["BG3/C4"],
            },
        ]
    )
    if len(rows) != 12 or len({row["run_id"] for row in rows}) != 12:
        raise AssertionError("registered E1.3b run registry is not exactly 12 units")
    return rows


def gpu_commands() -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

repo_e13b={str(REPO_ROOT)!r}
runner_e13b={str(SCRIPT_PATH)!r}
cd "$repo_e13b"

run_gpu_e13b() {{
  local rc_e13b=0
  flock -w 7200 /tmp/forge-gpu.lock \\
    timeout --signal=TERM --kill-after=5s 590s \\
    env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \\
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
      python3 "$runner_e13b" "$@" || rc_e13b=$?
  sleep 30
  return "$rc_e13b"
}}

# CPU-only sealed preparation, synthetic checks, and A1 audit.
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \\
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
  python3 "$runner_e13b" prepare
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \\
  python3 "$runner_e13b" self-test
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \\
  python3 "$runner_e13b" audit

# C2: all-sub-2048 contiguous WikiText sanity anchor.
run_gpu_e13b score --case c2

# C3: five long/reference pairs.  The 2,560 pair is also C1/R1.
for length_e13b in 1024 1536 2048 2304 2560; do
  run_gpu_e13b score --case ramp --length "$length_e13b" --view long
  run_gpu_e13b score --case ramp --length "$length_e13b" --view reference
done

# C4: WikiText-256 prefix plus the sealed w11 guide window.
run_gpu_e13b score --case c4

# Append the complete BG snapshot; nonzero means a receipt is absent/error.
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \\
  python3 "$runner_e13b" analyze --require-complete
"""


def _write_control_npz(wikitext: np.ndarray, guide_w11: np.ndarray) -> None:
    if CONTROL_ARRAYS.exists():
        raise FileExistsError(f"append-only control payload exists: {CONTROL_ARRAYS}")
    temporary = CONTROL_ARRAYS.with_name(CONTROL_ARRAYS.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        wikitext_0_2560=np.ascontiguousarray(wikitext, dtype=np.int64),
        guide_w11=np.ascontiguousarray(guide_w11, dtype=np.int64),
    )
    temporary.replace(CONTROL_ARRAYS)


def validate_prepared() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not CONTROL_MANIFEST.is_file() or not CONTROL_ARRAYS.is_file():
        raise FileNotFoundError("run E1.3b prepare before scoring")
    manifest = read_json(CONTROL_MANIFEST)
    if manifest.get("schema") != "moe_e1_3b_controls_v1":
        raise RuntimeError("unexpected E1.3b control manifest schema")
    if manifest.get("control_arrays_sha256") != sha256_file(CONTROL_ARRAYS):
        raise RuntimeError("E1.3b control payload hash mismatch")
    with np.load(CONTROL_ARRAYS, allow_pickle=False) as stored:
        arrays = {
            name: np.ascontiguousarray(stored[name], dtype=np.int64)
            for name in stored.files
        }
    wiki = arrays.get("wikitext_0_2560")
    guide = arrays.get("guide_w11")
    if wiki is None or wiki.shape != (WIKITEXT_STREAM_TOKENS,):
        raise RuntimeError(f"bad prepared WikiText shape: {None if wiki is None else wiki.shape}")
    if guide is None or guide.shape != (SHORT_WINDOW_TOKENS,):
        raise RuntimeError(f"bad prepared guide shape: {None if guide is None else guide.shape}")
    if manifest["wikitext"]["token_ids_sha256"] != sha256_array(wiki):
        raise RuntimeError("prepared WikiText token hash mismatch")
    if manifest["guide_w11"]["token_ids_sha256"] != sha256_array(guide):
        raise RuntimeError("prepared guide token hash mismatch")
    if len(manifest.get("registered_runs", [])) != 12:
        raise RuntimeError("prepared run registry is not 12 units")
    return manifest, arrays


def prepare(args: argparse.Namespace) -> int:
    ensure_output(args.output_dir)
    model_dir = ensure_model(args.model_dir)
    write_once_text(GPU_SCRIPT, gpu_commands())
    GPU_SCRIPT.chmod(0o755)

    if CONTROL_MANIFEST.is_file() or CONTROL_ARRAYS.is_file():
        manifest, _arrays = validate_prepared()
        print(
            json.dumps(
                {
                    "status": "already_complete",
                    "manifest": str(CONTROL_MANIFEST),
                    "arrays": str(CONTROL_ARRAYS),
                    "gpu_script": str(GPU_SCRIPT),
                    "registered_runs": len(manifest["registered_runs"]),
                }
            ),
            flush=True,
        )
        return 0

    for required in (RT1_MANIFEST, RT1_WINDOWS, E1_BASE_W11, E12_GUIDE256_W11):
        if not required.is_file():
            raise FileNotFoundError(required)
    rt1_manifest = read_json(RT1_MANIFEST)
    if rt1_manifest.get("windows_file_sha256") != sha256_file(RT1_WINDOWS):
        raise RuntimeError("frozen RT1 window payload hash mismatch")
    generic = rt1_manifest["corpora"]["generic"]
    allowed = Path("/home/vader/.cache/huggingface/hub/datasets--wikitext").resolve()
    for source in generic["sources"]:
        if not Path(source["path"]).resolve().is_relative_to(allowed):
            raise RuntimeError("frozen RT1 generic source escapes WikiText cache")

    import gpt_oss20b_router_telemetry as rt1
    from transformers import AutoTokenizer

    text, reconstructed_sources = rt1.build_generic_corpus()
    canonical = text.encode("utf-8")
    if sha256_bytes(canonical) != generic["corpus_sha256"]:
        raise RuntimeError("reconstructed WikiText corpus differs from frozen RT1")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    all_ids = rt1.tokenizer_ids(tokenizer, text)
    if int(all_ids.size) != int(generic["corpus_token_count"]):
        raise RuntimeError("reconstructed WikiText token count differs from frozen RT1")
    if all_ids.size < WIKITEXT_STREAM_TOKENS:
        raise RuntimeError("frozen WikiText corpus is too short for E1.3b")
    wikitext = np.ascontiguousarray(
        all_ids[:WIKITEXT_STREAM_TOKENS], dtype=np.int64
    )
    with np.load(RT1_WINDOWS, allow_pickle=False) as stored:
        generic_windows = np.ascontiguousarray(stored["generic"], dtype=np.int64)
    if generic_windows.shape != (16, SHORT_WINDOW_TOKENS):
        raise RuntimeError(f"unexpected frozen generic window shape {generic_windows.shape}")
    if not np.array_equal(wikitext[:SHORT_WINDOW_TOKENS], generic_windows[0]):
        raise RuntimeError("WikiText stream prefix differs from frozen RT1 window 0")

    e1_manifest, e1_arrays = e1.load_prepared(E1_OUTPUT)
    guide_w11 = np.ascontiguousarray(
        e1_arrays["behavioral_ids"][C4_WINDOW_INDEX], dtype=np.int64
    )
    base_w11 = read_json(E1_BASE_W11)
    guide256_w11 = read_json(E12_GUIDE256_W11)
    guide_targets = np.ascontiguousarray(guide_w11[1:], dtype=np.int64)
    target_hash = sha256_array(guide_targets)
    if base_w11.get("status") != "complete" or guide256_w11.get("status") != "complete":
        raise RuntimeError("sealed w11 comparison receipts are incomplete")
    if base_w11["arms"]["base"]["target_ids_sha256"] != target_hash:
        raise RuntimeError("w11 guide target hash differs from E1.1 base receipt")
    if guide256_w11["arms"]["teacher"]["target_ids_sha256"] != target_hash:
        raise RuntimeError("w11 guide target hash differs from E1.2 guide256 receipt")

    _write_control_npz(wikitext, guide_w11)
    manifest: dict[str, Any] = {
        "schema": "moe_e1_3b_controls_v1",
        "created_at": now_iso(),
        "status": "complete",
        "order": str(ORDER_PATH),
        "order_sha256": sha256_file(ORDER_PATH),
        "model_dir": str(model_dir),
        "model_revision": model_dir.name,
        "tokenizer_json_sha256": sha256_file(model_dir / "tokenizer.json"),
        "control_arrays": str(CONTROL_ARRAYS),
        "control_arrays_sha256": sha256_file(CONTROL_ARRAYS),
        "wikitext": {
            "recipe": (
                "tokens 0..2559 of the canonical frozen RT1 WikiText generic "
                "corpus stream; test parquet then validation parquet; plain text "
                "joined exactly as RT1; tokenizer adds no special tokens"
            ),
            "token_start": 0,
            "token_stop_exclusive": WIKITEXT_STREAM_TOKENS,
            "token_count": WIKITEXT_STREAM_TOKENS,
            "token_ids_sha256": sha256_array(wikitext),
            "canonical_corpus_sha256": sha256_bytes(canonical),
            "canonical_corpus_token_count": int(all_ids.size),
            "rt1_manifest": str(RT1_MANIFEST),
            "rt1_manifest_sha256": sha256_file(RT1_MANIFEST),
            "rt1_windows": str(RT1_WINDOWS),
            "rt1_windows_sha256": sha256_file(RT1_WINDOWS),
            "rt1_frozen_sources": generic["sources"],
            "reconstructed_sources": reconstructed_sources,
            "first_512_crosscheck_sha256": sha256_array(generic_windows[0]),
        },
        "guide_w11": {
            "recipe": "sealed E1 behavioral heldout window 11, exactly 512 tokens",
            "window_index": C4_WINDOW_INDEX,
            "token_count": SHORT_WINDOW_TOKENS,
            "token_ids_sha256": sha256_array(guide_w11),
            "target_ids_sha256": target_hash,
            "source_window": e1_manifest["windows"]["behavioral_heldout"][C4_WINDOW_INDEX],
            "e1_prepared_windows": e1_manifest["prepared_windows"],
            "base_reference": str(E1_BASE_W11),
            "base_reference_sha256": sha256_file(E1_BASE_W11),
            "guide256_reference": str(E12_GUIDE256_W11),
            "guide256_reference_sha256": sha256_file(E12_GUIDE256_W11),
        },
        "registered_controls": {
            "c1": (
                "WikiText tokens 0..2559; last 511 targets; R1 is the exact "
                "same targets in tokens 2048..2559 alone"
            ),
            "c2": "WikiText tokens 0..767; score last 511 targets",
            "c3": {
                "lengths": list(RAMP_LENGTHS),
                "each": "long sequence and exact same-target 512-token reference",
            },
            "c4": (
                "WikiText tokens 0..255 plus sealed guide window 11; score "
                "guide targets 1..511"
            ),
        },
        "scored_target_tokens_per_run": N_TARGETS,
        "registered_runs": registered_runs(),
        "registered_run_count": 12,
        "base_arm_only": True,
        "teacher_enabled": False,
        "expert_enabled": False,
        "attention_mode": "standard",
        "eval_fix": "original",
        "gpu_script": str(GPU_SCRIPT),
        "gpu_script_sha256": sha256_file(GPU_SCRIPT),
    }
    write_new_json(CONTROL_MANIFEST, manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": str(CONTROL_MANIFEST),
                "arrays": str(CONTROL_ARRAYS),
                "gpu_script": str(GPU_SCRIPT),
                "registered_runs": 12,
            }
        ),
        flush=True,
    )
    return 0


def case_inputs(
    wikitext: np.ndarray,
    guide_w11: np.ndarray,
    *,
    case_name: str,
    length: int | None,
    view: str | None,
) -> dict[str, Any]:
    if case_name == "ramp":
        if length not in RAMP_LENGTHS or view not in {"long", "reference"}:
            raise ValueError("ramp scoring requires registered --length and --view")
        source_start = 0 if view == "long" else int(length) - SHORT_WINDOW_TOKENS
        source_stop = int(length)
        full_sequence = np.ascontiguousarray(
            wikitext[source_start:source_stop], dtype=np.int64
        )
        run_id = f"ramp_{length}_{view}"
        role = "C1" if length == 2560 and view == "long" else (
            "R1" if length == 2560 else "C3"
        )
        source = {
            "kind": "contiguous_wikitext",
            "token_start": source_start,
            "token_stop_exclusive": source_stop,
            "parent_stream_start": 0,
            "parent_stream_stop_exclusive": int(length),
        }
        target_positions = {
            "corpus_start": int(length) - N_TARGETS,
            "corpus_stop_exclusive": int(length),
        }
    elif case_name == "c2":
        if length is not None or view is not None:
            raise ValueError("C2 accepts no --length or --view")
        full_sequence = np.ascontiguousarray(wikitext[:C2_LENGTH], dtype=np.int64)
        run_id = "c2_wikitext_768"
        role = "C2"
        source = {
            "kind": "contiguous_wikitext",
            "token_start": 0,
            "token_stop_exclusive": C2_LENGTH,
        }
        target_positions = {
            "corpus_start": C2_LENGTH - N_TARGETS,
            "corpus_stop_exclusive": C2_LENGTH,
        }
    elif case_name == "c4":
        if length is not None or view is not None:
            raise ValueError("C4 accepts no --length or --view")
        full_sequence = np.concatenate(
            [wikitext[:C4_WIKITEXT_PREFIX], guide_w11]
        ).astype(np.int64, copy=False)
        run_id = "c4_w11_wikitext256"
        role = "C4"
        source = {
            "kind": "wikitext_guide_splice",
            "prefix": {
                "corpus": "wikitext",
                "token_start": 0,
                "token_stop_exclusive": C4_WIKITEXT_PREFIX,
            },
            "window": {"corpus": "sealed_guide_w11", "token_start": 0, "token_stop_exclusive": 512},
        }
        target_positions = {
            "guide_w11_start": 1,
            "guide_w11_stop_exclusive": SHORT_WINDOW_TOKENS,
        }
    else:
        raise ValueError("score requires --case ramp|c2|c4")

    if full_sequence.size < SHORT_WINDOW_TOKENS:
        raise RuntimeError("control sequence is shorter than the scoring window")
    inputs = np.ascontiguousarray(full_sequence[:-1][None, :], dtype=np.int64)
    targets = np.ascontiguousarray(full_sequence[-N_TARGETS:], dtype=np.int64)
    predictors = np.ascontiguousarray(
        full_sequence[-SHORT_WINDOW_TOKENS:-1], dtype=np.int64
    )
    if inputs.shape[1] != full_sequence.size - 1:
        raise AssertionError("input shift contract failed")
    if targets.shape != (N_TARGETS,) or predictors.shape != (N_TARGETS,):
        raise AssertionError("target/predictor shape contract failed")
    return {
        "run_id": run_id,
        "role": role,
        "case": case_name,
        "length": length,
        "view": view,
        "full_sequence": np.ascontiguousarray(full_sequence),
        "inputs": inputs,
        "targets": targets,
        "predictors": predictors,
        "source": source,
        "target_positions": target_positions,
    }


def receipt_path(run_id: str, attempt: int) -> Path:
    suffix = "" if attempt == 0 else f"_attempt{attempt:02d}"
    return OUTPUT / "runs" / f"{run_id}{suffix}.json"


def cuda_probe() -> dict[str, Any]:
    nodes: list[str] = []
    for path in sorted(Path("/dev").glob("nvidia*")):
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
        smi = {"error_type": type(exc).__name__, "error": str(exc)}
    return {
        "character_device_nodes": nodes,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_smi": smi,
    }


def require_cuda() -> dict[str, Any]:
    probe = cuda_probe()
    required = {"/dev/nvidia0", "/dev/nvidiactl"}
    if not required.issubset(set(probe["character_device_nodes"])):
        raise RuntimeError(
            "CUDA device nodes unavailable; run E1.3b score modes on the lead "
            "GPU under the registered flock/590s wrapper"
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


def run_base_forward(
    *,
    args: argparse.Namespace,
    data: dict[str, Any],
    receipt: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    """Run the unchanged E1 base attention/model path once per loaded block."""

    runtime = e1.load_runtime()
    tc = runtime["tc"]
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
            e1.configure_block(block)
            if block.self_attn.attention_mode != "standard":
                raise RuntimeError("E1.3b base arm unexpectedly left standard attention")
            if block.self_attn.inject_kv is not None or int(block.self_attn.graft_seats) != 0:
                raise RuntimeError("E1.3b base arm unexpectedly mounted a graft")
            hidden, kv, route = block(hidden, cos, sin)
            tc.synchronize()
            receipt["layers"].append(
                {
                    "layer": int(layer),
                    "layer_type": cfg.layer_types[layer],
                    "attention_backend": block.self_attn.last_attention_backend,
                    "sliding_window": block.self_attn.sliding_window,
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
        raise RuntimeError("scored target hash mismatch")
    return score


def score(args: argparse.Namespace) -> int:
    ensure_output(args.output_dir)
    ensure_model(args.model_dir)
    if args.case_name is None:
        raise ValueError("score requires --case ramp|c2|c4")
    if args.attempt < 0:
        raise ValueError("--attempt must be nonnegative")
    manifest, arrays = validate_prepared()
    data = case_inputs(
        arrays["wikitext_0_2560"],
        arrays["guide_w11"],
        case_name=args.case_name,
        length=args.length,
        view=args.view,
    )
    registered = {row["run_id"] for row in manifest["registered_runs"]}
    if data["run_id"] not in registered:
        raise RuntimeError(f"unregistered E1.3b run {data['run_id']}")
    path = receipt_path(data["run_id"], args.attempt)
    if path.is_file():
        prior = read_json(path)
        if prior.get("status") == "complete":
            print(json.dumps({"status": "already_complete", "receipt": str(path)}))
            return 0
        raise FileExistsError(
            f"append-only attempt exists at {path}; select a new --attempt"
        )
    probe = require_cuda()
    started = time.perf_counter()
    receipt: dict[str, Any] = {
        "schema": "moe_e1_3b_score_v1",
        "created_at": now_iso(),
        "status": "starting",
        "argv": sys.argv,
        "required_shell_wrapper": GPU_WRAPPER,
        "order": str(ORDER_PATH),
        "run_id": data["run_id"],
        "role": data["role"],
        "case": data["case"],
        "length": data["length"],
        "view": data["view"],
        "attempt": int(args.attempt),
        "model_dir": str(args.model_dir.resolve()),
        "compute_dtype": "bfloat16",
        "attention_mode": "standard",
        "expert_mode": "resident_packed_mxfp4",
        "eval_fix": "original",
        "base_arm_only": True,
        "teacher_enabled": False,
        "expert_adapter_enabled": False,
        "position_offset": 0,
        "graft_seats": 0,
        "execution_schedule": "one_base_sequence_call_per_loaded_block",
        "full_sequence_tokens": int(data["full_sequence"].size),
        "input_tokens": int(data["inputs"].shape[1]),
        "scored_target_tokens": int(data["targets"].size),
        "full_sequence_ids_sha256": sha256_array(data["full_sequence"]),
        "input_ids_sha256": sha256_array(data["inputs"]),
        "predictor_ids_sha256": sha256_array(data["predictors"]),
        "target_ids_sha256": sha256_array(data["targets"]),
        "source": data["source"],
        "target_positions": data["target_positions"],
        "control_manifest": str(CONTROL_MANIFEST),
        "control_manifest_sha256": sha256_file(CONTROL_MANIFEST),
        "control_arrays": str(CONTROL_ARRAYS),
        "control_arrays_sha256": sha256_file(CONTROL_ARRAYS),
        "cuda_environment": probe,
        "gpu_before": nvidia_smi(),
        "script_sha256_at_run": sha256_file(SCRIPT_PATH),
        "e1_script_sha256_at_run": sha256_file(E1_PATH),
        "core_script_sha256_at_run": sha256_file(CORE_PATH),
        "completed_layers": 0,
    }
    atomic_write_json(path, receipt)
    try:
        result = run_base_forward(args=args, data=data, receipt=receipt, path=path)
        receipt.update(
            {
                "status": "complete",
                "score": result,
                "attention_backend_counts": dict(
                    Counter(row["attention_backend"] for row in receipt["layers"])
                ),
                "wall_seconds": float(time.perf_counter() - started),
                "gpu_after": nvidia_smi(),
            }
        )
        atomic_write_json(path, receipt)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "receipt": str(path),
                    "run_id": data["run_id"],
                    "score": result,
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
        atomic_write_json(path, receipt)
        raise


def source_anchor(path: Path, needle: str, occurrence: int = 0) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    matches = [index + 1 for index, line in enumerate(lines) if needle in line]
    if len(matches) <= occurrence:
        raise RuntimeError(
            f"static-audit anchor missing in {path}: occurrence {occurrence} of {needle}"
        )
    return {
        "path": str(path),
        "line": int(matches[occurrence]),
        "needle": needle,
        "file_sha256": sha256_file(path),
    }


def anchor_ref(anchor: dict[str, Any]) -> str:
    return f"{anchor['path']}:{anchor['line']}"


def build_audit_payload() -> dict[str, Any]:
    required_paths = (
        DIAG_PATH,
        E1_PATH,
        CONTEXT_LADDER_PATH,
        STREAM_PATH,
        CORE_PATH,
        TC_INIT_PATH,
        TC_KERNEL_PATH,
        H4_64K_PARENT,
        H4_64K_RUN,
        H5_96K_PARENT,
        H5_96K_CAPTURE,
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    anchors = {
        "e1_force_standard": source_anchor(
            E1_PATH, 'block.self_attn.attention_mode = "standard"', occurrence=2
        ),
        "e13_diag_rope": source_anchor(
            DIAG_PATH, 'cos, sin = runtime["gpt_oss_yarn_rope_tables"]('
        ),
        "e13_diag_block_call": source_anchor(
            DIAG_PATH, "hidden, kv, route = block(hidden, cos, sin)"
        ),
        "ladder_attention_arg": source_anchor(
            CONTEXT_LADDER_PATH, 'setting["attention_mode"],'
        ),
        "ladder_apa_scope": source_anchor(
            CONTEXT_LADDER_PATH, 'if setting["attention_mode"] == "apa_selective":'
        ),
        "stream_mode_resolution": source_anchor(
            STREAM_PATH, "resolve_gpt_oss_attention_mode("
        ),
        "stream_rope_length": source_anchor(
            STREAM_PATH, "rope_len = int(ids.shape[1] + mounted_graft_tokens)"
        ),
        "stream_rope_build": source_anchor(
            STREAM_PATH, "cos, sin = gpt_oss_yarn_rope_tables(cfg, rope_len)"
        ),
        "stream_mode_assignment": source_anchor(
            STREAM_PATH, "block.self_attn.attention_mode = attention_audit["
        ),
        "stream_block_call": source_anchor(
            STREAM_PATH, "h, kv, route_info = block(h, cos, sin)"
        ),
        "core_mode_resolve_standard": source_anchor(
            CORE_PATH, 'if requested_mode == "standard":'
        ),
        "core_mode_resolve_full_apa": source_anchor(
            CORE_PATH, 'elif layer_type == "full_attention":'
        ),
        "core_yarn_builder": source_anchor(
            CORE_PATH, "def gpt_oss_yarn_rope_tables("
        ),
        "core_rope_position_slice": source_anchor(
            CORE_PATH, "cseg = cos.slice(0, position_offset + int(shift), L)"
        ),
        "core_mask_builder": source_anchor(
            CORE_PATH, "def _gpt_oss_attention_mask("
        ),
        "core_mask_causal": source_anchor(CORE_PATH, "allowed = k_abs <= q_abs"),
        "core_apa_key_quantization": source_anchor(
            CORE_PATH, "kq = _quantize_keys(k, R, CB, BND)"
        ),
        "core_full_apa_sink": source_anchor(
            CORE_PATH, "attn = tc.apa_selective_attention_sink("
        ),
        "core_standard_sliding_sink": source_anchor(
            CORE_PATH, "attn = sliding_sink_attention_tc("
        ),
        "core_standard_full_mask": source_anchor(
            CORE_PATH, "mask = _gpt_oss_attention_mask("
        ),
        "core_standard_full_sink": source_anchor(
            CORE_PATH, "attn = sink_attention_tc("
        ),
        "core_standard_sink_softmax": source_anchor(
            CORE_PATH, "combined = tc.cat([scores, sink_logits], dim=-1)"
        ),
        "tc_fused_sink_contract": source_anchor(
            TC_INIT_PATH, "def apa_selective_attention_sink("
        ),
        "tc_fused_sink_kernel": source_anchor(
            TC_KERNEL_PATH, "NDArray apa_selective_attention_sink("
        ),
        "h4_pass": source_anchor(H4_64K_PARENT, '"classification": "pass"'),
        "h4_backend": source_anchor(
            H4_64K_PARENT, '"apa_selective_sink_fused": 12'
        ),
        "h5_pass": source_anchor(H5_96K_PARENT, '"classification": "pass"'),
        "h5_apa_command": source_anchor(
            H5_96K_PARENT, '"apa_selective"', occurrence=0
        ),
        "h5_rope_length": source_anchor(
            H5_96K_CAPTURE, '"rope_table_length": 98304'
        ),
        "h5_standard_sliding_backend": source_anchor(
            H5_96K_CAPTURE, '"attention_backend": "standard_sink_sliding_chunked"'
        ),
        "h5_fused_full_backend": source_anchor(
            H5_96K_CAPTURE, '"attention_backend": "apa_selective_sink_fused"'
        ),
    }

    h4_parent = read_json(H4_64K_PARENT)
    h4_runs = [
        row
        for row in h4_parent.get("runs", [])
        if int(row.get("target_tokens", -1)) == 65536
    ]
    if len(h4_runs) != 1:
        raise RuntimeError("H4 64K receipt does not contain exactly one registered run")
    h4 = h4_runs[0]
    h4_run = read_json(H4_64K_RUN)
    h5_parent = read_json(H5_96K_PARENT)
    h5_capture = read_json(H5_96K_CAPTURE)
    h5_counts = dict(Counter(row["attention_backend"] for row in h5_capture["layers"]))
    if not (
        h4_parent.get("status") == "ok"
        and h4.get("classification") == "pass"
        and h4_run.get("status") == "ok"
        and h5_parent.get("status") == "ok"
        and h5_parent.get("classification") == "pass"
        and h5_capture.get("status") == "ok"
    ):
        raise RuntimeError("one or more proven long-context receipts are not passing")
    if h5_counts != {
        "standard_sink_sliding_chunked": 12,
        "apa_selective_sink_fused": 12,
    }:
        raise RuntimeError(f"unexpected 96K attention backend counts: {h5_counts}")

    result_verbatim = (
        "A1 STATIC AUDIT — RoPE/YARN table construction and executed position "
        f"handling do not diverge: E1.3 calls the shared builder at {anchor_ref(anchors['e13_diag_rope'])} "
        f"and the stream lineage calls it at {anchor_ref(anchors['stream_rope_build'])}; both enter the "
        f"same builder at {anchor_ref(anchors['core_yarn_builder'])}, call blocks without a position_offset, "
        f"and use the same core slice at {anchor_ref(anchors['core_rope_position_slice'])}. The generic stream "
        f"formula at {anchor_ref(anchors['stream_rope_length'])} can add mounted graft seats, but the passing "
        "H4 run and the unmounted 96K capture execute that term as zero. The decisive executed divergence is "
        f"full attention: E1 forces standard at {anchor_ref(anchors['e1_force_standard'])}, selecting the "
        f"explicit causal-mask builder at {anchor_ref(anchors['core_standard_full_mask'])} and standard sink "
        f"softmax at {anchor_ref(anchors['core_standard_full_sink'])}; the passing ladder/96K commands request "
        f"apa_selective through {anchor_ref(anchors['ladder_attention_arg'])} and "
        f"{anchor_ref(anchors['stream_mode_assignment'])}, selecting quantized selective keys at "
        f"{anchor_ref(anchors['core_apa_key_quantization'])} and fused causal sink attention at "
        f"{anchor_ref(anchors['core_full_apa_sink'])}, which does not construct the Python full-attention mask. "
        f"Sink semantics exist in both paths but their implementations diverge: standard concatenates the "
        f"learned sink into an explicit softmax at {anchor_ref(anchors['core_standard_sink_softmax'])}, whereas "
        f"the proven path folds it into the fused kernel whose contract begins at "
        f"{anchor_ref(anchors['tc_fused_sink_contract'])}. Sliding layers do not diverge: both use "
        f"standard_sink_sliding_chunked at {anchor_ref(anchors['core_standard_sliding_sink'])}, confirmed 12/12 "
        "in the 96K receipt. Therefore the 64K/96K passes do not validate E1's standard full-attention "
        "mask/sink backend. No static defect inside that backend was proven, so --eval-fix e13b is not "
        "implemented and all existing defaults remain unchanged."
    )
    return {
        "schema": "moe_e1_3b_a1_static_audit_v1",
        "created_at": now_iso(),
        "status": "complete",
        "order": str(ORDER_PATH),
        "result_verbatim": result_verbatim,
        "verdict": "ONE_EXECUTED_FULL_ATTENTION_DIVERGENCE; NO_STATIC_DEFECT_PROVEN",
        "anchors": anchors,
        "same": [
            {
                "topic": "RoPE/YARN table math",
                "e1_diag": "shared gpt_oss_yarn_rope_tables(cfg, input_length)",
                "proven_lineage": "same builder; input length plus zero mounted seats",
            },
            {
                "topic": "position and graft shift in compared executions",
                "e1_diag": "position_offset=0, shift=0, no mounted graft",
                "proven_lineage": "position_offset=0; H4 and 96K capture are unmounted",
            },
            {
                "topic": "sliding attention and sinks",
                "e1_diag": "standard_sink_sliding_chunked on 12 sliding layers",
                "proven_lineage": "same backend on 12 sliding layers",
            },
            {
                "topic": "block/model class and dtype",
                "e1_diag": "GptOssDiagnosticBlockTC, bfloat16, resident packed MXFP4",
                "proven_lineage": "same",
            },
        ],
        "divergences": [
            {
                "topic": "full-attention mode selection",
                "e1_diag": "forced standard for every layer",
                "proven_lineage": "apa_selective for full layers, standard for sliding",
                "anchors": ["e1_force_standard", "ladder_attention_arg", "stream_mode_assignment"],
            },
            {
                "topic": "full-attention causal mask",
                "e1_diag": "materialized _gpt_oss_attention_mask added to exact scores",
                "proven_lineage": "fused causal index bounds; no Python mask tensor",
                "anchors": ["core_mask_builder", "core_standard_full_mask", "core_full_apa_sink"],
            },
            {
                "topic": "full-attention score path",
                "e1_diag": "exact QK matmul",
                "proven_lineage": "APA quantized bulk keys with selective exact refinement",
                "anchors": ["core_apa_key_quantization", "core_full_apa_sink"],
            },
            {
                "topic": "full-attention learned sink implementation",
                "e1_diag": "explicit sink column concatenation and composite softmax",
                "proven_lineage": "sink folded into fused online softmax denominator",
                "anchors": ["core_standard_sink_softmax", "tc_fused_sink_contract", "tc_fused_sink_kernel"],
            },
            {
                "topic": "generic mounted-path table extent",
                "e1_diag": "input length only",
                "proven_lineage": "input length plus mounted graft tokens",
                "executed_difference": False,
                "reason": "both cited passing executions are unmounted/capture-only",
                "anchors": ["stream_rope_length"],
            },
        ],
        "long_context_receipts": {
            "h4_64k": {
                "parent": str(H4_64K_PARENT),
                "parent_sha256": sha256_file(H4_64K_PARENT),
                "run": str(H4_64K_RUN),
                "run_sha256": sha256_file(H4_64K_RUN),
                "classification": h4["classification"],
                "actual_input_tokens": h4["actual_input_tokens"],
                "attention_backend_counts": h4["attention_backend_counts"],
            },
            "h5_96k": {
                "parent": str(H5_96K_PARENT),
                "parent_sha256": sha256_file(H5_96K_PARENT),
                "capture": str(H5_96K_CAPTURE),
                "capture_sha256": sha256_file(H5_96K_CAPTURE),
                "classification": h5_parent["classification"],
                "target_tokens": h5_parent["target_tokens"],
                "rope_table_length": h5_capture["rope_table_length"],
                "completed_layers": h5_capture["completed_layers"],
                "attention_backend_counts": h5_counts,
            },
        },
        "source_receipt_limitation": (
            "The July stream receipts do not record script/core source hashes at run time. "
            "This audit seals both receipt hashes and the current lineage source hashes, but "
            "cannot prove byte identity between July source and today's files."
        ),
        "defect_found": False,
        "eval_fix": None,
        "default_behavior_changed": False,
    }


def audit_artifact_path(payload: dict[str, Any]) -> Path:
    audit_key = canonical_json_sha256(
        {
            "result_verbatim": payload["result_verbatim"],
            "anchors": {
                name: {
                    "path": anchor["path"],
                    "line": anchor["line"],
                    "file_sha256": anchor["file_sha256"],
                }
                for name, anchor in payload["anchors"].items()
            },
            "harness_sha256": sha256_file(SCRIPT_PATH),
        }
    )[:16]
    return OUTPUT / "audit" / f"a1_static_audit_{audit_key}.json"


def latest_audit_path() -> Path | None:
    payload = build_audit_payload()
    path = audit_artifact_path(payload)
    return path if path.is_file() else None


def audit(args: argparse.Namespace) -> int:
    ensure_output(args.output_dir)
    payload = build_audit_payload()
    path = audit_artifact_path(payload)
    if path.is_file():
        prior = read_json(path)
        if prior.get("result_verbatim") != payload["result_verbatim"]:
            raise RuntimeError(f"existing A1 audit differs: {path}")
        status = "already_complete"
    else:
        write_new_json(path, payload)
        status = "complete"
    print(
        json.dumps(
            {
                "status": status,
                "audit": str(path),
                "result_verbatim": payload["result_verbatim"],
            }
        ),
        flush=True,
    )
    return 0


def add_check(
    checks: list[dict[str, Any]], name: str, passed: bool, observed: Any
) -> None:
    checks.append({"name": name, "passed": bool(passed), "observed": observed})


def self_test(args: argparse.Namespace) -> int:
    ensure_output(args.output_dir)
    ensure_model(args.model_dir)
    checks: list[dict[str, Any]] = []
    wiki = np.arange(5000, 5000 + WIKITEXT_STREAM_TOKENS, dtype=np.int64)
    guide = np.arange(100000, 100000 + SHORT_WINDOW_TOKENS, dtype=np.int64)
    registry = registered_runs()
    add_check(checks, "registered_run_count", len(registry) == 12, len(registry))
    add_check(
        checks,
        "registered_run_ids_unique",
        len({row["run_id"] for row in registry}) == 12,
        [row["run_id"] for row in registry],
    )
    for length in RAMP_LENGTHS:
        long = case_inputs(
            wiki, guide, case_name="ramp", length=length, view="long"
        )
        ref = case_inputs(
            wiki, guide, case_name="ramp", length=length, view="reference"
        )
        add_check(
            checks,
            f"ramp_{length}_same_targets",
            np.array_equal(long["targets"], ref["targets"]),
            {
                "long": sha256_array(long["targets"]),
                "reference": sha256_array(ref["targets"]),
            },
        )
        add_check(
            checks,
            f"ramp_{length}_lengths",
            long["full_sequence"].size == length
            and ref["full_sequence"].size == SHORT_WINDOW_TOKENS
            and long["targets"].size == N_TARGETS,
            {
                "long": int(long["full_sequence"].size),
                "reference": int(ref["full_sequence"].size),
                "targets": int(long["targets"].size),
            },
        )
    c2 = case_inputs(wiki, guide, case_name="c2", length=None, view=None)
    add_check(
        checks,
        "c2_exact_slice",
        np.array_equal(c2["full_sequence"], wiki[:768])
        and np.array_equal(c2["targets"], wiki[257:768]),
        {
            "full_tokens": int(c2["full_sequence"].size),
            "target_tokens": int(c2["targets"].size),
        },
    )
    c4 = case_inputs(wiki, guide, case_name="c4", length=None, view=None)
    add_check(
        checks,
        "c4_exact_splice",
        np.array_equal(c4["full_sequence"][:256], wiki[:256])
        and np.array_equal(c4["full_sequence"][256:], guide)
        and np.array_equal(c4["targets"], guide[1:]),
        {
            "full_tokens": int(c4["full_sequence"].size),
            "target_hash": sha256_array(c4["targets"]),
        },
    )
    shell = gpu_commands()
    for name, needle in (
        ("gpu_flock", "flock -w 7200 /tmp/forge-gpu.lock"),
        ("gpu_timeout", "timeout --signal=TERM --kill-after=5s 590s"),
        ("gpu_single_device", "CUDA_VISIBLE_DEVICES=0"),
        ("gpu_sleep", "sleep 30"),
        ("gpu_c2", "score --case c2"),
        ("gpu_c4", "score --case c4"),
        ("gpu_ramp_views", "--view reference"),
        ("gpu_complete_analysis", "analyze --require-complete"),
    ):
        add_check(checks, name, needle in shell, needle)
    add_check(
        checks,
        "no_teacher_or_fix_in_gpu_script",
        "teacher" not in shell and "eval-fix" not in shell,
        "teacher/eval-fix absent",
    )
    try:
        audit_probe = build_audit_payload()
        audit_ok = audit_probe["defect_found"] is False and len(
            audit_probe["divergences"]
        ) == 5
        audit_observed: Any = {
            "verdict": audit_probe["verdict"],
            "divergences": len(audit_probe["divergences"]),
        }
    except Exception as exc:
        audit_ok = False
        audit_observed = {"error_type": type(exc).__name__, "error": str(exc)}
    add_check(checks, "a1_anchor_probe", audit_ok, audit_observed)

    passed = all(row["passed"] for row in checks)
    script_hash = sha256_file(SCRIPT_PATH)
    payload = {
        "schema": "moe_e1_3b_cpu_self_test_v1",
        "created_at": now_iso(),
        "status": "pass" if passed else "fail",
        "evidence_class": "synthetic CPU machinery check; not model behavior",
        "script": str(SCRIPT_PATH),
        "script_sha256": script_hash,
        "checks": checks,
        "passed": sum(1 for row in checks if row["passed"]),
        "total": len(checks),
        "gpu_used": False,
    }
    path = OUTPUT / "self_test" / f"cpu_self_test_{script_hash[:16]}.json"
    if path.is_file():
        prior = read_json(path)
        if prior.get("status") != payload["status"]:
            raise RuntimeError(f"existing self-test status differs: {path}")
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


def completed_receipt(run_id: str) -> tuple[dict[str, Any] | None, str]:
    path = receipt_path(run_id, 0)
    if not path.is_file():
        return None, "missing"
    payload = read_json(path)
    if payload.get("status") != "complete":
        return None, str(payload.get("status", "unknown"))
    payload["_path"] = str(path)
    payload["_sha256"] = sha256_file(path)
    return payload, "complete"


def score_summary(receipt: dict[str, Any]) -> dict[str, Any]:
    arm = receipt["score"]
    return {
        "run_id": receipt["run_id"],
        "receipt": receipt["_path"],
        "receipt_sha256": receipt["_sha256"],
        "target_ids_sha256": receipt["target_ids_sha256"],
        "token_count": int(arm["token_count"]),
        "nll_sum": float(arm["nll_sum"]),
        "mean_nll": float(arm["mean_nll"]),
        "ppl": float(arm["ppl"]),
    }


def fmt_number(value: float | None, digits: int = 9) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def classify_complete_mechanism(
    ramp: Sequence[dict[str, Any]],
    c2: dict[str, Any],
    c4: dict[str, Any],
    base_w11: dict[str, Any],
    guide256: dict[str, Any],
    audit_payload: dict[str, Any],
) -> dict[str, Any]:
    by_length = {int(row["length"]): row for row in ramp}
    ratios = {
        length: float(row["long"]["mean_nll"] / row["reference"]["mean_nll"])
        for length, row in by_length.items()
    }
    first_hot = next(
        (length for length in RAMP_LENGTHS if ratios[length] >= POSITION_HOT_MEAN_NLL_RATIO),
        None,
    )
    preboundary_sane = all(ratios[length] < POSITION_HOT_MEAN_NLL_RATIO for length in (1024, 1536, 2048))
    reference_median = float(
        np.median([row["reference"]["mean_nll"] for row in ramp])
    )
    c2_reference_ratio = float(c2["mean_nll"] / reference_median)
    c2_sane = c2_reference_ratio <= C2_REFERENCE_MEDIAN_MEAN_NLL_RATIO
    positional_signal = bool(
        ratios[2560] >= POSITION_HOT_MEAN_NLL_RATIO
        and preboundary_sane
        and first_hot in {2304, 2560}
        and c2_sane
    )
    all_ramp_near_reference = all(
        (1.0 / POSITION_HOT_MEAN_NLL_RATIO) <= value < POSITION_HOT_MEAN_NLL_RATIO
        for value in ratios.values()
    )
    base_nll = float(base_w11["arms"]["base"]["mean_nll"])
    c4_ppl_ratio = float(math.exp(c4["mean_nll"] - base_nll))
    guide_ppl_ratio = float(
        math.exp(guide256["arms"]["teacher"]["mean_nll"] - base_nll)
    )
    splice_signal = max(c4_ppl_ratio, guide_ppl_ratio) >= SUB2048_SPLICE_PPL_RATIO

    candidate_anchor = audit_payload["anchors"]["core_standard_full_mask"]
    candidate_ref = anchor_ref(candidate_anchor)
    if positional_signal and splice_signal:
        label = "MIXED"
        sentence = (
            f"MIXED — BG1-BG2 show a late positional break (first hot ramp length "
            f"{first_hot}, C1/R1 mean-NLL ratio {ratios[2560]:.6f}x, C2/reference-median "
            f"ratio {c2_reference_ratio:.6f}x), localizing the positional component to "
            f"the standard full-attention candidate branch at {candidate_ref}; BG3 also "
            f"shows a below-2048 splice/content component (WikiText256 {c4_ppl_ratio:.6f}x "
            f"and guide256 {guide_ppl_ratio:.6f}x PPL versus the same w11 base)."
        )
    elif positional_signal:
        label = "POSITIONAL-BUG"
        sentence = (
            f"POSITIONAL-BUG — BG1-BG2 show a late break first hot at length {first_hot} "
            f"with C1/R1 mean-NLL ratio {ratios[2560]:.6f}x and a sane C2 anchor; A1 "
            f"localizes the unvalidated executed divergence to {candidate_ref}."
        )
    elif all_ramp_near_reference:
        label = "CONTENT-REAL"
        sentence = (
            f"CONTENT-REAL — all five BG2 same-target long/reference mean-NLL ratios "
            f"remain within the descriptive near-reference band, including C1/R1 "
            f"{ratios[2560]:.6f}x; BG3 attributes the w11 effect to prefix/splice content."
        )
    else:
        label = "MIXED"
        sentence = (
            "MIXED — BG1-BG3 do not form either the registered clean late-position "
            "signature or a flat same-target content signature; the split is an irregular "
            f"ramp (first descriptive hot length {first_hot}) plus sub-2048 splice ratio "
            f"{c4_ppl_ratio:.6f}x."
        )
    return {
        "label": label,
        "sentence": sentence,
        "descriptive_thresholds_not_registered_gates": {
            "position_hot_mean_nll_ratio": POSITION_HOT_MEAN_NLL_RATIO,
            "sub2048_splice_ppl_ratio": SUB2048_SPLICE_PPL_RATIO,
            "c2_vs_reference_median_mean_nll_ratio": C2_REFERENCE_MEDIAN_MEAN_NLL_RATIO,
        },
        "first_hot_ramp_length": first_hot,
        "ramp_mean_nll_ratios": {str(key): value for key, value in ratios.items()},
        "preboundary_sane": preboundary_sane,
        "c2_reference_median_mean_nll": reference_median,
        "c2_reference_median_ratio": c2_reference_ratio,
        "c2_sane": c2_sane,
        "positional_signal": positional_signal,
        "all_ramp_near_reference": all_ramp_near_reference,
        "c4_wikitext256_ppl_ratio_vs_base": c4_ppl_ratio,
        "guide256_ppl_ratio_vs_base": guide_ppl_ratio,
        "sub2048_splice_signal": splice_signal,
        "candidate_source": candidate_ref,
    }


def ramp_markdown(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| Length | long mean_nll | reference mean_nll | ratio | long ppl | reference ppl |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if not row.get("complete"):
            lines.append(f"| {row['length']} | n/a | n/a | n/a | n/a | n/a |")
            continue
        long = row["long"]
        ref = row["reference"]
        ratio = long["mean_nll"] / ref["mean_nll"]
        lines.append(
            f"| {row['length']} | {long['mean_nll']:.9f} | {ref['mean_nll']:.9f} | "
            f"{ratio:.9f}x | {long['ppl']:.6f} | {ref['ppl']:.6f} |"
        )
    return "\n".join(lines)


def render_report(analysis: dict[str, Any]) -> str:
    gates = analysis["gates"]
    bg1 = analysis["bg1"]
    bg3 = analysis["bg3"]
    lines = [
        f"<!-- E13B_SNAPSHOT:{analysis['snapshot_id']} -->",
        "# MOE-E1.3b Position Controls and Long-Context Path Audit",
        "",
        f"Snapshot generated: `{analysis['created_at']}`",
        "",
        "Evidence class: base-arm behavioral measurement on one frozen model/corpus; "
        "the A1 portion is a static/current-lineage audit.",
        "",
        "## BG verdicts",
        "",
        gates["BG1"]["line"],
        gates["BG2"]["line"],
        gates["BG3"]["line"],
        gates["BG4"]["line"],
        "",
        "## BG1 — C1/R1 and C2",
        "",
    ]
    if bg1.get("complete"):
        lines.extend(
            [
                f"- C1 WikiText 2,560: mean_nll={bg1['c1']['mean_nll']:.9f}, ppl={bg1['c1']['ppl']:.6f}",
                f"- R1 same targets in 512 tokens: mean_nll={bg1['r1']['mean_nll']:.9f}, ppl={bg1['r1']['ppl']:.6f}",
                f"- C1/R1 mean-NLL ratio: {bg1['mean_nll_ratio']:.9f}x",
                f"- C2 WikiText 768: mean_nll={bg1['c2']['mean_nll']:.9f}, ppl={bg1['c2']['ppl']:.6f}",
            ]
        )
    else:
        lines.append("C1, R1, and/or C2 GPU receipts are not complete; numbers are n/a.")
    lines.extend(
        [
            "",
            "## BG2 — C3 ramp",
            "",
            analysis["ramp_table_markdown"],
            "",
            "## BG3 — C4 guide-window cross",
            "",
        ]
    )
    if bg3.get("complete"):
        lines.extend(
            [
                f"- w11 base: mean_nll={bg3['base']['mean_nll']:.9f}, ppl={bg3['base']['ppl']:.6f}",
                f"- C4 WikiText256 + w11: mean_nll={bg3['c4']['mean_nll']:.9f}, ppl={bg3['c4']['ppl']:.6f}, PPL ratio vs base={bg3['c4_ppl_ratio_vs_base']:.9f}x",
                f"- E1.2 guide256 + w11: mean_nll={bg3['guide256']['mean_nll']:.9f}, ppl={bg3['guide256']['ppl']:.6f}, PPL ratio vs base={bg3['guide256_ppl_ratio_vs_base']:.9f}x",
            ]
        )
    else:
        lines.append("C4 GPU receipt is not complete; its number is n/a. The sealed guide256 comparator remains recorded.")
    lines.extend(
        [
            "",
            "## A1 audit verdict (verbatim)",
            "",
            analysis["a1_result_verbatim"],
            "",
            "## Mechanism sentence",
            "",
            analysis["mechanism"]["sentence"],
            "",
            "The mechanism label is derived only after BG1-BG3 are complete. The "
            "numeric descriptive rails are printed in the analysis artifact and do "
            "not replace the registered gates.",
            "",
            "## Lead run script",
            "",
            f"`{GPU_SCRIPT}`",
            "",
            "## Limitations",
            "",
            "- No GPU was available or used for the CPU build/audit handoff.",
            "- The July 64K/96K receipts do not contain source hashes at run time; A1 "
            "seals their artifact hashes and today's current-lineage source hashes.",
            "- No static defect was proven, so no `--eval-fix e13b` was added.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(args: argparse.Namespace) -> int:
    ensure_output(args.output_dir)
    ensure_model(args.model_dir)
    manifest, _arrays = validate_prepared()
    audit_path = latest_audit_path()
    if audit_path is None:
        raise FileNotFoundError("run E1.3b audit before analyze")
    audit_payload = read_json(audit_path)

    receipt_states: dict[str, str] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for row in manifest["registered_runs"]:
        payload, state = completed_receipt(row["run_id"])
        receipt_states[row["run_id"]] = state
        if payload is not None:
            receipts[row["run_id"]] = payload

    ramp: list[dict[str, Any]] = []
    for length in RAMP_LENGTHS:
        long_id = f"ramp_{length}_long"
        ref_id = f"ramp_{length}_reference"
        long_receipt = receipts.get(long_id)
        ref_receipt = receipts.get(ref_id)
        row: dict[str, Any] = {
            "length": int(length),
            "complete": long_receipt is not None and ref_receipt is not None,
            "long_state": receipt_states[long_id],
            "reference_state": receipt_states[ref_id],
        }
        if row["complete"]:
            if long_receipt["target_ids_sha256"] != ref_receipt["target_ids_sha256"]:
                raise RuntimeError(f"ramp {length} long/reference target mismatch")
            row["long"] = score_summary(long_receipt)
            row["reference"] = score_summary(ref_receipt)
            row["mean_nll_ratio"] = (
                row["long"]["mean_nll"] / row["reference"]["mean_nll"]
            )
        ramp.append(row)

    c2_receipt = receipts.get("c2_wikitext_768")
    c4_receipt = receipts.get("c4_w11_wikitext256")
    ramp_complete = all(row["complete"] for row in ramp)
    bg1_complete = ramp[-1]["complete"] and c2_receipt is not None
    bg3_complete = c4_receipt is not None
    all_complete = ramp_complete and c2_receipt is not None and c4_receipt is not None

    base_w11_receipt = read_json(E1_BASE_W11)
    guide256_receipt = read_json(E12_GUIDE256_W11)
    base_w11 = base_w11_receipt["arms"]["base"]
    guide256 = guide256_receipt["arms"]["teacher"]
    if base_w11["target_ids_sha256"] != guide256["target_ids_sha256"]:
        raise RuntimeError("sealed BG3 comparator target hashes differ")
    if c4_receipt is not None and c4_receipt["target_ids_sha256"] != base_w11["target_ids_sha256"]:
        raise RuntimeError("C4 target hash differs from sealed w11 comparator")

    if bg1_complete:
        c1 = ramp[-1]["long"]
        r1 = ramp[-1]["reference"]
        c2 = score_summary(c2_receipt)
        bg1 = {
            "complete": True,
            "c1": c1,
            "r1": r1,
            "c2": c2,
            "mean_nll_ratio": float(c1["mean_nll"] / r1["mean_nll"]),
        }
        bg1_line = (
            f"BG1 GREEN — C1 mean_nll={c1['mean_nll']:.9f}, ppl={c1['ppl']:.6f}; "
            f"R1 mean_nll={r1['mean_nll']:.9f}, ppl={r1['ppl']:.6f}; C1/R1 "
            f"mean-NLL ratio={bg1['mean_nll_ratio']:.9f}x; C2 mean_nll={c2['mean_nll']:.9f}, "
            f"ppl={c2['ppl']:.6f}."
        )
    else:
        bg1 = {"complete": False}
        bg1_line = "BG1 NOT_MEASURED — C1/R1 and/or C2 GPU numbers are n/a."
    if ramp_complete:
        bg2_line = "BG2 GREEN — all five C3 long/reference pairs are complete; see the ramp table."
    else:
        bg2_line = f"BG2 NOT_MEASURED — {sum(1 for row in ramp if row['complete'])}/5 ramp pairs are complete."

    c4_summary = score_summary(c4_receipt) if c4_receipt is not None else None
    base_summary = {
        "token_count": int(base_w11["token_count"]),
        "nll_sum": float(base_w11["nll_sum"]),
        "mean_nll": float(base_w11["mean_nll"]),
        "ppl": float(base_w11["ppl"]),
        "target_ids_sha256": base_w11["target_ids_sha256"],
    }
    guide_summary = {
        "token_count": int(guide256["token_count"]),
        "nll_sum": float(guide256["nll_sum"]),
        "mean_nll": float(guide256["mean_nll"]),
        "ppl": float(guide256["ppl"]),
        "target_ids_sha256": guide256["target_ids_sha256"],
    }
    if bg3_complete:
        c4_ratio = float(math.exp(c4_summary["mean_nll"] - base_summary["mean_nll"]))
        guide_ratio = float(math.exp(guide_summary["mean_nll"] - base_summary["mean_nll"]))
        bg3 = {
            "complete": True,
            "c4": c4_summary,
            "base": base_summary,
            "guide256": guide_summary,
            "c4_ppl_ratio_vs_base": c4_ratio,
            "guide256_ppl_ratio_vs_base": guide_ratio,
        }
        bg3_line = (
            f"BG3 GREEN — C4 mean_nll={c4_summary['mean_nll']:.9f}, ppl={c4_summary['ppl']:.6f}, "
            f"PPL ratio vs w11 base={c4_ratio:.9f}x; guide256 comparator "
            f"mean_nll={guide_summary['mean_nll']:.9f}, ppl={guide_summary['ppl']:.6f}, "
            f"ratio={guide_ratio:.9f}x."
        )
    else:
        bg3 = {
            "complete": False,
            "base": base_summary,
            "guide256": guide_summary,
            "guide256_ppl_ratio_vs_base": float(
                math.exp(guide_summary["mean_nll"] - base_summary["mean_nll"])
            ),
        }
        bg3_line = "BG3 NOT_MEASURED — C4 GPU number is n/a; sealed guide256 context is retained."

    if all_complete:
        mechanism = classify_complete_mechanism(
            ramp,
            score_summary(c2_receipt),
            c4_summary,
            base_w11_receipt,
            guide256_receipt,
            audit_payload,
        )
        bg4_line = f"BG4 GREEN — A1 complete; {mechanism['sentence']}"
    else:
        mechanism = {
            "label": "NOT_MEASURED",
            "sentence": "NOT_MEASURED — BG1-BG3 GPU receipts are incomplete; no mechanism label is assigned.",
        }
        bg4_line = "BG4 NOT_MEASURED — A1 is complete, but mechanism awaits complete BG1-BG3 receipts."

    evidence_key = {
        "manifest_sha256": sha256_file(CONTROL_MANIFEST),
        "audit_sha256": sha256_file(audit_path),
        "receipt_states": receipt_states,
        "receipt_hashes": {
            key: value["_sha256"] for key, value in receipts.items()
        },
    }
    snapshot_id = canonical_json_sha256(evidence_key)[:16]
    analysis: dict[str, Any] = {
        "schema": "moe_e1_3b_analysis_v1",
        "snapshot_id": snapshot_id,
        "created_at": now_iso(),
        "status": "complete" if all_complete else "not_measured",
        "order": str(ORDER_PATH),
        "evidence_class": "base-arm behavioral measurement on one frozen model/corpus",
        "gates": {
            "BG1": {"verdict": "GREEN" if bg1_complete else "NOT_MEASURED", "line": bg1_line},
            "BG2": {"verdict": "GREEN" if ramp_complete else "NOT_MEASURED", "line": bg2_line},
            "BG3": {"verdict": "GREEN" if bg3_complete else "NOT_MEASURED", "line": bg3_line},
            "BG4": {"verdict": "GREEN" if all_complete else "NOT_MEASURED", "line": bg4_line},
        },
        "bg1": bg1,
        "bg2": {"complete": ramp_complete, "ramp": ramp},
        "bg3": bg3,
        "bg4": {
            "complete": all_complete,
            "a1_audit": str(audit_path),
            "a1_audit_sha256": sha256_file(audit_path),
            "mechanism": mechanism,
        },
        "a1_result_verbatim": audit_payload["result_verbatim"],
        "mechanism": mechanism,
        "ramp_table_markdown": ramp_markdown(ramp),
        "receipt_states": receipt_states,
        "completed_gpu_runs": len(receipts),
        "registered_gpu_runs": 12,
        "gpu_script": str(GPU_SCRIPT),
        "gpu_script_sha256": sha256_file(GPU_SCRIPT),
        "control_manifest": str(CONTROL_MANIFEST),
        "control_manifest_sha256": sha256_file(CONTROL_MANIFEST),
        "limitations": [
            "CPU build/audit used no GPU and produces no BG behavior numbers.",
            audit_payload["source_receipt_limitation"],
            "No static defect was proven; no eval-fix e13b path was added.",
        ],
    }
    analysis_path = OUTPUT / "analysis" / f"analysis_{snapshot_id}.json"
    if analysis_path.is_file():
        prior = read_json(analysis_path)
        if prior.get("snapshot_id") != snapshot_id:
            raise RuntimeError(f"analysis snapshot collision: {analysis_path}")
    else:
        write_new_json(analysis_path, analysis)
    report_text = render_report(analysis)
    marker = f"<!-- E13B_SNAPSHOT:{snapshot_id} -->"
    append_text_once(REPORT, marker, report_text)
    print(
        json.dumps(
            {
                "status": analysis["status"],
                "analysis": str(analysis_path),
                "report": str(REPORT),
                "gates": {
                    name: gate["verdict"] for name, gate in analysis["gates"].items()
                },
                "completed_gpu_runs": len(receipts),
                "mechanism": mechanism["label"],
            }
        ),
        flush=True,
    )
    if args.require_complete and not all_complete:
        return 3
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "prepare":
        return prepare(args)
    if args.mode == "self-test":
        return self_test(args)
    if args.mode == "audit":
        return audit(args)
    if args.mode == "score":
        return score(args)
    if args.mode == "analyze":
        return analyze(args)
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
