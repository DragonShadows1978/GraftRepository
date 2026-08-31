#!/usr/bin/env python3
"""Self-leasing GPU runner for ORDER GRM-DET1.

Every GPU child is fresh, holds ``/tmp/forge-gpu.lock`` for at most 580
seconds (hard cap 590), and is separated from the next lease by 30 seconds.
CPU threshold fitting happens after the calibration lease is released and
before either eval lease begins.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")

from scripts.grm_det1_common import (  # noqa: E402
    DETError,
    canonical_json_bytes,
    file_record,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_content_addressed,
    write_json_exclusive,
)


LOCK_PATH = Path("/tmp/forge-gpu.lock")
MAX_LEASE_SECONDS = 590
DEFAULT_LEASE_SECONDS = 580
GAP_SECONDS = 30
E2E = ROOT / "scripts" / "grm_e2e_session.py"
WRAPPER = ROOT / "scripts" / "grm_det1_e2e.py"
ANALYZE = ROOT / "scripts" / "grm_det1_analyze.py"
MODEL_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
MODEL_DIR = Path(
    "/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/"
    f"snapshots/{MODEL_REVISION}"
)
NATIVE_LIB = ROOT / "cpp" / "build" / "libgrm_runtime.so"
STAGE_ORDER = (
    "g1_baseline",
    "g1_off",
    "calibration",
    "eval_e2e_1",
    "eval_e2e_2",
    "eval_e2e_3",
    "eval_e2e_4",
    "eval_sup_1",
    "eval_sup_2",
    "eval_sup_3",
    "eval_sup_4",
)
E2E_SEGMENTS = {
    "calibration": {
        "detector_stage": "calibration", "resume": False,
        "stop_after_turns": 17, "expected_rows": 2,
        "marker": "e2e_stage_complete.json",
    },
    "eval_e2e_1": {
        "detector_stage": "eval", "resume": False,
        "stop_after_turns": 10, "expected_rows": 2,
        "marker": "e2e_segment_1_complete.json",
    },
    "eval_e2e_2": {
        "detector_stage": "eval", "resume": True,
        "stop_after_turns": 17, "expected_rows": 4,
        "marker": "e2e_segment_2_complete.json",
    },
    "eval_e2e_3": {
        "detector_stage": "eval", "resume": True,
        "stop_after_turns": 25, "expected_rows": 10,
        "marker": "e2e_segment_3_complete.json",
    },
    "eval_e2e_4": {
        "detector_stage": "eval", "resume": True,
        "stop_after_turns": 31, "expected_rows": 14,
        "marker": "e2e_segment_4_complete.json",
    },
}


@contextmanager
def gpu_lease(seconds: int, wait_seconds: int) -> Iterator[None]:
    if seconds <= 0 or seconds > MAX_LEASE_SECONDS:
        raise DETError(f"lease must be in 1..{MAX_LEASE_SECONDS}s")
    handle = LOCK_PATH.open("a+")
    wait_started = time.monotonic()
    last_notice = -30
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            waited = int(time.monotonic() - wait_started)
            if waited >= int(wait_seconds):
                handle.close()
                raise DETError(f"GPU lock unavailable after {wait_seconds}s")
            if waited - last_notice >= 30:
                print(f"waiting_for_gpu_lock_s={waited}", flush=True)
                last_notice = waited
            time.sleep(5)
    old_handler = signal.getsignal(signal.SIGALRM)

    def expired(_signum, _frame):
        raise TimeoutError(f"GPU lease exceeded {seconds}s")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(int(seconds))
    started = time.monotonic()
    try:
        print(f"gpu_lease_acquired={LOCK_PATH} cap_s={seconds}", flush=True)
        yield
    finally:
        elapsed = time.monotonic() - started
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        print(f"gpu_lease_released_s={elapsed:.3f}", flush=True)


def _one(directory: Path, pattern: str) -> Path:
    paths = sorted(Path(directory).glob(pattern))
    if len(paths) != 1:
        raise DETError(f"expected one {pattern} under {directory}, found {len(paths)}")
    return paths[0]


def _record_path(record: Mapping[str, Any]) -> Path:
    path = Path(str(record["path"]))
    return path if path.is_absolute() else ROOT / path


def _validate_file_record(record: Mapping[str, Any], label: str) -> Path:
    path = _record_path(record)
    if not path.is_file() or file_record(path) != dict(record):
        raise DETError(f"{label} file record no longer validates: {record}")
    return path


def _write_bytes_exclusive_or_verify(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise DETError(f"immutable output collision: {path}")
        return path
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
    return path


def _validate_visibility() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").strip()
    if not visible or "," in visible:
        raise DETError(f"DET1 requires one visible GPU, got {visible!r}")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    return visible


def _gpu_snapshot() -> dict[str, Any]:
    process = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=10,
    )
    return {
        "returncode": int(process.returncode),
        "stdout": process.stdout.strip(),
        "stderr": process.stderr.strip(),
    }


def _git_value(*args: str) -> str:
    process = subprocess.run(
        ["git", *args], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return process.stdout.strip() if process.returncode == 0 else ""


def _registration(run_dir: Path) -> Path:
    return _one(run_dir, "registration_*.json")


def _check_inventory(registration: Mapping[str, Any]) -> None:
    for shown, expected in registration["production_source_inventory"].items():
        path = Path(shown)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise DETError(f"registered source disappeared: {path}")
        observed = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if observed != expected:
            raise DETError(
                f"source drift after pre-data registration: {shown}: "
                f"expected={expected} observed={observed}")
    for shown, expected in registration.get("model_snapshot_inventory", {}).items():
        path = Path(shown)
        observed = {
            "bytes": path.stat().st_size if path.is_file() else None,
            "resolved_blob": path.resolve().name if path.is_file() else None,
            "verification": "huggingface_content_addressed_blob_identity",
        }
        if observed != expected:
            raise DETError(
                f"model snapshot drift after pre-data registration: {shown}: "
                f"expected={expected} observed={observed}")


def _resolve_runtime_frame(
    run_dir: Path,
    *,
    lease_seconds: int,
    wait_seconds: int,
    visible_device: str,
) -> Path:
    registration_path = _registration(run_dir)
    registration = read_json(registration_path)
    _check_inventory(registration)
    existing = sorted(run_dir.glob("runtime_frame_*.json"))
    if existing:
        if len(existing) != 1:
            raise DETError("multiple runtime frames exist")
        path = existing[0]
        frame = read_json(path)
        if frame.get("registration") != file_record(registration_path):
            raise DETError("runtime frame is not bound to the current registration")
        if frame.get("native_library") != file_record(NATIVE_LIB):
            raise DETError("native runtime drifted from the frozen runtime frame")
        if str(frame.get("cuda_visible_devices")) != str(visible_device):
            raise DETError("CUDA_VISIBLE_DEVICES differs from the frozen runtime frame")
        if int(frame["gpu_lease"]["actual_cap_seconds"]) != int(lease_seconds):
            raise DETError("lease cap differs from the frozen runtime frame")
        return path
    from scripts import grm_e2e_session as e2e
    from core.grm_admission import adm_decisive_enabled
    from core.graft_arena import ArenaCache
    from scripts.grm_probe_ladder import probe_ladder_enabled
    from core.grm_supersession import sup_resolve_enabled

    parsed = e2e.parse_args([])
    ladder = bool(probe_ladder_enabled(SimpleNamespace(probe_ladder=None)))
    l2 = bool(sup_resolve_enabled(None))
    adm_decisive = bool(adm_decisive_enabled(None))
    route_query_lex = bool(ArenaCache._route_query_lex_enabled())
    if not ladder or not l2:
        raise DETError(
            "ORDER GRM-DET1 fixes ladder ON and L2 ON; ambient runtime resolved "
            f"ladder={ladder} L2={l2}")
    frame = {
        "schema": "grm.det1.runtime_frame.v1",
        "resolved_utc": utc_now(),
        "resolved_unix_ns": time.time_ns(),
        "registration": file_record(registration_path),
        "git": {
            "head": _git_value("rev-parse", "HEAD"),
            "branch": _git_value("branch", "--show-current"),
            "status_short": _git_value("status", "--short"),
        },
        "ambient_env_before_pin": {
            key: os.environ.get(key, "")
            for key in (
                "GRM_PROBE_LADDER", "GRM_SUP_RESOLVE", "GRM_ADM_DECISIVE",
                "GRM_GQA_CUDA_ROUTE", "GRM_GRAFT_STORAGE_BITS",
                "GRM_ROUTE_QUERY_LEX",
            )
        },
        "resolved_flags": {
            "probe_ladder": ladder,
            "sup_resolve": l2,
            "adm_decisive": adm_decisive,
            "admission_mode": "A-DEC" if adm_decisive else "fixed-k3",
            "topk": int(parsed.topk),
            "turn_pipeline": str(parsed.turn_pipeline),
            "length_debias": False,
            "max_trips": 1,
            "live_turns": 2,
            "arena_width": 96,
            "max_live": 4096,
            "ngen": 32,
            "gqa_cuda_route": True,
            "graft_storage_bits": 8,
            "route_query_lex": route_query_lex,
            "vram_budget_mb": None,
        },
        "model": {
            "repository": "openai/gpt-oss-20b",
            "revision": MODEL_REVISION,
            "path": str(MODEL_DIR),
        },
        "native_library": file_record(NATIVE_LIB),
        "cuda_visible_devices": str(visible_device),
        "gpu_lease": {
            "lock": str(LOCK_PATH),
            "default_cap_seconds": DEFAULT_LEASE_SECONDS,
            "hard_cap_seconds": MAX_LEASE_SECONDS,
            "actual_cap_seconds": int(lease_seconds),
            "lock_wait_seconds": int(wait_seconds),
            "inter_stage_gap_seconds": GAP_SECONDS,
        },
        "source_inventory_match": True,
    }
    return write_content_addressed(run_dir, "runtime_frame", frame)


def _base_e2e_command(
    entry: Path,
    session_dir: Path,
    mode: str,
    runtime_frame: Mapping[str, Any],
) -> list[str]:
    if mode not in ("smoke", "full"):
        raise DETError(f"bad E2E mode {mode}")
    command = [
        sys.executable,
        str(entry),
        "--mode", mode,
        "--session-dir", str(session_dir),
        "--model-dir", str(MODEL_DIR),
        "--native-lib", str(NATIVE_LIB),
        "--ngen", "24" if mode == "smoke" else "32",
        "--max-trips", "1",
        "--live-turns", "1" if mode == "smoke" else "2",
        "--arena-width", "96",
        "--max-live", "4096",
        "--topk", "3",
        "--turn-pipeline", "single",
        "--restart-after", "5" if mode == "smoke" else "17",
        "--probe-ladder",
        "--sup-resolve",
        "--skip-gpu-idle-check",
    ]
    command.append(
        "--adm-decisive"
        if runtime_frame["resolved_flags"]["adm_decisive"]
        else "--no-adm-decisive")
    return command


def _run_process(command: Sequence[str], env: Mapping[str, str], timeout: int) -> int:
    process = subprocess.Popen(
        list(command), cwd=ROOT, env=dict(env), start_new_session=True)
    try:
        return int(process.wait(timeout=max(1, int(timeout))))
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    prior = env.get("PYTHONPATH", "")
    env.update({
        "PYTHONPATH": str(ROOT) + (os.pathsep + prior if prior else ""),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return env


def _runtime_env(runtime_frame: Mapping[str, Any]) -> dict[str, str]:
    env = _base_env()
    flags = runtime_frame["resolved_flags"]
    env.update({
        "CUDA_VISIBLE_DEVICES": str(runtime_frame["cuda_visible_devices"]),
        "GRM_PROBE_LADDER": "1" if flags["probe_ladder"] else "0",
        "GRM_SUP_RESOLVE": "1" if flags["sup_resolve"] else "0",
        "GRM_ADM_DECISIVE": "1" if flags["adm_decisive"] else "0",
        "GRM_GQA_CUDA_ROUTE": "1" if flags["gqa_cuda_route"] else "0",
        "GRM_GRAFT_STORAGE_BITS": str(flags["graft_storage_bits"]),
        "GRM_ROUTE_QUERY_LEX": "1" if flags["route_query_lex"] else "0",
    })
    return env


def _receipt_context(
    runtime_frame: Mapping[str, Any],
    *,
    lease_seconds: int,
    actual_flag_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    flags = dict(runtime_frame["resolved_flags"])
    flags.update(dict(actual_flag_overrides or {}))
    pins = _runtime_env(runtime_frame)
    pin_keys = (
        "CUDA_VISIBLE_DEVICES", "GRM_PROBE_LADDER", "GRM_SUP_RESOLVE",
        "GRM_ADM_DECISIVE", "GRM_GQA_CUDA_ROUTE",
        "GRM_GRAFT_STORAGE_BITS", "GRM_ROUTE_QUERY_LEX",
    )
    return {
        "resolved_production_flags": dict(runtime_frame["resolved_flags"]),
        "actual_leg_flags": flags,
        "cuda_visible_devices": str(runtime_frame["cuda_visible_devices"]),
        "lease_seconds": int(lease_seconds),
        "environment_pins": {key: pins[key] for key in pin_keys},
    }


def _canonical_session(session_dir: Path) -> dict[str, Any]:
    paths = {
        "transcript.jsonl": session_dir / "transcript.jsonl",
        "probe_scorecard.json": session_dir / "probe_scorecard.json",
    }
    return {
        "comparison": "raw_file_bytes",
        "files": {
            name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
    }


def _g1_leg(
    run_dir: Path,
    leg: str,
    runtime_frame_path: Path,
    *,
    lease_seconds: int,
) -> None:
    stage_dir = run_dir / "g1" / leg
    marker = stage_dir / "stage_complete.json"
    if marker.exists():
        receipt = read_json(marker)
        if receipt.get("status") != "COMPLETE":
            raise DETError(f"G1 {leg} marker is not COMPLETE")
        _validate_file_record(receipt["canonical"], f"G1 {leg} canonical")
        print(f"stage={leg} status=already_complete", flush=True)
        return
    attempts = sorted(stage_dir.glob("attempt_*"))
    attempt_dir = stage_dir / f"attempt_{len(attempts) + 1:03d}"
    session_dir = attempt_dir / "session"
    entry = E2E if leg == "baseline" else WRAPPER
    runtime_frame = read_json(runtime_frame_path)
    env = _runtime_env(runtime_frame)
    env["GRM_DET1_ENABLED"] = "0"
    command = _base_e2e_command(entry, session_dir, "smoke", runtime_frame)
    started = time.monotonic()
    returncode = _run_process(command, env, int(lease_seconds) - 12)
    if returncode not in (0, 2):
        raise DETError(f"G1 {leg} E2E returned {returncode}")
    canonical = _canonical_session(session_dir)
    canonical_path = write_json_exclusive(attempt_dir / "canonical.json", canonical)
    write_json_exclusive(marker, {
        "schema": "grm.det1.g1_leg.v1",
        "leg": leg,
        "status": "COMPLETE",
        "command": command,
        "returncode": returncode,
        "elapsed_seconds": time.monotonic() - started,
        "runtime_frame": file_record(runtime_frame_path),
        "runtime": _receipt_context(
            runtime_frame,
            lease_seconds=lease_seconds,
            actual_flag_overrides={
                "mode": "smoke", "ngen": 24, "live_turns": 1,
                "restart_after": 5,
            },
        ),
        "canonical": file_record(canonical_path),
        "gpu": _gpu_snapshot(),
    })


def _finalize_g1(run_dir: Path, runtime_frame_path: Path) -> Path:
    existing = sorted((run_dir / "g1").glob("g1_receipt_*.json"))
    if existing:
        path = _one(run_dir / "g1", "g1_receipt_*.json")
        receipt = read_json(path)
        if receipt.get("status") != "PASS" or receipt.get("byte_identical") is not True:
            raise DETError("existing DET-G1 receipt is not PASS")
        _validate_file_record(receipt["baseline"], "DET-G1 baseline")
        _validate_file_record(receipt["instrumentation_off"], "DET-G1 off")
        return path
    baseline_marker = read_json(run_dir / "g1" / "baseline" / "stage_complete.json")
    off_marker = read_json(run_dir / "g1" / "off" / "stage_complete.json")
    baseline = _validate_file_record(baseline_marker["canonical"], "DET-G1 baseline")
    off = _validate_file_record(off_marker["canonical"], "DET-G1 off")
    base_bytes = canonical_json_bytes(read_json(baseline))
    off_bytes = canonical_json_bytes(read_json(off))
    runtime_frame = read_json(runtime_frame_path)
    receipt = {
        "schema": "grm.det1.g1.v1",
        "created_utc": utc_now(),
        "status": "PASS" if base_bytes == off_bytes else "RED",
        "byte_identical": base_bytes == off_bytes,
        "baseline_sha256": hashlib.sha256(base_bytes).hexdigest(),
        "instrumentation_off_sha256": hashlib.sha256(off_bytes).hexdigest(),
        "baseline": file_record(baseline),
        "instrumentation_off": file_record(off),
        "runtime_frame": file_record(runtime_frame_path),
        "resolved_flags": dict(runtime_frame["resolved_flags"]),
        "cuda_visible_devices": str(runtime_frame["cuda_visible_devices"]),
        "comparison": "raw transcript.jsonl + probe_scorecard.json file bytes",
        "scope": (
            "complete deterministic production transcript and scorecard; "
            "timing/session-path artifacts are excluded"
        ),
    }
    path = write_content_addressed(run_dir / "g1", "g1_receipt", receipt)
    if receipt["status"] != "PASS":
        raise DETError("DET-G1 flags-off bytes differ from production baseline")
    return path


def _e2e_stage(
    run_dir: Path,
    *,
    segment_name: str,
    registration_path: Path,
    runtime_frame_path: Path,
    lease_seconds: int,
) -> None:
    if segment_name not in E2E_SEGMENTS:
        raise DETError(f"bad E2E segment {segment_name}")
    spec = E2E_SEGMENTS[segment_name]
    detector_stage = str(spec["detector_stage"])
    stage_dir = run_dir / (
        "calibration" if detector_stage == "calibration" else "eval")
    marker = stage_dir / str(spec["marker"])
    if marker.exists():
        receipt = read_json(marker)
        if receipt.get("status") != "COMPLETE":
            raise DETError(f"{segment_name} marker is not COMPLETE")
        rows_path = _validate_file_record(receipt["rows"], f"{segment_name} rows")
        if len(read_jsonl(rows_path)) != int(spec["expected_rows"]):
            raise DETError(f"{segment_name} marker row count no longer validates")
        print(f"stage={segment_name} status=already_complete", flush=True)
        return
    if bool(spec["resume"]):
        prior_name = {
            "eval_e2e_2": "e2e_segment_1_complete.json",
            "eval_e2e_3": "e2e_segment_2_complete.json",
            "eval_e2e_4": "e2e_segment_3_complete.json",
        }[segment_name]
        prior = read_json(stage_dir / prior_name)
        session_dir = Path(str(prior["session_dir"]))
        prior_snapshot = _validate_file_record(
            prior["rows"], f"{segment_name} prior rows")
        prior_session = Path(str(prior["session_dir"]))
        attempts = sorted(stage_dir.glob(f"{segment_name}_attempt_*"))
        attempt_dir = stage_dir / f"{segment_name}_attempt_{len(attempts) + 1:03d}"
        session_dir = attempt_dir / "session"
        rows_path = attempt_dir / "rows.jsonl"
        # Fork only completed immutable checkpoints. A killed segment leaves
        # its attempt directory as evidence; rerun starts from the prior
        # marker without truncating or trusting partial state.
        shutil.copytree(prior_session, session_dir, symlinks=True)
        _write_bytes_exclusive_or_verify(rows_path, prior_snapshot.read_bytes())
    else:
        campaigns = sorted(stage_dir.glob("e2e_campaign_*"))
        campaign = stage_dir / f"e2e_campaign_{len(campaigns) + 1:03d}"
        session_dir = campaign / "session"
        rows_path = campaign / "rows.jsonl"
    runtime_frame = read_json(runtime_frame_path)
    env = _runtime_env(runtime_frame)
    env.update({
        "GRM_DET1_ENABLED": "1",
        "GRM_DET1_REGISTRATION": str(registration_path),
        "GRM_DET1_RUNTIME_FRAME": str(runtime_frame_path),
        "GRM_DET1_ROWS": str(rows_path),
        "GRM_DET1_STAGE": detector_stage,
        "GRM_DET1_STOP_AFTER_TURNS": str(spec["stop_after_turns"]),
    })
    command = _base_e2e_command(
        WRAPPER, session_dir, "full", runtime_frame)
    if bool(spec["resume"]):
        command.append("--resume")
    started = time.monotonic()
    returncode = _run_process(command, env, int(lease_seconds) - 12)
    if returncode not in (0, 2):
        raise DETError(f"DET1 {segment_name} E2E returned {returncode}")
    rows = read_jsonl(rows_path)
    expected = int(spec["expected_rows"])
    if len(rows) != expected:
        raise DETError(
            f"{segment_name} E2E expected {expected} rows, observed {len(rows)}")
    snapshot_path = _write_bytes_exclusive_or_verify(
        rows_path.parent / f"rows_snapshot_{expected:02d}.jsonl",
        rows_path.read_bytes(),
    )
    if detector_stage == "calibration":
        _write_bytes_exclusive_or_verify(
            stage_dir / "e2e_rows.jsonl", rows_path.read_bytes())
    if segment_name == "eval_e2e_4":
        _write_bytes_exclusive_or_verify(
            stage_dir / "e2e_rows.jsonl", rows_path.read_bytes())
    write_json_exclusive(marker, {
        "schema": "grm.det1.e2e_stage.v1",
        "stage": detector_stage,
        "segment": segment_name,
        "status": "COMPLETE",
        "command": command,
        "returncode": returncode,
        "elapsed_seconds": time.monotonic() - started,
        "row_count": len(rows),
        "rows": file_record(snapshot_path),
        "active_rows_path": str(rows_path),
        "session_config": file_record(session_dir / "run_config.json"),
        "scorecard": file_record(session_dir / "probe_scorecard.json"),
        "runtime_frame": file_record(runtime_frame_path),
        "runtime": _receipt_context(runtime_frame, lease_seconds=lease_seconds),
        "session_dir": str(session_dir),
        "stop_after_turns": int(spec["stop_after_turns"]),
        "resume": bool(spec["resume"]),
        "gpu": _gpu_snapshot(),
    })


def _install_fixture_nodes(repo, fixture: Mapping[str, Any]) -> dict[str, int]:
    arena = repo.arena
    node_to_idx: dict[str, int] = {}
    for node in fixture["nodes"]:
        index = int(arena.deposit(str(node["text"])))
        node_to_idx[str(node["node_id"])] = index
        arena.grafts[index]["node_id"] = index
        arena.grafts[index]["kind"] = "fact"
        arena.grafts[index]["metadata"] = {
            "kind": "fact", "active": True,
            "supersedes": [], "superseded_by": [],
        }
    for node in fixture["nodes"]:
        index = node_to_idx[str(node["node_id"])]
        older = [node_to_idx[str(parent)] for parent in node["supersedes"]]
        arena.grafts[index]["metadata"]["supersedes"] = older
        for old in older:
            arena.grafts[old]["metadata"]["superseded_by"].append(index)
    arena._bump_cuda_gqa_epoch()
    # Direct fixture deposits must still satisfy the production native-store
    # identity contract before routing or the first mount.  Syncing final
    # metadata transfers the revision graph while preserving the battery's
    # explicit all-revisions-active routing surface; native apply_revision
    # would instead retire older candidates and change the fixture.
    for index in range(len(arena.grafts)):
        repo._native_sync_node(index)
    return node_to_idx


def _sup_stage(
    run_dir: Path,
    *,
    segment_index: int,
    registration_path: Path,
    runtime_frame_path: Path,
    lease_seconds: int,
) -> None:
    stage_dir = run_dir / "eval"
    fixtures = sorted(
        (ROOT / "tests" / "fixtures" / "supersession_battery").glob("*.json"))
    if len(fixtures) != 4 or not 1 <= int(segment_index) <= len(fixtures):
        raise DETError(
            f"expected four supersession fixture segments, got {len(fixtures)}")
    fixture_path = fixtures[int(segment_index) - 1]
    final_segment = int(segment_index) == len(fixtures)
    marker = stage_dir / (
        "sup_stage_complete.json" if final_segment
        else f"sup_segment_{int(segment_index)}_complete.json")
    if marker.exists():
        receipt = read_json(marker)
        if receipt.get("status") != "COMPLETE":
            raise DETError(f"eval_sup_{segment_index} marker is not COMPLETE")
        rows_path = _validate_file_record(
            receipt["rows"], f"eval_sup_{segment_index} rows")
        expected = 10 if final_segment else int(receipt["fixture_row_count"])
        if len(read_jsonl(rows_path)) != expected:
            raise DETError(
                f"eval_sup_{segment_index} completed row count no longer validates")
        print(f"stage=eval_sup_{segment_index} status=already_complete", flush=True)
        return

    registration = read_json(registration_path)
    runtime_frame = read_json(runtime_frame_path)
    registered = {
        str(row["probe_id"]): row
        for row in registration["fixtures"]
        if row["source_family"] == "supersession_battery_on_gpt_oss"
    }
    started = time.monotonic()
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    session_id = str(fixture["session_id"])
    fixture_dir = stage_dir / "sup_fixtures" / session_id
    fixture_marker = fixture_dir / "complete.json"
    expected_fixture_rows = 2 * len(fixture["probes"])
    model_info = None
    if fixture_marker.exists():
        completed = read_json(fixture_marker)
        if not (
            completed.get("status") == "COMPLETE"
            and completed.get("source") == file_record(fixture_path)
            and completed.get("runtime_frame") == file_record(runtime_frame_path)
        ):
            raise DETError(f"sup fixture {session_id} marker binding drifted")
        fixture_rows_path = _validate_file_record(
            completed["rows"], f"sup fixture {session_id}")
        if len(read_jsonl(fixture_rows_path)) != expected_fixture_rows:
            raise DETError(f"sup fixture {session_id} row count drifted")
    else:
        from transformers import AutoTokenizer
        import tensor_cuda as tc
        from core import kv_graft
        from core.gpt_oss20b_tc import GptOss20B_TC, gpt_oss_grm_dialect_kwargs
        from core.graft_repository import GraftRepository
        from scripts import grm_e2e_session as e2e
        from scripts.grm_det1_e2e import evaluate_registered_probe

        attempts = sorted(fixture_dir.glob("attempt_*"))
        attempt_dir = fixture_dir / f"attempt_{len(attempts) + 1:03d}"
        fixture_rows_path = attempt_dir / "rows.jsonl"
        repo_dir = attempt_dir / "repository"
        model = None
        tokenizer = None
        try:
            model, model_info = GptOss20B_TC.from_pretrained(MODEL_DIR)
            tokenizer = AutoTokenizer.from_pretrained(
                str(MODEL_DIR), local_files_only=True)
            encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
            decode = lambda ids: tokenizer.decode(
                ids, clean_up_tokenization_spaces=False)
            dialect = gpt_oss_grm_dialect_kwargs(model.config)
            repo = GraftRepository(
                model, encode, decode, str(repo_dir),
                autosave=False,
                arena_cls=e2e.GptOssGQAArenaCache,
                native_lib_path=str(NATIVE_LIB),
                native_auto=False,
                vram_budget_mb=None,
                route_layer=int(dialect["route_layer"]),
                arena_width=96,
                topk=3,
                live_turns=2,
                max_live=4096,
                sink_text=e2e.HARMONY_SINK,
                prompt_template=e2e.harmony_turn,
                stop_sequences=e2e.HARMONY_STOPS,
                storage_bits=8,
                revision_resolution=True,
                decisive_admission=bool(
                    runtime_frame["resolved_flags"]["adm_decisive"]),
            )
            try:
                _install_fixture_nodes(repo, fixture)
                for probe in fixture["probes"]:
                    evaluate_registered_probe(
                        repo, e2e, e2e.probe_multimount_chat,
                        fixture=registered[str(probe["probe_id"])],
                        stage="eval",
                        rows_path=fixture_rows_path,
                        runtime_frame=runtime_frame,
                        registration_path=registration_path,
                        topk=3,
                        ngen=32,
                        max_trips=1,
                        turn_idx=None,
                        canonical_defer_memory=True,
                    )
            finally:
                repo.close()
                kv_graft.clear_injection(model)
                del repo
        finally:
            if tokenizer is not None:
                del tokenizer
            if model is not None:
                del model
            gc.collect()
            if hasattr(tc, "empty_cache"):
                tc.empty_cache()
        fixture_rows = read_jsonl(fixture_rows_path)
        if len(fixture_rows) != expected_fixture_rows:
            raise DETError(
                f"sup fixture {session_id} expected {expected_fixture_rows} "
                f"rows, observed {len(fixture_rows)}")
        write_json_exclusive(fixture_marker, {
            "schema": "grm.det1.sup_fixture.v1",
            "status": "COMPLETE",
            "segment_index": int(segment_index),
            "session_id": session_id,
            "source": file_record(fixture_path),
            "rows": file_record(fixture_rows_path),
            "runtime_frame": file_record(runtime_frame_path),
            "runtime": _receipt_context(
                runtime_frame, lease_seconds=lease_seconds),
        })

    receipt = {
        "schema": "grm.det1.sup_stage.v1",
        "stage": "eval",
        "status": "COMPLETE",
        "segment_index": int(segment_index),
        "fixture_source": file_record(fixture_path),
        "fixture_row_count": expected_fixture_rows,
        "elapsed_seconds": time.monotonic() - started,
        "row_count": expected_fixture_rows,
        "rows": file_record(fixture_rows_path),
        "model_info": model_info,
        "runtime_frame": file_record(runtime_frame_path),
        "runtime": _receipt_context(runtime_frame, lease_seconds=lease_seconds),
        "command": f"in-process GPT-OSS supersession fixture {fixture_path.name}",
        "gpu": _gpu_snapshot(),
    }
    if final_segment:
        completed_row_paths = []
        for source in fixtures:
            value = json.loads(source.read_text(encoding="utf-8"))
            completed = read_json(
                stage_dir / "sup_fixtures" / str(value["session_id"])
                / "complete.json")
            if not (
                completed.get("status") == "COMPLETE"
                and completed.get("source") == file_record(source)
                and completed.get("runtime_frame")
                    == file_record(runtime_frame_path)
            ):
                raise DETError(
                    f"sup fixture {value['session_id']} marker binding drifted")
            completed_row_paths.append(_validate_file_record(
                completed["rows"], f"sup fixture {value['session_id']}"))
        rows_path = stage_dir / "sup_rows.jsonl"
        combined = b"".join(path.read_bytes() for path in completed_row_paths)
        _write_bytes_exclusive_or_verify(rows_path, combined)
        rows = read_jsonl(rows_path)
        if len(rows) != 10:
            raise DETError(
                f"sup eval expected 10 paired rows, observed {len(rows)}")
        receipt.update({
            "row_count": len(rows),
            "rows": file_record(rows_path),
            "segment_rows": file_record(fixture_rows_path),
            "all_fixture_rows": [file_record(path) for path in completed_row_paths],
        })
    write_json_exclusive(marker, receipt)


def _require_g1_pass(run_dir: Path, runtime_frame_path: Path) -> Path:
    return _finalize_g1(run_dir, runtime_frame_path)


def _validated_thresholds(run_dir: Path, runtime_frame_path: Path) -> Path:
    path = _one(run_dir, "thresholds_*.json")
    value = read_json(path)
    if value.get("frozen_before_eval") is not True:
        raise DETError("threshold receipt is not frozen-before-eval")
    if value.get("runtime_frame") != file_record(runtime_frame_path):
        raise DETError("threshold receipt is bound to a different runtime frame")
    if value.get("registration") != file_record(_registration(run_dir)):
        raise DETError("threshold receipt is bound to a different registration")
    _validate_file_record(value["calibration_rows"], "threshold calibration rows")
    return path


def child_stage(
    run_dir: Path,
    stage: str,
    *,
    lease_seconds: int,
    wait_seconds: int,
) -> None:
    registration_path = _registration(run_dir)
    visible = _validate_visibility()
    runtime_frame_path = _resolve_runtime_frame(
        run_dir,
        lease_seconds=int(lease_seconds),
        wait_seconds=int(wait_seconds),
        visible_device=visible,
    )
    os.environ.update(_runtime_env(read_json(runtime_frame_path)))
    if stage == "g1_baseline":
        _g1_leg(
            run_dir, "baseline", runtime_frame_path,
            lease_seconds=int(lease_seconds))
    elif stage == "g1_off":
        if not (run_dir / "g1" / "baseline" / "stage_complete.json").is_file():
            raise DETError("G1 off leg opened before baseline completed")
        _g1_leg(
            run_dir, "off", runtime_frame_path,
            lease_seconds=int(lease_seconds))
    elif stage == "calibration":
        _require_g1_pass(run_dir, runtime_frame_path)
        _e2e_stage(
            run_dir, segment_name="calibration",
            registration_path=registration_path,
            runtime_frame_path=runtime_frame_path,
            lease_seconds=int(lease_seconds))
    elif stage in (
        "eval_e2e_1", "eval_e2e_2", "eval_e2e_3", "eval_e2e_4",
    ):
        _require_g1_pass(run_dir, runtime_frame_path)
        _validated_thresholds(run_dir, runtime_frame_path)
        _e2e_stage(
            run_dir, segment_name=stage,
            registration_path=registration_path,
            runtime_frame_path=runtime_frame_path,
            lease_seconds=int(lease_seconds))
    elif stage.startswith("eval_sup_"):
        _require_g1_pass(run_dir, runtime_frame_path)
        _validated_thresholds(run_dir, runtime_frame_path)
        if not (run_dir / "eval" / "e2e_segment_4_complete.json").is_file():
            raise DETError("sup eval opened before E2E eval completed")
        segment_index = int(stage.rsplit("_", 1)[1])
        if segment_index > 1:
            prior = run_dir / "eval" / f"sup_segment_{segment_index - 1}_complete.json"
            if not prior.is_file():
                raise DETError(
                    f"eval_sup_{segment_index} opened before prior sup segment")
        _sup_stage(
            run_dir,
            segment_index=segment_index,
            registration_path=registration_path,
            runtime_frame_path=runtime_frame_path,
            lease_seconds=int(lease_seconds))
    else:
        raise DETError(f"unknown child stage {stage}")


def _stage_complete(run_dir: Path, stage: str) -> bool:
    paths = {
        "g1_baseline": run_dir / "g1" / "baseline" / "stage_complete.json",
        "g1_off": run_dir / "g1" / "off" / "stage_complete.json",
        "calibration": run_dir / "calibration" / "e2e_stage_complete.json",
        "eval_e2e_1": run_dir / "eval" / "e2e_segment_1_complete.json",
        "eval_e2e_2": run_dir / "eval" / "e2e_segment_2_complete.json",
        "eval_e2e_3": run_dir / "eval" / "e2e_segment_3_complete.json",
        "eval_e2e_4": run_dir / "eval" / "e2e_segment_4_complete.json",
        "eval_sup_1": run_dir / "eval" / "sup_segment_1_complete.json",
        "eval_sup_2": run_dir / "eval" / "sup_segment_2_complete.json",
        "eval_sup_3": run_dir / "eval" / "sup_segment_3_complete.json",
        "eval_sup_4": run_dir / "eval" / "sup_stage_complete.json",
    }
    return paths[stage].is_file()


def _run_stage_parent(
    run_dir: Path,
    stage: str,
    *,
    lease_seconds: int,
    wait_seconds: int,
) -> None:
    if _stage_complete(run_dir, stage):
        print(f"stage={stage} status=already_complete", flush=True)
        return
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--run-dir", str(run_dir), "--child-stage", stage,
        "--lease-seconds", str(lease_seconds),
        "--lock-wait-seconds", str(wait_seconds),
    ]
    env = _base_env()
    started = time.monotonic()
    returncode = _run_process(
        command, env, int(wait_seconds) + int(lease_seconds) + 60)
    if returncode != 0:
        raise DETError(f"GPU child stage {stage} returned {returncode}")
    print(f"stage={stage} completed_s={time.monotonic() - started:.3f}", flush=True)


def _fit_thresholds(run_dir: Path) -> None:
    if sorted(run_dir.glob("thresholds_*.json")):
        runtime_frame_path = _one(run_dir, "runtime_frame_*.json")
        _validated_thresholds(run_dir, runtime_frame_path)
        print("thresholds=status_already_frozen", flush=True)
        return
    command = [
        sys.executable, str(ANALYZE), "fit", "--run-dir", str(run_dir),
    ]
    returncode = subprocess.run(command, cwd=ROOT, env=_base_env(), check=False).returncode
    if returncode != 0:
        raise DETError(f"CPU threshold fit returned {returncode}")


def run_parent(
    run_dir: Path,
    requested: str,
    *,
    lease_seconds: int,
    wait_seconds: int,
) -> None:
    stages = list(STAGE_ORDER) if requested == "all" else [requested]
    previous_ran = False
    for stage in stages:
        if previous_ran:
            print(f"inter_stage_gpu_gap_s={GAP_SECONDS}", flush=True)
            time.sleep(GAP_SECONDS)
        _run_stage_parent(
            run_dir, stage,
            lease_seconds=int(lease_seconds),
            wait_seconds=int(wait_seconds),
        )
        previous_ran = True
        if stage == "g1_off":
            runtime_frame_path = _one(run_dir, "runtime_frame_*.json")
            path = _finalize_g1(run_dir, runtime_frame_path)
            print(f"det_g1_receipt={path}", flush=True)
        if stage == "calibration":
            _fit_thresholds(run_dir)
    if requested == "all":
        command = [
            sys.executable, str(ANALYZE), "report", "--run-dir", str(run_dir),
        ]
        returncode = subprocess.run(
            command, cwd=ROOT, env=_base_env(), check=False).returncode
        if returncode != 0:
            raise DETError(f"CPU final report returned {returncode}")


def selftest() -> dict[str, Any]:
    lease_guard = False
    try:
        with gpu_lease(MAX_LEASE_SECONDS + 1, 0):
            pass
    except DETError:
        lease_guard = True
    if not lease_guard:
        raise DETError("GPU lease hard-cap selftest failed")
    from scripts.grm_det1_register import _fixtures

    fixtures = _fixtures()
    e2e = [row for row in fixtures if row["source_family"] == "certified_34_turn"]
    for name, spec in E2E_SEGMENTS.items():
        split = str(spec["detector_stage"])
        variants = 1 if split == "calibration" else 2
        expected = variants * sum(
            row["split"] == split
            and int(row["turn"]) < int(spec["stop_after_turns"])
            for row in e2e
        )
        if expected != int(spec["expected_rows"]):
            raise DETError(
                f"{name} row layout mismatch: expected {expected}, "
                f"spec={spec['expected_rows']}")
    sup_sources = sorted(
        (ROOT / "tests" / "fixtures" / "supersession_battery").glob("*.json"))
    if len(sup_sources) != 4 or sum(
        len(json.loads(path.read_text(encoding="utf-8"))["probes"])
        for path in sup_sources
    ) != 5:
        raise DETError("supersession stage layout is not four fixtures/five probes")

    class _FakeArena:
        def __init__(self):
            self.grafts = []
            self.epoch_bumps = 0

        def deposit(self, text):
            self.grafts.append({"text": str(text)})
            return len(self.grafts) - 1

        def _bump_cuda_gqa_epoch(self):
            self.epoch_bumps += 1

    class _FakeRepo:
        def __init__(self):
            self.arena = _FakeArena()
            self.synced = []

        def _native_sync_node(self, index):
            self.synced.append(int(index))

    fake_repo = _FakeRepo()
    fixture = json.loads(sup_sources[0].read_text(encoding="utf-8"))
    _install_fixture_nodes(fake_repo, fixture)
    if fake_repo.synced != list(range(len(fixture["nodes"]))) or not all(
        node["metadata"]["active"] is True for node in fake_repo.arena.grafts
    ):
        raise DETError("fixture native identity sync selftest failed")
    return {
        "schema": "grm.det1.gpu_selftest.v1",
        "status": "PASS",
        "checks": {
            "lease_cap_guard": "PASS",
            "stage_order": list(STAGE_ORDER),
            "gap_seconds": GAP_SECONDS,
            "registered_stage_row_layout": "PASS",
            "sup_fixture_shards": "4 fixtures / 5 probes",
            "sup_native_identity_sync_active_revisions": "PASS",
        },
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--stage", choices=("all", *STAGE_ORDER), default="all")
    parser.add_argument("--child-stage", choices=STAGE_ORDER)
    parser.add_argument("--lease-seconds", type=int, default=DEFAULT_LEASE_SECONDS)
    parser.add_argument("--lock-wait-seconds", type=int, default=7200)
    parser.add_argument("--selftest", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.selftest:
        print(json.dumps(selftest(), sort_keys=True))
        return 0
    if args.run_dir is None:
        raise DETError("--run-dir is required")
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise DETError(f"run directory does not exist: {run_dir}")
    artifact_root = (ROOT / "artifacts" / "grm_det1").resolve()
    if not run_dir.is_relative_to(artifact_root):
        raise DETError(
            f"ORDER GRM-DET1 confines run artifacts to {artifact_root}: {run_dir}")
    if args.child_stage:
        with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
            child_stage(
                run_dir,
                str(args.child_stage),
                lease_seconds=int(args.lease_seconds),
                wait_seconds=int(args.lock_wait_seconds),
            )
    else:
        run_parent(
            run_dir, str(args.stage),
            lease_seconds=int(args.lease_seconds),
            wait_seconds=int(args.lock_wait_seconds),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
