#!/usr/bin/env python3
"""Registered GRM-SUP-L2-ON acceptance and queued GPU runner.

With no flags this performs CPU-only validation of the frozen historical
anchors.  ``--run-gpu`` acquires ``/tmp/forge-gpu.lock`` for each GPU-bearing
leg, runs the no-flag default and ``GRM_SUP_RESOLVE=0`` escape batteries, then
runs the shared and full GRM pytest suites.  Every output goes to a new,
timestamped directory; existing artifacts are never overwritten.

The July receipts predate the later ``dialect`` receipt-envelope field.  The
registered *transcript* hash therefore canonicalizes every probe row after
removing only envelope keys (schema, record_type, dialect, mode).  Generated
answer bytes, classification, ranks, mounts, arena info, fixture identity, and
all other probe fields remain covered byte-for-byte.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "grm_supersession"
LEGACY_ANCHOR = ARTIFACT_ROOT / "g0_baseline.jsonl"
L2_ANCHOR = ARTIFACT_ROOT / "diag_resolve_only.jsonl"
L2_REPEAT = ARTIFACT_ROOT / "postmerge_resolve.jsonl"
GPU_LOCK = Path("/tmp/forge-gpu.lock")
MAX_LEASE_SECONDS = 590

LEGACY_RECEIPT_SHA256 = (
    "7bc14aaf1eb58a61b6df12586341f409f1768a537b5910b17dd67df9541e8ccd"
)
L2_RECEIPT_SHA256 = (
    "8053f9c3437bb6ff4c0e81533159b2e5870c32ca20f895e799ca61f8585d2a78"
)
EXPECTED_CHANGED_PROBES = frozenset(
    ("harbor_restatement", "lumen_head", "orion_current")
)
ENVELOPE_KEYS = frozenset(("schema", "record_type", "dialect", "mode"))


class GateError(RuntimeError):
    pass


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise GateError(f"{path}:{line_no}: JSONL row is not an object")
            rows.append(value)
    return rows


def probe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row.get("record_type") == "supersession_probe_receipt"
    ]


def transcript_projection_bytes(
    rows: list[dict[str, Any]], *, scenario: str | None = None,
) -> bytes:
    projected = []
    for row in probe_rows(rows):
        if scenario is not None and row.get("scenario") != scenario:
            continue
        projected.append({
            key: value for key, value in row.items() if key not in ENVELOPE_KEYS
        })
    if not projected:
        raise GateError(f"no probe rows found for scenario={scenario!r}")
    return b"".join(
        json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8") + b"\n"
        for row in projected
    )


def summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hits = [
        row for row in rows
        if row.get("record_type") == "supersession_battery_summary"
    ]
    if len(hits) != 1:
        raise GateError(f"expected one battery summary, found {len(hits)}")
    return hits[0]


def changed_probe_fields(
    legacy: list[dict[str, Any]], l2: list[dict[str, Any]],
) -> dict[str, list[str]]:
    def indexed(rows):
        return {
            str(row["probe_id"]): {
                key: value for key, value in row.items()
                if key not in ENVELOPE_KEYS
            }
            for row in probe_rows(rows)
        }

    before, after = indexed(legacy), indexed(l2)
    if before.keys() != after.keys():
        raise GateError("legacy and L2 anchors cover different probe IDs")
    changed = {}
    for probe_id in before:
        keys = sorted({
            key for key in before[probe_id].keys() | after[probe_id].keys()
            if before[probe_id].get(key) != after[probe_id].get(key)
        })
        if keys:
            changed[probe_id] = keys
    return changed


def coordination_pin_acceptance() -> dict[str, Any]:
    """Require every in-flight CMC/ADM ArenaCache frame to pin L1/L2."""
    sources = {
        "cmc1_1": ROOT / "scripts" / "grm_cmc1_gpu_arms.py",
        "adm1": ROOT / "scripts" / "grm_adm1_gpu.py",
    }
    result = {}
    trees = {}
    for label, path in sources.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        trees[label] = tree
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ArenaCache"
        ]
        if not calls:
            raise GateError(f"{label} has no auditable ArenaCache constructor")
        pinned_lines = []
        revision_resolution_pins = []
        for call in calls:
            keywords = {item.arg: item.value for item in call.keywords if item.arg}
            for key in ("length_debias", "revision_resolution"):
                value = keywords.get(key)
                if not (
                    isinstance(value, ast.Constant)
                    and (
                        value.value is False
                        or (key == "revision_resolution" and value.value is True)
                    )
                ):
                    required = "False" if key == "length_debias" else "True or False"
                    raise GateError(
                        f"{path}:{call.lineno}: {key} is not explicitly {required}"
                    )
                if key == "revision_resolution":
                    revision_resolution_pins.append(bool(value.value))
            pinned_lines.append(int(call.lineno))
        result[label] = {
            "path": str(path.relative_to(ROOT)),
            "arena_constructor_lines": pinned_lines,
            "length_debias": False,
            "revision_resolution": (
                revision_resolution_pins[0]
                if len(set(revision_resolution_pins)) == 1
                else revision_resolution_pins
            ),
        }

    adm_e2e_pins = [
        item.value
        for function in trees["adm1"].body
        if isinstance(function, ast.FunctionDef)
        and function.name == "run_e2e_frame"
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.List)
        and any(
            isinstance(target, ast.Name) and target.id == "command"
            for target in node.targets
        )
        for item in node.value.elts
        if isinstance(item, ast.Constant)
        and item.value in ("--no-sup-resolve", "--sup-resolve")
    ]
    if len(adm_e2e_pins) != 1:
        raise GateError(
            "ADM1 e2e subprocess does not have exactly one explicit L2 pin"
        )
    result["adm1"]["e2e_cli"] = adm_e2e_pins[0]
    return result


def anchor_acceptance() -> dict[str, Any]:
    hashes = {
        "legacy_receipt": sha256_file(LEGACY_ANCHOR),
        "l2_receipt": sha256_file(L2_ANCHOR),
        "l2_repeat_receipt": sha256_file(L2_REPEAT),
    }
    if hashes["legacy_receipt"] != LEGACY_RECEIPT_SHA256:
        raise GateError("registered legacy receipt hash drifted")
    if hashes["l2_receipt"] != L2_RECEIPT_SHA256:
        raise GateError("registered resolve-only receipt hash drifted")
    if hashes["l2_repeat_receipt"] != L2_RECEIPT_SHA256:
        raise GateError("post-merge resolve repeat is not byte-identical")

    legacy = read_jsonl(LEGACY_ANCHOR)
    l2 = read_jsonl(L2_ANCHOR)
    baseline_summary = summary_row(legacy)
    default_summary = summary_row(l2)
    baseline_counts = baseline_summary["classification_counts"]
    default_counts = default_summary["classification_counts"]
    fresh = default_summary["threshold_registration_inputs"]

    if default_counts.get("stale") != 0:
        raise GateError(f"L2 stale count is not zero: {default_counts}")
    if default_counts.get("wrong-fact", 0) > baseline_counts.get("wrong-fact", 0):
        raise GateError("L2 wrong-fact count regressed above baseline")
    if fresh.get("fresh_control_correct", 0) < 1 or fresh.get(
        "fresh_control_total") != 2:
        raise GateError(f"fresh-control floor failed: {fresh}")

    changes = changed_probe_fields(legacy, l2)
    if frozenset(changes) != EXPECTED_CHANGED_PROBES:
        raise GateError(
            "non-supersession transcript drift: "
            f"expected {sorted(EXPECTED_CHANGED_PROBES)}, got {sorted(changes)}"
        )

    legacy_fresh = transcript_projection_bytes(
        legacy, scenario="fresh_fact_controls")
    l2_fresh = transcript_projection_bytes(l2, scenario="fresh_fact_controls")
    if legacy_fresh != l2_fresh:
        raise GateError("L2 changed a non-lineage fresh-control transcript")

    return {
        "status": "PASS",
        "coordination_pins": coordination_pin_acceptance(),
        "registered_anchors": hashes,
        "transcript_projection": {
            "legacy_sha256": sha256_bytes(transcript_projection_bytes(legacy)),
            "l2_default_sha256": sha256_bytes(transcript_projection_bytes(l2)),
            "fresh_controls_sha256": sha256_bytes(l2_fresh),
            "envelope_keys_excluded": sorted(ENVELOPE_KEYS),
        },
        "classification_counts": {
            "legacy": baseline_counts,
            "l2_default": default_counts,
        },
        "fresh_controls": {
            "correct": fresh["fresh_control_correct"],
            "total": fresh["fresh_control_total"],
            "legacy_projection_byte_identical": True,
        },
        "changed_probes": changes,
    }


def acquire_lock(handle, wait_seconds: int) -> float:
    started = time.monotonic()
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return time.monotonic() - started
        except BlockingIOError:
            if time.monotonic() - started >= wait_seconds:
                raise GateError(
                    f"GPU lock unavailable after {wait_seconds}s: {GPU_LOCK}"
                )
            time.sleep(5)


def run_locked(
    name: str,
    argv: list[str],
    *,
    run_dir: Path,
    env: dict[str, str],
    wait_seconds: int,
    command_timeout: int,
) -> dict[str, Any]:
    log_path = run_dir / f"{name}.log"
    lock_handle = GPU_LOCK.open("a+")
    waited = acquire_lock(lock_handle, wait_seconds)
    started = time.monotonic()
    try:
        with log_path.open("xb") as log:
            completed = subprocess.run(
                argv,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=command_timeout,
                check=False,
            )
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise GateError(
            f"{name} failed rc={completed.returncode}; see {log_path}"
        )
    return {
        "name": name,
        "argv": argv,
        "waited_for_lock_s": waited,
        "elapsed_s": elapsed,
        "log": str(log_path.relative_to(ROOT)),
        "log_sha256": sha256_file(log_path),
    }


def gpu_acceptance(args: argparse.Namespace) -> dict[str, Any]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = args.artifact_root / f"l2_default_on_{stamp}_{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    prior_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + prior_pythonpath if prior_pythonpath else ""
    )

    legs = []
    default_receipt = run_dir / "default_on.jsonl"
    default_env = env.copy()
    default_env.pop("GRM_SUP_RESOLVE", None)
    legs.append(run_locked(
        "battery_default_on",
        [
            sys.executable,
            "tests/test_grm_supersession_battery.py",
            "--run-gpu",
            "--receipt-jsonl", str(default_receipt),
        ],
        run_dir=run_dir,
        env=default_env,
        wait_seconds=args.wait_seconds,
        command_timeout=args.command_timeout,
    ))

    escape_receipt = run_dir / "escape_off.jsonl"
    escape_env = env.copy()
    escape_env["GRM_SUP_RESOLVE"] = "0"
    legs.append(run_locked(
        "battery_escape_off",
        [
            sys.executable,
            "tests/test_grm_supersession_battery.py",
            "--run-gpu",
            "--receipt-jsonl", str(escape_receipt),
        ],
        run_dir=run_dir,
        env=escape_env,
        wait_seconds=args.wait_seconds,
        command_timeout=args.command_timeout,
    ))

    anchor = anchor_acceptance()
    live_default = read_jsonl(default_receipt)
    live_escape = read_jsonl(escape_receipt)
    default_projection = transcript_projection_bytes(live_default)
    escape_projection = transcript_projection_bytes(live_escape)
    if sha256_bytes(default_projection) != anchor[
        "transcript_projection"
    ]["l2_default_sha256"]:
        raise GateError("new no-env default transcript differs from L2 anchor")
    if sha256_bytes(escape_projection) != anchor[
        "transcript_projection"
    ]["legacy_sha256"]:
        raise GateError("GRM_SUP_RESOLVE=0 transcript differs from legacy anchor")

    shared = [
        "tests/test_grm_supersession_battery.py",
        "tests/test_grm_importance_telemetry.py",
        "tests/test_grm_fold_recovered_guard.py",
    ]
    legs.append(run_locked(
        "shared_suite",
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *shared],
        run_dir=run_dir,
        env=default_env,
        wait_seconds=args.wait_seconds,
        command_timeout=args.command_timeout,
    ))

    full_suite = sorted(str(path.relative_to(ROOT))
                        for path in ROOT.glob("tests/test_grm_*.py"))
    full_suite.append("tests/test_gqa_ragged_cuda_bank.py")
    legs.append(run_locked(
        "full_grm_suite",
        [
            sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            *full_suite,
        ],
        run_dir=run_dir,
        env=default_env,
        wait_seconds=args.wait_seconds,
        command_timeout=args.command_timeout,
    ))

    result = {
        "schema": "grm_sup_l2_default_on_gate_v1",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.relative_to(ROOT)),
        "anchor_acceptance": anchor,
        "live_receipts": {
            "default_on": {
                "path": str(default_receipt.relative_to(ROOT)),
                "receipt_sha256": sha256_file(default_receipt),
                "transcript_sha256": sha256_bytes(default_projection),
                "summary": summary_row(live_default),
            },
            "escape_off": {
                "path": str(escape_receipt.relative_to(ROOT)),
                "receipt_sha256": sha256_file(escape_receipt),
                "transcript_sha256": sha256_bytes(escape_projection),
                "summary": summary_row(live_escape),
            },
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                ROOT / "core" / "grm_supersession.py",
                ROOT / "core" / "graft_arena.py",
                ROOT / "tests" / "test_grm_supersession_battery.py",
                ROOT / "scripts" / "grm_cmc1_gpu_arms.py",
                ROOT / "scripts" / "grm_adm1_gpu.py",
                Path(__file__).resolve(),
            )
        },
        "legs": legs,
    }
    result_path = run_dir / "gate_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-gpu", action="store_true",
        help="run live batteries and both registered pytest suites behind flock",
    )
    parser.add_argument("--wait-seconds", type=int, default=21600)
    parser.add_argument("--command-timeout", type=int, default=580)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not (1 <= int(args.command_timeout) <= MAX_LEASE_SECONDS):
        raise GateError(
            f"command timeout must be in 1..{MAX_LEASE_SECONDS}s"
        )
    result = gpu_acceptance(args) if args.run_gpu else anchor_acceptance()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
