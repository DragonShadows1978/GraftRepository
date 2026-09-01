#!/usr/bin/env python3
"""Fail-closed DET1.5 race campaign on the DET1.4 fork substrate.

The public commands are resumable parents.  GPU work is always performed by
fresh child processes while the parent owns the standing bounded GPU lease.
The private ``_worker`` command is deliberately absent from the lead-facing
inventory and must never be used as a campaign entry point.

This file does not reinterpret the registered DET1 prediction.  It validates
the immutable base registration, the successful DET1.4 zero fork gate, the
later race authorization, and the DET1.7 lived-admission plant registry before
any detector stage may advance.  Historical reconstructed DET1 rows are never
inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import datetime as dt
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_det1_common import (  # noqa: E402
    DETECTORS,
    DETError,
    MECHANISTIC,
    VERBAL_QUESTION,
    canonical_json_bytes,
    append_jsonl_once,
    contains_value,
    file_record,
    fit_thresholds,
    normalize_value_text,
    race_metrics,
    read_json,
    read_jsonl,
    utc_now,
    write_content_addressed,
    write_json_exclusive,
)


FROZEN_RUN = ROOT / "artifacts/grm_det1/run_20260831T160525Z_2"
ORDER = ROOT / "orders/GRM_DET1_5_RACE_CAMPAIGN.md"
DET1_ORDER = ROOT / "orders/GRM_DET1_DEMAND_DETECTOR_RACE.md"
DET1_6_ORDER = ROOT / "orders/GRM_DET1_6_PLANT_ON_FORK.md"
DET1_7_ORDER = ROOT / "orders/GRM_DET1_7_PLANT_REALIGN.md"
REGISTRATION = FROZEN_RUN / "registration_62cb6c09cbec211d.json"
RUNTIME_FRAME = FROZEN_RUN / "runtime_frame_28b3196f8fb04a41.json"
PRIOR_AMENDMENT = FROZEN_RUN / "det1_4_source_amendment.json"
RACE_AMENDMENT = FROZEN_RUN / "det1_5_race_authorization_amendment.json"
DELTA_AMENDMENT = FROZEN_RUN / "det1_6_fork_hydration_delta_amendment.json"
DET1_7_SOURCE_AUTH = (
    FROZEN_RUN / "det1_7_precollection_source_authorization_r2.json"
)
DET1_7_TERMINAL_AMENDMENT = FROZEN_RUN / "det1_7_plant_registry_amendment.json"
PARENT_ZERO_MARKER = FROZEN_RUN / "det1_4/zero_gate/stage_complete.json"

CAMPAIGN_RELATIVE = Path("det1_4/campaign")
LEASE_SECONDS = 580
MAX_LEASE_SECONDS = 590
GAP_SECONDS = 30
DEFAULT_WAIT_SECONDS = 7200

STAGE_ORDER = (
    "cross_process_zero",
    "plant_registration",
    "g0",
    "g1",
    "calibration",
    "eval_mechanistic",
    "eval_verbal",
    "analyze",
)
GPU_STAGES = frozenset(STAGE_ORDER[:-1])
STAGE_DIRS = {
    "cross_process_zero": Path("cross_process_zero"),
    "plant_registration": Path("det1_7/plant_registration"),
    "g0": Path("g0"),
    "g1": Path("g1"),
    "calibration": Path("calibration"),
    "eval_mechanistic": Path("eval"),
    "eval_verbal": Path("eval"),
    "analyze": Path("analysis"),
}
STAGE_MARKERS = {
    "cross_process_zero": Path("cross_process_zero/stage_complete.json"),
    "plant_registration": Path("det1_7/plant_registration/stage_complete.json"),
    "g0": Path("g0/stage_complete.json"),
    "g1": Path("g1/stage_complete.json"),
    "calibration": Path("calibration/stage_complete.json"),
    "eval_mechanistic": Path("eval/mechanistic_stage_complete.json"),
    "eval_verbal": Path("eval/verbal_stage_complete.json"),
    "analyze": Path("analysis/stage_complete.json"),
}
STAGE_SHARD_DIRS = {
    "plant_registration": Path("det1_7/plant_registration/shards"),
    "g0": Path("g0/shards"),
    "g1": Path("g1/shards"),
    "calibration": Path("calibration/shards"),
    "eval_mechanistic": Path("eval/mechanistic_shards"),
    "eval_verbal": Path("eval/verbal_shards"),
}
STAGE_SPECS = {
    "plant_registration": (
        "e2e-cal", "e2e-1", "e2e-2", "e2e-3", "e2e-4",
        "sup-1", "sup-2", "sup-3", "sup-4",
    ),
    "g0": (
        "e2e-1", "e2e-2", "e2e-3", "e2e-4",
        "sup-1", "sup-2", "sup-3", "sup-4",
    ),
    "g1": ("control", "all-hooks"),
    "calibration": ("e2e-cal",),
    "eval_mechanistic": (
        "e2e-1", "e2e-2", "e2e-3", "e2e-4",
        "sup-1", "sup-2", "sup-3", "sup-4",
    ),
    "eval_verbal": (
        "e2e-1", "e2e-2", "e2e-3", "e2e-4",
        "sup-1", "sup-2", "sup-3", "sup-4",
    ),
}
STAGE_SCHEMAS = {
    "cross_process_zero": "grm.det1_5.cross_process_zero.v1",
    "plant_registration": "grm.det1_7.plant_registration.v1",
    "g0": "grm.det1_5.g0.v1",
    "g1": "grm.det1_5.g1.v1",
    "calibration": "grm.det1_5.calibration.v1",
    "eval_mechanistic": "grm.det1_5.mechanistic_eval.v1",
    "eval_verbal": "grm.det1_5.chronological_verbal.v1",
    "analyze": "grm.det1_5.analysis.v1",
}

ZERO_PASS_STATUS = "PASS_EXACT_CANONICAL_VALUE_BYTES"
PLANTED_INTERVENTION = "REGISTERED_PLANTED_MISS_WITHHOLDING"
DELTA_RECEIPT_SCHEMA = "grm.det1_6.fork_hydration_delta.v1"
DELTA_PASS_STATUS = "PASS_EXACT_REGISTERED_DELTA_CANONICAL_VALUE_BYTES"
DELTA_AMENDMENT_SCHEMA = "grm.det1_6.fork_hydration_delta_amendment.v1"
DELTA_AMENDMENT_STATUS = "AUTHORIZED_FORK_HYDRATION_DELTA_SOURCE_REBINDING"
RACE_AMENDMENT_SCHEMA = "grm.det1_5.race_authorization_amendment.v1"
RACE_AMENDMENT_STATUS = "AUTHORIZED_RACE_ON_FORKED_SUBSTRATE"
DET1_7_SOURCE_AUTH_SCHEMA = "grm.det1_7.precollection_source_authorization.v1"
DET1_7_SOURCE_AUTH_STATUS = "AUTHORIZED_LIVED_PLANT_REGISTRY_COLLECTION_ONLY"
DET1_7_TERMINAL_SCHEMA = "grm.det1_7.plant_registry_amendment.v1"
DET1_7_TERMINAL_STATUS = "AUTHORIZED_RACE_WITH_LIVED_PLANT_REGISTRY"
REGISTRATION_QUALIFICATION = "LIVED_TARGET_QUALIFICATION_ABLATION"

DET1_6_CHANGED_SOURCES = (
    "scripts/grm_det1_3_snapshot.py",
    "scripts/grm_det1_4_gpu.py",
    "scripts/grm_det1_5_analyze.py",
    "scripts/grm_det1_5_gpu.py",
    "scripts/grm_det1_5_workers.py",
    "tests/test_grm_det1_3_snapshot.py",
    "tests/test_grm_det1_5_campaign.py",
)
DET1_6_CHANGE_PURPOSES = {
    "scripts/grm_det1_3_snapshot.py": (
        "fork_hydration_delta_and_exact_snapshot_comparator"
    ),
    "scripts/grm_det1_4_gpu.py": "append_only_source_supersession_plumbing",
    "scripts/grm_det1_5_analyze.py": "terminal_amendment_evidence_validation",
    "scripts/grm_det1_5_gpu.py": "campaign_delta_application_and_receipt_gate",
    "scripts/grm_det1_5_workers.py": "terminal_amendment_shard_provenance",
    "tests/test_grm_det1_3_snapshot.py": "delta_comparator_cpu_contracts",
    "tests/test_grm_det1_5_campaign.py": "campaign_receipt_cpu_contracts",
}

DET1_7_CHANGED_SOURCES = {
    "scripts/grm_det1_5_analyze.py": "bind_analysis_to_lived_plant_registry",
    "scripts/grm_det1_5_gpu.py": "collect_freeze_and_apply_lived_plant_targets",
    "scripts/grm_det1_5_lead.sh": "run_registration_before_detector_campaign",
    "scripts/grm_det1_5_workers.py": "served_only_registration_workers_and_registry_provenance",
    "tests/test_grm_det1_5_campaign.py": "campaign_registration_and_target_binding_contracts",
}
DET1_7_ADDED_SOURCES = {
    "scripts/grm_det1_7_registry.py": "pure_lived_admission_registry_derivation",
    "scripts/grm_det1_7_source_auth.py": "append_only_precollection_source_authorization",
    "tests/test_grm_det1_7_registry.py": "plant_registry_cpu_contracts",
    "tests/test_grm_det1_7_source_auth.py": "source_authorization_cpu_contracts",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DETError(message)


def _plant_registry_entry(
    registry: Mapping[str, Any], base_fixture_id: str,
) -> dict[str, Any]:
    entries = registry.get("entries") or ()
    matches = [
        dict(value) for value in entries
        if isinstance(value, Mapping)
        and value.get("fixture_id") == str(base_fixture_id)
    ]
    _require(len(matches) == 1,
             f"plant registry lacks exactly one entry for {base_fixture_id}")
    return matches[0]


def effective_fixture_for_base(
    registry: Mapping[str, Any], base_fixture: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a base slot to its explicitly registered effective fixture."""
    entry = _plant_registry_entry(registry, str(base_fixture["fixture_id"]))
    effective_id = str(entry.get("effective_fixture_id", ""))
    if effective_id == str(base_fixture["fixture_id"]):
        return dict(base_fixture)
    effective = entry.get("effective_fixture")
    _require(isinstance(effective, Mapping),
             f"substituted entry lacks effective fixture: {base_fixture['fixture_id']}")
    projected = dict(effective)
    _require(projected.get("fixture_id") == effective_id,
             f"effective fixture identity drift: {base_fixture['fixture_id']}")
    selector = projected.get("selector")
    _require(isinstance(selector, Mapping) and bool(selector),
             f"effective fixture selector is absent: {base_fixture['fixture_id']}")
    source_family = str(projected.get("source_family", ""))
    if source_family == "certified_34_turn":
        _require(set(selector) == {"turn"}
                 and isinstance(selector.get("turn"), int)
                 and not isinstance(selector.get("turn"), bool),
                 f"effective E2E selector is invalid: {base_fixture['fixture_id']}")
        projected["turn"] = int(selector["turn"])
    elif source_family == "supersession_battery_on_gpt_oss":
        _require(set(selector) == {"probe_id"}
                 and isinstance(selector.get("probe_id"), str)
                 and bool(selector.get("probe_id")),
                 "effective supersession selector is invalid: "
                 f"{base_fixture['fixture_id']}")
        projected["probe_id"] = str(selector["probe_id"])
    else:
        raise DETError(
            f"effective fixture has unknown source family: {source_family!r}")
    projected["base_fixture_id"] = str(base_fixture["fixture_id"])
    projected["substitution"] = dict(entry.get("substitution") or {})
    return projected


def bind_plant_profile(
    profile: Mapping[str, Any],
    entry: Mapping[str, Any],
    registry_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind runtime diagnostics to the pre-eval lived-admission target."""
    aliases = [int(value) for value in entry.get("alias_ids") or ()]
    target = int(entry.get("selected_target_id", -1))
    actual = [int(value) for value in
              entry.get("actual_authoritative_mounts") or ()]
    _require(bool(aliases) and target in aliases,
             "plant registry target/alias unit is empty or inconsistent")
    _require(set(aliases).issubset(actual),
             "plant registry names aliases without lived source seats")
    bound = dict(profile)
    bound.update({
        "logical_router_rank1": target,
        "production_admitted_rank1": target,
        "router_rank1_admitted": True,
        "logical_alias_ids": aliases,
        "target_contains_expected": bool(
            (entry.get("expected_value_coverage") or {}).get(str(target))),
        "plant_target_source": "DET1_7_LIVED_ADMISSION_REGISTRY",
        "plant_registry": dict(registry_record),
        "plant_entry_sha256": hashlib.sha256(
            canonical_json_bytes(dict(entry))).hexdigest(),
        "plant_entry": dict(entry),
    })
    return bound


def plant_registry_path(run_dir: Path = FROZEN_RUN) -> Path:
    directory = campaign_root(run_dir) / STAGE_DIRS["plant_registration"]
    paths = sorted(directory.glob("plant_registry_*.json"))
    _require(len(paths) == 1,
             f"expected exactly one frozen DET1.7 plant registry, got {len(paths)}")
    return paths[0]


def campaign_root(run_dir: Path = FROZEN_RUN) -> Path:
    """Return the one fresh campaign namespace for a frozen run."""
    return Path(run_dir).resolve() / CAMPAIGN_RELATIVE


def stage_shard_root(run_dir: Path, stage: str) -> Path:
    _require(stage in STAGE_SHARD_DIRS, f"stage has no shard namespace: {stage}")
    return campaign_root(run_dir) / STAGE_SHARD_DIRS[stage]


def campaign_driver_inventory() -> dict[str, Any]:
    """Return the hash-bound execution inventory without probing CUDA."""
    sources = _amendment_sources()
    return {
        "schema": "grm.det1_5.campaign_driver_inventory.v1",
        "status": "PASS_CPU_ONLY",
        "sources": sources,
        "lead_script": sources["scripts/grm_det1_5_lead.sh"],
        "stage_order": list(STAGE_ORDER),
        "shard_plan": {
            stage: list(specs) for stage, specs in STAGE_SPECS.items()
        },
        "gpu_child_count": sum(len(specs) for specs in STAGE_SPECS.values()),
        "lease_seconds_per_child": LEASE_SECONDS,
        "inter_child_gap_seconds": GAP_SECONDS,
    }


def _path_from_record(record: Mapping[str, Any]) -> Path:
    value = record.get("path")
    _require(isinstance(value, str) and bool(value), "file record lacks path")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _validate_file_record(record: Mapping[str, Any], label: str) -> Path:
    _require(isinstance(record, Mapping), f"{label} is not a file record")
    path = _path_from_record(record)
    _require(path.is_file(), f"{label} path is missing: {path}")
    _require(file_record(path) == dict(record), f"{label} file record drifted")
    return path


def _one(directory: Path, pattern: str) -> Path:
    values = sorted(Path(directory).glob(pattern))
    _require(
        len(values) == 1,
        f"expected exactly one {pattern} under {directory}, got {len(values)}",
    )
    return values[0]


def _write_bytes_exclusive_or_verify(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        _require(path.read_bytes() == payload, f"immutable output collision: {path}")
    return path


def _write_json_exclusive_or_verify(path: Path, value: Mapping[str, Any]) -> Path:
    return _write_bytes_exclusive_or_verify(path, canonical_json_bytes(value))


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    chunks = []
    for row in rows:
        chunks.append(json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8") + b"\n")
    return b"".join(chunks)


def _write_jsonl_exclusive_or_verify(
    path: Path, rows: Sequence[Mapping[str, Any]],
) -> Path:
    return _write_bytes_exclusive_or_verify(path, _jsonl_bytes(rows))


def _process_instance() -> dict[str, Any]:
    from scripts.grm_det1_3_snapshot import process_instance_sha256

    pid = os.getpid()
    stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    close = stat_text.rfind(")")
    _require(close >= 0, "cannot parse current process start ticks")
    start_ticks = int(stat_text[close + 2:].split()[19])
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="utf-8").strip()
    return {
        "process_pid": pid,
        "process_start_ticks": start_ticks,
        "boot_id": boot_id,
        "process_instance_sha256": process_instance_sha256(
            pid, start_ticks, boot_id),
    }


def aggregate_process_instances(values: Sequence[str]) -> str:
    """Digest an ordered, unique process-instance list for stage receipts."""
    normalized = [str(value) for value in values]
    _require(bool(normalized), "process-instance list is empty")
    _require(len(normalized) == len(set(normalized)),
             "process-instance list contains duplicates")
    _require(all(len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    ) for value in normalized), "process-instance list contains a non-hex identity")
    payload = json.dumps(
        normalized, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _record_digest(record: Mapping[str, Any], label: str) -> tuple[int, str]:
    _require(isinstance(record, Mapping), f"{label} is not a byte record")
    size = record.get("bytes")
    digest = record.get("sha256")
    _require(isinstance(size, int) and size >= 0, f"{label}.bytes is invalid")
    _require(
        isinstance(digest, str) and len(digest) == 64,
        f"{label}.sha256 is invalid",
    )
    return int(size), str(digest)


def _zero_process(receipt: Mapping[str, Any]) -> str:
    direct = receipt.get("process_instance_sha256")
    if isinstance(direct, str) and direct:
        return direct
    process = receipt.get("process") or {}
    direct = process.get("process_instance_sha256")
    if isinstance(direct, str) and direct:
        return direct
    fork_process = receipt.get("fork_process") or {}
    direct = fork_process.get("process_instance_sha256")
    if isinstance(direct, str) and direct:
        return direct
    fork_record = receipt.get("fork_snapshot") or {}
    if isinstance(fork_record, Mapping) and fork_record.get("path"):
        manifest_path = _validate_file_record(fork_record, "zero fork snapshot")
        provenance = (read_json(manifest_path).get("provenance") or {})
        direct = provenance.get("process_instance_sha256")
        if isinstance(direct, str) and direct:
            return direct
    raise DETError("zero receipt lacks a verifiable process-instance identity")


def _zero_canonical_binding(receipt: Mapping[str, Any]) -> str:
    for source in (receipt, receipt.get("substrate_comparison") or {},
                   receipt.get("restore") or {}):
        for key in (
            "canonical_state_sha256",
            "canonical_value_state_sha256",
            "source_manifest_sha256",
        ):
            value = source.get(key) if isinstance(source, Mapping) else None
            if isinstance(value, str) and len(value) == 64:
                return value
    lived = receipt.get("lived_snapshot") or {}
    digest = lived.get("sha256")
    _require(
        isinstance(digest, str) and len(digest) == 64,
        "zero receipt lacks a canonical source-state binding",
    )
    return str(digest)


def _validate_zero_receipt(receipt: Mapping[str, Any], label: str) -> dict[str, Any]:
    _require(receipt.get("status") == "PASS", f"{label} status is not PASS")
    _require(receipt.get("gate_pass") is True, f"{label} gate_pass is not true")
    _require(receipt.get("intervention") == "ZERO", f"{label} is not ZERO")
    substrate = receipt.get("substrate_comparison") or {}
    _require(
        substrate.get("status") == ZERO_PASS_STATUS,
        f"{label} substrate status is not {ZERO_PASS_STATUS}",
    )
    _require(substrate.get("gate_pass") is True,
             f"{label} substrate gate_pass is not true")
    answer = receipt.get("answer") or {}
    _require(answer.get("normalized_match") is True,
             f"{label} normalized answer does not match lived")
    normalized = normalize_value_text(str(answer.get("normalized", "")))
    lived_normalized = normalize_value_text(str(answer.get("lived_normalized", "")))
    _require(bool(normalized) and normalized == lived_normalized,
             f"{label} normalized answer bytes differ")
    lived = receipt.get("lived_snapshot") or {}
    _record_digest(lived, f"{label}.lived_snapshot")
    return {
        "process_instance_sha256": _zero_process(receipt),
        "canonical_state_sha256": _zero_canonical_binding(receipt),
        "lived_snapshot": dict(lived),
        "answer_normalized": normalized,
    }


def validate_cross_process_zero(
    parent: Mapping[str, Any], child: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate two independent ZERO forks without weakening byte semantics."""
    left = _validate_zero_receipt(parent, "parent zero")
    right = _validate_zero_receipt(child, "child zero")
    _require(
        left["process_instance_sha256"] != right["process_instance_sha256"],
        "cross-process zero did not use a distinct process instance",
    )
    _require(
        _record_digest(left["lived_snapshot"], "parent lived snapshot")
        == _record_digest(right["lived_snapshot"], "child lived snapshot"),
        "cross-process zero receipts use different lived snapshots",
    )
    _require(
        left["canonical_state_sha256"] == right["canonical_state_sha256"],
        "cross-process canonical state bindings differ",
    )
    _require(
        left["answer_normalized"] == right["answer_normalized"],
        "cross-process normalized answers differ",
    )
    return {
        "schema": STAGE_SCHEMAS["cross_process_zero"],
        "status": "PASS",
        "gate_pass": True,
        "processes_distinct": True,
        "parent_process_instance_sha256": left["process_instance_sha256"],
        "child_process_instance_sha256": right["process_instance_sha256"],
        "canonical_state_sha256": left["canonical_state_sha256"],
        "lived_snapshot_sha256": left["lived_snapshot"]["sha256"],
        "normalized_answer": left["answer_normalized"],
        "byte_semantics": "C_CONTIGUOUS_HOST_VALUE_BYTES",
    }


def _pair_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_pairs: int,
) -> dict[str, dict[str, Mapping[str, Any]]]:
    by_fixture: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    seen_ids: set[str] = set()
    for row in rows:
        fixture_id = str(row.get("fixture_id", ""))
        variant = str(row.get("variant", ""))
        row_id = str(row.get("row_id", ""))
        _require(bool(fixture_id) and bool(row_id), "row lacks fixture_id/row_id")
        _require(row_id not in seen_ids, f"duplicate row_id: {row_id}")
        _require(variant in ("served", "planted_miss"),
                 f"invalid variant for {row_id}: {variant}")
        _require(variant not in by_fixture[fixture_id],
                 f"duplicate {fixture_id}:{variant}")
        seen_ids.add(row_id)
        by_fixture[fixture_id][variant] = row
    _require(len(by_fixture) == expected_pairs,
             f"expected {expected_pairs} fixture pairs, got {len(by_fixture)}")
    for fixture_id, variants in by_fixture.items():
        _require(set(variants) == {"served", "planted_miss"},
                 f"fixture is not paired: {fixture_id}")
    return dict(by_fixture)


def validate_served_control(
    row: Mapping[str, Any], where: str = "served control",
) -> dict[str, Any]:
    """Validate the complete registered served-side task projection."""
    try:
        aliases = {int(value) for value in row.get("plant_alias_ids") or ()}
        mounted = {int(value) for value in row.get("mounted_ids") or ()}
        target = int(row.get("plant_target_id", -1))
    except (TypeError, ValueError) as exc:
        raise DETError(f"{where} has malformed plant/mount identifiers") from exc
    registry = row.get("plant_registry") or {}
    entry_digest = str(row.get("plant_entry_sha256", ""))
    checks = {
        "answer_correct": row.get("answer_correct") is True,
        "target_contains_expected": row.get("target_contains_expected") is True,
        "registered_target_is_alias": target in aliases,
        "registered_target_mounted": target in mounted,
        "all_registered_aliases_mounted": bool(aliases) and aliases <= mounted,
        "mounted_contains_expected": row.get("mounted_contains_expected") is True,
        "full_index_contains_all_aliases": (
            row.get("full_index_contains_all_aliases") is True),
        "plant_target_from_lived_registry": (
            row.get("plant_target_source")
            == "DET1_7_LIVED_ADMISSION_REGISTRY"),
        "plant_registry_bound": (
            isinstance(registry, Mapping)
            and isinstance(registry.get("sha256"), str)
            and len(str(registry.get("sha256"))) == 64),
        "plant_entry_bound": (
            len(entry_digest) == 64
            and all(character in "0123456789abcdef"
                    for character in entry_digest)),
    }
    for name, passed in checks.items():
        _require(passed, f"{where} failed served check: {name}")
    return {
        "status": "PASS",
        "checks": checks,
        "plant_target_id": target,
        "plant_alias_ids": sorted(aliases),
        "mounted_alias_ids": sorted(aliases & mounted),
        "plant_registry_sha256": registry.get("sha256"),
        "plant_entry_sha256": entry_digest,
    }


def validate_g0_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate the registered 12-pair planted-miss gate."""
    _require(len(rows) == 24,
             f"DET-G0 requires 12 served + 12 planted rows, got {len(rows)}")
    pairs = _pair_rows(rows, expected_pairs=12)
    delta_count = 0
    delta_receipt_count = 0
    observed_delta_field_count = 0
    alias_count = 0
    for fixture_id, variants in pairs.items():
        served = variants["served"]
        planted = variants["planted_miss"]
        validate_served_control(served, f"served control {fixture_id}")
        for binding in (
            "plant_target_id", "plant_alias_ids", "plant_registry",
            "plant_entry_sha256", "plant_target_source",
        ):
            _require(served.get(binding) == planted.get(binding),
                     f"DET-G0 pair plant binding differs: "
                     f"{fixture_id}.{binding}")
        _require(planted.get("target_contains_expected") is True,
                 f"DET-G0 planted target lacks expected value: {fixture_id}")
        checks = planted.get("plant_checks") or {}
        for key in (
            "withheld_aliases_absent",
            "logical_target_absent",
            "registered_plant_target_absent",
            "registered_plant_aliases_absent",
            "expected_value_absent_from_mounted_text",
            "withheld_aliases_remain_in_full_detector_index",
            "admission_ranking_unchanged",
            "admission_branch_unchanged",
            "only_aliases_removed_from_admission_plan",
            "withheld_aliases_absent_from_every_ladder_attempt",
        ):
            _require(checks.get(key) is True,
                     f"DET-G0 planted check failed: {fixture_id}.{key}")
        restore = planted.get("fork_restore") or planted.get("restore") or {}
        _require(restore.get("intervention") == PLANTED_INTERVENTION,
                 f"wrong planted intervention: {fixture_id}")
        _require(restore.get("same_process_index_verified") is True,
                 f"same_process_index_verified failed: {fixture_id}")
        deltas = list(restore.get("field_deltas") or ())
        _require(bool(deltas),
                 f"planted fork_restore.field_deltas is empty: {fixture_id}")
        delta = restore.get("delta_receipt") or {}
        _require(delta.get("schema") == DELTA_RECEIPT_SCHEMA,
                 f"fork-hydration delta receipt schema drift: {fixture_id}")
        _require(delta.get("status") == DELTA_PASS_STATUS,
                 f"fork-hydration delta receipt status is not PASS: {fixture_id}")
        _require(delta.get("gate_pass") is True,
                 f"fork-hydration delta receipt gate_pass failed: {fixture_id}")
        target_absence = delta.get("target_absence") or {}
        _require(target_absence.get("gate_pass") is True,
                 f"fork-hydration target absence failed: {fixture_id}")
        absence_checks = target_absence.get("checks") or {}
        _require(
            isinstance(absence_checks, Mapping)
            and bool(absence_checks)
            and all(value is True for value in absence_checks.values()),
            f"fork-hydration target-absence checks failed: {fixture_id}",
        )
        strict_zero = delta.get("strict_zero_comparator") or {}
        _require(
            strict_zero.get("schema")
            == "grm.det1_4.zero_fork_substrate_comparison.v1"
            and strict_zero.get("status") == "DIVERGENT"
            and strict_zero.get("gate_pass") is False
            and isinstance(strict_zero.get("counts"), Mapping)
            and isinstance(strict_zero.get("row_count"), int)
            and int(strict_zero["row_count"]) > 0,
            f"fork-hydration strict ZERO comparator proof failed: {fixture_id}",
        )
        _require(delta.get("exact_divergence_set") is True,
                 f"fork-hydration exact divergence set failed: {fixture_id}")
        _require(delta.get("non_delta_fields_equal") is True,
                 f"fork-hydration non-delta equality failed: {fixture_id}")
        _require(delta.get("retained_arrays_exact") is True,
                 f"fork-hydration retained arrays differ: {fixture_id}")
        transformed = delta.get("transformed_arrays") or []
        _require(
            isinstance(transformed, list)
            and bool(transformed)
            and all(
                isinstance(item, Mapping)
                and item.get("retained_bytes_exact") is True
                for item in transformed
            ),
            f"fork-hydration transformed arrays are not exact: {fixture_id}",
        )
        for key in (
            "unexpected_divergent_fields",
            "missing_expected_divergent_fields",
            "expected_fork_value_mismatches",
        ):
            value = delta.get(key)
            _require(isinstance(value, list) and not value,
                     f"fork-hydration {key} is nonempty or malformed: {fixture_id}")
        expected_fields = delta.get("expected_divergent_fields")
        observed_fields = delta.get("observed_non_equal_fields")
        _require(isinstance(expected_fields, list) and bool(expected_fields),
                 f"fork-hydration expected divergent fields are empty: {fixture_id}")
        _require(isinstance(observed_fields, list),
                 f"fork-hydration observed divergent fields are malformed: {fixture_id}")
        _require(
            len(expected_fields) == len(set(map(str, expected_fields)))
            and len(observed_fields) == len(set(map(str, observed_fields))),
            f"fork-hydration divergence fields contain duplicates: {fixture_id}",
        )
        _require(
            {str(value) for value in expected_fields}
            == {str(value) for value in observed_fields},
            f"fork-hydration observed divergence set differs from expected: {fixture_id}",
        )
        delta_snapshot = restore.get("delta_fork_snapshot")
        _require(
            isinstance(delta_snapshot, Mapping)
            and isinstance(delta_snapshot.get("path"), str)
            and bool(delta_snapshot.get("path"))
            and isinstance(delta_snapshot.get("bytes"), int)
            and int(delta_snapshot.get("bytes")) > 0
            and isinstance(delta_snapshot.get("sha256"), str)
            and len(str(delta_snapshot.get("sha256"))) == 64,
            f"fork-hydration delta_fork_snapshot is not a file record: {fixture_id}",
        )
        delta_count += len(deltas)
        delta_receipt_count += 1
        observed_delta_field_count += len(observed_fields)
        aliases = {int(value) for value in planted.get("logical_alias_ids", ())}
        mounts = {int(value) for value in planted.get("mounted_ids", ())}
        if aliases:
            _require(not aliases & mounts,
                     f"planted aliases remain mounted: {fixture_id}")
            alias_count += len(aliases)
    return {
        "schema": STAGE_SCHEMAS["g0"],
        "status": "PASS",
        "gate_pass": True,
        "counts": {
            "fixtures": len(pairs),
            "served": 12,
            "planted_miss": 12,
        },
        "registered_delta_count": delta_count,
        "fork_hydration_delta_receipt_count": delta_receipt_count,
        "observed_delta_field_count": observed_delta_field_count,
        "withheld_alias_count": alias_count,
        "full_index_same_process_verified_count": 12,
    }


def _hook_set(value: Mapping[str, Any]) -> frozenset[str]:
    hooks = value.get("active_hooks", value.get("active_detectors", ()))
    return frozenset(str(item) for item in (hooks or ()))


def validate_g1_pair(
    control: Mapping[str, Any], hooked: Mapping[str, Any],
) -> dict[str, Any]:
    """Require raw byte identity between bare and all-mechanistic-hook arms."""
    _require(not _hook_set(control), "DET-G1 control unexpectedly has active hooks")
    _require(_hook_set(hooked) == frozenset(MECHANISTIC),
             "DET-G1 active hooks are not exactly D-LQR/D-NGH/D-ENT")
    _require(control.get("d_verb_isolated") is True
             and hooked.get("d_verb_isolated") is True,
             "DET-G1 must keep D-VERB outside both scored processes")
    left_process = str(control.get("process_instance_sha256", ""))
    right_process = str(hooked.get("process_instance_sha256", ""))
    _require(bool(left_process) and bool(right_process),
             "DET-G1 arm lacks process identity")
    left = control.get("byte_records") or {}
    right = hooked.get("byte_records") or {}
    required = {"transcript", "scorecard", "served_projection"}
    _require(required <= set(left) and required <= set(right),
             "DET-G1 lacks transcript/scorecard/served_projection byte records")
    _require(set(left) == set(right), "DET-G1 byte-record field sets differ")
    comparisons = {}
    for name in sorted(left):
        left_digest = _record_digest(left[name], f"control byte_records.{name}")
        right_digest = _record_digest(right[name], f"hooked byte_records.{name}")
        _require(left_digest == right_digest,
                 f"DET-G1 byte identity failed for {name}")
        comparisons[name] = {
            "bytes": left_digest[0],
            "sha256": left_digest[1],
            "byte_identical": True,
        }
    return {
        "schema": STAGE_SCHEMAS["g1"],
        "status": "PASS",
        "gate_pass": True,
        "control_process_instance_sha256": left_process,
        "hooked_process_instance_sha256": right_process,
        "active_hooks": list(MECHANISTIC),
        "d_verb_isolated": True,
        "byte_identity": {name: True for name in comparisons},
        "byte_records": comparisons,
    }


def _registered_split(registration: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    split = registration.get("split_rule") or {}
    calibration = [str(value) for value in split.get("calibration", ())]
    evaluation = [
        *[str(value) for value in split.get("eval_e2e", ())],
        *[str(value) for value in split.get("eval_supersession", ())],
    ]
    _require(len(calibration) == 2 and len(set(calibration)) == 2,
             "registration calibration split is not exactly two fixtures")
    _require(split.get("calibration_pair_count") == 2,
             "registration calibration_pair_count is not 2")
    _require(len(evaluation) == 12 and len(set(evaluation)) == 12,
             "registration evaluation split is not exactly twelve fixtures")
    _require(split.get("eval_pair_count") == 12,
             "registration eval_pair_count is not 12")
    _require(not set(calibration) & set(evaluation),
             "registration calibration/evaluation splits overlap")
    return calibration, evaluation


def effective_registration_projection(
    registration: Mapping[str, Any], registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Substitute only fixture identities while preserving the frozen split."""
    projected = json.loads(json.dumps(registration))
    mapping = {
        str(entry["fixture_id"]): str(entry["effective_fixture_id"])
        for entry in registry.get("entries") or ()
    }
    base_cal, base_eval = _registered_split(registration)
    _require(set(mapping) == set(base_cal) | set(base_eval),
             "plant registry does not map the complete frozen split")
    split = projected["split_rule"]
    split["calibration"] = [mapping[value] for value in base_cal]
    split["eval_e2e"] = [mapping[str(value)] for value in
                         registration["split_rule"]["eval_e2e"]]
    split["eval_supersession"] = [
        mapping[str(value)] for value in
        registration["split_rule"]["eval_supersession"]]
    return projected


def validate_split(
    registration: Mapping[str, Any],
    calibration_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the frozen 2-served calibration / 12+12 evaluation split."""
    calibration_ids, evaluation_ids = _registered_split(registration)
    _require(len(calibration_rows) == 2,
             f"calibration requires 2 served rows, got {len(calibration_rows)}")
    observed_cal = []
    for row in calibration_rows:
        _require(row.get("variant") == "served",
                 "calibration contains a non-served row")
        observed_cal.append(str(row.get("fixture_id", "")))
    _require(observed_cal == calibration_ids,
             f"calibration order/identity drift: {observed_cal}")
    observed_eval_ids = {str(row.get("fixture_id", "")) for row in eval_rows}
    _require(not set(observed_cal) & observed_eval_ids,
             "observed calibration/evaluation fixtures are not disjoint")
    pairs = _pair_rows(eval_rows, expected_pairs=12)
    _require(list(pairs) == evaluation_ids,
             f"evaluation order/identity drift: {list(pairs)}")
    _require(not set(observed_cal) & set(pairs),
             "observed calibration/evaluation fixtures overlap")
    return {
        "schema": STAGE_SCHEMAS["calibration"],
        "status": "PASS",
        "calibration_fixture_ids": calibration_ids,
        "calibration_served_count": 2,
        "evaluation_fixture_ids": evaluation_ids,
        "evaluation_served_count": 12,
        "evaluation_planted_miss_count": 12,
        "calibration_count": 2,
        "eval_counts": {"served": 12, "planted_miss": 12},
        "disjoint": True,
    }


def merge_detector_rows(
    mechanistic_rows: Sequence[Mapping[str, Any]],
    verbal_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join isolated D-VERB observations without changing mechanistic rows."""
    _require(len(mechanistic_rows) == 24 and len(verbal_rows) == 24,
             "detector row keys require exactly 24+24 rows")
    verbal_by_id: dict[str, Mapping[str, Any]] = {}
    for row in verbal_rows:
        row_id = str(row.get("row_id", ""))
        _require(bool(row_id) and row_id not in verbal_by_id,
                 f"duplicate/missing verbal row_id: {row_id}")
        verbal_by_id[row_id] = row
    merged = []
    seen: set[str] = set()
    for mech in mechanistic_rows:
        row_id = str(mech.get("row_id", ""))
        _require(row_id in verbal_by_id and row_id not in seen,
                 f"unpaired/duplicate mechanistic row: {row_id}")
        verbal = verbal_by_id[row_id]
        for key in ("fixture_id", "variant"):
            _require(str(mech.get(key)) == str(verbal.get(key)),
                     f"detector row identity drift for {row_id}.{key}")
        payload = verbal.get("verbal")
        _require(isinstance(payload, Mapping), f"verbal payload missing: {row_id}")
        row = dict(mech)
        row["verbal"] = dict(payload)
        row["verbal_evidence"] = {
            "schema": verbal.get("schema"),
            "process_instance_sha256": verbal.get("process_instance_sha256"),
            "verbal_started_unix_ns": verbal.get("verbal_started_unix_ns"),
        }
        merged.append(row)
        seen.add(row_id)
    _require(seen == set(verbal_by_id), "verbal rows contain unjoined extras")
    return merged


def validate_verbal_chronology(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Validate isolated lived D-VERB execution and strict after-ordering."""
    process = str(receipt.get("process_instance_sha256", ""))
    mechanistic_process = str(
        receipt.get("mechanistic_process_instance_sha256", ""))
    _require(bool(process) and bool(mechanistic_process) and process != mechanistic_process,
             "D-VERB did not run in a distinct process")
    _require(receipt.get("verbal_question_exact") == VERBAL_QUESTION,
             "D-VERB registered verbal question drifted")
    verbal_processes = [str(value) for value in (
        receipt.get("process_instance_sha256s") or ())]
    mechanistic_processes = [str(value) for value in (
        receipt.get("mechanistic_process_instance_sha256s") or ())]
    if verbal_processes or mechanistic_processes:
        _require(bool(verbal_processes) and bool(mechanistic_processes),
                 "D-VERB process-set binding is incomplete")
        _require(aggregate_process_instances(verbal_processes) == process,
                 "D-VERB process aggregate drifted")
        _require(aggregate_process_instances(mechanistic_processes)
                 == mechanistic_process,
                 "mechanistic process aggregate drifted")
        _require(set(verbal_processes).isdisjoint(mechanistic_processes),
                 "D-VERB process set overlaps mechanistic workers")
    rows = list(receipt.get("rows") or ())
    _require(len(rows) == 24, f"D-VERB chronology requires 24 rows, got {len(rows)}")
    pairs = _pair_rows(rows, expected_pairs=12)
    del pairs
    strict_after = 0
    for row in rows:
        row_id = str(row.get("row_id", ""))
        _require(row.get("prompt_suffix_exact") == VERBAL_QUESTION,
                 f"D-VERB prompt suffix drift: {row_id}")
        if verbal_processes:
            _require(str(row.get("process_instance_sha256", ""))
                     in set(verbal_processes),
                     f"D-VERB row process is not registered: {row_id}")
        mech_ns = row.get("mechanistic_completed_unix_ns")
        verbal_ns = row.get("verbal_started_unix_ns")
        _require(isinstance(mech_ns, int) and isinstance(verbal_ns, int),
                 f"D-VERB chronology timestamps missing: {row_id}")
        _require(verbal_ns > mech_ns,
                 f"D-VERB did not start strictly after mechanistic row: {row_id}")
        strict_after += 1
    return {
        "schema": STAGE_SCHEMAS["eval_verbal"],
        "status": "PASS",
        "gate_pass": True,
        "process_instance_sha256": process,
        "mechanistic_process_instance_sha256": mechanistic_process,
        "processes_distinct": True,
        "row_count": len(rows),
        "strict_after_count": strict_after,
        "verbal_question_exact": VERBAL_QUESTION,
    }


def build_lead_commands(
    run_dir: Path = FROZEN_RUN,
    lease_seconds: int = LEASE_SECONDS,
    wait_seconds: int = DEFAULT_WAIT_SECONDS,
    gap_seconds: int = GAP_SECONDS,
) -> list[list[str]]:
    """Pure deterministic lead-command inventory; performs no allocation."""
    _require(0 < int(lease_seconds) <= MAX_LEASE_SECONDS,
             "lead lease exceeds standing cap")
    script = str(Path(__file__).resolve())
    common = ["--run-dir", str(Path(run_dir))]
    gpu_tail = [
        "--lease-seconds", str(int(lease_seconds)),
        "--lock-wait-seconds", str(int(wait_seconds)),
        "--gap-seconds", str(int(gap_seconds)),
    ]
    commands = [
        [sys.executable, script, "cross-process-zero", *common, *gpu_tail],
        [sys.executable, script, "author-det1-7-source", *common],
        [sys.executable, script, "plant-registration", *common, *gpu_tail],
        [sys.executable, script, "author-det1-7-terminal", *common],
        *[
            [sys.executable, script, stage, *common, *gpu_tail]
            for stage in (
                "g0", "g1", "calibration",
                "eval-mechanistic", "eval-verbal",
            )
        ],
    ]
    analyzer = ROOT / "scripts/grm_det1_5_analyze.py"
    commands.append([
        sys.executable, str(analyzer), "report", "--run-dir", str(Path(run_dir)),
    ])
    return commands


def _parent_zero() -> tuple[Path, dict[str, Any]]:
    _require(PARENT_ZERO_MARKER.is_file(), "DET1.4 zero marker is missing")
    marker = read_json(PARENT_ZERO_MARKER)
    _require(marker.get("schema") == "grm.det1_4.stage_marker.v1"
             and marker.get("stage") == "zero_gate"
             and marker.get("status") == "COMPLETE",
             "DET1.4 zero marker is not COMPLETE")
    receipt_path = _validate_file_record(marker.get("receipt") or {},
                                         "DET1.4 zero receipt")
    receipt = read_json(receipt_path)
    _validate_zero_receipt(receipt, "DET1.4 parent zero")
    return receipt_path, receipt


def _amendment_sources() -> dict[str, dict[str, Any]]:
    candidates = (
        ROOT / "scripts/grm_det1_5_gpu.py",
        ROOT / "scripts/grm_det1_5_workers.py",
        ROOT / "scripts/grm_det1_5_analyze.py",
        ROOT / "scripts/grm_det1_5_lead.sh",
        ROOT / "tests/test_grm_det1_5_campaign.py",
    )
    values = {}
    for path in candidates:
        if path.is_file():
            values[str(path.relative_to(ROOT))] = file_record(path)
    required = (
        "scripts/grm_det1_5_gpu.py",
        "scripts/grm_det1_5_workers.py",
        "scripts/grm_det1_5_analyze.py",
        "scripts/grm_det1_5_lead.sh",
        "tests/test_grm_det1_5_campaign.py",
    )
    missing = [source for source in required if source not in values]
    _require(not missing,
             f"race amendment cannot omit execution sources: {missing}")
    return values


def _historical_source_record(
    source_path: str,
    record: Mapping[str, Any],
    where: str,
) -> dict[str, Any]:
    _require(isinstance(record, Mapping), f"{where} is not a source record")
    byte_count = record.get("bytes")
    digest = record.get("sha256")
    _require(isinstance(byte_count, int) and byte_count >= 0,
             f"{where}.bytes is invalid")
    _require(isinstance(digest, str) and len(digest) == 64,
             f"{where}.sha256 is invalid")
    shown = record.get("path", source_path)
    _require(shown == source_path, f"{where}.path drift")
    return {"path": source_path, "bytes": byte_count, "sha256": digest}


def _pre_delta_source_inventory() -> dict[str, dict[str, Any]]:
    """Recover the complete source inventory frozen by DET1.4 and DET1.5."""
    det14 = read_json(PRIOR_AMENDMENT)
    _require(det14.get("schema") == "grm.det1_4.source_amendment.v1"
             and det14.get("status") == "AUTHORIZED_FORK_FROM_SNAPSHOT_AMENDMENT",
             "DET1.4 source amendment is not the authorized predecessor")
    det15 = read_json(RACE_AMENDMENT)
    _require(det15.get("schema") == RACE_AMENDMENT_SCHEMA
             and det15.get("status") == RACE_AMENDMENT_STATUS,
             "DET1.5 race amendment is not the authorized predecessor")

    inventory: dict[str, dict[str, Any]] = {}

    def add(source_path: str, record: Mapping[str, Any], where: str) -> None:
        normalized = _historical_source_record(source_path, record, where)
        prior = inventory.get(source_path)
        _require(prior is None or prior == normalized,
                 f"pre-DET1.6 source lineage conflicts for {source_path}")
        inventory[source_path] = normalized

    for field in ("allowed_source_changes", "added_sources", "test_sources"):
        section = det14.get(field) or {}
        _require(isinstance(section, Mapping),
                 f"DET1.4 amendment {field} is malformed")
        for source_path, raw in section.items():
            _require(isinstance(source_path, str) and isinstance(raw, Mapping),
                     f"DET1.4 amendment {field} entry is malformed")
            record = raw.get("after") if field == "allowed_source_changes" else raw
            add(source_path, record or {}, f"DET1.4 amendment {field}.{source_path}")

    source_inventory = det15.get("source_inventory") or {}
    _require(isinstance(source_inventory, Mapping),
             "DET1.5 amendment source inventory is malformed")
    for source_path, record in source_inventory.items():
        _require(isinstance(source_path, str),
                 "DET1.5 amendment source path is malformed")
        add(source_path, record or {},
            f"DET1.5 amendment source_inventory.{source_path}")
    return inventory


def _delta_invariants() -> dict[str, Any]:
    return {
        "fork_substrate_only": True,
        "reconstruction_as_counterfactual": "FORBIDDEN",
        "registered_planted_miss_only": True,
        "withheld_seats_hydrated": False,
        "target_absence_required": True,
        "non_delta_bytes": "EXACT_CANONICAL_VALUE_BYTES",
        "zero_comparator_reused_under_delta": True,
        "historical_rows_reusable": False,
        "pre_det1_6_g0_attempts_reusable": False,
        "campaign_rerun_from": "DET_G0",
        "race_resume_authorized": True,
        "amendment_is_evidence": False,
    }


def _pre_delta_g0_output_records(run_dir: Path) -> list[dict[str, Any]]:
    root = campaign_root(run_dir) / STAGE_SHARD_DIRS["g0"]
    records: list[dict[str, Any]] = []
    for spec in ("e2e-1", "e2e-2", "e2e-3", "e2e-4", "sup-1", "sup-2"):
        path = root / spec / "attempt_001" / "worker_output.json"
        _require(path.is_file(), f"pre-DET1.6 G0 attempt is missing: {path}")
        value = read_json(path)
        _require(value.get("source_amendment") is None,
                 f"pre-DET1.6 G0 attempt unexpectedly has terminal provenance: {path}")
        records.append(file_record(path))
    return records


def validate_det1_7_source_authorization(
    run_dir: Path = FROZEN_RUN,
) -> dict[str, Any]:
    from scripts.grm_det1_7_source_auth import (
        validate_precollection_source_authorization,
    )

    run_dir = Path(run_dir).resolve()
    _require(run_dir == FROZEN_RUN.resolve(),
             "DET1.7 source authorization is bound to the frozen run")
    return validate_precollection_source_authorization(
        run_dir / DET1_7_SOURCE_AUTH.name,
        repo_root=ROOT,
        order_path=DET1_7_ORDER,
        registration_path=REGISTRATION,
        runtime_frame_path=RUNTIME_FRAME,
        det1_6_amendment_path=DELTA_AMENDMENT,
        changed_sources=DET1_7_CHANGED_SOURCES,
        added_sources=DET1_7_ADDED_SOURCES,
    )


def author_det1_7_source_authorization(
    run_dir: Path = FROZEN_RUN,
) -> dict[str, Any]:
    from scripts.grm_det1_7_source_auth import (
        author_precollection_source_authorization,
    )

    run_dir = Path(run_dir).resolve()
    _require(run_dir == FROZEN_RUN.resolve(),
             "DET1.7 source authorization is bound to the frozen run")
    path = run_dir / DET1_7_SOURCE_AUTH.name
    if path.exists():
        return validate_det1_7_source_authorization(run_dir)
    return author_precollection_source_authorization(
        path,
        repo_root=ROOT,
        order_path=DET1_7_ORDER,
        registration_path=REGISTRATION,
        runtime_frame_path=RUNTIME_FRAME,
        det1_6_amendment_path=DELTA_AMENDMENT,
        changed_sources=DET1_7_CHANGED_SOURCES,
        added_sources=DET1_7_ADDED_SOURCES,
    )


def load_det1_7_precollection_context(
    run_dir: Path = FROZEN_RUN,
) -> dict[str, Any]:
    authorization = validate_det1_7_source_authorization(run_dir)
    _require(authorization.get("collection_authorized") is True
             and authorization.get("race_resume_authorized") is False,
             "DET1.7 precollection authority crossed into evaluation")
    return {
        "precollection_authorization": file_record(
            Path(run_dir).resolve() / DET1_7_SOURCE_AUTH.name),
    }


def validate_delta_amendment(run_dir: Path = FROZEN_RUN) -> dict[str, Any]:
    """Validate the append-only DET1.6 rebinding and every source it covers."""
    run_dir = Path(run_dir).resolve()
    _require(run_dir == FROZEN_RUN.resolve(),
             "DET1.6 amendment is bound to the one frozen DET1 run")
    path = run_dir / DELTA_AMENDMENT.name
    _require(path.is_file(), f"DET1.6 delta amendment is missing: {path}")
    value = read_json(path)
    _require(value.get("schema") == DELTA_AMENDMENT_SCHEMA,
             "DET1.6 delta-amendment schema drift")
    _require(value.get("status") == DELTA_AMENDMENT_STATUS,
             "DET1.6 delta-amendment status drift")
    for key, expected in (
        ("order", file_record(DET1_6_ORDER)),
        ("registration", file_record(REGISTRATION)),
        ("runtime_frame", file_record(RUNTIME_FRAME)),
        ("det1_4_source_amendment", file_record(PRIOR_AMENDMENT)),
        ("det1_5_race_amendment", file_record(RACE_AMENDMENT)),
        ("zero_gate_marker", file_record(PARENT_ZERO_MARKER)),
        ("cross_process_zero_marker", file_record(
            campaign_root(run_dir) / STAGE_MARKERS["cross_process_zero"])),
    ):
        _require(value.get(key) == expected,
                 f"DET1.6 delta-amendment {key} drift")
    _require(
        value.get("historical_g0_attempt_outputs")
        == _pre_delta_g0_output_records(run_dir),
        "DET1.6 historical G0 attempt binding drift",
    )

    historical = _pre_delta_source_inventory()
    changes = value.get("allowed_source_changes") or {}
    _require(isinstance(changes, Mapping),
             "DET1.6 allowed_source_changes is malformed")
    _require(set(changes) == set(DET1_6_CHANGED_SOURCES),
             "DET1.6 amendment source allowlist drift")

    rebindings: dict[str, dict[str, Any]] = {}
    for source_path, before in historical.items():
        current = file_record(ROOT / source_path)
        if source_path not in changes:
            _require(current == before,
                     f"unregistered source drift after DET1.5: {source_path}")
            continue
        entry = changes[source_path]
        _require(isinstance(entry, Mapping),
                 f"DET1.6 source rebinding is malformed: {source_path}")
        _require(entry.get("purpose") == DET1_6_CHANGE_PURPOSES[source_path],
                 f"DET1.6 source purpose drift: {source_path}")
        _require(entry.get("before") == before,
                 f"DET1.6 source predecessor drift: {source_path}")
        _require(entry.get("after") == current,
                 f"DET1.6 source current binding drift: {source_path}")
        _require(entry.get("before") != entry.get("after"),
                 f"DET1.6 source change is empty: {source_path}")
        rebindings[source_path] = {
            "before": dict(entry["before"]),
            "after": dict(entry["after"]),
        }

    _require(set(rebindings) == set(DET1_6_CHANGED_SOURCES),
             "DET1.6 changed sources are not all predecessor-bound")
    invariants = value.get("invariants") or {}
    _require(invariants == _delta_invariants(),
             "DET1.6 delta-amendment invariants drift")
    return {
        "schema": DELTA_AMENDMENT_SCHEMA,
        "status": DELTA_AMENDMENT_STATUS,
        "record": file_record(path),
        "source_rebindings": rebindings,
        "source_count": len(rebindings),
        "race_resume_authorized": True,
        "amendment_is_evidence": False,
    }


def author_delta_amendment(run_dir: Path = FROZEN_RUN) -> dict[str, Any]:
    """Write once the DET1.6 source rebinding authorized by its order."""
    run_dir = Path(run_dir).resolve()
    _require(run_dir == FROZEN_RUN.resolve(),
             "DET1.6 amendment is bound to the one frozen DET1 run")
    path = run_dir / DELTA_AMENDMENT.name
    if path.exists():
        return validate_delta_amendment(run_dir)

    historical = _pre_delta_source_inventory()
    changes: dict[str, dict[str, Any]] = {}
    for source_path in DET1_6_CHANGED_SOURCES:
        _require(source_path in historical,
                 f"DET1.6 source lacks a frozen predecessor: {source_path}")
        current = file_record(ROOT / source_path)
        _require(current != historical[source_path],
                 f"DET1.6 source did not change: {source_path}")
        changes[source_path] = {
            "purpose": DET1_6_CHANGE_PURPOSES[source_path],
            "before": historical[source_path],
            "after": current,
        }
    for source_path, before in historical.items():
        if source_path not in changes:
            _require(file_record(ROOT / source_path) == before,
                     f"cannot authorize unrelated source drift: {source_path}")

    value = {
        "schema": DELTA_AMENDMENT_SCHEMA,
        "status": DELTA_AMENDMENT_STATUS,
        "created_utc": utc_now(),
        "order": file_record(DET1_6_ORDER),
        "registration": file_record(REGISTRATION),
        "runtime_frame": file_record(RUNTIME_FRAME),
        "det1_4_source_amendment": file_record(PRIOR_AMENDMENT),
        "det1_5_race_amendment": file_record(RACE_AMENDMENT),
        "zero_gate_marker": file_record(PARENT_ZERO_MARKER),
        "cross_process_zero_marker": file_record(
            campaign_root(run_dir) / STAGE_MARKERS["cross_process_zero"]),
        "historical_g0_attempt_outputs": _pre_delta_g0_output_records(run_dir),
        "allowed_source_changes": changes,
        "invariants": _delta_invariants(),
    }
    write_json_exclusive(path, value)
    return validate_delta_amendment(run_dir)


def validate_race_amendment(
    run_dir: Path = FROZEN_RUN,
    *,
    source_rebindings: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    path = Path(run_dir) / RACE_AMENDMENT.name
    _require(path.is_file(), f"race-authorization amendment is missing: {path}")
    value = read_json(path)
    _require(value.get("schema") == RACE_AMENDMENT_SCHEMA,
             "race-amendment schema drift")
    _require(value.get("status") == RACE_AMENDMENT_STATUS,
             "race-amendment status is not authorized")
    for key, expected in (
        ("order", file_record(ORDER)),
        ("det1_order", file_record(DET1_ORDER)),
        ("registration", file_record(REGISTRATION)),
        ("runtime_frame", file_record(RUNTIME_FRAME)),
        ("prior_amendment", file_record(PRIOR_AMENDMENT)),
    ):
        _require(value.get(key) == expected, f"race-amendment {key} drift")
    parent_zero_path, _parent_zero_value = _parent_zero()
    _require(value.get("zero_gate_marker") == file_record(PARENT_ZERO_MARKER),
             "race amendment does not bind DET1.4 zero marker")
    _require(value.get("zero_gate_receipt") == file_record(parent_zero_path),
             "race amendment does not bind DET1.4 zero receipt")
    zero_marker = campaign_root(run_dir) / STAGE_MARKERS["cross_process_zero"]
    _require(value.get("cross_process_zero_marker") == file_record(zero_marker),
             "race amendment does not bind current cross-process-zero marker")
    marker = read_json(zero_marker)
    zero_receipt = _validate_file_record(marker.get("receipt") or {},
                                         "amendment zero receipt")
    _require(value.get("cross_process_zero_receipt") == file_record(zero_receipt),
             "race amendment does not bind current zero receipt")
    sources = value.get("source_inventory") or {}
    _require(isinstance(sources, Mapping), "race-amendment source inventory malformed")
    if source_rebindings is None and DELTA_AMENDMENT.is_file():
        source_rebindings = validate_delta_amendment(run_dir)["source_rebindings"]
    superseded_sources: list[str] = []
    for shown, record in sources.items():
        current = file_record(ROOT / str(shown))
        if current != record:
            rebinding = (source_rebindings or {}).get(str(shown))
            _require(isinstance(rebinding, Mapping),
                     f"race-amendment source binding drift: {shown}")
            _require(rebinding.get("before") == record,
                     f"race-amendment predecessor binding drift: {shown}")
            _require(rebinding.get("after") == current,
                     f"race-amendment superseding binding drift: {shown}")
            superseded_sources.append(str(shown))
    required_sources = (
        "scripts/grm_det1_5_gpu.py",
        "scripts/grm_det1_5_workers.py",
        "scripts/grm_det1_5_analyze.py",
        "scripts/grm_det1_5_lead.sh",
        "tests/test_grm_det1_5_campaign.py",
    )
    for source in required_sources:
        _require(source in sources,
                 f"race amendment omits execution source: {source}")
    lead = value.get("lead_script") or {}
    # The source-inventory loop above has already required either exact bytes
    # or an explicit transitive rebinding.  Requiring the predecessor bytes a
    # second time here would reject the very lead-script rebinding DET1.7
    # authorized; this check is identity-only after that byte validation.
    lead_path = _path_from_record(lead)
    _require(lead_path.is_file(), "race amendment lead_script is missing")
    _require(lead_path.resolve()
             == (ROOT / "scripts/grm_det1_5_lead.sh").resolve(),
             "race amendment binds the wrong lead script")
    _require(sources.get("scripts/grm_det1_5_lead.sh") == lead,
             "lead script is not identically bound in source_inventory")
    invariants = value.get("invariants") or {}
    _require(invariants.get("race_resume_authorized") is True,
             "race amendment does not authorize resume")
    _require(invariants.get("fork_substrate_only") is True,
             "race amendment does not require fork substrate")
    _require(invariants.get("historical_rows_reusable") is False,
             "race amendment allows historical row reuse")
    return {
        "schema": RACE_AMENDMENT_SCHEMA,
        "status": RACE_AMENDMENT_STATUS,
        "record": file_record(path),
        "source_count": len(sources),
        "superseded_sources": sorted(superseded_sources),
        "race_resume_authorized": True,
    }


def author_amendment(run_dir: Path = FROZEN_RUN) -> dict[str, Any]:
    """Write once, or revalidate, the zero-chained race authorization."""
    run_dir = Path(run_dir)
    path = run_dir / RACE_AMENDMENT.name
    if path.exists():
        return validate_race_amendment(run_dir)
    campaign = campaign_root(run_dir)
    zero_marker = campaign / STAGE_MARKERS["cross_process_zero"]
    marker_value = read_json(zero_marker)
    _require(marker_value.get("status") == "COMPLETE",
             "cross-process zero must complete before amendment")
    zero_receipt = _validate_file_record(marker_value.get("receipt") or {},
                                         "cross-process zero receipt")
    zero_value = read_json(zero_receipt)
    _require(zero_value.get("status") == "PASS"
             and zero_value.get("gate_pass") is True,
             "cross-process zero did not PASS")
    parent_zero_receipt, _parent_zero_value = _parent_zero()
    sources = _amendment_sources()
    lead = sources.get("scripts/grm_det1_5_lead.sh")
    _require(lead is not None,
             "race amendment requires scripts/grm_det1_5_lead.sh")
    value = {
        "schema": RACE_AMENDMENT_SCHEMA,
        "status": RACE_AMENDMENT_STATUS,
        "created_utc": utc_now(),
        "order": file_record(ORDER),
        "det1_order": file_record(DET1_ORDER),
        "registration": file_record(REGISTRATION),
        "runtime_frame": file_record(RUNTIME_FRAME),
        "prior_amendment": file_record(PRIOR_AMENDMENT),
        "zero_gate_marker": file_record(PARENT_ZERO_MARKER),
        "zero_gate_receipt": file_record(parent_zero_receipt),
        "cross_process_zero_marker": file_record(zero_marker),
        "cross_process_zero_receipt": file_record(zero_receipt),
        "lead_script": lead,
        "source_inventory": sources,
        "invariants": {
            "race_resume_authorized": True,
            "authorization_boundary": "AFTER_CROSS_PROCESS_ZERO_BEFORE_DET_G0",
            "fork_substrate_only": True,
            "reconstruction_as_counterfactual": "FORBIDDEN",
            "historical_rows_reusable": False,
            "fresh_namespace": str(CAMPAIGN_RELATIVE),
            "d_verb_execution": "SEPARATE_CHRONOLOGICAL_LIVED_PROCESS",
            "stage_order": list(STAGE_ORDER),
            "verdict_vocabulary": ["SUPPORTED", "PARTIAL", "REFUTED"],
            "amendment_is_evidence": False,
        },
    }
    write_json_exclusive(path, value)
    return validate_race_amendment(run_dir)


def _marker_path(run_dir: Path, stage: str) -> Path:
    _require(stage in STAGE_MARKERS, f"unknown stage: {stage}")
    return campaign_root(run_dir) / STAGE_MARKERS[stage]


def _write_stage_marker(
    run_dir: Path,
    stage: str,
    receipt_path: Path,
    prerequisites: Sequence[Path],
) -> Path:
    marker = {
        "schema": "grm.det1_5.stage_marker.v1",
        "stage": stage,
        "status": "COMPLETE",
        "created_utc": utc_now(),
        "receipt": file_record(receipt_path),
        "prerequisites": [file_record(path) for path in prerequisites],
        "registration": file_record(REGISTRATION),
        "runtime_frame": file_record(RUNTIME_FRAME),
    }
    if stage == "cross_process_zero":
        marker["source_amendment"] = file_record(DELTA_AMENDMENT)
    elif stage == "plant_registration":
        marker["det1_7_provenance"] = {
            "precollection_authorization": file_record(
                Path(run_dir).resolve() / DET1_7_SOURCE_AUTH.name),
        }
    else:
        # Preserve the exact DET1.6 predecessor binding while the terminal
        # DET1.7 envelope authorizes the narrower registration amendment.
        marker["source_amendment"] = file_record(DELTA_AMENDMENT)
        marker["det1_7_provenance"] = {
            "precollection_authorization": file_record(
                Path(run_dir).resolve() / DET1_7_SOURCE_AUTH.name),
            "plant_registry_record": file_record(plant_registry_path(run_dir)),
            "terminal_amendment": file_record(
                Path(run_dir).resolve() / DET1_7_TERMINAL_AMENDMENT.name),
        }
    return _write_json_exclusive_or_verify(_marker_path(run_dir, stage), marker)


def _validate_stage_marker(run_dir: Path, stage: str) -> tuple[Path, Path, dict[str, Any]]:
    marker_path = _marker_path(run_dir, stage)
    _require(marker_path.is_file(), f"stage marker is missing: {stage}")
    marker = read_json(marker_path)
    _require(marker.get("schema") == "grm.det1_5.stage_marker.v1"
             and marker.get("stage") == stage
             and marker.get("status") == "COMPLETE",
             f"stage marker is not COMPLETE: {stage}")
    _require(marker.get("registration") == file_record(REGISTRATION),
             f"stage registration binding drift: {stage}")
    _require(marker.get("runtime_frame") == file_record(RUNTIME_FRAME),
             f"stage runtime binding drift: {stage}")
    if stage == "plant_registration":
        _require(
            marker.get("det1_7_provenance") == {
                "precollection_authorization": file_record(
                    Path(run_dir).resolve() / DET1_7_SOURCE_AUTH.name),
            },
            "plant-registration marker source authorization drifted",
        )
    elif stage != "cross_process_zero":
        _require(
            marker.get("det1_7_provenance") == {
                "precollection_authorization": file_record(
                    Path(run_dir).resolve() / DET1_7_SOURCE_AUTH.name),
                "plant_registry_record": file_record(
                    plant_registry_path(run_dir)),
                "terminal_amendment": file_record(
                    Path(run_dir).resolve()
                    / DET1_7_TERMINAL_AMENDMENT.name),
            },
            f"stage DET1.7 binding drift: {stage}",
        )
    receipt_path = _validate_file_record(marker.get("receipt") or {},
                                         f"{stage} receipt")
    receipt = read_json(receipt_path)
    expected_schema = STAGE_SCHEMAS.get(stage)
    if expected_schema is not None:
        _require(receipt.get("schema") == expected_schema,
                 f"stage receipt schema drift: {stage}")
        _require(receipt.get("status") == "PASS",
                 f"stage receipt did not PASS: {stage}")
        if stage == "plant_registration":
            _require(
                receipt.get("precollection_authorization") == file_record(
                    Path(run_dir).resolve() / DET1_7_SOURCE_AUTH.name),
                "plant-registration receipt source authorization drifted",
            )
        elif stage != "cross_process_zero":
            _require(
                receipt.get("plant_registry") == file_record(
                    plant_registry_path(run_dir))
                and receipt.get("terminal_amendment") == file_record(
                    Path(run_dir).resolve()
                    / DET1_7_TERMINAL_AMENDMENT.name),
                f"stage receipt DET1.7 binding drift: {stage}",
            )
    return marker_path, receipt_path, receipt


def _stage_is_complete(run_dir: Path, stage: str) -> bool:
    path = _marker_path(run_dir, stage)
    if not path.exists():
        return False
    _validate_stage_marker(run_dir, stage)
    return True


def _terminal_invariants() -> dict[str, Any]:
    return {
        "amendment_is_evidence": False,
        "detector_arms_changed": False,
        "threshold_policy_changed": False,
        "adjudication_vocabulary_changed": False,
        "historical_rows_reusable": False,
        "plant_registry_frozen_before_g0": True,
        "plant_registry_frozen_before_evaluation": True,
        "race_resume_authorized": True,
        "served_planted_pairing_preserved": True,
        "eval_served_count": 12,
        "eval_planted_miss_count": 12,
    }


def validate_det1_7_terminal_amendment(
    run_dir: Path = FROZEN_RUN,
) -> dict[str, Any]:
    from scripts.grm_det1_7_registry import validate_plant_registry

    run_dir = Path(run_dir).resolve()
    path = run_dir / DET1_7_TERMINAL_AMENDMENT.name
    _require(path.is_file(), f"DET1.7 terminal amendment is missing: {path}")
    value = read_json(path)
    _require(value.get("schema") == DET1_7_TERMINAL_SCHEMA
             and value.get("status") == DET1_7_TERMINAL_STATUS,
             "DET1.7 terminal amendment schema/status drifted")
    source_auth = validate_det1_7_source_authorization(run_dir)
    marker_path, receipt_path, registration_receipt = _validate_stage_marker(
        run_dir, "plant_registration")
    registry_path = plant_registry_path(run_dir)
    registry = read_json(registry_path)
    registry_validation = validate_plant_registry(registry, record_root=ROOT)
    for key, expected in (
        ("order", file_record(DET1_7_ORDER)),
        ("base_registration", file_record(REGISTRATION)),
        ("runtime_frame", file_record(RUNTIME_FRAME)),
        ("det1_6_predecessor", file_record(DELTA_AMENDMENT)),
        ("precollection_authorization", source_auth["record"]),
        ("plant_registration_marker", file_record(marker_path)),
        ("plant_registration_receipt", file_record(receipt_path)),
        ("plant_registry", file_record(registry_path)),
    ):
        _require(value.get(key) == expected,
                 f"DET1.7 terminal amendment {key} drifted")
    _require(registration_receipt.get("plant_registry") == file_record(
        registry_path), "plant-registration receipt binds another registry")
    _require(value.get("invariants") == _terminal_invariants(),
             "DET1.7 terminal amendment invariants drifted")
    return {
        "schema": DET1_7_TERMINAL_SCHEMA,
        "status": DET1_7_TERMINAL_STATUS,
        "record": file_record(path),
        "plant_registry": file_record(registry_path),
        "plant_registry_validation": registry_validation,
        "race_resume_authorized": True,
        "amendment_is_evidence": False,
    }


def author_det1_7_terminal_amendment(
    run_dir: Path = FROZEN_RUN,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    path = run_dir / DET1_7_TERMINAL_AMENDMENT.name
    if path.exists():
        return validate_det1_7_terminal_amendment(run_dir)
    source_auth = validate_det1_7_source_authorization(run_dir)
    marker_path, receipt_path, registration_receipt = _validate_stage_marker(
        run_dir, "plant_registration")
    registry_path = plant_registry_path(run_dir)
    _require(registration_receipt.get("plant_registry") == file_record(
        registry_path), "plant-registration stage did not freeze this registry")
    _require(not _marker_path(run_dir, "g0").exists(),
             "cannot author plant registry amendment after G0 completion")
    value = {
        "schema": DET1_7_TERMINAL_SCHEMA,
        "status": DET1_7_TERMINAL_STATUS,
        "created_utc": utc_now(),
        "order": file_record(DET1_7_ORDER),
        "base_registration": file_record(REGISTRATION),
        "runtime_frame": file_record(RUNTIME_FRAME),
        "det1_6_predecessor": file_record(DELTA_AMENDMENT),
        "precollection_authorization": source_auth["record"],
        "plant_registration_marker": file_record(marker_path),
        "plant_registration_receipt": file_record(receipt_path),
        "plant_registry": file_record(registry_path),
        "invariants": _terminal_invariants(),
    }
    write_json_exclusive(path, value)
    return validate_det1_7_terminal_amendment(run_dir)


def load_det1_7_registered_context(
    run_dir: Path = FROZEN_RUN,
) -> dict[str, Any]:
    terminal = validate_det1_7_terminal_amendment(run_dir)
    registry_path = plant_registry_path(run_dir)
    return {
        "precollection_authorization": file_record(
            Path(run_dir).resolve() / DET1_7_SOURCE_AUTH.name),
        "plant_registry_record": file_record(registry_path),
        "plant_registry": read_json(registry_path),
        "terminal_amendment": terminal["record"],
    }


def inventory(run_dir: Path = FROZEN_RUN) -> dict[str, Any]:
    """CPU-only validation of frozen bindings and fresh campaign progress."""
    run_dir = Path(run_dir)
    _require(run_dir.resolve() == FROZEN_RUN.resolve(),
             "DET1.5 is bound to the one frozen DET1 run")
    from scripts import grm_det1_4_gpu as det14

    det1_7_auth = validate_det1_7_source_authorization(run_dir)
    det14_args = argparse.Namespace(
        run_dir=run_dir,
        require_amendment=True,
        write_receipt=False,
        det1_6_source_rebindings=det1_7_auth[
            "det1_4_source_rebindings"],
    )
    prior = det14.inventory(det14_args)
    parent_zero_path, parent_zero = _parent_zero()
    stages = []
    for stage in STAGE_ORDER:
        complete = _stage_is_complete(run_dir, stage)
        stages.append({
            "stage": stage,
            "status": "COMPLETE" if complete else "PENDING",
            "marker": str(_marker_path(run_dir, stage).relative_to(ROOT)),
        })
    amendment = None
    if (run_dir / RACE_AMENDMENT.name).exists():
        amendment = validate_race_amendment(
            run_dir,
            source_rebindings=det1_7_auth["det1_5_source_rebindings"],
        )
    return {
        "schema": "grm.det1_5.inventory.v1",
        "status": "PASS_CPU_INVENTORY",
        "created_utc": utc_now(),
        "run_dir": str(run_dir),
        "campaign_root": str(campaign_root(run_dir)),
        "det1_4_inventory_status": prior.get("status"),
        "parent_zero_receipt": file_record(parent_zero_path),
        "parent_zero_status": parent_zero["substrate_comparison"]["status"],
        "race_amendment": amendment,
        "source_amendment": {
            "record": file_record(DELTA_AMENDMENT),
            "superseded_by": det1_7_auth["record"],
        },
        "det1_7_precollection_authorization": det1_7_auth,
        "campaign_drivers": campaign_driver_inventory(),
        "stages": stages,
        "historical_rows_reused": False,
        "gpu_allocations_attempted": 0,
    }


def gpu_preflight(run_dir: Path = FROZEN_RUN, *, write_receipt: bool = False) -> tuple[dict[str, Any], int]:
    """Use the DET1.4 frozen-device preflight without allocating a model."""
    inventory(run_dir)
    from scripts import grm_det1_4_gpu as det14

    det1_7_auth = validate_det1_7_source_authorization(run_dir)
    args = argparse.Namespace(
        run_dir=Path(run_dir),
        require_amendment=True,
        write_receipt=False,
        det1_6_source_rebindings=det1_7_auth[
            "det1_4_source_rebindings"],
    )
    value, code = det14.gpu_preflight(args)
    result = {
        "schema": "grm.det1_5.gpu_preflight.v1",
        "status": "READY" if code == 0 else "BLOCKED",
        "created_utc": utc_now(),
        "det1_4_preflight": value,
        "source_amendment": file_record(DELTA_AMENDMENT),
        "det1_7_precollection_authorization": det1_7_auth["record"],
        "gpu_allocations_attempted": 0,
    }
    if write_receipt:
        directory = campaign_root(run_dir) / "preflight"
        path = write_content_addressed(directory, "gpu_preflight", result)
        result["receipt_file"] = file_record(path)
    return result, 0 if code == 0 else 2


def _base_env() -> dict[str, str]:
    from scripts.grm_det1_gpu import _base_env as det1_base_env

    env = det1_base_env()
    env["PYTHONPATH"] = str(ROOT)
    env["GRM_DET1_FORK_FROM_SNAPSHOT_REQUIRED"] = "1"
    return env


def _run_worker(
    run_dir: Path,
    worker: str,
    attempt_dir: Path,
    *,
    lease_seconds: int,
    wait_seconds: int,
    extra: Sequence[str] = (),
) -> dict[str, Any]:
    _require(0 < int(lease_seconds) <= MAX_LEASE_SECONDS,
             "GPU lease exceeds standing cap")
    output = attempt_dir / "worker_output.json"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
        worker,
        "--run-dir", str(run_dir),
        "--attempt-dir", str(attempt_dir),
        "--output", str(output),
        "--lease-seconds", str(int(lease_seconds)),
        *[str(value) for value in extra],
    ]
    from scripts.grm_det1_gpu import gpu_lease

    completed: subprocess.CompletedProcess[bytes] | None = None
    launched = False
    outcome = "NOT_LAUNCHED"
    with gpu_lease(int(lease_seconds), int(wait_seconds)):
        try:
            launched = True
            outcome = "RUNNING"
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=_base_env(),
                check=False,
                timeout=int(lease_seconds),
            )
            outcome = "RETURNED"
        except subprocess.TimeoutExpired:
            outcome = "TIMED_OUT_AND_REAPED"
            raise
        except BaseException:
            outcome = "PARENT_OBSERVED_EXCEPTION"
            raise
        finally:
            if launched:
                finished = {
                    "schema": "grm.det1_5.worker_attempt_finished.v1",
                    "status": outcome,
                    "finished_utc": utc_now(),
                    "finished_unix_ns": time.time_ns(),
                    "worker": worker,
                    "attempt_dir": str(attempt_dir),
                    "parent_process_instance": _process_instance(),
                    "returncode": (
                        int(completed.returncode)
                        if completed is not None else None),
                }
                _write_json_exclusive_or_verify(
                    attempt_dir / "attempt_finished.json", finished)
    _require(completed is not None, f"GPU worker {worker} did not return")
    _require(completed.returncode == 0,
             f"GPU worker {worker} returned {completed.returncode}")
    _require(output.is_file(), f"GPU worker {worker} did not write output")
    value = read_json(output)
    _require(value.get("status") in ("PASS", "COMPLETE"),
             f"GPU worker {worker} did not complete")
    return value


def _next_attempt(directory: Path, prefix: str = "attempt") -> Path:
    existing = sorted(directory.glob(f"{prefix}_*")) if directory.exists() else []
    return directory / f"{prefix}_{len(existing) + 1:03d}"


def _completed_shard_output(
    shard_dir: Path,
    *,
    stage: str,
    spec: str,
) -> dict[str, Any] | None:
    """Return the one immutable completed shard, ignoring killed attempts."""
    completed: list[dict[str, Any]] = []
    for output_path in sorted(shard_dir.glob("attempt_*/worker_output.json")):
        value = read_json(output_path)
        if value.get("status") not in ("PASS", "COMPLETE"):
            continue
        finished_path = output_path.parent / "attempt_finished.json"
        if not finished_path.is_file():
            continue
        finished = read_json(finished_path)
        parent_process = finished.get("parent_process_instance") or {}
        parent_sha = str(parent_process.get("process_instance_sha256", ""))
        _require(
            finished.get("schema")
            == "grm.det1_5.worker_attempt_finished.v1"
            and finished.get("status") == "RETURNED"
            and finished.get("returncode") == 0
            and finished.get("worker") == stage.replace("_", "-")
            and Path(str(finished.get("attempt_dir", ""))).resolve()
            == output_path.parent.resolve()
            and isinstance(finished.get("finished_unix_ns"), int)
            and not isinstance(finished.get("finished_unix_ns"), bool)
            and int(finished["finished_unix_ns"]) > 0
            and len(parent_sha) == 64
            and all(character in "0123456789abcdef" for character in parent_sha),
            f"{stage}/{spec} parent finish observation did not pass",
        )
        if value.get("schema") != "grm.det1_7.worker_shard.v1":
            _require(
                value.get("schema") in (None, "grm.det1_5.worker_shard.v1"),
                f"{stage}/{spec} has an unrecognized historical shard schema",
            )
            # DET1.7 makes every earlier successful output historical.  It is
            # intentionally neither reused nor counted as a duplicate.
            continue
        _validate_worker_shard_output(value, stage=stage, spec=spec)
        completed.append(value)
    _require(
        len(completed) <= 1,
        f"{stage}/{spec} has duplicate completed shard attempts",
    )
    return completed[0] if completed else None


def _validate_worker_shard_output(
    value: Mapping[str, Any],
    *,
    stage: str,
    spec: str,
    require_delta: bool = True,
) -> Path:
    receipt_path = _validate_file_record(
        value.get("receipt_file") or {}, f"{stage}/{spec} shard receipt")
    receipt = read_json(receipt_path)
    _require(
        receipt.get("schema") == "grm.det1_7.worker_shard.v1"
        and receipt.get("status") == "COMPLETE"
        and receipt.get("stage") == stage
        and receipt.get("spec") == spec,
        f"{stage}/{spec} shard envelope drifted",
    )
    _require(
        set(value) == set(receipt) | {"receipt_file"}
        and all(value.get(key) == expected for key, expected in receipt.items()),
        f"{stage}/{spec} worker output differs from its immutable receipt",
    )
    for key, expected in (
        ("registration", file_record(REGISTRATION)),
        ("runtime_frame", file_record(RUNTIME_FRAME)),
    ):
        _require(receipt.get(key) == expected,
                 f"{stage}/{spec} shard {key} binding drifted")
    del require_delta
    expected_phase = (
        "PRECOLLECTION_SOURCE_AUTHORIZATION"
        if stage == "plant_registration"
        else "POST_REGISTRATION_EXACT_BINDING"
    )
    expected_provenance = (
        load_det1_7_precollection_context(FROZEN_RUN)
        if stage == "plant_registration"
        else {
            key: value for key, value in
            load_det1_7_registered_context(FROZEN_RUN).items()
            if key != "plant_registry"
        }
    )
    _require(receipt.get("det1_7_provenance_phase") == expected_phase,
             f"{stage}/{spec} shard DET1.7 phase drifted")
    _require(receipt.get("det1_7_provenance") == expected_provenance,
             f"{stage}/{spec} shard DET1.7 provenance drifted")
    process = str(receipt.get("process_instance_sha256", ""))
    _require(len(process) == 64 and all(
        character in "0123456789abcdef" for character in process),
        f"{stage}/{spec} shard process identity is malformed")
    return receipt_path


def _worker_gap_anchor_ns(run_dir: Path) -> int | None:
    """Return the latest parent-observed child end, including failed attempts."""
    campaign = campaign_root(run_dir)
    outputs = list(campaign.glob("**/attempt_*/worker_output.json"))
    finished = list(campaign.glob("**/attempt_*/attempt_finished.json"))
    if not outputs and not finished:
        return None
    anchors = [path.stat().st_mtime_ns for path in outputs]
    for path in finished:
        value = read_json(path)
        process = value.get("parent_process_instance") or {}
        process_sha = str(process.get("process_instance_sha256", ""))
        _require(
            value.get("schema") == "grm.det1_5.worker_attempt_finished.v1"
            and value.get("status") in {
                "RETURNED", "TIMED_OUT_AND_REAPED", "PARENT_OBSERVED_EXCEPTION"
            }
            and isinstance(value.get("finished_unix_ns"), int)
            and not isinstance(value.get("finished_unix_ns"), bool)
            and int(value["finished_unix_ns"]) > 0
            and isinstance(value.get("worker"), str)
            and bool(value.get("worker"))
            and Path(str(value.get("attempt_dir", ""))).resolve()
            == path.parent.resolve()
            and len(process_sha) == 64
            and all(character in "0123456789abcdef" for character in process_sha),
            f"worker attempt-finished record drifted: {path}",
        )
        anchors.append(max(path.stat().st_mtime_ns,
                           int(value["finished_unix_ns"])))
    return max(anchors)


def _enforce_worker_gap(run_dir: Path, gap_seconds: int) -> None:
    """Keep the standing wall-clock gap across resumptions and stage CLIs."""
    gap = int(gap_seconds)
    _require(gap >= 0, "inter-worker gap must be nonnegative")
    if gap == 0:
        return
    last_written_ns = _worker_gap_anchor_ns(run_dir)
    if last_written_ns is None:
        return
    remaining_ns = last_written_ns + gap * 1_000_000_000 - time.time_ns()
    _require(remaining_ns <= (gap + 5) * 1_000_000_000,
             "worker gap anchor is implausibly future-dated")
    if remaining_ns > 0:
        time.sleep(remaining_ns / 1_000_000_000)


def _calibration_thresholds(
    directory: Path,
    rows: Sequence[Mapping[str, Any]],
    rows_path: Path,
    run_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Fit once and revalidate the immutable pre-eval threshold freeze."""
    existing = sorted(directory.glob("thresholds_*.json"))
    _require(len(existing) <= 1,
             "calibration has duplicate threshold receipts")
    fitted = fit_thresholds(rows)
    expected_projection = {
        key: fitted[key]
        for key in (
            "fit_fixture_ids", "fit_row_ids", "fit_side", "D-NGH", "D-ENT"
        )
    }
    if existing:
        path = existing[0]
        value = read_json(path)
        _require(path.stem.endswith(file_record(path)["sha256"][:16]),
                 "calibration threshold receipt is not content-addressed")
        _require(value.get("schema") == "grm.det1_5.thresholds.v1",
                 "calibration threshold schema drifted")
        _require(value.get("frozen_before_eval") is True,
                 "calibration threshold is not frozen")
        for key, expected in expected_projection.items():
            _require(value.get(key) == expected,
                     f"calibration threshold fit drifted: {key}")
        for key, expected in (
            ("registration", file_record(REGISTRATION)),
            ("runtime_frame", file_record(RUNTIME_FRAME)),
            ("race_authorization_amendment", file_record(
                Path(run_dir).resolve() / RACE_AMENDMENT.name)),
            ("source_amendment", file_record(
                Path(run_dir).resolve() / DELTA_AMENDMENT.name)),
            ("precollection_authorization", file_record(
                Path(run_dir).resolve() / DET1_7_SOURCE_AUTH.name)),
            ("plant_registry", file_record(plant_registry_path(run_dir))),
            ("terminal_amendment", file_record(
                Path(run_dir).resolve() / DET1_7_TERMINAL_AMENDMENT.name)),
            ("calibration_rows", file_record(rows_path)),
        ):
            _require(value.get(key) == expected,
                     f"calibration threshold binding drifted: {key}")
        return path, value
    value = {
        **fitted,
        "schema": "grm.det1_5.thresholds.v1",
        "frozen_before_eval": True,
        "registration": file_record(REGISTRATION),
        "runtime_frame": file_record(RUNTIME_FRAME),
        "race_authorization_amendment": file_record(
            Path(run_dir).resolve() / RACE_AMENDMENT.name),
        "source_amendment": file_record(
            Path(run_dir).resolve() / DELTA_AMENDMENT.name),
        "precollection_authorization": file_record(
            Path(run_dir).resolve() / DET1_7_SOURCE_AUTH.name),
        "plant_registry": file_record(plant_registry_path(run_dir)),
        "terminal_amendment": file_record(
            Path(run_dir).resolve() / DET1_7_TERMINAL_AMENDMENT.name),
        "calibration_rows": file_record(rows_path),
    }
    return write_content_addressed(directory, "thresholds", value), value


def cross_process_zero(
    run_dir: Path = FROZEN_RUN,
    *,
    lease_seconds: int = LEASE_SECONDS,
    wait_seconds: int = DEFAULT_WAIT_SECONDS,
) -> dict[str, Any]:
    """CPU-verify the already completed lived/fork processes in ZERO PASS."""
    del lease_seconds, wait_seconds
    if _stage_is_complete(run_dir, "cross_process_zero"):
        return _validate_stage_marker(run_dir, "cross_process_zero")[2]
    parent_path, parent = _parent_zero()
    directory = campaign_root(run_dir) / "cross_process_zero"
    lived_path = _validate_file_record(
        parent.get("lived_snapshot") or {}, "ZERO lived snapshot")
    fork_path = _validate_file_record(
        parent.get("fork_snapshot") or {}, "ZERO fork snapshot")
    lived = read_json(lived_path)
    fork = read_json(fork_path)
    lived_process = str((lived.get("provenance") or {}).get(
        "process_instance_sha256", ""))
    fork_process = str((fork.get("provenance") or {}).get(
        "process_instance_sha256", ""))
    _require(len(lived_process) == 64 and len(fork_process) == 64,
             "ZERO manifests lack process-instance provenance")
    binding = str(parent["lived_snapshot"]["sha256"])
    source_projection = {
        **parent,
        "process_instance_sha256": lived_process,
        "canonical_state_sha256": binding,
    }
    fork_projection = {
        **parent,
        "process_instance_sha256": fork_process,
        "canonical_state_sha256": binding,
    }
    validated = validate_cross_process_zero(source_projection, fork_projection)
    counts = (parent.get("substrate_comparison") or {}).get("counts") or {}
    _require(counts == {"EQUAL": 345},
             f"ZERO canonical comparator count drifted: {counts}")
    receipt = {
        **validated,
        "created_utc": utc_now(),
        "order": file_record(ORDER),
        "registration": file_record(REGISTRATION),
        "runtime_frame": file_record(RUNTIME_FRAME),
        "prior_amendment": file_record(PRIOR_AMENDMENT),
        "parent_zero_gate": file_record(parent_path),
        "zero_gate": file_record(parent_path),
        "lived_snapshot": file_record(lived_path),
        "fork_snapshot": file_record(fork_path),
        "lived_process_instance_sha256": lived_process,
        "fork_process_instance_sha256": fork_process,
        "substrate_status": ZERO_PASS_STATUS,
        "substrate_counts": {"EQUAL": 345},
        "parent_substrate_status": ZERO_PASS_STATUS,
        "child_substrate_status": ZERO_PASS_STATUS,
        "canonical_comparison_counts": {"EQUAL": 345},
        "verification_mode": "CPU_EXISTING_PASS_MANIFEST_PROVENANCE",
        "gpu_allocations_attempted": 0,
        "race_resume_authorized": False,
    }
    receipt_path = write_content_addressed(directory, "cross_process_zero", receipt)
    _write_stage_marker(
        run_dir,
        "cross_process_zero",
        receipt_path,
        [PARENT_ZERO_MARKER, lived_path, fork_path],
    )
    return receipt


def _cross_zero_worker(args: argparse.Namespace) -> dict[str, Any]:
    """Fresh-process reuse of the qualified DET1.4 ZERO implementation."""
    from scripts import grm_det1_4_gpu as det14

    delta_amendment = validate_delta_amendment(Path(args.run_dir))
    # _run_zero_gate uses FROZEN_RUN only for its append-only attempt output;
    # all evidence inputs remain DET1.4's separately frozen absolute bindings.
    det14.FROZEN_RUN = Path(args.attempt_dir) / "worker_namespace"
    value, code = det14._run_zero_gate(argparse.Namespace(
        run_dir=Path(args.run_dir),
        lease_seconds=int(args.lease_seconds),
        lock_wait_seconds=0,
        require_amendment=True,
        write_receipt=False,
        det1_6_source_rebindings=delta_amendment["source_rebindings"],
    ))
    _require(code == 0 and value.get("status") == "PASS",
             "fresh-process DET1.4 ZERO worker failed")
    return value


def _sha_array(value: Any) -> str:
    import numpy as np

    array = value if isinstance(value, np.ndarray) else value.numpy()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _live_token_ids(arena: Any) -> list[int]:
    """Recover the exact current live-token ledger or fail before capture."""
    values: list[int] = []
    for graft_index, count in arena.live_segs:
        _require(graft_index is not None,
                 "inline capture encountered an anonymous live segment")
        graft = arena.grafts[int(graft_index)]
        ids = [int(value) for value in arena.encode(str(graft.get("text", "")))]
        _require(len(ids) == int(count),
                 "inline capture live-token ledger length differs from cache")
        values.extend(ids)
    return values


def _attempt_schedule(
    arena: Any,
    e2e: Any,
    question: str,
    profile: Mapping[str, Any],
    *,
    topk: int,
    max_trips: int,
) -> list[tuple[list[int], bool]]:
    """Use the registered production helpers to enumerate every ladder rung."""
    ranking = [int(value) for value in profile["raw_ranking"]]
    identifiers = e2e._probe_identifier_tokens(arena, question)
    point_lookup = bool(identifiers)
    precise = e2e._probe_rank1_covers_identifiers(
        arena, ranking, identifiers)
    baseline = e2e.build_probe_ladder_attempts(
        ranking=ranking,
        topk=max(1, int(topk)),
        precise=precise,
        point_lookup=point_lookup,
        max_trips=int(max_trips),
    )
    admission = profile.get("served_admission_profile")
    if not admission:
        attempts = baseline
    else:
        first = (
            [int(value) for value in admission["rank_plan"]],
            point_lookup,
        )
        attempts = [first]
        for candidate in [*baseline[1:], *baseline[:1]]:
            normalized = (
                [int(value) for value in candidate[0]], bool(candidate[1]))
            if normalized not in attempts:
                attempts.append(normalized)
        attempts = attempts[:max(1, int(max_trips) + 1)]
    _require(bool(attempts), "production probe ladder is empty")
    return [(list(picks), bool(clean)) for picks, clean in attempts]


def _admission_snapshot(
    *,
    arena: Any,
    profile: Mapping[str, Any],
    schedule: Sequence[tuple[Sequence[int], bool]],
    ordinal: int,
    planned: Sequence[int],
    fitted: Sequence[int],
    probe_key_sha256: str,
) -> dict[str, Any]:
    admission = dict(profile.get("served_admission_profile") or {})
    ranking = [int(value) for value in admission.get(
        "ranking", profile.get("raw_ranking", ()))]
    rank_plan = [int(value) for value in admission.get(
        "rank_plan", profile.get("served_effective_admission_plan", ()))]
    actual = [int(value) for value in arena.cur_mounts]
    return {
        "schema": "grm.det1_3.admission_snapshot.v2",
        "probe_key_sha256": probe_key_sha256,
        "ranking": ranking,
        "identifier_tokens": [
            str(value) for value in admission.get("identifier_tokens", ())],
        "identified_candidates": [
            int(value) for value in admission.get("identified_candidates", ())],
        "policy_branch": admission.get("policy_branch"),
        "rank_plan": rank_plan,
        "ladder_attempts": [
            {
                "ordinal": int(index),
                "planned": [int(value) for value in picks],
                "clean_room": bool(clean),
            }
            for index, (picks, clean) in enumerate(schedule)
        ],
        "current_attempt_ordinal": int(ordinal),
        "current_planned": [int(value) for value in planned],
        "current_clean_room": bool(schedule[int(ordinal)][1]),
        "current_fitted": actual,
        "current_dropped": [
            int(value) for value in planned if int(value) not in set(actual)],
        "selection_state": "PENDING_AT_PREFILL",
        "final_mounts": actual,
        "route_backend": admission.get("route_backend", profile.get("route_backend")),
        "route_margin_1_2": admission.get("route_margin_1_2"),
        "margin_threshold": admission.get("margin_threshold"),
        "rule_sha256": admission.get("rule_sha256"),
    }


def _answer_correct(answer: str, fixture: Mapping[str, Any]) -> bool:
    expected = [str(value) for value in fixture.get("expected_values", ())]
    rejected = [
        *[str(value) for value in fixture.get(
            "stale_values", fixture.get("old_values", ()))],
        *[str(value) for value in fixture.get("wrong_fact_values", ())],
    ]
    return bool(
        any(contains_value(answer, value) for value in expected)
        and not any(contains_value(answer, value) for value in rejected)
    )


def _fork_attempt(
    *,
    arena: Any,
    e2e: Any,
    fixture: Mapping[str, Any],
    profile: Mapping[str, Any],
    schedule: Sequence[tuple[Sequence[int], bool]],
    ordinal: int,
    planned: Sequence[int],
    fitted: Sequence[int],
    original_attempt: Any,
    identity: Mapping[str, Any],
    snapshot_dir: Path,
    runtime: Mapping[str, Any],
    process: Mapping[str, Any],
    capture_prompt: str,
    active_detectors: Sequence[str],
    variants: Sequence[str],
    verbal: bool,
    ngen: int,
) -> dict[str, dict[str, Any]]:
    """Capture one lived rung, then consume all variants inline in-process."""
    from scripts.grm_det1_3_snapshot import (
        ARM_PROTOCOLS,
        capture_arena_snapshot,
        compare_fork_hydration_delta,
        finalize_snapshot_with_answer,
        load_snapshot,
        restore_prefill_fork,
        stop_at_next_forward,
    )
    from scripts.grm_det1_4_gpu import _continue_forked_prefill
    from scripts.grm_det1_common import DetectorObserver, verbal_result
    from scripts.grm_det1_e2e import _PredictionObserver
    from scripts.grm_det1_3_gpu import _is_refusal

    aliases = {int(value) for value in profile["logical_alias_ids"]}
    question = str(fixture["question"])
    probe_key = arena._probe_key(question)
    probe_key_sha256 = _sha_array(probe_key)
    live_ids = _live_token_ids(arena)
    provenance = {
        "schema": "grm.det1_3.capture_provenance.v2",
        "arm": "lived",
        "protocol": ARM_PROTOCOLS["lived"],
        "protocol_source_sha256": file_record(Path(__file__).resolve())["sha256"],
        "run_id": FROZEN_RUN.name,
        "order_sha256": file_record(ORDER)["sha256"],
        "det1_7_order_sha256": file_record(DET1_7_ORDER)["sha256"],
        "source_amendment_sha256": file_record(DELTA_AMENDMENT)["sha256"],
        "det1_7_precollection_authorization_sha256": file_record(
            DET1_7_SOURCE_AUTH)["sha256"],
        "race_authorization_amendment_sha256": (
            file_record(RACE_AMENDMENT)["sha256"]
        ),
        "registration_sha256": file_record(REGISTRATION)["sha256"],
        "runtime_frame_sha256": file_record(RUNTIME_FRAME)["sha256"],
        "fixture_sha256": str((fixture.get("source") or {}).get("sha256", "")),
        "fixture_id": str(fixture["fixture_id"]),
        "probe_id": str(fixture.get("probe_id", fixture["fixture_id"])),
        "probe_question_sha256": hashlib.sha256(
            capture_prompt.encode("utf-8")).hexdigest(),
        "capture_boundary": "before_probe_prefill",
        "selection_boundary": "before_generation_and_grounding_selection",
        "probe_driver": "grm_det1_5.inline_full_ladder",
        "topk": int(getattr(arena, "topk", 3)),
        "max_trips": len(schedule) - 1,
        "probe_ladder": True,
        "defer_memory": True,
        "attempt_ordinal": int(ordinal),
        **dict(process),
    }
    if profile.get("plant_registry"):
        provenance.update({
            "det1_7_plant_registry_sha256": str(
                (profile.get("plant_registry") or {}).get("sha256", "")),
            "det1_7_terminal_amendment_sha256": file_record(
                DET1_7_TERMINAL_AMENDMENT)["sha256"],
        })

    def admission_supplier() -> dict[str, Any]:
        return _admission_snapshot(
            arena=arena,
            profile=profile,
            schedule=schedule,
            ordinal=ordinal,
            planned=planned,
            fitted=fitted,
            probe_key_sha256=probe_key_sha256,
        )

    with stop_at_next_forward(
        arena,
        snapshot_dir,
        label="lived",
        provenance=lambda: provenance,
        question=capture_prompt,
        admission_plan=admission_supplier,
        live_token_ids=live_ids,
        sink_text=e2e.HARMONY_SINK,
        sink_token_ids=arena.encode(e2e.HARMONY_SINK),
        explicit_identity=identity,
    ) as captured:
        lived_answer, lived_info = original_attempt(
            capture_prompt,
            [int(value) for value in fitted],
            int(ngen),
            False,
            arena.stop_sequences or (),
            defer_memory=True,
        )
    manifest_path = Path(captured["manifest_path"])
    linked = {
        "schema": "grm.det1_5.inline_lived_answer.v1",
        "process_instance_sha256": process["process_instance_sha256"],
        "captured_attempt_ordinal": int(ordinal),
        "attempt_answer": str(lived_answer),
        "attempt_mounts": [int(value) for value in arena.cur_mounts],
        "attempt_answer_correct": _answer_correct(str(lived_answer), fixture),
        "attempt_refusal": _is_refusal(str(lived_answer)),
        "probe_answer": str(lived_answer),
        "probe_selected_attempt": int(ordinal),
        "probe_mounts": [int(value) for value in arena.cur_mounts],
        "probe_answer_correct": _answer_correct(str(lived_answer), fixture),
        "probe_refusal": _is_refusal(str(lived_answer)),
    }
    manifest_path = finalize_snapshot_with_answer(manifest_path, linked)
    snapshot = load_snapshot(manifest_path)
    source_mounts = {
        int(value) for value in
        (snapshot.get("state") or {}).get("arena.cur_mounts", ())
    }
    results: dict[str, dict[str, Any]] = {}
    for variant in variants:
        _require(variant in ("served", "planted_miss"),
                 f"unsupported fork variant: {variant}")
        source_mounted_withheld = (
            sorted(aliases & source_mounts) if variant == "planted_miss" else []
        )
        withheld_for_rung = (
            sorted(aliases) if source_mounted_withheld else []
        )
        restore = restore_prefill_fork(
            arena,
            manifest_path,
            withheld_mounts=withheld_for_rung,
            target_identity=identity,
            expected_frame_sha256=str(snapshot["identity"]["frame_sha256"]),
            require_same_process_index=True,
        )
        if withheld_for_rung:
            delta_provenance = {
                **provenance,
                "schema": "grm.det1_4.fork_capture_provenance.v1",
                "arm": "fork",
                "protocol": ARM_PROTOCOLS["fork"],
                "probe_driver": "grm_det1_6.inline_fork_hydration_delta",
            }
            delta_manifest = capture_arena_snapshot(
                arena,
                snapshot_dir / "planted_delta_fork",
                label="fork",
                provenance=delta_provenance,
                question=capture_prompt,
                prompt_ids=restore["prompt_ids"],
                admission_plan=restore["admission_state"],
                live_token_ids=restore["live_token_ids"],
                sink_text=e2e.HARMONY_SINK,
                sink_token_ids=restore["sink_token_ids"],
                explicit_identity=identity,
            )
            delta_receipt = compare_fork_hydration_delta(
                manifest_path,
                delta_manifest,
                withheld_mounts=withheld_for_rung,
            )
            _require(
                delta_receipt.get("schema") == DELTA_RECEIPT_SCHEMA
                and delta_receipt.get("status") == DELTA_PASS_STATUS
                and delta_receipt.get("gate_pass") is True,
                "fork-hydration delta comparator failed closed for "
                f"{fixture['fixture_id']} rung {ordinal}: "
                f"status={delta_receipt.get('status')!r}",
            )
            restore["delta_receipt"] = dict(delta_receipt)
            restore["delta_fork_snapshot"] = file_record(delta_manifest)
        if verbal:
            verbal_started_unix_ns = time.time_ns()
            with _PredictionObserver(arena, int(ngen)) as observer:
                answer, masks = _continue_forked_prefill(
                    arena, restore["prompt_ids"], int(ngen))
            token_ids = observer.finish()
            verbal_payload = verbal_result(answer, token_ids, arena.decode)
            signals = None
        else:
            verbal_started_unix_ns = None
            selected = tuple(str(value) for value in active_detectors)
            if selected:
                with DetectorObserver(
                    arena,
                    aliases,
                    int(ngen),
                    active_detectors=selected,
                ) as observer:
                    answer, masks = _continue_forked_prefill(
                        arena, restore["prompt_ids"], int(ngen))
                signals = observer.finish()
            else:
                answer, masks = _continue_forked_prefill(
                    arena, restore["prompt_ids"], int(ngen))
                signals = {
                    "schema": "grm.det1.signals.v1",
                    "active_detectors": [],
                    "ngh_schedule": None,
                    "token_indexing": "zero_based_generated_prediction",
                    "tokens": [],
                    "prediction_token_ids": [],
                    "unused_final_cache_flush_dropped": False,
                }
            verbal_payload = None
        mounts = [int(value) for value in arena.cur_mounts]
        mounted_catalog = {
            str(index): {
                "graft_id": int(index),
                "text": str(arena.grafts[index].get("text", "")),
                "text_sha256": hashlib.sha256(
                    str(arena.grafts[index].get("text", "")).encode("utf-8")
                ).hexdigest(),
            }
            for index in mounts
        }
        mounted_text = "\n".join(
            str(mounted_catalog[str(index)]["text"]) for index in mounts)
        grounded, _contributors = arena._grounding_attribution(
            answer, mounts, capture_prompt)
        results[variant] = {
            "answer": str(answer),
            "answer_correct": _answer_correct(str(answer), fixture),
            "mounted_ids": mounts,
            "mounted_graft_catalog": mounted_catalog,
            "mounted_contains_expected": any(
                contains_value(mounted_text, str(value))
                for value in fixture.get("expected_values", ())),
            "full_index_contains_all_aliases": aliases <= {
                int(value) for value in arena._route_cand_base()},
            "signals": signals,
            "verbal": verbal_payload,
            "fork_restore": restore,
            "mask_consumption": masks,
            "grounded": bool(grounded),
            "attempt_ordinal": int(ordinal),
            "planned": [int(value) for value in planned],
            "requested_picks": [
                int(value) for value in fitted
                if variant != "planted_miss" or int(value) not in aliases
            ],
            "snapshot": file_record(manifest_path),
            "lived_answer": str(lived_answer),
            "lived_info": dict(lived_info or {}),
            "process_instance_sha256": process["process_instance_sha256"],
            "verbal_started_unix_ns": verbal_started_unix_ns,
            "registered_aliases_requested": (
                sorted(aliases) if variant == "planted_miss" else []),
            "withheld_aliases_on_this_rung": withheld_for_rung,
            "source_mounted_withheld_aliases": source_mounted_withheld,
        }
    return results


def _registration_target_candidate(
    *,
    fixture: Mapping[str, Any],
    result: Mapping[str, Any],
) -> tuple[int, dict[str, list[str]], str]:
    """Derive the candidate solely from the selected lived snapshot/mounts."""
    from scripts.grm_det1_3_snapshot import load_snapshot

    snapshot_record = result.get("snapshot") or {}
    snapshot_path = _path_from_record(snapshot_record)
    snapshot = load_snapshot(snapshot_path)
    state = snapshot.get("state") or {}
    actual = [int(value) for value in state.get(
        "admission.authoritative_mounts", ())]
    for field in (
        "admission.current_fitted", "admission.final_mounts", "arena.cur_mounts",
    ):
        _require([int(value) for value in state.get(field, ())] == actual,
                 f"registration lived mount projection differs at {field}")
    _require(actual == [int(value) for value in result.get("mounted_ids", ())],
             "selected served mounts differ from its lived snapshot")
    catalog = result.get("mounted_graft_catalog") or {}
    expected = [str(value) for value in fixture.get("expected_values", ())]
    coverage: dict[str, list[str]] = {}
    for graft_id in actual:
        record = catalog.get(str(graft_id)) or {}
        text_value = str(record.get("text", ""))
        hits = [value for value in expected if contains_value(text_value, value)]
        if hits:
            coverage[str(graft_id)] = hits

    rank_plan = [int(value) for value in state.get("admission.rank_plan", ())]
    planned = [int(value) for value in state.get(
        "admission.current_planned", ())]
    order = list(dict.fromkeys([*rank_plan, *planned, *sorted(actual)]))
    branch = str(state.get("admission.policy_branch", ""))
    if branch == "declared_synthesis_identified_set":
        identified = {int(value) for value in state.get(
            "admission.identified_candidates", ())}
        candidates = [
            value for value in order if value in identified and value in actual
        ]
        expected_keys = {normalize_value_text(value) for value in expected}
        essential: list[int] = []
        for candidate in candidates:
            remaining = {
                normalize_value_text(value)
                for graft_id, values in coverage.items()
                if int(graft_id) != candidate
                for value in values
            }
            if not expected_keys.issubset(remaining):
                essential.append(candidate)
        _require(len(essential) == 1,
                 "declared synthesis has no unique mounted expected-value "
                 f"breaker candidate: {essential}")
        return essential[0], coverage, "DECLARED_SYNTHESIS_BREAKING_MEMBER"

    target = next((value for value in order if value in actual), None)
    _require(target is not None,
             "lived admission has no mounted winner in admission order")
    _require(bool(coverage.get(str(target))),
             "lived admission winner does not carry the registered expected value")
    return int(target), coverage, "ORDINARY_ACTUAL_WINNER"


def _qualify_registration_target(
    *,
    arena: Any,
    e2e: Any,
    fixture: Mapping[str, Any],
    result: Mapping[str, Any],
    target_id: int,
    identity: Mapping[str, Any],
    process: Mapping[str, Any],
    capture_prompt: str,
    ngen: int,
) -> dict[str, Any]:
    """Run an explicit one-member fork ablation for synthesis registration."""
    from scripts.grm_det1_3_snapshot import (
        capture_arena_snapshot,
        compare_fork_hydration_delta,
        load_snapshot,
        restore_prefill_fork,
    )
    from scripts.grm_det1_4_gpu import _continue_forked_prefill
    from scripts.grm_det1_3_gpu import _is_refusal

    source_record = dict(result.get("snapshot") or {})
    source_path = _path_from_record(source_record)
    source = load_snapshot(source_path)
    actual_ordered = [
        int(value) for value in
        (source.get("state") or {}).get("admission.authoritative_mounts", ())
    ]
    actual = set(actual_ordered)
    _require(int(target_id) in actual,
             "registration ablation target has no lived source seat")
    restore = restore_prefill_fork(
        arena,
        source_path,
        withheld_mounts=[int(target_id)],
        target_identity=identity,
        expected_frame_sha256=str(source["identity"]["frame_sha256"]),
        require_same_process_index=True,
    )
    qualification_dir = source_path.parent / "registration_target_ablation"
    provenance = {
        **dict(source.get("provenance") or {}),
        "schema": "grm.det1_7.registration_ablation_provenance.v1",
        "arm": "fork",
        "probe_driver": "grm_det1_7.target_only_registration_qualification",
        "withheld_target_id": int(target_id),
        "process_instance_sha256": process["process_instance_sha256"],
    }
    fork_manifest = capture_arena_snapshot(
        arena,
        qualification_dir,
        label="fork",
        provenance=provenance,
        question=capture_prompt,
        prompt_ids=restore["prompt_ids"],
        admission_plan=restore["admission_state"],
        live_token_ids=restore["live_token_ids"],
        sink_text=e2e.HARMONY_SINK,
        sink_token_ids=restore["sink_token_ids"],
        explicit_identity=identity,
    )
    delta = compare_fork_hydration_delta(
        source_path, fork_manifest, withheld_mounts=[int(target_id)])
    _require(
        delta.get("schema") == DELTA_RECEIPT_SCHEMA
        and delta.get("status") == DELTA_PASS_STATUS
        and delta.get("gate_pass") is True,
        "registration target-only ablation failed exact delta validation",
    )
    answer, _masks = _continue_forked_prefill(
        arena, restore["prompt_ids"], int(ngen))
    answer_correct = _answer_correct(str(answer), fixture)
    refusal = _is_refusal(str(answer))
    breaks_probe = not answer_correct or refusal
    stale_values = [str(value) for value in fixture.get(
        "stale_values", fixture.get("old_values", ()))]
    wrong_values = [str(value) for value in fixture.get(
        "wrong_fact_values", ())]
    if refusal:
        classification = "refusal"
    elif any(contains_value(str(answer), value) for value in stale_values):
        classification = "stale"
    elif any(contains_value(str(answer), value) for value in wrong_values):
        classification = "wrong_fact"
    else:
        classification = "incorrect"
    return {
        "schema": "grm.det1_7.target_only_behavioral_breaker.v1",
        "source_manifest_payload_sha256": source[
            "manifest_payload_sha256"],
        "withheld_target_id": int(target_id),
        "withheld_alias_ids": [int(target_id)],
        "served": {
            "answer": str((source.get("linked_answer") or {})[
                "probe_answer"]),
            "answer_correct": True,
            "refusal": False,
            "mounted_ids": actual_ordered,
        },
        "counterfactual": {
            "answer": str(answer),
            "answer_correct": bool(answer_correct),
            "refusal": bool(refusal),
            "mounted_ids": [int(value) for value in arena.cur_mounts],
            "classification": classification,
        },
        "delta_receipt": dict(delta),
    }


def _select_ladder(
    attempts: Sequence[Mapping[str, Any]], variant: str,
) -> tuple[dict[str, Any], int]:
    candidates = [dict(value[variant]) for value in attempts]
    _require(bool(candidates), f"no forked ladder attempts for {variant}")
    selected = next((row for row in candidates if row["grounded"]), candidates[0])
    return selected, int(selected["attempt_ordinal"])


def _mechanistic_row(
    *,
    fixture: Mapping[str, Any],
    variant: str,
    split: str,
    profile: Mapping[str, Any],
    result: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    started_ns: int,
    completed_ns: int,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    aliases = {int(value) for value in profile["logical_alias_ids"]}
    mounted = {int(value) for value in result["mounted_ids"]}
    planted = variant == "planted_miss"
    signals = dict(result["signals"] or {})
    signals["ladder_attempt_selection"] = {
        "production_trip": int(result["attempt_ordinal"]),
        "selected_call_ordinal": int(result["attempt_ordinal"]),
        "attempt_count": len(attempts),
        "attempts": [
            {
                "call_ordinal": int(row[variant]["attempt_ordinal"]),
                "requested_picks": list(row[variant]["requested_picks"]),
                "mounted_ids": list(row[variant]["mounted_ids"]),
                "answer_prefix": str(row[variant]["answer"])[:160],
            }
            for row in attempts
        ],
    }
    admission = dict(profile.get("served_admission_profile") or {})
    unfiltered = [int(value) for value in admission.get(
        "rank_plan", profile.get("raw_ranking", ()))]
    filtered = [value for value in unfiltered if value not in aliases]
    selected_restore = dict(result.get(
        "selected_rung_restore", result.get("fork_restore") or {}))
    restored_admission = dict(selected_restore.get("admission_state") or {})
    ranking_unchanged = (
        [int(value) for value in restored_admission.get("ranking", ())]
        == [int(value) for value in admission.get("ranking", ())]
    )
    branch_unchanged = (
        restored_admission.get("policy_branch")
        == admission.get("policy_branch")
    )
    selected_source_mounts = [int(value) for value in
                              selected_restore.get("source_mounts", ())]
    selected_kept_mounts = [
        value for value in selected_source_mounts if value not in aliases]
    only_aliases_removed = bool(
        [int(value) for value in selected_restore.get("fork_mounts", ())]
        == selected_kept_mounts
        and [int(value) for value in result.get("requested_picks", ())]
        == selected_kept_mounts
        and [int(value) for value in restored_admission.get(
            "current_fitted", selected_kept_mounts)] == selected_kept_mounts
    )
    return {
        "schema": "grm.det1_5.mechanistic_row.v1",
        "row_id": f"{fixture['fixture_id']}:{variant}",
        "fixture_id": str(fixture["fixture_id"]),
        "source_family": str(fixture["source_family"]),
        "split": split,
        "stage": split,
        "variant": variant,
        "label": int(planted),
        "evaluation_started_unix_ns": int(started_ns),
        "mechanistic_completed_unix_ns": int(completed_ns),
        "process_instance_sha256": str(result["process_instance_sha256"]),
        "question": str(fixture["question"]),
        "expected_values": [str(value) for value in fixture["expected_values"]],
        "stale_values": [str(value) for value in fixture.get(
            "stale_values", fixture.get("old_values", ()))],
        "wrong_fact_values": [
            str(value) for value in fixture.get("wrong_fact_values", ())],
        "raw_router_rank1": int(profile["raw_rank1"]),
        "logical_router_rank1": int(profile["logical_router_rank1"]),
        "production_admitted_rank1": int(profile["production_admitted_rank1"]),
        "router_rank1_admitted": bool(profile["router_rank1_admitted"]),
        "raw_ranking_ids": [int(value) for value in profile["raw_ranking"]],
        "post_l2_ranking_ids": [int(value) for value in profile["effective_ranking"]],
        "logical_alias_ids": sorted(aliases),
        "plant_target_id": int(profile["logical_router_rank1"]),
        "plant_alias_ids": sorted(aliases),
        "plant_target_source": profile.get("plant_target_source"),
        "plant_registry": dict(profile.get("plant_registry") or {}),
        "plant_entry_sha256": profile.get("plant_entry_sha256"),
        "target_contains_expected": bool(profile["target_contains_expected"]),
        "served_admission_profile": admission,
        "served_effective_admission_plan": [
            int(value) for value in profile["served_effective_admission_plan"]],
        "expected_planted_admission_plan": filtered,
        "production_route_limit": int(profile["production_route_limit"]),
        "route_backend": str(profile["route_backend"]),
        "answer": str(result["answer"]),
        "answer_correct": bool(result["answer_correct"]),
        "mounted_ids": sorted(mounted),
        "mounted_contains_expected": bool(result["mounted_contains_expected"]),
        "full_index_contains_all_aliases": bool(
            result["full_index_contains_all_aliases"]),
        "signals": signals,
        "fork_restore": dict(result["fork_restore"]),
        "fork_source_snapshot": dict(result["snapshot"]),
        "plant_checks": {
            "applicable": planted,
            "withheld_aliases_absent": not bool(aliases & mounted) if planted else None,
            "logical_target_absent": (
                int(profile["logical_router_rank1"]) not in mounted
                if planted else None),
            "registered_plant_target_absent": (
                int(profile["logical_router_rank1"]) not in mounted
                if planted else None),
            "registered_plant_aliases_absent": (
                not bool(aliases & mounted) if planted else None),
            "raw_router_rank1_absent": (
                int(profile["raw_rank1"]) not in mounted if planted else None),
            "expected_value_absent_from_mounted_text": (
                not bool(result["mounted_contains_expected"]) if planted else None),
            "withheld_aliases_remain_in_full_detector_index": (
                bool(result["full_index_contains_all_aliases"]) if planted else None),
            "admission_ranking_unchanged": (
                ranking_unchanged if planted else None),
            "admission_branch_unchanged": (
                branch_unchanged if planted else None),
            "only_aliases_removed_from_admission_plan": (
                only_aliases_removed if planted else None),
            "withheld_aliases_absent_from_every_ladder_attempt": (
                all(
                    not aliases & (
                        set(row[variant]["mounted_ids"])
                        | set(row[variant]["requested_picks"])
                    )
                    for row in attempts
                ) if planted else None),
        },
        "runtime_frame": dict(runtime),
        "registration_path": str(REGISTRATION.resolve()),
    }


def _measure_fixture_inline(
    *,
    repo: Any,
    e2e: Any,
    original: Any,
    fixture: Mapping[str, Any],
    split: str,
    rows_path: Path,
    runtime: Mapping[str, Any],
    registration: Mapping[str, Any],
    model: Any,
    tokenizer: Any,
    topk: int,
    ngen: int,
    max_trips: int,
    turn_idx: int | None,
    canonical_defer_memory: bool,
    active_detectors: Sequence[str],
    variants: Sequence[str],
    verbal: bool,
    mechanistic_times: Mapping[str, int] | None,
    snapshot_root: Path,
    plant_registry: Mapping[str, Any] | None = None,
    registration_capture: bool = False,
) -> tuple[str, dict[str, Any]]:
    from scripts.grm_det1_e2e import (
        _restore_counterfactual,
        _snapshot_counterfactual,
    )
    from scripts.grm_det1_3_gpu import _frame_identity
    from scripts.grm_det1_common import route_fixture_profile

    arena = repo.arena
    base_fixture_id = str(fixture.get("base_fixture_id", fixture["fixture_id"]))
    if registration_capture:
        _require(
            plant_registry is None
            and tuple(variants) == ("served",)
            and not active_detectors
            and not verbal,
            "plant registration must be served-only with every detector dark",
        )
        registry_record: dict[str, Any] | None = None
        registry_entry: dict[str, Any] | None = None
    else:
        _require(isinstance(plant_registry, Mapping),
                 "race fixture opened without the frozen DET1.7 plant registry")
        registry_entry = _plant_registry_entry(plant_registry, base_fixture_id)
        _require(
            str(registry_entry.get("effective_fixture_id"))
            == str(fixture["fixture_id"]),
            f"effective fixture differs from plant registry slot {base_fixture_id}",
        )
        registry_record = file_record(plant_registry_path(FROZEN_RUN))
    live = {int(graft) for graft, _count in arena.live_segs if graft is not None}
    base = _snapshot_counterfactual(arena)
    try:
        profile = route_fixture_profile(
            arena,
            str(fixture["question"]),
            str(fixture["expected_values"][0]),
            live_excluded=live,
            route_limit=max(int(topk), (int(max_trips) + 1) * int(topk)),
        )
        if registry_entry is not None:
            profile = bind_plant_profile(
                profile, registry_entry, registry_record or {})
    finally:
        _restore_counterfactual(arena, base)
    schedule = _attempt_schedule(
        arena, e2e, str(fixture["question"]), profile,
        topk=int(topk), max_trips=int(max_trips))
    identity = _frame_identity(model, tokenizer, e2e, registration, runtime)
    process = _process_instance()
    started_ns = time.time_ns()
    attempt_rows: list[dict[str, Any]] = []
    original_attempt = arena._attempt
    capture_prompt = (
        str(fixture["question"]) + "\n\n" + VERBAL_QUESTION
        if verbal else str(fixture["question"])
    )
    try:
        for ordinal, (planned, clean) in enumerate(schedule):
            _restore_counterfactual(arena, base)
            if clean:
                arena.caches, arena.pos, arena.live_segs = None, 0, []
                arena.cur_mounts, arena.cur_mount_n = [], 0
            for layer in arena.m.layers:
                layer.self_attn.live_shift = arena.live_shift
            fitted = sorted(e2e._budget_fit_mounts(arena, planned))
            if not fitted and planned:
                continue
            attempt_rows.append(_fork_attempt(
                arena=arena,
                e2e=e2e,
                fixture=fixture,
                profile=profile,
                schedule=schedule,
                ordinal=ordinal,
                planned=planned,
                fitted=fitted,
                original_attempt=original_attempt,
                identity=identity,
                snapshot_dir=snapshot_root / f"rung_{ordinal:02d}",
                runtime=runtime,
                process=process,
                capture_prompt=capture_prompt,
                active_detectors=active_detectors,
                variants=variants,
                verbal=verbal,
                ngen=int(ngen),
            ))
        _require(bool(attempt_rows), "no production ladder rung could be captured")
        if verbal:
            _require(mechanistic_times is not None,
                     "D-VERB worker lacks mechanistic ladder bindings")
            selected = {}
            for variant in variants:
                row_id = f"{fixture['fixture_id']}:{variant}"
                binding = mechanistic_times.get(row_id)
                _require(isinstance(binding, Mapping),
                         f"D-VERB lacks mechanistic binding for {row_id}")
                selected_ordinal = int(binding["selected_attempt_ordinal"])
                candidates = [
                    dict(row[variant]) for row in attempt_rows
                    if int(row[variant]["attempt_ordinal"]) == selected_ordinal
                ]
                _require(len(candidates) == 1,
                         f"D-VERB cannot bind mechanistic ladder rung for {row_id}")
                selected[variant] = candidates[0]
                _require(
                    [int(value) for value in selected[variant]["mounted_ids"]]
                    == [int(value) for value in binding["mounted_ids"]],
                    f"D-VERB fork mounts differ from mechanistic row for {row_id}",
                )
        else:
            selected = {
                variant: _select_ladder(attempt_rows, variant)[0]
                for variant in variants
            }
        registration_observation: dict[str, Any] | None = None
        if registration_capture:
            served = selected["served"]
            observation_status = "LAWFUL_LIVED_TARGET"
            unplantable_reason: str | None = None
            target_id: int | None = None
            coverage: dict[str, list[str]] = {}
            selection_class: str | None = None
            breaker: dict[str, Any] | None = None
            from scripts.grm_det1_3_gpu import _is_refusal
            if served.get("answer_correct") is not True or _is_refusal(
                str(served.get("answer", ""))
            ):
                observation_status = "UNPLANTABLE"
                unplantable_reason = "LIVED_SERVED_CONTROL_INCORRECT_OR_REFUSAL"
            else:
                try:
                    target_id, coverage, selection_class = (
                        _registration_target_candidate(
                            fixture=fixture, result=served))
                    if selection_class == "DECLARED_SYNTHESIS_BREAKING_MEMBER":
                        breaker = _qualify_registration_target(
                            arena=arena,
                            e2e=e2e,
                            fixture=fixture,
                            result=served,
                            target_id=target_id,
                            identity=identity,
                            process=process,
                            capture_prompt=capture_prompt,
                            ngen=int(ngen),
                        )
                        if (breaker.get("counterfactual") or {}).get(
                            "answer_correct") is not False:
                            observation_status = "UNPLANTABLE"
                            unplantable_reason = (
                                "DECLARED_SYNTHESIS_MEMBER_DID_NOT_BREAK_PROBE")
                except DETError as exc:
                    observation_status = "UNPLANTABLE"
                    unplantable_reason = str(exc)
            if base_fixture_id == str(fixture["fixture_id"]):
                substitution = {"status": "NONE"}
            else:
                source = dict(fixture.get("source") or {})
                source_sha = str(source.get("sha256", ""))
                stale = fixture.get("stale_values")
                if stale is None:
                    stale = fixture.get("old_values") or []
                selector = (
                    {"turn": int(fixture["turn"])}
                    if fixture.get("source_family") == "certified_34_turn"
                    else {"probe_id": str(fixture["probe_id"])}
                )
                effective_fixture = {
                    "fixture_id": str(fixture["fixture_id"]),
                    "split": str(fixture["split"]),
                    "source_family": str(fixture["source_family"]),
                    "session_id": (
                        f"certified_34_turn:{source_sha}"
                        if fixture.get("source_family") == "certified_34_turn"
                        else str(fixture["session_id"])),
                    "selector": selector,
                    "question": str(fixture["question"]),
                    "expected_values": [str(value) for value in
                                        fixture.get("expected_values", ())],
                    "stale_values": [str(value) for value in stale],
                    "wrong_fact_values": [str(value) for value in
                                           fixture.get("wrong_fact_values", ())],
                    "source": source,
                }
                substitution = {
                    "status": "SUBSTITUTED",
                    "original_fixture_id": base_fixture_id,
                    "effective_fixture": effective_fixture,
                    "reason": str((fixture.get("substitution") or {}).get(
                        "reason", "NO_LAWFUL_LIVED_MOUNTED_TARGET")),
                }
            registration_observation = {
                "schema": "grm.det1_7.plant_observation.v1",
                "row_id": f"{fixture['fixture_id']}:plant_registration",
                "fixture_id": base_fixture_id,
                "effective_fixture_id": str(fixture["fixture_id"]),
                "candidate_for_fixture_id": str(fixture.get(
                    "candidate_for_fixture_id", base_fixture_id)),
                "status": observation_status,
                "unplantable_reason": unplantable_reason,
                "snapshot_path": str(_path_from_record(
                    served.get("snapshot") or {})),
                "source_snapshot": dict(served.get("snapshot") or {}),
                "evidence_utc": utc_now(),
                "expected_value_coverage": coverage,
                "alias_ids": [target_id] if target_id is not None else [],
                "selected_target_id": target_id,
                "selection_class": selection_class,
                "behavioral_breaker": breaker,
                "served_answer": str(served.get("answer", "")),
                "served_answer_correct": served.get("answer_correct") is True,
                "mounted_ids": [int(value) for value in
                                served.get("mounted_ids", ())],
                "substitution": substitution,
                "effective_fixture": {
                    key: fixture[key] for key in (
                        "fixture_id", "source_family", "split", "turn",
                        "session_id", "probe_id", "question", "expected_values",
                        "stale_values", "old_values", "wrong_fact_values", "source",
                    ) if key in fixture
                },
                "detector_arms_active": [],
                "race_row": False,
            }
        elif "planted_miss" in selected:
            rung_restores = [
                dict(row["planted_miss"]["fork_restore"])
                for row in attempt_rows
            ]
            changed = [
                restore for restore in rung_restores
                if restore.get("intervention") == PLANTED_INTERVENTION
            ]
            _require(bool(changed),
                     "registered planted target had no mounted seats in any "
                     "lived ladder rung; fork-hydration withholding cannot apply")
            aggregate_deltas = [
                dict(delta)
                for restore in changed
                for delta in (restore.get("field_deltas") or ())
            ]
            _require(bool(aggregate_deltas),
                     "planted ladder registered intervention has no field deltas")
            planted = selected["planted_miss"]
            planted["selected_rung_restore"] = dict(planted["fork_restore"])
            _require(
                planted["selected_rung_restore"].get("intervention")
                == PLANTED_INTERVENTION
                and bool(planted["selected_rung_restore"].get("field_deltas")),
                "selected planted-miss ladder rung is a state no-op",
            )
            planted["fork_restore"] = {
                "schema": "grm.det1_5.ladder_fork_restore.v1",
                "intervention": PLANTED_INTERVENTION,
                "same_process_index_verified": all(
                    restore.get("same_process_index_verified") is True
                    for restore in rung_restores
                ),
                "field_deltas": aggregate_deltas,
                "rung_restores": rung_restores,
                "registered_delta_rung_count": len(changed),
                "selected_attempt_ordinal": int(planted["attempt_ordinal"]),
                "selected_rung_intervention": planted[
                    "selected_rung_restore"].get("intervention"),
                "delta_receipt": dict(
                    planted["selected_rung_restore"]["delta_receipt"]),
                "delta_fork_snapshot": dict(
                    planted["selected_rung_restore"]["delta_fork_snapshot"]),
            }
    finally:
        _restore_counterfactual(arena, base)

    # Only the untouched production call advances the lived chronological path.
    canonical_answer, canonical_info = original(
        repo,
        str(fixture["question"]),
        topk=int(topk),
        ngen=int(ngen),
        defer_memory=bool(canonical_defer_memory),
        turn_idx=turn_idx,
        probe_ladder=True,
        max_trips=int(max_trips),
    )
    if not verbal and "served" in selected:
        _require(
            str(canonical_answer) == str(selected["served"]["answer"])
            and [int(value) for value in arena.cur_mounts]
            == [int(value) for value in selected["served"]["mounted_ids"]],
            f"inline served fork differs from lived production for {fixture['fixture_id']}",
        )
    completed_ns = time.time_ns()
    if registration_capture:
        assert registration_observation is not None
        append_jsonl_once(
            rows_path, registration_observation, key="row_id")
    elif verbal:
        _require(mechanistic_times is not None,
                 "D-VERB worker lacks mechanistic chronology bindings")
        for variant in variants:
            row_id = f"{fixture['fixture_id']}:{variant}"
            binding = mechanistic_times[row_id]
            _require(isinstance(binding, Mapping),
                     f"D-VERB chronology binding is malformed for {row_id}")
            mech_ns = int(binding["mechanistic_completed_unix_ns"])
            verbal_ns = selected[variant].get("verbal_started_unix_ns")
            _require(isinstance(verbal_ns, int) and verbal_ns > mech_ns,
                     f"observed D-VERB pass did not start after {row_id}")
            row = {
                "schema": "grm.det1_5.verbal_row.v1",
                "row_id": row_id,
                "fixture_id": str(fixture["fixture_id"]),
                "source_family": str(fixture["source_family"]),
                "split": "eval",
                "stage": "eval",
                "variant": variant,
                "evaluation_started_unix_ns": verbal_ns,
                "mechanistic_completed_unix_ns": mech_ns,
                "verbal_started_unix_ns": verbal_ns,
                "question": str(fixture["question"]),
                "verbal_question_exact": VERBAL_QUESTION,
                "prompt_suffix_exact": VERBAL_QUESTION,
                "selected_attempt_ordinal": int(
                    selected[variant]["attempt_ordinal"]),
                "mounted_ids": [
                    int(value) for value in selected[variant]["mounted_ids"]],
                "plant_target_id": int(profile["logical_router_rank1"]),
                "plant_alias_ids": [int(value) for value in
                                    profile["logical_alias_ids"]],
                "plant_target_source": profile.get("plant_target_source"),
                "plant_registry": dict(profile.get("plant_registry") or {}),
                "plant_entry_sha256": profile.get("plant_entry_sha256"),
                "mechanistic_selected_attempt_ordinal": int(
                    binding["selected_attempt_ordinal"]),
                "mechanistic_mounted_ids": [
                    int(value) for value in binding["mounted_ids"]],
                "verbal": dict(selected[variant]["verbal"]),
                "runtime_frame": dict(runtime),
                "registration_path": str(REGISTRATION.resolve()),
                "process_instance_sha256": process["process_instance_sha256"],
                "fork_restore": dict(selected[variant]["fork_restore"]),
            }
            append_jsonl_once(rows_path, row, key="row_id")
    else:
        for variant in variants:
            row = _mechanistic_row(
                fixture=fixture,
                variant=variant,
                split=split,
                profile=profile,
                result=selected[variant],
                attempts=attempt_rows,
                started_ns=started_ns,
                completed_ns=completed_ns,
                runtime=runtime,
            )
            append_jsonl_once(rows_path, row, key="row_id")
    return canonical_answer, canonical_info


def _load_rows_record(receipt: Mapping[str, Any], label: str) -> tuple[Path, list[dict[str, Any]]]:
    path = _validate_file_record(receipt.get("rows") or {}, label)
    return path, read_jsonl(path)


def _select_registration_observations(
    registration: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Choose a lawful lived observation per base slot, or fail closed."""
    by_slot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    row_ids: set[str] = set()
    for raw in candidates:
        row = dict(raw)
        _require(row.get("schema") == "grm.det1_7.plant_observation.v1",
                 "plant-registration candidate has the wrong schema")
        row_id = str(row.get("row_id", ""))
        slot = str(row.get("candidate_for_fixture_id", row.get("fixture_id", "")))
        _require(bool(row_id) and row_id not in row_ids and bool(slot),
                 "plant-registration candidate id/slot is missing or duplicate")
        row_ids.add(row_id)
        by_slot[slot].append(row)
    base_ids = [str(row["fixture_id"]) for row in registration["fixtures"]]
    _require(set(by_slot) == set(base_ids),
             "plant-registration candidate slots differ from base registration")
    selected: list[dict[str, Any]] = []
    substitutions: list[dict[str, Any]] = []
    for slot in base_ids:
        values = by_slot[slot]
        primary = [
            row for row in values
            if str(row.get("effective_fixture_id")) == slot
        ]
        _require(len(primary) == 1,
                 f"plant-registration slot lacks one primary observation: {slot}")
        chosen = primary[0]
        if chosen.get("status") != "LAWFUL_LIVED_TARGET":
            reserves = [
                row for row in values
                if str(row.get("effective_fixture_id")) != slot
                and row.get("status") == "LAWFUL_LIVED_TARGET"
            ]
            _require(len(reserves) == 1,
                     f"no unique lawful same-session substitution for {slot}: "
                     f"primary_reason={chosen.get('unplantable_reason')!r}")
            chosen = reserves[0]
            substitution = dict(chosen.get("substitution") or {})
            _require(substitution.get("status") == "SUBSTITUTED",
                     f"replacement candidate lacks explicit substitution: {slot}")
            substitutions.append({
                "base_fixture_id": slot,
                "effective_fixture_id": chosen.get("effective_fixture_id"),
                "reason": substitution.get("reason"),
                "same_certified_session": True,
            })
        normalized = dict(chosen)
        normalized["fixture_id"] = slot
        selected.append(normalized)
    effective_ids = [str(value["effective_fixture_id"]) for value in selected]
    _require(len(effective_ids) == len(set(effective_ids)) == 14,
             "plant-registration effective fixtures are not one-to-one")
    return selected, substitutions


def plant_registration(
    run_dir: Path = FROZEN_RUN,
    *,
    lease_seconds: int = LEASE_SECONDS,
    wait_seconds: int = DEFAULT_WAIT_SECONDS,
    gap_seconds: int = GAP_SECONDS,
) -> dict[str, Any]:
    """Capture lived admissions, freeze their targets, and emit no race rows."""
    from scripts.grm_det1_7_registry import (
        derive_plant_registry,
        validate_plant_registry,
        write_content_addressed_registry,
    )

    run_dir = Path(run_dir).resolve()
    if _stage_is_complete(run_dir, "plant_registration"):
        return _validate_stage_marker(run_dir, "plant_registration")[2]
    source_auth = validate_det1_7_source_authorization(run_dir)
    prerequisite = _require_prior_stage(run_dir, "cross_process_zero")
    preflight, code = gpu_preflight(run_dir)
    _require(code == 0,
             f"GPU preflight blocked plant registration: {preflight}")
    directory = campaign_root(run_dir) / STAGE_DIRS["plant_registration"]
    outputs: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for spec in STAGE_SPECS["plant_registration"]:
        shard_dir = stage_shard_root(run_dir, "plant_registration") / spec
        output = _completed_shard_output(
            shard_dir, stage="plant_registration", spec=spec)
        if output is None:
            _enforce_worker_gap(run_dir, int(gap_seconds))
            output = _run_worker(
                run_dir,
                "plant-registration",
                _next_attempt(shard_dir),
                lease_seconds=int(lease_seconds),
                wait_seconds=int(wait_seconds),
                extra=("--spec", spec),
            )
        receipt_path = _validate_worker_shard_output(
            output, stage="plant_registration", spec=spec)
        observations_path = _validate_file_record(
            output.get("observations") or {},
            f"plant-registration/{spec} observations",
        )
        observed = read_jsonl(observations_path)
        _require(int(output.get("observation_count", -1)) == len(observed),
                 f"plant-registration/{spec} observation count drifted")
        _require(output.get("detector_hooks") == []
                 and output.get("variants") == ["served"],
                 f"plant-registration/{spec} activated an evaluation arm")
        outputs.append(output)
        shard_records.append(file_record(receipt_path))
        candidates.extend(observed)

    _require(len(candidates) == 15,
             f"plant registration expected 14 slots + one reserve, got "
             f"{len(candidates)}")
    selected, substitutions = _select_registration_observations(
        read_json(REGISTRATION), candidates)
    candidates_path = directory / "candidate_observations.jsonl"
    selected_path = directory / "selected_observations.jsonl"
    _write_jsonl_exclusive_or_verify(candidates_path, candidates)
    _write_jsonl_exclusive_or_verify(selected_path, selected)
    registry = derive_plant_registry(
        REGISTRATION,
        DET1_7_ORDER,
        selected,
        created_utc=utc_now(),
        new_eval_evidence_utc=(),
        record_root=ROOT,
    )
    registry_path = write_content_addressed_registry(
        directory,
        registry,
        record_root=ROOT,
        stem="plant_registry",
    )
    validation = validate_plant_registry(registry, record_root=ROOT)
    table = [
        {
            "base_fixture_id": entry["fixture_id"],
            "effective_fixture_id": entry["effective_fixture_id"],
            "split": entry["split"],
            "policy_branch": entry["policy_branch"],
            "actual_lived_mounts": entry["actual_authoritative_mounts"],
            "plant_target_id": entry["selected_target_id"],
            "plant_alias_ids": entry["alias_ids"],
            "selection_rule": entry["rule_id"],
            "source_snapshot": entry["source_snapshot"],
            "substitution": entry["substitution"],
        }
        for entry in registry["entries"]
    ]
    process_ids = [str(value["process_instance_sha256"]) for value in outputs]
    _require(len(process_ids) == len(set(process_ids)),
             "plant-registration reused a process across bounded shards")
    receipt = {
        "schema": STAGE_SCHEMAS["plant_registration"],
        "status": "PASS",
        "created_utc": utc_now(),
        "registration": file_record(REGISTRATION),
        "runtime_frame": file_record(RUNTIME_FRAME),
        "precollection_authorization": source_auth["record"],
        "shard_receipts": shard_records,
        "candidate_observations": file_record(candidates_path),
        "selected_observations": file_record(selected_path),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "plant_registry": file_record(registry_path),
        "plant_registry_payload_sha256": registry[
            "registry_payload_sha256"],
        "plant_registry_file_sha256": file_record(registry_path)["sha256"],
        "registry_validation": validation,
        "per_turn_table": table,
        "substitutions": substitutions,
        "counts": {"served": 12, "planted_miss": 12, "eval_pairs": 12},
        "detector_arms_active": [],
        "race_rows_emitted": False,
        "thresholds_fitted": False,
        "frozen_before_g0": True,
        "frozen_before_eval": True,
        "process_instance_sha256s": process_ids,
        "process_instance_sha256": aggregate_process_instances(process_ids),
        "process_instance_identity": (
            "AGGREGATE_ORDERED_PROCESS_LIST_NOT_OS_IDENTITY"
        ),
        "process_identity_semantics": (
            "SHA256_OF_ORDERED_OS_PROCESS_INSTANCE_LIST"
        ),
        "gpu_allocations_claimed_by_parent": 0,
    }
    receipt_path = write_content_addressed(
        directory, "plant_registration_receipt", receipt)
    _write_stage_marker(
        run_dir,
        "plant_registration",
        receipt_path,
        [prerequisite, run_dir / DET1_7_SOURCE_AUTH.name],
    )
    return receipt


def _require_prior_stage(run_dir: Path, stage: str) -> Path:
    return _validate_stage_marker(run_dir, stage)[0]


def _run_row_stage(
    run_dir: Path,
    stage: str,
    *,
    lease_seconds: int,
    wait_seconds: int,
    gap_seconds: int = GAP_SECONDS,
) -> dict[str, Any]:
    """Run bounded worker shards and finalize one fresh stage receipt."""
    if _stage_is_complete(run_dir, stage):
        return _validate_stage_marker(run_dir, stage)[2]
    registered_context = load_det1_7_registered_context(run_dir)
    plant_registry = registered_context["plant_registry"]
    predecessor = {
        "g0": "plant_registration",
        "g1": "g0",
        "calibration": "g1",
        "eval_mechanistic": "calibration",
        "eval_verbal": "eval_mechanistic",
    }[stage]
    prerequisite = _require_prior_stage(run_dir, predecessor)
    preflight, code = gpu_preflight(run_dir)
    _require(code == 0, f"GPU preflight blocked {stage}: {preflight}")
    directory = campaign_root(run_dir) / STAGE_DIRS[stage]
    specs = STAGE_SPECS[stage]
    _require(int(gap_seconds) >= 0, "inter-worker gap must be nonnegative")
    outputs: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    for spec in specs:
        shard_dir = stage_shard_root(run_dir, stage) / spec
        output = _completed_shard_output(
            shard_dir, stage=stage, spec=spec)
        if output is None:
            # Revalidation consumes no lease. Every newly launched child,
            # including one after a parent restart, observes the wall gap.
            _enforce_worker_gap(run_dir, int(gap_seconds))
            attempt = _next_attempt(shard_dir)
            output = _run_worker(
                run_dir,
                stage.replace("_", "-"),
                attempt,
                lease_seconds=int(lease_seconds),
                wait_seconds=int(wait_seconds),
                extra=("--spec", spec),
            )
        shard_path = _validate_worker_shard_output(
            output, stage=stage, spec=spec)
        _require(output.get("process_instance_sha256"),
                 f"{stage}/{spec} lacks process identity")
        outputs.append(output)
        shard_records.append(file_record(shard_path))

    process_ids = [str(value["process_instance_sha256"]) for value in outputs]
    _require(len(process_ids) == len(set(process_ids)),
             f"{stage} reused a process across bounded shards")
    common = {
        "schema": STAGE_SCHEMAS[stage],
        "status": "PASS",
        "registration": file_record(REGISTRATION),
        "runtime_frame": file_record(RUNTIME_FRAME),
        "race_authorization_amendment": file_record(RACE_AMENDMENT),
        "source_amendment": file_record(DELTA_AMENDMENT),
        "precollection_authorization": registered_context[
            "precollection_authorization"],
        "plant_registry": registered_context["plant_registry_record"],
        "terminal_amendment": registered_context["terminal_amendment"],
        "shard_receipts": shard_records,
        "process_instance_sha256s": process_ids,
        "process_instance_sha256": aggregate_process_instances(process_ids),
        "process_instance_identity": "AGGREGATE_ORDERED_PROCESS_LIST_NOT_OS_IDENTITY",
        "process_identity_semantics": "SHA256_OF_ORDERED_OS_PROCESS_INSTANCE_LIST",
        "lease_seconds_per_child": int(lease_seconds),
        "inter_child_gap_seconds": int(gap_seconds),
    }
    registration = read_json(REGISTRATION)
    effective_registration = effective_registration_projection(
        registration, plant_registry)
    runtime = read_json(RUNTIME_FRAME)
    rows_path: Path | None = None
    rows: list[dict[str, Any]] = []
    if stage != "g1":
        rows_name = (
            "verbal_rows.jsonl" if stage == "eval_verbal"
            else "mechanistic_rows.jsonl"
        )
        rows_path = directory / rows_name
        for spec, output in zip(specs, outputs, strict=True):
            shard_rows_path = _validate_file_record(
                output.get("rows") or {}, f"{stage}/{spec} rows")
            shard_rows = read_jsonl(shard_rows_path)
            _require(int(output.get("row_count", -1)) == len(shard_rows),
                     f"{stage}/{spec} row count binding drifted")
            rows.extend(shard_rows)
        row_ids = [str(row.get("row_id", "")) for row in rows]
        _require(all(row_ids) and len(row_ids) == len(set(row_ids)),
                 f"{stage} shard rows overlap or lack row IDs")
        _write_jsonl_exclusive_or_verify(rows_path, rows)
    if stage == "g0":
        assert rows_path is not None
        gate = validate_g0_rows(rows)
        receipt = {
            **common,
            **gate,
            "rows": file_record(rows_path),
            "dedicated_gate_rows": True,
        }
        receipt_path = write_content_addressed(directory, "g0_receipt", receipt)
    elif stage == "g1":
        by_arm = {str(value.get("arm")): value for value in outputs}
        _require(set(by_arm) == {"control", "all-hooks"},
                 "DET-G1 worker arms are incomplete")
        for arm, output in by_arm.items():
            records = output.get("byte_records") or {}
            _require(set(records) == {
                "transcript", "scorecard", "served_projection"},
                f"DET-G1 {arm} byte-record inventory is incomplete")
            arm_root = (stage_shard_root(run_dir, "g1") / arm).resolve()
            for name, record in records.items():
                path = _validate_file_record(
                    record, f"DET-G1 {arm}.{name}")
                _require(path.resolve().is_relative_to(arm_root),
                         f"DET-G1 {arm}.{name} escapes its fresh arm")
        control_path = _validate_file_record(
            by_arm["control"].get("receipt_file") or {}, "G1 control receipt")
        hooked_path = _validate_file_record(
            by_arm["all-hooks"].get("receipt_file") or {}, "G1 hooked receipt")
        gate = validate_g1_pair(by_arm["control"], by_arm["all-hooks"])
        receipt = {
            **common,
            **gate,
            "byte_identical": True,
            "all_hooks_active": True,
            "control": file_record(control_path),
            "hooked": file_record(hooked_path),
        }
        receipt_path = write_content_addressed(directory, "g1_receipt", receipt)
    elif stage == "calibration":
        assert rows_path is not None
        calibration_ids, _evaluation_ids = _registered_split(
            effective_registration)
        _require(
            [str(row.get("fixture_id")) for row in rows] == calibration_ids
            and len(rows) == 2
            and all(row.get("variant") == "served" for row in rows),
            "fresh calibration rows differ from registered two served fixtures",
        )
        served_validations = [
            validate_served_control(row, f"calibration {row.get('fixture_id')}")
            for row in rows
        ]
        threshold_path, _thresholds = _calibration_thresholds(
            directory, rows, rows_path, run_dir)
        receipt = {
            **common,
            "gate_pass": True,
            "rows": file_record(rows_path),
            "thresholds": file_record(threshold_path),
            "calibration_count": 2,
            "calibration_fixture_ids": calibration_ids,
            "served_control_validations": served_validations,
            "frozen_before_eval": True,
        }
        receipt_path = write_content_addressed(
            directory, "calibration_receipt", receipt)
    elif stage == "eval_mechanistic":
        assert rows_path is not None
        calibration_path = campaign_root(run_dir) / "calibration/mechanistic_rows.jsonl"
        split = validate_split(
            effective_registration, read_jsonl(calibration_path), rows)
        repeated_g0 = validate_g0_rows(rows)
        receipt = {
            **common,
            "gate_pass": True,
            "rows": file_record(rows_path),
            "split_validation": split,
            "repeated_g0_validation": repeated_g0,
            "row_count": 24,
        }
        receipt_path = write_content_addressed(
            directory, "mechanistic_eval_receipt", receipt)
    else:
        assert rows_path is not None
        mech_receipt = _validate_stage_marker(
            run_dir, "eval_mechanistic")[2]
        chronological_sessions = [
            value.get("chronological_session") for value in outputs
            if value.get("chronological_session") is not None
        ]
        receipt = {
            **common,
            "gate_pass": True,
            "mechanistic_process_instance_sha256": mech_receipt[
                "process_instance_sha256"],
            "mechanistic_process_instance_sha256s": mech_receipt[
                "process_instance_sha256s"],
            "verbal_question_exact": VERBAL_QUESTION,
            "rows": file_record(rows_path),
            "row_count": 24,
            "chronological_session": chronological_sessions,
        }
        validation = validate_verbal_chronology({**receipt, "rows": rows})
        receipt.update(validation)
        receipt["chronology_validation"] = validation
        receipt_path = write_content_addressed(
            directory, "chronological_verbal_receipt", receipt)
    _write_stage_marker(
        run_dir,
        stage,
        receipt_path,
        [
            prerequisite,
            Path(run_dir) / DET1_7_TERMINAL_AMENDMENT.name,
            plant_registry_path(run_dir),
        ],
    )
    return receipt


def analyze(run_dir: Path = FROZEN_RUN) -> dict[str, Any]:
    if _stage_is_complete(run_dir, "analyze"):
        return _validate_stage_marker(run_dir, "analyze")[2]
    prerequisite = _require_prior_stage(run_dir, "eval_verbal")
    analyzer = ROOT / "scripts/grm_det1_5_analyze.py"
    _require(analyzer.is_file(), f"DET1.5 analyzer is missing: {analyzer}")
    command = [
        sys.executable, str(analyzer), "report", "--run-dir", str(run_dir),
    ]
    completed = subprocess.run(command, cwd=ROOT, env=_base_env(), check=False)
    _require(completed.returncode == 0,
             f"DET1.5 analyzer returned {completed.returncode}")
    directory = campaign_root(run_dir) / "analysis"
    receipt_path = _one(directory, "analysis_receipt_*.json")
    receipt = read_json(receipt_path)
    _require(receipt.get("status") == "PASS",
             "DET1.5 analyzer receipt is not PASS")
    verdict = receipt.get("prediction_verdict")
    _require(verdict in ("SUPPORTED", "PARTIAL", "REFUTED"),
             f"analyzer did not issue registered verdict: {verdict}")
    _write_stage_marker(run_dir, "analyze", receipt_path, [prerequisite])
    return receipt


def campaign(
    run_dir: Path = FROZEN_RUN,
    *,
    lease_seconds: int = LEASE_SECONDS,
    wait_seconds: int = DEFAULT_WAIT_SECONDS,
    gap_seconds: int = GAP_SECONDS,
) -> dict[str, Any]:
    """Run or revalidate the complete ordered campaign."""
    completed = []
    cross_process_zero(
        run_dir, lease_seconds=lease_seconds, wait_seconds=wait_seconds)
    completed.append("cross_process_zero")
    author_det1_7_source_authorization(run_dir)
    plant_registration(
        run_dir,
        lease_seconds=lease_seconds,
        wait_seconds=wait_seconds,
        gap_seconds=gap_seconds,
    )
    completed.append("plant_registration")
    author_det1_7_terminal_amendment(run_dir)
    for stage in ("g0", "g1", "calibration", "eval_mechanistic", "eval_verbal"):
        if completed:
            _require(int(gap_seconds) >= 0, "inter-stage gap must be nonnegative")
            time.sleep(int(gap_seconds))
        _run_row_stage(
            run_dir,
            stage,
            lease_seconds=lease_seconds,
            wait_seconds=wait_seconds,
            gap_seconds=gap_seconds,
        )
        completed.append(stage)
    result = analyze(run_dir)
    completed.append("analyze")
    _analysis_marker, analysis_path, _analysis_value = _validate_stage_marker(
        run_dir, "analyze")
    return {
        "schema": "grm.det1_5.campaign_complete.v1",
        "status": "COMPLETE",
        "created_utc": utc_now(),
        "stages": completed,
        "prediction_verdict": result.get("prediction_verdict"),
        "analysis_receipt": file_record(analysis_path),
    }


def selftest(_args: argparse.Namespace | None = None) -> dict[str, Any]:
    """Deterministic CPU-only contract tests; never probes CUDA."""
    fake_digest = "1" * 64
    fake_lived = {"path": "fixture", "bytes": 1, "sha256": fake_digest}
    base_zero = {
        "status": "PASS",
        "gate_pass": True,
        "intervention": "ZERO",
        "substrate_comparison": {
            "status": ZERO_PASS_STATUS,
            "gate_pass": True,
        },
        "answer": {
            "normalized": "**Nacre\u20116\u2011Blue**",
            "lived_normalized": "Nacre-6-Blue",
            "normalized_match": True,
        },
        "lived_snapshot": fake_lived,
        "canonical_state_sha256": fake_digest,
    }
    left = {**base_zero, "process_instance_sha256": "a" * 64}
    right = {**base_zero, "process_instance_sha256": "b" * 64}
    zero = validate_cross_process_zero(left, right)

    g0_rows = []
    for index in range(12):
        fixture = f"fixture_{index:02d}"
        g0_rows.extend((
            {
                "row_id": f"{fixture}:served",
                "fixture_id": fixture,
                "variant": "served",
                "answer_correct": True,
                "target_contains_expected": True,
                "router_rank1_admitted": True,
                "raw_router_rank1": index,
                "logical_router_rank1": index,
                "logical_alias_ids": [index],
                "plant_target_id": index,
                "plant_alias_ids": [index],
                "plant_target_source": "DET1_7_LIVED_ADMISSION_REGISTRY",
                "plant_registry": {
                    "path": "plant_registry.json", "bytes": 1,
                    "sha256": "a" * 64,
                },
                "plant_entry_sha256": "b" * 64,
                "mounted_ids": [index],
                "mounted_contains_expected": True,
                "full_index_contains_all_aliases": True,
            },
            {
                "row_id": f"{fixture}:planted_miss",
                "fixture_id": fixture,
                "variant": "planted_miss",
                "target_contains_expected": True,
                "logical_alias_ids": [index],
                "plant_target_id": index,
                "plant_alias_ids": [index],
                "plant_target_source": "DET1_7_LIVED_ADMISSION_REGISTRY",
                "plant_registry": {
                    "path": "plant_registry.json", "bytes": 1,
                    "sha256": "a" * 64,
                },
                "plant_entry_sha256": "b" * 64,
                "mounted_ids": [],
                "plant_checks": {
                    "withheld_aliases_absent": True,
                    "logical_target_absent": True,
                    "raw_router_rank1_absent": True,
                    "registered_plant_target_absent": True,
                    "registered_plant_aliases_absent": True,
                    "expected_value_absent_from_mounted_text": True,
                    "withheld_aliases_remain_in_full_detector_index": True,
                    "admission_ranking_unchanged": True,
                    "admission_branch_unchanged": True,
                    "only_aliases_removed_from_admission_plan": True,
                    "withheld_aliases_absent_from_every_ladder_attempt": True,
                },
                "fork_restore": {
                    "intervention": PLANTED_INTERVENTION,
                    "same_process_index_verified": True,
                    "field_deltas": [{"field": "state.arena.cur_mounts"}],
                    "delta_receipt": {
                        "schema": DELTA_RECEIPT_SCHEMA,
                        "status": DELTA_PASS_STATUS,
                        "gate_pass": True,
                        "target_absence": {
                            "gate_pass": True,
                            "checks": {
                                "registered_aliases_absent_from_fork_mounts": True,
                            },
                        },
                        "strict_zero_comparator": {
                            "schema": (
                                "grm.det1_4.zero_fork_substrate_comparison.v1"
                            ),
                            "status": "DIVERGENT",
                            "gate_pass": False,
                            "counts": {"EQUAL": 10, "DIVERGENT": 1},
                            "row_count": 11,
                        },
                        "exact_divergence_set": True,
                        "non_delta_fields_equal": True,
                        "retained_arrays_exact": True,
                        "transformed_arrays": [{
                            "field": "state.arena.cur_mounts",
                            "retained_bytes_exact": True,
                        }],
                        "unexpected_divergent_fields": [],
                        "missing_expected_divergent_fields": [],
                        "expected_fork_value_mismatches": [],
                        "expected_divergent_fields": [
                            "state.arena.cur_mounts"],
                        "observed_non_equal_fields": [
                            "state.arena.cur_mounts"],
                    },
                    "delta_fork_snapshot": {
                        "path": f"fixture_{index:02d}/delta_fork/manifest.json",
                        "bytes": 1,
                        "sha256": "e" * 64,
                    },
                },
            },
        ))
    g0 = validate_g0_rows(g0_rows)

    byte_records = {
        name: {"bytes": index + 1, "sha256": str(index + 2) * 64}
        for index, name in enumerate(("transcript", "scorecard", "served_projection"))
    }
    g1 = validate_g1_pair(
        {
            "process_instance_sha256": "c" * 64,
            "active_hooks": [],
            "d_verb_isolated": True,
            "byte_records": byte_records,
        },
        {
            "process_instance_sha256": "d" * 64,
            "active_hooks": list(MECHANISTIC),
            "d_verb_isolated": True,
            "byte_records": byte_records,
        },
    )
    commands = build_lead_commands(FROZEN_RUN)
    _require(all(isinstance(command, list) and command for command in commands),
             "lead command builder selftest failed")
    _require(not any("_worker" in command for command in commands),
             "private worker leaked into lead commands")
    drivers = campaign_driver_inventory()
    _require(drivers["gpu_child_count"] == 36,
             "bounded GPU shard count drifted")
    _require(STAGE_SHARD_DIRS["eval_mechanistic"]
             != STAGE_SHARD_DIRS["eval_verbal"],
             "mechanistic and D-VERB shards share a namespace")
    return {
        "schema": "grm.det1_5.selftest.v1",
        "status": "PASS_CPU_ONLY",
        "created_utc": utc_now(),
        "checks": {
            "cross_process_zero": zero["status"],
            "g0_12_plus_12": g0["status"],
            "g1_all_hooks_byte_identity": g1["status"],
            "registered_planted_intervention": PLANTED_INTERVENTION,
            "lead_commands_pure": "PASS",
            "campaign_driver_inventory": "PASS",
            "bounded_gpu_child_count": drivers["gpu_child_count"],
            "eval_shard_namespaces_distinct": True,
            "verdict_vocabulary_unchanged": list(("SUPPORTED", "PARTIAL", "REFUTED")),
        },
        "gpu_allocations_attempted": 0,
    }


def det1_7_cpu_preflight(
    run_dir: Path = FROZEN_RUN,
) -> dict[str, Any]:
    """Inventory extant lived evidence without treating it as a full registry."""
    run_dir = Path(run_dir).resolve()
    _require(run_dir == FROZEN_RUN.resolve(),
             "DET1.7 CPU preflight is bound to the frozen run")
    registration = read_json(REGISTRATION)
    _calibration, evaluation = _registered_split(registration)
    root = campaign_root(run_dir) / STAGE_SHARD_DIRS["g0"]
    table: list[dict[str, Any]] = []
    missing: list[str] = []
    incorrect: list[str] = []
    synthesis_pending: list[str] = []
    for fixture_id in evaluation:
        paths = sorted(root.glob(
            f"*/attempt_002/snapshots/{fixture_id}/rung_00/manifest.json"))
        if len(paths) != 1:
            missing.append(fixture_id)
            table.append({
                "fixture_id": fixture_id,
                "status": "MISSING_DET1_6_LIVED_FORK_SNAPSHOT",
                "candidate_target_id": None,
            })
            continue
        path = paths[0]
        manifest = read_json(path)
        state = manifest.get("state") or {}
        mounts = [int(value) for value in state.get(
            "admission.authoritative_mounts", ())]
        plan = [int(value) for value in state.get("admission.rank_plan", ())]
        planned = [int(value) for value in state.get(
            "admission.current_planned", ())]
        ordered = list(dict.fromkeys([*plan, *planned, *mounts]))
        branch = str(state.get("admission.policy_branch", ""))
        identified = {int(value) for value in state.get(
            "admission.identified_candidates", ())}
        if branch == "declared_synthesis_identified_set":
            candidates = [value for value in ordered
                          if value in identified and value in mounts]
            target = candidates[0] if len(candidates) == 1 else None
            synthesis_pending.append(fixture_id)
            status = "CANDIDATE_REQUIRES_TARGET_ONLY_BREAKER"
        else:
            target = next((value for value in ordered if value in mounts), None)
            status = "CANDIDATE_FROM_LIVED_ACTUAL_WINNER"
        linked = manifest.get("linked_answer") or {}
        served_correct = (
            linked.get("attempt_answer_correct") is True
            and linked.get("probe_answer_correct") is True
            and linked.get("attempt_refusal") is False
            and linked.get("probe_refusal") is False
        )
        if not served_correct:
            incorrect.append(fixture_id)
            status = "UNPLANTABLE_LIVED_SERVED_CONTROL"
        table.append({
            "fixture_id": fixture_id,
            "status": status,
            "candidate_target_id": target,
            "actual_lived_mounts": mounts,
            "policy_branch": branch,
            "rank_plan": plan,
            "identified_candidates": sorted(identified),
            "served_correct_nonrefusal": served_correct,
            "source_snapshot": file_record(path),
        })
    receipt = {
        "schema": "grm.det1_7.cpu_evidence_preflight.v1",
        "status": "BLOCKED_PENDING_LIVED_REGISTRATION_COLLECTION",
        "created_utc": utc_now(),
        "order": file_record(DET1_7_ORDER),
        "base_registration": file_record(REGISTRATION),
        "base_registration_sha256": file_record(REGISTRATION)["sha256"],
        "extant_eval_table": table,
        "extant_snapshot_count": len(evaluation) - len(missing),
        "missing_fixture_ids": missing,
        "invalid_served_fixture_ids": incorrect,
        "synthesis_breaker_pending_fixture_ids": synthesis_pending,
        "required_same_session_substitution": {
            "original_fixture_id": "e2e_t30_atlas_tone",
            "candidate_fixture_id": "e2e_t33_polaris_mark",
            "source_turn": 32,
            "probe_turn": 33,
            "expected_values": ["Marble-4-Juliet"],
            "measured": False,
        },
        "registry_freeze_ready": False,
        "registry_file_sha256": None,
        "historical_cross_model_rows_used": False,
        "gpu_allocations_attempted": 0,
        "cpu_selftest": selftest(),
    }
    path = write_content_addressed(
        run_dir / "det1_7", "cpu_evidence_preflight", receipt)
    result = dict(receipt)
    result["receipt_file"] = file_record(path)
    return result


def _worker_main(args: argparse.Namespace) -> dict[str, Any]:
    if args.worker == "cross-zero":
        return _cross_zero_worker(args)
    from scripts.grm_det1_5_workers import run_worker

    return run_worker(args.worker, args)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, default=FROZEN_RUN)


def _add_gpu(parser: argparse.ArgumentParser) -> None:
    _add_common(parser)
    parser.add_argument("--lease-seconds", type=int, default=LEASE_SECONDS)
    parser.add_argument("--lock-wait-seconds", type=int, default=DEFAULT_WAIT_SECONDS)
    parser.add_argument("--gap-seconds", type=int, default=GAP_SECONDS)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "inventory", "selftest", "author-amendment",
        "author-delta-amendment", "author-det1-7-source",
        "author-det1-7-terminal", "det1-7-cpu-preflight", "analyze",
    ):
        _add_common(sub.add_parser(name))
    preflight = sub.add_parser("gpu-preflight")
    _add_common(preflight)
    preflight.add_argument("--write-receipt", action="store_true")
    for name in (
        "cross-process-zero", "plant-registration", "g0", "g1", "calibration",
        "eval-mechanistic", "eval-verbal", "campaign",
    ):
        _add_gpu(sub.add_parser(name))
    worker = sub.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("worker")
    worker.add_argument("--run-dir", type=Path, required=True)
    worker.add_argument("--attempt-dir", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--lease-seconds", type=int, required=True)
    worker.add_argument("--spec")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    code = 0
    try:
        if args.command == "inventory":
            output = inventory(args.run_dir)
        elif args.command == "selftest":
            output = selftest(args)
        elif args.command == "gpu-preflight":
            output, code = gpu_preflight(
                args.run_dir, write_receipt=bool(args.write_receipt))
        elif args.command == "cross-process-zero":
            output = cross_process_zero(
                args.run_dir,
                lease_seconds=args.lease_seconds,
                wait_seconds=args.lock_wait_seconds,
            )
        elif args.command == "author-amendment":
            output = author_amendment(args.run_dir)
        elif args.command == "author-delta-amendment":
            output = author_delta_amendment(args.run_dir)
        elif args.command == "author-det1-7-source":
            output = author_det1_7_source_authorization(args.run_dir)
        elif args.command == "plant-registration":
            output = plant_registration(
                args.run_dir,
                lease_seconds=args.lease_seconds,
                wait_seconds=args.lock_wait_seconds,
                gap_seconds=args.gap_seconds,
            )
        elif args.command == "author-det1-7-terminal":
            output = author_det1_7_terminal_amendment(args.run_dir)
        elif args.command == "det1-7-cpu-preflight":
            output = det1_7_cpu_preflight(args.run_dir)
        elif args.command in (
            "g0", "g1", "calibration", "eval-mechanistic", "eval-verbal",
        ):
            output = _run_row_stage(
                args.run_dir,
                args.command.replace("-", "_"),
                lease_seconds=args.lease_seconds,
                wait_seconds=args.lock_wait_seconds,
                gap_seconds=args.gap_seconds,
            )
        elif args.command == "analyze":
            output = analyze(args.run_dir)
        elif args.command == "campaign":
            output = campaign(
                args.run_dir,
                lease_seconds=args.lease_seconds,
                wait_seconds=args.lock_wait_seconds,
                gap_seconds=args.gap_seconds,
            )
        else:
            output = _worker_main(args)
            _write_json_exclusive_or_verify(Path(args.output), output)
    except BaseException as exc:
        output = {
            "schema": "grm.det1_5.cli_error.v1",
            "status": "FAIL_CLOSED",
            "created_utc": utc_now(),
            "command": getattr(args, "command", None),
            "error": f"{type(exc).__name__}: {exc}",
            "gpu_results_claimed": False,
            "prediction_verdict": None,
        }
        code = 2
        if getattr(args, "command", None) == "_worker":
            try:
                _write_json_exclusive_or_verify(Path(args.output), output)
            except BaseException:
                pass
    sys.stdout.buffer.write(canonical_json_bytes(output))
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
