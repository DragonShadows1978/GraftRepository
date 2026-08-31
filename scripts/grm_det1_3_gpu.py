#!/usr/bin/env python3
"""Fresh-process DET1.3 lived/replay snapshot and evidence runner.

``preflight`` is CPU-only and adjudicates the evidence already on disk.
``capture`` constructs one GPT-OSS arm in a fresh process, snapshots immediately
before the Harbor probe prefill, and continues that exact process to bind its
answer.  The lived arm feeds the fixture's scripted turns chronologically
through the GPT-OSS live cache; the replay arm reproduces DET1's independent
fixture deposits plus counterfactual snapshot/restore path.  This new same-model
lived protocol is intentionally identified as new evidence, not silently
substituted for ADM2's MiniCPM producer receipt.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_det1_common import (  # noqa: E402
    DETError,
    contains_value,
    file_record,
    read_json,
    sha256_file,
)
from scripts.grm_det1_3_snapshot import (  # noqa: E402
    ARM_PROTOCOLS,
    compare_snapshots,
    finalize_snapshot_with_answer,
    load_snapshot,
    process_instance_sha256,
    stop_at_next_forward,
)


ORDER = ROOT / "orders" / "GRM_DET1_3_REPLAY_FIDELITY.md"
AMENDMENT_NAME = "det1_3_source_amendment.json"
FIXTURE = (
    ROOT / "tests" / "fixtures" / "supersession_battery"
    / "correction_then_restatement.json"
)
PRIOR_AMENDMENTS = (
    ("det1_1_source_amendment.json", "grm.det1.source_amendment.v1"),
    ("det1_2_source_amendment.json", "grm.det1_2.source_amendment.v1"),
    (
        "det1_2_replay_gate_source_amendment.json",
        "grm.det1_2.replay_gate_source_amendment.v1",
    ),
)
EXPECTED_ADDED = frozenset((
    "scripts/grm_det1_3_snapshot.py",
    "scripts/grm_det1_3_gpu.py",
))
EXPECTED_CHANGED = frozenset(("scripts/grm_det1_e2e.py",))
EXPECTED_TESTS = frozenset(("tests/test_grm_det1_3_snapshot.py",))


def _record(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file():
        raise DETError(f"DET1.3 source disappeared: {relative}")
    record = file_record(path)
    return {"bytes": record["bytes"], "sha256": record["sha256"]}


def _one(directory: Path, pattern: str) -> Path:
    values = sorted(Path(directory).glob(pattern))
    if len(values) != 1:
        raise DETError(f"expected one {pattern} under {directory}, got {len(values)}")
    return values[0]


def _validate_record(value: Mapping[str, Any], label: str) -> Path:
    path = Path(str(value.get("path", "")))
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file() or file_record(path) != dict(value):
        raise DETError(f"{label} file record no longer validates: {value}")
    return path


def _apply_layer(
    inventory: dict[str, dict[str, Any]],
    layer: Mapping[str, Any],
    *,
    label: str,
) -> None:
    changes = layer.get("allowed_source_changes") or {}
    additions = layer.get("added_sources") or {}
    if not isinstance(changes, dict) or not isinstance(additions, dict):
        raise DETError(f"{label} source inventory is malformed")
    for shown, change in changes.items():
        if shown not in inventory or change.get("before") != inventory[shown]:
            raise DETError(f"{label} before-record does not chain for {shown}")
        inventory[shown] = dict(change["after"])
    for shown, record in additions.items():
        if shown in inventory:
            raise DETError(f"{label} re-adds an existing source: {shown}")
        inventory[shown] = dict(record)


def check_inventory(run_dir: Path) -> Path:
    """Validate registration -> DET1.1 -> DET1.2 -> review -> DET1.3."""
    registration_path = _one(run_dir, "registration_*.json")
    registration = read_json(registration_path)
    inventory = {
        str(path): dict(record)
        for path, record in registration["production_source_inventory"].items()
    }
    previous_path = None
    for name, schema in PRIOR_AMENDMENTS:
        path = run_dir / name
        value = read_json(path)
        if value.get("schema") != schema:
            raise DETError(f"{name} schema drifted")
        if value.get("registration") != file_record(registration_path):
            raise DETError(f"{name} registration binding drifted")
        if previous_path is not None and value.get("prior_amendment") != file_record(
            previous_path
        ):
            raise DETError(f"{name} amendment chain drifted")
        _apply_layer(inventory, value, label=name)
        previous_path = path

    amendment_path = run_dir / AMENDMENT_NAME
    amendment = read_json(amendment_path)
    if amendment.get("schema") != "grm.det1_3.source_amendment.v1":
        raise DETError("DET1.3 source-amendment schema drifted")
    if amendment.get("status") != "AUTHORIZED_DIAGNOSTIC_AMENDMENT":
        raise DETError("DET1.3 amendment is not diagnostic-only authorized")
    if amendment.get("order") != file_record(ORDER):
        raise DETError("DET1.3 amendment is not bound to the live order")
    if amendment.get("registration") != file_record(registration_path):
        raise DETError("DET1.3 amendment registration binding drifted")
    if amendment.get("prior_amendment") != file_record(previous_path):
        raise DETError("DET1.3 amendment chain drifted")
    if set(amendment.get("allowed_source_changes") or {}) != EXPECTED_CHANGED:
        raise DETError("DET1.3 changed-source set drifted")
    if set(amendment.get("added_sources") or {}) != EXPECTED_ADDED:
        raise DETError("DET1.3 added-source set drifted")
    if set(amendment.get("test_sources") or {}) != EXPECTED_TESTS:
        raise DETError("DET1.3 test-source set drifted")
    _apply_layer(inventory, amendment, label=AMENDMENT_NAME)

    drift = {
        shown: {"expected": expected, "observed": _record(shown)}
        for shown, expected in inventory.items()
        if _record(shown) != expected
    }
    if drift:
        raise DETError(f"DET1.3 terminal source inventory drifted: {drift}")
    test_drift = {
        shown: {"expected": expected, "observed": _record(shown)}
        for shown, expected in amendment["test_sources"].items()
        if _record(shown) != expected
    }
    if test_drift:
        raise DETError(f"DET1.3 test source inventory drifted: {test_drift}")
    for layer_name in (*[name for name, _schema in PRIOR_AMENDMENTS], AMENDMENT_NAME):
        layer = read_json(run_dir / layer_name)
        for index, record in enumerate(layer.get("frozen_artifacts_unchanged") or ()):
            # Historical source records are intentionally superseded by the
            # overlay chain. Artifact records remain immutable.
            if str(record.get("path", "")).startswith(("scripts/", "core/")):
                continue
            _validate_record(record, f"{layer_name} frozen artifact {index}")
    for shown, expected in registration.get("model_snapshot_inventory", {}).items():
        path = Path(shown)
        observed = {
            "bytes": path.stat().st_size if path.is_file() else None,
            "resolved_blob": path.resolve().name if path.is_file() else None,
            "verification": "huggingface_content_addressed_blob_identity",
        }
        if observed != expected:
            raise DETError(
                f"DET1.3 model snapshot drifted: {shown}: "
                f"expected={expected} observed={observed}")
    return amendment_path


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def evidence_preflight(run_dir: Path) -> dict[str, Any]:
    """Adjudicate the purported live/replay pair without loading a model."""
    amendment_path = check_inventory(run_dir)
    registry_path = ROOT / "config" / "grm_live_registered_baselines.json"
    registry = read_json(registry_path)
    family = registry["families"]["supersession_battery_on_gpt_oss"]
    producer_path = ROOT / family["source"]["path"]
    producer = next(
        row for row in _jsonl(producer_path)
        if row.get("record_type") == "supersession_probe_receipt"
        and row.get("probe_id") == "harbor_restatement"
    )
    bisect_path = _one(run_dir / "det1_2", "bisect_receipt_*.json")
    bisect = read_json(bisect_path)
    control_path = _validate_record(bisect["arms"]["control_a"], "control_a")
    profile_path = _validate_record(bisect["arms"]["profile_only"], "profile_only")
    control = read_json(control_path)
    profile = read_json(profile_path)
    runtime_path = _one(run_dir, "runtime_frame_*.json")
    runtime = read_json(runtime_path)
    rows = [
        {
            "field": "model",
            "producer": "MiniCPM3-4B INT4 MLA",
            "replay": "GPT-OSS-20B resident_packed_mxfp4",
            "status": "DIVERGENT_INCOMPATIBLE_FRAME",
        },
        {
            "field": "checkpoint_revision",
            "producer": "NOT_RECORDED_DYNAMIC_MINICPM_SNAPSHOT",
            "replay": runtime["model"]["revision"],
            "status": "UNAVAILABLE_LEFT",
        },
        {
            "field": "topology",
            "producer": {"layers": 62, "hidden": 2560, "heads": 40},
            "replay": {
                "layers": control["model_info"]["layers"],
                "hidden": control["model_info"]["hidden_dim"],
                "heads": control["model_info"]["num_heads"],
                "kv_heads": control["model_info"]["num_kv_heads"],
            },
            "status": "DIVERGENT_INCOMPATIBLE_FRAME",
        },
        {
            "field": "cache_payload",
            "producer": {"kind": "MLA", "components": ["c", "kpe"]},
            "replay": {"kind": "GQA", "components": ["k", "v"]},
            "status": "DIVERGENT_INCOMPATIBLE_FRAME",
        },
        {
            "field": "arena",
            "producer": {"class": "ArenaCache", "width": 256, "route_layer": 44},
            "replay": {"class": "GptOssGQAArenaCache", "width": 96, "route_layer": 1},
            "status": "DIVERGENT_INCOMPATIBLE_FRAME",
        },
        {
            "field": "live_turn_policy",
            "producer": 0,
            "replay": 2,
            "status": "DIVERGENT_INCOMPATIBLE_FRAME",
        },
        {
            "field": "sink_and_prompt",
            "producer": "<conversation> plus plain User/Assistant",
            "replay": "GPT-OSS Harmony system/final template",
            "status": "DIVERGENT_INCOMPATIBLE_FRAME",
        },
        {
            "field": "probe_execution",
            "producer": {"max_trips": 0, "defer_memory": False},
            "replay": {"max_trips": 1, "defer_memory": True},
            "status": "DIVERGENT",
        },
        {
            "field": "raw_ranking",
            "producer": producer["ranking_nodes"],
            "replay": profile["profile"]["raw_ranking"],
            "status": "DIVERGENT",
        },
        {
            "field": "admission_rank_plan",
            "producer": producer["arena_info"]["admission_rank_plan"],
            "replay": profile["profile"]["served_admission_profile"]["rank_plan"],
            "status": "DIVERGENT",
        },
        {
            "field": "final_fixture_mount",
            "producer": producer["mounted_indices"],
            "replay": control["incident_projection"]["mounted_ids"],
            "status": "EQUAL_FIXTURE_LOCAL_ID_ONLY",
        },
        {
            "field": "answer",
            "producer": producer["answer_text"],
            "replay": control["incident_projection"]["answer"],
            "status": "DIVERGENT_CROSS_MODEL_BEHAVIOR",
        },
    ]
    for field in (
        "token_ids", "arena_kv_bytes", "rope_rows", "textual_sink_rows",
        "learned_sink_rows", "attention_masks", "live_token_ledger",
    ):
        rows.append({
            "field": field,
            "producer": "NOT_PERSISTED",
            "replay": "NOT_PERSISTED",
            "status": "NOT_CAPTURED_AND_NOT_COMPARABLE_FRAME",
        })
    return {
        "schema": "grm.det1_3.evidence_preflight.v1",
        "status": "INCOMPATIBLE_FRAME",
        "gate_pass": False,
        "order": file_record(ORDER),
        "source_amendment": file_record(amendment_path),
        "registration": file_record(_one(run_dir, "registration_*.json")),
        "registry": file_record(registry_path),
        "producer_receipt": file_record(producer_path),
        "bisect_receipt": file_record(bisect_path),
        "rows": rows,
        "adjudication": {
            "confirmed_class": "A_BASELINE_ORACLE_CONSTRUCTION_BUG",
            "same_model_state_divergence": "OPEN_NO_LIVED_GPT_OSS_SNAPSHOT",
            "reason": (
                "a cross-model task-acceptance comparator was interpreted as "
                "a same-model state/purity oracle"
            ),
        },
        "law": (
            "cross-frame task receipts cannot authorize state-byte claims; "
            "counterfactual purity requires complete same-frame live/replay snapshots"
        ),
        "race_resume_authorized": False,
    }


def _write_exclusive_or_verify(path: Path, value: Mapping[str, Any]) -> Path:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise DETError(f"immutable DET1.3 output collision: {path}")
        return path
    with path.open("xb") as handle:
        handle.write(payload)
    return path


def _gpu_probe() -> dict[str, Any]:
    process = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "returncode": int(process.returncode),
        "stdout": process.stdout.strip(),
        "stderr": process.stderr.strip(),
        "device_nodes": {
            path: Path(path).exists()
            for path in ("/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm")
        },
    }


def _process_instance() -> dict[str, Any]:
    pid = os.getpid()
    stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    close = stat_text.rfind(")")
    if close < 0:
        raise DETError("cannot parse /proc process start time")
    fields_after_comm = stat_text[close + 2:].split()
    # fields_after_comm[0] is stat field 3; starttime is stat field 22.
    start_ticks = int(fields_after_comm[19])
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="utf-8").strip()
    value = {
        "process_pid": int(pid),
        "process_start_ticks": start_ticks,
        "boot_id": boot_id,
    }
    value["process_instance_sha256"] = process_instance_sha256(
        value["process_pid"], value["process_start_ticks"], value["boot_id"])
    return value


def _is_refusal(answer: str) -> bool:
    lowered = str(answer).casefold().replace("\u2019", "'").replace("\u2018", "'")
    return any(marker in lowered for marker in (
        "i'm sorry", "i am sorry", "cannot help", "can't help",
        "do not know", "don't know", "cannot answer", "can't answer",
    ))


def _fixture_answer_correct(answer: str, fixture: Mapping[str, Any]) -> bool:
    probe = fixture["probes"][0]
    expected = [str(value) for value in probe["expected_values"]]
    rejected = [
        *[str(value) for value in probe["stale_values"]],
        *[str(value) for value in probe["wrong_fact_values"]],
    ]
    return bool(
        any(contains_value(answer, value) for value in expected)
        and not any(contains_value(answer, value) for value in rejected)
    )


def compare_with_observer_controls(
    lived_snapshot: Path,
    replay_snapshot: Path,
    lived_control_path: Path,
    replay_control_path: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Elevate state equality only after independent bare-answer controls."""
    comparison = compare_snapshots(lived_snapshot, replay_snapshot)
    snapshots = {
        "lived": load_snapshot(lived_snapshot),
        "replay": load_snapshot(replay_snapshot),
    }
    control_paths = {
        "lived": Path(lived_control_path),
        "replay": Path(replay_control_path),
    }
    controls = {arm: read_json(path) for arm, path in control_paths.items()}
    fixture = read_json(FIXTURE)
    errors: list[str] = []
    process_instances = {
        str(snapshots[arm].get("provenance", {}).get("process_instance_sha256"))
        for arm in ("lived", "replay")
    }
    for arm in ("lived", "replay"):
        snapshot = snapshots[arm]
        linked = snapshot.get("linked_answer") or {}
        provenance = snapshot.get("provenance") or {}
        control = controls[arm]
        control_provenance = control.get("execution_provenance") or {}
        if control.get("schema") != "grm.det1_3.same_model_answer.v1":
            errors.append(f"{arm.upper()}_CONTROL_SCHEMA")
        if control.get("status") != "COMPLETE" or control.get("arm") != arm:
            errors.append(f"{arm.upper()}_CONTROL_ROLE")
        for field in (
            "arm", "protocol", "protocol_source_sha256", "run_id",
            "order_sha256", "source_amendment_sha256", "registration_sha256",
            "runtime_frame_sha256", "fixture_sha256", "fixture_id", "probe_id",
            "probe_question_sha256", "probe_driver", "topk", "max_trips",
            "probe_ladder", "defer_memory", "boot_id",
        ):
            lv = provenance.get(field, "<MISSING>")
            rv = control_provenance.get(field, "<MISSING>")
            status = "EQUAL" if lv == rv and lv != "<MISSING>" else "DIVERGENT"
            comparison["rows"].append({
                "group": "observer_control",
                "field": f"observer_control.{arm}.provenance.{field}",
                "status": status,
                "left": lv,
                "right": rv,
            })
            if status != "EQUAL":
                errors.append(f"{arm.upper()}_CONTROL_PROVENANCE:{field}")
        control_process = control_provenance.get("process_instance_sha256")
        try:
            expected_control_process = process_instance_sha256(
                int(control_provenance["process_pid"]),
                int(control_provenance["process_start_ticks"]),
                str(control_provenance["boot_id"]),
            )
        except (KeyError, TypeError, ValueError):
            expected_control_process = None
        if control_process != expected_control_process:
            errors.append(f"{arm.upper()}_CONTROL_PROCESS_DIGEST_INVALID")
        if not control_process or control_process in process_instances:
            errors.append(f"{arm.upper()}_CONTROL_NOT_FRESH")
        process_instances.add(str(control_process))
        projections = (
            ("answer", "probe_answer"),
            ("mounted_ids", "probe_mounts"),
            ("selected_attempt", "probe_selected_attempt"),
            ("answer_correct", "probe_answer_correct"),
            ("answer_refusal", "probe_refusal"),
        )
        for control_field, linked_field in projections:
            lv = linked.get(linked_field, "<MISSING>")
            rv = control.get(control_field, "<MISSING>")
            status = "EQUAL" if lv == rv and lv != "<MISSING>" else "DIVERGENT"
            comparison["rows"].append({
                "group": "observer_control",
                "field": f"observer_control.{arm}.{linked_field}",
                "status": status,
                "left": lv,
                "right": rv,
            })
            if status != "EQUAL":
                errors.append(f"{arm.upper()}_CONTROL_BEHAVIOR:{linked_field}")
    lived_linked = snapshots["lived"].get("linked_answer") or {}
    lived_control = controls["lived"]
    lived_premise_checks = {
        "linked_answer_correct_recomputed": _fixture_answer_correct(
            str(lived_linked.get("probe_answer", "")), fixture),
        "linked_answer_not_refusal": (
            lived_linked.get("probe_refusal") is False
            and not _is_refusal(str(lived_linked.get("probe_answer", "")))
        ),
        "bare_control_correct_recomputed": _fixture_answer_correct(
            str(lived_control.get("answer", "")), fixture),
        "bare_control_not_refusal": (
            lived_control.get("answer_refusal") is False
            and not _is_refusal(str(lived_control.get("answer", "")))
        ),
    }
    legacy_correctness = {
        "linked_answer_receipted_correct": lived_linked.get(
            "probe_answer_correct") is True,
        "bare_control_receipted_correct": lived_control.get(
            "answer_correct") is True,
    }
    for field, passed in lived_premise_checks.items():
        comparison["rows"].append({
            "group": "lived_premise",
            "field": f"lived_premise.{field}",
            "status": "EQUAL" if passed else "DIVERGENT",
            "left": True,
            "right": bool(passed),
        })
    if not all(lived_premise_checks.values()):
        errors.append("LIVED_ANSWERING_PREMISE_NOT_REPRODUCED")
    comparison["normalization_adjudication"] = {
        "schema": "grm.det1_4.value_normalization_adjudication.v1",
        "rule": (
            "strip Markdown emphasis and map U+2010/U+2011 to ASCII hyphen "
            "for expected/stale/wrong-value comparison only"
        ),
        "raw_answer_behavior_comparison_unchanged": True,
        "normalized_recomputation_authoritative": True,
        "legacy_receipted_correctness": legacy_correctness,
    }
    comparison["behavioral_control_errors"] = errors
    comparison["lived_answering_premise"] = lived_premise_checks
    comparison["capture_behavioral_nonperturbation_pass"] = not errors
    comparison["controls"] = {
        arm: file_record(path) for arm, path in control_paths.items()
    }
    if comparison.get("state_gate_pass") and not errors:
        comparison["status"] = "PASS_CANONICAL_VALUE_BYTES_EQUAL"
        comparison["gate_pass"] = True
    elif comparison.get("state_gate_pass"):
        comparison["status"] = "CAPTURE_BEHAVIORAL_CONTROL_DIVERGENT"
        comparison["gate_pass"] = False
    comparison["race_resume_authorized"] = False
    comparison["race_resume_blocker"] = (
        "diagnostic-only source amendment requires a separately chained "
        "fix/fork amendment even after this fidelity gate passes"
    )
    comparison["counts"] = {
        key: sum(row["status"] == key for row in comparison["rows"])
        for key in (
            "EQUAL", "DIVERGENT", "MISSING_LEFT", "MISSING_RIGHT",
            "NOT_COMPARABLE_FRAME", "INVALID_PROVENANCE",
        )
    }
    if output_path is not None:
        _write_exclusive_or_verify(Path(output_path), comparison)
    return comparison


def _validate_production_gate_inputs(
    run_dir: Path,
    lived_snapshot: Path,
    replay_snapshot: Path,
    lived_control: Path,
    replay_control: Path,
) -> None:
    """Bind the production gate to this run's immutable capture chain."""
    from scripts.grm_det1_gpu import NATIVE_LIB

    amendment_path = check_inventory(run_dir)
    registration_path = _one(run_dir, "registration_*.json")
    runtime_path = _one(run_dir, "runtime_frame_*.json")
    runtime = read_json(runtime_path)
    if runtime.get("registration") != file_record(registration_path):
        raise DETError("DET1.3 comparison runtime registration binding drifted")
    if runtime.get("native_library") != file_record(NATIVE_LIB):
        raise DETError("DET1.3 comparison native runtime drifted")
    anchors = {
        "order": file_record(ORDER),
        "source_amendment": file_record(amendment_path),
        "runtime_frame": file_record(runtime_path),
        "fixture": file_record(FIXTURE),
    }
    expected_hashes = {
        "order_sha256": anchors["order"]["sha256"],
        "source_amendment_sha256": anchors["source_amendment"]["sha256"],
        "registration_sha256": file_record(registration_path)["sha256"],
        "runtime_frame_sha256": anchors["runtime_frame"]["sha256"],
        "fixture_sha256": anchors["fixture"]["sha256"],
    }
    snapshot_inputs = {
        "lived": Path(lived_snapshot),
        "replay": Path(replay_snapshot),
    }
    control_inputs = {
        "lived": Path(lived_control),
        "replay": Path(replay_control),
    }
    for arm, supplied in snapshot_inputs.items():
        manifest = Path(load_snapshot(supplied)["_manifest_path"]).resolve()
        try:
            relative = manifest.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise DETError(f"{arm} snapshot escapes DET1.3 run: {manifest}") from exc
        parts = relative.parts
        if not (
            len(parts) == 6
            and parts[:3] == ("det1_3", "snapshots", arm)
            and parts[3].startswith("attempt_")
            and parts[4:] == ("snapshot", "manifest.json")
        ):
            raise DETError(f"{arm} snapshot is not a registered arm artifact: {relative}")
        receipt_path = manifest.parents[1] / "capture_receipt.json"
        receipt = read_json(receipt_path)
        if (
            receipt.get("schema") != "grm.det1_3.capture_linked_answer.v1"
            or receipt.get("status") != "COMPLETE"
            or receipt.get("arm") != arm
            or receipt.get("protocol") != ARM_PROTOCOLS[arm]
            or receipt.get("snapshot") != file_record(manifest)
        ):
            raise DETError(f"{arm} capture receipt does not bind the snapshot")
        for field, expected in anchors.items():
            if receipt.get(field) != expected:
                raise DETError(f"{arm} capture receipt {field} binding drifted")
        provenance = load_snapshot(manifest).get("provenance") or {}
        for field, expected in expected_hashes.items():
            if provenance.get(field) != expected:
                raise DETError(f"{arm} snapshot current-run {field} binding drifted")

    for arm, supplied in control_inputs.items():
        path = supplied.expanduser().resolve()
        try:
            relative = path.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise DETError(f"{arm} control escapes DET1.3 run: {path}") from exc
        parts = relative.parts
        if not (
            len(parts) == 5
            and parts[:3] == ("det1_3", "answers", arm)
            and parts[3].startswith("attempt_")
            and parts[4] == "answer_receipt.json"
        ):
            raise DETError(f"{arm} control is not a registered arm artifact: {relative}")
        receipt = read_json(path)
        if (
            receipt.get("schema") != "grm.det1_3.same_model_answer.v1"
            or receipt.get("status") != "COMPLETE"
            or receipt.get("arm") != arm
            or receipt.get("protocol") != ARM_PROTOCOLS[arm]
        ):
            raise DETError(f"{arm} bare control role/protocol drifted")
        for field, expected in anchors.items():
            if receipt.get(field) != expected:
                raise DETError(f"{arm} bare control {field} binding drifted")
        provenance = receipt.get("execution_provenance") or {}
        for field, expected in expected_hashes.items():
            if provenance.get(field) != expected:
                raise DETError(f"{arm} control current-run {field} binding drifted")


def _write_failure(run_dir: Path, arm: str, exc: BaseException) -> Path:
    value = {
        "schema": "grm.det1_3.capture_failure.v1",
        "status": "BLOCKED_BEFORE_SNAPSHOT",
        "arm": str(arm),
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "traceback": traceback.format_exc(),
        "gpu": _gpu_probe(),
        "order": file_record(ORDER),
        "source_amendment": file_record(run_dir / AMENDMENT_NAME),
    }
    digest = hashlib.sha256(
        json.dumps(value, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return _write_exclusive_or_verify(
        run_dir / "det1_3" / "failures" / f"{arm}_{digest[:16]}.json", value)


def _split_fixture_turn(text: str) -> tuple[str, str]:
    prefix = "User: "
    marker = "\nAssistant: "
    if not text.startswith(prefix) or marker not in text:
        raise DETError(f"fixture turn is not a complete User/Assistant pair: {text!r}")
    user, assistant = text[len(prefix):].split(marker, 1)
    return user, assistant


def _install_lived_nodes(
    repo, e2e, fixture: Mapping[str, Any],
) -> tuple[dict[str, int], dict[int, list[int]]]:
    arena = repo.arena
    node_to_idx = {}
    token_ledgers = {}
    for node in fixture["nodes"]:
        user, assistant = _split_fixture_turn(str(node["text"]))
        turn_text = e2e.harmony_turn(user, assistant)
        token_ids = [int(value) for value in arena.encode(turn_text)]
        before_count = len(arena.grafts)
        returned = arena.feed(turn_text, deposit=True)
        after_count = len(arena.grafts)
        if after_count != before_count + 1:
            raise DETError(
                "chronological lived feed did not deposit exactly one graft: "
                f"before={before_count} after={after_count}")
        # Non-ephemeral ArenaCache.feed deposits but intentionally returns
        # None; ephemeral feed returns the deposit index. Derive the common
        # authoritative index from the validated append.
        index = after_count - 1
        if returned is not None and int(returned) != index:
            raise DETError(
                f"lived feed returned graft {returned}, appended graft {index}")
        node_to_idx[str(node["node_id"])] = index
        token_ledgers[index] = token_ids
        graft = arena.grafts[index]
        graft["node_id"] = index
        graft["kind"] = "fact"
        graft["metadata"] = {
            "kind": "fact", "active": True,
            "supersedes": [], "superseded_by": [],
        }
    for node in fixture["nodes"]:
        index = node_to_idx[str(node["node_id"])]
        older = [node_to_idx[str(value)] for value in node["supersedes"]]
        arena.grafts[index]["metadata"]["supersedes"] = older
        for old in older:
            arena.grafts[old]["metadata"]["superseded_by"].append(index)
    arena._bump_cuda_gqa_epoch()
    for index in range(len(arena.grafts)):
        repo._native_sync_node(index)
    return node_to_idx, token_ledgers


def _sha_array(value: Any) -> str:
    array = value if isinstance(value, np.ndarray) else value.numpy()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _frame_identity(
    model,
    tokenizer,
    e2e,
    registration: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    import tensor_cuda as tc

    tokenizer_path = Path(model.config.model_dir) / "tokenizer.json"
    engine_module = getattr(tc, "_C", tc)
    engine_path = Path(engine_module.__file__).resolve()
    prompt_contract = {
        "sink": e2e.HARMONY_SINK,
        "stops": list(e2e.HARMONY_STOPS),
        "function_source": inspect.getsource(e2e.harmony_turn),
    }
    return {
        "model_revision": str(model.config.revision),
        "model_weights_inventory_sha256": hashlib.sha256(
            json.dumps(
                registration["model_snapshot_inventory"],
                sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "engine_build_sha256": sha256_file(engine_path),
        "engine_path": str(engine_path),
        "prompt_template_sha256": hashlib.sha256(
            json.dumps(prompt_contract, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "native_library_sha256": runtime["native_library"]["sha256"],
        "runtime_contract_sha256": hashlib.sha256(
            json.dumps(
                {
                    "model": runtime["model"],
                    "resolved_flags": runtime["resolved_flags"],
                    "native_library": runtime["native_library"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "dialect": "gqa",
    }


def _run_protocol(
    run_dir: Path,
    arm: str,
    *,
    capture: bool,
    lease_seconds: int,
) -> tuple[Path | None, dict[str, Any] | None]:
    from scripts.grm_det1_2_gpu import _load_model_repo
    from scripts.grm_det1_gpu import (
        NATIVE_LIB,
        _install_fixture_nodes,
        _runtime_env,
        _validate_visibility,
    )
    from scripts.grm_det1_e2e import _mechanistic_counterfactual

    amendment_path = check_inventory(run_dir)
    registration_path = _one(run_dir, "registration_*.json")
    registration = read_json(registration_path)
    runtime_path = _one(run_dir, "runtime_frame_*.json")
    runtime = read_json(runtime_path)
    if runtime.get("registration") != file_record(registration_path):
        raise DETError("DET1.3 runtime frame registration binding drifted")
    if runtime.get("native_library") != file_record(NATIVE_LIB):
        raise DETError("DET1.3 native runtime drifted from the frozen frame")
    frozen_lease = int(runtime["gpu_lease"]["actual_cap_seconds"])
    if int(lease_seconds) != frozen_lease:
        raise DETError(
            "DET1.3 lease differs from frozen runtime frame: "
            f"requested={lease_seconds} frozen={frozen_lease}")
    os.environ.update(_runtime_env(runtime))
    visible = _validate_visibility()
    if str(visible) != str(runtime["cuda_visible_devices"]):
        raise DETError("DET1.3 visible GPU differs from the frozen runtime frame")
    fixture = read_json(FIXTURE)
    attempt_root = run_dir / "det1_3" / ("snapshots" if capture else "answers") / arm
    attempt_dir = attempt_root / f"attempt_{len(list(attempt_root.glob('attempt_*'))) + 1:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    e2e = model = tokenizer = repo = None
    try:
        e2e, model, tokenizer, repo, model_info = _load_model_repo(attempt_dir, runtime)
        if arm == "lived":
            _node_to_idx, live_token_ledgers = _install_lived_nodes(repo, e2e, fixture)
            protocol = ARM_PROTOCOLS["lived"]
            protocol_source = inspect.getsource(_install_lived_nodes)
        elif arm == "replay":
            _install_fixture_nodes(repo, fixture)
            live_token_ledgers = {}
            protocol = ARM_PROTOCOLS["replay"]
            protocol_source = inspect.getsource(_install_fixture_nodes)
        else:
            raise DETError(f"unsupported DET1.3 arm {arm}")

        arena = repo.arena

        def current_live_ids() -> list[int]:
            values = []
            for graft_index, count in arena.live_segs:
                if graft_index is None or int(graft_index) not in live_token_ledgers:
                    raise DETError(
                        "protocol cannot reconstruct exact model-visible live token ledger")
                ids = live_token_ledgers[int(graft_index)]
                if len(ids) != int(count):
                    raise DETError(
                        "live token ledger length differs from arena.live_segs")
                values.extend(ids)
            return values

        probe = fixture["probes"][0]
        question = str(probe["question"])
        expected = [str(value) for value in probe["expected_values"]]
        rejected = [
            *[str(value) for value in probe["stale_values"]],
            *[str(value) for value in probe["wrong_fact_values"]],
        ]
        process_instance = _process_instance()
        source_bundle = (
            protocol_source
            + inspect.getsource(e2e.probe_multimount_chat)
            + inspect.getsource(type(arena)._attempt)
        )
        base_provenance = {
            "schema": "grm.det1_3.capture_provenance.v2",
            "arm": arm,
            "protocol": protocol,
            "protocol_source_sha256": hashlib.sha256(
                source_bundle.encode("utf-8")).hexdigest(),
            "run_id": run_dir.name,
            "order_sha256": file_record(ORDER)["sha256"],
            "source_amendment_sha256": file_record(amendment_path)["sha256"],
            "registration_sha256": file_record(registration_path)["sha256"],
            "runtime_frame_sha256": file_record(runtime_path)["sha256"],
            "fixture_sha256": file_record(FIXTURE)["sha256"],
            "fixture_id": str(fixture["session_id"]),
            "probe_id": str(probe["probe_id"]),
            "probe_question_sha256": hashlib.sha256(
                question.encode("utf-8")).hexdigest(),
            "capture_boundary": "before_probe_prefill",
            "selection_boundary": "before_generation_and_grounding_selection",
            "probe_driver": "grm_e2e_session.probe_multimount_chat",
            "topk": 3,
            "max_trips": 1,
            "probe_ladder": True,
            "defer_memory": True,
            **process_instance,
        }

        if not capture:
            if arm == "replay":
                result = _mechanistic_counterfactual(
                    repo, e2e, e2e.probe_multimount_chat,
                    question=question, aliases=set(), planted_miss=False,
                    topk=3, ngen=32, max_trips=1, turn_idx=None,
                    expected=expected, stale=rejected,
                    active_detectors=(), observer_shell=False,
                )
                answer = str(result["answer"])
                mounts = [int(value) for value in result["mounted_ids"]]
                selected_attempt = int(
                    result.get("arena_info", {}).get("trip", 0) or 0)
            else:
                answer, _info = e2e.probe_multimount_chat(
                    repo, question, topk=3, ngen=32, defer_memory=True,
                    turn_idx=None, probe_ladder=True, max_trips=1,
                )
                mounts = [int(value) for value in arena.cur_mounts]
                selected_attempt = int((_info or {}).get("trip", 0) or 0)
            receipt = {
                "schema": "grm.det1_3.same_model_answer.v1",
                "status": "COMPLETE",
                "arm": arm,
                "protocol": protocol,
                "execution_provenance": {
                    **base_provenance,
                    "execution_mode": "bare_answer_control",
                },
                "answer": answer,
                "mounted_ids": mounts,
                "selected_attempt": selected_attempt,
                "answer_refusal": _is_refusal(answer),
                "answer_correct": bool(
                    any(contains_value(answer, value) for value in expected)
                    and not any(contains_value(answer, value) for value in rejected)
                ),
                "model_info": model_info,
                "order": file_record(ORDER),
                "source_amendment": file_record(amendment_path),
                "runtime_frame": file_record(runtime_path),
                "fixture": file_record(FIXTURE),
            }
            path = _write_exclusive_or_verify(attempt_dir / "answer_receipt.json", receipt)
            return path, receipt

        holder: dict[str, Any] = {
            "profile": None,
            "probe_key_sha256": None,
            "attempt_context": None,
            "attempt_results": [],
            "captured": None,
        }
        original_admission = e2e.decisive_admission_profile
        original_probe_key = arena._probe_key
        original_attempt = arena._attempt

        def admission_wrapped(*args, **kwargs):
            value = original_admission(*args, **kwargs)
            holder["profile"] = dict(value)
            return value

        def probe_key_wrapped(*args, **kwargs):
            value = original_probe_key(*args, **kwargs)
            holder["probe_key_sha256"] = _sha_array(value)
            return value

        def admission_supplier() -> dict[str, Any]:
            profile = dict(holder.get("profile") or {})
            context = dict(holder.get("attempt_context") or {})
            picks = [int(value) for value in context.get("fitted", ())]
            planned = [int(value) for value in context.get("planned", ())]
            rank_plan = [int(value) for value in profile.get("rank_plan", planned)]
            return {
                "schema": "grm.det1_3.admission_snapshot.v2",
                "probe_key_sha256": holder.get("probe_key_sha256"),
                "ranking": [int(value) for value in profile.get("ranking", ())],
                "identifier_tokens": [
                    str(value) for value in profile.get("identifier_tokens", ())],
                "identified_candidates": [
                    int(value) for value in profile.get("identified_candidates", ())],
                "policy_branch": profile.get("policy_branch"),
                "rank_plan": rank_plan,
                "ladder_attempts": list(context.get("ladder_attempts", ())),
                "current_attempt_ordinal": context.get("ordinal"),
                "current_planned": planned,
                "current_clean_room": context.get("clean_room"),
                "current_fitted": picks,
                "current_dropped": [
                    value for value in planned if value not in set(picks)],
                "selection_state": "PENDING_AT_PREFILL",
                "final_mounts": [int(value) for value in arena.cur_mounts],
                "route_backend": profile.get("route_backend"),
                "route_margin_1_2": profile.get("route_margin_1_2"),
                "margin_threshold": profile.get("margin_threshold"),
                "rule_sha256": profile.get("rule_sha256"),
            }

        identity = _frame_identity(model, tokenizer, e2e, registration, runtime)

        def attempt_wrapped(user_text, picks, *args, **kwargs):
            frame = inspect.currentframe()
            caller = None if frame is None else frame.f_back
            try:
                if caller is None or caller.f_code is not e2e._probe_ladder_chat.__code__:
                    raise DETError(
                        "DET1.3 attempt hook was not called by the registered ladder")
                local = caller.f_locals
                schedule = [
                    {
                        "ordinal": int(index),
                        "planned": [int(value) for value in planned],
                        "clean_room": bool(clean),
                    }
                    for index, (planned, clean) in enumerate(local["attempts"])
                ]
                context = {
                    "ordinal": int(local["trip"]),
                    "planned": [int(value) for value in local["planned"]],
                    "clean_room": bool(local["clean"]),
                    "fitted": [int(value) for value in picks],
                    "ladder_attempts": schedule,
                }
            finally:
                del frame
                del caller
            holder["attempt_context"] = context
            if holder["captured"] is not None:
                answer, info = original_attempt(user_text, picks, *args, **kwargs)
                holder["attempt_results"].append({
                    "ordinal": context["ordinal"],
                    "answer": str(answer),
                    "mounts": [int(value) for value in arena.cur_mounts],
                })
                return answer, info

            def provenance_supplier() -> dict[str, Any]:
                return {**base_provenance, "attempt_ordinal": context["ordinal"]}

            with stop_at_next_forward(
                arena,
                attempt_dir / "snapshot",
                label=arm,
                provenance=provenance_supplier,
                question=question,
                admission_plan=admission_supplier,
                live_token_ids=current_live_ids,
                sink_text=e2e.HARMONY_SINK,
                sink_token_ids=arena.encode(e2e.HARMONY_SINK),
                explicit_identity=identity,
            ) as capture_result:
                answer, info = original_attempt(user_text, picks, *args, **kwargs)
            captured = {
                "manifest_path": Path(capture_result["manifest_path"]),
                "ordinal": context["ordinal"],
                "answer": str(answer),
                "mounts": [int(value) for value in arena.cur_mounts],
            }
            holder["captured"] = captured
            holder["attempt_results"].append(dict(captured, manifest_path=None))
            return answer, info

        e2e.decisive_admission_profile = admission_wrapped
        arena._probe_key = probe_key_wrapped
        arena._attempt = attempt_wrapped
        probe_answer = None
        probe_mounts: list[int] = []
        probe_selected_attempt = None
        try:
            if arm == "replay":
                result = _mechanistic_counterfactual(
                    repo, e2e, e2e.probe_multimount_chat,
                    question=question, aliases=set(), planted_miss=False,
                    topk=3, ngen=32, max_trips=1, turn_idx=None,
                    expected=expected, stale=rejected,
                    active_detectors=(), observer_shell=False,
                )
                probe_answer = str(result["answer"])
                probe_mounts = [int(value) for value in result["mounted_ids"]]
                probe_selected_attempt = int(
                    result.get("arena_info", {}).get("trip", 0) or 0)
            else:
                probe_answer, probe_info = e2e.probe_multimount_chat(
                    repo, question, topk=3, ngen=32, defer_memory=True,
                    turn_idx=None, probe_ladder=True, max_trips=1,
                )
                probe_answer = str(probe_answer)
                probe_mounts = [int(value) for value in arena.cur_mounts]
                probe_selected_attempt = int((probe_info or {}).get("trip", 0) or 0)
        finally:
            e2e.decisive_admission_profile = original_admission
            arena._probe_key = original_probe_key
            arena._attempt = original_attempt
        captured = holder.get("captured")
        if captured is None or probe_answer is None or probe_selected_attempt is None:
            raise DETError("DET1.3 target prefill/linked answer was not completed")
        attempt_answer = str(captured["answer"])
        linked_answer = {
            "schema": "grm.det1_3.same_process_linked_answer.v1",
            "process_instance_sha256": process_instance["process_instance_sha256"],
            "captured_attempt_ordinal": int(captured["ordinal"]),
            "attempt_answer": attempt_answer,
            "attempt_mounts": [int(value) for value in captured["mounts"]],
            "attempt_answer_correct": bool(
                any(contains_value(attempt_answer, value) for value in expected)
                and not any(contains_value(attempt_answer, value) for value in rejected)
            ),
            "attempt_refusal": _is_refusal(attempt_answer),
            "probe_answer": probe_answer,
            "probe_selected_attempt": int(probe_selected_attempt),
            "probe_mounts": probe_mounts,
            "probe_answer_correct": bool(
                any(contains_value(probe_answer, value) for value in expected)
                and not any(contains_value(probe_answer, value) for value in rejected)
            ),
            "probe_refusal": _is_refusal(probe_answer),
        }
        manifest_path = finalize_snapshot_with_answer(
            Path(captured["manifest_path"]), linked_answer)
        receipt = {
            "schema": "grm.det1_3.capture_linked_answer.v1",
            "status": "COMPLETE",
            "arm": arm,
            "protocol": protocol,
            "snapshot": file_record(manifest_path),
            "linked_answer": linked_answer,
            "attempt_results": holder["attempt_results"],
            "order": file_record(ORDER),
            "source_amendment": file_record(amendment_path),
            "runtime_frame": file_record(runtime_path),
            "fixture": file_record(FIXTURE),
        }
        path = _write_exclusive_or_verify(
            attempt_dir / "capture_receipt.json", receipt)
        return path, receipt
    finally:
        repo = tokenizer = model = e2e = None
        gc.collect()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preflight", "capture", "answer", "compare"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("lived", "replay"))
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--lived-control", type=Path)
    parser.add_argument("--replay-control", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lease-seconds", type=int, default=580)
    parser.add_argument("--lock-wait-seconds", type=int, default=7200)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    allowed = (ROOT / "artifacts" / "grm_det1").resolve()
    if not run_dir.is_dir() or not run_dir.is_relative_to(allowed):
        raise DETError(f"run dir must be beneath {allowed}: {run_dir}")
    if args.command == "preflight":
        value = evidence_preflight(run_dir)
        digest = hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        path = _write_exclusive_or_verify(
            run_dir / "det1_3" / f"evidence_preflight_{digest[:16]}.json", value)
        print(json.dumps({"status": value["status"], "receipt": str(path)}))
        return 2
    if args.command == "compare":
        if any(value is None for value in (
            args.left, args.right, args.lived_control, args.replay_control,
            args.output,
        )):
            raise DETError(
                "compare requires --left/--right finalized snapshots and "
                "--lived-control/--replay-control bare answer receipts plus "
                "an append-only --output")
        output_path = args.output.expanduser().resolve()
        comparison_root = (run_dir / "det1_3" / "comparisons").resolve()
        if not output_path.is_relative_to(comparison_root):
            raise DETError(f"comparison output must be beneath {comparison_root}")
        _validate_production_gate_inputs(
            run_dir,
            args.left,
            args.right,
            args.lived_control,
            args.replay_control,
        )
        value = compare_with_observer_controls(
            args.left,
            args.right,
            args.lived_control,
            args.replay_control,
            output_path=output_path,
        )
        print(json.dumps({"status": value["status"], "gate_pass": value["gate_pass"]}))
        return 0 if value["gate_pass"] else 2
    if args.arm is None:
        raise DETError(f"{args.command} requires --arm")
    from scripts.grm_det1_gpu import gpu_lease

    try:
        with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
            path, _value = _run_protocol(
                run_dir,
                str(args.arm),
                capture=args.command == "capture",
                lease_seconds=int(args.lease_seconds),
            )
    except BaseException as exc:
        failure = _write_failure(run_dir, str(args.arm), exc)
        print(json.dumps({"status": "BLOCKED", "failure": str(failure)}))
        raise
    print(json.dumps({"status": "COMPLETE", "artifact": str(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
