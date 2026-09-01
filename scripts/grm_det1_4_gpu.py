#!/usr/bin/env python3
"""Inventory, preflight, and isolated ZERO fork gate for ORDER GRM-DET1.4.

Inventory and preflight are CPU-only.  ``zero-gate`` is the separately invoked
GPU stage that hydrates the frozen lived Harbor snapshot and resumes directly
at its captured prefill; it never authorizes detector or race evidence because
that requires an inline same-process lived capture and full routing index.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import stat
import sys
import time
from typing import Any, Mapping

from grm_det1_common import (
    DETError,
    ROOT,
    canonical_json_bytes,
    file_record,
    normalize_value_text,
    read_json,
    utc_now,
    write_content_addressed,
    write_json_exclusive,
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FROZEN_RUN = ROOT / "artifacts" / "grm_det1" / "run_20260831T160525Z_2"
ORDER = ROOT / "orders" / "GRM_DET1_4_FORK_FROM_SNAPSHOT.md"
REGISTRATION = FROZEN_RUN / "registration_62cb6c09cbec211d.json"
RUNTIME_FRAME = FROZEN_RUN / "runtime_frame_28b3196f8fb04a41.json"
RUN_MANIFEST = FROZEN_RUN / "run_manifest.json"
PRIOR_AMENDMENT = FROZEN_RUN / "det1_3_source_amendment.json"
DET1_4_AMENDMENT = FROZEN_RUN / "det1_4_source_amendment.json"
CAPTURE_RECEIPT = (
    FROZEN_RUN
    / "det1_3/snapshots/lived/attempt_005/capture_receipt.json"
)
LIVED_MANIFEST = (
    FROZEN_RUN
    / "det1_3/snapshots/lived/attempt_005/snapshot/manifest.json"
)
FIXTURE = (
    ROOT / "tests/fixtures/supersession_battery/correction_then_restatement.json"
)
OLD_COMPARE = FROZEN_RUN / "det1_3/comparisons/compare_result.json"
TENSOR_CUDA_ROOT = Path("/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
TENSOR_CUDA_EXTENSION = (
    TENSOR_CUDA_ROOT
    / "tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so"
)

STAGE_ORDER = (
    "zero_gate",
    "det_g0",
    "all_hooks_g1",
    "calibration",
    "eval_e2e",
    "eval_sup",
    "report",
)
REQUIRED_DEVICE_NODES = (
    Path("/dev/nvidia0"),
    Path("/dev/nvidiactl"),
    Path("/dev/nvidia-uvm"),
)
FORK_LAW = (
    "a counterfactual replay may only fork from a lived snapshot's actual "
    "bytes; reconstruction is not a valid counterfactual substrate."
)
OLD_COMPARE_LAW = (
    "replay may serve as a counterfactual only when both complete same-frame "
    "snapshots have equal canonical bytes and independent bare-answer controls "
    "prove that capture did not move behavior"
)
HISTORICAL_DISPOSITION = "HISTORICAL_ONLY_REJECTED_AS_DET1_4_EVIDENCE"
REQUIRED_AMENDMENT_SOURCES = (
    "core/gpt_oss20b_tc.py",
    "scripts/grm_det1_3_gpu.py",
    "scripts/grm_det1_3_snapshot.py",
    "scripts/grm_det1_4_gpu.py",
    "scripts/grm_det1_baseline_registry.py",
    "scripts/grm_det1_common.py",
    "scripts/grm_det1_e2e.py",
    "tests/test_grm_det1_3_snapshot.py",
    "tests/test_grm_det1_baseline_registry.py",
)


# These records are frozen evidence, not a discovery heuristic.  Any drift is
# an error even if the replacement JSON happens to have plausible fields.
FROZEN_RECORDS: dict[Path, tuple[int, str]] = {
    ORDER: (
        2018,
        "7123a2bcc17046a5eb3e9d03d6816c503fc53e5379d8a7de6fbd57683bc2b764",
    ),
    REGISTRATION: (
        21406,
        "62cb6c09cbec211d8cf897b1e67975ff7cbcd30f6d949f2c0b7d65dcc121fcc7",
    ),
    RUNTIME_FRAME: (
        2180,
        "28b3196f8fb04a4123fcf21b1e61f8bb1b8a0e64fadcfc6643801f3e19c4ac02",
    ),
    RUN_MANIFEST: (
        434,
        "c12a17363d4d38eee68db24e634b7ba434f45ec14c74f204aa1cf813e2a1eb7f",
    ),
    PRIOR_AMENDMENT: (
        4658,
        "671aff9d29a4b7751d24d7c5170bbf31d6285cab4e91b6b00e6521338128b256",
    ),
    CAPTURE_RECEIPT: (
        2184,
        "8dd524fa6c814edf2c4eca028324de20b7f451890ff768994997a22739acceaa",
    ),
    LIVED_MANIFEST: (
        140808,
        "b8ee7257ec3eebb2888b707849af4acc1c0d55845d793560cafdb58bc816cbec",
    ),
    OLD_COMPARE: (
        465551,
        "1b1e9b087aba33acd7028ed282d2231b43c47291a6bf2001aa459027d88ef344",
    ),
    TENSOR_CUDA_EXTENSION: (
        10798464,
        "7f438cc7ceb66a16b492d59185f2bd9940f0ecfa44d05514499c89ac2195b250",
    ),
}

HISTORICAL_ANCHORS: dict[str, tuple[Path, int, str]] = {
    "g1_flags_off_only": (
        FROZEN_RUN / "g1/g1_receipt_d83aff09c3478d0d.json",
        1619,
        "d83aff09c3478d0d480e7c48382c3a8deb9c00c3449e4d755c8410c59cf330d0",
    ),
    "calibration_rows": (
        FROZEN_RUN / "calibration/e2e_rows.jsonl",
        23840,
        "96dff529784e4b448e8c2ca4af1558919a299126c73aa409ee5e341ba52f8eab",
    ),
    "thresholds": (
        FROZEN_RUN / "thresholds_e5a0298a2486af7b.json",
        2207,
        "e5a0298a2486af7b2b7a906625125be640bfe8690f250fe7947fa6186dde67c3",
    ),
    "eval_e2e_rows": (
        FROZEN_RUN / "eval/e2e_rows.jsonl",
        160823,
        "d055741fc4288f7d39aa57ca645d2e10b23b7c913fc388a5a68b1cd741def495",
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DETError(message)


def _frozen_record(path: Path) -> dict[str, Any]:
    expected_bytes, expected_sha256 = FROZEN_RECORDS[path]
    _require(path.is_file(), f"missing frozen file: {path}")
    record = file_record(path)
    _require(
        record["bytes"] == expected_bytes,
        f"frozen byte-count drift for {record['path']}: "
        f"{record['bytes']} != {expected_bytes}",
    )
    _require(
        record["sha256"] == expected_sha256,
        f"frozen digest drift for {record['path']}: "
        f"{record['sha256']} != {expected_sha256}",
    )
    return record


def _record_equal(actual: Mapping[str, Any], expected: Mapping[str, Any], where: str) -> None:
    for key in ("path", "bytes", "sha256"):
        _require(
            actual.get(key) == expected.get(key),
            f"{where}.{key} does not bind the frozen record",
        )


def _compact_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_order(record: Mapping[str, Any]) -> dict[str, Any]:
    text = ORDER.read_text(encoding="utf-8")
    semantic_text = " ".join(text.replace("**", "").split())
    _require(FORK_LAW in semantic_text,
             "DET1.4 order is missing the registered fork law")
    _require("214 divergent" in semantic_text and "1191 equal" in semantic_text,
             "DET1.4 order is missing the adjudicated comparison counts")
    _require("ZERO interventions" in semantic_text,
             "DET1.4 order is missing the zero gate")
    return {"record": dict(record), "registered_law": FORK_LAW}


def _validate_registration(record: Mapping[str, Any]) -> dict[str, Any]:
    value = read_json(REGISTRATION)
    _require(value.get("schema") == "grm.det1.registration.v1",
             "wrong registration schema")
    _require(value.get("registered_pre_data") is True,
             "registration is not frozen pre-data")
    _require(
        value.get("verbal_question_exact")
        == "Do you have the information needed to answer? YES/NO",
        "registered verbal question drift",
    )
    fixtures = value.get("fixtures")
    _require(isinstance(fixtures, list) and len(fixtures) == 14,
             "registration must contain the frozen 14 fixtures")
    split = value.get("split_rule") or {}
    _require(split.get("calibration_pair_count") == 2,
             "frozen calibration pair count is not 2")
    _require(split.get("eval_pair_count") == 12,
             "frozen evaluation pair count is not 12")
    _require(len(split.get("eval_e2e") or []) == 7,
             "frozen E2E evaluation split is not 7 pairs")
    _require(len(split.get("eval_supersession") or []) == 5,
             "frozen supersession split is not 5 pairs")
    return {
        "record": dict(record),
        "schema": value["schema"],
        "registered_pre_data": True,
        "fixture_count": 14,
        "calibration_pair_count": 2,
        "eval_pair_count": 12,
    }


def _validate_runtime(record: Mapping[str, Any], registration: Mapping[str, Any]) -> dict[str, Any]:
    value = read_json(RUNTIME_FRAME)
    _require(value.get("schema") == "grm.det1.runtime_frame.v1",
             "wrong runtime-frame schema")
    _require(value.get("source_inventory_match") is True,
             "frozen runtime source inventory did not match")
    _require(value.get("cuda_visible_devices") == "0",
             "frozen runtime was not pinned to CUDA device 0")
    _record_equal(value.get("registration") or {}, registration,
                  "runtime_frame.registration")
    model = value.get("model") or {}
    _require(
        model.get("revision") == "6cee5e81ee83917806bbde320786a8fb61efebee",
        "frozen model revision drift",
    )
    flags = value.get("resolved_flags") or {}
    _require(flags.get("adm_decisive") is True and flags.get("gqa_cuda_route") is True,
             "frozen runtime lacks registered A-DEC/GQA routing")
    return {
        "record": dict(record),
        "schema": value["schema"],
        "source_inventory_match": True,
        "cuda_visible_devices": "0",
        "model_revision": model["revision"],
        "resolved_flags": flags,
    }


def _validate_prior_amendment(record: Mapping[str, Any]) -> dict[str, Any]:
    value = read_json(PRIOR_AMENDMENT)
    _require(value.get("schema") == "grm.det1_3.source_amendment.v1",
             "wrong DET1.3 amendment schema")
    _require(value.get("status") == "AUTHORIZED_DIAGNOSTIC_AMENDMENT",
             "wrong DET1.3 amendment status")
    invariants = value.get("invariants") or {}
    _require(invariants.get("race_resume_authorized") is False,
             "DET1.3 amendment unexpectedly authorizes race resume")
    _require(invariants.get("replay_fix_authorized") is False,
             "DET1.3 amendment unexpectedly authorizes replay fixes")
    return {
        "record": dict(record),
        "schema": value["schema"],
        "status": value["status"],
        "race_resume_authorized": False,
        "replay_fix_authorized": False,
    }


def _source_entry(value: Mapping[str, Any], path: str) -> Mapping[str, Any] | None:
    for field in ("added_sources", "test_sources"):
        added = value.get(field) or {}
        if isinstance(added, Mapping) and isinstance(added.get(path), Mapping):
            return added[path]
    changed = value.get("allowed_source_changes") or {}
    if isinstance(changed, Mapping) and isinstance(changed.get(path), Mapping):
        entry = changed[path]
        if isinstance(entry.get("after"), Mapping):
            return entry["after"]
    return None


def _validate_det1_4_amendment(
    *,
    required: bool,
    bindings: Mapping[str, Mapping[str, Any]],
    source_rebindings: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not DET1_4_AMENDMENT.exists():
        _require(not required, f"required DET1.4 amendment is missing: {DET1_4_AMENDMENT}")
        return {
            "path": str(DET1_4_AMENDMENT.relative_to(ROOT)),
            "status": "PENDING_REQUIRED_BEFORE_ZERO_GATE",
            "required_by_cli": False,
        }
    _require(DET1_4_AMENDMENT.is_file(), "DET1.4 amendment path is not a file")
    value = read_json(DET1_4_AMENDMENT)
    _require(value.get("schema") == "grm.det1_4.source_amendment.v1",
             "wrong DET1.4 amendment schema")
    _require(value.get("status") == "AUTHORIZED_FORK_FROM_SNAPSHOT_AMENDMENT",
             "wrong DET1.4 amendment status")
    for key in ("order", "registration", "runtime_frame", "prior_amendment"):
        _record_equal(value.get(key) or {}, bindings[key], f"det1_4_amendment.{key}")
    source_bindings: dict[str, Any] = {}
    superseded_sources: list[str] = []
    for source_path in REQUIRED_AMENDMENT_SOURCES:
        source_entry = _source_entry(value, source_path)
        _require(source_entry is not None,
                 f"DET1.4 amendment does not bind {source_path}")
        current = file_record(ROOT / source_path)
        matches_original = (
            source_entry.get("bytes") == current["bytes"]
            and source_entry.get("sha256") == current["sha256"]
        )
        if not matches_original:
            rebinding = (source_rebindings or {}).get(source_path)
            _require(
                isinstance(rebinding, Mapping),
                f"DET1.4 amendment binding does not match current bytes: {source_path}",
            )
            historical = {
                "path": source_path,
                "bytes": source_entry.get("bytes"),
                "sha256": source_entry.get("sha256"),
            }
            _record_equal(
                rebinding.get("before") or {}, historical,
                f"DET1.4 superseding amendment.{source_path}.before",
            )
            _record_equal(
                rebinding.get("after") or {}, current,
                f"DET1.4 superseding amendment.{source_path}.after",
            )
            superseded_sources.append(source_path)
        source_bindings[source_path] = current
    invariants = value.get("invariants") or {}
    _require(invariants.get("reconstruction_as_counterfactual") in (False, "FORBIDDEN"),
             "DET1.4 amendment does not forbid reconstructed counterfactual state")
    race_authorized = invariants.get("race_resume_authorized")
    _require(
        race_authorized is False
        or (
            isinstance(race_authorized, str)
            and race_authorized.upper().startswith(("PENDING", "CONDITIONAL"))
        ),
        "DET1.4 amendment prematurely authorizes race resume",
    )
    invariant_text = json.dumps(invariants, sort_keys=True).upper().replace("-", "_")
    for required_term in ("ZERO_GATE", "G0", "G1", "CALIBRATION"):
        _require(required_term in invariant_text,
                 f"DET1.4 amendment race condition omits {required_term}")
    return {
        "record": file_record(DET1_4_AMENDMENT),
        "schema": value["schema"],
        "status": value["status"],
        "required_by_cli": bool(required),
        "source_bindings": source_bindings,
        "superseded_sources": sorted(superseded_sources),
        "race_resume_authorized": race_authorized,
    }


def _validate_lived_capture(
    receipt_record: Mapping[str, Any],
    manifest_record: Mapping[str, Any],
    runtime_record: Mapping[str, Any],
    prior_record: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = read_json(CAPTURE_RECEIPT)
    _require(receipt.get("schema") == "grm.det1_3.capture_linked_answer.v1",
             "wrong lived capture receipt schema")
    _require(receipt.get("status") == "COMPLETE" and receipt.get("arm") == "lived",
             "lived capture receipt is not COMPLETE/lived")
    _record_equal(receipt.get("snapshot") or {}, manifest_record,
                  "capture_receipt.snapshot")
    _record_equal(receipt.get("runtime_frame") or {}, runtime_record,
                  "capture_receipt.runtime_frame")
    _record_equal(receipt.get("source_amendment") or {}, prior_record,
                  "capture_receipt.source_amendment")

    linked = receipt.get("linked_answer") or {}
    lived_answer = "The current Harbor token value is **Nacre\u20116\u2011Blue**."
    _require(linked.get("attempt_answer") == lived_answer,
             "lived Harbor attempt answer bytes drift")
    _require(linked.get("probe_answer") == lived_answer,
             "lived Harbor probe answer bytes drift")
    _require(linked.get("attempt_refusal") is False and linked.get("probe_refusal") is False,
             "lived Harbor control is a refusal")
    normalized = normalize_value_text(lived_answer)
    _require("nacre-6-blue" in normalized,
             "registered DET1.4 value normalization does not recover Harbor value")

    manifest = read_json(LIVED_MANIFEST)
    _require(manifest.get("schema") == "grm.det1_3.model_visible_snapshot.v2",
             "wrong lived snapshot schema")
    _require(manifest.get("complete") is True and
             manifest.get("capture_finalized") is True,
             "lived snapshot is not complete/finalized")
    _require(manifest.get("label") == "lived" and
             manifest.get("phase") == "before_probe_prefill",
             "lived snapshot label/phase drift")
    payload_digest = manifest.get("manifest_payload_sha256")
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    _require(_compact_digest(payload) == payload_digest,
             "lived manifest payload digest is invalid")
    identity = manifest.get("identity") or {}
    _require(
        identity.get("frame_sha256")
        == "2bfd827b4d031ef719fe8fc805612c49932ab3f91e4b8e03fb53889c364338c5",
        "lived snapshot frame identity drift",
    )
    _require(identity.get("engine_build_sha256") == FROZEN_RECORDS[TENSOR_CUDA_EXTENSION][1],
             "lived snapshot TensorCUDA build identity drift")
    availability = manifest.get("availability") or {}
    _require(not availability.get("missing_groups"), "lived snapshot has missing groups")
    _require(not availability.get("unavailable_fields"),
             "lived snapshot has unavailable fields")
    semantics = manifest.get("byte_semantics") or {}
    _require(semantics.get("comparison_bytes") == "C_contiguous_host_value_bytes",
             "lived snapshot byte semantics drift")

    arrays = manifest.get("arrays")
    _require(isinstance(arrays, dict) and len(arrays) == 199,
             "lived snapshot must bind exactly 199 blobs")
    expected_groups = {
        "arena_kv": 96,
        "masks": 24,
        "positions_rope": 2,
        "sink_rows": 72,
        "text_tokens": 4,
        "live_tokens": 1,
    }
    groups: Counter[str] = Counter()
    seen_blobs: set[Path] = set()
    snapshot_root = LIVED_MANIFEST.parent.resolve()
    blob_bytes = 0
    for field, entry in arrays.items():
        _require(isinstance(entry, dict), f"array record is not an object: {field}")
        relative = entry.get("blob")
        _require(isinstance(relative, str) and relative,
                 f"array record lacks blob path: {field}")
        blob = (snapshot_root / relative).resolve()
        _require(blob.is_relative_to(snapshot_root),
                 f"array blob escapes snapshot root: {field}")
        _require(blob not in seen_blobs, f"duplicate blob path: {relative}")
        seen_blobs.add(blob)
        _require(blob.is_file(), f"missing lived blob: {blob}")
        size = blob.stat().st_size
        _require(size == entry.get("bytes"), f"blob byte-count drift: {field}")
        digest = hashlib.sha256(blob.read_bytes()).hexdigest()
        _require(digest == entry.get("sha256"), f"blob digest drift: {field}")
        blob_bytes += size
        groups[str(entry.get("group"))] += 1
    _require(dict(groups) == expected_groups,
             f"lived snapshot blob group counts drift: {dict(groups)}")
    return {
        "capture_receipt": dict(receipt_record),
        "snapshot_manifest": dict(manifest_record),
        "snapshot_schema": manifest["schema"],
        "snapshot_frame_sha256": identity["frame_sha256"],
        "snapshot_blob_count": len(arrays),
        "snapshot_blob_bytes": blob_bytes,
        "snapshot_array_group_counts": dict(groups),
        "byte_semantics": semantics,
        "lived_answer_raw": lived_answer,
        "lived_answer_normalized": normalized,
        "old_correct_flags": {
            "attempt_answer_correct": linked.get("attempt_answer_correct"),
            "probe_answer_correct": linked.get("probe_answer_correct"),
            "disposition": "KNOWN_PRE_DET1_4_COMPARATOR_NORMALIZATION_BUG",
        },
    }


def _validate_old_compare(record: Mapping[str, Any]) -> dict[str, Any]:
    value = read_json(OLD_COMPARE)
    _require(value.get("schema") == "grm.det1_3.dual_snapshot_comparison.v2",
             "wrong old comparison schema")
    expected_counts = {
        "DIVERGENT": 214,
        "EQUAL": 1191,
        "INVALID_PROVENANCE": 0,
        "MISSING_LEFT": 12,
        "MISSING_RIGHT": 2,
        "NOT_COMPARABLE_FRAME": 0,
    }
    _require(value.get("counts") == expected_counts,
             f"old comparison counts drift: {value.get('counts')}")
    _require(value.get("law") == OLD_COMPARE_LAW, "old comparison law drift")
    _require(value.get("status") == "DIVERGENT",
             "old comparison no longer records DIVERGENT")
    _require(value.get("state_gate_pass") is False and
             value.get("gate_pass") is False and
             value.get("race_resume_authorized") is False,
             "old comparison unexpectedly authorizes replay/race")
    rows = value.get("rows") or []
    _require(len(rows) == 1423, "old comparison row count drift")
    divergent_groups = Counter(
        str(row.get("group")) for row in rows if row.get("status") == "DIVERGENT"
    )
    expected_divergent = {
        "admission_dynamic": 86,
        "arena_kv": 96,
        "linked_behavior": 4,
        "lived_premise": 4,
        "masks": 24,
    }
    # The receipt used one of two equivalent names for the admission group;
    # normalize only that historical label, never counts or statuses.
    if "dynamic_scalar" in divergent_groups:
        divergent_groups["admission_dynamic"] += divergent_groups.pop("dynamic_scalar")
    _require(dict(divergent_groups) == expected_divergent,
             f"old divergent group counts drift: {dict(divergent_groups)}")
    return {
        "record": dict(record),
        "schema": value["schema"],
        "status": value["status"],
        "counts": expected_counts,
        "divergent_group_counts": dict(divergent_groups),
        "law": value["law"],
        "race_resume_authorized": False,
        "DET1_4_adjudication": FORK_LAW,
    }


def _historical_evidence() -> dict[str, Any]:
    anchors: dict[str, Any] = {}
    for label, (path, expected_bytes, expected_sha256) in HISTORICAL_ANCHORS.items():
        _require(path.is_file(), f"missing historical anchor: {path}")
        record = file_record(path)
        _require(record["bytes"] == expected_bytes and
                 record["sha256"] == expected_sha256,
                 f"historical anchor drift: {record['path']}")
        anchors[label] = {
            "record": record,
            "disposition": HISTORICAL_DISPOSITION,
        }
    old_markers = sorted(
        str(path.relative_to(ROOT))
        for directory in (FROZEN_RUN / "g1", FROZEN_RUN / "calibration", FROZEN_RUN / "eval")
        for path in directory.rglob("*complete*.json")
        if path.is_file()
    )
    return {
        "policy": (
            "Pre-DET1.4 G1, calibration, thresholds, evaluation rows, and "
            "stage markers are append-only history and cannot satisfy any "
            "fork-substrate stage."
        ),
        "anchors": anchors,
        "rejected_stage_markers": [
            {"path": path, "disposition": HISTORICAL_DISPOSITION}
            for path in old_markers
        ],
    }


def _command_binding(command: str, args: argparse.Namespace) -> dict[str, Any]:
    script = Path(__file__).resolve()
    normalized = [sys.executable, str(script), command, "--run-dir", str(FROZEN_RUN)]
    if getattr(args, "require_amendment", False):
        normalized.append("--require-amendment")
    if getattr(args, "write_receipt", False):
        normalized.append("--write-receipt")
    return {
        "argv_exact": list(sys.argv),
        "shell_command_exact": shlex.join(sys.argv),
        "normalized_command": shlex.join(normalized),
        "executable": sys.executable,
        "script": file_record(script),
    }


def inventory(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    _require(run_dir == FROZEN_RUN.resolve(),
             f"DET1.4 is frozen to {FROZEN_RUN}; got {run_dir}")
    records = {path: _frozen_record(path) for path in FROZEN_RECORDS if path != TENSOR_CUDA_EXTENSION}
    order = _validate_order(records[ORDER])
    registration = _validate_registration(records[REGISTRATION])
    runtime = _validate_runtime(records[RUNTIME_FRAME], records[REGISTRATION])
    run_manifest = read_json(RUN_MANIFEST)
    _require(run_manifest.get("schema") == "grm.det1.run_manifest.v1",
             "wrong frozen run-manifest schema")
    _record_equal(run_manifest.get("registration") or {}, records[REGISTRATION],
                  "run_manifest.registration")
    prior = _validate_prior_amendment(records[PRIOR_AMENDMENT])
    amendment = _validate_det1_4_amendment(
        required=bool(args.require_amendment),
        bindings={
            "order": records[ORDER],
            "registration": records[REGISTRATION],
            "runtime_frame": records[RUNTIME_FRAME],
            "prior_amendment": records[PRIOR_AMENDMENT],
        },
        source_rebindings=getattr(args, "det1_6_source_rebindings", None),
    )
    lived = _validate_lived_capture(
        records[CAPTURE_RECEIPT],
        records[LIVED_MANIFEST],
        records[RUNTIME_FRAME],
        records[PRIOR_AMENDMENT],
    )
    comparison = _validate_old_compare(records[OLD_COMPARE])
    return {
        "schema": "grm.det1_4.inventory.v1",
        "status": "PASS_FROZEN_INVENTORY_ONLY",
        "created_utc": utc_now(),
        "run_dir": str(run_dir),
        "command": _command_binding("inventory", args),
        "order": order,
        "run_manifest": {"record": records[RUN_MANIFEST], "status": run_manifest.get("status")},
        "registration": registration,
        "runtime_frame": runtime,
        "prior_det1_3_amendment": prior,
        "det1_4_amendment": amendment,
        "lived_capture": lived,
        "old_reconstructed_comparison": comparison,
        "historical_evidence": _historical_evidence(),
        "pending_stage_order": [
            {"ordinal": ordinal, "stage": stage, "status": "PENDING"}
            for ordinal, stage in enumerate(STAGE_ORDER)
        ],
        "claims": {
            "fork_gate_implemented": True,
            "zero_gate_passed": False,
            "race_attempted": False,
            "race_resume_authorized": False,
            "scope": "CPU_ONLY_FROZEN_EVIDENCE_INVENTORY",
        },
    }


def _device_node(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "required": True}
    try:
        info = path.stat()
    except OSError as exc:
        result.update({
            "exists": False,
            "character_device": False,
            "readable": False,
            "writable": False,
            "ready": False,
            "error": f"{type(exc).__name__}: {exc}",
        })
        return result
    result.update({
        "exists": True,
        "character_device": stat.S_ISCHR(info.st_mode),
        "major": os.major(info.st_rdev),
        "minor": os.minor(info.st_rdev),
        "readable": os.access(path, os.R_OK),
        "writable": os.access(path, os.W_OK),
    })
    result["ready"] = bool(
        result["character_device"] and result["readable"] and result["writable"]
    )
    return result


def _tensor_cuda_probe(nodes_ready: bool) -> dict[str, Any]:
    extension = _frozen_record(TENSOR_CUDA_EXTENSION)
    result: dict[str, Any] = {
        "package_root": str(TENSOR_CUDA_ROOT),
        "extension": extension,
        "import_ok": False,
        "runtime_probe": "NOT_ATTEMPTED_DEVICE_NAMESPACE_INCOMPLETE",
        "cuda_available": False,
        "gpu_allocations_attempted": 0,
    }
    try:
        root_text = str(TENSOR_CUDA_ROOT)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        module = importlib.import_module("tensor_cuda")
        result["import_ok"] = True
        result["module_path"] = str(Path(module.__file__).resolve())
        if nodes_ready:
            # cudaDeviceSynchronize is a context/runtime reachability check.  It
            # creates no Tensor and requests no device allocation.
            module.synchronize()
            result["runtime_probe"] = "cudaDeviceSynchronize_PASS_NO_ALLOCATION"
            result["cuda_available"] = True
    except Exception as exc:  # the exact loader/runtime failure belongs in evidence
        result["runtime_probe"] = "FAIL"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def gpu_preflight(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    frozen = inventory(args)
    nodes = [_device_node(path) for path in REQUIRED_DEVICE_NODES]
    nodes_ready = all(item["ready"] for item in nodes)
    tensor_cuda = _tensor_cuda_probe(nodes_ready)
    ready = bool(nodes_ready and tensor_cuda["cuda_available"])
    result: dict[str, Any] = {
        "schema": "grm.det1_4.gpu_preflight.v1",
        "status": "READY_NO_RACE_ATTEMPTED" if ready else "BLOCKED_NO_RACE_ATTEMPTED",
        "created_utc": utc_now(),
        "run_dir": str(FROZEN_RUN),
        "command": _command_binding("gpu-preflight", args),
        "inventory_binding": {
            "schema": frozen["schema"],
            "status": frozen["status"],
            "order": frozen["order"]["record"],
            "registration": frozen["registration"]["record"],
            "runtime_frame": frozen["runtime_frame"]["record"],
            "prior_det1_3_amendment": frozen["prior_det1_3_amendment"]["record"],
            "det1_4_amendment": frozen["det1_4_amendment"],
            "lived_capture_receipt": frozen["lived_capture"]["capture_receipt"],
            "lived_snapshot_manifest": frozen["lived_capture"]["snapshot_manifest"],
            "old_compare": frozen["old_reconstructed_comparison"]["record"],
        },
        "device_nodes": nodes,
        "device_namespace_ready": nodes_ready,
        "tensor_cuda": tensor_cuda,
        "pending_stage_order": frozen["pending_stage_order"],
        "claims": {
            "gpu_allocations_attempted": 0,
            "model_loaded": False,
            "fork_gate_implemented": True,
            "zero_gate_passed": False,
            "race_attempted": False,
            "race_resume_authorized": False,
            "scope": "CPU_ONLY_DEVICE_NAMESPACE_AND_RUNTIME_PREFLIGHT",
        },
    }
    if args.write_receipt:
        receipt = write_content_addressed(
            FROZEN_RUN / "det1_4" / "preflight",
            "gpu_preflight",
            result,
        )
        result["receipt_file"] = file_record(receipt)
    return result, 0 if ready else 2


def _current_process() -> dict[str, Any]:
    from scripts.grm_det1_3_snapshot import process_instance_sha256

    pid = os.getpid()
    stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    close = stat_text.rfind(")")
    _require(close >= 0, "cannot parse current process start time")
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


def _snapshot_utf8(snapshot: Mapping[str, Any], field: str) -> str:
    from scripts.grm_det1_3_snapshot import load_snapshot_array

    return load_snapshot_array(snapshot, field).tobytes(order="C").decode("utf-8")


def _continue_forked_prefill(arena: Any, prompt_ids: list[int], ngen: int) -> tuple[str, dict[str, Any]]:
    """Continue exactly at the captured prefill boundary without routing."""
    from core import kv_graft
    from scripts.grm_det1_3_snapshot import verify_fork_masks_consumed

    stops = tuple(arena.stop_sequences or ())
    row = arena._forward(prompt_ids)
    mask_receipt = verify_fork_masks_consumed(arena)
    kv_graft.clear_injection(arena.m)
    out = [int(row.argmax())]
    stopped = False
    for _ in range(int(ngen) - 1):
        if any(stop in arena.decode(out) for stop in stops):
            stopped = True
            break
        row = arena._forward([out[-1]])
        out.append(int(row.argmax()))
    if not stopped and not any(stop in arena.decode(out) for stop in stops):
        arena._forward([out[-1]])
    answer = arena.decode(out)
    for stop in stops:
        if stop in answer:
            answer = answer.split(stop)[0]
    return answer.strip(), mask_receipt


def _run_zero_gate(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    from scripts.grm_det1_2_gpu import _load_model_repo
    from scripts.grm_det1_3_gpu import (
        _fixture_answer_correct,
        _frame_identity,
        _install_lived_nodes,
        _is_refusal,
    )
    from scripts.grm_det1_3_snapshot import (
        ARM_PROTOCOLS,
        capture_arena_snapshot,
        compare_fork_substrate,
        finalize_snapshot_with_answer,
        load_snapshot,
        restore_prefill_fork,
    )
    from scripts.grm_det1_gpu import (
        NATIVE_LIB,
        _runtime_env,
        _validate_visibility,
    )

    runtime = read_json(RUNTIME_FRAME)
    registration = read_json(REGISTRATION)
    _require(runtime.get("native_library") == file_record(NATIVE_LIB),
             "DET1.4 native runtime differs from the frozen frame")
    os.environ.update(_runtime_env(runtime))
    visible = _validate_visibility()
    _require(str(visible) == str(runtime["cuda_visible_devices"]),
             "DET1.4 visible GPU differs from the frozen frame")
    frozen_lease = int(runtime["gpu_lease"]["actual_cap_seconds"])
    _require(int(args.lease_seconds) == frozen_lease,
             f"zero-gate lease must equal frozen {frozen_lease}s")

    attempt_root = FROZEN_RUN / "det1_4" / "zero_gate"
    attempt_dir = attempt_root / f"attempt_{len(list(attempt_root.glob('attempt_*'))) + 1:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    fixture = read_json(FIXTURE)
    source = load_snapshot(LIVED_MANIFEST)
    source_provenance = source.get("provenance") or {}
    repo = model = tokenizer = e2e = None
    started = time.monotonic()
    try:
        e2e, model, tokenizer, repo, model_info = _load_model_repo(attempt_dir, runtime)
        _install_lived_nodes(repo, e2e, fixture)
        arena = repo.arena
        identity = _frame_identity(model, tokenizer, e2e, registration, runtime)
        restore = restore_prefill_fork(
            arena,
            LIVED_MANIFEST,
            target_identity=identity,
            expected_frame_sha256=source["identity"]["frame_sha256"],
            # The frozen Harbor capture predates this process.  This isolated
            # answer gate installs no detectors and makes no full-index claim.
            require_same_process_index=False,
        )
        _require(restore["intervention"] == "ZERO", "zero gate mutated the fork")
        _require(restore["zero_intervention_no_deltas"] is True,
                 "zero gate reports a non-empty delta set")

        process = _current_process()
        provenance = {
            **source_provenance,
            **process,
            "schema": "grm.det1_4.fork_capture_provenance.v1",
            "arm": "fork",
            "protocol": ARM_PROTOCOLS["fork"],
            "protocol_source_sha256": file_record(Path(__file__).resolve())["sha256"],
            "order_sha256": file_record(ORDER)["sha256"],
            "source_amendment_sha256": file_record(DET1_4_AMENDMENT)["sha256"],
            "registration_sha256": file_record(REGISTRATION)["sha256"],
            "runtime_frame_sha256": file_record(RUNTIME_FRAME)["sha256"],
            "capture_boundary": "before_probe_prefill",
            "selection_boundary": "before_generation_and_grounding_selection",
        }
        question = _snapshot_utf8(source, "text.question_utf8")
        sink_text = _snapshot_utf8(source, "text.sink_utf8")
        fork_manifest = capture_arena_snapshot(
            arena,
            attempt_dir / "fork_snapshot",
            label="fork",
            provenance=provenance,
            question=question,
            prompt_ids=restore["prompt_ids"],
            admission_plan=restore["admission_state"],
            live_token_ids=restore["live_token_ids"],
            sink_text=sink_text,
            sink_token_ids=restore["sink_token_ids"],
            explicit_identity=identity,
        )
        answer, masks = _continue_forked_prefill(
            arena, restore["prompt_ids"], int(runtime["resolved_flags"]["ngen"]))
        correct = _fixture_answer_correct(answer, fixture)
        refusal = _is_refusal(answer)
        expected_answer = str((source.get("linked_answer") or {}).get("probe_answer", ""))
        normalized_match = normalize_value_text(answer) == normalize_value_text(
            expected_answer)
        linked = {
            "schema": "grm.det1_4.zero_fork_linked_answer.v1",
            "process_instance_sha256": process["process_instance_sha256"],
            "captured_attempt_ordinal": int(provenance["attempt_ordinal"]),
            "attempt_answer": answer,
            "attempt_mounts": list(arena.cur_mounts),
            "attempt_answer_correct": bool(correct),
            "attempt_refusal": bool(refusal),
            "probe_answer": answer,
            "probe_selected_attempt": int(provenance["attempt_ordinal"]),
            "probe_mounts": list(arena.cur_mounts),
            "probe_answer_correct": bool(correct),
            "probe_refusal": bool(refusal),
        }
        fork_manifest = finalize_snapshot_with_answer(fork_manifest, linked)
        substrate = compare_fork_substrate(LIVED_MANIFEST, fork_manifest)
        gate_pass = bool(
            substrate["gate_pass"]
            and masks["gate_pass"]
            and correct
            and not refusal
            and normalized_match
        )
        receipt = {
            "schema": "grm.det1_4.zero_fork_gate.v1",
            "status": "PASS" if gate_pass else "FAIL_CLOSED",
            "created_utc": utc_now(),
            "gate_pass": gate_pass,
            "intervention": "ZERO",
            "elapsed_seconds": time.monotonic() - started,
            "order": file_record(ORDER),
            "source_amendment": file_record(DET1_4_AMENDMENT),
            "registration": file_record(REGISTRATION),
            "runtime_frame": file_record(RUNTIME_FRAME),
            "fixture": file_record(FIXTURE),
            "lived_snapshot": file_record(LIVED_MANIFEST),
            "fork_snapshot": file_record(fork_manifest),
            "restore": restore,
            "substrate_comparison": substrate,
            "mask_consumption": masks,
            "answer": {
                "raw": answer,
                "lived_raw": expected_answer,
                "raw_exact_match": answer == expected_answer,
                "normalized": normalize_value_text(answer),
                "lived_normalized": normalize_value_text(expected_answer),
                "normalized_match": normalized_match,
                "fixture_correct": bool(correct),
                "refusal": bool(refusal),
                "normalization_rule": (
                    "strip Markdown emphasis and normalize U+2010/U+2011 to ASCII hyphen"
                ),
            },
            "model_info": model_info,
            "claims": {
                "canonical_value_bytes_equal": bool(substrate["gate_pass"]),
                "literal_device_storage_bytes_equal": False,
                "detector_index_verified": False,
                "race_resume_authorized": False,
            },
        }
        receipt_path = write_content_addressed(attempt_dir, "zero_fork_gate", receipt)
        if gate_pass:
            write_json_exclusive(attempt_root / "stage_complete.json", {
                "schema": "grm.det1_4.stage_marker.v1",
                "stage": "zero_gate",
                "status": "COMPLETE",
                "receipt": file_record(receipt_path),
            })
        return {**receipt, "receipt_file": file_record(receipt_path)}, 0 if gate_pass else 2
    finally:
        if repo is not None:
            try:
                repo.close()
            except BaseException:
                pass
        model = tokenizer = repo = e2e = None
        gc.collect()
        try:
            import tensor_cuda as tc
            if hasattr(tc, "empty_cache"):
                tc.empty_cache()
        except BaseException:
            pass


def zero_gate(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    required = argparse.Namespace(**vars(args))
    required.require_amendment = True
    required.write_receipt = False
    preflight, code = gpu_preflight(required)
    if code != 0:
        return preflight, code
    from scripts.grm_det1_gpu import gpu_lease

    with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
        return _run_zero_gate(args)


def selftest(args: argparse.Namespace) -> dict[str, Any]:
    value = inventory(args)
    _require([item["stage"] for item in value["pending_stage_order"]] == list(STAGE_ORDER),
             "stage order selftest failed")
    _require(all(item["status"] == "PENDING" for item in value["pending_stage_order"]),
             "pending-state selftest failed")
    _require(normalize_value_text("**Nacre\u20106\u2011Blue**") == "nacre-6-blue",
             "value normalization selftest failed")
    _require(canonical_json_bytes({"b": 1, "a": 2}).startswith(b"{\n"),
             "canonical JSON helper selftest failed")
    return {
        "schema": "grm.det1_4.selftest.v1",
        "status": "PASS_CPU_ONLY",
        "created_utc": utc_now(),
        "inventory_status": value["status"],
        "pending_stage_order": value["pending_stage_order"],
        "gpu_allocations_attempted": 0,
        "fork_gate_implemented": True,
        "race_attempted": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("inventory", "gpu-preflight", "selftest", "zero-gate"):
        child = subparsers.add_parser(name)
        child.add_argument("--run-dir", type=Path, default=FROZEN_RUN)
        child.add_argument(
            "--require-amendment",
            action="store_true",
            help="fail unless the separately chained DET1.4 source amendment is present",
        )
        if name == "gpu-preflight":
            child.add_argument(
                "--write-receipt",
                action="store_true",
                help="write an append-only content-addressed preflight receipt",
            )
        if name == "zero-gate":
            child.add_argument("--lease-seconds", type=int, default=580)
            child.add_argument("--lock-wait-seconds", type=int, default=7200)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        if args.command == "inventory":
            output = inventory(args)
            code = 0
        elif args.command == "gpu-preflight":
            output, code = gpu_preflight(args)
        elif args.command == "zero-gate":
            output, code = zero_gate(args)
        else:
            output = selftest(args)
            code = 0
    except Exception as exc:
        output = {
            "schema": "grm.det1_4.cli_error.v1",
            "status": "FAIL_CLOSED",
            "created_utc": utc_now(),
            "command": args.command,
            "argv_exact": list(sys.argv),
            "error": f"{type(exc).__name__}: {exc}",
            "fork_gate_implemented": True,
            "race_attempted": False,
        }
        code = 2
    sys.stdout.buffer.write(canonical_json_bytes(output))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
