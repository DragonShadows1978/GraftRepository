#!/usr/bin/env python3
"""Fresh-process Harbor hook bisection for ORDER GRM-DET1.2.

The incident is sensitive to CUDA process history.  Consequently every arm
loads a fresh GPT-OSS model, creates a fresh repository, serves exactly one
Harbor turn, writes an immutable receipt, and exits.  The matrix includes the
DET1 route-profile prepass and an empty observer shell as nuisance controls;
without them a hook could be blamed for a forward performed before any hook is
installed.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")

from scripts.grm_det1_common import (  # noqa: E402
    DETError,
    MECHANISTIC,
    VERBAL_QUESTION,
    contains_value,
    file_record,
    read_json,
    route_fixture_profile,
    utc_now,
    write_content_addressed,
    write_json_exclusive,
)
from scripts.grm_det1_baseline_registry import (  # noqa: E402
    compare_served_to_live_registry,
)
from scripts.grm_det1_gpu import (  # noqa: E402
    DEFAULT_LEASE_SECONDS,
    GAP_SECONDS,
    MODEL_DIR,
    NATIVE_LIB,
    _base_env,
    _gpu_snapshot,
    _install_fixture_nodes,
    _live_baseline_anchor,
    _one,
    _registration,
    _receipt_context,
    _run_process,
    _runtime_env,
    _validate_file_record,
    _validate_visibility,
    gpu_lease,
)
from scripts.grm_det1_e2e import (  # noqa: E402
    _mechanistic_counterfactual,
    _restore_counterfactual,
    _snapshot_counterfactual,
    _verbal_counterfactual,
)


ORDER = ROOT / "orders" / "GRM_DET1_2_OBSERVER_LEAK.md"
DET1_1_AMENDMENT = "det1_1_source_amendment.json"
DET1_2_AMENDMENT = "det1_2_source_amendment.json"
DET1_2_REPLAY_AMENDMENT = "det1_2_replay_gate_source_amendment.json"
DET1_2_CHANGED = frozenset((
    "scripts/grm_det1_common.py",
    "scripts/grm_det1_e2e.py",
))
DET1_2_ADDED = frozenset(("scripts/grm_det1_2_gpu.py",))
DET1_2_REPLAY_CHANGED = frozenset(("scripts/grm_det1_2_gpu.py",))
FIXTURE_SOURCE = (
    ROOT / "tests" / "fixtures" / "supersession_battery"
    / "correction_then_restatement.json"
)
FIXTURE_ID = "sup_harbor_restatement"
INCIDENT_GLOB = "det1_1_guard_failure_fefc72a057a8dcb4.json"
BASE_ARMS: dict[str, dict[str, Any]] = {
    "control_a": {"profile": False, "hooks": (), "shell": False, "verbal": False},
    "control_b": {"profile": False, "hooks": (), "shell": False, "verbal": False},
    "profile_only": {"profile": True, "hooks": (), "shell": False, "verbal": False},
    "observer_shell": {"profile": False, "hooks": (), "shell": True, "verbal": False},
    "profile_observer_shell": {
        "profile": True, "hooks": (), "shell": True, "verbal": False,
    },
    "d_lqr": {"profile": False, "hooks": ("D-LQR",), "shell": False, "verbal": False},
    "d_ngh": {"profile": False, "hooks": ("D-NGH",), "shell": False, "verbal": False},
    "d_ent": {"profile": False, "hooks": ("D-ENT",), "shell": False, "verbal": False},
    # In the incident path the guard adjudicates the served result before
    # this separate pass begins.  The arm preserves that causal order.
    "d_verb": {"profile": False, "hooks": (), "shell": False, "verbal": True},
    "all_mechanistic_no_profile": {
        "profile": False,
        "hooks": tuple(MECHANISTIC),
        "shell": False,
        "verbal": False,
    },
    "all_legacy": {
        "profile": True,
        "hooks": tuple(MECHANISTIC),
        "shell": False,
        "verbal": True,
    },
}
HOOK_TO_ARM = {
    "D-LQR": "d_lqr",
    "D-NGH": "d_ngh",
    "D-ENT": "d_ent",
    "D-VERB": "d_verb",
}


def _pair_name(left: str, right: str) -> str:
    return "pair_" + left.removeprefix("D-").lower() + "_" + right.removeprefix("D-").lower()


def _pair_arm(left: str, right: str) -> dict[str, Any]:
    hooks = tuple(value for value in MECHANISTIC if value in (left, right))
    return {
        "profile": False,
        "hooks": hooks,
        "shell": False,
        "verbal": "D-VERB" in (left, right),
    }


PAIR_ARMS = {
    _pair_name(left, right): _pair_arm(left, right)
    for left, right in itertools.combinations(MECHANISTIC, 2)
}
PROFILE_HOOK_ARMS = {
    "profile_" + arm: {**BASE_ARMS[arm], "profile": True}
    for arm in HOOK_TO_ARM.values()
}
PROFILE_PAIR_ARMS = {
    "profile_" + arm: {**spec, "profile": True}
    for arm, spec in PAIR_ARMS.items()
}
ALL_ARMS = {
    **BASE_ARMS,
    **PROFILE_HOOK_ARMS,
    **PAIR_ARMS,
    **PROFILE_PAIR_ARMS,
}
INITIAL_ARMS = (
    "control_a",
    "control_b",
    "profile_only",
    "observer_shell",
    "profile_observer_shell",
    "d_lqr",
    "d_ngh",
    "d_ent",
    "d_verb",
    "profile_d_lqr",
    "profile_d_ngh",
    "profile_d_ent",
    "profile_d_verb",
    "all_mechanistic_no_profile",
    "all_legacy",
)
PAIR_STAGE_ARMS = (*PAIR_ARMS, *PROFILE_PAIR_ARMS)


def _source_line(relative: str, needle: str) -> dict[str, Any]:
    path = ROOT / relative
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if needle in line:
            return {"path": relative, "line": number, "needle": needle}
    raise DETError(f"DET1.2 source locator disappeared: {relative}: {needle}")


def _mechanism_inventory() -> dict[str, Any]:
    return {
        "profile_prepass": {
            "sites": [
                _source_line(
                    "scripts/grm_det1_common.py",
                    "probe_key = arena._probe_key(question)",
                ),
                _source_line(
                    "core/grm_admission.py",
                    "probe_key = arena._probe_key(question)",
                ),
            ],
            "extra_query_forwards": 2,
            "mechanism": (
                "two extra full model query forwards before the served turn; "
                "no arena position or KV commit, but CUDA kernel/allocator "
                "warmth is the candidate perturbation channel"
            ),
        },
        "D-LQR": {
            "arm_site": _source_line(
                "scripts/grm_det1_common.py",
                "observer._route_attn._capture_q = True",
            ),
            "capture_site": _source_line(
                "core/gpt_oss20b_tc.py", "self._captured_q = q.numpy()"),
            "mechanism": (
                "pre-RoPE BF16 query capture; q.numpy performs a device cast "
                "and synchronous D2H copy inside the production forward; its "
                "candidate channel is allocation/synchronization history"
            ),
        },
        "D-NGH": {
            "tap_site": _source_line(
                "scripts/grm_det1_common.py",
                "observer._mass_rows.append(observer._full_mass(",
            ),
            "kernel_site": _source_line(
                "scripts/grm_det1_common.py",
                "scores = gpt.tc.matmul(q_last, key_h",
            ),
            "mechanism": (
                "second L=1 query-key GEMM plus mask add, sink concat, "
                "softmax, fp32 cast, and D2H copy inline after real attention; "
                "fresh tensors leave positions, masks, KV bytes, dtype, and RNG "
                "unchanged; CUDA shape warmth and allocator ordering are the "
                "candidate perturbation mechanism pending the singleton receipt"
            ),
        },
        "D-ENT": {
            "site": _source_line(
                "scripts/grm_det1_common.py",
                "entropy, margin = _entropy_and_margin(logits)",
            ),
            "mechanism": "CPU-only computation on the already-returned host logit row",
        },
        "D-VERB": {
            "site": _source_line(
                "scripts/grm_det1_e2e.py", "def _verbal_counterfactual("),
            "mechanism": (
                "separate extra generation pass after served guard adjudication; "
                "temporally incapable of causing the recorded first served refusal"
            ),
        },
    }


def _incident(run_dir: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    paths = sorted(Path(run_dir).rglob(INCIDENT_GLOB))
    if len(paths) != 1:
        raise DETError(
            f"expected one lead-verified incident {INCIDENT_GLOB}, found {len(paths)}")
    path = paths[0]
    value = read_json(path)
    comparison = dict(value.get("served_baseline_comparison") or {})
    observed = dict(comparison.get("observed") or {})
    projection = {
        "answer": str(observed.get("answer", "")),
        "mounted_ids": [int(item) for item in observed.get("mounted_ids", ())],
    }
    if value.get("status") != "STOPPED" or value.get("fixture_id") != FIXTURE_ID:
        raise DETError(f"lead incident binding drifted: {path}")
    return path, value, projection


def _registered_fixture(registration: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        dict(row) for row in registration["fixtures"]
        if row.get("fixture_id") == FIXTURE_ID
    ]
    if len(rows) != 1:
        raise DETError(f"registration lacks exactly one {FIXTURE_ID} fixture")
    return rows[0]


def _fixture_aliases_without_forward(
    fixture: Mapping[str, Any],
    node_to_idx: Mapping[str, int],
    expected_values: Sequence[str],
) -> set[int]:
    """Resolve Harbor's logical revision unit without a model forward."""

    graph = {str(node["node_id"]): set() for node in fixture["nodes"]}
    for node in fixture["nodes"]:
        child = str(node["node_id"])
        for parent_value in node.get("supersedes", ()):
            parent = str(parent_value)
            graph[child].add(parent)
            graph[parent].add(child)
    seeds = {
        str(node["node_id"])
        for node in fixture["nodes"]
        if any(
            contains_value(str(node.get("text", "")), str(value))
            for value in expected_values
        )
    }
    if not seeds:
        raise DETError("Harbor fixture has no expected-value alias seed")
    found = set(seeds)
    pending = list(seeds)
    while pending:
        node = pending.pop()
        for linked in graph[node]:
            if linked not in found:
                found.add(linked)
                pending.append(linked)
    aliases = {int(node_to_idx[node]) for node in found}
    if not aliases:
        raise DETError("Harbor static logical alias unit is empty")
    return aliases


def _current_record(shown: str) -> dict[str, Any]:
    path = ROOT / shown
    if not path.is_file():
        raise DETError(f"DET1.2 source disappeared: {shown}")
    record = file_record(path)
    return {"bytes": record["bytes"], "sha256": record["sha256"]}


def _check_det1_2_inventory(
    run_dir: Path,
    registration: Mapping[str, Any],
    registration_path: Path,
) -> Path:
    """Validate the append-only DET1.1 -> DET1.2 -> replay-gate chain."""

    prior_path = Path(run_dir) / DET1_1_AMENDMENT
    base_path = Path(run_dir) / DET1_2_AMENDMENT
    current_path = Path(run_dir) / DET1_2_REPLAY_AMENDMENT
    prior = read_json(prior_path)
    base = read_json(base_path)
    current = read_json(current_path)
    if prior.get("schema") != "grm.det1.source_amendment.v1":
        raise DETError("DET1.1 source amendment schema drifted")
    if base.get("schema") != "grm.det1_2.source_amendment.v1":
        raise DETError("DET1.2 source amendment schema drifted")
    if base.get("status") != "AUTHORIZED_DIAGNOSTIC_AMENDMENT":
        raise DETError("DET1.2 source amendment is not authorized")
    if base.get("order") != file_record(ORDER):
        raise DETError("DET1.2 source amendment is not bound to the live order")
    if base.get("registration") != file_record(registration_path):
        raise DETError("DET1.2 source amendment registration binding drifted")
    if base.get("prior_amendment") != file_record(prior_path):
        raise DETError("DET1.2 source amendment does not chain from DET1.1")
    if current.get("schema") != "grm.det1_2.replay_gate_source_amendment.v1":
        raise DETError("DET1.2 replay-gate amendment schema drifted")
    if current.get("status") != "AUTHORIZED_DIAGNOSTIC_REVIEW_AMENDMENT":
        raise DETError("DET1.2 replay-gate amendment is not authorized")
    if current.get("order") != file_record(ORDER):
        raise DETError("DET1.2 replay-gate amendment order binding drifted")
    if current.get("registration") != file_record(registration_path):
        raise DETError("DET1.2 replay-gate registration binding drifted")
    if current.get("prior_amendment") != file_record(base_path):
        raise DETError("DET1.2 replay-gate amendment chain drifted")
    if prior.get("registration") != file_record(registration_path):
        raise DETError("DET1.1 source amendment registration binding drifted")
    prior_frozen = prior.get("frozen_artifacts_unchanged") or []
    if not prior_frozen:
        raise DETError("DET1.1 source amendment lost its frozen bindings")
    for index, record in enumerate(prior_frozen):
        # DET1.1 froze grm_det1_common.py before this order amended it.  Its
        # old record remains valid historical evidence, but cannot match the
        # current source by design.  All artifact records remain immutable.
        if record.get("path") == "scripts/grm_det1_common.py":
            continue
        _validate_file_record(record, f"DET1.1 frozen artifact {index}")

    allowed = base.get("allowed_source_changes") or {}
    if not isinstance(allowed, dict) or set(allowed) != DET1_2_CHANGED:
        raise DETError("DET1.2 changed-source set drifted")
    registered = registration["production_source_inventory"]
    prior_allowed = prior.get("allowed_source_changes") or {}
    expected_before = {
        "scripts/grm_det1_common.py": registered["scripts/grm_det1_common.py"],
        "scripts/grm_det1_e2e.py": prior_allowed["scripts/grm_det1_e2e.py"]["after"],
    }
    for shown, change in allowed.items():
        if change.get("before") != expected_before[shown]:
            raise DETError(f"DET1.2 before-record drifted: {shown}")
        observed = _current_record(shown)
        if change.get("after") != observed:
            raise DETError(
                f"DET1.2 changed source drifted: {shown}: "
                f"authorized={change.get('after')} observed={observed}")

    review_allowed = current.get("allowed_source_changes") or {}
    if (
        not isinstance(review_allowed, dict)
        or set(review_allowed) != DET1_2_REPLAY_CHANGED
    ):
        raise DETError("DET1.2 replay-gate changed-source set drifted")
    base_added = base.get("added_sources") or {}
    for shown, change in review_allowed.items():
        if change.get("before") != base_added.get(shown):
            raise DETError(f"DET1.2 replay-gate before-record drifted: {shown}")
        observed = _current_record(shown)
        if change.get("after") != observed:
            raise DETError(
                f"DET1.2 replay-gate source drifted: {shown}: "
                f"authorized={change.get('after')} observed={observed}")

    # Validate every source frozen by the original registration, applying
    # the append-only amendment overlays in order.  The harness imports
    # grm_det1_gpu, so checking only the two newly edited files would leave a
    # large unsealed execution surface.
    for shown, registered_record in registered.items():
        expected = registered_record
        if shown in prior_allowed:
            expected = prior_allowed[shown]["after"]
        if shown in allowed:
            expected = allowed[shown]["after"]
        observed = _current_record(shown)
        if observed != expected:
            raise DETError(
                f"DET1.2 registered source drifted: {shown}: "
                f"expected={expected} observed={observed}")

    prior_added = prior.get("added_sources") or {}
    if not isinstance(prior_added, dict):
        raise DETError("DET1.1 added-source inventory drifted")
    for shown, expected in prior_added.items():
        observed = _current_record(shown)
        if observed != expected:
            raise DETError(
                f"DET1.1 added source drifted under DET1.2: {shown}: "
                f"expected={expected} observed={observed}")

    added = base_added
    if not isinstance(added, dict) or set(added) != DET1_2_ADDED:
        raise DETError("DET1.2 added-source set drifted")
    for shown, expected in added.items():
        if shown in review_allowed:
            expected = review_allowed[shown]["after"]
        observed = _current_record(shown)
        if expected != observed:
            raise DETError(
                f"DET1.2 added source drifted: {shown}: "
                f"authorized={expected} observed={observed}")
    frozen = base.get("frozen_artifacts_unchanged") or []
    if not isinstance(frozen, list) or len(frozen) < 4:
        raise DETError("DET1.2 amendment lacks frozen incident bindings")
    for index, record in enumerate(frozen):
        _validate_file_record(record, f"DET1.2 frozen artifact {index}")
    review_frozen = current.get("frozen_artifacts_unchanged") or []
    if not isinstance(review_frozen, list) or len(review_frozen) < 3:
        raise DETError("DET1.2 replay-gate amendment lacks frozen bindings")
    for index, record in enumerate(review_frozen):
        _validate_file_record(record, f"DET1.2 replay-gate frozen artifact {index}")
    for shown, expected in registration.get("model_snapshot_inventory", {}).items():
        path = Path(shown)
        observed = {
            "bytes": path.stat().st_size if path.is_file() else None,
            "resolved_blob": path.resolve().name if path.is_file() else None,
            "verification": "huggingface_content_addressed_blob_identity",
        }
        if observed != expected:
            raise DETError(
                f"DET1.2 model snapshot drifted: {shown}: "
                f"expected={expected} observed={observed}")
    return current_path


def _arm_marker(run_dir: Path, arm: str) -> Path:
    return Path(run_dir) / "det1_2" / "bisect" / "arms" / arm / "complete.json"


def _load_arm(run_dir: Path, arm: str) -> tuple[Path, dict[str, Any]]:
    marker_path = _arm_marker(run_dir, arm)
    marker = read_json(marker_path)
    if marker.get("status") != "COMPLETE" or marker.get("arm") != arm:
        raise DETError(f"invalid DET1.2 arm marker: {marker_path}")
    receipt_path = _validate_file_record(marker["receipt"], f"DET1.2 arm {arm}")
    receipt = read_json(receipt_path)
    registration_path = _registration(run_dir)
    runtime_frame_path = _one(run_dir, "runtime_frame_*.json")
    runtime_frame = read_json(runtime_frame_path)
    amendment_path = Path(run_dir) / DET1_2_REPLAY_AMENDMENT
    incident_path, _incident_value, _incident_projection = _incident(run_dir)
    spec = ALL_ARMS[arm]
    if not (
        receipt.get("schema") == "grm.det1_2.hook_arm.v1"
        and receipt.get("status") == "COMPLETE"
        and receipt.get("arm") == arm
        and receipt.get("order") == file_record(ORDER)
        and receipt.get("source_amendment") == file_record(amendment_path)
        and receipt.get("registration") == file_record(registration_path)
        and receipt.get("runtime_frame") == file_record(runtime_frame_path)
        and receipt.get("incident") == file_record(incident_path)
        and receipt.get("fixture_source") == file_record(FIXTURE_SOURCE)
        and receipt.get("profile_prepass") is bool(spec["profile"])
        and receipt.get("observer_shell") is bool(spec["shell"])
        and receipt.get("active_mechanistic_hooks") == list(spec["hooks"])
        and receipt.get("d_verb_requested") is bool(spec["verbal"])
        and receipt.get("baseline_anchor") == _live_baseline_anchor(runtime_frame)
    ):
        raise DETError(f"invalid DET1.2 arm receipt: {receipt_path}")
    projection_path = _validate_file_record(
        receipt["projection"], f"DET1.2 arm {arm} projection")
    _validate_file_record(receipt["signals"], f"DET1.2 arm {arm} signals")
    projection = read_json(projection_path)
    fixture = _registered_fixture(read_json(registration_path))
    fresh_comparison = compare_served_to_live_registry(fixture, projection)
    stored_comparison = receipt.get("live_baseline_comparison") or {}
    if not (
        receipt.get("live_match") is fresh_comparison["live_match"]
        and stored_comparison.get("live_match") is fresh_comparison["live_match"]
        and stored_comparison.get("anchor") == fresh_comparison["anchor"]
        and stored_comparison.get("observed") == fresh_comparison["observed"]
        and receipt.get("baseline_anchor") == fresh_comparison["anchor"]
    ):
        raise DETError(f"DET1.2 arm {arm} live comparison binding drifted")
    verbal_record = receipt.get("verbal")
    requested = bool(spec["verbal"])
    should_execute = bool(requested and fresh_comparison["live_match"])
    if not (
        receipt.get("d_verb_requested") is requested
        and receipt.get("d_verb_executed_after_live_guard") is should_execute
        and receipt.get("d_verb_skipped_by_live_guard")
            is bool(requested and not should_execute)
    ):
        raise DETError(f"DET1.2 arm {arm} D-VERB causal-order binding drifted")
    if should_execute:
        if not verbal_record:
            raise DETError(f"DET1.2 arm {arm} executed D-VERB without evidence")
        _validate_file_record(verbal_record, f"DET1.2 arm {arm} verbal")
    elif verbal_record is not None:
        raise DETError(f"DET1.2 arm {arm} has unexpected D-VERB evidence")
    if not isinstance(receipt.get("model_info"), dict):
        raise DETError(f"DET1.2 arm {arm} lacks model identity")
    return receipt_path, receipt


def _projection_bytes(receipt: Mapping[str, Any]) -> bytes:
    path = _validate_file_record(receipt["projection"], "DET1.2 projection")
    return path.read_bytes()


def _load_model_repo(attempt_dir: Path, runtime_frame: Mapping[str, Any]):
    from transformers import AutoTokenizer
    from core.gpt_oss20b_tc import GptOss20B_TC, gpt_oss_grm_dialect_kwargs
    from core.graft_repository import GraftRepository
    from scripts import grm_e2e_session as e2e

    model, model_info = GptOss20B_TC.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), local_files_only=True)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    decode = lambda ids: tokenizer.decode(ids, clean_up_tokenization_spaces=False)
    dialect = gpt_oss_grm_dialect_kwargs(model.config)
    repo = GraftRepository(
        model,
        encode,
        decode,
        str(attempt_dir / "repository"),
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
        decisive_admission=bool(runtime_frame["resolved_flags"]["adm_decisive"]),
    )
    return e2e, model, tokenizer, repo, model_info


def run_arm(
    run_dir: Path,
    arm: str,
    *,
    lease_seconds: int,
) -> Path:
    if arm not in ALL_ARMS:
        raise DETError(f"unknown DET1.2 arm: {arm}")
    marker_path = _arm_marker(run_dir, arm)
    registration_path = _registration(run_dir)
    registration = read_json(registration_path)
    amendment_path = _check_det1_2_inventory(
        run_dir, registration, registration_path)
    runtime_frame_path = _one(run_dir, "runtime_frame_*.json")
    runtime_frame = read_json(runtime_frame_path)
    if runtime_frame.get("registration") != file_record(registration_path):
        raise DETError("DET1.2 runtime frame registration binding drifted")
    if runtime_frame.get("native_library") != file_record(NATIVE_LIB):
        raise DETError("DET1.2 native runtime drifted from the frozen frame")
    frozen_lease = int(runtime_frame["gpu_lease"]["actual_cap_seconds"])
    if int(lease_seconds) != frozen_lease:
        raise DETError(
            "DET1.2 lease differs from frozen runtime frame: "
            f"requested={lease_seconds} frozen={frozen_lease}")
    flags = runtime_frame["resolved_flags"]
    expected_settings = {
        "arena_width": 96,
        "topk": 3,
        "live_turns": 2,
        "max_live": 4096,
        "graft_storage_bits": 8,
        "sup_resolve": True,
        "ngen": 32,
        "max_trips": 1,
    }
    setting_drift = {
        key: {"frozen": flags.get(key), "harness": expected}
        for key, expected in expected_settings.items()
        if flags.get(key) != expected
    }
    if setting_drift:
        raise DETError(f"DET1.2 hardcoded Harbor settings drifted: {setting_drift}")
    os.environ.update(_runtime_env(runtime_frame))
    visible = _validate_visibility()
    if str(visible) != str(runtime_frame["cuda_visible_devices"]):
        raise DETError("DET1.2 visible GPU differs from the frozen runtime frame")

    incident_path, _incident_value, incident_projection = _incident(run_dir)
    registered_fixture = _registered_fixture(registration)
    if registered_fixture.get("source") != file_record(FIXTURE_SOURCE):
        raise DETError("DET1.2 Harbor fixture source drifted from registration")
    if marker_path.is_file():
        receipt_path, _receipt = _load_arm(run_dir, arm)
        print(f"arm={arm} status=already_complete receipt={receipt_path}", flush=True)
        return receipt_path
    fixture = json.loads(FIXTURE_SOURCE.read_text(encoding="utf-8"))
    attempts_root = marker_path.parent
    attempts = sorted(attempts_root.glob("attempt_*"))
    attempt_dir = attempts_root / f"attempt_{len(attempts) + 1:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)

    spec = ALL_ARMS[arm]
    model = tokenizer = repo = None
    started = time.monotonic()
    try:
        e2e, model, tokenizer, repo, model_info = _load_model_repo(
            attempt_dir, runtime_frame)
        node_to_idx = _install_fixture_nodes(repo, fixture)
        arena = repo.arena
        expected = [str(value) for value in registered_fixture["expected_values"]]
        stale = [
            str(value) for value in (
                list(registered_fixture.get("stale_values", ()))
                + list(registered_fixture.get("wrong_fact_values", ()))
            )
        ]
        profile = None
        aliases = _fixture_aliases_without_forward(
            fixture, node_to_idx, expected)
        alias_source = "fixture_revision_graph_no_model_forward"
        if bool(spec["profile"]):
            snapshot = _snapshot_counterfactual(arena)
            try:
                profile = route_fixture_profile(
                    arena,
                    str(registered_fixture["question"]),
                    expected[0],
                    live_excluded=set(),
                    route_limit=6,
                )
            finally:
                _restore_counterfactual(arena, snapshot)
            aliases = {int(value) for value in profile["logical_alias_ids"]}
            alias_source = "route_fixture_profile"

        served = _mechanistic_counterfactual(
            repo,
            e2e,
            e2e.probe_multimount_chat,
            question=str(registered_fixture["question"]),
            aliases=aliases,
            planted_miss=False,
            topk=3,
            ngen=32,
            max_trips=1,
            turn_idx=None,
            expected=expected,
            stale=stale,
            active_detectors=tuple(spec["hooks"]),
            ngh_schedule="legacy_inline",
            observer_shell=bool(spec["shell"]),
        )
        projection = {
            "answer": str(served["answer"]),
            "mounted_ids": [int(value) for value in served["mounted_ids"]],
        }
        comparison = compare_served_to_live_registry(registered_fixture, served)
        verbal = None
        verbal_requested = bool(spec["verbal"])
        verbal_executed = bool(
            verbal_requested and comparison.get("live_match") is True)
        if verbal_executed:
            verbal = _verbal_counterfactual(
                arena,
                prompt=(
                    str(registered_fixture["question"])
                    + "\n\n" + VERBAL_QUESTION
                ),
                mounts=served["mounted_ids"],
                ngen=32,
                clean_room=bool(
                    served["arena_info"].get("clean_room")
                    or served["arena_info"].get("point_lookup")
                ),
            )

        projection_path = write_json_exclusive(
            attempt_dir / "served_projection.json", projection)
        signals_path = write_json_exclusive(
            attempt_dir / "signals.json", dict(served["signals"]))
        verbal_path = (
            write_json_exclusive(attempt_dir / "verbal.json", verbal)
            if verbal is not None else None
        )
        receipt = {
            "schema": "grm.det1_2.hook_arm.v1",
            "status": "COMPLETE",
            "created_utc": utc_now(),
            "arm": arm,
            "fresh_process_per_arm": True,
            "order": file_record(ORDER),
            "incident": file_record(incident_path),
            "registration": file_record(registration_path),
            "source_amendment": file_record(amendment_path),
            "runtime_frame": file_record(runtime_frame_path),
            "fixture_source": file_record(FIXTURE_SOURCE),
            "fixture_id": FIXTURE_ID,
            "profile_prepass": bool(spec["profile"]),
            "observer_shell": bool(spec["shell"]),
            "active_mechanistic_hooks": list(spec["hooks"]),
            "d_verb_requested": verbal_requested,
            "d_verb_executed_after_live_guard": verbal_executed,
            "d_verb_skipped_by_live_guard": bool(
                verbal_requested and not verbal_executed),
            "ngh_schedule": (
                "legacy_inline" if "D-NGH" in spec["hooks"] else None),
            "projection": file_record(projection_path),
            "signals": file_record(signals_path),
            "verbal": file_record(verbal_path) if verbal_path else None,
            "live_baseline_comparison": comparison,
            "baseline_anchor": comparison.get("anchor"),
            "live_match": comparison.get("live_match") is True,
            "incident_projection": incident_projection,
            "exact_incident_match": projection == incident_projection,
            "profile": profile,
            "logical_alias_ids": sorted(aliases),
            "logical_alias_source": alias_source,
            "elapsed_seconds": time.monotonic() - started,
            "model_info": model_info,
            "runtime": _receipt_context(
                runtime_frame, lease_seconds=int(lease_seconds)),
            "gpu": _gpu_snapshot(),
        }
        receipt_path = write_content_addressed(
            attempt_dir, "arm_receipt", receipt)
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_exclusive(marker_path, {
            "schema": "grm.det1_2.hook_arm_marker.v1",
            "status": "COMPLETE",
            "arm": arm,
            "receipt": file_record(receipt_path),
        })
        print(f"arm={arm} receipt={receipt_path}", flush=True)
        return receipt_path
    finally:
        if repo is not None:
            try:
                repo.close()
            except BaseException:
                pass
        if model is not None:
            try:
                from core import kv_graft
                kv_graft.clear_injection(model)
            except BaseException:
                pass
        del repo, tokenizer, model
        gc.collect()
        try:
            import tensor_cuda as tc
            if hasattr(tc, "empty_cache"):
                tc.empty_cache()
        except BaseException:
            pass


def _run_child(
    run_dir: Path,
    arm: str,
    *,
    lease_seconds: int,
    wait_seconds: int,
) -> bool:
    if _arm_marker(run_dir, arm).is_file():
        _load_arm(run_dir, arm)
        print(f"arm={arm} status=already_complete", flush=True)
        return False
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-dir", str(run_dir),
        "--child-arm", arm,
        "--lease-seconds", str(int(lease_seconds)),
        "--lock-wait-seconds", str(int(wait_seconds)),
    ]
    rc = _run_process(
        command,
        _base_env(),
        int(wait_seconds) + int(lease_seconds) + 60,
    )
    if rc != 0:
        raise DETError(f"DET1.2 arm {arm} returned {rc}")
    return True


def _write_arm_failure(run_dir: Path, arm: str, exc: BaseException) -> Path:
    attempts_root = _arm_marker(run_dir, arm).parent
    attempts = sorted(attempts_root.glob("attempt_*"))
    directory = attempts_root / "failures"
    receipt = {
        "schema": "grm.det1_2.hook_arm_failure.v1",
        "status": "STOPPED",
        "created_utc": utc_now(),
        "arm": arm,
        "order": file_record(ORDER),
        "source_amendment": (
            file_record(Path(run_dir) / DET1_2_REPLAY_AMENDMENT)
            if (Path(run_dir) / DET1_2_REPLAY_AMENDMENT).is_file() else None
        ),
        "registration": file_record(_registration(run_dir)),
        "runtime_frame": file_record(_one(run_dir, "runtime_frame_*.json")),
        "failure_type": type(exc).__name__,
        "failure": str(exc),
        "traceback": traceback.format_exc(),
        "latest_attempt_directory": (
            str(attempts[-1]) if attempts else None),
        "occurred_before_arm_marker": True,
        "gpu": _gpu_snapshot(),
    }
    path = write_content_addressed(directory, "arm_failure", receipt)
    print(f"det1_2_arm_failure={path}", flush=True)
    return path


def _adjudicate(
    receipts: Mapping[str, Mapping[str, Any]],
    incident_projection: Mapping[str, Any],
) -> dict[str, Any]:
    required = set(INITIAL_ARMS)
    missing = sorted(required - set(receipts))
    if missing:
        raise DETError(f"DET1.2 adjudication missing initial arms: {missing}")

    def projection(arm: str) -> dict[str, Any]:
        return read_json(_validate_file_record(
            receipts[arm]["projection"], f"{arm} projection"))

    def is_incident(arm: str) -> bool:
        return projection(arm) == dict(incident_projection)

    control_bytes = _projection_bytes(receipts["control_a"])
    controls_reproducible = (
        control_bytes == _projection_bytes(receipts["control_b"])
        and receipts["control_a"].get("live_match") is True
        and receipts["control_b"].get("live_match") is True
    )
    control_projection = projection("control_a")
    profile_bytes = _projection_bytes(receipts["profile_only"])
    shell_bytes = _projection_bytes(receipts["observer_shell"])
    profile_shell_bytes = _projection_bytes(receipts["profile_observer_shell"])
    nuisance = {
        "profile_only": {
            "baseline_arm": "control_a",
            "clean": profile_bytes == control_bytes,
            "exact_incident": is_incident("profile_only"),
            "projection": projection("profile_only"),
        },
        "observer_shell": {
            "baseline_arm": "control_a",
            "clean": shell_bytes == control_bytes,
            "exact_incident": is_incident("observer_shell"),
            "projection": projection("observer_shell"),
        },
        "profile_observer_shell": {
            "baseline_arm": "profile_only",
            "clean": profile_shell_bytes == profile_bytes,
            "exact_incident": is_incident("profile_observer_shell"),
            "projection": projection("profile_observer_shell"),
        },
    }
    legacy_all_reproduced = is_incident("all_legacy")
    unprofiled_singletons = {
        detector: {
            "arm": arm,
            "baseline_arm": (
                "control_a" if detector == "D-VERB" else "observer_shell"),
            "perturbing": (
                _projection_bytes(receipts[arm])
                != (control_bytes if detector == "D-VERB" else shell_bytes)
            ),
            "exact_incident": is_incident(arm),
        }
        for detector, arm in HOOK_TO_ARM.items()
    }
    profiled_singletons = {
        detector: {
            "arm": "profile_" + arm,
            "baseline_arm": (
                "profile_only"
                if detector == "D-VERB" else "profile_observer_shell"),
            "perturbing": (
                _projection_bytes(receipts["profile_" + arm])
                != (profile_bytes if detector == "D-VERB" else profile_shell_bytes)
            ),
            "exact_incident": is_incident("profile_" + arm),
        }
        for detector, arm in HOOK_TO_ARM.items()
    }
    d_verb_replay_consistent = (
        _projection_bytes(receipts[HOOK_TO_ARM["D-VERB"]]) == control_bytes
        and _projection_bytes(
            receipts["profile_" + HOOK_TO_ARM["D-VERB"]]
        ) == profile_bytes
    )
    # D-VERB is invoked only after the served projection and live guard have
    # been frozen.  Its arm is a temporal-order check, never a causal arm for
    # this first served refusal.
    causative = tuple(MECHANISTIC)
    singleton_incident = [
        detector for detector in causative
        if profiled_singletons[detector]["exact_incident"]
    ]
    singleton_perturbing = [
        detector for detector in causative
        if profiled_singletons[detector]["perturbing"]
    ]
    result: dict[str, Any] = {
        "controls_reproducible": controls_reproducible,
        "control_projection": control_projection,
        "nuisance_controls": nuisance,
        "legacy_all_reproduced": legacy_all_reproduced,
        "unprofiled_singletons": unprofiled_singletons,
        "incident_path_singletons": profiled_singletons,
        "incident_path_singleton_exact_incident": singleton_incident,
        "incident_path_singleton_purity_leaks": singleton_perturbing,
        "nuisance_purity_leaks": [
            name for name, value in nuisance.items() if not value["clean"]
        ],
        "D-VERB_causal_status": "TEMPORALLY_EXCLUDED_FROM_FIRST_SERVED_PROJECTION",
        "D-VERB_replay_consistent": d_verb_replay_consistent,
        "pairwise_needed": False,
        "convicted_hooks": [],
        "convicted_common_prepass": False,
        "minimal_interactions": [],
    }
    if not controls_reproducible:
        result["status"] = "INVALID_NONDETERMINISTIC_CONTROLS"
    elif not d_verb_replay_consistent:
        # Both projections are frozen before the extra D-VERB attempt begins.
        # Drift here is therefore fresh-process replay nondeterminism, not a
        # causal D-VERB effect, and invalidates every downstream conviction.
        result["status"] = "INVALID_D_VERB_REPLAY_DRIFT"
    elif not legacy_all_reproduced:
        result["status"] = "INCONCLUSIVE_INCIDENT_NOT_REPRODUCED"
    elif nuisance["profile_only"]["exact_incident"]:
        result["status"] = "CONVICTED_COMMON_PROFILE_PREPASS"
        result["convicted_common_prepass"] = True
    elif nuisance["observer_shell"]["exact_incident"]:
        result["status"] = "CONVICTED_OBSERVER_SHELL"
    elif nuisance["profile_observer_shell"]["exact_incident"]:
        result["status"] = "CONVICTED_PROFILE_OBSERVER_SHELL_INTERACTION"
    elif singleton_incident:
        independent = [
            detector for detector in singleton_incident
            if unprofiled_singletons[detector]["exact_incident"]
        ]
        conditioned = [
            detector for detector in singleton_incident
            if detector not in independent
        ]
        result["status"] = (
            "CONVICTED_SINGLETON_HOOKS"
            if not conditioned else
            "CONVICTED_PROFILE_CONDITIONED_SINGLETON_HOOKS"
        )
        result["convicted_hooks"] = singleton_incident
        result["independent_singleton_hooks"] = independent
        result["profile_conditioned_singleton_hooks"] = conditioned
    else:
        result["status"] = (
            "CONVICTED_PURITY_LEAKS_EXACT_INCIDENT_PAIRWISE_REQUIRED"
            if singleton_perturbing else "PAIRWISE_REQUIRED"
        )
        result["convicted_hooks"] = singleton_perturbing
        result["pairwise_needed"] = True
        if set(PAIR_STAGE_ARMS) <= set(receipts):
            interactions = []
            for left, right in itertools.combinations(
                MECHANISTIC, 2
            ):
                plain_arm = _pair_name(left, right)
                profiled_arm = "profile_" + plain_arm
                if is_incident(profiled_arm):
                    interactions.append({
                        "hooks": [left, right],
                        "profile_conditioned": not is_incident(plain_arm),
                        "unprofiled_exact_incident": is_incident(plain_arm),
                    })
            result["minimal_interactions"] = interactions
            result["pairwise_needed"] = False
            if interactions:
                result["status"] = "CONVICTED_PAIRWISE_INTERACTION"
                result["convicted_hooks"] = sorted({
                    hook for value in interactions for hook in value["hooks"]
                })
            else:
                result["status"] = "CONVICTED_TRIPLE_MECHANISTIC_INTERACTION"
                result["convicted_hooks"] = list(MECHANISTIC)
                result["minimal_interactions"] = [{
                    "hooks": list(MECHANISTIC),
                    "profile_conditioned": not is_incident(
                        "all_mechanistic_no_profile"),
                    "unprofiled_exact_incident": is_incident(
                        "all_mechanistic_no_profile"),
                    "basis": (
                        "all_legacy reproduced while every profiled proper "
                        "singleton and pair subset was nonincident"
                    ),
                }]
    return result


def run_parent(
    run_dir: Path,
    *,
    lease_seconds: int,
    wait_seconds: int,
) -> Path:
    registration_path = _registration(run_dir)
    registration = read_json(registration_path)
    amendment_path = _check_det1_2_inventory(
        run_dir, registration, registration_path)
    incident_path, _incident_value, incident_projection = _incident(run_dir)
    initial = INITIAL_ARMS
    ran_previous = False
    for arm in initial:
        will_run = not _arm_marker(run_dir, arm).is_file()
        if ran_previous and will_run:
            print(f"inter_arm_gpu_gap_s={GAP_SECONDS}", flush=True)
            time.sleep(GAP_SECONDS)
        ran_previous = _run_child(
            run_dir,
            arm,
            lease_seconds=int(lease_seconds),
            wait_seconds=int(wait_seconds),
        ) or ran_previous
    records = {arm: _load_arm(run_dir, arm)[1] for arm in initial}
    verdict = _adjudicate(records, incident_projection)
    if verdict["pairwise_needed"]:
        for arm in PAIR_STAGE_ARMS:
            will_run = not _arm_marker(run_dir, arm).is_file()
            if ran_previous and will_run:
                print(f"inter_arm_gpu_gap_s={GAP_SECONDS}", flush=True)
                time.sleep(GAP_SECONDS)
            ran_previous = _run_child(
                run_dir,
                arm,
                lease_seconds=int(lease_seconds),
                wait_seconds=int(wait_seconds),
            ) or ran_previous
        records.update({
            arm: _load_arm(run_dir, arm)[1] for arm in PAIR_STAGE_ARMS
        })
        verdict = _adjudicate(records, incident_projection)

    receipt = {
        "schema": "grm.det1_2.hook_bisect.v1",
        "created_utc": utc_now(),
        "status": verdict["status"],
        "order": file_record(ORDER),
        "incident": file_record(incident_path),
        "registration": file_record(registration_path),
        "source_amendment": file_record(amendment_path),
        "runtime_frame": file_record(_one(run_dir, "runtime_frame_*.json")),
        "fresh_process_per_arm": True,
        "comparison": "raw canonical JSON bytes of answer plus authoritative mounts",
        "comparison_scope": (
            "incident-symptom bisection only; this projection is not the "
            "active-hooks DET-G1 served-path byte-identity gate"
        ),
        "arms": {
            arm: file_record(_load_arm(run_dir, arm)[0]) for arm in records
        },
        "verdict": verdict,
        "mechanism_inventory": _mechanism_inventory(),
        "conviction_law": (
            "controls must duplicate and match live; all_legacy must reproduce "
            "the lead incident; profile/shell controls take precedence; a hook "
            "is independently convicted only by an exact singleton incident match"
        ),
    }
    path = write_content_addressed(run_dir / "det1_2", "bisect_receipt", receipt)
    print(f"det1_2_bisect_receipt={path}", flush=True)
    return path


def selftest() -> dict[str, Any]:
    if (
        len(PAIR_ARMS) != 3
        or len(PROFILE_PAIR_ARMS) != 3
        or set(HOOK_TO_ARM.values()) - set(BASE_ARMS)
        or set(INITIAL_ARMS) - set(ALL_ARMS)
    ):
        raise DETError("DET1.2 arm matrix construction failed")
    if set(BASE_ARMS["all_legacy"]["hooks"]) != set(MECHANISTIC):
        raise DETError("DET1.2 all-legacy arm omits a mechanistic hook")
    if BASE_ARMS["d_verb"]["hooks"] or not BASE_ARMS["d_verb"]["verbal"]:
        raise DETError("DET1.2 D-VERB causal-order arm drifted")
    if any(BASE_ARMS[arm]["profile"] for arm in HOOK_TO_ARM.values()):
        raise DETError("DET1.2 unprofiled singleton arm gained a prepass")
    if not all(spec["profile"] for spec in PROFILE_HOOK_ARMS.values()):
        raise DETError("DET1.2 incident-path singleton lost its prepass")
    if not BASE_ARMS["profile_observer_shell"]["shell"]:
        raise DETError("DET1.2 profile-plus-shell nuisance control drifted")
    with tempfile.TemporaryDirectory(prefix="grm-det1-2-selftest-") as raw:
        directory = Path(raw)
        control = {"answer": "live", "mounted_ids": [1]}
        incident = {"answer": "refusal", "mounted_ids": [1]}
        nuisance_wrong = {"answer": "third", "mounted_ids": [1]}
        fake_call = [0]

        def fake_receipts(overrides: Mapping[str, Mapping[str, Any]]):
            fake_call[0] += 1
            out = {}
            for arm in (*INITIAL_ARMS, *PAIR_STAGE_ARMS):
                value = dict(overrides.get(arm, control))
                path = write_json_exclusive(
                    directory / f"{fake_call[0]}_{arm}.json", value)
                out[arm] = {"projection": file_record(path), "live_match": True}
            return out

        conditioned = _adjudicate(fake_receipts({
            "all_legacy": incident,
            "profile_d_ngh": incident,
            "profile_only": nuisance_wrong,
            "profile_observer_shell": nuisance_wrong,
            "profile_d_verb": nuisance_wrong,
        }), incident)
        if not (
            conditioned["status"]
                == "CONVICTED_PROFILE_CONDITIONED_SINGLETON_HOOKS"
            and conditioned["convicted_hooks"] == ["D-NGH"]
            and "profile_only" in conditioned["nuisance_purity_leaks"]
        ):
            raise DETError(
                f"DET1.2 conditional-singleton adjudication failed: {conditioned}")

        triple = _adjudicate(fake_receipts({"all_legacy": incident}), incident)
        if not (
            triple["status"] == "CONVICTED_TRIPLE_MECHANISTIC_INTERACTION"
            and set(triple["convicted_hooks"]) == set(MECHANISTIC)
        ):
            raise DETError(f"DET1.2 triple adjudication failed: {triple}")

        replay_drift = _adjudicate(fake_receipts({
            "all_legacy": incident,
            "d_verb": incident,
        }), incident)
        if replay_drift["status"] != "INVALID_D_VERB_REPLAY_DRIFT":
            raise DETError(
                f"DET1.2 D-VERB replay-drift gate failed: {replay_drift}")
    return {
        "schema": "grm.det1_2.gpu_selftest.v1",
        "status": "PASS",
        "checks": {
            "fresh_process_arm_count_before_pairs": len(INITIAL_ARMS),
            "unprofiled_pair_count": len(PAIR_ARMS),
            "incident_path_pair_count": len(PROFILE_PAIR_ARMS),
            "profile_nuisance_control": "PRESENT",
            "observer_shell_controls": "UNPROFILED_AND_PROFILED",
            "singleton_characterization": "UNPROFILED_AND_INCIDENT_PATH",
            "legacy_d_ngh_schedule": "legacy_inline",
            "d_verb_order": "after_served_projection",
            "adjudication_branches": (
                "CONDITIONAL_SINGLETON_TRIPLE_AND_D_VERB_DRIFT_PASS"
            ),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--child-arm", choices=tuple(ALL_ARMS))
    parser.add_argument("--lease-seconds", type=int, default=DEFAULT_LEASE_SECONDS)
    parser.add_argument("--lock-wait-seconds", type=int, default=7200)
    parser.add_argument("--selftest", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.selftest:
        print(json.dumps(selftest(), sort_keys=True))
        return 0
    if args.run_dir is None:
        raise DETError("--run-dir is required")
    run_dir = args.run_dir.expanduser().resolve()
    allowed = (ROOT / "artifacts" / "grm_det1").resolve()
    if not run_dir.is_dir() or not run_dir.is_relative_to(allowed):
        raise DETError(f"DET1.2 run dir must be beneath {allowed}: {run_dir}")
    if args.child_arm:
        with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
            try:
                run_arm(
                    run_dir,
                    str(args.child_arm),
                    lease_seconds=int(args.lease_seconds),
                )
            except BaseException as exc:
                _write_arm_failure(run_dir, str(args.child_arm), exc)
                raise
    else:
        run_parent(
            run_dir,
            lease_seconds=int(args.lease_seconds),
            wait_seconds=int(args.lock_wait_seconds),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
