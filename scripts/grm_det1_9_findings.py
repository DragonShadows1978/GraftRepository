#!/usr/bin/env python3
"""Record the DET1.9 campaign-r4 lived-control finding, without diagnosis.

This module is deliberately separate from registry selection and race analysis.
It replays the exact persisted t30/t33 evidence projections, freezes the ordered
finding text, and writes one append-only content-addressed JSON receipt.  It
does not investigate a cause or change any race adjudication.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.grm_det1_common import (
    canonical_json_bytes,
    file_record,
    utc_now,
    write_content_addressed,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN_RUN = ROOT / "artifacts/grm_det1/run_20260831T160525Z_2"
ORDER = ROOT / "orders/GRM_DET1_9_SUBSTITUTION_POOL.md"
REGISTRATION = FROZEN_RUN / "registration_62cb6c09cbec211d.json"
PREDECESSOR_AUTHORIZATION = (
    FROZEN_RUN / "det1_8_precollection_source_authorization.json"
)
E2E4_ATTEMPT = (
    FROZEN_RUN
    / "det1_4/campaign/det1_7/plant_registration/shards/e2e-4/attempt_001"
)
FINDINGS_DIR = FROZEN_RUN / "det1_9/findings"

SCHEMA = "grm.det1_9.campaign_r4_lived_control_finding.v1"
STATUS = "OPEN_POST_RACE_INVESTIGATION"
ORDER_ID = "GRM-DET1.9"
CAMPAIGN_ID = "r4"
RUN_ID = "run_20260831T160525Z_2"
T30_FIXTURE = "e2e_t30_atlas_tone"
T33_FIXTURE = "e2e_t33_polaris_mark"
T30_EXPECTED = "Cobalt-1-India"
T30_ANSWER = "Cobalt 1 India"
T33_EXPECTED = "Marble-4-Juliet"
T33_ANSWER = "I’m sorry, but I don’t have that information."
UNPLANTABLE_REASON = "LIVED_SERVED_CONTROL_INCORRECT_OR_REFUSAL"

T30_FINDING_TEXT = (
    "OPEN POST-RACE INVESTIGATION ITEM (knife-edge class candidate; do not "
    "investigate here): campaign r4 e2e_t30_atlas_tone's LIVED served control "
    "is incorrect (\"Cobalt 1 India\" versus exact \"Cobalt-1-India\") despite "
    "rank-1 correctly routed and mounted ([30]/[30]) — a no-co-mount, no-plant "
    "lived failure."
)
T33_FINDING_TEXT = (
    "Turn 33's e2e_t33_polaris_mark row also records probe_pass=false and "
    "served a refusal rather than exact \"Marble-4-Juliet\"; it is not a lawful "
    "substitute."
)


class FindingReceiptError(RuntimeError):
    """The ordered finding is not proved by the bound campaign evidence."""


@dataclass(frozen=True)
class FindingEvidencePaths:
    """All immutable files bound by the finding receipt."""

    order: Path
    base_registration: Path
    predecessor_authorization: Path
    worker_shard: Path
    chronological_session: Path
    observations: Path
    probe_scorecard: Path
    transcript: Path
    instrumentation: Path
    t30_snapshot_manifest: Path
    t33_snapshot_manifest: Path


PRODUCTION_EVIDENCE = FindingEvidencePaths(
    order=ORDER,
    base_registration=REGISTRATION,
    predecessor_authorization=PREDECESSOR_AUTHORIZATION,
    worker_shard=E2E4_ATTEMPT / "worker_shard_c83930bfa37df0c2.json",
    chronological_session=(
        E2E4_ATTEMPT / "chronological_session_0ffdb3d4be05e9b1.json"
    ),
    observations=E2E4_ATTEMPT / "observations.jsonl",
    probe_scorecard=E2E4_ATTEMPT / "session/probe_scorecard.json",
    transcript=E2E4_ATTEMPT / "session/transcript.jsonl",
    instrumentation=E2E4_ATTEMPT / "session/instrumentation.jsonl",
    t30_snapshot_manifest=(
        E2E4_ATTEMPT / "snapshots/e2e_t30_atlas_tone/rung_00/manifest.json"
    ),
    t33_snapshot_manifest=(
        E2E4_ATTEMPT / "snapshots/e2e_t33_polaris_mark/rung_00/manifest.json"
    ),
)

EVIDENCE_KEYS = tuple(FindingEvidencePaths.__dataclass_fields__)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FindingReceiptError(message)


def _validate_created_utc(value: Any) -> str:
    _require(isinstance(value, str) and value.endswith("Z"),
             "created_utc must have a UTC Z suffix")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FindingReceiptError("created_utc is not ISO-8601") from exc
    _require(
        parsed.tzinfo is not None
        and parsed.utcoffset() == timezone.utc.utcoffset(parsed),
        "created_utc is not UTC",
    )
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FindingReceiptError(f"cannot read {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            _require(
                isinstance(value, dict),
                f"{label}:{line_number} is not a JSON object",
            )
            rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FindingReceiptError(f"cannot read {label}: {path}") from exc
    return rows


def _inside_root(path: Path, root: Path, label: str) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(Path(root).resolve())
    except ValueError as exc:
        raise FindingReceiptError(f"{label} escapes the repository: {path}") from exc
    _require(resolved.is_file(), f"{label} is absent: {resolved}")
    return resolved


def _record(path: Path, *, repo_root: Path, label: str) -> dict[str, Any]:
    return file_record(_inside_root(path, repo_root, label), root=repo_root)


def _validate_record(
    value: Any, *, repo_root: Path, label: str
) -> tuple[Path, dict[str, Any]]:
    _require(isinstance(value, Mapping), f"{label} is not a file record")
    record = dict(value)
    _require(
        set(record) == {"path", "bytes", "sha256"},
        f"{label} has non-canonical fields",
    )
    shown = record.get("path")
    _require(isinstance(shown, str) and bool(shown), f"{label} path is empty")
    shown_path = Path(shown)
    _require(not shown_path.is_absolute(), f"{label} path must be repository-relative")
    path = _inside_root(Path(repo_root) / shown_path, repo_root, label)
    observed = file_record(path, root=repo_root)
    _require(record == observed, f"{label} record drift")
    return path, observed


def _unique_turn(
    rows: Sequence[Mapping[str, Any]], turn: int, label: str
) -> dict[str, Any]:
    selected = [dict(row) for row in rows if row.get("turn") == turn]
    _require(len(selected) == 1, f"{label} lacks exactly one turn {turn} row")
    return selected[0]


def _unique_effective_fixture(
    rows: Sequence[Mapping[str, Any]], fixture_id: str
) -> dict[str, Any]:
    selected = [
        dict(row) for row in rows
        if row.get("effective_fixture_id") == fixture_id
    ]
    _require(
        len(selected) == 1,
        f"observations lack exactly one {fixture_id} effective row",
    )
    return selected[0]


def _require_snapshot_base(
    manifest: Mapping[str, Any],
    *,
    fixture_id: str,
    registration_sha256: str,
    authorization_sha256: str,
    process_instance_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _require(
        manifest.get("schema") == "grm.det1_3.model_visible_snapshot.v2",
        f"{fixture_id} snapshot schema drift",
    )
    _require(
        manifest.get("complete") is True
        and manifest.get("capture_finalized") is True,
        f"{fixture_id} snapshot is incomplete",
    )
    _require(
        manifest.get("label") == "lived"
        and manifest.get("phase") == "before_probe_prefill",
        f"{fixture_id} snapshot is not finalized LIVED evidence",
    )
    provenance = manifest.get("provenance")
    state = manifest.get("state")
    answer = manifest.get("linked_answer")
    _require(isinstance(provenance, Mapping), f"{fixture_id} provenance missing")
    _require(isinstance(state, Mapping), f"{fixture_id} snapshot state missing")
    _require(isinstance(answer, Mapping), f"{fixture_id} linked answer missing")
    _require(
        provenance.get("arm") == "lived"
        and provenance.get("fixture_id") == fixture_id
        and provenance.get("probe_id") == fixture_id
        and provenance.get("registration_sha256") == registration_sha256
        and provenance.get("det1_7_precollection_authorization_sha256")
        == authorization_sha256
        and provenance.get("process_instance_sha256")
        == process_instance_sha256,
        f"{fixture_id} snapshot provenance drift",
    )
    _require(
        answer.get("process_instance_sha256") == process_instance_sha256,
        f"{fixture_id} linked answer process drift",
    )
    return dict(provenance), dict(state), dict(answer)


def _require_observation_base(
    row: Mapping[str, Any],
    *,
    effective_fixture_id: str,
    snapshot_record: Mapping[str, Any],
    process_instance_sha256: str,
) -> dict[str, Any]:
    _require(
        row.get("schema") == "grm.det1_7.plant_observation.v1",
        f"{effective_fixture_id} observation schema drift",
    )
    _require(
        row.get("effective_fixture_id") == effective_fixture_id
        and row.get("status") == "UNPLANTABLE"
        and row.get("unplantable_reason") == UNPLANTABLE_REASON,
        f"{effective_fixture_id} observation status drift",
    )
    _require(
        row.get("race_row") is False
        and row.get("detector_arms_active") == [],
        f"{effective_fixture_id} observation is not collection-only",
    )
    _require(
        row.get("selected_target_id") is None
        and row.get("behavioral_breaker") is None,
        f"{effective_fixture_id} observation unexpectedly selected a plant",
    )
    _require(
        row.get("process_instance_sha256") == process_instance_sha256,
        f"{effective_fixture_id} observation process drift",
    )
    _require(
        row.get("source_snapshot") == dict(snapshot_record),
        f"{effective_fixture_id} observation snapshot binding drift",
    )
    effective = row.get("effective_fixture")
    _require(
        isinstance(effective, Mapping)
        and effective.get("fixture_id") == effective_fixture_id
        and effective.get("split") == "eval",
        f"{effective_fixture_id} effective fixture drift",
    )
    return dict(effective)


def _evidence_projection(
    records: Mapping[str, Any], *, repo_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(
        set(records) == set(EVIDENCE_KEYS),
        "evidence record enumeration drift",
    )
    paths: dict[str, Path] = {}
    normalized: dict[str, dict[str, Any]] = {}
    for key in EVIDENCE_KEYS:
        path, record = _validate_record(
            records[key], repo_root=repo_root, label=key.replace("_", " ")
        )
        paths[key] = path
        normalized[key] = record

    order_text = paths["order"].read_text(encoding="utf-8")
    _require(ORDER_ID in order_text, "bound order is not GRM-DET1.9")
    _require(
        "open post-race investigation item" in order_text.casefold()
        and "probe_pass=false" in order_text,
        "bound order omits the finding requirement",
    )

    registration = _read_json(paths["base_registration"], "base registration")
    _require(
        registration.get("schema") == "grm.det1.registration.v1",
        "base registration schema drift",
    )
    fixtures = registration.get("fixtures")
    _require(isinstance(fixtures, list), "base registration fixtures missing")
    registered_t30 = [
        row for row in fixtures
        if isinstance(row, Mapping) and row.get("fixture_id") == T30_FIXTURE
    ]
    _require(len(registered_t30) == 1, "registered t30 fixture drift")
    _require(
        registered_t30[0].get("turn") == 30
        and registered_t30[0].get("split") == "eval"
        and registered_t30[0].get("expected_values") == [T30_EXPECTED],
        "registered t30 exact control drift",
    )

    authorization = _read_json(
        paths["predecessor_authorization"], "predecessor authorization"
    )
    _require(
        authorization.get("schema")
        == "grm.det1_8.precollection_source_authorization.v1"
        and authorization.get("status")
        == "AUTHORIZED_LIVED_PLANT_REGISTRY_COLLECTION_ONLY",
        "predecessor authorization schema/status drift",
    )
    _require(
        authorization.get("registration") == normalized["base_registration"],
        "predecessor authorization registration binding drift",
    )

    worker = _read_json(paths["worker_shard"], "e2e-4 worker shard")
    _require(
        worker.get("schema") == "grm.det1_7.worker_shard.v1"
        and worker.get("status") == "COMPLETE"
        and worker.get("stage") == "plant_registration"
        and worker.get("spec") == "e2e-4"
        and worker.get("variants") == ["served"],
        "e2e-4 worker identity/status drift",
    )
    _require(
        worker.get("historical_rows_reused") is False
        and worker.get("detector_hooks") == [],
        "e2e-4 worker is not lived collection-only evidence",
    )
    _require(
        worker.get("registration") == normalized["base_registration"]
        and worker.get("observations") == normalized["observations"]
        and worker.get("chronological_session")
        == normalized["chronological_session"],
        "e2e-4 worker file bindings drift",
    )
    worker_provenance = worker.get("det1_7_provenance")
    _require(
        isinstance(worker_provenance, Mapping)
        and worker_provenance.get("precollection_authorization")
        == normalized["predecessor_authorization"],
        "e2e-4 worker authorization binding drift",
    )
    process_instance = worker.get("process_instance_sha256")
    _require(
        isinstance(process_instance, str) and len(process_instance) == 64,
        "e2e-4 process identity missing",
    )

    chronological = _read_json(
        paths["chronological_session"], "chronological session"
    )
    _require(
        chronological.get("schema") == "grm.det1_5.chronological_session.v1"
        and chronological.get("status") == "COMPLETE_BOUNDARY"
        and chronological.get("stage") == "plant_registration"
        and chronological.get("spec") == "e2e-4"
        and chronological.get("process_instance_sha256") == process_instance,
        "chronological session identity/status drift",
    )
    chronological_files = chronological.get("files")
    _require(isinstance(chronological_files, Mapping),
             "chronological session files missing")
    for filename, key in (
        ("instrumentation.jsonl", "instrumentation"),
        ("probe_scorecard.json", "probe_scorecard"),
        ("transcript.jsonl", "transcript"),
    ):
        _require(
            chronological_files.get(filename) == normalized[key],
            f"chronological {filename} binding drift",
        )

    scorecard = _read_json(paths["probe_scorecard"], "probe scorecard")
    probes = scorecard.get("probes")
    _require(
        scorecard.get("schema") == "grm_e2e_probe_scorecard_v1"
        and isinstance(probes, list),
        "probe scorecard schema/rows drift",
    )
    _require(
        scorecard.get("total") == len(probes)
        and scorecard.get("passed")
        == sum(row.get("pass") is True for row in probes if isinstance(row, Mapping))
        and scorecard.get("all_passed") is False,
        "probe scorecard aggregate drift",
    )
    t30_score = _unique_turn(probes, 30, "probe scorecard")
    t33_score = _unique_turn(probes, 33, "probe scorecard")

    transcript_rows = _read_jsonl(paths["transcript"], "transcript")
    instrumentation_rows = _read_jsonl(
        paths["instrumentation"], "instrumentation"
    )
    t30_transcript = _unique_turn(transcript_rows, 30, "transcript")
    t33_transcript = _unique_turn(transcript_rows, 33, "transcript")
    t30_instrumentation = _unique_turn(
        instrumentation_rows, 30, "instrumentation"
    )
    t33_instrumentation = _unique_turn(
        instrumentation_rows, 33, "instrumentation"
    )
    for turn, transcript, instrumentation, score in (
        (30, t30_transcript, t30_instrumentation, t30_score),
        (33, t33_transcript, t33_instrumentation, t33_score),
    ):
        _require(
            transcript.get("kind") == "probe"
            and instrumentation.get("kind") == "probe",
            f"turn {turn} is not a probe",
        )
        _require(
            "plant_mode" in transcript
            and transcript.get("plant_mode") is None
            and "plant_mode" in instrumentation
            and instrumentation.get("plant_mode") is None,
            f"turn {turn} applied a plant",
        )
        _require(
            instrumentation.get("probe_score") == score,
            f"turn {turn} instrumentation/scorecard drift",
        )
        _require(
            transcript.get("assistant") == score.get("answer"),
            f"turn {turn} transcript/scorecard answer drift",
        )

    observations = _read_jsonl(paths["observations"], "observations")
    t30_observation = _unique_effective_fixture(observations, T30_FIXTURE)
    t33_observation = _unique_effective_fixture(observations, T33_FIXTURE)
    t30_manifest = _read_json(
        paths["t30_snapshot_manifest"], "t30 snapshot manifest"
    )
    t33_manifest = _read_json(
        paths["t33_snapshot_manifest"], "t33 snapshot manifest"
    )
    _, t30_state, t30_linked = _require_snapshot_base(
        t30_manifest,
        fixture_id=T30_FIXTURE,
        registration_sha256=normalized["base_registration"]["sha256"],
        authorization_sha256=normalized["predecessor_authorization"]["sha256"],
        process_instance_sha256=process_instance,
    )
    _, t33_state, t33_linked = _require_snapshot_base(
        t33_manifest,
        fixture_id=T33_FIXTURE,
        registration_sha256=normalized["base_registration"]["sha256"],
        authorization_sha256=normalized["predecessor_authorization"]["sha256"],
        process_instance_sha256=process_instance,
    )
    t30_effective = _require_observation_base(
        t30_observation,
        effective_fixture_id=T30_FIXTURE,
        snapshot_record=normalized["t30_snapshot_manifest"],
        process_instance_sha256=process_instance,
    )
    t33_effective = _require_observation_base(
        t33_observation,
        effective_fixture_id=T33_FIXTURE,
        snapshot_record=normalized["t33_snapshot_manifest"],
        process_instance_sha256=process_instance,
    )

    _require(
        t30_effective.get("expected_values") == [T30_EXPECTED]
        and t30_observation.get("fixture_id") == T30_FIXTURE
        and t30_observation.get("candidate_for_fixture_id") == T30_FIXTURE
        and t30_observation.get("served_answer") == T30_ANSWER
        and t30_observation.get("served_answer_correct") is False
        and t30_observation.get("mounted_ids") == [30]
        and t30_observation.get("substitution") == {"status": "NONE"},
        "t30 observation projection drift",
    )
    t30_route = t30_score.get("route_ranking")
    t30_admission = t30_score.get("admission")
    _require(
        isinstance(t30_route, Mapping) and isinstance(t30_admission, Mapping),
        "t30 routing evidence missing",
    )
    _require(
        t30_score.get("fact_id") == "atlas tone"
        and t30_score.get("accepts") == [T30_EXPECTED]
        and t30_score.get("expected") == T30_EXPECTED
        and t30_score.get("answer") == T30_ANSWER
        and t30_score.get("pass") is False
        and t30_score.get("contains_expected") is False
        and t30_score.get("mount_plan") == [30]
        and t30_score.get("mounted_ids") == [30],
        "t30 served-control projection drift",
    )
    _require(
        t30_admission.get("admission_rank_plan") == [30]
        and t30_admission.get("admission_identified_candidates") == [30]
        and t30_route.get("source_node_id") == 30
        and t30_route.get("source_rank") == 1
        and isinstance(t30_route.get("ranking_ids"), list)
        and t30_route["ranking_ids"][:1] == [30]
        and t30_route.get("mount_plan") == [30]
        and t30_route.get("mount_fitted") == [30]
        and t30_route.get("mount_dropped_for_width") == [],
        "t30 rank-1 [30]/[30] routing projection drift",
    )
    _require(
        t30_state.get("admission.authoritative_mounts") == [30]
        and t30_state.get("admission.final_mounts") == [30]
        and t30_state.get("admission.rank_plan") == [30]
        and t30_state.get("admission.current_planned") == [30]
        and t30_state.get("arena.cur_mounts") == [30],
        "t30 lived snapshot mount projection drift",
    )
    _require(
        t30_linked.get("attempt_answer") == T30_ANSWER
        and t30_linked.get("probe_answer") == T30_ANSWER
        and t30_linked.get("attempt_answer_correct") is False
        and t30_linked.get("probe_answer_correct") is False
        and t30_linked.get("attempt_refusal") is False
        and t30_linked.get("probe_refusal") is False
        and t30_linked.get("attempt_mounts") == [30]
        and t30_linked.get("probe_mounts") == [30],
        "t30 linked served answer projection drift",
    )

    t33_substitution = t33_observation.get("substitution")
    _require(
        t33_effective.get("expected_values") == [T33_EXPECTED]
        and t33_observation.get("fixture_id") == T30_FIXTURE
        and t33_observation.get("candidate_for_fixture_id") == T30_FIXTURE
        and t33_observation.get("served_answer") == T33_ANSWER
        and t33_observation.get("served_answer_correct") is False
        and isinstance(t33_substitution, Mapping)
        and t33_substitution.get("status") == "SUBSTITUTED"
        and t33_substitution.get("original_fixture_id") == T30_FIXTURE,
        "t33 reserve observation projection drift",
    )
    _require(
        t33_score.get("fact_id") == "polaris mark"
        and t33_score.get("accepts") == [T33_EXPECTED]
        and t33_score.get("expected") == T33_EXPECTED
        and t33_score.get("answer") == T33_ANSWER
        and t33_score.get("pass") is False
        and t33_score.get("contains_expected") is False,
        "t33 probe_pass=false refusal projection drift",
    )
    _require(
        t33_linked.get("attempt_answer") == T33_ANSWER
        and t33_linked.get("probe_answer") == T33_ANSWER
        and t33_linked.get("attempt_answer_correct") is False
        and t33_linked.get("probe_answer_correct") is False
        and t33_linked.get("attempt_refusal") is True
        and t33_linked.get("probe_refusal") is True,
        "t33 linked refusal projection drift",
    )
    _require(
        t33_state.get("admission.authoritative_mounts")
        == t33_score.get("mounted_ids")
        and t33_state.get("admission.final_mounts")
        == t33_score.get("mounted_ids"),
        "t33 lived snapshot/scorecard mount drift",
    )

    t30_projection = {
        "answer_refusal": False,
        "co_mount": False,
        "expected_exact": T30_EXPECTED,
        "lived": True,
        "mounted_ids": [30],
        "observation_status": "UNPLANTABLE",
        "plant_applied": False,
        "probe_pass": False,
        "rank_plan": [30],
        "route_and_mount_projection": "[30]/[30]",
        "served_answer": T30_ANSWER,
        "source_node_id": 30,
        "source_rank": 1,
        "unplantable_reason": UNPLANTABLE_REASON,
    }
    t33_projection = {
        "expected_exact": T33_EXPECTED,
        "lived": True,
        "not_lawful_substitute": True,
        "observation_status": "UNPLANTABLE",
        "plant_applied": False,
        "probe_pass": False,
        "served_answer": T33_ANSWER,
        "served_refusal": True,
        "unplantable_reason": UNPLANTABLE_REASON,
    }
    return t30_projection, t33_projection


def _ordered_findings(
    t30_projection: Mapping[str, Any], t33_projection: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            "evidence_projection": dict(t30_projection),
            "fixture_id": T30_FIXTURE,
            "sequence": 1,
            "text": T30_FINDING_TEXT,
            "turn": 30,
        },
        {
            "evidence_projection": dict(t33_projection),
            "fixture_id": T33_FIXTURE,
            "sequence": 2,
            "text": T33_FINDING_TEXT,
            "turn": 33,
        },
    ]


def build_finding_receipt(
    evidence_paths: FindingEvidencePaths = PRODUCTION_EVIDENCE,
    *,
    created_utc: str,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    """Validate the lived evidence and derive the exact finding receipt."""
    _validate_created_utc(created_utc)
    root = Path(repo_root).resolve()
    records = {
        key: _record(
            getattr(evidence_paths, key),
            repo_root=root,
            label=key.replace("_", " "),
        )
        for key in EVIDENCE_KEYS
    }
    t30_projection, t33_projection = _evidence_projection(records, repo_root=root)
    receipt = {
        "campaign": {
            "campaign_id": CAMPAIGN_ID,
            "collection_stage": "plant_registration",
            "lived_collected": True,
            "run_id": RUN_ID,
            "worker_spec": "e2e-4",
        },
        "created_utc": created_utc,
        "evidence_records": records,
        "ordered_findings": _ordered_findings(t30_projection, t33_projection),
        "schema": SCHEMA,
        "scope": {
            "candidate_class": "KNIFE_EDGE",
            "causal_conclusion": None,
            "cause_investigated": False,
            "do_not_investigate_here": True,
            "finding_only": True,
            "investigation_status": STATUS,
            "race_adjudication_changed": False,
        },
        "status": STATUS,
    }
    validate_finding_receipt(receipt, repo_root=root)
    return receipt


def validate_finding_receipt(
    receipt: Mapping[str, Any], *, repo_root: Path = ROOT
) -> dict[str, Any]:
    """Re-read every bound file and replay both exact finding projections."""
    _require(isinstance(receipt, Mapping), "finding receipt is not an object")
    value = dict(receipt)
    _require(
        set(value)
        == {
            "campaign",
            "created_utc",
            "evidence_records",
            "ordered_findings",
            "schema",
            "scope",
            "status",
        },
        "finding receipt top-level fields drift",
    )
    _require(value.get("schema") == SCHEMA, "finding receipt schema drift")
    _require(value.get("status") == STATUS, "finding receipt status drift")
    _validate_created_utc(value.get("created_utc"))
    _require(
        value.get("campaign")
        == {
            "campaign_id": CAMPAIGN_ID,
            "collection_stage": "plant_registration",
            "lived_collected": True,
            "run_id": RUN_ID,
            "worker_spec": "e2e-4",
        },
        "finding receipt campaign projection drift",
    )
    _require(
        value.get("scope")
        == {
            "candidate_class": "KNIFE_EDGE",
            "causal_conclusion": None,
            "cause_investigated": False,
            "do_not_investigate_here": True,
            "finding_only": True,
            "investigation_status": STATUS,
            "race_adjudication_changed": False,
        },
        "finding scope drift or causal upgrade",
    )
    records = value.get("evidence_records")
    _require(isinstance(records, Mapping), "finding evidence records missing")
    t30_projection, t33_projection = _evidence_projection(
        records, repo_root=Path(repo_root).resolve()
    )
    expected_findings = _ordered_findings(t30_projection, t33_projection)
    _require(
        value.get("ordered_findings") == expected_findings,
        "ordered finding text/projection drift",
    )
    return {
        "campaign_id": CAMPAIGN_ID,
        "finding_count": 2,
        "schema": "grm.det1_9.campaign_r4_lived_control_finding_validation.v1",
        "status": "PASS_EXACT_OPEN_FINDING_NO_CAUSAL_CONCLUSION",
        "t30_probe_pass": False,
        "t33_probe_pass": False,
    }


def write_finding_receipt(
    receipt: Mapping[str, Any],
    *,
    directory: Path = FINDINGS_DIR,
    repo_root: Path = ROOT,
) -> Path:
    """Validate and exclusively write a content-addressed finding receipt."""
    validate_finding_receipt(receipt, repo_root=repo_root)
    return write_content_addressed(
        directory, "campaign_r4_lived_control_finding", dict(receipt)
    )


def author_finding_receipt(
    evidence_paths: FindingEvidencePaths = PRODUCTION_EVIDENCE,
    *,
    created_utc: str | None = None,
    directory: Path = FINDINGS_DIR,
    repo_root: Path = ROOT,
) -> tuple[Path, dict[str, Any]]:
    """Build, validate, and append the campaign-r4 finding receipt."""
    receipt = build_finding_receipt(
        evidence_paths,
        created_utc=created_utc or utc_now(),
        repo_root=repo_root,
    )
    path = write_finding_receipt(
        receipt, directory=directory, repo_root=repo_root
    )
    return path, receipt


def validate_finding_receipt_file(
    path: Path, *, repo_root: Path = ROOT
) -> dict[str, Any]:
    """Validate canonical bytes, content-address name, bindings, and semantics."""
    receipt_path = _inside_root(path, repo_root, "finding receipt")
    value = _read_json(receipt_path, "finding receipt")
    payload = canonical_json_bytes(value)
    _require(receipt_path.read_bytes() == payload,
             "finding receipt is not canonically encoded")
    digest = hashlib.sha256(payload).hexdigest()
    _require(
        receipt_path.name
        == f"campaign_r4_lived_control_finding_{digest[:16]}.json",
        "finding receipt filename is not content-addressed",
    )
    return validate_finding_receipt(value, repo_root=repo_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--created-utc",
        default=None,
        help="explicit UTC timestamp; defaults to the current UTC time",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FINDINGS_DIR,
        help="append-only content-addressed receipt directory",
    )
    parser.add_argument(
        "--validate",
        type=Path,
        default=None,
        help="validate an existing finding receipt instead of authoring one",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.validate is not None:
        result = validate_finding_receipt_file(args.validate)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    path, receipt = author_finding_receipt(
        created_utc=args.created_utc,
        directory=args.output_dir,
    )
    print(json.dumps({
        "finding_text": [row["text"] for row in receipt["ordered_findings"]],
        "receipt": file_record(path),
        "status": receipt["status"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
