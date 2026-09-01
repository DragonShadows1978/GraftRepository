from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.grm_det1_common import canonical_json_bytes, file_record
from scripts.grm_det1_9_findings import (
    FindingEvidencePaths,
    FindingReceiptError,
    SCHEMA,
    STATUS,
    T30_ANSWER,
    T30_EXPECTED,
    T30_FINDING_TEXT,
    T30_FIXTURE,
    T33_ANSWER,
    T33_EXPECTED,
    T33_FINDING_TEXT,
    T33_FIXTURE,
    UNPLANTABLE_REASON,
    author_finding_receipt,
    build_finding_receipt,
    validate_finding_receipt,
    validate_finding_receipt_file,
)


CREATED_UTC = "2026-09-01T03:04:05Z"
PROCESS_INSTANCE = "1" * 64


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.write_text(payload, encoding="utf-8")


def _score_rows(*, t30_source_rank: int = 1, t33_pass: bool = False):
    t30 = {
        "accepts": [T30_EXPECTED],
        "admission": {
            "admission_identified_candidates": [30],
            "admission_rank_plan": [30],
        },
        "answer": T30_ANSWER,
        "contains_expected": False,
        "expected": T30_EXPECTED,
        "fact_id": "atlas tone",
        "mount_plan": [30],
        "mounted_ids": [30],
        "pass": False,
        "route_ranking": {
            "mount_dropped_for_width": [],
            "mount_fitted": [30],
            "mount_plan": [30],
            "ranking_ids": [30, 4],
            "source_node_id": 30,
            "source_rank": t30_source_rank,
        },
        "turn": 30,
    }
    t33 = {
        "accepts": [T33_EXPECTED],
        "admission": {
            "admission_identified_candidates": [],
            "admission_rank_plan": [4, 19, 8],
        },
        "answer": T33_ANSWER,
        "contains_expected": False,
        "expected": T33_EXPECTED,
        "fact_id": "polaris mark",
        "mount_plan": [4, 19, 8],
        "mounted_ids": [4],
        "pass": t33_pass,
        "route_ranking": {
            "mount_dropped_for_width": [19, 8],
            "mount_fitted": [4],
            "mount_plan": [4, 19, 8],
            "ranking_ids": [4, 19, 8],
            "source_node_id": 35,
            "source_rank": None,
        },
        "turn": 33,
    }
    return t30, t33


def _snapshot(
    *,
    fixture_id: str,
    registration_sha256: str,
    authorization_sha256: str,
    answer: str,
    mounts: list[int],
    rank_plan: list[int],
    refusal: bool,
) -> dict:
    return {
        "capture_finalized": True,
        "complete": True,
        "label": "lived",
        "linked_answer": {
            "attempt_answer": answer,
            "attempt_answer_correct": False,
            "attempt_mounts": mounts,
            "attempt_refusal": refusal,
            "probe_answer": answer,
            "probe_answer_correct": False,
            "probe_mounts": mounts,
            "probe_refusal": refusal,
            "process_instance_sha256": PROCESS_INSTANCE,
        },
        "phase": "before_probe_prefill",
        "provenance": {
            "arm": "lived",
            "det1_7_precollection_authorization_sha256": authorization_sha256,
            "fixture_id": fixture_id,
            "probe_id": fixture_id,
            "process_instance_sha256": PROCESS_INSTANCE,
            "registration_sha256": registration_sha256,
        },
        "schema": "grm.det1_3.model_visible_snapshot.v2",
        "state": {
            "admission.authoritative_mounts": mounts,
            "admission.current_planned": rank_plan,
            "admission.final_mounts": mounts,
            "admission.rank_plan": rank_plan,
            "arena.cur_mounts": mounts,
        },
    }


def _observation(
    *,
    effective_fixture_id: str,
    expected: str,
    answer: str,
    mounts: list[int],
    snapshot_record: dict,
    substitution: dict,
) -> dict:
    return {
        "behavioral_breaker": None,
        "candidate_for_fixture_id": T30_FIXTURE,
        "detector_arms_active": [],
        "effective_fixture": {
            "expected_values": [expected],
            "fixture_id": effective_fixture_id,
            "split": "eval",
        },
        "effective_fixture_id": effective_fixture_id,
        "fixture_id": T30_FIXTURE,
        "mounted_ids": mounts,
        "process_instance_sha256": PROCESS_INSTANCE,
        "race_row": False,
        "schema": "grm.det1_7.plant_observation.v1",
        "selected_target_id": None,
        "served_answer": answer,
        "served_answer_correct": False,
        "source_snapshot": snapshot_record,
        "status": "UNPLANTABLE",
        "substitution": substitution,
        "unplantable_reason": UNPLANTABLE_REASON,
    }


def _fixture(
    tmp_path: Path,
    *,
    t30_source_rank: int = 1,
    t33_pass: bool = False,
    plant_mode=None,
) -> tuple[Path, FindingEvidencePaths]:
    root = tmp_path / "repo"
    order = root / "orders/GRM_DET1_9_SUBSTITUTION_POOL.md"
    registration = root / "artifacts/run/registration.json"
    authorization = root / "artifacts/run/det1_8_authorization.json"
    attempt = root / "artifacts/run/campaign/shards/e2e-4/attempt_001"
    worker = attempt / "worker_shard.json"
    chronological = attempt / "chronological_session.json"
    observations = attempt / "observations.jsonl"
    scorecard = attempt / "session/probe_scorecard.json"
    transcript = attempt / "session/transcript.jsonl"
    instrumentation = attempt / "session/instrumentation.jsonl"
    t30_manifest = attempt / "snapshots/t30/manifest.json"
    t33_manifest = attempt / "snapshots/t33/manifest.json"

    _write(
        order,
        "# ORDER GRM-DET1.9\n\n"
        "Write an open post-race investigation item and note probe_pass=false.\n",
    )
    _write_json(registration, {
        "fixtures": [{
            "expected_values": [T30_EXPECTED],
            "fixture_id": T30_FIXTURE,
            "split": "eval",
            "turn": 30,
        }],
        "schema": "grm.det1.registration.v1",
    })
    registration_record = file_record(registration, root=root)
    _write_json(authorization, {
        "registration": registration_record,
        "schema": "grm.det1_8.precollection_source_authorization.v1",
        "status": "AUTHORIZED_LIVED_PLANT_REGISTRY_COLLECTION_ONLY",
    })
    authorization_record = file_record(authorization, root=root)

    _write_json(t30_manifest, _snapshot(
        fixture_id=T30_FIXTURE,
        registration_sha256=registration_record["sha256"],
        authorization_sha256=authorization_record["sha256"],
        answer=T30_ANSWER,
        mounts=[30],
        rank_plan=[30],
        refusal=False,
    ))
    _write_json(t33_manifest, _snapshot(
        fixture_id=T33_FIXTURE,
        registration_sha256=registration_record["sha256"],
        authorization_sha256=authorization_record["sha256"],
        answer=T33_ANSWER,
        mounts=[4],
        rank_plan=[4, 19, 8],
        refusal=True,
    ))
    t30_snapshot_record = file_record(t30_manifest, root=root)
    t33_snapshot_record = file_record(t33_manifest, root=root)
    _write_jsonl(observations, [
        _observation(
            effective_fixture_id=T30_FIXTURE,
            expected=T30_EXPECTED,
            answer=T30_ANSWER,
            mounts=[30],
            snapshot_record=t30_snapshot_record,
            substitution={"status": "NONE"},
        ),
        _observation(
            effective_fixture_id=T33_FIXTURE,
            expected=T33_EXPECTED,
            answer=T33_ANSWER,
            mounts=[4],
            snapshot_record=t33_snapshot_record,
            substitution={
                "effective_fixture": {"fixture_id": T33_FIXTURE},
                "original_fixture_id": T30_FIXTURE,
                "reason": "same-session reserve",
                "status": "SUBSTITUTED",
            },
        ),
    ])

    t30_score, t33_score = _score_rows(
        t30_source_rank=t30_source_rank,
        t33_pass=t33_pass,
    )
    probes = [t30_score, t33_score]
    _write_json(scorecard, {
        "all_passed": False,
        "passed": sum(row["pass"] is True for row in probes),
        "probes": probes,
        "schema": "grm_e2e_probe_scorecard_v1",
        "total": len(probes),
    })
    _write_jsonl(transcript, [
        {
            "assistant": T30_ANSWER,
            "kind": "probe",
            "plant_mode": plant_mode,
            "turn": 30,
        },
        {
            "assistant": T33_ANSWER,
            "kind": "probe",
            "plant_mode": plant_mode,
            "turn": 33,
        },
    ])
    _write_jsonl(instrumentation, [
        {
            "kind": "probe",
            "plant_mode": plant_mode,
            "probe_score": t30_score,
            "turn": 30,
        },
        {
            "kind": "probe",
            "plant_mode": plant_mode,
            "probe_score": t33_score,
            "turn": 33,
        },
    ])

    _write_json(chronological, {
        "files": {
            "instrumentation.jsonl": file_record(instrumentation, root=root),
            "probe_scorecard.json": file_record(scorecard, root=root),
            "transcript.jsonl": file_record(transcript, root=root),
        },
        "process_instance_sha256": PROCESS_INSTANCE,
        "schema": "grm.det1_5.chronological_session.v1",
        "spec": "e2e-4",
        "stage": "plant_registration",
        "status": "COMPLETE_BOUNDARY",
    })
    _write_json(worker, {
        "chronological_session": file_record(chronological, root=root),
        "det1_7_provenance": {
            "precollection_authorization": authorization_record,
        },
        "detector_hooks": [],
        "historical_rows_reused": False,
        "observations": file_record(observations, root=root),
        "process_instance_sha256": PROCESS_INSTANCE,
        "registration": registration_record,
        "schema": "grm.det1_7.worker_shard.v1",
        "spec": "e2e-4",
        "stage": "plant_registration",
        "status": "COMPLETE",
        "variants": ["served"],
    })
    return root, FindingEvidencePaths(
        order=order,
        base_registration=registration,
        predecessor_authorization=authorization,
        worker_shard=worker,
        chronological_session=chronological,
        observations=observations,
        probe_scorecard=scorecard,
        transcript=transcript,
        instrumentation=instrumentation,
        t30_snapshot_manifest=t30_manifest,
        t33_snapshot_manifest=t33_manifest,
    )


def test_authors_exact_ordered_content_addressed_open_finding(tmp_path: Path):
    root, evidence = _fixture(tmp_path)
    output_dir = root / "artifacts/run/det1_9/findings"
    path, receipt = author_finding_receipt(
        evidence,
        created_utc=CREATED_UTC,
        directory=output_dir,
        repo_root=root,
    )

    assert receipt["schema"] == SCHEMA
    assert receipt["status"] == STATUS
    assert [row["text"] for row in receipt["ordered_findings"]] == [
        T30_FINDING_TEXT,
        T33_FINDING_TEXT,
    ]
    assert receipt["ordered_findings"][0]["evidence_projection"][
        "route_and_mount_projection"
    ] == "[30]/[30]"
    assert receipt["scope"]["cause_investigated"] is False
    assert receipt["scope"]["causal_conclusion"] is None
    assert path.read_bytes() == canonical_json_bytes(receipt)
    assert path.parent == output_dir
    assert path.name.startswith("campaign_r4_lived_control_finding_")

    validation = validate_finding_receipt_file(path, repo_root=root)
    assert validation["status"] == "PASS_EXACT_OPEN_FINDING_NO_CAUSAL_CONCLUSION"
    assert validation["t30_probe_pass"] is False
    assert validation["t33_probe_pass"] is False


@pytest.mark.parametrize(
    ("fixture_kwargs", "message"),
    [
        ({"t30_source_rank": 2}, "rank-1"),
        ({"t33_pass": True}, "probe_pass=false"),
        ({"plant_mode": "planted_miss"}, "applied a plant"),
    ],
)
def test_fails_closed_when_ordered_finding_evidence_is_not_exact(
    tmp_path: Path, fixture_kwargs: dict, message: str
):
    root, evidence = _fixture(tmp_path, **fixture_kwargs)
    with pytest.raises(FindingReceiptError, match=message):
        build_finding_receipt(
            evidence,
            created_utc=CREATED_UTC,
            repo_root=root,
        )


def test_rejects_text_drift_and_causal_upgrade(tmp_path: Path):
    root, evidence = _fixture(tmp_path)
    receipt = build_finding_receipt(
        evidence,
        created_utc=CREATED_UTC,
        repo_root=root,
    )

    text_drift = deepcopy(receipt)
    text_drift["ordered_findings"][0]["text"] += " Cause identified."
    with pytest.raises(FindingReceiptError, match="ordered finding text"):
        validate_finding_receipt(text_drift, repo_root=root)

    causal_upgrade = deepcopy(receipt)
    causal_upgrade["scope"]["causal_conclusion"] = "routing caused the failure"
    causal_upgrade["scope"]["cause_investigated"] = True
    with pytest.raises(FindingReceiptError, match="causal upgrade"):
        validate_finding_receipt(causal_upgrade, repo_root=root)

