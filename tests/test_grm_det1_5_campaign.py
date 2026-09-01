from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import pytest

from scripts.grm_det1_5_analyze import prediction_verdict
from scripts.grm_det1_5_gpu import (
    _worker_gap_anchor_ns,
    build_lead_commands,
    merge_detector_rows,
    validate_cross_process_zero,
    validate_g0_rows,
    validate_g1_pair,
    validate_split,
    validate_verbal_chronology,
)
from scripts.grm_det1_common import DETECTORS, DETError, VERBAL_QUESTION


def _sha(character: str) -> str:
    return hashlib.sha256(character.encode("utf-8")).hexdigest()


def test_worker_gap_anchor_survives_timeout_without_worker_output(tmp_path: Path):
    run_dir = tmp_path / "run"
    attempt = (
        run_dir / "det1_4/campaign/g0/shards/e2e-1/attempt_001"
    )
    attempt.mkdir(parents=True)
    finished_ns = time.time_ns()
    (attempt / "attempt_finished.json").write_text(json.dumps({
        "schema": "grm.det1_5.worker_attempt_finished.v1",
        "status": "TIMED_OUT_AND_REAPED",
        "finished_unix_ns": finished_ns,
        "worker": "g0",
        "attempt_dir": str(attempt),
        "parent_process_instance": {
            "process_instance_sha256": _sha("parent"),
        },
        "returncode": None,
    }), encoding="utf-8")

    anchor = _worker_gap_anchor_ns(run_dir)
    assert anchor is not None and anchor >= finished_ns


def _zero_receipt(process: str, *, state: str = "s") -> dict:
    return {
        "schema": "grm.det1_4.zero_fork_gate.v1",
        "status": "PASS",
        "gate_pass": True,
        "intervention": "ZERO",
        "process_instance_sha256": _sha(process),
        "canonical_state_sha256": _sha(state),
        "lived_snapshot": {
            "path": "artifacts/grm_det1/run/det1_3/lived/manifest.json",
            "bytes": 1234,
            "sha256": _sha("l"),
        },
        "substrate_comparison": {
            "status": "PASS_EXACT_CANONICAL_VALUE_BYTES",
            "gate_pass": True,
        },
        "answer": {
            "normalized": "the current harbor token value is nacre-6-blue.",
            "lived_normalized": (
                "the current harbor token value is nacre-6-blue."
            ),
            "normalized_match": True,
        },
    }


def test_cross_process_zero_binds_same_canonical_state_in_distinct_processes():
    result = validate_cross_process_zero(
        _zero_receipt("a"), _zero_receipt("b"))

    assert result["schema"] == "grm.det1_5.cross_process_zero.v1"
    assert result["status"] == "PASS"
    assert result["gate_pass"] is True


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda child: child.update(process_instance_sha256=_sha("a")),
         "distinct process"),
        (lambda child: child.update(canonical_state_sha256=_sha("x")),
         "canonical"),
        (lambda child: child["lived_snapshot"].update(sha256=_sha("x")),
         "lived snapshot"),
        (lambda child: child["substrate_comparison"].update(status="DIVERGENT"),
         "PASS_EXACT_CANONICAL_VALUE_BYTES"),
        (lambda child: child["answer"].update(normalized_match=False),
         "answer"),
    ],
)
def test_cross_process_zero_rejects_provenance_or_value_drift(mutation, match):
    parent = _zero_receipt("a")
    child = _zero_receipt("b")
    mutation(child)

    with pytest.raises(DETError, match=match):
        validate_cross_process_zero(parent, child)


def _delta_receipt(field: str) -> dict:
    return {
        "schema": "grm.det1_6.fork_hydration_delta.v1",
        "status": "PASS_EXACT_REGISTERED_DELTA_CANONICAL_VALUE_BYTES",
        "gate_pass": True,
        "target_absence": {
            "gate_pass": True,
            "checks": {"registered_aliases_absent_from_fork_mounts": True},
        },
        "strict_zero_comparator": {
            "schema": "grm.det1_4.zero_fork_substrate_comparison.v1",
            "status": "DIVERGENT",
            "gate_pass": False,
            "counts": {"EQUAL": 10, "DIVERGENT": 1},
            "row_count": 11,
        },
        "exact_divergence_set": True,
        "non_delta_fields_equal": True,
        "retained_arrays_exact": True,
        "transformed_arrays": [{
            "field": field,
            "retained_bytes_exact": True,
        }],
        "unexpected_divergent_fields": [],
        "missing_expected_divergent_fields": [],
        "expected_fork_value_mismatches": [],
        "expected_divergent_fields": [field],
        "observed_non_equal_fields": [field],
    }


def _paired_rows(pair_count: int = 12) -> list[dict]:
    rows: list[dict] = []
    for index in range(pair_count):
        fixture = f"eval_{index:02d}"
        rows.extend([
            {
                "schema": "grm.det1_5.mechanistic_row.v1",
                "row_id": f"{fixture}:served",
                "fixture_id": fixture,
                "variant": "served",
                "answer_correct": True,
                "target_contains_expected": True,
                "router_rank1_admitted": True,
                "raw_router_rank1": index,
                "logical_router_rank1": index,
                "logical_alias_ids": [index],
                "mounted_ids": [index],
                "mounted_contains_expected": True,
                "full_index_contains_all_aliases": True,
            },
            {
                "schema": "grm.det1_5.mechanistic_row.v1",
                "row_id": f"{fixture}:planted_miss",
                "fixture_id": fixture,
                "variant": "planted_miss",
                "answer_correct": False,
                "target_contains_expected": True,
                "plant_checks": {
                    "withheld_aliases_absent": True,
                    "logical_target_absent": True,
                    "raw_router_rank1_absent": True,
                    "expected_value_absent_from_mounted_text": True,
                    "withheld_aliases_remain_in_full_detector_index": True,
                    "admission_ranking_unchanged": True,
                    "admission_branch_unchanged": True,
                    "only_aliases_removed_from_admission_plan": True,
                    "withheld_aliases_absent_from_every_ladder_attempt": True,
                },
                "fork_restore": {
                    "intervention": (
                        "REGISTERED_PLANTED_MISS_WITHHOLDING"
                    ),
                    "same_process_index_verified": True,
                    "field_deltas": [{
                        "field": "injection.layer_00.k",
                        "reason": "registered_planted_miss_withholding",
                    }],
                    "delta_receipt": _delta_receipt(
                        "arrays.injection.layer_00.k"),
                    "delta_fork_snapshot": {
                        "path": f"artifacts/delta/{fixture}/manifest.json",
                        "bytes": 1234,
                        "sha256": _sha(f"delta-{fixture}"),
                    },
                },
            },
        ])
    return rows


def test_det_g0_accepts_exactly_twelve_verified_pairs():
    result = validate_g0_rows(_paired_rows())

    assert result["schema"] == "grm.det1_5.g0.v1"
    assert result["status"] == "PASS"
    assert result["gate_pass"] is True
    assert result["counts"] == {
        "fixtures": 12,
        "served": 12,
        "planted_miss": 12,
    }
    assert result["fork_hydration_delta_receipt_count"] == 12
    assert result["observed_delta_field_count"] == 12


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda rows: rows[0].update(answer_correct=False), "served"),
        (lambda rows: rows[1]["plant_checks"].update(
            withheld_aliases_absent=False), "withheld_aliases_absent"),
        (lambda rows: rows[1]["fork_restore"].update(field_deltas=[]),
         "field_deltas"),
        (lambda rows: rows[1]["fork_restore"].update(
            same_process_index_verified=False), "same_process_index"),
        (lambda rows: rows[1]["fork_restore"]["delta_receipt"].update(
            schema="wrong"), "schema"),
        (lambda rows: rows[1]["fork_restore"]["delta_receipt"].update(
            status="FAIL_CLOSED"), "status"),
        (lambda rows: rows[1]["fork_restore"]["delta_receipt"].update(
            gate_pass=False), "gate_pass"),
        (lambda rows: rows[1]["fork_restore"]["delta_receipt"][
            "target_absence"].update(gate_pass=False), "target absence"),
        (lambda rows: rows[1]["fork_restore"]["delta_receipt"][
            "target_absence"]["checks"].update(
                registered_aliases_absent_from_fork_mounts=False),
         "target-absence checks"),
        (lambda rows: rows[1]["fork_restore"]["delta_receipt"][
            "strict_zero_comparator"].update(status="PASS"), "strict ZERO"),
        (lambda rows: rows[1]["fork_restore"]["delta_receipt"].update(
            exact_divergence_set=False), "exact divergence"),
        (lambda rows: rows[1]["fork_restore"]["delta_receipt"].update(
            non_delta_fields_equal=False), "non-delta equality"),
        (lambda rows: rows[1]["fork_restore"]["delta_receipt"].update(
            retained_arrays_exact=False), "retained arrays"),
        (lambda rows: rows[1]["fork_restore"]["delta_receipt"][
            "transformed_arrays"][0].update(retained_bytes_exact=False),
         "transformed arrays"),
        (lambda rows: rows[1]["fork_restore"]["delta_receipt"].update(
            unexpected_divergent_fields=["state.arena.pos"]),
         "unexpected_divergent_fields"),
        (lambda rows: rows[1]["fork_restore"]["delta_receipt"].update(
            missing_expected_divergent_fields=["state.arena.cur_mounts"]),
         "missing_expected_divergent_fields"),
        (lambda rows: rows[1]["fork_restore"]["delta_receipt"].update(
            expected_fork_value_mismatches=["arrays.injection.layer_00.k"]),
         "expected_fork_value_mismatches"),
        (lambda rows: rows[1]["fork_restore"]["delta_receipt"].update(
            observed_non_equal_fields=["state.arena.pos"]),
         "observed divergence set"),
        (lambda rows: rows[1]["fork_restore"].pop("delta_fork_snapshot"),
         "delta_fork_snapshot"),
        (lambda rows: rows.pop(), "12"),
    ],
)
def test_det_g0_rejects_an_ineffective_or_unpaired_plant(mutate, match):
    rows = _paired_rows()
    mutate(rows)

    with pytest.raises(DETError, match=match):
        validate_g0_rows(rows)


def _byte_record(name: str, digest: str) -> dict:
    return {
        "path": f"artifacts/grm_det1/run/{name}",
        "bytes": 100 + len(name),
        "sha256": _sha(digest),
    }


def _g1_leg(*, hooked: bool) -> dict:
    return {
        "process_instance_sha256": _sha("h" if hooked else "c"),
        "active_hooks": (
            ["D-LQR", "D-NGH", "D-ENT"] if hooked else []
        ),
        "d_verb_isolated": True,
        "byte_records": {
            "transcript": _byte_record("transcript.jsonl", "t"),
            "scorecard": _byte_record("scorecard.json", "s"),
            "served_projection": _byte_record("served_projection.json", "p"),
        },
    }


def test_det_g1_requires_all_mechanistic_hooks_and_exact_served_bytes():
    result = validate_g1_pair(_g1_leg(hooked=False), _g1_leg(hooked=True))

    assert result["schema"] == "grm.det1_5.g1.v1"
    assert result["status"] == "PASS"
    assert result["gate_pass"] is True
    assert set(result["byte_identity"]) == {
        "transcript", "scorecard", "served_projection",
    }
    assert all(result["byte_identity"].values())


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda hooked: hooked.update(active_hooks=["D-LQR", "D-NGH"]),
         "active hooks"),
        (lambda hooked: hooked.update(d_verb_isolated=False), "D-VERB"),
        (lambda hooked: hooked["byte_records"]["transcript"].update(
            sha256=_sha("x")), "transcript"),
        (lambda hooked: hooked["byte_records"].pop("scorecard"),
         "scorecard"),
    ],
)
def test_det_g1_rejects_partial_hooks_or_any_byte_drift(mutate, match):
    control = _g1_leg(hooked=False)
    hooked = _g1_leg(hooked=True)
    mutate(hooked)

    with pytest.raises(DETError, match=match):
        validate_g1_pair(control, hooked)


def _registration() -> dict:
    return {
        "schema": "grm.det1.registration.v1",
        "registered_pre_data": True,
        "split_rule": {
            "calibration": ["cal_00", "cal_01"],
            "calibration_pair_count": 2,
            "eval_e2e": [f"eval_{index:02d}" for index in range(7)],
            "eval_supersession": [
                f"eval_{index:02d}" for index in range(7, 12)
            ],
            "eval_pair_count": 12,
            "query_and_certified_fact_disjoint": True,
            "unit": "underlying probe; paired variants never cross splits",
        },
    }


def _calibration_rows() -> list[dict]:
    return [
        {
            "row_id": f"cal_{index:02d}:served",
            "fixture_id": f"cal_{index:02d}",
            "variant": "served",
        }
        for index in range(2)
    ]


def test_calibration_and_eval_are_exact_registered_disjoint_splits():
    result = validate_split(
        _registration(), _calibration_rows(), _paired_rows())

    assert result["schema"] == "grm.det1_5.calibration.v1"
    assert result["status"] == "PASS"
    assert result["calibration_count"] == 2
    assert result["eval_counts"] == {"served": 12, "planted_miss": 12}


@pytest.mark.parametrize(
    ("registration_mutation", "calibration_mutation", "eval_mutation", "match"),
    [
        (lambda value: None, lambda rows: rows.pop(), lambda rows: None, "2"),
        (lambda value: None,
         lambda rows: rows[0].update(variant="planted_miss"),
         lambda rows: None, "served"),
        (lambda value: None, lambda rows: None,
         lambda rows: rows[0].update(fixture_id="cal_00"), "disjoint"),
        (lambda value: value["split_rule"].update(eval_pair_count=11),
         lambda rows: None, lambda rows: None, "12"),
    ],
)
def test_split_validation_rejects_leakage_or_cardinality_drift(
    registration_mutation, calibration_mutation, eval_mutation, match,
):
    registration = _registration()
    calibration = _calibration_rows()
    evaluation = _paired_rows()
    registration_mutation(registration)
    calibration_mutation(calibration)
    eval_mutation(evaluation)

    with pytest.raises(DETError, match=match):
        validate_split(registration, calibration, evaluation)


def _mechanistic_rows() -> list[dict]:
    rows = _paired_rows()
    for ordinal, row in enumerate(rows):
        row["signals"] = {"tokens": [], "ordinal": ordinal}
        row["mechanistic_completed_unix_ns"] = 1_000 + ordinal * 10
    return rows


def _verbal_rows() -> list[dict]:
    rows = []
    for ordinal, mechanistic in enumerate(_mechanistic_rows()):
        rows.append({
            "schema": "grm.det1_5.verbal_row.v1",
            "row_id": mechanistic["row_id"],
            "fixture_id": mechanistic["fixture_id"],
            "variant": mechanistic["variant"],
            "prompt_suffix_exact": VERBAL_QUESTION,
            "mechanistic_completed_unix_ns": (
                mechanistic["mechanistic_completed_unix_ns"]
            ),
            "verbal_started_unix_ns": (
                mechanistic["mechanistic_completed_unix_ns"] + 1
            ),
            "verbal": {
                "answer": "NO",
                "trigger": True,
                "latency_token_index": 0,
            },
        })
    return rows


def _verbal_receipt() -> dict:
    return {
        "schema": "grm.det1_5.chronological_verbal.v1",
        "status": "PASS",
        "process_instance_sha256": _sha("v"),
        "mechanistic_process_instance_sha256": _sha("m"),
        "verbal_question_exact": VERBAL_QUESTION,
        "rows": _verbal_rows(),
    }


def test_d_verb_is_a_distinct_later_chronological_lived_process():
    result = validate_verbal_chronology(_verbal_receipt())

    assert result["schema"] == "grm.det1_5.chronological_verbal.v1"
    assert result["status"] == "PASS"
    assert result["row_count"] == 24
    assert result["processes_distinct"] is True


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda receipt: receipt.update(
            process_instance_sha256=receipt[
                "mechanistic_process_instance_sha256"]), "distinct"),
        (lambda receipt: receipt.update(
            verbal_question_exact="Do you know?"), "verbal question"),
        (lambda receipt: receipt["rows"][0].update(
            verbal_started_unix_ns=receipt["rows"][0][
                "mechanistic_completed_unix_ns"]), "after"),
        (lambda receipt: receipt["rows"].pop(), "24"),
    ],
)
def test_d_verb_chronology_fails_closed_on_leak_or_reordering(mutate, match):
    receipt = _verbal_receipt()
    mutate(receipt)

    with pytest.raises(DETError, match=match):
        validate_verbal_chronology(receipt)


def test_mechanistic_and_verbal_rows_join_one_to_one_without_signal_loss():
    mechanistic = _mechanistic_rows()
    verbal = _verbal_rows()
    merged = merge_detector_rows(mechanistic, verbal)

    assert len(merged) == 24
    assert [row["row_id"] for row in merged] == [
        row["row_id"] for row in mechanistic
    ]
    assert [row["signals"] for row in merged] == [
        row["signals"] for row in mechanistic
    ]
    assert all(row["verbal"]["trigger"] is True for row in merged)


def test_detector_row_join_rejects_a_missing_or_mislabelled_verbal_row():
    mechanistic = _mechanistic_rows()
    missing = _verbal_rows()[:-1]
    with pytest.raises(DETError, match="row keys"):
        merge_detector_rows(mechanistic, missing)

    mismatched = _verbal_rows()
    mismatched[0]["variant"] = "planted_miss"
    with pytest.raises(DETError, match="identity"):
        merge_detector_rows(mechanistic, mismatched)


def _metric_row(detector: str, f1: float, viable: bool) -> dict:
    return {
        "detector": detector,
        "tp": 12 if f1 else 0,
        "positive_n": 12,
        "fp": 0,
        "negative_n": 12,
        "recall": f1,
        "false_positive_rate": 0.0,
        "precision": f1,
        "f1": f1,
        "median_trigger_latency_token": 0.0 if f1 else None,
        "viable": viable,
    }


@pytest.mark.parametrize(
    ("scores", "viability", "expected"),
    [
        ((0.95, 0.90, 0.85, 0.50), (True, True, True, False), "SUPPORTED"),
        ((0.80, 0.95, 0.70, 0.50), (True, True, False, False), "PARTIAL"),
        ((0.95, 0.80, 0.70, 0.95), (True, True, False, True), "REFUTED"),
        # Raw F1 equality refutes even when D-VERB itself is not viable.
        ((0.95, 0.80, 0.70, 0.95), (True, True, False, False), "REFUTED"),
    ],
)
def test_prediction_verdict_uses_only_registered_vocabulary(
    scores, viability, expected,
):
    table = [
        _metric_row(name, score, viable)
        for name, score, viable in zip(DETECTORS, scores, viability, strict=True)
    ]

    assert prediction_verdict(table) == expected


def test_prediction_verdict_rejects_the_registered_uncovered_corner():
    # D-LQR wins, but another viable mechanism trails D-VERB.  DET1 explicitly
    # left this corner unadjudicated; DET1.5 must fail rather than mint a label.
    table = [
        _metric_row("D-LQR", 0.95, True),
        _metric_row("D-NGH", 0.40, True),
        _metric_row("D-ENT", 0.30, False),
        _metric_row("D-VERB", 0.50, False),
    ]

    with pytest.raises(DETError, match="adjudication gap"):
        prediction_verdict(table)


def test_prediction_verdict_rejects_missing_duplicate_or_reordered_detectors():
    table = [
        _metric_row(name, 0.9 - index / 10, name != "D-VERB")
        for index, name in enumerate(DETECTORS)
    ]
    for malformed in (
        table[:-1],
        [*table[:-1], deepcopy(table[-2])],
        [table[1], table[0], *table[2:]],
    ):
        with pytest.raises(DETError, match="four rows|detector"):
            prediction_verdict(malformed)


def test_gpu_lead_plan_is_argument_vectorized_and_carries_lease_discipline(
    tmp_path: Path,
):
    commands = build_lead_commands(tmp_path, 580, 7200)

    assert commands
    assert all(isinstance(command, list) and command for command in commands)
    assert all(all(isinstance(part, str) for part in command)
               for command in commands)
    flattened = [" ".join(command) for command in commands]
    for stage in (
        "cross-process-zero", "g0", "g1", "calibration",
        "eval-mechanistic", "eval-verbal", "analyze",
    ):
        assert any(stage in command for command in flattened), stage
    for command in commands[:-1]:
        assert command[command.index("--lease-seconds") + 1] == "580"
        assert command[command.index("--lock-wait-seconds") + 1] == "7200"
    assert "grm_det1_5_analyze.py" in flattened[-1]
