from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from scripts.grm_det1_5_analyze import (
    _process_instances as analyzer_process_instances,
    prediction_verdict,
)
from scripts.grm_det1_5_gpu import (
    _qualify_registration_target,
    _select_registration_observations,
    _worker_gap_anchor_ns,
    bind_plant_profile,
    build_lead_commands,
    effective_fixture_for_base,
    effective_registration_projection,
    merge_detector_rows,
    selftest,
    validate_cross_process_zero,
    validate_g0_rows,
    validate_g1_pair,
    validate_split,
    validate_verbal_chronology,
)
from scripts.grm_det1_common import DETECTORS, DETError, VERBAL_QUESTION


def _sha(character: str) -> str:
    return hashlib.sha256(character.encode("utf-8")).hexdigest()


def test_campaign_cpu_selftest_includes_det1_8_member_gate():
    receipt = selftest()

    assert receipt["status"] == "PASS_CPU_ONLY"
    assert receipt["checks"]["g0_12_plus_12"] == "PASS"
    assert receipt["gpu_allocations_attempted"] == 0


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


def test_registration_receipt_labels_ordered_process_aggregate_semantics():
    process_ids = [_sha("registration-a"), _sha("registration-b")]
    aggregate = hashlib.sha256(
        json.dumps(
            process_ids,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    receipt = {
        "process_instance_sha256s": process_ids,
        "process_instance_sha256": aggregate,
        "process_instance_identity": (
            "AGGREGATE_ORDERED_PROCESS_LIST_NOT_OS_IDENTITY"
        ),
        "process_identity_semantics": (
            "SHA256_OF_ORDERED_OS_PROCESS_INSTANCE_LIST"
        ),
    }

    assert analyzer_process_instances(receipt, "plant_registration") == (
        process_ids
    )


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
        "member_snapshot_coverage": {
            "gate_pass": True,
            "selected_member_payload_absent": True,
            "all_other_member_payload_bytes_exact": True,
            "lived": {"gate_pass": True},
            "fork": {"gate_pass": True},
        },
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
    registry_record = {
        "path": "artifacts/grm_det1/run/plant_registry_deadbeef.json",
        "bytes": 1234,
        "sha256": "a" * 64,
    }
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
                "plant_target_id": index,
                "plant_alias_ids": [index],
                "plant_target_source": "DET1_7_LIVED_ADMISSION_REGISTRY",
                "plant_registry": registry_record,
                "plant_entry_sha256": _sha(f"entry-{index}"),
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
                "plant_target_id": index,
                "plant_alias_ids": [index],
                "plant_target_source": "DET1_7_LIVED_ADMISSION_REGISTRY",
                "plant_registry": registry_record,
                "plant_entry_sha256": _sha(f"entry-{index}"),
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


def test_lived_admission_target_overrides_stale_raw_rank_one_diagnostic():
    entry = {
        "selected_target_id": 1,
        "alias_ids": [1],
        "actual_authoritative_mounts": [1],
        "expected_value_coverage": {"1": True},
    }
    registry_record = {
        "path": "artifacts/grm_det1/run/plant_registry_feedface.json",
        "bytes": 321,
        "sha256": "f" * 64,
    }

    bound = bind_plant_profile(
        {
            "raw_router_rank1": 2,
            "logical_router_rank1": 2,
            "logical_alias_ids": [2],
        },
        entry,
        registry_record,
    )

    # Solace's lived A-DEC mount was graft 1 even though the raw/rank-plan
    # diagnostic began at graft 2.  The plant must withhold the lived seat.
    assert bound["raw_router_rank1"] == 2
    assert bound["logical_router_rank1"] == 1
    assert bound["logical_alias_ids"] == [1]
    assert bound["plant_target_source"] == (
        "DET1_7_LIVED_ADMISSION_REGISTRY"
    )
    assert bound["plant_registry"] == registry_record


def test_registration_selects_only_unique_lawful_same_session_reserve():
    fixture_ids = [f"slot_{index:02d}" for index in range(14)]
    registration = {"fixtures": [
        {"fixture_id": fixture_id} for fixture_id in fixture_ids
    ]}
    candidates = [
        {
            "schema": "grm.det1_7.plant_observation.v1",
            "row_id": f"{fixture_id}:plant_registration",
            "fixture_id": fixture_id,
            "candidate_for_fixture_id": fixture_id,
            "effective_fixture_id": fixture_id,
            "status": (
                "UNPLANTABLE" if fixture_id == "slot_12"
                else "LAWFUL_LIVED_TARGET"
            ),
            "unplantable_reason": (
                "served answer incorrect" if fixture_id == "slot_12"
                else None
            ),
        }
        for fixture_id in fixture_ids
    ]
    candidates.append({
        "schema": "grm.det1_7.plant_observation.v1",
        "row_id": "polaris:plant_registration",
        "fixture_id": "polaris",
        "candidate_for_fixture_id": "slot_12",
        "effective_fixture_id": "polaris",
        "status": "LAWFUL_LIVED_TARGET",
        "substitution": {
            "status": "SUBSTITUTED",
            "reason": "primary served control failed",
        },
    })

    selected, substitutions = _select_registration_observations(
        registration, candidates)

    assert len(selected) == 14
    assert selected[12]["fixture_id"] == "slot_12"
    assert selected[12]["effective_fixture_id"] == "polaris"
    assert substitutions == [{
        "base_fixture_id": "slot_12",
        "effective_fixture_id": "polaris",
        "reason": "primary served control failed",
        "same_certified_session": True,
    }]


def test_registration_ablation_uses_fork_protocol_and_member_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    import scripts.grm_det1_3_snapshot as snapshot_module

    source_path = tmp_path / "lived_manifest.json"
    source_path.write_text("{}\n", encoding="utf-8")
    fork_path = tmp_path / "fork_manifest.json"
    source = {
        "state": {"admission.authoritative_mounts": [10, 20]},
        "identity": {"frame_sha256": _sha("frame")},
        "provenance": {
            "schema": "grm.det1_3.capture_provenance.v2",
            "arm": "lived",
            "protocol": snapshot_module.ARM_PROTOCOLS["lived"],
        },
        "manifest_payload_sha256": _sha("source-manifest"),
        "linked_answer": {"probe_answer": "The value is right."},
    }
    captured_provenance: dict = {}

    def fake_load(path: Path) -> dict:
        return source if Path(path) == source_path else {
            "state": {"admission.authoritative_mounts": [20]},
        }

    def fake_coverage(value: dict, _label: str) -> dict:
        return {
            "gate_pass": True,
            "mounted_members": list(
                value["state"]["admission.authoritative_mounts"]),
        }

    def fake_restore(arena, _path, **_kwargs):
        arena.cur_mounts = [20]
        return {
            "prompt_ids": [1, 2],
            "admission_state": {"final_mounts": [20]},
            "live_token_ids": [],
            "sink_token_ids": [3],
        }

    def fake_capture(_arena, _directory, **kwargs):
        captured_provenance.update(kwargs["provenance"])
        return fork_path

    delta = _delta_receipt("arrays.cache.layer_00.k")
    delta.update({
        "source_mounts": [10, 20],
        "fork_mounts": [20],
        "withheld_logical_aliases": [10],
        "source_mounted_aliases": [10],
        "already_absent_aliases": [],
        "member_snapshot_coverage": {
            "gate_pass": True,
            "selected_member_payload_absent": True,
            "all_other_member_payload_bytes_exact": True,
        },
    })
    monkeypatch.setattr(snapshot_module, "load_snapshot", fake_load)
    monkeypatch.setattr(
        snapshot_module, "require_snapshot_member_coverage", fake_coverage)
    monkeypatch.setattr(snapshot_module, "restore_prefill_fork", fake_restore)
    monkeypatch.setattr(snapshot_module, "capture_arena_snapshot", fake_capture)
    monkeypatch.setattr(
        snapshot_module, "compare_fork_hydration_delta",
        lambda *_args, **_kwargs: delta,
    )
    monkeypatch.setitem(
        sys.modules,
        "scripts.grm_det1_4_gpu",
        SimpleNamespace(
            _continue_forked_prefill=(
                lambda *_args, **_kwargs: ("obsolete", {})
            ),
        ),
    )
    arena = SimpleNamespace(
        cur_mounts=[10, 20],
        encode=lambda _text: [3],
    )

    breaker = _qualify_registration_target(
        arena=arena,
        e2e=SimpleNamespace(HARMONY_SINK="<sink>"),
        fixture={
            "fixture_id": "declared_synthesis",
            "expected_values": ["right"],
            "wrong_fact_values": ["obsolete"],
        },
        result={"snapshot": {"path": str(source_path)}},
        target_id=10,
        identity={"frame_sha256": _sha("frame")},
        process={"process_instance_sha256": _sha("process")},
        capture_prompt="What is the value?",
        ngen=1,
    )

    assert captured_provenance["arm"] == "fork"
    assert captured_provenance["protocol"] == snapshot_module.ARM_PROTOCOLS["fork"]
    assert breaker["withheld_target_id"] == 10
    assert breaker["counterfactual"]["mounted_ids"] == [20]
    assert breaker["counterfactual"]["classification"] == "wrong_fact"
    assert breaker["delta_receipt"]["member_snapshot_coverage"][
        "all_other_member_payload_bytes_exact"] is True


def test_effective_registration_projection_preserves_2_plus_12_split():
    registration = _registration()
    base_ids = [
        *registration["split_rule"]["calibration"],
        *registration["split_rule"]["eval_e2e"],
        *registration["split_rule"]["eval_supersession"],
    ]
    registry = {"entries": [
        {
            "fixture_id": fixture_id,
            "effective_fixture_id": (
                "e2e_t33_polaris_mark"
                if fixture_id == "eval_11" else fixture_id
            ),
        }
        for fixture_id in base_ids
    ]}

    projected = effective_registration_projection(registration, registry)
    calibration, evaluation = _registered_split_for_test(projected)

    assert calibration == ["cal_00", "cal_01"]
    assert len(evaluation) == 12
    assert evaluation[-1] == "e2e_t33_polaris_mark"
    assert "eval_11" not in evaluation


@pytest.mark.parametrize(
    ("source_family", "selector", "runtime_key", "runtime_value"),
    [
        ("certified_34_turn", {"turn": 33}, "turn", 33),
        (
            "supersession_battery_on_gpt_oss",
            {"probe_id": "reserve-probe"},
            "probe_id",
            "reserve-probe",
        ),
    ],
)
def test_substituted_selector_projects_to_worker_runtime_identity(
    source_family, selector, runtime_key, runtime_value,
):
    base = {"fixture_id": "base-slot"}
    effective = {
        "fixture_id": "reserve-fixture",
        "source_family": source_family,
        "selector": selector,
    }
    registry = {"entries": [{
        "fixture_id": "base-slot",
        "effective_fixture_id": "reserve-fixture",
        "effective_fixture": effective,
        "substitution": {"status": "SUBSTITUTED"},
    }]}

    projected = effective_fixture_for_base(registry, base)

    assert projected[runtime_key] == runtime_value
    assert projected["base_fixture_id"] == "base-slot"
    assert projected["fixture_id"] == "reserve-fixture"


def _registered_split_for_test(registration: dict) -> tuple[list[str], list[str]]:
    split = registration["split_rule"]
    return (
        list(split["calibration"]),
        [*split["eval_e2e"], *split["eval_supersession"]],
    )


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
        "cross-process-zero", "author-det1-7-source", "plant-registration",
        "author-det1-7-terminal", "g0", "g1", "calibration",
        "eval-mechanistic", "eval-verbal", "analyze",
    ):
        assert any(stage in command for command in flattened), stage
    gpu_commands = [
        command for command in commands[:-1] if "--lease-seconds" in command
    ]
    assert gpu_commands
    for command in gpu_commands:
        assert command[command.index("--lease-seconds") + 1] == "580"
        assert command[command.index("--lock-wait-seconds") + 1] == "7200"
    assert "grm_det1_5_analyze.py" in flattened[-1]
