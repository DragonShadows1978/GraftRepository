from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.grm_det1_7_registry import (
    RegistryError,
    canonical_json_bytes,
    canonical_sha256,
    derive_plant_registry,
    validate_plant_registry,
    write_content_addressed_registry,
)


EVIDENCE_UTC = "2026-08-31T20:00:00Z"
CREATED_UTC = "2026-09-01T00:00:00Z"
EVAL_UTC = "2026-09-01T01:00:00Z"


def _base_registration() -> dict:
    fixtures = []
    for index in range(14):
        fixture_id = f"fixture_{index:02d}"
        family = "certified_34_turn" if index < 9 else "synthetic_supersession"
        fixture = {
            "fixture_id": fixture_id,
            "source_family": family,
            "split": "calibration" if index < 2 else "eval",
            "question": f"question {index}",
            "expected_values": [f"value-{index}"],
            "source": {
                "path": "fixture_source.json",
                "bytes": 1,
                "sha256": "a" * 64 if index < 9 else chr(98 + index % 5) * 64,
            },
        }
        if family == "certified_34_turn":
            fixture["turn"] = index * 2 + 1
        else:
            fixture["session_id"] = f"session_{index}"
            fixture["probe_id"] = f"probe_{index}"
        fixtures.append(fixture)
    calibration = [row["fixture_id"] for row in fixtures[:2]]
    evaluation = [row["fixture_id"] for row in fixtures[2:]]
    return {
        "schema": "grm.det1.registration.v1",
        "order": "GRM-DET1",
        "fixtures": fixtures,
        "split_rule": {
            "calibration": calibration,
            "calibration_pair_count": 2,
            "eval_pair_count": 12,
            "eval_e2e": evaluation[:7],
            "eval_supersession": evaluation[7:],
        },
        "detectors": {
            "D-LQR": {"trigger": "lqr", "threshold_fit": "none"},
            "D-NGH": {"trigger": "ngh", "threshold_fit": "minimum"},
            "D-ENT": {"trigger": "ent", "threshold_fit": "envelope"},
            "D-VERB": {"trigger": "whole-word NO"},
        },
        "prediction_adjudication": {
            "PARTIAL": "p",
            "REFUTED": "r",
            "SUPPORTED": "s",
            "UNADJUDICATED": "u",
        },
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sign_snapshot(manifest: dict) -> None:
    manifest.pop("manifest_payload_sha256", None)
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)


def _snapshot(
    fixture_id: str,
    registration_sha256: str,
    *,
    actual: list[int],
    rank_plan: list[int],
    current_planned: list[int] | None = None,
    ranking: list[int] | None = None,
    identified: list[int] | None = None,
    branch: str = "exactly_one_identifier_decisive_rank1",
    answer: str = "registered-value",
) -> dict:
    current_planned = current_planned or list(rank_plan)
    ranking = ranking or list(rank_plan)
    identified = identified if identified is not None else list(actual)
    process = "d" * 64
    manifest = {
        "schema": "grm.det1_3.model_visible_snapshot.v2",
        "label": "lived",
        "phase": "before_probe_prefill",
        "complete": True,
        "capture_finalized": True,
        "provenance": {
            "arm": "lived",
            "capture_boundary": "before_probe_prefill",
            "fixture_id": fixture_id,
            "registration_sha256": registration_sha256,
            "process_instance_sha256": process,
            "attempt_ordinal": 0,
        },
        "state": {
            "arena.cur_mounts": list(actual),
            "admission.authoritative_mounts": list(actual),
            "admission.final_mounts": list(actual),
            "admission.rank_plan": list(rank_plan),
            "admission.current_planned": list(current_planned),
            "admission.ranking": list(ranking),
            "admission.identified_candidates": list(identified),
            "admission.policy_branch": branch,
            "admission.full_plan": {
                "final_mounts": list(actual),
                "rank_plan": list(rank_plan),
                "current_planned": list(current_planned),
                "ranking": list(ranking),
                "identified_candidates": list(identified),
                "policy_branch": branch,
            },
        },
        "linked_answer": {
            "process_instance_sha256": process,
            "captured_attempt_ordinal": 0,
            "attempt_answer": answer,
            "attempt_mounts": list(actual),
            "attempt_answer_correct": True,
            "attempt_refusal": False,
            "probe_answer": answer,
            "probe_selected_attempt": 0,
            "probe_mounts": list(actual),
            "probe_answer_correct": True,
            "probe_refusal": False,
        },
    }
    _sign_snapshot(manifest)
    return manifest


def _delta_receipt(aliases: list[int], remaining: list[int]) -> dict:
    return {
        "schema": "grm.det1_6.fork_hydration_delta.v1",
        "status": "PASS_EXACT_REGISTERED_DELTA_CANONICAL_VALUE_BYTES",
        "gate_pass": True,
        "exact_divergence_set": True,
        "non_delta_fields_equal": True,
        "retained_arrays_exact": True,
        "already_absent_aliases": [],
        "unexpected_divergent_fields": [],
        "missing_expected_divergent_fields": [],
        "expected_fork_value_mismatches": [],
        "withheld_logical_aliases": list(aliases),
        "source_mounted_aliases": list(aliases),
        "fork_mounts": list(remaining),
        "target_absence": {
            "gate_pass": True,
            "checks": {
                "at_least_one_target_had_lived_seats": True,
                "registered_aliases_absent_from_fork_mounts": True,
                "fork_mounts_equal_source_minus_target": True,
            },
        },
    }


def _case(tmp_path: Path) -> tuple[Path, Path, dict[str, dict]]:
    order_path = tmp_path / "orders/GRM_DET1_7_PLANT_REALIGN.md"
    order_path.parent.mkdir(parents=True)
    order_path.write_text("# ORDER GRM-DET1.7\n", encoding="utf-8")
    order_path.with_name("GRM_DET1_9_SUBSTITUTION_POOL.md").write_text(
        "# ORDER GRM-DET1.9\n", encoding="utf-8"
    )
    order_path.with_name("GRM_DET1_11_ACHIEVED_COUNT.md").write_text(
        "# ORDER GRM-DET1.11\n", encoding="utf-8"
    )
    registration_path = tmp_path / "registration.json"
    registration = _base_registration()
    _write_json(registration_path, registration)
    registration_sha = __import__("hashlib").sha256(
        registration_path.read_bytes()
    ).hexdigest()
    observations: dict[str, dict] = {}
    for index, fixture in enumerate(registration["fixtures"]):
        fixture_id = fixture["fixture_id"]
        actual = [100 + index]
        rank_plan = list(actual)
        branch = "exactly_one_identifier_decisive_rank1"
        identified = list(actual)
        coverage = {}
        if index == 0:
            # Non-declared insurance branches select the first *actual* mount,
            # not raw ranking/rank_plan[0].
            rank_plan = [999, actual[0]]
            branch = "one_off_rank_identifier_insurance_k3"
        if index == 1:
            actual = [101, 201]
            rank_plan = list(actual)
            identified = list(actual)
            branch = "declared_synthesis_identified_set"
            coverage = {"101": [fixture["expected_values"][0]], "201": []}
        manifest = _snapshot(
            fixture_id,
            registration_sha,
            actual=actual,
            rank_plan=rank_plan,
            ranking=rank_plan,
            identified=identified,
            branch=branch,
            answer=fixture["expected_values"][0],
        )
        snapshot_path = tmp_path / "snapshots" / fixture_id / "manifest.json"
        _write_json(snapshot_path, manifest)
        observations[fixture_id] = {
            "snapshot_path": snapshot_path,
            "evidence_utc": EVIDENCE_UTC,
            "expected_value_coverage": coverage,
            "substitution": {"status": "NONE"},
        }
        if branch == "declared_synthesis_identified_set":
            receipt = _delta_receipt([101], [201])
            observations[fixture_id]["behavioral_breaker"] = {
                "schema": "grm.det1_7.target_only_behavioral_breaker.v1",
                "source_manifest_payload_sha256": manifest[
                    "manifest_payload_sha256"
                ],
                "withheld_target_id": 101,
                "withheld_alias_ids": [101],
                "served": {
                    "answer": fixture["expected_values"][0],
                    "answer_correct": True,
                    "refusal": False,
                    "mounted_ids": [101, 201],
                },
                "counterfactual": {
                    "answer": "information unavailable",
                    "answer_correct": False,
                    "refusal": False,
                    "mounted_ids": [201],
                    "classification": "incorrect",
                },
                "delta_receipt": receipt,
            }
    return registration_path, order_path, observations


def _derive(tmp_path: Path, **kwargs) -> dict:
    registration, order, observations = _case(tmp_path)
    return derive_plant_registry(
        registration,
        order,
        observations,
        created_utc=CREATED_UTC,
        new_eval_evidence_utc=kwargs.pop("new_eval_evidence_utc", [EVAL_UTC]),
        record_root=tmp_path,
        **kwargs,
    )


def _resign_registry(registry: dict) -> None:
    registry.pop("registry_payload_sha256", None)
    registry["registry_payload_sha256"] = canonical_sha256(registry)


def test_derives_exact_base_slots_and_lived_targets_then_validates(tmp_path: Path):
    registry = _derive(tmp_path)

    assert registry["fixture_count"] == 14
    assert registry["pair_counts"]["eval_served_rows"] == 12
    assert registry["pair_counts"]["eval_planted_miss_rows"] == 12
    assert registry["entries"][0]["rank_plan"] == [999, 100]
    assert registry["entries"][0]["selected_target_id"] == 100
    assert registry["entries"][0]["alias_ids"] == [100]
    assert registry["entries"][1]["selected_target_id"] == 101
    assert registry["entries"][1]["rule_id"].startswith("LIVED_UNIQUE_ESSENTIAL")
    assert registry["entries"][1]["behavioral_breaker"]["counterfactual"][
        "classification"
    ] == "incorrect"
    assert all(row["substitution"] == {"status": "NONE"} for row in registry["entries"])

    receipt = validate_plant_registry(registry, record_root=tmp_path)
    assert receipt == {
        "schema": "grm.det1_7.plant_registry_validation.v1",
        "status": "PASS",
        "registry_payload_sha256": registry["registry_payload_sha256"],
        "fixture_count": 14,
        "calibration_count": 2,
        "eval_count": 12,
        "eval_served_count": 12,
        "eval_planted_count": 12,
        # DET1.11: a full-population run reports achieved == registered.
        "achieved_eval_pairs": 12,
        "registered_eval_pairs": 12,
        "excluded_eval_pairs": 0,
        "achieved_eval_pair_floor": 8,
        "achieved_of_registered": "12 of 12",
        "excluded_fixture_ids": [],
        "substitution_count": 0,
    }


def test_content_addressed_writer_is_deterministic_and_non_overwriting(tmp_path: Path):
    registry = _derive(tmp_path)
    first = write_content_addressed_registry(
        tmp_path / "out", registry, record_root=tmp_path
    )
    second = write_content_addressed_registry(
        tmp_path / "out", deepcopy(registry), record_root=tmp_path
    )

    assert first == second
    assert first.read_bytes() == canonical_json_bytes(registry) + b"\n"


@pytest.mark.parametrize(
    ("coverage", "match"),
    [
        ({"101": [], "201": []}, "lacks mounted expected-value coverage"),
        ({"101": ["value-1"], "201": ["value-1"]}, "exactly one"),
    ],
)
def test_declared_synthesis_none_or_tied_essential_carrier_fails_closed(
    tmp_path: Path, coverage: dict, match: str
):
    registration, order, observations = _case(tmp_path)
    observations["fixture_01"]["expected_value_coverage"] = coverage

    with pytest.raises(RegistryError, match=match):
        derive_plant_registry(
            registration,
            order,
            observations,
            created_utc=CREATED_UTC,
            record_root=tmp_path,
        )


def test_aliases_may_name_only_lived_mounted_seats(tmp_path: Path):
    registration, order, observations = _case(tmp_path)
    observations["fixture_00"]["alias_ids"] = [100, 999]

    with pytest.raises(RegistryError, match="without lived mounted seats"):
        derive_plant_registry(
            registration,
            order,
            observations,
            created_utc=CREATED_UTC,
            record_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda m: m.update(complete=False), "incomplete"),
        (lambda m: m.update(capture_finalized=False), "not finalized"),
        (lambda m: m.update(label="replay"), "not lived"),
        (lambda m: m.update(phase="after_probe"), "not lived"),
        (
            lambda m: m["linked_answer"].update(probe_answer_correct=False),
            "not correct",
        ),
        (
            lambda m: m["state"].update(**{"admission.final_mounts": []}),
            "no lived mounted seats",
        ),
    ],
)
def test_unqualified_lived_snapshot_fails_closed(
    tmp_path: Path, mutate, match: str
):
    registration, order, observations = _case(tmp_path)
    path = observations["fixture_00"]["snapshot_path"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    _sign_snapshot(manifest)
    _write_json(path, manifest)

    with pytest.raises(RegistryError, match=match):
        derive_plant_registry(
            registration,
            order,
            observations,
            created_utc=CREATED_UTC,
            record_root=tmp_path,
        )


def test_snapshot_payload_digest_tamper_fails_closed(tmp_path: Path):
    registration, order, observations = _case(tmp_path)
    path = observations["fixture_00"]["snapshot_path"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["label"] = "replay"
    _write_json(path, manifest)

    with pytest.raises(RegistryError, match="payload digest mismatch"):
        derive_plant_registry(
            registration,
            order,
            observations,
            created_utc=CREATED_UTC,
            record_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("created", "eval_stamps", "match"),
    [
        ("2026-08-31T19:59:59Z", [], "before its lived evidence"),
        (CREATED_UTC, ["2026-08-31T23:59:59Z"], "not frozen before new eval"),
    ],
)
def test_freeze_chronology_rejects_post_data_registration(
    tmp_path: Path, created: str, eval_stamps: list[str], match: str
):
    registration, order, observations = _case(tmp_path)
    with pytest.raises(RegistryError, match=match):
        derive_plant_registry(
            registration,
            order,
            observations,
            created_utc=created,
            new_eval_evidence_utc=eval_stamps,
            record_root=tmp_path,
        )


def test_same_session_substitution_is_enumerated(tmp_path: Path):
    registration, order, observations = _case(tmp_path)
    base = json.loads(registration.read_text(encoding="utf-8"))["fixtures"][2]
    fixture_id = "fixture_02"
    replacement = "replacement_turn_31"
    path = observations[fixture_id]["snapshot_path"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["provenance"]["fixture_id"] = replacement
    manifest["linked_answer"]["attempt_answer"] = "replacement-value"
    manifest["linked_answer"]["probe_answer"] = "replacement-value"
    _sign_snapshot(manifest)
    _write_json(path, manifest)
    observations[fixture_id]["substitution"] = {
        "status": "SUBSTITUTED",
        "original_fixture_id": fixture_id,
        "effective_fixture": {
            "fixture_id": replacement,
            "split": "eval",
            "source_family": "certified_34_turn",
            "session_id": f"certified_34_turn:{'a' * 64}",
            "selector": {"turn": 31},
            "question": "What is the Polaris tone?",
            "expected_values": ["replacement-value"],
            "stale_values": ["value-2"],
            "wrong_fact_values": ["wrong-value"],
            "source": base["source"],
        },
        "reason": "original probe had no lawful mounted seat",
    }

    registry = derive_plant_registry(
        registration,
        order,
        observations,
        created_utc=CREATED_UTC,
        record_root=tmp_path,
    )

    assert registry["substitution_count"] == 1
    assert registry["substitutions"][0]["effective_fixture"]["fixture_id"] == replacement
    assert registry["entries"][2]["fixture_id"] == fixture_id
    assert registry["entries"][2]["effective_fixture_id"] == replacement
    assert registry["entries"][2]["effective_fixture"]["expected_values"] == [
        "replacement-value"
    ]


def test_uncertified_cross_session_substitution_fails_closed(tmp_path: Path):
    registration, order, observations = _case(tmp_path)
    base = json.loads(registration.read_text(encoding="utf-8"))["fixtures"][2]
    fixture_id = "fixture_02"
    observations[fixture_id]["substitution"] = {
        "status": "SUBSTITUTED",
        "original_fixture_id": fixture_id,
        "effective_fixture": {
            "fixture_id": "replacement",
            "split": "eval",
            "source_family": "certified_34_turn",
            "session_id": "another-session",
            "selector": {"turn": 7},
            "question": "replacement question",
            "expected_values": ["replacement-value"],
            "stale_values": [],
            "wrong_fact_values": [],
            "source": base["source"],
        },
        "reason": "no seat",
    }
    with pytest.raises(RegistryError, match="not certified by this campaign"):
        derive_plant_registry(
            registration,
            order,
            observations,
            created_utc=CREATED_UTC,
            record_root=tmp_path,
        )


def test_certified_cross_session_supersession_substitution_is_accepted(
    tmp_path: Path,
):
    registration, order, observations = _case(tmp_path)
    base_registration = json.loads(registration.read_text(encoding="utf-8"))
    base = base_registration["fixtures"][2]
    donor = base_registration["fixtures"][9]
    fixture_id = str(base["fixture_id"])
    replacement = "reserve_meridian_docket"
    path = observations[fixture_id]["snapshot_path"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["provenance"]["fixture_id"] = replacement
    manifest["linked_answer"]["attempt_answer"] = donor["expected_values"][0]
    manifest["linked_answer"]["probe_answer"] = donor["expected_values"][0]
    _sign_snapshot(manifest)
    _write_json(path, manifest)
    observations[fixture_id]["substitution"] = {
        "status": "SUBSTITUTED",
        "original_fixture_id": fixture_id,
        "effective_fixture": {
            "fixture_id": replacement,
            "split": "eval",
            "source_family": donor["source_family"],
            "session_id": donor["session_id"],
            "selector": {"probe_id": "reserve_meridian_docket"},
            "question": "What is the current Meridian docket value?",
            "expected_values": list(donor["expected_values"]),
            "stale_values": [],
            "wrong_fact_values": [],
            "source": dict(donor["source"]),
        },
        "reason": "primary lived served control failed",
    }

    registry = derive_plant_registry(
        registration,
        order,
        observations,
        created_utc=CREATED_UTC,
        record_root=tmp_path,
    )

    entry = registry["entries"][2]
    assert entry["effective_fixture_id"] == replacement
    assert entry["effective_fixture"]["session_id"] == donor["session_id"]
    assert registry["substitution_policy"]["amendment_order"] == "GRM-DET1.9"
    assert registry["substitution_count"] == 1


def test_duplicate_effective_replacements_fail_closed(tmp_path: Path):
    registration, order, observations = _case(tmp_path)
    base_registration = json.loads(registration.read_text(encoding="utf-8"))
    for index, selector in ((2, 31), (3, 33)):
        fixture_id = f"fixture_{index:02d}"
        path = observations[fixture_id]["snapshot_path"]
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["provenance"]["fixture_id"] = "same_replacement"
        manifest["linked_answer"]["attempt_answer"] = "replacement-value"
        manifest["linked_answer"]["probe_answer"] = "replacement-value"
        _sign_snapshot(manifest)
        _write_json(path, manifest)
        observations[fixture_id]["substitution"] = {
            "status": "SUBSTITUTED",
            "original_fixture_id": fixture_id,
            "effective_fixture": {
                "fixture_id": "same_replacement",
                "split": "eval",
                "source_family": "certified_34_turn",
                "session_id": f"certified_34_turn:{'a' * 64}",
                "selector": {"turn": selector},
                "question": "replacement question",
                "expected_values": ["replacement-value"],
                "stale_values": [],
                "wrong_fact_values": [],
                "source": base_registration["fixtures"][index]["source"],
            },
            "reason": "no lawful seat",
        }

    with pytest.raises(RegistryError, match="duplicate effective"):
        derive_plant_registry(
            registration,
            order,
            observations,
            created_utc=CREATED_UTC,
            record_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: value.clear(), "must carry"),
        (
            lambda value: value.update(withheld_alias_ids=[201]),
            "exact target aliases",
        ),
        (
            lambda value: value["delta_receipt"].update(
                non_delta_fields_equal=False
            ),
            "exact target-only delta",
        ),
        (
            lambda value: value["counterfactual"].update(
                answer="value-1", classification="incorrect"
            ),
            "remains semantically correct",
        ),
    ],
)
def test_declared_synthesis_requires_target_only_behavioral_breaker(
    tmp_path: Path, mutate, match: str
):
    registration, order, observations = _case(tmp_path)
    mutate(observations["fixture_01"]["behavioral_breaker"])
    with pytest.raises(RegistryError, match=match):
        derive_plant_registry(
            registration,
            order,
            observations,
            created_utc=CREATED_UTC,
            record_root=tmp_path,
        )


def test_registry_policy_or_entry_tamper_cannot_be_resigned(tmp_path: Path):
    registry = _derive(tmp_path)
    registry["policy_bindings"]["threshold_policy"]["projection"]["D-NGH"][
        "threshold_fit"
    ] = "changed"
    registry["entries"][0]["selected_target_id"] = 999
    _resign_registry(registry)

    with pytest.raises(RegistryError, match="binding changed"):
        validate_plant_registry(registry, record_root=tmp_path)
