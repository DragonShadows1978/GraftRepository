import json
from pathlib import Path

import pytest

from scripts.grm_det1_baseline_registry import (
    BaselineRegistryError,
    DEFAULT_REGISTRY,
    REGISTRY_ENV,
    compare_served_to_live_registry,
    load_live_registry,
    registered_projection,
    registry_path,
)


HARBOR = {
    "fixture_id": "sup_harbor_restatement",
    "source_family": "supersession_battery_on_gpt_oss",
    "source": {
        "bytes": 2303,
        "path": "tests/fixtures/supersession_battery/correction_then_restatement.json",
        "sha256": "ea97d4007d66357442f0424f76200984e8c2f39311618a4b892e88db9ad90f4b",
    },
    "session_id": "correction_then_restatement",
    "scenario": "correction_then_restatement",
    "probe_id": "harbor_restatement",
    "question": "What is the current Harbor token value?",
    "target_node": "harbor_c",
    "expected_values": ["nacre-6-blue"],
    "stale_values": ["morrow-5-red"],
    "wrong_fact_values": ["opal-7-green"],
}
ANSWER = "The current Harbor token value is Nacre-6-Blue."
ORION = {
    "fixture_id": "sup_orion_current",
    "source_family": "supersession_battery_on_gpt_oss",
    "source": {
        "bytes": 2101,
        "path": "tests/fixtures/supersession_battery/short_correction_long_competitor.json",
        "sha256": "c06decbe127dd228bdf6529b002ccdaaf5f628addda003692e5e49d9e47c1221",
    },
    "session_id": "short_correction_long_competitor",
    "scenario": "short_correction_long_competitor",
    "probe_id": "orion_current",
    "question": "What is the current Orion pin value?",
    "target_node": "orion_b",
    "expected_values": ["kestrel-9-tango"],
    "stale_values": ["auric-4-alpha"],
    "wrong_fact_values": ["vortex-3-sierra"],
}


def test_live_harbor_registration_is_post_adm2():
    registry, anchor = load_live_registry()
    active = registered_projection(HARBOR, registry=registry)
    retired = registered_projection(
        HARBOR,
        registry=registry,
        previous_registration_id=registry["previous_registrations"][0]["id"],
    )
    assert anchor["registration_hash"] == (
        "e2ff807356604911520eac922befd82384fef6354720b37cd9b76d119c8d9378"
    )
    assert active["answer"] == retired["answer"] == ANSWER
    assert active["mounted_ids"] == [1]
    assert active["mounted_nodes"] == ["harbor_b"]
    assert retired["mounted_ids"] == [2]
    assert retired["mounted_nodes"] == ["harbor_c"]
    previous = registry["previous_registrations"][0]
    assert previous["registration_receipt"]["sha256"] == (
        "f2d82d105b7ccaba1c1dfd65b54e56c9f2829e3064f54ea471b4d691e22ca211"
    )
    assert previous["families"]["supersession_battery_on_gpt_oss"]["source"][
        "sha256"
    ] == "ba550e37915b09bbf4f80347b32386b3500ea360afef28b054f91463dce4e6b1"
    assert active["registered_source_row"]["arena_info"]["mounts"] == [2]
    assert retired["registered_source_row"]["arena_info"]["mounts"] == [3]


def test_served_comparison_distinguishes_live_retired_and_leak():
    live = compare_served_to_live_registry(
        HARBOR, {"answer": ANSWER, "mounted_ids": [1], "answer_correct": True})
    assert live["live_match"] is True
    assert live["previous_registration_matches"] == []

    retired = compare_served_to_live_registry(
        HARBOR, {"answer": ANSWER, "mounted_ids": [2], "answer_correct": True})
    assert retired["live_match"] is False
    assert retired["previous_registration_matches"] == [
        "GRM-SUP-L2-default-on-20260830T165648866160Z_4123111"
    ]

    leak = compare_served_to_live_registry(
        HARBOR, {"answer": ANSWER, "mounted_ids": [3], "answer_correct": True})
    assert leak["live_match"] is False
    assert leak["previous_registration_matches"] == []


def test_orion_cross_model_uses_det_fixture_semantics_not_producer_bytes():
    short = compare_served_to_live_registry(
        ORION,
        {"answer": "Kestrel-9-Tango", "mounted_ids": [1], "answer_correct": True},
    )
    assert short["live_match"] is True
    assert short["observed_mounted_nodes"] == ["orion_b"]

    producer_bytes = compare_served_to_live_registry(
        ORION,
        {
            "answer": (
                "The current Orion pin value is Kestrel-9-Tango, "
                "replacing Auric-4-Alpha."
            ),
            "mounted_ids": [1],
            "answer_correct": False,
        },
    )
    assert producer_bytes["live_match"] is False
    observed_class = producer_bytes["checks"][
        "det_fixture_observed_classification"
    ]
    assert observed_class["classification"] == "stale"
    assert observed_class["rejected_value_matches"] == ["auric-4-alpha"]

    inconsistent = compare_served_to_live_registry(
        ORION,
        {"answer": "Kestrel-9-Tango", "mounted_ids": [1], "answer_correct": False},
    )
    assert inconsistent["live_match"] is False
    assert inconsistent["checks"]["served_answer_correct_consistent"] is False


def test_cross_model_value_semantics_normalize_emphasis_and_hyphen_glyphs():
    result = compare_served_to_live_registry(
        ORION,
        {
            "answer": "**Kestrel\u20119\u2010Tango**",
            "mounted_ids": [1],
            "answer_correct": True,
        },
    )
    assert result["live_match"] is True
    assert result["checks"]["det_fixture_observed_classification"][
        "classification"
    ] == "correct"


def test_same_model_e2e_contract_remains_exact_bytes_and_ordered_mounts():
    fixture = {
        "fixture_id": "e2e_t05_orion_pin",
        "source_family": "certified_34_turn",
        "turn": 5,
    }
    registry, _anchor = load_live_registry()
    expected = registered_projection(fixture, registry=registry)
    exact = compare_served_to_live_registry(
        fixture,
        {"answer": expected["answer"], "mounted_ids": expected["mounted_ids"]},
    )
    assert exact["live_match"] is True
    changed = compare_served_to_live_registry(
        fixture,
        {"answer": expected["answer"] + " ", "mounted_ids": expected["mounted_ids"]},
    )
    assert changed["live_match"] is False
    assert changed["checks"]["answer_exact_bytes"] is False


def test_registry_source_hash_drift_fails_closed(tmp_path: Path):
    registry, _anchor = load_live_registry()
    registry["families"]["supersession_battery_on_gpt_oss"]["source"][
        "sha256"
    ] = "0" * 64
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(BaselineRegistryError, match="hash/size"):
        load_live_registry(path)


def test_comparator_and_fixture_identity_tamper_fail_closed(tmp_path: Path):
    registry, _anchor = load_live_registry()
    registry["families"]["supersession_battery_on_gpt_oss"]["comparator"][
        "answer"
    ] = "exact_bytes"
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(BaselineRegistryError, match="comparator"):
        load_live_registry(path)

    registry, _anchor = load_live_registry()
    bad_fixture = dict(HARBOR)
    bad_fixture["source"] = {**HARBOR["source"], "sha256": "0" * 64}
    with pytest.raises(BaselineRegistryError, match="fixture source"):
        registered_projection(bad_fixture, registry=registry)


def test_ambient_registry_environment_cannot_override_production(monkeypatch):
    monkeypatch.setenv(REGISTRY_ENV, "/tmp/private-stale-registry.json")
    assert registry_path() == DEFAULT_REGISTRY.resolve()
