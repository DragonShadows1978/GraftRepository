"""CPU-only contract tests for GRM-ADM1.3 dual-frame evaluation."""

from __future__ import annotations

from types import SimpleNamespace

from scripts.grm_adm1_3_dual_frame import (
    ARMS,
    CLASSES,
    adoption_rule,
    anchor_matches,
    build_dec_driver_attempts,
    corpus_frame_direction,
    frame_disagreement_text,
    gate_adoption,
    result_exit_code,
)
from scripts.grm_adm1_3_prod_e2e import install_patch


def _metric(n: int, value: float | None):
    return {
        "n": n,
        "estimate": value,
        "ci95": None if n == 0 else [value, value],
    }


def _table(dec_point=1.0, k3_point=1.0, dec_wrong=0.0, k3_wrong=0.0):
    table = {}
    for cls in CLASSES:
        n = 0 if cls == "AMBIGUOUS" else 1
        table[cls] = {}
        for arm in ARMS:
            recall = 1.0 if n else None
            wrong = 0.0 if n else None
            if cls == "POINT-LOOKUP" and arm == "A-DEC":
                recall, wrong = dec_point, dec_wrong
            if cls == "POINT-LOOKUP" and arm == "A-k3":
                recall, wrong = k3_point, k3_wrong
            table[cls][arm] = {
                "recall": _metric(n, recall),
                "wrong_read": _metric(n, wrong),
                "miss": _metric(n, None if n == 0 else 1.0 - recall),
                "mount_count": _metric(n, None if n == 0 else 1.0),
            }
    return table


def _synth_rows():
    rows = []
    for probe_id, fixture, nodes in (
        ("e", "E2E-34-CROSS-FACT", [1, 2]),
        ("p", "P4-REPLICATION-CROSS-FACT", [3, 4]),
    ):
        rows.append({
            "probe_id": probe_id,
            "fixture": fixture,
            "class_label": "CROSS-FACT / SYNTHESIS",
            "split": "eval",
            "identified_candidates": nodes,
            "required_node_ids": [nodes[-1]],
            "arms": {
                "A-DEC": {
                    "policy_branch": "declared_synthesis_identified_set",
                    "recall": 1,
                    "miss": 0,
                    "ladder_initial_policy_plan": nodes,
                    "production_ladder": {"probe_ladder": True},
                    "mounted_nodes": nodes,
                },
                "A-k3": {"recall": 1, "miss": 0, "mounted_nodes": nodes},
            },
        })
    return rows


def test_dec_plan_is_trip_zero_then_registered_retry():
    attempts = build_dec_driver_attempts(
        ranking=[5, 6, 7, 4],
        topk=3,
        precise=[5],
        point_lookup=True,
        max_trips=1,
        policy_plan=[5, 7],
    )
    assert attempts == [([5, 7], True), ([5, 6, 7], True)]


def test_anchor_match_is_exact_on_answer_mount_and_metrics():
    result = {
        "answer": "Alpha",
        "mounted_nodes": [1, 3],
        "answer_correct": True,
        "recall": 1,
        "wrong_read": 0,
    }
    anchor = {"a_k3_expectation": dict(result)}
    assert anchor_matches(anchor, result)
    changed = dict(result, mounted_nodes=[1])
    assert not anchor_matches(anchor, changed)


def test_adoption_rule_recommends_only_when_all_registered_cells_pass():
    passed = adoption_rule(_table(), _synth_rows())
    assert passed["recommendation"] == "ADOPT-RECOMMENDED"
    recall_fail = adoption_rule(_table(dec_point=0.0), _synth_rows())
    assert recall_fail["recommendation"] == "NOT-RECOMMENDED"
    assert "POINT-LOOKUP" in recall_fail["failures"][0]
    wrong_fail = adoption_rule(
        _table(dec_wrong=1.0, k3_wrong=0.0), _synth_rows())
    assert wrong_fail["recommendation"] == "NOT-RECOMMENDED"
    assert any("wrong-read" in value for value in wrong_fail["failures"])


def test_missing_declared_synthesis_replication_blocks_adoption():
    result = adoption_rule(_table(), _synth_rows()[:1])
    assert result["recommendation"] == "NOT-RECOMMENDED"
    assert not result["declared_synthesis_fixture_coverage"]


def test_changed_declared_synthesis_final_mount_receipt_blocks_adoption():
    rows = _synth_rows()
    rows[0]["arms"]["A-DEC"]["mounted_nodes"] = [1]
    result = adoption_rule(_table(), rows)
    assert result["recommendation"] == "NOT-RECOMMENDED"
    assert not result["declared_synthesis_checks"][0]["final_mount_receipt_preserved"]


def test_both_arms_losing_registered_synthesis_receipt_blocks_adoption():
    rows = _synth_rows()
    rows[0]["arms"]["A-DEC"].update(
        {"mounted_nodes": [99], "recall": 0, "miss": 1})
    rows[0]["arms"]["A-k3"].update(
        {"mounted_nodes": [99], "recall": 0, "miss": 1})
    result = adoption_rule(_table(), rows)
    assert result["recommendation"] == "NOT-RECOMMENDED"
    assert not result["declared_synthesis_checks"][0]["final_mount_receipt_preserved"]


def test_red_data_gate_forces_not_recommended_and_nonzero_exit():
    adoption = adoption_rule(_table(), _synth_rows())
    gates = {
        "ADM1.3-G0-FRESH-ANCHORS": "RED",
        "F-PROD-LADDER-ON-L2-ON": "PASS",
        "F-ISO-REUSE": "PASS",
        "FROZEN-RULE-UNCHANGED": "PASS",
        "PRODUCTION-FILES-UNCHANGED-DURING-RUN": "PASS",
        "NO-PRODUCTION-WIRING": "PASS",
        "DUAL-FRAME-REPORT": "PASS",
    }
    gated = gate_adoption(adoption, gates)
    assert gated["recommendation"] == "NOT-RECOMMENDED"
    assert gated["data_validity_gate_failures"] == ["ADM1.3-G0-FRESH-ANCHORS"]
    assert result_exit_code({"gates": gates}) == 2


def test_frame_disagreement_claims_rescue_and_compression_only_when_measured():
    iso = {
        "A-k1": [17, 20], "A-k2": [8, 20],
        "A-k3": [4, 20], "A-DEC": [17, 20],
    }
    prod = {arm: [16, 20] for arm in ARMS}
    direction = corpus_frame_direction(prod, iso)
    assert direction["arm_differences_compressed"]
    assert direction["rescued_arms"] == ["A-k2", "A-k3"]
    assert direction["regressed_arms"] == ["A-k1", "A-DEC"]
    text = frame_disagreement_text(direction, prod, iso, 3)
    assert "do not support the full expected" in text

    rescued = {arm: [18, 20] for arm in ARMS}
    direction = corpus_frame_direction(rescued, iso)
    text = frame_disagreement_text(direction, rescued, iso, 3)
    assert "consistent with" in text


def test_prod_wrapper_is_inert_with_flag_off():
    original = object()
    fake = SimpleNamespace(probe_multimount_chat=original)
    assert install_patch(fake, enabled=False) is False
    assert fake.probe_multimount_chat is original
