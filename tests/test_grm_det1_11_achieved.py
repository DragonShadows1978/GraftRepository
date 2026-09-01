"""GRM-DET1.11 contracts: the achieved-pair floor and achieved-count reporting.

Two things must hold and keep holding:

1. The campaign runs at the ACHIEVED lawful pair count, and REFUSES below the
   registered floor of eight pairs rather than running a hollow race.
2. The achieved N is reported prominently — in the race table and in the
   verdict sentence — so no reader can mistake an amended run for a full one.

The adjudication vocabulary is explicitly NOT amended, and one test pins that
too: DET1.11 changes the population, never the verdict words.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_det1_common import (  # noqa: E402
    ACHIEVED_EVAL_PAIR_FLOOR,
    DETError,
    REGISTERED_CALIBRATION_PAIR_COUNT,
    REGISTERED_EVAL_PAIR_COUNT,
    race_metrics,
)
from scripts.grm_det1_5_analyze import (  # noqa: E402
    VERDICTS,
    _achieved_clause,
    _prediction_sentence,
    race_table_markdown,
)
from scripts.grm_det1_5_gpu import (  # noqa: E402
    merge_detector_rows,
    validate_g0_rows,
)
from scripts.grm_det1_7_registry import (  # noqa: E402
    ACHIEVED_SELECTION_RULE_ID,
    EXCLUSION_STATUS,
    RegistryError,
    _achieved_counts_projection,
    _achieved_pair_counts,
    _exclusion_map,
    _require_achieved_floor,
)


# --- the registered floor ------------------------------------------------


def test_floor_is_registered_at_eight_pairs_before_any_gate():
    """The floor is a registered constant, not a tunable read off results."""
    assert ACHIEVED_EVAL_PAIR_FLOOR == 8
    assert REGISTERED_EVAL_PAIR_COUNT == 12
    assert REGISTERED_CALIBRATION_PAIR_COUNT == 2
    assert ACHIEVED_EVAL_PAIR_FLOOR < REGISTERED_EVAL_PAIR_COUNT


def _entries(eval_pairs: int, calibration_pairs: int = 2) -> list[dict]:
    rows = [
        {"fixture_id": f"cal_{index}", "split": "calibration"}
        for index in range(calibration_pairs)
    ]
    rows += [
        {"fixture_id": f"eval_{index}", "split": "eval"}
        for index in range(eval_pairs)
    ]
    return rows


def _exclusions(count: int, *, start: int = 0) -> list[dict]:
    return [
        {
            "fixture_id": f"eval_{start + index}",
            "split": "eval",
            "status": EXCLUSION_STATUS,
            "primary_reason": "LIVED_SERVED_CONTROL_INCORRECT_OR_REFUSAL",
            "served_answer": "Birch-2-Beacon.",
            "reserve_pool_exhausted": True,
            "selection_rule_id": ACHIEVED_SELECTION_RULE_ID,
        }
        for index in range(count)
    ]


@pytest.mark.parametrize("achieved", [12, 11, 10, 9, 8])
def test_achieved_populations_at_or_above_the_floor_are_accepted(achieved):
    excluded = _exclusions(REGISTERED_EVAL_PAIR_COUNT - achieved)
    _require_achieved_floor(_entries(achieved), excluded)


@pytest.mark.parametrize("achieved", [7, 6, 2, 0])
def test_achieved_population_below_the_floor_refuses(achieved):
    excluded = _exclusions(REGISTERED_EVAL_PAIR_COUNT - achieved)
    with pytest.raises(RegistryError, match="floor not met"):
        _require_achieved_floor(_entries(achieved), excluded)


def test_calibration_is_never_shrunk():
    """Calibration fits thresholds; DET1.11 does not license dropping it."""
    with pytest.raises(RegistryError, match="calibration is never shrunk"):
        _require_achieved_floor(_entries(12, calibration_pairs=1), [])


def test_achieved_plus_excluded_must_reconstruct_the_registered_population():
    with pytest.raises(RegistryError, match="must equal the registered"):
        _require_achieved_floor(_entries(10), _exclusions(1))


def test_achieved_population_cannot_exceed_the_registered_one():
    with pytest.raises(RegistryError, match="must equal the registered"):
        _require_achieved_floor(_entries(13), [])


def test_race_metrics_refuses_below_the_floor_and_runs_at_it():
    rows = _race_rows(ACHIEVED_EVAL_PAIR_FLOOR)
    table, _winners, _verdict = race_metrics(rows, _thresholds())
    assert table[0]["positive_n"] == ACHIEVED_EVAL_PAIR_FLOOR
    assert table[0]["negative_n"] == ACHIEVED_EVAL_PAIR_FLOOR

    with pytest.raises(DETError, match="floor not met"):
        race_metrics(_race_rows(ACHIEVED_EVAL_PAIR_FLOOR - 1), _thresholds())


def test_race_metrics_refuses_unbalanced_arms():
    # Both arms clear the floor; only the imbalance is at fault, so the
    # imbalance is the error the caller must see.
    rows = _race_rows(ACHIEVED_EVAL_PAIR_FLOOR + 1)
    rows = [row for row in rows if row["variant"] == "served"] + [
        row for row in rows if row["variant"] == "planted_miss"
    ][:-1]
    with pytest.raises(DETError, match="unbalanced"):
        race_metrics(rows, _thresholds())


# --- achieved-count reporting -------------------------------------------


def test_verdict_sentence_names_the_achieved_population():
    for verdict in VERDICTS:
        sentence = _prediction_sentence(verdict, 10)
        assert sentence.startswith(f"PREDICTION VERDICT: {verdict}")
        assert "at 10 of 12 registered pairs" in sentence


def test_verdict_sentence_still_reads_naturally_at_full_population():
    sentence = _prediction_sentence("SUPPORTED", REGISTERED_EVAL_PAIR_COUNT)
    assert "at 12 of 12 registered pairs" in sentence


def test_verdict_vocabulary_is_unchanged_by_the_amendment():
    """DET1.11 moves the population, never the adjudication words."""
    assert VERDICTS == ("SUPPORTED", "PARTIAL", "REFUTED")
    with pytest.raises(DETError, match="forbidden prediction verdict"):
        _prediction_sentence("UNADJUDICATED", 10)


def test_achieved_clause_refuses_counts_outside_the_registered_band():
    with pytest.raises(DETError, match="outside the registered floor"):
        _achieved_clause(ACHIEVED_EVAL_PAIR_FLOOR - 1)
    with pytest.raises(DETError, match="outside the registered floor"):
        _achieved_clause(REGISTERED_EVAL_PAIR_COUNT + 1)


def test_race_table_leads_with_the_achieved_population_and_exclusions():
    table = [
        {
            "detector": name,
            "tp": 8, "positive_n": 10, "fp": 1, "negative_n": 10,
            "recall": 0.8, "false_positive_rate": 0.1,
            "precision": 0.888, "f1": 0.842,
            "median_trigger_latency_token": 3.0,
            "viable": True,
        }
        for name in ("D-LQR", "D-NGH", "D-ENT", "D-VERB")
    ]
    text = race_table_markdown(
        table,
        achieved_pairs=10,
        excluded_fixture_ids=["sup_lumen_head", "sup_orion_current"],
    )
    header = text.splitlines()[0]
    assert "at 10 of 12 registered pairs" in header
    assert "sup_lumen_head" in text and "sup_orion_current" in text
    assert "Excluded (2)" in text
    # The four detector rows survive the added header.
    assert text.count("| D-") == 4


def test_race_table_without_achieved_pairs_is_the_bare_registered_table():
    table = [
        {
            "detector": name,
            "tp": 9, "positive_n": 12, "fp": 1, "negative_n": 12,
            "recall": 0.75, "false_positive_rate": 0.083,
            "precision": 0.9, "f1": 0.818,
            "median_trigger_latency_token": None,
            "viable": False,
        }
        for name in ("D-LQR", "D-NGH", "D-ENT", "D-VERB")
    ]
    text = race_table_markdown(table)
    assert text.splitlines()[0].startswith("| Detector |")


# --- achieved-count projections -----------------------------------------


def test_achieved_counts_projection_reports_n_of_registered():
    projection = _achieved_counts_projection(_entries(10), _exclusions(2))
    assert projection["achieved_eval_pairs"] == 10
    assert projection["registered_eval_pairs"] == 12
    assert projection["excluded_eval_pairs"] == 2
    assert projection["achieved_of_registered"] == "10 of 12"
    assert projection["meets_floor"] is True
    assert projection["is_full_registered_population"] is False


def test_full_population_projection_is_marked_as_such():
    projection = _achieved_counts_projection(_entries(12), [])
    assert projection["is_full_registered_population"] is True
    assert projection["achieved_of_registered"] == "12 of 12"


def test_pair_counts_are_derived_from_entries_not_asserted():
    counts = _achieved_pair_counts(_entries(9))
    assert counts == {
        "all_fixture_pairs": 11,
        "calibration_fixture_pairs": 2,
        "eval_fixture_pairs": 9,
        "eval_served_rows": 9,
        "eval_planted_miss_rows": 9,
    }


# --- exclusion records ---------------------------------------------------


_FIXTURES = {
    "eval_0": {"fixture_id": "eval_0", "split": "eval"},
    "cal_0": {"fixture_id": "cal_0", "split": "calibration"},
}


def test_exclusion_requires_a_lived_failure_reason():
    row = dict(_exclusions(1)[0])
    row.pop("primary_reason")
    with pytest.raises(RegistryError, match="primary_reason"):
        _exclusion_map([row], _FIXTURES)


def test_exclusion_may_not_name_a_calibration_slot():
    row = dict(_exclusions(1)[0])
    row["fixture_id"] = "cal_0"
    with pytest.raises(RegistryError, match="evaluation slots only"):
        _exclusion_map([row], _FIXTURES)


def test_exclusion_may_not_name_an_uncertified_fixture():
    row = dict(_exclusions(1)[0])
    row["fixture_id"] = "not_registered"
    with pytest.raises(RegistryError, match="does not\\s+certify"):
        _exclusion_map([row], _FIXTURES)


def test_exclusion_requires_the_registered_status_word():
    row = dict(_exclusions(1)[0])
    row["status"] = "DROPPED"
    with pytest.raises(RegistryError, match="must carry status"):
        _exclusion_map([row], _FIXTURES)


def test_duplicate_exclusions_fail_closed():
    rows = _exclusions(1) + _exclusions(1)
    with pytest.raises(RegistryError, match="duplicate exclusion"):
        _exclusion_map(rows, _FIXTURES)


def test_exclusion_records_carry_the_served_answer_for_the_receipts():
    mapped = _exclusion_map(_exclusions(1), _FIXTURES)
    assert mapped["eval_0"]["served_answer"] == "Birch-2-Beacon."
    assert mapped["eval_0"]["reserve_pool_exhausted"] is True
    assert mapped["eval_0"]["selection_rule_id"] == ACHIEVED_SELECTION_RULE_ID


# --- gate arity follows the achieved population -------------------------


def test_g0_gate_arity_follows_the_achieved_pair_count():
    with pytest.raises(DETError, match="requires 10 served"):
        validate_g0_rows([], expected_pairs=10)


def test_detector_join_refuses_arms_below_the_floor():
    rows = [
        {"row_id": f"r{index}", "fixture_id": f"f{index // 2}",
         "variant": "served" if index % 2 else "planted_miss"}
        for index in range(2 * (ACHIEVED_EVAL_PAIR_FLOOR - 1))
    ]
    with pytest.raises(DETError, match="floor not met"):
        merge_detector_rows(rows, rows)


# --- helpers -------------------------------------------------------------


def _thresholds() -> dict:
    return {
        "D-NGH": {"mounted_mass_min": 0.5},
        "D-ENT": {"margin_min": 0.5},
    }


def _race_rows(pairs: int) -> list[dict]:
    rows = []
    for index in range(pairs):
        for variant in ("planted_miss", "served"):
            rows.append({
                "row_id": f"{index}:{variant}",
                "fixture_id": f"fixture_{index}",
                "variant": variant,
                "lqr": {
                    "triggered": variant == "planted_miss",
                    "trigger_token_index": 2,
                },
                "ngh": {"mounted_mass": 0.9, "trigger_token_index": 3},
                "ent": {"margin": 0.9, "trigger_token_index": 4},
                "verbal": {"answer": "YES", "trigger_token_index": 5},
            })
    return rows
