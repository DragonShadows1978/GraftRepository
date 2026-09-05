"""GRM-WC1 — CPU tests for the PURE parts of the arena-width sweep driver.

Nothing here touches a GPU, a model, or the frozen run tree.  What these tests
cover is exactly the machinery that could silently corrupt the measurement:

  * FRAME DERIVATION — the width injection must move ONE field and nothing
    else, and the single-variable check must actually catch a second change
    rather than rubber-stamping it;
  * TABLE ASSEMBLY — a battery that was not run must read as "not run", never
    as "scored zero";
  * REGRESSION-VS-96 — a regression and a recovery must not be confused, and
    a probe set that differs between two widths must be reported as
    non-comparable instead of producing a bogus regression list;
  * SPLIT CENSUS — the width-guard family counts read only the flags the
    existing split writers set;
  * PREDICTION SCORING — an unrunnable prediction is UNSCORED, never a
    silent HIT.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_wc1_common import (  # noqa: E402
    BATTERIES,
    REFERENCE_WIDTH,
    WIDTH_GRID,
    WC1Error,
    battery_result,
    mean_or_none,
    prediction_verdicts,
    probe_row,
    regressions_vs_reference,
    split_census,
    sweep_table,
    unchanged_fields,
    width_frame_payload,
)


BASE_FRAME = {
    "resolved_flags": {
        "adm_decisive": True,
        "arena_width": 96,
        "live_turns": 2,
        "ngen": 32,
        "topk": 3,
    },
    "model": {"path": "/models/gpt-oss-20b", "revision": "deadbeef"},
    "schema": "grm.det1.runtime_frame.v1",
}


# ---------------------------------------------------------------------------
# The width grid itself
# ---------------------------------------------------------------------------

def test_width_grid_is_the_registered_one():
    """The grid is the order's, in ascending order, with 96 on it."""
    assert WIDTH_GRID == (64, 96, 128, 192, 256)
    assert list(WIDTH_GRID) == sorted(WIDTH_GRID)
    assert REFERENCE_WIDTH in WIDTH_GRID


def test_batteries_are_the_three_the_order_names():
    assert BATTERIES == ("sup", "census", "longhorizon")


# ---------------------------------------------------------------------------
# Frame derivation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width", WIDTH_GRID)
def test_width_frame_sets_only_the_width(width):
    derived = width_frame_payload(BASE_FRAME, width)
    assert derived["resolved_flags"]["arena_width"] == width
    assert unchanged_fields(BASE_FRAME, derived)
    # Every other flag survives byte-for-byte.
    for key, value in BASE_FRAME["resolved_flags"].items():
        if key == "arena_width":
            continue
        assert derived["resolved_flags"][key] == value
    assert derived["model"] == BASE_FRAME["model"]


def test_width_frame_records_the_override_provenance():
    derived = width_frame_payload(BASE_FRAME, 256)
    override = derived["wc1_override"]
    assert override["field"] == "resolved_flags.arena_width"
    assert override["frozen_value"] == 96
    assert override["override_value"] == 256


def test_width_frame_does_not_mutate_the_frozen_frame():
    before = json.dumps(BASE_FRAME, sort_keys=True)
    width_frame_payload(BASE_FRAME, 192)
    assert json.dumps(BASE_FRAME, sort_keys=True) == before


def test_single_variable_check_catches_a_second_change():
    """The check must FAIL when anything besides the width moves.

    A check that only ever returns True would certify a confounded run.
    """
    derived = width_frame_payload(BASE_FRAME, 128)
    derived["resolved_flags"]["topk"] = 5
    assert not unchanged_fields(BASE_FRAME, derived)


def test_single_variable_check_catches_a_new_top_level_key():
    derived = width_frame_payload(BASE_FRAME, 128)
    derived["smuggled"] = True
    assert not unchanged_fields(BASE_FRAME, derived)


@pytest.mark.parametrize("bad", [0, -1, -96])
def test_width_frame_refuses_a_non_positive_width(bad):
    with pytest.raises(WC1Error):
        width_frame_payload(BASE_FRAME, bad)


def test_width_frame_refuses_a_frame_with_no_width():
    with pytest.raises(WC1Error):
        width_frame_payload({"resolved_flags": {"topk": 3}}, 128)


# ---------------------------------------------------------------------------
# Battery cells and table assembly
# ---------------------------------------------------------------------------

def _cell(width, battery, verdicts):
    return battery_result(
        width, battery,
        [probe_row(f"p{i}", ok) for i, ok in enumerate(verdicts)])


def test_battery_result_counts_correct_probes():
    cell = _cell(96, "sup", [True, False, True])
    assert cell["correct"] == 2
    assert cell["total"] == 3
    assert cell["width"] == 96


def test_battery_result_refuses_an_unknown_battery():
    with pytest.raises(WC1Error):
        battery_result(96, "not_a_battery", [])


def test_sweep_table_reports_a_missing_battery_as_none_not_zero():
    """A battery that did not run must never read as a failing battery."""
    cells = {
        96: {"sup": _cell(96, "sup", [True] * 9)},
        128: {"sup": _cell(128, "sup", [True] * 9)},
    }
    table = sweep_table(cells)
    row96 = next(r for r in table if r["width"] == 96)
    assert row96["sup"]["score"] == "9/9"
    assert row96["census"] is None
    assert row96["longhorizon"] is None
    assert row96[f"census_regressions_vs_{REFERENCE_WIDTH}"] is None


def test_sweep_table_reference_row_has_an_empty_regression_list():
    cells = {96: {"sup": _cell(96, "sup", [True] * 9)}}
    row = sweep_table(cells)[0]
    assert row[f"sup_regressions_vs_{REFERENCE_WIDTH}"] == []


def test_sweep_table_is_ordered_by_width():
    cells = {w: {"sup": _cell(w, "sup", [True])} for w in (256, 64, 128)}
    assert [r["width"] for r in sweep_table(cells)] == [64, 128, 256]


def test_sweep_table_regressions_when_reference_missing_are_none():
    cells = {128: {"sup": _cell(128, "sup", [True, False])}}
    row = sweep_table(cells)[0]
    assert row[f"sup_regressions_vs_{REFERENCE_WIDTH}"] is None


# ---------------------------------------------------------------------------
# Regression vs the reference width
# ---------------------------------------------------------------------------

def test_regression_is_a_probe_96_gets_right_and_this_width_does_not():
    ref = _cell(96, "sup", [True, True, False])
    cell = _cell(64, "sup", [True, False, False])
    out = regressions_vs_reference(cell, ref)
    assert out["regressions"] == ["p1"]
    assert out["recoveries"] == []
    assert out["delta_correct"] == -1
    assert out["comparable"]


def test_recovery_is_not_counted_as_a_regression():
    ref = _cell(96, "sup", [True, False])
    cell = _cell(192, "sup", [True, True])
    out = regressions_vs_reference(cell, ref)
    assert out["regressions"] == []
    assert out["recoveries"] == ["p1"]
    assert out["delta_correct"] == 1


def test_identical_columns_have_no_regressions_either_way():
    ref = _cell(96, "census", [True, True, False])
    cell = _cell(128, "census", [True, True, False])
    out = regressions_vs_reference(cell, ref)
    assert out["regression_count"] == 0
    assert out["recovery_count"] == 0
    assert out["delta_correct"] == 0


def test_a_differing_probe_set_is_reported_non_comparable():
    """A silent set difference would fabricate a regression count."""
    ref = battery_result(96, "sup", [probe_row("a", True),
                                     probe_row("b", True)])
    cell = battery_result(64, "sup", [probe_row("a", True),
                                      probe_row("c", True)])
    out = regressions_vs_reference(cell, ref)
    assert not out["comparable"]
    assert out["missing_from_this_width"] == ["b"]
    assert out["absent_from_reference"] == ["c"]
    # 'b' is missing, not failing — it must NOT appear as a regression.
    assert out["regressions"] == []


# ---------------------------------------------------------------------------
# Split census
# ---------------------------------------------------------------------------

def test_split_census_reads_only_the_existing_width_guard_flags():
    nodes = [
        {"metadata": {"width_guard_parent": True}},
        {"metadata": {"width_guard_child": True}},
        {"metadata": {"width_guard_child": True}},
        {"metadata": {}},
        {},
    ]
    out = split_census(nodes)
    assert out["nodes"] == 5
    assert out["split_parents"] == 1
    assert out["split_children"] == 2
    assert out["split_nodes"] == 3
    assert out["ephemeral_split_members"] == 0


def test_split_census_counts_the_ephemeral_pair_too():
    nodes = [
        {"ephemeral_split": True},
        {"ephemeral_split_of": 0},
    ]
    out = split_census(nodes)
    assert out["split_parents"] == 1
    assert out["split_children"] == 1
    assert out["ephemeral_split_members"] == 2


def test_split_census_of_an_unsplit_repository_is_all_zero():
    out = split_census([{"metadata": {}} for _ in range(7)])
    assert out["split_nodes"] == 0
    assert out["nodes"] == 7


# ---------------------------------------------------------------------------
# mean_or_none
# ---------------------------------------------------------------------------

def test_mean_of_an_empty_sample_is_none_not_zero():
    assert mean_or_none([]) is None
    assert mean_or_none([None, None]) is None


def test_mean_ignores_missing_values_but_keeps_real_zeros():
    assert mean_or_none([1.0, 3.0]) == 2.0
    assert mean_or_none([0.0, 0.0]) == 0.0
    assert mean_or_none([None, 2.0, 4.0]) == 3.0


# ---------------------------------------------------------------------------
# Prediction scoring
# ---------------------------------------------------------------------------

def _full_cells(sup, census, lh):
    """Cells for every registered width from three per-width verdict maps."""
    cells = {}
    for width in WIDTH_GRID:
        cells[width] = {
            "sup": _cell(width, "sup", sup[width]),
            "census": _cell(width, "census", census[width]),
            "longhorizon": _cell(width, "longhorizon", lh[width]),
        }
    return cells


def test_prediction_96_not_optimum_hits_when_128_and_192_match_or_beat():
    sup = {w: [True] * 9 for w in WIDTH_GRID}
    census = {w: [True] * 9 + [False] for w in WIDTH_GRID}
    lh = {w: [True] * 4 for w in WIDTH_GRID}
    cells = _full_cells(sup, census, lh)
    table = sweep_table(cells)
    out = prediction_verdicts(
        table, [{"id": "P1", "check": "96_not_optimum", "claim": "x"}], cells)
    assert out[0]["verdict"] == "HIT"


def test_prediction_96_not_optimum_misses_when_192_loses_a_probe():
    sup = {w: [True] * 9 for w in WIDTH_GRID}
    sup[192] = [True] * 8 + [False]
    census = {w: [True] * 10 for w in WIDTH_GRID}
    lh = {w: [True] * 4 for w in WIDTH_GRID}
    cells = _full_cells(sup, census, lh)
    out = prediction_verdicts(
        sweep_table(cells),
        [{"id": "P1", "check": "96_not_optimum", "claim": "x"}], cells)
    assert out[0]["verdict"] == "MISS"


def test_prediction_64_needs_two_sup_regressions():
    sup = {w: [True] * 9 for w in WIDTH_GRID}
    sup[64] = [False, False] + [True] * 7
    census = {w: [True] * 10 for w in WIDTH_GRID}
    lh = {w: [True] * 4 for w in WIDTH_GRID}
    cells = _full_cells(sup, census, lh)
    out = prediction_verdicts(
        sweep_table(cells),
        [{"id": "P2", "check": "64_loses_two_sup", "claim": "x"}], cells)
    assert out[0]["verdict"] == "HIT"
    assert out[0]["evidence"]["correct_lost"] == 2


def test_prediction_64_misses_on_a_single_regression():
    sup = {w: [True] * 9 for w in WIDTH_GRID}
    sup[64] = [False] + [True] * 8
    census = {w: [True] * 10 for w in WIDTH_GRID}
    lh = {w: [True] * 4 for w in WIDTH_GRID}
    cells = _full_cells(sup, census, lh)
    out = prediction_verdicts(
        sweep_table(cells),
        [{"id": "P2", "check": "64_loses_two_sup", "claim": "x"}], cells)
    assert out[0]["verdict"] == "MISS"


def test_prediction_256_wall_needs_both_a_regression_and_a_wall_rise():
    sup = {w: [True] * 9 for w in WIDTH_GRID}
    census = {w: [True] * 10 for w in WIDTH_GRID}
    census[256] = [False] + [True] * 9
    lh = {w: [True] * 4 for w in WIDTH_GRID}
    cells = _full_cells(sup, census, lh)
    # A census regression alone, with no wall data, is NOT the prediction.
    out = prediction_verdicts(
        sweep_table(cells),
        [{"id": "P3", "check": "256_yarn_wall", "claim": "x"}], cells)
    assert out[0]["verdict"] == "MISS"
    # With the wall rising too, it hits.
    cells[256]["census"]["mean_wall_ms_per_turn"] = 900.0
    cells[96]["census"]["mean_wall_ms_per_turn"] = 400.0
    out = prediction_verdicts(
        sweep_table(cells),
        [{"id": "P3", "check": "256_yarn_wall", "claim": "x"}], cells)
    assert out[0]["verdict"] == "HIT"
    assert out[0]["evidence"]["wall_rose"] is True


def test_prediction_longhorizon_ignores_widths_below_96():
    sup = {w: [True] * 9 for w in WIDTH_GRID}
    census = {w: [True] * 10 for w in WIDTH_GRID}
    lh = {w: [True] * 4 for w in WIDTH_GRID}
    lh[64] = [False] * 4          # below the prediction's floor
    cells = _full_cells(sup, census, lh)
    out = prediction_verdicts(
        sweep_table(cells),
        [{"id": "P4", "check": "longhorizon_4_of_4", "claim": "x"}], cells)
    assert out[0]["verdict"] == "HIT"
    assert 64 not in out[0]["evidence"]


def test_prediction_longhorizon_misses_when_a_width_at_or_above_96_drops():
    sup = {w: [True] * 9 for w in WIDTH_GRID}
    census = {w: [True] * 10 for w in WIDTH_GRID}
    lh = {w: [True] * 4 for w in WIDTH_GRID}
    lh[192] = [True, True, True, False]
    cells = _full_cells(sup, census, lh)
    out = prediction_verdicts(
        sweep_table(cells),
        [{"id": "P4", "check": "longhorizon_4_of_4", "claim": "x"}], cells)
    assert out[0]["verdict"] == "MISS"


def test_an_unknown_check_is_unscored_never_a_silent_hit():
    cells = {96: {"sup": _cell(96, "sup", [True])}}
    out = prediction_verdicts(
        sweep_table(cells),
        [{"id": "PX", "check": "no_such_check", "claim": "x"}], cells)
    assert out[0]["verdict"] == "UNSCORED"


def test_a_prediction_about_a_width_that_did_not_run_is_unscored():
    cells = {96: {"sup": _cell(96, "sup", [True] * 9)}}
    out = prediction_verdicts(
        sweep_table(cells),
        [{"id": "P2", "check": "64_loses_two_sup", "claim": "x"}], cells)
    assert out[0]["verdict"] == "UNSCORED"
