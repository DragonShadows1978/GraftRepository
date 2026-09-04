#!/usr/bin/env python3
"""GRM-RS4 G1 — CPU tests for the live-band row split.

ORDER: ``orders/GRM_RS4_CEILING_REPARTITION.md`` (gate G1: "new CPU tests for
the row-split arithmetic (bands sum to one; boundaries derived)").

NO GPU, NO MODEL, NO ARENA.  Every test here runs on integers and small numpy
arrays, so the partition the arms rely on is pinned by a suite that can run
anywhere.  The functions under test are the ones the GPU module IMPORTS, not
copies of them.

THE THINGS THESE TESTS PIN.

  1. THE BOUNDARIES ARE DERIVED.  ``live_band_bounds`` computes every edge
     from ``S``, ``n_sink``, ``cur_mount_n`` and the three token counts; the
     tests vary each input and assert the edges move with it, so a literal
     row index smuggled into the module would fail rather than pass silently.
  2. THE SUB-BANDS PARTITION THE LIVE BAND.  For any probability row, the four
     sub-bands sum to exactly the live mass the unmodified instrument reports,
     and the sub-bands plus sink plus mount plus learned sink sum to one.
  3. THE THREE UNTOUCHED BANDS STAY UNTOUCHED, so the bit-identity gate has
     something to be true about.
  4. AN OVERRUN RAISES.  A derivation that no longer matches the rows the
     kernel scored is a hard failure, never a clamp — that is what keeps a
     mis-attributed band from being reported as a measurement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_rs4_registration import (  # noqa: E402
    ARMS, BAND_NAMES, LIVE_SUBBANDS, RS4Error,
)
from scripts.grm_rs4_row_split import (  # noqa: E402
    BASE_NAMES, PARTITION_HI, PARTITION_LO, ROW_NAMES, SUBBAND_SUM_ATOL,
    live_band_bounds, mean_over_positions, partition_sums_to_one,
    split_full_row, subbands_sum_to_live,
)

REGISTRATION = ROOT / "artifacts" / "grm_rs4" / "registration.json"

#: The RS3 C5 geometry the registration cross-checks against: 19 sink rows,
#: nothing mounted, 211 fed live tokens, a 36-token harmony prompt, and the
#: 266-row cache the sliding receipt recorded at the first answer position.
RS3_C5_S = 266
RS3_C5_N_SINK = 19
RS3_C5_FED = 211
RS3_C5_QUESTION = 36


def _probs(heads: int, S: int, seed: int = 0) -> np.ndarray:
    """A normalized (heads, S+1) probability row, like the softmax produces.

    The last column is the learned sink, exactly as ``_full_mass`` lays it out.
    Normalizing per head is what makes the partition law meaningful: these are
    the same shape of rows the kernel's softmax hands the observer.
    """
    rng = np.random.default_rng(seed)
    raw = rng.random((heads, S + 1)).astype(np.float32) + 0.01
    return (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)


# ======================================================================
# 1. The boundaries are DERIVED
# ======================================================================
def test_bounds_reproduce_the_rs3_c5_geometry():
    """The registered cross-check, restated as a test.

    266 cache rows, 19 of sink, nothing mounted, 211 fed: the question must
    occupy the 36 rows the tokenizer independently counts, and the answer band
    must be empty at that first answer position.
    """
    bounds = live_band_bounds(
        S=RS3_C5_S, n_sink=RS3_C5_N_SINK, cur_mount_n=0,
        prior_ntok=0, fed_ntok=RS3_C5_FED, question_ntok=RS3_C5_QUESTION)
    assert bounds["live0"] == RS3_C5_N_SINK
    assert bounds["fed_text_band"] == [19, 230]
    assert bounds["question_band"] == [230, 266]
    assert bounds["answer_band"] == [266, 266]
    assert bounds["answer_rows"] == 0
    assert bounds["live_rows"] == RS3_C5_FED + RS3_C5_QUESTION


def test_live0_tracks_the_mount_width():
    """``live0`` is ``n_sink + cur_mount_n`` — ``_full_mass``'s own expression.

    Widening the mount must push the live band start by exactly that much, or
    the sub-bands would be cut out of rows the mount owns.
    """
    base = live_band_bounds(S=300, n_sink=19, cur_mount_n=0, prior_ntok=0,
                            fed_ntok=0, question_ntok=36)
    wider = live_band_bounds(S=300, n_sink=19, cur_mount_n=61, prior_ntok=0,
                             fed_ntok=0, question_ntok=36)
    assert base["live0"] == 19
    assert wider["live0"] == 80
    assert wider["question_band"][0] - base["question_band"][0] == 61
    assert wider["mount_band"] == [19, 80]
    assert base["mount_band"] == [19, 19]


@pytest.mark.parametrize("fed", [0, 1, 61, 211])
def test_fed_width_moves_the_question_band(fed):
    """The question band starts exactly ``fed`` rows later, never a fixed row."""
    bounds = live_band_bounds(S=400, n_sink=19, cur_mount_n=0, prior_ntok=0,
                              fed_ntok=fed, question_ntok=36)
    assert bounds["fed_text_band"] == [19, 19 + fed]
    assert bounds["question_band"] == [19 + fed, 19 + fed + 36]
    assert bounds["fed_text_rows"] == fed


@pytest.mark.parametrize("question", [12, 36, 37, 38])
def test_question_width_moves_the_answer_band(question):
    """The answer band is the REMAINDER: it absorbs any question width."""
    bounds = live_band_bounds(S=400, n_sink=19, cur_mount_n=0, prior_ntok=0,
                              fed_ntok=211, question_ntok=question)
    assert bounds["question_rows"] == question
    assert bounds["answer_band"] == [19 + 211 + question, 400]
    assert bounds["answer_rows"] == 400 - 19 - 211 - question


def test_prior_live_band_is_separate_from_fed_text():
    """A C0-style serve's prior turns get their OWN band.

    Folding prior conversation into ``fed_text`` would inflate the very number
    this order exists to measure, so the split keeps them apart.
    """
    bounds = live_band_bounds(S=400, n_sink=19, cur_mount_n=0, prior_ntok=90,
                              fed_ntok=211, question_ntok=36)
    assert bounds["prior_live_band"] == [19, 109]
    assert bounds["fed_text_band"] == [109, 320]
    assert bounds["question_band"] == [320, 356]
    assert bounds["answer_band"] == [356, 400]


def test_sub_bands_are_contiguous_and_cover_the_live_band():
    """No gaps, no overlaps, and the union is exactly ``[live0, S)``."""
    bounds = live_band_bounds(S=333, n_sink=19, cur_mount_n=17, prior_ntok=7,
                              fed_ntok=100, question_ntok=36)
    spans = [bounds[f"{name}_band"] for name in
             ("prior_live", "fed_text", "question", "answer")]
    assert spans[0][0] == bounds["live_band"][0]
    assert spans[-1][1] == bounds["live_band"][1]
    for left, right in zip(spans, spans[1:]):
        assert left[1] == right[0]
    assert sum(hi - lo for lo, hi in spans) == bounds["live_rows"]


# ======================================================================
# 2. The partition laws
# ======================================================================
def test_sub_bands_sum_to_the_live_band():
    """The four sub-bands add back to the live mass the base tool reports.

    This is what makes the split a RE-PARTITION of RS3's own number rather
    than a second computation that happens to look similar.
    """
    bounds = live_band_bounds(S=RS3_C5_S, n_sink=RS3_C5_N_SINK, cur_mount_n=0,
                              prior_ntok=0, fed_ntok=RS3_C5_FED,
                              question_ntok=RS3_C5_QUESTION)
    row = split_full_row(_probs(8, RS3_C5_S, seed=1), bounds)
    total = sum(row[name] for name in LIVE_SUBBANDS)
    assert abs(row["live_mass"] - total) <= SUBBAND_SUM_ATOL
    assert subbands_sum_to_live(row)


def test_bands_sum_to_one():
    """Sink + mount + the four sub-bands + learned sink ~= 1.0."""
    bounds = live_band_bounds(S=200, n_sink=19, cur_mount_n=40, prior_ntok=11,
                              fed_ntok=50, question_ntok=36)
    row = split_full_row(_probs(8, 200, seed=2), bounds)
    total = sum(row[name] for name in BAND_NAMES)
    assert PARTITION_LO <= total <= PARTITION_HI
    assert partition_sums_to_one(row)


def test_live_mass_is_not_double_counted_in_the_vocabulary():
    """``BAND_NAMES`` must not contain ``live_mass`` beside its own sub-bands.

    Counting both would make every partition sum to roughly two, which is the
    single easiest way for this instrument to be quietly wrong.
    """
    assert "live_mass" not in BAND_NAMES
    assert set(LIVE_SUBBANDS) <= set(BAND_NAMES)
    assert "live_mass" in ROW_NAMES  # kept for the agreement check
    assert set(BASE_NAMES) <= set(ROW_NAMES)


def test_unchanged_bands_match_a_direct_full_mass_style_sum():
    """Sink, mount, live and learned sink come out of RS4's cut UNCHANGED.

    Reproduced here with the same slicing ``_full_mass`` performs, so a change
    to RS4's band helper that perturbed a band it is not supposed to touch
    fails on CPU rather than only at the bit-identity gate.
    """
    S, n_sink, cur_mount_n = 240, 19, 30
    values = _probs(6, S, seed=3)
    bounds = live_band_bounds(S=S, n_sink=n_sink, cur_mount_n=cur_mount_n,
                              prior_ntok=0, fed_ntok=100, question_ntok=36)
    row = split_full_row(values, bounds)
    end = n_sink + cur_mount_n
    assert row["physical_sink_mass"] == pytest.approx(
        float(np.mean(values[:, :n_sink].sum(axis=1))), abs=0.0)
    assert row["mounted_mass"] == pytest.approx(
        float(np.mean(values[:, n_sink:end].sum(axis=1))), abs=0.0)
    assert row["live_mass"] == pytest.approx(
        float(np.mean(values[:, end:S].sum(axis=1))), abs=0.0)
    assert row["learned_sink_mass"] == pytest.approx(
        float(np.mean(values[:, S])), abs=0.0)


def test_empty_sub_band_is_exactly_zero():
    """A zero-width band contributes exactly 0.0, not a near-zero.

    On C2/C3l there is no fed text at all, and the receipt must say 0.0 so a
    reader can tell "this band does not exist here" from "this band was read
    and found small".
    """
    bounds = live_band_bounds(S=100, n_sink=19, cur_mount_n=10, prior_ntok=0,
                              fed_ntok=0, question_ntok=36)
    row = split_full_row(_probs(4, 100, seed=4), bounds)
    assert row["prior_live_mass"] == 0.0
    assert row["fed_text_mass"] == 0.0
    assert row["question_mass"] > 0.0
    assert row["answer_mass"] > 0.0
    assert subbands_sum_to_live(row)


def test_row_carries_the_derived_widths_beside_the_masses():
    """A receipt must show how many rows each mass was read over.

    Without the widths a reader cannot tell a small band from an empty one,
    which is the distinction the whole order turns on.
    """
    bounds = live_band_bounds(S=266, n_sink=19, cur_mount_n=0, prior_ntok=0,
                              fed_ntok=211, question_ntok=36)
    row = split_full_row(_probs(4, 266, seed=5), bounds)
    assert row["fed_text_rows"] == 211
    assert row["question_rows"] == 36
    assert row["answer_rows"] == 0
    assert row["prior_live_rows"] == 0
    assert row["live_rows"] == 247


# ======================================================================
# 3. Failures are raised, never clamped
# ======================================================================
def test_overrunning_derivation_raises():
    """Widths that outrun the scored cache are a hard failure."""
    with pytest.raises(RS4Error, match="overrun"):
        live_band_bounds(S=100, n_sink=19, cur_mount_n=0, prior_ntok=0,
                         fed_ntok=211, question_ntok=36)


def test_mount_overrunning_the_cache_raises():
    with pytest.raises(RS4Error, match="overruns the cache"):
        live_band_bounds(S=50, n_sink=19, cur_mount_n=40, prior_ntok=0,
                         fed_ntok=0, question_ntok=0)


@pytest.mark.parametrize("field", ["prior_ntok", "fed_ntok", "question_ntok"])
def test_negative_widths_raise(field):
    kwargs = dict(S=300, n_sink=19, cur_mount_n=0, prior_ntok=0, fed_ntok=0,
                  question_ntok=36)
    kwargs[field] = -1
    with pytest.raises(RS4Error, match="negative"):
        live_band_bounds(**kwargs)


def test_probability_array_shape_is_checked():
    bounds = live_band_bounds(S=100, n_sink=19, cur_mount_n=0, prior_ntok=0,
                              fed_ntok=0, question_ntok=36)
    with pytest.raises(RS4Error, match="columns"):
        split_full_row(_probs(4, 99), bounds)
    with pytest.raises(RS4Error, match="probability array"):
        split_full_row(np.zeros((2, 3, 4), dtype=np.float32), bounds)


# ======================================================================
# 4. The two-level reduction
# ======================================================================
def test_mean_over_positions_matches_a_hand_reduction():
    """Mean over layers inside a step, then mean over steps — RS2's reduction."""
    bounds = live_band_bounds(S=120, n_sink=19, cur_mount_n=0, prior_ntok=0,
                              fed_ntok=40, question_ntok=36)
    steps = [
        [split_full_row(_probs(4, 120, seed=10 * step + layer), bounds)
         for layer in range(3)]
        for step in range(2)
    ]
    summary = mean_over_positions(steps, ROW_NAMES)
    assert summary["layers_observed"] == 3
    assert summary["answer_positions"] == 2
    for name in ROW_NAMES:
        expected = float(np.mean([
            float(np.mean([float(row[name]) for row in step]))
            for step in steps]))
        assert summary["mean_over_answer_positions"][name] == pytest.approx(
            expected, abs=0.0)
    assert summary["subbands_sum_to_live"]
    assert summary["partition_sums_to_one"]


def test_mean_over_positions_rejects_drifting_layer_counts():
    bounds = live_band_bounds(S=120, n_sink=19, cur_mount_n=0, prior_ntok=0,
                              fed_ntok=40, question_ntok=36)
    row = split_full_row(_probs(4, 120), bounds)
    with pytest.raises(RS4Error, match="drifted"):
        mean_over_positions([[row, row], [row]], ROW_NAMES)


def test_mean_over_positions_with_no_rows_is_not_a_partition():
    summary = mean_over_positions([], ROW_NAMES)
    assert summary["answer_positions"] == 0
    assert not subbands_sum_to_live(summary["mean_over_answer_positions"])
    assert not partition_sums_to_one(summary["mean_over_answer_positions"])


# ======================================================================
# 5. The registration is present and self-consistent
# ======================================================================
@pytest.mark.skipif(not REGISTRATION.is_file(),
                    reason="RS4 registration not written yet")
def test_registration_crosscheck_agrees():
    """The registered derivation cross-check must have passed.

    The registration records that ``266 - 19 - 211`` equals the tokenizer's own
    count of the harbor probe's harmony prompt. If that ever disagrees, the row
    split is not arithmetic over RS3's numbers and no RS4 result means anything.
    """
    payload = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    cross = payload["derivation_crosscheck_against_rs3"]
    assert cross["agree"] is True
    assert cross["implied_question_ntok"] == RS3_C5_QUESTION
    assert cross["independently_tokenized_question_ntok"] == RS3_C5_QUESTION
    assert cross["rs3_live_tokens_after_feed"] == RS3_C5_FED
    assert cross["rs3_sliding_S_min"] == RS3_C5_S


@pytest.mark.skipif(not REGISTRATION.is_file(),
                    reason="RS4 registration not written yet")
def test_registration_geometry_matches_rs3():
    payload = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    geometry = payload["geometry_REGISTERED_BEFORE_ANY_GATE"]
    assert geometry["n_sink"] == RS3_C5_N_SINK
    assert geometry["arena_width"] == 96
    assert geometry["live_shift"] == 115
    assert geometry["agrees_with_rs3_registration"] is True


@pytest.mark.skipif(not REGISTRATION.is_file(),
                    reason="RS4 registration not written yet")
def test_registration_names_the_four_arms_and_the_sub_bands():
    payload = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    assert tuple(payload["arm_order"]) == ARMS
    split = payload["live_band_split_REGISTERED_BEFORE_ANY_GATE"]
    assert tuple(split["sub_bands_in_physical_row_order"]) == LIVE_SUBBANDS
    assert tuple(split["full_band_vocabulary"]) == BAND_NAMES


@pytest.mark.skipif(not REGISTRATION.is_file(),
                    reason="RS4 registration not written yet")
def test_registered_question_lengths_are_all_positive():
    payload = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    questions = payload["question_lengths_REGISTERED_BEFORE_ANY_GATE"]
    assert questions
    for probe_id, entry in questions.items():
        assert entry["question_ntok"] > 0, probe_id
        assert entry["question"]
        assert entry["prompt_sha256"]


@pytest.mark.skipif(not REGISTRATION.is_file(),
                    reason="RS4 registration not written yet")
def test_registration_carries_rs3s_own_framing_of_the_pair():
    """RS3's 0.74-vs-0.41 numbers are carried, not retyped.

    The report has to place RS4's residual beside RS3's, so the registration
    reads RS3's published pair off its own results file.
    """
    payload = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    pairs = payload["rs3_ceiling_pair_AS_RS3_FRAMED_IT"]
    harbor = pairs["sup_harbor_restatement:registered"]
    assert harbor["rs3_C5_live_mass_full_layers"] == pytest.approx(0.742,
                                                                  abs=5e-4)
    assert harbor["rs3_C3l_mounted_mass_full_layers"] == pytest.approx(
        0.413, abs=5e-4)
    assert harbor["rs3_framing_residual"] == pytest.approx(0.329, abs=5e-4)
    assert harbor["rs3_C5_mounted_mass_full_layers"] == 0.0
