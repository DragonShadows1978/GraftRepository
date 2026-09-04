#!/usr/bin/env python3
"""GRM-RS4 G1 — CPU tests for the live-row baseline and the two gates.

ORDER: ``orders/GRM_RS4_CEILING_REPARTITION.md`` (gate G1).

Companion to ``test_grm_rs4_row_split.py``, which pins the band ARITHMETIC.
This file pins the two things that sit either side of it:

  * THE BASELINE DERIVATION — how many live rows were already in the cache
    when a serve's prompt went in.  This is the input RS4's first C0 run got
    wrong, and that failure is pinned here as a regression test so it cannot
    come back;
  * THE GATE LOGIC — G2's bit-equality comparison and G3's residual, exercised
    on synthetic rows so the verdicts are checkable without a card.

NO GPU.  ``RowSplitMassObserver``'s derivation methods are exercised against a
stub arena: they read only ``n_sink``, ``cur_mount_n`` and the recorded
pre-forward position, so a ``SimpleNamespace`` is enough and no model is built.
The METHODS are the real ones, bound to the stub, so these tests pin the
shipped code rather than a copy of it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_rs4_registration import RS4Error  # noqa: E402
from scripts.grm_rs4_results import (  # noqa: E402
    G2_BANDS, gate_g2, gate_g3, partition_audit, residual_table,
)

REGISTRATION = ROOT / "artifacts" / "grm_rs4" / "registration.json"
RESULTS = ROOT / "artifacts" / "grm_rs4" / "grm_rs4_results.json"


class _StubObserver:
    """Just enough of ``RowSplitMassObserver`` to exercise the derivation."""

    def __init__(self, *, n_sink: int, cur_mount_n: int, fed_ntok: int,
                 question_ntok: int, pos_before_forward: int | None):
        from scripts.grm_rs4_ceiling_gpu import RowSplitMassObserver

        self.arena = SimpleNamespace(n_sink=int(n_sink),
                                     cur_mount_n=int(cur_mount_n))
        self.fed_ntok = int(fed_ntok)
        self.question_ntok = int(question_ntok)
        self._pos_before_forward = pos_before_forward
        self.baseline_ntok = None
        self.baseline_receipt = None
        self.prior_ntok = None
        self.bounds_seen: list[dict] = []
        self._cls = RowSplitMassObserver
        # Bind the REAL methods onto this instance, so ``_bounds_for``'s own
        # ``self._baseline_from_first_forward(...)`` call resolves to the real
        # one too. Binding rather than copying is the point: these tests pin
        # the shipped code, not a re-implementation of it.
        self._baseline_from_first_forward = (
            RowSplitMassObserver._baseline_from_first_forward.__get__(self))
        self._bounds_for = RowSplitMassObserver._bounds_for.__get__(self)

    def baseline(self, S: int) -> int:
        return self._baseline_from_first_forward(int(S))

    def bounds(self, S: int) -> dict:
        return self._bounds_for(int(S))


# ======================================================================
# 1. The baseline derivation
# ======================================================================
def test_baseline_reproduces_the_rs3_c5_feed():
    """``S - live0 - question_ntok`` recovers the fed token count.

    RS3's C5 harbor receipt: 266 cache rows, 19 of sink, nothing mounted, a
    36-token harmony prompt. The baseline must come out as 211 — exactly the
    ``live_tokens_after_feed`` that receipt records.
    """
    obs = _StubObserver(n_sink=19, cur_mount_n=0, fed_ntok=211,
                        question_ntok=36, pos_before_forward=211)
    assert obs.baseline(266) == 211
    receipt = obs.baseline_receipt
    assert receipt["baseline_ntok_derived"] == 211
    assert receipt["fed_ntok_from_feed_receipt"] == 211
    assert receipt["prior_ntok"] == 0
    assert receipt["two_derivations_agree"] is True


def test_baseline_is_zero_on_a_cleared_mounted_arm():
    """C2/C3l: ``_clear_boat`` runs first, so the live band is question only.

    At the prompt forward the cache is sink + mount + question and nothing
    else, which is what makes the mounted arms' live band a question-and-answer
    band rather than a residual on the mount.
    """
    obs = _StubObserver(n_sink=19, cur_mount_n=61, fed_ntok=0,
                        question_ntok=36, pos_before_forward=0)
    assert obs.baseline(19 + 61 + 36) == 0
    bounds = obs.bounds(19 + 61 + 36)
    assert bounds["fed_text_rows"] == 0
    assert bounds["prior_live_rows"] == 0
    assert bounds["question_rows"] == 36
    assert bounds["answer_rows"] == 0


def test_baseline_does_not_use_pos_at_observer_entry():
    """REGRESSION TEST for RS4's first C0 failure.

    ``ArenaCache.eb1_begin_turn``'s ephemeral branch clears the boat at every
    production turn-open, so live rows present when the observer was entered
    may not be in the cache the kernel then scores. RS4's first C0 run read
    ``arena.pos`` at entry, derived 48 phantom prior rows, and raised:
    ``live0=83 prior=48 fed=0 question=37 ends at 168 but S=120``.

    Here the cache holds ONLY the question (S = live0 + 37), which is exactly
    that situation. The baseline must be 0 — the number the cache supports —
    and no phantom rows may be invented from a stale counter.
    """
    obs = _StubObserver(n_sink=19, cur_mount_n=64, fed_ntok=0,
                        question_ntok=37, pos_before_forward=0)
    assert obs.baseline(83 + 37) == 0
    bounds = obs.bounds(120)
    assert bounds["live0"] == 83
    assert bounds["question_band"] == [83, 120]
    assert bounds["answer_band"] == [120, 120]
    assert obs.baseline_receipt["why_not_pos_at_observer_entry"]


def test_baseline_is_cached_after_the_first_forward():
    """Later forwards grow ``S`` with answer rows; the baseline must not move."""
    obs = _StubObserver(n_sink=19, cur_mount_n=0, fed_ntok=211,
                        question_ntok=36, pos_before_forward=211)
    first = obs.bounds(266)
    assert first["answer_rows"] == 0
    obs._pos_before_forward = 247          # after the prompt forward
    later = obs.bounds(280)
    assert later["fed_text_band"] == first["fed_text_band"]
    assert later["question_band"] == first["question_band"]
    assert later["answer_rows"] == 14


def test_two_derivations_disagreeing_raises():
    """``S``-derived and ``arena.pos``-derived baselines must agree.

    They are the same quantity by two routes. A disagreement means the row
    split would cut the live band at rows the kernel did not score, and that
    is a hard failure rather than a preference for one route.
    """
    obs = _StubObserver(n_sink=19, cur_mount_n=0, fed_ntok=211,
                        question_ntok=36, pos_before_forward=999)
    with pytest.raises(RS4Error, match="baseline disagrees"):
        obs.baseline(266)


def test_question_overrunning_the_cache_raises():
    obs = _StubObserver(n_sink=19, cur_mount_n=0, fed_ntok=0,
                        question_ntok=400, pos_before_forward=0)
    with pytest.raises(RS4Error, match="question alone overruns"):
        obs.baseline(100)


def test_feed_receipt_larger_than_the_baseline_raises():
    """A feed claiming more tokens than the cache holds is a contradiction."""
    obs = _StubObserver(n_sink=19, cur_mount_n=0, fed_ntok=500,
                        question_ntok=36, pos_before_forward=211)
    with pytest.raises(RS4Error, match="feed and the cache disagree"):
        obs.bounds(266)


# ======================================================================
# 2. G2 — the bit-equality gate
# ======================================================================
def _c0_row(**over):
    row = {
        "arm": "C0", "probe_id": "p", "variant": "registered",
        "served": "answer", "correct": True, "registered_probe": True,
        "mounted_mass_full_layers": 0.25,
        "live_mass_full_layers": 0.40,
        "sink_mass_full_layers": 0.05,
        "learned_sink_mass_full_layers": 0.30,
        "mounted_mass_sliding_layers": 0.02,
        "live_mass_sliding_layers": 0.50,
    }
    row.update(over)
    return row


def _registration_with(expected):
    return {"rs3_c0_rows_RS4_MUST_REPRODUCE": expected,
            "rs3_ceiling_pair_AS_RS3_FRAMED_IT": {}}


def test_g2_passes_on_float_equal_rows():
    row = _c0_row()
    reg = _registration_with({"p:registered": dict(row)})
    result = gate_g2(reg, [row])
    assert result["verdict"] == "PASS"
    assert result["drift"] == []
    assert result["rows_checked"] == 1
    assert all(result["table"][0]["bands_bit_equal"].values())


@pytest.mark.parametrize("band", G2_BANDS)
def test_g2_fails_on_any_band_drifting(band):
    """FLOAT equality, not a tolerance: a last-bit change is drift."""
    expected = _c0_row()
    observed = _c0_row(**{band: float(expected[band]) * (1.0 + 2.0e-16)})
    assert observed[band] != expected[band]
    reg = _registration_with({"p:registered": expected})
    result = gate_g2(reg, [observed])
    assert result["verdict"] == "FAIL"
    assert result["drift"] == ["p:registered"]
    assert result["table"][0]["bands_bit_equal"][band] is False


def test_g2_fails_on_a_different_served_string():
    expected = _c0_row()
    reg = _registration_with({"p:registered": expected})
    result = gate_g2(reg, [_c0_row(served="something else")])
    assert result["verdict"] == "FAIL"
    assert result["table"][0]["served_equal"] is False


def test_g2_fails_on_a_flipped_correctness():
    expected = _c0_row()
    reg = _registration_with({"p:registered": expected})
    result = gate_g2(reg, [_c0_row(correct=False)])
    assert result["verdict"] == "FAIL"
    assert result["table"][0]["correct_equal"] is False


def test_g2_reports_a_missing_row_rather_than_skipping_it():
    reg = _registration_with({"p:registered": _c0_row()})
    result = gate_g2(reg, [])
    assert result["verdict"] == "FAIL"
    assert "no RS4 C0 row" in result["drift"][0]


# ======================================================================
# 3. G3 — the residual
# ======================================================================
def _arm_row(arm, **over):
    row = {
        "arm": arm, "probe_id": "sup_harbor_restatement",
        "variant": "registered", "session_id": "s", "correct": True,
        "registered_probe": True,
        "mounted_mass_full_layers": 0.0, "live_mass_full_layers": 0.0,
        "sink_mass_full_layers": 0.03,
        "learned_sink_mass_full_layers": 0.23,
        "prior_live_mass_full_layers": 0.0,
        "fed_text_mass_full_layers": 0.0,
        "question_mass_full_layers": 0.0,
        "answer_mass_full_layers": 0.0,
    }
    row.update(over)
    return row


def _residual_rows():
    ceiling = _arm_row(
        "C5", live_mass_full_layers=0.74, fed_text_mass_full_layers=0.44,
        question_mass_full_layers=0.15, answer_mass_full_layers=0.15)
    mount = _arm_row(
        "C3l", mounted_mass_full_layers=0.41, live_mass_full_layers=0.29,
        question_mass_full_layers=0.13, answer_mass_full_layers=0.16)
    return [ceiling, mount]


def test_residual_table_reports_both_framings():
    """RS3's subtraction and RS4's, side by side on one row."""
    table = residual_table(_registration_with({}), _residual_rows())
    assert len(table) == 1
    row = table[0]
    assert row["rs3_framing_residual"] == pytest.approx(0.74 - 0.41)
    assert row["rs4_residual_fed_minus_mount"] == pytest.approx(0.44 - 0.41)
    assert row["c5_question_plus_answer"] == pytest.approx(0.30)
    assert row["c5_question_plus_answer_minus_c3l_live_band"] == pytest.approx(
        0.30 - 0.29)


def test_residual_table_ignores_non_registered_rows():
    """A probe every arm serves identically carries no arm contrast."""
    rows = [dict(r, registered_probe=False) for r in _residual_rows()]
    assert residual_table(_registration_with({}), rows) == []


@pytest.mark.skipif(not REGISTRATION.is_file(),
                    reason="RS4 registration not written yet")
def test_g3_verdicts_follow_the_registered_thresholds():
    """The three registered predictions, judged on synthetic refuser rows."""
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    predictions = gate_g3(registration, _residual_rows())["predictions"]
    # fed 0.44 is inside 0.45 +/- 0.05
    assert predictions["P2_c5_fed_text_mass"]["verdict"] == "HIT"
    # residual 0.03 is at or below 0.08
    assert predictions["P3_residual"]["verdict"] == "HIT"
    # question 0.15 is outside 0.29 +/- 0.05
    assert predictions[
        "P1_c5_question_row_mass_matches_mounted_live_band"][
            "verdict"] == "MISS"


@pytest.mark.skipif(not REGISTRATION.is_file(),
                    reason="RS4 registration not written yet")
def test_g3_reports_inconclusive_between_the_registered_branches():
    """A residual between 0.08 and 0.20 is neither branch, and says so."""
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    ceiling, mount = _residual_rows()
    ceiling["fed_text_mass_full_layers"] = 0.56   # residual 0.15
    result = gate_g3(registration, [ceiling, mount])
    assert result["predictions"]["P3_residual"]["verdict"].startswith(
        "INCONCLUSIVE")


@pytest.mark.skipif(not REGISTRATION.is_file(),
                    reason="RS4 registration not written yet")
def test_g3_reports_the_mechanism_branch_when_the_residual_is_large():
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    ceiling, mount = _residual_rows()
    ceiling["fed_text_mass_full_layers"] = 0.70   # residual 0.29
    result = gate_g3(registration, [ceiling, mount])
    assert "mechanism question stands" in result["predictions"][
        "P3_residual"]["verdict"]


# ======================================================================
# 4. The partition audit
# ======================================================================
def test_partition_audit_flags_a_row_whose_sub_bands_do_not_close():
    good = {"subbands_sum_to_live_full": True,
            "partition_sums_to_one_full": True,
            "recut_vs_base_abs_delta": {"full_attention": {"live_mass": 0.0}}}
    bad = dict(good, subbands_sum_to_live_full=False)
    assert partition_audit([good])["subbands_sum_to_live_all_rows"] is True
    assert partition_audit([good, bad])[
        "subbands_sum_to_live_all_rows"] is False
    assert partition_audit([good])["recut_vs_base_max_abs_delta"] == 0.0


# ======================================================================
# 5. The measured results, if they are on disk
# ======================================================================
@pytest.mark.skipif(not RESULTS.is_file(), reason="RS4 results not built yet")
def test_measured_g2_passed_and_the_recut_was_exact():
    """The shipped receipt's own verdicts, asserted rather than described."""
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert payload["G2_c0_reproduces_rs3"]["verdict"] == "PASS"
    assert payload["G2_c0_reproduces_rs3"]["drift"] == []
    audit = payload["partition_audit"]
    assert audit["subbands_sum_to_live_all_rows"] is True
    assert audit["partition_sums_to_one_all_rows"] is True
    # The recut reproduced every UNTOUCHED band exactly, not approximately.
    assert audit["recut_vs_base_max_abs_delta"] == 0.0
    assert audit["recut_vs_base_comparisons"] > 0


@pytest.mark.skipif(not RESULTS.is_file(), reason="RS4 results not built yet")
def test_measured_sub_bands_close_on_every_table_row():
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    for row in payload["table"]:
        assert row["subbands_sum_to_live_full"] is True, row["probe_id"]
        assert row["partition_sums_to_one_full"] is True, row["probe_id"]
