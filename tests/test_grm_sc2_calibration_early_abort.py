"""GRM-SC2 — wider calibration (rule unchanged) + early abort at the fire token.

ORDER: ``orders/GRM_SC2_CALIBRATION_EARLY_ABORT.md``.
REGISTRATION: ``artifacts/grm_sc2/registration.json``.

WHAT THESE TESTS PIN.

Part A (CPU, from persisted receipts):

1. RULE EQUALITY.  SC2's minima-based evaluator agrees with the FROZEN DET1
   ``fit_thresholds`` / ``detector_decision`` exactly -- same envelope value by
   float equality, same fire verdict and same fire index on every frozen race
   row at every point of the registered sweep grid.  This is what licenses
   Part A to rule on rows that persisted a minimum but not a token array.
2. SPLIT DETERMINISM.  The registered provenance split is a pure function of
   what is on disk: repeated runs produce identical set ids in identical order.
3. CANDIDATE NOT READ BY PRODUCTION.  ``registered_threshold`` returns the
   CARRIED race value while ``adopted_by`` is unset, even though the config
   now carries a ``candidate_threshold`` right next to it.

Part B (CPU fixture level; the GPU gates are the real measurement):

4. EARLY ABORT OFF is byte-identical to SC1: no abort is ever raised and the
   decision the turn acts on is the same one SC1 acted on.
5. THE RESUME PATH SERVES THE ORIGINAL TEXT.  An attempt aborted at the fire
   token and then resumed produces the SAME text, byte for byte, as the same
   attempt run without any abort.  This is the property that makes defaulting
   the abort ON safe: a failed demand trip still serves what it served before.
6. The registered flag semantics and the receipt-field contract.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import grm_demand  # noqa: E402
from core.graft_arena import ArenaCache  # noqa: E402
from scripts import grm_sc2_calibration as sc2  # noqa: E402
from scripts.grm_det1_common import (  # noqa: E402
    detector_decision,
    fit_thresholds,
)

CARRIED = 0.3380523274342219


# ===========================================================================
# Part A / 1. Rule equality with the frozen DET1 functions
# ===========================================================================


def test_envelope_rule_reproduces_the_frozen_det1_threshold_exactly():
    """The carried rule, carried: EXACT float equality, not a tolerance.

    If SC2's envelope drifted from ``fit_thresholds`` by even one ULP, the
    'rule unchanged' claim would be false and every downstream number would
    describe a different detector than the one production runs.
    """
    rows = sc2._read_jsonl(sc2.RACE_CALIBRATION_ROWS)
    det1 = float(fit_thresholds(rows)["D-NGH"]["threshold"])
    ours = sc2.envelope_threshold([
        sc2.ServedRow(
            row_id=str(row["row_id"]), probe_id=str(row["fixture_id"]),
            min_mass=min(sc2._race_token_masses(row)),
            provenance="race_calibration_lived", source="test")
        for row in rows if row.get("variant") == "served"])
    assert ours == det1
    assert ours == CARRIED


def test_minima_verdict_equals_detector_decision_on_every_frozen_row():
    """``min(tokens) < t`` IS ``detector_decision(row, "D-NGH", t)``.

    Checked on every frozen race row at every registered sweep point.  This
    equality is load-bearing: SC1.1 and SC1.2 persisted per-turn minima rather
    than token arrays, and without this pin Part A would be asserting a verdict
    it could not derive.
    """
    result = sc2.rule_equality_check()
    assert result["equality_holds"] is True
    assert result["mismatch_count"] == 0
    assert result["mismatches"] == []
    assert result["decisions_checked"] == (
        result["rows_checked"] * result["grid_points"])
    assert result["envelope_exact_float_equality"] is True


def test_fire_index_matches_first_strictly_below_on_frozen_rows():
    """Where DET1 fires, its index is the FIRST strictly-below position.

    Pinned directly against ``detector_decision`` rather than inferred, since
    the early abort fires on exactly this index and must not fire one token
    early or late.
    """
    checked = 0
    for row in (sc2._read_jsonl(sc2.RACE_CALIBRATION_ROWS)
                + sc2._read_jsonl(sc2.RACE_EVAL_ROWS)):
        masses = sc2._race_token_masses(row)
        for point in sc2.sweep_grid():
            fired, index = detector_decision(
                row, "D-NGH", {"D-NGH": {"threshold": float(point)}})
            if not fired:
                assert sc2._first_below(masses, point) is None
                continue
            assert index == sc2._first_below(masses, point)
            checked += 1
    assert checked > 0


def test_equality_at_the_threshold_is_not_a_fire():
    """``trigger_if_strictly_below``: the fitting turn sits exactly ON the line."""
    assert sc2.fires(CARRIED, CARRIED) is False
    assert sc2.fires(CARRIED - 1e-15, CARRIED) is True
    assert sc2.fires(CARRIED + 1e-15, CARRIED) is False


# ===========================================================================
# Part A / 2. Split determinism
# ===========================================================================


def _split_inputs():
    inv = sc2.inventory()
    return inv, sc2.deduplicate(inv)


def test_registered_split_is_deterministic_across_repeated_runs():
    first = sc2.registered_split(*_split_inputs())
    second = sc2.registered_split(*_split_inputs())
    assert [r.row_id for r in first["calibration"]] == [
        r.row_id for r in second["calibration"]]
    assert [r.row_id for r in first["held_out"]] == [
        r.row_id for r in second["held_out"]]
    assert [r.min_mass for r in first["calibration"]] == [
        r.min_mass for r in second["calibration"]]


def test_split_sets_are_non_empty_and_match_the_registration():
    registration = json.loads(
        (ROOT / "artifacts" / "grm_sc2" / "registration.json").read_text())
    split = sc2.registered_split(*_split_inputs())
    assert split["calibration"], "calibration set must be non-empty"
    assert split["held_out"], "held-out set must be non-empty"
    registered = registration["PART_A"][
        "SPLIT_RULE_REGISTERED_BEFORE_COMPUTATION"]
    assert len(split["calibration"]) == registered["calibration_set_n"]
    assert len(split["held_out"]) == registered["held_out_set_n"]
    assert sorted(r.row_id for r in split["held_out"]) == sorted(
        registered["held_out_set_ids"])


def test_held_out_contains_the_current_false_fire():
    """The order requires ``sup_solace_fresh`` on the held-out side."""
    split = sc2.registered_split(*_split_inputs())
    assert "sup_solace_fresh" in {row.probe_id for row in split["held_out"]}


def test_sc1_2_arm0_rows_fold_into_the_race_rows_losslessly():
    """De-duplication is measured, not assumed: every delta must be 0.0."""
    dedup = sc2.deduplicate(sc2.inventory())
    assert dedup["sc1_2_folded_count"] == 7
    assert dedup["sc1_2_kept_as_distinct"] == []
    for fold in dedup["sc1_2_folds"]:
        assert fold["delta"] == 0.0


def test_the_solace_instrument_disagreement_is_carried_not_collapsed():
    """Two lived instruments, two numbers, both kept.

    ``sup_solace_fresh`` measures 0.3181... on the lived battery and
    0.29139... under the SC1.1 G2 same-process fork.  Collapsing them would
    silently pick one, and it is the single lowest served number in the whole
    inventory -- the one the envelope is most sensitive to.
    """
    split = sc2.registered_split(*_split_inputs())
    solace = sorted(
        row.min_mass for row in split["held_out"]
        if row.probe_id == "sup_solace_fresh")
    assert solace == [0.29139505078395206, 0.3181141105790933]


def test_sweep_grid_is_the_registered_grid_and_nothing_else():
    grid = sc2.sweep_grid()
    assert len(grid) == 41
    assert grid[0] == 0.05
    assert grid[-1] == 0.45
    assert all(round(b - a, 10) == 0.01 for a, b in zip(grid, grid[1:]))


def test_planted_rows_are_all_exactly_zero_mass():
    """Why recall is near-trivial, stated as a test rather than a footnote.

    A withheld node is not mounted, so there is no mounted band to read and
    the mass is exactly 0.0. Recall is therefore 'is the line above zero',
    not evidence of fine discrimination -- and the receipt says so.
    """
    planted = sc2.inventory()["planted"]
    assert planted
    assert {row.min_mass for row in planted} == {0.0}


# ===========================================================================
# Part A / 3. The candidate is recorded but NOT read by production
# ===========================================================================


def test_production_reads_threshold_not_candidate_threshold():
    """The whole adoption contract, in one assertion.

    The config carries both numbers.  Production must return the CARRIED one
    while ``adopted_by`` is unset -- which this order does not set.
    """
    config = grm_demand.load_registered()
    assert "candidate_threshold" in config
    assert config["candidate_threshold"] != config["threshold"]
    assert not config.get("adopted_by")
    assert grm_demand.registered_threshold() == config["threshold"]
    assert grm_demand.registered_threshold() != config["candidate_threshold"]
    assert grm_demand.registered_threshold() == CARRIED


def test_candidate_is_only_read_once_adopted_by_is_set(tmp_path):
    """The gate field is the ONLY thing that switches production over."""
    config = dict(grm_demand.load_registered())
    path = tmp_path / "demand.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert grm_demand.registered_threshold(path) == config["threshold"]

    config["adopted_by"] = "David, order GRM-SC3 (hypothetical)"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert grm_demand.registered_threshold(path) == config["candidate_threshold"]


def test_adopted_by_without_a_candidate_is_refused(tmp_path):
    """A half-applied adoption edit fails loudly instead of falling back."""
    config = dict(grm_demand.load_registered())
    config.pop("candidate_threshold", None)
    config["adopted_by"] = "someone"
    path = tmp_path / "demand.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(grm_demand.DemandError, match="candidate_threshold"):
        grm_demand.registered_threshold(path)


def test_candidate_provenance_names_the_rule_and_the_calibration_set():
    config = grm_demand.load_registered()
    provenance = config["candidate_threshold_provenance"]
    assert provenance["rule"].startswith(
        "minimum_of_served_turn_minima_strict_envelope")
    assert provenance["calibration_set_n"] == len(
        provenance["calibration_set_ids"])
    assert provenance["order"].endswith("GRM_SC2_CALIBRATION_EARLY_ABORT.md")
    assert config["candidate_threshold_status"].startswith(
        "REPORTED, NOT ADOPTED")
    assert config["adoption_contract"][
        "adopted_by_is_NOT_set_by_this_order"] is True


def test_the_candidate_is_the_envelope_over_the_calibration_set():
    """The reported number is the rule applied to the registered set. No fudge."""
    split = sc2.registered_split(*_split_inputs())
    candidate = sc2.envelope_threshold(split["calibration"])
    config = grm_demand.load_registered()
    assert candidate == config["candidate_threshold"]
    assert candidate == min(row.min_mass for row in split["calibration"])


# ===========================================================================
# Part B / the flag
# ===========================================================================


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF", "False"])
def test_early_abort_explicit_false_tokens_turn_it_off(value):
    assert grm_demand.early_abort_enabled(
        environ={grm_demand.ENV_EARLY_ABORT: value}) is False


@pytest.mark.parametrize("value", ["", "1", "true", "on", "banana", "maybe"])
def test_early_abort_defaults_on_including_unknown_tokens(value):
    """Registered default ON; an unknown token fails TO the default.

    Deliberately the opposite direction from ``GRM_DEMAND_NGH``, because this
    switch cannot change what the turn serves -- only how many tokens are
    generated getting there.
    """
    assert grm_demand.early_abort_enabled(
        environ={grm_demand.ENV_EARLY_ABORT: value}) is True


def test_early_abort_defaults_on_when_unset():
    assert grm_demand.early_abort_enabled(environ={}) is True


def test_early_abort_explicit_argument_beats_the_environment():
    assert grm_demand.early_abort_enabled(
        True, environ={grm_demand.ENV_EARLY_ABORT: "0"}) is True
    assert grm_demand.early_abort_enabled(False, environ={}) is False


def test_demand_flag_still_fails_closed_to_off():
    """SC2 must not have loosened SC1's switch while adding its own."""
    assert grm_demand.demand_enabled(environ={}) is False
    assert grm_demand.demand_enabled(
        environ={grm_demand.ENV_NAME: "banana"}) is False
    assert grm_demand.demand_enabled(
        environ={grm_demand.ENV_NAME: "1"}) is True


def test_registered_config_documents_the_early_abort_default():
    flag = grm_demand.load_registered()["early_abort_flag"]
    assert flag["name"] == grm_demand.ENV_EARLY_ABORT
    assert flag["default"].startswith("ON")
    assert flag["off_path"] == "byte-identical to SC1 behaviour"


def test_demand_abort_is_not_an_ordinary_exception():
    """It must survive a broad ``except Exception`` on the generation path.

    A control-flow signal a stray handler can swallow would become a silent
    behaviour change, which is the one failure mode this seam cannot have.
    """
    assert issubclass(grm_demand.DemandAbort, BaseException)
    assert not issubclass(grm_demand.DemandAbort, Exception)


# ===========================================================================
# Part B / a real arena with the real abort/resume seam
# ===========================================================================


class _ResumeArena:
    """A deterministic greedy generator carrying the real ``_attempt``.

    ``_forward`` returns a one-hot row whose argmax is a pure function of how
    many forwards have run, so the "model" is deterministic and the un-aborted
    and resumed runs are directly comparable.  Everything ``_attempt`` touches
    is present; nothing else is.
    """

    PAYLOAD = ()

    _attempt = ArenaCache._attempt
    _finish_attempt = ArenaCache._finish_attempt
    _suspend_attempt = ArenaCache._suspend_attempt
    _resume_attempt = ArenaCache._resume_attempt

    def __init__(self):
        self.caches = "cache"
        self.pos = 0
        self.live_segs = []
        self.cur_mounts = []
        self.cur_mount_n = 0
        self.grafts = []
        self.cache_deposits = False
        self.committed = 0
        self.m = SimpleNamespace(layers=[SimpleNamespace(
            self_attn=SimpleNamespace(
                layer_type="full_attention", attention_mode="standard",
                layer_idx=0))])

    def _resolve_revision_mounts(self, picks):
        return list(picks)

    def _ensure_h(self, picks):
        return None

    def _telemetry_enabled(self):
        return False

    def encode(self, text):
        return [7, 8, 9]

    def _format_step_prompt(self, user_text):
        return str(user_text)

    def _format_step_turn(self, user_text, txt):
        return f"{user_text}|{txt}"

    def decode(self, ids):
        return " ".join(str(int(i)) for i in ids)

    def evict(self):
        return 0

    def _cache_len(self):
        return 32

    def _bump_cuda_gqa_epoch(self):
        return None

    def deposit(self, text):
        return 0

    def swap(self, picks):
        self.cur_mounts = list(picks)

    def _forward(self, ids, last_only=True):
        self.committed += 1
        row = np.zeros(64, dtype=np.float32)
        # Position-dependent: any divergence in the resume path shows up as a
        # DIFFERENT token rather than an accidental match.
        row[(self.committed * 3) % 64] = 1.0
        return row


class _ScriptedObserver:
    """The real abort decision, driven by a scripted mass sequence.

    This wraps ``arena._forward`` exactly the way ``DemandObserver`` does --
    append the row, then raise ``DemandAbort`` on the first mass strictly
    below the line -- so ``_attempt``'s catch/suspend seam is the real one.
    Only the attention arithmetic is replaced, because a CPU fixture has no
    attention operands to compute it from.
    """

    def __init__(self, arena, masses, threshold, *, early_abort):
        self.arena = arena
        self.queue = list(masses)
        self.threshold = float(threshold)
        self.early_abort = bool(early_abort)
        self.aborted_at = None
        self.records = []

    def __enter__(self):
        self._original = self.arena._forward

        def forward_wrapper(ids, last_only=True):
            logits = self._original(ids, last_only=last_only)
            mass = self.queue.pop(0) if self.queue else 1.0
            index = len(self.records)
            self.records.append({
                "token_index": index,
                "prediction_token_id": int(np.asarray(logits).argmax()),
                "mounted_mass": float(mass),
            })
            if (self.early_abort and self.aborted_at is None
                    and float(mass) < self.threshold):
                self.aborted_at = int(index)
                raise grm_demand.DemandAbort(index, float(mass))
            return logits

        self.arena._forward = forward_wrapper
        return self

    def __exit__(self, *exc):
        self.arena._forward = self._original
        return False

    def rows(self):
        return list(self.records)


@pytest.fixture(autouse=True)
def _no_injection_teardown(monkeypatch):
    """``clear_injection`` walks real attention modules; stub it out.

    These tests are about the abort/resume seam, not about graft unmounting.
    The call is pinned as HAPPENING (see
    ``test_abort_at_position_zero_still_clears_the_bootstrap_injection``);
    what it does to a real model is SC1's business and is covered there.
    """
    calls: list[object] = []
    monkeypatch.setattr(
        "core.graft_arena.kv_graft.clear_injection",
        lambda model, *a, **k: calls.append(model))
    return calls


def _run_attempt(masses, *, early_abort, ngen=6):
    arena = _ResumeArena()
    observer = _ScriptedObserver(
        arena, masses, CARRIED, early_abort=early_abort)
    with observer:
        out = arena._attempt("q", [], ngen, False, [], defer_memory=False)
    return arena, observer, out


def _refill(suspended, observer):
    """What the demand block does: recover the emitted ids from the rows."""
    rows = observer.rows()
    index = int(suspended["abort_token_index"])
    suspended["out"] = [
        int(row["prediction_token_id"]) for row in rows[:index + 1]]
    suspended["tokens_generated_before_abort"] = len(suspended["out"])
    return suspended


def test_abort_suspends_at_the_first_strictly_below_token():
    masses = [0.9, 0.9, 0.1, 0.05, 0.9, 0.9, 0.9]
    _arena, observer, out = _run_attempt(masses, early_abort=True)
    assert isinstance(out, dict)
    assert out["grm_sc2_suspended_attempt"] is True
    # FIRST strictly-below, not the deepest: token 2, though token 3 is lower.
    assert out["abort_token_index"] == 2
    assert observer.aborted_at == 2


def test_abort_does_not_fire_on_equality():
    """``trigger_if_strictly_below``: equality is not a fire, mid-generation too."""
    masses = [0.9, CARRIED, 0.9, 0.9, 0.9, 0.9, 0.9]
    _arena, observer, out = _run_attempt(masses, early_abort=True)
    assert isinstance(out, tuple)
    assert observer.aborted_at is None


def test_flag_off_never_aborts_and_returns_the_ordinary_pair():
    """OFF generates the whole answer, exactly as SC1 did."""
    masses = [0.9, 0.1, 0.05, 0.02, 0.9, 0.9, 0.9]
    arena, observer, out = _run_attempt(masses, early_abort=False)
    assert isinstance(out, tuple)
    txt, info = out
    assert isinstance(txt, str) and txt
    assert observer.aborted_at is None
    assert "mounts" in info and "resident" in info
    # The DECISION is unchanged by the flag -- only the token count differs.
    assert grm_demand.decide(
        observer.rows(), CARRIED)["demand_token_index"] == 1


def test_flag_off_is_byte_identical_to_a_run_with_no_observer_at_all():
    """The OFF path must not perturb generation even by an extra forward."""
    masses = [0.9] * 8
    arena_a, _obs, out_a = _run_attempt(masses, early_abort=False)
    arena_b = _ResumeArena()
    out_b = arena_b._attempt("q", [], 6, False, [], defer_memory=False)
    assert out_a[0] == out_b[0]
    assert arena_a.committed == arena_b.committed


def test_resume_produces_byte_identical_text_to_the_unaborted_attempt():
    """THE contract that makes the abort safe to default on.

    Same masses, same model: one run never aborted, one run aborted then
    resumed from the fire token.  The served text must match byte for byte --
    not 'closely', not 'semantically'.  If it did not, a failed demand trip
    would serve something the stack had never served before, which is exactly
    the regression the suspended-attempt fallback exists to prevent.
    """
    masses = [0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9]
    _ref_arena, _ref_obs, ref_out = _run_attempt(masses, early_abort=False)
    ref_txt, _ref_info = ref_out
    assert isinstance(ref_txt, str) and ref_txt

    arena, observer, suspended = _run_attempt(masses, early_abort=True)
    assert isinstance(suspended, dict)
    assert suspended["abort_token_index"] == 2
    _refill(suspended, observer)

    resumed_txt, resumed_info = arena._resume_attempt(
        suspended, False, [], False)
    assert resumed_txt == ref_txt
    assert "mounts" in resumed_info


@pytest.mark.parametrize("fire_at", [0, 1, 2, 3])
def test_resume_is_byte_identical_wherever_the_fire_lands(fire_at):
    """Not one lucky index: every abort position resumes to the same text."""
    masses = [0.9] * 8
    masses[fire_at] = 0.1
    _ref_arena, _ref_obs, (ref_txt, _ref_info) = _run_attempt(
        masses, early_abort=False)
    arena, observer, suspended = _run_attempt(masses, early_abort=True)
    assert suspended["abort_token_index"] == fire_at
    _refill(suspended, observer)
    resumed_txt, _info = arena._resume_attempt(suspended, False, [], False)
    assert resumed_txt == ref_txt


def test_aborted_attempt_generates_strictly_fewer_tokens():
    """The saving is real: fewer forwards, not the same work rearranged."""
    masses = [0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9]
    ref_arena, _obs, _out = _run_attempt(masses, early_abort=False)
    arena, _observer, suspended = _run_attempt(masses, early_abort=True)
    assert isinstance(suspended, dict)
    assert arena.committed < ref_arena.committed
    assert suspended["abort_token_index"] == 1


def test_abort_at_position_zero_still_clears_the_bootstrap_injection(
    _no_injection_teardown,
):
    """The injection fires ONCE, abort or no abort.

    A fire on the prompt forward unwinds before the line that clears it, so
    the abort path clears it itself. Missing that would let the bootstrap
    injection fire a second time on resume.
    """
    masses = [0.1, 0.9, 0.9, 0.9, 0.9, 0.9]
    _arena, _observer, suspended = _run_attempt(masses, early_abort=True)
    assert isinstance(suspended, dict)
    assert suspended["abort_token_index"] == 0
    assert len(_no_injection_teardown) == 1


def test_a_suspended_attempt_runs_no_deposit_or_live_segment_bookkeeping():
    """An answer that does not exist yet must not be recorded as a turn."""
    masses = [0.1, 0.9, 0.9, 0.9, 0.9, 0.9]
    arena, _observer, suspended = _run_attempt(masses, early_abort=True)
    assert isinstance(suspended, dict)
    assert arena.live_segs == []
    assert suspended["abort_token_index"] == 0


def test_resume_runs_the_bookkeeping_exactly_once():
    """One live segment for one served turn, whatever route it took."""
    masses = [0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9]
    arena, observer, suspended = _run_attempt(masses, early_abort=True)
    _refill(suspended, observer)
    assert arena.live_segs == []
    arena._resume_attempt(suspended, False, [], False)
    assert len(arena.live_segs) == 1


def test_a_suspended_handle_cannot_be_resumed_twice():
    masses = [0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9]
    arena, observer, suspended = _run_attempt(masses, early_abort=True)
    _refill(suspended, observer)
    arena._resume_attempt(suspended, False, [], False)
    with pytest.raises(RuntimeError, match="already resumed"):
        arena._resume_attempt(suspended, False, [], False)


def test_resume_refuses_a_handle_with_no_recovered_tokens():
    """Honest failure over a silently different answer."""
    masses = [0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9]
    arena, _observer, suspended = _run_attempt(masses, early_abort=True)
    suspended["out"] = []
    with pytest.raises(grm_demand.DemandError, match="observer rows"):
        arena._resume_attempt(suspended, False, [], False)


def test_resume_refuses_something_that_is_not_a_suspension_handle():
    arena, _observer, _out = _run_attempt([0.9] * 8, early_abort=False)
    with pytest.raises(ValueError, match="suspended attempt handle"):
        arena._resume_attempt({"nope": True}, False, [], False)


def test_suspension_handle_carries_the_registered_receipt_inputs():
    masses = [0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9]
    _arena, observer, suspended = _run_attempt(masses, early_abort=True)
    _refill(suspended, observer)
    assert suspended["abort_token_index"] == 2
    assert suspended["tokens_generated_before_abort"] == 3
    assert suspended["abort_mounted_mass"] == 0.1
    assert suspended["resumed"] is False


# ===========================================================================
# Part B / the observer's own abort behaviour
# ===========================================================================


def _bare_observer(masses, *, early_abort):
    arena = SimpleNamespace(
        m=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(
            layer_type="full_attention", attention_mode="standard",
            layer_idx=0))]),
        n_sink=4, cur_mount_n=8)
    observer = grm_demand.DemandObserver(
        arena, len(masses), CARRIED, early_abort=early_abort)
    observer.expected_full_layers = 1
    return observer


def test_observer_finish_keeps_the_fire_row_on_an_aborted_run():
    """The flush-drop convention applies to COMPLETED attempts only.

    An aborted attempt never runs the final commit forward, so there is no
    unused flush row to drop.  Dropping one anyway would discard the FIRE row
    whenever the fire landed on the last captured position, turning a fired
    turn into an un-fired one.
    """
    observer = _bare_observer([0.9, 0.1], early_abort=True)
    observer.records = [
        {"token_index": 0, "mounted_mass": 0.9, "prediction_token_id": 1},
        {"token_index": 1, "mounted_mass": 0.1, "prediction_token_id": 2},
    ]
    observer.aborted_at = 1
    observer.ngen = 2
    rows = observer.finish()
    assert len(rows) == 2
    assert grm_demand.decide(rows, CARRIED)["demand_fired"] is True
    assert grm_demand.decide(rows, CARRIED)["demand_token_index"] == 1


def test_observer_finish_still_drops_the_flush_row_on_a_completed_run():
    """SC1's convention is untouched where it still applies."""
    observer = _bare_observer([0.9, 0.9], early_abort=False)
    observer.records = [
        {"token_index": 0, "mounted_mass": 0.9, "prediction_token_id": 1},
        {"token_index": 1, "mounted_mass": 0.9, "prediction_token_id": 2},
    ]
    observer.aborted_at = None
    observer.ngen = 1
    assert len(observer.finish()) == 1


def test_observer_default_is_not_aborting():
    """A caller that never asks for the abort never gets one."""
    observer = _bare_observer([0.1], early_abort=False)
    assert observer.early_abort is False
    assert observer.aborted_at is None


# ===========================================================================
# Part B / receipt fields
# ===========================================================================


def test_the_registered_receipt_fields_are_all_demand_prefixed():
    """``core.grm_three_pass`` persists by the ``demand_`` prefix.

    A new field without the prefix would silently vanish from every receipt.
    """
    fields = grm_demand.demand_info_fields(
        supported=True, early_abort=True, abort_token_index=0,
        tokens_generated_before_abort=1, tokens_saved=5,
        resumed_original=False, wall_ms_attempt=1.0, wall_ms_trip=2.0,
        wall_ms_resume=3.0)
    registered = {
        "demand_early_abort", "demand_abort_token_index",
        "demand_tokens_generated_before_abort", "demand_tokens_saved",
        "demand_resumed_original", "demand_wall_ms_attempt",
        "demand_wall_ms_trip", "demand_wall_ms_resume",
    }
    assert registered <= set(fields)
    assert all(key.startswith("demand_") for key in fields)


def test_early_abort_fields_are_absent_when_not_supplied():
    """SC1 receipts must not sprout SC2 keys on paths SC2 never touched."""
    fields = grm_demand.demand_info_fields(supported=True)
    assert "demand_early_abort" not in fields
    assert "demand_abort_token_index" not in fields
    assert "demand_resumed_original" not in fields
    assert "demand_tokens_saved" not in fields


# ===========================================================================
# Part B / the whole demand block, through ``ArenaCache.step``
# ===========================================================================


_QUESTION = "What is the current Solace key value?"

#: The admission profile SC1's step-level tests pin, carried verbatim so the
#: SC2 step tests exercise the same routing shape the SC1 ones did.
_LIVED_PROFILE = {
    "ranking": [3, 2, 1, 0],
    "identified_candidates": [3],
    "identifier_tokens": ["meridian", "docket"],
    "identifier_hit_count": 1,
    "route_margin_1_2": 1.036860985747615,
    "rank_plan": [3],
    "policy_branch": "exactly_one_identifier_decisive_rank1",
    "route_backend": "python",
    "margin_threshold": 0.1385774091529802,
    "rule_sha256":
        "c304609f81475bd2ae3399ad180d3bb00cc5d2b44b1a49db570387c810defb91",
    "route_margin_evaluated": True,
}


def _abort_step_arena(*, answers=None, grounded=None, abort_at=None):
    """A ``step``-shaped arena whose ``_attempt`` honours the abort seam.

    ``_attempt`` is stubbed (a real one needs a real model), but the stub
    reproduces the ONE behaviour Part B depends on: when the demand observer
    would fire at ``abort_at``, it returns a suspension handle instead of a
    finished answer, and it finishes that handle when resumed.  That is the
    contract ``step`` is written against, so this exercises the real
    ``_serve`` / ``_demand_trip`` / resume wiring.
    """
    arena = ArenaCache.__new__(ArenaCache)
    arena.m = SimpleNamespace(layers=[
        SimpleNamespace(self_attn=SimpleNamespace(
            live_shift=None, layer_type="full_attention",
            attention_mode="standard", layer_idx=0)),
        SimpleNamespace(self_attn=SimpleNamespace(
            live_shift=None, layer_type="sliding_attention",
            attention_mode="standard", layer_idx=1)),
    ])
    arena.live_shift = 9
    arena.n_sink = 4
    arena.stop_sequences = ()
    arena.ephemeral = False
    arena.live_segs = []
    arena.topk = 3
    arena.decisive_admission = True
    arena.caches = None
    arena.pos = 0
    arena.cur_mounts = []
    arena.cur_mount_n = 0
    arena.width = 96
    arena.prompt_template = None
    arena.recency_mounts = 0
    arena.grafts = [
        {"text": f"node-{i}", "ntok": n, "kind": "fact"}
        for i, n in enumerate([20, 25, 40, 30])
    ]
    arena._s4_turn = 0
    arena.route = lambda text, *, exclude, limit: [
        i for i in range(4) if i not in set(exclude)]
    arena._rare_tokens = lambda _text: set()
    arena._descent_expand = lambda picks, _kinds, qrare=None: list(picks)
    arena._resolve_revision_mounts = lambda picks: list(picks)
    arena._next_s4_turn = lambda: 1
    arena._bump_cuda_gqa_epoch = lambda: None
    arena._commit_s4_attempt = lambda *_a, **_k: None
    arena.decode = lambda ids: " ".join(str(i) for i in ids)
    answers = dict(answers or {})
    grounded = dict(grounded or {})

    def _grounding(_ans, picks, _q):
        key = tuple(sorted(int(v) for v in picks))
        if key in grounded:
            return bool(grounded[key]), list(picks)
        return bool(picks), list(picks)

    arena._grounding_attribution = _grounding
    arena._grounding_receipt = lambda *_a, **_k: None
    served: list[list[int]] = []
    resumes: list[dict] = []

    def attempt(_question, picks, _ngen, _deposit, _stops,
                defer_memory=False, suspended=None):
        if suspended is not None:
            resumes.append(suspended)
            suspended["resumed"] = True
            key = tuple(sorted(int(v) for v in suspended["picks"]))
            served.append(list(key))
            return answers.get(key, "answer"), {"mounts": [v + 1 for v in key]}
        key = tuple(sorted(int(v) for v in picks))
        arena.cur_mounts = list(picks)
        served.append(list(key))
        if abort_at is not None and key == abort_at[0]:
            return {
                "grm_sc2_suspended_attempt": True,
                "abort_token_index": int(abort_at[1]),
                "abort_mounted_mass": 0.1,
                "tokens_generated_before_abort": int(abort_at[1]) + 1,
                "user_text": _question, "picks": list(picks),
                "seg_start_ntok": 3, "out": [], "cached_out": 0,
                "ngen": int(_ngen), "deposit": bool(_deposit),
                "stops": list(_stops), "defer_memory": bool(defer_memory),
                "resumed": False,
            }
        return answers.get(key, "answer"), {"mounts": [v + 1 for v in key]}

    arena._attempt = attempt
    return arena, served, resumes


def _pin_profile(monkeypatch):
    monkeypatch.setattr(
        "core.graft_arena.decisive_admission_profile",
        lambda _arena, _text, *, exclude, route_limit: {
            **_LIVED_PROFILE,
            "ranking": [i for i in _LIVED_PROFILE["ranking"]
                        if i not in set(exclude)],
            "rank_plan": [i for i in _LIVED_PROFILE["rank_plan"]
                          if i not in set(exclude)]
            or [i for i in _LIVED_PROFILE["ranking"]
                if i not in set(exclude)][:1],
        },
    )


def _pin_observer(monkeypatch, sequence):
    """Scripted observer rows, one row-list per generation attempt."""
    queue = list(sequence)

    class _FakeObserver:
        def __init__(self, arena, ngen, threshold, early_abort=False):
            self._rows = queue.pop(0) if queue else []
            self.early_abort = bool(early_abort)
            self.aborted_at = None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def finish(self):
            return list(self._rows)

    monkeypatch.setattr(grm_demand, "DemandObserver", _FakeObserver)


def _rows(*masses):
    return [
        {"token_index": i, "mounted_mass": float(m),
         "prediction_token_id": 1000 + i}
        for i, m in enumerate(masses)
    ]


def test_step_aborted_attempt_takes_the_trip_and_serves_it_when_grounded(
    monkeypatch,
):
    """The saving case: the wrong answer is never finished.

    The suspended attempt is ABANDONED, ``demand_resumed_original`` is False,
    and what goes out is the grounded trip -- the same text SC1 served, minus
    every token after the fire index.
    """
    monkeypatch.setenv("GRM_DEMAND_NGH", "1")
    monkeypatch.delenv(grm_demand.ENV_EARLY_ABORT, raising=False)
    _pin_profile(monkeypatch)
    _pin_observer(monkeypatch, [_rows(0.9, 0.1), _rows(0.9, 0.8)])
    arena, served, resumes = _abort_step_arena(
        answers={(3,): "wrong", (0,): "RIGHT", (1,): "RIGHT", (2,): "RIGHT"},
        abort_at=((3,), 1),
    )
    ans, info = arena.step(_QUESTION, ngen=4, deposit=False, max_trips=1)
    assert ans == "RIGHT"
    assert info["demand_served"] == "demand_trip"
    assert info["demand_trip_taken"] is True
    assert info["demand_early_abort"] is True
    assert info["demand_abort_token_index"] == 1
    assert info["demand_tokens_generated_before_abort"] == 2
    assert info["demand_resumed_original"] is False
    assert resumes == [], "a grounded trip must never resume the original"


def test_step_aborted_attempt_resumes_the_original_when_the_trip_fails(
    monkeypatch,
):
    """The fallback case: a failed trip serves what it always served.

    This is the property the whole early abort rests on. If the trip does not
    ground, the suspended attempt is resumed and finished, and the turn puts
    out the original answer -- not a truncated one, and not a new one.
    """
    monkeypatch.setenv("GRM_DEMAND_NGH", "1")
    monkeypatch.delenv(grm_demand.ENV_EARLY_ABORT, raising=False)
    _pin_profile(monkeypatch)
    _pin_observer(monkeypatch, [_rows(0.9, 0.1), _rows(0.9, 0.8)])
    arena, served, resumes = _abort_step_arena(
        answers={(3,): "original-answer", (0,): "trip", (1,): "trip",
                 (2,): "trip"},
        grounded={(0,): False, (1,): False, (2,): False},
        abort_at=((3,), 1),
    )
    ans, info = arena.step(_QUESTION, ngen=4, deposit=False, max_trips=1)
    assert ans == "original-answer"
    assert info["demand_served"] == "original"
    assert info["demand_trip_grounded"] is False
    assert info["demand_resumed_original"] is True
    assert info["demand_tokens_saved"] == 0
    assert len(resumes) == 1
    assert resumes[0]["abort_token_index"] == 1


def test_step_early_abort_off_is_the_sc1_path(monkeypatch):
    """OFF: no abort, no SC2 receipt fields on the abort keys, SC1 behaviour."""
    monkeypatch.setenv("GRM_DEMAND_NGH", "1")
    monkeypatch.setenv(grm_demand.ENV_EARLY_ABORT, "0")
    _pin_profile(monkeypatch)
    _pin_observer(monkeypatch, [_rows(0.9, 0.1), _rows(0.9, 0.8)])
    arena, served, resumes = _abort_step_arena(
        answers={(3,): "wrong", (0,): "RIGHT", (1,): "RIGHT", (2,): "RIGHT"},
        abort_at=None,
    )
    ans, info = arena.step(_QUESTION, ngen=4, deposit=False, max_trips=1)
    assert ans == "RIGHT"
    assert info["demand_served"] == "demand_trip"
    assert info["demand_early_abort"] is False
    assert "demand_abort_token_index" not in info
    assert "demand_resumed_original" not in info
    assert resumes == []


def test_step_demand_off_never_consults_the_abort_flag(monkeypatch):
    """Demand OFF: nothing changes, whatever the abort flag says."""
    monkeypatch.delenv("GRM_DEMAND_NGH", raising=False)
    monkeypatch.setenv(grm_demand.ENV_EARLY_ABORT, "1")
    _pin_profile(monkeypatch)
    arena, served, resumes = _abort_step_arena()
    ans, info = arena.step(_QUESTION, ngen=4, deposit=False, max_trips=1)
    assert ans == "answer"
    assert not [key for key in info if key.startswith("demand_")]
    assert resumes == []


def test_step_the_demand_trip_itself_is_never_early_aborted(monkeypatch):
    """Cap 1: a second fire is RECORDED, never acted on.

    Aborting the trip would throw away the only answer the turn has left,
    so the abort is suppressed for the duration of the trip. The refire is
    still recorded, exactly as SC1 recorded it.
    """
    monkeypatch.setenv("GRM_DEMAND_NGH", "1")
    monkeypatch.delenv(grm_demand.ENV_EARLY_ABORT, raising=False)
    _pin_profile(monkeypatch)
    _pin_observer(monkeypatch, [_rows(0.9, 0.1), _rows(0.1, 0.05)])
    arena, served, resumes = _abort_step_arena(
        answers={(3,): "wrong", (0,): "RIGHT", (1,): "RIGHT", (2,): "RIGHT"},
        abort_at=((3,), 1),
    )
    _ans, info = arena.step(_QUESTION, ngen=4, deposit=False, max_trips=1)
    assert info["demand_refired"] is True
    assert info["demand_refire_acted_on"] is False
    # Exactly two generations: the aborted original and the one trip.
    assert len(served) == 2


def test_step_abort_and_decision_cannot_disagree(monkeypatch):
    """A suspension whose rows say 'never fired' is a contradiction.

    It would mean the abort and the carried decision rule had drifted apart,
    and the turn would be acting on a fire that the receipt denies. Loud
    failure, never a quiet reconciliation.
    """
    monkeypatch.setenv("GRM_DEMAND_NGH", "1")
    monkeypatch.delenv(grm_demand.ENV_EARLY_ABORT, raising=False)
    _pin_profile(monkeypatch)
    # Rows say nothing fired; the stubbed attempt suspends anyway.
    _pin_observer(monkeypatch, [_rows(0.9, 0.9), _rows(0.9, 0.8)])
    arena, _served, _resumes = _abort_step_arena(abort_at=((3,), 1))
    with pytest.raises(grm_demand.DemandError, match="diverged"):
        arena.step(_QUESTION, ngen=4, deposit=False, max_trips=1)


def test_step_receipt_carries_the_wall_clock_fields(monkeypatch):
    monkeypatch.setenv("GRM_DEMAND_NGH", "1")
    monkeypatch.delenv(grm_demand.ENV_EARLY_ABORT, raising=False)
    _pin_profile(monkeypatch)
    _pin_observer(monkeypatch, [_rows(0.9, 0.1), _rows(0.9, 0.8)])
    arena, _served, _resumes = _abort_step_arena(
        answers={(3,): "wrong", (0,): "RIGHT", (1,): "RIGHT", (2,): "RIGHT"},
        abort_at=((3,), 1),
    )
    _ans, info = arena.step(_QUESTION, ngen=4, deposit=False, max_trips=1)
    assert info["demand_wall_ms_attempt"] >= 0.0
    assert info["demand_wall_ms_trip"] >= 0.0


def test_receipt_fields_survive_the_route_receipt_passthrough():
    from core.grm_three_pass import (
        ROUTE_RECEIPT_INFO_PREFIXES,
        _route_receipt_generic_info,
    )

    assert grm_demand.DEMAND_INFO_PREFIX in ROUTE_RECEIPT_INFO_PREFIXES
    fields = grm_demand.demand_info_fields(
        supported=True, early_abort=True, abort_token_index=3,
        tokens_generated_before_abort=4, tokens_saved=11,
        resumed_original=True, wall_ms_attempt=1.5)
    carried = _route_receipt_generic_info(dict(fields))
    for key, value in fields.items():
        assert carried.get(key) == value
