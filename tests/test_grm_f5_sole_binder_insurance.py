"""GRM-F5 — sole-binder insurance in the admission plan.

WHAT THIS MODULE PINS
---------------------
F3 proved, and pinned as characterization tests, that both shipping admission
rules have a branch NAMED for an identifier insurance they do not provide: a
sole identifier binder outside the top-3 window is silently dropped from the
plan (`docs/GRM_F3_ROUTING_AT_DISTANCE_LEDGER.md`, row 10 --
`artifacts/grm_f3/route_rows.json[9]`: `identified_candidates == [64]`,
`route_margin_1_2 == 0.007692198706398479`, `rank_plan == [168, 132, 116]`,
node 64 -- the intact Breakwater record -- never mounted).

F5 installs the registered fix behind `GRM_ROUTE_SOLE_BINDER_INSURANCE`,
default OFF.  ON, when the scan yields EXACTLY ONE binder, that binder is in
the plan: it takes the plan's LAST slot, every other member keeps its
membership and relative order, nothing is rescored.

THE F3 PINS ARE NOT EDITED.  `tests/test_grm_f3_routing_at_distance.py`'s
`test_identifier_insurance_branch_*` and two-binder tests still assert what
the code does with the flag OFF, and they still pass.  What lands here are
their ON-arm counterparts: the same frozen receipt values, the flag ON,
asserting the fix.  That is the Feathers discipline -- the pin records the
old behaviour, the new test records the new one, neither is rewritten to
agree with the other.

THE RED->GREEN CONTRACT, stated so a reader can check it without running
anything: `test_row10_red_under_off_margin_first` and
`test_row10_green_under_on_margin_first` are the same call with the same
arguments and only the flag differing.  RED (OFF) = plan `[168, 132, 116]`,
node 64 absent.  GREEN (ON) = node 64 in the plan.

Prior art
---------
* RT1 (`core.grm_admission.demote_non_binding_split_members`; GRM
  contributors, 2026) -- TAKEN VERBATIM: the reorder-never-score stance and
  its "stable, only the named node moves" discipline, which is what the
  control tests below assert.  OURS: nothing of the stance.
* Maximal Marginal Relevance -- Carbonell & Goldstein, SIGIR 1998.  TAKEN:
  the general shape of reordering a scored list under a secondary criterion
  without rescoring.  NOT taken: the diversity objective or its lambda; F5
  has no tunable.  Cited by F3 for the same reason.
* Characterization tests -- Feathers, *Working Effectively with Legacy Code*
  (2004), ch. 13.  TAKEN: the practice, and specifically the rule that a fix
  ADDS a counterpart rather than editing the pin.
* The CPU-double / literal-strings style follows
  `tests/test_grm_f3_routing_at_distance.py` and the C7 / SCOUT-FIX-8
  modules in this suite (GRM contributors, 2026).
* UNVERIFIED against the wider literature (no network in this sandbox); lead
  to check.  Search terms: "must-include constraint top-k retrieval",
  "constrained re-ranking guarantee matched entity", "hard inclusion
  constraint re-ranking", "maximal marginal relevance reorder".

Every test here is PURE CPU on literal values: no repository, no GPU.
"""
import json
from pathlib import Path

import pytest

from core.grm_admission import (
    MARGIN_THRESHOLD,
    ROUTE_SOLE_BINDER_INSURANCE_ENV,
    SOLE_BINDER_BRANCH_ALL_TOKENS_BIND,
    SOLE_BINDER_BRANCH_MARGIN_FIRST,
    insure_sole_binder,
    margin_first_plan,
    policy_plan,
    route_sole_binder_insurance_enabled,
)

ROOT = Path(__file__).resolve().parents[1]

# --- LT1.1 r2 arm A+ `recall_3_150`, turn 164, verbatim from the F3 table ---
# artifacts/grm_f3/route_rows.json[9] (repo-relative pin).
ROW10_RANKING = [168, 132, 116, 152, 167, 42]
ROW10_MARGIN = 0.007692198706398479
ROW10_HITS = [64]
ROW10_OFF_PLAN = [168, 132, 116]
ROW10_ON_PLAN = [168, 132, 64]

# --- arm A+ `recall_3_100`, turn 114 (F3 row 9): the SAME defect one probe
# earlier, where the binder is inside the 6-wide ranking but outside top-3.
ROW9_RANKING = [116, 110, 42, 84, 64, 55]
ROW9_MARGIN = 0.06734233962214775
ROW9_HITS = [64]
ROW9_OFF_PLAN = [116, 110, 42]
ROW9_ON_PLAN = [116, 110, 64]


@pytest.fixture
def flag(monkeypatch):
    """Set or clear `GRM_ROUTE_SOLE_BINDER_INSURANCE` for one test."""
    def _set(value):
        if value is None:
            monkeypatch.delenv(ROUTE_SOLE_BINDER_INSURANCE_ENV, raising=False)
        else:
            monkeypatch.setenv(ROUTE_SOLE_BINDER_INSURANCE_ENV, value)
    return _set


# ---------------------------------------------------------------------------
# The flag itself
# ---------------------------------------------------------------------------

def test_flag_defaults_off_and_unknown_tokens_fail_closed_to_off(flag):
    """Default OFF; an unknown token keeps the pre-F5 world.

    The direction matters and is the opposite of RT1's.  RT1 fails closed to
    ON because RT1 IS the default; F5 fails closed to OFF because every
    frozen receipt on disk was recorded with F5 absent, so a typo must not
    silently move a plan.
    """
    flag(None)
    assert route_sole_binder_insurance_enabled() is False
    for token in ('0', 'false', 'no', 'off', '', 'banana', 'ON1', '2'):
        flag(token)
        assert route_sole_binder_insurance_enabled() is False, token
    for token in ('1', 'true', 'yes', 'on', 'ON', ' True '):
        flag(token)
        assert route_sole_binder_insurance_enabled() is True, token


def test_explicit_kwarg_outranks_the_environment(flag):
    """Both entrypoints take `sole_binder_insurance` explicitly.

    The same precedence A-DEC uses (explicit caller > env > default), so a
    replay harness can pin an arm without mutating the process environment.
    """
    flag('1')
    assert margin_first_plan(
        ranking=ROW10_RANKING, route_margin_1_2=ROW10_MARGIN,
        identified_candidates=ROW10_HITS, scores=None,
        sole_binder_insurance=False)[0] == ROW10_OFF_PLAN
    flag('0')
    assert margin_first_plan(
        ranking=ROW10_RANKING, route_margin_1_2=ROW10_MARGIN,
        identified_candidates=ROW10_HITS, scores=None,
        sole_binder_insurance=True)[0] == ROW10_ON_PLAN
    flag('1')
    assert policy_plan(
        ranking=ROW10_RANKING, identified_candidates=ROW10_HITS,
        route_margin_1_2=ROW10_MARGIN,
        sole_binder_insurance=False)[0] == ROW10_OFF_PLAN
    flag('0')
    assert policy_plan(
        ranking=ROW10_RANKING, identified_candidates=ROW10_HITS,
        route_margin_1_2=ROW10_MARGIN,
        sole_binder_insurance=True)[0] == ROW10_ON_PLAN


# ---------------------------------------------------------------------------
# The registered fixture: F3 row 10, RED under OFF -> GREEN under ON
# ---------------------------------------------------------------------------

def test_row10_red_under_off_margin_first(flag):
    """RED arm.  The frozen r2 row, flag OFF: the sole binder is dropped.

    The F3 defect restated as the RED half of F5's own fixture, from the row
    the lead named in the order (`recall_3_150`, where node 64 is not even in
    the 6-wide ranking).
    """
    flag(None)
    assert ROW10_MARGIN < MARGIN_THRESHOLD
    plan, branch, _ = margin_first_plan(
        ranking=ROW10_RANKING, route_margin_1_2=ROW10_MARGIN,
        identified_candidates=ROW10_HITS, scores=None)
    assert plan == ROW10_OFF_PLAN
    assert branch == 'margin_insurance_k3_identifier_tiebreak'
    assert 64 not in plan, 'RED: sole identifier binder not in plan'


def test_row10_green_under_on_margin_first(flag):
    """GREEN arm.  Same call, flag ON: node 64 is in the plan.

    Reorder-never-score: the plan head is untouched, the plan keeps width 3,
    the node that leaves is the LAST slot -- the one the ranker trusted least
    of the three -- and the RANKING itself is returned unreordered.
    """
    flag('1')
    plan, branch, ranking = margin_first_plan(
        ranking=ROW10_RANKING, route_margin_1_2=ROW10_MARGIN,
        identified_candidates=ROW10_HITS, scores=None)
    assert 64 in plan, 'GREEN: sole identifier binder IS in the plan'
    assert plan == ROW10_ON_PLAN
    assert branch == SOLE_BINDER_BRANCH_MARGIN_FIRST
    assert plan[:2] == ROW10_OFF_PLAN[:2], 'head untouched'
    assert len(plan) == len(ROW10_OFF_PLAN), 'width unchanged'
    assert ranking == ROW10_RANKING, 'the RANKING is never reordered'


def test_row10_red_under_off_all_tokens_bind(flag):
    """RED arm through the OTHER shipping rule.  Same drop."""
    flag(None)
    plan, branch = policy_plan(
        ranking=ROW10_RANKING, identified_candidates=ROW10_HITS,
        route_margin_1_2=ROW10_MARGIN)
    assert plan == ROW10_OFF_PLAN
    assert branch == 'one_off_rank_identifier_insurance_k3'
    assert 64 not in plan, 'RED: the "insurance" branch does not insure'


def test_row10_green_under_on_all_tokens_bind(flag):
    """GREEN arm through the frozen `all_tokens_bind` rule.

    Both shipping rules share the defect (F3's finding, and why the fix is
    not a FIX-6 regression), so both get the flag.
    """
    flag('1')
    plan, branch = policy_plan(
        ranking=ROW10_RANKING, identified_candidates=ROW10_HITS,
        route_margin_1_2=ROW10_MARGIN)
    assert 64 in plan, 'GREEN: the insurance branch now insures'
    assert plan == ROW10_ON_PLAN
    assert branch == SOLE_BINDER_BRANCH_ALL_TOKENS_BIND


def test_row9_same_defect_same_fix_both_rules(flag):
    """F3 row 9 (`recall_3_100`): binder inside the ranking, outside top-3.

    Row 10's binder is not in the 6-wide ranking at all; row 9's is, at rank
    5.  The rule does not care -- it consumes the identifier scan, not the
    ranking -- and this pins that it does not, so a later "only insure what
    the router ranked" narrowing would break here.
    """
    flag(None)
    assert margin_first_plan(
        ranking=ROW9_RANKING, route_margin_1_2=ROW9_MARGIN,
        identified_candidates=ROW9_HITS, scores=None)[0] == ROW9_OFF_PLAN
    assert policy_plan(
        ranking=ROW9_RANKING, identified_candidates=ROW9_HITS,
        route_margin_1_2=ROW9_MARGIN)[0] == ROW9_OFF_PLAN
    flag('1')
    assert margin_first_plan(
        ranking=ROW9_RANKING, route_margin_1_2=ROW9_MARGIN,
        identified_candidates=ROW9_HITS, scores=None)[0] == ROW9_ON_PLAN
    assert policy_plan(
        ranking=ROW9_RANKING, identified_candidates=ROW9_HITS,
        route_margin_1_2=ROW9_MARGIN)[0] == ROW9_ON_PLAN


# ---------------------------------------------------------------------------
# Controls: the cases the rule must NOT touch, with the flag ON
# ---------------------------------------------------------------------------

def test_control_sole_binder_already_rank1_plan_unchanged(flag):
    """CONTROL (the order's first control): sole binder already at rank 1.

    `policy_plan` reaches this row by `exactly_one_identifier_decisive_rank1`
    -- a SINGLETON plan the flag must not widen.  `margin_first_plan` takes
    its k3 branch with the binder already inside the window.  Pinned so a
    later "always substitute" simplification, which would evict node 132 for
    a node already present, breaks here.
    """
    flag('1')
    plan, branch = policy_plan(
        ranking=ROW10_RANKING, identified_candidates=[168],
        route_margin_1_2=ROW10_MARGIN)
    assert plan == [168] and branch == 'exactly_one_identifier_decisive_rank1'

    plan, branch, _ = margin_first_plan(
        ranking=ROW10_RANKING, route_margin_1_2=ROW10_MARGIN,
        identified_candidates=[168], scores=None)
    assert plan == ROW10_OFF_PLAN, 'plan unchanged: binder already present'
    assert branch == 'margin_insurance_k3_identifier_tiebreak'


def test_control_sole_binder_already_last_slot_plan_unchanged(flag):
    """CONTROL: the binder is the plan's LAST slot already.

    The adversarial neighbour of the rank-1 control -- substituting here
    would be a self-eviction.  ON and OFF must be identical.
    """
    flag(None)
    off_mf = margin_first_plan(
        ranking=ROW10_RANKING, route_margin_1_2=ROW10_MARGIN,
        identified_candidates=[116], scores=None)
    off_pp = policy_plan(
        ranking=ROW10_RANKING, identified_candidates=[116],
        route_margin_1_2=ROW10_MARGIN)
    flag('1')
    assert margin_first_plan(
        ranking=ROW10_RANKING, route_margin_1_2=ROW10_MARGIN,
        identified_candidates=[116], scores=None) == off_mf
    assert policy_plan(
        ranking=ROW10_RANKING, identified_candidates=[116],
        route_margin_1_2=ROW10_MARGIN) == off_pp
    assert off_pp[0] == ROW10_OFF_PLAN and 116 in off_pp[0]


def test_control_two_or_more_binders_existing_rule_unchanged(flag):
    """CONTROL (the order's second control): >= 2 binders, existing rule.

    `declared_synthesis_identified_set` already admits EVERY identified
    candidate including unranked ones -- that asymmetry is exactly what made
    the one-binder case a defect -- so F5 declines to touch it.  ON must
    equal OFF, branch and plan.
    """
    flag(None)
    off = policy_plan(
        ranking=ROW10_RANKING, identified_candidates=[64, 99],
        route_margin_1_2=ROW10_MARGIN)
    flag('1')
    on = policy_plan(
        ranking=ROW10_RANKING, identified_candidates=[64, 99],
        route_margin_1_2=ROW10_MARGIN)
    assert on == off
    assert on[1] == 'declared_synthesis_identified_set'
    assert 64 in on[0] and 99 in on[0]


def test_control_two_binders_under_margin_first_is_also_unchanged(flag):
    """CONTROL, and a RED note the lead should read.

    `margin_first_plan` has NO `declared_synthesis_identified_set` branch: at
    two binders it returns `ranking[:3]` and admits NEITHER (here, neither 64
    nor 99).  F5 leaves that exactly as it found it -- the order scopes this
    flag to the SOLE-binder case -- so this control pins ON == OFF and, with
    it, that the >= 2 binder asymmetry SURVIVES in the margin_first rule.
    That residual is reported, not fixed, by F5.
    """
    flag(None)
    off = margin_first_plan(
        ranking=ROW10_RANKING, route_margin_1_2=ROW10_MARGIN,
        identified_candidates=[64, 99], scores=None)
    flag('1')
    on = margin_first_plan(
        ranking=ROW10_RANKING, route_margin_1_2=ROW10_MARGIN,
        identified_candidates=[64, 99], scores=None)
    assert on == off
    assert on[0] == ROW10_OFF_PLAN
    assert 64 not in on[0] and 99 not in on[0], (
        'RESIDUAL, registered not fixed: margin_first admits neither binder')


def test_control_zero_binders_unchanged(flag):
    """CONTROL: no binder at all -- nothing to insure, through both rules."""
    flag(None)
    off_pp = policy_plan(
        ranking=ROW10_RANKING, identified_candidates=[],
        route_margin_1_2=ROW10_MARGIN)
    off_mf = margin_first_plan(
        ranking=ROW10_RANKING, route_margin_1_2=ROW10_MARGIN,
        identified_candidates=[], scores=None)
    flag('1')
    assert policy_plan(
        ranking=ROW10_RANKING, identified_candidates=[],
        route_margin_1_2=ROW10_MARGIN) == off_pp
    assert off_pp[1] == 'ambiguous_zero_identifier_hits_k3'
    assert margin_first_plan(
        ranking=ROW10_RANKING, route_margin_1_2=ROW10_MARGIN,
        identified_candidates=[], scores=None) == off_mf


def test_control_decisive_margin_is_never_widened(flag):
    """CONTROL: a margin-decisive rank-1 plan stays a singleton, ON.

    F3 row 8 (`A+ recall_3_50`, margin 0.322, sole binder 28 AT rank 1), then
    the harder case where the decisive rank-1 is NOT the binder.  A decisive
    margin means the plan is a deliberate singleton, not an insurance window;
    widening it would be a scoring decision, which this rule never makes.
    """
    flag('1')
    plan, branch, _ = margin_first_plan(
        ranking=[28, 42, 18, 20, 46, 12], route_margin_1_2=0.32210014929976594,
        identified_candidates=[28], scores=None)
    assert plan == [28] and branch == 'fit_margin_decisive_rank1'

    plan, branch, _ = margin_first_plan(
        ranking=[28, 42, 18, 20, 46, 12], route_margin_1_2=0.32210014929976594,
        identified_candidates=[46], scores=None)
    assert plan == [28] and branch == 'fit_margin_decisive_rank1'
    plan, branch = policy_plan(
        ranking=[28, 42, 18, 20, 46, 12], identified_candidates=[46],
        route_margin_1_2=0.32210014929976594)
    assert plan == [28] and branch == 'fit_margin_decisive_rank1'


def test_control_empty_ranking_unchanged(flag):
    """CONTROL: an empty plan has no slot to spend and stays empty."""
    flag('1')
    assert margin_first_plan(
        ranking=[], route_margin_1_2=0.0,
        identified_candidates=[64], scores=None) == ([], 'empty_ranking', [])
    assert policy_plan(
        ranking=[], identified_candidates=[64],
        route_margin_1_2=0.0) == ([], 'empty_ranking')


def test_control_tie_group_reorder_still_runs_and_then_insures(flag):
    """CONTROL: FIX-6's exact-score tie-break is not replaced, only followed.

    With `scores` supplied the tie group is reordered FIRST (pre-F5
    behaviour, byte-identical) and the insurance then acts on the resulting
    window.  Here the tie-break alone already seats the binder, so ON == OFF:
    the rule adds nothing where the old one worked.
    """
    scores = {168: 1.0, 132: 1.0, 116: 1.0, 152: 0.4, 167: 0.3, 42: 0.2}
    flag(None)
    off = margin_first_plan(
        ranking=ROW10_RANKING, route_margin_1_2=0.0,
        identified_candidates=[116], scores=scores)
    flag('1')
    on = margin_first_plan(
        ranking=ROW10_RANKING, route_margin_1_2=0.0,
        identified_candidates=[116], scores=scores)
    assert off[0][0] == 116, 'FIX-6 tie-break hoisted the binder'
    assert on == off


def test_tie_break_cannot_reach_an_untied_binder_and_f5_can(flag):
    """The exact mechanism F3 named, as a fixture.

    Node 64's score is NOT tied with rank 1, so FIX-6's tie-break provably
    cannot move it -- the branch named "identifier_tiebreak" has no tie to
    break.  OFF, the binder is dropped even with `scores` present; ON, it is
    seated.  The difference between the two rules in one test.
    """
    scores = {168: 1.0, 132: 0.9, 116: 0.8, 152: 0.7, 167: 0.6, 42: 0.5,
              64: 0.1}
    flag(None)
    off, off_branch, _ = margin_first_plan(
        ranking=ROW10_RANKING, route_margin_1_2=ROW10_MARGIN,
        identified_candidates=ROW10_HITS, scores=scores)
    assert off == ROW10_OFF_PLAN and 64 not in off
    assert off_branch == 'margin_insurance_k3_identifier_tiebreak'
    flag('1')
    on, on_branch, _ = margin_first_plan(
        ranking=ROW10_RANKING, route_margin_1_2=ROW10_MARGIN,
        identified_candidates=ROW10_HITS, scores=scores)
    assert on == ROW10_ON_PLAN and on_branch == SOLE_BINDER_BRANCH_MARGIN_FIRST


# ---------------------------------------------------------------------------
# The helper in isolation
# ---------------------------------------------------------------------------

def test_insure_sole_binder_is_stable_and_minimal():
    """The helper's whole contract, independent of either rule's branches."""
    assert insure_sole_binder(
        plan=[1, 2, 3], identified_candidates=[9], enabled=True) == (
            [1, 2, 9], True)
    assert insure_sole_binder(
        plan=[1, 2, 3], identified_candidates=[9], enabled=False) == (
            [1, 2, 3], False)
    for present in (1, 2, 3):
        assert insure_sole_binder(
            plan=[1, 2, 3], identified_candidates=[present],
            enabled=True) == ([1, 2, 3], False)
    assert insure_sole_binder(
        plan=[1, 2, 3], identified_candidates=[], enabled=True) == (
            [1, 2, 3], False)
    assert insure_sole_binder(
        plan=[1, 2, 3], identified_candidates=[9, 10], enabled=True) == (
            [1, 2, 3], False)
    assert insure_sole_binder(
        plan=[], identified_candidates=[9], enabled=True) == ([], False)
    # width-1 plan: the binder replaces it outright, the only substitution a
    # width-1 window admits.
    assert insure_sole_binder(
        plan=[1], identified_candidates=[9], enabled=True) == ([9], True)
    # the caller's list is never mutated in place
    original = [1, 2, 3]
    insure_sole_binder(plan=original, identified_candidates=[9], enabled=True)
    assert original == [1, 2, 3]


def test_generator_identified_candidates_are_not_consumed_twice():
    """`identified_candidates` is declared Iterable and is now read twice.

    Before F5 `margin_first_plan` consumed it exactly once (into `hits`); the
    insurance reads it again.  A generator argument must therefore be
    materialized, not silently yield an empty second pass -- which would make
    the flag a no-op for any caller passing a generator.
    """
    plan, branch, _ = margin_first_plan(
        ranking=ROW10_RANKING, route_margin_1_2=ROW10_MARGIN,
        identified_candidates=(n for n in [64]), scores=None,
        sole_binder_insurance=True)
    assert plan == ROW10_ON_PLAN and branch == SOLE_BINDER_BRANCH_MARGIN_FIRST


# ---------------------------------------------------------------------------
# The registered receipts
# ---------------------------------------------------------------------------

def test_flag_contract_is_registered_and_pins_are_repo_relative():
    """`artifacts/grm_f5/flag_contract.json` exists and pins nothing absolute.

    The registration rule carried from round 1: every registration pins
    REPO-RELATIVE paths only.
    """
    contract = ROOT / 'artifacts/grm_f5/flag_contract.json'
    body = json.loads(contract.read_text())
    assert body['env_name'] == ROUTE_SOLE_BINDER_INSURANCE_ENV
    assert body['default'] == 'OFF'
    assert body['off_is_byte_identical'] is True
    for key, value in body['repo_relative_pins'].items():
        assert not value.startswith('/'), (key, value)
        assert (ROOT / value).exists(), (key, value)


def test_f3_row_fixture_values_match_the_frozen_route_table():
    """The fixture constants ARE the F3 receipts, not a retyping of them.

    Guards against the fixture drifting from `artifacts/grm_f3/route_rows.json`
    while still passing -- which would make every RED->GREEN claim above a
    claim about a row that no longer exists.
    """
    rows = json.loads((ROOT / 'artifacts/grm_f3/route_rows.json').read_text())
    row = next(r for r in rows
               if r['arm'] == 'A+' and r['pid'] == 'recall_3_150')
    assert row['ranking'] == ROW10_RANKING
    assert row['margin'] == ROW10_MARGIN
    assert row['receipt_idcands'] == ROW10_HITS == row['cpu_hits']
    assert row['plan'] == ROW10_OFF_PLAN
    assert row['branch'] == 'margin_insurance_k3_identifier_tiebreak'

    row9 = next(r for r in rows
                if r['arm'] == 'A+' and r['pid'] == 'recall_3_100')
    assert row9['ranking'] == ROW9_RANKING
    assert row9['margin'] == ROW9_MARGIN
    assert row9['receipt_idcands'] == ROW9_HITS
    assert row9['plan'] == ROW9_OFF_PLAN


def test_c2_132_plan_replay_receipt_is_recorded_off_byte_identical():
    """The OFF byte-identity gate's receipt, as produced by this seat.

    The replay itself is `scripts/grm_f5_c2_replay_gate.py` (132 recorded C2
    executions, every OFF plan reproduced byte-for-byte against its
    GPU-recorded `rank_plan`); this pins the receipts it wrote so a later
    tree cannot quietly lose them.
    """
    body = json.loads(
        (ROOT / 'artifacts/grm_f5/c2_replay_off.json').read_text())
    assert body['rows'] == 132
    assert body['off_byte_identical'] == 132
    assert body['verdict'] == 'PASS'

    risk = json.loads(
        (ROOT / 'artifacts/grm_f5/c2_replay_on_risk.json').read_text())
    assert risk['rows'] == 132 and risk['evaluable'] == 112
    assert risk['on_changed'] == 0, (
        'registered BEFORE the r3 run: the rule changes none of the 112 '
        'evaluable C2 plans; a later non-zero here is a result, not a tuning '
        'target')


# ---------------------------------------------------------------------------
# The receipt, end to end through decisive_admission_profile
# ---------------------------------------------------------------------------

class _OffRankBinderArena:
    """Arena double: five candidates, the SOLE binder scored last.

    Deliberately shaped as F3 row 10 in miniature -- the binder's score is
    NOT tied with rank 1 (so FIX-6's tie-break provably cannot reach it) and
    the margin is below threshold (so the insurance branch is the one taken).
    Follows the `_ProfileArena` double in `tests/test_grm_admission.py` (GRM
    contributors, 2026); the shape is reused, the scores are new.
    """

    def __init__(self):
        self.grafts = [
            {"text": "unrelated filler one"},
            {"text": "unrelated filler two"},
            {"text": "unrelated filler three"},
            {"text": "unrelated filler four"},
            {"text": "The current Praxis dock value is Quartz-8-Jade."},
        ]
        self.last_route_backend = "python"

    def _route_cand_base(self):
        return [0, 1, 2, 3, 4]

    def _probe_key(self, _question):
        return [1.0]

    def route(self, _question, *, exclude, limit, probe_key):
        return [0, 1, 2, 3, 4][:limit]

    def _vector_route_scores(self, _probe_key, _eligible):
        return {0: 1.0, 1: 0.95, 2: 0.9, 3: 0.85, 4: 0.1}

    def _length_debias_scores(self, base, _eligible):
        return base

    def _normalize_scores(self, base):
        return base

    def _query_lex_tokens(self, _question):
        return {"praxis", "dock"}

    def _lex_bonus(self, _qlex, _graft):
        return 0.0

    def _rare_tokens(self, _question):
        return set()


QUESTION = "What is the current Praxis dock value?"


def _profile(monkeypatch, value):
    from core.grm_admission import decisive_admission_profile
    if value is None:
        monkeypatch.delenv(ROUTE_SOLE_BINDER_INSURANCE_ENV, raising=False)
    else:
        monkeypatch.setenv(ROUTE_SOLE_BINDER_INSURANCE_ENV, value)
    return decisive_admission_profile(
        _OffRankBinderArena(), QUESTION, exclude=set())


def test_receipt_off_is_byte_identical_and_carries_no_f5_key(monkeypatch):
    """OFF: the profile has no `sole_binder_inserted` key at all.

    The absence IS the contract: every frozen receipt on disk was recorded
    before F5 existed, and an unconditional additive key would break
    `tests/test_grm_scout_fix4.py::test_nonrecency_existing_fixture_receipt_
    byte_identical`, which compares EVERY legacy receipt byte.
    """
    unset = _profile(monkeypatch, None)
    zero = _profile(monkeypatch, '0')
    assert 'sole_binder_inserted' not in unset
    assert 'sole_binder_inserted' not in zero
    assert unset == zero
    assert unset['identified_candidates'] == [4]
    assert unset['identifier_hit_count'] == 1
    assert 4 not in unset['rank_plan'], 'RED end to end: binder dropped'
    assert unset['rank_plan'] == [0, 1, 2]


def test_receipt_on_records_sole_binder_inserted_true(monkeypatch):
    """ON, inserted: the key is true and the branch name says so."""
    on = _profile(monkeypatch, '1')
    assert on['sole_binder_inserted'] is True
    assert 4 in on['rank_plan'], 'GREEN end to end: binder planned'
    assert on['rank_plan'] == [0, 1, 4]
    assert on['policy_branch'] in (
        SOLE_BINDER_BRANCH_MARGIN_FIRST, SOLE_BINDER_BRANCH_ALL_TOKENS_BIND)

    off = _profile(monkeypatch, None)
    assert on['ranking'] == off['ranking'], 'the RANKING never moves'
    assert on['identified_candidates'] == off['identified_candidates']
    assert on['route_margin_1_2'] == off['route_margin_1_2']
    assert on['rule_sha256'] == off['rule_sha256']
    assert {k: v for k, v in on.items()
            if k not in ('rank_plan', 'policy_branch',
                         'sole_binder_inserted')} == {
        k: v for k, v in off.items() if k != 'rank_plan'
        and k != 'policy_branch'}, 'only the plan and its branch differ'


def test_receipt_on_is_never_silent_when_nothing_was_inserted(monkeypatch):
    """ON, not inserted: the key is present and FALSE, never absent.

    NEVER SILENT, the law P2A/P2C/RT1 obey -- within the flag's own world a
    reader must not be able to mistake a missing field for an insurance that
    did not get recorded.  Here the question binds nothing, so the scan
    yields zero binders and there is nothing to insure.
    """
    from core.grm_admission import decisive_admission_profile
    monkeypatch.setenv(ROUTE_SOLE_BINDER_INSURANCE_ENV, '1')
    profile = decisive_admission_profile(
        _OffRankBinderArena(), 'What is the current Nowhere gate value?',
        exclude=set())
    assert profile['identified_candidates'] == []
    assert 'sole_binder_inserted' in profile
    assert profile['sole_binder_inserted'] is False


# ---------------------------------------------------------------------------
# The ON-arm counterparts to F3's DEFECT-PINNED tests
# ---------------------------------------------------------------------------
#
# The order makes `tests/test_grm_f3_routing_at_distance.py` READ-ONLY, and
# item 2 asks for "ON-arm counterparts asserting the fix rather than editing
# the pinned ones".  Both constraints point the same way, so the counterparts
# live HERE, named after the tests they answer, and the F3 module is not
# touched at all.  Each pair below is: the pinned OFF assertion re-executed
# verbatim (so a drift in the pin is caught here too), then the ON assertion.

def test_counterpart_to_f3_insurance_branch_drops_sole_binder_margin_first(
        flag):
    """Answers `test_identifier_insurance_branch_drops_the_sole_binder_margin_first`.

    That F3 pin uses row 9's values (`[116, 110, 42, 84, 64, 55]`, margin
    0.06734233962214775) and asserts `64 not in plan`.  Re-executed OFF here,
    then ON.
    """
    flag(None)
    plan, branch, _ = margin_first_plan(
        ranking=ROW9_RANKING, route_margin_1_2=ROW9_MARGIN,
        identified_candidates=ROW9_HITS, scores=None)
    assert branch == 'margin_insurance_k3_identifier_tiebreak'
    assert plan == ROW9_OFF_PLAN and 64 not in plan, 'the F3 pin still holds'

    flag('1')
    plan, branch, _ = margin_first_plan(
        ranking=ROW9_RANKING, route_margin_1_2=ROW9_MARGIN,
        identified_candidates=ROW9_HITS, scores=None)
    assert branch == SOLE_BINDER_BRANCH_MARGIN_FIRST
    assert plan == ROW9_ON_PLAN and 64 in plan, 'and F5 answers it'


def test_counterpart_to_f3_insurance_branch_drops_sole_binder_all_tokens_bind(
        flag):
    """Answers `test_identifier_insurance_branch_drops_the_sole_binder_all_tokens_bind`."""
    flag(None)
    plan, branch = policy_plan(
        ranking=ROW9_RANKING, identified_candidates=ROW9_HITS,
        route_margin_1_2=ROW9_MARGIN)
    assert branch == 'one_off_rank_identifier_insurance_k3'
    assert 64 not in plan, 'the F3 pin still holds'

    flag('1')
    plan, branch = policy_plan(
        ranking=ROW9_RANKING, identified_candidates=ROW9_HITS,
        route_margin_1_2=ROW9_MARGIN)
    assert branch == SOLE_BINDER_BRANCH_ALL_TOKENS_BIND
    assert plan == ROW9_ON_PLAN and 64 in plan, 'and F5 answers it'


def test_counterpart_to_f3_two_or_more_binders_asymmetry_is_closed(flag):
    """Answers `test_two_or_more_binders_are_all_admitted_regardless_of_rank`.

    That pin states the asymmetry that made the one-binder case a defect:
    with TWO binders the frozen `all_tokens_bind` rule admits every
    identified candidate; with ONE it admitted none.  ON, the discontinuity
    is closed -- in BOTH directions the sole binder is now planned -- while
    the two-binder branch it was measured against is untouched.

    HONEST SCOPE: closed for `policy_plan` only.  `margin_first_plan` has no
    declared-synthesis branch at all and still admits neither of two binders;
    that residual is registered in
    `artifacts/grm_f5/flag_contract.json:known_residual_not_fixed_by_this_item`
    and pinned by `test_control_two_binders_under_margin_first_is_also_unchanged`.
    """
    flag('1')
    two, two_branch = policy_plan(
        ranking=ROW9_RANKING, identified_candidates=[64, 99],
        route_margin_1_2=ROW9_MARGIN)
    assert two_branch == 'declared_synthesis_identified_set'
    assert 64 in two and 99 in two, 'the >= 2 rule is untouched'

    one, one_branch = policy_plan(
        ranking=ROW9_RANKING, identified_candidates=[64],
        route_margin_1_2=ROW9_MARGIN)
    assert 64 in one, 'and the == 1 case no longer admits none'
    assert one_branch == SOLE_BINDER_BRANCH_ALL_TOKENS_BIND
