"""GRM-LSR-P2A — fit-stage honesty (SHUTTLE) and not-in-memory abstention.

The defect these tests pin (Phase 1 FINAL, adjudication
``artifacts/lsr_p1/lsr_p1_adjudication_31e5c89453f4aa42.json``): A-DEC
planned exactly the answer-bearing node (``rank_plan = [X]``), the probe
ladder's later rung widened the plan to ``[X, Y, Z]``, and expansion-ordered
budget truncation seated the WRONG lineage end ``[Y]`` because ``X`` alone
already exceeded the 96-seat arena.  ``X`` was PLANNED and never seated, and
nothing in ``info`` said so.

Every test here is CPU-only: it drives the stubbed arena harness that
``tests/test_grm_admission.py`` established, so no GPU lease is consumed.
"""

from types import SimpleNamespace

import pytest

from core.graft_arena import ArenaCache
from core.grm_admission import (
    ABSTAIN_REASON_IDENTIFIER_UNBOUND,
    ABSTENTION_TEMPLATE,
    abstention_text,
    fit_info_fields,
    identifier_unbound_abstention,
    plan_priority_fit,
    shuttle_trip_cap,
)


# ---------------------------------------------------------------------------
# plan_priority_fit — the seating law itself
# ---------------------------------------------------------------------------


def test_plan_members_are_seated_before_filler_in_plan_order():
    # Filler 1 is cheap and comes FIRST in expansion order; under the old
    # expansion-ordered truncation it would have eaten the seats the plan
    # member needs.  Plan priority must seat 3 anyway.
    receipt = plan_priority_fit(
        plan=[3],
        candidates=[1, 2, 3],
        ntok={1: 40, 2: 40, 3: 50},
        budget=96,
    )
    assert receipt["fit_seated_planned"] == [3]
    assert receipt["fit_seated"] == [1, 3]
    assert receipt["fit_dropped_planned"] == []
    assert receipt["fit_dropped_filler"] == [2]
    assert receipt["fit_shuttle_pending"] == []


def test_plan_order_is_preserved_and_not_resorted_by_candidate_order():
    receipt = plan_priority_fit(
        plan=[9, 4],
        candidates=[4, 9],
        ntok={4: 10, 9: 10},
        budget=96,
    )
    # Seating happens in PLAN order; the seated set is sorted for assembly.
    assert receipt["fit_seated_planned"] == [9, 4]
    assert receipt["fit_seated"] == [4, 9]


def test_non_cofitting_plan_members_shuttle_rather_than_drop():
    # Two 60-seat plan members, width 96: exactly the registered
    # declared_synthesis shuttle case.
    receipt = plan_priority_fit(
        plan=[5, 6],
        candidates=[5, 6],
        ntok={5: 60, 6: 60},
        budget=96,
    )
    assert receipt["fit_seated_planned"] == [5]
    assert receipt["fit_shuttle_pending"] == [6]
    # A member that merely does not CO-fit is never a "drop".
    assert receipt["fit_dropped_planned"] == []
    assert receipt["fit_unseatable"] == []


def test_unseatable_plan_member_is_an_explicit_degrade():
    # The lived ADMISSION-PRUNE shape: node 3 is the 650-char competitor and
    # does not fit even alone.
    receipt = plan_priority_fit(
        plan=[3],
        candidates=[3, 2, 1],
        ntok={3: 130, 2: 66, 1: 25},
        budget=96,
    )
    assert receipt["fit_unseatable"] == [3]
    assert receipt["fit_dropped_planned"] == [3]
    assert receipt["fit_shuttle_pending"] == []
    # Filler still seats — but it is labeled filler, not a stand-in.
    assert sorted(receipt["fit_seated_filler"]) == [1, 2]
    assert receipt["fit_seated_planned"] == []


def test_shuttle_trip_cap_is_len_rank_plan():
    assert shuttle_trip_cap([]) == 0
    assert shuttle_trip_cap([7]) == 1
    assert shuttle_trip_cap([7, 8, 9]) == 3


def test_fit_info_fields_are_always_present_even_when_nothing_dropped():
    fields = fit_info_fields(plan_priority_fit(
        plan=[1], candidates=[1], ntok={1: 5}, budget=96))
    for key in (
        "fit_planned", "fit_seated", "fit_dropped_planned",
        "fit_dropped_filler", "fit_unseatable", "fit_shuttle",
        "fit_shuttle_trips", "served_without_plan_head",
    ):
        assert key in fields
    assert fields["fit_dropped_planned"] == []
    assert fields["fit_shuttle"] is False


# ---------------------------------------------------------------------------
# abstention — structural, no thresholds
# ---------------------------------------------------------------------------


def test_abstains_only_when_identifier_tokens_bind_nothing():
    unbound = identifier_unbound_abstention({
        "identifier_tokens": ["zephyr", "manifest"],
        "identified_candidates": [],
    })
    assert unbound is not None
    assert unbound["abstained"] is True
    assert unbound["abstain_reason"] == ABSTAIN_REASON_IDENTIFIER_UNBOUND
    assert unbound["abstain_identifier_tokens"] == ["zephyr", "manifest"]
    assert unbound["abstain_text"] == ABSTENTION_TEMPLATE.format(
        tokens="zephyr, manifest")


def test_bound_identifier_never_abstains():
    assert identifier_unbound_abstention({
        "identifier_tokens": ["meridian", "docket"],
        "identified_candidates": [3],
    }) is None


def test_non_identifier_question_never_abstains_this_round():
    # Ruling 2.3: the ambiguous/topical class gets NO abstention and no
    # threshold this round.
    assert identifier_unbound_abstention({
        "identifier_tokens": [],
        "identified_candidates": [],
    }) is None
    assert identifier_unbound_abstention(None) is None


def test_abstention_text_is_a_constant_template():
    assert abstention_text(["alpha"]) == (
        "Not in memory: no stored record matches alpha.")


# ---------------------------------------------------------------------------
# arena step() — the wired behaviour
# ---------------------------------------------------------------------------


def _arena_probe(*, decisive: bool, ntok, width, grounded=True):
    """Stubbed CPU arena, extending the harness in tests/test_grm_admission."""
    arena = ArenaCache.__new__(ArenaCache)
    arena.m = SimpleNamespace(
        layers=[SimpleNamespace(self_attn=SimpleNamespace(live_shift=None))])
    arena.live_shift = 9
    arena.stop_sequences = ()
    arena.ephemeral = False
    arena.live_segs = []
    arena.topk = 3
    arena.decisive_admission = bool(decisive)
    arena.caches = None
    arena.pos = 0
    arena.cur_mounts = []
    arena.cur_mount_n = 0
    arena.width = int(width)
    arena.prompt_template = None
    arena.grafts = [
        {"text": f"node-{i}", "ntok": int(n), "kind": "fact"}
        for i, n in enumerate(ntok)
    ]
    arena._s4_turn = 0
    route_calls = []
    arena.route = lambda text, *, exclude, limit: (
        route_calls.append((text, set(exclude), limit))
        or list(range(len(ntok))))
    arena._rare_tokens = lambda _text: set()
    arena._descent_expand = lambda picks, _kinds, qrare=None: list(picks)
    arena._resolve_revision_mounts = lambda picks: list(picks)
    arena._next_s4_turn = lambda: 1
    arena._bump_cuda_gqa_epoch = lambda: None
    arena._commit_s4_attempt = lambda *_a, **_k: None
    arena._grounding_attribution = lambda _ans, picks, _q: (
        bool(grounded) and bool(picks), list(picks))
    served = []

    def attempt(_question, picks, _ngen, _deposit, _stops):
        arena.cur_mounts = list(picks)
        served.append(list(picks))
        return "answer", {"mounts": [v + 1 for v in picks]}

    arena._attempt = attempt
    return arena, route_calls, served


def _pin_profile(monkeypatch, profile):
    monkeypatch.setattr(
        "core.graft_arena.decisive_admission_profile",
        lambda _arena, _text, *, exclude, route_limit: dict(profile),
    )


_LIVED_PROFILE = {
    # The lived sup_reserve_meridian_docket shape, exactly.
    "ranking": [3, 2, 1, 0],
    "identified_candidates": [3],
    "identifier_tokens": ["meridian", "docket"],
    "identifier_hit_count": 1,
    "route_margin_1_2": 1.036860985747615,
    "rank_plan": [3],
    "policy_branch": "exactly_one_identifier_decisive_rank1",
}


def test_lived_admission_prune_shape_no_longer_substitutes_silently(monkeypatch):
    _pin_profile(monkeypatch, _LIVED_PROFILE)
    # ntok: node 3 = the 650-char competitor, unseatable alone at width 96.
    arena, _calls, _served = _arena_probe(
        decisive=True, ntok=[20, 25, 66, 130], width=96)
    _answer, info = arena.step("meridian docket?", ngen=1, deposit=False)
    assert info["fit_planned"] == [3]
    # The plan head is UNSEATABLE and that is stated, not hidden.
    assert info["fit_unseatable"] == [3]
    assert info["fit_dropped_planned"] == [3]
    assert info["served_without_plan_head"] is True
    # The lower-ranked node still serves, but it is LABELED, never a
    # substitution the receipt is silent about.
    assert 3 not in info["fit_seated"]


def test_shuttle_serializes_a_declared_synthesis_pair(monkeypatch):
    # Two 60-seat plan members, width 96 -> neither co-fits; both are served,
    # across two trips.
    _pin_profile(monkeypatch, {
        "ranking": [0, 1],
        "identified_candidates": [0, 1],
        "identifier_tokens": ["twin", "pair"],
        "identifier_hit_count": 2,
        "route_margin_1_2": 0.0,
        "rank_plan": [0, 1],
        "policy_branch": "declared_synthesis_identified_set",
    })
    arena, _calls, served = _arena_probe(
        decisive=True, ntok=[60, 60], width=96, grounded=False)
    _answer, info = arena.step("twin pair?", ngen=1, deposit=False,
                               max_trips=1)
    assert info["fit_shuttle"] is True
    assert info["fit_shuttle_trips"] == [[1]]
    # Both plan members received a seat across the turn's trips.
    seated_across_turn = {v for trip in served for v in trip}
    assert seated_across_turn == {0, 1}
    # Neither was DROPPED: co-fitting was impossible, so it shuttled.
    assert info["fit_dropped_planned"] == []
    assert info["fit_unseatable"] == []


def test_shuttle_trip_count_never_exceeds_len_rank_plan(monkeypatch):
    _pin_profile(monkeypatch, {
        "ranking": [0, 1, 2],
        "identified_candidates": [0, 1, 2],
        "identifier_tokens": ["a", "b"],
        "identifier_hit_count": 3,
        "route_margin_1_2": 0.0,
        "rank_plan": [0, 1, 2],
        "policy_branch": "declared_synthesis_identified_set",
    })
    arena, _calls, _served = _arena_probe(
        decisive=True, ntok=[60, 60, 60], width=96, grounded=False)
    _answer, info = arena.step("a b?", ngen=1, deposit=False, max_trips=1)
    assert len(info["fit_shuttle_trips"]) <= shuttle_trip_cap([0, 1, 2])


def test_shuttle_fires_on_fit_drop_not_on_grounding_failure(monkeypatch):
    # Trip 0 GROUNDS (wrong-but-grounded). The old ladder ended the turn
    # there. The shuttle rung must still exist because the FIT dropped a
    # planned member.
    _pin_profile(monkeypatch, {
        "ranking": [0, 1],
        "identified_candidates": [0, 1],
        "identifier_tokens": ["twin", "pair"],
        "identifier_hit_count": 2,
        "route_margin_1_2": 0.0,
        "rank_plan": [0, 1],
        "policy_branch": "declared_synthesis_identified_set",
    })
    arena, _calls, _served = _arena_probe(
        decisive=True, ntok=[60, 60], width=96, grounded=True)
    _answer, info = arena.step("twin pair?", ngen=1, deposit=False,
                               max_trips=1)
    assert info["fit_shuttle"] is True
    assert info["fit_shuttle_trips"] == [[1]]


def test_legacy_path_is_byte_identical(monkeypatch):
    """GRM_ADM_DECISIVE=0: no plan, no fit fields, legacy route call."""
    monkeypatch.setenv("GRM_ADM_DECISIVE", "0")
    arena, calls, served = _arena_probe(
        decisive=False, ntok=[1, 1, 1], width=16)
    answer, info = arena.step("praxis dock?", ngen=1, deposit=False)
    assert answer == "answer"
    assert calls == [("praxis dock?", set(), 3)]
    assert served == [[0, 1, 2]]
    assert arena.cur_mounts == [0, 1, 2]
    for key in (
        "fit_planned", "fit_seated", "fit_dropped_planned",
        "fit_dropped_filler", "fit_unseatable", "fit_shuttle",
        "fit_shuttle_trips", "served_without_plan_head",
        "abstained", "abstain_reason",
    ):
        assert key not in info, f"legacy info gained {key}"
    assert "admission_policy" not in info


def test_step_abstains_when_identifiers_bind_nothing(monkeypatch):
    _pin_profile(monkeypatch, {
        "ranking": [0, 1],
        "identified_candidates": [],
        "identifier_tokens": ["zephyr", "manifest"],
        "identifier_hit_count": 0,
        "route_margin_1_2": 0.0,
        "rank_plan": [0, 1, 2],
        "policy_branch": "ambiguous_zero_identifier_hits_k3",
    })
    arena, _calls, served = _arena_probe(
        decisive=True, ntok=[10, 10, 10], width=96)
    answer, info = arena.step("zephyr manifest?", ngen=1, deposit=False)
    assert info["abstained"] is True
    assert info["abstain_reason"] == ABSTAIN_REASON_IDENTIFIER_UNBOUND
    assert info["abstain_identifier_tokens"] == ["zephyr", "manifest"]
    assert answer == (
        "Not in memory: no stored record matches zephyr, manifest.")
    # Nothing was confabulated: no model forward ran at all.
    assert served == []
    assert info["mounts"] == []


def test_step_does_not_abstain_when_the_identifier_binds(monkeypatch):
    _pin_profile(monkeypatch, _LIVED_PROFILE)
    arena, _calls, served = _arena_probe(
        decisive=True, ntok=[20, 25, 66, 40], width=96)
    _answer, info = arena.step("meridian docket?", ngen=1, deposit=False)
    assert "abstained" not in info
    assert served, "a bound identifier must serve, not abstain"
    assert 3 in info["fit_seated"]
    assert info["served_without_plan_head"] is False


def test_abstention_output_deposits_as_recall_not_as_a_routable_fact(monkeypatch):
    _pin_profile(monkeypatch, {
        "ranking": [0],
        "identified_candidates": [],
        "identifier_tokens": ["zephyr"],
        "identifier_hit_count": 0,
        "route_margin_1_2": 0.0,
        "rank_plan": [0],
        "policy_branch": "ambiguous_zero_identifier_hits_k3",
    })
    arena, _calls, _served = _arena_probe(
        decisive=True, ntok=[10], width=96)
    deposits = []

    def deposit(text):
        arena.grafts.append({"text": text, "ntok": 3, "kind": "turn"})
        deposits.append(text)
        return len(arena.grafts) - 1

    arena.deposit = deposit
    _answer, info = arena.step("zephyr?", ngen=1, deposit=True)
    # The turn IS deposited (the question is a lived turn) ...
    assert len(deposits) == 1
    gidx = info["abstain_deposited_graft"]
    # ... but pinned kind="recall", which _route_cand_base excludes, so the
    # abstention can never be routed later as if it were a stored fact.
    assert arena.grafts[gidx]["kind"] == "recall"
    assert [
        i for i, g in enumerate(arena.grafts)
        if not g.get("retired") and g.get("kind", "turn") != "recall"
    ] == [0]


@pytest.mark.parametrize("width,expect_head", [(96, False), (256, True)])
def test_arena_width_decides_the_lived_defect(monkeypatch, width, expect_head):
    """The ADM1.3-vs-lived reconciliation, as a behavioural pin.

    The same fixture is green at width 256 and prunes at width 96 — that is
    exactly why ADM1.3's F-PROD was all-green while the lived campaign was
    not (adjudication 31e5c894, difference_1_arena_width).
    """
    _pin_profile(monkeypatch, _LIVED_PROFILE)
    arena, _calls, _served = _arena_probe(
        decisive=True, ntok=[20, 25, 66, 130], width=width)
    _answer, info = arena.step("meridian docket?", ngen=1, deposit=False)
    assert (3 in info["fit_seated"]) is expect_head
    assert info["served_without_plan_head"] is not expect_head
