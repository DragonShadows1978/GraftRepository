"""GRM-RT1 — a fit-time split child must not outrank its competing fact node.

MEASURED (RS1 A0 ``fresh_fact_controls``, reproduced by RS3 C0; the channel
decomposition is ``artifacts/grm_rt1/grm_rt1_diagnosis.json``): on
``sup_solace_fresh`` the router returns ``[2, 4, 3, 1, 0]`` — the width-guard
INDEX PARENT 2 and both its children 4 and 3 sweep the top three ranks while
graft 1, the solace FACT node and the only node in the repository whose own
text binds the question's identifier, is pushed to rank 4.

The cause is neither identifier evidence nor L2/lineage.  The lexical channel
is a FOUR-WAY TIE at ``lex_bonus = 1.000`` (the competitor's own text says
"the Praxis dock and Solace key references are index context", so both query
content words hit every family member as well as the fact node), and the
latent channel then decides — with the split family holding three tickets to
the fact node's one, because ``_guard_deposit_width`` gives the index parent
``child_cents`` and ``_cent_score`` takes the max over the family.
``route_margin_1_2 == 0.0`` is the fingerprint of exactly that structure.

THE RULE these tests pin: **a split child, or its index parent, that does not
bind the question's identifier tokens in its OWN text never outranks a
candidate that does; and a split member's binding is judged on its own text,
never on the union routing surface it inherited from its parent.**

Structural: no threshold is introduced and nothing is refit.  Governed by
``GRM_LSR_FIXES`` — with the switch OFF the rule does not run at all and the
ranking is the router's verbatim.

Every test here is CPU-only, on the stubbed arena harness the P2A/P2C tests
established, so no GPU lease is consumed.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.graft_arena import ArenaCache
from core.grm_admission import (
    RT1_RULE,
    AdmissionPolicyError,
    admission_info_fields,
    decisive_admission_profile,
    demote_non_binding_split_members,
    identifier_unbound_abstention,
    rt1_enabled,
    rt1_info_fields,
    split_member_binds_own_text,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT / "tests/fixtures/supersession_battery/fresh_fact_controls.json")

#: The lived ranking, verbatim from the RS1 A0 receipt
#: (``grm_rs1_A0_fresh_fact_controls_dbfce088cddf1fc1.json``, probe
#: ``sup_solace_fresh``, ``info["ranking_ids"]``).
LIVED_SOLACE_RANKING = [2, 4, 3, 1, 0]

#: The lived split family, from the same receipt's ``fit_split_parent`` /
#: ``fit_split_children``.
LIVED_SPLIT_PARENT = 2
LIVED_SPLIT_CHILDREN = [3, 4]


# ---------------------------------------------------------------------------
# The rule, as a pure function
# ---------------------------------------------------------------------------


def test_rule_is_stated_once_and_names_the_own_text_clause():
    """The rule string is the contract; both halves must be in it."""
    assert "own text" in RT1_RULE
    assert "never outranks" in RT1_RULE
    assert "inherited" in RT1_RULE


def test_solace_ranking_flips_to_the_fact_node():
    """THE LIVED CASE: ``[2, 4, 3, 1, 0]`` -> graft 1 at rank 1.

    Graft 1 is the only candidate whose own text binds "solace key"; 2, 3 and
    4 are the split family and bind nothing.  All three are demoted below it.
    """
    ranking, demoted = demote_non_binding_split_members(
        ranking=LIVED_SOLACE_RANKING,
        binding={1},
        split_members={LIVED_SPLIT_PARENT, *LIVED_SPLIT_CHILDREN},
    )
    assert ranking[0] == 1, ranking
    assert ranking == [1, 0, 2, 4, 3], ranking
    assert demoted == [2, 4, 3], demoted


def test_tundra_ranking_is_untouched_because_the_family_binds():
    """CONTROL: on ``sup_reserve_tundra_ledger`` the family DOES bind.

    ``identified_candidates`` was ``[2, 4]`` in the lived receipt, so the rule
    has no non-binding member above a binder and must not move anything.
    """
    ranking, demoted = demote_non_binding_split_members(
        ranking=LIVED_SOLACE_RANKING,
        binding={2, 4},
        split_members={LIVED_SPLIT_PARENT, *LIVED_SPLIT_CHILDREN},
    )
    assert ranking == LIVED_SOLACE_RANKING
    assert demoted == []


def test_relative_order_within_each_group_is_the_routers():
    """The rule REORDERS between two groups; it never re-sorts inside one."""
    ranking, demoted = demote_non_binding_split_members(
        ranking=[7, 5, 9, 1, 3],
        binding={1, 3},
        split_members={7, 5, 9},
    )
    # Binders keep 1-before-3; demoted members keep 7-before-5-before-9.
    assert ranking == [1, 3, 7, 5, 9]
    assert demoted == [7, 5, 9]


def test_no_op_when_nothing_binds():
    """Nothing to protect: the router's order stands verbatim."""
    ranking, demoted = demote_non_binding_split_members(
        ranking=LIVED_SOLACE_RANKING, binding=set(),
        split_members={2, 3, 4})
    assert ranking == LIVED_SOLACE_RANKING
    assert demoted == []


def test_rt1_cannot_touch_the_t33_carve_out():
    """SCOPE, pinned: ``e2e_t33_polaris_mark`` is out of RT1's reach.

    t33's identifier binds NO node in the repository, so P2A Ruling 2's
    abstention fires before any ranking is consulted — and the RT1 rule is
    itself a no-op with zero binders, whatever the split family looks like.
    The order asks whether the rule touches t33; this is the structural
    answer, and the G3 census confirmed it empirically (t33 served the same
    abstention string RS3's census recorded).
    """
    profile = {
        "identifier_tokens": ["polaris", "mark"],
        "identified_candidates": [],
    }
    assert identifier_unbound_abstention(profile) is not None
    ranking, demoted = demote_non_binding_split_members(
        ranking=LIVED_SOLACE_RANKING, binding=set(),
        split_members={2, 3, 4})
    assert ranking == LIVED_SOLACE_RANKING
    assert demoted == []


def test_no_op_when_every_binder_is_itself_a_split_member():
    """The rule protects a NON-member against a member, never a family
    against itself: a split node whose own chunk binds is a legitimate answer,
    and re-sorting one family against another is not RT1's business.
    """
    ranking, demoted = demote_non_binding_split_members(
        ranking=[2, 4, 3], binding={4}, split_members={2, 3, 4})
    assert ranking == [2, 4, 3]
    assert demoted == []


def test_no_op_when_no_split_member_outranks_a_binder():
    """Already obeying the rule: nothing moves."""
    ranking, demoted = demote_non_binding_split_members(
        ranking=[1, 2, 4, 3], binding={1}, split_members={2, 3, 4})
    assert ranking == [1, 2, 4, 3]
    assert demoted == []


def test_only_members_above_the_highest_binder_are_demoted():
    """A non-binding member already BELOW the binder is left where it is —
    the rule fixes an inversion, it does not sweep the tail."""
    ranking, demoted = demote_non_binding_split_members(
        ranking=[2, 1, 4], binding={1}, split_members={2, 4})
    assert demoted == [2]
    assert ranking == [1, 4, 2]


def test_the_rule_never_promotes_a_non_binder_above_a_binder():
    """PROPERTY, over every binding subset of a small ranking: after the rule
    no non-binding split member precedes any binding non-member, and no
    candidate is invented or lost."""
    base = [0, 1, 2, 3, 4]
    members = {2, 3, 4}
    for size in range(len(base) + 1):
        for binding in combinations(base, size):
            binders = set(binding)
            ranking, _demoted = demote_non_binding_split_members(
                ranking=base, binding=binders, split_members=members)
            assert sorted(ranking) == sorted(base), "no candidate invented"
            protected = [v for v in ranking if v in binders - members]
            if not protected:
                continue
            cut = ranking.index(protected[0])
            for value in ranking[:cut]:
                assert not (value in members and value not in binders)


# ---------------------------------------------------------------------------
# The OWN-TEXT clause — the half the index parent needs
# ---------------------------------------------------------------------------


def _fixture_texts():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return {node["node_id"]: node["text"] for node in data["nodes"]}


def test_own_text_predicate_binds_only_the_fact_node_on_solace():
    """The FROZEN ADM1 predicate, on the REGISTERED fixture texts."""
    texts = _fixture_texts()
    qrare = ArenaCache._rare_tokens("What is the current Solace key value?")
    ordered = ["solace", "key"]
    assert split_member_binds_own_text(
        text=texts["solace_fact"],
        ordered_identifier_tokens=ordered,
        rare_identifier_tokens=qrare,
    ) is True
    # The competitor MENTIONS "Solace key" but never as "current solace key
    # value" — the predicate is a phrase test, not a bag of words.
    assert split_member_binds_own_text(
        text=texts["sable_competitor"],
        ordered_identifier_tokens=ordered,
        rare_identifier_tokens=qrare,
    ) is False


def test_lexical_channel_cannot_discriminate_which_is_why_the_rule_exists():
    """The DIAGNOSIS, pinned: lex_bonus TIES the competitor with the answer.

    Both query content words ("solace", "key") appear in the competitor's own
    text, so the lexical channel gives BOTH nodes a full 1.000 and cannot
    separate them.  The identifier predicate can.  This is the measurement
    that makes an identifier-shaped rule the right instrument rather than a
    lexical re-weighting.
    """
    texts = _fixture_texts()
    qlex = ArenaCache._query_lex_tokens(
        "What is the current Solace key value?")
    assert qlex == {"solace", "key"}

    def lex_bonus(text):
        have = set(ArenaCache._rare_tokens(text))
        if not (qlex <= have):
            have |= ArenaCache._node_text_tokens(text)
        return len(qlex & have) / len(qlex)

    assert lex_bonus(texts["solace_fact"]) == 1.0
    assert lex_bonus(texts["sable_competitor"]) == 1.0


def test_index_parent_is_judged_on_its_own_text_not_the_inherited_union():
    """``_guard_deposit_width`` gives the parent the UNION rare surface.

    A parent whose own text binds nothing must NOT inherit a child's binding.
    The union text is constructed here to show it WOULD bind if the rule read
    the inherited surface — which is exactly what the rule forbids.
    """
    parent_own = "index of the following material, no facts of its own"
    child_text = "the current solace key value is raven-9-ivory"
    ordered = ["solace", "key"]
    assert split_member_binds_own_text(
        text=parent_own, ordered_identifier_tokens=ordered,
        rare_identifier_tokens=set()) is False
    assert split_member_binds_own_text(
        text=child_text, ordered_identifier_tokens=ordered,
        rare_identifier_tokens=set()) is True
    # The union WOULD bind; the rule judges the parent on `parent_own` alone.
    assert split_member_binds_own_text(
        text=f"{parent_own} {child_text}", ordered_identifier_tokens=ordered,
        rare_identifier_tokens=set()) is True


# ---------------------------------------------------------------------------
# Split-family membership — read from flags the split writers already set
# ---------------------------------------------------------------------------


def _membership_arena(nodes):
    arena = ArenaCache.__new__(ArenaCache)
    arena.grafts = list(nodes)
    return arena


def test_membership_reads_the_persisted_width_guard_flags():
    arena = _membership_arena([
        {"metadata": {}},
        {"metadata": {"width_guard_parent": True}},
        {"metadata": {"width_guard_child": True}},
    ])
    assert arena._split_family_members([0, 1, 2]) == {1, 2}


def test_membership_reads_the_ephemeral_split_flags():
    arena = _membership_arena([
        {"metadata": {}},
        {"metadata": {}, "ephemeral_split": True},
        {"metadata": {}, "ephemeral_split_of": 1},
    ])
    assert arena._split_family_members([0, 1, 2]) == {1, 2}


def test_membership_is_empty_for_a_repository_that_never_split():
    arena = _membership_arena([{"metadata": {}} for _ in range(4)])
    assert arena._split_family_members([0, 1, 2, 3]) == set()


def test_membership_tolerates_an_out_of_range_candidate():
    """A ranking naming a node the arena no longer has must not raise: the
    rule's job is to reorder, never to become a new failure mode."""
    arena = _membership_arena([{"metadata": {"width_guard_child": True}}])
    assert arena._split_family_members([0, 99]) == {0}


# ---------------------------------------------------------------------------
# The switch: GRM_LSR_FIXES governs the rule, OFF is byte-identical
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_ambient_rt1_escape(monkeypatch):
    """The isolation escape is a MEASUREMENT variable, absent in production.

    Clearing it for every test here means each test states its own switch
    conditions and an ambient value in the seat's shell can never quietly
    steer a result.
    """
    monkeypatch.delenv("GRM_RT1_RULE", raising=False)


def test_rt1_rides_the_lsr_fixes_switch_and_defaults_on(monkeypatch):
    monkeypatch.delenv("GRM_LSR_FIXES", raising=False)
    assert rt1_enabled() is True


@pytest.mark.parametrize("token", ["0", "false", "no", "off", "OFF", "False"])
def test_rt1_off_tokens(monkeypatch, token):
    monkeypatch.setenv("GRM_LSR_FIXES", token)
    assert rt1_enabled() is False


def test_rt1_unknown_token_fails_closed_to_on(monkeypatch):
    """Same direction P2C fails: a mistyped value KEEPS the fix."""
    monkeypatch.setenv("GRM_LSR_FIXES", "maybe")
    assert rt1_enabled() is True


def test_rt1_defers_to_the_arenas_own_resolver(monkeypatch):
    """RT1 and P2C can never disagree about the switch on a live arena."""
    monkeypatch.delenv("GRM_RT1_RULE", raising=False)
    assert rt1_enabled(
        SimpleNamespace(_lsr_fixes_enabled=lambda: False)) is False
    assert rt1_enabled(
        SimpleNamespace(_lsr_fixes_enabled=lambda: True)) is True


# --- the measurement-only isolation escape (amendment_a1_rt1_only_switch) ---


def test_rt1_only_escape_is_absent_by_default_and_changes_nothing(monkeypatch):
    """ABSENT in production: with it unset, the family switch decides."""
    monkeypatch.delenv("GRM_RT1_RULE", raising=False)
    monkeypatch.setenv("GRM_LSR_FIXES", "1")
    assert rt1_enabled() is True
    monkeypatch.setenv("GRM_LSR_FIXES", "0")
    assert rt1_enabled() is False


def test_rt1_only_escape_turns_rt1_off_while_p2c_stays_on(monkeypatch):
    """The whole point: a SINGLE-VARIABLE control for the G2 OFF arm.

    ``GRM_LSR_FIXES`` governs P2A+P2C+RT1 together, so measuring RT1 by
    turning it off would also remove the fit-time split and the arm would
    differ from its control by two changes.  This escape isolates RT1.
    """
    monkeypatch.setenv("GRM_LSR_FIXES", "1")
    monkeypatch.setenv("GRM_RT1_RULE", "0")
    assert rt1_enabled() is False
    # And P2C's own resolver is untouched — it still reads the family switch.
    assert ArenaCache._lsr_fixes_enabled() is True


def test_rt1_only_escape_takes_precedence_over_the_arena_resolver(monkeypatch):
    monkeypatch.setenv("GRM_RT1_RULE", "0")
    assert rt1_enabled(
        SimpleNamespace(_lsr_fixes_enabled=lambda: True)) is False
    monkeypatch.setenv("GRM_RT1_RULE", "1")
    assert rt1_enabled(
        SimpleNamespace(_lsr_fixes_enabled=lambda: False)) is True


def test_rt1_only_escape_fails_closed_to_on(monkeypatch):
    """Same direction the family switch fails: a mistyped value KEEPS RT1."""
    monkeypatch.setenv("GRM_RT1_RULE", "maybe")
    assert rt1_enabled() is True


# ---------------------------------------------------------------------------
# End to end through decisive_admission_profile — the lived shape
# ---------------------------------------------------------------------------


#: Texts at the LIVED graft indices: 0 praxis fact, 1 solace fact, 2 the split
#: index parent, 3 and 4 its two children.  Only graft 1 binds
#: "current solace key value".
_SOLACE_TEXTS = [
    "the current praxis dock value is quartz-8-jade",
    "the current solace key value is raven-9-ivory",
    "long planning transcript naming the praxis dock and the solace key "
    "as index context and owning neither fact",
    "long planning transcript naming the praxis dock and the solace key "
    "while cataloging search terms but owning neither fact",
    "after that background they recorded that the current tundra ledger "
    "value is sable-0-copper and the solace key reference is index context",
]


def _profile_arena(texts, *, split_members=()):
    """A stubbed arena carrying only what ``decisive_admission_profile``
    reads.  The router is PINNED to the lived ranking so the test measures
    the admission rule, not a re-derived latent score."""
    arena = ArenaCache.__new__(ArenaCache)
    members = {int(v) for v in split_members}
    arena.grafts = [
        {"text": text, "ntok": len(str(text).split()), "kind": "turn",
         "rare": ArenaCache._rare_tokens(text), "sources": [], "cent": None,
         "metadata": ({"width_guard_child": True} if index in members else {})}
        for index, text in enumerate(texts)
    ]
    arena._route_cand_base = lambda: list(range(len(texts)))
    arena._probe_key = lambda text: text
    arena.last_route_backend = "python"
    arena.route = lambda question, *, exclude, limit, probe_key=None: [
        value for value in LIVED_SOLACE_RANKING if value not in set(exclude)
    ][:limit]
    # A score map that REPRODUCES the pinned ranking, so the frozen A-DEC
    # reconstruction guard passes (that guard checks the ROUTER against the
    # score law, and RT1 must leave it checking exactly that) AND reproduces
    # the LIVED TIE: ranks 1 and 2 carry the SAME score, because the index
    # parent's ``_cent_score`` IS its winning child's via ``child_cents``.
    # That tie is what makes ``route_margin_1_2 == 0.0`` in the receipts and
    # what sends A-DEC down ``one_off_rank_identifier_insurance_k3`` instead
    # of the fitted-margin branch.  Ties break by ascending graft id in the
    # reference sort, so 2 must precede 4 by id as well as by score.
    pinned = {2: 5.0, 4: 5.0, 3: 3.0, 1: 2.0, 0: 1.0}
    arena._vector_route_scores = lambda p, cand: {
        int(i): pinned[int(i)] for i in cand if int(i) in pinned}
    arena._length_debias_scores = lambda base, cand: base
    arena._normalize_scores = lambda base: base
    arena._lex_bonus = lambda qlex, g: 0.0
    return arena


def test_profile_flips_the_branch_from_insurance_to_decisive(monkeypatch):
    """THE FIX, end to end.

    BEFORE: ranking ``[2, 4, 3, 1, 0]``, the sole binder off rank 1, margin
    0.0 (a tie, not above the fitted threshold) -> branch
    ``one_off_rank_identifier_insurance_k3`` and ``rank_plan = [2, 4, 3]``:
    the plan is made entirely of a competitor's chunks and the turn refuses.

    AFTER: the three non-binding split members are demoted, graft 1 is rank 1
    -> branch ``exactly_one_identifier_decisive_rank1`` and
    ``rank_plan = [1]``: the fact node, alone.
    """
    monkeypatch.setenv("GRM_LSR_FIXES", "1")
    arena = _profile_arena(_SOLACE_TEXTS, split_members=(2, 3, 4))
    profile = decisive_admission_profile(
        arena, "What is the current Solace key value?", exclude=(),
        route_limit=5)
    assert profile["admission_ranking_before_demotion"] == LIVED_SOLACE_RANKING
    assert profile["ranking"][0] == 1, profile["ranking"]
    assert profile["rank_plan"] == [1], profile["rank_plan"]
    assert profile["policy_branch"] == "exactly_one_identifier_decisive_rank1"
    assert profile["admission_split_child_demoted"] is True
    assert profile["admission_split_demoted_ids"] == [2, 4, 3]
    assert profile["admission_split_family_ids"] == [2, 3, 4]


def test_profile_with_the_switch_off_reproduces_the_lived_defect(monkeypatch):
    """OFF is the LEGACY PATH: the same wrong ranking, the same wrong branch,
    the same wrong plan the lived run recorded."""
    monkeypatch.setenv("GRM_LSR_FIXES", "0")
    arena = _profile_arena(_SOLACE_TEXTS, split_members=(2, 3, 4))
    profile = decisive_admission_profile(
        arena, "What is the current Solace key value?", exclude=(),
        route_limit=5)
    assert profile["ranking"] == LIVED_SOLACE_RANKING
    assert profile["rank_plan"] == [2, 4, 3]
    assert profile["policy_branch"] == "one_off_rank_identifier_insurance_k3"
    assert profile["admission_split_child_demoted"] is False
    assert profile["admission_split_demoted_ids"] == []


def test_profile_is_identical_on_off_when_nothing_ever_split(monkeypatch):
    """A repository the width guard never touched: ON and OFF must agree
    field for field.  This is the byte-identity claim for the overwhelming
    majority of traffic, not merely for the flag."""
    question = "What is the current Solace key value?"
    monkeypatch.setenv("GRM_LSR_FIXES", "1")
    on = decisive_admission_profile(
        _profile_arena(_SOLACE_TEXTS), question, exclude=(), route_limit=5)
    monkeypatch.setenv("GRM_LSR_FIXES", "0")
    off = decisive_admission_profile(
        _profile_arena(_SOLACE_TEXTS), question, exclude=(), route_limit=5)
    assert on == off, "no split family => the rule must be invisible"


def test_the_reconstruction_guard_still_checks_the_router(monkeypatch):
    """The frozen A-DEC guard compares the ROUTER against the score law.

    RT1's reorder happens after the router returns, so the guard is checked
    against ``ranking_before`` — otherwise it would fire on the very reorder
    it is meant to be blind to.  A genuinely inconsistent router must STILL
    raise.
    """
    monkeypatch.setenv("GRM_LSR_FIXES", "1")
    arena = _profile_arena(_SOLACE_TEXTS, split_members=(2, 3, 4))
    # Scores that do NOT reproduce the pinned router order.
    arena._vector_route_scores = lambda p, cand: {
        int(i): float(int(i)) for i in cand}
    with pytest.raises(AdmissionPolicyError):
        decisive_admission_profile(
            arena, "What is the current Solace key value?", exclude=(),
            route_limit=5)


def test_tundra_profile_still_routes_to_the_chunk_that_carries_its_value(
        monkeypatch):
    """CONTROL, end to end: the tundra probe must not regress.

    Graft 4's own text carries ``sable-0-copper`` for the Tundra ledger and
    binds "current tundra ledger value".  The ONLY binder is therefore itself
    a split member, so the rule's second no-op condition fires: RT1 protects a
    non-member against a member and never re-sorts a family against itself.
    The ranking is untouched and 4 stays in the plan.
    """
    monkeypatch.setenv("GRM_LSR_FIXES", "1")
    arena = _profile_arena(_SOLACE_TEXTS, split_members=(2, 3, 4))
    on = decisive_admission_profile(
        arena, "What is the current Tundra ledger value?", exclude=(),
        route_limit=5)
    assert on["admission_split_child_demoted"] is False
    assert on["ranking"] == LIVED_SOLACE_RANKING
    assert 4 in on["rank_plan"], on["rank_plan"]
    assert on["identified_candidates"] == [4]
    # And every DECISION field is identical to the legacy path: the rule
    # examined the family and changed nothing.  ``admission_split_family_ids``
    # is deliberately allowed to differ — ON names the family it looked at,
    # OFF names none because the rule never ran.  That is the NEVER-SILENT
    # receipt doing its job, not a behaviour change.
    monkeypatch.setenv("GRM_LSR_FIXES", "0")
    off = decisive_admission_profile(
        _profile_arena(_SOLACE_TEXTS, split_members=(2, 3, 4)),
        "What is the current Tundra ledger value?", exclude=(), route_limit=5)
    informational = {"admission_split_family_ids"}
    assert {k: v for k, v in on.items() if k not in informational} == {
        k: v for k, v in off.items() if k not in informational}
    assert off["admission_split_family_ids"] == []


# ---------------------------------------------------------------------------
# Receipts — NEVER SILENT
# ---------------------------------------------------------------------------


def test_receipt_fields_are_present_even_when_nothing_was_demoted():
    fields = rt1_info_fields()
    assert fields["admission_split_child_demoted"] is False
    assert fields["admission_split_demoted_ids"] == []
    assert fields["admission_split_family_ids"] == []
    assert fields["admission_ranking_before_demotion"] == []


def test_admission_info_fields_carry_the_rt1_receipt():
    fields = admission_info_fields({
        "policy_branch": "exactly_one_identifier_decisive_rank1",
        "rank_plan": [1],
        "identifier_hit_count": 1,
        "identified_candidates": [1],
        "route_margin_1_2": 0.0,
        "route_margin_evaluated": True,
        "admission_split_demoted_ids": [2, 4, 3],
        "admission_split_family_ids": [2, 3, 4],
        "admission_ranking_before_demotion": LIVED_SOLACE_RANKING,
    })
    assert fields["admission_split_child_demoted"] is True
    assert fields["admission_split_demoted_ids"] == [2, 4, 3]
    assert fields["admission_ranking_before_demotion"] == LIVED_SOLACE_RANKING


def test_admission_info_fields_tolerate_a_pre_rt1_profile():
    """An older frame's profile has no RT1 keys; the receipt must report the
    rule as inert rather than raising."""
    fields = admission_info_fields({
        "policy_branch": "exactly_one_identifier_decisive_rank1",
        "rank_plan": [1],
        "identifier_hit_count": 1,
        "identified_candidates": [1],
        "route_margin_1_2": 0.5,
        "ranking": [1, 0],
    })
    assert fields["admission_split_child_demoted"] is False
    assert fields["admission_ranking_before_demotion"] == [1, 0]
