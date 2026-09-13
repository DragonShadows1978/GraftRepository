"""GRM-F3 — routing at distance: why the intact record is never ranked.

WHAT THIS MODULE PINS
---------------------
The F3 diagnosis found NO single mechanism behind >= 5 of the 7 miss rows
(max 3/7), so per the order's item 3 NO FIX LANDS and `core/` is untouched.
What lands instead is this characterization suite: every mechanism the
diagnosis named is pinned as an executable statement of CURRENT behaviour, so
that a later fix flips a RED here rather than changing behaviour silently.

Three of these tests are DELIBERATE CHARACTERIZATIONS OF A DEFECT
(`test_identifier_insurance_branch_*`).  They assert what the code does today,
which is WRONG, and each carries an explicit `DEFECT:` note saying so.  A fix
for the registered proposal MUST break them; that is their purpose.  They are
not a claim that the behaviour is correct.

The suite splits into two halves:

* PURE-CPU pins (no marker, always collected) — the admission rule, the
  identifier predicate and the grounding predicate are ordinary deterministic
  code, so the mechanisms are reproducible from literal strings with no
  repository, no artifacts and no GPU.
* CAMPAIGN-RECEIPT pins (`@pytest.mark.campaign_receipt`) — the ones that read
  the gitignored r2 receipts under `artifacts/grm_d1/lt1_1/`.  These are
  sha-bound to the LT1.1 r2 registration and are skipped by default, exactly
  as tests/conftest.py requires.

Prior art
---------
Characterization tests (pin current behaviour, including defects, before
changing it): Feathers, "Working Effectively with Legacy Code" (2004), ch. 13
— taken: the practice and the discipline of labelling a pinned defect as such.
The CPU-double / receipt-replay split follows the existing GRM C7 and
SCOUT-FIX-8 modules in this suite (GRM contributors, 2026) — taken: the
`campaign_receipt` marker convention and the "reproduce the production
predicate from literal strings" pattern.  No prior art known to me for the
specific finding pinned here (an identifier-insurance admission branch that
does not admit the identified candidate).  Unverified against the current
literature (no network in this sandbox); lead to check.  Search terms:
"retrieval re-ranking guarantee include matched entity", "must-include
constraint top-k retrieval", "characterization test legacy defect pin".
"""
import json
from pathlib import Path

import pytest

from core.graft_arena import ArenaCache
from core.grm_admission import (
    MARGIN_THRESHOLD,
    is_identifier_binding,
    margin_first_plan,
    normalized_words,
    policy_plan,
    shaped_identifier_tokens,
)

ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / 'artifacts/grm_d1/lt1_1'
REGISTRATION = 'artifacts/grm_d1/lt1_1/registration.json'


# --------------------------------------------------------- GRM-D2 re-pin
#
# This file is a CHARACTERIZATION suite (Feathers 2004, ch. 13): its
# "identifier insurance branch drops the sole binder" tests exist to record
# a DEFECT exactly as it behaved, so the fix can be shown against it.  F5
# is that fix, and GRM-D2 (2026-09-11) made F5 a shipped default -- which
# means that on a bare tree the defect no longer reproduces and these
# characterizations would silently become assertions about the fix instead.
#
# THE ASSERTIONS ARE UNCHANGED.  `GRM_LEGACY_DEFAULTS=1` restores the
# pre-F5 world, so the characterization still characterizes the thing it
# was written for.  The ON-arm counterparts live in
# tests/test_grm_f5_sole_binder_insurance.py, exactly as F5's module
# docstring promised ("the F3 pinned-defect tests stay untouched and the
# ON-arm counterparts are ADDED beside them").
@pytest.fixture(autouse=True)
def _grm_d2_legacy_defaults(monkeypatch):
    from core import grm_legacy_defaults as legacy_defaults
    monkeypatch.setenv(legacy_defaults.ENV_NAME, "1")

#: The r2 question and the three node texts the diagnosis turns on, quoted
#: verbatim from the frozen session repositories.  Kept as literals so the
#: PURE-CPU half needs no artifacts at all.
QUESTION = "What did we settle on for Breakwater's map position?"

#: arm A, node 61 (era, LIVE): carries the correct value but names the entity
#: "the Beacon" — the F2 chronicle-fold alias capture.
ARM_A_ERA_61 = (
    'ARCHIVE NOTE. CHRONICLE — In this period,  the Kestrel’s cargo '
    'allowance is 37 crates, the Lantern’s launch date is 18 October '
    '2196, the Galactic Hub still feels like a social place, the '
    'Promenade’s lamp spacing is 9 voxels, the maintenance crew of the '
    'Beacon is located at Iona Vale, and the Beacon’s map position is '
    '(-31, 48, 12).\n'
)

#: arm A, node 41 (fact, LIVE): a DIFFERENT entity's coordinates.  This is the
#: node the ladder served at d=100.
ARM_A_FACT_41 = (
    "Actually, Spindle's map position will be (64, -17, 8), replacing the "
    "earlier choice."
)

#: arm A+, node 64 (era, LIVE): carries the correct value AND names the entity
#: correctly ("the breakwater's coordinates").
ARM_APLUS_ERA_64 = (
    'ARCHIVE NOTE. Combined record: during these conversations,  the '
    'maintenance crew of the comm tower, named Iona Vale, recorded that the '
    'breakwater’s coordinates are (-31, 48, 12). The Forge, formerly '
    'known as Foundry, now holds an iron reserve of 91 ingots, surpassing any '
    'earlier count. Meanwhile, the water reserve at Orchard, also called '
    '“the Cistern,” is logged at 152 litres, again surpassing prior '
    'records.\n'
)


def _identifier_kwargs(question=QUESTION):
    """The production identifier surface for `admission_rule == margin_first`.

    `decisive_admission_profile` overwrites `ordered`/`rare` with
    `shaped_identifier_tokens` under margin_first (grm_admission.py, the
    `if margin_first:` branch), so the scan reduces to a rare-token subset
    test.  Reproduced here exactly, not approximated.
    """
    ordered = shaped_identifier_tokens(ArenaCache, question)
    return dict(ordered_identifier_tokens=ordered, rare_identifier_tokens=set(ordered))


# ---------------------------------------------------------------------------
# PURE-CPU pins — no artifacts, no repository, no GPU.
# ---------------------------------------------------------------------------

def test_shaped_identifier_for_the_r2_question_is_the_entity_name():
    assert shaped_identifier_tokens(ArenaCache, QUESTION) == ['breakwater']


def test_alias_capture_removes_the_identifier_from_the_intact_record():
    """Arm A cause: the digest holds the VALUE but not the IDENTIFIER.

    This is the F2 defect's routing consequence.  Node 61 is the only LIVE
    node in arm A carrying (-31, 48, 12), and the identifier scan cannot see
    it, because the chronicle fold rebound the fact onto "the Beacon".
    """
    words = set(normalized_words(ARM_A_ERA_61))
    assert '48' in words and '12' in words, 'the value IS in the node'
    assert 'beacon' in words, 'the fold wrote the alias'
    assert 'breakwater' not in words, 'the fold dropped the asked-for entity'
    assert not is_identifier_binding(
        candidate_text=ARM_A_ERA_61, **_identifier_kwargs())


def test_arm_aplus_digest_keeps_the_identifier_and_does_bind():
    """The A+ contrast: same value, entity preserved, so the scan finds it."""
    assert is_identifier_binding(
        candidate_text=ARM_APLUS_ERA_64, **_identifier_kwargs())


def test_grounding_rejects_a_correct_answer_against_an_alias_captured_record():
    """Arm A's serving cause, reproduced from the production predicate.

    The question carries no rare token ("breakwater" has no digit and is not
    ALL-CAPS), so grounding's identifier short-circuit does not fire and the
    verdict falls to `content <= have`.  A CORRECT answer naming Breakwater
    contributes "breakwater" to `content`, which the alias-captured node can
    never supply -- so the correct answer is ruled UNGROUNDED, while the
    confabulation read off the wrong-entity fact node grounds cleanly.
    """
    assert ArenaCache._rare_tokens(QUESTION) == set(), (
        'no rare token in the question: the identifier short-circuit is dead')

    def content(answer):
        return ((ArenaCache._rare_tokens(answer) | ArenaCache._caps_tokens(answer))
                - ArenaCache._rare_tokens(QUESTION)
                - ArenaCache._caps_tokens(QUESTION, skip_sentence_initial=False))

    def have(text):
        return (ArenaCache._rare_tokens(text)
                | ArenaCache._caps_tokens(text, False))

    correct = 'The map position for Breakwater is (-31, 48, 12).'
    confabulated = 'We settled on (64, -17, 8) for Breakwater’s map position.'

    # The CORRECT answer, against the node that actually holds the value.
    assert 'breakwater' in content(correct)
    assert 'breakwater' not in have(ARM_A_ERA_61)
    assert not content(correct) <= have(ARM_A_ERA_61), (
        'grounding false-negative on the intact record')

    # The CONFABULATION, against the wrong-entity node the ladder descended to.
    assert content(confabulated) <= have(ARM_A_FACT_41), (
        'grounding false-positive on the wrong-entity record')


def test_identifier_insurance_branch_drops_the_sole_binder_margin_first():
    """DEFECT (characterization, margin_first / FIX-6).

    Reproduces arm A+ `recall_3_100` verbatim from its frozen receipt values:
    the scan identified node 64 as the SOLE identifier binder, the margin was
    below threshold, and the branch named
    `margin_insurance_k3_identifier_tiebreak` returned `ranking[:3]` -- which
    does NOT contain node 64.

    DEFECT: a branch whose name says "identifier tiebreak" performs no
    tiebreak when the binder's score is not tied with rank 1.  The sole
    identified candidate is silently dropped from the plan.  A fix for the
    registered F3 proposal MUST break this assertion.
    """
    ranking = [116, 110, 42, 84, 64, 55]
    margin = 0.06734233962214775
    assert margin < MARGIN_THRESHOLD
    plan, branch, _ = margin_first_plan(
        ranking=ranking, route_margin_1_2=margin,
        identified_candidates=[64], scores=None)
    assert branch == 'margin_insurance_k3_identifier_tiebreak'
    assert plan == [116, 110, 42]
    assert 64 not in plan, 'DEFECT PINNED: sole identifier binder not in plan'


def test_identifier_insurance_branch_drops_the_sole_binder_all_tokens_bind():
    """DEFECT (characterization, the frozen all_tokens_bind rule).

    The same receipt through the OTHER admission rule.  The branch is even
    called `one_off_rank_identifier_insurance_k3`, yet it "insures" nothing:
    it returns `ranking[:3]`, and the off-rank binder it is named for is
    outside that window.  Both shipping rules share the defect, so it is NOT a
    FIX-6 regression -- that matters for the proposal's scope.
    """
    plan, branch = policy_plan(
        ranking=[116, 110, 42, 84, 64, 55], identified_candidates=[64],
        route_margin_1_2=0.06734233962214775)
    assert branch == 'one_off_rank_identifier_insurance_k3'
    assert 64 not in plan, 'DEFECT PINNED: "insurance" branch does not insure'


def test_two_or_more_binders_are_all_admitted_regardless_of_rank():
    """The asymmetry that makes the defect a defect, not a design choice.

    With TWO binders the frozen rule admits EVERY identified candidate, even
    ones the router never ranked (`declared_synthesis_identified_set`).  With
    ONE it admits none.  Any principled fix removes that discontinuity.
    """
    plan, branch = policy_plan(
        ranking=[116, 110, 42, 84, 64, 55], identified_candidates=[64, 99],
        route_margin_1_2=0.06734233962214775)
    assert branch == 'declared_synthesis_identified_set'
    assert 64 in plan and 99 in plan


def test_margin_above_threshold_is_the_only_reason_d50_survived():
    """The d=50 -> d=100 contrast, both rows from their frozen receipts.

    Nothing about the REPOSITORY changed the verdict: the same binder, the
    same rule.  Only the score margin moved -- 0.322 (decisive, plan = the
    binder) to 0.067 (not decisive, plan = three non-binders).
    """
    d50, branch50, _ = margin_first_plan(
        ranking=[28, 42, 18, 20, 46, 12], route_margin_1_2=0.32210014929976594,
        identified_candidates=[28], scores=None)
    assert branch50 == 'fit_margin_decisive_rank1' and d50 == [28]

    d100, branch100, _ = margin_first_plan(
        ranking=[116, 110, 42, 84, 64, 55], route_margin_1_2=0.06734233962214775,
        identified_candidates=[64], scores=None)
    assert branch100 == 'margin_insurance_k3_identifier_tiebreak'
    assert d100 == [116, 110, 42]


def test_mounts_receipt_field_is_one_based_and_is_not_a_graft_index():
    """The round-1 diagnostic's mis-read, pinned at its source.

    `graft_arena` writes `info["mounts"] = [i + 1 for i in picks]` -- a
    1-BASED display convention.  `miss_causes.json` resolved `served_nodes`
    from it, landing on graft `i+1` (an unrelated turn) in all 7 miss rows,
    which is where the order's "the mounted node is an unrelated fact" premise
    came from.  `mounted_ids` / `fit_seated` are the graft indices.
    """
    source = (ROOT / 'core/graft_arena.py').read_text(encoding='utf-8')
    assert '"mounts": [i + 1 for i in picks]' in source, (
        'the 1-based mounts convention moved; re-check any consumer that '
        'treats info["mounts"] as graft indices')


# ---------------------------------------------------------------------------
# CAMPAIGN-RECEIPT pins — read the gitignored LT1.1 r2 receipts.
# ---------------------------------------------------------------------------

def _probe(arm_dir, cell, probe_id):
    path = R2 / arm_dir / 'cells' / cell / 'probes.jsonl'
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record['probe_id'] == probe_id:
            return record
    raise AssertionError('%s not found in %s' % (probe_id, path))


def _manifest(arm_dir, cell):
    return json.loads((R2 / arm_dir / 'cells' / cell / 'session/repository'
                       / 'manifest.json').read_text(encoding='utf-8'))


@pytest.mark.campaign_receipt(registration=REGISTRATION)
@pytest.mark.parametrize('arm_dir,cell,probe_id,expected_hits', [
    ('run_A', 'A-057-064', 'recall_3_50', []),
    ('run_A', 'A-111-118', 'recall_3_100', []),
    ('run_A', 'A-157-164', 'recall_3_150', []),
    ('run_A', 'A-197-200', 'recap_3', []),
    ('run_Aplus', 'A-057-064', 'recall_2_50', []),
    ('run_Aplus', 'A-111-118', 'recall_2_100', []),
    ('run_Aplus', 'A-157-164', 'recall_2_150', []),
    ('run_Aplus', 'A-057-064', 'recall_3_50', [28]),
    ('run_Aplus', 'A-111-118', 'recall_3_100', [64]),
    ('run_Aplus', 'A-157-164', 'recall_3_150', [64]),
])
def test_cpu_identifier_scan_reproduces_every_gpu_route_receipt(
        arm_dir, cell, probe_id, expected_hits):
    """All 10 route-table rows: the CPU scan == the GPU receipt, exactly.

    This is the proof that the diagnosis traverses the production path: the
    identifier scan is re-run over the frozen session repository with the same
    eligibility law `_route_cand_base` applies (not retired, not recall-kind),
    and must return the receipt's `identified_candidates` verbatim.
    """
    record = _probe(arm_dir, cell, probe_id)
    nodes = _manifest(arm_dir, cell)['nodes']
    kwargs = _identifier_kwargs(record['question'])
    eligible = [i for i, g in enumerate(nodes)
                if not g.get('retired') and g.get('kind', 'turn') != 'recall']
    hits = [i for i in eligible
            if is_identifier_binding(
                candidate_text=str(nodes[i].get('text') or ''), **kwargs)]
    receipt = record['memory']['route_info']['_route_observation'][
        'admission_profile']['identified_candidates']
    assert hits == expected_hits
    assert hits == receipt, 'CPU scan diverged from the frozen GPU receipt'


@pytest.mark.campaign_receipt(registration=REGISTRATION)
@pytest.mark.parametrize('arm_dir,cell,probe_id', [
    ('run_A', 'A-111-118', 'recall_3_100'),
    ('run_A', 'A-157-164', 'recall_3_150'),
    ('run_A', 'A-197-200', 'recap_3'),
    ('run_Aplus', 'A-111-118', 'recall_2_100'),
    ('run_Aplus', 'A-157-164', 'recall_2_150'),
    ('run_Aplus', 'A-111-118', 'recall_3_100'),
    ('run_Aplus', 'A-157-164', 'recall_3_150'),
])
def test_miss_rows_mounts_field_is_seated_plus_one(arm_dir, cell, probe_id):
    """All 7 miss rows: `mounts` == `fit_seated` + 1.

    Receipt-side confirmation of the 1-based convention pinned above, and the
    evidence that the order's "the mounted node is an unrelated fact" premise
    is a diagnostic artifact rather than a router behaviour.
    """
    route = _probe(arm_dir, cell, probe_id)['memory']['route_info']
    seated = [int(v) for v in route['fit_seated']]
    assert [int(v) for v in route['mounts']] == [i + 1 for i in seated]


@pytest.mark.campaign_receipt(registration=REGISTRATION)
def test_arm_a_ladder_rejected_the_intact_record_then_served_a_confabulation():
    """arm A `recall_3_100`, the sharpest row: first-grounded-wins loses.

    Trip 0 mounted node 61 -- the ONLY live node carrying (-31, 48, 12) -- and
    grounding returned false.  Trip 3 mounted node 41, a different entity's
    coordinates, grounding returned true, and the ladder served it.  The
    ranker DID reach the intact record; the grounding gate discarded it.
    """
    route = _probe('run_A', 'A-111-118', 'recall_3_100')['memory']['route_info']
    profile = route['_route_observation']['admission_profile']
    trips = route['_route_observation']['trips']

    assert profile['ranking'][0] == 61, 'the intact record ranked FIRST'
    assert 61 in profile['rank_plan'], 'and it was planned'
    assert trips[0]['mount_set'] == [61] and trips[0]['grounded'] is False
    assert trips[-1]['mount_set'] == [41] and trips[-1]['grounded'] is True
    assert route['fit_seated'] == [41]
    assert route['served_without_plan_head'] is True

    nodes = _manifest('run_A', 'A-111-118')['nodes']
    assert 'breakwater' not in set(normalized_words(nodes[61]['text']))
    assert '(-31, 48, 12)' in nodes[61]['text']


@pytest.mark.campaign_receipt(registration=REGISTRATION)
@pytest.mark.parametrize('probe_id', ['recall_2_100', 'recall_2_150'])
def test_arm_aplus_commtower_rows_are_not_routing_misses(probe_id):
    """RED against the order's premise for 2 of its 7 rows.

    These were filed as "fabricated a Commtower crew ... the mounted node is
    an unrelated fact".  The receipt says otherwise: node 64, whose text reads
    "the maintenance crew of the comm tower, named Iona Vale", ranked FIRST,
    was the whole plan, was seated, and GROUNDED.  The route was correct end
    to end; the reader confabulated with the answer in context.  Nothing in
    admission or routing can fix these rows.
    """
    cell = 'A-111-118' if probe_id.endswith('100') else 'A-157-164'
    record = _probe('run_Aplus', cell, probe_id)
    route = record['memory']['route_info']
    profile = route['_route_observation']['admission_profile']
    trips = route['_route_observation']['trips']

    assert profile['ranking'][0] == 64
    assert profile['rank_plan'] == [64]
    assert profile['policy_branch'] == 'fit_margin_decisive_rank1'
    assert route['fit_seated'] == [64]
    assert len(trips) == 1 and trips[0]['grounded'] is True

    nodes = _manifest('run_Aplus', cell)['nodes']
    assert 'Iona Vale' in nodes[64]['text'], 'the served node HELD the answer'
    assert 'Iona Vale' not in record['memory']['answer']


@pytest.mark.campaign_receipt(registration=REGISTRATION)
def test_arm_aplus_recall_3_100_is_a_scorer_miss_not_a_routing_miss():
    """RED against the order's premise for a 3rd row.

    The answer IS the expected value, written without parentheses and with
    U+2011/U+202F.  The registered secondary scorer (F4) marks it correct.
    """
    from scripts.grm_d1_scorer_v2 import score_v2_full
    record = _probe('run_Aplus', 'A-111-118', 'recall_3_100')
    assert record['memory']['score']['category'] == 'wrong_value'
    column2 = score_v2_full(record['memory']['answer'], record['expected'])
    assert column2['exact_correct'] is True
    assert column2['matched_variant'] == '-31, 48, 12'


@pytest.mark.campaign_receipt(registration=REGISTRATION)
def test_no_single_mechanism_covers_five_of_seven_miss_rows():
    """The order's item-3 gate, evaluated: 3/7 max, so NO FIX LANDS.

    Pinned so the "deliver the proposal only" decision is auditable rather
    than asserted.
    """
    causes = {
        ('A', 'recall_3_100'): 'alias_capture_grounding_false_negative',
        ('A', 'recall_3_150'): 'alias_capture_grounding_false_negative',
        ('A', 'recap_3'): 'alias_capture_grounding_false_negative',
        ('A+', 'recall_2_100'): 'reader_dilution_intact_mounted_and_grounded',
        ('A+', 'recall_2_150'): 'reader_dilution_intact_mounted_and_grounded',
        ('A+', 'recall_3_100'): 'scorer_miss_answer_glyph_correct',
        ('A+', 'recall_3_150'): 'identifier_insurance_branch_drops_sole_binder',
    }
    assert len(causes) == 7
    counts = {}
    for cause in causes.values():
        counts[cause] = counts.get(cause, 0) + 1
    assert max(counts.values()) == 3
    assert max(counts.values()) < 5, 'item 3 gate: proposal only, no fix'


# ---------------------------------------------------------------------------
# F4 — registered secondary scorer.  Pure CPU, no artifacts.
# ---------------------------------------------------------------------------

def test_f4_abstention_extension_covers_the_one_observed_r2_phrasing():
    from scripts.grm_d1_scorer_v2 import abstains_v2, score_v2_full
    observed = ('We’re still working on that. It’s a good question '
                'to keep in mind as we design the rest of the game.')
    assert abstains_v2(observed)
    assert score_v2_full(observed, '16 beds')['category'] == 'abstention'


def test_f4_extension_does_not_rescue_a_wrong_value_or_move_the_primary():
    from scripts.grm_lt1 import score as primary
    from scripts.grm_d1_scorer_v2 import abstains_v2, score_v2_full
    wrong = 'We settled on (64, -17, 8) for Breakwater’s map position.'
    assert not abstains_v2(wrong)
    assert score_v2_full(wrong, '(-31, 48, 12)')['category'] == 'wrong_value'
    # The registered PRIMARY is untouched: it still banks the observed
    # abstention as a wrong value, which is exactly why column 2 exists.
    observed = 'We’re still working on that.'
    assert primary(observed, '16 beds')['category'] == 'wrong_value'


def test_f4_primary_alternation_is_a_verbatim_copy_of_the_registered_scorer():
    """If the primary's regex ever moves, this RED says the copy is stale."""
    import re as _re
    from scripts.grm_d1_scorer_v2 import PRIMARY_ABSTAIN_ALTERNATION
    source = (ROOT / 'scripts/grm_lt1.py').read_text(encoding='utf-8')
    for alternative in PRIMARY_ABSTAIN_ALTERNATION.split('|'):
        assert alternative in source, (
            'copied alternative %r is no longer in scripts/grm_lt1.py'
            % alternative)
    assert _re.compile(PRIMARY_ABSTAIN_ALTERNATION)
