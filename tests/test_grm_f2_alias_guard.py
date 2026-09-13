"""GRM-F2 — chronicle folds must not rebind facts onto an alias.  CPU only.

WHAT THIS FILE IS EVIDENCE FOR, AND WHAT IT IS NOT.
Every test runs the REAL production objects — ``GraftRepository``,
``ArenaCache`` (via ``scripts.grm_c7_diagnose.CPUArena``, which replaces the
numerical/payload seams only), the real ``consolidate``, the real FIX-3
Harmony wrapper, the real FIX-5 ``[source N]`` enumeration, the real coverage
gate — against deterministic CPU model doubles.  That makes it PLUMBING and
DETECTOR evidence: which window a fold gets, what the enumeration says, what
the coverage gate scores, and whether the attribution QC accepts or rejects
the digest that comes back.

It is NOT model-quality evidence, and it is not a claim that GPT-OSS-20B will
write an entity-correct digest once the alias turn is gone.  The receipt below
is a MEASURED production failure; the CPU reproduction shows the mechanism and
the guard's response.  Whether the guard changes the LT1.1 r3 arm-A numbers is
the lead's registered GPU contrast (``artifacts/grm_f2/flag_contract.json``).

THE RECEIPT (LT1.1 r2 arm A, ``artifacts/grm_d1/REPORT.md`` §3.3).
Digest node 24 of the arm-A session store folded turns ``[13, 14, 17, 18]``
and read "the maintenance crew of the Beacon is located at Iona Vale, and the
map position of the Beacon is (-31, 48, 12)".  The sources say Commtower's
crew and Breakwater's position; "the Beacon" is an alias for LANTERN, a third
entity owning neither fact.  Verbatim source and digest texts are frozen in
``artifacts/grm_f2/receipt_digest24.json``, copied from the read-only r2
run_A cell manifest at
``artifacts/grm_d1/lt1_1/run_A/cells/A-025-032/session/repository/``.

THE TWO DOUBLES, AND WHY BOTH.
``FaithfulDigestModel`` is an EXTRACTIVE double: it reads only the FIX-5
``[source N]`` lines out of the prompt it is handed and writes them back as
prose, carrying each source span's own entity with its own value.  It never
reads the mounted K/V, the receipt, or any expected answer.  It is the
control: it shows the production consolidation path REACHES a correct digest
on this window, so a later RED cannot be blamed on the harness.

``RecordedDigest24Model`` is a RECORDED-OUTPUT double: it replays the exact
digest 24 text from the receipt.  It is the failure mode.  The fixture must
fail on the ENTITY CHECK and not on the stub, which is asserted directly —
``test_recorded_digest24_passes_coverage_and_qc`` pins that the recorded text
scores ``best_cov == 1.0``, ``hit_count == 5/5`` and passes ``_digest_qc``,
so the only thing that can reject it is attribution.

THE ENTITY CHECK, as the order defines it: every (entity, attribute, value)
in the sources must appear in the digest with the SAME entity, or with an
alias registered for THAT entity only.  Implementation and its stated limits:
``core/grm_fold_alias_guard.py``.

Prior art: ``tests/test_grm_scout_fix5.py`` and
``tests/test_grm_a1_alias_fold.py`` (GRM contributors, 2026) supply the
repository/arena CPU double, the ``EnumeratedModel`` prompt-reading stimulus
idiom, the ``seed`` + ``_sync_lifecycle`` deposit shape, the byte-identity
record shape and the receipt-replay pattern; all reused.  The literature
annotation for the check itself is in ``core/grm_fold_alias_guard.py``'s
PRIOR ART block (Maynez 2020, Nan 2021, Goodrich 2019 — all UNVERIFIED, lead
to check).  No prior art known to me for this exact fixture composition.
"""
import json
import os
import re
from pathlib import Path

import pytest

from core.graft_arena import ArenaCache
from core import grm_fold_alias_guard as guard
from core import grm_legacy_defaults as legacy_defaults
from scripts.grm_c7_diagnose import Model, repository

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------- GRM-D2 re-pin
#
# GRM-D2 (2026-09-11) flipped F2's shipped default from OFF to ON.  THIS
# SUITE'S ASSERTIONS ARE UNCHANGED: they pin the pre-D2 behaviour, which is
# what ``GRM_LEGACY_DEFAULTS=1`` selects.  ``_grm_d2_legacy_defaults`` pins
# the process environment for the arena-driven tests; ``legacy()`` builds
# the explicit ``environ=`` dicts the resolver tests pass directly, which
# never consult ``os.environ``.

def legacy(**extra):
    """An ``environ`` mapping pinned to the pre-D2 defaults."""
    return {legacy_defaults.ENV_NAME: "1", **extra}


@pytest.fixture(autouse=True)
def _grm_d2_legacy_defaults(monkeypatch):
    monkeypatch.setenv(legacy_defaults.ENV_NAME, "1")
RECEIPT = json.loads(
    (ROOT / 'artifacts/grm_f2/receipt_digest24.json').read_text())

#: The four source turns of the receipt, in the order the fold saw them.
SOURCE_IDS = [int(v) for v in RECEIPT['sources']]
SOURCES = [RECEIPT['source_texts'][str(i)] for i in SOURCE_IDS]
#: Digest 24 verbatim, WITHOUT the "ARCHIVE NOTE. " prefix
#: ``_deposit_consolidation`` adds — that prefix is the depositor's, not the
#: model's, so a replay double must not emit it.
DIGEST24 = RECEIPT['digest_text'].split('ARCHIVE NOTE.', 1)[1].strip()

#: The five facts of this window, by ``ArenaCache._fact_set``.  Pinned as a
#: literal so a change to the fact vocabulary shows up here as a failure
#: rather than silently re-aiming the whole file.
NEED = {'12', '31', '48', 'iona', 'vale'}


class FaithfulDigestModel(Model):
    """Extractive double: prose from the FIX-5 enumeration, entity intact.

    Reads ONLY ``[source N]`` lines out of the prompt it is given.  It never
    inspects the mounted K/V (``self.injected``), the receipt, or an expected
    answer — the same independence contract as
    ``tests/test_grm_scout_fix5.EnumeratedModel``, which this is modelled on.
    Each enumerated span is rewritten and joined with ", and ", so the entity
    that owned a value in the source still owns it in the digest.
    """

    def __call__(self, ids, kv_caches=None, **kwargs):
        if kv_caches is None:
            prompt = self.codec.decode(ids[0])
            spans = re.findall(r'^\[source \d+\] (.+)$', prompt, re.M)
            if spans:
                body = ', and '.join(s.rstrip('. ') for s in spans)
                self.fold_output = (
                    'the archival record states that ' + body + '.<|end|>')
        return super().__call__(ids, kv_caches=kv_caches, **kwargs)


class RecordedDigest24Model(Model):
    """Recorded-output double: replays the receipt's digest 24 verbatim."""

    def __call__(self, ids, kv_caches=None, **kwargs):
        if kv_caches is None:
            self.fold_output = DIGEST24 + '<|end|>'
        return super().__call__(ids, kv_caches=kv_caches, **kwargs)


def seed(repo, texts=SOURCES):
    """Deposit the receipt's source turns as ordinary turn nodes."""
    ids = []
    for text in texts:
        idx = repo.arena.deposit(text)
        repo.arena.grafts[idx]['kind'] = 'turn'
        ids.append(idx)
    repo._sync_lifecycle()
    return ids


def fold_state(repo, ids, didx):
    """The observables a fold changes, as one comparable record."""
    return {
        'digest_idx': didx,
        'accepted': didx is not None,
        'result': dict(repo.arena.last_consolidation_result),
        'retired': [bool(repo.arena.grafts[i].get('retired')) for i in ids],
        'nodes': [{k: g.get(k) for k in
                   ('text', 'kind', 'retired', 'sources', 'ntok', 'no_fold',
                    'metadata')}
                  for g in repo.arena.grafts],
        'fold_history': repo.fold_history,
        'guard_history': repo.fold_guard_history,
    }


# ----------------------------------------------------------------- the flag

def test_flag_default_off_and_fails_closed(monkeypatch):
    """Default OFF; an unknown token fails CLOSED, like A1 / L2 / A-DEC."""
    monkeypatch.delenv(guard.ENV_NAME, raising=False)
    assert guard.fold_alias_guard_enabled() is False
    assert guard.env_fold_alias_guard_override() is None
    for token, want in (('1', True), ('true', True), ('YES', True),
                        ('on', True), ('0', False), ('off', False),
                        ('', False), ('banana', False), ('2', False)):
        # 'banana' / '2' still resolve False: GRM-D2 moved the fail-closed
        # rung from the override helper (which now returns None) to the
        # shipped default, and under this suite's legacy pin the shipped
        # default IS False.  Same verdict, stated one rung later.
        assert guard.fold_alias_guard_enabled(
            environ=legacy(**{guard.ENV_NAME: token})) is want, token
    # An explicit constructor value outranks the environment, both ways.
    assert guard.fold_alias_guard_enabled(
        True, environ=legacy(**{guard.ENV_NAME: '0'})) is True
    assert guard.fold_alias_guard_enabled(
        False, environ=legacy(**{guard.ENV_NAME: '1'})) is False


def test_flag_off_byte_identical(tmp_path, monkeypatch):
    """OFF must be byte-identical to the pre-F2 branch.

    The comparison is the flag ABSENT from the environment (the pre-F2 world,
    where no such variable existed) versus explicitly ``0``.  Every observable
    F2 could touch is captured: the node table with its metadata, the fold
    history, the consolidation receipt, and the model's own call log (inputs,
    outputs, injected text, position offsets) — so a changed PROMPT, a changed
    window or a changed number of generation steps would fail this too.
    """
    records = {}
    for label, flag in (('absent', None), ('explicit_off', '0')):
        monkeypatch.delenv(guard.ENV_NAME, raising=False)
        if flag is not None:
            monkeypatch.setenv(guard.ENV_NAME, flag)
        repo = repository(tmp_path / label, monkeypatch)
        repo.arena.m = RecordedDigest24Model(repo.arena.m.codec)
        try:
            assert repo.fold_alias_guard is False
            assert repo.arena.fold_attribution_guard is None
            ids = seed(repo)
            didx, _ = repo.arena.consolidate(ids)
            records[label] = {
                'state': fold_state(repo, ids, didx),
                'calls': repo.arena.m.calls,
            }
        finally:
            repo.close()
    assert (json.dumps(records['absent'], sort_keys=True, default=str)
            == json.dumps(records['explicit_off'], sort_keys=True,
                          default=str))
    # And OFF really does deposit the defective digest — the RED it is the
    # baseline for.
    assert records['absent']['state']['accepted'] is True


def test_flag_off_leaves_the_window_and_the_hook_untouched(
        tmp_path, monkeypatch):
    """With the flag OFF neither seam is reachable."""
    monkeypatch.delenv(guard.ENV_NAME, raising=False)
    repo = repository(tmp_path / 'repo', monkeypatch)
    try:
        ids = seed(repo)
        jobs = [('digest', list(ids))]
        assert repo._guard_fold_windows(jobs) is jobs
        assert repo.fold_guard_history == []
        assert repo.arena.fold_attribution_guard is None
    finally:
        repo.close()


def test_planner_only_stub_keeps_the_pre_f2_plan(tmp_path):
    """A repository built WITHOUT ``__init__`` must not see this treatment.

    ``tests/test_grm_s4_fold_order.py::_planner_repo`` builds a planner-only
    stub with ``object.__new__(GraftRepository)`` and a minimal attribute set
    — the existing idiom ``_librarian_jobs`` already honours for
    ``fold_order``.  Such a stub has no ``fold_alias_guard`` attribute at all,
    and F2 must hand its plan back unchanged rather than raising.

    RED receipt: the first regression run of this branch failed 8 tests in
    ``test_grm_s4_fold_order.py`` with
    ``AttributeError: 'GraftRepository' object has no attribute
    'fold_alias_guard'`` (artifacts/grm_f2/regression_red.log).
    """
    from types import SimpleNamespace

    from core.graft_repository import GraftRepository

    repo = object.__new__(GraftRepository)
    repo.path = str(tmp_path)
    repo.native_store = None
    repo.fold_order = 'age'
    repo.arena = SimpleNamespace(grafts=[], live_segs=[],
                                 ENABLE_ERA_FOLDING=True)
    assert not hasattr(repo, 'fold_alias_guard')
    jobs = [('digest', [0, 1])]
    assert repo._guard_fold_windows(jobs) is jobs
    assert not hasattr(repo, 'fold_guard_history')


# ------------------------------------------- the mechanism, measured on disk

def test_receipt_is_the_lt1_1_r2_arm_a_digest_24():
    """Pin what this file is about, straight from the frozen receipt."""
    assert SOURCE_IDS == [13, 14, 17, 18]
    assert RECEIPT['digest_node'] == 24
    # NB the double space after "the" is the depositor's, verbatim from the
    # store; the assertion keeps the receipt's bytes rather than tidying them.
    assert 'maintenance crew of the Beacon' in RECEIPT['digest_text']
    assert 'map position of the Beacon' in RECEIPT['digest_text']
    assert "Commtower's maintenance crew will be Iona Vale." in SOURCES[0]
    assert "Breakwater's map position will be (-31, 48, 12)." in SOURCES[1]
    assert "call Kestrel 'the Hauler'" in SOURCES[2]
    assert "call Lantern 'the Beacon'" in SOURCES[3]


def test_alias_turns_contribute_no_facts_and_no_enumeration(tmp_path,
                                                            monkeypatch):
    """The measurement that makes exclusion (a) FREE.

    Dropping the two alias turns leaves ``need`` and the FIX-5 enumerated
    span list byte-identical.  Their whole contribution to the fold is the
    K/V mount.  This is the evidence for choosing (a) over a coverage tweak.
    """
    repo = repository(tmp_path / 'repo', monkeypatch)
    try:
        arena = repo.arena
        assert arena._fact_set(SOURCES) == NEED
        assert arena._fact_set(SOURCES[:2]) == NEED
        for text in SOURCES[2:]:
            assert arena._fact_set([text]) == set(), text

        def enumerated(texts):
            return [re.findall(r'^\[source \d+\] (.+)$', prompt, re.M)
                    for prompt in arena._consolidation_prompts(False, texts)]

        with_alias = enumerated(SOURCES)
        without_alias = enumerated(SOURCES[:2])
        assert with_alias == without_alias
        assert with_alias[0] == [
            "Commtower's maintenance crew will be Iona Vale.",
            "Breakwater's map position will be (-31, 48, 12)."]
        for lines in with_alias:
            joined = ' '.join(lines)
            for absent in ('Beacon', 'Lantern', 'Hauler', 'Kestrel'):
                assert absent not in joined
    finally:
        repo.close()


def test_coverage_is_blind_to_attribution(tmp_path, monkeypatch):
    """Why the existing gate cannot catch this: ``need`` holds no entity.

    ``_fact_set``'s multi-word rule needs two consecutive capitalized words,
    and ``_caps_tokens`` matches ``^[A-Z][\\w-]+$`` on a whitespace-split
    token — which the POSSESSIVE surfaces "Commtower's" and "Breakwater's"
    both fail while "Beacon" passes.  So the rebound digest scores 1.0.
    """
    repo = repository(tmp_path / 'repo', monkeypatch)
    try:
        arena = repo.arena
        assert 'commtower' not in NEED and 'breakwater' not in NEED
        assert arena._caps_tokens(
            "Commtower's maintenance crew", False) == set()
        assert arena._caps_tokens(
            "Breakwater's map position", False) == set()
        assert 'beacon' in arena._caps_tokens(DIGEST24, False)
        assert arena._coverage(DIGEST24, NEED) == 1.0
        # The possessive-tolerant proxy does see them — that is the fix.
        assert 'commtower' in guard.entity_tokens(SOURCES[0])
        assert 'breakwater' in guard.entity_tokens(SOURCES[1])
    finally:
        repo.close()


# ---------------------------------------------------- reproduction: double 1

def test_faithful_double_reaches_an_entity_correct_digest(tmp_path,
                                                          monkeypatch):
    """CONTROL: the production path reaches a correct digest on this window.

    An extractive double that reads only the FIX-5 enumeration writes a
    digest that keeps Commtower with its crew and Breakwater with its
    coordinates.  It passes coverage, QC, AND the attribution check — so a
    RED from the recorded double below cannot be blamed on the harness.
    """
    monkeypatch.setenv(guard.ENV_NAME, '1')
    repo = repository(tmp_path / 'repo', monkeypatch)
    repo.arena.m = FaithfulDigestModel(repo.arena.m.codec)
    try:
        assert repo.fold_alias_guard is True
        ids = seed(repo)
        didx, text = repo.arena.consolidate(ids)
        assert didx is not None, repo.arena.last_consolidation_attempts
        assert repo.arena.last_consolidation_result['best_cov'] == 1.0
        assert 'Commtower' in text and 'Breakwater' in text
        assert 'Beacon' not in text
        receipt = repo.arena.last_consolidation_result['attribution']
        assert receipt['attribution_ok'] is True, receipt
        assert receipt['violations'] == []
    finally:
        repo.close()


# ---------------------------------------------------- reproduction: double 2

def test_recorded_digest24_passes_coverage_and_qc(tmp_path, monkeypatch):
    """The failure is the ENTITY CHECK, not the stub — pinned explicitly.

    The recorded digest 24 text clears every gate that existed before F2:
    shape QC, list QC, and coverage at a perfect 5/5.  Nothing but
    attribution can reject it.
    """
    monkeypatch.delenv(guard.ENV_NAME, raising=False)
    repo = repository(tmp_path / 'repo', monkeypatch)
    repo.arena.m = RecordedDigest24Model(repo.arena.m.codec)
    try:
        arena = repo.arena
        ids = seed(repo)
        didx, text = arena.consolidate(ids)
        assert didx is not None
        assert arena._digest_qc(text, None, forbid_lists=True) is True
        assert arena.last_consolidation_result['best_cov'] == 1.0
        assert arena.last_consolidation_result['need_count'] == len(NEED)
        assert arena.last_consolidation_result['hit_count'] == len(NEED)
        assert arena.grafts[didx]['sources'] == ids
        assert all(arena.grafts[i]['retired'] for i in ids)
    finally:
        repo.close()


def test_red_flag_off_deposits_the_rebound_digest(tmp_path, monkeypatch):
    """RED: with the flag OFF the defect reproduces end to end."""
    monkeypatch.delenv(guard.ENV_NAME, raising=False)
    repo = repository(tmp_path / 'repo', monkeypatch)
    repo.arena.m = RecordedDigest24Model(repo.arena.m.codec)
    try:
        ids = seed(repo)
        assert repo._fold_once(jobs=[('digest', list(ids))]) is True
        event = repo.fold_history[-1]
        assert event['accepted'] is True
        didx = event['digest_idx']
        text = repo.arena.grafts[didx]['text']
        assert 'the Beacon' in text
        assert 'Commtower' not in text and 'Breakwater' not in text
        # And the alias record is destroyed to produce it.
        assert all(repo.arena.grafts[i]['retired'] for i in ids)
        assert repo.fold_guard_history == []
        # The check names exactly what went wrong.
        violations = guard.attribution_violations(
            repo.arena._fact_set, SOURCES, text)
        assert len(violations) == 5, violations
        assert {v['value'] for v in violations} == NEED
        assert {tuple(v['entity']) for v in violations} == {
            ('commtower',), ('breakwater',)}
        for violation in violations:
            assert 'beacon' in violation['digest_entities']
            assert violation['reason'] == guard.REASON_ATTRIBUTION
    finally:
        repo.close()


def test_green_flag_on_b_rejects_the_rebound_digest(tmp_path, monkeypatch):
    """GREEN (b): the attribution QC aborts the fold; sources survive.

    The window is handed to ``consolidate`` DIRECTLY (all four sources), so
    this isolates (b) from (a): even with the alias turn present and the
    model replaying the defective text, the digest is never deposited.
    """
    monkeypatch.setenv(guard.ENV_NAME, '1')
    repo = repository(tmp_path / 'repo', monkeypatch)
    repo.arena.m = RecordedDigest24Model(repo.arena.m.codec)
    try:
        ids = seed(repo)
        before = len(repo.arena.grafts)
        didx, text = repo.arena.consolidate(ids)
        assert didx is None and text is None
        # FIX-3's abort contract, unchanged: nothing was deposited, nothing
        # was retired, every source is still an active reader and router.
        assert len(repo.arena.grafts) == before
        assert not any(repo.arena.grafts[i].get('retired') for i in ids)
        receipt = repo.arena.last_consolidation_result['attribution']
        assert receipt['attribution_ok'] is False
        assert len(receipt['violations']) == 5
        assert receipt['alias_bindings'] == {'beacon': 'lantern',
                                             'hauler': 'kestrel'}
        assert repo.fold_guard_history[-1]['reason'] == (
            guard.REASON_ATTRIBUTION)
    finally:
        repo.close()


def test_green_flag_on_a_excludes_the_alias_turns_from_the_window(
        tmp_path, monkeypatch):
    """GREEN (a): the planner drops the fact-less alias turns.

    ``_guard_fold_windows`` is the seam both ``_librarian_jobs`` return
    paths pass through.  The excluded nodes stay ACTIVE and are NOT marked
    ``no_fold`` — they remain available to A1's alias merge.
    """
    monkeypatch.setenv(guard.ENV_NAME, '1')
    repo = repository(tmp_path / 'repo', monkeypatch)
    repo.arena.m = FaithfulDigestModel(repo.arena.m.codec)
    try:
        ids = seed(repo)
        jobs = repo._guard_fold_windows([('digest', list(ids))])
        assert jobs == [('digest', [ids[0], ids[1]])]
        record = repo.fold_guard_history[-1]
        assert record['reason'] == guard.REASON_WINDOW_EXCLUDED
        assert record['excluded'] == [ids[2], ids[3]]
        assert record['kept'] == [ids[0], ids[1]]
        # The excluded nodes are untouched: active, foldable, alias-mergeable.
        for i in (ids[2], ids[3]):
            assert not repo.arena.grafts[i].get('retired')
            assert not repo.arena.grafts[i].get('no_fold')
        # And the reduced window still folds, entity-correct.
        didx, text = repo.arena.consolidate(jobs[0][1])
        assert didx is not None
        assert 'Commtower' in text and 'Breakwater' in text
        assert 'Beacon' not in text
    finally:
        repo.close()


def test_green_end_to_end_through_fold_once(tmp_path, monkeypatch):
    """(a) and (b) together on the production ``_fold_once`` path.

    With the guard ON and the recorded double still replaying the defective
    text, the window is first reduced to ``[13, 14]`` by (a); the model then
    emits digest 24 anyway, and (b) rejects it.  ``_fold_once`` takes its
    EXISTING abort branch: sources stay active and are exempted so the
    planner advances.  Nothing about that branch is new.
    """
    monkeypatch.setenv(guard.ENV_NAME, '1')
    repo = repository(tmp_path / 'repo', monkeypatch)
    repo.arena.m = RecordedDigest24Model(repo.arena.m.codec)
    try:
        ids = seed(repo)
        jobs = repo._guard_fold_windows([('digest', list(ids))])
        assert repo._fold_once(jobs=jobs) is True
        event = repo.fold_history[-1]
        assert event['accepted'] is False
        assert event['sources'] == [ids[0], ids[1]]
        assert repo.folds_aborted == 1
        assert repo.arena.grafts[ids[0]]['no_fold'] is True
        assert repo.arena.grafts[ids[1]]['no_fold'] is True
        # The alias turns were excluded, not folded and not exempted.
        for i in (ids[2], ids[3]):
            assert not repo.arena.grafts[i].get('no_fold')
        # No digest naming "the Beacon" was ever deposited.
        assert not any(g.get('kind') == 'digest'
                       and 'the Beacon' in g.get('text', '')
                       for g in repo.arena.grafts)
        reasons = [r['reason'] for r in repo.fold_guard_history]
        assert guard.REASON_WINDOW_EXCLUDED in reasons
        assert guard.REASON_ATTRIBUTION in reasons
    finally:
        repo.close()


# ------------------------------------------------------------- the check API

def test_entity_check_definition():
    """The rule, stated as four properties of ``attribution_violations``."""
    fact_set = ArenaCache._fact_set
    src = ["Commtower's maintenance crew will be Iona Vale."]
    # 1. SAME entity -> no violation.
    assert guard.attribution_violations(
        fact_set, src,
        'the maintenance crew of Commtower is Iona Vale.') == []
    # 2. DIFFERENT entity -> violation.
    wrong = guard.attribution_violations(
        fact_set, src, 'the maintenance crew of Breakwater is Iona Vale.')
    assert len(wrong) == 2 and {v['value'] for v in wrong} == {'iona', 'vale'}
    # 3. A REGISTERED alias for THAT entity -> no violation.
    with_alias = src + ["Let's call Commtower 'the Spire' from now on."]
    assert guard.attribution_violations(
        fact_set, with_alias,
        'the maintenance crew of the Spire is Iona Vale.') == []
    # 4. An alias registered for a DIFFERENT entity is NOT accepted — this is
    #    the receipt's exact shape, in miniature.
    other = src + ["Let's call Lantern 'the Beacon' from now on."]
    assert len(guard.attribution_violations(
        fact_set, other,
        'the maintenance crew of the Beacon is Iona Vale.')) == 2


def test_check_is_silent_on_a_value_the_digest_dropped():
    """A missing value is COVERAGE's question, never attribution's."""
    fact_set = ArenaCache._fact_set
    src = ["Breakwater's map position will be (-31, 48, 12)."]
    assert guard.attribution_violations(
        fact_set, src, 'the archival record mentions Breakwater.') == []


def test_check_is_silent_when_the_source_clause_has_no_entity():
    """The guard never guesses an owner it cannot read."""
    fact_set = ArenaCache._fact_set
    assert guard.attribution_violations(
        fact_set, ['the spacing will be 9 voxels.'],
        'the spacing of Promenade is 9 voxels.') == []


def test_exclusion_keeps_a_fact_bearing_alias_turn():
    """Excluding a turn that carries a fact would LOSE the fact.

    The exclusion is narrow by construction: both conditions (parses as an
    alias edge AND contributes nothing to ``_fact_set``) are required.
    """
    fact_set = ArenaCache._fact_set
    factless = "Let's call Lantern 'the Beacon' from now on."
    bearing = ("Let's call Kestrel 'the Hauler' from now on; "
               "its cargo allowance is 37 crates.")
    plain = "Commtower's maintenance crew will be Iona Vale."
    assert fact_set([factless]) == set()
    assert fact_set([bearing])
    assert guard.fold_window_excludes(
        fact_set, [plain, factless, bearing]) == [1]


def test_exclusion_never_empties_a_window():
    """All-alias windows are handed back unchanged, not emptied."""
    fact_set = ArenaCache._fact_set
    aliases = ["Let's call Lantern 'the Beacon' from now on.",
               "Let's call Kestrel 'the Hauler' from now on."]
    assert guard.fold_window_excludes(fact_set, aliases) == []
    assert guard.fold_window_excludes(fact_set, aliases[:1]) == []


def test_alias_binding_uses_a1s_parser_only():
    """No second alias detector: what A1 does not parse, F2 does not invent."""
    assert guard.alias_bindings(SOURCES) == {'beacon': 'lantern',
                                             'hauler': 'kestrel'}
    assert guard.alias_bindings(['Breakwater sits near Commtower.']) == {}
    # Scoping: an alias is accepted for its OWN base and for nothing else.
    bindings = guard.alias_bindings(SOURCES)
    assert 'beacon' in guard.accepted_names('lantern', bindings)
    assert 'beacon' not in guard.accepted_names('commtower', bindings)
    assert 'beacon' not in guard.accepted_names('breakwater', bindings)
    assert guard.accepted_names('commtower', bindings) == {'commtower'}


def test_entity_proxy_is_glyph_normalized():
    """FIX-8/SC1.1 projection: U+2011 is U+002D for the entity scan too."""
    assert guard.entity_tokens('Comm‑tower') == guard.entity_tokens(
        'Comm-tower')


# ----------------------------------------------------------------- controls

def test_control_non_alias_folds_are_byte_identical_on_vs_off(tmp_path,
                                                              monkeypatch):
    """A window with no fact-less alias turn must be unchanged by the flag.

    This is the OFF-vs-ON control the order asks for, run on the SAME
    production path with the SAME double: every observable — node table, fold
    history, consolidation receipt, the model's whole call log — must match.
    """
    plain = [SOURCES[0], SOURCES[1],
             '<|start|>user<|message|>Promenade’s lamp spacing will be '
             '9 voxels.<|end|>',
             '<|start|>user<|message|>Medibay’s bed count will be 16 '
             'beds.<|end|>']
    records = {}
    for label, flag in (('off', '0'), ('on', '1')):
        monkeypatch.setenv(guard.ENV_NAME, flag)
        repo = repository(tmp_path / label, monkeypatch)
        repo.arena.m = FaithfulDigestModel(repo.arena.m.codec)
        try:
            assert repo.fold_alias_guard is (flag == '1')
            ids = seed(repo, plain)
            jobs = repo._guard_fold_windows([('digest', list(ids))])
            assert jobs == [('digest', list(ids))], jobs
            didx, _ = repo.arena.consolidate(jobs[0][1])
            assert didx is not None
            records[label] = {
                'state': fold_state(repo, ids, didx),
                'calls': repo.arena.m.calls,
            }
        finally:
            repo.close()
    # The guard receipt is the ONE permitted difference: it is new bookkeeping
    # on the ON arm, it is EMPTY here, and it is never read by a serving path.
    for record in records.values():
        assert record['state']['guard_history'] == []
        record['state'].pop('guard_history')
        record['state']['result'].pop('attribution', None)
    assert (json.dumps(records['off'], sort_keys=True, default=str)
            == json.dumps(records['on'], sort_keys=True, default=str))


FIX5_REG = json.loads(
    (ROOT / 'artifacts/grm_scout_fix5/registration.json').read_text())


@pytest.mark.parametrize('start_path', FIX5_REG['fold_starts'],
                         ids=lambda p: Path(p).parent.name[:3])
def test_control_fix5_gpu_contrast_digests_still_pass(start_path):
    """The 13 FIX-5 GPU-contrast folds' RECORDED digests still pass QC.

    These are real GPT-OSS-20B digests from the FIX-5 GPU run, recorded under
    ``artifacts/grm_scout_fix5/gpu/<turn>/receipt.json``, paired with the
    source windows their ``start.json`` registers.  If the attribution check
    cost an accepted fold anywhere, it would show here — a false positive on
    a digest the project already accepted.  All 13 must report ZERO
    violations, and none of their windows may lose a source to exclusion.
    """
    start = json.loads((ROOT / start_path).read_text())
    turn = int(start['turn'])
    receipt = json.loads(
        (ROOT / f'artifacts/grm_scout_fix5/gpu/{turn:03d}/receipt.json'
         ).read_text())
    assert receipt['accepted'] is True
    digest_text = receipt['digest_text']
    assert digest_text
    sources = [s['text'] for s in start['sources']]
    violations = guard.attribution_violations(
        ArenaCache._fact_set, sources, digest_text)
    assert violations == [], (turn, violations)
    assert guard.fold_window_excludes(
        ArenaCache._fact_set, sources) == [], turn


def test_control_flag_contract_is_registered():
    """The named optional flag F1 registers on the LT1.1 r3 run."""
    contract = json.loads(
        (ROOT / 'artifacts/grm_f2/flag_contract.json').read_text())
    assert contract['env_name'] == guard.ENV_NAME == 'GRM_FOLD_ALIAS_GUARD'
    assert contract['default'] == 'OFF'
    assert contract['off_is_byte_identical'] is True
    for path in contract['repo_relative_pins'].values():
        assert not os.path.isabs(path), path
        assert (ROOT / path).exists(), path
    for path in contract['core_files']:
        assert not os.path.isabs(path), path
        assert (ROOT / path).exists(), path
