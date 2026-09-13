"""FIX-3 CPU treatment pins. Prior art: local GRM C7 diagnostic numerical
doubles and EB1 Harmony contract (GRM contributors, 2026). Borrow the real
repository/arena paths; new stimuli cover stops and punctuation collapse.
No prior art known to me for this exact QC rule. No model-quality claim.
"""
import pytest

from core.graft_arena import ArenaCache


# --------------------------------------------------------- GRM-D2 re-pin
#
# GRM-D2 (2026-09-11) flipped the shipped admission rule to `margin_first`
# and turned F1 / F2 / F5 / A1 ON.  F1 in particular changes the fold lifecycle this suite measures: with retention ON a fold no longer retires its sources, so a coverage-bar assertion written against the substituting fold reads a different node table.
#
# THIS SUITE'S ASSERTIONS ARE UNCHANGED -- they are the receipt for the
# pre-D2 behaviour this suite was written to pin, and `GRM_LEGACY_DEFAULTS=1`
# restores exactly that world.  The flipped behaviours have their own suites
# (tests/test_grm_f1_fold_retain.py, test_grm_f2_alias_guard.py,
# test_grm_f5_sole_binder_insurance.py, test_grm_a1_alias_fold.py).
@pytest.fixture(autouse=True)
def _grm_d2_legacy_defaults(monkeypatch):
    from core import grm_legacy_defaults as legacy_defaults
    monkeypatch.setenv(legacy_defaults.ENV_NAME, "1")

from scripts.grm_c7_diagnose import repository
from scripts.grm_e2e_session import harmony_turn, HARMONY_STOPS

SOURCE = 'The current C7-Fresh-0 value is Basalt-811.'


@pytest.mark.parametrize('kind', ['turn', 'digest', 'era'])
@pytest.mark.parametrize('stop', HARMONY_STOPS)
def test_fold_harmony_wrapper_and_stop(tmp_path, monkeypatch, kind, stop):
    repo = repository(tmp_path / 'repo', monkeypatch)
    try:
        a = repo.arena
        idx = repo.add_document(SOURCE)
        a.grafts[idx]['kind'] = kind
        a.m.fold_output = SOURCE + stop + ' LEAK'
        assert repo._fold_once(jobs=[('digest', [idx])])
        first = a.m.calls[0]['input']
        raw = a._consolidation_prompts(kind != 'turn', [a.grafts[idx]['text']])[0]
        user, primer = raw.removeprefix('User: ').rsplit('\nAssistant:', 1)
        assert first == harmony_turn(user, None) + primer
        assert len(a.m.calls) == len(a.encode(SOURCE + stop))
        text = a.last_consolidation_attempts[0]['text']
        assert stop not in text and 'LEAK' not in text
        assert a.last_consolidation_result['accepted'] is True
    finally:
        repo.close()


@pytest.mark.parametrize('template', [None, 'Question: {user}\nAnswer:'])
def test_fold_explicit_budget_and_legacy_prompt(tmp_path, monkeypatch, template):
    repo = repository(tmp_path / 'repo', monkeypatch)
    try:
        a = repo.arena
        idx = repo.add_document(SOURCE)
        a.grafts[idx]['kind'] = 'turn'
        a.prompt_template = template
        a.m.fold_output = SOURCE + '<|end|> more unique prose words here'
        a.consolidate([idx], ngen=19)
        assert a.m.calls[0]['input'] == a._consolidation_prompts(False, [a.grafts[idx]['text']])[0]
        assert len(a.m.calls) == 19
        assert '<|end|>' in a.last_consolidation_attempts[0]['text']
        a.m.calls.clear()
        a.prompt_template = harmony_turn
        a.m.fold_output = 'ordinary words without facts'
        a.consolidate([idx], ngen=4)
        assert len(a.m.calls) == 3 * 4
        assert a.CONSOLIDATE_NGEN == 120 and a.MIN_FOLD_KEEP == .70
    finally:
        repo.close()


@pytest.mark.parametrize('run', ['………', '......', '!!!!!!', '?!' * 4, '. ' * 8])
def test_digest_qc_rejects_punctuation_collapse(run):
    assert not ArenaCache._digest_qc(SOURCE + ' ' + run)


@pytest.mark.parametrize('text', [SOURCE, 'Wait... the current C7-Fresh-0 value is Basalt-811.',
    'The current C7-Fresh-0 value is Basalt-811; source [A] confirms it!'])
def test_digest_qc_accepts_ordinary_punctuation(text):
    assert ArenaCache._digest_qc(text)


def test_degenerate_qc_precedes_coverage_even_with_list_relaxation(tmp_path, monkeypatch):
    repo = repository(tmp_path / 'repo', monkeypatch)
    try:
        a = repo.arena
        idx = repo.add_document(SOURCE)
        a.m.fold_output = SOURCE + ' ……………<|end|>'
        a.ALLOW_HIGH_COVERAGE_LIST_DIGESTS = True
        def no_coverage(*args):
            pytest.fail('degenerate digest reached coverage')
        monkeypatch.setattr(a, '_coverage', no_coverage)
        assert a.consolidate([idx]) == (None, None)
        assert len(a.last_consolidation_attempts) == 3
        assert all(not r['qc'] for r in a.last_consolidation_attempts)
    finally:
        repo.close()


@pytest.mark.parametrize('kept,accepted', [(6, False), (7, True)])
def test_fold_coverage_bar_remains_point_seven(tmp_path, monkeypatch, kept, accepted):
    # Prior art: existing GRM MIN_FOLD_KEEP fidelity law (2026). Exercise
    # both sides with real fact extraction/coverage, no mocked score.
    repo = repository(tmp_path / 'repo', monkeypatch)
    try:
        a = repo.arena
        source = 'The recorded values are ' + ' '.join(f'X-{i}' for i in range(10)) + '.'
        idx = repo.add_document(source)
        a.grafts[idx]['kind'] = 'turn'
        a.m.fold_output = 'The recorded values are ' + ' '.join(f'X-{i}' for i in range(kept)) + '.<|end|>'
        result = a.consolidate([idx])
        assert (result[0] is not None) is accepted
        assert a.last_consolidation_result['best_cov'] == kept / 10
        assert a.last_consolidation_result['accepted'] is accepted
        assert bool(a.grafts[idx].get('retired')) is accepted
    finally:
        repo.close()
