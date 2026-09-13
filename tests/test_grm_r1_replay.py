"""GRM-R1 replay-worker contracts: registration, RED fixtures, scorer, resume.

Prior art: GRM C7/FIX8 CPU contract tests (GRM contributors, 2026) --
TAKEN: monkeypatched ``sha`` drift injection, O_EXCL owner and orphan
fail-closed tests, and the "expected value never reaches the reader" check.
OURS: the parity-barrier RED fixture and the two-arm rule-pin assertions.
No prior art known to me for this exact composition.  Author checks, not
blind verification and never a model-quality claim.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from scripts import grm_r1_replay as run


# --------------------------------------------------------------------------
# registration + cohort
# --------------------------------------------------------------------------

@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/registration.json + orders/GRM_R1_MARGIN_FIRST_REPLAY.md')
def test_registration_cohort_budget_and_plan_wording():
    r = run.verify()
    assert r['schema'] == 'grm.r1.replay.v1'
    assert r['cell_count'] == 31 and r['distinct_question_ids'] == 9
    cells = run.cells()
    assert len(cells) == 31
    assert len({c['id'] for c in cells}) == 31
    assert len({c['question_id'] for c in cells}) == 9
    # Budget must fit the registered 1.0 GPU-h cap with room to spare.
    assert r['reserved_seconds'] == sum(r['batch_lease_seconds'].values())
    assert r['reserved_seconds'] <= r['gpu_cap_seconds'] == 3600
    assert r['gpu_cap_gpu_hours'] == 1.0
    # Prediction and verdict rule are recorded verbatim from the plan.
    assert '<= 2 of 31' in r['prediction']
    assert '0 correct->wrong on supersession' in r['prediction']
    assert 'correct->wrong = 0 on sup and <= 1 elsewhere' in r['verdict_rule']
    # The 20 unresolved FIX-6 executions are excluded, never imputed.
    assert 'NOT in this cohort' in r['unresolved_policy']
    assert 'Nothing is imputed' in r['unresolved_policy']


@pytest.mark.campaign_receipt(registration='artifacts/grm_r1/registration.json (cell checkpoints pinned under absolute /mnt/ForgeRealm/wt/grm-c2/artifacts/grm_c2/epochs/scout-fix-2/, pruned with the grm-c2 fork)')
def test_every_registered_cell_input_hash_verifies():
    for c in run.cells():
        descriptor = run.verify_cell_inputs(c)
        assert descriptor['files'] == c['checkpoint_files']
        assert len(c['checkpoint_files']) >= 1


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/registration.json + orders/GRM_R1_MARGIN_FIRST_REPLAY.md')
def test_batches_partition_the_cohort_in_order_and_by_side():
    r = run.verify()
    ids = [c['id'] for c in run.cells()]
    flat = [i for b in r['batch_ids'] for i in r['batches'][b]]
    assert flat == ids                      # order preserved, nothing dropped
    by_id = {c['id']: c for c in run.cells()}
    for b in r['batch_ids']:
        sides = {by_id[i]['side'] for i in r['batches'][b]}
        assert len(sides) == 1, f'batch {b} mixes sides: {sides}'


# --------------------------------------------------------------------------
# RED fixture: a checkpoint hash mismatch is RED, never a retry
# --------------------------------------------------------------------------

@pytest.mark.campaign_receipt(registration='artifacts/grm_r1/registration.json (cell checkpoints pinned under absolute /mnt/ForgeRealm/wt/grm-c2/artifacts/grm_c2/epochs/scout-fix-2/, pruned with the grm-c2 fork)')
def test_checkpoint_file_hash_mismatch_is_red(monkeypatch):
    c = run.cells()[0]
    name = sorted(c['checkpoint_files'])[0]
    target = Path(c['checkpoint']).parent / 'repository' / name
    original = run.sha
    monkeypatch.setattr(
        run, 'sha',
        lambda p: 'deadbeef' if Path(p) == target else original(p))
    with pytest.raises(ValueError, match='R1_CHECKPOINT_FILE_SHA_MISMATCH'):
        run.verify_cell_inputs(c)


@pytest.mark.campaign_receipt(registration='artifacts/grm_r1/registration.json (cell checkpoints pinned under absolute /mnt/ForgeRealm/wt/grm-c2/artifacts/grm_c2/epochs/scout-fix-2/, pruned with the grm-c2 fork)')
def test_checkpoint_descriptor_hash_mismatch_is_red(monkeypatch):
    c = run.cells()[0]
    target = Path(c['checkpoint'])
    original = run.sha
    monkeypatch.setattr(
        run, 'sha',
        lambda p: 'deadbeef' if Path(p) == target else original(p))
    with pytest.raises(ValueError, match='R1_CHECKPOINT_SHA_MISMATCH'):
        run.verify_cell_inputs(c)


@pytest.mark.campaign_receipt(registration='artifacts/grm_r1/registration.json (cell checkpoints pinned under absolute /mnt/ForgeRealm/wt/grm-c2/artifacts/grm_c2/epochs/scout-fix-2/, pruned with the grm-c2 fork)')
def test_policy_state_and_source_worker_drift_are_red(monkeypatch):
    c = run.cells()[0]
    original = run.sha
    monkeypatch.setattr(
        run, 'sha',
        lambda p: ('bad' if Path(p) == run.ROOT / c['policy_state']
                   else original(p)))
    with pytest.raises(ValueError, match='R1_POLICY_STATE_SHA_MISMATCH'):
        run.verify_cell_inputs(c)
    monkeypatch.setattr(
        run, 'sha',
        lambda p: ('bad' if Path(p) == Path(c['source_worker'])
                   else original(p)))
    with pytest.raises(ValueError, match='R1_SOURCE_WORKER_SHA_MISMATCH'):
        run.verify_cell_inputs(c)


def test_registered_input_drift_is_red(monkeypatch):
    original = run.sha
    target = run.ROOT / 'core/grm_admission.py'
    monkeypatch.setattr(
        run, 'sha',
        lambda p: 'bad' if Path(p) == target else original(p))
    with pytest.raises(ValueError, match='R1_INPUT_SHA_MISMATCH'):
        run.verify()


def test_a_hash_mismatch_never_becomes_a_retry():
    """The mismatch raises; nothing in the worker catches and re-runs it."""
    source = (run.ROOT / 'scripts/grm_r1_replay.py').read_text()
    assert 'R1_CHECKPOINT_FILE_SHA_MISMATCH' in source
    for banned in ('for attempt in', 'while True:', 'max_retries'):
        assert banned not in source, f'retry machinery present: {banned}'


# --------------------------------------------------------------------------
# RED fixture: the arm-OFF parity barrier stops the run
# --------------------------------------------------------------------------

@pytest.mark.campaign_receipt(registration='artifacts/grm_r1/registration.json (cell checkpoints pinned under absolute /mnt/ForgeRealm/wt/grm-c2/artifacts/grm_c2/epochs/scout-fix-2/, pruned with the grm-c2 fork)')
def test_parity_barrier_stops_the_run(tmp_path, monkeypatch):
    """A replayed OFF plan that is not the recorded plan is RED and STOPS."""
    c = copy.deepcopy(run.cells()[0])
    calls = []
    real_fake_arm = run.fake_arm

    def drifting(cell, descriptor, arm, session):
        value = real_fake_arm(cell, descriptor, arm, session)
        calls.append(arm)
        if arm == 'off':
            value['rank_plan'] = value['rank_plan'] + [9999]
        return value

    monkeypatch.setattr(run, 'fake_arm', drifting)
    with pytest.raises(ValueError, match='R1_OFF_PLAN_PARITY_RED_STOP'):
        run.run_cell(c, tmp_path, None, fake=True)
    # The RED receipt is written BEFORE the stop, so the failure is auditable.
    receipt = json.loads((tmp_path / 'cells' / (c['id'] + '.json')).read_text())
    assert receipt['off_plan_parity'] is False
    assert receipt['off_plan_bytes_hex'] != receipt['recorded_off_plan_bytes_hex']
    assert calls == ['off', 'on']        # both arms ran; only the check failed


@pytest.mark.campaign_receipt(registration='artifacts/grm_r1/registration.json (cell checkpoints pinned under absolute /mnt/ForgeRealm/wt/grm-c2/artifacts/grm_c2/epochs/scout-fix-2/, pruned with the grm-c2 fork)')
def test_parity_barrier_passes_on_the_recorded_plan(tmp_path):
    c = run.cells()[0]
    value, _ = run.run_cell(copy.deepcopy(c), tmp_path, None, fake=True)
    assert value['off_plan_parity'] is True
    assert value['arms']['off']['rank_plan'] == c['off_plan']
    assert value['arms']['on']['rank_plan'] == c['on_plan']
    assert value['on_plan_matches_registered'] is True


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/registration.json + orders/GRM_R1_MARGIN_FIRST_REPLAY.md')
def test_batch_records_the_red_and_fails_closed(tmp_path, monkeypatch):
    r = run.verify()
    batch_id = r['batch_ids'][0]
    real_fake_arm = run.fake_arm

    def drifting(cell, descriptor, arm, session):
        value = real_fake_arm(cell, descriptor, arm, session)
        if arm == 'off':
            value['rank_plan'] = []
        return value

    monkeypatch.setattr(run, 'fake_arm', drifting)
    with pytest.raises(RuntimeError, match='R1_OFF_PLAN_PARITY_RED_STOP'):
        run.batch(batch_id, fake=True, root=tmp_path)
    controller = json.loads(
        (tmp_path / 'gpu' / batch_id / 'controller.json').read_text())
    assert controller['status'] == 'FAILED'
    assert controller['red']
    assert 'R1_OFF_PLAN_PARITY_RED_STOP' in controller['red'][0]
    # A failed batch is charged its full pessimistic reservation.
    assert controller['charged_seconds'] >= controller['reservation_seconds']
    # ... and the campaign refuses to continue over it.
    with pytest.raises(ValueError, match='R1_FAILED_CAMPAIGN_STOP'):
        run.campaign_state(r, tmp_path)


# --------------------------------------------------------------------------
# the two arms differ ONLY by the pinned rule, over ONE shared state
# --------------------------------------------------------------------------

def test_rule_pin_per_arm_and_readback(monkeypatch):
    from core.grm_admission import admission_rule
    monkeypatch.setenv(run.RULE_ENV, 'some_ambient_leftover')
    assert run.pin_rule('off') == 'all_tokens_bind'
    # GRM-D2 (2026-09-11) CHANGED THIS ASSERTION, and the change is real
    # rather than a re-pin.  The OFF arm used to express itself by leaving
    # the variable ABSENT -- a correct pin only while absent meant
    # `all_tokens_bind`.  D2 made `margin_first` the shipped rule, so an
    # absent variable would now give the OFF arm the ON rule; `pin_rule`
    # therefore pins `all_tokens_bind` EXPLICITLY and its own read-back is
    # what caught the regression.  The property tested is unchanged: the
    # arm overrides an ambient leftover and the resolver agrees.
    assert os.environ[run.RULE_ENV] == 'all_tokens_bind'
    assert admission_rule() == 'all_tokens_bind'
    assert run.pin_rule('on') == 'margin_first'
    assert os.environ[run.RULE_ENV] == 'margin_first'
    assert admission_rule() == 'margin_first'
    with pytest.raises(ValueError, match='R1_UNKNOWN_ARM'):
        run.pin_rule('sideways')


def test_rule_is_pinned_after_environment_strips_grm_vars(monkeypatch):
    """FIX-6's state contract: environment(flags) wipes GRM_*, so pin after."""
    from scripts.grm_c2_cells import environment, flags_for
    monkeypatch.setenv(run.RULE_ENV, 'margin_first')
    env = environment(flags_for('profile'))
    assert run.RULE_ENV not in env, 'environment() must strip the rule'
    for key in list(os.environ):
        if key.startswith('GRM_'):
            monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        if key.startswith('GRM_'):
            monkeypatch.setenv(key, value)
    assert run.RULE_ENV not in os.environ
    assert run.pin_rule('on') == 'margin_first'   # pinned AFTER, so it sticks
    monkeypatch.delenv(run.RULE_ENV, raising=False)


@pytest.mark.campaign_receipt(registration='artifacts/grm_r1/registration.json (cell checkpoints pinned under absolute /mnt/ForgeRealm/wt/grm-c2/artifacts/grm_c2/epochs/scout-fix-2/, pruned with the grm-c2 fork)')
def test_both_arms_share_one_verified_state(tmp_path):
    c = run.cells()[0]
    value, _ = run.run_cell(copy.deepcopy(c), tmp_path, None, fake=True)
    proof = value['same_state_proof']
    assert proof['checkpoint_sha256'] == c['checkpoint_sha256']
    assert proof['checkpoint_files_verified'] == len(c['checkpoint_files'])
    assert proof['arms_share_one_checkpoint'] and proof['per_arm_private_copy']
    assert proof['differing_environment_keys'] == [run.RULE_ENV]
    off, on = value['arms']['off'], value['arms']['on']
    assert off['checkpoint_sha256'] == on['checkpoint_sha256']
    assert off['question'] == on['question']
    assert off['expected_values'] == on['expected_values']
    assert off['admission_rule'] == 'all_tokens_bind'
    assert on['admission_rule'] == 'margin_first'


# --------------------------------------------------------------------------
# scorer fixtures: exact / value-span / abstain / wrong
# --------------------------------------------------------------------------

def test_scorer_exact_match():
    v = run.score_answer('Kestrel-9-Tango', ['Kestrel-9-Tango'], [])
    assert v['correct'] and v['expected_hits'] == ['Kestrel-9-Tango']
    assert not v['abstained'] and not v['rejected_hits']


def test_scorer_value_span_inside_a_sentence():
    """The LT1/C5 rule is a value SPAN, not full-string equality."""
    v = run.score_answer('The current orion pin value is Kestrel-9-Tango.',
                         ['Kestrel-9-Tango'], [])
    assert v['correct'] and v['expected_hits'] == ['Kestrel-9-Tango']


def test_scorer_value_span_normalizes_emphasis_and_unicode_hyphens():
    v = run.score_answer('It is **Kestrel‑9‑Tango**.',
                         ['Kestrel-9-Tango'], [])
    assert v['correct']


def test_scorer_value_span_respects_word_boundaries():
    """A longer token that merely contains the value is NOT a hit."""
    v = run.score_answer('XKestrel-9-TangoX', ['Kestrel-9-Tango'], [])
    assert not v['correct'] and not v['expected_hits']


def test_scorer_abstain_is_wrong_but_flagged_separately():
    v = run.score_answer("I don't know.", ['Kestrel-9-Tango'], [])
    assert not v['correct']
    assert v['abstained'] is True


def test_scorer_wrong_value():
    v = run.score_answer('Auric-4-Alpha', ['Kestrel-9-Tango'], [])
    assert not v['correct'] and not v['abstained']
    assert v['expected_hits'] == []


def test_scorer_rejected_value_defeats_a_hit():
    """Supersession guard: naming the stale value alongside is NOT correct."""
    v = run.score_answer('It was Auric-4-Alpha, now Kestrel-9-Tango.',
                         ['Kestrel-9-Tango'], ['Auric-4-Alpha'])
    assert v['expected_hits'] == ['Kestrel-9-Tango']
    assert v['rejected_hits'] == ['Auric-4-Alpha']
    assert not v['correct']


@pytest.mark.campaign_receipt(registration='artifacts/grm_r1/registration.json (cell checkpoints pinned under absolute /mnt/ForgeRealm/wt/grm-c2/artifacts/grm_c2/epochs/scout-fix-2/, pruned with the grm-c2 fork)')
def test_expected_values_come_from_the_recorded_checkpoint_only():
    for c in run.cells():
        descriptor = run.read(Path(c['checkpoint']))
        expected, rejected = run.expected_values(descriptor['context'],
                                                 c['battery'])
        assert expected and all(isinstance(v, str) and v for v in expected)
        if c['battery'] == 'sup':
            assert rejected, 'sup cells must carry their competitor guards'
        else:
            assert rejected == []


# --------------------------------------------------------------------------
# transition classification
# --------------------------------------------------------------------------

@pytest.mark.parametrize('off_ok,on_ok,expected', [
    (True, True, 'unchanged_correct'),
    (False, False, 'unchanged_wrong'),
    (True, False, 'correct_to_wrong'),
    (False, True, 'wrong_to_correct'),
])
def test_classify_primary_transitions(off_ok, on_ok, expected):
    off = {'correct': off_ok, 'abstained': False, 'answer': 'a'}
    on = {'correct': on_ok, 'abstained': False, 'answer': 'b'}
    value = run.classify(off, on)
    assert value['transition'] == expected
    assert value['answer_changed'] is True


def test_classify_abstention_is_reported_alongside_not_folded_in():
    off = {'correct': True, 'abstained': False, 'answer': 'Kestrel-9-Tango'}
    on = {'correct': False, 'abstained': True, 'answer': "I don't know."}
    value = run.classify(off, on)
    assert value['transition'] == 'correct_to_wrong'
    assert value['abstain_transition'] == 'abstain_gained'
    off2 = {'correct': False, 'abstained': True, 'answer': "I don't know."}
    on2 = {'correct': True, 'abstained': False, 'answer': 'Kestrel-9-Tango'}
    value2 = run.classify(off2, on2)
    assert value2['transition'] == 'wrong_to_correct'
    assert value2['abstain_transition'] == 'abstain_lost'


def test_classify_identical_answers_are_unchanged():
    off = {'correct': True, 'abstained': False, 'answer': 'Kestrel-9-Tango'}
    value = run.classify(off, dict(off))
    assert value['transition'] == 'unchanged_correct'
    assert value['answer_changed'] is False
    assert value['abstain_transition'] is None


# --------------------------------------------------------------------------
# dry run
# --------------------------------------------------------------------------

@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/registration.json + orders/GRM_R1_MARGIN_FIRST_REPLAY.md')
def test_dry_run_enumerates_every_cell_with_an_estimate_and_no_gpu():
    value = run.dry_run()
    assert value['status'] == 'READY_FOR_LEAD'
    assert value['gpu_executed'] is False
    assert value['measurement_status'] == 'NOT_RUN'
    assert len(value['cells']) == value['cell_count'] == 31
    for row in value['cells']:
        assert row['estimate_seconds'] > 0
        assert row['off_plan'] != row['on_plan'], 'every cell is a plan change'
        assert row['checkpoint_files'] >= 1
    assert value['reserved_gpu_hours'] <= 1.0
    assert value['estimated_seconds'] <= value['reserved_seconds']


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/registration.json + orders/GRM_R1_MARGIN_FIRST_REPLAY.md')
def test_dry_run_never_imports_a_gpu_loader():
    import sys
    before = set(sys.modules)
    run.dry_run()
    new = set(sys.modules) - before
    assert not [m for m in new if 'cuda' in m.lower()]


# --------------------------------------------------------------------------
# summary + verdict rule
# --------------------------------------------------------------------------

@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/registration.json + orders/GRM_R1_MARGIN_FIRST_REPLAY.md')
def test_summary_is_not_measured_before_any_run(tmp_path):
    value = run.summary(tmp_path)
    assert value['status'] == 'NOT_MEASURED'
    assert value['adopt'] is False and value['prediction_held'] is None


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/registration.json + orders/GRM_R1_MARGIN_FIRST_REPLAY.md')
def test_summary_refuses_to_adopt_from_a_fake_run(tmp_path):
    r = run.verify()
    for batch_id in r['batch_ids']:
        run.batch(batch_id, fake=True, root=tmp_path)
    value = run.summary(tmp_path)
    assert value['complete'] is True and value['cells_measured'] == 31
    assert value['off_plan_parity'] is True
    # A fake run measures no answers, so it can never produce a verdict.
    assert value['answers_measured'] is False
    assert value['status'] == 'NOT_MEASURED'
    assert value['adopt'] is False
    assert value['fake_run'] is True
    assert 'answers are NOT measured' in value['scope']


def _measure(tmp_path, r):
    """Mark every fake receipt as measured, so the verdict rule can be tested."""
    for batch_id in r['batch_ids']:
        for p in sorted((tmp_path / 'gpu' / batch_id / 'cells').glob('*.json')):
            row = json.loads(p.read_text())
            for arm in row['arms'].values():
                arm['answer_measured'] = True
            p.write_text(json.dumps(row, indent=2, sort_keys=True) + '\n')


def _promote(tmp_path, r, cell_id, transition):
    for batch_id in r['batch_ids']:
        p = tmp_path / 'gpu' / batch_id / 'cells' / (cell_id + '.json')
        if p.exists():
            row = json.loads(p.read_text())
            row['transition'] = transition
            row['answer_changed'] = transition != 'unchanged_correct'
            p.write_text(json.dumps(row, indent=2, sort_keys=True) + '\n')
            return
    raise AssertionError('cell not found: ' + cell_id)


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/registration.json + orders/GRM_R1_MARGIN_FIRST_REPLAY.md')
def test_verdict_rule_is_applied_verbatim(tmp_path):
    """correct->wrong: 0 on sup and <=1 elsewhere; measured answers required."""
    r = run.verify()
    for batch_id in r['batch_ids']:
        run.batch(batch_id, fake=True, root=tmp_path)
    _measure(tmp_path, r)
    value = run.summary(tmp_path)
    assert value['answers_measured'] is True
    assert value['status'] == 'ADOPT' and value['adopt'] is True
    assert value['prediction_held'] is True

    # One correct->wrong outside sup is still adoptable (<= 1 elsewhere).
    census = next(c['id'] for c in run.cells() if c['battery'] != 'sup')
    _promote(tmp_path, r, census, 'correct_to_wrong')
    value = run.summary(tmp_path)
    assert value['correct_to_wrong_other'] == 1 and value['adopt'] is True

    # A second one is not.
    census2 = next(c['id'] for c in run.cells()
                   if c['battery'] != 'sup' and c['id'] != census)
    _promote(tmp_path, r, census2, 'correct_to_wrong')
    value = run.summary(tmp_path)
    assert value['correct_to_wrong_other'] == 2
    assert value['adopt'] is False and value['status'] == 'DO_NOT_ADOPT'

    # ANY correct->wrong on sup is disqualifying, even with none elsewhere.
    _promote(tmp_path, r, census, 'unchanged_correct')
    _promote(tmp_path, r, census2, 'unchanged_correct')
    sup = next(c['id'] for c in run.cells() if c['battery'] == 'sup')
    _promote(tmp_path, r, sup, 'correct_to_wrong')
    value = run.summary(tmp_path)
    assert value['correct_to_wrong_sup'] == 1
    assert value['correct_to_wrong_other'] == 0
    assert value['adopt'] is False and value['status'] == 'DO_NOT_ADOPT'


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/registration.json + orders/GRM_R1_MARGIN_FIRST_REPLAY.md')
def test_prediction_is_recorded_but_is_not_the_adoption_gate(tmp_path):
    r = run.verify()
    for batch_id in r['batch_ids']:
        run.batch(batch_id, fake=True, root=tmp_path)
    _measure(tmp_path, r)
    changed = 0
    for batch_id in r['batch_ids']:
        for p in sorted((tmp_path / 'gpu' / batch_id / 'cells').glob('*.json')):
            if changed >= 3:
                break
            row = json.loads(p.read_text())
            row['answer_changed'] = True
            changed += 1
            p.write_text(json.dumps(row, indent=2, sort_keys=True) + '\n')
    value = run.summary(tmp_path)
    assert value['answers_changed'] == 3
    assert value['prediction_held'] is False
    assert value['adopt'] is True and value['status'] == 'ADOPT'


# --------------------------------------------------------------------------
# resume after interrupt
# --------------------------------------------------------------------------

@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/registration.json + orders/GRM_R1_MARGIN_FIRST_REPLAY.md')
def test_resume_after_interrupt_skips_complete_and_continues(tmp_path, capsys):
    r = run.verify()
    first, second = r['batch_ids'][0], r['batch_ids'][1]
    run.batch(first, fake=True, root=tmp_path)
    charged, complete = run.campaign_state(r, tmp_path)
    assert complete == [first] and charged > 0

    # Re-running a COMPLETE batch is a no-op, not a second execution.
    before = (tmp_path / 'gpu' / first / 'controller.json').read_bytes()
    capsys.readouterr()
    run.batch(first, fake=True, root=tmp_path)
    assert (tmp_path / 'gpu' / first / 'controller.json').read_bytes() == before
    assert json.loads(capsys.readouterr().out)['status'] == 'ALREADY_COMPLETE'

    # The campaign resumes at the next batch and its charge accumulates.
    run.batch(second, fake=True, root=tmp_path)
    charged2, complete2 = run.campaign_state(r, tmp_path)
    assert complete2 == [first, second] and charged2 > charged


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/registration.json + orders/GRM_R1_MARGIN_FIRST_REPLAY.md')
def test_resume_refuses_to_skip_an_unrun_prior_batch(tmp_path):
    r = run.verify()
    # Raised by the pre-flight guard, BEFORE any reservation is written,
    # so it is not wrapped into the controller's RuntimeError.
    with pytest.raises(ValueError, match='R1_PRIOR_BATCH_INCOMPLETE'):
        run.batch(r['batch_ids'][1], fake=True, root=tmp_path)
    assert not (tmp_path / 'gpu' / r['batch_ids'][1]).exists()
    assert not (tmp_path / 'replay.active').exists()   # owner file released


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/registration.json + orders/GRM_R1_MARGIN_FIRST_REPLAY.md')
def test_interrupted_batch_leaves_an_orphan_reservation_that_fails_closed(tmp_path):
    r = run.verify()
    batch_id = r['batch_ids'][0]
    # Simulate a kill between the reservation write and the controller write.
    run.write(tmp_path / 'gpu' / batch_id / 'reservation.json',
              {'seconds': r['batch_lease_seconds'][batch_id],
               'batch': batch_id, 'registration_sha256': run.sha(run.REG)})
    with pytest.raises(ValueError, match='R1_ORPHAN_RESERVATION_STOP'):
        run.campaign_state(r, tmp_path)
    # ... and the worker will not silently re-run over it either.
    with pytest.raises(ValueError, match='R1_ORPHAN_RESERVATION_STOP'):
        run.batch(r['batch_ids'][1], fake=True, root=tmp_path)
    assert not (tmp_path / 'replay.active').exists()   # owner file released


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/registration.json + orders/GRM_R1_MARGIN_FIRST_REPLAY.md')
def test_single_owner_file_is_never_stolen(tmp_path):
    r = run.verify()
    owner = tmp_path / 'replay.active'
    owner.parent.mkdir(parents=True, exist_ok=True)
    owner.write_text('another owner')
    with pytest.raises(FileExistsError):
        run.batch(r['batch_ids'][0], fake=True, root=tmp_path)
    assert owner.read_text() == 'another owner'   # untouched


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/registration.json + orders/GRM_R1_MARGIN_FIRST_REPLAY.md')
def test_unknown_batch_and_unknown_cell_are_rejected(tmp_path):
    with pytest.raises(ValueError, match='R1_UNKNOWN_BATCH'):
        run.batch('NOPE', fake=True, root=tmp_path)
    with pytest.raises(ValueError, match='R1_UNKNOWN_CELL'):
        run.cell_by_id('not-a-cell')


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/registration.json + orders/GRM_R1_MARGIN_FIRST_REPLAY.md')
def test_receipts_are_create_only(tmp_path):
    r = run.verify()
    batch_id = r['batch_ids'][0]
    run.batch(batch_id, fake=True, root=tmp_path)
    cell_id = r['batches'][batch_id][0]
    p = tmp_path / 'gpu' / batch_id / 'cells' / (cell_id + '.json')
    assert p.exists()
    with pytest.raises(FileExistsError):
        run.write(p, {'overwrite': 'attempt'})
