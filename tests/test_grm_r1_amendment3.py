"""GRM-R1 amendment 3: a satisfied re-arm scope stops gating the campaign.

The defect: `resume_scope()` re-applied amendment 1's retain/reissue
cross-check on every invocation, so once the re-armed batch COMPLETED its
own success made the numbers disagree and the next batch was refused
(`R1_AMENDMENT_RETAIN_MISMATCH` raised from `--batch R2`).

`test_completed_rearm_no_longer_blocks_the_next_batch` is the RED-before /
GREEN-after fixture: it drives the whole sequence on the fake path --
re-arm, run to completion, then run the NEXT batch -- and asserts the next
batch runs.  Under the pre-amendment-3 worker its final step raised.

Prior art: GRM C7/C2 amendment-chain tests (GRM contributors, 2026) --
TAKEN: chain drift injection and scope fixtures.  OURS: the satisfied-scope
assertions.  No prior art known to me for this exact composition.  No
model-quality claim.
"""
from __future__ import annotations

import json

import pytest

from scripts import grm_r1_replay as run


def _chain_into(tmp_path, rearm=None, links=(1, 2, 3)):
    """Copy the real amendment chain into ``tmp_path``, optionally re-scoped."""
    for index in links:
        a = json.loads((run.OUT / f'amendment_{index}.json').read_text())
        if rearm is not None:
            a['rearm'] = rearm
        if index > 1:
            a['previous_amendment_sha256'] = run.sha(
                tmp_path / f'amendment_{index - 1}.json')
        run.write(tmp_path / f'amendment_{index}.json', a)
        (tmp_path / f'amendment_{index}.sha256').write_text(
            run.sha(tmp_path / f'amendment_{index}.json')
            + f'  amendment_{index}.json\n')


def _failed_controller(ids):
    return {'status': 'FAILED',
            'error': "RuntimeError('cudaMalloc failed: out of memory')",
            'red': [], 'elapsed_seconds': 138.5, 'charged_seconds': 263.0,
            'reservation_seconds': 263, 'fake': True, 'attempt': 1,
            'amendment': None, 'cells_run': ids, 'cells_retained': [],
            'registration_sha256': run.sha(run.REG)}


def _rearm(batch_id, keep, drop):
    return {batch_id: {'retain': keep, 'reissue': drop, 'lease_seconds': 200,
                       'prior_status': 'FAILED', 'prior_error': 'oom',
                       'prior_charged_seconds': 263.0,
                       'measured_per_cell_seconds': 18.0,
                       'estimate_basis': 'test fixture'}}


def _rearmed_and_completed(tmp_path):
    """Drive a batch to: failed -> re-armed -> re-issued -> COMPLETE."""
    r = run.verify(tmp_path)
    first = r['batch_ids'][0]
    ids = r['batches'][first]
    run.batch(first, fake=True, root=tmp_path)
    d = tmp_path / 'gpu' / first
    keep, drop = ids[:5], ids[5:]
    for cell_id in drop:
        (d / 'cells' / (cell_id + '.json')).unlink()
    (d / 'controller.json').unlink()
    run.write(d / 'controller.json', _failed_controller(ids))
    _chain_into(tmp_path, rearm=_rearm(first, keep, drop))
    run.batch(first, fake=True, root=tmp_path)          # the re-issue
    assert json.loads(
        (d / 'controller.json').read_text())['status'] == 'COMPLETE'
    return r, first, ids


# --------------------------------------------------------------------------
# THE FIXTURE: RED before amendment 3, GREEN after
# --------------------------------------------------------------------------

def test_completed_rearm_no_longer_blocks_the_next_batch(tmp_path):
    """The lead's exact sequence: R1 re-armed and completed, then R2 runs.

    Before amendment 3 the final ``run.batch(second, ...)`` raised
    ``R1_AMENDMENT_RETAIN_MISMATCH``, because the completed batch now had 8
    receipts where the amendment recorded 5 retained.
    """
    r, first, ids = _rearmed_and_completed(tmp_path)
    second = r['batch_ids'][1]

    # The next batch runs instead of being refused by the satisfied scope.
    run.batch(second, fake=True, root=tmp_path)

    controller = json.loads(
        (tmp_path / 'gpu' / second / 'controller.json').read_text())
    assert controller['status'] == 'COMPLETE'
    # It ran as a NORMAL first run, not on the amendment's re-issue path.
    assert controller['amendment'] is None
    assert controller['attempt'] == 1
    assert controller['cells_run'] == r['batches'][second]
    assert controller['cells_retained'] == []
    # And the completed re-arm is recorded as complete, not re-run.
    _, complete = run.campaign_state(run.verify(tmp_path), tmp_path)
    assert first in complete and second in complete


def test_satisfied_scope_records_receipts_and_empties_reissue(tmp_path):
    r, first, ids = _rearmed_and_completed(tmp_path)
    scope = run.resume_scope(run.verify(tmp_path), tmp_path)[first]
    assert scope['scope_satisfied'] is True
    assert scope['reissue'] == []
    assert scope['retain'] == ids
    # Every cell receipt is recorded by hash, as the order requires.
    assert set(scope['receipts']) == set(ids)
    d = tmp_path / 'gpu' / first / 'cells'
    for cell_id, digest in scope['receipts'].items():
        assert digest == run.sha(d / (cell_id + '.json'))


def test_satisfied_scope_is_a_record_not_a_reissue_instruction(tmp_path):
    """A satisfied scope must not push a later batch onto the re-issue path."""
    r, first, _ = _rearmed_and_completed(tmp_path)
    second = r['batch_ids'][1]
    run.batch(second, fake=True, root=tmp_path)
    d = tmp_path / 'gpu' / second
    # No archived controller: the re-issue path was never taken here.
    assert not list(d.glob('controller_attempt_*.json'))
    assert not list(d.glob('reservation_attempt_*.json'))


def test_a_completed_rearm_is_not_rerun(tmp_path):
    r, first, _ = _rearmed_and_completed(tmp_path)
    d = tmp_path / 'gpu' / first
    before = {p.name: p.read_bytes() for p in (d / 'cells').glob('*.json')}
    run.batch(first, fake=True, root=tmp_path)      # ALREADY_COMPLETE
    after = {p.name: p.read_bytes() for p in (d / 'cells').glob('*.json')}
    assert after == before


# --------------------------------------------------------------------------
# what is deliberately NOT relaxed
# --------------------------------------------------------------------------

def test_open_batch_still_stops_on_a_genuine_retain_disagreement(tmp_path):
    """While the batch is OPEN the cross-check is unchanged."""
    r = run.verify(tmp_path)
    first = r['batch_ids'][0]
    ids = r['batches'][first]
    run.batch(first, fake=True, root=tmp_path)
    d = tmp_path / 'gpu' / first
    keep, drop = ids[:5], ids[5:]
    for cell_id in drop:
        (d / 'cells' / (cell_id + '.json')).unlink()
    (d / 'controller.json').unlink()
    run.write(d / 'controller.json', _failed_controller(ids))
    # Amendment claims a retain list that disagrees with disk, batch OPEN.
    _chain_into(tmp_path, rearm=_rearm(first, keep[:3], drop))
    with pytest.raises(ValueError, match='R1_AMENDMENT_RETAIN_MISMATCH'):
        run.resume_scope(run.verify(tmp_path), tmp_path)


def test_open_batch_still_stops_on_a_genuine_reissue_disagreement(tmp_path):
    r = run.verify(tmp_path)
    first = r['batch_ids'][0]
    ids = r['batches'][first]
    run.batch(first, fake=True, root=tmp_path)
    d = tmp_path / 'gpu' / first
    keep, drop = ids[:5], ids[5:]
    for cell_id in drop:
        (d / 'cells' / (cell_id + '.json')).unlink()
    (d / 'controller.json').unlink()
    run.write(d / 'controller.json', _failed_controller(ids))
    _chain_into(tmp_path, rearm=_rearm(first, keep, drop[:1]))
    with pytest.raises(ValueError, match='R1_AMENDMENT_REISSUE_MISMATCH'):
        run.resume_scope(run.verify(tmp_path), tmp_path)


def test_complete_with_missing_cells_is_a_new_red(tmp_path):
    """'COMPLETE' only satisfies the amendment if the cohort really is done."""
    r, first, ids = _rearmed_and_completed(tmp_path)
    # Remove a receipt while the controller still says COMPLETE.
    (tmp_path / 'gpu' / first / 'cells' / (ids[-1] + '.json')).unlink()
    with pytest.raises(ValueError, match='R1_AMENDMENT_SCOPE_UNSATISFIED'):
        run.resume_scope(run.verify(tmp_path), tmp_path)


def test_retained_receipt_binding_is_still_checked(tmp_path):
    """Satisfaction never waives the per-receipt binding/parity checks."""
    r, first, ids = _rearmed_and_completed(tmp_path)
    p = tmp_path / 'gpu' / first / 'cells' / (ids[0] + '.json')
    row = json.loads(p.read_text())
    row['off_plan_parity'] = False
    p.unlink()
    run.write(p, row)
    with pytest.raises(ValueError,
                       match='R1_RETAINED_RECEIPT_NOT_PARITY_CLEAN'):
        run.resume_scope(run.verify(tmp_path), tmp_path)


# --------------------------------------------------------------------------
# the amendment itself
# --------------------------------------------------------------------------

def test_amendment_3_chains_to_amendment_2():
    a = run.amendment()
    assert a['amendment'] == 3
    assert a['registration_sha256'] == run.sha(run.REG)
    assert a['previous_amendment_sha256'] == run.sha(
        run.OUT / 'amendment_2.json')
    assert len(a['_chain']) == 3
    assert a['acceptance_unchanged'] is True
    r = run.verify()
    assert a['prediction'] == r['prediction']
    assert a['verdict_rule'] == r['verdict_rule']


def test_amendment_3_records_the_completed_rearm():
    a = run.amendment()
    status = a['rearm_status']['R1']
    assert status['satisfied'] is True
    assert status['controller_status'] == 'COMPLETE'
    assert status['cells'] == 8
    assert len(status['receipts_sha256']) == 8
    d = run.OUT / 'gpu/R1/cells'
    for cell_id, digest in status['receipts_sha256'].items():
        assert digest == run.sha(d / (cell_id + '.json'))
    # The re-arm entry is kept verbatim for the record.
    assert a['rearm_retained_for_the_record'] is True
    assert len(a['rearm']['R1']['retain']) == 5


def test_real_campaign_is_open_with_r1_complete():
    """The live receipts: R1 satisfied, campaign no longer self-closed."""
    r = run.verify()
    scope = run.resume_scope(r, run.OUT)['R1']
    assert scope['scope_satisfied'] is True and scope['reissue'] == []
    charged, complete = run.campaign_state(r, run.OUT)
    assert complete == ['R1']
    assert charged > 0
    # R2 is a normal first run: the amendment does not scope it.
    assert run.resume_scope(r, run.OUT).get('R2') is None
