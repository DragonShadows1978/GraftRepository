"""GRM-R1 amendment 1: device release, re-arm scope, retained receipts.

Prior art: GRM C7/C2 sha-bound amendment tests and FIX8 resume tests (GRM
contributors, 2026) -- TAKEN: chain/scope drift injection and the
"completed work is never re-run" contract.  OURS: the arena-payload release
assertions and the retain/reissue mismatch fixtures.  No prior art known to
me for this exact composition.  No model-quality claim.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import grm_r1_replay as run


def _chain_into(tmp_path, mutate=None):
    """Copy the REAL amendment chain into tmp_path, optionally mutating link 1.

    Amendment 2 chains to amendment 1, so a fixture that plants only link 1
    would break the chain.  ``mutate`` receives each link's dict.
    """
    for index in (1, 2, 3, 4):
        a = json.loads((run.OUT / f'amendment_{index}.json').read_text())
        if mutate is not None:
            mutate(a, index)
        if index > 1:
            a['previous_amendment_sha256'] = run.sha(
                tmp_path / f'amendment_{index - 1}.json')
        run.write(tmp_path / f'amendment_{index}.json', a)
        (tmp_path / f'amendment_{index}.sha256').write_text(
            run.sha(tmp_path / f'amendment_{index}.json')
            + f'  amendment_{index}.json\n')

# --------------------------------------------------------------------------
# the fix: device payloads are released between arms
# --------------------------------------------------------------------------

def test_release_arm_drops_every_device_payload():
    """The growth cause: grafts[i]['h'] are device-resident and survive close."""
    repo = run._FakeRepo(payloads=26)
    assert sum(g['h'] is not None for g in repo.arena.grafts) == 26
    released = run.release_arm(repo)
    assert released['payloads_freed'] == 26
    assert sum(g['h'] is not None for g in repo.arena.grafts) == 0
    # close() is still called -- the native store has its own lifecycle.
    assert repo.closed is True
    # and the live KV cache is dropped too.
    assert repo.arena.reset_calls == 1


def test_release_arm_preserves_node_text_and_identity():
    """Only the DEVICE payload goes; persisted text/lineage is untouched."""
    repo = run._FakeRepo(payloads=3)
    before = [g['text'] for g in repo.arena.grafts]
    run.release_arm(repo)
    assert [g['text'] for g in repo.arena.grafts] == before
    assert len(repo.arena.grafts) == 3


def test_release_arm_is_idempotent_and_never_raises():
    repo = run._FakeRepo(payloads=4)
    assert run.release_arm(repo)['payloads_freed'] == 4
    assert run.release_arm(repo)['payloads_freed'] == 0   # nothing left to free


def test_release_arm_survives_a_repo_that_cannot_close():
    class Stubborn(run._FakeRepo):
        def close(self):
            raise RuntimeError('native store already gone')

    repo = Stubborn(payloads=5)
    # A failure to close must not strand the payloads: they are freed first.
    released = run.release_arm(repo)
    assert released['payloads_freed'] == 5
    assert sum(g['h'] is not None for g in repo.arena.grafts) == 0


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/amendment_1.json')
def test_release_path_is_exercised_on_the_fake_run(tmp_path):
    """Order item 3: the release path runs on the fake gate, not only on GPU."""
    r = run.verify(tmp_path)
    batch_id = r['batch_ids'][0]
    run.batch(batch_id, fake=True, root=tmp_path)
    cell_id = r['batches'][batch_id][0]
    row = json.loads(
        (tmp_path / 'gpu' / batch_id / 'cells' / (cell_id + '.json')).read_text())
    device = row['device_memory']
    for arm in ('off', 'on'):
        released = device['arms'][arm]['released']
        assert released['payloads_freed'] > 0, 'release path did not run'
        assert released['pool_emptied'] is True


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/amendment_1.json')
def test_device_probe_is_recorded_and_never_fabricated(tmp_path):
    """A fake run has no device: the probe records None, not a made-up 0."""
    r = run.verify(tmp_path)
    batch_id = r['batch_ids'][0]
    run.batch(batch_id, fake=True, root=tmp_path)
    cell_id = r['batches'][batch_id][0]
    row = json.loads(
        (tmp_path / 'gpu' / batch_id / 'cells' / (cell_id + '.json')).read_text())
    device = row['device_memory']
    assert device['before_cell_mib'] is None
    assert device['after_cell_mib'] is None
    for arm in ('off', 'on'):
        assert device['arms'][arm]['before_mib'] is None
        assert device['arms'][arm]['after_mib'] is None


def test_device_memory_mib_returns_int_or_none():
    value = run.device_memory_mib()
    assert value is None or (isinstance(value, int) and value >= 0)


# --------------------------------------------------------------------------
# the amendment: binding and scope
# --------------------------------------------------------------------------

@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/amendment_1.json')
def test_amendment_is_sha_bound_to_registration_and_order():
    a = run.amendment()
    assert a is not None, 'amendment 1 must exist for this suite'
    assert a['schema'] == 'grm.r1.amendment.v1'
    assert a['registration_sha256'] == run.sha(run.REG)
    assert a['order_sha256'] == run.sha(
        run.ROOT / 'orders/GRM_R1_MARGIN_FIRST_REPLAY.md')
    # The plan's prediction and verdict rule are carried through UNCHANGED.
    r = run.verify()
    assert a['prediction'] == r['prediction']
    assert a['verdict_rule'] == r['verdict_rule']
    assert a['acceptance_unchanged'] is True


def test_amendment_sha_drift_is_rejected(monkeypatch):
    original = run.sha
    target = run.OUT / 'amendment_1.json'
    monkeypatch.setattr(
        run, 'sha', lambda p: 'bad' if Path(p) == target else original(p))
    with pytest.raises(ValueError, match='R1_AMENDMENT_SHA_MISMATCH'):
        run.amendment()


def test_amendment_chain_drift_is_rejected(tmp_path):
    a = dict(run.amendment())
    a['registration_sha256'] = 'not-the-registration'
    run.write(tmp_path / 'amendment_1.json', a)
    (tmp_path / 'amendment_1.sha256').write_text(
        run.sha(tmp_path / 'amendment_1.json') + '  amendment_1.json\n')
    with pytest.raises(ValueError, match='R1_AMENDMENT_CHAIN_MISMATCH'):
        run.amendment(tmp_path)


def test_amendment_cannot_rebind_an_unlisted_source(tmp_path):
    def poison(a, index):
        if index == 4:
            a['rebound_inputs'] = dict(a['rebound_inputs'])
            a['rebound_inputs']['core/graft_arena.py.evil'] = 'deadbeef'

    _chain_into(tmp_path, poison)
    with pytest.raises(ValueError, match='R1_AMENDMENT_REBIND_OUT_OF_SCOPE'):
        run.verify(tmp_path)


def test_rebinding_does_not_weaken_the_other_inputs(monkeypatch):
    """Only the listed sources are re-bound; the rest still fail closed."""
    original = run.sha
    target = run.ROOT / 'core/grm_admission.py'
    monkeypatch.setattr(
        run, 'sha', lambda p: 'bad' if Path(p) == target else original(p))
    with pytest.raises(ValueError, match='R1_INPUT_SHA_MISMATCH'):
        run.verify()


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/amendment_1.json')
def test_amendment_scope_matches_the_receipts_on_disk():
    r = run.verify()
    scope = run.resume_scope(r, run.OUT)
    assert 'R1' in scope, 'the failed batch is re-armed'
    # The amendment RECORDED retain 5 / reissue 3 ...
    recorded = run.read(run.OUT / 'amendment_1.json')['rearm']['R1']
    assert len(recorded['retain']) == 5 and len(recorded['reissue']) == 3
    assert not set(recorded['retain']) & set(recorded['reissue'])
    assert recorded['retain'] + recorded['reissue'] == r['batches']['R1']
    # ... and the re-issue has since completed, so the scope is satisfied
    # and covers the whole cohort (amendment 3).
    assert scope['R1']['scope_satisfied'] is True
    assert scope['R1']['reissue'] == []
    assert scope['R1']['retain'] == r['batches']['R1']
    # R2-R5 are untouched by the amendment.
    # amendment 1 left R2-R5 alone; later links re-arm what later fails.
    assert (run.read(run.OUT / 'amendment_1.json')['unchanged_batches']
            == ['R2', 'R3', 'R4', 'R5'])


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/amendment_1.json')
def test_amendment_budget_stays_under_the_cap():
    a = run.amendment()
    r = run.verify()
    assert a['total_reserved_seconds'] <= r['gpu_cap_seconds']
    # The FAILED attempt's charge is not refunded -- it stays on the books.
    assert a['prior_charged_seconds'] > 0


# --------------------------------------------------------------------------
# retained receipts are never re-run and never rewritten
# --------------------------------------------------------------------------

@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/amendment_1.json')
def test_retained_receipts_must_be_parity_clean_and_bound():
    r = run.verify()
    scope = run.resume_scope(r, run.OUT)
    for cell_id in scope['R1']['retain']:
        row = run.read(run.OUT / 'gpu/R1/cells' / (cell_id + '.json'))
        assert row['off_plan_parity'] is True
        assert row['on_plan_matches_registered'] is True
        assert row['registration_sha256'] == run.sha(run.REG)
        # They carry REAL measured answers -- that is why they are kept.
        assert all(a['answer_measured'] for a in row['arms'].values())


def _open_rearmed_batch(tmp_path):
    """A batch left FAILED with some receipts missing: the OPEN state."""
    r = run.verify(tmp_path)
    batch_id = r['batch_ids'][0]
    ids = r['batches'][batch_id]
    run.batch(batch_id, fake=True, root=tmp_path)
    d = tmp_path / 'gpu' / batch_id
    keep, drop = ids[:5], ids[5:]
    for cell_id in drop:
        (d / 'cells' / (cell_id + '.json')).unlink()
    (d / 'controller.json').unlink()
    run.write(d / 'controller.json',
              {'status': 'FAILED', 'error': 'oom', 'red': [],
               'elapsed_seconds': 1.0, 'charged_seconds': 263.0,
               'reservation_seconds': 263, 'fake': True, 'attempt': 1,
               'amendment': None, 'cells_run': ids, 'cells_retained': [],
               'registration_sha256': run.sha(run.REG)})
    return r, batch_id, keep, drop


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/amendment_1.json')
def test_retain_list_disagreeing_with_disk_is_a_red_stop(tmp_path):
    # The cross-check is a PRECONDITION, so it is exercised on an OPEN batch
    # (amendment 3); a completed one is satisfied and no longer checked.
    r, batch_id, keep, drop = _open_rearmed_batch(tmp_path)
    a = dict(run.amendment())
    a['rearm'] = {batch_id: {'retain': keep + ['invented-cell'],
                             'reissue': drop, 'lease_seconds': 200}}
    with pytest.raises(ValueError, match='R1_AMENDMENT_RETAIN_MISMATCH'):
        run.resume_scope(r, tmp_path, a)


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/amendment_1.json')
def test_reissue_list_disagreeing_with_disk_is_a_red_stop(tmp_path):
    r, batch_id, keep, drop = _open_rearmed_batch(tmp_path)
    a = dict(run.amendment())
    a['rearm'] = {batch_id: {'retain': keep, 'reissue': drop[:1],
                             'lease_seconds': 200}}
    with pytest.raises(ValueError, match='R1_AMENDMENT_REISSUE_MISMATCH'):
        run.resume_scope(r, tmp_path, a)


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/amendment_1.json')
def test_completed_cells_retained_and_missing_reissued_on_the_fake_path(tmp_path):
    """Order item 3: reproduce 'completed retained, one cell re-issued'.

    Runs a batch, deletes SOME cell receipts to simulate the OOM death, writes
    a matching amendment, and asserts the re-issue runs ONLY the missing cells
    while the retained receipts survive byte-for-byte.
    """
    r = run.verify(tmp_path)
    batch_id = r['batch_ids'][0]
    run.batch(batch_id, fake=True, root=tmp_path)
    ids = r['batches'][batch_id]
    d = tmp_path / 'gpu' / batch_id

    keep, drop = ids[:5], ids[5:]
    assert keep and drop
    kept_bytes = {c: (d / 'cells' / (c + '.json')).read_bytes() for c in keep}
    for cell_id in drop:
        (d / 'cells' / (cell_id + '.json')).unlink()
    # Simulate the failed controller the OOM left behind.
    (d / 'controller.json').unlink()
    run.write(d / 'controller.json',
              {'status': 'FAILED',
               'error': "RuntimeError('cudaMalloc failed: out of memory')",
               'red': [], 'elapsed_seconds': 138.5, 'charged_seconds': 263.0,
               'reservation_seconds': 263, 'fake': True, 'attempt': 1,
               'amendment': None, 'cells_run': ids, 'cells_retained': [],
               'registration_sha256': run.sha(run.REG)})

    rearm = {batch_id: {'retain': keep, 'reissue': drop,
                        'lease_seconds': 200,
                        'prior_status': 'FAILED', 'prior_error': 'oom',
                        'prior_charged_seconds': 263.0,
                        'measured_per_cell_seconds': 18.0,
                        'estimate_basis': 'test fixture'}}
    _chain_into(tmp_path, lambda a, index: a.__setitem__('rearm', rearm))

    # The campaign is un-closed by the amendment ...
    charged, complete = run.campaign_state(run.verify(tmp_path), tmp_path)
    assert charged == 263.0 and complete == []
    # ... and the re-issue runs ONLY the missing cells.
    run.batch(batch_id, fake=True, root=tmp_path)
    controller = json.loads((d / 'controller.json').read_text())
    assert controller['status'] == 'COMPLETE'
    assert controller['cells_run'] == drop
    assert controller['cells_retained'] == keep
    assert controller['amendment'] == 'amendment_1'
    assert controller['attempt'] == 2
    # The prior attempt's controller is archived, not destroyed.
    assert (d / 'controller_attempt_1.json').exists()
    # Every retained receipt is byte-for-byte what it was.
    for cell_id, before in kept_bytes.items():
        assert (d / 'cells' / (cell_id + '.json')).read_bytes() == before
    # And every cell now has a receipt.
    for cell_id in ids:
        assert (d / 'cells' / (cell_id + '.json')).exists()


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/amendment_1.json')
def test_reissue_refuses_when_nothing_is_missing(tmp_path):
    r = run.verify(tmp_path)
    batch_id = r['batch_ids'][0]
    run.batch(batch_id, fake=True, root=tmp_path)
    d = tmp_path / 'gpu' / batch_id
    (d / 'controller.json').unlink()
    run.write(d / 'controller.json',
              {'status': 'FAILED', 'error': 'x', 'red': [],
               'elapsed_seconds': 1.0, 'charged_seconds': 263.0,
               'reservation_seconds': 263, 'fake': True, 'attempt': 1,
               'amendment': None, 'cells_run': [], 'cells_retained': [],
               'registration_sha256': run.sha(run.REG)})
    _chain_into(tmp_path,
                lambda a, index: a.__setitem__('rearm', {batch_id: {'retain': r['batches'][batch_id], 'reissue': [],
                                 'lease_seconds': 60, 'prior_status': 'FAILED',
                                 'prior_error': 'x', 'prior_charged_seconds': 263.0,
                                 'measured_per_cell_seconds': 18.0,
                                 'estimate_basis': 'test fixture'}}))
    with pytest.raises(RuntimeError, match='R1_NOTHING_TO_REISSUE'):
        run.batch(batch_id, fake=True, root=tmp_path)


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/amendment_1.json')
def test_an_unamended_failed_batch_still_closes_the_campaign(tmp_path):
    """The amendment un-closes ONLY what it names; nothing else is relaxed."""
    r = run.verify(tmp_path)
    second = r['batch_ids'][1]
    d = tmp_path / 'gpu' / second
    run.write(d / 'controller.json',
              {'status': 'FAILED', 'error': 'unrelated', 'red': [],
               'elapsed_seconds': 1.0, 'charged_seconds': 10.0,
               'reservation_seconds': 194, 'fake': True, 'attempt': 1,
               'amendment': None, 'cells_run': [], 'cells_retained': [],
               'registration_sha256': run.sha(run.REG)})
    _chain_into(tmp_path,
                lambda a, index: a.__setitem__('rearm', {'R1': a['rearm']['R1']}))
    with pytest.raises(ValueError, match='R1_FAILED_CAMPAIGN_STOP'):
        run.campaign_state(run.verify(tmp_path), tmp_path)
