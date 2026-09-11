"""GRM-R1 amendment 4: idle-card pre-check and the NON_FIT rail.

The R4 OOM was NOT a fit problem: R4's first cell is the same size as one R3
completed on the same frame, and R4 is the smallest profile batch by total
payload.  The card was held by a display-side program with no compute
process listed.  These tests cover the pre-check that declines to start on a
busy card, and the rail that records an unloadable cell instead of failing
the campaign.

Prior art: GRM FIX-8 `grm_scout_fix8_resume` memory-probe tests and C7/FIX8
create-only failure receipts (GRM contributors, 2026) -- TAKEN: the probe
shape and the record-don't-retry contract.  OURS: the NON_FIT accounting
assertions.  No prior art known to me for this exact composition.  No
model-quality claim.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import grm_r1_replay as run


def _chain_into(tmp_path, rearm=None, links=(1, 2, 3, 4)):
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


# --------------------------------------------------------------------------
# the counts that refute the fit hypothesis
# --------------------------------------------------------------------------

def _payload_mib(cells, cell_id):
    root = Path(cells[cell_id]['checkpoint']).parent / 'repository'
    return sum(p.stat().st_size for p in root.glob('nodes/*.npz')) / 1048576


def _nodes(cells, cell_id):
    root = Path(cells[cell_id]['checkpoint']).parent / 'repository'
    return len(json.loads((root / 'manifest.json').read_text())['nodes'])


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/amendment_4.json')
def test_r4_cells_are_not_larger_than_cells_r3_completed():
    """The evidence for NOT implementing lever 2(a)."""
    cells = {c['id']: c for c in run.cells()}
    r = run.verify()
    r3_max = max(_payload_mib(cells, c) for c in r['batches']['R3'])
    r4_max = max(_payload_mib(cells, c) for c in r['batches']['R4'])
    r5_max = max(_payload_mib(cells, c) for c in r['batches']['R5'])
    # R4's largest cell is no larger than R3's largest -- and R3 completed.
    assert r4_max <= r3_max + 0.1
    assert r5_max < r4_max
    # R4's first cell is the same size as one R3 ran to completion.
    assert _nodes(cells, r['batches']['R4'][0]) == 25
    assert any(_nodes(cells, c) == 25 for c in r['batches']['R3'])
    # And R4 is the smaller batch in total.
    assert (sum(_payload_mib(cells, c) for c in r['batches']['R4'])
            < sum(_payload_mib(cells, c) for c in r['batches']['R3']))


def test_amendment_records_the_counts_and_declines_lever_2a():
    a = run.amendment()
    counts = a['node_payload_counts']
    assert set(counts) == {'R1', 'R2', 'R3', 'R4', 'R5'}
    assert counts['R4']['total_mib'] < counts['R3']['total_mib']
    assert counts['R4']['max_nodes'] == counts['R3']['max_nodes'] == 25
    assert a['lever_2a_host_resident_payloads']['implemented'] is False
    assert a['fit_analysis']['headroom_multiple_over_largest_cell'] >= 10


# --------------------------------------------------------------------------
# the idle-card pre-check
# --------------------------------------------------------------------------

def test_device_snapshot_shape():
    snapshot = run.device_snapshot()
    assert snapshot['status'] in ('OK', 'ERROR')
    if snapshot['status'] == 'OK':
        for key in ('memory_used_mib', 'memory_total_mib', 'memory_free_mib'):
            assert isinstance(snapshot[key], int)
        assert (snapshot['memory_used_mib'] + snapshot['memory_free_mib']
                == snapshot['memory_total_mib'])
        assert isinstance(snapshot['processes'], list)


def test_idle_gate_refuses_a_busy_card():
    busy = {'status': 'OK', 'memory_used_mib': 3144, 'memory_total_mib': 12282,
            'memory_free_mib': 9138, 'compute_list_empty': True,
            'other_process_pids': []}
    ok, reason = run.idle_gate(busy)
    assert ok is False and 'DEVICE_BUSY' in reason
    assert '3144' in reason


def test_idle_gate_accepts_an_idle_card():
    idle = {'status': 'OK', 'memory_used_mib': 320, 'memory_total_mib': 12282,
            'memory_free_mib': 11962, 'compute_list_empty': True,
            'other_process_pids': []}
    ok, reason = run.idle_gate(idle)
    assert ok is True and 'IDLE' in reason


def test_idle_gate_ignores_the_compute_process_list():
    """THE R4 CASE: memory held, no compute process listed. Must still refuse.

    FIX-8's parse_memory states the rule: never infer an idle card merely
    from an empty compute list.
    """
    holder = {'status': 'OK', 'memory_used_mib': 3144,
              'memory_total_mib': 12282, 'memory_free_mib': 9138,
              'compute_list_empty': True, 'other_process_pids': []}
    assert run.idle_gate(holder)[0] is False


def test_idle_gate_fails_closed_on_a_broken_probe():
    ok, reason = run.idle_gate({'status': 'ERROR', 'error': 'nvidia-smi gone'})
    assert ok is False and 'DEVICE_PROBE_FAILED' in reason


def test_await_idle_is_bounded_and_records_attempts(monkeypatch):
    busy = {'status': 'OK', 'memory_used_mib': 5000, 'memory_total_mib': 12282,
            'memory_free_mib': 7282, 'compute_list_empty': True,
            'other_process_pids': [], 'time_unix': 0.0}
    monkeypatch.setattr(run, 'device_snapshot', lambda: dict(busy))
    ok, receipt = run.await_idle(wait_seconds=0)
    assert ok is False
    assert receipt['attempts'] and receipt['attempts'][0]['ok'] is False
    assert 'DEVICE_BUSY' in receipt['reason']


def test_await_idle_returns_as_soon_as_the_card_frees(monkeypatch):
    states = [{'status': 'OK', 'memory_used_mib': 5000,
               'memory_total_mib': 12282, 'memory_free_mib': 7282,
               'compute_list_empty': True, 'other_process_pids': [],
               'time_unix': 0.0},
              {'status': 'OK', 'memory_used_mib': 320,
               'memory_total_mib': 12282, 'memory_free_mib': 11962,
               'compute_list_empty': True, 'other_process_pids': [],
               'time_unix': 1.0}]
    monkeypatch.setattr(run, 'device_snapshot', lambda: states.pop(0))
    ok, receipt = run.await_idle(wait_seconds=5, poll_seconds=0)
    assert ok is True
    assert len(receipt['attempts']) == 2


def test_the_gate_never_signals_anything():
    """It declines; it does not kill, signal or clear. Asserted on the source."""
    source = (run.ROOT / 'scripts/grm_r1_replay.py').read_text()
    for banned in ('os.kill', 'signal.SIGKILL', 'SIGTERM', 'pkill', 'killall'):
        assert banned not in source, f'signalling present: {banned}'


# --------------------------------------------------------------------------
# the NON_FIT rail
# --------------------------------------------------------------------------

def test_is_oom_matches_only_device_oom():
    assert run.is_oom(RuntimeError('cudaMalloc failed: out of memory'))
    assert run.is_oom(RuntimeError('CUDA error: out of memory'))
    assert not run.is_oom(RuntimeError('something else entirely'))
    assert not run.is_oom(ValueError('R1_OFF_PLAN_PARITY_RED_STOP: x'))


def test_non_fit_receipt_is_create_only_and_unmeasured(tmp_path):
    c = run.cells()[0]
    value = run.non_fit_receipt(
        c, tmp_path, RuntimeError('cudaMalloc failed: out of memory'),
        {'status': 'OK', 'memory_used_mib': 3144, 'memory_total_mib': 12282},
        stage='load_or_harvest')
    assert value['status'] == 'NON_FIT'
    assert value['measured'] is False and value['answer_measured'] is False
    assert value['arms'] == {} and value['off_plan_parity'] is None
    assert value['stage'] == 'load_or_harvest'
    assert value['device_memory']['memory_used_mib'] == 3144
    assert 'NOT a measurement' in value['evidence_class']
    p = tmp_path / 'cells' / (c['id'] + '.json')
    assert p.exists()
    with pytest.raises(FileExistsError):
        run.write(p, {'overwrite': 'attempt'})


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/amendment_4.json')
def test_batch_continues_past_a_non_fit_cell(tmp_path, monkeypatch):
    """THE RAIL: one unloadable cell no longer fails the whole batch."""
    r = run.verify(tmp_path)
    batch_id = r['batch_ids'][0]
    ids = r['batches'][batch_id]
    target = ids[0]
    real = run.fake_arm

    def oom_on_first(cell, descriptor, arm, session):
        if cell['id'] == target:
            raise RuntimeError('cudaMalloc failed: out of memory')
        return real(cell, descriptor, arm, session)

    monkeypatch.setattr(run, 'fake_arm', oom_on_first)
    run.batch(batch_id, fake=True, root=tmp_path)

    d = tmp_path / 'gpu' / batch_id
    controller = json.loads((d / 'controller.json').read_text())
    assert controller['status'] == 'COMPLETE'
    assert controller['non_fit_cells'] == [target]
    # The NON_FIT cell has a receipt ...
    row = json.loads((d / 'cells' / (target + '.json')).read_text())
    assert row['status'] == 'NON_FIT'
    # ... and every OTHER cell ran normally.
    for cell_id in ids[1:]:
        other = json.loads((d / 'cells' / (cell_id + '.json')).read_text())
        assert other.get('status') != 'NON_FIT'
        assert other['off_plan_parity'] is True


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/amendment_4.json')
def test_non_fit_is_unmeasured_never_unchanged(tmp_path, monkeypatch):
    r = run.verify(tmp_path)
    batch_id = r['batch_ids'][0]
    target = r['batches'][batch_id][0]
    real = run.fake_arm

    def oom_on_first(cell, descriptor, arm, session):
        if cell['id'] == target:
            raise RuntimeError('cudaMalloc failed: out of memory')
        return real(cell, descriptor, arm, session)

    monkeypatch.setattr(run, 'fake_arm', oom_on_first)
    run.batch(batch_id, fake=True, root=tmp_path)
    value = run.summary(tmp_path)
    assert value['non_fit_count'] == 1
    assert target in value['non_fit_cells']
    # It is NOT folded into any transition bucket.
    assert sum(value['transitions'].values()) < len(r['batches'][batch_id])
    # A campaign with a NON_FIT cell is not complete and cannot adopt.
    assert value['complete'] is False
    assert value['adopt'] is False
    assert value['status'] == 'NOT_MEASURED'


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/amendment_4.json')
def test_a_non_oom_error_still_stops_the_campaign(tmp_path, monkeypatch):
    """The rail is narrow: only an OOM is survivable."""
    r = run.verify(tmp_path)
    batch_id = r['batch_ids'][0]
    target = r['batches'][batch_id][0]
    real = run.fake_arm

    def boom(cell, descriptor, arm, session):
        if cell['id'] == target:
            raise RuntimeError('a completely different failure')
        return real(cell, descriptor, arm, session)

    monkeypatch.setattr(run, 'fake_arm', boom)
    with pytest.raises(RuntimeError, match='a completely different failure'):
        run.batch(batch_id, fake=True, root=tmp_path)
    controller = json.loads(
        (tmp_path / 'gpu' / batch_id / 'controller.json').read_text())
    assert controller['status'] == 'FAILED'


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/amendment_4.json')
def test_parity_red_still_stops_and_is_not_swallowed_as_non_fit(tmp_path,
                                                                monkeypatch):
    r = run.verify(tmp_path)
    batch_id = r['batch_ids'][0]
    real = run.fake_arm

    def drifting(cell, descriptor, arm, session):
        value = real(cell, descriptor, arm, session)
        if arm == 'off':
            value['rank_plan'] = []
        return value

    monkeypatch.setattr(run, 'fake_arm', drifting)
    with pytest.raises(RuntimeError, match='R1_OFF_PLAN_PARITY_RED_STOP'):
        run.batch(batch_id, fake=True, root=tmp_path)


# --------------------------------------------------------------------------
# the amendment
# --------------------------------------------------------------------------

@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/amendment_4.json')
def test_amendment_4_chains_to_amendment_3():
    a = run.amendment()
    assert a['amendment'] == 4
    assert a['registration_sha256'] == run.sha(run.REG)
    assert a['previous_amendment_sha256'] == run.sha(
        run.OUT / 'amendment_3.json')
    assert len(a['_chain']) == 4
    assert a['acceptance_unchanged'] is True
    r = run.verify()
    assert a['prediction'] == r['prediction']
    assert a['verdict_rule'] == r['verdict_rule']


def test_amendment_4_documents_both_defects():
    a = run.amendment()
    ids = {d['id'] for d in a['defects']}
    assert ids == {'R1-D6-foreign-device-holder',
                   'R1-D7-one-unloadable-cell-failed-the-campaign'}
    holder = next(d for d in a['defects']
                  if d['id'] == 'R1-D6-foreign-device-holder')
    assert 'NOT a payload-fit problem' in holder['not_the_cause']
    assert 'never' in holder['fix'].casefold()


# --------------------------------------------------------------------------
# re-arm from a FAILED batch with ZERO retained receipts
# --------------------------------------------------------------------------

@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/amendment_4.json')
def test_rearm_from_failed_with_zero_receipts(tmp_path):
    """R4's exact shape: FAILED, 0 cell receipts, every cell to re-issue.

    Amendments 1-3 only ever re-armed a batch that had SOME receipts. R4
    failed 13 s in with none at all, so `retain` is empty -- the re-issue
    must still run, and must run the WHOLE cohort.
    """
    r = run.verify(tmp_path)
    batch_id = r['batch_ids'][0]
    ids = r['batches'][batch_id]
    d = tmp_path / 'gpu' / batch_id
    # A batch that died before writing any receipt.
    run.write(d / 'reservation.json',
              {'seconds': 238, 'batch': batch_id,
               'registration_sha256': run.sha(run.REG)})
    run.write(d / 'controller.json',
              {'status': 'FAILED',
               'error': 'RuntimeError: cudaMalloc failed: out of memory',
               'red': [], 'elapsed_seconds': 13.4, 'charged_seconds': 238.0,
               'reservation_seconds': 238, 'fake': True, 'attempt': 1,
               'amendment': None, 'cells_run': ids, 'cells_retained': [],
               'registration_sha256': run.sha(run.REG)})
    assert not (d / 'cells').exists()

    _chain_into(tmp_path, rearm={batch_id: {
        'retain': [], 'reissue': ids, 'lease_seconds': 238,
        'prior_status': 'FAILED', 'prior_error': 'oom',
        'prior_charged_seconds': 238.0,
        'estimate_basis': 'test fixture'}})

    scope = run.resume_scope(run.verify(tmp_path), tmp_path)[batch_id]
    assert scope['retain'] == [] and scope['reissue'] == ids
    assert scope['scope_satisfied'] is False

    run.batch(batch_id, fake=True, root=tmp_path)

    controller = json.loads((d / 'controller.json').read_text())
    assert controller['status'] == 'COMPLETE'
    assert controller['cells_run'] == ids          # the WHOLE cohort re-ran
    assert controller['cells_retained'] == []
    assert controller['amendment'] == 'amendment_1'
    # The failed attempt is archived, and its charge is not refunded.
    assert (d / 'controller_attempt_1.json').exists()
    charged, complete = run.campaign_state(run.verify(tmp_path), tmp_path)
    assert batch_id in complete
    assert charged > 238.0
    for cell_id in ids:
        assert (d / 'cells' / (cell_id + '.json')).exists()


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/amendment_4.json')
def test_real_r4_is_rearmed_with_zero_retained():
    """The live receipts: R4 FAILED with 0 receipts, re-armed for all 8."""
    r = run.verify()
    scope = run.resume_scope(r, run.OUT)['R4']
    assert scope['retain'] == []
    assert len(scope['reissue']) == 8
    assert scope['scope_satisfied'] is False
    # R5 has no controller at all: a normal first run, not a re-arm.
    assert run.resume_scope(r, run.OUT).get('R5') is None
    charged, complete = run.campaign_state(r, run.OUT)
    assert complete == ['R1', 'R2', 'R3']
