"""C8 amendment CPU author baseline, not independent blind verification.

Prior art: C8/C2 (GRM, 2026) immutable receipt tests, hand-computed boundary
tests and fake subprocess orchestration. Taken: adversarial stale/forged
inputs and simulated controller state; ours: amendment/resume/report cases.
No prior art known to me for this precise adapter beyond those local systems.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from contextlib import contextmanager

import pytest

from scripts import grm_c8_cells as c
from scripts import grm_c8_profiler as p


def fake_rows(count, fraction=.25, demand=False):
    result = []
    for turn in range(count):
        stages = {s: {'wall_ms': 0., 'gpu_ms': 0., 'peak_pool_used_bytes': 10,
                      'device_used_boundary_peak_bytes': 20} for s in p.STAGES}
        stages['decode']['wall_ms'] = fraction * 100
        stages['other']['wall_ms'] = (1 - fraction) * 100
        row = {'status': 'COMPLETE', 'turn_wall_ms': 100., 'stages': stages, 'turn': turn}
        if demand:
            row.update(turn=33, demand_info={'demand_fired': True, 'demand_trip_taken': True})
        result.append(row)
    return result


def make_receipt(cell, *, status='COMPLETE', fraction=.25, amendment_sha=None):
    directory = c.OUT / 'cells' / cell['id']
    binding = {'registration_sha256': c.ANCHOR,
               'amendment_sha256': amendment_sha or c.AMENDMENT_ANCHOR}
    c.create(directory / 'reservation.json', {'seconds': 285, 'cell': cell, **binding})
    c.create(directory / 'worker.json', {'status': 'COMPLETE', 'cell': cell,
        'rows': fake_rows(cell['turns'], fraction, cell['battery'] == 'demand'), **binding})
    c.create(directory / 'controller.json', {'status': status,
        'worker_sha256': c.sha(directory / 'worker.json'),
        'charged_seconds': cell['estimate_seconds'], 'ended_epoch': 1., **binding})
    return directory


def test_amendment_cap_all_cells_and_original_registration_immutable():
    before = c.REG.read_bytes()
    original = c.registration()
    effective = c.effective_registration()
    assert c.sha(c.AMENDMENT) == c.AMENDMENT_ANCHOR
    assert original['budget_seconds'] == 1800
    assert effective['budget_seconds'] == 2700
    assert effective['cells'] == original['cells'] and len(effective['cells']) == 24
    assert sum(x['estimate_seconds'] for x in effective['cells']) == 2591 < 2700
    assert effective['decision_rule'] == original['decision_rule']
    assert c.REG.read_bytes() == before and c.sha(c.REG) == c.ANCHOR
    c.fit(effective)


def test_preflight_reads_amendment_without_gpu(capsys):
    assert c.main(['--preflight']) == 0
    value = json.loads(capsys.readouterr().out)
    assert value['budget_seconds'] == 2700 and value['estimated_seconds'] == 2591
    assert value['amendment_sha256'] == c.sha(c.AMENDMENT)


def test_forged_amendment_refused_before_reservation(tmp_path, monkeypatch):
    forged = c.read(c.AMENDMENT)
    forged['budget_seconds'] = 99999
    path = tmp_path / 'forged.json'
    c.create(path, forged)
    monkeypatch.setattr(c, 'AMENDMENT', path)
    monkeypatch.setattr(c, 'OUT', tmp_path / 'must_not_exist')
    with pytest.raises(ValueError, match='amendment SHA differs from code anchor'):
        c.main(['--cell', 'gpu-identity', '--profile-turn', '--resume'])
    assert not c.OUT.exists()


@pytest.mark.parametrize('field,value,error', [
    ('registration_sha256', '0' * 64, 'stale amendment registration'),
    ('sequence', 0, 'stale amendment registration'),
    ('budget_seconds', 99999, 'unauthorized amendment cap'),
    ('runner_template_sha256', '0' * 64, 'amended runner template changed'),
])
def test_stale_amendment_semantics_even_with_rehashed_file(tmp_path, monkeypatch, field, value, error):
    # Defense-in-depth: simulate re-pinning a modified document; semantic
    # and source checks still refuse it. This is not a signature test.
    forged = c.read(c.AMENDMENT)
    forged[field] = value
    path = tmp_path / 'stale.json'
    c.create(path, forged)
    monkeypatch.setattr(c, 'AMENDMENT', path)
    monkeypatch.setattr(c, 'AMENDMENT_ANCHOR', c.sha(path))
    with pytest.raises(ValueError, match=error):
        c.main(['--preflight'])


def test_stale_amendment_input_refused(tmp_path, monkeypatch):
    original_sha = c.sha
    target = 'orders/GRM_C8_AMENDMENT_1.md'
    monkeypatch.setattr(c, 'sha', lambda path: '0' * 64 if str(path).endswith(target) else original_sha(path))
    with pytest.raises(ValueError, match='amendment bound input changed'):
        c.main(['--preflight'])


def test_missing_amendment_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'AMENDMENT', tmp_path / 'missing.json')
    with pytest.raises(FileNotFoundError):
        c.main(['--preflight'])


@pytest.mark.parametrize('state,expected', [('COMPLETE', 0), ('RED', 1), ('unfinished', 1), ('empty', 1)])
def test_resume_skips_all_started_states_without_launch(tmp_path, monkeypatch, capsys, state, expected):
    r = c.effective_registration()
    cell = r['cells'][0]
    monkeypatch.setattr(c, 'OUT', tmp_path)
    directory = tmp_path / 'cells' / cell['id']
    if state in ('COMPLETE', 'RED'):
        make_receipt(cell, status=state)
    elif state == 'unfinished':
        c.create(directory / 'reservation.json', {'cell': cell})
    else:
        directory.mkdir(parents=True)
    before = {str(x): x.read_bytes() for x in directory.rglob('*') if x.is_file()}
    def forbidden(*a, **kw):
        pytest.fail('started cell reached launch, cooldown, or new reservation')
    monkeypatch.setattr(c.subprocess, 'run', forbidden)
    monkeypatch.setattr(c.time, 'sleep', forbidden)
    monkeypatch.setattr(c, 'create', forbidden)
    assert c.run_cell(r, cell, resume=True) == expected
    record = json.loads(capsys.readouterr().out)
    assert record['retry'] is False and record['status'].startswith('SKIPPED_')
    assert {str(x): x.read_bytes() for x in directory.rglob('*') if x.is_file()} == before


def test_resume_stale_receipt_is_red(tmp_path, monkeypatch):
    r = c.effective_registration()
    monkeypatch.setattr(c, 'OUT', tmp_path)
    make_receipt(r['cells'][0], amendment_sha='0' * 64)
    assert c.run_cell(r, r['cells'][0], resume=True) == 1


def test_prior_red_blocks_unstarted_cell(tmp_path, monkeypatch):
    r = c.effective_registration()
    monkeypatch.setattr(c, 'OUT', tmp_path)
    make_receipt(r['cells'][0])
    make_receipt(r['cells'][1], status='RED')
    with pytest.raises(ValueError, match='prior RED'):
        c.run_cell(r, r['cells'][2], resume=True)
    assert not (tmp_path / 'cells' / r['cells'][2]['id']).exists()


def test_forecast_all_24_reservations_fit_and_overrun_refused():
    r = c.effective_registration()
    used = 0
    for cell in r['cells']:
        reserved = c.reservation_seconds(r, cell, used)
        assert cell['estimate_seconds'] <= reserved - 5
        assert used + reserved <= 2700 and reserved <= 285
        used += cell['estimate_seconds']
    assert used == 2591 and reserved == 169
    with pytest.raises(ValueError, match='remaining budget cannot fit'):
        c.reservation_seconds(r, r['cells'][-1], 2636)


def test_unstarted_final_cell_uses_remaining_lease_cpu_fake(tmp_path, monkeypatch):
    r = c.effective_registration()
    monkeypatch.setattr(c, 'OUT', tmp_path)
    for cell in r['cells'][:-1]:
        make_receipt(cell)
    cell = r['cells'][-1]
    binding = {'registration_sha256': c.ANCHOR, 'amendment_sha256': c.AMENDMENT_ANCHOR}
    monkeypatch.setattr(c, 'bindings', lambda: binding)
    observed = []
    @contextmanager
    def lease(seconds, wait):
        observed.append((seconds, wait))
        yield
    monkeypatch.setitem(sys.modules, 'scripts.grm_cmc1_gpu_arms', types.SimpleNamespace(gpu_lease=lease))
    def launch(command, **kw):
        assert kw['timeout'] == 164
        assert kw['env']['GRM_DEMAND_NGH'] == '1'
        assert command[-3:] == ['--worker', cell['id'], '--profile-turn']
        c.create(tmp_path / 'cells' / cell['id'] / 'worker.json', {
            'status': 'COMPLETE', 'cell': cell, 'rows': fake_rows(1, demand=True), **binding})
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(c.subprocess, 'run', launch)
    timer = iter((100., 160.))
    monkeypatch.setattr(c.time, 'monotonic', lambda: next(timer))
    assert c.run_cell(r, cell, resume=True) == 0
    assert observed == [(169, 240)]
    assert c.completed(cell['id'])['rows'][0]['demand_info']['demand_trip_taken']
    assert c.read(tmp_path / 'cells' / cell['id'] / 'controller.json')['charged_seconds'] == 60.


@pytest.mark.parametrize('fraction,expected', [(.25, 'session routing/admission wins'),
    (.5, 'APA decode integration competitive; lead reports both')])
def test_complete_summary_stage_table_demand_and_decision(tmp_path, monkeypatch, fraction, expected):
    r = c.effective_registration()
    monkeypatch.setattr(c, 'OUT', tmp_path)
    monkeypatch.setattr(c, 'bindings', lambda: {})
    for cell in r['cells']:
        make_receipt(cell, fraction=fraction)
    result = c.summary()
    assert result['status'] == 'COMPLETE' and result['allocation_decision'] == expected
    assert result['demand_trip_turn']['turn'] == 33
    assert result['demand_trip_turn']['observation']['demand_trip_taken'] is True
    text = c.render_summary(result)
    assert '| stage | metric | mean | p50 | p95 |' in text
    for stage in p.STAGES:
        assert f'| {stage} | wall_ms |' in text
    assert f'Decode share of turn wall: {fraction:.6f}' in text
    assert f'Allocation decision: {expected}' in text
    assert 'lead\'s error' in text and 'S=2,048' in text
    for b in result['batteries'].values():
        assert b['stages']['decode']['wall_ms'] == dict(mean=fraction * 100, p50=fraction * 100, p95=fraction * 100)


@pytest.mark.parametrize('defect', ['missing', 'red', 'demand_not_taken', 'identity_missing', 'conflict'])
def test_summary_withholds_allocation_for_missing_red_demand_identity_or_conflict(tmp_path, monkeypatch, defect):
    r = c.effective_registration()
    monkeypatch.setattr(c, 'OUT', tmp_path)
    monkeypatch.setattr(c, 'bindings', lambda: {})
    for cell in r['cells']:
        if defect == 'missing' and cell['battery'] == 'census':
            continue
        if defect == 'identity_missing' and cell['battery'] == 'identity':
            continue
        directory = make_receipt(cell, status='RED' if defect == 'red' and cell['battery'] == 'sup' else 'COMPLETE',
            fraction=.75 if defect == 'conflict' and cell['battery'] == 'census' else .25)
        if defect == 'demand_not_taken' and cell['battery'] == 'demand':
            # A self-consistent worker/controller hash is insufficient if the
            # registered demand observation itself fails.
            worker = c.read(directory / 'worker.json')
            worker['rows'][0]['demand_info']['demand_trip_taken'] = False
            (directory / 'worker.json').write_text(json.dumps(worker))
            ctl = c.read(directory / 'controller.json')
            ctl['worker_sha256'] = c.sha(directory / 'worker.json')
            (directory / 'controller.json').write_text(json.dumps(ctl))
    result = c.summary()
    assert result['allocation_decision'] is None
    assert result['conflicting_batteries'] is (defect == 'conflict')
    assert result['status'] == ('COMPLETE' if defect == 'conflict' else 'BLOCKED_NOT_MEASURED')


def test_lead_commands_all_cells_in_order_resume_and_final_summary(tmp_path):
    # Exercise the real executable shell file using fake python/timeout on
    # PATH. No real cell process, lease, GPU, wait, or signal is launched.
    script = c.ROOT / 'artifacts/grm_c8/lead_commands.txt'
    assert os.access(script, os.X_OK)
    log = tmp_path / 'calls.jsonl'
    shim = tmp_path / 'python'
    shim.write_text('''#!/usr/bin/python3
import json, os, sys
with open(os.environ['C8_TEST_CALL_LOG'], 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
raise SystemExit(7 if '--cell' in sys.argv and os.environ.get('C8_TEST_RED_CELL') in sys.argv else 0)
''')
    shim.chmod(0o755)
    timeout = tmp_path / 'timeout'
    timeout.write_text('#!/bin/bash\nshift 3\nexec "$@"\n')
    timeout.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path) + ':' + os.environ['PATH'], C8_TEST_CALL_LOG=str(log))
    proc = subprocess.run([str(script)], env=env, capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0][-1] == '--dry-run' and calls[1][-1] == '--preflight'
    cells = [args for args in calls if '--cell' in args]
    assert [args[args.index('--cell') + 1] for args in cells] == [x['id'] for x in c.registration()['cells']]
    assert all('--resume' in args and '--profile-turn' in args for args in cells)
    assert calls[-1][-1] == '--summary'
    log.write_text('')
    env['C8_TEST_RED_CELL'] = 'sup-fresh_fact_controls'
    proc = subprocess.run([str(script)], env=env, capture_output=True, text=True, timeout=10)
    assert proc.returncode == 7
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len([args for args in calls if '--cell' in args]) == 3
    assert calls[-1][-1] == '--summary'
