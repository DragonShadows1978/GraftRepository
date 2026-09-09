"""Author CPU operands only. Prior art: house X1 synthetic fixtures (2026).
New: r3 lifecycle/failure/budget boundaries; never GPU timing evidence.
"""
import json
import shutil
from unittest.mock import patch

import pytest

from scripts import grm_x1_campaign as c
from scripts import grm_x1_units as u
from scripts.grm_x1_register import create, raw_json, sha
from test_grm_x1_campaign import synthetic_rows


@pytest.fixture
def r3_tree(tmp_path, monkeypatch):
    original = c.ROOT
    out = tmp_path / 'artifacts/grm_x1'
    frozen_sources = c.source_manifest()
    shutil.copytree(c.OUT, out, ignore=shutil.ignore_patterns('sessions', 'r3'))
    registration = c.read(out / 'continuation_03_registration.json')
    for name in ['orders/GRM_X1_AMENDMENT_3.md', 'orders/GRM_X1_AMENDMENT_2.md']:
        target = tmp_path / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(original / name, target)
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'OUT', out)
    monkeypatch.setattr(c, 'FIX', out / 'fixtures')
    monkeypatch.setattr(c, 'source_manifest', lambda: frozen_sources.copy())
    monkeypatch.setattr(c, 'verify_fixtures', lambda: {'status': 'PASS'})
    monkeypatch.setattr(c, 'dependency_inventory', lambda: c.read(out / 'handoff_manifest.json')['dependencies'])
    return out, registration, frozen_sources


def sample(unit):
    oracle = synthetic_rows()
    data = oracle if unit['phase'] == 'oracle' else [{**r, 'arm': 'N'} for r in oracle if r['arm'] == 'C']
    return [r for r in data if r['query_id'] in unit['query_ids'] and r['multiplicity'] == unit['multiplicity']
            and r['arm'] in unit['arms'] and r['condition'] in unit['conditions']]


def simulated(unit, status='COMPLETE', attempt=1, wall=10, rows=None, non_fit=False, claim_only=False):
    fp = sha(c.OUT / 'continuation_03.json')
    create(c.OUT / 'claims/r3' / f'{u.key(unit, attempt)}_{fp}.json', raw_json({
        'cell': unit['cell'], 'unit_id': unit['unit_id'], 'attempt': attempt,
        'fingerprint': fp, 'reserved_gpu_s': 285}))
    if claim_only:
        return None
    return u.record(unit, attempt, fp, status, sample(unit) if rows is None else rows, wall,
                    {'type': 'SyntheticError', 'message': 'CPU operand'} if status == 'RED' else None, non_fit)


def test_r3_enumeration_order(r3_tree):
    layout = u.units()
    assert len(layout) == len({x['unit_id'] for x in layout}) == 144
    assert [x['query_ordinal'] for x in layout[:12]] == list(range(0, 24, 2))
    for cell in c.read(c.FIX / 'cells.json'):
        selected = [x for x in layout if x['cell'] == cell['cell']]
        assert [x['query_ids'][0] for x in selected] == cell['query_ids']
        assert len(selected) == (12 if cell['phase'] == 'oracle' else 24)
        assert all(x['arms'] == cell['arms'] and x['conditions'] == cell['conditions'] for x in selected)


def test_r3_create_only(r3_tree):
    unit = u.units()[0]
    path = simulated(unit)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        u.record(unit, 1, u.fingerprint(), 'COMPLETE', sample(unit), 11)
    assert path.read_bytes() == before
    value = c.read(path)
    value['worker_wall_s'] = 9
    path.write_bytes(raw_json(value))
    with pytest.raises(ValueError, match='content digest'):
        u.state()


def test_r3_aggregate_sum(r3_tree):
    all_rows = []
    layout = u.units()
    for unit in layout:
        data = sample(unit)
        simulated(unit, rows=data)
        all_rows.extend(data)
    current = u.state()
    reference = c.aggregate(all_rows)
    assert current['rows'] == 576
    for field in ('table', 'source_strata', 'query_strata', 'predictions', 'kill', 'oracle_positive'):
        assert current[field] == reference[field]
    assert current['r3_worker_wall_s'] == 1440
    assert current['charged_gpu_s'] == 2010
    assert sum(x['worker_wall_s'] for x in current['cells'].values()) == 1440
    assert sum(x['rows'] for x in current['cells'].values()) == 576
    assert set(current['cell_status'].values()) == {'COMPLETE'}
    assert u.next_unit('oracle_m1_s0')[1] == 'CELL_COMPLETE'


def test_r3_resume_next_missing(r3_tree):
    first = u.next_unit()[0]
    simulated(first)
    assert u.next_unit(first['cell'])[0]['unit_id'] == u.units()[1]['unit_id']
    simulated(u.units()[2])  # Hole stays next, despite a later receipt.
    assert u.next_unit()[0]['unit_id'] == u.units()[1]['unit_id']
    assert u.next_unit('natural_m1')[1] == 'NATURAL_LOCKED'


def test_r3_red_advances_and_second_red_stops(r3_tree):
    first = u.units()[0]
    simulated(first, 'RED', rows=[])
    assert u.next_unit(first['cell'])[0] == u.units()[1]
    assert u.next_unit(first['cell'], first['unit_id'])[2] == 2
    simulated(first, 'RED', attempt=2, rows=[])
    current = u.state()
    assert current['stop'] and current['units'][first['unit_id']]['red_count'] == 2
    assert u.next_unit()[1] == 'STOP_SECOND_RED_OR_INCOMPLETE'


def test_r3_non_fit_stays_local_and_no_forward_chunking(r3_tree):
    first = u.units()[0]
    simulated(first, 'RED', wall=280.01, rows=sample(first)[:1], non_fit=True)
    current = u.state()
    assert current['units'][first['unit_id']]['fit_status'] == 'NON_FIT'
    assert current['rows'] == 1 and current['planning_estimate_s'] is None
    assert u.next_unit(first['cell'])[0] == u.units()[1]
    with pytest.raises(ValueError, match='non-timeout'):
        u.next_unit(first['cell'], first['unit_id'])


def test_r3_incomplete_claim_stops(r3_tree):
    simulated(u.units()[0], claim_only=True)
    current = u.state()
    assert current['charged_gpu_s'] == 855
    assert current['outstanding_reserved_s'] == 285
    assert u.next_unit()[1] == 'STOP_SECOND_RED_OR_INCOMPLETE'


def test_r3_historical_bytes(r3_tree):
    _, reg, _ = r3_tree
    before = {p: (c.ROOT / p).read_bytes() for p in reg['historical_files']}
    simulated(u.units()[0], 'RED', rows=[])
    current = u.state()
    assert len(current['historical_red']) == 2
    assert sorted(len(r['rows']) for r in current['historical_red']) == [0, 17]
    for p, data in before.items():
        assert (c.ROOT / p).read_bytes() == data
        assert sha(c.ROOT / p) == reg['historical_files'][p]


def test_r3_dry_run(r3_tree):
    matrix = c.dry_run()
    assert matrix['unit_count'] == 144
    assert [cell['unit_count'] for cell in matrix['cells']] == [12] * 6 + [24] * 3
    assert matrix['planning_estimate_s'] is None
    assert matrix['rails'] == {'worker_s': 285, 'work_alarm_s': 280, 'outer_s': 590, 'cooldown_s': 30, 'lock_wait_s': 250}
    assert not list((c.OUT / 'claims/r3').glob('*.json'))


@pytest.mark.parametrize('wall,nonfit', [(33.5, False), (33.55, True), (40, True)])
def test_r3_budget(r3_tree, wall, nonfit):
    simulated(u.units()[0], wall=wall)
    current = u.state()
    assert current['planning_estimate_s'] == wall
    assert current['projected_total_s'] == 570 + 144 * wall
    assert current['budget_nonfit'] == nonfit
    if nonfit:
        assert current['campaign_verdict'] == 'NON_FIT_BUDGET'
        with patch.object(u.subprocess, 'run', side_effect=AssertionError('must not launch')):
            assert u.controller('oracle_m1_s0')['status'] == 'NON_FIT'
    else:
        assert u.next_unit()[1] == 'READY'


def test_r3_estimate_is_first_completion_and_red_retry_not_double_counted(r3_tree):
    first, second = u.units()[:2]
    simulated(second, wall=10)
    simulated(first, 'RED', rows=sample(first)[:1], wall=5)
    simulated(first, attempt=2, wall=12)
    current = u.state()
    assert current['planning_estimate_s'] == 10
    assert current['rows'] == 12  # selected latest attempt, RED row retained in attempts
    assert current['r3_worker_wall_s'] == 27
    assert current['failed'] and not current['oracle_positive']


@pytest.mark.parametrize('attack', ['source', 'order', 'history', 'manifest', 'commands'])
def test_r3_bindings(r3_tree, attack):
    out, reg, sources = r3_tree
    if attack == 'source':
        sources['scripts/grm_x1_units.py'] = '0' * 64
    elif attack == 'order':
        (c.ROOT / 'orders/GRM_X1_AMENDMENT_3.md').write_text('forged')
    elif attack == 'history':
        (c.ROOT / next(iter(reg['historical_files']))).write_text('forged')
    elif attack == 'commands':
        (out / 'lead_commands.txt').write_text('forged')
    else:
        value = c.read(out / 'continuation_03.json')
        value['budget']['total_s'] += 285
        (out / 'continuation_03.json').write_bytes(raw_json(value))
        (out / 'continuation_03.sha256').write_text(sha(out / 'continuation_03.json'))
    with pytest.raises(ValueError, match='binding|drift'):
        u.validate()


@pytest.mark.parametrize('timeout', [False, True])
def test_r3_worker_cpu(r3_tree, monkeypatch, timeout):
    unit = u.units()[0]
    calls = []
    def fake_execute(selected, rows, directory):
        calls.append(selected)
        create(directory / 'captured.json', b'{}')
        rows.extend(sample(selected)[:1] if timeout else sample(selected))
        if timeout:
            raise TimeoutError('synthetic CPU rail')
    monkeypatch.setattr(u, 'lease_check', lambda: None)
    monkeypatch.setattr(u, 'execute', fake_execute)
    monkeypatch.setattr(u.signal, 'signal', lambda *a: None)
    alarms = []
    monkeypatch.setattr(u.signal, 'setitimer', lambda *a: alarms.append(a))
    code = u.worker(unit['cell'], u.fingerprint(), unit['unit_id'], 1)
    assert code == int(timeout) and calls == [unit]
    assert 0 < alarms[0][1] <= 280 and alarms[-1][1] == 0
    current = u.state()
    receipt = next(iter(current['receipts'].values()))
    assert receipt['session_digests']['captured.json'] == u.hashlib.sha256(b'{}').hexdigest()
    assert receipt['status'] == ('RED' if timeout else 'COMPLETE')
    assert receipt['fit_status'] == ('NON_FIT' if timeout else 'FIT')
    assert u.next_unit(unit['cell'])[0] == u.units()[1]


def test_r3_run_unit_preserves_capture_and_global_order(monkeypatch, tmp_path):
    from scripts import grm_x1_gpu as gpu
    selected = u.units()[1]
    calls = []
    monkeypatch.setattr(gpu, 'run_cell', lambda *args: calls.append(args))
    rows = []
    gpu.run_unit(selected, rows, tmp_path)
    assert calls == [(selected, rows, tmp_path)]
    assert selected['query_ordinal'] == 2
    with pytest.raises(ValueError, match='exactly one query'):
        gpu.run_unit({**selected, 'query_ids': []}, rows, tmp_path)


def test_r3_lead_loop(r3_tree, monkeypatch):
    calls = []
    def launch(args, **kwargs):
        assert args[:5] == ['flock', '--wait', '250', '--no-fork', '/tmp/forge-gpu.lock']
        assert 'timeout' not in kwargs
        uid = args[args.index('--unit') + 1]
        unit = next(x for x in u.units() if x['unit_id'] == uid)
        calls.append(uid)
        simulated(unit, 'RED' if len(calls) == 1 else 'COMPLETE', rows=[] if len(calls) == 1 else None)
        return type('Result', (), {'returncode': 1 if len(calls) == 1 else 0})()
    monkeypatch.setattr(u.subprocess, 'run', launch)
    with patch.object(u.time, 'sleep') as cooldown:
        for i in range(12):
            result = u.controller('oracle_m1_s0')
            assert result['status'] == ('UNIT_RED' if i == 0 else 'CELL_COMPLETE' if i == 11 else 'UNIT_COMPLETE')
        assert u.controller('oracle_m1_s0')['status'] == 'CELL_COMPLETE'
        assert cooldown.call_count == 12
        assert all(call.args == (30,) for call in cooldown.call_args_list)
    assert calls == [x['unit_id'] for x in u.units()[:12]]
    assert u.state()['cells']['oracle_m1_s0']['red_units'] == 1
    commands = (c.OUT / 'lead_commands.txt').read_text()
    assert 'while true; do' in commands and 'CELL_COMPLETE|NON_FIT) break' in commands
    assert 'NON_FIT_BUDGET' in commands and 'run "$cell"' in commands
