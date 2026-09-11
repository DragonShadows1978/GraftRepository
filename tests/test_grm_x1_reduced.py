"""Author CPU gates only. Prior art: house X1 (2026) synthetic lifecycle tests.
New: reduced populations, immutable adoption and diagnostic-only labels.
"""
import copy
import json
import shutil
import subprocess
from unittest.mock import patch

import pytest

from scripts import grm_x1_campaign as c
from scripts import grm_x1_units as u
from scripts import grm_x1_reduced as r
from scripts.grm_x1_register import create, raw_json, sha
from test_grm_x1_units import sample


@pytest.fixture
def r4_tree(tmp_path, monkeypatch):
    root, original_out = c.ROOT, c.OUT
    sources = c.source_manifest()
    out = tmp_path / 'artifacts/grm_x1'
    shutil.copytree(original_out, out)
    reg = c.read(out / 'continuation_04_registration.json')
    for name in [*c.read(out / 'continuation_03.json')['bindings'],
                 *c.read(out / 'registration.json')['source_shas'],
                 'orders/GRM_X1_AMENDMENT_4.md', 'scripts/grm_x1_campaign.py']:
        target = tmp_path / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'OUT', out)
    monkeypatch.setattr(c, 'FIX', out / 'fixtures')
    monkeypatch.setattr(c, 'source_manifest', lambda: sources.copy())
    monkeypatch.setattr(c, 'dependency_inventory', lambda: c.read(out / 'handoff_manifest.json')['dependencies'])
    # Real immutable fixture checks remain active.
    return out, reg, sources


def simulated(unit, status='COMPLETE', attempt=1, wall=10, rows=None, claim_only=False, non_fit=False):
    fp = sha(u.continuation_path())
    create(c.OUT / 'claims/r4' / f'{u.key(unit, attempt)}_{fp}.json', raw_json({
        'cell': unit['cell'], 'unit_id': unit['unit_id'], 'attempt': attempt,
        'fingerprint': fp, 'reserved_gpu_s': 285}))
    if claim_only:
        return
    data = sample(unit) if rows is None else rows
    data = [{**row, 'answer': row.get('answer', 'CPU synthetic answer')} for row in data]
    return u.record(unit, attempt, fp, status, data, wall,
                    {'type': 'SyntheticError'} if status == 'RED' else None, non_fit)


@pytest.mark.campaign_receipt(registration='artifacts/grm_x1/continuation_04_registration.json + artifacts/grm_x1/claims/r4/')
def test_r4_continuation_accepted(r4_tree):
    out, reg, _ = r4_tree
    assert u.validate()['reuse']['status'] == 'VALID_BYTE_IDENTICAL'
    assert u.fingerprint() == sha(out / 'continuation_04_source_amendment_01.json')
    current = u.state()
    assert current['active_epoch'] == 'r4' and current['campaign_verdict'] == 'PENDING'
    assert current['charged_gpu_s'] == 570 + reg['reuse']['worker_wall_s']
    assert current['r4_worker_wall_s'] == 0 and current['rows'] == 6
    assert current['reservation_cap_s'] == 10800 and not current['budget_nonfit']
    assert len(current['historical_red']) == 2
    with pytest.raises(FileExistsError, match='immutable'):
        r.seal()


@pytest.mark.parametrize('field', ['budget', 'cells', 'reuse', 'commands', 'order', 'checksum', 'registration', 'source_seal'])
def test_r4_forged_continuation(r4_tree, field):
    out, reg, _ = r4_tree
    path = out / 'continuation_04.json'
    value = c.read(path)
    if field == 'budget':
        value['budget']['total_s'] = 99999
    elif field == 'cells':
        value['cells'].append('oracle_m10_s0')
    elif field == 'reuse':
        value['reuse']['worker_wall_s'] = 0
    elif field == 'commands':
        (out / 'lead_commands.txt').write_text('forged')
    elif field == 'order':
        (c.ROOT / 'orders/GRM_X1_AMENDMENT_4.md').write_text('forged')
    elif field == 'source_seal':
        supplement = out / 'continuation_04_source_amendment_01.json'
        changed = c.read(supplement)
        changed['sources']['scripts/grm_x1_units.py'] = '0' * 64
        supplement.write_bytes(raw_json(changed))
        (out / 'continuation_04_source_amendment_01.sha256').write_text(sha(supplement))
    elif field == 'registration':
        (out / 'continuation_04_registration.json').write_text('{}')
    path.write_bytes(raw_json(value))
    (out / 'continuation_04.sha256').write_text('0' * 64 if field == 'checksum' else sha(path))
    with pytest.raises(ValueError, match='binding|mismatch|drift'):
        u.validate()


@pytest.mark.parametrize('source', ['scripts/grm_x1_units.py', 'scripts/grm_x1_gpu.py', 'core/grm_x1_addresses.py'])
def test_r4_stale_continuation(r4_tree, source):
    r4_tree[2][source] = '0' * 64
    with pytest.raises(ValueError, match='binding|drift'):
        u.validate()


@pytest.mark.campaign_receipt(registration='artifacts/grm_x1/continuation_04_registration.json + artifacts/grm_x1/claims/r4/')
def test_r4_historical_bytes(r4_tree):
    _, reg, _ = r4_tree
    before = {p: (c.ROOT / p).read_bytes() for p in reg['historical_files']}
    simulated(u.units()[1])
    u.state()
    for path, original in before.items():
        assert (c.ROOT / path).read_bytes() == original
        assert sha(c.ROOT / path) == reg['historical_files'][path]
    historical = c.ROOT / reg['reuse']['receipt_path']
    historical.write_text('{}')
    with pytest.raises(ValueError, match='historical'):
        u.validate()


@pytest.mark.campaign_receipt(registration='artifacts/grm_x1/continuation_04_registration.json + artifacts/grm_x1/claims/r4/')
def test_r4_dry_run_72(r4_tree):
    dry = c.dry_run()
    assert dry['unit_count'] == len(u.units()) == 72
    assert [x['cell'] for x in dry['cells']] == ['oracle_m1_s0','oracle_m100_s0','natural_m1','natural_m100']
    assert [x['unit_count'] for x in dry['cells']] == [12,12,24,24]
    assert [x['query_ordinal'] for x in u.units()[:12]] == list(range(0,24,2))
    assert sum(x['turns'] for x in dry['cells']) == 240
    assert sum(x['units'] for x in dry['not_run']) == 72
    assert all(x['status'] == 'NOT RUN' for x in dry['not_run'])
    assert sum(x['additional_estimated_gpu_s'] for x in dry['not_run']) == pytest.approx(72*dry['planning_estimate_s'])
    assert not list((c.OUT/'claims/r4').glob('*.json'))


@pytest.mark.campaign_receipt(registration='artifacts/grm_x1/continuation_04_registration.json + artifacts/grm_x1/claims/r4/')
def test_r4_reuse_identical_unit(r4_tree):
    out, reg, _ = r4_tree
    old = next(x for x in c.read(out/'continuation_03.json')['units'] if x['unit_id'] == reg['reuse']['unit_id'])
    assert raw_json(old) == raw_json(u.units()[0])
    assert u.rows_digest(old) == reg['reuse']['unit_definition_sha256']
    assert u.next_unit()[0] == u.units()[1]
    with pytest.raises(ValueError, match='retry requires'):
        u.next_unit(retry_unit=old['unit_id'])
    simulated(old)
    with pytest.raises(ValueError, match='duplicate unit evidence'):
        u.state()


@pytest.mark.campaign_receipt(registration='artifacts/grm_x1/continuation_04_registration.json + artifacts/grm_x1/claims/r4/')
def test_r4_unit_definition_drift(r4_tree, monkeypatch):
    original = u.units
    def altered(full=False):
        layout = original(full=full)
        layout[0]['query_ordinal'] += 1
        return layout
    monkeypatch.setattr(u, 'units', altered)
    with pytest.raises(ValueError, match='byte-identical'):
        u.validate()


@pytest.mark.campaign_receipt(registration='artifacts/grm_x1/continuation_04_registration.json + artifacts/grm_x1/claims/r4/')
def test_r4_create_only(r4_tree):
    unit = u.units()[1]
    path = simulated(unit)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        simulated(unit)
    assert path.read_bytes() == before
    value = c.read(path)
    assert all('served_text_class' in row for row in value['rows'])
    value['rows'][0]['served_text_class'] = 'forged'
    value['rows_sha256'] = u.rows_digest(value['rows'])
    value['content_sha256'] = u.rows_digest({k:v for k,v in value.items() if k!='content_sha256'})
    path.write_bytes(raw_json(value))
    with pytest.raises(ValueError, match='served text class'):
        u.state()


@pytest.mark.campaign_receipt(registration='artifacts/grm_x1/continuation_04_registration.json + artifacts/grm_x1/claims/r4/')
def test_r4_aggregate_sum(r4_tree):
    reg = r4_tree[1]
    all_rows = list(c.read(c.ROOT/reg['reuse']['receipt_path'])['rows'])
    for unit in u.units()[1:]:
        path = simulated(unit)
        all_rows.extend(c.read(path)['rows'])
    current = u.state()
    reference = c.aggregate(all_rows, cells=u.active_cells())
    for field in ('table','source_strata','query_strata','predictions','kill','oracle_positive'):
        assert current[field] == reference[field]
    assert current['rows'] == 240 and sum(x['rows'] for x in current['cells'].values()) == 240
    assert current['r4_worker_wall_s'] == 710
    assert current['charged_gpu_s'] == 570 + 710 + reg['reuse']['worker_wall_s']
    assert sum(current['served_text_counts'].values()) == 240
    assert set(current['cell_status'].values()) == {'COMPLETE'}


@pytest.mark.campaign_receipt(registration='artifacts/grm_x1/continuation_04_registration.json + artifacts/grm_x1/claims/r4/')
def test_r4_resume_next_missing(r4_tree):
    simulated(u.units()[2])
    assert u.next_unit()[0] == u.units()[1]
    simulated(u.units()[1])
    assert u.next_unit()[0] == u.units()[3]
    assert u.next_unit('natural_m1')[1] == 'NATURAL_LOCKED'
    with pytest.raises(ValueError, match='unregistered cell'):
        u.next_unit('oracle_m10_s0')


@pytest.mark.campaign_receipt(registration='artifacts/grm_x1/continuation_04_registration.json + artifacts/grm_x1/claims/r4/')
def test_r4_red_advances_second_red_stops(r4_tree):
    unit = u.units()[1]
    simulated(unit, 'RED', rows=[])
    assert u.next_unit()[0] == u.units()[2]
    assert u.next_unit(retry_unit=unit['unit_id'])[2] == 2
    simulated(unit, 'RED', attempt=2, rows=[])
    assert u.next_unit()[1] == 'STOP_SECOND_RED_OR_INCOMPLETE'
    assert not u.state()['oracle_positive']


@pytest.mark.campaign_receipt(registration='artifacts/grm_x1/continuation_04_registration.json + artifacts/grm_x1/claims/r4/')
@pytest.mark.parametrize('scenario', ['projection','admission','incomplete','timeout'])
def test_r4_budget(r4_tree, monkeypatch, scenario):
    current = u.state()
    assert current['projected_total_s'] == pytest.approx(570 + 72 * 106.25843796180561)
    unit = u.units()[1]
    if scenario == 'projection':
        for selected in u.units()[1:18]:
            simulated(selected, wall=279)
        assert u.state()['budget_nonfit']
        with patch.object(u.subprocess, 'run', side_effect=AssertionError('must not launch')):
            assert u.controller()['reason'] == 'NON_FIT_BUDGET'
    elif scenario == 'admission':
        current['charged_gpu_s'] = 10800 - 285
        monkeypatch.setattr(u,'state',lambda:current)
        assert u.next_unit()[1] == 'READY'
        current['charged_gpu_s'] += .001
        assert u.next_unit()[1] == 'NON_FIT_BUDGET'
    elif scenario == 'incomplete':
        simulated(unit, claim_only=True)
        assert u.state()['charged_gpu_s'] == current['charged_gpu_s'] + 285
        assert u.next_unit()[1] == 'STOP_SECOND_RED_OR_INCOMPLETE'
    else:
        simulated(unit, 'RED', rows=[], wall=280.01, non_fit=True)
        assert u.next_unit()[0] == u.units()[2]
        with pytest.raises(ValueError, match='non-timeout'):
            u.next_unit(retry_unit=unit['unit_id'])


@pytest.mark.campaign_receipt(registration='artifacts/grm_x1/continuation_04_registration.json + artifacts/grm_x1/claims/r4/')
@pytest.mark.parametrize('failure', [None,'minimum','spread','gain','coverage','fault','retains','incomplete','duplicate','natural_missing'])
def test_r4_natural_gate(r4_tree, monkeypatch, failure):
    rows = [row for unit in u.units()[:24] for row in sample(unit)]
    if failure == 'minimum':
        for row in rows:
            if row['arm']=='B': row['exact_answer']=False
    elif failure in ('spread','retains'):
        arm = 'B' if failure == 'spread' else 'C'
        next(row for row in rows if row['arm']==arm and row['condition']=='present')['exact_answer']=False
    elif failure == 'gain':
        for row in rows:
            if row['arm']=='A': row['exact_answer']=True
    elif failure == 'coverage':
        next(row for row in rows if row['arm']=='B')['address_coverage']=False
    elif failure == 'fault':
        next(row for row in rows if row['arm']=='C' and row['condition']=='absent')['false_answer']=True
    elif failure == 'incomplete': rows.pop()
    elif failure == 'duplicate': rows.append(rows[0])
    result = c.aggregate(rows, cells=u.active_cells())
    assert result['oracle_positive'] == (failure in (None,'natural_missing'))
    current = u.state()
    current['oracle_positive'] = result['oracle_positive']
    monkeypatch.setattr(u,'state',lambda:current)
    assert u.next_unit('natural_m1')[1] == ('READY' if result['oracle_positive'] else 'NATURAL_LOCKED')
    if failure == 'natural_missing':
        natural = [row for unit in u.units()[24:] for row in sample(unit)]
        assert c.aggregate(rows+natural[:-1],cells=u.active_cells())['predictions']['S5'] is None
        assert c.aggregate(rows+natural,cells=u.active_cells())['predictions']['S5'] is True


@pytest.mark.campaign_receipt(registration='artifacts/grm_x1/continuation_04_registration.json + artifacts/grm_x1/claims/r4/')
@pytest.mark.parametrize('answer,abstained,label', [
    ("I’m sorry, but I can’t help with that.",False,'refusal-style'),
    ("I'm sorry, but I can't help with that",False,'refusal-style'),
    ('I cannot assist with your request.',False,'refusal-style'),
    ('I cannot answer',True,'abstention'),
    ('I cannot answer from the mounted memory: PAGE_FAULT.',True,'abstention'),
    ('cobalt-3-comet',False,'answer'),
    ('I cannot help with that. cobalt-3-comet',False,'answer'),
    ('The answer contains the words I cannot help with that.',False,'answer')])
def test_r4_served_text_classes(r4_tree, answer, abstained, label):
    row = {'answer':answer,'abstained':abstained,'false_answer':not abstained,'exact_answer':False}
    before = copy.deepcopy(row)
    annotated = r.classify_rows([row])[0]
    assert annotated.pop('served_text_class') == label and annotated == before and row == before
    actual = u.state()
    assert actual['served_text_counts'] == {'answer':2,'abstention':1,'refusal-style':3}
    assert actual['table']['A/1/present']['false_rate'] == 1


@pytest.mark.campaign_receipt(registration='artifacts/grm_x1/continuation_04_registration.json + artifacts/grm_x1/claims/r4/')
@pytest.mark.parametrize('timeout',[False,True])
def test_r4_worker_cpu(r4_tree, monkeypatch, timeout):
    unit = u.units()[1]
    def execute(selected,rows,directory):
        assert selected == unit
        create(directory/'captured.json',b'{}')
        rows.extend({**row,'answer':'CPU synthetic answer'} for row in sample(unit)[:1 if timeout else 6])
        if timeout: raise TimeoutError('synthetic CPU rail')
    monkeypatch.setattr(u,'lease_check',lambda:None)
    monkeypatch.setattr(u,'execute',execute)
    monkeypatch.setattr(u.signal,'signal',lambda *a:None)
    monkeypatch.setattr(u.signal,'setitimer',lambda *a:None)
    assert u.worker(unit['cell'],u.fingerprint(),unit['unit_id'],1) == int(timeout)
    receipt = u.state()['units'][unit['unit_id']]['attempts'][0]
    assert receipt['schema'] == 'grm.x1.gpu-unit.v4'
    assert all('served_text_class' in row for row in receipt['rows'])
    assert receipt['session_digests']['captured.json'] == u.hashlib.sha256(b'{}').hexdigest()
    assert u.next_unit()[0] == u.units()[2]


@pytest.mark.parametrize('scenario,expected_calls,exit_code', [('normal',71,0),('budget',1,2),('locked',24,0),('stop',1,1)])
def test_r4_lead_loop(r4_tree, tmp_path, scenario, expected_calls, exit_code):
    # House X1 r3 loop protocol (2026), CPU stub exercises real shell parser.
    stubroot = tmp_path/'stub'
    (stubroot/'scripts').mkdir(parents=True)
    commands = (c.OUT/'lead_commands.txt').read_text().replace('cd /mnt/ForgeRealm/wt/grm-x1',f'cd {stubroot}')
    loop = stubroot/'loop.sh'; loop.write_text(commands)
    stub = stubroot/'scripts/grm_x1_lead_gpu.sh'
    stub.write_text('''#!/usr/bin/env bash
exec python3 stub.py "$@"
''')
    (stubroot/'stub.py').write_text('''import json,sys,pathlib
p=pathlib.Path('calls.json'); calls=json.loads(p.read_text()) if p.exists() else []
if sys.argv[1]!='run': print('{}');sys.exit(0)
cell=sys.argv[2];calls.append(cell);p.write_text(json.dumps(calls))
scenario=SCENARIO
status='UNIT_RED' if len(calls)==1 else 'UNIT_COMPLETE'; reason='PENDING'
if calls.count(cell)==(11 if cell=='oracle_m1_s0' else 12 if cell.startswith('oracle') else 24): status='CELL_COMPLETE'
if scenario=='budget':status='NON_FIT';reason='NON_FIT_BUDGET'
if scenario=='locked' and cell.startswith('natural'):status='NATURAL_LOCKED'
if scenario=='stop':status='STOP_SECOND_RED_OR_INCOMPLETE'
print(json.dumps({'worker_progress':'CPU stub'}));print(json.dumps({'status':status,'reason':reason},indent=2))
'''.replace('SCENARIO',repr(scenario)))
    completed = subprocess.run(['bash',str(loop)],capture_output=True,text=True)
    assert completed.returncode == exit_code, completed.stderr
    calls = json.loads((stubroot/'calls.json').read_text())
    assert len(calls)==expected_calls
    assert set(calls) <= {'oracle_m1_s0','oracle_m100_s0','natural_m1','natural_m100'}
