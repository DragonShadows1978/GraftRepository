"""CPU-only author gates for lead A2; no blind verification/GPU claim.
Prior art: house C4/EB1/DET1 (2026) CPU fake lease and persisted session tests.
New cases encode the lead order's exact recovery and cell-boundary conditions.
"""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import shlex
import shutil
from types import SimpleNamespace

import pytest
from scripts import grm_c4_campaign as c
from scripts import grm_c4_resume_a2 as a2


def persist(session, stop, sentinel='PASS', start=0):
    session.mkdir(parents=True, exist_ok=True)
    for name in ('instrumentation.jsonl', 'transcript.jsonl'):
        (session/name).write_text(''.join(json.dumps({'turn': t})+'\n' for t in range(start, stop)))
    (session/'restart.json').write_text(json.dumps({'after_turn': stop-1}))
    (session/'repository').mkdir(exist_ok=True)
    (session/'repository/state').write_text(sentinel)
    (session/'probe_scorecard.json').write_text('{"probes": []}')


def bind_state(session):
    return [c.record(p) for p in sorted(session.rglob('*')) if p.is_file()]


@pytest.fixture
def layout(tmp_path, monkeypatch):
    # Register three two-turn segments before exercising the production worker.
    reg = c.binding()
    reg = {**reg, 'legacy_sources': reg['sources'], 'legacy_amendment': reg['amendment'],
        'sources': [], 'amendment': {'path': 'synthetic-a4', 'sha256': 'cpu-only', 'bytes': 0},
        'units_per_new_cell': [{'battery': 'longhorizon', 'spec': f's{i}',
            'stop_after_turns': 2*i, 'resume': f's{i-1}' if i > 1 else None} for i in range(1,4)]}
    monkeypatch.setattr(c, 'OUT', tmp_path/'original')
    monkeypatch.setattr(a2, 'OUT', tmp_path/'amended')
    monkeypatch.setattr(a2, 'binding', lambda: reg)
    cell = 'c64_w96'
    def receipt(spec, status='PASS', error=None, serving=True):
        stop = next(u['stop_after_turns'] for u in a2.lh_units(reg) if u['spec'] == spec)
        session = c.OUT/'runs'/cell/'longhorizon'/spec/'session'
        persist(session, stop, 'RED-DO-NOT-RESUME' if status == 'RED' else 'PASS')
        row = {'cell':cell, 'battery':'longhorizon', 'spec':spec, 'status':status,
            'error':error, 'registration':c.record(c.REG), 'sources':reg['legacy_sources'],
            'amendment':reg['legacy_amendment'], 'events':[{'event':'geometry'}] + ([{'event':'attempt_residency'}] if serving else []),
            'finished_unix':0, 'gpu_seconds':7,
            'result': {'session_dir':str(session), 'stop_after_turns':stop, 'resumed':spec != 's1'},
            'session_state':bind_state(session)}
        c.write_once(a2.receipt_path(cell, 'longhorizon', spec), row)
        return row
    receipt('s1')
    receipt('s2', 'RED', a2.ROOT_ERROR, serving=False)
    from scripts import grm_cmc1_gpu_arms as leases
    @contextmanager
    def lease(seconds, wait):
        assert (seconds, wait) == (285,0)
        yield
    monkeypatch.setattr(leases, 'gpu_lease', lease)
    # Avoid cooldown waits while retaining real accounting's assertions.
    monkeypatch.setattr(a2.time, 'time', lambda: 1000)
    controls = {'serving': True, 'turn_delta': 0, 'fail': False}
    @contextmanager
    def harness(chunk, width, run_root, events):
        from scripts import grm_e2e_session as e2e
        def run_turn(repo, event, turn_idx, **kwargs):
            return None
        def run_shard(spec, *, run_dir):
            index = int(spec[1:])-1
            dest = run_dir/spec/'session'
            source = run_dir/f's{index}'/'session'
            assert (source/'repository/state').read_text() == 'PASS'
            shutil.copytree(source, dest)
            if controls['fail']:
                raise TimeoutError('synthetic rail')
            stop = 2*(index+1)
            for turn in range(2*index, stop + controls['turn_delta']):
                e2e.run_turn(None, {}, turn)
            persist(dest, stop)
            events.append({'event':'geometry'})
            if spec == 's3' and controls['serving']:
                events.append({'event':'attempt_residency'})
            return {'spec':spec, 'session_dir':str(dest), 'stop_after_turns':stop, 'resumed':True}
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(e2e, 'run_turn', run_turn)
            yield None, None, SimpleNamespace(run_shard=run_shard)
    monkeypatch.setattr(c, 'harness', harness)
    return SimpleNamespace(reg=reg, cell=cell, receipt=receipt, controls=controls)


def rewrite_original(layout, spec, **changes):
    path = a2.receipt_path(layout.cell, 'longhorizon', spec)
    row = c.read(path)
    row.update(changes)
    path.write_text(json.dumps(row))


def test_three_segments_probe_free_middle_passes_original_bytes_unchanged(layout, monkeypatch):
    before = {p:p.read_bytes() for p in c.OUT.rglob('*') if p.is_file()}
    a2.worker(layout.cell, 's2')
    path = a2.receipt_path(layout.cell, 'longhorizon', 's2', amended=True)
    row = c.read(path)
    assert row['status'] == 'PASS'
    assert row['amendment'] == layout.reg['amendment']
    assert row['turn_range']['completed'] == [2,3]
    assert 'cell_serving_guard' not in row
    assert row['original_receipt'] == c.record(a2.receipt_path(layout.cell, 'longhorizon', 's2'))
    assert row['prior_receipt'] == c.record(a2.receipt_path(layout.cell, 'longhorizon', 's1'))
    assert row['campaign_gpu_seconds_before'] == 14  # original RED time counted
    monkeypatch.setattr(a2.time, 'time', lambda: 2000)
    a2.worker(layout.cell, 's3')
    final = c.read(a2.receipt_path(layout.cell, 'longhorizon', 's3', amended=True))
    assert final['status'] == 'PASS' and final['cell_serving_guard']['attempt_residency_count'] == 2
    assert final['prior_receipt'] == c.record(path)
    assert all(p.read_bytes() == data for p,data in before.items())
    with pytest.raises(AssertionError, match='used once'):
        a2.worker(layout.cell, 's2')
    with pytest.raises(AssertionError, match='used once'):
        a2.worker(layout.cell, 's3')


def test_cell_with_zero_serving_overall_reds_only_on_final(layout, monkeypatch):
    rewrite_original(layout, 's1', events=[{'event':'geometry'}])
    layout.controls['serving'] = False
    a2.worker(layout.cell, 's2')
    assert c.read(a2.receipt_path(layout.cell,'longhorizon','s2',amended=True))['status'] == 'PASS'
    monkeypatch.setattr(a2.time, 'time', lambda: 2000)
    with pytest.raises(AssertionError, match='No serving observed'):
        a2.worker(layout.cell, 's3')
    row = c.read(a2.receipt_path(layout.cell,'longhorizon','s3',amended=True))
    assert row['error'] == a2.ROOT_ERROR and row['cell_serving_guard']['attempt_residency_count'] == 0
    with pytest.raises(AssertionError, match='used once'):
        a2.worker(layout.cell, 's3')
    with pytest.raises(AssertionError, match='Other RED'):
        a2.accounting(layout.reg)


@pytest.mark.parametrize('error', ['No serving observed', 'AssertionError: No serving observed ', 'TimeoutError: No serving observed', 'AssertionError: No arena observed', a2.DEPENDENCY_ERROR])
def test_root_error_requires_exact_stored_string(layout, error):
    rewrite_original(layout, 's2', error=error)
    with pytest.raises(AssertionError, match='Other RED'):
        a2.recovery_ready(layout.reg, layout.cell, 's2')


def test_dependency_error_requires_root_and_transitive_suffix(layout):
    layout.receipt('s3', 'RED', a2.DEPENDENCY_ERROR, serving=False)
    assert a2.eligible_suffix(layout.reg, layout.cell) == ['s2','s3']
    with pytest.raises(AssertionError, match='Prior unit missing/RED'):
        a2.worker(layout.cell, 's3')
    assert not a2.receipt_path(layout.cell,'longhorizon','s3',amended=True).exists()


def test_unrelated_dependent_red_is_stop(layout):
    layout.receipt('s3', 'RED', 'TimeoutError: synthetic rail')
    with pytest.raises(AssertionError, match='Other RED'):
        a2.worker(layout.cell, 's2')


def test_existing_pass_and_unevidenced_cell_not_rerunnable(layout):
    with pytest.raises(AssertionError, match='not re-runnable'):
        a2.worker(layout.cell, 's1')
    assert a2.eligible_suffix(layout.reg, 'c96_w64') == []
    with pytest.raises(AssertionError, match='not re-runnable'):
        a2.worker('c96_w64', 's2')


@pytest.mark.parametrize('amended', [False, True])
def test_orphan_claim_blocks_campaign(layout, amended):
    path = a2.receipt_path('c96_w64','longhorizon','s1',amended=amended)
    c.write_once(path.with_suffix('.attempt.json'), {'started_unix':1})
    with pytest.raises(AssertionError, match='Unfinished worker claim'):
        a2.worker(layout.cell, 's2')
    assert not a2.receipt_path(layout.cell,'longhorizon','s2',amended=True).exists()


def test_one_attempt_even_after_failure_or_abandonment(layout):
    layout.controls['fail'] = True
    with pytest.raises(TimeoutError, match='synthetic rail'):
        a2.worker(layout.cell, 's2')
    path = a2.receipt_path(layout.cell,'longhorizon','s2',amended=True)
    raw = path.read_bytes()
    with pytest.raises(AssertionError, match='used once'):
        a2.worker(layout.cell, 's2')
    assert path.read_bytes() == raw
    path.unlink()  # fixture-only model of an outer-killed worker: claim remains
    with pytest.raises(AssertionError, match='used once'):
        a2.worker(layout.cell, 's2')


@pytest.mark.parametrize('delta', [-1, 1])
def test_wrong_turn_advancement_reds_probe_free_segment(layout, delta):
    layout.controls['turn_delta'] = delta
    with pytest.raises(AssertionError, match='Registered turn range mismatch'):
        a2.worker(layout.cell, 's2')


@pytest.mark.parametrize('corruption', ['missing', 'changed', 'unlisted'])
def test_resume_bound_state_must_be_complete_and_unchanged(layout, corruption):
    path,row = a2.pass_receipt(layout.reg,layout.cell,'longhorizon','s1')
    session = Path(row['result']['session_dir'])
    if corruption == 'missing':
        rewrite_original(layout,'s1',session_state=[])
    elif corruption == 'changed':
        (session/'repository/state').write_text('drift')
    else:
        (session/'unexpected').write_text('unlisted')
    with pytest.raises(AssertionError, match='bound session state|Resumable state drift'):
        a2.worker(layout.cell,'s2')


def test_saved_boundary_and_turn_range_are_checked(layout):
    _,row = a2.pass_receipt(layout.reg,layout.cell,'longhorizon','s1')
    session=Path(row['result']['session_dir'])
    persist(session,1)
    rewrite_original(layout,'s1',session_state=bind_state(session))
    with pytest.raises(AssertionError, match='Registered turn range mismatch'):
        a2.worker(layout.cell,'s2')


def test_source_and_amendment_compatibility_is_exact(layout):
    rewrite_original(layout,'s1',sources=[])
    with pytest.raises(AssertionError, match='Mixed session source'):
        a2.worker(layout.cell,'s2')
    rewrite_original(layout,'s1',sources=layout.reg['legacy_sources'],amendment=layout.reg['amendment'])
    with pytest.raises(AssertionError, match='Mixed session amendment'):
        a2.worker(layout.cell,'s2')


def test_budget_and_cooldown_unchanged(layout, monkeypatch):
    rewrite_original(layout,'s2',gpu_seconds=4510)
    with pytest.raises(AssertionError, match='GPU budget non-fit'):
        a2.worker(layout.cell,'s2')
    rewrite_original(layout,'s2',gpu_seconds=7,finished_unix=990)
    with pytest.raises(AssertionError, match='30s cooldown'):
        a2.worker(layout.cell,'s2')


def test_command_file_only_eligible_suffixes_and_cell_scores():
    reg=a2.binding()
    lines=(c.OUT/'lead_commands_a2.txt').read_text().splitlines()
    assert 'set -e' in lines
    found=[]
    for i,line in enumerate(lines):
        if ' worker ' in line:
            args=shlex.split(line)
            assert args[:3] == ['timeout','--signal=KILL','590s']
            assert lines[i+1] == 'sleep 30'
            found.append((args[args.index('--cell')+1],args[args.index('--spec')+1]))
    a=c.read(a2.AMENDMENT)
    assert found == [(cell,spec) for cell in c.executable_cells(reg) for spec in a['eligible_suffixes_at_registration'][cell]]
    for cell in c.executable_cells(reg):
        if a['eligible_suffixes_at_registration'][cell]:
            index=lines.index(f'python scripts/grm_c4_resume_a2.py score --cell {cell}')
            assert lines[index-2].endswith(f"--spec {a['eligible_suffixes_at_registration'][cell][-1]}")
    assert not any(' --battery sup ' in line or ' --battery census ' in line for line in lines)


def test_original_binding_and_receipts_byte_unchanged():
    reg=c.binding()
    assert reg['amendment'] == c.record(c.OUT/'amendment_a3.json')
    a2.binding()
    before=c.read(a2.OUT/'original_receipts_before.json')
    assert len(before['files']) > 0
    for row in before['files']:
        assert c.record(row['path']) == row


def test_registered_turn_plan_confirms_deposit_only_segment():
    from scripts import grm_eb1_longhorizon_gpu as lh
    reg=c.binding()
    plan=lh.build_long_script()
    expected=[5,9,13,16,19,22,24,26,30,33,70,80,90,100]
    assert [i for i,t in enumerate(plan) if t['kind']=='probe'] == expected
    assert [p['turn'] for p in reg['fixtures']['longhorizon']['probes']] == expected
    assert [t['kind'] for t in plan[40:48]] == ['fact','supersede']*4
    assert c.LH_STOPS['c4-lh-05']==40 and c.LH_STOPS['c4-lh-06']==48


def test_a2_binding_refuses_order_source_and_sha_drift(tmp_path, monkeypatch):
    legacy=c.binding()
    monkeypatch.setattr(c,'binding',lambda:legacy)
    order=tmp_path/'order.md'
    order.write_text('order')
    source=tmp_path/'source.py'
    source.write_text('source')
    amendment=tmp_path/'amendment_a4.json'
    c.write_once(amendment,{'order':c.record(order),'registration':c.record(c.REG),
        'previous_amendment':legacy['amendment'],'sources':[c.record(source)]})
    amendment.with_suffix('.sha256').write_text(c.record(amendment)['sha256'])
    monkeypatch.setattr(a2,'AMENDMENT',amendment)
    a2.binding()
    order.write_text('drift')
    with pytest.raises(AssertionError,match='order drift'):
        a2.binding()
    order.write_text('order')
    source.write_text('drift')
    with pytest.raises(AssertionError,match='source drift'):
        a2.binding()
    source.write_text('source')
    with amendment.open('a') as f:
        f.write(' ')
    with pytest.raises(AssertionError,match='amendment SHA mismatch'):
        a2.binding()


def test_serving_only_in_earlier_segment_satisfies_cell(layout, monkeypatch):
    layout.controls['serving']=False
    a2.worker(layout.cell,'s2')
    monkeypatch.setattr(a2.time,'time',lambda:2000)
    a2.worker(layout.cell,'s3')
    row=c.read(a2.receipt_path(layout.cell,'longhorizon','s3',amended=True))
    assert row['status']=='PASS' and row['cell_serving_guard']['attempt_residency_count']==1


def test_score_matches_original_arithmetic_and_preserves_receipts(tmp_path, monkeypatch):
    reg=c.binding()
    monkeypatch.setattr(c,'OUT',tmp_path/'original')
    monkeypatch.setattr(c,'binding',lambda:reg)
    monkeypatch.setattr(a2,'OUT',tmp_path/'amended')
    amended={**reg,'legacy_sources':reg['sources'],'legacy_amendment':reg['amendment'],
        'sources':[], 'amendment':{'path':'synthetic-a4','sha256':'cpu','bytes':0}}
    monkeypatch.setattr(a2,'binding',lambda:amended)
    cell='c64_w96'
    root=c.OUT/'runs'/cell
    events=[{'event':'geometry'}, {'event':'attempt_residency','mounted_token_seats':10}]
    for unit in reg['units_per_new_cell']:
        probes=[{'verdict':{'correct':True}}]*(3 if unit['spec']==c.SUP[0] else 2)
        row={'cell':cell,'battery':unit['battery'],'spec':unit['spec'],
            'status':'PASS','registration':c.record(c.REG),'amendment':reg['amendment'],
            'sources':reg['sources'],'events':events,'result':{'probes':probes}}
        c.write_once(a2.receipt_path(cell,unit['battery'],unit['spec']),row)
    census={'probes':[{'turn_row_found':True,'verdict':{'correct':True}} for _ in range(10)]}
    long={'probes':[{'turn_row_found':True,'correct':True} for _ in range(14)],'measured_count':14,'correct_count':14}
    @contextmanager
    def harness(*args):
        yield None,SimpleNamespace(score_run=lambda *a,**k:copy.deepcopy(census)),SimpleNamespace(score_run=lambda *a,**k:copy.deepcopy(long))
    monkeypatch.setattr(c,'harness',harness)
    monkeypatch.setattr(a2,'harness',harness)
    baseline=c.score(cell)
    # Fixture mutation models the observed original RED and new effective PASS
    # suffix. Real artifacts are never changed by the test.
    for unit in a2.lh_units(reg)[5:]:
        path=a2.receipt_path(cell,'longhorizon',unit['spec'])
        row=c.read(path)
        row.update(sources=amended['sources'],amendment=amended['amendment'])
        session=a2.OUT/'runs'/cell/'longhorizon'/unit['spec']/'session'
        persist(session,unit['stop_after_turns'])
        row['result'].update(session_dir=str(session),stop_after_turns=unit['stop_after_turns'])
        row['session_state']=bind_state(session)
        if unit['spec']=='c4-lh-13':
            row['cell_serving_guard']={'attempt_residency_count':13}
        c.write_once(a2.receipt_path(cell,'longhorizon',unit['spec'],amended=True),row)
        if unit['spec']=='c4-lh-06':
            original=c.read(path)
            original.update(status='RED',error=a2.ROOT_ERROR)
            path.write_text(json.dumps(original))
        else:
            path.unlink()
    before={p:p.read_bytes() for p in c.OUT.rglob('*') if p.is_file()}
    result=a2.score(cell)
    for key in ('scores','counts','measurements','sup_probes','census_probes','longhorizon_probes','prediction_met_in_cell','factorial_verdict'):
        assert result[key]==baseline[key]
    assert result['scores']=={'sup':9,'census':10,'longhorizon':14}
    assert result['measurement_receipts'][13]['path']==str(a2.receipt_path(cell,'longhorizon','c4-lh-06',amended=True))
    assert all(p.read_bytes()==data for p,data in before.items())
