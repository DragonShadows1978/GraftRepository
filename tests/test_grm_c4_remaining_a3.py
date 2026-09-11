"""CPU author gates, not blind verification or GPU evidence.
Prior art: local C4/A4 fake leases and persisted-state gates (house, 2026).
Adds remaining-cell scope, three-root budget and four-cell assembly cases.
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
from scripts import grm_c4_remaining_a3 as a3


def persist(session, stop):
    session.mkdir(parents=True, exist_ok=True)
    for name in ('instrumentation.jsonl','transcript.jsonl'):
        (session/name).write_text(''.join(json.dumps({'turn':i})+'\n' for i in range(stop)))
    (session/'restart.json').write_text(json.dumps({'after_turn':stop-1}))
    (session/'repository').mkdir(exist_ok=True)
    (session/'repository/state').write_text('bound PASS')


def state(session):
    return [c.record(p) for p in sorted(session.rglob('*')) if p.is_file()]


def probe_rows(reg):
    sup = [dict(p, verdict={'correct':True}) for ps in reg['fixtures']['sup'].values() for p in ps]
    census = [dict(p, verdict={'correct':True}, turn_row_found=True) for p in reg['fixtures']['census']]
    lh = [dict(p, correct=True, turn_row_found=True) for p in reg['fixtures']['longhorizon']['probes']]
    return sup, census, lh


@pytest.fixture
def layout(tmp_path, monkeypatch):
    reg = a3.binding()
    monkeypatch.setattr(c,'OUT',tmp_path/'original')
    monkeypatch.setattr(a2,'OUT',tmp_path/'a4')
    monkeypatch.setattr(a3,'OUT',tmp_path/'a5')
    monkeypatch.setattr(a3,'binding',lambda:reg)
    monkeypatch.setattr(a3.time,'time',lambda:100000)
    # The preceding A4 cell score is a separate dependency gate; worker tests
    # isolate the new executor, while summary tests below use real load_score.
    original_load = a3.load_score
    monkeypatch.setattr(a3,'load_score',lambda *args: {})
    controls = {'no_serving':False,'wrong_turns':False,'fail':False}
    from scripts import grm_cmc1_gpu_arms as leases
    @contextmanager
    def lease(seconds, wait):
        assert (seconds,wait)==(285,0)
        yield
    monkeypatch.setattr(leases,'gpu_lease',lease)
    _, census_probes, lh_probes = probe_rows(reg)
    @contextmanager
    def harness(chunk,width,root,events):
        from scripts import grm_e2e_session as e2e
        def turn(*args,**kwargs):
            return None
        def run(spec, *, run_dir, **kwargs):
            battery = 'longhorizon' if spec.startswith('c4-lh') else 'census'
            units = [u for u in reg['units_per_new_cell'] if u['battery']==battery]
            index = next(i for i,u in enumerate(units) if u['spec']==spec)
            dest = run_dir/('arm1' if battery=='census' else '')/spec/'session'
            if index:
                prior = dest.parent.parent/units[index-1]['spec']/'session'
                assert (prior/'repository/state').read_text()=='bound PASS'
                shutil.copytree(prior,dest)
            stop = units[index].get('stop_after_turns',(index+1)*8)
            start = units[index-1].get('stop_after_turns',index*8) if index else 0
            persist(dest,stop)
            events.append({'event':'geometry'})
            probe_turns = {p['turn'] for p in reg['fixtures']['longhorizon']['probes']}
            if not controls['no_serving'] and (battery=='census' or any(t in probe_turns for t in range(start,stop))):
                events.append({'event':'attempt_residency','mounted_token_seats':12})
            for t in range(start,stop-int(controls['wrong_turns'])):
                e2e.run_turn(None,{},t)
            if controls['fail']:
                raise TimeoutError('synthetic rail')
            return {'session_dir':str(dest),'stop_after_turns':stop,'resumed':bool(index)}
        def serve(spec,**kwargs):
            events.extend([{'event':'geometry'},{'event':'attempt_residency','mounted_token_seats':12}])
            return {'probes':[dict(p,verdict={'correct':True}) for p in reg['fixtures']['sup'][spec]]}
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(e2e,'run_turn',turn)
            yield SimpleNamespace(serve_fixture=serve), SimpleNamespace(run_shard=run,
                score_run=lambda *a,**k: {'probes':copy.deepcopy(census_probes)}), SimpleNamespace(run_shard=run,
                score_run=lambda *a,**k: {'probes':copy.deepcopy(lh_probes),'measured_count':14,'correct_count':14})
    monkeypatch.setattr(c,'harness',harness)
    return SimpleNamespace(reg=reg,controls=controls,load_score=original_load,monkeypatch=monkeypatch,next_time=200000)


def run_units(layout, units, cell='c96_w64'):
    # Advance fake clock, never sleep or acquire a real lease.
    for u in units:
        now=layout.next_time
        layout.next_time+=1000
        layout.monkeypatch.setattr(a3.time,'time',lambda n=now:n)
        a3.worker(cell,u['battery'],u['spec'])


@pytest.mark.campaign_receipt(registration='artifacts/grm_c4/registration.json + amendment_a1..a3.json (pin absolute /mnt/ForgeRealm/wt/grm-c4/ paths)')
def test_real_dry_schedule_and_commands():
    reg=a3.binding()
    units=a3.schedule(reg)
    assert len(units)==42 and sum(u['estimate_seconds'] for u in units)==3200
    lines=(c.OUT/'lead_commands_a3.txt').read_text().splitlines()
    assert 'set -e' in lines and lines[-1].endswith(' summary')
    found=[]
    for i,line in enumerate(lines):
        if ' worker ' not in line:
            continue
        args=shlex.split(line)
        assert args[:3]==['timeout','--signal=KILL','590s']
        assert lines[i+1]=='sleep 30'
        found.append(tuple(args[args.index(k)+1] for k in ('--cell','--battery','--spec')))
    assert found==[(u['cell'],u['battery'],u['spec']) for u in units]
    for cell in a3.CELLS:
        idx=lines.index(f'python scripts/grm_c4_remaining_a3.py score --cell {cell}')
        assert lines[idx-2].endswith('--spec c4-lh-13')
    assert lines==a3.command_text(reg).splitlines()


@pytest.mark.parametrize('field',['sha','previous_amendment','order','sources','scheduled_units','budget','receipt_root','score_cells','executor'])
def test_forged_or_stale_amendment_refused(tmp_path,monkeypatch,field):
    row=c.read(a3.AMENDMENT)
    if field=='sha':
        row['immutable']=False
    elif field=='sources':
        row[field]=[]
    elif field=='scheduled_units':
        row[field]=row[field][1:]
    elif field=='budget':
        row[field]['gpu_seconds']=99999
    elif field=='score_cells':
        row[field]=['c64_w96']
    elif field=='receipt_root':
        row[field]=str(c.OUT/'runs')
    else:
        row[field]['sha256']='forged'
    path=tmp_path/'amendment.json'
    c.write_once(path,row)
    path.with_suffix('.sha256').write_text('stale' if field=='sha' else c.record(path)['sha256'])
    monkeypatch.setattr(a3,'AMENDMENT',path)
    with pytest.raises(AssertionError):
        a3.binding()


@pytest.mark.campaign_receipt(registration='artifacts/grm_c4/registration.json + amendment_a1..a3.json (pin absolute /mnt/ForgeRealm/wt/grm-c4/ paths)')
def test_existing_receipts_and_original_bindings_unchanged():
    assert c.binding()['amendment']==c.record(c.OUT/'amendment_a3.json')
    assert a2.binding()['amendment']==c.record(a2.AMENDMENT)
    a3.binding()
    before=c.read(a3.OUT/'original_receipts_before.json')['files']
    assert before
    assert all(c.record(r['path'])==r for r in before)


@pytest.mark.campaign_receipt(registration='artifacts/grm_c4/registration.json + amendment_a1..a3.json (pin absolute /mnt/ForgeRealm/wt/grm-c4/ paths)')
def test_all_batteries_resume_guard_score_layout_and_create_only(layout):
    reg=layout.reg
    run_units(layout,reg['units_per_new_cell'])
    middle=c.read(a3.receipt_path('c96_w64','longhorizon','c4-lh-06'))
    assert middle['status']=='PASS' and middle['turn_range']['completed']==list(range(40,48))
    assert not any(e['event']=='attempt_residency' for e in middle['events'])
    final=c.read(a3.receipt_path('c96_w64','longhorizon','c4-lh-13'))
    assert final['cell_serving_guard']['attempt_residency_count']>0
    before={p:p.read_bytes() for p in (a3.OUT/'runs').rglob('*') if p.is_file()}
    result=a3.score('c96_w64')
    assert result['scores']==reg['counts'] and len(result['measurement_receipts'])==21
    assert all(p.read_bytes()==data for p,data in before.items())
    assert not c.OUT.exists() and not a2.OUT.exists()
    with pytest.raises(FileExistsError):
        a3.score('c96_w64')
    with pytest.raises(AssertionError,match='No retries'):
        a3.worker('c96_w64','longhorizon','c4-lh-06')


@pytest.mark.campaign_receipt(registration='artifacts/grm_c4/registration.json + amendment_a1..a3.json (pin absolute /mnt/ForgeRealm/wt/grm-c4/ paths)')
def test_out_of_order_or_original_claim_refused_before_work(layout):
    with pytest.raises(AssertionError,match='Prior unit missing'):
        a3.worker('c96_w64','census','e2e-1')
    with pytest.raises(AssertionError,match='not authorized'):
        a3.worker('c64_w96','sup',c.SUP[0])
    claim=c.OUT/'runs/c96_w64'/f'sup_{c.SUP[0]}.attempt.json'
    c.write_once(claim,{'started_unix':0})
    raw=claim.read_bytes()
    with pytest.raises(AssertionError,match='No retries'):
        a3.worker('c96_w64','sup',c.SUP[0])
    assert claim.read_bytes()==raw and not a3.OUT.exists()


@pytest.mark.campaign_receipt(registration='artifacts/grm_c4/registration.json + amendment_a1..a3.json (pin absolute /mnt/ForgeRealm/wt/grm-c4/ paths)')
def test_previous_cell_score_is_required(layout,monkeypatch):
    monkeypatch.setattr(a3,'load_score',layout.load_score)
    with pytest.raises(AssertionError,match='Prior cell score missing'):
        a3.worker('c96_w64','sup',c.SUP[0])
    assert not a3.OUT.exists()


@pytest.mark.campaign_receipt(registration='artifacts/grm_c4/registration.json + amendment_a1..a3.json (pin absolute /mnt/ForgeRealm/wt/grm-c4/ paths)')
@pytest.mark.parametrize('kind',['state','turns','zero_serving','rail'])
def test_bound_state_turns_serving_and_failure_stop(layout,kind):
    units=layout.reg['units_per_new_cell']
    run_units(layout,units[:9])
    layout.monkeypatch.setattr(a3.time,'time',lambda:900000)
    if kind=='state':
        row=c.read(a3.receipt_path('c96_w64','longhorizon','c4-lh-01'))
        (Path(row['result']['session_dir'])/'unlisted').write_text('drift')
        with pytest.raises(AssertionError,match='state drift'):
            a3.worker('c96_w64','longhorizon','c4-lh-02')
    elif kind=='turns':
        layout.controls['wrong_turns']=True
        with pytest.raises(AssertionError,match='turn range mismatch'):
            a3.worker('c96_w64','longhorizon','c4-lh-02')
    elif kind=='rail':
        layout.controls['fail']=True
        with pytest.raises(TimeoutError,match='synthetic rail'):
            a3.worker('c96_w64','longhorizon','c4-lh-02')
        with pytest.raises(AssertionError,match='Other RED'):
            a3.accounting(layout.reg)
    else:
        first=a3.receipt_path('c96_w64','longhorizon','c4-lh-01')
        row=c.read(first); row['events']=[{'event':'geometry'}]
        first.write_text(json.dumps(row))  # synthetic fixture mutation only
        layout.controls['no_serving']=True
        run_units(layout,units[9:-1])
        layout.monkeypatch.setattr(a3.time,'time',lambda:900000)
        with pytest.raises(AssertionError,match='No serving observed'):
            a3.worker('c96_w64','longhorizon','c4-lh-13')


@pytest.mark.campaign_receipt(registration='artifacts/grm_c4/registration.json + amendment_a1..a3.json (pin absolute /mnt/ForgeRealm/wt/grm-c4/ paths)')
@pytest.mark.parametrize('case',['charge','cap','cooldown','orphan','other_red'])
def test_three_root_accounting(layout,case):
    old=layout.reg['a4_context']
    for root,amended,seconds,status in ((c.OUT,False,73,'RED'),(a2.OUT,True,17,'PASS')):
        c.write_once(root/'runs/c64_w96/longhorizon_c4-lh-06.json',{
            'cell':'c64_w96','battery':'longhorizon','spec':'c4-lh-06',
            'registration':c.record(c.REG),'sources':old['sources' if amended else 'legacy_sources'],
            'amendment':old['amendment' if amended else 'legacy_amendment'],
            'status':status,'error':a2.ROOT_ERROR,'finished_unix':0,'gpu_seconds':seconds})
    layout.monkeypatch.setattr(a2,'eligible_suffix',lambda *args:['c4-lh-06'])
    run_units(layout,layout.reg['units_per_new_cell'][:1])
    p=a3.receipt_path('c96_w64','sup',c.SUP[0]); row=c.read(p)
    assert row['campaign_gpu_seconds_before']==90
    row['finished_unix']=0
    row['gpu_seconds']=11
    # Prior art: C4 reserve-boundary gate (house, 2026). Lead A4 changes
    # the cap only; derive an over-cap fixture from the effective budget.
    if case=='cap': row['gpu_seconds']=layout.reg['budget']['gpu_seconds']-90-285+1
    if case=='cooldown': row['finished_unix']=199999
    if case=='other_red': row['status']='RED'
    p.write_text(json.dumps(row))
    if case=='orphan':
        c.write_once(a2.OUT/'runs/c64_w96/longhorizon_c4-lh-07.attempt.json',{})
    if case=='charge':
        assert a3.accounting(layout.reg)==101
    else:
        with pytest.raises(AssertionError,match={'cap':'budget non-fit','cooldown':'cooldown','orphan':'Unfinished worker','other_red':'Other RED'}[case]):
            a3.accounting(layout.reg)


def four_cells(layout, monkeypatch):
    reg=layout.reg
    sup,census,lh=probe_rows(reg)
    monkeypatch.setattr(a3,'load_score',layout.load_score)
    def a4_pass(context,cell,battery,spec):
        p=a2.OUT/'runs'/cell/f'{battery}_{spec}.json'
        row=c.read(p)
        assert row['status']=='PASS'
        return p,row
    monkeypatch.setattr(a2,'pass_receipt',a4_pass)
    for cell in c.executable_cells(reg):
        context=reg['a4_context'] if cell=='c64_w96' else reg
        root=a2.OUT if cell=='c64_w96' else a3.OUT
        cfg=next(x for x in reg['cells'] if x['id']==cell)
        receipts=[]
        for u in reg['units_per_new_cell']:
            p=root/'runs'/cell/f"{u['battery']}_{u['spec']}.json"
            row={'cell':cell,'battery':u['battery'],'spec':u['spec'],'status':'PASS',
                'registration':c.record(c.REG),'sources':context['sources'],'amendment':context['amendment'],
                'geometry_pin':reg['geometry'],'chunk':cfg['chunk'],'width':cfg['width']}
            if u['spec']=='c4-lh-13':
                session=root/'runs'/cell/'longhorizon/c4-lh-13/session'
                persist(session,104)
                row.update(result={'session_dir':str(session)},session_state=state(session),cell_serving_guard={'attempt_residency_count':1})
            c.write_once(p,row); receipts.append(c.record(p))
        c.write_once(a3.score_path(cell),{'cell':cell,'registration':c.record(c.REG),
            'amendment':context['amendment'],'sources':context['sources'],
            'geometry':reg['geometry'],'counts':reg['counts'],'scores':reg['counts'],
            'sup_probes':copy.deepcopy(sup),'census_probes':copy.deepcopy(census),'longhorizon_probes':copy.deepcopy(lh),
            'measurement_receipts':receipts,'measurements':{b:{'attempt_observations':1} for b in reg['counts']}})


@pytest.mark.campaign_receipt(registration='artifacts/grm_c4/registration.json + amendment_a1..a3.json (pin absolute /mnt/ForgeRealm/wt/grm-c4/ paths)')
@pytest.mark.parametrize('outcome',['SUPPORTED_FINITE','REJECTED_FINITE','MIXED'])
def test_synthetic_four_cell_summary(layout,monkeypatch,outcome):
    four_cells(layout,monkeypatch)
    if outcome!='SUPPORTED_FINITE':
        cells=['c64_w96'] if outcome=='REJECTED_FINITE' else list(c.executable_cells(layout.reg))
        for cell in cells:
            p=a3.score_path(cell); row=c.read(p)
            next(x for x in row['longhorizon_probes'] if x['turn']==33)['correct']=False
            row['scores']['longhorizon']=13; p.write_text(json.dumps(row))
    before={p:p.read_bytes() for root in (c.OUT,a2.OUT,a3.OUT) for p in (root/'runs').rglob('*') if p.is_file()}
    summary=a3.cross_summary()
    assert summary['factorial_verdict']==outcome
    assert summary['prediction_met']==(outcome=='SUPPORTED_FINITE')
    assert summary['cells']['c96_w96']['scores']=={'sup':9,'census':9,'longhorizon':13}
    assert summary['diagonal_comparison']['historical']=={'sup':8,'census':10,'longhorizon':14}
    assert summary['diagonal_comparison']['delta_correct']['sup']==1
    rendered=a3.format_summary(summary)
    assert '96 |' in rendered and '64 |' in rendered and '13/14' in rendered and '4/4' not in rendered
    assert all(p.read_bytes()==data for p,data in before.items())
    with pytest.raises(FileExistsError): a3.cross_summary()


@pytest.mark.campaign_receipt(registration='artifacts/grm_c4/registration.json + amendment_a1..a3.json (pin absolute /mnt/ForgeRealm/wt/grm-c4/ paths)')
@pytest.mark.parametrize('bad',['missing','stale','identity','probes','observation'])
def test_cross_summary_refuses_incomplete_or_mixed_evidence(layout,monkeypatch,bad):
    four_cells(layout,monkeypatch)
    p=a3.score_path('c64_w64'); row=c.read(p)
    if bad=='missing':
        p.unlink()
    else:
        if bad=='stale': row['measurement_receipts'][0]['sha256']='stale'
        if bad=='identity': row['amendment']={}
        if bad=='probes': row['longhorizon_probes'][0]['turn']=33
        if bad=='observation': row['measurements']['census']['attempt_observations']=0
        p.write_text(json.dumps(row))
    with pytest.raises(AssertionError): a3.cross_summary()
    assert not (a3.OUT/'cross_summary.json').exists()
