"""CPU author gates only; no GPU or blind verification claim.
Prior art: local C4/A2/A3 fake lease/state/manifest gates (house,2026).
Adds independent-cell execution and half-range resume/admission attacks.
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
from scripts import grm_c4_split_a5 as a5


def persist(session,stop):
    session.mkdir(parents=True,exist_ok=True)
    for name in ('instrumentation.jsonl','transcript.jsonl'):
        (session/name).write_text(''.join(json.dumps({'turn':i})+'\n' for i in range(stop)))
    (session/'restart.json').write_text(json.dumps({'after_turn':stop-1}))
    (session/'repository').mkdir(exist_ok=True)
    (session/'repository/state').write_text('bound PASS')


def state(session):
    return [c.record(p) for p in sorted(session.rglob('*')) if p.is_file()]


@pytest.fixture
def layout(tmp_path,monkeypatch):
    reg=copy.deepcopy(a5.binding())
    # Synthetic timing admission only, pre-registered for exercising otherwise
    # blocked resume paths. Live registered NON_FIT is tested separately below.
    for u in reg['a5_plan']['scheduled_units']:
        u['planning_status']='FIT'; u['cap_status']='FIT'
    monkeypatch.setattr(c,'OUT',tmp_path/'original')
    monkeypatch.setattr(a2,'OUT',tmp_path/'a2')
    monkeypatch.setattr(a3,'OUT',tmp_path/'a3')
    monkeypatch.setattr(a5,'OUT',tmp_path/'a5')
    c.write_once(a5.OUT/'before.json',{'files':[]})
    monkeypatch.setattr(a5,'binding',lambda:reg)
    clock=[100000]
    monkeypatch.setattr(a5.time,'time',lambda:clock[0])
    controls={'wrong_turns':False,'fail':False,'no_serving':False,'lease_calls':0}
    from scripts import grm_cmc1_gpu_arms as leases
    @contextmanager
    def lease(seconds,wait):
        assert (seconds,wait)==(285,0)
        controls['lease_calls']+=1
        yield
    monkeypatch.setattr(leases,'gpu_lease',lease)
    sup=[dict(p,verdict={'correct':True}) for ps in reg['fixtures']['sup'].values() for p in ps]
    census=[dict(p,verdict={'correct':True},turn_row_found=True) for p in reg['fixtures']['census']]
    probes=[dict(p,correct=True,turn_row_found=True) for p in reg['fixtures']['longhorizon']['probes']]
    @contextmanager
    def harness(chunk,width,root,events):
        from scripts import grm_e2e_session as e2e
        # Verify the real adapter's stop table, not a parallel fake schedule.
        assert c.LH_STOPS=={u['spec']:u['stop_after_turns'] for u in a2.lh_units(reg)}
        def run(spec,*,run_dir,**kwargs):
            battery='longhorizon' if spec.startswith('c4-lh') else 'census'
            units=[u for u in reg['units_per_new_cell'] if u['battery']==battery]
            index=next(i for i,u in enumerate(units) if u['spec']==spec)
            start=units[index-1].get('stop_after_turns',index*8) if index else 0
            stop=units[index].get('stop_after_turns',(index+1)*8)
            dest=run_dir/('arm1' if battery=='census' else '')/spec/'session'
            if index:
                source=dest.parent.parent/units[index-1]['spec']/'session'
                assert (source/'repository/state').read_text()=='bound PASS'
                a2.saved_turns(source,start)
                shutil.copytree(source,dest)
            persist(dest,stop)
            events.append({'event':'geometry'})
            if not controls['no_serving'] and (battery=='census' or any(start<=p['turn']<stop for p in probes)):
                events.append({'event':'attempt_residency','mounted_token_seats':12})
            for t in range(start,stop-int(controls['wrong_turns'])):
                e2e.run_turn(None,{},t)
            if controls['fail']:
                raise TimeoutError('synthetic rail')
            return {'session_dir':str(dest),'stop_after_turns':stop,'resumed':bool(index)}
        def serve(spec,**kwargs):
            events.extend([{'event':'geometry'},{'event':'attempt_residency','mounted_token_seats':12}])
            return {'probes':[dict(p,verdict={'correct':True}) for p in reg['fixtures']['sup'][spec]]}
        def lh_score(root):
            a2.saved_turns(root/'c4-lh-13b/session',104)
            return {'probes':copy.deepcopy(probes),'measured_count':14,'correct_count':14}
        with patch_turn(monkeypatch,e2e):
            yield SimpleNamespace(serve_fixture=serve),SimpleNamespace(run_shard=run,score_run=lambda *a,**k:{'probes':copy.deepcopy(census)}),SimpleNamespace(run_shard=run,score_run=lh_score)
    monkeypatch.setattr(c,'harness',harness)
    return SimpleNamespace(reg=reg,clock=clock,controls=controls,tmp=tmp_path)


@contextmanager
def patch_turn(monkeypatch,e2e):
    with monkeypatch.context() as mp:
        mp.setattr(e2e,'run_turn',lambda *a,**k:None)
        yield


def run(layout,cell,units=None):
    for u in units or layout.reg['units_per_new_cell']:
        layout.clock[0]+=1000
        a5.worker(cell,u['battery'],u['spec'])


def seed_legacy(layout):
    cell='c64_w96';reg=layout.reg;old=reg['legacy_context']['a4_context']
    for u in reg['units_per_new_cell'][:-4]:
        amended=u['battery']=='longhorizon' and int(u['spec'][-2:])>=6
        root=a2.OUT if amended else c.OUT
        session=root/'runs'/cell/u['battery']/('arm1' if u['battery']=='census' else '')/u['spec']/'session'
        stop=u.get('stop_after_turns',32)
        persist(session,stop)
        row={'cell':cell,'battery':u['battery'],'spec':u['spec'],'status':'PASS',
            'registration':c.record(c.REG),'amendment':old['amendment' if amended else 'legacy_amendment'],
            'sources':old['sources' if amended else 'legacy_sources'],
            'events':[{'event':'geometry'},{'event':'attempt_residency','mounted_token_seats':12}],
            'gpu_seconds':1,'finished_unix':0,'result':{'session_dir':str(session)},'session_state':state(session)}
        if u['battery']=='sup':
            row['result']['probes']=[dict(p,verdict={'correct':True}) for p in reg['fixtures']['sup'][u['spec']]]
        c.write_once(root/'runs'/cell/f"{u['battery']}_{u['spec']}.json",row)
    # Synthetic legacy A2 eligibility uses original failure at06, as real A2.
    red=c.read(a2.receipt_path(cell,'longhorizon','c4-lh-06',amended=True))
    red.update(status='RED',error=a2.ROOT_ERROR,sources=old['legacy_sources'],amendment=old['legacy_amendment'])
    c.write_once(a2.receipt_path(cell,'longhorizon','c4-lh-06'),red)
    pins=[c.record(p) for root in (c.OUT,a2.OUT) for p in root.rglob('*') if p.is_file()]
    (a5.OUT/'before.json').write_text(json.dumps({'files':pins}))
    return pins


def test_registered_plan_honesty_and_commands():
    reg=a5.binding();plan=reg['a5_plan'];units=a5.schedule(reg)
    assert len(units)==50
    assert [sum(u['cell']==cell for u in units) for cell in a5.CELLS]==[4,23,23]
    halves=plan['units_per_cell'][-4:]
    assert [u['stop_after_turns'] for u in halves]==[92,96,100,104]
    assert [u['estimate_seconds'] for u in halves]==[284,63,66,348]
    assert [u['planning_status'] for u in halves]==['NON_FIT','FIT','FIT','NON_FIT']
    assert plan['timing_evidence'][-1]['turn_seconds'][2]['seconds']>225
    assert plan['projection']['total_seconds']==pytest.approx(7898.6785489856265)
    assert plan['projection']['overflow_seconds']==pytest.approx(2498.6785489856265)
    assert sum(u['estimate_seconds'] for u in units)==5741
    assert any(u['cap_status']=='NON_FIT' for u in units)
    lines=a5.COMMANDS.read_text().splitlines()
    assert a5.COMMANDS.read_text()==a5.command_text(units)
    found=[]
    for i,line in enumerate(lines):
        if ' worker ' not in line:continue
        args=shlex.split(line)
        assert args[:3]==['timeout','--signal=KILL','590s'] and lines[i+1]=='sleep 30'
        found.append(tuple(args[args.index(k)+1] for k in ('--cell','--battery','--spec')))
    assert found==[(u['cell'],u['battery'],u['spec']) for u in units]
    assert sum(' score ' in s for s in lines)==3 and lines[-1].endswith(' summary')


@pytest.mark.parametrize('field',['sha','order','previous_amendment','sources','plan','decoupling','receipt_root','budget','registration','legacy_executor','extra'])
def test_forged_or_stale_refused(tmp_path,monkeypatch,field):
    row=c.read(a5.AMENDMENT)
    if field=='sha':row['immutable']=False
    elif field=='sources':row['sources']=[]
    elif field=='plan':row['plan']['scheduled_units'][0]['planning_status']='FIT'
    elif field=='decoupling':row['decoupling']['within_cell_dependencies_required']=False
    elif field=='receipt_root':row[field]=str(c.OUT/'runs')
    elif field=='budget':row[field]['gpu_seconds']=99999
    elif field=='extra':row[field]=True
    else:row[field]['sha256']='forged'
    path=tmp_path/'amendment.json';c.write_once(path,row)
    path.with_suffix('.sha256').write_text('stale' if field=='sha' else c.record(path)['sha256'])
    monkeypatch.setattr(a5,'AMENDMENT',path)
    with pytest.raises(AssertionError):a5.binding()


def test_stale_source_refused(monkeypatch):
    original=c.record
    def changed(path):
        pin=original(path)
        if str(path)==str(Path(a5.__file__)):
            pin['sha256']='changed'
        return pin
    monkeypatch.setattr(c,'record',changed)
    with pytest.raises(AssertionError):a5.binding()


@pytest.mark.parametrize('cell',['c96_w64','c64_w64'])
def test_decoupled_start_without_earlier_score(layout,cell):
    run(layout,cell,layout.reg['units_per_new_cell'][:1])
    assert c.read(a5.receipt_path(cell,'sup',c.SUP[0]))['status']=='PASS'
    assert not a5.score_path('c64_w96').exists()


def test_synthetic_full_batteries_half_resume_score_and_cross(layout):
    pins=seed_legacy(layout)
    run(layout,'c64_w96',layout.reg['units_per_new_cell'][-4:])
    for cell in a5.CELLS[1:]:run(layout,cell)
    for cell in a5.CELLS:
        row=a5.score(cell)
        assert row['scores']==layout.reg['counts'] and len(row['measurement_receipts'])==23
        for spec,start,stop in [('c4-lh-12a',88,92),('c4-lh-12b',92,96),('c4-lh-13a',96,100),('c4-lh-13b',100,104)]:
            receipt=c.read(a5.receipt_path(cell,'longhorizon',spec))
            assert receipt['turn_range']['completed']==list(range(start,stop))
        with pytest.raises(FileExistsError):a5.score(cell)
    assert all(c.record(p['path'])==p for p in pins)
    result=a5.cross_summary()
    assert result['factorial_verdict']=='SUPPORTED_FINITE'
    assert result['cells']['c96_w96']['scores']['longhorizon']==13
    assert '13/14' in a5.format_summary(result)
    with pytest.raises(AssertionError,match='No retries'):
        a5.worker('c64_w96','longhorizon','c4-lh-12a')


@pytest.mark.parametrize('bad',['state','turns','prior','geometry','amendment','serving'])
def test_resume_and_identity_attacks(layout,bad):
    cell='c96_w64';units=layout.reg['units_per_new_cell']
    run(layout,cell,units[:-4])
    path=a5.receipt_path(cell,'longhorizon','c4-lh-11');row=c.read(path)
    if bad=='state':(Path(row['result']['session_dir'])/'unlisted').write_text('stale')
    elif bad=='turns':layout.controls['wrong_turns']=True
    elif bad=='prior':row['prior_receipt']['sha256']='forged'
    elif bad=='geometry':row['chunk']=0
    elif bad=='amendment':row['amendment']={}
    else:
        for u in a2.lh_units(layout.reg)[:-4]:
            p=a5.receipt_path(cell,'longhorizon',u['spec']);r=c.read(p)
            r['events']=[{'event':'geometry'}]
            # Rebind dependent links solely in this synthetic attack fixture.
            prev=u['resume'];r['prior_receipt']=c.record(a5.receipt_path(cell,'longhorizon',prev)) if prev else None
            p.write_text(json.dumps(r))
        layout.controls['no_serving']=True
    if bad in ('prior','geometry','amendment'):path.write_text(json.dumps(row))
    layout.clock[0]+=1000
    if bad=='serving':
        run(layout,cell,units[-4:-1])
        layout.clock[0]+=1000
        with pytest.raises(AssertionError,match='No serving observed'):a5.worker(cell,'longhorizon','c4-lh-13b')
    else:
        with pytest.raises(AssertionError):a5.worker(cell,'longhorizon','c4-lh-12a')


def test_registered_nonfit_no_gpu_and_independent_cell(layout):
    unit=layout.reg['a5_plan']['scheduled_units'][0];unit['planning_status']='NON_FIT'
    result=a5.worker('c64_w96','longhorizon','c4-lh-12a')
    assert result['status']=='NON_FIT' and layout.controls['lease_calls']==0
    assert not a5.receipt_path('c64_w96','longhorizon','c4-lh-12a').with_suffix('.attempt.json').exists()
    run(layout,'c96_w64',layout.reg['units_per_new_cell'][:1])
    assert layout.controls['lease_calls']==1
    with pytest.raises(AssertionError,match='No retries'):a5.worker('c64_w96','longhorizon','c4-lh-12a')


def test_cap_overflow_and_missing_dependency_are_nonfit(layout):
    layout.reg['a5_plan']['scheduled_units'][4]['cap_status']='NON_FIT'
    row=a5.worker('c96_w64','sup',c.SUP[0])
    assert 'cap overflow' in row['reason']
    row=a5.worker('c64_w64','longhorizon','c4-lh-12b')
    assert 'missing' in row['reason'] and layout.controls['lease_calls']==0


def test_partial_summary_has_no_battery_score(layout):
    pins=seed_legacy(layout)
    run(layout,'c96_w64',layout.reg['units_per_new_cell'][:1])
    for cell in a5.CELLS:
        row=a5.score(cell)
        assert row['status']=='NON_FIT' and 'scores' not in row
    result=a5.cross_summary()
    assert result['factorial_verdict']=='INCONCLUSIVE' and result['prediction_met'] is None
    assert len(result['cells']['c64_w96']['completed_segments'])==19
    assert all('scores' not in result['cells'][cell] for cell in a5.CELLS)
    assert 'NON_FIT (completed:' in a5.format_summary(result)
    assert all(c.record(p['path'])==p for p in pins)


@pytest.mark.parametrize('bad',['partial_score','stale_partial','stale_score','probes'])
def test_summary_refuses_forged_partial_and_stale_full(layout,bad):
    cell='c96_w64'
    if bad in ('stale_score','probes'):
        run(layout,cell)
    a5.score(cell)
    p=a5.score_path(cell);row=c.read(p)
    if bad=='partial_score':row['scores']=layout.reg['counts']
    elif bad=='stale_partial':row['completed_segments']=['fake']
    elif bad=='stale_score':row['measurement_receipts'][0]['sha256']='forged'
    else:row['longhorizon_probes'][0]['turn']=1000
    p.write_text(json.dumps(row))
    with pytest.raises(AssertionError):a5.cross_summary()
    assert not (a5.OUT/'cross_summary.json').exists()


@pytest.mark.parametrize('case',['charge','reserve','cooldown','orphan','historical_drift'])
def test_accounting(layout,case):
    pins=seed_legacy(layout)
    assert a5.accounting(layout.reg)==20 #19PASS + original RED
    if case=='charge':return
    run(layout,'c96_w64',layout.reg['units_per_new_cell'][:1])
    path=a5.receipt_path('c96_w64','sup',c.SUP[0]);row=c.read(path)
    if case=='reserve':row['gpu_seconds']=5400-20-285+0.001
    elif case=='cooldown':row['finished_unix']=layout.clock[0]
    elif case=='orphan':c.write_once(a5.receipt_path('c64_w64','sup',c.SUP[0]).with_suffix('.attempt.json'),{})
    else:Path(pins[0]['path']).write_text('stale')
    path.write_text(json.dumps(row))
    if case!='cooldown':layout.clock[0]+=1000
    with pytest.raises((AssertionError,ValueError)):a5.accounting(layout.reg)


def test_real_existing_bytes_unchanged():
    before=c.read(a5.OUT/'before.json')['files']
    assert len(before)==2567
    assert all(c.record(p['path'])==p for p in before)
    assert not (a5.OUT/'runs').exists()
    a2.binding();a3.binding();a5.binding()
