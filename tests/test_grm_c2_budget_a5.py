"""CPU author baseline. Prior art: C2 amendment1/3 fake dispatch tests (2026).
Taken: temporary epochs, fake subprocess receipts, cap and integrity assertions.
New: named reservation-only RED relocation and unchanged42/remaining10 tests.
No prior art known to me beyond these local systems for this test adapter.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
TARGET = Path(os.environ.get('GRM_C2_A5_TEST_ADAPTER', str(ROOT / 'artifacts/grm_c2/a5/resume.py')))
spec = importlib.util.spec_from_file_location('c2_budget_a5', TARGET)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
b = m.base


@pytest.fixture
def installed(monkeypatch):
    for module, names in [(b, ('OUT', '__file__', 'effective_registration', 'charged_seconds',
                              'receipt_bindings', 'dry_run', 'campaign_lock', 'run_campaign', 'summary')),
                          (m.epoch, ('verify_amendment',)), (b.old, ('OUT',)),
                          (m, ('_INSTALLED', '_RESUMING'))]:
        for name in names:
            monkeypatch.setattr(module, name, getattr(module, name))
    m.install()
    return m.effective_registration()


def test_live42_cap4800_and_original_red_still_ineligible(installed):
    r = installed
    assert r['budget_seconds'] == 4800
    assert r['cells'] == m.read(b.REGISTRATION)['cells']
    assert (r['worker_seconds'], r['outer_seconds'], r['cooldown_seconds'], r['lock_wait_seconds']) == (285,590,30,240)
    assert b.charged_seconds(r) == pytest.approx(3420.8233032692224)
    d = m.dry_run()
    assert list(d['states'].values()).count('COMPLETE') == 42
    assert list(d['states'].values()).count('UNSTARTED') == 9
    assert d['states']['profile-longhistory-7'] == 'RED'
    assert d['reservation_only_eligibility'] == 'ONE_TIME_ON_EXPLICIT_RESUME'
    assert d['budget_seconds'] == 4800
    assert b.__file__ == str(TARGET)


@pytest.mark.parametrize('field,value', [('budget_seconds',9999),('red_cell','other'),('completed_receipts',{})])
def test_forgery_recomputed_sidecar_refused(installed,tmp_path,monkeypatch,field,value):
    a = copy.deepcopy(installed['budget_amendment']); a[field] = value
    p = tmp_path/'forged.json'; b.create(p,a)
    p.with_suffix('.sha256').write_text(m.sha(p))
    monkeypatch.setattr(m,'AMENDMENT',p)
    with pytest.raises(ValueError,match='forged or stale amendment 5'):m.verify_amendment()


@pytest.mark.parametrize('target', ['scripts/grm_c2_epoch3.py','artifacts/grm_c2/lead_commands.txt',
    'artifacts/grm_c2/a5/lead_commands.txt.before',
    'artifacts/grm_c2/epochs/scout-fix-2/cells/profile-longhistory-6/worker.json'])
def test_stale_source_command_or_completed_receipt_refused(installed,monkeypatch,target):
    original=m.sha
    monkeypatch.setattr(m,'sha',lambda p:'0'*64 if Path(p)==ROOT/target else original(p))
    with pytest.raises(ValueError,match='stale amendment 5 binding|changed completed receipt'):
        m.verify_amendment()


@pytest.fixture
def isolated(installed,tmp_path,monkeypatch):
    r=copy.deepcopy(installed)
    live=b.OUT
    monkeypatch.setattr(b,'OUT',tmp_path/'epoch')
    monkeypatch.setattr(b.old,'OUT',b.OUT)
    for c in r['cells']:
        original=live/'cells'/c['id']
        if not original.exists():continue
        for name in ('controller.json','reservation.json'):
            p=original/name
            if p.exists():
                target=b.OUT/'cells'/c['id']/name
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(p.read_bytes())
    monkeypatch.setattr(m,'effective_registration',lambda:r)
    monkeypatch.setattr(b.time,'sleep',lambda seconds:None)
    monkeypatch.setattr(b.old,'sources',lambda:{})
    return r


def test_one_time_archive_and_crash_before_first_worker(isolated):
    r=isolated; cell=r['budget_amendment']['red_cell']
    p=b.OUT/'cells'/cell/'controller.json'; before=p.read_bytes()
    with m._BASE_LOCK(): assert m.reeligible_once(r)
    archive=b.OUT/'reservation_red_a5'/cell/'controller.json'
    assert archive.read_bytes()==before and not p.exists()
    assert b.cell_state(b.old.cell_by_id(r,cell))=='UNSTARTED'
    assert not m.reeligible_once(r) # crash after archive: no second reset needed
    p.parent.mkdir();p.write_bytes(before) # a later RED must remain terminal
    assert not m.reeligible_once(r) and p.read_bytes()==before
    assert b.cell_state(b.old.cell_by_id(r,cell))=='RED'
    assert b.charged_seconds(r)==pytest.approx(3420.8233032692224)


@pytest.mark.parametrize('defect',['reservation','worker','charge','error','worker_sha','status'])
def test_reservation_only_exception_is_narrow(isolated,defect):
    a=isolated['budget_amendment'];d=b.OUT/'cells'/a['red_cell'];p=d/'controller.json'
    if defect in ('reservation','worker'):
        (d/(defect+'.json')).write_text('{}')
    else:
        c=m.read(p)
        key={'charge':'charged_seconds','error':'error','worker_sha':'worker_sha256','status':'status'}[defect]
        c[key]={'charge':1,'error':'ValueError: worker failed','worker_sha':'a'*64,'status':'COMPLETE'}[defect]
        p.write_text(json.dumps(c)); a['red_controller_sha256']=m.sha(p)
    with pytest.raises(ValueError,match='reservation-only RED'):m.reeligible_once(isolated)
    assert p.exists() and not (b.OUT/'reservation_red_a5'/a['red_cell']).exists()


def test_underfunded_exception_preserves_red(isolated):
    isolated['budget_seconds']=3600
    with pytest.raises(ValueError,match='budget rail'):m.reeligible_once(isolated)
    assert (b.OUT/'cells'/isolated['budget_amendment']['red_cell']/'controller.json').exists()


def test_exact_reservation_boundary_retains_historical_charge(isolated,monkeypatch):
    r=isolated;total=b.charged_seconds(r)
    assert total==pytest.approx(3420.8233032692224)
    r['budget_seconds']=total+285-0.001
    with pytest.raises(ValueError,match='budget rail'):m.reeligible_once(r)
    r['budget_seconds']=total+285
    assert m.reeligible_once(r)
    orphan=next(c for c in r['cells'] if b.cell_state(c)=='UNSTARTED')
    (b.OUT/'cells'/orphan['id']).mkdir()
    assert b.charged_seconds(r)==pytest.approx(total+285)


def test_resume_dispatches_remaining10_once_and_preserves42(isolated,monkeypatch):
    r=isolated;calls=[]
    frozen={p:p.read_bytes() for p in (b.OUT/'cells').glob('*/controller.json')
            if m.read(p)['status']=='COMPLETE'}
    red_before=(b.OUT/'cells'/r['budget_amendment']['red_cell']/'controller.json').read_bytes()
    def execute(command,**kw):
        assert command[:4]==['timeout','--signal=TERM','--kill-after=5','585']
        assert command[-3]==str(TARGET)
        assert kw['pass_fds'] and kw['env']['GRM_C2_CAMPAIGN_FD']==str(kw['pass_fds'][0])
        c=b.old.cell_by_id(r,command[-1]);calls.append(c['id'])
        assert all(b.cell_state(b.old.cell_by_id(r,d))=='COMPLETE' for d in c['depends'])
        assert b.charged_seconds(r)+285<=r['budget_seconds']
        d=b.OUT/'cells'/c['id'];d.mkdir()
        b.create(d/'controller.json',dict(status='COMPLETE',charged_seconds=81,finished_at=0,cell=c,**b.receipt_bindings()))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(b.subprocess,'run',execute)
    monkeypatch.setattr(b,'score_battery',lambda r,battery:{'status':'GREEN'})
    with pytest.raises(ValueError,match='explicit'):m.run_campaign(r,resume=False)
    assert m.run_campaign(r,resume=True)==0
    assert len(calls)==10 and calls[0]=='profile-longhistory-7'
    assert m.run_campaign(r,resume=True)==0 and len(calls)==10
    assert all(p.read_bytes()==data for p,data in frozen.items())
    assert (b.OUT/'reservation_red_a5'/r['budget_amendment']['red_cell']/'controller.json').read_bytes()==red_before
    assert b.charged_seconds(r)==pytest.approx(4230.8233032692224)


def test_new_red_stops_and_never_resets(isolated,monkeypatch):
    r=isolated;calls=[]
    def execute(command,**kw):
        c=b.old.cell_by_id(r,command[-1]);calls.append(c['id'])
        d=b.OUT/'cells'/c['id'];d.mkdir()
        b.create(d/'controller.json',dict(status='RED',charged_seconds=22,finished_at=0,cell=c,**b.receipt_bindings()))
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(b.subprocess,'run',execute)
    monkeypatch.setattr(b,'score_battery',lambda r,battery:{'status':'GREEN'})
    assert m.run_campaign(r,resume=True)==1 and calls==['profile-longhistory-7']
    assert not m.reeligible_once(r)
    with pytest.raises(ValueError,match='never retried'):
        b.run_cell(b.old.cell_by_id(r,calls[0]),r)


def test_campaign_mutex_refuses_before_reset(isolated):
    with m._BASE_LOCK():
        with pytest.raises(ValueError,match='another C2 foreground controller'):
            m.run_campaign(isolated,resume=True)
    assert (b.OUT/'cells'/isolated['budget_amendment']['red_cell']/'controller.json').exists()


def test_a4_reader_still_accepts_frozen_source_and_sup():
    path=ROOT/'artifacts/grm_c2/a4_staging/scripts/grm_c2_score_a4.py'
    spec=importlib.util.spec_from_file_location('a4_compat',path);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
    s.verify_amendment()
    assert m.read(s.EPOCH/'scores_a4/sup.json')


def test_fresh_process_dry_run_and_direct_cell_refusal():
    env=dict(b.os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    p=subprocess.run([sys.executable,str(TARGET),'--dry-run'],cwd=ROOT,env=env,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert json.loads(p.stdout)['budget_seconds']==4800
    p=subprocess.run([sys.executable,str(TARGET),'--cell','profile-longhistory-7'],cwd=ROOT,env=env,capture_output=True,text=True)
    assert p.returncode==1 and 'direct --cell disabled' in p.stderr


@pytest.mark.parametrize('later_state', ['COMPLETE', 'RED', 'ORPHAN'])
def test_second_attempt_is_never_reset(isolated, later_state):
    # Prior art: C2 A3 create-only terminal states (project contributors, 2026).
    # Reuse states; ours: apply each state after the A5 one-time archive.
    cell = isolated['budget_amendment']['red_cell']
    assert m.reeligible_once(isolated)
    d = b.OUT / 'cells' / cell
    d.mkdir()
    if later_state != 'ORPHAN':
        b.create(d / 'controller.json', dict(status=later_state, charged_seconds=1,
                 **b.receipt_bindings()))
    before = {p.name: p.read_bytes() for p in d.iterdir()}
    assert not m.reeligible_once(isolated)
    assert d.exists() and {p.name:p.read_bytes() for p in d.iterdir()} == before


def test_incomplete_dependency_refuses_reset(isolated):
    cell = b.old.cell_by_id(isolated, isolated['budget_amendment']['red_cell'])
    dep = b.OUT / 'cells' / cell['depends'][0] / 'controller.json'
    value = m.read(dep); value['status'] = 'RED'; dep.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='dependency incomplete'):
        m.reeligible_once(isolated)
    assert (b.OUT / 'cells' / cell['id'] / 'controller.json').exists()


@pytest.mark.parametrize('headroom', [-0.001, 0.0])
def test_unchanged_controller_reservation_and_worker_entry(isolated, monkeypatch, headroom):
    # Prior art: C2 A1/A3 fake-process boundary tests (project contributors, 2026).
    # Reuse the real reservation controller with fake lease/process; ours:
    # verify the exact raised-cap rail and the worker descendant adapter path.
    from contextlib import contextmanager
    from types import ModuleType
    r = isolated
    assert m.reeligible_once(r)
    used = b.charged_seconds(r)
    r['budget_seconds'] = used + r['worker_seconds'] + headroom
    cell = b.old.cell_by_id(r, r['budget_amendment']['red_cell'])
    calls = []
    fake = ModuleType('scripts.grm_cmc1_gpu_arms')
    @contextmanager
    def lease(seconds, lock_wait):
        assert (seconds, lock_wait) == (285, 240)
        yield
    fake.gpu_lease = lease
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(b.old, 'environment', lambda flags: {})
    monkeypatch.setattr(b.old, 'flags_for', lambda side: {})
    monkeypatch.setattr(b, 'evidence_rows', lambda cell, value: [])
    def execute(command, **kw):
        calls.append(command)
        assert command == [sys.executable, str(TARGET), '--worker', cell['id']]
        assert kw['timeout'] == 280
        b.create(b.OUT / 'cells' / cell['id'] / 'worker.json', {})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(b.subprocess, 'run', execute)
    result = b.run_cell(cell, r)
    receipt = m.read(b.OUT / 'cells' / cell['id'] / 'controller.json')
    if headroom < 0:
        assert result == 1 and not calls
        assert receipt['error'] == 'ValueError: budget rail: cannot reserve next 285-second cell'
        assert receipt['charged_seconds'] == 0 and receipt['worker_sha256'] is None
        assert sorted(p.name for p in (b.OUT / 'cells' / cell['id']).iterdir()) == ['controller.json']
    else:
        assert result == 0 and len(calls) == 1 and receipt['status'] == 'COMPLETE'
        assert m.read(b.OUT / 'cells' / cell['id'] / 'reservation.json')['seconds'] == 285
