"""A8 CPU author baseline, never GPU/blind-verification evidence.
Prior art: C4/A5/A6/A7 (house, 2026), exact bytes, fake leases/state and
adversarial manifest/accounting fixtures; extend to incident recovery.
"""
import copy
import json
from pathlib import Path
import shlex
from types import SimpleNamespace
import subprocess

import pytest
from scripts import grm_c4_campaign as c
from scripts import grm_c4_split_a5 as a5
from scripts import grm_c4_skip_a6 as a6
from scripts import grm_c4_recovery_a8 as a8
from test_grm_c4_split_a5 import layout, run
from test_grm_c4_skip_a6 import a6_layout

LIVE_OUT = a8.OUT
TRUNCATED = a8.OLD_OUT/'runs/c96_w64/census_e2e-2.json'
SHA = '28eb9c9dd370c07f00f2e5e10e413f987c1d80bde3d2ab11f67fe96b008b03c6'


def marker(path, reg):
    cell,battery,spec=a8.identity(path)
    c.write_once(path.with_suffix('.attempt.json'),dict(cell=cell,battery=battery,spec=spec,
        registration=c.record(c.REG),amendment=reg['amendment'],budget_amendment=reg['budget_amendment'],started_unix=1))


def test_exact_corrupt_bytes_and_live_inventory():
    reg=a8.binding(); rows=reg['a8_incidents']
    assert len(rows)==1 and rows[0]['classification']=='CORRUPT_DISK_FULL'
    row=rows[0]
    assert row['receipt']['sha256']==SHA and row['receipt']['bytes']==16384
    assert Path(row['archive']['path']).read_bytes()==TRUNCATED.read_bytes()
    with pytest.raises(json.JSONDecodeError,match='line 545 column 3'):
        json.loads(Path(row['archive']['path']).read_bytes())
    assert row['recorded_wall_seconds'] is None and row['charge_seconds']==0
    assert row['claim']['sha256']=='ecd7924d1de68f43b734cdf2f28524918fe7cde95e04182d54a35cbb6ed766f8'
    for spec in c.SUP:
        assert c.read(a8.OLD_OUT/'runs/c96_w64'/f'sup_{spec}.json')['status']=='PASS'
    assert c.read(a8.OLD_OUT/'runs/c96_w64/census_e2e-1.json')['status']=='PASS'
    assert not [p for root in a8.original_roots() for p in (root/'runs').glob('*/*.attempt.json')
                if not p.with_name(p.name.replace('.attempt.json','.json')).exists()]


@pytest.mark.parametrize('data,wall',[(b'{"gpu_seconds":12.5,"events":[',12.5),
    (b'{"events":[{"gpu_seconds":123},',None),
    (b'{"x":"gpu_seconds:999",',None),(b'{"gpu_seconds":0,',0)])
def test_only_complete_top_level_corrupt_wall(data,wall):
    assert a8.recorded_wall(data)==wall


@pytest.mark.parametrize('value',['NaN','Infinity','-1','true','"9"'])
def test_invalid_corrupt_wall_refused(value):
    with pytest.raises(AssertionError,match='Invalid recorded'):
        a8.recorded_wall(('{"gpu_seconds":'+value+',').encode())


@pytest.fixture
def recovery(a6_layout,monkeypatch):
    previous=copy.deepcopy(a6_layout.reg)
    old=a5.OUT
    monkeypatch.setattr(a8,'OLD_OUT',old)
    monkeypatch.setattr(a8,'OUT',a6_layout.tmp/'a8')
    reg={**copy.deepcopy(previous),'a8_previous':previous,'a8_incidents':[],
         'amendment':{'path':'synthetic-a8','sha256':'synthetic-a8','bytes':1},'sources':[]}
    monkeypatch.setattr(a8,'binding',lambda:reg)
    monkeypatch.setattr(a8,'disk_preflight',lambda:20_000_000_000)
    return SimpleNamespace(base=a6_layout,previous=previous,reg=reg,old=old)


def test_abandoned_claim_charged_full_reservation(recovery):
    r=recovery; path=r.old/'runs/c96_w64/sup_correction_then_restatement.json'
    marker(path,r.previous)
    row=a8.classify(path,path.with_suffix('.attempt.json'),r.previous)
    assert row['classification']=='ABANDONED_DISK_FULL' and row['charge_seconds']==285
    assert row['reruns_allowed']==1 and not path.exists()
    r.reg['a8_incidents']=[row]
    with a8.executor(): assert a8.accounting(r.reg,reserve=False)==285


@pytest.mark.parametrize('kind',['corrupt','abandoned'])
@pytest.mark.parametrize('outcome',['PASS','RED','unfinished'])
def test_incident_rerun_once_and_accounting(recovery,kind,outcome):
    r=recovery; cell='c96_w64'; spec=c.SUP[0]
    path=r.old/'runs'/cell/f'sup_{spec}.json'; marker(path,r.previous)
    if kind=='corrupt': path.write_bytes(TRUNCATED.read_bytes())
    # Real bytes are census-shaped internally, but invalid JSON classification
    # is tied to the authorized filename/claim; no payload identity is inferred.
    incident=a8.classify(path,path.with_suffix('.attempt.json'),r.previous)
    r.reg['a8_incidents']=[incident]
    pins=[c.record(p) for p in path.parent.glob('*.json')]
    dest=a8.OUT/'runs'/cell/path.name
    if outcome=='unfinished':
        marker(dest,r.reg)
    else:
        # SUP fake runner normally cannot fail; force the measured harness to
        # fail geometry validation after fake lease acquisition for RED.
        if outcome=='RED':
            from contextlib import contextmanager
            from unittest.mock import patch
            @contextmanager
            def broken(*args):
                yield SimpleNamespace(serve_fixture=lambda *a,**k: {}),None,None
            with patch.object(a8,'OLD_HARNESS',broken):
                with pytest.raises(AssertionError,match='No arena'):
                    a8.worker(cell,'sup',spec)
        else: a8.worker(cell,'sup',spec)
        assert c.read(dest)['status']==outcome
        with a8.executor():
            assert a8.accounting(r.reg,reserve=False)==pytest.approx(incident['charge_seconds']+c.read(dest)['gpu_seconds'])
    calls=r.base.controls['lease_calls']
    with pytest.raises(AssertionError,match='No retries'):
        a8.worker(cell,'sup',spec)
    assert r.base.controls['lease_calls']==calls
    assert all(c.record(p['path'])==p for p in pins)
    if outcome=='unfinished':
        with a8.executor(),pytest.raises(AssertionError,match='Unfinished worker claim'):
            a8.accounting(r.reg,reserve=False)


def test_census_recovery_copies_intact_prior_and_continues(recovery):
    r=recovery; cell='c96_w64'
    # Original five completed units; their session bytes and receipts must
    # survive both the exact corrupt rerun and subsequent e2e-3 resume.
    run(r.base,cell,r.previous['units_per_new_cell'][:5])
    path=r.old/'runs'/cell/'census_e2e-2.json'; marker(path,r.previous)
    path.write_bytes(TRUNCATED.read_bytes())
    r.reg['a8_incidents']=[a8.classify(path,path.with_suffix('.attempt.json'),r.previous)]
    pins=[c.record(p) for p in r.old.rglob('*') if p.is_file()]
    for spec in ('e2e-2','e2e-3'):
        r.base.clock[0]+=1000
        a8.worker(cell,'census',spec)
        row=c.read(a8.OUT/'runs'/cell/f'census_{spec}.json')
        assert row['status']=='PASS'
        assert str(a8.OUT/'runs') in row['result']['session_dir']
    first=c.read(a8.OUT/'runs'/cell/'census_e2e-2.json')
    assert first['prior_receipt']==c.record(r.old/'runs'/cell/'census_e2e-1.json')
    with a8.executor(): a5.pass_receipt(r.reg,cell,'census','e2e-3')
    assert r.base.controls['lease_calls']==7
    assert all(c.record(p['path'])==p for p in pins)


@pytest.mark.parametrize('free,accept',[(0,False),(19_999_999_999,False),(20_000_000_000,True),(20_000_000_001,True)])
def test_mock_df_boundary_prints_number_before_refusal(monkeypatch,capsys,free,accept):
    def df(args,**kwargs):
        assert args==['df','-B1','--output=avail','/mnt/ForgeRealm']
        assert kwargs==dict(check=True,capture_output=True,text=True)
        return SimpleNamespace(stdout=f'       Avail\n{free}\n')
    monkeypatch.setattr(a8.subprocess,'run',df)
    if accept: assert a8.disk_preflight()==free
    else:
        with pytest.raises(AssertionError,match='Disk preflight refused'): a8.disk_preflight()
    assert json.loads(capsys.readouterr().out)['available_bytes']==free


@pytest.mark.parametrize('output',['','Avail\n','Avail\nbad\n','Filesystem\n999999999999\n','Avail\n1\n2\n'])
def test_malformed_df_fails_closed(monkeypatch,output):
    monkeypatch.setattr(a8.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=output))
    with pytest.raises((AssertionError,ValueError)): a8.disk_preflight()


def test_df_failure_propagates(monkeypatch):
    def fail(*args,**kwargs): raise subprocess.CalledProcessError(1,args[0])
    monkeypatch.setattr(a8.subprocess,'run',fail)
    with pytest.raises(subprocess.CalledProcessError): a8.disk_preflight()


def test_low_space_worker_stops_before_binding_or_lease(monkeypatch):
    monkeypatch.setattr(a8.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='Avail\n123\n'))
    monkeypatch.setattr(a8,'binding',lambda:pytest.fail('Low-space worker reached binding/lease'))
    with pytest.raises(AssertionError,match='123 bytes free'): a8.worker('c96_w64','census','e2e-2')


@pytest.mark.parametrize('kind',['claim','corrupt'])
def test_unregistered_damage_blocks_accounting(recovery,kind):
    path=a8.OUT/'runs/c96_w64/sup_correction_then_restatement.json'
    marker(path,recovery.reg)
    if kind=='corrupt': path.write_text('{')
    with a8.executor(),pytest.raises(AssertionError,match='campaign blocked'):
        a8.accounting(recovery.reg,reserve=False)
    assert recovery.base.controls['lease_calls']==0


@pytest.mark.parametrize('field',['incidents','budget','policy','sources','extra','sha'])
def test_forged_amendment_refused(tmp_path,monkeypatch,field):
    row=c.read(a8.AMENDMENT)
    if field=='sha':row['immutable']=False
    elif field in ('incidents','sources'):row[field]=[]
    elif field=='budget':row[field]['gpu_seconds']=99999
    else:row[field]='forged'
    path=tmp_path/'amendment.json';c.write_once(path,row)
    path.with_suffix('.sha256').write_text('stale' if field=='sha' else c.record(path)['sha256'])
    monkeypatch.setattr(a8,'AMENDMENT',path)
    with pytest.raises(AssertionError):a8.binding()


@pytest.mark.parametrize('target',['source','test','before','pre','command','archive','original','claim'])
def test_stale_bound_bytes_refused(monkeypatch,target):
    row=c.read(a8.AMENDMENT)
    paths={'source':Path(a8.__file__),'test':Path(__file__),'before':a8.OUT/'before.json',
           'pre':a8.OUT/'pre_registration.json','command':a8.COMMANDS,
           'archive':Path(row['incidents'][0]['archive']['path']),
           'original':TRUNCATED,'claim':TRUNCATED.with_suffix('.attempt.json')}
    original=c.record
    def record(path):
        pin=original(path)
        if str(path)==str(paths[target]):pin['sha256']='stale'
        return pin
    monkeypatch.setattr(c,'record',record)
    with pytest.raises(AssertionError):a8.binding()


def test_live_projection_order_admission_and_immutable_originals():
    reg=a8.binding()
    with a8.executor():
        assert a8.binding()['amendment']==reg['amendment']
        p=a8.projection(reg)
        assert p['recorded_seconds_including_incident_charges']==pytest.approx(2415.558284636121)
        assert p['remaining_estimate_seconds']==3396
        assert p['total_seconds']<6600 and p['peak_reservation_seconds']<6600
        assert a5.admission(reg,'c96_w64','census','e2e-2') is None
        receipt,prior=a5.ready(reg,'c96_w64','census','e2e-2')
        assert receipt==a8.OUT/'runs/c96_w64/census_e2e-2.json'
        assert prior[0]==a8.OLD_OUT/'runs/c96_w64/census_e2e-1.json'
    found=[]
    for line in a8.COMMANDS.read_text().splitlines():
        if ' worker ' not in line and ' skip ' not in line:continue
        args=shlex.split(line)
        found.append(tuple(args[args.index(k)+1] for k in ('--cell','--battery','--spec')))
    assert found==[(u['cell'],u['battery'],u['spec']) for u in a8.remaining(reg['a8_previous'],reg['a8_incidents'])]
    assert found[0]==('c96_w64','census','e2e-2')
    assert all(c.record(pin['path'])==pin for pin in c.read(a8.OUT/'before.json')['files'])
    assert not (a8.OUT/'runs').exists()
