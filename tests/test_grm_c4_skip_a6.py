"""CPU author gates; no GPU or blind-review claim.
Prior art: local A5 synthetic lease/state/receipt tests (house, 2026), reused;
A6 adds skip accounting, cap boundaries and forbidden executed-skip attacks.
"""
import copy
import json
from pathlib import Path
import shlex

import pytest
from scripts import grm_c4_campaign as c
from scripts import grm_c4_split_a5 as a5
from scripts import grm_c4_skip_a6 as a6
from test_grm_c4_split_a5 import layout, run, seed_legacy


@pytest.fixture
def a6_layout(layout, monkeypatch):
    # Reuse A5's fake lease/session machinery, restore the REAL A6 skip plan
    # after A5's full-resume fixture overrides it. All writes stay in tmp_path.
    row = c.read(a6.AMENDMENT)
    layout.reg.update(a5_plan=copy.deepcopy(row['plan']), budget=copy.deepcopy(row['budget']),
                      amendment=c.record(a6.AMENDMENT), budget_amendment=c.record(a6.AMENDMENT),
                      sources=copy.deepcopy(row['sources']))
    monkeypatch.setattr(a6, 'binding', lambda: layout.reg)
    with a6.executor():
        yield layout


def test_registered_projection_commands_and_unchanged_rules():
    reg = a6.binding(); old = a6.LEGACY_BINDING(); plan = reg['a5_plan']
    assert reg['budget'] == {**old['budget'], 'gpu_seconds': 6600}
    assert reg['units_per_new_cell'] == old['units_per_new_cell']
    assert reg['legacy_context'] == old['legacy_context']
    assert len(a5.schedule(reg)) == 50
    assert [(u['cell'],u['spec']) for u in plan['skip_units']] == [
        (cell,spec) for cell in a5.CELLS for spec in ('c4-lh-12a','c4-lh-13b')]
    assert [u['estimate_seconds'] for u in plan['skip_units']] == [284,348]*3
    assert len(plan['dependency_blocked_fitting_units']) == 6
    p = plan['projection']
    assert p['recorded_seconds'] == pytest.approx(2157.6785489856265)
    assert p['registered_seconds'] == 3845
    assert p['total_seconds'] == pytest.approx(6002.6785489856265)
    assert p['headroom_seconds'] == pytest.approx(597.3214510143735)
    assert p['skipped_estimate_seconds'] == 1896
    assert p['status'] == 'FIT' and p['overflow_seconds'] == 0
    assert p['peak_reservation_seconds'] <= 6600
    units = a5.schedule(reg)
    assert all(u['cap_status']=='FIT' for u in units if u['dispatch']=='WORKER')
    assert a6.COMMANDS.read_text() == a6.command_text(units)
    lines = a6.COMMANDS.read_text().splitlines(); found=[]
    for i,line in enumerate(lines):
        if ' worker ' not in line and ' skip ' not in line: continue
        args=shlex.split(line)
        if 'skip' in args:
            assert args[:3] == ['python','scripts/grm_c4_skip_a6.py','skip']
            assert lines[i+1] != 'sleep 30'
        else:
            assert args[:3] == ['timeout','--signal=KILL','590s']
            assert lines[i+1] == 'sleep 30'
        found.append(tuple(args[args.index(k)+1] for k in ('--cell','--battery','--spec')))
    assert found == [(u['cell'],u['battery'],u['spec']) for u in units]
    assert sum(' skip ' in x for x in lines) == 6
    assert sum(' worker ' in x for x in lines) == 44
    assert sum(' score ' in x for x in lines) == 3
    assert lines[-1] == 'python scripts/grm_c4_skip_a6.py summary'


@pytest.mark.parametrize('field', ['sha','order','previous_amendment','registration',
    'plan','budget','previous_budget','sources','executor','receipt_root','policy','extra'])
def test_forged_or_stale_amendment_refused(tmp_path, monkeypatch, field):
    row=c.read(a6.AMENDMENT)
    if field=='sha': row['immutable']=False
    elif field=='plan': row['plan']['scheduled_units'][0]['dispatch']='WORKER'
    elif field in ('budget','previous_budget'): row[field]['gpu_seconds']=99999
    elif field=='sources': row[field]=[]
    elif field in ('receipt_root','policy','extra'): row[field]='forged'
    else: row[field]['sha256']='forged'
    path=tmp_path/'amendment.json';c.write_once(path,row)
    path.with_suffix('.sha256').write_text('stale' if field=='sha' else c.record(path)['sha256'])
    monkeypatch.setattr(a6,'AMENDMENT',path)
    with pytest.raises(AssertionError): a6.binding()


@pytest.mark.parametrize('target', ['source','order','predecessor','executor','commands'])
def test_stale_bound_bytes_refused(monkeypatch,target):
    paths={'source':Path(a6.__file__),'order':a6.ORDER,'predecessor':a5.AMENDMENT,
           'executor':Path(a5.__file__),'commands':a6.COMMANDS}
    original=c.record
    def changed(path):
        pin=original(path)
        if str(path)==str(paths[target]): pin['sha256']='stale'
        return pin
    monkeypatch.setattr(c,'record',changed)
    with pytest.raises(AssertionError): a6.binding()


def test_all_registered_skips_zero_attempt_lease_reserve_or_charge(a6_layout,monkeypatch):
    def forbidden(*a,**k): pytest.fail('Skip reached reservation/readiness/harness')
    for method in ('ready','accounting','split_harness'):
        monkeypatch.setattr(a5,method,forbidden)
    for u in a6_layout.reg['a5_plan']['skip_units']:
        row=a5.worker(u['cell'],u['battery'],u['spec'])
        path=a5.receipt_path(u['cell'],u['battery'],u['spec'])
        assert row['status']=='NON_FIT' and row['gpu_executed'] is False
        assert row['gpu_seconds']==0 and 'finished_unix' not in row
        assert not path.with_suffix('.attempt.json').exists()
        a6.validate_receipt(path,a6_layout.reg,u['cell'],u['battery'],u['spec'])
        with pytest.raises(AssertionError,match='No retries'):
            a5.worker(u['cell'],u['battery'],u['spec'])
    assert a6_layout.controls['lease_calls']==0


def test_all_fitting_units_dispatched_dependencies_preserved_partial_summary(a6_layout):
    pins=seed_legacy(a6_layout)
    before=a5.accounting(a6_layout.reg,reserve=False)
    for u in a5.schedule(a6_layout.reg):
        a6_layout.clock[0]+=1000
        a5.worker(u['cell'],u['battery'],u['spec'])
    assert a6_layout.controls['lease_calls']==38 #19 reachable units in each new cell
    for cell in a5.CELLS:
        for spec in ('c4-lh-12b','c4-lh-13a'):
            path=a5.receipt_path(cell,'longhorizon',spec)
            row=c.read(path)
            assert row['status']=='NON_FIT' and 'prior c4-lh-12a NON_FIT' in row['reason']
            assert row['gpu_seconds']==0 and not path.with_suffix('.attempt.json').exists()
        result=a5.score(cell)
        assert result['status']=='NON_FIT' and len(result['completed_segments'])==19
        assert len(result['failed_segments'])==4 and not result['missing_segments']
        assert 'scores' not in result
    result=a5.cross_summary()
    assert result['factorial_verdict']=='INCONCLUSIVE' and result['prediction_met'] is None
    assert all('scores' not in result['cells'][cell] for cell in a5.CELLS)
    assert a5.format_summary(result).count('NON_FIT (completed:')==3
    assert all(c.record(p['path'])==p for p in pins)
    new=[c.read(p) for p in (a5.OUT/'runs').glob('*/*.json')
         if p.name!='score.json' and not p.name.endswith('.attempt.json')]
    assert a5.accounting(a6_layout.reg,reserve=False)==pytest.approx(
        before+sum(r['gpu_seconds'] for r in new))


@pytest.mark.parametrize('attack', ['pass','charge','attempt','source','reason'])
def test_forged_skip_receipt_refused(a6_layout,attack):
    cell='c64_w96';spec='c4-lh-12a'
    row=a5.worker(cell,'longhorizon',spec);path=a5.receipt_path(cell,'longhorizon',spec)
    if attack=='pass': row['status']='PASS'
    elif attack=='charge': row['gpu_seconds']=285
    elif attack=='attempt': c.write_once(path.with_suffix('.attempt.json'),{})
    elif attack=='source': row['sources']=[]
    else: row['reason']='forged'
    path.write_text(json.dumps(row))
    with pytest.raises(AssertionError): a5.cross_summary()
    assert not (a5.OUT/'cross_summary.json').exists()


@pytest.mark.parametrize('attack', ['scores','completed','amendment'])
def test_forged_or_stale_partial_score_refused(a6_layout,attack):
    cell='c96_w64';row=a5.score(cell);path=a5.score_path(cell)
    if attack=='scores': row['scores']=a6_layout.reg['counts']
    elif attack=='completed': row['completed_segments']=['fake']
    else: row['amendment']={}
    path.write_text(json.dumps(row))
    with pytest.raises(AssertionError): a5.cross_summary()


@pytest.mark.parametrize('used,accepted', [(6315,True),(6315.001,False)])
def test_actual_cap_reserves_285_at_6600(a6_layout,used,accepted):
    run(a6_layout,'c96_w64',a6_layout.reg['units_per_new_cell'][:1])
    path=a5.receipt_path('c96_w64','sup',c.SUP[0]);row=c.read(path)
    row['gpu_seconds']=used;path.write_text(json.dumps(row));a6_layout.clock[0]+=1000
    if accepted: assert a5.accounting(a6_layout.reg)==used
    else:
        with pytest.raises(AssertionError,match='budget non-fit'): a5.accounting(a6_layout.reg)


def test_existing_artifacts_byte_unchanged_and_no_real_attempts():
    pins=c.read(a6.OUT/'before.json')['files']
    assert len(pins)>2567
    assert all(c.record(p['path'])==p for p in pins)
    assert not (a5.OUT/'runs').exists()
    assert not (a6.OUT/'runs').exists()
    a6.LEGACY_BINDING();a6.binding()
