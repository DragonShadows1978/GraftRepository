"""Prior art: C7 rehashed-forgery and checkpoint-boundary CPU gates
(GRM contributors, 2026). Reuse fail-closed negative tests; no new algorithm.
"""
import copy
from contextlib import nullcontext
import os
from types import SimpleNamespace

import pytest
from scripts import grm_c7_common as c, grm_c7_fix4 as f, grm_c7_run as run


def test_registration_resume_checkpoint_and_historical_charge():
    r = c.verify()
    a = r['fix4']
    assert a['resume_cell'] == 'A-024-031'
    assert a['historical_charged_seconds'] == 523.1634768834338
    assert a['historical_reserved_seconds'] == 0
    assert a['budget_seconds_arm_A'] == 7200
    cp = c.OUT/'r2/cells/A-017-023/checkpoint'
    state = f.resume_state(cp, 24, run.binding('A'), r)
    assert state['next_turn'] == 24
    with pytest.raises(ValueError, match='CHECKPOINT'):
        c.validate_checkpoint(cp, 24, run.binding('A'))
    with pytest.raises(ValueError, match='FIX4_CHECKPOINT'):
        f.resume_state(cp, 25, run.binding('A'), r)
    assert f.cell_directory(run.OUT, 'A-017-023') == cp.parent
    assert f.cell_directory(run.OUT, 'A-024-031') == f.ATTEMPT/'cells/A-024-031'
    old_failure = c.read(c.OUT/'r2/cells/A-024-031/controller.json')
    assert old_failure['status'] == 'FAILED'
    assert old_failure['charged_seconds'] == 280
    assert run.summary('A')['status'] == 'NOT_MEASURED'


@pytest.mark.parametrize('attack,error', [('scope','SCOPE'), ('chain','CHAIN'),
    ('budget','PROTOCOL'), ('cells','PROTOCOL'), ('quarantine','PROTOCOL'),
    ('before','BEFORE'), ('history','HISTORY'), ('charge','HISTORICAL_CHARGE')])
def test_forged_continuation_rejected(tmp_path, attack, error):
    r = c.verify()
    a = copy.deepcopy(r['fix4'])
    inputs = dict(r['effective_inputs'])
    for name, change in a['overrides'].items():
        inputs[name] = change['before_sha256']
    if attack == 'scope': a['overrides'].pop('core/graft_arena.py')
    elif attack == 'chain': a['previous_amendment_sha256'] = '0'*64
    elif attack == 'budget': a['budget_seconds_arm_A'] += 1
    elif attack == 'cells': a['cells'][3]['start'] += 1
    elif attack == 'quarantine': a['quarantine_memory_probe_ids'] = []
    elif attack == 'before': a['overrides']['core/graft_arena.py']['before_sha256'] = '0'*64
    elif attack == 'history': a['historical_files'][next(iter(a['historical_files']))] = '0'*64
    elif attack == 'charge': a['historical_charged_seconds'] -= 280
    path = tmp_path/'registration.json'
    c.create(path, a)
    path.with_suffix('.sha256').write_text(c.sha(path))
    with pytest.raises(ValueError, match='FIX4_'+error+'_MISMATCH'):
        f.verify_fix4(r, inputs, path)


def test_owned_child_retains_fix4_and_budget_counts_old_failures(tmp_path, monkeypatch):
    from scripts import grm_cmc1_gpu_arms as leases
    r = c.read(c.REG)
    r['effective_inputs'] = {}
    cell = next(x for x in r['cells'] if x['id'] == 'A-024-031')
    old = tmp_path/'old'
    c.create(old/'A-017-023/controller.json', {'status':'COMPLETE','charged_seconds':243.163476883434})
    c.create(old/'A-024-031/controller.json', {'status':'FAILED','charged_seconds':280})
    monkeypatch.setattr(run, 'OUT', tmp_path/'attempt')
    run.OUT.mkdir()
    monkeypatch.setattr(run, 'verify', lambda: r)
    monkeypatch.setattr(run, 'binding', lambda arm: {'arm':arm})
    monkeypatch.setattr(f, 'cell_directory', lambda out, cid: old/cid)
    monkeypatch.setattr(f, 'accounting_directories', lambda out: [old,out/'cells'])
    monkeypatch.setattr(leases, 'gpu_lease', lambda *a: nullcontext())
    monkeypatch.setattr(run.time, 'sleep', lambda _: None)
    calls, charges = [], []
    real_check = run.reserve_check
    def charge(r, cell, done, reserved):
        charges.append((done,reserved))
        real_check(r,cell,done,reserved)
    monkeypatch.setattr(run, 'reserve_check', charge)
    def child(cmd, **kw):
        calls.append((kw['env'].get('GRM_C7_FIX4'),kw['env'].get('GRM_C7_REVISION')))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(run.subprocess, 'run', child)
    assert run.run_leased(cell) == 0
    assert charges == [(523.163476883434,0)]
    assert calls == [('1','r2')]
    with pytest.raises(ValueError, match='STARTED_CELL_NEVER_RETRIED'):
        run.run_leased(cell)
