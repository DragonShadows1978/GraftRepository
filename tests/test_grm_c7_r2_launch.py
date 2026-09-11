"""Prior art: C7 owned-child lease marker and C2 environment isolation
(GRM contributors, 2026). Exercise the real controller with a fake lease
and owned-child launcher; no GPU, external process or sleep is performed.
New regression: campaign revision survives filtering; no new algorithm.
"""
from contextlib import nullcontext
import os
from types import SimpleNamespace
import pytest

from scripts import grm_c7_common as common
from scripts import grm_c7_run as run
from scripts import grm_cmc1_gpu_arms as leases
from scripts import grm_c7_register_r2 as register


def test_leased_child_keeps_r2_after_c2_environment_filter(tmp_path, monkeypatch):
    r = common.read(common.REG)
    cell = r['cells'][0]
    r['effective_inputs'] = {}
    monkeypatch.setenv('GRM_C7_REVISION', 'r2')
    monkeypatch.setenv('GRM_UNREGISTERED_SWITCH', 'must-be-removed')
    monkeypatch.setattr(run, 'OUT', tmp_path)
    monkeypatch.setattr(run, 'verify', lambda: r)
    monkeypatch.setattr(run, 'binding', lambda arm: {'revision':'r2','arm':arm})
    monkeypatch.setattr(leases, 'gpu_lease', lambda *args: nullcontext())
    monkeypatch.setattr(run.time, 'sleep', lambda seconds: None)
    calls = []
    def child(cmd, **kwargs):
        # Keep unrelated host environment (potential credentials) out of
        # pytest failure reprs and receipts; observe only campaign markers.
        env = kwargs['env']
        calls.append((cmd, {'revision':env.get('GRM_C7_REVISION'),
                           'parent':env.get('GRM_C7_LEASE_PARENT'),
                           'unregistered_present':'GRM_UNREGISTERED_SWITCH' in env}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(run.subprocess, 'run', child)
    assert run.run_leased(cell) == 0
    assert len(calls) == 1
    cmd, env = calls[0]
    assert cmd[-2:] == ['--worker', cell['id']]
    assert env['revision'] == 'r2'
    assert env['parent'] == str(os.getpid())
    assert not env['unregistered_present']
    assert not (tmp_path/'A.active').exists()
    assert common.read(tmp_path/'cells'/cell['id']/'controller.json')['status'] == 'COMPLETE'


@pytest.mark.parametrize('attack,error', [('checksum','SHA'), ('chain','CHAIN'),
                                        ('scope','SCOPE'), ('before','BEFORE')])
def test_launch_amendment_rehashed_forgery_rejected(tmp_path, attack, error):
    r = common.verify()
    path = common.OUT/'r2/launch_amendment/amendment.json'
    a = common.read(path)
    inputs = dict(r['effective_inputs'])
    for name, change in a['overrides'].items():
        inputs[name] = change['before_sha256']
    for name in a['new_inputs']:
        inputs.pop(name, None)
    r['amendment_r2_sha256'] = r['amendment_r2_base_sha256']
    if attack == 'chain': a['previous_amendment_sha256'] = '0'*64
    elif attack == 'scope': a['overrides'].pop('scripts/grm_c7_run.py')
    elif attack == 'before': a['overrides']['scripts/grm_c7_run.py']['before_sha256'] = '0'*64
    copied = tmp_path/'amendment.json'
    common.create(copied, a)
    copied.with_suffix('.sha256').write_text(common.sha(copied))
    if attack == 'checksum': copied.write_text('{}')
    with pytest.raises(ValueError, match='R2_LAUNCH_'+error+'_MISMATCH'):
        register.verify_launch(r, inputs, copied)
