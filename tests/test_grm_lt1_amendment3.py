"""Author CPU adversarial source-chain contracts.
Prior art: GRM contributors, C7/LT1 checksum, scope and checkpoint mutation
contracts (2026), verified local source; reuse temporary corrupt copies.
No prior art known to me for exact cases. Not independent blind review.
"""
import json
from pathlib import Path
import pytest
from scripts import grm_lt1 as lt
from scripts import grm_lt1_worker as worker
from scripts.grm_lt1_amendment3 import PROTOCOL


def amended(tmp_path, monkeypatch, mutate, rehash=True):
    a = lt.read(lt.AMEND3)
    digest = lt.sha(lt.AMEND3)
    mutate(a)
    p = tmp_path/'amendment.json'
    lt.create(p, a)
    p.with_suffix('.sha256').write_text((lt.sha(p) if rehash else digest)+'  amendment.json\n')
    monkeypatch.setattr(lt, 'AMEND3', p)


def test_amendment3_preserves_protocol_and_binds_all_core():
    r = lt.verify(); a = lt.read(lt.AMEND3); previous = lt.read(lt.AMEND2)
    assert a['protocol'] == {k:previous[k] for k in PROTOCOL}
    assert len(a['core_shas']) == 26
    assert set(lt.binding('A')['core_shas']) == {n for n in r['effective_inputs'] if n.startswith('core/')}
    assert all(lt.binding('A')['core_shas'][n] == lt.sha(lt.ROOT/n) for n in a['core_shas'])
    assert r['cells'] == previous['cells']
    # Amendment1 already changed effective status; stored status stays immutable.
    assert r['arms'] == {k:dict(v,status='FIT_ESTIMATE') for k,v in previous['arms'].items()}
    assert lt.sha(lt.ROOT/'fixtures/lt1/manifest.json') == previous['fixture_manifest_sha256']


def test_forged_amendment_checksum_refused(tmp_path, monkeypatch):
    amended(tmp_path, monkeypatch, lambda a:a.update(status='forged'), rehash=False)
    with pytest.raises(ValueError, match='AMENDMENT3_SHA_MISMATCH'): lt.verify()


@pytest.mark.parametrize('field', ['registration_sha256', 'previous_amendment_sha256', 'order_sha256', 'supersedes_amendment_sha256'])
def test_rehashed_stale_amendment_chain_refused(tmp_path, monkeypatch, field):
    amended(tmp_path, monkeypatch, lambda a:a.update({field:'0'*64}))
    with pytest.raises(ValueError, match='AMENDMENT3_CHAIN_MISMATCH'): lt.verify()


@pytest.mark.parametrize('field,value', [('admission_rule','all_tokens'), ('budget_gpu_seconds',20000),
    ('cells',[]), ('arms',{}), ('fixture_manifest_sha256','0'*64)])
def test_rehashed_protocol_forgery_refused(tmp_path, monkeypatch, field, value):
    amended(tmp_path, monkeypatch, lambda a:a['protocol'].update({field:value}))
    with pytest.raises(ValueError, match='AMENDMENT3_PROTOCOL_MISMATCH'): lt.verify()


@pytest.mark.parametrize('field', ['before_sha256', 'after_sha256'])
def test_rehashed_core_pin_forgery_refused(tmp_path, monkeypatch, field):
    amended(tmp_path, monkeypatch, lambda a:a['core_shas']['core/graft_arena.py'].update({field:'0'*64}))
    with pytest.raises(ValueError, match='AMENDMENT3_CORE_PIN_MISMATCH'): lt.verify()


def test_rehashed_core_omission_refused(tmp_path, monkeypatch):
    amended(tmp_path, monkeypatch, lambda a:a['core_shas'].pop('core/grm_native.py'))
    with pytest.raises(ValueError, match='AMENDMENT3_CORE_SCOPE_MISMATCH'): lt.verify()


def test_rehashed_override_forgery_refused(tmp_path, monkeypatch):
    amended(tmp_path, monkeypatch, lambda a:a['overrides']['core/graft_arena.py'].update(after_sha256='0'*64))
    with pytest.raises(ValueError, match='AMENDMENT3_SOURCE_PIN_MISMATCH'): lt.verify()


def test_rehashed_fixture_override_refused(tmp_path, monkeypatch):
    amended(tmp_path, monkeypatch, lambda a:a['overrides'].update({'fixtures/lt1/dialogue.json':{}}))
    with pytest.raises(ValueError, match='AMENDMENT3_OVERRIDE_SCOPE_MISMATCH'): lt.verify()


def test_rehashed_new_input_shadow_refused(tmp_path, monkeypatch):
    amended(tmp_path, monkeypatch, lambda a:a['new_inputs'].update({'core/graft_arena.py':'0'*64}))
    with pytest.raises(ValueError, match='AMENDMENT3_NEW_INPUT_SCOPE_MISMATCH'): lt.verify()


def test_missing_amendment_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(lt, 'AMEND3', tmp_path/'absent.json')
    with pytest.raises(ValueError, match='AMENDMENT3_REQUIRED'): lt.verify()


def test_stale_working_source_refused(monkeypatch):
    real = lt.sha
    monkeypatch.setattr(lt, 'sha', lambda p:'0'*64 if Path(p)==lt.ROOT/'core/graft_arena.py' else real(p))
    with pytest.raises(ValueError, match='INPUT_SHA_MISMATCH: core/graft_arena.py'): lt.verify()


def test_old_checkpoint_binding_refused(tmp_path):
    (tmp_path/'repository').mkdir()
    old = lt.binding('A'); old['amendment_sha256'] = lt.sha(lt.AMEND2); old.pop('core_shas')
    lt.checkpoint(tmp_path, dict(next_turn=9, process_id='old'), old)
    with pytest.raises(ValueError, match='CHECKPOINT_INTEGRITY'): lt.validate_checkpoint(tmp_path,9,lt.binding('A'))


def test_preflight_requires_current_receipt_and_fix4(tmp_path, monkeypatch):
    realout=lt.OUT; monkeypatch.setenv('GRM_ADMISSION_RULE','margin_first')
    # Only receipt lookup moves; registration and all source paths stay real.
    monkeypatch.setattr(lt, 'OUT', tmp_path)
    p=tmp_path/'amendment3/r3/cpu_receipt.json'
    assert 'CPU_GATES_NOT_GREEN' in lt.preflight()['reasons']
    lt.create(p, dict(status='PASS', binding=lt.binding('CPU'), fix4_check='RED'))
    assert 'FIX4_PREREQUISITE_NOT_GREEN' in lt.preflight()['reasons']
    data=lt.read(p); data['fix4_check']='PASS'; data['binding']['amendment_sha256']=lt.sha(lt.AMEND2)
    p.write_text(json.dumps(data))
    assert 'CPU_RECEIPT_BINDING_MISMATCH' in lt.preflight()['reasons']
    data['binding']=lt.binding('CPU');p.write_text(json.dumps(data))
    assert lt.preflight()['status']=='READY'


def test_dry_run_never_enters_worker(monkeypatch, capsys):
    monkeypatch.setattr(lt, 'preflight', lambda:dict(status='READY'))
    monkeypatch.setattr(worker, 'resume', lambda:pytest.fail('GPU worker entered'))
    monkeypatch.setattr(lt.sys, 'argv', ['lt1','--dry-run'])
    assert lt.main()==0
    assert 'CPU_DRY_RUN' in capsys.readouterr().out
