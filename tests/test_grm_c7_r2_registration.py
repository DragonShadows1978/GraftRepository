"""Prior art: local C7 immutable-chain negative tests (GRM, 2026).
Borrow rehashed-forgery tests and stale checkpoint controls; use small
amendment-only copies so no r1 model payload is copied or changed.
"""
import copy
import json
import shutil

import pytest

from scripts import grm_c7_common as c
from scripts import grm_c7_run as run
from scripts import grm_c7_register_r2 as r2
from scripts import grm_c7_fix4 as fix4


def test_r2_layout_bindings_fixtures_and_no_r1_resume(tmp_path):
    r = c.verify()
    a = c.read(c.OUT/'r2/amendment.json')
    assert run.OUT == (fix4.ATTEMPT if fix4.enabled() else c.OUT/'r2')
    assert r['cells'] == c.read(c.REG)['cells'] == a['cells']
    assert len([x for x in r['cells'] if x['arm'] == 'A']) == 39
    assert a['fixture_sha256'] == c.sha(c.FIX)
    assert a['acceptance'] == c.read(c.OUT/'amendment_lead_1.json')['acceptance']
    b = run.binding('A')
    assert r['amendment_r2_base_sha256'] == c.sha(c.OUT/'r2/amendment.json')
    assert b['amendment_r2_sha256'] == c.sha(c.OUT/'r2/launch_amendment/amendment.json')
    assert b['fold_core_sha256'] == c.sha(c.ROOT/'core/graft_arena.py')
    state = {'next_turn': 9, 'process_id': 'cpu-r2'}
    c.checkpoint(tmp_path, state, b)
    assert c.validate_checkpoint(tmp_path, 9, b) == state
    stale = {k:v for k,v in b.items() if k not in ('revision','amendment_r2_sha256','fold_core_sha256')}
    with pytest.raises(ValueError, match='CHECKPOINT'):
        c.validate_checkpoint(tmp_path, 9, stale)
    # FIX-4 explicitly imports three completed cells, with old rows quarantined.
    assert run.summary('A')['status'] == ('NOT_MEASURED' if fix4.enabled() else 'INCOMPLETE')
    assert run.dry_run()['receipt_directory'] == str(run.OUT)


@pytest.mark.parametrize('attack,error', [
    ('missing','AMENDMENT_REQUIRED'), ('checksum','SHA_MISMATCH'),
    ('previous','CHAIN_MISMATCH'), ('scope','SCOPE_MISMATCH'),
    ('acceptance','PROTOCOL_MISMATCH'), ('cells','PROTOCOL_MISMATCH'),
    ('budget','PROTOCOL_MISMATCH'), ('directory','PROTOCOL_MISMATCH'),
    ('archive','BEFORE_MISMATCH'), ('core','CORE_MISMATCH'),
    ('questions','QUESTION_MISMATCH'),
])
def test_r2_forged_amendment_fails_closed(tmp_path, attack, error):
    r = c.verify()
    inputs = dict(r['effective_inputs'])
    a = c.read(c.OUT/'r2/amendment.json')
    for name, change in a['overrides'].items():
        inputs[name] = change['before_sha256']
    for name in a['new_inputs']:
        inputs.pop(name, None)
    (tmp_path/'r2').mkdir()
    for name in ('registration.json','fixture.json','fixture_manifest.json'):
        shutil.copyfile(c.OUT/name, tmp_path/name)
    shutil.copyfile(c.OUT/'r2/effective_questions.json', tmp_path/'r2/effective_questions.json')
    if attack == 'previous': a['previous_amendment_sha256'] = '0'*64
    elif attack == 'scope': a['overrides'].pop('core/graft_arena.py')
    elif attack == 'acceptance': a['acceptance']['coverage'] = .01
    elif attack == 'cells': a['cells'][0]['stop'] += 1
    elif attack == 'budget': a['budget_seconds_arm_A'] += 1
    elif attack == 'directory': a['receipt_directory'] = 'artifacts/grm_c7'
    elif attack == 'archive': a['overrides']['core/graft_arena.py']['before_archive'] = 'core/graft_arena.py'
    elif attack == 'core': a['core_sha256']['core/graft_arena.py'] = '0'*64
    elif attack == 'questions': (tmp_path/'r2/effective_questions.json').write_text('[]')
    path = tmp_path/'r2/amendment.json'
    c.create(path,a)
    path.with_suffix('.sha256').write_text(c.sha(path))
    if attack == 'missing': path.unlink()
    elif attack == 'checksum': path.write_text('{}')
    with pytest.raises(ValueError, match='R2_'+error):
        r2.verify_r2(copy.deepcopy(r), inputs, c.ROOT, tmp_path)


@pytest.mark.parametrize('name', sorted(r2.CHANGED | {'scripts/grm_c7_register_r2.py'}))
def test_live_r2_source_tampering_is_rejected(monkeypatch, name):
    original = c.sha
    target = c.ROOT/name
    monkeypatch.setattr(c, 'sha', lambda p: '0'*64 if p == target else original(p))
    with pytest.raises(ValueError, match='INPUT_SHA_MISMATCH'):
        c.verify()


def test_r1_execution_fails_closed_after_core_change(monkeypatch):
    monkeypatch.delenv('GRM_C7_REVISION', raising=False)
    monkeypatch.delenv('GRM_C7_FIX4', raising=False)
    with pytest.raises(ValueError, match='INPUT_SHA_MISMATCH'):
        c.verify()
