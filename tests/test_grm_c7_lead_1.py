"""Prior art: C7 A1/A2 negative binding tests and C2 manifest projection
(GRM contributors, 2026). Taken: isolated forged files and checkpoint guards.
New: SCOUT-FIX-2 authorization cases; no new verification algorithm.
"""
import copy
import shutil

import pytest

from scripts import grm_c7_common as c
from scripts import grm_c7_run as run


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    # Only artifact copies are changed; registered source/order never mutated.
    shutil.copytree(c.OUT, tmp_path/'artifacts')
    monkeypatch.setattr(c, 'OUT', tmp_path/'artifacts')
    monkeypatch.setattr(run, 'OUT', c.OUT)
    return c.OUT/'amendment_lead_1.json'


def rewrite(path, amendment):
    import json
    path.write_text(json.dumps(amendment, sort_keys=True, indent=2)+'\n')
    path.with_suffix('.sha256').write_text(c.sha(path)+'\n')


@pytest.mark.parametrize('attack,expected', [
    ('missing', 'AMENDMENT_REQUIRED'),
    ('missing_checksum', 'AMENDMENT_REQUIRED'),
    ('stale_checksum', 'SHA_MISMATCH'),
    ('registration', 'CHAIN_MISMATCH'),
    ('previous', 'CHAIN_MISMATCH'),
    ('order_hash', 'ORDER_MISMATCH'),
    ('order_path', 'ORDER_MISMATCH'),
    ('old_source', 'SOURCE_MISMATCH'),
    ('stale_source', 'SOURCE_MISMATCH'),
    ('forged_source', 'SOURCE_MISMATCH'),
    ('extra_override', 'SCOPE_MISMATCH'),
    ('missing_input', 'SCOPE_MISMATCH'),
    ('weaken_acceptance', 'ACCEPTANCE_MISMATCH'),
    ('exclude_children', 'ACCEPTANCE_MISMATCH'),
    ('before_archive', 'BEFORE_MISMATCH'),
    ('commands', 'COMMANDS_MISMATCH'),
])
def test_forged_or_stale_amendment_refused_even_when_rehashed(isolated, attack, expected):
    a = c.read(isolated)
    source = a['overrides'][c.SOURCE_PATH]
    if attack == 'missing':
        isolated.unlink()
    elif attack == 'missing_checksum':
        isolated.with_suffix('.sha256').unlink()
    elif attack == 'stale_checksum':
        isolated.write_text('{}')
    else:
        if attack == 'registration':
            a['registration_sha256'] = '0'*64
        elif attack == 'previous':
            a['previous_amendment_sha256'] = '0'*64
        elif attack == 'order_hash':
            a['order']['sha256'] = '0'*64
        elif attack == 'order_path':
            a['order']['path'] = 'orders/GRM_C7_300_TURN_RESIDENCY.md'
        elif attack == 'old_source':
            source['before_sha256'] = c.SOURCE_AFTER
        elif attack == 'stale_source':
            source['after_sha256'] = c.SOURCE_BEFORE
        elif attack == 'forged_source':
            source['after_sha256'] = '0'*64
        elif attack == 'extra_override':
            a['overrides']['scripts/grm_c7_register.py'] = copy.deepcopy(source)
        elif attack == 'missing_input':
            a['new_inputs'].pop('tests/test_grm_scout_fix2_capture.py')
        elif attack == 'weaken_acceptance':
            a['acceptance']['max_exact_answer_error_delta_per_distance'] = .99
        elif attack == 'exclude_children':
            a['registered_items']['restart_metadata_scope'] = 'parents only'
        elif attack == 'before_archive':
            source['before_archive'] = c.SOURCE_PATH
        elif attack == 'commands':
            a['lead_commands_sha256'] = '0'*64
        rewrite(isolated, a)
    # Every public receipt entry point verifies before returning a binding.
    for entry in (c.verify, run.dry_run, lambda: run.binding('A'), lambda: run.summary('A')):
        with pytest.raises(ValueError, match='LEAD_1_'+expected):
            entry()


@pytest.mark.parametrize('changed', [c.SOURCE_PATH, c.LEAD_ORDER,
                                    'artifacts/grm_c7/lead_commands.txt'])
def test_live_bytes_must_match_authorized_amendment(isolated, monkeypatch, changed):
    original_sha = c.sha
    target = c.OUT/'lead_commands.txt' if changed.endswith('lead_commands.txt') else c.ROOT/changed
    monkeypatch.setattr(c, 'sha', lambda p: '0'*64 if str(p) == str(target) else original_sha(p))
    with pytest.raises(ValueError, match='INPUT_SHA_MISMATCH|LEAD_1_ORDER_MISMATCH|LEAD_1_COMMANDS_MISMATCH'):
        run.dry_run()


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_c7/amendment_lead_1.json')
def test_receipt_and_checkpoint_bind_amended_source_and_keep_39_cells(tmp_path):
    r = c.verify()
    a = c.read(c.OUT/'amendment_lead_1.json')
    original = c.read(c.REG)
    assert r['cells'] == original['cells']
    assert r['acceptance']['restart'] == original['acceptance']['restart']
    assert a['acceptance'] == r['acceptance']
    assert a['acceptance_wording_changed'] is False
    assert r['effective_inputs'][c.SOURCE_PATH] == c.SOURCE_AFTER == c.sha(c.ROOT/c.SOURCE_PATH)
    dry = run.dry_run()
    assert len([x for x in dry['cells'] if x['arm'] == 'A']) == 39
    assert len([x for x in dry['cells'] if x['arm'] == 'B']) == 39
    assert dry['arms']['B']['status'] == 'NON_FIT'
    binding = run.binding('A')
    for receipt in (dry, binding, run.summary('A')['binding']):
        assert receipt['amendment_lead_1_sha256'] == c.sha(c.OUT/'amendment_lead_1.json')
        assert receipt['amended_source_sha256'] == c.SOURCE_AFTER
    state = {'next_turn': 9, 'process_id': 'synthetic-c7-lead-1'}
    c.checkpoint(tmp_path, state, binding)
    assert c.validate_checkpoint(tmp_path, 9, binding) == state
    for field in ('amended_source_sha256', 'amendment_lead_1_sha256'):
        stale = dict(binding, **{field: '0'*64})
        with pytest.raises(ValueError, match='CHECKPOINT'):
            c.validate_checkpoint(tmp_path, 9, stale)
    historical = {k: v for k, v in binding.items()
                  if k not in ('amended_source_sha256', 'amendment_lead_1_sha256')}
    with pytest.raises(ValueError, match='CHECKPOINT'):
        c.validate_checkpoint(tmp_path, 9, historical)


def test_restart_projection_detects_lost_split_child_capture(tmp_path):
    from scripts.grm_c2_cells import manifest_projection
    from tests.test_grm_scout_fix2_capture import CAPTURE, expected, repository
    repo = repository(tmp_path)
    parent = repo.add_document('A1 B2 C3 D4 E5 F6')
    children = repo.cull_graft(parent, max_tokens=3)['children']
    repo.save()
    persisted = c.read(tmp_path/'manifest.json')['nodes']
    loaded_repo = repository(tmp_path)
    loaded = [loaded_repo._node_manifest(g) for g in loaded_repo.arena.grafts]
    assert manifest_projection(persisted) == manifest_projection(loaded)
    for child in children:
        assert loaded[child]['capture'] == expected(CAPTURE, parent)
        corrupt = copy.deepcopy(loaded)
        corrupt[child].pop('capture')
        assert manifest_projection(persisted) != manifest_projection(corrupt)
        corrupt = copy.deepcopy(loaded)
        corrupt[child]['capture']['capture_parent_graft_id'] = -1
        assert manifest_projection(persisted) != manifest_projection(corrupt)
