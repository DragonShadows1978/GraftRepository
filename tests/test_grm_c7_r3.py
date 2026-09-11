"""Prior art: GRM contributors (2026), C7 CPU doubles and exact contracts.
Reuse production mutation/oracle calls and immutable-byte checks; new r3
stimuli only. No prior art known to me for this exact suite. No GPU.
"""
import json
import os
from pathlib import Path

import pytest

from scripts import grm_c7_common as common
from scripts import grm_c7_run as run

BASE = common.OUT
R3 = BASE / 'r3'


def fixture():
    path = R3/'fixture.json'
    return common.read(path if path.exists() else BASE/'fixture.json')


def test_turn2_real_correction():
    f = fixture()
    event = f['turns'][1]
    assert event['kind'] == 'supersede'
    assert event['old_value'] == 'Mica-431'
    query, replacement = event['correction_command'].split(' => ')
    assert query == 'correct memory: current C7-Vesper-0 value is Mica-431'
    assert replacement == event['user']
    assert 'Flint-512' in replacement


def test_plain_questions(monkeypatch):
    monkeypatch.setenv('GRM_C7_REVISION', 'r3')
    for p in fixture()['probes']:
        expected = p['question'].replace('; if unspecified, reply UNKNOWN.', '.')
        assert run.effective_question(p['question']) == expected
        assert run.probe_row(p, p['turn'], {}, {})['question'] == expected


@pytest.mark.parametrize('answer', ['unknown', 'Unknown', 'uNkNoWn'])
def test_control_casefold(answer):
    assert common.score(answer, {'answerable':False,'expected':'UNKNOWN'})['exact_correct']


def test_answerable_case_sensitive_and_no_substring_credit():
    for answer in ['basalt-811','UNKNOWN','Basalt-810 Basalt-811']:
        assert not common.score(answer, {'answerable':True,'expected':'Basalt-811'})['exact_correct']
    for answer in ['unknown Jasper-711','I do not know']:
        assert not common.score(answer, {'answerable':False,'expected':'UNKNOWN'})['exact_correct']


def test_correction_real_repository_and_reload(tmp_path, monkeypatch):
    from scripts.grm_c7_diagnose import repository
    repo = repository(tmp_path/'repo', monkeypatch)
    f = fixture()
    try:
        original = repo.add_document(f['turns'][0]['user'])
        event = f['turns'][1]
        repo.add_document(event['user'])
        assert event['kind'] == 'supersede'
        repo.apply_memory_command(event['correction_command'])
        assert repo.arena.grafts[original]['retired']
        replacements = [g for g in repo.arena.grafts
                        if original in g.get('metadata', {}).get('supersedes', [])]
        assert replacements
        assert all('Flint-511' in g['text'] and 'Flint-512' in g['text'] for g in replacements)
        for event in f['turns'][2:]:
            if event['kind'] == 'supersede' and event['fact_id'].startswith('C7-Vesper-'):
                repo.add_document(event['user'])
                repo.apply_memory_command(event['correction_command'])
        active = '\n'.join(g['text'] for g in repo.arena.grafts if not g.get('retired'))
        assert 'Mica-431' not in active and 'Mica-432' not in active
        assert 'Onyx-911' in active and 'Onyx-912' in active
        repo.flush_now()
        before = common.read(tmp_path/'repo/manifest.json')['nodes']
    finally:
        repo.close()
    loaded = repository(tmp_path/'repo', monkeypatch)
    try:
        from scripts.grm_c2_cells import manifest_projection
        assert manifest_projection(before) == manifest_projection(
            [loaded._node_manifest(g) for g in loaded.arena.grafts])
    finally:
        loaded.close()


def turn_spans(path):
    raw = path.read_text()
    offset = raw.index('[', raw.index('"turns":')) + 1
    spans = []
    decoder = json.JSONDecoder()
    while True:
        while raw[offset] in ' \n\t,':
            offset += 1
        if raw[offset] == ']':
            return spans
        _, end = decoder.raw_decode(raw, offset)
        spans.append(raw[offset:end].encode())
        offset = end


def test_fixture_exact_scope_and_all_aliases_retained():
    from scripts.grm_c7_register_r3 import fixture_bytes
    original = common.read(BASE/'fixture.json')
    changed = fixture()
    a, b = turn_spans(BASE/'fixture.json'), turn_spans(R3/'fixture.json')
    assert len(a) == len(b) == 300
    assert [i+1 for i,(x,y) in enumerate(zip(a,b)) if x != y] == [2]
    for key in original:
        if key != 'turns':
            assert changed[key] == original[key]
    assert (R3/'fixture.json').read_bytes() == fixture_bytes()
    assert len(changed['probes']) == 60
    for cls in common.CLASSES:
        assert sum(p['class']==cls for p in changed['probes']) == 15
        assert sum(p['class']==cls and p['answerable'] for p in changed['probes']) == 10


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_c7/r3/registration.json')
def test_registration_and_budget():
    r = common.verify()
    assert len(r['cells']) == 39 and set(r['arms']) == {'A'}
    assert r['budget_seconds_arm_A'] == 7920
    assert r['arms']['A']['flags']['arena_width'] == 96
    assert r['fixes'] == {'FIX-3':True,'FIX-4':True,'FIX-5':True,'FIX-7':False}
    assert run.OUT == R3 and run.FIX == R3/'fixture.json'
    assert run.binding('A')['registration_sha256'] == common.sha(R3/'registration.json')
    cell = r['cells'][0]
    run.reserve_check(r, cell, 7640, 0)
    with pytest.raises(ValueError, match='BUDGET_RAIL'):
        run.reserve_check(r, cell, 7640, 1)


@pytest.mark.parametrize('free,passes', [(19_999_999_999,False),(20_000_000_000,True),(21_000_000_000,True)])
def test_preflight_boundary(tmp_path, monkeypatch, free, passes):
    from scripts import grm_c7_register_r3 as reg
    from types import SimpleNamespace
    monkeypatch.setattr(reg.shutil, 'disk_usage', lambda p: SimpleNamespace(free=free))
    if passes:
        assert reg.preflight(tmp_path)['pass']
    else:
        with pytest.raises(ValueError, match='R3_FREE_SPACE_RED'):
            reg.preflight(tmp_path)


@pytest.mark.campaign_receipt(registration='artifacts/grm_c7/registration.json (immutable_inputs pins absolute /mnt/ForgeRealm/wt/grm-c3/orders/GRM_C3_DNGH_DECOY_CALIBRATION.md, pruned with the grm-c3 fork)')
@pytest.mark.parametrize('attack', ['checksum','input','budget','cells','fix7'])
def test_registration_tamper(tmp_path, attack):
    from scripts import grm_c7_register_r3 as reg
    r = common.read(reg.REG)
    if attack == 'budget': r['budget_seconds_arm_A'] = 7921
    if attack == 'cells': r['cells'].pop()
    if attack == 'fix7': r['fixes']['FIX-7'] = True
    if attack == 'input':
        p=tmp_path/'input.txt'; p.write_text('before')
        r['immutable_inputs'][str(p)] = common.sha(p)
        p.write_text('after')
    path = tmp_path/'registration.json'
    common.create(path, r)
    path.with_suffix('.sha256').write_text(common.sha(path))
    if attack == 'checksum': path.write_text('{}')
    with pytest.raises(ValueError, match='R3_'):
        reg.verify_r3(path)


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_c7/r3/registration.json')
def test_resume_complete_only_with_checkpoint(tmp_path):
    from scripts import grm_c7_register_r3 as reg
    r = common.verify(); cell = r['cells'][0]; binding=run.binding('A')
    assert reg.resume_check(cell['id'], tmp_path)['status'] == 'UNSTARTED'
    target=tmp_path/'cells'/cell['id']; target.mkdir(parents=True)
    with pytest.raises(ValueError, match='ORPHAN'):
        reg.resume_check(cell['id'], tmp_path)
    common.create(target/'controller.json', {'status':'FAILED','cell':cell,'binding':binding})
    with pytest.raises(ValueError, match='INCOMPLETE'):
        reg.resume_check(cell['id'], tmp_path)
    (target/'controller.json').write_text(json.dumps({'status':'COMPLETE','cell':cell,'binding':binding}))
    cp=target/'checkpoint'; cp.mkdir()
    common.checkpoint(cp, {'next_turn':9,'process_id':'test-cpu'}, binding)
    common.create(target/'worker.json', {'cell':cell,'binding':binding,
        'checkpoint_sha256':common.sha(cp/'checkpoint.json')})
    assert reg.resume_check(cell['id'], tmp_path)['status'] == 'COMPLETE'
    (cp/'state.json').write_text('{}')
    with pytest.raises(ValueError, match='CHECKPOINT'):
        reg.resume_check(cell['id'], tmp_path)


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_c7/r3/registration.json')
def test_r3_child_keeps_revision_and_no_alias_switch(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from scripts import grm_c7_register_r3 as reg
    from scripts import grm_cmc1_gpu_arms as leases
    r=common.verify()
    monkeypatch.setattr(run, 'OUT', tmp_path)
    monkeypatch.setattr(reg, 'check_ready', lambda:r)
    monkeypatch.setattr(reg, 'preflight', lambda:None)
    monkeypatch.setattr(leases, 'gpu_lease', lambda *a:nullcontext())
    monkeypatch.setattr(run.time, 'sleep', lambda s:None)
    observed=[]
    def child(cmd, **kwargs):
        env=kwargs['env']
        observed.append({k:v for k,v in env.items() if k.startswith('GRM_')})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(run.subprocess, 'run', child)
    assert run.run_leased(r['cells'][0]) == 0
    assert len(observed) == 1
    assert observed[0]['GRM_C7_REVISION'] == 'r3'
    assert 'GRM_ALIAS_FOLLOW' not in observed[0]
    assert 'GRM_C7_FIX4' not in observed[0]
    assert observed[0]['GRM_C7_LEASE_PARENT'] == str(os.getpid())


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_c7/r3/registration.json')
def test_oracle_plain_source_and_answerability(tmp_path, monkeypatch):
    from scripts.grm_c7_diagnose import repository
    repo=repository(tmp_path/'repo', monkeypatch)
    try:
        for p in fixture()['probes']:
            upper=run.oracle(repo.arena,p,32)
            expected=p['question'].replace('; if unspecified, reply UNKNOWN.', '.')
            assert upper['question'] == expected
            assert upper['live_prompt'].endswith(expected)
            assert 'if unspecified' not in upper['wrapped_prompt']
            assert upper['source_texts'] == p['oracle_source_texts']
            assert upper['mounted_ids'] == []
    finally:
        repo.close()


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_c7/r3/registration.json')
def test_plain_restart_sentinels(monkeypatch, tmp_path):
    from types import SimpleNamespace
    r=common.verify(); f=fixture(); calls=[]
    def read_only(repo, question, **kwargs):
        calls.append(question)
        assert kwargs['defer_memory'] is True
        return 'unknown', {}
    monkeypatch.setattr(run, 'seats', lambda *a: {})
    rows=run.run_sentinels(SimpleNamespace(arena=None),
        SimpleNamespace(_probe_ladder_chat=read_only), f, r, tmp_path,
        {'turn':100,'process_id':'cpu'}, 'restart_before')
    assert len(rows) == len(calls) == 4
    expected=[next(p['question'] for p in f['probes'] if p['id']==pid)
              .replace('; if unspecified, reply UNKNOWN.', '.') for pid in r['restart_sentinels']]
    assert calls == expected
