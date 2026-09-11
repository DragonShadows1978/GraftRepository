"""RD1 author CPU gates, not a blind audit or GPT-OSS reproduction.
Prior art: C7 diagnostic Codec/Model/CPUArena and checkpoint fixtures (GRM
contributors, 2026), locally verified. Only numerical boundaries are fake;
production repository loading and _attempt execute. Ours: amendment contracts;
no prior art known to me for this exact test composition.
"""
import copy
import json
from pathlib import Path

import pytest

from scripts import grm_rd1_replay as rd
from scripts import grm_rd1 as old
from scripts import grm_c7_common as common
from scripts import grm_c7_diagnose as diag
from core.graft_arena import ArenaCache


def fake_case(tmp_path, monkeypatch, prefix=False):
    cp = tmp_path/'checkpoint'
    cp.mkdir()
    repo = diag.repository(cp/'repository', monkeypatch)
    repo.add_document('The current C7-Test value is Basalt-811.')
    codec = repo.arena.m.codec
    repo.flush_now()
    state = {'next_turn': 1, 'turn_records': {}, 'transcript': [], 'probe_rows': [],
             'process_id': 'fake-predecessor', 'cold_nodes': [], 'fold_failures': []}
    common.checkpoint(cp, state, {'fixture': 'fake-checkpoint'})
    repo.close()
    cell = {'checkpoint': str(cp), 'worker_checkpoint_sha256': old.sha(cp/'checkpoint.json'),
            'historical_binding': {'fixture': 'fake-checkpoint'}, 'historical_controller_status': 'COMPLETE',
            'start': 1, 'id': 'FAKE'}
    turn = 2 if prefix else 1
    expected = 'Prefix-222' if prefix else 'Basalt-811'
    p = {'id': 'fake_probe', 'class': 'fresh', 'answerable': True, 'expected': expected}
    prompt = 'What is the current C7-Test value? Reply only with the answer' + old.SUFFIX
    oracle = 'Exact source records, in chronological order:\nThe current C7-Test value is '+expected+'.\nUse only those records.\n'+prompt
    row = {'probe': p, 'cell_id': 'FAKE', 'checkpoint': str(cp),
           'checkpoint_sha256': old.sha(cp/'checkpoint.json'), 'prefix_turns': list(range(1,turn)),
           'historical': {'probe_id': p['id'], 'turn': turn, 'question': prompt,
               'memory': {'answer': expected, 'residency': {'mounted_ids': [1 if prefix else 0]},
                          'route_info': {'frame_ephemeral': True, '_deferred_memory': {'deposited': False}}},
               'oracle': {'answer': expected, 'live_prompt': oracle}}}
    # Reuse the tokenizer vocabulary across fake NPZ save/load; never inject
    # expected outputs. The Model derives its answer from the loaded tokens.
    monkeypatch.setattr(diag, 'Codec', lambda: codec)
    loaded = []
    def loader(session, registration):
        loaded_repo = diag.repository(session/'repository', monkeypatch)
        assert type(loaded_repo.arena)._attempt is ArenaCache._attempt
        loaded.append(loaded_repo)
        return loaded_repo, {'backend': 'CPU diagnostic Model; not numerical GPT-OSS'}
    def advance(repo, state, event, turn, session, registration):
        node = repo.add_document(event['user'])
        state['turn_records'][str(turn)] = {'memory_node_id': node}
    fixture = {'turns': ([{'kind': 'plant', 'user': 'The current C7-Test value is Prefix-222.'}] if prefix else []) +
                       [{'kind': 'probe', 'probe_id': p['id']}],
               'forced_folds': [], 'pressure_after_turns': []}
    r = {'flags': {'arena_width': 96}, 'model_frame': {'sink_text': old.PREFIX.split('<|start|>user')[0],
         'sink_token_ids': codec.encode(old.PREFIX.split('<|start|>user')[0])}}
    return cell, row, fixture, r, loader, advance, loaded


def run_fake(tmp_path, monkeypatch, *, prefix=False, arm='A0', side='memory', modify=None):
    cell, row, fixture, r, loader, advance, loaded = fake_case(tmp_path, monkeypatch, prefix)
    q = {'arm': arm, **old.request(row, side, arm)}
    if modify:
        modify(q, row)
    out = tmp_path/'result'
    out.mkdir()
    result = rd.replay_batch(r, {'cell': cell}, out, [q], [row], fixture,
                             loader=loader, advance=advance)
    receipt = old.read(out/('fake_probe_'+side+'.json'))
    return result, receipt, loaded


@pytest.mark.campaign_receipt(registration='artifacts/grm_rd1/registration.json (scripts/grm_rd1.py SOURCE = /mnt/ForgeRealm/wt/grm-c7, pruned with the grm-c7 fork)')
def test_registration_and_138_intended_diffs():
    r = rd.verify()
    qs = old.read(rd.OUT/'requests.json')
    assert r['request_count'] == len(qs) == 138
    assert r['estimate_seconds'] == pytest.approx(3129.8965672597046)
    assert r['gpu_cap_seconds'] == 3600 and r['status'] == 'FIT_ESTIMATE'
    assert len(r['batches']) == 48 and max(b['estimate_seconds'] for b in r['batches']) < 285
    assert all(sum(q['arm'] == a and q['side'] == s for q in qs) == n
               for a in rd.ARMS for s,n in [('memory',30),('oracle',16)])
    allowed = {'A0': set(), 'A2': {'prompt','wrapped_prompt'}, 'A4': {'prompt','wrapped_prompt','reasoning'}}
    for d in old.read(rd.OUT/'arm_diffs.json'):
        assert set(d['changes']) == allowed[d['arm']]
    for row in old.read(old.OUT/'cohort.json'):
        rd.assert_deferred_history(row)
    assert r['arms']['A1']['status'] == r['arms']['A3']['status'] == 'NOT_RUN'


@pytest.mark.campaign_receipt(registration='artifacts/grm_rd1/registration.json (scripts/grm_rd1.py SOURCE = /mnt/ForgeRealm/wt/grm-c7, pruned with the grm-c7 fork)')
def test_checkpoint_tree_and_boundary(tmp_path, monkeypatch):
    for c in rd.verify()['batches'][:16]:
        old.validate_bound_state(c['cell'])
    cell, *_ = fake_case(tmp_path, monkeypatch)
    old.validate_bound_state(cell)
    with (Path(cell['checkpoint'])/'state.json').open('a') as f:
        f.write(' ')
    with pytest.raises(ValueError, match='CHECKPOINT_INTEGRITY'):
        old.validate_bound_state(cell)


@pytest.mark.parametrize('side', ['memory','oracle'])
def test_fake_checkpoint_production_attempt_A0(tmp_path, monkeypatch, side):
    result, receipt, loaded = run_fake(tmp_path, monkeypatch, side=side)
    assert result['status'] == 'COMPLETE' and result['scored_probe_attempts'] == 1
    assert receipt['served_text'] == 'Basalt-811' and receipt['r2_byte_equal']
    assert receipt['mounted_ids'] == ([0] if side == 'memory' else [])
    assert loaded[0].arena.m.calls[0]['input'] == receipt['request']['wrapped_prompt']
    assert loaded[0].arena.m.calls[0]['live_shift'] == loaded[0].arena.live_shift
    assert len(loaded[0].arena.grafts) == 1


def test_fake_prefix_creates_missing_mount(tmp_path, monkeypatch):
    result, receipt, _ = run_fake(tmp_path, monkeypatch, prefix=True)
    assert receipt['mounted_ids'] == [1] and receipt['served_text'] == 'Prefix-222'
    assert result['events'][0]['kind'] == 'original_nonprobe_reconstruction'
    assert len(old.read(tmp_path/'checkpoint/repository/manifest.json')['nodes']) == 1


@pytest.mark.parametrize('arm', ['A0','A2','A4'])
def test_fake_arm_fields_and_isolation(tmp_path, monkeypatch, arm):
    result, receipt, loaded = run_fake(tmp_path, monkeypatch, arm=arm)
    q = receipt['request']
    assert q['answer_budget'] == 32
    assert ('Reasoning: medium.' in loaded[0].arena.m.calls[0]['input']) == (arm == 'A4')
    assert (old.SUFFIX in loaded[0].arena.m.calls[0]['input']) == (arm == 'A0')
    assert loaded[0].arena.prompt_template is diag.e2e.harmony_turn
    assert loaded[0].arena.caches is None and loaded[0].arena.cur_mounts == []
    assert loaded[0].arena._deferred_route_keys == {}


def test_mount_mismatch_rejected_before_forward(tmp_path, monkeypatch):
    cell,row,fixture,r,loader,advance,loaded = fake_case(tmp_path,monkeypatch)
    q = {'arm':'A0', **old.request(row,'memory','A0')}
    q['mounted_ids'] = [9]
    out=tmp_path/'result';out.mkdir()
    with pytest.raises(ValueError,match='REQUEST_RESIDENCY_MISMATCH'):
        rd.replay_batch(r,{'cell':cell},out,[q],[row],fixture,loader=loader,advance=advance)
    assert loaded[0].arena.m.calls == []


def test_A0_difference_receipted_and_stops(tmp_path, monkeypatch):
    def change(q,row):
        row['historical']['memory']['answer'] = 'Different-999'
    with pytest.raises(ValueError, match='A0_R2_TEXT_DIFFERENCE'):
        run_fake(tmp_path,monkeypatch,modify=change)
    row=old.read(tmp_path/'result/fake_probe_memory.json')
    assert row['served_text'] == 'Basalt-811' and not row['r2_byte_equal']
    assert row['difference']['actual_utf8_hex'] == 'Basalt-811'.encode().hex()


@pytest.mark.campaign_receipt(registration='artifacts/grm_rd1/registration.json (scripts/grm_rd1.py SOURCE = /mnt/ForgeRealm/wt/grm-c7, pruned with the grm-c7 fork)')
def test_A0_campaign_barrier(tmp_path):
    with pytest.raises(ValueError,match='A0_INCOMPLETE'):
        rd.verify_a0(rd.verify(),tmp_path)


@pytest.mark.campaign_receipt(registration='artifacts/grm_rd1/registration.json (scripts/grm_rd1.py SOURCE = /mnt/ForgeRealm/wt/grm-c7, pruned with the grm-c7 fork)')
def test_budget_and_incomplete_summary_fail_closed(tmp_path, monkeypatch):
    r=rd.verify()
    s=rd.summary(r,tmp_path)
    assert s['status'] == 'NOT_MEASURED' and len(s['table']) == 40
    assert s['observed'] == 0 and all(x['exact'] is None for x in s['table'])
    monkeypatch.setenv('GRM_RD1_LEAD_GPU','1')
    old.write(tmp_path/'orphan/reservation.json',{'seconds':3600})
    with pytest.raises(ValueError,match='BUDGET_RAIL'):
        rd.launch(r,r['batches'][0],tmp_path)
    assert not (tmp_path/r['batches'][0]['id']).exists()


@pytest.mark.campaign_receipt(registration='artifacts/grm_rd1/registration.json (scripts/grm_rd1.py SOURCE = /mnt/ForgeRealm/wt/grm-c7, pruned with the grm-c7 fork)')
def test_dry_run_no_gpu(tmp_path, monkeypatch):
    monkeypatch.delenv('GRM_RD1_LEAD_GPU',raising=False)
    r=rd.verify()
    with pytest.raises(ValueError,match='BLOCKED_NO_GPU_IN_SANDBOX'):
        rd.launch(r,r['batches'][0],tmp_path)


def test_registered_mutations_at_least_80_percent(tmp_path, monkeypatch):
    # Prior art: HOUSE_RULES section 8 and C7 source-copy mutation gate (2026).
    # Mutate in-memory function copies, then run the SAME baseline gates.
    # Registered defects, threshold .80; source files are never rewritten.
    import inspect
    mutants = [
        ('erase_mount_guard', 'replay_attempt', "require(wanted == historical_ids, 'REQUEST_RESIDENCY_MISMATCH')", 'pass', test_mount_mismatch_rejected_before_forward, {}),
        ('force_A0_equal', 'replay_attempt', "equal = str(answer).encode('utf-8') == expected.encode('utf-8')", 'equal = True', test_A0_difference_receipted_and_stops, {}),
        ('force_A4_low', 'replay_attempt', "'Reasoning: ' + q['reasoning'] + '.'", "'Reasoning: low.'", test_fake_arm_fields_and_isolation, {'arm':'A4'}),
        ('erase_prompt_suffix_change', 'replay_attempt', "answer, info = a._attempt(q['prompt'], wanted", "answer, info = a._attempt(row['historical']['question'], wanted", test_fake_arm_fields_and_isolation, {'arm':'A2'}),
        ('skip_prefix_advance', 'replay_batch', 'advance(repo, state, event, turn, session, r)', 'pass', test_fake_prefix_creates_missing_mount, {}),
    ]
    receipts=[]
    for name,function,needle,replacement,gate,kwargs in mutants:
        text=inspect.getsource(getattr(rd,function))
        assert text.count(needle)==1
        ns=dict(vars(rd))
        exec(compile(text.replace(needle,replacement), '<RD1 registered mutant '+name+'>', 'exec'),ns)
        directory=tmp_path/name;directory.mkdir()
        with monkeypatch.context() as mp:
            mp.setattr(rd,function,ns[function])
            try:
                gate(directory,mp,**kwargs)
            except (AssertionError,ValueError,pytest.fail.Exception) as exc:
                status='KILLED'; error=f'{type(exc).__name__}: {exc}'
            else:
                status='SURVIVED';error=None
        receipts.append({'name':name,'status':status,'error':error})
    result={'mutants':receipts,'kill_rate':sum(x['status']=='KILLED' for x in receipts)/5,'threshold':0.8}
    print(json.dumps(result,sort_keys=True))
    assert result['kill_rate']>=0.8
