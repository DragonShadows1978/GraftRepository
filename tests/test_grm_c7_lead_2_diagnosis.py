"""Registered diagnosis pins, not repair gates or GPT-OSS validation.

Prior art: local GRM LSR-P2B/WC1/EB1 CPU fixtures (GRM contributors, 2026).
Taken: counterfactual input pairs through actual serving paths; new: C7
UNKNOWN poisoning and fold call observations. No prior art known to me for
this exact set of regression stimuli. No production algorithm is changed.
"""
import inspect
import json

from core.graft_arena import ArenaCache
from core.grm_admission import ordered_identifier_tokens, is_identifier_binding
from scripts import grm_c7_run as c7
from scripts import grm_e2e_session as e2e
from scripts.grm_c7_diagnose import ROOT, record, repository


def fixture():
    return json.loads((ROOT / 'artifacts/grm_c7/fixture.json').read_text())


def test_identifier_unknown_suffix_blocks_real_repository_admission(tmp_path, monkeypatch):
    repo = repository(tmp_path / 'repo', monkeypatch)
    p = fixture()['probes'][0]
    idx = repo.add_document(p['oracle_source_texts'][0])
    original = p['question']
    cases = [original, original.replace('UNKNOWN', 'unknown'),
             original.replace('UNKNOWN', 'unknown').replace('C7-Fresh-0', 'c7-fresh-0'),
             original.split(' Reply only')[0]]
    rows = []
    try:
        for question in cases:
            repo.arena.reset_live_cache()
            answer, info = e2e._probe_ladder_chat(repo, question, topk=3,
                                                 ngen=32, max_trips=1, defer_memory=True)
            rows.append({'question': question, 'answer': answer,
                'identifiers': info.get('abstain_identifier_tokens',
                                       info.get('admission_identified_candidates')),
                'mounted_ids': list(repo.arena.cur_mounts), 'info': info})
        record('admission', rows)
        assert rows[0]['answer'] == 'Not in memory: no stored record matches c7-fresh-0, unknown.'
        assert rows[0]['mounted_ids'] == []
        assert all(r['mounted_ids'] == [idx] for r in rows[1:])
        assert all(r['answer'] == 'Basalt-811' for r in rows[1:])
    finally:
        repo.close()


def test_eb1_wc1_identifier_controls_and_all_c7_binding_pairs(tmp_path, monkeypatch):
    repo = repository(tmp_path / 'repo', monkeypatch)
    rows = []
    try:
        idx = repo.add_document('The current orion pin value is Auric-4-Alpha.')
        q = 'What is the current orion pin value?'
        answer, info = e2e._probe_ladder_chat(repo, q, topk=3, ngen=32,
                                             max_trips=1, defer_memory=True)
        assert repo.arena.cur_mounts == [idx]
        assert answer == 'Auric-4-Alpha'
        for p in fixture()['probes']:
            for variant, question in [('original', p['question']),
                                      ('lower_unknown', p['question'].replace('UNKNOWN', 'unknown'))]:
                ordered, rare = ordered_identifier_tokens(repo.arena, question)
                bindings = [is_identifier_binding(candidate_text=s,
                    ordered_identifier_tokens=ordered, rare_identifier_tokens=rare)
                    for s in p['oracle_source_texts']]
                rows.append({'probe_id': p['id'], 'variant': variant,
                             'identifiers': ordered, 'bindings': bindings})
        record('binding_pairs', {'eb1_wc1_question': q, 'answer': answer,
                                 'mounts': [idx], 'info': info, 'pairs': rows})
        assert not any(any(r['bindings']) for r in rows if r['variant'] == 'original')
        assert all(any(r['bindings']) for r in rows if r['variant'] == 'lower_unknown')
    finally:
        repo.close()


def test_oracle_exact_source_reaches_real_attempt_harmony_live_input(tmp_path, monkeypatch):
    repo = repository(tmp_path / 'repo', monkeypatch)
    p = fixture()['probes'][0]
    try:
        before = (repo.arena.caches, repo.arena.pos, list(repo.arena.live_segs),
                  list(repo.arena.cur_mounts), repo.arena.cur_mount_n)
        result = c7.oracle(repo.arena, p, 32)
        initial = [r for r in repo.arena.m.calls if r['initial']]
        record('oracle', {'result': result, 'model_calls': repo.arena.m.calls,
                          'expected_formatted_input': e2e.harmony_turn(result['live_prompt'], None)})
        assert result['answer'] == 'Basalt-811'
        assert result['mounted_ids'] == []
        assert len(initial) == 1
        assert initial[0]['input'] == e2e.harmony_turn(result['live_prompt'], None)
        assert p['oracle_source_texts'][0] in initial[0]['input']
        assert (repo.arena.caches, repo.arena.pos, repo.arena.live_segs,
                repo.arena.cur_mounts, repo.arena.cur_mount_n) == before
        # This pins an omitted setup, not the GPT-OSS consequence of it.
        assert initial[0]['live_shift'] is None
    finally:
        repo.close()


def test_fold_uses_core_raw_prompts_120_steps_and_ignores_harmony_stop(tmp_path, monkeypatch):
    repo = repository(tmp_path / 'repo', monkeypatch)
    try:
        idx = repo.add_document(fixture()['probes'][0]['oracle_source_texts'][0])
        # C7 folds fed turn nodes; add_document defaults to kind=document.
        # Diagnostic fixture correction only; preserve all registered pins.
        repo.arena.grafts[idx]['kind'] = 'turn'
        repo.arena.m.fold_output = '<|end|> …'
        assert repo._fold_once(jobs=[('digest', [idx])])
        calls = repo.arena.m.calls
        initial = [r for r in calls if r['initial']]
        record('fold', {'model_calls': calls, 'attempts': repo.arena.last_consolidation_attempts,
                        'result': repo.arena.last_consolidation_result,
                        'no_fold': repo.arena.grafts[idx].get('no_fold'),
                        'configured_stops': list(repo.arena.stop_sequences)})
        assert [r['input'] for r in initial] == list(ArenaCache.DIGEST_PROMPTS)
        assert len(calls) == 3 * 120
        assert all('<|end|>' in r['text'] for r in repo.arena.last_consolidation_attempts)
        assert all(r['live_shift'] is None for r in calls)
        assert repo.arena.last_consolidation_result['accepted'] is False
        assert repo.arena.grafts[idx]['no_fold'] is True
    finally:
        repo.close()


def test_fold_recorded_ellipsis_can_pass_core_qc_but_coverage_rejects():
    paths = sorted((ROOT / 'artifacts/grm_c7/cells').glob('A-*/folds/*/result.json'))
    rows = []
    for path in paths:
        result = json.loads(path.read_text())
        for attempt in result['attempts']:
            if '…………' in attempt['text']:
                rows.append({'path': str(path.relative_to(ROOT)), 'text': attempt['text'],
                    'recorded_qc': attempt['qc'], 'coverage': attempt['coverage'],
                    'replayed_qc': ArenaCache._digest_qc(attempt['text'], None, forbid_lists=True)})
    record('ellipsis_qc', rows)
    assert rows
    assert any(r['recorded_qc'] and r['replayed_qc'] for r in rows)
    assert all(r['coverage'] < .70 for r in rows)


def test_probe_question_expected_omitted_at_c7_serialization():
    rows = [json.loads(s) for p in sorted((ROOT / 'artifacts/grm_c7/cells').glob('A-*/probes.jsonl'))
            for s in p.read_text().splitlines()]
    probes = {p['id']: p for p in fixture()['probes']}
    record('probe_schema', {'row_count': len(rows), 'missing_question': sum('question' not in r for r in rows),
        'missing_expected': sum('expected' not in r for r in rows),
        'first_probe': probes[rows[0]['probe_id']], 'first_row_keys': sorted(rows[0]),
        'worker_source': inspect.getsource(c7.worker)})
    assert len(rows) == 60
    assert all('question' not in r and 'expected' not in r for r in rows)
    assert all(probes[r['probe_id']]['question'] and probes[r['probe_id']]['expected'] for r in rows)
