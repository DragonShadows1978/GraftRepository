"""C7 r2 repair gates. Prior art: C7 lead-2 counterfactual replay and EB1
position setup (GRM contributors, 2026), local source verified. Borrow fake
numerical boundaries, live-source ceiling and immutable amendment checks.
Ours: repair assertions; no prior art known to me for this exact composition.
"""
import inspect
import json

import pytest

from core.grm_admission import ordered_identifier_tokens, is_identifier_binding
from scripts import grm_c7_run as c7
from scripts import grm_c7_common as common
from scripts.grm_c7_diagnose import repository
from scripts.grm_e2e_session import _probe_ladder_chat, harmony_turn


def fixture():
    return json.loads(common.FIX.read_text())


def test_fixture_identifiers_mount_and_live_oracle_answers(tmp_path, monkeypatch):
    repo = repository(tmp_path / 'repo', monkeypatch)
    p = fixture()['probes'][0]
    try:
        idx = repo.add_document(p['oracle_source_texts'][0])
        question = c7.effective_question(p['question'])
        answer, info = _probe_ladder_chat(repo, question, topk=3,
                                          ngen=32, max_trips=1, defer_memory=True)
        assert repo.arena.cur_mounts == [idx]
        assert answer == 'Basalt-811'
        # Ensure the oracle cannot answer from a stale injected source.
        repo.arena.m.injected = ''
        repo.arena.m.calls.clear()
        upper = c7.oracle(repo.arena, p, 32)
        assert upper['answer'] == 'Basalt-811'
        assert upper['mounted_ids'] == []
        assert 'C7-Fresh-0 value is Basalt-811' in repo.arena.m.calls[0]['input']
        assert upper['question'] == question
        assert info.get('abstain_reason') != 'identifier_unbound'
        for probe in fixture()['probes']:
            q = c7.effective_question(probe['question'])
            assert q == probe['question'].removesuffix('UNKNOWN.') + 'unknown.'
            ordered, rare = ordered_identifier_tokens(repo.arena, q)
            assert 'unknown' not in rare
            assert any(is_identifier_binding(candidate_text=s,
                ordered_identifier_tokens=ordered, rare_identifier_tokens=rare)
                for s in probe['oracle_source_texts'])
    finally:
        repo.close()


def test_oracle_sets_restores_layer_positions_and_records_wrapped_ids(tmp_path, monkeypatch):
    repo = repository(tmp_path / 'repo', monkeypatch)
    a = repo.arena
    p = fixture()['probes'][0]
    try:
        a.m.layers[0].self_attn.live_shift = 991
        before = (a.caches, a.pos, list(a.live_segs), list(a.cur_mounts), a.cur_mount_n)
        result = c7.oracle(a, p, 32)
        assert a.m.calls[0]['live_shift'] == a.live_shift
        assert result['answer'] == 'Basalt-811'
        assert result['prompt_token_ids'] == a.encode(harmony_turn(result['live_prompt'], None))
        assert a.decode(result['prompt_token_ids']) == a.m.calls[0]['input']
        assert a.m.layers[0].self_attn.live_shift == 991
        assert (a.caches, a.pos, a.live_segs, a.cur_mounts, a.cur_mount_n) == before
        def fail(*args, **kwargs):
            raise RuntimeError('oracle fault')
        monkeypatch.setattr(a, '_attempt', fail)
        with pytest.raises(RuntimeError, match='oracle fault'):
            c7.oracle(a, p, 32)
        assert a.m.layers[0].self_attn.live_shift == 991
        assert (a.caches, a.pos, a.live_segs, a.cur_mounts, a.cur_mount_n) == before
    finally:
        repo.close()


def test_probe_row_question_expected_and_worker_uses_effective_question():
    p = fixture()['probes'][0]
    row = c7.probe_row(p, p['turn'], {'answer': 'x'}, {'answer': 'y'})
    assert row['question'] == c7.effective_question(p['question'])
    assert row['registered_question'] == p['question']
    assert row['expected'] == p['expected']
    assert row['memory']['answer'] == 'x' and row['oracle']['answer'] == 'y'
    # Wiring guard complements the real ladder + real oracle session above.
    assert "effective_question(event['user'])" in inspect.getsource(c7.worker)
    assert 'probe_row(p, turn, memory, upper)' in inspect.getsource(c7.worker)
    assert "effective_question(p['question'])" in inspect.getsource(c7.run_sentinels)
