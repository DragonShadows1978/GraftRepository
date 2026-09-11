"""CPU author pins, not a GPT-OSS quality gate.

Prior art: GRM contributors (2026), C7 receipt replays and real serving-path
doubles; reuse these with the actual attention constructor. No prior art
known to me for this exact test composition.
"""
import json

import pytest

from core.gpt_oss20b_tc import GptOss20BConfig, GptOssAttentionTC
from core.grm_admission import (ordered_identifier_tokens, is_identifier_binding,
                                decisive_admission_profile)
from scripts import grm_c7_run as c7
from scripts.grm_c7_common import ROOT, FIX, read
from scripts.grm_c7_diagnose import Codec, Model, repository
from scripts.grm_e2e_session import _probe_ladder_chat, harmony_turn


def real_attention():
    # The real constructor assigns metadata/None only; no weights or CUDA.
    return GptOssAttentionTC(GptOss20BConfig(layer_types=('full_attention',)), 0)


def test_fake_attention_surface_matches_real_constructor():
    real = real_attention()
    fake = Model(Codec()).layers[0].self_attn
    # Enumerate the real class/instance, rather than maintaining a hand list.
    assert set(vars(fake)) == set(vars(real))
    assert set(dir(fake)) == set(dir(real))
    assert not hasattr(real, 'live_shift')
    assert not hasattr(fake, 'live_shift')


def test_real_attention_without_live_shift_oracle_restores_absence(tmp_path, monkeypatch):
    repo = repository(tmp_path/'repo', monkeypatch)
    a = repo.arena
    att = real_attention()
    a.m.layers[0].self_attn = att
    before = (a.caches, a.pos, list(a.live_segs), list(a.cur_mounts), a.cur_mount_n)
    try:
        result = c7.oracle(a, read(FIX)['probes'][0], 32)
        assert result['answer'] == 'Basalt-811'
        assert a.m.calls[0]['live_shift'] == a.live_shift
        assert a.m.calls[0]['input'] == result['wrapped_prompt']
        assert a.encode(result['wrapped_prompt']) == result['prompt_token_ids']
        assert not hasattr(att, 'live_shift')
        assert (a.caches, a.pos, a.live_segs, a.cur_mounts, a.cur_mount_n) == before
    finally:
        repo.close()


@pytest.mark.parametrize('previous', ['absent', None, 991])
def test_oracle_restores_optional_shift_after_exception(tmp_path, monkeypatch, previous):
    repo = repository(tmp_path/'repo', monkeypatch)
    a = repo.arena
    att = real_attention()
    a.m.layers[0].self_attn = att
    if previous != 'absent':
        att.live_shift = previous
    def fail(*args, **kwargs):
        assert att.live_shift == a.live_shift
        raise RuntimeError('registered oracle fault')
    monkeypatch.setattr(a, '_attempt', fail)
    try:
        with pytest.raises(RuntimeError, match='registered oracle fault'):
            c7.oracle(a, read(FIX)['probes'][0], 32)
        assert hasattr(att, 'live_shift') == (previous != 'absent')
        if previous != 'absent':
            assert att.live_shift == previous
    finally:
        repo.close()


@pytest.mark.parametrize('serving_path', ['core', 'ladder'])
def test_alias_recency_exclusion_reproduces_in_core_and_ladder(tmp_path, monkeypatch, serving_path):
    repo = repository(tmp_path/'repo', monkeypatch)
    a = repo.arena
    fixture = read(FIX)
    try:
        for turn in fixture['turns'][:10]:
            idx = a.feed(harmony_turn(turn['user'], turn['assistant']))
            a.grafts[idx]['kind'] = 'turn'
        a.recency_mounts = 2
        p = next(p for p in fixture['probes'] if p['id'] == 'c7_alias_0_d005')
        question = c7.effective_question(p['question'])
        ordered, rare = ordered_identifier_tokens(a, question)
        assert is_identifier_binding(candidate_text=a.grafts[8]['text'],
            ordered_identifier_tokens=ordered, rare_identifier_tokens=rare)
        if serving_path == 'core':
            answer, info = a.step(question, ngen=32, deposit=False, defer_memory=True)
        else:
            answer, info = _probe_ladder_chat(repo, question, topk=3, ngen=32,
                                             max_trips=1, defer_memory=True)
        assert not info.get('abstained', False)
        assert info['served_from'] == 'recency_mount'
        assert info['served_from_node_ids'] == [8]
        assert 'Not in memory:' not in answer
        assert info['admission_identified_candidates'] == []
        assert info['recency_mounted_ids'] == [8, 9]
        assert a.cur_mounts == [8]  # only the binding nominee, no admission mount
        assert a.m.calls
        assert 'C7-Signal-0 is an alias for C7-AliasBase-0.' in a.m.calls[0]['injected']
        assert 8 not in info['admission_rank_plan']
        assert 8 not in info['admission_ranking_before_demotion']
        # Read-only counterfactual, not a proposed recency-policy change.
        profile = decisive_admission_profile(a, question, exclude=(), route_limit=6)
        assert profile['identified_candidates'] == [8]
    finally:
        repo.close()


def test_recorded_oracle_prompt_contains_exact_source_and_corrections_answer():
    paths = sorted((ROOT/'artifacts/grm_c7/r2/cells').glob('*/probes.jsonl'))
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
    assert len(rows) == 8
    for row in rows:
        upper = row['oracle']
        assert upper['wrapped_prompt'] == harmony_turn(upper['live_prompt'], None)
        assert all(text in upper['wrapped_prompt'] for text in upper['source_texts'])
    answers = {r['probe_id']: r['oracle']['answer'] for r in rows}
    assert answers['c7_fresh_0_d005'] == 'unknown'
    assert answers['c7_correction_0_d005'] == 'Onyx-911'
    assert answers['c7_correction_1_d005'] == 'Onyx-912'
