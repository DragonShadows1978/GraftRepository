"""CPU-only gates registered by amendment7; no model loading.

Prior art: RD1/C7 fake numerical seams and SC1.1 glyph tests (GRM, 2026).
Reuse the separation between control-flow checks and model evidence. Ours:
these receipt-bound assertions; no prior art known to me for this composition.
"""
import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import grm_c7_amendment7 as d
from scripts import grm_c7_middle_replay as run
from scripts.grm_c7_common import score
from core.grm_admission import is_identifier_binding, normalized_words
from core.grm_text_norm import normalize_glyphs


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_c7/amendment_7.json + orders/GRM_C7_AMENDMENT_7.md')
def test_registered_cohort_prompt_budget():
    r=d.verify();qs=d.read(d.OUT/'requests.json')
    assert len(qs)==40 and len({q['probe_id'] for q in qs})==40
    assert sum(q['probe']['expected']=='UNKNOWN' for q in qs)==20
    for cls in ('fresh','folded'):
        assert sum(q['probe']['class']==cls and q['probe']['expected']!='UNKNOWN' for q in qs)==10
    ids=[p for ps in r['batches'].values() for p in ps]
    assert len(ids)==40 and set(ids)=={q['probe_id'] for q in qs}
    assert len(r['batches'])*r['worker_seconds']==1680<=r['gpu_cap_seconds']==1800
    import re
    rare=lambda s:{w.lower() for w in re.findall(r'[A-Za-z0-9][\w:.,\-]*',s) if any(x.isdigit() for x in w) or (w.isupper() and len(w)>=3)}
    for q in qs:
        assert q['middle']==d.MIDDLE+q['plain']
        assert rare(q['plain'])==rare(q['middle'])
        assert 'UNKNOWN' not in d.MIDDLE
        assert q['probe']['expected'] not in d.MIDDLE
        if not q['mounted_ids']:
            assert q['historical']['route_info']['abstain_reason']=='identifier_unbound'


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_c7/amendment_7.json + orders/GRM_C7_AMENDMENT_7.md')
def test_actual_identifier_red_and_glyph_counterfactual():
    # Prior art: SC1.1 existing normalizer (2026). Counterfactual INPUT only;
    # never patch or replace the production predicate or mutate a digest.
    qs=d.read(d.OUT/'requests.json')
    results=[]
    for node_id,token in ((10,'c7-archive-0'),(16,'c7-fresh-0')):
        p=d.R3/'cells/A-032-039/checkpoint/repository/manifest.json'
        node=d.read(p)['nodes'][node_id]
        args=dict(ordered_identifier_tokens=[token],rare_identifier_tokens=[token])
        assert token in node['rare']
        assert '\u2011' in node['text']
        assert not is_identifier_binding(candidate_text=node['text'],**args)
        assert is_identifier_binding(candidate_text=normalize_glyphs(node['text']),**args)
        assert is_identifier_binding(candidate_text=normalize_glyphs(node['text']).upper(),**args)
        removed=normalize_glyphs(node['text']).casefold().replace(token, 'other')
        assert not is_identifier_binding(candidate_text=removed,**args)
        results.append({'id':node_id,'token':token,'raw_words':normalized_words(node['text']),
            'normalized_words':normalized_words(normalize_glyphs(node['text'])),'raw_binding':False,'glyph_projected_binding':True})
    # Different digits must stay different under the existing projection.
    assert not is_identifier_binding(candidate_text='C7‑Fresh‑9',ordered_identifier_tokens=['c7-fresh-0'],rare_identifier_tokens=['c7-fresh-0'])
    assert not is_identifier_binding(candidate_text=normalize_glyphs('C7‑Fresh‑9'),ordered_identifier_tokens=['c7-fresh-0'],rare_identifier_tokens=['c7-fresh-0'])


def test_frozen_score_hand_cases():
    fresh={'expected':'Basalt-811','answerable':True}
    control={'expected':'UNKNOWN','answerable':False}
    assert score('Basalt‑811',fresh)['exact_correct']
    assert score('unknown',fresh)['abstention_error']==1
    assert score('Basalt-812',fresh)['wrong_value_error']==1
    assert score('unknown',control)['exact_correct']
    assert score('The inspection password is Flint-511.',control)['unsupported_answer_error']==1
    assert score('Not in memory: no stored record matches c7-fresh-0.',fresh)['abstention_error']==1


class FakeArena:
    """Numerical seam only: this checks replay orchestration, not quality."""
    def __init__(self,answers):
        self.answers=iter(answers);self.cur_mounts=[];self.grafts=[{'h':1} for _ in range(8)]
        self.m=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace())])
        self.live_shift=115;self.stop_sequences=['<|end|>'];self.calls=[]
    def reset_live_cache(self):self.clean=True;self.cur_mounts=[]
    def eb1_begin_turn(self):pass
    def _resolve_revision_mounts(self,p):return p
    def _ensure_h(self,p):pass
    def _format_step_prompt(self,p):return '<user>'+p
    def encode(self,p):return list(p.encode())
    def _attempt(self,p,picks,n,deposit,stops,defer_memory):
        assert self.clean and n==32 and not deposit and defer_memory
        assert self.m.layers[0].self_attn.live_shift==115
        self.clean=False;self.cur_mounts=list(picks);self.calls.append((p,list(picks)))
        return next(self.answers),{}


def test_pair_isolation_and_mismatch_stop(tmp_path):
    q=copy.deepcopy(d.read(d.OUT/'requests.json')[0]);q['mounted_ids']=[6]
    repo=SimpleNamespace(arena=FakeArena([q['historical']['answer'],'unknown']))
    result=run.pair(repo,q,tmp_path)
    assert len(repo.arena.calls)==2
    assert repo.arena.calls==[(q['plain'],[6]),(q['middle'],[6])]
    assert not hasattr(repo.arena.m.layers[0].self_attn,'live_shift')
    assert result['A1']['score']['abstention_error']==1
    bad=tmp_path/'bad';bad.mkdir()
    repo=SimpleNamespace(arena=FakeArena(['wrong baseline']))
    with pytest.raises(ValueError,match='A0_R3_BYTE_MISMATCH_STOP'):run.pair(repo,q,bad)
    assert len(repo.arena.calls)==1
    assert (bad/(q['probe_id']+'_A0.json')).exists()
    assert not (bad/(q['probe_id']+'_A1.json')).exists()


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_c7/amendment_7.json + orders/GRM_C7_AMENDMENT_7.md')
def test_refused_rows_are_not_generated(tmp_path):
    q=next(q for q in d.read(d.OUT/'requests.json') if not q['mounted_ids'])
    run.carry_refusal(q,tmp_path)
    r=d.read(tmp_path/(q['probe_id']+'_A1.json'))
    assert r['answer']==q['historical']['answer']
    assert 'zero model calls' in r['evidence_class']
    assert run.summary()['status']=='NOT_MEASURED'
