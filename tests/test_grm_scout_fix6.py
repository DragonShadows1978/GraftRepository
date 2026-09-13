"""Prior art: LT1 offline Rule2 and GRM A-DEC (contributors, 2026).
Independent reference-pair contracts; no prior art known for exact cases.
"""
import json
from pathlib import Path
import pytest
from core import grm_admission as adm
from core.graft_arena import ArenaCache
from scripts.grm_lt1_admission import evaluate

class FrozenArena:
    """Replay exact stored scores through the core scoring/decision boundary."""
    _rare_tokens=staticmethod(ArenaCache._rare_tokens)
    _query_lex_tokens=staticmethod(ArenaCache._query_lex_tokens)
    _norm_text=staticmethod(ArenaCache._norm_text)
    _QUERY_LEX_STOP=ArenaCache._QUERY_LEX_STOP
    def __init__(self,s):self.s=s;self.grafts=s['nodes'];self.last_route_backend='frozen';self.lsr_fixes=True
    def _route_cand_base(self):return self.s['eligible']
    def _probe_key(self,q):return q
    def route(self,q,**kwargs):return self.s['ranking'][:kwargs['limit']]
    def _split_family_members(self,r):return self.s.get('split_members',[])
    def _vector_route_scores(self,*_):return {int(i):v for i,v in self.s['scores'].items()}
    def _length_debias_scores(self,b,e):return b
    def _normalize_scores(self,b):return b
    def _lex_bonus(self,*_):return 0

@pytest.mark.parametrize('margin',[0.,.01,adm.MARGIN_THRESHOLD,adm.MARGIN_THRESHOLD+1e-12])
def test_rule2_exact_reference_and_refusal(monkeypatch,margin):
    monkeypatch.setenv('GRM_ADMISSION_RULE','margin_first')
    s=dict(question='What did we settle on at Harbor?',nodes=[dict(text='Promenade 9'),dict(text='Harbor 12'),dict(text='elsewhere')],eligible=[0,1,2],ranking=[0,1,2],split_members=[0],margin=margin,scores={'0':margin,'1':0.,'2':-1.})
    p=adm.decisive_admission_profile(FrozenArena(s),s['question'],exclude=[])
    assert p['rank_plan']==evaluate(s,2)['plan']
    assert adm.identifier_unbound_abstention(p) is None
    assert p['admission_rule']=='margin_first'
    assert adm.admission_info_fields(p)['admission_rule']=='margin_first'

@pytest.mark.parametrize('value',[None,'','typo','all_tokens_bind'])
def test_default_and_unknown_stay_rule0(monkeypatch,value):
    """Rule 0 for the unset/empty/unknown cases, under the PRE-D2 default.

    GRM-D2 (2026-09-11) made `margin_first` the shipped rule, so "unset"
    and "unknown token" now resolve to rule 2 on a tree with no umbrella.
    The assertions here are UNCHANGED and remain the receipt for rule 0's
    reference pair; `GRM_LEGACY_DEFAULTS=1` is what makes `None` / `''` /
    `'typo'` mean rule 0 again.  `'all_tokens_bind'` is an EXPLICIT pin and
    would pass with or without the umbrella -- it outranks it.
    """
    monkeypatch.setenv('GRM_LEGACY_DEFAULTS','1')
    if value is None:monkeypatch.delenv('GRM_ADMISSION_RULE',raising=False)
    else:monkeypatch.setenv('GRM_ADMISSION_RULE',value)
    s=dict(question='What is at Harbor?',nodes=[dict(text='Harbor 12')],eligible=[0],ranking=[0],split_members=[],margin=0.,scores={'0':0.})
    p=adm.decisive_admission_profile(FrozenArena(s),s['question'],exclude=[])
    assert p['rank_plan']==evaluate(s,0)['proposed_plan']
    assert bool(adm.identifier_unbound_abstention(p))==evaluate(s,0)['refused']
    assert p['admission_rule']=='all_tokens_bind'

def test_rule2_forces_margin_large_bank(monkeypatch):
    monkeypatch.setenv('GRM_ADMISSION_RULE','margin_first')
    s=dict(question='What is at Absentport?',nodes=[dict(text='unrelated') for _ in range(20)],eligible=list(range(20)),ranking=list(range(6)),split_members=[],margin=1.,scores={str(i):float(20-i) for i in range(20)})
    p=adm.decisive_admission_profile(FrozenArena(s),s['question'],exclude=[],route_limit=6)
    assert p['route_margin_evaluated'] and p['rank_plan']==[0]
    assert adm.identifier_unbound_abstention(p) is None

def test_empty_and_profile_bound_rule(monkeypatch):
    monkeypatch.setenv('GRM_ADMISSION_RULE','margin_first')
    s=dict(question='Missing?',nodes=[],eligible=[],ranking=[],scores={})
    p=adm.decisive_admission_profile(FrozenArena(s),s['question'],exclude=[])
    assert p['rank_plan']==[] and p['admission_rule']=='margin_first'
    # Receipt/profile decision must survive an ambient flag change.
    monkeypatch.delenv('GRM_ADMISSION_RULE')
    assert adm.identifier_unbound_abstention(dict(p,identifier_tokens=['missing'])) is None
