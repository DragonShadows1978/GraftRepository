"""Prior art: GRM ADM1/EB1 and P2B receipt invariants (GRM, 2026).
Apply those contracts to FIX-4 boundary cases; no new selection algorithm.
"""
from copy import deepcopy

import pytest
from core.grm_admission import decisive_admission_profile, identifier_serving_decision
from core.grm_three_pass import build_route_receipt
from scripts.grm_c7_diagnose import repository
from scripts.grm_e2e_session import harmony_turn
from test_grm_scout_fix4 import serve


@pytest.mark.parametrize('path',['core','ladder'])
def test_all_nodes_live_still_reads_once_and_receipt_survives(tmp_path,monkeypatch,path):
    repo=repository(tmp_path/'repo',monkeypatch)
    a=repo.arena
    try:
        a.feed(harmony_turn('The current Live-431 value is Basalt-811.','Recorded.'))
        a.recency_mounts=1
        answer,info=serve(repo,path,'What is the current Live-431 value?')
        assert answer=='Basalt-811'
        assert a.cur_mounts==[0]
        assert info['admission_rank_plan']==[]
        assert info['admission_ranking_before_demotion']==[]
        assert info['recency_seats']['charged_ntok']==a.grafts[0]['ntok']
        receipt=build_route_receipt(session_id='cpu',turn_id='1',info=info,arena=a)
        assert receipt['fit']['info_pass_through']['served_from']=='recency_mount'
        assert receipt['fit']['info_pass_through']['served_from_node_ids']==[0]
    finally:
        repo.close()


def test_shared_decision_is_inert_when_admission_has_a_binder(tmp_path,monkeypatch):
    repo=repository(tmp_path/'repo',monkeypatch)
    a=repo.arena
    try:
        for _ in range(2):
            a.feed(harmony_turn('The current Live-431 value is Basalt-811.','Recorded.'))
        question='What is the current Live-431 value?'
        profile=decisive_admission_profile(a,question,exclude={1},route_limit=3)
        before=deepcopy(profile)
        assert profile['identified_candidates']==[0]
        assert identifier_serving_decision(a,question,profile,exclude={1}) is None
        assert profile==before
    finally:
        repo.close()
