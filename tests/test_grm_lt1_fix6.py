"""Prior art: LT1 shared worker CPU process gates (GRM contributors, 2026).
Reuse all 52 cells/checkpoint checks; Rule2 admission and receipt assertions
are new. No prior art known to me for exact test composition.
"""
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest
from scripts import grm_lt1 as lt, grm_lt1_worker as worker
from core import grm_admission as adm
from scripts.grm_lt1_admission import evaluate
from test_grm_scout_fix6 import FrozenArena


def test_frozen_cpu_arena_70_profiles(monkeypatch):
    monkeypatch.setenv('GRM_ADMISSION_RULE','margin_first')
    paths=list((lt.OUT/'amendment1/offline/lt1_states').glob('*.json'))
    assert len(paths)==70
    for path in paths:
        s=lt.read(path)
        p=adm.decisive_admission_profile(FrozenArena(s),s['question'],exclude=s['exclude'],route_limit=6)
        assert p['rank_plan']==evaluate(s,2)['plan']
        assert p['rank_plan'] and adm.identifier_unbound_abstention(p) is None


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_lt1/ (FIX-6 amendment)')
def test_amendment_and_preflight_enforce_flag(monkeypatch):
    r=lt.verify()
    assert r['effective_admission_rule']=='margin_first'
    assert r['projection']['budget_seconds']==10800
    assert all(a['admission_rule']=='margin_first' for a in r['arms'].values())
    assert len(r['cells'])==52
    monkeypatch.delenv('GRM_ADMISSION_RULE',raising=False)
    assert 'REGISTERED_ADMISSION_RULE_MISMATCH' in lt.preflight()['reasons']
    assert 'amendment2/run_margin_first' in str(worker.RUN)
    assert os.access(lt.OUT/'lead_commands.txt',os.X_OK)


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_lt1/ (FIX-6 amendment)')
def test_worker_rejects_unregistered_rule(tmp_path,monkeypatch):
    r=lt.verify();monkeypatch.delenv('GRM_ADMISSION_RULE',raising=False)
    with pytest.raises(ValueError,match='REGISTERED_ADMISSION_RULE_MISMATCH'):
        worker.execute(r['cells'][0],tmp_path,r,lambda *_:pytest.fail('loader must not run'))


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_lt1/ (FIX-6 amendment)')
@pytest.mark.parametrize('arm',['A','B'])
def test_200_turn_shared_worker_cpu(tmp_path,arm):
    r=lt.verify();run=tmp_path/'cells';run.mkdir()
    env=dict(os.environ,GRM_ADMISSION_RULE='margin_first',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    workers=[]
    for c in [c for c in r['cells'] if c['arm']==arm]:
        d=run/c['id'];d.mkdir()
        result=subprocess.run([sys.executable,'-m','scripts.grm_lt1_worker_cpu',str(run),c['id']],cwd=lt.ROOT,env=env,capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
        w=lt.read(d/'worker.json');workers.append(w)
        assert w['admission_rule']=='margin_first' and w['pid']!=w['previous_pid']
        if c['depends']:assert w['metadata_retained']
        if c['start'] in (71,141):assert w['restart_retained_scores']
        lt.validate_checkpoint(d/'checkpoint',c['stop']+1,lt.binding(arm))
    rows=[json.loads(x) for p in run.glob('*/probes.jsonl') for x in p.read_text().splitlines()]
    assert len(rows)==35 and len({x['probe_id'] for x in rows})==35
    for row in rows:
        info=row['memory']['route_info']
        assert info['admission_rule']=='margin_first'
        assert info['admission_rank_plan'] and not info.get('abstained',False)
        assert row['admission_rule']==row['memory']['admission_rule']=='margin_first'
        assert not set(row['source_ids'])&set(row['recency_ids'])
        assert row['oracle']['score']['exact_correct']
    lt.create(lt.OUT/'amendment3/r3'/f'worker_fake_{arm}.json',dict(status='PASS',evidence_class='CPU fake shared worker, 200 turns with 26 new processes; no model quality claim',admitted=35,cells=26,workers=workers,rows=rows,recap=lt.read(next(run.glob('*/recap.json'))),binding=lt.binding(arm)))
