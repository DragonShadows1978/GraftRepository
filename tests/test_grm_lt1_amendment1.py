"""Author CPU contracts, not blind verification or GPT-OSS quality.
Prior art: C7 a3 doubles, C2 checkpoint tests (GRM contributors, 2026).
New counterexamples exercise lead's three rules; no prior art known to me
for the exact test composition. No fixture-derived threshold adjustment.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from scripts import grm_lt1 as lt
from scripts import grm_lt1_worker as worker
from scripts.grm_lt1_admission import evaluate,shaped_tokens
from core.grm_admission import MARGIN_THRESHOLD
# Import archived-independent pure r1 checks without depending on tests package layout.
import importlib.util
_spec=importlib.util.spec_from_file_location('lt1_old_tests',Path(__file__).with_name('test_grm_lt1.py'))
_old=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_old)
for _name in ('test_recency_gate_rejects_recent_source',
    'test_instruction_collision_and_missing_oracle_are_rejected',
    'test_value_span_and_disjoint_error_categories',
    'test_fake_attention_matches_actual_constructor',
    'test_visible_reader_requires_sources_and_current_revision',
    'test_c7_checkpoint_roundtrip_and_corruption'):
    globals()[_name]=getattr(_old,_name)



def test_fixture_only_assistant_fact_changes_and_effective_registration():
    r=lt.verify();f=lt.read(lt.FIX)
    old=lt.read(lt.OUT/'amendment1/before/fixtures/lt1/dialogue.json')
    changed=[]
    for x,y in zip(f['turns'],old['turns']):
        if x!=y:
            assert x['kind']=='fact'
            changed.append(x['turn']);x['assistant']=y['assistant']
    assert f==old and len(changed)==60
    assert len(lt.fixture_gate(f))==35
    assert r['status']=='FIT_ESTIMATE' and r['projection']['budget_seconds']==10800
    assert lt.read(lt.REG)['status']=='NON_FIT'
    assert len(r['cells'])==52
    for a in ('A','B'):
        cs=[c for c in r['cells'] if c['arm']==a]
        assert [t for c in cs for t in range(c['start'],c['stop']+1)]==list(range(1,201))
        assert all(c['stop']-c['start']<8 and c['estimate_seconds']<=280 for c in cs)


def test_identifier_shape_numbers_names_hyphens_and_common_words():
    assert shaped_tokens("What did we settle on for Promenade's lamp spacing?")==['promenade']
    assert shaped_tokens('What is the allocation for Iona Vale at dock-amber 42?')==['iona','vale','dock-amber','42']
    assert shaped_tokens('what did we settle on for lamp spacing?')==[]


def state(question,margin):
    return dict(question=question,nodes=[dict(text='Promenade lamp spacing 9 voxels'),dict(text='Harbor 12 berths')],
                eligible=[0,1],ranking=[0,1],split_members=[],margin=margin,scores={'0':margin,'1':0.0})


def test_core_rule0_refusal_rule1_mount_and_no_identifier_fallback():
    s=state("What did we settle on for Promenade's lamp spacing?",0.3)
    assert evaluate(s,0)['refused']
    assert evaluate(s,1)['plan']==[0]
    s['question']='what did we settle on for lamp spacing?'
    assert evaluate(s,1)['plan']==[0,1] and not evaluate(s,1)['refused']
    s['question']='What is missing at Absentport?'
    assert evaluate(s,1)['refused'] and evaluate(s,2)['plan']==[0]


def test_rule2_strict_threshold_and_identifier_only_exact_tiebreak():
    s=state('What is at Harbor?',MARGIN_THRESHOLD)
    assert evaluate(s,2)['plan']==[0,1]
    s['margin']=MARGIN_THRESHOLD+1e-12
    assert evaluate(s,2)['plan']==[0]
    s.update(margin=0,scores={'0':0.,'1':0.})
    assert evaluate(s,2)['plan']==[1,0]
    s.update(margin=.01,scores={'0':.01,'1':0.})
    assert evaluate(s,2)['plan']==[0,1]
    s['margin']=None
    assert evaluate(s,2)['status']=='UNRESOLVED'


def test_orphan_red_and_checkpoint_process_integrity(tmp_path):
    d=tmp_path/'cells/A-001-008';d.mkdir(parents=True)
    lt.create(d/'reservation.json',dict(seconds=285))
    with pytest.raises(ValueError,match='ORPHAN'):worker.accounting(tmp_path)
    lt.create(d/'controller.json',dict(status='RED',charged_seconds=285))
    with pytest.raises(ValueError,match='PRIOR_CELL_RED'):worker.accounting(tmp_path)
    cp=tmp_path/'checkpoint';cp.mkdir();(cp/'repository').mkdir()
    lt.checkpoint(cp,dict(next_turn=9,process_id='bound-process'),lt.binding('A'))
    manifest=lt.read(cp/'checkpoint.json');manifest['process_id']='other-process'
    (cp/'checkpoint.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='CHECKPOINT_STATE_MISMATCH'):lt.validate_checkpoint(cp,9,lt.binding('A'))
    with pytest.raises(TimeoutError):worker.check_deadline(0)


def test_empty_summary_does_not_claim_quality():
    for arm in ('A','B'):
        s=lt.summary(arm)
        assert s['status']=='NOT_RUN' and s['recap'] is None and s['restart_retained'] is None
        assert all(x['memory']['exact_rate'] is None for x in s['by_distance'])


@pytest.mark.parametrize('arm',['A','B'])
def test_worker_all_8_turn_cells_resume_and_designated_restarts(tmp_path,arm):
    # Shared GPU worker body, fresh CPU process for EVERY registered cell.
    r=lt.verify();run=tmp_path/'cells';run.mkdir()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    for c in [c for c in r['cells'] if c['arm']==arm]:
        d=run/c['id'];d.mkdir()
        completed=subprocess.run([sys.executable,'-m','scripts.grm_lt1_worker_cpu',str(run),c['id']],
                cwd=lt.ROOT,env=env,capture_output=True,text=True)
        assert completed.returncode==0,completed.stdout+completed.stderr
        w=lt.read(d/'worker.json');assert w['admission_rule']==0
        assert w['pid']!=w['previous_pid']
        if c['depends']:assert w['metadata_retained']
        if c['start'] in (71,141):assert w['restart_retained_scores']
        lt.validate_checkpoint(d/'checkpoint',c['stop']+1,lt.binding(arm))
    rows=[json.loads(x) for p in run.glob('*/probes.jsonl') for x in p.read_text().splitlines()]
    assert len(rows)==35 and len({r['probe_id'] for r in rows})==35
    assert all(r['oracle']['score']['exact_correct'] for r in rows)
    assert all(r['memory']['score']['category']=='abstention' for r in rows)
    assert all(not set(r['source_ids'])&set(r['recency_ids']) for r in rows)
    assert len(list(run.glob('*/recap.json')))==1
    receipt=lt.OUT/'amendment1'/f'worker_fake_{arm}.json'
    lt.create(receipt,dict(status='PASS',evidence_class='author CPU fake shared worker body; no GPU',
        cells=26,probes=35,oracle_exact=35,rule0_abstentions=35,
        workers=[lt.read(p) for p in sorted(run.glob('*/worker.json'))],binding=lt.binding(arm)))
