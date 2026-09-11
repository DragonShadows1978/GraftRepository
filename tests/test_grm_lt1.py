"""Registered LT1 author gates. Prior art: C7 r2/a3 CPU parity, checkpoint
and defect injection tests (GRM, 2026); new fixture constraints and scorer
hand cases. No prior art known to me for this exact test composition.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from scripts import grm_lt1 as lt
from scripts.grm_lt1_cpu import visible_answer


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_lt1/registration.json')
def test_registration_and_frozen_fixture():
    r=lt.verify(); f=lt.read(lt.FIX)
    rows=lt.fixture_gate(f)
    lt.create(lt.OUT/'fixture_gate_receipt.json',dict(status='PASS',rows=rows))
    assert len(rows)==35
    assert r['status']=='NON_FIT'
    assert len(r['cells'])==52
    for arm in ('A','B'):
        cells=[c for c in r['cells'] if c['arm']==arm]
        assert [t for c in cells for t in range(c['start'],c['stop']+1)]==list(range(1,201))
        assert all(c['stop']-c['start']<8 for c in cells)
        assert {70,140}<={c['stop'] for c in cells}


def test_recency_gate_rejects_recent_source():
    f=lt.read(lt.FIX);p=f['probes'][0];p['source_turns']=[p['turn']-1]
    with pytest.raises(ValueError,match='SOURCE_TOO_RECENT'):lt.fixture_gate(f)


def test_instruction_collision_and_missing_oracle_are_rejected():
    f=lt.read(lt.FIX);f['probes'][0]['question']+=' Reply UNKNOWN.'
    with pytest.raises(ValueError,match='UPPERCASE_INSTRUCTION_TOKEN'):lt.fixture_gate(f)
    f=lt.read(lt.FIX);f['probes'][0]['oracle_source_texts']=[]
    with pytest.raises(ValueError,match='ORACLE_SOURCE_MISMATCH'):lt.fixture_gate(f)


def test_value_span_and_disjoint_error_categories():
    assert lt.score('We agreed on １６\u00a0beds.','16 beds')['exact_correct']
    assert lt.score('It was north—east.','north-east')['exact_correct']
    assert not lt.score('116 beds','16 beds')['exact_correct']
    assert not lt.score('16 comfortable beds','16 beds')['exact_correct']
    for text,cat in [('unknown','abstention'),('12 beds','wrong_value'),("I cannot assist with that request.",'refusal'),('','abstention')]:
        s=lt.score(text,'16 beds');assert s['category']==cat
        assert sum(s[k] for k in ['wrong_value_error','abstention_error','refusal_error'])==1
    # This is exactly a span metric: mentioning old and current values passes
    # if the current span appears; do not silently add a contradiction policy.
    assert lt.score('12 beds became 16 beds','16 beds')['exact_correct']


def test_fake_attention_matches_actual_constructor():
    from core.gpt_oss20b_tc import GptOss20BConfig,GptOssAttentionTC
    from scripts.grm_c7_diagnose import Model,Codec
    real=GptOssAttentionTC(GptOss20BConfig(layer_types=('full_attention',)),0)
    fake=Model(Codec()).layers[0].self_attn
    assert vars(fake).keys()==vars(real).keys()
    assert set(dir(fake))==set(dir(real))
    assert not hasattr(fake,'live_shift')


def test_visible_reader_requires_sources_and_current_revision():
    assert visible_answer('',"What did we settle on for Medibay's bed count?")=='unknown'
    text="Medibay's bed count will be 12 beds. Actually, Medibay's bed count will be 16 beds, replacing the earlier choice."
    assert visible_answer(text,"What did we settle on for Medibay's bed count?")=='16 beds'
    text="Kestrel's cargo allowance will be 37 crates. Let's call Kestrel 'the Hauler' from now on."
    assert visible_answer(text,"What did we settle on for the Hauler's cargo allowance?")=='37 crates'


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_lt1/registration.json')
@pytest.mark.parametrize('arm',['A','B'])
def test_full_fixture_fake_mounts_answers_and_restarts(tmp_path,arm):
    directory=tmp_path/arm;directory.mkdir()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    for start,stop in [(1,70),(71,140),(141,200)]:
        result=subprocess.run([sys.executable,'-m','scripts.grm_lt1_cpu','--stage',str(directory),
            '--arm',arm,'--start',str(start),'--stop',str(stop)],cwd=lt.ROOT,env=env,capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
    state=lt.read(directory/'state.json')
    lt.create(lt.OUT/f'fake_{arm}_receipt.json',state)
    assert len(set(state['process_ids']))==3
    assert state['next_turn']==201
    assert len(state['rows'])==35
    assert all(x['oracle_score']['exact_correct'] for x in state['rows'])
    failures=[dict(id=x['probe_id'],answer=x['answer'],mounts=x['mounted_ids'],sources=x['source_ids'])
        for x in state['rows'] if not x['memory_score']['exact_correct'] or not set(x['source_ids'])&set(x['mounted_ids'])]
    assert not failures,failures


def test_c7_checkpoint_roundtrip_and_corruption(tmp_path):
    binding=lt.binding('A'); (tmp_path/'repository').mkdir()
    (tmp_path/'repository/payload').write_text('cpu payload')
    lt.checkpoint(tmp_path,dict(next_turn=71,process_id='cpu-checkpoint',transcript=['natural dialogue']),binding)
    assert lt.validate_checkpoint(tmp_path,71,binding)['next_turn']==71
    (tmp_path/'repository/payload').write_text('corrupt')
    with pytest.raises(ValueError):lt.validate_checkpoint(tmp_path,71,binding)


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_lt1/registration.json')
def test_empty_summary_and_budget_fail_closed():
    for arm in ('A','B'):
        s=lt.summary(arm)
        assert s['status']=='NOT_RUN' and s['recap'] is None
        assert all(r['memory']['exact_rate'] is None for r in s['by_distance'])
    assert 'NON_FIT' in lt.preflight()['reasons']
