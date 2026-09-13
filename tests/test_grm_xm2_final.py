"""Author CPU tests, not Qwen GPU parity.
Prior art: Qwen Team (2026) template/delimiters; GRM XM1 numpy loader doubles
and recorded RS4 partitions (2026). Ours: recorded thought trace + explicitly
synthetic final continuation, input-query alignment and legacy byte identity.
"""
import ast
import copy
import json
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace
import numpy as np
import pytest
from scripts import grm_xm1_parity as xm
from scripts import grm_xm1_qwen_final as final
from scripts.grm_xm1_cpu import cpu_loader,Tensor
from scripts.grm_xm1_registration import read,sha

REG=read(xm.REG)
OUT=xm.ROOT/'artifacts/grm_xm2'

@pytest.fixture(scope='module')
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(OUT/'sources'),local_files_only=True)


def test_recorded_thinking_traces_have_zero_final_positions(tok):
    paths=sorted((xm.ROOT/'artifacts/grm_xm1/amendment_1_run/cells/gpu/qwen35').glob('*.json'))
    assert len(paths)==10
    for path in paths:
        r=read(path)['result']
        prompt=tok.decode(r['question_token_ids'],skip_special_tokens=False)
        assert prompt.endswith('<think>\n')
        assert final.final_indices(tok,r['generated_token_ids'],prompt)==[],path
        assert len(r['observed_rows'])==len(r['generated_token_ids'])==32


def test_recorded_trace_continuation_selects_input_queries(tok,monkeypatch):
    trace=read(OUT/'recorded_trace_cpu.json')
    assert trace['evidence_class']=='recorded GPU thinking prefix + synthetic CPU final continuation'
    source=xm.ROOT/trace['source_receipt']
    assert sha(source)==trace['source_sha256']
    original=read(source)['result']['generated_token_ids']
    assert trace['generated_token_ids'][1:33]==original
    cell=next(c for c in REG['cells'] if c['model']=='qwen35' and c['arm']=='C5' and c['probe']=='sup_praxis_fresh')
    loaded=cpu_loader('qwen35',REG)
    loaded.tokenizer=tok
    loaded.model.codec=SimpleNamespace(words=range(248320))
    real_forward=loaded.forward
    question=tok.apply_chat_template([{'role':'user','content':REG['probes'][cell['probe']]['question']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    qids=tok.encode(question,add_special_tokens=False)
    # Bounded production decode is 32 tokens. Replay a shorter recorded prefix
    # through it; full 32-token original is checked independently above/below.
    seq=[248068]+original[:5]+trace['generated_token_ids'][33:]
    first_final=8  # opening + 5 recorded + closing + whitespace
    expected=list(range(first_final,len(seq)-1))
    assert final.final_indices(tok,seq,question)==expected
    index=0;started=False;inputs=[]
    def forward(ids,caches=None,offset=0):
        nonlocal index,started
        logits,caches=real_forward(ids,caches,offset)
        if ids==qids: started=True
        if started:
            inputs.append(list(ids))
            assert index<len(seq)
            values=np.zeros((1,1,248320),np.float32);values[0,0,seq[index]]=1
            index+=1
            logits=Tensor(values)
        return logits,caches
    monkeypatch.setattr(loaded,'forward',forward)
    result=xm.native_cell(loaded,cell,REG)
    assert result['served']=='Quartz-8-Jade.'
    assert result['decode_status']=='FINAL'
    assert result['final_token_indices']==expected
    assert result['final_observation_indices']==[i+1 for i in expected]
    assert all(inputs[i+1]==[seq[i]] for i in expected)
    assert 0 not in result['final_observation_indices']
    assert result['per_layer']['full_attention']['answer_positions']==len(expected)
    # Independent arithmetic on raw rows catches prefill inclusion and shifts.
    rows=result['observed_rows']
    expected_mass=np.mean([r['fed_text_mass'] for i in expected for r in rows[i+1]])
    got=result['per_layer']['full_attention']['mean_over_answer_positions']['fed_text_mass']
    assert got==pytest.approx(expected_mass,abs=1e-12)
    full=final.final_indices(tok,trace['generated_token_ids'],question)
    assert full==trace['expected_final_token_indices']
    assert tok.decode([trace['generated_token_ids'][i] for i in full])=='Quartz-8-Jade.'


@pytest.mark.parametrize('probe',list(REG['probes']))
@pytest.mark.parametrize('arm',['C3l','C5'])
def test_off_result_bytes_equal_preserved_worker(probe,arm,monkeypatch):
    snapshot=OUT/'sources/grm_xm1_parity_amendment_1.py'
    assert sha(snapshot)==read(xm.AMENDMENT)['pins']['scripts/grm_xm1_parity.py']
    # Compile the actual frozen function, independently of the renamed body.
    tree=ast.parse(snapshot.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='native_cell')
    ns=dict(vars(xm));exec(compile(ast.Module(body=[fn],type_ignores=[]),str(snapshot),'exec'),ns)
    cell=dict(model='qwen35',arm=arm,probe=probe)
    monkeypatch.setenv('GRM_QWEN35_FINAL_CHANNEL','0')
    before=cpu_loader('qwen35',REG);after=cpu_loader('qwen35',REG)
    expected=ns['native_cell'](before,cell,REG)
    observed=xm.native_cell(after,cell,REG)
    encode=lambda x:json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()
    assert encode(observed)==encode(expected)
    assert after.model.calls==before.model.calls
    with xm.legacy_environment(REG): assert not final.enabled()


@pytest.mark.parametrize('pin,near',[('off',False),('off',True),('live',False),('live',True)])
def test_rs3_geometry_on_native_qwen_cpu_double(pin,near):
    cell=dict(model='qwen35',arm='C3l',probe='sup_praxis_fresh',capture_pin=pin,seat_near_live=near)
    result=xm.native_cell(cpu_loader('qwen35',REG),cell,REG)
    g=result['seating'];n=g['mount_ntok'];width=g['live_shift']
    assert result['capture']['capture_shift']==(width if pin=='live' else 0)
    assert g['mount_pos0']==(width-n if near else 0)
    assert g['plan_head_last_position']==(width-1 if near else n-1)
    assert g['seat_near_live']==near
    assert result['capture']['payload_sha256']


def test_flags_and_boundaries(tok,monkeypatch):
    monkeypatch.delenv('GRM_QWEN35_FINAL_CHANNEL',raising=False);assert final.enabled()
    monkeypatch.setenv('GRM_QWEN35_FINAL_CHANNEL','garbage')
    with pytest.raises(ValueError,match='invalid GRM'):final.enabled()
    enc=lambda s:tok.encode(s,add_special_tokens=False)
    assert final.final_indices(tok,[], '<think>\n')==[]
    assert final.final_indices(tok,enc('still reasoning'), '<think>\n')==[]
    assert final.final_indices(tok,enc('</think>\n\n')+[tok.eos_token_id], '<think>\n')==[]
    text='<think>hidden</think>\n\nYes.<|im_end|>ignored'
    ids=enc(text);sel=final.final_indices(tok,ids,'<think>\n\n</think>\n\n')
    assert tok.decode([ids[i] for i in sel])=='Yes.'


def test_source_diagnosis_is_exact_and_not_same_fact():
    d=read(OUT/'diagnosis.json');rows={r['probe']:r for r in d['rows']}
    for r in rows.values():
        for path,digest in r['receipt_pins'].items():assert sha(xm.ROOT/path)==digest
        assert [x['layer_index'] for x in r['layers']]==[3,7,11,15,19,23,27,31]
        assert all(t['source_ids_match'] for t in r['tokenizer'].values())
    p=rows['sup_praxis_fresh'];s=rows['sup_solace_fresh']
    assert list(p['tokenizer']['C3l']['expected_mentions'].values())==[2]
    assert list(p['tokenizer']['C5']['expected_mentions'].values())==[0]
    assert list(s['tokenizer']['C3l']['expected_mentions'].values())==[0]
    assert list(s['tokenizer']['C5']['expected_mentions'].values())==[2]
    assert all(x['fed_minus_mount']>0.10 for x in p['layers'])
    assert all(x['fed_minus_mount']>0.10 for x in s['layers'][1:])


def test_registration_matrix_budget_and_dry_run():
    from scripts.grm_xm1_x2_run import validate
    reg,_=validate()
    assert len(reg['cells'])==16
    assert sum(c['reservation_s'] for c in reg['cells'])==reg['total_reserved_s']
    assert reg['total_reserved_s']+reg['prior_reserved_s']<=1800
    assert len({c['id'] for c in reg['cells']})==16
    assert sum(c['arm']=='C5' for c in reg['cells'])==5
    for probe in ('sup_praxis_fresh','sup_solace_fresh'):
        assert {(c['capture_pin'],c['seat_near_live']) for c in reg['cells'] if c['probe']==probe and c['arm']=='C3l'}=={('off',False),('off',True),('live',False),('live',True)}
    commands=[s for s in (OUT/'lead_commands.txt').read_text().splitlines() if s and not s.startswith('#')]
    assert len(commands)==16
    for command in commands:
        assert 'flock' not in command
        run=subprocess.run(shlex.split(command)+['--dry-run'],cwd=xm.ROOT,text=True,capture_output=True,timeout=20)
        assert run.returncode==0,(command,run.stdout,run.stderr)
        r=json.loads(run.stdout);assert r['status']=='DRY_RUN' and r['gpu_executed'] is False
