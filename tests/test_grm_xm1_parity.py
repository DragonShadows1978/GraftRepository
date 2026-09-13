"""XM1 author CPU baseline; not blind verification or GPU certification.

Prior art: RS4 exact comparators/partition fixtures and LT1 receipt binding
(GRM contributors, 2026). Ours: adapter payload/dispatch and fail-closed cases.
"""
import copy
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import numpy as np
import pytest
from scripts import grm_xm1_parity as xm
from scripts.grm_xm1_cpu import cpu_loader
from scripts.grm_xm1_registration import read, sha, create, MODELS, ARMS

REG=read(xm.REG)
CELLS=REG['cells']


def test_00_rs4_historical_reassembly_is_float_equal():
    # Reassemble actual recorded raw masses, never synthesize the expected
    # row. This validates serialization/comparator, NOT a model re-execution.
    from scripts.grm_rs4_ceiling_gpu import assemble_table
    checked=0
    for key,expected in REG['rs4_references'].items():
        raw=read(xm.ROOT/'artifacts/grm_rs4'/expected['receipt'])
        got=next(r for r in assemble_table(raw['probes']) if r['probe_id']==expected['probe_id'] and r['variant']=='registered')
        assert xm.reference_diff(expected,got)=={},key
        checked+=1
    assert checked==10


# GRM-H4: sha-bound: registration source drift: core/grm_admission.py (D2 flip)
@pytest.mark.campaign_receipt(registration='artifacts/grm_xm1/registration.json')
def test_registration_and_pins_are_immutable_and_relative():
    assert xm.validate_registration()==REG
    assert len(REG['probes'])==5 and len(CELLS)==60
    assert list(REG['model_order'])==list(MODELS)
    assert REG['gpu_budget_s_per_model']==1800
    for model in MODELS:
        assert sum(c['estimate_s'] for c in CELLS if c['model']==model)<=1800
    for rel in REG['pins']:
        assert not Path(rel).is_absolute() and '..' not in Path(rel).parts


@pytest.mark.parametrize('cell',CELLS,ids=lambda c:f"{c['model']}-{c['arm']}-{c['probe']}")
def test_each_registered_cell_receipts_and_resumes(tmp_path,cell):
    got=xm.execute(cell,REG,tmp_path,'cpu',cpu_loader)
    assert got['status']==('PASS' if REG['adapters'][cell['model']]['status']=='READY' else 'NO_GRAFT_PATH'),got
    path=xm.cell_path(tmp_path,cell,'cpu')
    original=path.read_bytes()
    assert xm.execute(cell,REG,tmp_path,'cpu',lambda *_:pytest.fail('resume reloaded model'))==got
    assert path.read_bytes()==original
    assert got['memory']['peak'] is None and not got['gpu_executed']
    if got['status']=='PASS':
        r=got['result'];assert r['served']=='CPU-double'
        assert r['observed_rows'] and all(len(s)==2 for s in r['observed_rows'])
        for step in r['observed_rows']:
            for row in step:
                b=row['bounds']
                assert sum(b[k][1]-b[k][0] for k in ('sink_band','mount_band','prior_live_band','fed_text_band','question_band','answer_band'))==b['S']
                assert xm.partition_sums_to_one(row)
        if cell['arm']=='C3l':
            assert r['seating']['plan_head_last_position']==r['seating']['live_shift']-1
            assert r['seating']['mount_ntok']==len(r['source_token_ids'])
        else:
            assert r['seating']['mount_ntok']==0


@pytest.mark.parametrize('model',['gpt-oss','qwen35','minicpm3','trinity'])
def test_real_capture_source_and_constant_delta_seat(model):
    loaded=cpu_loader(model,REG)
    ids=loaded.tokenizer.encode('alpha beta gamma')
    payload=xm.capture(loaded,ids,80)
    original=copy.deepcopy(payload)
    seating=xm.seat(loaded,payload,80)
    assert seating['delta_positions']==80-len(ids)
    for i,att in loaded.attentions():
        for a,b in zip(payload[i],original[i]):np.testing.assert_array_equal(a,b)
        first,second=payload[i]
        a,b=att.inject_kv[:2]
        if model=='minicpm3':
            np.testing.assert_array_equal(a.numpy(),first)
            assert not np.array_equal(b.numpy(),second)
        else:
            np.testing.assert_array_equal(b.numpy(),second)
            if model=='trinity' and not att.is_local_attention:
                np.testing.assert_array_equal(a.numpy(),first)
            else:
                assert not np.array_equal(a.numpy(),first)
            if model=='qwen35':np.testing.assert_array_equal(a.numpy()[...,2:],first[...,2:])
        assert att.graft_seats==len(ids) and att.live_shift==80


def test_exact_barrier_rejects_every_single_float_bit_change():
    for expected in REG['rs4_references'].values():
        for key,value in expected.items():
            if '_mass_' in key and isinstance(value,float):
                observed=copy.deepcopy(expected)
                observed[key]=float(np.nextafter(value,np.inf))
                assert key in xm.reference_diff(expected,observed)
        assert xm.reference_diff(expected,{})
        assert 'served' in xm.reference_diff(expected,dict(expected,served='wrong'))


def test_cpu_receipts_cannot_unlock_gpu(tmp_path):
    cell=next(c for c in CELLS if c['model']=='gpt-oss')
    xm.execute(cell,REG,tmp_path,'cpu',cpu_loader)
    assert xm.barrier(tmp_path,REG)['status']=='BLOCKED'
    other=next(c for c in CELLS if c['model']=='qwen35')
    with pytest.raises(xm.XM1Error,match='parity barrier'):
        xm.execute(other,REG,tmp_path,'gpu',lambda *_:pytest.fail('barrier loaded GPU'))


def test_resume_rejects_unbound_and_create_refuses_overwrite(tmp_path):
    c=CELLS[0];got=xm.execute(c,REG,tmp_path,'cpu',cpu_loader)
    path=xm.cell_path(tmp_path,c,'cpu')
    with pytest.raises(FileExistsError):create(path,got)
    got['binding']['registration_sha256']='bad'
    path.write_text(json.dumps(got))
    with pytest.raises(xm.XM1Error,match='binding mismatch'):xm.execute(c,REG,tmp_path,'cpu',cpu_loader)


def test_environment_restored_and_round_two_disabled(monkeypatch):
    monkeypatch.setenv('GRM_FOLD_ALIAS_GUARD','1')
    monkeypatch.setenv('GRM_TEST_LEAK','1')
    with xm.legacy_environment(REG):
        import os
        assert 'GRM_TEST_LEAK' not in os.environ
        for key,value in REG['environment'].items():assert os.environ[key]==value
    import os
    assert os.environ['GRM_FOLD_ALIAS_GUARD']=='1'


def test_value_spans_use_rs4_comparator():
    p=REG['probes']['sup_harbor_restatement']
    assert xm.value_score('Nacre\u20116\u2011Blue',p)['correct']
    assert not xm.value_score('Nacre-6-Blue or Morrow-5-Red',p)['correct']
    assert xm.value_score("I don't have that information.",p)['refusal']


# GRM-H4: sha-bound: every pinned dry-run exits 1 on the same drift
@pytest.mark.campaign_receipt(registration='artifacts/grm_xm1/registration.json')
def test_lead_commands_all_accept_appended_dry_run():
    commands=[line for line in (xm.OUT/'lead_commands.txt').read_text().splitlines() if line and not line.startswith('#')]
    assert len(commands)==60
    for command in commands:
        result=subprocess.run(shlex.split(command)+['--dry-run'],cwd=xm.ROOT,text=True,capture_output=True,timeout=20)
        assert result.returncode==0,(command,result.stdout,result.stderr)
        payload=json.loads(result.stdout)
        assert payload['status']=='DRY_RUN' and payload['gpu_executed'] is False
