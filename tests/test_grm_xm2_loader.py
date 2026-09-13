"""Author CPU loader regression; no GPU fit or numerical claim.

Prior art: GRM XM1/LT1 loader doubles and immutable bindings (2026), reused.
Python unittest.mock/AST source replay; ours: shared-callable and actual
lm_head branch/config comparison through both worker entry points.
"""
import ast
from contextlib import nullcontext
import copy
import gc
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import grm_xm1_parity as xm
from scripts import grm_xm1_x2_run as x2
from scripts.grm_xm1_registration import read, sha

OUT = xm.ROOT/'artifacts/grm_xm2'


@pytest.fixture
def adapter_double(monkeypatch):
    # Replay the actual adapter's two load methods with tiny host arrays.
    # The fake QuantLinear records the whole matrix and load context; replacing
    # lm_head with a different mode breaks assertions below, not just a label.
    tree = ast.parse((xm.ROOT/'core/qwen35_tc.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='Qwen35_TC')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ('load_weights', 'from_pretrained')]
    assert len(methods)==2
    calls, configs = [], []
    class QuantLinear:
        WEIGHT_BITS = 4
        LOAD_CONTEXT = None
        def __init__(self, weight):
            calls.append((self.LOAD_CONTEXT, weight.shape, self.WEIGHT_BITS))
    class HostTensor:
        def __init__(self, array): self.array = array
        def to(self, _dtype): return self
        def numpy(self): return self.array
    weights = {
        'model.language_model.embed_tokens.weight': np.ones((3,2),np.float32),
        'model.language_model.norm.weight': np.zeros(2,np.float32),
        'lm_head.weight': np.full((3,2),2,np.float32),
    }
    class Shard:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def keys(self): return weights.keys()
        def get_tensor(self, name): return HostTensor(weights[name])
    monkeypatch.setitem(sys.modules, 'safetensors', SimpleNamespace(safe_open=lambda *_a,**_k:Shard()))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(float32='float32'))
    cfg = SimpleNamespace(num_layers=0, repository='Qwen/Qwen3.5-9B',
        revision=read(xm.XM2_REG)['model_revision'], hidden_dim=2,
        num_heads=1,num_kv_heads=1,head_dim=2,tie_word_embeddings=False,
        attention_layer_indices=lambda:[])
    tc = SimpleNamespace(no_grad=nullcontext, tensor=lambda a,**_k:a)
    block, linear, norm = SimpleNamespace(), SimpleNamespace(), SimpleNamespace()
    ns = dict(np=np, os=os, gc=gc, tc=tc, QuantLinearTC=QuantLinear,
        BlockTC=block, LinearTC=linear, RMSNormTC=norm,
        glob=SimpleNamespace(glob=lambda _p:['CPU_DOUBLE.safetensors']),
        Qwen35Config=SimpleNamespace(from_model_dir=lambda _p:cfg))
    body = ast.ClassDef(name='Adapter',bases=[],keywords=[],body=methods,decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[body],type_ignores=[])),
                 'core/qwen35_tc.py CPU load-method replay','exec'),ns)
    class CPUAdapter(ns['Adapter']):
        def __init__(self, config):
            self.config=config;self.layers=[]
            self.embed_tokens=SimpleNamespace();self.norm=SimpleNamespace()
            configs.append(dict(weight_bits=QuantLinear.WEIGHT_BITS,
                fused_decode=QuantLinear.FUSED_DECODE,compute_dtype=block.COMPUTE_DTYPE,
                linear_dtype=linear.DTYPE,fused_norm=norm.USE_FUSED,
                environment={k:v for k,v in os.environ.items() if k.startswith('GRM_')}))
    mod = SimpleNamespace(Qwen35_TC=CPUAdapter,tc=tc,F=SimpleNamespace(),
                          _snap=lambda:str(OUT/'sources'))
    import core
    monkeypatch.setattr(core,'qwen35_tc',mod,raising=False)
    monkeypatch.setitem(sys.modules,'core.qwen35_tc',mod)
    tokenizer_calls=[]
    def tokenizer(path, **kwargs):
        tokenizer_calls.append((path,kwargs));return object()
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer)))
    return calls,configs,tokenizer_calls,CPUAdapter


def test_xm1_xm2_loader_identity_and_actual_lm_head_mode(adapter_double,tmp_path,monkeypatch):
    assert x2.gpu_loader is xm.gpu_loader  # Loader-identity proof: imported callable, not a copy.
    reg=read(xm.REG)
    effective=xm.xm2_registration()
    cell=effective['cells'][0]
    monkeypatch.setattr(xm,'barrier',lambda *_a,**_k:{'status':'PASS'})
    monkeypatch.setattr(xm,'device_memory',lambda:{'kind':'CPU double'})
    seen=[]
    def native(loaded, actual_cell, actual_reg):
        assert actual_cell['model']=='qwen35' and actual_reg==reg
        seen.append(loaded.info)
        return dict(decode_status='FINAL',model_info=loaded.info)
    monkeypatch.setattr(xm,'native_cell',native)
    monkeypatch.setitem(sys.modules,'scripts.grm_cmc1_gpu_arms',
        SimpleNamespace(gpu_lease=lambda *_:nullcontext()))
    out=tmp_path/'xm2';out.mkdir();(out/'sources').symlink_to(OUT/'sources',target_is_directory=True)
    monkeypatch.setattr(x2,'OUT',out)
    monkeypatch.setattr(x2,'RUN',out/'amendment_1_run')
    effective=copy.deepcopy(effective)
    effective.update(prior_reserved_s=0,prior_attempt_pins={})
    # Actual XM1 default loader resolution (loader=None), then actual X2 run.
    monkeypatch.setenv('GRM_QWEN35_FINAL_CHANNEL','1')
    first=xm.execute(cell,reg,tmp_path/'xm1',mode='gpu')
    second=x2.run(cell,effective,reg)
    assert first['status']==second['status']=='PASS',(first,second)
    calls,configs,tokenizers,adapter=adapter_double
    assert len(seen)==len(configs)==len(tokenizers)==2
    assert seen[0]==seen[1]
    assert configs[0]==configs[1]
    assert calls==[('lm_head.weight',(3,2),4)]*2
    assert configs[0]['weight_bits']==4
    assert configs[0]['compute_dtype']==configs[0]['linear_dtype']=='bfloat16'
    assert configs[0]['fused_decode'] is configs[0]['fused_norm'] is True
    assert configs[0]['environment']=={**reg['environment'],'GRM_QWEN35_FINAL_CHANNEL':'1'}
    assert tokenizers[0]==tokenizers[1]==(str(OUT/'sources'),
        {'local_files_only':True,'trust_remote_code':True})
    assert second['binding']['amendment_sha256']==sha(x2.AMENDMENT)


# GRM-H4: artifact-bound: artifacts/grm_xm2/amendment_1/evidence_before.json was lost
@pytest.mark.campaign_receipt(registration='artifacts/grm_xm2/registration.json + artifacts/grm_xm2/registration_amendment_1.json')
def test_original_and_current_loader_bodies_equal():
    def body(path):
        t=ast.parse(path.read_text())
        return ast.dump(next(n for n in t.body if isinstance(n,ast.FunctionDef)
                            and n.name=='gpu_loader'),include_attributes=False)
    assert body(xm.ROOT/'scripts/grm_xm1_parity.py')==body(
        OUT/'sources/grm_xm1_parity_amendment_1.py')
    evidence=read(OUT/'amendment_1/evidence_before.json')
    for row in evidence['source_comparisons']:
        assert row['identical'] and sha(xm.ROOT/row['path'])==row['xm1_sha256']==row['xm2_sha256']


# GRM-H4: sha-bound: registration source drift: core/grm_admission.py (D2 flip)
@pytest.mark.campaign_receipt(registration='artifacts/grm_xm2/registration.json + artifacts/grm_xm2/registration_amendment_1.json')
def test_amendment_caps_and_immutable_originals():
    amendment=read(x2.AMENDMENT);reg,parent=x2.validate()
    assert sha(xm.XM2_REG)=='7d537ab25ba485094b0235c9b1407d75b542dcbba21ebdd587a6534d16d73c3d'
    assert amendment['registration_sha256']==sha(xm.XM2_REG)
    assert amendment['pins']['scripts/grm_xm1_x2_run.py']==sha(xm.ROOT/'scripts/grm_xm1_x2_run.py')
    for rel,digest in amendment['original_source_pins'].items():
        assert sha(OUT/'amendment_1/sources'/rel)==digest
        assert read(xm.XM2_REG)['pins'][rel]==digest
    assert reg['prior_reserved_s']==sum(read(xm.ROOT/p)['reservation_s'] for p in reg['prior_attempt_pins'])==330
    for rel,digest in reg['prior_attempt_pins'].items():assert sha(xm.ROOT/rel)==digest
    caps={c['id']:c['reservation_s'] for c in reg['cells']}
    for row in amendment['timing_evidence']:
        r=read(xm.ROOT/row['path'])
        assert sha(xm.ROOT/row['path'])==row['sha256']
        assert r['elapsed_s']==row['elapsed_s']<caps[Path(row['path']).stem]
    assert caps['C3l__sup_harbor_restatement']==130
    assert caps['C5__sup_reserve_meridian_docket']==80
    assert set(caps.values())=={80,90,130}
    assert reg['total_reserved_s']==1470
    assert reg['total_reserved_s']+reg['prior_reserved_s']==reg['gpu_budget_s']==1800
    assert x2.receipt_path(reg['cells'][0]).parent==x2.RUN/'cells'
    assert (OUT/'run/cells/C3l__sup_harbor_restatement.json').is_file()


# GRM-H4: sha-bound: registration source drift: core/grm_admission.py (D2 flip)
@pytest.mark.campaign_receipt(registration='artifacts/grm_xm2/registration.json + artifacts/grm_xm2/registration_amendment_1.json')
def test_amendment_parent_and_source_drift_fail_closed(tmp_path,monkeypatch):
    amendment=read(x2.AMENDMENT);path=tmp_path/'amendment.json'
    amendment['registration_sha256']='0'*64
    path.write_text(__import__('json').dumps(amendment))
    monkeypatch.setattr(xm,'XM2_AMENDMENT',path)
    with pytest.raises(xm.XM1Error,match='wrong registration binding'):x2.validate()
    amendment['registration_sha256']=sha(xm.XM2_REG)
    amendment['pins']['scripts/grm_xm1_x2_run.py']='0'*64
    path.write_text(__import__('json').dumps(amendment))
    with pytest.raises(xm.XM1Error,match='implementation source drift'):x2.validate()


# GRM-H4: sha-bound: registration source drift: core/grm_admission.py (D2 flip)
@pytest.mark.campaign_receipt(registration='artifacts/grm_xm2/registration.json + artifacts/grm_xm2/registration_amendment_1.json')
def test_wrong_weight_mode_rejected_before_gpu(monkeypatch):
    monkeypatch.setenv('TC_WEIGHT_BITS','3')
    with pytest.raises(xm.XM1Error,match='TC_WEIGHT_BITS=4'):x2.validate()
