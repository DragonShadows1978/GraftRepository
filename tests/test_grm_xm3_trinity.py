"""Author CPU gates; native attention AST, synthetic weights, no CUDA.

Prior art: GRM XM1 AST doubles, RS4 partitions, LT1 negative binding gates
(2026); Su et al. RoFormer (2021) trigonometric oracle, unverified — lead to
check 2104.09864. Ours: adversarial position controls and matched source test.
"""
import copy
import json
import shlex
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from scripts import grm_xm1_parity as xm
from scripts import grm_xm1_trinity_x3 as x3
from scripts.grm_xm1_cpu import cpu_loader, Model, Tensor, Engine

BASE = xm.read(xm.REG)
PROBES = ('sup_praxis_fresh', 'sup_harbor_restatement')


def cell(probe, arm):
    return dict(model='trinity', probe=probe, arm=arm)


@pytest.mark.parametrize('probe', PROBES)
@pytest.mark.parametrize('arm', ('C5', 'C3l'))
def test_off_canonical_bytes_identical(probe, arm):
    a = xm.native_cell(cpu_loader('trinity', BASE), cell(probe, arm), BASE)
    b = x3.matched_cell(cpu_loader('trinity', BASE), cell(probe, arm), BASE)
    assert json.dumps(a, sort_keys=True).encode() == json.dumps(b, sort_keys=True).encode()


@pytest.mark.parametrize('near', (False, True))
@pytest.mark.parametrize('pin', ('off', 'live'))
def test_native_injected_keys_against_independent_rotation(near, pin):
    loaded = cpu_loader('trinity', BASE)
    ids = loaded.tokenizer.encode('alpha beta gamma')
    n, width = len(ids), 96
    payload = xm.capture(loaded, ids, 0 if pin=='off' else width-len(ids))
    frozen = copy.deepcopy(payload)
    xm.seat(loaded, payload, width if near else n)
    for _, att in loaded.attentions():
        att.live_shift = width
    _, caches = loaded.forward([1, 2])
    for i, att in loaded.attentions():
        raw, values = payload[i]
        expected = raw.copy()
        if att.is_local_attention:
            # Independent explicit coordinate formula, not model.rotate/seat.
            theta = ((width-n if near else 0) + np.arange(n))[:, None] * np.array([1., .01])[None, :]
            u, v = raw[..., :2], raw[..., 2:]
            expected = np.concatenate([u*np.cos(theta)-v*np.sin(theta),
                                       v*np.cos(theta)+u*np.sin(theta)], axis=-1)
        np.testing.assert_allclose(caches[i][0].numpy()[:, :, :n], expected, atol=1e-5, rtol=0)
        np.testing.assert_array_equal(caches[i][1].numpy()[:, :, :n], values)
        for actual, original in zip(payload[i], frozen[i]):
            assert actual.tobytes() == original.tobytes()
        if not att.is_local_attention:
            assert caches[i][0].numpy()[:, :, :n].tobytes() == raw.tobytes()


@pytest.mark.parametrize('probe', PROBES)
def test_factorial_has_identical_tokens_and_explicit_geometry(probe):
    live = x3.matched_cell(cpu_loader('trinity', BASE), cell(probe,'C5'), BASE, enabled=True)
    n = len(live['source_token_ids'])
    for pin in ('off','live'):
        for near in (False,True):
            mount = x3.matched_cell(cpu_loader('trinity', BASE), cell(probe,'C3l'), BASE,
                                    enabled=True, capture_pin=pin, near=near)
            assert mount['source_token_ids'] == live['source_token_ids']
            assert mount['question_token_ids'] == live['question_token_ids']
            s = mount['seating']
            assert s['capture_shift'] == (96-n if pin=='live' else 0)
            assert s['mount_pos0'] == (96-n if near else 0)
            assert s['live_shift'] == live['seating']['question_pos0'] == 96
            assert s['payload_unchanged']
            assert not mount['decode_window']['semantic_answer_valid']
            if near and pin=='live':
                for kind in ('full_attention','sliding_attention'):
                    a=live['per_layer'][kind]['mean_over_answer_positions']['fed_text_mass']
                    b=mount['per_layer'][kind]['mean_over_answer_positions']['mounted_mass']
                    assert abs(a-b)<1e-6


def test_reject_width_and_unknown_pin():
    base=copy.deepcopy(BASE)
    base['probes'][PROBES[0]]['capture_texts']=[' '.join(['long']*200)]
    with pytest.raises(xm.XM1Error,match='96-seat'):
        x3.matched_cell(cpu_loader('trinity',base),cell(PROBES[0],'C5'),base,enabled=True)
    with pytest.raises(xm.XM1Error,match='unknown capture'):
        x3.matched_cell(cpu_loader('trinity',BASE),cell(PROBES[0],'C3l'),BASE,enabled=True,capture_pin='bad')


def test_historical_audit_no_vacuous_bands_or_template():
    audit=xm.read(x3.OUT/'historical_audit_amendment_1.json')
    assert len(audit['cells'])==10
    assert audit['total_layer_rows']>10000
    for r in audit['cells']:
        assert r['source_ids_match'] and r['question_ids_match'] and r['rendered_match']
        assert r['skip_special_decode_matches'] and not r['band_errors']
        assert not r['value_span']['correct']
        assert r['S_max']<2048
    praxis=next(r for r in audit['cells'] if r['probe']==PROBES[0] and r['arm']=='C5')
    assert not any(praxis['expected_in_source'].values())
    harbor=next(r for r in audit['cells'] if r['probe']==PROBES[1] and r['arm']=='C5')
    assert all(harbor['expected_in_source'].values())


# GRM-H4: sha-bound: X3 source drift: scripts/grm_xm1_cpu.py (D2 fallout)
@pytest.mark.campaign_receipt(registration='artifacts/grm_xm3/registration.json + artifacts/grm_xm3/implementation_amendment_3.json')
def test_registration_dry_run_all_commands_no_gpu():
    reg=x3.validate()
    assert reg['gpu']['cells']*reg['gpu']['reservation_per_cell_s']<=1800
    commands=[l for l in (x3.OUT/'lead_commands.txt').read_text().splitlines() if l and not l.startswith('#')]
    assert len(commands)==14
    for line in commands:
        assert 'flock' not in line
        result=subprocess.run(shlex.split(line)+['--dry-run'],cwd=xm.ROOT,capture_output=True,text=True,timeout=20)
        assert result.returncode==0,(line,result.stderr)
        r=json.loads(result.stdout)
        assert r['status']=='DRY_RUN' and not r['gpu_executed']
    assert 'tensor_cuda' not in sys.modules


# GRM-H4: sha-bound: X3 source drift: scripts/grm_xm1_cpu.py (D2 fallout)
@pytest.mark.campaign_receipt(registration='artifacts/grm_xm3/registration.json + artifacts/grm_xm3/implementation_amendment_3.json')
def test_prerequisites_reject_missing_wrong_control_budget_gap(tmp_path):
    reg=x3.validate()
    with pytest.raises(xm.XM1Error,match='control missing'):
        x3.prerequisites(reg,PROBES[0],'C0',tmp_path)
    path=tmp_path/'cells'/f'C5__{PROBES[0]}.json'
    control=dict(binding=x3.cell_binding(PROBES[0],'C5'),status='RED_NO_ANSWER',result={'value_span':{'correct':False}})
    xm.create(path,control)
    with pytest.raises(xm.XM1Error,match='no correct answer'):
        x3.prerequisites(reg,PROBES[0],'C3l',tmp_path)
    control.update(status='PASS',result={'value_span':{'correct':True}})
    path.write_text(json.dumps(control))
    x3.prerequisites(reg,PROBES[0],'C3l',tmp_path)
    xm.create(tmp_path/'attempts/a.json',{'reservation_s':1800,'reserved_at':0})
    with pytest.raises(xm.XM1Error,match='budget exhausted'):
        x3.prerequisites(reg,PROBES[0],'C5',tmp_path)
    (tmp_path/'attempts/a.json').write_text(json.dumps({'reservation_s':120,'reserved_at':x3.time.time()}))
    with pytest.raises(xm.XM1Error,match='gap'):
        x3.prerequisites(reg,PROBES[0],'C5',tmp_path)


# GRM-H4: sha-bound: X3 source drift: scripts/grm_xm1_cpu.py (D2 fallout)
@pytest.mark.campaign_receipt(registration='artifacts/grm_xm3/registration.json + artifacts/grm_xm3/implementation_amendment_3.json')
def test_pin_drift_fails_closed(tmp_path,monkeypatch):
    reg=x3.validate()
    modified=copy.deepcopy(reg)
    modified['pins']['scripts/grm_xm1_parity.py']='0'*64
    p=tmp_path/'registration.json';p.write_text(json.dumps(modified))
    amendment=xm.read(x3.PINS);amendment['registration_sha256']=xm.sha(p)
    ap=tmp_path/'implementation.json';ap.write_text(json.dumps(amendment))
    monkeypatch.setattr(x3,'REG',p);monkeypatch.setattr(x3,'PINS',ap)
    with pytest.raises(xm.XM1Error,match='source drift'):
        x3.validate()


class CoupledModel(Model):
    """Propagate native attention outputs between layers, unlike XM1 base double.

    Prior art: XM1 native AST loader (2026); causal attention Vaswani et al.
    (2017), unverified — lead to check Attention Is All You Need. Ours: add
    hidden-state coupling so a full NoPE layer sees previous local output.
    """
    def __call__(self, ids, caches=None, kv_caches=None, position_offset=0, **kw):
        caches = caches if caches is not None else kv_caches
        n=ids.shape[1]
        x=Tensor(np.sin(ids.astype(np.float32)[...,None]*np.array([.1,.2,.3,.4],np.float32)))
        out=[]
        for i,layer in enumerate(self.layers):
            x,cache=layer.self_attn(x,self.rope_cos,self.rope_sin,position_offset,
                                   None if caches is None else caches[i])
            out.append(cache)
        logits=np.zeros((1,1,len(self.codec.words)),np.float32)
        logits[0,0,0 if n==1 else self.output]=1
        return Tensor(logits),out


def coupled_loader():
    loaded=cpu_loader('trinity',BASE)
    model=CoupledModel.__new__(CoupledModel)
    model.__dict__.update(loaded.model.__dict__)
    loaded.model=model
    def causal(q,k,v,**kw):
        L,S=q.shape[2],k.shape[2]
        scores=np.matmul(q.a,k.a.swapaxes(-1,-2))*kw['scale']
        if kw.get('is_causal'):
            scores=scores+np.where(np.arange(S)[None,:] <= (S-L+np.arange(L))[:,None],0.,-1e4)
        if kw.get('attn_mask') is not None:
            scores=scores+kw['attn_mask'].a
        return Engine.matmul(Tensor(scores).softmax(-1),v)
    loaded.module._trinity_scaled_attention=causal
    return loaded


@pytest.mark.parametrize('probe',PROBES)
def test_coupled_native_layers_matched_geometry(probe):
    feed=x3.matched_cell(coupled_loader(),cell(probe,'C5'),BASE,enabled=True)
    mount=x3.matched_cell(coupled_loader(),cell(probe,'C3l'),BASE,enabled=True,capture_pin='live',near=True)
    assert feed['generated_token_ids']==mount['generated_token_ids']
    for fs,ms in zip(feed['observed_rows'],mount['observed_rows']):
        for f,m in zip(fs,ms):
            assert abs(f['fed_text_mass']-m['mounted_mass'])<1e-6
            assert abs(f['question_mass']-m['question_mass'])<1e-6


# GRM-H4: sha-bound: X3 source drift: scripts/grm_xm1_cpu.py (D2 fallout)
@pytest.mark.campaign_receipt(registration='artifacts/grm_xm3/registration.json + artifacts/grm_xm3/implementation_amendment_3.json')
def test_gpu_entrypoint_mock_owns_lease_and_reservation(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from scripts import grm_cmc1_gpu_arms as cmc
    active=[];seen=[]
    @contextmanager
    def lease(seconds,wait):
        assert (seconds,wait)==(120,0)
        active.append(True);seen.append('lease')
        try: yield
        finally: active.pop()
    def loader(*_):
        assert active and len(list((tmp_path/'attempts').glob('*.json')))==1
        loaded=cpu_loader('trinity',BASE)
        loaded.model.output=loaded.tokenizer.encode(BASE['probes'][PROBES[0]]['expected_values'][0])[0]
        seen.append('loader')
        return loaded
    monkeypatch.setattr(cmc,'gpu_lease',lease)
    monkeypatch.setattr(xm,'gpu_loader',loader)
    monkeypatch.setattr(xm,'barrier',lambda *_:{'status':'PASS'})
    # Only output is redirected. Immutable registration and pins stay real.
    real_validate=x3.validate
    reg=real_validate()
    monkeypatch.setattr(x3,'validate',lambda:reg)
    monkeypatch.setattr(x3,'OUT',tmp_path)
    args=['--probe',PROBES[0],'--treatment','C5','--xm3-matched']
    assert x3.main(args)==0
    assert seen==['lease','loader']
    result=xm.read(tmp_path/'cells'/f'C5__{PROBES[0]}.json')
    assert result['result']['value_span']['correct']
    assert x3.main(args)==0
    assert seen==['lease','loader']


class X3GateReceipt:
    """Persist the last suite result and clean its exact registered basetemp.

    Prior art: pytest sessionfinish reporting/temporary-directory lifecycle
    (pytest contributors); GRM immutable receipts (2026). Ours: scoped receipt.
    """
    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        import shutil
        import time
        out=xm.ROOT/'artifacts/grm_xm3'
        terminal=session.config.pluginmanager.getplugin('terminalreporter')
        receipt=dict(evidence_class='author CPU pytest suite; not blind verification',
                     exitstatus=int(exitstatus),collected=session.testscollected,
                     passed=len(terminal.stats.get('passed',[])),failed=session.testsfailed,
                     argv=list(session.config.invocation_params.args),gpu_executed=False)
        base=session.config.getoption('basetemp')
        if base and Path(base).resolve()==(out/'tmp').resolve():
            shutil.rmtree(out/'tmp',ignore_errors=False)
            receipt['basetemp_cleaned']=not (out/'tmp').exists()
        xm.create(out/'pytest_runs'/f'{time.time_ns()}.json',receipt)
        with (out/'IMPLEMENTATION_LEDGER.md').open('a') as f:
            f.write('\nCPU suite receipt: '+json.dumps(receipt,sort_keys=True)+'\n')
        report=out/'REPORT.md'
        if report.exists():
            with report.open('a') as f:
                f.write('\nFinal pytest receipt (written by sessionfinish): '+json.dumps(receipt,sort_keys=True)+'\n')


@pytest.fixture(scope='session',autouse=True)
def record_suite_and_cleanup(request):
    request.config.pluginmanager.register(X3GateReceipt(),'xm3-gate-receipt')


def test_registered_gap_falsifier_not_hidden_by_correct_answer():
    def result(fed, mount, correct=True):
        return dict(value_span={'correct':correct},per_layer={
            kind:{'mean_over_answer_positions':{'fed_text_mass':fed,'mounted_mass':mount}}
            for kind in ('full_attention','sliding_attention')})
    assert x3.parity_comparison(result(.8,0),result(0,.5))['status']=='RED_PARITY'
    assert x3.parity_comparison(result(.8,0),result(0,.8))['status']=='PASS'
    assert x3.parity_comparison(result(.8,0),result(0,.8,False))['status']=='RED_PARITY'
    with pytest.raises(xm.XM1Error,match='missing/nonfinite'):
        x3.parity_comparison(result(float('nan'),0),result(0,.8))
