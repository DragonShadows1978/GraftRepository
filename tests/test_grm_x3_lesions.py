"""Author CPU baseline, not blind verification.
Prior art: house DET1 byte/delta oracle and adversarial mutation rules (2026).
Own cases target X3 mask/payload locality, scoring direction and receipt safety.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scripts import grm_x3_diagnostic as x

@pytest.fixture
def base():
    arrays={'tokens.prompt_ids':x.freeze_array(np.array([101,102],dtype=np.int64)),
        'model.rope_cos':x.freeze_array(np.arange(20,dtype=np.float32)),
        'sink.rows':x.freeze_array(np.array([1.,-0.],dtype=np.float32))}
    for li in range(3):
        for kind in ('injection','mounted_graft.9'):
            for name in ('k','v'):
                arrays[f'{kind}.layer_{li:02d}.{name}']=x.freeze_array(np.arange(40,dtype=np.float32).reshape(1,2,5,4)+li)
        mask=np.ones((1,1,2,7),np.uint8);mask[0,0,0,6]=0
        if li==1:mask[...,:2]=0
        arrays[f'mask.layer_{li:02d}.allowed']=x.freeze_array(mask)
    return x.Snapshot({'synthetic':True,'pos':0,'mount_ids':[9,4],'route':{'choice':[9,4]},'rng':[1,2,3]},arrays,{'target':(3,5),'sham':(2,3)},'injection',3)

def donor(base):
    return {k:np.full((1,2,2,4),-7.,np.float32) for k in base.arrays if k.startswith('injection.')}

@pytest.mark.parametrize('arm',x.ARMS)
def test_all_other_state_bitwise_identical(base,arm):
    before=base.fingerprint();other=x.fork(base,arm,donor(base))
    receipt=x.verify_delta(base,other,arm,donor(base))
    assert receipt['canonical_host_bytes_pinned'] and base.fingerprint()==before
    assert x.canonical(base.state)==x.canonical(other.state)
    for k,a in base.arrays.items():
        if k.startswith('injection.') and arm=='swap':
            assert a[:,:,:3,:].tobytes()==other.arrays[k][:,:,:3,:].tobytes()
        elif k.startswith('mask.') and arm in ('remove','sham'):
            lo,hi=base.spans['sham' if arm=='sham' else 'target']
            assert a[...,:lo].tobytes()==other.arrays[k][...,:lo].tobytes()
            assert a[...,hi:].tobytes()==other.arrays[k][...,hi:].tobytes()
            assert not other.arrays[k][...,lo:hi].any()
        else:assert a.tobytes()==other.arrays[k].tobytes()

@pytest.mark.parametrize('arm',['zero','same_payload'])
def test_controls_exact_and_same_payload_is_independent_copy(base,arm):
    other=x.fork(base,arm)
    assert other.fingerprint()==base.fingerprint()
    if arm=='same_payload':
        assert not np.shares_memory(other.arrays['injection.layer_00.k'],base.arrays['injection.layer_00.k'])
    with pytest.raises(ValueError):other.arrays['injection.layer_00.k'][0,0,0,0]=1

@pytest.mark.parametrize('field',['tokens.prompt_ids','model.rope_cos','sink.rows','mounted_graft.9.layer_01.v','injection.layer_00.k'])
def test_unrelated_byte_corruption_rejected(base,field):
    other=x.fork(base,'remove');bad=other.arrays[field].copy();bad.flat[0]+=1;other.arrays[field]=bad
    with pytest.raises(x.X3Error,match='unauthorized'):x.verify_delta(base,other,'remove')

def test_nested_scalar_drift_rejected(base):
    other=x.fork(base,'remove');other.state['route']['choice'].reverse()
    with pytest.raises(x.X3Error,match='scalar'):x.verify_delta(base,other,'remove')
    assert base.state['route']['choice']==[9,4]

@pytest.mark.parametrize('span',[(3,3),(-1,2),(4,6)])
def test_bad_ranges_fail_closed(base,span):
    base.spans['target']=span
    with pytest.raises(x.X3Error):x.fork(base,'remove')

def test_overlap_and_unknown_arms_fail(base):
    with pytest.raises(x.X3Error):x.fork(base,'foo')
    base.spans['sham']=(4,5)
    with pytest.raises(x.X3Error):x.fork(base,'remove')

@pytest.mark.parametrize('bad',[{}, {'injection.layer_00.k':np.zeros((1,2,1,4),np.float32)}, {'injection.layer_00.k':np.zeros((1,2,2,4),np.float64)}])
def test_donor_missing_wrong_length_and_wrong_dtype_rejected(base,bad):
    with pytest.raises(x.X3Error,match='donor'):x.fork(base,'swap',bad)

def test_remove_does_not_compact_or_shift_other_rows(base):
    other=x.fork(base,'remove')
    for k,a in base.arrays.items():assert a.shape==other.arrays[k].shape
    assert base.spans==other.spans
    assert all(base.arrays[k].tobytes()==v.tobytes() for k,v in other.arrays.items() if k.startswith('injection'))

def test_kl_independent_probability_oracle_and_direction():
    p=np.array([.8,.1,.1]);q=np.array([.1,.3,.6])
    expected=sum(float(a)*__import__('math').log(float(a/b)) for a,b in zip(p,q))
    d=x.distribution_delta(np.log(p),np.log(q));assert d['kl_nats']==pytest.approx(expected,abs=1e-14)
    assert d['top1_changed'] and d['base_top1']==0 and d['fork_top1']==2
    assert d['kl_nats']!=pytest.approx(x.distribution_delta(np.log(q),np.log(p))['kl_nats'])

def test_kl_extremes_shift_invariance_and_tie():
    d=x.distribution_delta(np.array([1000.,-1000.]),np.array([-1000.,1000.]))
    assert d['kl_nats']==pytest.approx(2000.)
    assert x.distribution_delta(np.array([3.,3.]),np.array([1003.,1003.]))['kl_nats']==0
    assert not x.distribution_delta(np.array([3.,3.]),np.array([1003.,1003.]))['top1_changed']

@pytest.mark.parametrize('bad',[[float('nan'),0.],[float('inf'),0.],[1.],[[1.,2.]]])
def test_invalid_logits_fail(bad):
    with pytest.raises(x.X3Error):x.distribution_delta(bad,bad)

def test_frozen_inputs_token_ids_shas_and_dry_run():
    from tokenizers import Tokenizer
    reg,rows=x.validate_fixtures();tok=Tokenizer.from_file(reg['tokenizer']['path'])
    assert len(rows)==20
    for r in rows:
        assert tok.encode(r['sham_text'],add_special_tokens=False).ids==r['sham_token_ids']
        assert len(r['base_token_ids'])==len(r['swap_token_ids'])
    plan=x.cells(rows);assert len(plan)==4
    assert sum(c['worker_cap_seconds'] for c in plan)<=1800
    assert [i for c in plan for i in c['snapshot_ids']]==[r['id'] for r in rows]
    assert all(c['arms']==['unforked',*x.ARMS] for c in plan)

def test_exact_answer_does_not_accept_substrings_or_praise():
    f={'expected_values':['Quartz-8-Jade'],'decoy_value':'Quartz-9-Jade','intended_group':'correct'}
    assert not x.outcome(' QUARTZ-8-JADE ',f)['exact_error']
    assert x.outcome('It is Quartz-8-Jade.',f)['exact_error']
    assert x.outcome("I don't know",f)['is_refusal']

def result(i,group='correct',mass=.5,effect=2.,error=False):
    d={'id':f'x3_{i:02d}','intended_group':group,'mass':mass,'outcome':{'exact_error':error,'category_realized':True,'truncated':False},
       'forks':{a:{'kl_nats':0.,'raw_logits_byte_equal':True} for a in x.ARMS}}
    d['forks']['remove']['kl_nats']=effect;d['forks']['swap']['kl_nats']=effect
    return d

def test_four_way_errors_empty_cells_and_fixed_thresholds():
    reg,_=x.validate_fixtures();t=reg['scoring']['mass_threshold']
    rows=[result(0,mass=t,effect=1,error=True),result(1,mass=t-1e-9,effect=1+1e-9,error=False)]
    s=x.summarize(rows,reg);table={(c['mass'],c['effect']):c for c in s['table']}
    assert table['high','low']['error_rate']==1
    assert table['low','high']['error_rate']==0
    assert table['high','high']['error_rate'] is None
    assert s['predictions']['P1'] is None and s['accuracy']['all']['lesion']==0

def test_no_vacuous_gpu_pass():
    reg,_=x.validate_fixtures();s=x.summarize([],reg)
    assert s['status'].startswith('RED') and all(p is None for p in s['predictions'].values())
    assert all(c['error_rate'] is None for c in s['table'])

def test_prediction_sets_and_category_guard():
    reg,_=x.validate_fixtures()
    rows=[result(i,group='correct' if i<8 else 'decoy' if i<14 else 'refusal',error=i>=8) for i in range(20)]
    s=x.summarize(rows,reg)
    assert s['predictions']['P1'] and s['predictions']['P2'] and s['predictions']['P3']
    rows[8]['outcome']['category_realized']=False
    s=x.summarize(rows,reg);assert s['predictions']['P2'] is None and s['predictions']['P3'] is None
    with pytest.raises(x.X3Error,match='duplicate'):x.summarize([rows[0],rows[0]],reg)

def test_comparable_controls_kill_without_hiding_measurement():
    reg,_=x.validate_fixtures();rows=[result(i,group='correct' if i<8 else 'decoy' if i<14 else 'refusal') for i in range(20)]
    for r in rows[:3]:r['forks']['sham']['kl_nats']=2
    s=x.summarize(rows,reg);assert s['status']=='RED_KILL' and s['controls_comparable_count']==3
    assert s['P1_joint_count']==17 and s['predictions']['P1'] is False

def test_create_only_receipts(tmp_path):
    p=tmp_path/'receipt.json';x.create(p,{'status':'RED'})
    with pytest.raises(FileExistsError):x.create(p,{'status':'PASS'})
    assert json.loads(p.read_text())['status']=='RED'

def test_resume_is_one_cell_and_failed_start_is_consumed(tmp_path,monkeypatch):
    from scripts import grm_x3_lead as lead
    monkeypatch.setattr(lead,'OUT',tmp_path)
    assert lead.choose_cell('resume')=='cell_00'
    d=tmp_path/'runs'/'abc'/'cell_00';x.create(d/'started.json',{'fingerprint':'abc'})
    with pytest.raises(x.X3Error,match='abandoned'):lead.choose_cell('resume')
    x.create(d/'receipt.json',{'status':'COMPLETE'})
    assert lead.choose_cell('resume')=='cell_01'
    with pytest.raises(x.X3Error,match='already'):lead.choose_cell('run','cell_00')
    with pytest.raises(x.X3Error,match='unknown'):lead.choose_cell('run','cell_20')

def test_leased_entry_requires_inherited_lock(monkeypatch):
    from scripts import grm_x3_lead as lead
    monkeypatch.delenv('GRM_X3_LEASED',raising=False)
    with pytest.raises(x.X3Error,match='shell'):lead.leased('cell_00','unused')

@pytest.mark.parametrize('arm',x.ARMS)
def test_real_det1_capture_and_x3_hydration_cpu_roundtrip(tmp_path,monkeypatch,arm):
    # Real capture + restore code; only device upload is replaced by numpy.
    # Uses house FakeArena, not a reimplementation of the manifest format.
    import importlib.util
    from scripts import grm_det1_3_snapshot as snap
    from scripts import grm_x3_gpu as gpu
    spec=importlib.util.spec_from_file_location('x3_house_test_helpers',x.ROOT/'tests/test_grm_det1_3_snapshot.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    arena=helper._multi_mount_arena('injection')
    prov=helper.provenance('lived','lived-process');prov.update(gpu.process_instance())
    path=snap.capture_arena_snapshot(arena,tmp_path/'captured',label='lived',provenance=prov,
        question='What is the value?',prompt_ids=[7,8,9],admission_plan=helper._declared_synthesis_admission(),
        live_token_ids=[],sink_text='<sink>',sink_token_ids=[1,2],explicit_identity=helper.IDENTITY)
    base=x.load_capture(path,{'sham':(2,3),'target':(3,5)})
    donor={k:np.full_like(v[:,:,3:5,:],7) for k,v in base.arrays.items() if k.startswith('injection.')}
    branch=x.fork(base,arm,donor)
    monkeypatch.setattr(snap,'_fork_upload',lambda a,source_dtype,tensor_factory=None:np.ascontiguousarray(a.copy()))
    receipt=gpu.hydrate(arena,path,base,branch,helper.IDENTITY)
    assert receipt['same_process_index_verified'] and receipt['zero_intervention_no_deltas']
    for li,layer in enumerate(arena.m.layers):
        for j,k in enumerate(('k','v')):
            assert layer.self_attn.inject_kv[j].tobytes()==branch.arrays[f'injection.layer_{li:02d}.{k}'].tobytes()
        assert layer.self_attn._det1_fork_allowed_mask.tobytes()==branch.arrays[f'mask.layer_{li:02d}.allowed'].tobytes()
    assert arena.pos==base.state['arena.pos'] and arena.cur_mounts==base.state['arena.cur_mounts']
    assert arena.m.rope_cos.tobytes()==base.arrays['model.rope_cos'].tobytes()
    assert x.load_capture(path,base.spans).fingerprint()==base.fingerprint()

def test_direct_attempt_pins_rs3_query_origin_and_width(monkeypatch):
    from core.graft_arena import GQAArenaCache
    from scripts.grm_x3_gpu import prepare_attempt
    monkeypatch.setenv('GRM_SEAT_NEAR_LIVE','1')
    arena=SimpleNamespace(m=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(live_shift=None))]),
        n_sink=2,width=8,live_shift=10,grafts=[{'ntok':1},{'ntok':2}])
    arena._rs3_seat_plan=lambda picks:GQAArenaCache._rs3_seat_plan(arena,picks)
    order,seat=prepare_attempt(arena,[1,0])
    assert order==[0,1] and seat['plan_head_last_position']==9
    assert arena.m.layers[0].self_attn.live_shift==10
    assert arena.caches is None and arena.pos==0 and arena.live_segs==[]
    arena.width=2
    with pytest.raises(x.X3Error,match='width'):prepare_attempt(arena,[1,0])
