"""CPU author baseline; no claim of blind verification.
Prior art: house X3/DET1 synthetic capture oracle (2026); inherited receipt layout.
New tests enforce amendment 2's registered controls, pins, quotas and sham floor.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys
import pytest
import numpy as np
from scripts import grm_x3_r2_diagnostic as x
from scripts.grm_x3_r2_scorer import value_span, outcome, normalize, DASH_POINTS


@pytest.mark.campaign_receipt(registration='artifacts/grm_x3/r2/registration.json + r1_evidence_before.json (5161 pinned files, incl. gitignored runs/*/donor_payload.npz)')
def test_registered_scorer_controls():
    reg,_=x.validate_fixtures()
    controls=json.loads((x.ROOT/reg['scorer']['controls']['path']).read_text())['cases']
    assert len(controls)==372
    assert {'changed_digit','omitted_token','swapped_relation','negation'} <= {c['kind'] for c in controls}
    for c in controls:
        assert value_span(c['answer'],c['value'],refusal=c.get('refusal',False)) is c['accept'], c


def test_all_registered_dashes():
    for point in DASH_POINTS:
        assert value_span('Answer: copper'+chr(point)+'731 before indigo-851.', 'copper-731 before indigo-851'), hex(point)
    assert normalize('ＣＯＰＰＥＲ－７３１\tbefore\nindigo-851')=='copper-731 before indigo-851'


@pytest.mark.campaign_receipt(registration='artifacts/grm_x3/r2/registration.json + r1_evidence_before.json (5161 pinned files, incl. gitignored runs/*/donor_payload.npz)')
@pytest.mark.parametrize('corruption',['forged','stale','manifest'])
def test_r2_registration_rejects_forged_and_stale(tmp_path,corruption):
    reg,_=x.validate_fixtures()
    if corruption=='stale':reg=json.loads((x.ROOT/'artifacts/grm_x3/registration.json').read_text())
    elif corruption=='forged':reg['sham_calibration']['threshold_nats']=100
    x.create(tmp_path/'registration.json',reg)
    manifest=json.loads((x.OUT/'fixtures/manifest.json').read_text())
    if corruption=='manifest':manifest['counts']['correct']=20
    x.create(tmp_path/'fixtures/manifest.json',manifest)
    with pytest.raises(x.X3Error,match='forged/stale'):x.validate_fixtures(tmp_path)


@pytest.mark.campaign_receipt(registration='artifacts/grm_x3/r2/registration.json + r1_evidence_before.json (5161 pinned files, incl. gitignored runs/*/donor_payload.npz)')
def test_r1_evidence_byte_unchanged():
    pinned=json.loads((x.OUT/'r1_evidence_before.json').read_text())
    assert len(pinned['files'])==5161
    for record in pinned['files']:
        assert x.sha(x.ROOT/record['path'])==record['sha256'], record['path']


@pytest.mark.campaign_receipt(registration='artifacts/grm_x3/r2/registration.json + r1_evidence_before.json (5161 pinned files, incl. gitignored runs/*/donor_payload.npz)')
def test_r2_disjoint_recipes_and_dry_run():
    reg,fixtures=x.validate_fixtures()
    old=[json.loads(p.read_text()) for p in (x.ROOT/'artifacts/grm_x3/fixtures').glob('x3_*.json')]
    assert len({f['entity'] for f in fixtures})==20
    oldtext='\n'.join(json.dumps(f) for f in old).casefold()
    for f in fixtures:
        assert f['entity'].casefold() not in oldtext
        assert f['question'] not in {o['question'] for o in old}
        for v in [*f['expected_values'],f['decoy_value']]:assert v.casefold() not in oldtext
    result=subprocess.run(['bash','scripts/grm_x3_r2_lead_gpu.sh','--dry-run'],cwd=x.ROOT,capture_output=True,text=True,check=True)
    cells=json.loads(result.stdout)
    assert len(cells)==4 and sum(len(c['snapshot_ids']) for c in cells)==20
    assert all(c['worker_cap_seconds']==285 for c in cells)
    assert [i for c in cells for i in c['snapshot_ids']]==[f['id'] for f in fixtures]


def make_rows():
    reg,fixtures=x.validate_fixtures();rows=[]
    for i,f in enumerate(fixtures):
        group=f['intended_group'];value=f['expected_values'][0] if group=='correct' else f['decoy_value'] if group=='decoy' else 'unknown'
        rows.append({'id':f['id'],'intended_group':group,'mass':.5,
            'outcome':outcome('Answer: **'+value.replace('-','\u2011')+'**.',f),
            'forks':{a:{'kl_nats':2. if a in ('remove','swap') else .078 if a=='sham' else 0.,
                         'raw_logits_byte_equal':a in ('same_payload','zero'),'top1_changed':False} for a in x.ARMS}})
    return reg,rows


@pytest.mark.campaign_receipt(registration='artifacts/grm_x3/r2/registration.json + r1_evidence_before.json (5161 pinned files, incl. gitignored runs/*/donor_payload.npz)')
def test_r2_summary_guards_and_floor():
    reg,rows=make_rows();s=x.summarize(rows,reg)
    assert s['predictions']["P1'"] is True and s['realized_strata']
    assert s['frozen_exact_equality_strata']==dict(correct=0,decoy=0,refusal=0)
    assert s['predictions']["P2'"] is True and s['predictions']["P3'"] is True
    for r in rows[:2]:r['forks']['sham']['kl_nats']=.10
    assert x.summarize(rows,reg)['predictions']["P1'"] is True
    rows[2]['forks']['sham']['kl_nats']=.10
    s=x.summarize(rows,reg)
    assert s['predictions']["P1'"] is False and s['status']=='RED_P1'
    rows[2]['forks']['sham']['kl_nats']=.078
    rows[3]['forks']['same_payload']['raw_logits_byte_equal']=False
    assert x.summarize(rows,reg)['predictions']["P1'"] is False
    for group in ('correct','decoy','refusal'):
        reg,rows=make_rows();members=[r for r in rows if r['intended_group']==group]
        for r in members[:2]:r['outcome']['category_realized']=False
        assert x.summarize(rows,reg)['realized_strata']
        members[2]['outcome']['category_realized']=False
        s=x.summarize(rows,reg)
        assert not s['realized_strata'] and s['predictions']["P2'"] is None
        assert s['predictions']["Q3'"] is None
    s=x.summarize([],reg)
    assert s['status']=='RED_STRATA_OR_INCOMPLETE' and all(v is None for v in s['predictions'].values())


@pytest.mark.campaign_receipt(registration='artifacts/grm_x3/r2/registration.json + r1_evidence_before.json (5161 pinned files, incl. gitignored runs/*/donor_payload.npz)')
def test_r2_thresholds_and_unchanged_kill():
    reg,rows=make_rows()
    for r in rows[:3]:r['forks']['sham']['kl_nats']=2.
    assert x.summarize(rows,reg)['status']=='RED_KILL'
    reg,rows=make_rows()
    for r in rows[:3]:r['forks']['remove']['kl_nats']=1.
    assert x.summarize(rows,reg)['predictions']["P2'"] is False
    reg,rows=make_rows()
    for r in rows:r['mass']=.1
    s=x.summarize(rows,reg)
    assert s['status']=='RED_KILL' and s['predictions']["Q3'"] is True


@pytest.mark.campaign_receipt(registration='artifacts/grm_x3/r2/registration.json + r1_evidence_before.json (5161 pinned files, incl. gitignored runs/*/donor_payload.npz)')
def test_r2_scorer_competing_answers_and_frozen_exact():
    _,fixtures=x.validate_fixtures();f=fixtures[0]
    a=outcome(f['expected_values'][0],f)
    assert not a['exact_error'] and a['category_realized']
    b=outcome('**'+f['expected_values'][0]+'**',f)
    assert b['exact_error'] and b['category_realized'] and not b['exact_equality_realization']
    both=outcome(f['expected_values'][0]+' or '+f['decoy_value'],f)
    assert not both['category_realized']

@pytest.fixture
def synthetic_run(tmp_path,monkeypatch):
    """Real four-cell receipt layout, real DET1 blobs, CPU logits; no GPU mocks.
    Prior art: house DET1 FakeArena/capture oracle (2026), reused unchanged.
    Own fixture exercises discovery through validation to registered reporting.
    """
    import importlib.util
    from scripts import grm_x3_r2_lead as lead, grm_det1_3_snapshot as snap
    reg,fixtures=x.validate_fixtures()
    spec=importlib.util.spec_from_file_location('x3_summary_helpers',x.ROOT/'tests/test_grm_det1_3_snapshot.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    out=tmp_path/'artifacts/grm_x3'
    x.create(out/'registration.json',reg)
    x.create(out/'fixtures/manifest.json',{'synthetic':True})
    x.create(out/'implementation_manifest.json',{'synthetic':True})
    for name in ('grm_x3_r2_lead.py','grm_x3_r2_diagnostic.py'):
        p=tmp_path/'scripts'/name;p.parent.mkdir(exist_ok=True);p.write_bytes((x.ROOT/'scripts'/name).read_bytes())
    rt={'registration_sha256':x.sha(out/'registration.json'),'fixtures_sha256':x.sha(out/'fixtures/manifest.json'),
        'implementation_sha256':x.sha(out/'implementation_manifest.json'),'model_id':reg['model_id'],'frame':reg['frame']}
    rt['fingerprint']=x.digest(rt);run=out/'runs'/rt['fingerprint']
    def record(path):return {'path':str(path.relative_to(tmp_path)),'sha256':x.sha(path)}
    for ci in range(4):
        d=run/f'cell_{ci:02d}';rows=[]
        for fixture in fixtures[ci*5:(ci+1)*5]:
            i=int(fixture['id'].split('_')[2]);rd=d/fixture['id']
            path=snap.capture_arena_snapshot(helper._multi_mount_arena('injection'),rd/'snapshot',label='lived',
                provenance=helper.provenance('lived','lived-process'),question='What is the value?',prompt_ids=[7,8,9],
                admission_plan=helper._declared_synthesis_admission(),live_token_ids=[],sink_text='<sink>',
                sink_token_ids=[1,2],explicit_identity=helper.IDENTITY)
            spans={'sham':(2,3),'target':(3,5)};base=x.load_capture(path,spans)
            payload={k:np.full_like(v[:,:,3:5,:],7) for k,v in base.arrays.items() if k.startswith('injection.')}
            np.savez(rd/'donor_payload.npz',**payload)
            logits=np.array([3.,0.]);np.save(rd/'unforked.npy',logits)
            group=fixture['intended_group']
            answer=fixture['expected_values'][0] if group=='correct' else fixture['decoy_value'] if group=='decoy' else 'unknown'
            r={'id':fixture['id'],'intended_group':group,'mass':.5 if i%4>=2 else .1,
                'outcome':x.outcome(answer,fixture),'snapshot':record(path),'spans':spans,
                'donor_payload':record(rd/'donor_payload.npz'),'unforked_logits':record(rd/'unforked.npy'),'forks':{}}
            for arm in x.ARMS:
                other=logits[::-1].copy() if arm in ('remove','swap') and i%2 else logits.copy()
                np.save(rd/f'{arm}.npy',other)
                r['forks'][arm]={**x.distribution_delta(logits,other),'logits':record(rd/f'{arm}.npy'),
                    'delta_pin':x.verify_delta(base,x.fork(base,arm,payload),arm,payload)}
            x.create(rd/'result.json',r);rows.append(r)
        x.create(d/'started.json',{'cell':d.name,'fingerprint':rt['fingerprint'],'runtime':rt})
        x.create(d/'worker_result.json',{'cell':d.name,'fingerprint':rt['fingerprint'],'results':rows})
        x.create(d/'receipt.json',{'cell':d.name,'fingerprint':rt['fingerprint'],'status':'COMPLETE',
            'worker_result_sha256':x.sha(d/'worker_result.json')})
    monkeypatch.setattr(lead,'ROOT',tmp_path);monkeypatch.setattr(lead,'OUT',out)
    monkeypatch.setattr(lead,'validate_fixtures',lambda:(reg,fixtures))
    # validate_worker imports this validator locally too.
    monkeypatch.setattr(x,'validate_fixtures',lambda:(reg,fixtures))
    return lead,run


@pytest.mark.campaign_receipt(registration='artifacts/grm_x3/r2/registration.json + r1_evidence_before.json (5161 pinned files, incl. gitignored runs/*/donor_payload.npz)')
def test_r2_summary_existing_run_layout(synthetic_run,monkeypatch):
    lead,run=synthetic_run
    monkeypatch.setattr(lead,'runtime',lambda:pytest.fail('live runtime queried'))
    s=lead.summary(run.name)
    assert s['observed_snapshots']==20 and not s['receipt_errors']
    assert s['realized_strata'] and s['frozen_exact_equality_strata']==dict(correct=8,decoy=6,refusal=6)
    assert len(s['source_receipts'])==32
    assert all(c['n']==5 for c in s['table'])
    assert s['predictions']["P1'"] is True
    assert s['status']=='RED_KILL'


def test_r2_cli_refuses_stale_fingerprint_before_gpu(monkeypatch):
    from scripts import grm_x3_r2_lead as lead
    monkeypatch.setattr(lead,'runtime',lambda:{'fingerprint':'registered'})
    monkeypatch.setattr(lead,'preflight',lambda:pytest.fail('GPU preflight reached'))
    for args in (['run','cell_00','stale'],['preflight','stale'],['run','cell_00']):
        monkeypatch.setattr(sys,'argv',['runner',*args])
        with pytest.raises(x.X3Error,match='fingerprint'):lead.main()
