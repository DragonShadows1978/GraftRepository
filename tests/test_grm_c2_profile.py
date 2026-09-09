"""Prior art: SCOUT-FIX-1 CPU manifest/receipt fakes and WC1 paired gates
(project contributors, 2026). Reuse persistence and exact named contracts;
new cases challenge C2 opt-in, missing evidence and process boundaries.
No prior art known to me beyond those local systems for this adapter's tests.
"""
import copy
import inspect
import json
import os
import subprocess
import sys
from types import SimpleNamespace
import pytest
from scripts import grm_c2_profile as p
from scripts import grm_c2_cells as c


def test_profile_off_unless_explicitly_selected():
    flags={'arena_width':256,'capture_pin':'off','seat_near_live':False,'unrelated':{'a':[1]}}
    before=copy.deepcopy(flags); env=dict(os.environ)
    result=p.select_profile(flags)
    assert result==before and result is not flags
    assert os.environ==env and flags==before


def test_profile_loading_sets_exact_registered_flags_and_nothing_else():
    r=p.read(p.REGISTRY); flags=copy.deepcopy(r['default_flags'])
    flags['unrelated']={'sentinel':[3,4]}; before=copy.deepcopy(flags); env=dict(os.environ)
    result=p.select_profile(flags,p.PROFILE_ID)
    assert r['flag_overrides']=={'arena_width':96,'capture_pin':'live','seat_near_live':True,'rt1_rule':True}
    assert result=={**before,**r['flag_overrides']}
    assert flags==before and os.environ==env
    result['unrelated']['sentinel'].append(5)
    assert flags==before
    with pytest.raises(ValueError):p.select_profile(flags,'typo')


def test_registry_matches_shipped_defaults_and_exact_frame():
    from core.graft_arena import ArenaCache
    from core.grm_frame import env_capture_pin,env_seat_near_live,ephemeral_frame_enabled
    from core.grm_admission import rt1_enabled
    r=p.read(p.REGISTRY)
    assert inspect.signature(ArenaCache.__init__).parameters['arena_width'].default==r['default_flags']['arena_width']==256
    assert env_capture_pin({})=='off' and not env_seat_near_live({})
    assert ephemeral_frame_enabled(environ={})
    assert r['frame']['n_sink']==len(r['frame']['sink_token_ids'])==19
    assert r['geometry']['profile']['live_shift']==115
    assert r['model']['revision']=='6cee5e81ee83917806bbde320786a8fb61efebee'


def test_cell_plan_complete_dependencies_and_nonfit_honesty():
    r=p.read(p.REGISTRATION);cells=r['cells'];seen=set()
    assert len(cells)==52 and r['estimated_seconds']==3140>r['budget_seconds']==2880
    for cell in cells:
        assert set(cell['depends'])<=seen
        assert cell['estimate_seconds']<=285
        seen.add(cell['id'])
    for side in ('profile','defaults'):
        for battery,total in [('sup',9),('census',10),('longhistory',14)]:
            restart=[i for x in cells if x['side']==side and x['battery']==battery and x['phase']=='restart' for i in x['probes']]
            assert len(restart)==len(set(restart))==total
    with pytest.raises(ValueError,match='NON_FIT_BUDGET'):c.run_leased(cells[0])


def test_worker_cannot_run_without_leased_parent(monkeypatch):
    monkeypatch.delenv('GRM_C2_LEASE_PARENT',raising=False)
    with pytest.raises(RuntimeError,match='foreground leased parent'):c.worker({})


def test_capture_explicit_unpinned_inherited_and_wrong_geometry():
    assert not p.capture_grade([{}],'profile',96,19)['valid']
    assert p.capture_grade([{}],'defaults',256,19)['rows'][0]['grade']=='INHERITED_UNATTESTED'
    off={'capture':{'capture_pin':'off','capture_shift_observed':None}}
    assert p.capture_grade([off],'defaults',256,19)['valid']
    assert not p.capture_grade([off],'profile',96,19)['valid']
    live={'capture':{'capture_pin':'live','capture_shift':115,'capture_shift_observed':115,'live_shift':115,'arena_width':96,'n_sink':19}}
    assert p.capture_grade([live],'profile',96,19)['valid']
    live['capture']['capture_shift_observed']=0
    assert not p.capture_grade([live],'profile',96,19)['valid']
    assert not p.capture_grade([],'profile',96,19)['valid']


def test_token_seats_are_sum_not_graft_count():
    assert p.token_seats([{'ntok':23},{'ntok':72}],[1,0])==95
    assert p.token_seats([{'ntok':23},{}],[0,1]) is None
    assert p.token_seats([],[])==0


def synthetic_rows():
    r=p.read(p.REGISTRY);rows=[]
    for battery,ids in r['probe_ids'].items():
        controls=r['original_correct_controls'][battery]
        for side in ('profile','defaults'):
            target=r['prediction'][side][battery]
            # Synthetic fixtures do not pretend to be measured historical misses.
            winners=(controls+[i for i in ids if i not in controls])[:target]
            for phase in ('fresh','restart'):
                for i in ids:
                    rows.append(dict(battery=battery,side=side,phase=phase,probe_id=i,
                      correct=i in winners,new_process=True,metadata_retained=True,
                      capture_valid=True,token_seats=75,rt1={k:[] for k in p.RT1_FIELDS}))
    return rows


def test_missing_measurement_is_not_a_pass():
    assert p.compare([])['adopt'] is False
    assert p.compare([])['fix_validation'] is False


@pytest.mark.parametrize('defect',['missing','duplicate','metadata','process','token_seats','rt1','capture','score'])
def test_gate_rejects_incomplete_or_changed_evidence(defect):
    rows=synthetic_rows()
    baseline=p.compare(rows)
    assert baseline['adopt'] and baseline['fix_validation']
    row=next(r for r in rows if r['side']=='profile' and r['phase']=='restart')
    if defect=='missing':rows.remove(row)
    elif defect=='duplicate':rows.append(copy.deepcopy(row))
    elif defect=='metadata':row['metadata_retained']=False
    elif defect=='process':row['new_process']=False
    elif defect=='token_seats':row['token_seats']=None
    elif defect=='rt1':row['rt1']={}
    elif defect=='capture':row['capture_valid']=False
    else:row['correct']=not row['correct']
    assert not p.compare(rows)['adopt']


def test_manifest_checkpoint_new_process_no_recapture(tmp_path):
    from tests.test_grm_scout_fix1_manifest import repository
    repo=repository(tmp_path/'original');i=repo.add_document('C2 persisted payload 1234')
    repo.arena.grafts[i].update(capture_pin='live',capture_shift=115,
        capture_shift_observed=115,n_sink=19,arena_width=96,live_shift=115)
    cp=c.snapshot(repo,tmp_path/'checkpoint',{'probe_id':'cpu'},'parent')
    c.validate_checkpoint(tmp_path/'checkpoint')
    command="""
import json,sys
from tests.test_grm_scout_fix1_manifest import repository
from scripts.grm_c2_cells import manifest_projection
r=repository(sys.argv[1])
def forbidden(*a,**k):raise AssertionError('reharvest')
r.arena.deposit=forbidden
r.arena._ensure_h([0])
print(json.dumps(manifest_projection([r._node_manifest(g) for g in r.arena.grafts])))
"""
    result=subprocess.run([sys.executable,'-c',command,str(tmp_path/'checkpoint/repository')],cwd=p.ROOT,
                          env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,check=True,timeout=30)
    # Runtime diagnostic output may precede the JSON result.
    assert json.loads(result.stdout.splitlines()[-1])==cp['manifest_projection']
    path=tmp_path/'checkpoint/repository/manifest.json';path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='changed'):c.validate_checkpoint(tmp_path/'checkpoint')
