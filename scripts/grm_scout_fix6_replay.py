"""CPU policy replay and C2 replay-cell registration; never loads GPU payloads.
Prior art: GRM LT1 amendment1 full-manifest repair and C2 restart cells (2026).
Reuse recorded states, eligibility and rankings; new adapter calls core Rule2.
No prior art known to me for this exact receipt composition.
"""
import copy
import json
from pathlib import Path
from core import grm_admission as adm
from core.graft_arena import ArenaCache
from scripts import grm_lt1 as lt, grm_lt1_offline as old
from scripts.grm_lt1_offline_supplement import c2_replay
from scripts.grm_lt1_admission import evaluate
OUT=lt.ROOT/'artifacts/grm_scout_fix6'

def main():
    rows=c2_replay();assert len(rows)==132
    comparisons=[];cells=[];unresolved=[]
    for row in rows:
        source=lt.read(row['source_path']);recorded=source['rows'][row['row_ordinal']]
        manifest=Path(row['state_path']);cp=manifest.parent.parent/'checkpoint.json'
        original_read=old.lt.read;repaired=copy.deepcopy(lt.read(cp));repaired['manifest_projection']=lt.read(manifest)['nodes']
        old.lt.read=lambda p:repaired if Path(p)==cp else original_read(p)
        try:s=old.frozen_c2(recorded,cp)
        finally:old.lt.read=original_read
        baseline=evaluate(s,0)
        before=json.dumps(recorded['route_receipt']['admission']['rank_plan'],separators=(',',':')).encode()
        after=json.dumps(baseline['proposed_plan'],separators=(',',':')).encode()
        assert before==after
        tokens=adm.shaped_identifier_tokens(ArenaCache,s['question'])
        hits=[i for i in s['eligible'] if tokens and adm.is_identifier_binding(candidate_text=s['nodes'][i]['text'] or '',ordered_identifier_tokens=tokens,rare_identifier_tokens=tokens)]
        reference=evaluate(s,2)
        if reference['status']=='UNRESOLVED':
            unresolved.append(dict(execution_id=row['execution_id'],question_id=row['question_id'],reason=reference['reason'],off_plan_bytes_hex=before.hex(),replayed_bytes_hex=after.hex(),checkpoint=str(cp),checkpoint_sha256=lt.sha(cp)))
            continue
        plan,branch,_=adm.margin_first_plan(ranking=s['ranking'],route_margin_1_2=s['margin'],identified_candidates=hits)
        assert plan==evaluate(s,2)['plan']
        changed=plan!=baseline['plan']
        comparison=dict(execution_id=row['execution_id'],question_id=row['question_id'],off_plan=baseline['proposed_plan'],on_plan=plan,off_plan_bytes_hex=before.hex(),replayed_bytes_hex=after.hex(),changed=changed,currently_correct=row['currently_correct'])
        comparisons.append(comparison)
        if changed and row['currently_correct']:
            name=row['execution_id'].replace(':','--');sp=OUT/'c2_states'/f'{name}.json'
            if sp.exists():assert lt.read(sp)==s
            else:lt.create(sp,s)
            c=source['cell']
            cells.append(dict(id='fix6-replay-'+name,side=c['side'],battery=c['battery'],phase='restart',probes=[row['question_id']],depends=[],lease_seconds=285,worker_seconds=280,outer_seconds=590,cooldown_seconds=30,
                source_execution_id=row['execution_id'],source_phase=c['phase'],question_id=row['question_id'],question=s['question'],source_worker=row['source_path'],source_worker_sha256=row['source_sha256'],row_ordinal=row['row_ordinal'],
                checkpoint=str(cp),checkpoint_sha256=lt.sha(cp),state_manifest=str(manifest),state_manifest_sha256=lt.sha(manifest),checkpoint_files=lt.read(cp)['files'],
                policy_state=str(sp.relative_to(lt.ROOT)),policy_state_sha256=lt.sha(sp),off_plan=baseline['proposed_plan'],on_plan=plan,admission_rule='margin_first'))
    assert len(cells)==31
    assert len(unresolved)==20 and len(comparisons)==112
    lt.create(OUT/'c2_unresolved.json',unresolved)
    lt.create(OUT/'c2_comparisons.json',comparisons)
    lt.create(OUT/'c2_replay_cells.json',dict(schema='grm.fix6.c2.replay.v1',status='REGISTERED_NOT_RUN',admission_rule='margin_first',cells=cells,
        state_contract='C2 restart worker validates checkpoint, copies its repository, loads the recorded context and calls score_probe. Lead imports these cell fields into grm-c2 registration and resolves each explicit checkpoint path; stock worker otherwise derives checkpoint root from OUT. Pin GRM_ADMISSION_RULE after environment(flags), which removes ambient GRM variables. Do not run this list as a fresh battery.',
        gpu_authorization='Not run by this seat. Lead must register C2 GPU budget and code bindings after FIX6 integration.',
        prior_art=__doc__))
    lt.create(OUT/'c2_receipt.json',dict(status='PASS',evidence_class='CPU real core policy replay on full recorded node states and GPU ranking/margin; no numerical router or reader rerun',off_byte_identical=len(rows),on_evaluable=len(comparisons),on_unresolved=len(unresolved),on_changed=sum(x['changed'] for x in comparisons),currently_correct_changed=len(cells),unique_question_ids=len({x['question_id'] for x in cells}),no_gpu=True))
    (OUT/'C2_QUESTION_IDS.md').write_text('# C2 registered Rule2 regression set\n\n31 execution cells; repeated question ids across arm/phase retained. Plan changes are risk, not measured answer regressions.\n\n| Execution | Question id | OFF plan | ON plan |\n|---|---|---|---|\n'+''.join(f"| {c['source_execution_id']} | {c['question_id']} | {c['off_plan']} | {c['on_plan']} |\n" for c in cells))
    print(json.dumps(lt.read(OUT/'c2_receipt.json')))
if __name__=='__main__':main()
