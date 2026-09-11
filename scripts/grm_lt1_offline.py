#!/usr/bin/env python3
"""Exhaustive receipt census and offline admission evaluation; CPU only.
Prior art: GRM C2 pre-probe sidecars, EB1/WC1 receipts (2026). Borrow stored
states and rankings; never fabricate embeddings or roll final metadata back.
New: LT1 counterfactual table adapter; no prior art known to me for exact mix.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from scripts import grm_lt1 as lt
from scripts.grm_lt1_admission import snapshot,evaluate
from scripts.grm_c7_run import emit
from core.graft_arena import ArenaCache
from scripts import grm_repo_paths as repo_paths
OUT=lt.OUT/'amendment1/offline'
# GRM-F6: was Path('/mnt/ForgeRealm/wt/grm-c2/artifacts/grm_c2/epochs/
# scout-fix-2') -- a pruned seat worktree, so `C2.glob(...)` returned ZERO
# rows and the census passed vacuously (F5's finding). The same epoch data
# is canonical in-repo. Prior art: repo-root-from-__file__ (setuptools /
# pytest rootdir, 2009-); vacuous-zero-is-an-error (dbt/Great Expectations
# row-count assertions, 2018). See scripts/grm_repo_paths.py.
C2=repo_paths.receipt_root('artifacts/grm_c2/epochs/scout-fix-2','GRM_C2_EPOCH_ROOT',
    'artifacts/grm_scout_fix8/green.json (c2[].source pins the absolute path)',
    '/mnt/ForgeRealm/wt/grm-c2/artifacts/grm_c2/epochs/scout-fix-2')
EB1=Path('/mnt/ForgeRealm/GraftRepository/artifacts/grm_eb1')
WC1=Path('/mnt/ForgeRealm/GraftRepository/artifacts/grm_wc1_opus')


def result(s):
    return {str(rule):evaluate(s,rule) for rule in (0,1,2)}


def lt1():
    import pytest
    from scripts.grm_c7_diagnose import repository
    from scripts.grm_e2e_session import harmony_turn
    fixture=lt.read(lt.FIX);probes={p['turn']:p for p in fixture['probes']};rows=[]
    for arm in ('A','B'):
        with tempfile.TemporaryDirectory(prefix='lt1-offline-') as temp, pytest.MonkeyPatch.context() as patch:
            repo=repository(Path(temp)/'repository',patch);a=repo.arena;a.width=96 if arm=='A' else 256;a.recency_mounts=2
            turn_nodes={}
            try:
                for e in fixture['turns']:
                    if e['kind']=='probe':
                        p=probes[e['turn']];rec=a.eb1_begin_turn()
                        s=snapshot(a,p['question'],rec);s['source_ids']=[turn_nodes[t] for t in p['source_turns']]
                        path=OUT/'lt1_states'/f"{arm}-{p['id']}.json";lt.create(path,s)
                        decisions=result(s);base=s['baseline_profile']
                        from core.grm_admission import identifier_unbound_abstention
                        expected=[] if identifier_unbound_abstention(base) else base['rank_plan']
                        if decisions['0']['plan']!=expected:raise ValueError('REAL_CORE_RULE0_PARITY')
                        rows.append(dict(battery='LT1',execution_id=arm+':'+p['id'],question_id=p['id'],arm=arm,
                            question=p['question'],currently_correct=False,currently_refused=True,
                            state_path=str(path),state_sha256=lt.sha(path),source_ids=s['source_ids'],rules=decisions))
                    elif e['kind']!='recap':
                        i=a.feed(harmony_turn(e['user'],e['assistant']));a.grafts[i]['kind']='turn';turn_nodes[e['turn']]=i
            finally:repo.close()
    return rows


def unresolved(reason):
    return dict(status='UNRESOLVED',reason=reason)


def frozen_c2(row,checkpoint):
    cp=lt.read(checkpoint);nodes=cp['manifest_projection'];context=cp['context']
    q=context['probe']['question'] if 'probe' in context else context['event']['user']
    rr=row['route_receipt'];ad=rr['admission'];info=rr['fit']['info_pass_through']
    # Prior art: EB1 begin-turn nominations, real source method (GRM, 2026).
    # Sup banks have recency disabled. Other pre-probe sidecars are ephemeral
    # with recency=2; live segments are cleared by EB1 at probe entry.
    if 'probe' in context:excluded=[]
    else:
        a=SimpleNamespace(grafts=nodes,ephemeral=True,recency_mounts=2,live_segs=[],cur_mounts=[],cur_mount_n=0)
        excluded=ArenaCache.eb1_begin_turn(a)
    eligible=[i for i,n in enumerate(nodes) if not n.get('retired') and n.get('kind','turn')!='recall' and i not in excluded]
    ranking=ad.get('ranking_before_demotion')
    if ranking is None:ranking=info.get('admission_ranking_before_demotion',rr['route']['ranking_ids'])
    if not set(ranking)<=set(eligible):raise ValueError('RECORDED_RANKING_OUTSIDE_FROZEN_ELIGIBILITY')
    # A skipped margin is unknown, not a measured tie at zero.
    margin=ad['route_margin_1_2'] if ad.get('route_margin_evaluated') else None
    return dict(question=q,nodes=nodes,eligible=eligible,ranking=ranking,
        split_members=ad.get('split_family_ids',info.get('admission_split_family_ids',[])),margin=margin,
        exclude=excluded,evidence_class='recorded GPU ranking and CPU real admission over pre-probe checkpoint')


def c2():
    rows=[]
    for worker in repo_paths.require_rows(sorted(C2.glob('cells/*/worker.json')),C2,'cells/*/worker.json','C2 worker census'):
        w=lt.read(worker)
        for ordinal,row in enumerate(w['rows']):
            cell=w['cell'];pid=row['probe_id'];cp=C2/'checkpoints'/cell['side']/cell['battery']/pid/'checkpoint.json'
            record=dict(battery='C2',execution_id=cell['id']+':'+pid,question_id=pid,
                        currently_correct=row['correct'],currently_refused=refused(row['served_answer']),
                        source_path=str(worker),source_sha256=lt.sha(worker),row_ordinal=ordinal)
            try:
                if not cp.exists():raise ValueError('MISSING_PREPROBE_CHECKPOINT')
                s=frozen_c2(row,cp);record.update(question=s['question'],state_path=str(cp),state_sha256=lt.sha(cp))
                # Check immutable state metadata against recorded sidecar file
                # hashes, without opening numerical payloads or loading CUDA.
                manifest=cp.parent/'repository/manifest.json'
                if lt.sha(manifest)!=lt.read(cp)['files']['manifest.json']:raise ValueError('PREPROBE_MANIFEST_SHA_MISMATCH')
                decisions=result(s)
                recorded=row['route_receipt']['admission']['rank_plan']
                baseline=decisions['0']
                parity=baseline['status']=='EVALUATED' and baseline['proposed_plan']==recorded
                record['baseline_plan_parity']=parity
                if not parity:decisions={str(i):unresolved('RULE0_REPLAY_PARITY_NOT_ESTABLISHED') for i in (0,1,2)}
                record['rules']=decisions
            except (ValueError,KeyError) as exc:
                record['rules']={str(i):unresolved(str(exc)) for i in (0,1,2)}
            rows.append(record)
    return rows


def refused(answer):
    low=answer.casefold()
    return 'not in memory' in low or lt.score(answer,'__not_a_real_value__')['category'] in ('abstention','refusal')


def session_rows(base,battery):
    rows=[]
    for transcript in sorted(base.rglob('transcript.jsonl')):
        # C2 archives repeat history in each cell/session; only new turns with
        # corresponding local instrumentation were actually served in that cell.
        instruments=transcript.with_name('instrumentation.jsonl')
        ins={r['turn']:r for r in map(json.loads,instruments.read_text().splitlines())} if instruments.exists() else {}
        scorepath=transcript.with_name('probe_scorecard.json')
        scores={r['turn']:r for r in lt.read(scorepath).get('probes',[])} if scorepath.exists() else {}
        for t in map(json.loads,transcript.read_text().splitlines()):
            turn=t['turn'];instrument=ins.get(turn)
            if not instrument:continue
            if battery=='C2' and t['kind']=='probe':continue # worker rows are authoritative scored execution census
            # Feed-complete deposits did not serve a question through admission.
            if instrument.get('plant_mode')=='feed_complete' or t.get('plant_mode')=='feed_complete':continue
            if t['kind'] in ('fact','supersede') and not instrument.get('infer_calls'):continue
            score=scores.get(turn);correct=score.get('contains_expected') if score else None
            pid=f"{t['kind']}_t{turn:03d}_"+str(t.get('fact_id') or 'unscored').replace(' ','_')
            rr=instrument.get('route_receipt',{});ad=rr.get('admission',{})
            recorded=dict(status='RECORDED_ONLY',plan=ad.get('rank_plan'),mounts=bool(rr.get('fit',{}).get('final_mounts')),
                          reason='NO_EXACT_PREPROBE_STATE; not real-core reevaluation')
            rows.append(dict(battery=battery,execution_id=str(transcript.relative_to(base))+':'+str(turn),question_id=pid,
                question=t['user'],currently_correct=correct,currently_refused=refused(t['assistant']),
                source_path=str(transcript),source_sha256=lt.sha(transcript),instrumentation_path=str(instruments),
                instrumentation_sha256=lt.sha(instruments),turn=turn,
                rules={'0':recorded,'1':unresolved('NO_EXACT_PREPROBE_STATE'),'2':unresolved('NO_EXACT_PREPROBE_STATE_OR_EXACT_MARGIN')}))
    return rows


def wc1_sup():
    rows=[]
    for p in sorted(WC1.glob('runs/w*/sup_*.json')):
        d=lt.read(p)
        for i,r in enumerate(d['rows']):
            rows.append(dict(battery='WC1',execution_id=str(p.relative_to(WC1))+':'+r['probe_id'],question_id=r['probe_id'],
                currently_correct=r['correct'],currently_refused=refused(r['served_answer']),
                source_path=str(p),source_sha256=lt.sha(p),row_ordinal=i,
                rules={'0':dict(status='RECORDED_ONLY',plan=r.get('fit',{}).get('admission_rank_plan'),
                               reason='SUP_RECEIPT_LACKS_PREPROBE_STATE_AND_MARGIN'),
                       '1':unresolved('SUP_RECEIPT_LACKS_PREPROBE_STATE'),
                       '2':unresolved('SUP_RECEIPT_LACKS_PREPROBE_STATE_AND_MARGIN')}))
    return rows


def table(rows):
    table=[]
    for battery in ('LT1','EB1','C2','WC1'):
        selected=[r for r in rows if r['battery']==battery]
        for rule in ('0','1','2'):
            valid=[r for r in selected if r['rules'][rule]['status']=='EVALUATED']
            changed=[];gains=[]
            for r in valid:
                base=r['rules']['0'];alt=r['rules'][rule]
                if base['status']!='EVALUATED':continue
                if r['currently_correct'] and alt['plan']!=base['plan']:changed.append(r['execution_id'])
                if r['currently_refused'] and alt['mounts']:gains.append(r['execution_id'])
            table.append(dict(battery=battery,rule=int(rule),served_n=len(selected),evaluated_n=len(valid),
                mounts=sum(r['rules'][rule]['mounts'] for r in valid),
                unresolved_ids=[r['execution_id'] for r in selected if r not in valid],
                correct_mount_plan_change_ids=changed,refused_to_mount_ids=gains,
                status='COMPLETE' if len(valid)==len(selected) and selected else 'RED_INCOMPLETE',
                note='admission plan changes are regression RISK, not measured reader regressions; gains are mount permission only'))
    return table


def main():
    lt.verify();OUT.mkdir(parents=True,exist_ok=False)
    rows=lt1()+c2()+session_rows(C2,'C2')+session_rows(EB1,'EB1')+session_rows(WC1,'WC1')+wc1_sup()
    for row in rows:emit(OUT/'questions.jsonl',row)
    lt.create(OUT/'table.json',table(rows))
    lt.create(OUT/'receipt.json',dict(status='RED_INCOMPLETE_HISTORICAL_REPLAY',admission_rule=0,
        binding=lt.binding('A'),question_rows_sha256=lt.sha(OUT/'questions.jsonl'),table_sha256=lt.sha(OUT/'table.json'),
        counts=dict(Counter(r['battery'] for r in rows)),evidence_class='CPU offline admission; zero reader calls; historical missing-state rows unresolved'))
    print(json.dumps([{k:v for k,v in t.items() if not k.endswith('_ids')} | {
        'regression_risk_n':len(t['correct_mount_plan_change_ids']),'gain_n':len(t['refused_to_mount_ids']),
        'unresolved_n':len(t['unresolved_ids'])} for t in table(rows)],indent=2))

if __name__=='__main__':main()
