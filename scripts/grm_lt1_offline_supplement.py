"""Read-only census correction and sufficient-statistic replay supplement.
Prior art: GRM C2 full manifest (2026); canonical JSON/hash deduplication,
Python hashlib/JSON (Python contributors). New: eliminate copied transcript
prefixes using entire timing-bearing instrumentation records, never by question
text. No prior art known to me for this exact audit adapter.
"""
import copy
import hashlib
import json
import re
from pathlib import Path
from scripts import grm_lt1_offline as old
from scripts import grm_lt1 as lt
from scripts.grm_lt1_admission import evaluate,shaped_tokens
from core import grm_admission as adm
from core.graft_arena import ArenaCache
OUT=lt.OUT/'amendment1/offline_supplement'
REG=lt.OUT/'amendment1/offline_supplement_registration.json'


def is_refused(answer):
    # Prior art: LT1/C7 disjoint answer labels (GRM, 2026). Include explicit
    # reader lack-of-information statements, distinct from admission refusal.
    text=answer.casefold().replace('’', "'")
    return old.refused(answer) or bool(re.search(r"(?:don't|do not) have (?:that|the|this) information|(?:can't|cannot) help with that",text))


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def partial(question,rr):
    """Exact policy decisions where absent node/score fields cannot affect them.

    Prior art: A-DEC branch short circuit (GRM, 2026). No imputed scores:
    zero identifier-shaped tokens implies zero Rule1 binders for ANY state;
    positive margin implies singleton rank-1 tie group for Rule2.
    """
    ad=rr.get('admission',{});route=rr.get('route',{});info=rr.get('fit',{}).get('info_pass_through',{})
    raw=ad.get('ranking_before_demotion',info.get('admission_ranking_before_demotion'))
    if raw is None:
        if ad.get('split_child_demoted',info.get('admission_split_child_demoted',False)):
            return {str(i):old.unresolved('MISSING_RAW_PREDEMOTION_RANKING') for i in (0,1,2)}
        raw=route.get('ranking_ids')
    if raw is None or ad.get('policy')!='A-DEC':
        return {str(i):old.unresolved('MISSING_ADEC_PROFILE_OR_RANKING') for i in (0,1,2)}
    margin=ad.get('route_margin_1_2') if ad.get('route_margin_evaluated') else None
    # This verifies the real frozen policy on recorded sufficient statistics;
    # binder enumeration / RT1 cannot be rechecked without historical nodes.
    tokens,_=adm.ordered_identifier_tokens(ArenaCache,question)
    hits=ad.get('identified_candidates',[])
    refusal=adm.identifier_unbound_abstention(dict(identifier_tokens=tokens,identified_candidates=hits))
    baseline=dict(status='RECORDED_PROFILE',plan=[] if refusal else ad.get('rank_plan',[]),
        mounts=bool(ad.get('rank_plan') and not refusal),refused=bool(refusal),
        reason='recorded binder/RT1 profile; whole-state Rule0 replay unavailable')
    s=dict(question=question,nodes=[],eligible=[],ranking=raw,split_members=[],margin=margin)
    rule1=evaluate(s,1) if not shaped_tokens(question) else old.unresolved('IDENTIFIER_SHAPED_QUERY_REQUIRES_PREPROBE_NODES')
    rule2=evaluate(s,2) if margin is not None and (margin>0 or not shaped_tokens(question)) else old.unresolved('MISSING_EXACT_MARGIN_OR_TIED_BINDERS')
    for value in (rule1,rule2):
        if value['status']=='EVALUATED':value['evidence_class']='exact policy short circuit on recorded ranking; node state not decision-relevant'
    return {'0':baseline,'1':rule1,'2':rule2}


def c2_replay():
    rows=[]
    for p in sorted(old.C2.glob('cells/*/worker.json')):
        w=lt.read(p)
        for ordinal,r in enumerate(w['rows']):
            c=w['cell'];pid=r['probe_id'];cp=old.C2/'checkpoints'/c['side']/c['battery']/pid/'checkpoint.json'
            checkpoint=lt.read(cp);manifest=cp.parent/'repository/manifest.json'
            if lt.sha(manifest)!=checkpoint['files']['manifest.json']:raise ValueError('STATE_MANIFEST_SHA')
            nodes=lt.read(manifest)['nodes'];context=checkpoint['context']
            q=context['probe']['question'] if 'probe' in context else context['event']['user']
            # Mechanical r1 adapter defect: manifest_projection intentionally
            # omits top-level kind; full manifest preserves recall eligibility.
            # Borrow full recorded nodes, never infer kinds from their text.
            repaired=copy.deepcopy(checkpoint);repaired['manifest_projection']=nodes
            # Reuse adapter with a read proxy restricted to this one JSON input.
            original_read=old.lt.read
            def read(path):return repaired if Path(path)==cp else original_read(path)
            old.lt.read=read
            try:s=old.frozen_c2(r,cp)
            finally:old.lt.read=original_read
            decisions=old.result(s);baseline=decisions['0'];recorded=r['route_receipt']['admission']['rank_plan']
            parity=baseline['status']=='EVALUATED' and baseline['proposed_plan']==recorded
            if not parity:decisions={str(i):old.unresolved('RULE0_STATE_REPLAY_PARITY_FAILED') for i in (0,1,2)}
            rows.append(dict(battery='C2',execution_id=c['id']+':'+pid,question_id=pid,question=q,scored=True,
                currently_correct=r['correct'],currently_refused=is_refused(r['served_answer']),
                source_path=str(p),source_sha256=lt.sha(p),row_ordinal=ordinal,
                state_path=str(manifest),state_sha256=lt.sha(manifest),checkpoint_sha256=lt.sha(cp),
                baseline_plan_parity=parity,rules=decisions))
    return rows


def sessions(base,battery):
    unique={};copies=[]
    for row in old.session_rows(base,battery):
        instrument_path=Path(row['instrumentation_path'])
        instruments=[json.loads(x) for x in instrument_path.read_text().splitlines()]
        instrument=next(i for i in instruments if i['turn']==row['turn'])
        key=digest(instrument)
        if key in unique:
            copies.append(dict(canonical_execution_id=unique[key]['execution_id'],copied_path=row['source_path'],
                               turn=row['turn'],instrument_record_sha256=key));continue
        transcript=[json.loads(x) for x in Path(row['source_path']).read_text().splitlines()]
        answer=next(t['assistant'] for t in transcript if t['turn']==row['turn'])
        row['currently_refused']=is_refused(answer)
        row['instrument_record_sha256']=key;row['scored']=row['currently_correct'] is not None
        row['rules']=partial(row['question'],instrument.get('route_receipt',{}))
        unique[key]=row
    return list(unique.values()),copies


def eb1_sup():
    summary=old.EB1/'grm_eb1_summary.json';rows=[]
    for r in lt.read(summary)['G2_sup_battery_spec_frame']['rows']:
        rows.append(dict(battery='EB1',execution_id='G2:'+r['probe_id'],question_id=r['probe_id'],scored=True,
            currently_correct=r['correct'],currently_refused=is_refused(r['served_answer']),
            source_path=str(summary),source_sha256=lt.sha(summary),
            rules={str(i):old.unresolved('EB1_G2_SUMMARY_NO_PREPROBE_STATE_OR_FULL_ADMISSION_PROFILE') for i in (0,1,2)}))
    return rows


def summarize(rows):
    table=[]
    for battery in ('LT1','EB1','C2','WC1'):
        for scope in ('scored','unscored'):
            selected=[r for r in rows if r['battery']==battery and r['scored']==(scope=='scored')]
            if not selected:continue
            for rule in ('0','1','2'):
                valid=[r for r in selected if r['rules'][rule]['status']=='EVALUATED']
                risk=[];gain=[];refused_mount=[]
                for r in valid:
                    base=r['rules']['0'];alt=r['rules'][rule]
                    if base.get('plan') is None:continue
                    if r['currently_correct'] and alt['plan']!=base['plan']:risk.append(r['execution_id'])
                    if r['currently_refused'] and alt['mounts']:
                        refused_mount.append(r['execution_id'])
                        if not base.get('mounts'):gain.append(r['execution_id'])
                table.append(dict(battery=battery,scope=scope,rule=int(rule),total=len(selected),evaluated=len(valid),
                    mounts=sum(r['rules'][rule]['mounts'] for r in valid),
                    correct_plan_change_ids=risk,new_admission_gain_ids=gain,currently_refused_would_mount_ids=refused_mount,
                    unresolved_ids=[r['execution_id'] for r in selected if r['rules'][rule]['status']!='EVALUATED'],
                    note='Risk is plan change, NOT observed answer regression. Existing reader refusals that already had admission are separate from new admission gains.'))
    return table


def main():
    reg=lt.read(REG)
    for path,d in reg['inputs'].items():
        if lt.sha(lt.ROOT/path)!=d:raise ValueError('SUPPLEMENT_INPUT_SHA')
    OUT.mkdir(exist_ok=False)
    old_rows=[json.loads(x) for x in (old.OUT/'questions.jsonl').read_text().splitlines()]
    rows=[dict(r,scored=True) for r in old_rows if r['battery']=='LT1']+c2_replay()
    copies=[]
    for base,battery in ((old.EB1,'EB1'),(old.C2,'C2'),(old.WC1,'WC1')):
        new,duplicates=sessions(base,battery);rows.extend(new);copies.extend(duplicates)
    rows+=eb1_sup()+[dict(r,scored=True) for r in old.wc1_sup()]
    for r in rows:old.emit(OUT/'questions.jsonl',r)
    lt.create(OUT/'copied_prefixes.json',copies)
    lt.create(OUT/'table.json',summarize(rows))
    lt.create(OUT/'receipt.json',dict(status='RED_INCOMPLETE_HISTORICAL_INPUTS',registration_sha256=lt.sha(REG),
        rows_sha256=lt.sha(OUT/'questions.jsonl'),table_sha256=lt.sha(OUT/'table.json'),
        copied_prefixes_sha256=lt.sha(OUT/'copied_prefixes.json'),deduplicated_copies=len(copies),
        no_gpu=True,no_reader=True,whole_state_replay='LT1 and C2 scored; remaining sufficient-statistic decisions labelled individually'))
    for t in summarize(rows):
        print(t['battery'],t['scope'],t['rule'],'n',t['total'],'eval',t['evaluated'],'mount',t['mounts'],
              'risk',len(t['correct_plan_change_ids']),'newgain',len(t['new_admission_gain_ids']),
              'refused_mount',len(t['currently_refused_would_mount_ids']),'unresolved',len(t['unresolved_ids']))
if __name__=='__main__':main()
