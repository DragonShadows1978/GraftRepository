#!/usr/bin/env python3
"""FIX8 lead replay plus isolated execution of the existing A7 contrast.

Prior art: C7 amendment7/RD1 copied checkpoints, production _attempt/ladder,
C2 environment and CMC1 foreground leases (GRM contributors, 2026). Reuse
these contracts, frozen scorer, and A7 campaign unchanged. Ours: explicit
archived-core input mapping and 24-row mount+read campaign. No prior art
known to me for this exact composition. Never a GPU loader on dry-run.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.grm_c7_amendment7 import read, sha, write, need
OUT=ROOT/'artifacts/grm_scout_fix8'
REG=OUT/'replay_registration.json'
A7=ROOT/'artifacts/grm_c7/r3/amendment_7'
CORE=('core/grm_text_norm.py','core/grm_admission.py','core/graft_arena.py')


def verify():
    need(sha(REG)==REG.with_suffix('.sha256').read_text().split()[0],'FIX8_REGISTRATION_SHA_MISMATCH')
    r=read(REG)
    for p,h in r['inputs'].items():need(sha(p)==h,'FIX8_INPUT_SHA_MISMATCH: '+p)
    qs=read(OUT/'requests.json')
    need(len(qs)==len({q['probe_id'] for q in qs})==24,'FIX8_COHORT_MISMATCH')
    need([pid for ids in r['batches'].values() for pid in ids]==[q['probe_id'] for q in qs],'FIX8_BATCH_COHORT_MISMATCH')
    need(len(r['batches'])*r['lease_seconds']==1680<=r['gpu_cap_seconds']==1800,'FIX8_BUDGET_MISMATCH')
    for q in qs:
        cp=ROOT/q['checkpoint'];d=read(cp/'checkpoint.json')
        need(sha(cp/'checkpoint.json')==q['checkpoint_sha256'],'FIX8_CHECKPOINT_SHA_MISMATCH')
        need(d['binding']['registration_sha256']==sha(ROOT/'artifacts/grm_c7/r3/registration.json'),'FIX8_CHECKPOINT_BINDING_MISMATCH')
        for name in ('state.json','repository/manifest.json'):
            need(sha(cp/name)==d['files'][name],'FIX8_CHECKPOINT_METADATA_MISMATCH')
    return r


def verify_middle(payloads=True):
    # Prior art: C7 explicit before/after compatibility maps (2026).
    # Verify actual archived bytes against ORIGINAL A7 hashes. No hash is
    # replaced or waived; the loaded origin is asserted below and recorded.
    need(sha(A7/'registration.json')==read(REG)['middle_registration_sha256'],'A7_REGISTRATION_CHANGED')
    old=read(A7/'registration.json')
    need(sha(A7/'registration.json')==(A7/'registration.sha256').read_text().split()[0],'A7_SHA_MISMATCH')
    for p,h in old['inputs'].items():
        relative=str(Path(p).relative_to(ROOT)) if Path(p).is_relative_to(ROOT) else None
        actual=OUT/'before'/relative if relative in CORE else Path(p)
        need(sha(actual)==h,'A7_ARCHIVED_INPUT_SHA_MISMATCH: '+str(actual))
    for q in read(A7/'requests.json'):
        cp=ROOT/q['checkpoint'];d=read(cp/'checkpoint.json')
        need(sha(cp/'checkpoint.json')==q['checkpoint_sha256'],'A7_CHECKPOINT_SHA_MISMATCH')
        need(d['binding']['registration_sha256']==sha(ROOT/'artifacts/grm_c7/r3/registration.json'),'A7_CHECKPOINT_BINDING_MISMATCH')
        for name in ('repository/manifest.json','state.json'):
            need(sha(cp/name)==d['files'][name],'A7_METADATA_MISMATCH')
        if payloads:
            for i,h in q['mounted_payload_sha256'].items():need(sha(cp/f'repository/nodes/{int(i):04d}.npz')==h,'A7_PAYLOAD_MISMATCH')
    return old


def middle_runner():
    # A fresh process imports the old three modules from their immutable
    # archive. Every other module remains the byte-pinned A7 input. This
    # changes no live file or source hash and never runs a second contrast.
    verify()
    need(not any(name in sys.modules for name in ('core.grm_text_norm','core.grm_admission','core.graft_arena')),'MIDDLE_REQUIRES_FRESH_PROCESS')
    import core
    core.__path__.insert(0,str(OUT/'before/core'))
    origins={}
    for path in CORE:
        name=path[:-3].replace('/','.')
        module=importlib.import_module(name)
        need(Path(module.__file__).resolve()==(OUT/'before'/path).resolve(),'MIDDLE_CORE_ORIGIN_MISMATCH')
        origins[name]={'path':module.__file__,'sha256':sha(module.__file__)}
    from scripts import grm_c7_middle_replay as middle
    middle.verify=verify_middle  # explicit checked path mapping, frozen policy unchanged
    return middle,origins


def serve(repo,q,destination):
    """Real mount decision + fit + reader, no historical-mount injection."""
    from scripts.grm_e2e_session import _probe_ladder_chat
    from scripts.grm_c7_common import score
    a=repo.arena
    a.reset_live_cache();a._deferred_route_keys={}
    original=[g.get('text') for g in a.grafts]
    answer,info=_probe_ladder_chat(repo,q['plain'],topk=3,ngen=32,max_trips=1,defer_memory=True)
    need([g.get('text') for g in a.grafts[:len(original)]]==original,'EXISTING_SOURCE_TEXT_MUTATED')
    # Production fit may deposit split children in this private copy. Preserve
    # their provenance; no history rewrite and no synthetic source rescue.
    obs=info.get('_route_observation',{})
    profile=obs.get('admission_profile') or {}
    mounts=list(a.cur_mounts)
    result={'probe_id':q['probe_id'],'answer':str(answer),'score':score(answer,q['probe']),
        'historical_answer':q['historical']['answer'],'historical_score':q['historical']['score'],
        'question':q['plain'],'checkpoint':q['checkpoint'],'checkpoint_sha256':q['checkpoint_sha256'],
        'mounted_ids':mounts,'admission_profile':profile,'route_info':info,
        'identifier_unbound':info.get('abstain_reason')=='identifier_unbound',
        'admitted':bool(profile.get('identified_candidates') or info.get('served_from_node_ids')),
        'mounted_texts':{str(i):a.grafts[i].get('text') for i in mounts},
        'split_children':[{k:g.get(k) for k in ('text','ntok','sources','kind')} for g in a.grafts[len(original):]],
        'registration_sha256':sha(REG),'evidence_class':'production checkpoint mount decision + fit + reader, cell-end state; no preceding-turn replay'}
    write(destination/(q['probe_id']+'.json'),result)
    a.reset_live_cache();a._deferred_route_keys={}
    return result


def campaign_state(r):
    controllers=list((OUT/'gpu').glob('*/controller.json'))
    need(all(p.parent.name in r['batches'] for p in controllers),'UNKNOWN_BATCH_RECEIPT')
    for p in controllers:
        c=read(p)
        need(c['registration_sha256']==sha(REG),'RESULT_BINDING_MISMATCH')
        need(c['status']=='COMPLETE','FAILED_CAMPAIGN_STOP')
        need(c['charged_seconds']>=r['lease_seconds'],'INVALID_CHARGE')
    reservations=list((OUT/'gpu').glob('*/reservation.json'))
    need(all(p.with_name('controller.json').exists() for p in reservations),'ORPHAN_RESERVATION_STOP')
    need(all((p.parent/'reservation.json').exists() for p in controllers),'MISSING_RESERVATION')
    for p in reservations:
        value=read(p)
        need(value['registration_sha256']==sha(REG) and value['seconds']==r['lease_seconds'],'RESERVATION_BINDING_MISMATCH')
    return sum(read(p)['charged_seconds'] for p in controllers)


def batch(batch_id):
    r=verify();need(batch_id in r['batches'],'UNKNOWN_BATCH')
    need(shutil.disk_usage(OUT).free>=r['free_space_min_bytes'],'FREE_SPACE_BELOW_20_GIB')
    # Prior art: C7 atomic owner and pessimistic budget, no retries (2026).
    owner=OUT/'replay.active'
    fd=os.open(owner,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        charged=campaign_state(r)
        for b in list(r['batches'])[:list(r['batches']).index(batch_id)]:
            need(read(OUT/'gpu'/b/'controller.json')['status']=='COMPLETE','PRIOR_BATCH_INCOMPLETE')
        need(charged+r['lease_seconds']<=r['gpu_cap_seconds'],'GPU_BUDGET_RAIL')
        destination=OUT/'gpu'/batch_id;destination.mkdir(parents=True,exist_ok=False)
        write(destination/'reservation.json',{'seconds':r['lease_seconds'],'registration_sha256':sha(REG)})
        started=None;elapsed=0.;status='FAILED';error=None;loaded=None
        try:
            from scripts.grm_c2_cells import environment
            env=environment(r['flags'])
            for k in list(os.environ):
                if k.startswith('GRM_'):del os.environ[k]
            os.environ.update({k:v for k,v in env.items() if k.startswith('GRM_')})
            from scripts.grm_cmc1_gpu_arms import gpu_lease
            from scripts.grm_c7_middle_replay import open_copy
            qs={q['probe_id']:q for q in read(OUT/'requests.json')}
            with gpu_lease(r['lease_seconds'],0):
                started=time.monotonic()
                for pid in r['batches'][batch_id]:
                    q=qs[pid]
                    repo,loaded=open_copy(q,destination/'sessions'/pid,r,loaded)
                    try:serve(repo,q,destination)
                    finally:repo.close()
                elapsed=time.monotonic()-started
                need(elapsed<=r['lease_seconds'],'LEASE_OVERRUN_RED')
                write(destination/'model.json',loaded[2])
                status='COMPLETE'
        except BaseException as exc:
            error=f'{type(exc).__name__}: {exc}'
        finally:
            if started is not None:elapsed=time.monotonic()-started
            write(destination/'controller.json',{'status':status,'error':error,
                'elapsed_seconds':elapsed,'charged_seconds':max(r['lease_seconds'],elapsed),
                'registration_sha256':sha(REG)})
            time.sleep(30)  # foreground lead-only cooldown outside GPU lease
        if status!='COMPLETE':raise RuntimeError(error)
    finally:owner.unlink()


def summary():
    r=verify();values=[];complete=True;charge=0.
    for b,ids in r['batches'].items():
        d=OUT/'gpu'/b;cp=d/'controller.json'
        if not cp.exists():complete=False
        else:
            c=read(cp);need(c['registration_sha256']==sha(REG),'RESULT_BINDING_MISMATCH')
            complete &= c['status']=='COMPLETE';charge+=c['charged_seconds']
        for pid in ids:
            p=d/(pid+'.json')
            if not p.exists():complete=False;continue
            row=read(p);need(row['registration_sha256']==sha(REG) and row['probe_id']==pid,'ROW_BINDING_MISMATCH')
            values.append(row)
    complete &= len(values)==24 and charge<=r['gpu_cap_seconds']
    groups={}
    qs={q['probe_id']:q for q in read(OUT/'requests.json')}
    for group in ('fresh','folded','controls'):
        cohort=[v for v in values if (qs[v['probe_id']]['probe']['expected']=='UNKNOWN' if group=='controls' else
            qs[v['probe_id']]['probe']['expected']!='UNKNOWN' and qs[v['probe_id']]['probe']['class']==group)]
        groups[group]={'rows':len(cohort),**{k:sum(v['score'][k] for v in cohort) for k in
            ('exact_correct','wrong_value_error','abstention_error','unsupported_answer_error')}}
    admitted=sum(v['admitted'] and not v['identifier_unbound'] and bool(v['mounted_ids']) for v in values)
    return {'status':'PASS_ADMISSION' if complete and admitted==24 else 'RED' if complete else 'NOT_MEASURED',
        'complete':complete,'rows':len(values),'admitted_and_mounted':admitted,'charged_seconds':charge,
        'groups':groups,'quality_rule':'Frozen scorer, all observed outcomes reported; no quality success threshold or product certification registered.'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dry-run',action='store_true');ap.add_argument('--batch');ap.add_argument('--summary',action='store_true')
    ap.add_argument('--middle-check',action='store_true');ap.add_argument('--middle-batch');ap.add_argument('--middle-summary',action='store_true');a=ap.parse_args()
    if a.middle_check or a.middle_batch or a.middle_summary:
        middle,origins=middle_runner()
        if a.middle_batch:
            d=A7/'gpu'/a.middle_batch
            if d.exists():
                c=read(d/'controller.json')
                need(c['registration_sha256']==sha(A7/'registration.json') and c['status']=='COMPLETE','A7_STARTED_BATCH_NEVER_RETRIED')
                print(json.dumps({'status':'ALREADY_COMPLETE','batch':a.middle_batch}))
            else:
                # Receipt belongs to FIX8 bridge; A7 outcomes retain A7 binding.
                write(OUT/'middle_origins'/f'{a.middle_batch}.json',{'origins':origins,'middle_registration_sha256':sha(A7/'registration.json'),'bridge_registration_sha256':sha(REG)})
                middle.batch(a.middle_batch)
        elif a.middle_summary:
            value=middle.summary();print(json.dumps(value,indent=2))
            need(value['complete'] and value['A0_parity'],'A7_INCOMPLETE_OR_PARITY_FAILURE')
        else:
            middle.preflight();print(json.dumps({'status':'VERIFIED_EXISTING_A7','origins':origins,'gpu_executed':False},indent=2))
    elif a.batch:batch(a.batch)
    elif a.summary:print(json.dumps(summary(),indent=2))
    else:
        r=verify();print(json.dumps({'status':'REGISTERED_NOT_RUN','gpu_executed':False,'registration_sha256':sha(REG),'rows':24,
            'reserved_seconds':1680,'cap_seconds':1800,'middle_registration_sha256':r['middle_registration_sha256']},indent=2))

if __name__=='__main__':main()
