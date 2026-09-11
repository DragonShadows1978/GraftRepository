#!/usr/bin/env python3
"""Lead-only fixed-residency middle-prompt replay; CPU dry run by default.

Prior art: RD1 amendment-1 replay (GRM contributors, 2026), C7 oracle cache
isolation, C2 environment and CMC1 foreground lease. Reuse production reader,
scorer, checkpoint payloads, and lease; ours is paired middle-prompt dispatch.
No prior art known to me for this exact composition. No core changes.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.grm_c7_amendment7 import OUT, REG, read, sha, write, need, verify
from scripts.grm_c7_common import score


def preflight():
    r=verify()
    need(shutil.disk_usage(OUT).free>=r['free_space_min_bytes'],'FREE_SPACE_BELOW_20_GIB')
    return r


def pair(repo,q,destination):
    # Prior art: RD1 replay_attempt/C7 oracle (2026): isolate transient cache,
    # layer shift and deferred keys for each production _attempt. All answers
    # are observed before scorer invocation; expected values never enter input.
    wanted=q['mounted_ids'];a=repo.arena
    need(wanted and len(wanted)==len(set(wanted)),'INVALID_RECORDED_MOUNTS')
    need(list(a._resolve_revision_mounts(wanted))==wanted,'L2_CHANGED_RECORDED_MOUNTS')
    original_count=len(a.grafts)
    out={}
    for arm,key in [('A0','plain'),('A1','middle')]:
        a.reset_live_cache();a.eb1_begin_turn()
        shifts=[(x.self_attn,hasattr(x.self_attn,'live_shift'),getattr(x.self_attn,'live_shift',None)) for x in a.m.layers]
        try:
            for att,_,_ in shifts:att.live_shift=a.live_shift
            a._ensure_h(wanted)
            need(all(a.grafts[i].get('h') is not None for i in wanted),'MISSING_PAYLOAD')
            wrapped=a._format_step_prompt(q[key]);tokens=list(a.encode(wrapped))
            ans,info=a._attempt(q[key],wanted,32,False,tuple(a.stop_sequences),defer_memory=True)
            need(list(a.cur_mounts)==wanted,'ACTUAL_MOUNTS_CHANGED')
            need(len(a.grafts)==original_count,'UNEXPECTED_DEPOSIT')
            value={'probe_id':q['probe_id'],'arm':arm,'answer':str(ans),'score':score(ans,q['probe']),
                'evidence_class':'production fixed-mount checkpoint reader',
                'mounted_ids':list(a.cur_mounts),'mounted_payload_sha256':q['mounted_payload_sha256'],
                'prompt':q[key],'wrapped_prompt':wrapped,'prompt_token_ids':tokens,
                'checkpoint':q['checkpoint'],'checkpoint_sha256':q['checkpoint_sha256'],
                'historical_answer':q['historical']['answer'],
                'r3_byte_equal':str(ans).encode()==q['historical']['answer'].encode(),
                'info':info,'registration_sha256':sha(REG)}
            write(destination/(q['probe_id']+'_'+arm+'.json'),value);out[arm]=value
            if arm=='A0':need(value['r3_byte_equal'],'A0_R3_BYTE_MISMATCH_STOP')
        finally:
            a.reset_live_cache();a._deferred_route_keys={}
            for att,exists,val in shifts:
                if exists:att.live_shift=val
                elif hasattr(att,'live_shift'):delattr(att,'live_shift')
    return out


def carry_refusal(q,destination):
    need(not q['mounted_ids'] and q['historical']['route_info'].get('abstain_reason')=='identifier_unbound','NOT_RECORDED_ADMISSION_REFUSAL')
    for arm in ('A0','A1'):
        write(destination/(q['probe_id']+'_'+arm+'.json'),{'probe_id':q['probe_id'],'arm':arm,
            'answer':q['historical']['answer'],'score':score(q['historical']['answer'],q['probe']),
            'evidence_class':'recorded admission policy carry-forward; zero model calls',
            'r3_byte_equal':True,'mounted_ids':[], 'registration_sha256':sha(REG)})


def open_copy(q,session,r,loaded=None):
    # Prior art: C7 checkpoint copies and e2e.load_model_and_repo (2026).
    # Preserve each cell-end repository; do not synthesize a memory bank.
    from scripts import grm_e2e_session as e
    cp=ROOT/q['checkpoint'];descriptor=read(cp/'checkpoint.json')
    for name,digest in descriptor['files'].items():need(sha(cp/name)==digest,'FULL_CHECKPOINT_TREE_MISMATCH: '+name)
    shutil.copytree(cp/'repository',session/'repository')
    f=r['flags']
    if loaded is None:
        args=e.parse_args(['--mode','full','--session-dir',str(session),'--model-dir',r['model']['path'],
            '--native-lib',r['native_library'],'--arena-width',str(f['arena_width']),
            '--ngen','32','--max-trips','1','--live-turns',str(f['live_turns']),
            '--max-live',str(f['max_live']),'--topk','3','--turn-pipeline',f['turn_pipeline'],
            '--restart-after','999','--skip-gpu-idle-check','--probe-ladder','--sup-resolve','--adm-decisive'])
        model,tok,repo,info=e.load_model_and_repo(args,session)
        loaded=(model,tok,info)
    else:
        model,tok,info=loaded
        repo=e.GraftRepository(model,lambda t:tok.encode(t,add_special_tokens=False),
            lambda ids:tok.decode(ids,clean_up_tokenization_spaces=False),str(session/'repository'),
            autosave=False,arena_cls=e.GptOssGQAArenaCache,native_lib_path=r['native_library'],native_auto=False,
            vram_budget_mb=None,route_layer=int(e.gpt_oss_grm_dialect_kwargs(model.config)['route_layer']),
            arena_width=f['arena_width'],topk=f['topk'],live_turns=f['live_turns'],max_live=f['max_live'],
            ephemeral=True,sink_text=e.HARMONY_SINK,prompt_template=e.harmony_turn,stop_sequences=e.HARMONY_STOPS,
            storage_bits=8,revision_resolution=True,decisive_admission=True)
    a=repo.arena
    need(a.ephemeral and a.width==96 and a.n_sink==19,'PROFILE_GEOMETRY_MISMATCH')
    need(list(a.encode(r['model_frame']['sink_text']))==r['model_frame']['sink_token_ids'],'SINK_TOKENS_MISMATCH')
    return repo,loaded


def batch(batch_id):
    r=preflight();need(batch_id in r['batches'],'UNKNOWN_BATCH')
    destination=OUT/'gpu'/batch_id
    # Prior art: C2/C7 atomic ownership and pessimistic reservations (2026).
    # Reject incomplete/orphaned batches, never clear another owner or retry.
    owner=OUT/'replay.active'
    fd=os.open(owner,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        prior_ids=r['batch_ids'][:r['batch_ids'].index(batch_id)]
        for b in prior_ids:need(read(OUT/'gpu'/b/'controller.json')['status']=='COMPLETE','PRIOR_BATCH_NOT_COMPLETE')
        controllers=list((OUT/'gpu').glob('*/controller.json'))
        need(all(read(p)['status']=='COMPLETE' for p in controllers),'FAILED_CAMPAIGN_STOP')
        reservations=list((OUT/'gpu').glob('*/reservation.json'))
        orphan=[p for p in reservations if not p.with_name('controller.json').exists()]
        need(not orphan,'ORPHAN_RESERVATION_STOP')
        charged=sum(read(p)['charged_seconds'] for p in controllers)
        need(charged+280<=1800,'BUDGET_RAIL')
        destination.mkdir(parents=True,exist_ok=False)
        write(destination/'reservation.json',{'seconds':280,'batch':batch_id,'registration_sha256':sha(REG)})
        qs={q['probe_id']:q for q in read(OUT/'requests.json')}
        started=None;elapsed=0;status='FAILED';error=None;loaded=None
        try:
            # Lead invocation only. No GPU imports or lease on --dry-run.
            from scripts.grm_c2_cells import environment
            env=environment(r['flags'])
            for k in list(os.environ):
                if k.startswith('GRM_'):del os.environ[k]
            os.environ.update({k:v for k,v in env.items() if k.startswith('GRM_')})
            from scripts.grm_cmc1_gpu_arms import gpu_lease
            with gpu_lease(r['lease_seconds'],0):
                started=time.monotonic()
                for pid in r['batches'][batch_id]:
                    q=qs[pid]
                    if not q['mounted_ids']:carry_refusal(q,destination);continue
                    repo,loaded=open_copy(q,destination/'sessions'/pid,r,loaded)
                    try:pair(repo,q,destination)
                    finally:repo.close()
                elapsed=time.monotonic()-started
                need(elapsed<=280,'LEASE_OVERRUN_RED')
                write(destination/'model.json',loaded[2] if loaded else {'gpu_model_loaded':False})
                status='COMPLETE'
        except BaseException as exc:
            error=f'{type(exc).__name__}: {exc}'
        finally:
            if started is not None:elapsed=time.monotonic()-started
            write(destination/'controller.json',{'status':status,'error':error,
                'charged_seconds':max(280,elapsed) if status!='COMPLETE' else elapsed,
                'reservation_seconds':280,'registration_sha256':sha(REG)})
            time.sleep(30)  # lead-only foreground cooldown, after lease release
        if status!='COMPLETE':raise RuntimeError(error)
    finally:owner.unlink()  # only this process's successfully created file


def summary():
    r=verify(payloads=False);qs=read(OUT/'requests.json');values={};complete=True
    for b,ids in r['batches'].items():
        d=OUT/'gpu'/b
        complete &= (d/'controller.json').exists() and read(d/'controller.json')['status']=='COMPLETE'
        for pid in ids:
            for arm in ('A0','A1'):
                p=d/(pid+'_'+arm+'.json')
                if p.exists():
                    row=read(p);need(row['registration_sha256']==sha(REG),'RESULT_BINDING_MISMATCH');values[(pid,arm)]=row
    complete &= len(values)==80
    parity=all(values.get((q['probe_id'],'A0'),{}).get('r3_byte_equal',False) for q in qs)
    groups={}
    for group in ('controls','fresh','folded'):
        cohort=[q for q in qs if (q['probe']['expected']=='UNKNOWN' if group=='controls' else q['probe']['expected']!='UNKNOWN' and q['probe']['class']==group)]
        groups[group]={arm:{k:sum(values.get((q['probe_id'],arm),{}).get('score',{}).get(k,0) for q in cohort)
            for k in ('exact_correct','wrong_value_error','abstention_error','unsupported_answer_error')}
            for arm in ('A0','A1')}
    passed=complete and parity and groups['controls']['A1']['unsupported_answer_error']<=5 and groups['fresh']['A1']['exact_correct']>=2
    return {'status':'PASS' if passed else 'RED' if complete else 'NOT_MEASURED',
        'complete':complete,'row_receipts':len(values),'A0_parity':parity,'groups':groups,
        'scope':'fixed-mount reader contrast, refusal carry-forwards separated; not admission or product certification'}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dry-run',action='store_true');ap.add_argument('--batch');ap.add_argument('--summary',action='store_true');a=ap.parse_args()
    if a.batch:batch(a.batch)
    elif a.summary:print(json.dumps(summary(),indent=2))
    else:
        r=preflight();print(json.dumps({'status':'NOT_RUN_NO_GPU_ORDER','gpu_executed':False,
            'registration_sha256':sha(REG),'cohort':r['cohort'],'batches':r['batches'],
            'reserved_seconds':1680,'cap_seconds':1800},indent=2))
