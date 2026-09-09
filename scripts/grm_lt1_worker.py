#!/usr/bin/env python3
"""LT1 foreground resumable worker/controller, Rule 0 only.

Prior art: GRM C7/C2/EB1 (contributors, 2026), reuse profile loader, real
ladder, a3 oracle, metadata projection, payload checks and SHA checkpoints.
Recovery concept: Mohan et al., ARIES (1992), unverified — lead to check
ARIES recovery; no WAL implementation borrowed. New: natural turn replay,
combined 3-hour accounting and no-kill cooperative deadline glue.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid
from scripts import grm_lt1 as lt
from scripts.grm_c7_common import hashes
from scripts.grm_c7_run import emit, lines, oracle, seats
RUN = lt.OUT/'amendment1/run_rule0'


def bind(arm):
    return lt.binding(arm)


def initial_state():
    return dict(next_turn=1,turn_nodes={},transcript=[],process_id=None,pid=None)


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('WORKER_COOPERATIVE_DEADLINE')


def observe(arena, directory, context):
    # Prior art: C7 per-attempt summed residency (GRM, 2026), same bound.
    original=arena._attempt
    def traced(*args,**kwargs):
        answer,info=original(*args,**kwargs)
        if context['phase']!='oracle':
            record_seats(arena,directory,context,info)
        return answer,info
    arena._attempt=traced


def record_seats(arena,directory,context,info=None):
    row=dict(context,**seats(arena,info),admission_rule=0)
    emit(directory/'residency.jsonl',row)
    if row['summed_token_seats']>2*row['width']+row['actual_recency_token_seats']:
        raise ValueError('RESIDENCY_BOUND_EXCEEDED')
    return row


def sentinels(repo,fixture,registration,directory,context):
    from scripts.grm_e2e_session import _probe_ladder_chat
    rows=[]
    for pid in registration['restart_sentinels']:
        p=next(p for p in fixture['probes'] if p['id']==pid)
        answer,info=_probe_ladder_chat(repo,p['question'],topk=3,ngen=32,max_trips=1,defer_memory=True)
        rows.append(dict(probe_id=pid,answer=str(answer),score=lt.score(answer,p['expected']),
                         route_info=info,residency=record_seats(repo.arena,directory,context,info),admission_rule=0))
    emit(directory/'restart.jsonl',dict(context,pid=os.getpid(),rows=rows,admission_rule=0))
    return rows


def execute(cell, directory, registration, loader, *, run=RUN, deadline=float('inf'), fake=False):
    """Shared worker body; CPU gates replace only model/payload loader boundary."""
    from scripts import grm_e2e_session as e2e
    from scripts.grm_c2_cells import manifest_projection, assert_payloads, strict_capture_grade
    fixture=lt.read(lt.FIX); flags=registration['arms'][cell['arm']]['flags']
    session=directory/'session';session.mkdir()
    state=initial_state();prior=None;previous_nodes=None
    if cell['depends']:
        previous=run/cell['depends'];prior=lt.read(previous/'worker.json')
        if prior['binding']!=bind(cell['arm']):raise ValueError('PRIOR_WORKER_BINDING')
        state=lt.validate_checkpoint(previous/'checkpoint',cell['start'],bind(cell['arm']))
        if not fake:assert_payloads(previous/'checkpoint/repository')
        previous_nodes=lt.read(previous/'checkpoint/repository/manifest.json')['nodes']
        shutil.copytree(previous/'checkpoint/repository',session/'repository')
    elif cell['start']!=1:raise ValueError('MISSING_INITIAL_DEPENDENCY')
    old_process=state['process_id'];old_pid=state['pid']
    state.update(process_id=str(uuid.uuid4()),pid=os.getpid())
    if old_process==state['process_id'] or old_pid==state['pid']:
        raise ValueError('RESTART_REQUIRES_NEW_PROCESS')
    repo,model_info=loader(session,flags,state)
    context=dict(turn=cell['start']-1,phase='load',process_id=state['process_id'])
    try:
        a=repo.arena
        if a.width!=flags['arena_width'] or not a.ephemeral:raise ValueError('PROFILE_GEOMETRY_MISMATCH')
        if not fake and (a.n_sink!=19 or list(a.encode(registration['model_frame']['sink_text']))!=registration['model_frame']['sink_token_ids']):
            raise ValueError('EB1_SINK_TOKEN_MISMATCH')
        if previous_nodes is not None:
            loaded=[repo._node_manifest(g) for g in a.grafts]
            if manifest_projection(previous_nodes)!=manifest_projection(loaded):raise ValueError('RESTART_METADATA_CHANGED')
        observe(a,directory,context)
        restart_ok=None
        if cell['start'] in (71,141):
            context['phase']='restart_after'
            after=sentinels(repo,fixture,registration,directory,context)
            before=prior['restart_before']
            restart_ok=bool(before and len(after)==len(before)==2 and all(
                x['probe_id']==y['probe_id'] and x['score']['category']==y['score']['category']
                and x['score']['exact_correct']==y['score']['exact_correct'] for x,y in zip(before,after)))
            if not restart_ok:raise ValueError('RESTART_SENTINEL_SCORE_CHANGED')
        probes={p['turn']:p for p in fixture['probes']};nrows=0
        for event in fixture['turns'][cell['start']-1:cell['stop']]:
            check_deadline(deadline)
            turn=event['turn'];context.update(turn=turn,phase='session')
            if event['kind']=='probe':
                p=probes[turn];source_ids=[state['turn_nodes'][str(t)] for t in p['source_turns']]
                nominees=a.eb1_begin_turn()
                if set(source_ids)&set(nominees):raise ValueError('SOURCE_IS_RECENCY_NOMINEE')
                # Prior art: C7 defer_memory probes (GRM, 2026). No probe answer
                # redeposit; source IDs are audit-only, never passed to route.
                answer,info=e2e._probe_ladder_chat(repo,p['question'],topk=3,ngen=32,max_trips=1,defer_memory=True)
                memory=dict(answer=str(answer),score=lt.score(answer,p['expected']),route_info=info,
                            residency=record_seats(a,directory,context,info),admission_rule=0)
                context['phase']='oracle';check_deadline(deadline)
                upper=oracle(a,p,32);upper['score']=lt.score(upper['answer'],p['expected'])
                upper['admission_rule']='oracle_no_admission'
                if not all(s in upper['wrapped_prompt'] for s in p['oracle_source_texts']):raise ValueError('ORACLE_SOURCE_MISSING')
                emit(directory/'probes.jsonl',dict(probe_id=p['id'],turn=turn,question=p['question'],
                     expected=p['expected'],source_ids=source_ids,recency_ids=nominees,
                     memory=memory,oracle=upper,binding=bind(cell['arm']),admission_rule=0))
                nrows+=1
            elif event['kind']=='recap':
                answer,info=e2e._probe_ladder_chat(repo,event['user'],topk=3,ngen=160,max_trips=1,defer_memory=True)
                lt.create(directory/'recap.json',dict(answer=str(answer),route_info=info,admission_rule=0,
                    matched=sum(lt.score(answer,p['expected'])['exact_correct'] for p in fixture['decisions']),out_of=5,binding=bind(cell['arm'])))
            else:
                # Prior art: LT1 r1 / EB1 frozen complete-turn replay (2026).
                # User corrections are ordinary prose, not hidden supersede calls.
                idx=a.feed(e2e.harmony_turn(event['user'],event['assistant']))
                a.grafts[idx]['kind']='turn';state['turn_nodes'][str(turn)]=idx
            state['transcript'].append(copy.deepcopy(event));state['next_turn']=turn+1
            context['phase']='post_turn';record_seats(a,directory,context)
        restart_before=None
        if cell['stop'] in (70,140):
            context['phase']='restart_before';restart_before=sentinels(repo,fixture,registration,directory,context)
        check_deadline(deadline)
        repo.flush_now()
        nodes=lt.read(session/'repository/manifest.json')['nodes']
        capture=None if fake else strict_capture_grade([g for g in nodes if not g.get('retired')],
                'profile' if cell['arm']=='A' else 'defaults',a.width,a.n_sink)
        if capture is not None and not capture['valid']:raise ValueError('CAPTURE_PROVENANCE_RED')
        if fake:state['codec_words']=a.m.codec.words
        cp=directory/'checkpoint';cp.mkdir();shutil.copytree(session/'repository',cp/'repository')
        if not fake:assert_payloads(cp/'repository')
        lt.checkpoint(cp,state,bind(cell['arm']))
        check_deadline(deadline)
        return dict(cell=cell,binding=bind(cell['arm']),admission_rule=0,process_id=state['process_id'],pid=os.getpid(),
                    previous_process_id=old_process,previous_pid=old_pid,metadata_retained=True if prior else None,
                    restart_before=restart_before,restart_retained_scores=restart_ok,capture=capture,
                    rows=nrows,model_info=model_info,checkpoint_sha256=lt.sha(cp/'checkpoint.json'),
                    evidence_class='CPU fake worker' if fake else 'GPU worker E2E')
    finally:repo.close()


def worker(cell):
    if os.environ.get('GRM_LT1_LEASE_PARENT')!=str(os.getppid()):raise ValueError('WORKER_REQUIRES_LEASE_PARENT')
    # Prior art: POSIX interval alarm / Python signal exception (Python docs).
    # No kill syscall. Native calls may defer delivery: report actual overruns.
    deadline=time.monotonic()+cell['worker_seconds']
    def expired(*_):raise TimeoutError('WORKER_COOPERATIVE_DEADLINE')
    signal.signal(signal.SIGALRM,expired);signal.alarm(cell['worker_seconds'])
    def loader(session,flags,state):
        from scripts import grm_e2e_session as e2e
        from scripts.grm_c2_cells import args_for
        _,_,repo,info=e2e.load_model_and_repo(args_for(e2e,session,flags),session)
        return repo,info
    directory=RUN/'cells'/cell['id']
    try:
        result=execute(cell,directory,lt.verify(),loader,run=RUN/'cells',deadline=deadline)
        lt.create(directory/'worker.json',result)
    finally:signal.alarm(0)


def accounting(run=RUN):
    spent=0.0
    for p in (run/'cells').glob('*/reservation.json'):
        controller=p.with_name('controller.json')
        if not controller.exists():raise ValueError('ORPHAN_RESERVATION: '+str(p.parent))
        r=lt.read(controller)
        if r['status']!='COMPLETE':raise ValueError('PRIOR_CELL_RED: '+str(p.parent))
        spent+=r['charged_seconds']
    return spent


def pending(registration,run=RUN):
    # Prior art: C2 immutable reservations (2026), combined-arm ownership here.
    accounting(run)
    for cell in sorted(registration['cells'],key=lambda c:(c['start'],c['arm'])):
        d=run/'cells'/cell['id']
        if d.exists():
            c=lt.read(d/'controller.json');w=lt.read(d/'worker.json')
            if c['status']!='COMPLETE' or c['binding']!=bind(cell['arm']) or w['binding']!=bind(cell['arm']):
                raise ValueError('COMPLETED_CELL_BINDING_OR_STATUS')
            lt.validate_checkpoint(d/'checkpoint',cell['stop']+1,bind(cell['arm']))
            if lt.sha(d/'checkpoint/checkpoint.json')!=w['checkpoint_sha256']:raise ValueError('WORKER_CHECKPOINT_SHA')
            continue
        return cell
    return None


def run_cell(cell,registration):
    from scripts.grm_c2_cells import environment
    from scripts.grm_cmc1_gpu_arms import LOCK_PATH
    if accounting()+cell['lease_seconds']>10800:raise ValueError('COMBINED_GPU_BUDGET_RAIL')
    directory=RUN/'cells'/cell['id'];directory.mkdir(parents=True)
    lt.create(directory/'reservation.json',dict(seconds=285,cell=cell,binding=bind(cell['arm']),admission_rule=0))
    outer=time.monotonic();status='RED';error=None;charged=285.0;acquired=None
    try:
        # Prior art: CMC1 flock lease (2026), same shared lock, bounded wait.
        # Parent deliberately has no SIGALRM/subprocess timeout: those can kill
        # an owned child. Keep the lease until the child actually exits.
        with LOCK_PATH.open('a+') as lock:
            while True:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    if time.monotonic()-outer>=240:raise TimeoutError('GPU_LOCK_WAIT_RAIL')
                    print('LT1 waiting for shared GPU lock',flush=True);time.sleep(5)
            acquired=time.monotonic()
            busy=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True)
            if busy.returncode or busy.stdout.strip():raise ValueError('GPU_NOT_IDLE: '+busy.stdout.strip())
            env=environment(registration['arms'][cell['arm']]['flags']);env['GRM_LT1_LEASE_PARENT']=str(os.getpid())
            with (directory/'worker.log').open('x') as stream:
                child=subprocess.Popen([sys.executable,'-m','scripts.grm_lt1_worker','--worker',cell['id']],
                    cwd=lt.ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                # Foreground wait; interruptions do not release someone else's
                # compute lease while our child still runs, and never kill it.
                interrupted=False
                while True:
                    try:code=child.wait();break
                    except KeyboardInterrupt:interrupted=True
                charged=time.monotonic()-acquired
            if interrupted:raise ValueError('CONTROLLER_INTERRUPTED')
            if code:raise ValueError('WORKER_EXIT_'+str(code))
            if charged>285:raise ValueError('LEASE_DEADLINE_OVERRUN')
            lt.validate_checkpoint(directory/'checkpoint',cell['stop']+1,bind(cell['arm']))
            if not (directory/'worker.json').is_file():raise ValueError('MISSING_WORKER_RECEIPT')
            if time.monotonic()-outer+30>590:raise ValueError('OUTER_DEADLINE_OVERRUN')
            status='COMPLETE'
    except BaseException as exc:error=f'{type(exc).__name__}: {exc}'
    finally:
        if acquired is not None:charged=max(charged,time.monotonic()-acquired)
        lt.create(directory/'controller.json',dict(status=status,error=error,charged_seconds=charged,
             cell=cell,binding=bind(cell['arm']),admission_rule=0,outer_seconds_before_cooldown=time.monotonic()-outer))
        time.sleep(30)
    return status=='COMPLETE'


def resume():
    r=lt.verify();p=lt.preflight()
    if p['status']!='READY':raise ValueError(json.dumps(p))
    RUN.mkdir(parents=True,exist_ok=True)
    owner=RUN/'campaign.active';fd=os.open(owner,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        while True:
            cell=pending(r)
            if cell is None:return 0
            print('LT1 cell '+cell['id'],flush=True)
            if not run_cell(cell,r):return 2
    finally:owner.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--resume',action='store_true');args=p.parse_args()
    if args.worker:worker(next(c for c in lt.verify()['cells'] if c['id']==args.worker))
    elif args.resume:raise SystemExit(resume())
    else:p.error('use --resume or leased --worker')
