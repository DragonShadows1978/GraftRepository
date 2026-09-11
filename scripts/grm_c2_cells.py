#!/usr/bin/env python3
"""C2 additive cells, foreground lease controller and new-process replay.

Prior art: local DET1, EB1/WC1/RS3/RT1.1 and SCOUT-FIX-1 (contributors, 2026).
Import their battery, serving, comparator and manifest methods unchanged. Ours:
pre-probe persistence sidecars and paired replay receipts. Checkpoint/restart is
established systems practice; no new algorithm or novelty claim. No prior art
known to me for this specific glue beyond those local systems.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.grm_c2_profile import (OUT, ROOT, REGISTRY, REGISTRATION, PROFILE_ID,
    RT1_FIELDS, create, read, sha, select_profile, verify_registration,
    capture_grade, token_seats, compare)


def sources():
    # All local executable Python/C++ and config inputs, plus loaded modules,
    # are fingerprinted in every cell, including failures and blocked receipts.
    paths = [p for d in ('core','scripts','cpp','config') for p in (ROOT/d).rglob('*')
             if p.is_file() and p.suffix in ('.py','.cpp','.hpp','.h','.json')]
    paths += [Path(getattr(m,'__file__','')) for m in list(sys.modules.values())
              if getattr(m,'__file__',None) and Path(m.__file__).is_file()]
    return {str(p):sha(p) for p in sorted(set(paths))}


def cell_by_id(registration, cell_id):
    return next(c for c in registration['cells'] if c['id'] == cell_id)


def flags_for(side):
    r=read(REGISTRY)
    return select_profile(r['default_flags'], PROFILE_ID if side=='profile' else None)


def environment(flags):
    # Explicit experimental pin, separate from the pure opt-in loader. Remove
    # ambient GRM instrumentation/routing switches before pinning this frame.
    env={k:v for k,v in os.environ.items() if not k.startswith('GRM_')}
    keys={'GRM_PERSISTENT_BOAT':not flags['ephemeral'],
          'GRM_CAPTURE_PIN':flags['capture_pin'], 'GRM_SEAT_NEAR_LIVE':flags['seat_near_live'],
          'GRM_LSR_FIXES':flags['lsr_fixes'], 'GRM_RT1_RULE':flags['rt1_rule'],
          'GRM_DEMAND_NGH':flags['demand_ngh'],'GRM_GQA_CUDA_ROUTE':flags['gqa_cuda_route'],
          'GRM_GRAFT_STORAGE_BITS':flags['graft_storage_bits'],
          'GRM_ROUTE_QUERY_LEX':flags['route_query_lex'],
          'GRM_PROBE_LADDER':flags['probe_ladder'],'GRM_SUP_RESOLVE':flags['sup_resolve'],
          'GRM_ADM_DECISIVE':flags['adm_decisive']}
    env.update({k:str(int(v)) if isinstance(v,bool) else str(v) for k,v in keys.items()})
    return env


def args_for(e2e, session, flags):
    r=read(REGISTRY)
    args=e2e.parse_args(['--mode','full','--session-dir',str(session),
        '--model-dir',r['model']['path'], '--native-lib',
        '/mnt/ForgeRealm/GraftRepository/cpp/build/libgrm_runtime.so',
        '--arena-width',str(flags['arena_width']),'--ngen',str(flags['ngen']),
        '--max-trips',str(flags['max_trips']),'--live-turns',str(flags['live_turns']),
        '--max-live',str(flags['max_live']),'--topk',str(flags['topk']),
        '--turn-pipeline',flags['turn_pipeline'],'--restart-after','999',
        '--skip-gpu-idle-check','--probe-ladder','--sup-resolve','--adm-decisive'])
    return args


def manifest_projection(nodes):
    return [{k:n.get(k) for k in ('text','ntok','metadata','capture','retired','sources','provenance')}
            for n in nodes]


def tree_hashes(directory):
    return {str(p.relative_to(directory)):sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}


def snapshot(repo, target, context, process_id):
    repo.flush_now()
    shutil.copytree(repo.path, target/'repository')
    nodes=read(target/'repository/manifest.json')['nodes']
    record={'context':context,'process_id':process_id,'pid':os.getpid(),
            'manifest_projection':manifest_projection(nodes),
            'files':tree_hashes(target/'repository')}
    create(target/'checkpoint.json',record)
    return record


def assert_payloads(directory):
    # Prior art: B3 manifest and ordinary NPZ durability (project,2026).
    # Missing durable payload is a RED input, never a cue to recapture text.
    nodes=read(directory/'manifest.json')['nodes']
    for i,n in enumerate(nodes):
        if not n.get('retired') and (n.get('payload_pending') or
                                    not (directory/f'nodes/{i:04d}.npz').is_file()):
            raise ValueError(f'INHERITED_MISSING_PAYLOAD node={i}; reharvest forbidden')
    return nodes


def validate_checkpoint(path):
    record=read(path/'checkpoint.json')
    if tree_hashes(path/'repository') != record['files']:
        raise ValueError(f'checkpoint content changed: {path}')
    assert_payloads(path/'repository')
    return record


def strict_capture_grade(nodes, side, width, n_sink):
    # Prior art: RS3 deposit_from_cache (project, 2026) explicitly distinguishes
    # live-cache payload slices from a pinned harvest. Preserve that distinction;
    # C2 adds fail-closed schema/geometry checks, not payload recapture.
    result=capture_grade(nodes,side,width,n_sink)
    required=set(read(REGISTRY)['payload_provenance']['required'])
    for node,row in zip(nodes,result['rows']):
        capture=node.get('capture') or {}
        if capture and (not required <= capture.keys() or
                        capture.get('arena_width')!=width or
                        capture.get('n_sink')!=n_sink or
                        capture.get('live_shift')!=n_sink+width):
            row['grade']='INCOMPLETE_OR_WRONG_CAPTURE_GEOMETRY'
        elif capture.get('capture_payload_source')=='live_cache_slice':
            span=capture.get('capture_span_pos0')
            if capture.get('capture_pin_moves_payload') is not False or not isinstance(span,int) or span<n_sink+width:
                row['grade']='INVALID_CACHE_SLICE_PROVENANCE'
            elif row['grade'] in ('FRESH_LIVE_PINNED','FRESH_UNPINNED'):
                row['grade']='FRESH_CACHE_SLICE_PAYLOAD_NOT_MOVED_BY_PIN'
    allowed={'FRESH_LIVE_PINNED'} if side=='profile' else {'FRESH_UNPINNED'}
    allowed.add('FRESH_CACHE_SLICE_PAYLOAD_NOT_MOVED_BY_PIN')
    result['valid']=bool(result['rows']) and all(row['grade'] in allowed for row in result['rows'])
    return result


def observe(repo, info, cell, probe_id, checkpoint, process_id):
    from core.grm_three_pass import build_route_receipt
    arena=repo.arena
    route=build_route_receipt(session_id=cell['id'],turn_id=probe_id,info=info,arena=arena)
    nodes=checkpoint['manifest_projection']
    grade=strict_capture_grade(nodes,cell['side'],arena.width,arena.n_sink)
    mounts=list(arena.cur_mounts)
    return {'battery':cell['battery'],'side':cell['side'],'phase':cell['phase'],
            'probe_id':probe_id,'mounted_ids':mounts,'token_seats':token_seats(arena.grafts,mounts),
            'arena_cur_mount_n':int(arena.cur_mount_n),
            'rt1':{k:info[k] for k in RT1_FIELDS if k in info},'route_receipt':route,
            'capture':grade,'capture_valid':grade['valid'],
            'new_process':process_id != checkpoint['process_id'] and os.getpid()!=checkpoint['pid'],
            'metadata_retained':None,'geometry':{'n_sink':arena.n_sink,'arena_width':arena.width,'live_shift':arena.live_shift},
            'frame_ephemeral':arena.ephemeral}


def score_probe(repo, e2e, cell, context, checkpoint, process_id, session):
    from scripts.lsr_p2c_replay_gpu import answer_verdict
    flags=flags_for(cell['side'])
    pid=context['probe_id']
    if cell['battery']=='sup':
        probe=context['probe']
        answer,info=e2e._probe_ladder_chat(repo,probe['question'],topk=flags['topk'],
            ngen=flags['ngen'],max_trips=flags['max_trips'],defer_memory=True)
        row=observe(repo,info,cell,pid,checkpoint,process_id)
        verdict=answer_verdict(answer,expected_values=probe['expected_values'],rejected_values=probe['rejected_values'])
    else:
        # Use the same production run_turn and ladder; observe the served band
        # before pass3/deposit can change it. No routing or scoring substitution.
        captured=[]
        original=e2e._probe_ladder_chat
        def traced(*args,**kwargs):
            answer,info=original(*args,**kwargs)
            captured.append(observe(repo,info,cell,pid,checkpoint,process_id))
            return answer,info
        e2e._probe_ladder_chat=traced
        rows=[]
        try:
            e2e.run_turn(repo,context['event'],context['turn'],paths=e2e.stage_paths(session),
                transcript=copy.deepcopy(context['transcript']),
                turn_records={int(k):v for k,v in context['turn_records'].items()},
                probe_rows=rows,args=args_for(e2e,session,flags),resumed=cell['phase']=='restart')
        finally:
            e2e._probe_ladder_chat=original
        if len(captured)!=1 or len(rows)!=1:
            raise ValueError(f'expected one served probe trace, got {len(captured)}/{len(rows)}')
        row=captured[0]; answer=rows[0]['answer']
        # EB1/WC1 use contains_value on the frozen expected value, not driver pass.
        verdict=answer_verdict(answer,expected_values=[context['event']['expected']],rejected_values=[])
    row.update(served_answer=answer,correct=verdict['correct'],verdict=verdict)
    return row


def worker(cell):
    if os.environ.get('GRM_C2_LEASE_PARENT') != str(os.getppid()):
        raise RuntimeError('worker requires its foreground leased parent')
    from scripts import grm_e2e_session as e2e
    from scripts.grm_det1_3_gpu import _install_lived_nodes
    from scripts import lsr_p2c_replay_gpu as sup
    from core.graft_repository import GraftRepository
    registration=verify_registration(); registry=read(REGISTRY)
    flags=flags_for(cell['side']); session=OUT/'cells'/cell['id']/'session'
    session.parent.mkdir(parents=True,exist_ok=True)
    if session.exists(): raise ValueError('existing session: retry forbidden')
    process_id=str(uuid.uuid4()); rows=[]; retained=[]
    checkpoints=OUT/'checkpoints'/cell['side']/cell['battery']
    # Load model once per worker, then use ordinary repository constructors for
    # checkpoint reloads. Capture original constructor arguments, not a port.
    constructor=[]; original_ctor=e2e.GraftRepository
    def recording_ctor(*a,**kw):
        constructor.append((a,kw)); return original_ctor(*a,**kw)
    e2e.GraftRepository=recording_ctor
    model=tokenizer=repo=None
    try:
        if cell['phase']=='restart':
            contexts=[]
            for pid in cell['probes']:
                source=checkpoints/pid; checkpoint=validate_checkpoint(source)
                contexts.append((pid,source,checkpoint))
            for ordinal,(pid,source,checkpoint) in enumerate(contexts):
                dest=session/pid
                shutil.copytree(source/'repository',dest/'repository')
                if ordinal==0:
                    model,tokenizer,repo,_=e2e.load_model_and_repo(args_for(e2e,dest,flags),dest)
                else:
                    a,kw=constructor[0]; a=list(a); a[3]=str(dest/'repository')
                    repo=GraftRepository(*a,**kw)
                loaded=manifest_projection([repo._node_manifest(g) for g in repo.arena.grafts])
                same=loaded==checkpoint['manifest_projection']
                row=score_probe(repo,e2e,cell,checkpoint['context'],checkpoint,process_id,dest)
                row['metadata_retained']=same; rows.append(row)
                repo.close(); repo=None
        else:
            if cell.get('depends'):
                prior=OUT/'cells'/cell['depends'][-1]/'session'
                shutil.copytree(prior,session)
            else: session.mkdir()
            if cell.get('depends'):
                prior_receipt=read(OUT/'cells'/cell['depends'][-1]/'worker.json')
                if tree_hashes(session/'repository')!=prior_receipt['session_repository_sha256']:
                    raise ValueError('persisted continuation input changed')
                assert_payloads(session/'repository')
            model,tokenizer,repo,_=e2e.load_model_and_repo(args_for(e2e,session,flags),session)
            if repo.arena.width!=flags['arena_width'] or repo.arena.n_sink!=registry['frame']['n_sink'] or not repo.arena.ephemeral:
                raise ValueError('observed frame disagrees with registry')
            if cell['battery']=='sup':
                fixture=read(sup.SUP_FIXTURES/f'{cell["session"]}.json')
                _install_lived_nodes(repo,e2e,fixture)
                for probe in registry['sup_plans'][cell['session']]:
                    context={'probe_id':probe['probe_id'],'probe':probe}
                    cp=snapshot(repo,checkpoints/probe['probe_id'],context,process_id)
                    rows.append(score_probe(repo,e2e,cell,context,cp,process_id,session))
            else:
                statepath=session/'c2_state.json'
                state=read(statepath) if statepath.exists() else {'transcript':[],'turn_records':{},'probe_rows':[],'next_turn':0}
                if state['next_turn']!=cell['start']:raise ValueError('segment dependency state mismatch')
                transcript=state['transcript']; records={int(k):v for k,v in state['turn_records'].items()}; probes=state['probe_rows']
                for turn in range(cell['start'],cell['stop']):
                    event=registry['session_script'][turn]
                    if event['kind']=='probe':
                        pid=(f'lh_t{turn:03d}' if cell['battery']=='longhistory' else
                             next(p['probe_id'] for p in registry['census_probes'] if p['turn']==turn))
                        context={'probe_id':pid,'event':event,'turn':turn,'transcript':transcript,'turn_records':records}
                        cp=snapshot(repo,checkpoints/pid,context,process_id)
                        # Trace the original run while retaining its mutable session
                        # bookkeeping; score_probe's replay copies are restart-only.
                        original=e2e._probe_ladder_chat; captured=[]
                        def traced(*a,**kw):
                            answer,info=original(*a,**kw)
                            captured.append(observe(repo,info,cell,pid,cp,process_id))
                            return answer,info
                        e2e._probe_ladder_chat=traced
                    try:
                        e2e.run_turn(repo,event,turn,paths=e2e.stage_paths(session),transcript=transcript,
                            turn_records=records,probe_rows=probes,args=args_for(e2e,session,flags),resumed=cell['start']>0)
                    finally:
                        if event['kind']=='probe':e2e._probe_ladder_chat=original
                    if event['kind']=='probe':
                        from scripts.lsr_p2c_replay_gpu import answer_verdict
                        if len(captured)!=1:raise ValueError('missing or duplicate trace')
                        row=captured[0]; answer=probes[-1]['answer']
                        verdict=answer_verdict(answer,expected_values=[event['expected']],rejected_values=[])
                        row.update(served_answer=answer,correct=verdict['correct'],verdict=verdict); rows.append(row)
                # Existing copied continuation state is a working session, not a
                # registration or receipt; updating it is normal persistence.
                statepath.write_text(json.dumps({'transcript':transcript,'turn_records':records,'probe_rows':probes,'next_turn':cell['stop']},indent=2))
            repo.flush_now()
            end_nodes=read(session/'repository/manifest.json')['nodes']
            retained=strict_capture_grade(end_nodes,cell['side'],repo.arena.width,repo.arena.n_sink)
    finally:
        e2e.GraftRepository=original_ctor
        if repo is not None:repo.close()
    return {'cell':cell,'rows':rows,'end_capture':retained,'process_id':process_id,'pid':os.getpid(),
            'flags':flags,'registry_sha256':sha(REGISTRY),'registration_sha256':sha(REGISTRATION),
            'sources':sources(),'summed_token_seats':[r['token_seats'] for r in rows],
            'rt1_provenance':[r['rt1'] for r in rows],
            'session_repository_sha256':tree_hashes(session/'repository') if cell['phase']=='fresh' else None}


def dry_run():
    r=verify_registration()
    return {'status':'NON_FIT_BUDGET' if r['estimated_seconds']>r['budget_seconds'] else 'READY_FOR_LEAD',
            'gpu_executed':False,'budget_seconds':r['budget_seconds'],'estimated_seconds':r['estimated_seconds'],
            'worker_seconds':r['worker_seconds'],'outer_seconds':r['outer_seconds'],
            'cooldown_seconds':r['cooldown_seconds'],'cells':r['cells'],
            'registry_sha256':sha(REGISTRY),'registration_sha256':sha(REGISTRATION),
            'token_seats':None,'rt1_provenance':None,'measurement_status':'NOT_RUN',
            'sources':sources()}


def run_leased(cell):
    r=verify_registration()
    if r['estimated_seconds']>r['budget_seconds']:
        raise ValueError('NON_FIT_BUDGET: registered estimate 3140s > 2880s; separate lead amendment required')
    for dep in cell['depends']:
        if read(OUT/'cells'/dep/'controller.json')['status']!='COMPLETE':raise ValueError(f'dependency not complete: {dep}')
    directory=OUT/'cells'/cell['id']; directory.mkdir(parents=True,exist_ok=True)
    # A reservation remains charged at full cap if the controller dies.
    used=sum(read(p)['charged_seconds'] for p in (OUT/'cells').glob('*/controller.json'))
    reserved=sum(read(p)['seconds'] for p in (OUT/'cells').glob('*/reservation.json') if not p.with_name('controller.json').exists())
    if used+reserved+285>2880:raise ValueError('budget rail: cannot reserve next cell')
    create(directory/'reservation.json',{'seconds':285,'cell':cell['id']})
    from scripts.grm_cmc1_gpu_arms import gpu_lease
    status='FAILED'; elapsed=0; message=None
    try:
        with gpu_lease(285,240):
            started=time.monotonic()
            env=environment(flags_for(cell['side']));env['GRM_C2_LEASE_PARENT']=str(os.getpid())
            with (directory/'worker.log').open('x') as stream:
                result=subprocess.run([sys.executable,__file__,'--worker',cell['id']],cwd=ROOT,env=env,
                                      stdout=stream,stderr=subprocess.STDOUT,timeout=280)
            elapsed=time.monotonic()-started
            if result.returncode==0: status='COMPLETE'
            else: message=f'worker returncode={result.returncode}'
    except (Exception,BaseException) as exc:
        message=f'{type(exc).__name__}: {exc}'; elapsed=285
    finally:
        create(directory/'controller.json',{'status':status,'charged_seconds':elapsed,'error':message,
            'cell':cell,'sources':sources(),'registry_sha256':sha(REGISTRY),'token_seats':None,
            'rt1_provenance':None,'measurement_status':'see worker.json' if status=='COMPLETE' else 'NOT_COMPLETED'})
        time.sleep(30)  # foreground cooldown, no background waiter
    return 0 if status=='COMPLETE' else 1


def main():
    p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true')
    p.add_argument('--cell');p.add_argument('--worker');p.add_argument('--score',action='store_true')
    args=p.parse_args()
    if args.dry_run: print(json.dumps(dry_run(),indent=2));return 0
    if args.score:
        rows=[r for f in (OUT/'cells').glob('*/worker.json') for r in read(f)['rows']]
        result=compare(rows);print(json.dumps(result,indent=2));return 0 if result['adopt'] and result['fix_validation'] else 1
    r=verify_registration()
    if args.worker:
        cell=cell_by_id(r,args.worker)
        value=worker(cell);create(OUT/'cells'/cell['id']/'worker.json',value);return 0
    if args.cell:return run_leased(cell_by_id(r,args.cell))
    p.error('choose --dry-run, --cell, or --score')
if __name__=='__main__':raise SystemExit(main())
