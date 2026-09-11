#!/usr/bin/env python3
"""Lead-only X3 worker over captured live attempts; diagnostic code only.
Prior art: house DET1 capture/hydration and RS3 seating (2026), reused verbatim;
Meng et al. (2022, arXiv:2202.05262) motivates activation interventions. X3 adds
same-length donor and column-mask lesions, not model weight edits.
"""
from __future__ import annotations
import gc
import inspect
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.grm_x3_r2_diagnostic import (ARMS,OUT,X3Error,canonical,create,sha,digest,freeze_array,
    load_capture,fork,verify_delta,distribution_delta,outcome,validate_fixtures)
MODEL=Path('/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee')
NATIVE=Path('/mnt/ForgeRealm/GraftRepository/cpp/build/libgrm_runtime.so')
START=time.monotonic()

def deadline(*unused):
    raise TimeoutError('GRM-X3 worker 285-second rail reached; no retry authorized')

def check_time():
    if time.monotonic()-START>=280:deadline()

def process_instance():
    from scripts.grm_det1_3_snapshot import process_instance_sha256
    pid=os.getpid();s=Path(f'/proc/{pid}/stat').read_text();ticks=int(s[s.rfind(')')+2:].split()[19])
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    return {'process_pid':pid,'process_start_ticks':ticks,'boot_id':boot,'process_instance_sha256':process_instance_sha256(pid,ticks,boot)}

def save_array(path,array):
    with Path(path).open('xb') as f:np.save(f,np.asarray(array),allow_pickle=False)
    return {'path':str(Path(path).relative_to(ROOT)),'sha256':sha(path)}

def formatted(e2e,text):
    u,a=text[6:].split('\nAssistant: ',1);return e2e.harmony_turn(u,a)

def identity(model,e2e,runtime):
    import tensor_cuda as tc
    # Prior art: DET1 frame identity (2026); model weights are identified by
    # HF blob SHA names + sizes, not falsely reported as locally rehashed bytes.
    inventory={p.name:{'resolved_blob':p.resolve().name,'bytes':p.stat().st_size} for p in sorted(MODEL.glob('*.safetensors'))}
    if not inventory:raise X3Error('missing complete model files')
    engine=Path(getattr(tc,'_C',tc).__file__).resolve()
    return {'model_revision':MODEL.name,'model_weights_inventory_sha256':digest(inventory),
        'model_inventory_semantics':'HF cache blob names and byte sizes; weights not rehashed during bounded GPU lease',
        'tokenizer_sha256':sha(MODEL/'tokenizer.json'),'engine_build_sha256':sha(engine),
        'native_library_sha256':sha(NATIVE),'prompt_template_sha256':digest({'source':inspect.getsource(e2e.harmony_turn),'sink':e2e.HARMONY_SINK}),
        'dialect':'gqa','target_model_reasoning_effort':'low (Harmony prompt)','x3_runtime_fingerprint':runtime['fingerprint']}

def make_observer(arena,spans):
    from scripts.grm_det1_common import DetectorObserver
    class TargetObserver(DetectorObserver):
        def _full_mass(self,*args,**kwargs):
            # Reuse EXACT D-NGH arithmetic with its contiguous-band boundary
            # narrowed to this mount. Prior art: DET1 D-NGH (2026). Remaining
            # bands here are observer bookkeeping, not whole-arena categories.
            lo,hi=spans['target']
            proxy=SimpleNamespace(_gpt=self._gpt,arena=SimpleNamespace(n_sink=lo,cur_mount_n=hi-lo))
            result=DetectorObserver._full_mass(proxy,*args,**kwargs)
            a,b=spans['sham']
            proxy.arena=SimpleNamespace(n_sink=a,cur_mount_n=b-a)
            self._x3_sham_pending=getattr(self,'_x3_sham_pending',[])
            self._x3_sham_pending.append(DetectorObserver._full_mass(proxy,*args,**kwargs)['mounted_mass'])
            return result
        def _summarize_mass(self,rows):
            result=super()._summarize_mass(rows)
            pending=getattr(self,'_x3_sham_pending',[])
            self._validate_mass_count(pending)
            self.sham_masses=getattr(self,'sham_masses',[])
            self.sham_masses.append(float(np.mean(pending)))
            self._x3_sham_pending=[]
            return result
    return TargetObserver(arena=arena,ngen=40,logical_alias_ids=set(),active_detectors=frozenset({'D-NGH'}))

def hydrate(arena,path,base,branch,ident):
    from scripts.grm_det1_3_snapshot import restore_prefill_fork,_fork_upload,_fork_export
    m=json.loads(Path(path).read_text())
    receipt=restore_prefill_fork(arena,path,target_identity=ident,require_same_process_index=True,require_finalized_source=False)
    # ZERO restoration always comes first. Only fields changed by the pure
    # fork are uploaded next; metadata/index/routing is never re-run.
    for li,layer in enumerate(arena.m.layers):
        att=layer.self_attn
        att._det1_fork_allowed_mask=branch.arrays[f'mask.layer_{li:02d}.allowed'].copy()
        att._det1_fork_mask_consumed=False
        if base.active_prefix!='injection':raise X3Error('EB1 requires a bootstrap injection capture')
        vals=[]
        for name in ('k','v'):
            key=f'injection.layer_{li:02d}.{name}'
            value=_fork_upload(branch.arrays[key],m['arrays'][key]['source_dtype'])
            if _fork_export(value).tobytes()!=branch.arrays[key].tobytes():raise X3Error('device roundtrip changed canonical fork bytes')
            vals.append(value)
        att.inject_kv=(*vals,m['state'].get(f'injection.layer_{li:02d}.component_2',1.0))
    return receipt

def greedy(arena,ids,arm,spans,stops):
    from core import kv_graft
    from scripts.grm_det1_3_snapshot import verify_fork_masks_consumed
    check_time();first=arena._forward(ids);verify_fork_masks_consumed(arena);kv_graft.clear_injection(arena.m)
    out=[int(first.argmax())];ended=False
    for _ in range(40):
        text=arena.decode(out)
        if any(s in text for s in stops):ended=True;break
        if len(out)==40:break
        # Persist the mask intervention over decode, not merely prefill, so
        # exact-answer outcomes correspond to the same lesion throughout.
        for li,layer in enumerate(arena.m.layers):
            count=int(arena.caches[li][0].shape[2])+1
            window=layer.self_attn.sliding_window
            allowed=np.ones((1,1,1,count),dtype=np.uint8)
            if window is not None:allowed[...,:max(0,count-int(window))]=0
            if arm in ('remove','sham'):
                lo,hi=spans['sham' if arm=='sham' else 'target'];allowed[...,lo:hi]=0
            layer.self_attn._det1_fork_allowed_mask=allowed
            layer.self_attn._det1_fork_mask_consumed=False
        check_time();row=arena._forward([out[-1]]);verify_fork_masks_consumed(arena);out.append(int(row.argmax()))
    text=arena.decode(out)
    for stop in stops:text=text.split(stop)[0]
    return first,text.strip(),not ended,out

def prepare_attempt(arena,picks):
    # Prior art: RS2/RS3 _clear_boat (2026), production EB1 turn-open geometry.
    # Direct _attempt bypasses step's live_shift assignment; preserve that
    # assignment explicitly or the target query RoPE origin would be wrong.
    from scripts.grm_rs2_mount_read_gpu import _clear_boat
    if sum(int(arena.grafts[i]['ntok']) for i in picks)>arena.width:
        raise X3Error('actual graft rows exceed arena width')
    _clear_boat(arena)
    seat_order,seat=arena._rs3_seat_plan(picks)
    if not seat['seat_near_live']:raise X3Error('seat near live declined')
    return seat_order,seat

def run_fixture(model,tokenizer,e2e,fixture,directory,runtime,ident,reg):
    from core.graft_repository import GraftRepository
    from core.gpt_oss20b_tc import gpt_oss_grm_dialect_kwargs
    from core import kv_graft
    from scripts.grm_det1_3_gpu import _install_lived_nodes
    from scripts.grm_det1_3_snapshot import capture_arena_snapshot,load_snapshot
    check_time();directory.mkdir(parents=True,exist_ok=False)
    for layer in model.layers:
        att=layer.self_attn;att.inject_kv=None;att.live_shift=None;att.graft_seats=0
        att._det1_fork_allowed_mask=None;att._det1_fork_mask_consumed=False
    dialect=gpt_oss_grm_dialect_kwargs(model.config)
    repo=GraftRepository(model,lambda t:tokenizer.encode(t,add_special_tokens=False),
        lambda ids:tokenizer.decode(ids,clean_up_tokenization_spaces=False),str(directory/'repository'),
        autosave=False,arena_cls=e2e.GptOssGQAArenaCache,native_lib_path=str(NATIVE),native_auto=False,
        vram_budget_mb=None,route_layer=int(dialect['route_layer']),arena_width=96,topk=3,live_turns=2,max_live=4096,
        sink_text=e2e.HARMONY_SINK,prompt_template=e2e.harmony_turn,stop_sequences=e2e.HARMONY_STOPS,
        storage_bits=8,revision_resolution=True,decisive_admission=True)
    arena=repo.arena
    try:
        if not arena.ephemeral:raise X3Error('non-EB1 arena')
        node_map,_=_install_lived_nodes(repo,e2e,{'nodes':fixture['battery']['nodes']})
        def deposit(text):
            check_time();idx=int(arena.deposit(text,capture_pin='live'));repo._native_sync_node(idx);return idx
        base_text=formatted(e2e,fixture['base_text']);swap_text=formatted(e2e,fixture['swap_text'])
        if arena.encode(base_text)!=fixture['base_token_ids'] or arena.encode(swap_text)!=fixture['swap_token_ids'] or arena.encode(fixture['sham_text'])!=fixture['sham_token_ids']:raise X3Error('frozen tokenization mismatch')
        target=node_map[fixture['battery']['target_node']] if fixture['intended_group']=='correct' else deposit(base_text)
        donor=deposit(swap_text);sham=deposit(fixture['sham_text']);picks=[target,sham]
        arena._ensure_h([target,donor,sham])
        if arena.grafts[target]['ntok']!=arena.grafts[donor]['ntok']:raise X3Error('donor row count mismatch')
        if arena.caches is not None or arena.live_segs:raise X3Error('EB1 feed left persistent live state')
        seat_order,seat=prepare_attempt(arena,picks)
        spans={};offset=arena.n_sink
        for idx in seat_order:
            n=int(arena.grafts[idx]['ntok']);spans['target' if idx==target else 'sham']=(offset,offset+n);offset+=n
        donor_h=[]
        for block in arena.grafts[donor]['h']:
            donor_h.append({k:np.ascontiguousarray(v if isinstance(v,np.ndarray) else v.numpy()) for k,v in block.items()})
        # RS3 rotation is applied to donor rows by the SAME constant delta;
        # no positional recapture, repacking, padding or truncation is allowed.
        replacement={}
        for li,block in enumerate(donor_h):
            for k in ('k','v'):
                arr=arena._rs3_rotate_rows(block[k],2,int(seat['delta_positions'])) if k in arena.ROPE_KEYS else block[k]
                replacement[f'injection.layer_{li:02d}.{k}']=np.ascontiguousarray(arr)
        proc=process_instance();question=fixture['question']
        provenance={'schema':'grm.det1_3.capture_provenance.v2','arm':'lived',
            'protocol':'same_model_chronological_harmony_feed_v1','protocol_source_sha256':sha(Path(__file__)),
            'run_id':runtime['fingerprint'],'order_sha256':reg['order']['sha256'],
            'source_amendment_sha256':sha(OUT/'implementation_manifest.json'),
            'registration_sha256':sha(OUT/'registration.json'),'runtime_frame_sha256':digest(reg['frame']),
            'fixture_sha256':sha(OUT/'fixtures'/f'{fixture["id"]}.json'),'fixture_id':fixture['id'],
            'probe_id':fixture['battery']['key'],'probe_question_sha256':__import__('hashlib').sha256(question.encode()).hexdigest(),
            'capture_boundary':'before_probe_prefill','probe_driver':'X3 controlled direct _attempt after lived feeds; diagnostic',
            'topk':3,'max_trips':0,'probe_ladder':False,'defer_memory':False,'attempt_ordinal':0,
            'selection_boundary':'before_generation_and_grounding_selection',**proc}
        admission={'schema':'grm.x3.controlled_mount_plan.v1','probe_key_sha256':digest({'question':question,'route_not_recomputed':True}),
            'ranking':picks,'identifier_tokens':[],'identified_candidates':picks,'policy_branch':'X3_REGISTERED_CONTROLLED_PICKS_NOT_ROUTED',
            'rank_plan':picks,'ladder_attempts':[{'ordinal':0,'planned':picks,'clean_room':True}],
            'current_attempt_ordinal':0,'current_planned':picks,'current_clean_room':True,'current_fitted':picks,'current_dropped':[],
            'selection_state':'PENDING_AT_PREFILL','final_mounts':picks}
        holder={}
        observer=make_observer(arena,spans)
        with observer:
            original=arena._forward
            def capture(ids,*args,**kwargs):
                check_time()
                if not holder:
                    path=capture_arena_snapshot(arena,directory/'snapshot',label='lived',provenance=provenance,
                        question=question,prompt_ids=ids,admission_plan=admission,live_token_ids=[],
                        sink_text=e2e.HARMONY_SINK,sink_token_ids=arena.encode(e2e.HARMONY_SINK),explicit_identity=ident)
                    # Seal before baseline measurement and all fork gates.
                    holder.update(path=path,ids=list(ids),snapshot_sha256=sha(path))
                    if not load_snapshot(path)['complete']:raise X3Error('captured snapshot incomplete')
                row=original(ids,*args,**kwargs)
                if 'first' not in holder:holder['first']=np.asarray(row).copy()
                return row
            arena._forward=capture
            try:answer,info=arena._attempt(question,picks,40,False,e2e.HARMONY_STOPS,defer_memory=False)
            finally:arena._forward=original
        if arena.cur_mounts!=picks:raise X3Error('production revised controlled mount identities')
        base=load_capture(holder['path'],spans)
        if base.active_prefix!='injection':raise X3Error('expected EB1 cold bootstrap')
        # Donor upload uses capture source dtype; canonicalize exactly once,
        # before any comparison, to the same export representation as source.
        from scripts.grm_det1_3_snapshot import _fork_upload,_fork_export
        manifest=load_snapshot(holder['path'])
        replacement={k:_fork_export(_fork_upload(v,manifest['arrays'][k]['source_dtype'])) for k,v in replacement.items()}
        with (directory/'donor_payload.npz').open('xb') as f:np.savez(f,**replacement)
        donor_receipt={'path':str((directory/'donor_payload.npz').relative_to(ROOT)),'sha256':sha(directory/'donor_payload.npz')}
        base_logits=holder['first'];base_out=outcome(answer,fixture,truncated=len(observer.records)>=41)
        first_record=observer.records[0];mass=float(first_record['ngh']['mounted_mass'])
        result={'id':fixture['id'],'intended_group':fixture['intended_group'],'evidence_class':'E2E session receipt',
            'session_scope':'controlled captured attempt, not full natural battery acceptance','outcome':base_out,'mass':mass,
            'sham_mass':observer.sham_masses[0],'donor_payload':donor_receipt,
            'mass_telemetry_first_position':first_record,'snapshot':{'path':str(holder['path'].relative_to(ROOT)),'sha256':holder['snapshot_sha256']},
            'spans':spans,'seat':seat,'unforked_logits':save_array(directory/'unforked.npy',base_logits),'forks':{}}
        for arm in ARMS:
            check_time();branch=fork(base,arm,replacement);delta=verify_delta(base,branch,arm,replacement)
            hydrate_receipt=hydrate(arena,holder['path'],base,branch,ident)
            with make_observer(arena,spans) as obs:
                logits,text,truncated,ids=greedy(arena,holder['ids'],arm,spans,e2e.HARMONY_STOPS)
            result['forks'][arm]={**distribution_delta(base_logits,logits),'outcome':outcome(text,fixture,truncated),
                'mass_on_target':float(obs.records[0]['ngh']['mounted_mass']),
                'mass_on_lesioned_mount':obs.sham_masses[0] if arm=='sham' else float(obs.records[0]['ngh']['mounted_mass']),
                'base_mass_on_lesioned_mount':observer.sham_masses[0] if arm=='sham' else mass,'tokens':ids,'delta_pin':delta,
                'zero_hydration':{'same_process_index_verified':hydrate_receipt['same_process_index_verified'],'source_manifest_sha256':hydrate_receipt['source_manifest_sha256']},
                'logits':save_array(directory/f'{arm}.npy',logits)}
        if sha(holder['path'])!=holder['snapshot_sha256'] or base.fingerprint()!=result['forks']['zero']['delta_pin']['base_fingerprint']:raise X3Error('source mutated')
        create(directory/'result.json',result)
        return result
    finally:
        kv_graft.clear_injection(model);repo.close();del arena,repo;gc.collect()

def main(cell,run_dir,runtime):
    global START
    START=time.monotonic();signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,285)
    reg,fixtures=validate_fixtures()
    for k,v in reg['frame'].items():
        if k.startswith('GRM_'):os.environ[k]=v
    os.environ['TOKENIZERS_PARALLELISM']='false'
    sys.path.insert(0,'/mnt/ForgeRealm/Project-Tensor/tensor_cuda')
    from core.gpt_oss20b_tc import GptOss20B_TC
    from transformers import AutoTokenizer
    from scripts import grm_e2e_session as e2e
    check_time();model,model_info=GptOss20B_TC.from_pretrained(MODEL)
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    ident=identity(model,e2e,runtime);results=[]
    index=int(cell.split('_')[1]);directory=Path(run_dir)
    for fixture in fixtures[index*5:(index+1)*5]:
        results.append(run_fixture(model,tokenizer,e2e,fixture,directory/fixture['id'],runtime,ident,reg))
    create(directory/'worker_result.json',{'schema':'grm.x3.worker.v1','evidence_class':'E2E session receipt',
        'cell':cell,'fingerprint':runtime['fingerprint'],'model_id':reg['model_id'],'model_reasoning_effort':'low (frozen Harmony sink)',
        'identity':ident,'model_info':model_info,'results':results,'elapsed_seconds':time.monotonic()-START})
    signal.setitimer(signal.ITIMER_REAL,0)
