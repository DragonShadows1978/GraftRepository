#!/usr/bin/env python3
"""Create-only, fingerprinted, foreground lead dispatch.
Prior art: house DET1 leased runners (2026), util-linux flock system; reused
exclusive lease and content-addressed receipts. No new scheduling algorithm.
"""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.grm_x3_diagnostic import OUT,X3Error,create,sha,digest,validate_fixtures,cells,summarize
MODEL=Path('/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee')
NATIVE=Path('/mnt/ForgeRealm/GraftRepository/cpp/build/libgrm_runtime.so')
ENGINE=Path('/mnt/ForgeRealm/Project-Tensor/tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so')

def implementation():
    p=OUT/'implementation_manifest.json'
    if not p.exists():raise X3Error('implementation manifest missing; lead must not author new source pins')
    m=json.loads(p.read_text())
    for r in m['files']:
        if sha(ROOT/r['path'])!=r['sha256']:raise X3Error('implementation source drift: '+r['path'])
    return sha(p)

def runtime():
    reg,fixtures=validate_fixtures()
    inventory={p.name:{'blob':p.resolve().name,'size':p.stat().st_size} for p in sorted(MODEL.glob('*.safetensors'))}
    data={'registration_sha256':sha(OUT/'registration.json'),'fixtures_sha256':sha(OUT/'fixtures/manifest.json'),
        'implementation_sha256':implementation(),'model_id':reg['model_id'],'model_inventory':inventory,
        'native_sha256':sha(NATIVE) if NATIVE.is_file() else None,'engine_sha256':sha(ENGINE) if ENGINE.is_file() else None,
        'python':sys.version,'frame':reg['frame']}
    data['fingerprint']=digest(data);return data

def preflight():
    errors=[]
    for p in (NATIVE,ENGINE,MODEL/'tokenizer.json',MODEL/'model.safetensors.index.json'):
        if not p.is_file():errors.append('missing '+str(p))
    if (MODEL/'model.safetensors.index.json').exists():
        index=json.loads((MODEL/'model.safetensors.index.json').read_text())
        for name in sorted(set(index['weight_map'].values())):
            if not (MODEL/name).is_file():errors.append('missing weight shard '+name)
    for name in ('numpy','transformers','tokenizers'):
        if importlib.util.find_spec(name) is None:errors.append('missing Python package '+name)
    if not Path('/dev/nvidia0').exists():errors.append('NO GPU: /dev/nvidia0 absent in dispatched sandbox')
    return errors

def check_operator():
    # Foreground read-only check; a non-cooperating GPU user still has priority.
    if Path('/tmp/forge-gpu.operator').exists():raise X3Error('operator priority marker present')
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    if result.stdout.strip():raise X3Error('GPU occupied; operator right of way: '+result.stdout.strip())

def state_dirs():return sorted((OUT/'runs').glob('*/cell_??'))

def validate_worker(path,runtime_info):
    data=json.loads(path.read_text())
    if data['fingerprint']!=runtime_info['fingerprint']:raise X3Error('stale receipt fingerprint')
    import numpy as np
    from scripts.grm_x3_diagnostic import distribution_delta,load_capture,fork,verify_delta,outcome
    _,fixtures=validate_fixtures();fixture_map={f['id']:f for f in fixtures}
    for r in data['results']:
        if r['snapshot']['sha256']!=sha(ROOT/r['snapshot']['path']):raise X3Error('snapshot receipt drift')
        # Prior art: house DET1/X3 (2026). load_capture validates the manifest,
        # every blob and member coverage; a second full traversal is redundant.
        if r['donor_payload']['sha256']!=sha(ROOT/r['donor_payload']['path']):raise X3Error('donor receipt drift')
        with np.load(ROOT/r['donor_payload']['path'],allow_pickle=False) as archive:donor={k:archive[k] for k in archive.files}
        base=load_capture(ROOT/r['snapshot']['path'],{k:tuple(v) for k,v in r['spans'].items()})
        logits=np.load(ROOT/r['unforked_logits']['path'],allow_pickle=False)
        for arm,f in r['forks'].items():
            measured=distribution_delta(logits,np.load(ROOT/f['logits']['path'],allow_pickle=False))
            if any(f[k]!=v for k,v in measured.items()):raise X3Error('stored metric differs from raw logits')
            expected_pin=verify_delta(base,fork(base,arm,donor),arm,donor)
            if expected_pin!=f['delta_pin']:raise X3Error('fork delta fingerprint drift')
        expected_outcome=outcome(r['outcome']['answer'],fixture_map[r['id']],r['outcome']['truncated'])
        if expected_outcome!=r['outcome']:raise X3Error('answer scoring drift')
        for rec in [r['unforked_logits'],*[f['logits'] for f in r['forks'].values()]]:
            if rec['sha256']!=sha(ROOT/rec['path']):raise X3Error('logits receipt drift')
    return data

def summary(fingerprint=None):
    # Prior art: house DET1 content-addressed receipts and X3 (2026) runtime
    # pins. Select one recorded run, never merge environments or require the
    # historical worker source to equal this repaired reporting code. No prior
    # art known to me for this particular repair; no new experimental rule.
    reg,fixtures=validate_fixtures();rows=[];receipt_errors=[]
    runs=sorted(p for p in (OUT/'runs').glob('*') if p.is_dir())
    if fingerprint is None:
        if len(runs)!=1:raise X3Error('summary requires exactly one run or explicit fingerprint')
        run=runs[0]
    else:
        matches=[p for p in runs if p.name==fingerprint]
        if len(matches)!=1:raise X3Error('unknown summary run fingerprint')
        run=matches[0]
    fingerprint=run.name;runtime_info=None;sources=[]
    expected_cells={c['cell']:c['snapshot_ids'] for c in cells(fixtures)}
    unexpected={d.name for d in run.iterdir() if d.is_dir()}-set(expected_cells)
    if unexpected:raise X3Error('unexpected run cell directories: '+str(sorted(unexpected)))
    for cell,expected_ids in expected_cells.items():
        d=run/cell
        start=d/'started.json'
        if not start.exists():receipt_errors.append(str(d)+': missing start');continue
        started=json.loads(start.read_text())
        rt=started['runtime'];unsigned=dict(rt);unsigned.pop('fingerprint',None)
        if (started['cell']!=cell or started['fingerprint']!=fingerprint
                or rt['fingerprint']!=fingerprint or digest(unsigned)!=fingerprint):
            raise X3Error('recorded runtime fingerprint/cell drift')
        for key,path in (('registration_sha256',OUT/'registration.json'),
                         ('fixtures_sha256',OUT/'fixtures/manifest.json'),
                         ('implementation_sha256',OUT/'implementation_manifest.json')):
            if rt[key]!=sha(path):raise X3Error('historical input pin drift: '+key)
        if rt['model_id']!=reg['model_id'] or rt['frame']!=reg['frame']:raise X3Error('recorded model/frame drift')
        if runtime_info is not None and runtime_info!=rt:raise X3Error('mixed recorded runtimes')
        runtime_info=rt
        end=d/'receipt.json'
        if not end.exists():receipt_errors.append(str(d)+': abandoned/incomplete start; budget consumed');continue
        receipt=json.loads(end.read_text())
        if receipt['cell']!=cell or receipt['fingerprint']!=fingerprint:raise X3Error('cell receipt identity drift')
        if receipt['status']!='COMPLETE':receipt_errors.append(str(d)+': '+receipt['status']);continue
        worker=d/'worker_result.json'
        if receipt.get('worker_result_sha256')!=sha(worker):raise X3Error('worker result receipt drift')
        data=validate_worker(worker,rt)
        if data['cell']!=cell or [r['id'] for r in data['results']]!=expected_ids:
            raise X3Error('worker cell/snapshot allocation drift')
        for r in data['results']:
            result_path=d/r['id']/'result.json'
            if json.loads(result_path.read_text())!=r:raise X3Error('per-snapshot result differs from worker')
            fixture=next(f for f in fixtures if f['id']==r['id'])
            if r['intended_group']!=fixture['intended_group']:raise X3Error('result stratum drift')
            sources.append({'path':str(result_path.relative_to(ROOT)),'sha256':sha(result_path)})
        sources.extend({'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in (start,end,worker))
        rows.extend(data['results'])
    result=summarize(rows,reg);result['receipt_errors']=receipt_errors
    result['run_fingerprint']=fingerprint;result['recorded_runtime']=runtime_info
    result['source_receipts']=sources
    result['summary_sources']=[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
        for p in (ROOT/'scripts/grm_x3_lead.py',ROOT/'scripts/grm_x3_diagnostic.py')]
    if receipt_errors:result['status']='RED_RECEIPTS'
    return result

def choose_cell(command,requested=None):
    valid=[f'cell_{i:02d}' for i in range(4)]
    begun={d.name for d in state_dirs() if (d/'started.json').exists()}
    # Run/resume stop on any failed or interrupted GPU cell; nothing is retried.
    for d in state_dirs():
        if (d/'started.json').exists():
            if not (d/'receipt.json').exists() or json.loads((d/'receipt.json').read_text())['status']!='COMPLETE':
                raise X3Error('prior failed/abandoned start: stop at registered rail '+str(d))
    if command=='resume':return next((c for c in valid if c not in begun),None)
    if requested not in valid:raise X3Error('unknown GPU cell')
    if requested in begun:raise X3Error('cell already started; no retries or overwrites')
    if len(begun)>=4:raise X3Error('four-launch GPU budget exhausted')
    return requested

def leased(cell,expected):
    # Require shell-owned inherited lease; direct hidden worker calls fail closed.
    import fcntl
    if os.environ.get('GRM_X3_LEASED')!='1':raise X3Error('use lead shell runner')
    fd=9;st=os.fstat(fd);lock=Path('/tmp/forge-gpu.lock').stat()
    if (st.st_dev,st.st_ino)!=(lock.st_dev,lock.st_ino):raise X3Error('wrong lease descriptor')
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rt=runtime()
    if rt['fingerprint']!=expected:raise X3Error('runtime changed while waiting for lease')
    choose_cell('run',cell)
    errors=preflight()
    if errors:raise X3Error('; '.join(errors))
    check_operator()
    d=OUT/'runs'/rt['fingerprint']/cell;d.mkdir(parents=True,exist_ok=True)
    create(d/'started.json',{'cell':cell,'fingerprint':rt['fingerprint'],'started_unix':time.time(),
        'reserved_gpu_seconds':285,'evidence_class':'reasoning; budget reservation, not measurement','runtime':rt})
    start=time.monotonic();status='RED_FAILURE';failure=None
    try:
        from scripts.grm_x3_gpu import main as gpu_main
        gpu_main(cell,d,rt);status='COMPLETE'
    except BaseException:
        failure=traceback.format_exc();print(failure,file=sys.stderr)
    finally:
        import signal
        signal.setitimer(signal.ITIMER_REAL,0)
        elapsed=time.monotonic()-start
        if elapsed>285:status='RED_WORKER_OVERRUN'
        receipt={'schema':'grm.x3.cell_receipt.v1','evidence_class':'E2E session receipt','cell':cell,
            'fingerprint':rt['fingerprint'],'status':status,'elapsed_worker_seconds':elapsed,'failure':failure,
            'worker_result_sha256':sha(d/'worker_result.json') if (d/'worker_result.json').exists() else None,
            'process_policy':'no signals to other processes, no kill, foreground; Python alarm raises in own worker',
            'cooldown_seconds':30}
        create(d/'receipt.json',receipt)
        # Foreground cooldown, still holding GPU lease; no background wait.
        time.sleep(30)
    return 0 if status=='COMPLETE' else 1

def main():
    args=sys.argv[1:]
    if not args:raise X3Error('usage: list | run CELL | resume | summary | preflight')
    command=args[0]
    if command=='_leased':return leased(args[1],args[2])
    reg,fixtures=validate_fixtures()
    if command in ('list','--dry-run'):
        print(json.dumps(cells(fixtures),indent=2));return 0
    if command=='summary':
        if len(args)>2:raise X3Error('usage: summary [RUN_FINGERPRINT]')
        result=summary(args[1] if len(args)==2 else None);p=OUT/'summaries'/f'{digest(result)}.json'
        if p.exists():
            if json.loads(p.read_text())!=result:raise X3Error('summary address collision')
        else:create(p,result)
        print(json.dumps(result,indent=2));return 0
    rt=runtime()
    if command=='preflight':
        errors=preflight();print(json.dumps({'status':'BLOCKED' if errors else 'READY','errors':errors,'runtime':rt},indent=2));return 2 if errors else 0
    if command not in ('run','resume'):raise X3Error('unknown command')
    cell=choose_cell(command,args[1] if len(args)>1 else None)
    if cell is None:print('All four cells already complete.');return 0
    errors=preflight()
    if errors:raise X3Error('; '.join(errors))
    # One cell per invocation, including resume: worst-case 240 lock + 285
    # worker + 30 cooldown = 555 s, leaving 35 s setup before outer rail.
    start=time.monotonic()
    result=subprocess.run(['bash',str(ROOT/'scripts/grm_x3_lead_gpu.sh'),'_lease',cell,rt['fingerprint']],check=False)
    if time.monotonic()-start>590:raise X3Error('RED: outer 590-second rail exceeded')
    return result.returncode

if __name__=='__main__':
    try:raise SystemExit(main())
    except (X3Error,FileNotFoundError) as exc:print('RED/BLOCKED: '+str(exc),file=sys.stderr);raise SystemExit(2)
