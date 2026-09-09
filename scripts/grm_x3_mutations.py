#!/usr/bin/env python3
"""Bounded CPU mutation test; source is changed in a fresh process's memory only.
Prior art: house mutation-testing rule (2026), threshold >=0.80. The six concrete
X3 defects below are author-selected, not a blind adversary or novelty claim.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types
ROOT=Path(__file__).resolve().parents[1]
MUTANTS={
    'mask_wrong_span':('mask[...,lo:hi]=0;arrays[mask_key]','mask[...,0:hi-lo]=0;arrays[mask_key]'),
    'swap_noop':('dst[:,:,lo:hi,:]=donor;arrays[key]','dst[:,:,lo:hi,:]=src[:,:,lo:hi,:];arrays[key]'),
    'kl_reversed':('p,q=log_probs(base),log_probs(other)','p,q=log_probs(other),log_probs(base)'),
    'classifier_inverted':('eh=effect>1;err=', 'eh=effect<=1;err='),
    'scalar_check_missing':('if canonical(base.state)!=canonical(other.state) or base.spans!=other.spans or base.layers!=other.layers or base.active_prefix!=other.active_prefix:', 'if False:'),
    'receipt_overwrite':("with path.open('xb') as f:f.write(canonical(x))", "with path.open('wb') as f:f.write(canonical(x))")}

def run_child(name):
    sys.path.insert(0,str(ROOT))
    path=ROOT/'scripts/grm_x3_diagnostic.py';source=path.read_text();old,new=MUTANTS[name]
    if source.count(old)!=1:raise RuntimeError('mutant replacement did not match exactly once')
    module=types.ModuleType('scripts.grm_x3_diagnostic');module.__file__=str(path)
    sys.modules[module.__name__]=module
    exec(compile(source.replace(old,new),str(path)+':MUTANT:'+name,'exec'),module.__dict__)
    import pytest
    return pytest.main(['-q','-p','no:cacheprovider',str(ROOT/'tests/test_grm_x3_lesions.py')])

def main():
    if len(sys.argv)>1 and sys.argv[1]=='_mutant':return run_child(sys.argv[2])
    source=ROOT/'scripts/grm_x3_diagnostic.py';before=hashlib.sha256(source.read_bytes()).hexdigest()
    manifest=ROOT/'artifacts/grm_x3/mutation_registration.json'
    if not manifest.exists():raise RuntimeError('mutation registration must precede runs')
    reg=json.loads(manifest.read_text())
    if reg['mutants']!={k:list(v) for k,v in MUTANTS.items()} or reg['source_sha256']!=before:raise RuntimeError('mutation registration drift')
    records=[]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT))
    for name in MUTANTS:
        log=ROOT/'logs'/f'grm_x3_mutation_{name}.txt'
        start=time.monotonic()
        with log.open('x') as stream:
            child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'_mutant',name],env=env,stdout=stream,stderr=subprocess.STDOUT,check=False)
        # Pytest 1 means tests ran and found failures; 2+ is an invalid lane.
        records.append({'name':name,'exit_code':child.returncode,'killed':child.returncode==1,
            'valid_lane':child.returncode in (0,1),'seconds':time.monotonic()-start,
            'log':str(log.relative_to(ROOT)),'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()})
    valid=[r for r in records if r['valid_lane']];rate=sum(r['killed'] for r in valid)/len(valid) if valid else 0
    after=hashlib.sha256(source.read_bytes()).hexdigest()
    receipt={'evidence_class':'unit test; author mutation baseline, not blind verification','mutants':records,
        'valid_lanes':len(valid),'kill_fraction':rate,'required_fraction':.8,'source_unchanged':before==after,
        'status':'PASS' if len(valid)==6 and rate>=.8 and before==after else 'RED'}
    p=ROOT/'artifacts/grm_x3/mutation_receipt.json'
    with p.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps(receipt,indent=2));return 0 if receipt['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
