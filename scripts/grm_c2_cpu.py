"""C2 foreground CPU shard runner.

Prior art: WC1 G1 and SCOUT-FIX-1 (project, 2026), reused unmodified pytest
batteries and raw logs. Ours: per-file resumable receipt, no test rewrites.
No prior art known to me beyond those systems for this specific glue.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.grm_c2_profile import OUT, create, read, sha, verify_registration


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--index',type=int,required=True)
    args=p.parse_args()
    verify_registration()
    inventory=read(OUT/'amendment_A1_cpu.json')['cpu_files']
    filename=inventory[args.index]
    path=OUT/'cpu'/f'{args.index:03d}.json'
    if path.exists():
        print(json.dumps(read(path))); return 0
    path.parent.mkdir(exist_ok=True)
    log=path.with_suffix('.log')
    command=[sys.executable,'-m','pytest','-q',filename]
    started=time.monotonic()
    with log.open('x') as f:
        try:
            result=subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT,timeout=285)
            code=result.returncode
        except subprocess.TimeoutExpired:
            code=124
    body={'file':filename,'command':command,'returncode':code,'seconds':time.monotonic()-started,'log':str(log.relative_to(ROOT)),'sha256':sha(log)}
    create(path,body)
    print(json.dumps(body))
    return 0
if __name__=='__main__':
    raise SystemExit(main())
