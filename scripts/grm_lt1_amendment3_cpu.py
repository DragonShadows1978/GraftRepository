"""Foreground CPU suite receipts for LT1 source amendment 3.
Prior art: GRM contributors, LT1/C7 registered CPU receipts (2026), local
source verified; reuse pytest and SHA-bound receipts. Historical receipt
writes go to a fresh directory without changing assertions or sources.
No prior art known to me for this exact suite composition.
"""
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from scripts import grm_lt1 as lt

OUT = lt.OUT/'amendment3/r3'
SUITES = {
    'fix4': ['tests/test_grm_scout_fix4.py'],
    'fix5': ['tests/test_grm_scout_fix5.py'],
    'fix6': ['tests/test_grm_scout_fix6.py'],
    'lt1_current': ['tests/test_grm_lt1_fix6.py','tests/test_grm_lt1_controller.py','tests/test_grm_lt1_amendment3.py'],
    'lt1_amendment1_contracts': ['tests/test_grm_lt1_amendment1.py','-k','not test_worker_all_8_turn_cells_resume_and_designated_restarts'],
    'lt1_historical': ['tests/test_grm_lt1.py','tests/test_grm_lt1_amendment1.py'],
}


def child(name):
    import pytest
    if name == 'lt1_historical':
        original = lt.create
        def create(path, value):
            p = Path(path)
            if p.parent in (lt.OUT, lt.OUT/'amendment1') and p.suffix == '.json':
                p = OUT/'historical_receipts'/p.relative_to(lt.OUT)
            return original(p, value)
        lt.create = create
    return pytest.main(['-q','-p','no:cacheprovider','--tb=short',
                        '--junitxml='+str(OUT/(name+'.xml'))]+SUITES[name])


def main():
    if len(sys.argv)>1:
        return child(sys.argv[1])
    lt.verify()
    OUT.mkdir(exist_ok=True)
    results={}
    for name in SUITES:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
        env.pop('GRM_ADMISSION_RULE',None)
        env.pop('GRM_FIX4_FREEZE_BEFORE',None);env.pop('GRM_FIX5_FREEZE_BEFORE',None)
        env.pop('GRM_FIX5_RECEIPTS',None)
        log=OUT/(name+'.log')
        with log.open('x') as stream:
            r=subprocess.run([sys.executable,'-m','scripts.grm_lt1_amendment3_cpu',name],
                             cwd=lt.ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        root=ET.parse(OUT/(name+'.xml')).getroot()
        cases=list(root.iter('testcase'))
        failed=sum(c.find('failure') is not None for c in cases)
        errors=sum(c.find('error') is not None for c in cases)
        skipped=sum(c.find('skipped') is not None for c in cases)
        results[name]=dict(exit_code=r.returncode,passed=len(cases)-failed-errors-skipped,
            failed=failed,errors=errors,skipped=skipped,tests=[c.attrib for c in cases],
            log=str(log.relative_to(lt.ROOT)),log_sha256=lt.sha(log),binding=lt.binding('CPU'))
        lt.create(OUT/(name+'_receipt.json'),results[name])
        print(name, {k:v for k,v in results[name].items() if k in ('exit_code','passed','failed','errors','skipped')},flush=True)
    active={n:r for n,r in results.items() if n!='lt1_historical'}
    good=all(r['exit_code']==0 and r['passed']>0 and not r['skipped'] for r in active.values())
    lt.create(OUT/'cpu_receipt.json',dict(status='PASS' if good else 'RED',
        fix4_check='PASS' if results['fix4']['exit_code']==0 and results['fix4']['passed']==7 else 'RED',
        binding=lt.binding('CPU'),amendment_sha256=lt.sha(lt.AMEND3),suites=results,
        active_passed=sum(r['passed'] for r in active.values()),gpu_executed=False,
        evidence_class='author CPU suites and fake model; historical superseded protocol results reported separately; no blind validation or model quality claim'))
    return 0 if good else 2


if __name__ == '__main__':
    raise SystemExit(main())
