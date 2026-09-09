"""Rerun failed retained cases with their original suite environment.
Prior art: GRM contributors (2026), amendment3 CPU runner's environment
isolation and output redirection. Reuse unchanged tests; only move newly
produced CPU receipts out of immutable amendment3. No new test predicates.
"""
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from scripts import grm_lt1 as lt
from scripts import grm_lt1_amendment4 as a4


def main():
    os.environ.pop('GRM_ADMISSION_RULE',None)
    for name in ('GRM_FIX4_FREEZE_BEFORE','GRM_FIX5_FREEZE_BEFORE','GRM_FIX5_RECEIPTS'):
        os.environ.pop(name,None)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    original=lt.create
    def create(path,value):
        p=Path(path)
        if p in [lt.OUT/'amendment3/r3'/f'worker_fake_{arm}.json' for arm in 'AB']:
            p=a4.OUT/p.name
        return original(p,value)
    lt.create=create
    reg=lt.read(a4.OUT/'RETAINED_RERUN_REGISTRATION.json')
    assert reg['runner_sha256']==lt.sha(Path(__file__))
    assert reg['initial_junit_sha256']==lt.sha(a4.OUT/'retained_tests.xml')
    import pytest
    return pytest.main(['-q','-p','no:cacheprovider','--junitxml='+str(a4.OUT/'retained_rerun.xml')]+reg['nodeids'])

if __name__=='__main__':
    raise SystemExit(main())
