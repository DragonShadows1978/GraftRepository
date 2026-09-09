"""Registered r3 author CPU gates and source-copy mutation receipts.

Prior art: GRM contributors (2026), C7/r2 CPU runner and HOUSE_RULES
baseline-before-mutation >=0.80. Reuse immutable receipts and in-memory
source-copy defects. New r3 defect sites; no prior art known to me for
this exact set. No CUDA allocation, external process kill or blind review.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from scripts.grm_c7_common import ROOT, create, sha
from scripts.grm_c7_register_r3 import OUT, REG, TESTS, LEGACY_TESTS, verify_r3

MUTANTS = [
    ('case_sensitive_control','scripts/grm_c7_common.py',
     "exact = text.casefold() == normalize(probe['expected']).casefold()", 'exact = False',
     'tests/test_grm_c7_r3.py::test_control_casefold'),
    ('retain_abstention_clause','scripts/grm_c7_run.py',
     "return question[:-len(suffix)] + '.'", 'return question',
     'tests/test_grm_c7_r3.py::test_plain_questions'),
    ('wrong_budget_7200','scripts/grm_c7_run.py',
     "r.get('budget_seconds_arm_A', 7200)", '7200',
     'tests/test_grm_c7_r3.py::test_registration_and_budget'),
    ('drop_child_revision','scripts/grm_c7_run.py',
     "env['GRM_C7_REVISION'] = os.environ.get('GRM_C7_REVISION', 'r1')", 'pass  # missing revision',
     'tests/test_grm_c7_r3.py::test_r3_child_keeps_revision_and_no_alias_switch'),
    ('weaken_space_preflight','scripts/grm_c7_register_r3.py',
     'if free < minimum:', 'if False:',
     'tests/test_grm_c7_r3.py::test_preflight_boundary'),
]


def main():
    r=verify_r3()
    directory=OUT/'amendment_6'
    env=dict(os.environ, GRM_C7_REVISION='r3', CUDA_VISIBLE_DEVICES='',
             PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    cmd=[sys.executable,'-m','pytest','-q',*TESTS]
    result=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True)
    with (directory/'cpu_registered.log').open('x') as f:
        f.write(result.stdout+result.stderr)
    create(directory/'cpu_registered.json', {'command':cmd,'returncode':result.returncode,
        'registration_sha256':sha(REG),'executed_inputs':r['effective_inputs'],
        'evidence_class':'author CPU baseline; not model validation or blind review','gpu_executed':False})
    print(result.stdout+result.stderr,flush=True)
    if result.returncode:
        return result.returncode
    # Prior art: FIX4 frozen byte receipts (GRM, 2026). Run the unchanged
    # historical-prompt suite in its original prompt mode; r3 necessarily
    # changes its prompt/token bytes and has separate plain-prompt tests.
    legacy_cmd=[sys.executable,'-m','pytest','-q',*LEGACY_TESTS]
    legacy=subprocess.run(legacy_cmd,cwd=ROOT,env=dict(env,GRM_C7_REVISION='r1'),
                          capture_output=True,text=True)
    with (directory/'cpu_legacy_fix4.log').open('x') as f:
        f.write(legacy.stdout+legacy.stderr)
    create(directory/'cpu_legacy_fix4.json', {'command':legacy_cmd,'returncode':legacy.returncode,
        'revision':'r1 historical prompt mode','registration_sha256':sha(REG),
        'evidence_class':'author CPU historical byte regression','gpu_executed':False})
    print(legacy.stdout+legacy.stderr,flush=True)
    if legacy.returncode:
        return legacy.returncode
    rows=[]
    for name, source_path, old, new, test in MUTANTS:
        source=(ROOT/source_path).read_text()
        if source.count(old)!=1:
            raise ValueError('MUTANT_SITE_NOT_UNIQUE: '+name)
        with tempfile.TemporaryDirectory(prefix='grm-c7-r3-mutant-') as td:
            path=Path(td)/'mutant.py'; path.write_text(source.replace(old,new))
            code=('import importlib,pathlib,pytest,sys; '
                  'm=importlib.import_module(sys.argv[1]); '
                  'exec(compile(pathlib.Path(sys.argv[2]).read_text(),sys.argv[2],"exec"),m.__dict__); '
                  'raise SystemExit(pytest.main(["-q",sys.argv[3]]))')
            result=subprocess.run([sys.executable,'-c',code,source_path[:-3].replace('/','.'),str(path),test],
                cwd=ROOT,env=env,capture_output=True,text=True)
            status=('KILLED' if result.returncode==1 and ' failed' in result.stdout else
                    'SURVIVED' if result.returncode==0 else 'ERROR')
            with (directory/(name+'.log')).open('x') as f:
                f.write(result.stdout+result.stderr)
            rows.append({'id':name,'status':status,'returncode':result.returncode,
                         'mutant_sha256':sha(path)})
            print(name+': '+status,flush=True)
    non_error=[row for row in rows if row['status']!='ERROR']
    rate=sum(row['status']=='KILLED' for row in non_error)/len(non_error) if non_error else 0
    unchanged=verify_r3()['effective_inputs']==r['effective_inputs']
    passed=len(non_error)==len(MUTANTS) and rate>=.80 and unchanged
    create(directory/'cpu_mutations.json', {'rows':rows,'kill_rate':rate,'threshold':.80,
        'pass':passed,'registration_sha256':sha(REG),'live_sources_unchanged':unchanged,
        'evidence_class':'author source-copy mutation, no blind review'})
    if not passed:
        return 1
    receipts={str((directory/name).relative_to(ROOT)):sha(directory/name) for name in
              ('cpu_registered.json','cpu_registered.log','cpu_mutations.json',
               'cpu_legacy_fix4.json','cpu_legacy_fix4.log')}
    ready=directory/'ready.json'
    create(ready, {'status':'GREEN_CPU','registration_sha256':sha(REG),'receipts':receipts,
                   'gpu_executed':False,'model_quality':'NOT_RUN'})
    with ready.with_suffix('.sha256').open('x') as f:
        f.write(sha(ready)+'  ready.json\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
