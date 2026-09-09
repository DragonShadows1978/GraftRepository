"""Registered CPU gates and five source-copy mutants. No GPU work.

Prior art: C7 CPU scorer mutation harness and HOUSE_RULES section 8
(GRM contributors, 2026). Borrow baseline-before-mutation and >=0.80 kill
rate, source copies, immutable receipts. New: five FIX-3/C7 defect operators;
no prior art known to me for these exact operators. Author baseline only.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from scripts.grm_c7_common import ROOT, OUT, create, sha, verify

TESTS = ['tests/test_grm_scout_fix3.py', 'tests/test_grm_c7_r2.py',
         'tests/test_grm_c7_r2_registration.py', 'tests/test_grm_c7.py',
         'tests/test_grm_c7_a1.py', 'tests/test_grm_c7_a2.py',
         'tests/test_grm_fold_recovered_guard.py', 'tests/test_grm_s4_fold_order.py',
         'tests/test_grm_scout_fix2_capture.py',
         'tests/test_grm_c7_lead_1.py::test_receipt_and_checkpoint_bind_amended_source_and_keep_39_cells',
         'tests/test_grm_c7_lead_1.py::test_restart_projection_detects_lost_split_child_capture']


def main():
    if os.environ.get('GRM_C7_REVISION') != 'r2':
        raise ValueError('R2_ENVIRONMENT_REQUIRED')
    r = verify()
    directory = OUT/'r2'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    cmd = [sys.executable, '-m', 'pytest', '-q', *TESTS]
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=285)
    with (directory/'cpu_registered.log').open('x') as f:
        f.write(result.stdout+result.stderr)
    create(directory/'cpu_registered.json', {'command':cmd, 'returncode':result.returncode,
        'amendment_sha256':r['amendment_r2_sha256'], 'executed_inputs':r['effective_inputs'],
        'evidence_class':'author CPU baseline; not model validation or blind review',
        'gpu_executed':False})
    print(result.stdout+result.stderr, flush=True)
    if result.returncode:
        return result.returncode
    mutants = [
        ('remove_fold_wrapper', 'core/graft_arena.py', 'prompt = formatted + primer', 'prompt = prompt', 'tests/test_grm_scout_fix3.py'),
        ('remove_fold_stops', 'core/graft_arena.py', 'stops = self.stop_sequences if harmony else ()', 'stops = ()', 'tests/test_grm_scout_fix3.py'),
        ('disable_punctuation_qc', 'core/graft_arena.py', 'if re.search(r"(?:…\\s*){3,}|(?:[^\\w\\s]\\s*){6,}", text):', 'if False:', 'tests/test_grm_scout_fix3.py'),
        ('omit_oracle_shift', 'scripts/grm_c7_run.py', 'layer.self_attn.live_shift = arena.live_shift', 'pass  # mutant omits shift', 'tests/test_grm_c7_r2.py'),
        ('disable_instruction_lowercase', 'scripts/grm_c7_run.py', "return question[:-len(suffix)] + 'if unspecified, reply unknown.'", 'return question', 'tests/test_grm_c7_r2.py'),
    ]
    rows = []
    for name, source_path, old, new, test in mutants:
        source = (ROOT/source_path).read_text()
        if source.count(old) != 1:
            raise ValueError('MUTANT_SITE_NOT_UNIQUE: '+name)
        with tempfile.TemporaryDirectory(prefix='grm-c7-r2-mutant-') as td:
            path = Path(td)/'mutant.py'
            path.write_text(source.replace(old,new))
            module = source_path[:-3].replace('/', '.')
            code = ('import importlib, pathlib, pytest, sys; '
                    'm=importlib.import_module(sys.argv[1]); '
                    'exec(compile(pathlib.Path(sys.argv[2]).read_text(),sys.argv[2],"exec"),m.__dict__); '
                    'raise SystemExit(pytest.main(["-q",sys.argv[3]]))')
            result = subprocess.run([sys.executable,'-c',code,module,str(path),test],
                cwd=ROOT,env=env,capture_output=True,text=True,timeout=60)
            status = 'KILLED' if result.returncode == 1 and ' failed' in result.stdout else 'SURVIVED' if result.returncode == 0 else 'ERROR'
            with (directory/(name+'.log')).open('x') as f:
                f.write(result.stdout+result.stderr)
            rows.append({'id':name,'status':status,'returncode':result.returncode,'mutant_sha256':sha(path)})
    non_error = [x for x in rows if x['status'] != 'ERROR']
    rate = sum(x['status']=='KILLED' for x in non_error)/len(non_error) if non_error else 0
    passed = len(non_error)==5 and rate>=.80
    create(directory/'cpu_mutations.json', {'rows':rows,'kill_rate':rate,'threshold':.80,
        'pass':passed,'amendment_sha256':r['amendment_r2_sha256'],
        'live_sources_unchanged': verify()['effective_inputs']==r['effective_inputs'],
        'evidence_class':'author CPU source-copy mutation; not blind review'})
    print(json.dumps({'mutations':rows,'pass':passed}), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
