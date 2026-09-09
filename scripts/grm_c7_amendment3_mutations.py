"""Registered CPU source-copy mutants; never execute a GPU cell.

Prior art: local C7 r2 mutation runner / HOUSE_RULES section 8 (GRM
contributors, 2026). Reuse isolated source copies and >=0.80 non-error kill
rate. No prior art known to me for these exact three defect operators.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from scripts.grm_c7_common import ROOT, create, sha


def main():
    directory = ROOT/'artifacts/grm_c7/r2/amendment_3'
    if '62 passed' not in (directory/'cpu_green.log').read_text():
        raise ValueError('PASSING_BASELINE_REQUIRED')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    mutants = [
        ('direct_shift_read', 'scripts/grm_c7_run.py',
         "getattr(layer.self_attn, 'live_shift', missing_shift)",
         'layer.self_attn.live_shift'),
        ('leave_shift_after_oracle', 'scripts/grm_c7_run.py',
         "delattr(layer.self_attn, 'live_shift')", 'pass  # mutation leaves shift'),
        ('fake_precreates_live_shift', 'scripts/grm_c7_diagnose.py',
         'self.layers = [SimpleNamespace(self_attn=GptOssAttentionTC(cfg, 0))]',
         'self.layers = [SimpleNamespace(self_attn=GptOssAttentionTC(cfg, 0))]\n'
         '        self.layers[0].self_attn.live_shift = None'),
    ]
    before = {name: sha(ROOT/name) for _, name, _, _ in mutants}
    rows = []
    for name, path, old, new in mutants:
        source = (ROOT/path).read_text()
        if source.count(old) != 1:
            raise ValueError('MUTANT_SITE_NOT_UNIQUE: '+name)
        with tempfile.TemporaryDirectory(prefix='c7-a3-mutant-') as td:
            copy = Path(td)/'mutant.py'
            copy.write_text(source.replace(old, new))
            code = ('import importlib,pathlib,pytest,sys; '
                    'm=importlib.import_module(sys.argv[1]); '
                    'exec(compile(pathlib.Path(sys.argv[2]).read_text(),sys.argv[2],"exec"),m.__dict__); '
                    'raise SystemExit(pytest.main(["-q","tests/test_grm_c7_amendment3.py"]))')
            result = subprocess.run([sys.executable, '-c', code,
                path[:-3].replace('/', '.'), str(copy)], cwd=ROOT, env=env,
                capture_output=True, text=True, timeout=60)
            status = ('KILLED' if result.returncode == 1 and ' failed' in result.stdout
                      else 'SURVIVED' if result.returncode == 0 else 'ERROR')
            with (directory/(name+'.log')).open('x') as f:
                f.write(result.stdout+result.stderr)
            rows.append({'name':name, 'status':status, 'returncode':result.returncode,
                         'source_copy_sha256':sha(copy)})
    non_error = [r for r in rows if r['status'] != 'ERROR']
    rate = sum(r['status']=='KILLED' for r in non_error)/len(non_error) if non_error else 0
    unchanged = all(sha(ROOT/name)==value for name,value in before.items())
    passed = len(non_error)==3 and rate>=0.8 and unchanged
    create(directory/'cpu_mutations.json', {'rows':rows, 'kill_rate':rate,
        'pass':passed, 'live_sources_unchanged':unchanged, 'gpu_executed':False,
        'registration_sha256':sha(directory/'registration.json'),
        'evidence_class':'author CPU mutation tests, not blind verification'})
    print(rows, 'pass=', passed, flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
