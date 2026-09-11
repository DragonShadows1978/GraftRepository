#!/usr/bin/env python3
"""C7 author baseline and registered scorer mutations; no GPU imports.

Prior art: HOUSE_RULES section 8, local C2 CPU receipts (GRM, 2026).
Borrowed: baseline then mutation, frozen thresholds and disjoint receipts.
Ours: five small copies with score defects; no new testing algorithm and no
prior art known to me for these exact mutants. Not blind verification.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.grm_c7_common import ROOT, OUT, REG, create, sha, verify


def hand_gate(module):
    cases = [
        ('Onyx-911', True, 'Onyx-911', (0,0,0,0)),
        ('Flint-511 or Onyx-911', True, 'Onyx-911', (1,1,0,0)),
        ('UNKNOWN', True, 'Onyx-911', (1,0,1,0)),
        ('UNKNOWN', False, 'UNKNOWN', (0,0,0,0)),
        ('Onyx-911', False, 'UNKNOWN', (1,0,0,1)),
        ('Cedar-1011|Reed-611', True, 'Reed-611|Cedar-1011', (1,1,0,0)),
    ]
    for answer, answerable, expected, errors in cases:
        s = module.score(answer, {'answerable':answerable,'expected':expected})
        assert tuple(s[k] for k in ('exact_answer_error','wrong_value_error','abstention_error','unsupported_answer_error')) == errors


def mutations():
    from scripts import grm_c7_common
    hand_gate(grm_c7_common)
    source = (ROOT/'scripts/grm_c7_common.py').read_text()
    mutants = [
        ('substring_instead_of_exact', 'exact = text == normalize(probe[\'expected\'])',
         'exact = normalize(probe[\'expected\']) in text'),
        ('erase_abstention_error', "'abstention_error': int(required and abstained)", "'abstention_error': 0"),
        ('erase_unsupported_error', "'unsupported_answer_error': int(not required and not abstained)", "'unsupported_answer_error': 0"),
        ('invert_exact_failure', "'exact_answer_error': int(not exact)", "'exact_answer_error': int(exact)"),
        ('overlap_wrong_and_abstention', "'wrong_value_error': int(required and not exact and not abstained)",
         "'wrong_value_error': int(required and not exact)"),
    ]
    rows = []
    with tempfile.TemporaryDirectory(prefix='grm-c7-mutations-') as td:
        for name, old, new in mutants:
            if source.count(old) != 1:
                raise ValueError(f'MUTANT_SITE_NOT_UNIQUE: {name}')
            path = Path(td)/f'{name}.py'
            path.write_text(source.replace(old,new))
            spec = importlib.util.spec_from_file_location(name,path)
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
                hand_gate(module)
                status = 'SURVIVED'
            except AssertionError:
                status = 'KILLED'
            except Exception as exc:
                status = f'ERROR: {type(exc).__name__}: {exc}'
            rows.append({'id':name,'status':status,'mutant_sha256':sha(path)})
    non_error = [x for x in rows if not x['status'].startswith('ERROR')]
    rate = sum(x['status']=='KILLED' for x in non_error)/len(non_error) if non_error else 0
    return {'rows': rows, 'non_error_count':len(non_error), 'kill_rate':rate,
            'threshold':.80, 'pass':len(non_error)==5 and rate>=.80}


def main():
    r = verify()
    started = time.monotonic()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    result = subprocess.run([sys.executable,'-m','pytest','-q','tests/test_grm_c7.py'],
                            cwd=ROOT,env=env,capture_output=True,text=True)
    with (OUT/'cpu_baseline.log').open('x') as stream:
        stream.write(result.stdout+result.stderr)
    receipt = {'evidence_class':'author CPU baseline, not blind review or model validation',
               'registration_sha256':sha(REG), 'command':[sys.executable,'-m','pytest','-q','tests/test_grm_c7.py'],
               'returncode':result.returncode, 'gpu_executed':False,
               'wall_seconds':time.monotonic()-started, 'executed_inputs':r['immutable_inputs']}
    create(OUT/'cpu_baseline.json',receipt)
    print(result.stdout+result.stderr)
    if result.returncode:
        return result.returncode
    m = mutations()
    m.update(registration_sha256=sha(REG),evidence_class='CPU scorer mutation gate; five registered defects; source copies only')
    create(OUT/'cpu_mutations.json',m)
    print(json.dumps(m,indent=2))
    return 0 if m['pass'] else 1


if __name__=='__main__':
    raise SystemExit(main())
