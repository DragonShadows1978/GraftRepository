"""Run frozen amendment mutants on CPU, never modify original sources.

Prior art: DeMillo/Lipton/Sayward (1978), mutation testing (unverified — lead
to check: Hints on test data selection 1978); C8 in-memory module harness
(GRM, 2026). Taken: seeded-defect rejection and module copies. Ours: amendment
gate receipts. This is an author baseline, not blind verification.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.grm_c2_profile import create, read, sha


def main():
    out = ROOT / 'artifacts/grm_c8'
    registration = out / 'amendment_1_mutation_registration.json'
    if sha(registration) != '217a21786258227502476b9183cfb396708d2dbca44be2fa4f9f00aa06392405':
        raise ValueError('mutation registration changed')
    spec = read(registration)
    source = ROOT / 'scripts/grm_c8_cells.py'
    baseline = out / 'amendment_1_cpu_baseline.log'
    assert sha(source) == spec['runner_sha256']
    assert sha(baseline) == spec['baseline_sha256'] and '113 passed' in baseline.read_text()
    assert sha(ROOT / 'tests/test_grm_c8_amendment.py') == spec['tests_sha256']
    launcher = '''import sys, types, pytest, scripts
from pathlib import Path
source = Path(sys.argv[1])
module = types.ModuleType('scripts.grm_c8_cells')
module.__file__ = str(source)
sys.modules[module.__name__] = module
scripts.grm_c8_cells = module
lease_module = types.ModuleType('scripts.grm_cmc1_gpu_arms')
def forbidden_lease(*a, **kw):
    raise AssertionError('CPU MUTATION FORBIDS ANY GPU LEASE')
lease_module.gpu_lease = forbidden_lease
sys.modules[lease_module.__name__] = lease_module
code = source.read_text().replace(sys.argv[2], sys.argv[3])
exec(compile(code, str(source), 'exec'), module.__dict__)
raise SystemExit(pytest.main(['-q', 'tests/test_grm_c8_amendment.py::' + sys.argv[4]]))
'''
    results = []
    for mutant in spec['mutations']:
        original = source.read_text()
        assert original.count(mutant['old']) == 1
        log = out / ('amendment_1_mutation_' + mutant['name'] + '.log')
        with log.open('x') as stream:
            proc = subprocess.run([sys.executable, '-c', launcher, str(source),
                mutant['old'], mutant['new'], mutant['test']], cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'),
                stdout=stream, stderr=subprocess.STDOUT, timeout=60)
        text = log.read_text()
        valid = proc.returncode in (0, 1) and 'ERROR collecting' not in text
        killed = valid and proc.returncode == 1 and ' failed' in text
        results.append({'name': mutant['name'], 'returncode': proc.returncode,
            'valid': valid, 'killed': killed, 'log': str(log.relative_to(ROOT)),
            'log_sha256': sha(log), 'mutated_source_sha256': hashlib.sha256(
                original.replace(mutant['old'], mutant['new']).encode()).hexdigest()})
    valid = sum(x['valid'] for x in results)
    killed = sum(x['killed'] for x in results)
    receipt = {'evidence_class': 'author mutation baseline, not blind verification',
        'results': results, 'valid_mutants': valid, 'killed_mutants': killed,
        'kill_fraction': killed / valid if valid else 0,
        'pass': valid == 5 and killed / valid >= .8,
        'original_source_unchanged': sha(source) == spec['runner_sha256'],
        'mutation_registration_sha256': sha(registration), 'harness_sha256': sha(__file__),
        'baseline_sha256': sha(baseline)}
    create(out / 'amendment_1_mutation_results.json', receipt)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['pass'] and receipt['original_source_unchanged'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
