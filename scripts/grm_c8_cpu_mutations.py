"""Run the five pre-registered C8 mutants as in-memory source copies.

Prior art: mutation testing, DeMillo/Lipton/Sayward (1978; unverified — lead
search 'Hints on test data selection 1978'). Taken: seed defects and ask tests
to reject them. Ours: registered C8 substitutions and local receipt glue.
Original sources are read-only; no GPU, background jobs, or foreign signals.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import grm_c8_cells as c
from scripts.grm_c2_profile import create, sha


def main():
    r = c.registration()
    baseline = c.OUT / 'cpu_baseline.log'
    if '88 passed' not in baseline.read_text():
        raise RuntimeError('registered mutation runs require passing baseline')
    source = ROOT / 'scripts/grm_c8_profiler.py'
    before = sha(source)
    results = []
    for mutant in r['mutations']:
        original = source.read_text()
        if original.count(mutant['old']) != 1:
            raise RuntimeError(f'nonunique mutation target: {mutant["name"]}')
        code = original.replace(mutant['old'], mutant['new'])
        # Compile and execute with the original __file__ so source-map reads
        # still locate the unchanged product code; only profiler code mutates.
        launcher = """import sys,types,pytest,scripts
from pathlib import Path
source=Path(sys.argv[1])
code=source.read_text().replace(sys.argv[2],sys.argv[3])
module=types.ModuleType('scripts.grm_c8_profiler')
module.__file__=str(source)
sys.modules[module.__name__]=module
scripts.grm_c8_profiler=module
exec(compile(code,str(source),'exec'),module.__dict__)
raise SystemExit(pytest.main(['-q','tests/test_grm_c8_profiler.py::'+sys.argv[4]]))
"""
        path = c.OUT / ('mutation_' + mutant['name'] + '.log')
        command = [sys.executable, '-c', launcher, str(source), mutant['old'], mutant['new'], mutant['test']]
        started = time.monotonic()
        with path.open('x') as stream:
            proc = subprocess.run(command, cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'),
                stdout=stream, stderr=subprocess.STDOUT, timeout=120)
        text = path.read_text()
        # A syntax/collection error is INVALID, never a killed mutant.
        valid = proc.returncode in (0, 1) and 'ERROR collecting' not in text
        killed = valid and proc.returncode == 1 and ('AssertionError' in text or 'assert ' in text)
        results.append({'name': mutant['name'], 'returncode': proc.returncode,
            'valid': valid, 'killed': killed, 'seconds': time.monotonic()-started,
            'mutated_source_sha256': hashlib.sha256(code.encode()).hexdigest(),
            'log': str(path.relative_to(ROOT)), 'log_sha256': sha(path)})
    valid = sum(row['valid'] for row in results)
    killed = sum(row['killed'] for row in results)
    receipt = {'evidence_class': 'author mutation baseline, not blind verification',
        'results': results, 'valid_mutants': valid, 'killed_mutants': killed,
        'kill_fraction': killed/valid if valid else 0.,
        'pass': valid == 5 and killed/valid >= .8,
        'original_source_unchanged': before == sha(source),
        'source_sha256': before, 'harness_sha256': sha(Path(__file__)),
        'registration_sha256': sha(c.REG), 'baseline_sha256': sha(baseline)}
    create(c.OUT / 'mutation_results.json', receipt)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['pass'] and receipt['original_source_unchanged'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
