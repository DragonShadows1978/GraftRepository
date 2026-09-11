#!/usr/bin/env python3
"""GRM-C6 foreground CPU receipt driver. Original harnesses are imported by pytest.

Prior art: Python Software Foundation ast/hashlib/subprocess and pytest's JUnit
reporting (installed versions recorded); only audit orchestration is ours.
No runtime algorithm, assertion, threshold, skip or xfail is introduced here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--cell', action='append', default=[])
    args = parser.parse_args()
    regpath = HERE / 'registration.json'
    assert digest(regpath) == (HERE / 'registration.sha256').read_text().split()[0]
    reg = json.loads(regpath.read_text())
    cells = [c for c in reg['cpu_cells'] if not args.cell or c['id'] in args.cell]
    if args.dry_run:
        print(json.dumps({'cpu_cells': cells, 'gpu_execution': False,
                          'gpu_plan': reg['gpu_acceptance'],
                          'gpu_budget_s': reg['gpu_followup_budget_s']}, indent=2))
        return
    if not args.cell:
        parser.error('Name foreground cells explicitly; inspect --dry-run first')
    for cell in cells:
        receipt = HERE / (cell['id'] + '.json')
        if receipt.exists():
            raise RuntimeError('Immutable receipt already exists: ' + str(receipt))
        executed_sources = {p: digest(ROOT / p) for p in reg['fingerprints']}
        assert executed_sources == reg['fingerprints'], 'Source changed after registration'
        # Existing native CPU tests compile from a hard-coded sibling checkout.
        # Fingerprint those actual inputs without changing their import or path.
        external = {}
        for p in Path('/mnt/ForgeRealm/GraftRepository/cpp').glob('*'):
            if p.is_file():
                external[str(p)] = digest(p)
        log = HERE / (cell['id'] + '.log')
        xml = HERE / (cell['id'] + '.xml')
        command = [sys.executable, '-B', '-m', 'pytest', '-q', cell['file'],
                   '-p', 'no:cacheprovider', '--tb=short', '-ra',
                   '--junitxml=' + str(xml)]
        if cell['selection']:
            command.extend(['-k', ' or '.join(cell['selection'])])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                   PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=str(ROOT))
        # Resolve shipped defaults, preserving test-owned monkeypatch overrides.
        for key in list(env):
            if key.startswith('GRM_'):
                del env[key]
        start = time.monotonic()
        with log.open('w') as stream:
            try:
                rc = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT,
                                    timeout=cell['timeout_s']).returncode
            except subprocess.TimeoutExpired:
                rc = 124  # subprocess.run only kills its own child; never others.
        counts = {}
        if xml.exists():
            suites = ET.parse(xml).getroot()
            for key in ('tests', 'failures', 'errors', 'skipped'):
                counts[key] = sum(int(s.get(key, 0)) for s in suites.iter('testsuite'))
            counts['passed'] = counts['tests'] - sum(counts[k] for k in ('failures', 'errors', 'skipped'))
        result = {'cell': cell, 'command': command, 'rc': rc,
                  'wall_s': time.monotonic() - start, 'counts': counts,
                  'registration_sha256': digest(regpath),
                  'driver_sha256': digest(Path(__file__)),
                  'source_fingerprints': executed_sources,
                  'external_cpp_fingerprints': external,
                  'log_sha256': digest(log),
                  'xml_sha256': digest(xml) if xml.exists() else None,
                  'scope': 'CPU unit/suite run; GPU hidden, no model acceptance'}
        receipt.write_text(json.dumps(result, indent=2) + '\n')
        with (HERE / 'IMPLEMENTATION_LEDGER.md').open('a') as stream:
            stream.write(f"- CPU `{cell['id']}`: rc={rc}, {counts}, {result['wall_s']:.2f}s; receipt `{receipt.name}`; source/driver/registration/log hashes bound.\n")
        print(json.dumps({k: result[k] for k in ('cell', 'rc', 'wall_s', 'counts')}), flush=True)


if __name__ == '__main__':
    main()
