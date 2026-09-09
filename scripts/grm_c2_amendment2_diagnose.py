#!/usr/bin/env python3
"""CPU-only replay of the amendment-2 SCOUT-FIX-1 stop finding.

Prior art: C2 strict_capture_grade/evidence_rows and B3 ordinary manifests,
project contributors (2026), inspected locally. Taken: existing validators;
ours: exact persisted-evidence replay and source/receipt hash checks. No prior
art known to me for this diagnostic glue beyond these local systems.
This is a diagnostic, not a campaign runner or authorization for a retry.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.grm_c2_profile import read, sha
from scripts.grm_c2_cells import strict_capture_grade
from scripts.grm_c2_amended import effective_registration, evidence_rows, charged_seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--amendment-sha256', required=True)
    args = parser.parse_args()
    amendment = ROOT / 'artifacts/grm_c2/amendment_lead_2.json'
    if sha(amendment) != args.amendment_sha256:
        raise ValueError('diagnosis amendment digest mismatch')
    a = read(amendment)
    for name, digest in a['bindings'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'diagnosis input changed: {name}')
    base = ROOT / a['cell_directory']
    worker = read(base / 'worker.json')
    manifest = read(base / 'session/repository/manifest.json')
    fixture = read(ROOT / a['fixture'])
    grade = strict_capture_grade(manifest['nodes'], 'profile', 96, 19)
    assert grade == worker['end_capture'] == fixture
    assert grade['valid'] is False
    assert [r['node'] for r in grade['rows'] if r['capture'] is None] == [4, 5]
    for index, span in ((4, (0, 86)), (5, (86, 150))):
        node = manifest['nodes'][index]
        assert 'capture' not in node
        assert node['metadata']['width_guard_child'] is True
        assert node['metadata']['culled_from'] == 3
        assert (node['metadata']['token_start'], node['metadata']['token_end']) == span
        assert node['provenance'][0]['segment_type'] == 'width_guard_span'
    try:
        evidence_rows(worker['cell'], worker)
    except ValueError as error:
        assert str(error) == 'invalid persisted capture evidence'
    else:
        raise AssertionError('persisted RED was incorrectly accepted')
    registration = effective_registration()
    controller = read(base / 'controller.json')
    assert controller['status'] == 'RED'
    assert controller['charged_seconds'] == a['charged_seconds_preserved']
    assert charged_seconds(registration) == a['charged_seconds_preserved']
    assert registration['budget_seconds'] == 3600
    for name in a['source_matches_worker']:
        assert sha(ROOT / name) == worker['sources'][str(ROOT / name)]
    print(json.dumps({
        'status': 'DIAGNOSIS_REPRODUCED_CORE_GAP_STOP',
        'evidence_class': 'author-run CPU persisted-fixture replay; no core execution',
        'manifest_reproduces_exact_end_capture': True,
        'validator_error': 'ValueError: invalid persisted capture evidence',
        'affected_nodes': [4, 5], 'parent': 3,
        'source_matches_lead_worker': True,
        'original_amendment_verifies': True,
        'charged_seconds': charged_seconds(registration),
        'budget_seconds': registration['budget_seconds'],
        'gpu_executed': False, 'retry_enabled': False,
        'amendment_sha256': args.amendment_sha256,
    }, indent=2))


if __name__ == '__main__':
    main()
