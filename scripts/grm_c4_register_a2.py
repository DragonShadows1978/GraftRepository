#!/usr/bin/env python3
"""Register isolated lead amendment 2 before CPU gates.
Prior art: C4/RS3 (house, 2026) immutable order/source/SHA amendment chain.
This amendment changes only recovery/guard scope; no new algorithm claimed.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import grm_c4_campaign as c
from scripts import grm_c4_resume_a2 as a2


def main():
    reg = c.binding()
    assert reg['amendment'] == c.record(c.OUT/'amendment_a3.json')
    context = {**reg, 'legacy_sources': reg['sources'], 'legacy_amendment': reg['amendment']}
    command = c.OUT/'lead_commands_a2.txt'
    with command.open('x') as f:
        f.write(a2.command_text(context))
    paths = {Path(r['path']) for r in reg['sources']} | {
        Path(__file__).resolve(), Path(a2.__file__), a2.ORDER, command,
        c.ROOT/'tests/test_grm_c4_resume_a2.py',
        a2.OUT/'pre_registration.json', a2.OUT/'original_receipts_before.json'}
    payload = {
        'schema': 'grm.c4.amendment.v1', 'immutable': True,
        'lead_amendment_number': 2, 'artifact_sequence': 4,
        'order': c.record(a2.ORDER), 'registration': c.record(c.REG),
        'previous_amendment': c.record(c.OUT/'amendment_a3.json'),
        'activation': 'Explicit scripts/grm_c4_resume_a2.py entry point only; after original chain finishes. Nested amendment intentionally invisible to original binding glob.',
        'model_id': 'GPT-6 (Codex; specific deployment id not exposed)',
        'reasoning_effort': 'high (order)',
        'guard': {'scope': 'cell long-history segments only', 'evaluate': 'last registered long-history segment', 'minimum_attempt_residency': 1,
                  'segment_requires': ['exact registered completed turn indices', 'saved transcript/instrumentation full prefix and restart boundary', 'complete nonempty SHA-bound session state', 'geometry observation']},
        'rerun': {'maximum_attempts_per_suffix_unit': 1, 'root_error_exact': a2.ROOT_ERROR,
                  'dependent_error_exact': a2.DEPENDENCY_ERROR,
                  'missing_dependent_receipts': 'Allowed only in transitive suffix of an evidenced exact-error root; original pre-lease dependency failures emitted no receipt.',
                  'existing_pass': 'Never rerun or overwrite', 'other_red': 'Registered stop, including any amended RED',
                  'abandoned_claim': 'Registered stop, never clear marker',
                  'resume': 'Immediate effective PASS predecessor: first rerun resumes last original PASS; subsequent units resume amended PASS. Verify and copy bound state, never RED state.',
                  'source_compatibility': 'Exactly A3 source manifest/amendment for original receipts; exactly A4 manifest/amendment for new receipts. Preserve both identities.',
                  'accounting': 'All original and amended GPU seconds included; 4800 cap,285 reserve,285 lease,590 outer,30 cooldown unchanged.'},
        'confirmation': c.read(a2.OUT/'pre_registration.json')['confirmation'],
        'registered_probe_turns': [p['turn'] for p in reg['fixtures']['longhorizon']['probes']],
        'eligible_suffixes_at_registration': {cell: a2.eligible_suffix(context, cell) for cell in c.executable_cells(reg)},
        'unchanged': ['scoring', 'geometry', 'fixtures', 'counts', 'acceptance', 'prediction', 'rejection', 'units_per_new_cell', 'budget', 'execution_order', 'diagonal_comparison', 'diagonal_ruling'],
        'gates_before_gpu': c.read(a2.OUT/'pre_registration.json')['gates'],
        'prior_art': c.read(a2.OUT/'pre_registration.json')['prior_art'],
        'sources': [c.record(p) for p in sorted(paths)],
    }
    c.write_once(a2.AMENDMENT, payload)
    with a2.AMENDMENT.with_suffix('.sha256').open('x') as f:
        f.write(c.record(a2.AMENDMENT)['sha256']+'  amendment_a4.json\n')
    print(c.record(a2.AMENDMENT))


if __name__ == '__main__':
    main()
