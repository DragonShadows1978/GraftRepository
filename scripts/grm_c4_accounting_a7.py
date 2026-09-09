#!/usr/bin/env python3
"""A7 exact source replacement closure; no GPU imports or execution.

Prior art: local C4 cap A4 (house, 2026), verified in grm_c4_cap_a4.py:
reuse explicit before/after source pins without rewriting historical manifests.
A7 binds only the accounting repair and its compatibility plumbing. No novelty.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import grm_c4_campaign as c

OUT = c.OUT/'lead_a7'
ORDER = c.ROOT/'orders/GRM_C4_AMENDMENT_7.md'
PREVIOUS = c.OUT/'lead_a6/amendment_a8.json'
AMENDMENT = OUT/'amendment_a9.json'
COMMANDS = c.OUT/'lead_commands_a7.txt'
ORDER_SHA = 'e7176e6a0d11602f35f88a49cffcd65a13834d200a22c1ea9cbdda27b6af0b33'
PREVIOUS_SHA = '4343189f992408e57797dfc0ebd13dc6df2a7e086e28d874823bdaa546f2c78e'
BEFORE_SHA = 'd16fd22bcad2fff203d226abb59b81565a0717af51085c966b96a8f343965985'
REPLACEMENTS = ('scripts/grm_c4_split_a5.py', 'scripts/grm_c4_skip_a6.py')
ADDITIONS = ('scripts/grm_c4_accounting_a7.py', 'tests/test_grm_c4_accounting_a7.py',
             'orders/GRM_C4_AMENDMENT_7.md', 'artifacts/grm_c4/lead_commands_a7.txt',
             'artifacts/grm_c4/lead_a7/pre_registration.json',
             'artifacts/grm_c4/lead_a7/before.json',
             'artifacts/grm_c4/lead_a7/grm_c4_split_a5.py.before',
             'artifacts/grm_c4/lead_a7/grm_c4_skip_a6.py.before')


def expected_payload():
    assert c.record(ORDER)['sha256'] == ORDER_SHA, 'A7 order drift'
    assert c.record(PREVIOUS)['sha256'] == PREVIOUS_SHA, 'A7 predecessor drift'
    assert c.record(OUT/'before.json')['sha256'] == BEFORE_SHA, 'A7 before drift'
    before = {p['path']: p for p in c.read(OUT/'before.json')['files']}
    replacements = []
    for name in REPLACEMENTS:
        old = before[str(c.ROOT/name)]
        snapshot = c.record(OUT/(Path(name).name+'.before'))
        assert (snapshot['sha256'], snapshot['bytes']) == (old['sha256'], old['bytes']), 'A7 original source drift'
        replacements.append({'before': old, 'after': c.record(c.ROOT/name)})
    assert COMMANDS.read_bytes() == (c.OUT/'lead_commands_a6.txt').read_bytes(), 'A7 commands differ from A6'
    return {'schema': 'grm.c4.accounting-repair.v1', 'immutable': True,
            'lead_amendment_number': 7, 'artifact_sequence': 9,
            'order': c.record(ORDER), 'previous_amendment': c.record(PREVIOUS),
            'before': c.record(OUT/'before.json'), 'source_replacements': replacements,
            'added_sources': [c.record(c.ROOT/p) for p in ADDITIONS],
            'receipt_identity': 'A6 identities retained, as with the A4 source replacement overlay. A7 authorizes exact accounting source replacements only; historical manifests and receipts are not rewritten.',
            'policy': 'No budget, plan, projection, skip, retry, dependency, scoring or GPU lease changes. Metadata/attempt/score files are excluded with diagnostics; incomplete worker evidence blocks admission.',
            'prior_art': c.read(OUT/'pre_registration.json')['prior_art']}


def binding():
    pin = c.record(AMENDMENT)
    assert pin['sha256'] == AMENDMENT.with_suffix('.sha256').read_text().split()[0], 'A7 amendment SHA mismatch'
    row = c.read(AMENDMENT)
    assert row == expected_payload(), 'A7 amendment content/source mismatch'
    return row


def historical_record(path):
    # Prior art: A4 check_source (house,2026), explicit exact replacements.
    # The returned pin is deliberately the historical manifest identity,
    # NOT a claim about live bytes: binding verifies the live after-pin first.
    if str(path) in {str(c.ROOT/p) for p in REPLACEMENTS}:
        row = binding()
        return next(p['before'] for p in row['source_replacements'] if p['before']['path'] == str(path))
    return c.record(path)


def register():
    # Prior art: C4 create-only pre-gate registration (house,2026), reused.
    with COMMANDS.open('xb') as f:
        f.write((c.OUT/'lead_commands_a6.txt').read_bytes())
    c.write_once(AMENDMENT, expected_payload())
    with AMENDMENT.with_suffix('.sha256').open('x') as f:
        f.write(c.record(AMENDMENT)['sha256']+'  '+AMENDMENT.name+'\n')
    print(c.record(AMENDMENT))


if __name__ == '__main__':
    assert sys.argv[1:] == ['--register'], 'Only --register is supported'
    register()
