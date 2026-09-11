"""Lead amendment 4: runtime-only C4 cap overlay; no GPU imports or work.

Prior art: local C4/A4/A5 SHA manifests and DET1 lease accounting (house,
2026), verified in local sources. Reuses content binding and fail-closed
validation; adds exact source replacements and a cap-only overlay. No novelty.
"""
from pathlib import Path

from scripts import grm_c4_campaign as c

OUT = c.OUT / 'lead_a4'
AMENDMENT = OUT / 'amendment_a6.json'
ORDER = c.ROOT / 'orders/GRM_C4_AMENDMENT_4.md'
PREVIOUS = c.OUT / 'lead_a3/amendment_a5.json'
ORDER_SHA = 'ef9f3b291bf55077d2578c5a393d3e3dee89f921c4833cd6f3d70400951ea7e1'
PREVIOUS_SHA = 'ffc9a66afe9c27ae9d56b6d5b35ad119666db60d6f9459687dec67f35126cdf1'
REPLACEMENTS = (
    'scripts/grm_c4_resume_a2.py',
    'scripts/grm_c4_remaining_a3.py',
    'tests/test_grm_c4_remaining_a3.py',
)
ADDITIONS = (
    'scripts/grm_c4_cap_a4.py',
    'scripts/grm_c4_register_a4.py',
    'tests/test_grm_c4_cap_a4.py',
    'artifacts/grm_c4/lead_a4/pre_registration.json',
    'artifacts/grm_c4/lead_a4/before.json',
)


def expected_payload():
    # Prior art: A5 exact source closure (house, 2026). Fixed order/predecessor
    # pins prevent a recomputed sidecar from authorizing a different order.
    # SHA binding is integrity checking, not a signature against code writers.
    order, previous = c.record(ORDER), c.record(PREVIOUS)
    assert order['sha256'] == ORDER_SHA, 'cap order drift'
    assert previous['sha256'] == PREVIOUS_SHA, 'cap predecessor drift'
    old = c.read(PREVIOUS)
    assert c.record(old['previous_amendment']['path']) == old['previous_amendment'], 'cap A4 predecessor drift'
    assert old['registration'] == c.record(c.REG), 'cap registration drift'
    sources = {r['path']: r for r in old['sources']}
    replacements = [{'before': sources[str(c.ROOT / p)], 'after': c.record(c.ROOT / p)}
                    for p in REPLACEMENTS]
    changed = {str(c.ROOT / p) for p in REPLACEMENTS}
    for path, pin in sources.items():
        if path not in changed:
            assert c.record(path) == pin, f'source drift: {path}'
    return {
        'schema': 'grm.c4.cap-amendment.v1', 'immutable': True,
        'lead_amendment_number': 4, 'artifact_sequence': 6,
        'order': order, 'previous_amendment': previous,
        'registration': c.record(c.REG), 'previous_budget': old['budget'],
        'overrides': {'budget': {'gpu_seconds': 5400}},
        'source_replacements': replacements,
        'added_sources': [c.record(c.ROOT / p) for p in ADDITIONS],
        'receipt_identity': 'Retain A4/A5 amendment and source identities; attach budget_amendment separately to new worker claims and receipts.',
        'prior_art': 'Local C4/A4/A5 and DET1 (house, 2026): SHA manifests, exact closure and lease accounting. Added cap-only overlay and explicit source replacements; no novel algorithm claimed.',
    }


def binding():
    # Always load at runtime. Missing, stale or forged overlays never fall
    # back to a baked-in cap. Exact equality refuses extra/changed semantics.
    pin = c.record(AMENDMENT)
    assert pin['sha256'] == AMENDMENT.with_suffix('.sha256').read_text().split()[0], 'cap amendment SHA mismatch'
    row = c.read(AMENDMENT)
    assert row == expected_payload(), 'cap amendment content mismatch'
    assert type(row['overrides']['budget']['gpu_seconds']) is int, 'cap must be integer seconds'
    return {**row, 'record': pin}


def check_source(pin, cap):
    # Historical receipt manifests remain immutable. Only exact old->new
    # pairs in the validated overlay can substitute for current source bytes.
    for replacement in cap['source_replacements']:
        if pin == replacement['before']:
            assert c.record(pin['path']) == replacement['after'], 'cap replacement drift'
            return
    assert c.record(pin['path']) == pin, f"source drift: {pin['path']}"


def apply_budget(reg, cap):
    assert reg['budget'] == cap['previous_budget'], 'cap previous budget mismatch'
    return {**reg, 'budget': {**reg['budget'], **cap['overrides']['budget']},
            'budget_amendment': cap['record']}
