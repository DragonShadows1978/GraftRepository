#!/usr/bin/env python3
"""A6 admission/cap overlay on the unchanged A5 executor; CPU preflight.

Prior art: C4/A4/A5 and DET1 (house, 2026), verified local source: exact
SHA closure, reserve accounting, create-only NON_FIT receipts and dependency
gates. This adds the lead's fitting-only projection and explicit skip commands;
no novel algorithm claimed. A5's timing heuristic is inherited unchanged.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import grm_c4_campaign as c
from scripts import grm_c4_split_a5 as a5

OUT = c.OUT / 'lead_a6'
ORDER = c.ROOT / 'orders/GRM_C4_AMENDMENT_6.md'
AMENDMENT = OUT / 'amendment_a8.json'
COMMANDS = c.OUT / 'lead_commands_a6.txt'
ORDER_SHA = 'f003ef8c6b3f40bced73c5d753b94961bdbd4065d55d7ab8572313d3a01df8f0'
PREVIOUS_SHA = 'e3ab21e4974a11b2d8499233b220ed02dbd2778b9e2023ca1272f1ea416a6d65'
LEGACY_BINDING = a5.binding
LEGACY_VALIDATE_RECEIPT = a5.validate_receipt
ADDITIONS = ('scripts/grm_c4_skip_a6.py', 'scripts/grm_c4_register_a6.py',
             'tests/test_grm_c4_skip_a6.py', 'orders/GRM_C4_AMENDMENT_6.md',
             'artifacts/grm_c4/lead_commands_a6.txt',
             'artifacts/grm_c4/lead_a6/pre_registration.json',
             'artifacts/grm_c4/lead_a6/before.json')


def plan(previous):
    # Prior art: A5 admission and DET1 full-worker reservation (house, 2026).
    # A6 excludes only registered planning NON_FIT estimates, never dependency
    # savings. Preserve all membership, turn ranges and state prerequisites.
    result = copy.deepcopy(previous['a5_plan'])
    running = result['projection']['recorded_seconds']
    peak = running
    skips, dependent = [], []
    blocked_cells = set()
    for u in result['scheduled_units']:
        identity = {k: u[k] for k in ('cell', 'battery', 'spec', 'estimate_seconds')}
        u['projection_before_seconds'] = running
        if u.get('planning_status') == 'NON_FIT':
            skips.append(identity)
            blocked_cells.add(u['cell'])
            u.update(cap_status='NOT_APPLICABLE', dispatch='SKIP',
                     projected_charge_seconds=0)
            continue
        if u['cell'] in blocked_cells:
            dependent.append(identity)
        peak = max(peak, running + max(285, u['estimate_seconds']))
        u.update(cap_status='FIT' if running + max(285, u['estimate_seconds']) <= 6600 else 'NON_FIT',
                 dispatch='WORKER', projected_charge_seconds=u['estimate_seconds'])
        running += u['estimate_seconds']
    result.update(skip_units=skips, dependency_blocked_fitting_units=dependent)
    result['projection'] = {
        'recorded_seconds': previous['a5_plan']['projection']['recorded_seconds'],
        'registered_seconds': sum(u['projected_charge_seconds'] for u in result['scheduled_units']),
        'total_seconds': running, 'cap_seconds': 6600,
        'overflow_seconds': max(0, running-6600), 'headroom_seconds': 6600-running,
        'peak_reservation_seconds': peak,
        'status': 'FIT' if peak <= 6600 else 'NON_FIT',
        'skipped_estimate_seconds': sum(u['estimate_seconds'] for u in skips),
        'accounting': 'All recorded walls including REDs + all 44 fitting estimates; six registered planning NON_FIT units charge zero, with no attempt or lease. Dependency-blocked fitting estimates remain included. Actual admission retains reserve285, lease285, cooldown30, outer590.',
        'limit': 'Planning FIT is not execution completion. lh12b and lh13a in each cell require skipped lh12a state; unchanged dependencies refuse them without GPU. Every long-history battery remains NON_FIT.'}
    return result


def expected_payload(previous):
    assert c.record(ORDER)['sha256'] == ORDER_SHA, 'A6 order drift'
    assert c.record(a5.AMENDMENT)['sha256'] == PREVIOUS_SHA, 'A6 predecessor drift'
    return {'schema': 'grm.c4.skip-cap-amendment.v1', 'immutable': True,
        'lead_amendment_number': 6, 'artifact_sequence': 8,
        'order': c.record(ORDER), 'previous_amendment': c.record(a5.AMENDMENT),
        'registration': c.record(c.REG), 'plan': plan(previous),
        'previous_budget': previous['budget'],
        'budget': {**previous['budget'], 'gpu_seconds': 6600},
        'receipt_root': str(a5.OUT/'runs'),
        'sources': [c.record(c.ROOT/p) for p in ADDITIONS],
        'executor': c.record(a5.__file__),
        'policy': 'Explicit A6 entrypoint overlays the unchanged A5 executor at runtime. Planning NON_FIT skips create only a zero-work status receipt, never an attempt. No longer lease, retries, dependency bypass or partial battery scores.',
        'prior_art': c.read(OUT/'pre_registration.json')['prior_art']}


def binding():
    previous = LEGACY_BINDING()
    pin = c.record(AMENDMENT)
    assert pin['sha256'] == AMENDMENT.with_suffix('.sha256').read_text().split()[0], 'A6 amendment SHA mismatch'
    row = c.read(AMENDMENT)
    assert row == expected_payload(previous), 'A6 amendment content/source mismatch'
    assert type(row['budget']['gpu_seconds']) is int, 'A6 cap must be integer seconds'
    return {**previous, 'amendment': pin, 'sources': row['sources'],
        'budget': row['budget'], 'budget_amendment': pin, 'a5_plan': row['plan']}


def validate_receipt(path, reg, cell, battery, spec):
    # Prior art: A5 receipt identity checks (house, 2026). A6 also refuses
    # a forged PASS/RED or attempt for a registered never-attempted unit.
    row = LEGACY_VALIDATE_RECEIPT(path, reg, cell, battery, spec)
    unit = next((u for u in a5.schedule(reg)
                 if (u['cell'], u['battery'], u['spec']) == (cell, battery, spec)), None)
    if unit and unit['dispatch'] == 'SKIP':
        assert row['status'] == 'NON_FIT', 'Registered skip cannot have executed'
        assert row['reason'] == 'Registered split estimate >=200s: NON_FIT', 'Forged skip reason'
        assert not path.with_suffix('.attempt.json').exists(), 'Registered skip has an attempt'
    return row


@contextmanager
def executor():
    # Prior art: C4/A5 runtime adapter (house, 2026). Retain historical source
    # bytes and receipt root; A6 pins identify all new receipts and budget.
    with patch.object(a5, 'binding', binding), patch.object(a5, 'validate_receipt', validate_receipt):
        yield


def command_text(units):
    # Prior art: A5 foreground dispatch (house, 2026). CPU-only skips have
    # no timeout/lease/cooldown command and cannot consume a 285s reservation.
    lines = ['#!/bin/bash', 'set -e', f'cd {c.ROOT}',
        '# A6 replaces the unqueued A5 command list; run only after prior foreground work exits.',
        '# SKIP is CPU-only status recording. Fitting workers retain state dependencies.',
        'python scripts/grm_c4_skip_a6.py --dry-run', 'sleep 30']
    for cell in a5.CELLS:
        for u in units:
            if u['cell'] != cell:
                continue
            args = f"--cell {cell} --battery {u['battery']} --spec {u['spec']}"
            if u['dispatch'] == 'SKIP':
                lines.append(f'python scripts/grm_c4_skip_a6.py skip {args}')
            else:
                lines += [f'timeout --signal=KILL 590s python scripts/grm_c4_skip_a6.py worker {args}', 'sleep 30']
        lines.append(f'python scripts/grm_c4_skip_a6.py score --cell {cell}')
    return '\n'.join(lines+['python scripts/grm_c4_skip_a6.py summary'])+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', default='preflight',
                        choices=('preflight', 'skip', 'worker', 'score', 'summary'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--cell', choices=a5.CELLS)
    parser.add_argument('--battery', choices=('sup', 'census', 'longhorizon'))
    parser.add_argument('--spec')
    args = parser.parse_args()
    reg = binding()
    if args.dry_run or args.command == 'preflight':
        print(json.dumps({'gpu_executed': False, 'amendment': reg['amendment'],
            'budget': reg['budget'], 'unit_count': len(a5.schedule(reg)), **reg['a5_plan']}, indent=2))
        return
    with executor():
        if args.command in ('skip', 'worker'):
            if args.command == 'skip':
                assert any(u['dispatch']=='SKIP' and (u['cell'],u['battery'],u['spec']) ==
                           (args.cell,args.battery,args.spec) for u in a5.schedule(reg)), 'Not a registered skip'
            row = a5.worker(args.cell, args.battery, args.spec)
            if row is not None:
                print(json.dumps(row, indent=2))
        elif args.command == 'score':
            print(json.dumps(a5.score(args.cell), indent=2))
        else:
            print(a5.format_summary(a5.cross_summary()))


if __name__ == '__main__':
    main()
