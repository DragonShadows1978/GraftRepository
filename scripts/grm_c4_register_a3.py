#!/usr/bin/env python3
"""Create lead amendment 3 before gates; never edit earlier manifests.
Prior art: C4/A4 immutable registration and SHA source closure (house, 2026).
This order adds executor scope only; no novel algorithm claimed.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import grm_c4_campaign as c
from scripts import grm_c4_resume_a2 as a2
from scripts import grm_c4_remaining_a3 as a3


def main():
    reg = a2.binding()
    assert tuple(c.executable_cells(reg)[1:]) == a3.CELLS
    for root in (c.OUT, a2.OUT, a3.OUT):
        for cell in a3.CELLS:
            assert not any((root/'runs'/cell).glob('*')), 'Cell already started; registration refused'
    # Snapshot existing JSON receipts/claims and A4 sources, not future files.
    paths = set()
    for root in (c.OUT, a2.OUT):
        paths.update((root/'runs').glob('*/*.json'))
    snapshot = {'evidence_class': 'read-only pre-gate SHA inventory',
        'files': [c.record(p) for p in sorted(paths)]}
    c.write_once(a3.OUT/'original_receipts_before.json', snapshot)
    pre = {'order': c.record(a3.ORDER), 'previous_amendment': c.record(a2.AMENDMENT),
        'gates': ['dry-run: exactly 42 registered units, order, estimates and bound resume',
            'SHA/order/source/chain/scope/budget/receipt-root forgery or drift refused',
            'existing receipts byte-unchanged; one attempt; no earlier-tree writes',
            'CPU fake harness all batteries with A4 per-cell serving and bound resume',
            'three-root accounting: original RED charged; other RED/orphans/cooldown/non-fit stop',
            'synthetic four-cell summary: WC1 full14 diagonal, historical comparison, prediction outcomes; missing/stale/probe mismatch refused',
            'existing C4 and A4 CPU suites and shell syntax'],
        'prediction_interpretation': 'MET only if wide chunk64 meets registered 9/10/14 and named probes. Narrow-only required-probe/full-LH recovery gives REJECTED_FINITE; other failure is MIXED. Missing evidence refuses summary as INCONCLUSIVE.',
        'prior_art': 'Verified local C4/A4/WC1/RS3/EB1/DET1 (house, 2026): SHA manifests, create-only receipts, ordered saved-state resume, lease accounting, imported scorers and probe comparison. Added executor scope and cross-summary assembly; no novel algorithm claimed.',
        'model_id': 'GPT-6 (Codex; deployment identifier not exposed)', 'reasoning_effort': 'high (order)',
        'scope': 'CPU author gates only; no GPU, git, subagents, background waits, process signals or lock changes.'}
    c.write_once(a3.OUT/'pre_registration.json', pre)
    command = c.OUT/'lead_commands_a3.txt'
    with command.open('x') as f:
        f.write(a3.command_text(reg))
    sources = {Path(r['path']) for r in reg['sources']} | {
        Path(__file__).resolve(), Path(a3.__file__), a3.ORDER, command,
        c.ROOT/'tests/test_grm_c4_remaining_a3.py', a3.OUT/'pre_registration.json',
        a3.OUT/'original_receipts_before.json'}
    payload = {'schema': 'grm.c4.amendment.v1', 'immutable': True,
        'lead_amendment_number': 3, 'artifact_sequence': 5,
        'order': c.record(a3.ORDER), 'registration': c.record(c.REG),
        'previous_amendment': c.record(a2.AMENDMENT), 'executor': c.record(a3.__file__),
        'activation': 'Explicit A4 sibling entry point only, after A2 recovery and score; original and A4 manifests unchanged.',
        'scheduled_units': a3.schedule(reg), 'score_cells': list(a3.CELLS),
        'receipt_root': str(a3.OUT/'runs'), 'receipt_policy': 'Create-only claims, worker receipts, state destinations and scores. No retries or overwrite, including previous executor attempts.',
        'guard': c.read(a2.AMENDMENT)['guard'], 'budget': reg['budget'],
        'accounting': 'Count every completed original/A4/A5 worker wall exactly once, including original c64_w96 eligible RED suffix. Orphan claims and any other RED stop. Reserve 285, cooldown30, cap4800, lease285, outer590.',
        'unchanged': ['units_per_new_cell','fixtures','counts','geometry','prediction','acceptance','rejection','scoring','execution_order','diagonal_comparison','diagonal_ruling'],
        'gates_before_gpu': pre['gates'], 'prior_art': pre['prior_art'],
        'model_id': pre['model_id'], 'reasoning_effort': pre['reasoning_effort'],
        'sources': [c.record(p) for p in sorted(sources)]}
    c.write_once(a3.AMENDMENT, payload)
    with a3.AMENDMENT.with_suffix('.sha256').open('x') as f:
        f.write(c.record(a3.AMENDMENT)['sha256']+'  amendment_a5.json\n')
    print(c.record(a3.AMENDMENT))


if __name__ == '__main__':
    main()
