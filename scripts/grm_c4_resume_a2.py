#!/usr/bin/env python3
"""Lead amendment 2: isolated C4 one-time long-history recovery.

Prior art: local C4/WC1/EB1/DET1 (house, 2026), SHA-bound manifests,
create-only claims, saved-state copy/resume and unchanged battery scorers.
This order adds per-cell serving validation and narrowly authorized recovery.
No novel algorithm claimed. Original campaign and live run tree stay untouched.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import grm_c4_campaign as c

OUT = c.OUT / 'lead_a2'
AMENDMENT = OUT / 'amendment_a4.json'
ORDER = c.ROOT / 'orders/GRM_C4_AMENDMENT_2.md'
ROOT_ERROR = 'AssertionError: No serving observed'
DEPENDENCY_ERROR = 'AssertionError: Prior unit missing/RED'


def binding():
    # Prior art: C4/RS3 immutable SHA chain (house, 2026). Nested opt-in
    # amendment avoids changing the manifest of the still-running A3 chain.
    reg = c.binding()
    a = c.read(AMENDMENT)
    assert c.record(AMENDMENT)['sha256'] == AMENDMENT.with_suffix('.sha256').read_text().split()[0], 'amendment SHA mismatch'
    assert a['previous_amendment'] == reg['amendment'], 'amendment chain mismatch'
    assert a['registration'] == c.record(c.REG), 'amendment registration mismatch'
    assert c.record(a['order']['path']) == a['order'], 'amendment order drift'
    for row in a['sources']:
        assert c.record(row['path']) == row, f"source drift: {row['path']}"
    return {**reg, 'legacy_sources': reg['sources'], 'legacy_amendment': reg['amendment'],
            'sources': a['sources'], 'amendment': c.record(AMENDMENT)}


def lh_units(reg):
    return [u for u in reg['units_per_new_cell'] if u['battery'] == 'longhorizon']


def receipt_path(cell, battery, spec, *, amended=False):
    return (OUT if amended else c.OUT) / 'runs' / cell / f'{battery}_{spec}.json'


def validate_receipt(path, reg, cell, battery, spec):
    row = c.read(path)
    assert (row['cell'], row['battery'], row['spec']) == (cell, battery, spec), 'Receipt identity mismatch'
    assert row['registration'] == c.record(c.REG), 'Mixed session registration'
    amended = path == receipt_path(cell, battery, spec, amended=True)
    assert row['sources'] == reg['sources' if amended else 'legacy_sources'], 'Mixed session source manifest'
    assert row['amendment'] == reg['amendment' if amended else 'legacy_amendment'], 'Mixed session amendment'
    return row


def eligible_suffix(reg, cell):
    # Prior art: C4 dependency-ordered units and immutable attempt claims
    # (house, 2026). This order permits only the exact root failure and its
    # transitive dependency suffix; any unrelated RED or orphan claim stops.
    assert cell in c.executable_cells(reg), 'Only registered new cells may execute'
    suffix = []
    for unit in lh_units(reg):
        spec = unit['spec']
        path = receipt_path(cell, 'longhorizon', spec)
        marker = path.with_suffix('.attempt.json')
        if marker.exists():
            assert path.is_file(), 'Unfinished worker claim: campaign blocked'
        row = validate_receipt(path, reg, cell, 'longhorizon', spec) if path.exists() else None
        if not suffix:
            if row is None:
                return []  # no evidenced root; do not authorize unstarted cells
            if row['status'] == 'PASS':
                continue
            assert row['status'] == 'RED' and row.get('error') == ROOT_ERROR, 'Other RED: registered stop'
        elif row:
            assert row['status'] == 'RED' and row.get('error') in (ROOT_ERROR, DEPENDENCY_ERROR), 'Other RED or existing PASS in recovery suffix'
        suffix.append(spec)
    return suffix


def selected_path(reg, cell, battery, spec):
    new = receipt_path(cell, battery, spec, amended=True)
    if new.exists():
        assert battery == 'longhorizon' and spec in eligible_suffix(reg, cell), 'Unregistered recovery receipt'
        return new
    return receipt_path(cell, battery, spec)


def pass_receipt(reg, cell, battery, spec):
    path = selected_path(reg, cell, battery, spec)
    assert path.exists(), 'Prior unit missing/RED'
    row = validate_receipt(path, reg, cell, battery, spec)
    assert row['status'] == 'PASS', 'Prior unit missing/RED'
    return path, row


def state_files(row):
    # Prior art: EB1 saved repository state and C4 SHA inventory (house, 2026).
    # Verify complete inventory, not just the listed subset, before copying.
    session = Path(row['result']['session_dir'])
    files = sorted(p for p in session.rglob('*') if p.is_file())
    assert files and row.get('session_state'), 'Missing bound session state'
    assert not any(p.is_symlink() for p in session.rglob('*')), 'Symlink in resumable state'
    assert [c.record(p) for p in files] == row['session_state'], 'Resumable state drift'
    return session


def saved_turns(session, stop):
    for name in ('instrumentation.jsonl', 'transcript.jsonl'):
        rows = [json.loads(line) for line in (session / name).read_text().splitlines()]
        assert [r['turn'] for r in rows] == list(range(stop)), 'Registered turn range mismatch'
    assert c.read(session / 'restart.json')['after_turn'] == stop - 1, 'Saved boundary mismatch'


def segment_guard(reg, cell, spec, payload, completed_turns):
    # Prior art: C4 observed-event guard (house, 2026). Move its existential
    # serving check to the registered cell boundary; retain exact advancement
    # and state binding for every segment. No scoring threshold changes.
    units = lh_units(reg)
    index = next(i for i, u in enumerate(units) if u['spec'] == spec)
    stop = units[index]['stop_after_turns']
    start = units[index - 1]['stop_after_turns'] if index else 0
    assert completed_turns == list(range(start, stop)), 'Registered turn range mismatch'
    assert payload['result']['stop_after_turns'] == stop, 'Registered stop mismatch'
    assert payload['result']['resumed'] == bool(index), 'Resume mode mismatch'
    session = state_files(payload)
    saved_turns(session, stop)
    assert any(e['event'] == 'geometry' for e in payload['events']), 'No arena observed'
    payload['turn_range'] = {'start_inclusive': start, 'stop_exclusive': stop, 'completed': completed_turns}
    if index == len(units) - 1:
        priors = [pass_receipt(reg, cell, 'longhorizon', u['spec']) for u in units[:index]]
        events = [e for _, r in priors for e in r['events']] + payload['events']
        count = sum(e['event'] == 'attempt_residency' for e in events)
        payload['cell_serving_guard'] = {'attempt_residency_count': count,
            'prior_receipts': [c.record(p) for p, _ in priors]}
        assert count > 0, 'No serving observed'


def accounting(reg):
    # Prior art: C4/DET1 persistent lease accounting (house, 2026). Count both
    # original failed work and the one amended attempt; never waive other REDs.
    workers = []
    for root, amended in ((c.OUT, False), (OUT, True)):
        for claim in (root / 'runs').glob('*/*.attempt.json'):
            assert claim.with_name(claim.name.replace('.attempt.json', '.json')).is_file(), 'Unfinished worker claim: campaign blocked'
        for path in (root / 'runs').glob('*/*.json'):
            row = c.read(path)
            if 'finished_unix' not in row:
                assert row.get('status') not in ('PASS', 'RED'), 'Incomplete worker receipt: campaign blocked'
                continue
            row = validate_receipt(path, reg, row['cell'], row['battery'], row['spec'])
            if row['status'] != 'PASS':
                assert not amended and row['battery'] == 'longhorizon' and row['spec'] in eligible_suffix(reg, row['cell']), 'Other RED: registered stop'
            workers.append(row)
    used = sum(r['gpu_seconds'] for r in workers)
    assert used + 285 <= reg['budget']['gpu_seconds'], 'GPU budget non-fit; no run'
    if workers:
        assert time.time() - max(r['finished_unix'] for r in workers) >= 30, '30s cooldown not met'
    return used


def recovery_ready(reg, cell, spec):
    assert spec in eligible_suffix(reg, cell), 'Unit not re-runnable under amendment 2'
    path = receipt_path(cell, 'longhorizon', spec, amended=True)
    assert not path.exists() and not path.with_suffix('.attempt.json').exists(), 'No retries or overwrite: amendment used once'
    units = lh_units(reg)
    index = next(i for i, u in enumerate(units) if u['spec'] == spec)
    # Every earlier unit is still required, including sup/census: no silently
    # skipped dependencies when a shell continued after the original RED.
    for unit in reg['units_per_new_cell']:
        if unit['battery'] == 'longhorizon' and unit['spec'] == spec:
            break
        pass_receipt(reg, cell, unit['battery'], unit['spec'])
    prior = pass_receipt(reg, cell, 'longhorizon', units[index-1]['spec']) if index else None
    if prior:
        session = state_files(prior[1])
        saved_turns(session, units[index-1]['stop_after_turns'])
    return path, prior


@contextmanager
def observe_turns(completed):
    from scripts import grm_e2e_session as e2e
    original = e2e.run_turn
    def observed(repo, event, turn_idx, **kwargs):
        result = original(repo, event, turn_idx, **kwargs)
        completed.append(int(turn_idx))
        return result
    # EB1 installs its stop wrapper around this observer, so the last turn is
    # recorded before DETStageStop. Restore process-local patches on exit.
    with patch.object(e2e, 'run_turn', observed):
        yield


def worker(cell, spec):
    reg = binding()
    receipt, prior = recovery_ready(reg, cell, spec)
    cfg = next(cel for cel in reg['cells'] if cel['id'] == cell)
    run_root = OUT / 'runs' / cell
    events, completed = [], []
    payload = {'cell': cell, 'battery': 'longhorizon', 'spec': spec,
        'registration': c.record(c.REG), 'amendment': reg['amendment'],
        'sources': reg['sources'], 'geometry_pin': reg['geometry'],
        'chunk': cfg['chunk'], 'width': cfg['width'], 'status': 'RED',
        'prior_receipt': c.record(prior[0]) if prior else None,
        'original_receipt': c.record(receipt_path(cell, 'longhorizon', spec)) if receipt_path(cell, 'longhorizon', spec).exists() else None,
        'events': events, 'evidence_class': 'GPU E2E worker; one-time lead amendment 2'}
    from scripts.grm_cmc1_gpu_arms import gpu_lease
    with gpu_lease(285, 0):
        recovery_ready(reg, cell, spec)
        payload['campaign_gpu_seconds_before'] = accounting(reg)
        c.write_once(receipt.with_suffix('.attempt.json'), {
            'cell': cell, 'battery': 'longhorizon', 'spec': spec,
            'registration': c.record(c.REG), 'amendment': reg['amendment'], 'started_unix': time.time()})
        started = time.monotonic()
        try:
            dest = run_root / 'longhorizon' / spec / 'session'
            assert not dest.parent.exists(), 'Recovery session already exists'
            if prior:
                source = state_files(prior[1])
                seed = run_root / 'longhorizon' / prior[1]['spec'] / 'session'
                if source != seed:
                    assert not seed.exists(), 'Recovery seed already exists'
                    shutil.copytree(source, seed)
                    for rec in prior[1]['session_state']:
                        copied = c.record(seed / Path(rec['path']).relative_to(source))
                        assert (copied['sha256'], copied['bytes']) == (rec['sha256'], rec['bytes']), 'Copied state drift'
            with c.harness(cfg['chunk'], cfg['width'], run_root, events) as (_, _, lh), observe_turns(completed):
                result = lh.run_shard(spec, run_dir=run_root/'longhorizon')
                payload['result'] = result
                payload['session_state'] = [c.record(p) for p in sorted(Path(result['session_dir']).rglob('*')) if p.is_file()]
                segment_guard(reg, cell, spec, payload, completed)
                if prior:
                    state_files(prior[1])
                binding()
            assert time.monotonic() - started <= 285, 'Worker rail exceeded: NON_FIT'
            payload['status'] = 'PASS'
        except BaseException as exc:
            payload['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            payload['gpu_seconds'] = time.monotonic() - started
            payload['finished_unix'] = time.time()
            c.write_once(receipt, payload)


# Prior art: C4 score implementation (house, 2026), copied verbatim below
# except receipt/source selection, final guard validation and isolated paths.
# Original census/LH scorers, score arithmetic, membership and bars unchanged.
from scripts.grm_c4_campaign import (executable_cells, read, record, REG, harness, write_once)

def score(cell):
    reg = binding()
    assert cell in executable_cells(reg), 'Only registered new cells may score'
    cfg = next(c for c in reg['cells'] if c['id'] == cell)
    run_root = c.OUT / 'runs' / cell
    recovery_root = OUT / 'runs' / cell
    receipts = []
    for unit in reg['units_per_new_cell']:
        _, r = pass_receipt(reg, cell, unit['battery'], unit['spec'])
        receipts.append(r)
    state_files(receipts[-1])
    assert receipts[-1].get('cell_serving_guard', {}).get('attempt_residency_count', 0) > 0, 'Missing per-cell serving guard'
    with harness(cfg['chunk'], cfg['width'], recovery_root, []) as (_, census, lh):
        census_result = census.score_run(run_root/'census', arm=1)
        l = lh.score_run(recovery_root/'longhorizon')
    sup = [p for r in receipts if r['battery'] == 'sup' for p in r['result']['probes']]
    assert len(sup) == 9 and len(census_result['probes']) == 10 and l['measured_count'] == 14
    assert all(p['turn_row_found'] for p in census_result['probes'])
    scores = {'sup': sum(p['verdict']['correct'] for p in sup),
              'census': sum(p['verdict']['correct'] for p in census_result['probes']),
              'longhorizon': l['correct_count']}
    measurements = {}
    for battery in reg['counts']:
        ev = [e for r in receipts if r['battery'] == battery for e in r['events']]
        attempts = [e for e in ev if e['event'] == 'attempt_residency']
        turns = [e for e in ev if e['event'] == 'turn']
        assert attempts, 'No actual token residency observations'
        guards = [e for e in ev if e['event'] == 'guard' and e['split']]
        measurements[battery] = {
            'attempt_observations': len(attempts),
            'mean_mounted_token_seats_per_attempt': sum(e['mounted_token_seats'] for e in attempts)/len(attempts),
            'max_mounted_token_seats': max(e['mounted_token_seats'] for e in attempts),
            'mean_mounted_token_seats_per_step': (sum(e['mounted_token_seats'] for e in turns)/len(turns) if turns else None),
            'step_observations': len(turns),
            'deposit_split_events': sum(e['phase'] == 'deposit' for e in guards),
            'fit_repair_split_events': sum(e['phase'] == 'explicit_budget_repair' for e in guards),
            'new_split_children': sum(len(e['split']['children']) for e in guards),
            'route_observations': [e['info'] for e in turns],
            'final_split_census_each_segment': [e for e in ev if e['event'] == 'split_census'],
        }
    # Prior art: WC1/EB1 original verdicts (2026). C4 reports ALL 14 rows;
    # no substitution of the four distant spot probes for full long-history.
    payload = {'cell': cell, 'registration': record(REG), 'sources': reg['sources'],
               'geometry': reg['geometry'], 'counts': reg['counts'], 'scores': scores,
               'measurements': measurements, 'amendment': reg.get('amendment'),
               'sup_probes': sup, 'census_probes': census_result['probes'], 'longhorizon_probes': l['probes'],
               'prediction_met_in_cell': scores == reg['counts'],
               'factorial_verdict': 'INCONCLUSIVE: per-cell score only; lead must compare all fixed-geometry cells',
               'measurement_receipts': [record(selected_path(reg, cell, u['battery'], u['spec'])) for u in reg['units_per_new_cell']],
               'measurement_note': 'Per-attempt and turn token residency, capture/seat positions, split events and original routing info in bound worker receipts; historical WC1 graft-count column not used.'}
    if cell == 'c64_w64' and 'diagonal_comparison' in reg:
        # Prior art: WC1 per-battery deltas (house, 2026). Report componentwise
        # differences from the historical diagonal; no causal verdict inferred.
        comparison = reg['diagonal_comparison']
        baseline = comparison['historical_scores']
        payload['diagonal_comparison'] = {
            'registration': comparison, 'fixed_geometry_scores': scores,
            'delta_correct': {b: scores[b] - baseline[b] for b in scores},
            'note': 'Different historical geometry; compare probe identities too. This is not support for chunking by itself.'}
    write_once(recovery_root / 'score.json', payload)
    return payload


def command_text(reg):
    # Prior art: WC1/C4 dependency-ordered foreground commands (house, 2026).
    lines = ['#!/bin/bash', 'set -e', f'cd {c.ROOT}',
             '# Execute only after the original lead chain has finished.',
             '# Only evidenced eligible long-history suffixes; unchanged budget/rails.']
    for cell in c.executable_cells(reg):
        suffix = eligible_suffix(reg, cell)
        if not suffix:
            continue
        for spec in suffix:
            lines += [f'timeout --signal=KILL 590s python scripts/grm_c4_resume_a2.py worker --cell {cell} --spec {spec}', 'sleep 30']
        lines += [f'python scripts/grm_c4_resume_a2.py score --cell {cell}']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preflight', 'worker', 'score'))
    parser.add_argument('--cell')
    parser.add_argument('--spec')
    args = parser.parse_args()
    if args.command == 'worker':
        worker(args.cell, args.spec)
    elif args.command == 'score':
        print(json.dumps(score(args.cell), indent=2))
    else:
        reg = binding()
        print(json.dumps({'amendment': reg['amendment'], 'gpu_executed': False,
            'eligible_suffixes': {cell: eligible_suffix(reg, cell) for cell in c.executable_cells(reg)},
            'budget': reg['budget']}, indent=2))


if __name__ == '__main__':
    main()
