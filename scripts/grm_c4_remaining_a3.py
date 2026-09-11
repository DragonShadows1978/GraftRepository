#!/usr/bin/env python3
"""Lead amendment 3: A4 sibling for the two unstarted C4 cells.

Prior art: local C4/A4, WC1, RS3, EB1 and DET1 (house, 2026): immutable
SHA manifests, create-only claims, ordered resume, accounting and scorers.
This sibling extends executor eligibility, imports the A4 cell guard and
original scorer, and assembles the registered comparison. No novel algorithm.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import grm_c4_campaign as c
from scripts import grm_c4_resume_a2 as a2
from scripts import grm_c4_cap_a4 as cap_a4

OUT = c.OUT / 'lead_a3'
ORDER = c.ROOT / 'orders/GRM_C4_AMENDMENT_3.md'
AMENDMENT = OUT / 'amendment_a5.json'
CELLS = ('c96_w64', 'c64_w64')


def schedule(reg):
    return [dict(cell=cell, **u) for cell in CELLS for u in reg['units_per_new_cell']]


def binding():
    # Prior art: A4 (house, 2026) nested immutable chain; retain A4 unchanged
    # so the already queued recovery commands still validate their sources.
    previous = a2.binding()
    # Prior art: A5 exact SHA closure (house, 2026); lead A4 cap overlay
    # retains A5 receipt identity while authorizing only named replacements.
    cap = cap_a4.binding()
    row = c.read(AMENDMENT)
    assert c.record(AMENDMENT)['sha256'] == AMENDMENT.with_suffix('.sha256').read_text().split()[0], 'amendment SHA mismatch'
    assert row['previous_amendment'] == previous['amendment'], 'amendment chain mismatch'
    assert row['order'] == c.record(ORDER), 'amendment order drift'
    assert row['registration'] == c.record(c.REG), 'amendment registration mismatch'
    assert row['scheduled_units'] == schedule(previous), 'Unregistered schedule'
    assert row['executor']['path'] == str(Path(__file__).resolve()), 'executor drift'
    cap_a4.check_source(row['executor'], cap)
    assert row['receipt_root'] == str(OUT / 'runs'), 'receipt layout drift'
    assert row['score_cells'] == list(CELLS), 'Unregistered scores'
    assert row['budget'] == cap['previous_budget'], 'budget drift'
    required = {r['path'] for r in previous['sources']} | {
        str(Path(__file__).resolve()), str(ORDER),
        str(c.ROOT/'scripts/grm_c4_register_a3.py'),
        str(c.ROOT/'tests/test_grm_c4_remaining_a3.py'),
        str(c.OUT/'lead_commands_a3.txt'), str(OUT/'pre_registration.json'),
        str(OUT/'original_receipts_before.json')}
    assert {r['path'] for r in row['sources']} == required, 'source closure mismatch'
    for source in row['sources']:
        cap_a4.check_source(source, cap)
    return {**previous, 'a4_context': previous, 'sources': row['sources'],
            'amendment': c.record(AMENDMENT)}


def receipt_path(cell, battery, spec):
    return OUT/'runs'/cell/f'{battery}_{spec}.json'


def validate_receipt(path, reg, cell, battery, spec):
    row = c.read(path)
    assert path == receipt_path(cell, battery, spec), 'Receipt path mismatch'
    assert (cell, battery, spec) in [(u['cell'], u['battery'], u['spec']) for u in schedule(reg)], 'Unregistered receipt'
    assert (row['cell'], row['battery'], row['spec']) == (cell, battery, spec), 'Receipt identity mismatch'
    assert row['registration'] == c.record(c.REG), 'Mixed registration'
    assert row['amendment'] == reg['amendment'] and row['sources'] == reg['sources'], 'Mixed amendment/source'
    cfg = next(v for v in reg['cells'] if v['id'] == cell)
    assert row['geometry_pin'] == reg['geometry'] and (row['chunk'], row['width']) == (cfg['chunk'], cfg['width']), 'Mixed geometry'
    return row


def pass_receipt(reg, cell, battery, spec):
    path = receipt_path(cell, battery, spec)
    assert path.is_file(), 'Prior unit missing/RED'
    row = validate_receipt(path, reg, cell, battery, spec)
    assert row['status'] == 'PASS', 'Prior unit missing/RED'
    return path, row


def score_path(cell):
    return (a2.OUT if cell == 'c64_w96' else OUT)/'runs'/cell/'score.json'


def load_score(reg, cell):
    path = score_path(cell)
    assert path.is_file(), 'Prior cell score missing'
    row = c.read(path)
    context = reg['a4_context'] if cell == 'c64_w96' else reg
    assert row['cell'] == cell and row['registration'] == c.record(c.REG), 'Mixed score identity'
    assert row['sources'] == context['sources'] and row['amendment'] == context['amendment'], 'Mixed score source/amendment'
    assert row['geometry'] == reg['geometry'] and row['counts'] == reg['counts'], 'Mixed score geometry/counts'
    select = a2.pass_receipt if cell == 'c64_w96' else pass_receipt
    receipts = [select(context, cell, u['battery'], u['spec']) for u in reg['units_per_new_cell']]
    assert row['measurement_receipts'] == [c.record(p) for p, _ in receipts], 'Stale score receipts'
    a2.state_files(receipts[-1][1])
    assert receipts[-1][1].get('cell_serving_guard', {}).get('attempt_residency_count', 0) > 0, 'Missing per-cell serving guard'
    return row


def accounting(reg):
    # Prior art: A4/DET1 (house, 2026) charge original RED walls and all
    # successful/recovery work. Scan each location once; other REDs stop.
    old = reg['a4_context']
    workers = []
    for root in (c.OUT, a2.OUT, OUT):
        for claim in (root/'runs').glob('*/*.attempt.json'):
            assert claim.with_name(claim.name.replace('.attempt.json', '.json')).is_file(), 'Unfinished worker claim: campaign blocked'
        for path in (root/'runs').glob('*/*.json'):
            row = c.read(path)
            if 'finished_unix' not in row:
                assert row.get('status') not in ('PASS', 'RED'), 'Incomplete worker receipt'
                continue
            identity = (row['cell'], row['battery'], row['spec'])
            if root == OUT:
                row = validate_receipt(path, reg, *identity)
            else:
                assert path == a2.receipt_path(*identity, amended=root == a2.OUT), 'Receipt path mismatch'
                row = a2.validate_receipt(path, old, *identity)
            if row['status'] != 'PASS':
                assert root == c.OUT and row['cell'] == 'c64_w96' and row['battery'] == 'longhorizon' and row['spec'] in a2.eligible_suffix(old, row['cell']), 'Other RED: registered stop'
            assert math.isfinite(row['gpu_seconds']) and row['gpu_seconds'] >= 0, 'Invalid wall accounting'
            assert math.isfinite(row['finished_unix']), 'Invalid finish time'
            workers.append(row)
    used = sum(r['gpu_seconds'] for r in workers)
    assert used + 285 <= reg['budget']['gpu_seconds'], 'GPU budget non-fit; no run'
    if workers:
        assert time.time() - max(r['finished_unix'] for r in workers) >= 30, '30s cooldown not met'
    return used


def ready(reg, cell, battery, spec):
    assert (cell, battery, spec) in [(u['cell'], u['battery'], u['spec']) for u in schedule(reg)], 'Unit not authorized'
    path = receipt_path(cell, battery, spec)
    for root in (c.OUT, a2.OUT, OUT):
        other = root/'runs'/cell/path.name
        assert not other.exists() and not other.with_suffix('.attempt.json').exists(), 'No retries or overwrite'
    for earlier in c.executable_cells(reg):
        if earlier == cell:
            break
        load_score(reg, earlier)
    prior = None
    for unit in reg['units_per_new_cell']:
        if (unit['battery'], unit['spec']) == (battery, spec):
            break
        p, row = pass_receipt(reg, cell, unit['battery'], unit['spec'])
        if unit['battery'] == battery:
            prior = p, row
    if prior and battery in ('census', 'longhorizon'):
        session = a2.state_files(prior[1])
        if battery == 'longhorizon':
            a2.saved_turns(session, next(u['stop_after_turns'] for u in a2.lh_units(reg) if u['spec'] == prior[1]['spec']))
    root = OUT/'runs'/cell
    dest = root/battery/('arm1' if battery == 'census' else '')/spec/'session'
    assert not dest.parent.exists(), 'Session already exists; no overwrite'
    return path, prior


def worker(cell, battery, spec):
    reg = binding()
    receipt, prior = ready(reg, cell, battery, spec)
    cfg = next(v for v in reg['cells'] if v['id'] == cell)
    root = OUT/'runs'/cell
    events, completed = [], []
    payload = {'cell': cell, 'battery': battery, 'spec': spec, 'status': 'RED',
        'registration': c.record(c.REG), 'amendment': reg['amendment'], 'sources': reg['sources'],
        'budget_amendment': reg.get('budget_amendment'), 'budget': reg['budget'],
        'geometry_pin': reg['geometry'], 'chunk': cfg['chunk'], 'width': cfg['width'],
        'prior_receipt': c.record(prior[0]) if prior else None, 'events': events,
        'evidence_class': 'GPU E2E worker; A4 sibling, lead amendment 3'}
    from scripts.grm_cmc1_gpu_arms import gpu_lease
    with gpu_lease(285, 0):
        ready(reg, cell, battery, spec)
        payload['campaign_gpu_seconds_before'] = accounting(reg)
        c.write_once(receipt.with_suffix('.attempt.json'), {
            'cell': cell, 'battery': battery, 'spec': spec,
            'budget_amendment': reg.get('budget_amendment'),
            'registration': c.record(c.REG), 'amendment': reg['amendment'], 'started_unix': time.time()})
        started = time.monotonic()
        try:
            with c.harness(cfg['chunk'], cfg['width'], root, events) as (replay, census, lh), a2.observe_turns(completed):
                if battery == 'sup':
                    result = replay.serve_fixture(spec, arm=1, capture_pin='live', seat_near_live=True)
                elif battery == 'census':
                    result = census.run_shard(spec, arm=1, run_dir=root/'census', capture_pin='live', seat_near_live=True)
                else:
                    result = lh.run_shard(spec, run_dir=root/'longhorizon')
                payload['result'] = result
                if 'session_dir' in result:
                    payload['session_state'] = [c.record(p) for p in sorted(Path(result['session_dir']).rglob('*')) if p.is_file()]
                    a2.state_files(payload)
                if battery == 'longhorizon':
                    with patch.object(a2, 'pass_receipt', pass_receipt):
                        a2.segment_guard(reg, cell, spec, payload, completed)
                else:
                    assert any(e['event'] == 'geometry' for e in events), 'No arena observed'
                    assert any(e['event'] == 'attempt_residency' for e in events), 'No serving observed'
                if prior and battery in ('census', 'longhorizon'):
                    a2.state_files(prior[1])
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


def score(cell):
    # Prior art: C4 score (house, 2026), imported without arithmetic edits.
    # All batteries now share one root, including census and final LH state.
    reg = binding()
    assert cell in CELLS, 'Unit not authorized'
    rows = [pass_receipt(reg, cell, u['battery'], u['spec'])[1] for u in reg['units_per_new_cell']]
    a2.state_files(rows[-1])
    assert rows[-1].get('cell_serving_guard', {}).get('attempt_residency_count', 0) > 0, 'Missing per-cell serving guard'
    with patch.object(c, 'OUT', OUT), patch.object(c, 'binding', lambda: reg):
        return c.score(cell)


def probe_outcomes(row, reg):
    # Prior art: WC1/EB1 probe identity comparisons (house, 2026). Require
    # complete registered membership before deriving the named prediction.
    sup, census, lh = row['sup_probes'], row['census_probes'], row['longhorizon_probes']
    expected_sup = [p['probe_id'] for ps in reg['fixtures']['sup'].values() for p in ps]
    for rows, key, expected in ((sup, 'probe_id', expected_sup),
            (census, 'probe_id', [p['probe_id'] for p in reg['fixtures']['census']]),
            (lh, 'turn', [p['turn'] for p in reg['fixtures']['longhorizon']['probes']])):
        assert sorted(p[key] for p in rows) == sorted(expected), 'Probe membership mismatch'
    assert all(p['turn_row_found'] for p in census + lh), 'Missing probe observations'
    assert row['scores'] == {'sup': sum(p['verdict']['correct'] for p in sup),
        'census': sum(p['verdict']['correct'] for p in census), 'longhorizon': sum(p['correct'] for p in lh)}, 'Score/probe mismatch'
    return {'sup_reserve_juniper_pass': next(p['verdict']['correct'] for p in sup if p['probe_id'] == 'sup_reserve_juniper_pass'),
        'e2e_t33_polaris_mark': next(p['verdict']['correct'] for p in census if p['probe_id'] == 'e2e_t33_polaris_mark'),
        'longhorizon_turn_33': next(p['correct'] for p in lh if p['turn'] == 33)}


def cross_summary():
    # Prior art: registered C4 crossed controls and WC1 componentwise deltas
    # (house, 2026). No new statistical test or causal proof is introduced.
    reg = binding()
    assert c.record(reg['historical_results']['path']) == reg['historical_results'], 'Historical result drift'
    historical = c.read(reg['historical_results']['path'])
    def wc_scores(width):
        detail = historical['detail'][str(width)]
        correct, total = map(int, detail['longhorizon']['all_probes'].split('/'))
        assert total == 14, 'Historical full LH count mismatch'
        return {'sup': detail['sup']['correct'], 'census': detail['census']['correct'], 'longhorizon': correct}
    def wc_probes(width):
        # Prior art: WC1 archived per-probe records (house, 2026); resolve
        # archive relocation while preserving each original content hash.
        batteries, pins = {}, []
        for battery, detail in historical['detail'][str(width)].items():
            probes = []
            for source in detail['sources']:
                path = c.ARCHIVE/Path(source['path']).relative_to('artifacts/grm_wc1')
                pin = c.record(path)
                assert (pin['sha256'], pin['bytes']) == (source['sha256'], source['bytes']), 'Historical probe drift'
                assert pin in reg['sources'], 'Unbound historical probe source'
                payload = c.read(path)
                probes.extend(payload['all_probes'] if battery == 'longhorizon' else
                    [dict(p, verdict={'correct':p['correct']}) for p in payload['rows']])
                pins.append(pin)
            batteries[battery+'_probes'] = probes
        return probe_outcomes(dict(batteries, scores=wc_scores(width)), reg), pins
    assert reg['diagonal_ruling']['decision'] == 'CITE_WC1', 'Diagonal not authorized'
    assert wc_scores(96) == reg['diagonal_ruling']['scores'], 'WC1 diagonal mismatch'
    assert wc_scores(64) == reg['diagonal_comparison']['historical_scores'], 'WC1 historical mismatch'
    cells, required = {}, {}
    for cell in c.executable_cells(reg):
        row = load_score(reg, cell)
        required[cell] = probe_outcomes(row, reg)
        assert all(row['measurements'][b]['attempt_observations'] > 0 for b in reg['counts']), 'Missing residency observations'
        cells[cell] = {'scores': row['scores'], 'receipt': c.record(score_path(cell)), 'required_probes': required[cell]}
    historical_required, historical_pins = wc_probes(64)
    diagonal_required, diagonal_pins = wc_probes(96)
    cells['c96_w96'] = {'scores': wc_scores(96), 'receipt': reg['historical_results'],
        'required_probes': diagonal_required, 'probe_receipts': diagonal_pins,
        'evidence_class': 'WC1 archived E2E; registered geometry reuse; token residency unavailable'}
    wide = cells['c64_w96']['scores']
    met = wide == reg['counts'] and all(required['c64_w96'].values())
    # Registered rejection specifically concerns needing narrower seating.
    # Failure without a narrow recovery stays MIXED, not a refutation.
    wide_core = wide['longhorizon'] == 14 and all(required['c64_w96'].values())
    narrow_recovers = any(cells[x]['scores']['longhorizon'] == 14 and
        all(required[x].values()) for x in CELLS)
    verdict = 'SUPPORTED_FINITE' if met else ('REJECTED_FINITE' if not wide_core and narrow_recovers else 'MIXED')
    fixed, historic = cells['c64_w64']['scores'], wc_scores(64)
    result = {'evidence_class': 'CPU aggregation of bound E2E scores; finite registered batteries',
        'registration': c.record(c.REG), 'amendment': reg['amendment'], 'cells': cells,
        'counts': reg['counts'], 'prediction': reg['prediction'], 'prediction_met': met,
        'factorial_verdict': verdict, 'rejection': reg['rejection'],
        'diagonal_comparison': {'fixed': fixed, 'historical': historic,
            'delta_correct': {b: fixed[b]-historic[b] for b in fixed},
            'fixed_required_probes': required['c64_w64'],
            'historical_required_probes': historical_required, 'historical_probe_receipts': historical_pins,
            'note': reg['diagonal_comparison']['interpretation']},
        'limits': 'Missing or stale cells/observations refuse summary: INCONCLUSIVE. WC1 token residency unavailable. No universal or statistical causal claim.'}
    c.write_once(OUT/'cross_summary.json', result)
    return result


def format_summary(result):
    def scores(row):
        return ', '.join(f"{row[b]}/{result['counts'][b]}" for b in ('sup','census','longhorizon'))
    lines = ['Scores: sup, census, full long-history', 'chunk | width 64 | width 96', '--- | --- | ---']
    for chunk in (64, 96):
        lines.append(f"{chunk} | {scores(result['cells'][f'c{chunk}_w64']['scores'])} | {scores(result['cells'][f'c{chunk}_w96']['scores'])}")
    d = result['diagonal_comparison']
    lines += [f"(96,96): cited WC1 row (full 14-probe long-history).",
        f"(64,64) fixed: {scores(d['fixed'])}; historical: {scores(d['historical'])}; delta correct: {d['delta_correct']}",
        f"(64,64) required probes fixed: {d['fixed_required_probes']}; historical: {d['historical_required_probes']}",
        f"Registered prediction: {result['prediction']}",
        f"Prediction met: {result['prediction_met']}; outcome: {result['factorial_verdict']}"]
    return '\n'.join(lines)


def command_text(reg):
    # Prior art: A4/C4 foreground leased schedule (house, 2026).
    lines = ['#!/bin/bash', 'set -e', f'cd {c.ROOT}',
        '# Run after lead_commands_a2.txt completes successfully; do not overlap.',
        'python scripts/grm_c4_remaining_a3.py --dry-run', 'sleep 30']
    for cell in CELLS:
        for u in reg['units_per_new_cell']:
            lines += [f"timeout --signal=KILL 590s python scripts/grm_c4_remaining_a3.py worker --cell {cell} --battery {u['battery']} --spec {u['spec']}", 'sleep 30']
        lines += [f'python scripts/grm_c4_remaining_a3.py score --cell {cell}']
    return '\n'.join(lines + ['python scripts/grm_c4_remaining_a3.py summary']) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', default='preflight', choices=('preflight','worker','score','summary'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--cell', choices=CELLS)
    parser.add_argument('--battery', choices=('sup','census','longhorizon'))
    parser.add_argument('--spec')
    args = parser.parse_args()
    if args.dry_run or args.command == 'preflight':
        reg = binding()
        units = schedule(reg)
        print(json.dumps({'gpu_executed': False, 'amendment': reg['amendment'],
            'budget_amendment': reg['budget_amendment'],
            'scheduled_units': units, 'unit_count': len(units),
            'estimate_seconds': sum(u['estimate_seconds'] for u in units),
            'cooldown_seconds': 30*(len(units)+1), 'budget': reg['budget']}, indent=2))
    elif args.command == 'worker':
        worker(args.cell, args.battery, args.spec)
    elif args.command == 'score':
        print(json.dumps(score(args.cell), indent=2))
    else:
        print(format_summary(cross_summary()))


if __name__ == '__main__':
    main()
