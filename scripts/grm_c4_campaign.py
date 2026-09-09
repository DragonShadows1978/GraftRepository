#!/usr/bin/env python3
"""C4 register / dry-run / one leased worker / CPU score, additive imports only.

Prior art: WC1 (house, 2026) import/rebinding harness; EB1 and DET1 (house,
2026) resumable session segments, semantic scoring, and leased foreground
execution. C4 adds the crossed control schedule, SHA binding and observations.
No novel algorithm claimed. See registration and IMPLEMENTATION_LEDGER.md.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.grm_c4_adapter import deposit_chunking, fixed_geometry

SHARED = Path('/mnt/ForgeRealm/GraftRepository')
OUT = ROOT / 'artifacts/grm_c4'
REG = OUT / 'registration.json'
ARCHIVE = SHARED / 'artifacts/grm_wc1_opus'
DET = SHARED / 'artifacts/grm_det1/run_20260831T160525Z_2'
FRAME = DET / 'runtime_frame_28b3196f8fb04a41.json'
CENSUS = DET / 'det1_11/census/lived_serving_census_98ef71e88dec17a1.json'
MANIFEST = ARCHIVE / 'runs/w96/census_session/arm1/e2e-4/session/repository/manifest.json'
SUP = ('correction_then_restatement', 'fresh_fact_controls',
       'multi_hop_a_b_c', 'short_correction_long_competitor')
CENSUS_SPECS = ('e2e-1', 'e2e-2', 'e2e-3', 'e2e-4')
LH_STOPS = dict((f'c4-lh-{i+1:02}', stop) for i, stop in enumerate(range(8, 105, 8)))
NEW_CELLS = ('c64_w96', 'c96_w64')


def read(path):
    return json.loads(Path(path).read_text())


def record(path):
    p = Path(path)
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
            'bytes': p.stat().st_size}


def write_once(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write('\n')


def inventory():
    # Prior art: WC1 source receipts (2026). Conservative execution closure:
    # all local Python source in core/scripts, plus native library, fixtures,
    # model metadata, and frozen evidence. Model payload hashes are inherited
    # only where the existing frame provides them; no invented model checksum.
    paths = set(ROOT.glob('core/**/*.py')) | set(ROOT.glob('scripts/**/*.py'))
    paths |= set((ROOT / 'tests/fixtures/supersession_battery').glob('*.json'))
    paths |= set(ROOT.glob('tests/test_grm_c4*.py'))
    paths |= {FRAME, CENSUS, MANIFEST, SHARED / read(FRAME)['native_library']['path'],
              ARCHIVE / 'registration.json', ARCHIVE / 'grm_wc1_results.json'}
    model = Path(read(FRAME)['model']['path'])
    paths |= set(model.glob('*.json'))
    return [record(p) for p in sorted(paths)]


def historic_fingerprint():
    rows = []
    for r in read(ARCHIVE / 'registration.json')['sources'].values():
        p = ROOT / r['path']
        if not p.is_file():
            p = SHARED / r['path']
        actual = record(p) if p.is_file() else None
        rows.append({'wc1': r, 'resolved': actual,
                     'match': bool(actual and actual['sha256'] == r['sha256'])})
    return rows


def units():
    return ([{'battery': 'sup', 'spec': s, 'estimate_seconds': 60} for s in SUP]
            + [{'battery': 'census', 'spec': s, 'estimate_seconds': 80} for s in CENSUS_SPECS]
            + [{'battery': 'longhorizon', 'spec': s, 'stop_after_turns': n,
                'resume': list(LH_STOPS)[i-1] if i else None,
                'estimate_seconds': 80} for i, (s, n) in enumerate(LH_STOPS.items())])


def register():
    from scripts import grm_eb1_longhorizon_gpu as lh
    from scripts import lsr_p2c_replay_gpu as replay
    from scripts import lsr_p2c_e2e_gpu as census
    with patch.object(replay, 'CENSUS', CENSUS), patch.object(census, 'CENSUS', CENSUS):
        sup_probes = {s: replay.sup_probe_plan(s) for s in SUP}
        census_probes = census.e2e_census_probes()
        long_plan = lh.extension_plan()
    fp = historic_fingerprint()
    match = all(r['match'] for r in fp)
    estimated = sum(u['estimate_seconds'] for u in units())
    payload = {
        'schema': 'grm.c4.registration.v1', 'immutable': True,
        'order': 'User GRM-C4 order; immutable mission captured in IMPLEMENTATION_PLAN.md',
        'model_id': 'GPT-6 (Codex; no more specific deployment id exposed)',
        'reasoning_effort': 'high (order)', 'reader_model': read(FRAME)['model'],
        'chunk_flag': {'name': 'GRM_C4_DEPOSIT_CHUNK_TOKENS', 'default': 'OFF',
                       'scope': 'Only inside additive deposit_chunking context; no core edits',
                       'values': [64, 96], 'explicit_fit_budget_unchanged': True},
        'geometry': {'capture_pin': 'live', 'seat_near_live': True,
                     'live_shift': 'n_sink + 96', 'plan_head_last_position': 'n_sink + 95',
                     'sink': 'unchanged tokenizer-derived n_sink',
                     'live_turns': 2, 'max_live': 4096, 'ephemeral': True,
                     'remaining_flags': read(FRAME)['resolved_flags']},
        'cells': [
            {'id': 'c64_w64', 'chunk': 64, 'width': 64, 'status': 'HISTORICAL_ONLY_GEOMETRY_MISMATCH',
             'estimate_seconds': estimated, 'equal_standing': True,
             'reason': 'WC1 captured at n_sink+64. A fixed-geometry replacement is required for a full factorial; full campaign is budget non-fit.'},
            {'id': 'c96_w96', 'chunk': 96, 'width': 96,
             'status': 'CITE_WC1' if match else 'RED_FINGERPRINT_MISMATCH',
             'estimate_seconds': 0, 'equal_standing': True},
            *[{'id': c, 'chunk': int(c[1:3]), 'width': int(c[-2:]),
               'status': 'NEW', 'estimate_seconds': estimated, 'equal_standing': True} for c in NEW_CELLS]],
        'units_per_new_cell': units(), 'counts': {'sup': 9, 'census': 10, 'longhorizon': 14},
        'fixtures': {'sup': sup_probes, 'census': census_probes, 'longhorizon': long_plan},
        'prediction': 'c64_w96 recovers t33, reaches 14/14 long-history, and keeps Juniper',
        'acceptance': {'sup': 9, 'census': 10, 'longhorizon': 14,
                       'required_probes': ['sup_reserve_juniper_pass', 'e2e_t33_polaris_mark', 'longhorizon_turn_33']},
        'rejection': 'Chunking claim rejected if improvement needs narrower seating; wide chunk64 must recover t33 and retain Juniper. Missing cells/geometry/observations => INCONCLUSIVE, never support.',
        'decision_categories': ['SUPPORTED_FINITE', 'REJECTED_FINITE', 'MIXED', 'INCONCLUSIVE'],
        'budget': {'gpu_seconds': 3600, 'worker_seconds': 285, 'outer_seconds': 590,
                   'cooldown_seconds': 30, 'lock': '/tmp/forge-gpu.lock',
                   'lock_wait_seconds': 0, 'new_cells_estimate_seconds': 2*estimated,
                   'full_fixed_geometry_estimate_seconds': 3*estimated,
                   'full_fixed_geometry_status': 'NON_FIT',
                   'estimate_class': 'Planning assumption: sup 60s/fixture, census and 8-turn LH segments 80s each including load/save; unvalidated on this sandbox; stop at rail without retry.',
                   'stop': 'One timeout stops its chain; no retry or longer lease. Reserve full 285s before each worker against remaining total.'},
        'historical_fingerprints': fp,
        'historical_results': record(ARCHIVE / 'grm_wc1_results.json'),
        'historic_metrics': 'WC1 graft-count seat column INVALID; no historical actual token residency claimed. C64W64 historical scores are not a fixed-geometry control.',
        'gates': ['synthetic 64 split with exact parent preservation',
                  'OFF callable identity and byte-identical existing manifest behavior',
                  '96 chunk /64 width not clamped at deposit; explicit fit budget preserved',
                  'fixed numerical capture/live/seat positions across widths',
                  'actual residency sums ntok and checks cur_mount_n; not len(ids)',
                  'frozen 9/10/14 probe membership and full 104 turns',
                  'immutable SHA/source drift refusal and missing result refusal'],
        'prior_art': 'Verified local: LSR-P2C, WC1, RS3, EB1, DET1 house systems (2026). Borrow split, placement, replay, save/restore, scorer and lease. C4 adds crossed controls and measurements. No new algorithm claim. External factorial lead: Fisher, The Design of Experiments (1935), unverified — lead to check; search Fisher factorial experiments 1935.',
        'sources': inventory(),
    }
    write_once(REG, payload)
    with (OUT / 'registration.sha256').open('x') as f:
        f.write(record(REG)['sha256'] + '  registration.json\n')
    return payload


def binding():
    expected = (OUT / 'registration.sha256').read_text().split()[0]
    assert record(REG)['sha256'] == expected, 'registration SHA mismatch'
    reg = read(REG)
    previous = None
    for amendment in sorted(OUT.glob('amendment_a*.json')):
        expected_amendment = amendment.with_suffix('.sha256').read_text().split()[0]
        assert record(amendment)['sha256'] == expected_amendment, 'amendment SHA mismatch'
        update = read(amendment)
        assert update['registration'] == record(REG), 'amendment registration mismatch'
        if previous is not None:
            assert update['previous_amendment'] == previous, 'amendment chain mismatch'
        # Prior art: RS3/WC1 immutable amendments (house, 2026). Lead amendment
        # 1 authorizes only the added diagonal, budget and comparison schedule;
        # fixtures, geometry, prediction and rejection remain the base values.
        if 'overrides' in update:
            assert record(update['order']['path']) == update['order'], 'amendment order drift'
            allowed = {'cells', 'budget', 'execution_order', 'diagonal_comparison', 'diagonal_ruling'}
            assert set(update['overrides']) <= allowed, 'unapproved amendment field'
            reg = {**reg, **update['overrides']}
        previous = record(amendment)
        reg = {**reg, 'sources': update['sources'], 'amendment': previous}
    for r in reg['sources']:
        assert record(r['path']) == r, f"source drift: {r['path']}"
    return reg


def executable_cells(reg):
    # Prior art: WC1 registered cell enumeration (house, 2026). Use the active
    # SHA-bound schedule so merely adding a CLI choice cannot authorize a run.
    cells = tuple(c['id'] for c in reg['cells'] if c['status'] == 'NEW')
    order = tuple(reg.get('execution_order', cells))
    assert len(order) == len(cells) and set(order) == set(cells)
    return order


@contextmanager
def harness(chunk, width, run_root, events):
    from scripts import grm_wc1_sweep_gpu as wc
    from scripts import grm_det1_2_gpu as loader
    from scripts import lsr_p2c_replay_gpu as replay
    from scripts import lsr_p2c_e2e_gpu as census
    from scripts import grm_eb1_longhorizon_gpu as lh
    frame = read(FRAME)
    frame['resolved_flags']['arena_width'] = width
    frame['native_library']['path'] = str(SHARED / frame['native_library']['path'])
    frame_path = run_root / 'runtime_frame.json'
    if not frame_path.exists():
        write_once(frame_path, frame)
    assert read(frame_path) == frame
    with ExitStack() as stack:
        for module in (replay, census, lh):
            stack.enter_context(patch.object(module, 'RUNTIME_FRAME', frame_path))
        for module in (replay, census):
            stack.enter_context(patch.object(module, 'CENSUS', CENSUS))
        stack.enter_context(patch.object(lh, 'SHARDS', tuple(LH_STOPS)))
        stack.enter_context(patch.object(lh, 'SHARD_STOPS', LH_STOPS))
        stack.enter_context(patch.dict(os.environ, {
            'GRM_LSR_FIXES': '1', 'GRM_CAPTURE_PIN': 'live',
            'GRM_SEAT_NEAR_LIVE': '1', 'GRM_PERSISTENT_BOAT': '0',
            'GRM_GQA_CUDA_ROUTE': '1', 'GRM_GRAFT_STORAGE_BITS': '8',
            'GRM_ROUTE_QUERY_LEX': '1', 'GRM_PROBE_LADDER': '1',
            'GRM_SUP_RESOLVE': '1', 'GRM_ADM_DECISIVE': '1'}))
        stack.enter_context(patch.object(loader, 'NATIVE_LIB', Path(frame['native_library']['path'])))
        stack.enter_context(patch.object(loader, 'MODEL_DIR', Path(frame['model']['path'])))
        stack.enter_context(wc.width_patched_loader(width))
        stack.enter_context(fixed_geometry(width, events))
        stack.enter_context(deposit_chunking(chunk, events))
        yield replay, census, lh


def worker(cell, battery, spec):
    reg = binding()
    assert cell in executable_cells(reg), 'Only registered new cells may execute'
    cfg = next(c for c in reg['cells'] if c['id'] == cell)
    candidates = [u for u in reg['units_per_new_cell'] if u['battery'] == battery]
    index = next(i for i, u in enumerate(candidates) if u['spec'] == spec)
    run_root = OUT / 'runs' / cell
    receipt = run_root / f'{battery}_{spec}.json'
    marker = run_root / f'{battery}_{spec}.attempt.json'
    assert not receipt.exists() and not marker.exists(), 'No retries or overwrite'
    prior = None
    if index:
        prior = run_root / f"{battery}_{candidates[index-1]['spec']}.json"
        assert prior.exists() and read(prior)['status'] == 'PASS', 'Prior unit missing/RED'
        previous = read(prior)
        assert previous['registration'] == record(REG), 'Mixed session registration'
        assert previous['sources'] == reg['sources'], 'Mixed session source manifest'
        for state in previous.get('session_state', []):
            assert record(state['path']) == state, 'Resumable state drift'

    events = []
    payload = {'cell': cell, 'battery': battery, 'spec': spec, 'registration': record(REG),
               'geometry_pin': reg['geometry'], 'chunk': cfg['chunk'], 'width': cfg['width'],
               'sources': reg['sources'], 'prior_receipt': record(prior) if prior else None,
               'events': events, 'status': 'RED', 'amendment': reg.get('amendment'), 'evidence_class': 'GPU E2E worker'}
    from scripts.grm_cmc1_gpu_arms import gpu_lease
    # Prior art: WC1 foreground lease (2026). Fail immediately on lock contention;
    # never clear locks or signal other processes. Leased time excludes cooldown.
    with gpu_lease(285, 0):
        # Prior art: persistent claim/receipt bookkeeping from house DET1
        # (2026). An abandoned claim is unknown GPU usage, never zero usage.
        for claim in (OUT / 'runs').glob('*/*.attempt.json'):
            completed = claim.with_name(claim.name.replace('.attempt.json', '.json'))
            assert completed.is_file(), 'Unfinished worker claim: campaign blocked'
        existing = [read(p) for p in (OUT / 'runs').glob('*/*_*.json') if p.name != 'runtime_frame.json']
        workers = [r for r in existing if 'finished_unix' in r]
        assert not any(r['status'] != 'PASS' for r in workers), 'Campaign RED: no continuation without amendment'
        used = sum(r['gpu_seconds'] for r in workers)
        assert used + 285 <= reg['budget']['gpu_seconds'], 'GPU budget non-fit; no run'
        if workers:
            assert time.time() - max(r['finished_unix'] for r in workers) >= 30, '30s cooldown not met'
        # A killed/timed-out worker cannot silently reuse and delete partial
        # session data on a second invocation. This claim is never cleared.
        write_once(marker, {'registration': record(REG), 'cell': cell,
                            'battery': battery, 'spec': spec, 'started_unix': time.time()})
        started = time.monotonic()
        try:
            with harness(cfg['chunk'], cfg['width'], run_root, events) as (replay, census, lh):
                if battery == 'sup':
                    result = replay.serve_fixture(spec, arm=1, capture_pin='live', seat_near_live=True)
                elif battery == 'census':
                    result = census.run_shard(spec, arm=1, run_dir=run_root/'census', capture_pin='live', seat_near_live=True)
                else:
                    result = lh.run_shard(spec, run_dir=run_root/'longhorizon')
                payload['result'] = result
                # Bind resumable state, including full repository files, before
                # another unit consumes it. Never reconstruct from final counts.
                if 'session_dir' in result:
                    payload['session_state'] = [record(p) for p in sorted(Path(result['session_dir']).rglob('*')) if p.is_file()]
                assert any(e['event'] == 'geometry' for e in events), 'No arena observed'
                assert any(e['event'] == 'attempt_residency' for e in events), 'No serving observed'
                binding()
            assert time.monotonic() - started <= 285, 'Worker rail exceeded: NON_FIT'
            payload['status'] = 'PASS'
        except BaseException as exc:
            payload['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            payload['gpu_seconds'] = time.monotonic() - started
            payload['finished_unix'] = time.time()
            write_once(receipt, payload)


def score(cell):
    reg = binding()
    assert cell in executable_cells(reg), 'Only registered new cells may score'
    cfg = next(c for c in reg['cells'] if c['id'] == cell)
    run_root = OUT / 'runs' / cell
    receipts = []
    for unit in units():
        r = read(run_root / f"{unit['battery']}_{unit['spec']}.json")
        assert r['status'] == 'PASS' and r['registration'] == record(REG), 'Incomplete/mixed chain'
        assert r['sources'] == reg['sources'], 'Mixed score source manifest'
        receipts.append(r)
    with harness(cfg['chunk'], cfg['width'], run_root, []) as (_, census, lh):
        c = census.score_run(run_root/'census', arm=1)
        l = lh.score_run(run_root/'longhorizon')
    sup = [p for r in receipts if r['battery'] == 'sup' for p in r['result']['probes']]
    assert len(sup) == 9 and len(c['probes']) == 10 and l['measured_count'] == 14
    assert all(p['turn_row_found'] for p in c['probes'])
    scores = {'sup': sum(p['verdict']['correct'] for p in sup),
              'census': sum(p['verdict']['correct'] for p in c['probes']),
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
               'sup_probes': sup, 'census_probes': c['probes'], 'longhorizon_probes': l['probes'],
               'prediction_met_in_cell': scores == reg['counts'],
               'factorial_verdict': 'INCONCLUSIVE: per-cell score only; lead must compare all fixed-geometry cells',
               'measurement_receipts': [record(run_root / f"{u['battery']}_{u['spec']}.json") for u in units()],
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
    write_once(run_root / 'score.json', payload)
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['register', 'preflight', 'worker', 'score'], nargs='?', default='preflight')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--cell', choices=('c64_w96', 'c96_w64', 'c64_w64', 'c96_w96'))
    p.add_argument('--battery', choices=['sup', 'census', 'longhorizon'])
    p.add_argument('--spec')
    a = p.parse_args()
    if a.command == 'register':
        result = register()
    elif a.dry_run or a.command == 'preflight':
        r = binding()
        result = {k: r[k] for k in ('cells', 'units_per_new_cell', 'counts', 'geometry', 'budget')}
        result['registration'] = record(REG)
        result['amendment'] = r.get('amendment')
        result['execution_order'] = executable_cells(r)
        result['scheduled_units'] = [dict(cell=c, **u) for c in executable_cells(r)
                                     for u in r['units_per_new_cell']]
        for key in ('diagonal_ruling', 'diagonal_comparison'):
            if key in r:
                result[key] = r[key]
        result['gpu_executed'] = False
    elif a.command == 'score':
        result = score(a.cell)
    else:
        worker(a.cell, a.battery, a.spec)
        return
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
