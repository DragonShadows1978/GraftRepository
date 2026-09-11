#!/usr/bin/env python3
"""Lead A5: independent cells and create-only half-range plans.

Prior art: C4/A2/A3/A4, EB1, DET1 (house, 2026), verified local source:
SHA closure, saved-state resume, imported scorers, reserve accounting.
Adds schedule/admission/reporting only; no novel algorithm claimed.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import grm_c4_campaign as c
from scripts import grm_c4_resume_a2 as a2
from scripts import grm_c4_remaining_a3 as a3
from scripts import grm_c4_cap_a4 as cap
from scripts import grm_c4_accounting_a7 as accounting_a7

OUT = c.OUT/'lead_a5'
ORDER = c.ROOT/'orders/GRM_C4_AMENDMENT_5.md'
AMENDMENT = OUT/'amendment_a7.json'
COMMANDS = c.OUT/'lead_commands_a5.txt'
ORDER_SHA = '8eb729c2454a1ee1da1f43ff9a288472c74541720911623455037a4f9646f0e4'
PREVIOUS_SHA = 'f28963404f0e0498d11cbc8368c83846eb0b33945df95583ddb301174425359d'
CELLS = ('c64_w96', 'c96_w64', 'c64_w64')
ADDITIONS = ('scripts/grm_c4_split_a5.py', 'scripts/grm_c4_register_a5.py',
             'tests/test_grm_c4_split_a5.py', 'orders/GRM_C4_AMENDMENT_5.md',
             'artifacts/grm_c4/lead_commands_a5.txt',
             'artifacts/grm_c4/lead_a5/before.json',
             'artifacts/grm_c4/lead_a5/pre_registration.json')


def timing_evidence():
    rows = []
    for i in range(6, 13):
        path = a2.receipt_path('c64_w96', 'longhorizon', f'c4-lh-{i:02}', amended=True)
        row = c.read(path)
        instrumentation = Path(row['result']['session_dir'])/'instrumentation.jsonl'
        pin = c.record(instrumentation)
        assert pin in row['session_state'], 'Timing state drift'
        turns = [json.loads(s) for s in instrumentation.read_text().splitlines()][-8:]
        assert [t['turn'] for t in turns] == list(range((i-1)*8, i*8))
        rows.append({'spec': row['spec'], 'receipt': c.record(path), 'instrumentation': pin,
            'status': row['status'], 'wall_seconds': row['gpu_seconds'],
            'turn_seconds': [{'turn': t['turn'], 'kind': t['kind'], 'seconds': t['turn_wall_ms']/1000} for t in turns],
            'overhead_seconds': row['gpu_seconds']-sum(t['turn_wall_ms']/1000 for t in turns)})
    assert rows[-1]['status'] == 'RED' and c.read(rows[-1]['receipt']['path'])['error'] == 'AssertionError: Worker rail exceeded: NON_FIT'
    assert all(r['status'] == 'PASS' for r in rows[:-1])
    return rows


def plan(previous):
    # Prior art: C4 ordered turn boundaries (house, 2026), reused as halves.
    # Specific timing heuristic: no prior art known to me. Use measured turns
    # plus max observed startup/save overhead; extrapolate probe100 using the
    # largest adjacent increase at70/80/90. These estimates are NOT GPU gates.
    evidence = timing_evidence()
    turns = {t['turn']: t for r in evidence for t in r['turn_seconds']}
    overhead = math.ceil(max(r['overhead_seconds'] for r in evidence))
    ordinary = max(t['seconds'] for t in turns.values() if t['kind'] != 'probe')
    increment = max(turns[80]['seconds']-turns[70]['seconds'], turns[90]['seconds']-turns[80]['seconds'])
    estimates = [math.ceil(overhead+sum(turns[t]['seconds'] for t in range(lo, hi)))
                 for lo, hi in ((88,92),(92,96))]
    estimates += [math.ceil(overhead+4*ordinary),
                  math.ceil(overhead+turns[90]['seconds']+increment+3*ordinary)]
    units = [dict(u) for u in previous['units_per_new_cell'][:-2]]
    measured = {r['spec']: math.ceil(r['wall_seconds']) for r in evidence[:-1]}
    for u in units:
        if u['spec'] in measured:
            u['estimate_seconds'] = measured[u['spec']]
    resume = 'c4-lh-11'
    for spec, start, stop, seconds in zip(('c4-lh-12a','c4-lh-12b','c4-lh-13a','c4-lh-13b'),
                                         (88,92,96,100), (92,96,100,104), estimates):
        units.append({'battery':'longhorizon','spec':spec,'start_inclusive':start,
            'stop_after_turns':stop,'resume':resume,'estimate_seconds':seconds,
            'planning_status':'FIT' if seconds < 200 else 'NON_FIT',
            'reason':None if seconds < 200 else 'Turn-range split cannot meet <200s planning target'})
        resume = spec
    before = c.read(OUT/'before.json')['files']
    worker_pins = [p for p in before if Path(p['path']).suffix == '.json'
                   and Path(p['path']).parent.parent.name == 'runs'
                   and not p['path'].endswith('.attempt.json')]
    recorded = 0
    for pin in worker_pins:
        assert c.record(pin['path']) == pin, 'Historical receipt drift'
        row = c.read(pin['path'])
        if 'finished_unix' in row:
            recorded += row['gpu_seconds']
    scheduled = [dict(u, cell=cell) for cell in CELLS for u in (units[-4:] if cell == CELLS[0] else units)]
    running = recorded
    for u in scheduled:
        # Full projection includes every unit, including rejected units. Never
        # trim the battery or recycle hypothetical savings into authorization.
        fits = running + max(285, u['estimate_seconds']) <= 5400
        u['cap_status'] = 'FIT' if fits else 'NON_FIT'
        u['projection_before_seconds'] = running
        running += u['estimate_seconds']
    return {'units_per_cell': units, 'scheduled_units': scheduled,
        'timing_evidence': evidence, 'estimation': {'overhead_seconds':overhead,
            'ordinary_turn_seconds':ordinary,'probe_increment_seconds':increment,
            'method':'ceil(sum observed turns + max observed overhead); lh13 ordinary turns use observed max, probe100 uses probe90 plus max adjacent probe increase. Existing other estimates retained; lh06..11 updated to ceil observed walls for both remaining cells.',
            'limit':'No <200 estimate is possible for a unit containing measured turn90=225.512s. Such units are registered NON_FIT, not authorized for GPU. lh13 is an unvalidated extrapolation.'},
        'projection': {'recorded_seconds': recorded, 'registered_seconds': running-recorded,
            'total_seconds':running, 'cap_seconds':5400,'overflow_seconds':max(0,running-5400),
            'status':'NON_FIT' if running > 5400 else 'FIT',
            'accounting':'All recorded walls, including both REDs, plus full untrimmed plan; 285s reservation also checked per admission; cooldown excluded.'}}


def expected_payload(previous):
    assert c.record(ORDER)['sha256'] == ORDER_SHA, 'Order drift'
    assert c.record(cap.AMENDMENT)['sha256'] == PREVIOUS_SHA, 'Predecessor drift'
    return {'schema':'grm.c4.split-amendment.v1','immutable':True,
        'lead_amendment_number':5,'artifact_sequence':7,
        'order':c.record(ORDER),'previous_amendment':c.record(cap.AMENDMENT),
        'registration':c.record(c.REG),'plan':plan(previous),
        'decoupling':{'prior_cell_scores_required':False,'within_cell_dependencies_required':True},
        'receipt_root':str(OUT/'runs'),'budget':previous['budget'],
        'sources':[accounting_a7.historical_record(c.ROOT/p) for p in ADDITIONS],
        'legacy_executor':c.record(a3.__file__),
        'receipt_policy':'Create-only, one run each. Known planning/cap/dependency NON_FIT makes a receipt without GPU execution. Full score only after every registered PASS; partial summary contains no battery score.',
        'activation':'Explicit A5 successor to A3; historic entrypoints and source manifests remain byte-unchanged.',
        'prior_art':c.read(OUT/'pre_registration.json')['prior_art']}


def binding():
    previous = a3.binding()
    row = c.read(AMENDMENT)
    pin = c.record(AMENDMENT)
    assert pin['sha256'] == AMENDMENT.with_suffix('.sha256').read_text().split()[0], 'Amendment SHA mismatch'
    assert row == expected_payload(previous), 'Amendment content/source mismatch'
    return {**previous, 'legacy_context':previous, 'amendment':pin,
        'sources':row['sources'], 'units_per_new_cell':row['plan']['units_per_cell'],
        'a5_plan':row['plan']}


def schedule(reg):
    return reg['a5_plan']['scheduled_units']


def receipt_path(cell, battery, spec):
    return OUT/'runs'/cell/f'{battery}_{spec}.json'


def selected_path(reg, cell, battery, spec):
    if cell == 'c64_w96' and spec not in {u['spec'] for u in reg['units_per_new_cell'][-4:]}:
        return a2.selected_path(reg['legacy_context']['a4_context'], cell, battery, spec)
    return receipt_path(cell,battery,spec)


def validate_receipt(path, reg, cell, battery, spec):
    assert path == selected_path(reg,cell,battery,spec), 'Receipt path mismatch'
    assert any((u['battery'],u['spec']) == (battery,spec) for u in reg['units_per_new_cell']), 'Unregistered receipt'
    if path != receipt_path(cell,battery,spec):
        pins = {p['path']:p for p in c.read(OUT/'before.json')['files']}
        assert c.record(path) == pins[str(path)], 'Historical receipt drift'
        return a2.validate_receipt(path,reg['legacy_context']['a4_context'],cell,battery,spec)
    row = c.read(path)
    assert (row['cell'],row['battery'],row['spec']) == (cell,battery,spec), 'Receipt identity mismatch'
    assert row['registration'] == c.record(c.REG), 'Mixed registration'
    assert row['amendment'] == reg['amendment'] and row['sources'] == reg['sources'], 'Mixed amendment/source'
    cfg = next(x for x in reg['cells'] if x['id']==cell)
    assert row['geometry_pin'] == reg['geometry'] and (row['chunk'],row['width']) == (cfg['chunk'],cfg['width']), 'Mixed geometry'
    assert row['status'] in ('PASS','RED','NON_FIT'), 'Invalid receipt status'
    assert row['budget'] == reg['budget'] and row['budget_amendment'] == reg['budget_amendment'], 'Mixed budget'
    if row['status'] == 'NON_FIT':
        assert row['gpu_executed'] is False and row['gpu_seconds'] == 0, 'Invalid NON_FIT accounting'
    return row


def pass_receipt(reg, cell, battery, spec):
    path = selected_path(reg,cell,battery,spec)
    assert path.is_file(), 'Prior unit missing/RED'
    row = validate_receipt(path,reg,cell,battery,spec)
    assert row['status'] == 'PASS', 'Prior unit missing/RED'
    if path == receipt_path(cell,battery,spec) and battery in ('census','longhorizon'):
        units = [u for u in reg['units_per_new_cell'] if u['battery']==battery]
        index = next(i for i,u in enumerate(units) if u['spec']==spec)
        prior = selected_path(reg,cell,battery,units[index-1]['spec']) if index else None
        assert row['prior_receipt'] == (c.record(prior) if prior else None), 'Stale prior receipt'
    return path,row


def ready(reg, cell, battery, spec):
    assert any((u['cell'],u['battery'],u['spec']) == (cell,battery,spec) for u in schedule(reg)), 'Unit not authorized'
    path = receipt_path(cell,battery,spec)
    for root in (c.OUT,a2.OUT,a3.OUT,OUT):
        other = root/'runs'/cell/path.name
        assert not other.exists() and not other.with_suffix('.attempt.json').exists(), 'No retries or overwrite'
    # Prior art: local C4 dependency graph (house,2026). A5 removes ONLY
    # cross-cell score edges. Turn/state dependencies inside a cell remain.
    prior = None
    for u in reg['units_per_new_cell']:
        if (u['battery'],u['spec']) == (battery,spec):
            break
        p,row = pass_receipt(reg,cell,u['battery'],u['spec'])
        if u['battery'] == battery:
            prior = p,row
    if prior and battery in ('census','longhorizon'):
        session = a2.state_files(prior[1])
        if battery == 'longhorizon':
            a2.saved_turns(session,next(u['stop_after_turns'] for u in a2.lh_units(reg) if u['spec']==prior[1]['spec']))
    dest = OUT/'runs'/cell/battery/('arm1' if battery=='census' else '')/spec/'session'
    assert not dest.parent.exists(), 'Session already exists; no overwrite'
    return path,prior


def accounting(reg, *, reserve=True):
    # Prior art: DET1/A4 reserve accounting (house,2026). Charge immutable
    # historical REDs exactly; new RED blocks its cell but not independent cells.
    # A7: C4/A5 completed-worker accounting and claim guard (house,2026),
    # reused. Added filename filtering and per-file diagnostics; no novelty.
    # Metadata is not a receipt. Incomplete worker evidence still fails closed
    # so skipping it cannot undercount work and authorize a new lease.
    workers=[]
    completed=set()
    claims=[]
    blocked=[]
    pins={p['path']:p for p in c.read(OUT/'before.json')['files']}
    def skip(path, reason):
        print(json.dumps({'accounting_skip':str(path),'reason':reason}), file=sys.stderr)
    for root in (c.OUT,a2.OUT,a3.OUT,OUT):
        for path in sorted((root/'runs').glob('*/*.json')):
            if path.name.endswith('.attempt.json'):
                skip(path, 'attempt marker; not a completed receipt')
                claims.append(path)
                continue
            if path.name=='score.json':
                skip(path, 'score summary; not a worker receipt')
                continue
            if not path.name.startswith(('sup_', 'census_', 'longhorizon_')):
                skip(path, 'metadata/non-worker filename; not a worker receipt')
                continue
            try:
                row=c.read(path)
            except (ValueError, OSError) as exc:
                skip(path, f'unreadable worker receipt: {type(exc).__name__}: {exc}')
                blocked.append(str(path))
                continue
            if not isinstance(row, dict) or row.get('status') not in ('PASS','RED','NON_FIT'):
                skip(path, 'incomplete worker receipt: missing/invalid status')
                blocked.append(str(path))
                continue
            if not all(k in row for k in ('cell','battery','spec')):
                skip(path, 'incomplete worker receipt: missing cell/battery/spec')
                blocked.append(str(path))
                continue
            assert path.parent.name == row['cell'] and path.name == f"{row['battery']}_{row['spec']}.json", f'Worker identity mismatch: {path}'
            if row['status'] != 'NON_FIT' and not all(k in row for k in ('finished_unix','gpu_seconds')):
                skip(path, 'incomplete worker receipt: missing finished_unix/gpu_seconds')
                blocked.append(str(path))
                continue
            if root != OUT:
                assert str(path) in pins and c.record(path)==pins[str(path)], 'Unregistered or stale historical worker'
            else:
                row=validate_receipt(path,reg,row['cell'],row['battery'],row['spec'])
            if row['status']=='NON_FIT':
                assert row.get('gpu_executed') is False and row.get('gpu_seconds') == 0, f'Invalid NON_FIT accounting: {path}'
                skip(path, 'NON_FIT status receipt; no executed work, zero charge')
                continue
            assert all(type(row[k]) in (int,float) and math.isfinite(row[k]) and row[k]>=0
                       for k in ('gpu_seconds','finished_unix')), f'Invalid wall accounting: {path}'
            completed.add(path)
            workers.append(row)
    for claim in claims:
        receipt=claim.with_name(claim.name.removesuffix('.attempt.json')+'.json')
        if receipt not in completed:
            skip(claim, 'unfinished worker claim: no completed PASS/RED receipt; campaign blocked')
            blocked.append(str(claim))
    assert not blocked, 'Unfinished worker claim or incomplete worker receipt: campaign blocked: '+', '.join(blocked)
    used=sum(r['gpu_seconds'] for r in workers)
    if reserve:
        assert used+285<=reg['budget']['gpu_seconds'], 'GPU budget non-fit; no run'
        if workers:
            assert time.time()-max(r['finished_unix'] for r in workers)>=30, '30s cooldown not met'
    return used


@contextmanager
def split_harness(chunk,width,root,events):
    # Prior art: EB1 and C4 runtime shard-table adapter (house,2026). Change
    # boundaries only; preserve script, resume implementation and scorer.
    reg=binding()
    stops={u['spec']:u['stop_after_turns'] for u in a2.lh_units(reg)}
    with patch.object(c,'LH_STOPS',stops), c.harness(chunk,width,root,events) as modules:
        yield modules


def seed_prior(root, prior):
    # Prior art: A2 content-verified copy/resume (house,2026), reused unchanged.
    if not prior:
        return
    source=a2.state_files(prior[1])
    seed=root/'longhorizon'/prior[1]['spec']/'session'
    if source != seed:
        assert not seed.exists(), 'Recovery seed already exists'
        shutil.copytree(source,seed)
        for pin in prior[1]['session_state']:
            copied=c.record(seed/Path(pin['path']).relative_to(source))
            assert (copied['sha256'],copied['bytes'])==(pin['sha256'],pin['bytes']), 'Copied state drift'


def non_fit(reg,cell,battery,spec,reason):
    cfg=next(x for x in reg['cells'] if x['id']==cell)
    path=receipt_path(cell,battery,spec)
    assert not path.exists() and not path.with_suffix('.attempt.json').exists(), 'No retries or overwrite'
    row={'cell':cell,'battery':battery,'spec':spec,'status':'NON_FIT','reason':reason,
        'registration':c.record(c.REG),'amendment':reg['amendment'],'sources':reg['sources'],
        'budget':reg['budget'],'budget_amendment':reg['budget_amendment'],
        'geometry_pin':reg['geometry'],'chunk':cfg['chunk'],'width':cfg['width'],
        'gpu_executed':False,'gpu_seconds':0,'evidence_class':'Planning/dependency admission; no GPU execution'}
    c.write_once(path,row)
    return row


def admission(reg,cell,battery,spec):
    unit=next((u for u in schedule(reg) if (u['cell'],u['battery'],u['spec'])==(cell,battery,spec)),None)
    assert unit, 'Unit not authorized'
    for root in (c.OUT,a2.OUT,a3.OUT,OUT):
        p=root/'runs'/cell/f'{battery}_{spec}.json'
        assert not p.exists() and not p.with_suffix('.attempt.json').exists(), 'No retries or overwrite'
    if unit.get('planning_status')=='NON_FIT':
        return 'Registered split estimate >=200s: NON_FIT'
    if unit['cap_status']=='NON_FIT':
        return 'Registered full-plan cap overflow: NON_FIT'
    for u in reg['units_per_new_cell']:
        if (u['battery'],u['spec'])==(battery,spec):
            break
        p=selected_path(reg,cell,u['battery'],u['spec'])
        if not p.exists():
            return f"Incomplete cell; prior {u['spec']} missing: NON_FIT"
        row=validate_receipt(p,reg,cell,u['battery'],u['spec'])
        if row['status']!='PASS':
            return f"Incomplete cell; prior {u['spec']} {row['status']}: NON_FIT"
        pass_receipt(reg,cell,u['battery'],u['spec'])
    if accounting(reg,reserve=False)+285>reg['budget']['gpu_seconds']:
        return 'Actual wall plus285s reserve exceeds cap: NON_FIT'
    return None


def worker(cell, battery, spec):
    reg = binding()
    reason = admission(reg, cell, battery, spec)
    if reason:
        return non_fit(reg, cell, battery, spec, reason)
    receipt, prior = ready(reg, cell, battery, spec)
    cfg = next(v for v in reg['cells'] if v['id'] == cell)
    root = OUT/'runs'/cell
    events, completed = [], []
    payload = {'cell': cell, 'battery': battery, 'spec': spec, 'status': 'RED',
        'registration': c.record(c.REG), 'amendment': reg['amendment'], 'sources': reg['sources'],
        'budget_amendment': reg.get('budget_amendment'), 'budget': reg['budget'],
        'geometry_pin': reg['geometry'], 'chunk': cfg['chunk'], 'width': cfg['width'],
        'prior_receipt': c.record(prior[0]) if prior else None, 'events': events,
        'evidence_class': 'GPU E2E worker; lead A5 split and independent cells'}
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
            if battery == 'longhorizon':
                seed_prior(root, prior)
            with split_harness(cfg['chunk'], cfg['width'], root, events) as (replay, census, lh), a2.observe_turns(completed):
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
    reg = binding()
    assert cell in CELLS, 'Only registered new cells may score'
    partial = cell_progress(reg, cell)
    if partial['status'] != 'PASS':
        c.write_once(score_path(cell), dict(partial, cell=cell, registration=c.record(c.REG), amendment=reg['amendment'], sources=reg['sources']))
        return partial
    cfg = next(c for c in reg['cells'] if c['id'] == cell)
    run_root = (c.OUT if cell == 'c64_w96' else OUT) / 'runs' / cell
    recovery_root = OUT / 'runs' / cell
    receipts = []
    for unit in reg['units_per_new_cell']:
        _, r = pass_receipt(reg, cell, unit['battery'], unit['spec'])
        receipts.append(r)
    a2.state_files(receipts[-1])
    assert receipts[-1].get('cell_serving_guard', {}).get('attempt_residency_count', 0) > 0, 'Missing per-cell serving guard'
    with split_harness(cfg['chunk'], cfg['width'], recovery_root, []) as (_, census, lh):
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
    payload = {'cell': cell, 'registration': c.record(c.REG), 'sources': reg['sources'],
               'geometry': reg['geometry'], 'counts': reg['counts'], 'scores': scores,
               'measurements': measurements, 'amendment': reg.get('amendment'),
               'sup_probes': sup, 'census_probes': census_result['probes'], 'longhorizon_probes': l['probes'],
               'prediction_met_in_cell': scores == reg['counts'],
               'factorial_verdict': 'INCONCLUSIVE: per-cell score only; lead must compare all fixed-geometry cells',
               'measurement_receipts': [c.record(selected_path(reg, cell, u['battery'], u['spec'])) for u in reg['units_per_new_cell']],
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
    c.write_once(recovery_root / 'score.json', payload)
    return payload


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


def score_path(cell):
    return OUT/'runs'/cell/'score.json'


def cell_progress(reg,cell):
    # Prior art: C4 complete-battery score gate (house,2026). Preserve that
    # gate; add explicit NON_FIT lists rather than treating partial totals as scores.
    completed, missing, failed, pins = [], [], [], []
    for u in reg['units_per_new_cell']:
        path=selected_path(reg,cell,u['battery'],u['spec'])
        if not path.exists():
            missing.append(u['spec'])
            continue
        row=validate_receipt(path,reg,cell,u['battery'],u['spec'])
        pins.append(c.record(path))
        if row['status']=='PASS':
            pass_receipt(reg,cell,u['battery'],u['spec'])
            completed.append(u['spec'])
        else:
            failed.append({'spec':u['spec'],'status':row['status'],'reason':row.get('reason',row.get('error'))})
    return {'status':'PASS' if not missing and not failed else 'NON_FIT',
        'completed_segments':completed,'missing_segments':missing,'failed_segments':failed,
        'measurement_receipts':pins}


def load_score(reg,cell):
    path=score_path(cell)
    row=c.read(path)
    assert row['cell']==cell and row['registration']==c.record(c.REG), 'Mixed score identity'
    assert row['sources']==reg['sources'] and row['amendment']==reg['amendment'], 'Mixed score source/amendment'
    assert row['geometry']==reg['geometry'] and row['counts']==reg['counts'], 'Mixed score geometry/counts'
    receipts=[pass_receipt(reg,cell,u['battery'],u['spec']) for u in reg['units_per_new_cell']]
    assert row['measurement_receipts']==[c.record(p) for p,_ in receipts], 'Stale score receipts'
    a2.state_files(receipts[-1][1])
    assert receipts[-1][1].get('cell_serving_guard',{}).get('attempt_residency_count',0)>0, 'Missing per-cell serving guard'
    probe_outcomes(row,reg)
    assert all(row['measurements'][b]['attempt_observations']>0 for b in reg['counts']), 'Missing residency observations'
    return row

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
                assert pin in reg['legacy_context']['sources'], 'Unbound historical probe source'
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
        progress = cell_progress(reg, cell)
        if progress['status'] != 'PASS':
            if score_path(cell).exists():
                partial = c.read(score_path(cell))
                assert 'scores' not in partial, 'Partial score presented as battery score'
                assert partial == dict(progress, cell=cell, registration=c.record(c.REG), amendment=reg['amendment'], sources=reg['sources']), 'Stale partial score'
            cells[cell] = progress
            continue
        if not score_path(cell).exists():
            cells[cell] = dict(progress, status='NON_FIT', reason='Complete worker chain but no bound battery score')
            continue
        row = load_score(reg, cell)
        required[cell] = probe_outcomes(row, reg)
        cells[cell] = dict(progress, scores=row['scores'], receipt=c.record(score_path(cell)), required_probes=required[cell])
    historical_required, historical_pins = wc_probes(64)
    diagonal_required, diagonal_pins = wc_probes(96)
    cells['c96_w96'] = {'scores': wc_scores(96), 'receipt': reg['historical_results'],
        'required_probes': diagonal_required, 'probe_receipts': diagonal_pins,
        'evidence_class': 'WC1 archived E2E; registered geometry reuse; token residency unavailable'}
    if any(cells[cell]['status'] != 'PASS' for cell in CELLS):
        result = {'evidence_class':'CPU aggregation; incomplete finite battery, no partial score',
            'registration':c.record(c.REG),'amendment':reg['amendment'],'cells':cells,
            'counts':reg['counts'],'prediction':reg['prediction'],'prediction_met':None,
            'factorial_verdict':'INCONCLUSIVE','projection':reg['a5_plan']['projection'],
            'limits':'NON_FIT includes partial/unscored cells. No battery score or prediction inferred from partial evidence.',
            'diagonal_comparison':{'historical':wc_scores(64),'historical_required_probes':historical_required,
                'historical_probe_receipts':historical_pins}}
        if cells['c64_w64']['status'] == 'PASS':
            fixed=cells['c64_w64']['scores']
            result['diagonal_comparison'].update(fixed=fixed,
                delta_correct={b:fixed[b]-wc_scores(64)[b] for b in fixed},
                fixed_required_probes=required['c64_w64'])
        c.write_once(OUT/'cross_summary.json',result)
        return result
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
        'limits': 'Partial cells report NON_FIT without battery scores; stale or forged evidence refuses summary. WC1 token residency unavailable. No universal or statistical causal claim.'}
    c.write_once(OUT/'cross_summary.json', result)
    return result


def format_summary(result):
    def render(cell):
        row=result['cells'][cell]
        if row.get('status')=='NON_FIT':
            return 'NON_FIT (completed: '+(', '.join(row['completed_segments']) or 'none')+')'
        return ', '.join(f"{row['scores'][b]}/{result['counts'][b]}" for b in ('sup','census','longhorizon'))
    lines=['Scores: sup, census, full long-history', 'chunk | width64 | width96', '--- | --- | ---']
    for chunk in (64,96):
        lines.append(f"{chunk} | {render(f'c{chunk}_w64')} | {render(f'c{chunk}_w96')}")
    lines += ['(96,96): cited WC1, all14 long-history probes.',
              f"Registered prediction: {result['prediction']}",
              f"Prediction met: {result['prediction_met']}; outcome: {result['factorial_verdict']}",
              'Diagonal comparison: '+json.dumps(result['diagonal_comparison'],sort_keys=True)]
    return '\n'.join(lines)


def command_text(units):
    # Prior art: C4 foreground leased commands (house,2026). Keep every unit,
    # including explicit NON_FIT admission; no blanket shell error suppression.
    lines=['#!/bin/bash','set -e',f'cd {c.ROOT}',
        '# A5 successor to A2/A3; run only after prior foreground chain exits.',
        '# NON_FIT admissions exit0 with a create-only no-GPU receipt; integrity errors stop.',
        'python scripts/grm_c4_split_a5.py --dry-run','sleep 30']
    for cell in CELLS:
        for u in units:
            if u['cell'] != cell:
                continue
            lines += [f"timeout --signal=KILL 590s python scripts/grm_c4_split_a5.py worker --cell {cell} --battery {u['battery']} --spec {u['spec']}",'sleep 30']
        lines += [f'python scripts/grm_c4_split_a5.py score --cell {cell}']
    return '\n'.join(lines+['python scripts/grm_c4_split_a5.py summary'])+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',nargs='?',default='preflight',choices=('preflight','worker','score','summary'))
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--cell',choices=CELLS)
    parser.add_argument('--battery',choices=('sup','census','longhorizon'))
    parser.add_argument('--spec')
    args=parser.parse_args()
    if args.dry_run or args.command=='preflight':
        reg=binding()
        print(json.dumps({'gpu_executed':False,'amendment':reg['amendment'],
            'budget_amendment':reg['budget_amendment'],'budget':reg['budget'],
            'unit_count':len(schedule(reg)),**reg['a5_plan']},indent=2))
    elif args.command=='worker':
        worker(args.cell,args.battery,args.spec)
    elif args.command=='score':
        print(json.dumps(score(args.cell),indent=2))
    else:
        print(format_summary(cross_summary()))


if __name__=='__main__':
    main()
