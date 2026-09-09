"""r3 query-sized leases; CPU state inspection imports no GPU runtime.

Prior art: house X1 (2026), locally verified create-only claims/SHA receipts,
paired query templates and leased foreground workers. Borrow those mechanisms;
new integration is per-query checkpoints, resume and first-unit cost projection.
NIST FIPS 180-4 (2015) SHA-256: unverified — lead to check those search terms.
No new scientific algorithm or cryptographic authenticity claim.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from scripts import grm_x1_campaign as c
from scripts.grm_x1_register import create, raw_json, sha

REG_SHA = "0262299553a7123b297488383de9cfd4bf3922a1195e9f23e7068ac4f00a4729"


def units():
    # House X1 (2026): retain global ordinal for RNG/arm rotation; filtering
    # to one query changes checkpoint boundaries only, never the paired order.
    queries = {q['query_id']: (i, q) for i, q in enumerate(c.queries())}
    return [{**cell, 'unit_id': f"{cell['cell']}__{qid}", 'query_ids': [qid],
             'query_ordinal': queries[qid][0], 'family_id': queries[qid][1]['family_id']}
            for cell in c.read(c.FIX / 'cells.json') for qid in cell['query_ids']]


def sources():
    return c.source_manifest()


def payload():
    reg_path = c.OUT / 'continuation_03_registration.json'
    if sha(reg_path) != REG_SHA:
        raise ValueError('r3 registration binding mismatch')
    reg = c.read(reg_path)
    bindings = {'artifacts/grm_x1/continuation_03_registration.json': REG_SHA,
                'orders/GRM_X1_AMENDMENT_3.md': reg['order_sha256'],
                'artifacts/grm_x1/continuation_02.json': reg['predecessor_sha256']}
    previous = c.read(c.OUT / 'continuation_02.json')
    bindings.update(previous['bindings'])
    bindings.update(reg['historical_files'])
    for path, digest in bindings.items():
        if sha(c.ROOT / path) != digest:
            raise ValueError(f'r3 historical/authority binding drift: {path}')
    current = sources()
    allowed = {'scripts/grm_x1_campaign.py', 'scripts/grm_x1_units.py',
               'scripts/grm_x1_gpu.py', 'tests/test_grm_x1_campaign.py',
               'tests/test_grm_x1_units.py', 'scripts/grm_x1_cpu.py'}
    before = reg['source_shas_before']
    if set(before) - set(current) or any(current.get(p) != d for p, d in before.items() if p not in allowed):
        raise ValueError('r3 sources changed outside harness scope')
    if set(current) - set(before) - allowed:
        raise ValueError('r3 unregistered new source')
    return {'schema': 'grm.x1.continuation.v3', 'bindings': bindings,
            'sources': current, 'lead_commands_sha256': sha(c.OUT / 'lead_commands.txt'), 'units': units(), 'unit_scheme': reg['unit_scheme'],
            'budget': reg['budget'], 'rails': reg['rails'], 'failure_policy': reg['failure_policy'],
            'historical_files': reg['historical_files'],
            'receipt_directory': 'artifacts/grm_x1/receipts/r3',
            'claim_directory': 'artifacts/grm_x1/claims/r3',
            'evidence_class': 'reasoning: registration before CPU gates, no GPU execution'}


def seal():
    c.verify_fixtures()
    value = payload()
    path = c.OUT / 'continuation_03.json'
    checksum = c.OUT / 'continuation_03.sha256'
    if path.exists() or checksum.exists():
        raise FileExistsError('r3 continuation immutable')
    create(path, raw_json(value))
    create(checksum, f'{sha(path)}  continuation_03.json\n'.encode())
    return {'continuation': str(path), 'sha256': sha(path)}


def validate():
    path = c.OUT / 'continuation_03.json'
    if sha(path) != (c.OUT / 'continuation_03.sha256').read_text().split()[0]:
        raise ValueError('r3 continuation SHA mismatch')
    value = c.read(path)
    if value != payload():
        raise ValueError('r3 continuation policy/source binding mismatch')
    c.verify_fixtures()
    return value


def fingerprint():
    validate()
    frozen = c.read(c.OUT / 'handoff_manifest.json')
    current = c.dependency_inventory()
    if current != frozen['dependencies'] or current['missing']:
        raise ValueError('lead dependencies missing or drifted from handoff')
    return sha(c.OUT / 'continuation_03.json')


def key(unit, attempt):
    return f"{unit['unit_id']}__a{attempt}"


def rows_digest(rows):
    return hashlib.sha256(raw_json(rows)).hexdigest()


def check_rows(unit, rows, complete):
    expected = {(unit['query_ids'][0], arm, cond, unit['multiplicity'])
                for cond in unit['conditions'] for arm in unit['arms']}
    observed = [(r['query_id'], r['arm'], r['condition'], r['multiplicity']) for r in rows]
    if len(observed) != len(set(observed)) or not set(observed) <= expected or (complete and set(observed) != expected):
        raise ValueError('invalid/incomplete unit rows')
    if any(r['family_id'] != unit['family_id'] for r in rows):
        raise ValueError('unit family mismatch')


def record(unit, attempt, fp, status, rows, wall, error=None, non_fit=False, session_digests=None):
    # House X1 create-only receipts (2026). Deterministic attempt pathname
    # prevents a changed payload from creating a second receipt for one unit.
    check_rows(unit, rows, status == 'COMPLETE')
    value = {'schema': 'grm.x1.gpu-unit.v3', 'unit_id': unit['unit_id'],
             'cell': unit['cell'], 'attempt': attempt, 'fingerprint': fp,
             'registration_sha256': sha(c.OUT / 'registration.json'),
             'status': status, 'fit_status': 'NON_FIT' if non_fit else 'FIT' if status == 'COMPLETE' else 'UNKNOWN',
             'rows': rows, 'worker_wall_s': wall, 'rows_sha256': rows_digest(rows),
             'session_digests': session_digests or {}, 'error': error,
             'completed_unix_ns': time.time_ns(), 'evidence_class': 'E2E session receipt'}
    value['content_sha256'] = rows_digest(value)
    path = c.OUT / 'receipts/r3' / f'{key(unit, attempt)}_{fp}.json'
    create(path, raw_json(value))
    return path


def state():
    manifest = validate()
    fp = sha(c.OUT / 'continuation_03.json')
    layout = manifest['units']
    index = {u['unit_id']: u for u in layout}
    claims, receipts = {}, {}
    for folder, target in [('claims', claims), ('receipts', receipts)]:
        for path in sorted((c.OUT / folder / 'r3').glob('*.json')):
            if path.name.startswith('controller_'):
                continue
            value = c.read(path)
            uid, attempt = value['unit_id'], value['attempt']
            if uid not in index or type(attempt) is not int or attempt not in (1, 2):
                raise ValueError('invalid unit/attempt identity')
            unit = index[uid]
            identity = key(unit, attempt)
            if path.name != f'{identity}_{fp}.json' or value['fingerprint'] != fp or value['cell'] != unit['cell']:
                raise ValueError('invalid unit fingerprint/path/cell')
            if identity in target:
                raise ValueError('duplicate unit evidence')
            if folder == 'claims':
                if value['reserved_gpu_s'] != 285:
                    raise ValueError('invalid unit reservation')
            else:
                if value.get('content_sha256') != rows_digest({k: v for k, v in value.items() if k != 'content_sha256'}):
                    raise ValueError('unit receipt content digest mismatch')
                if (value['status'] not in ('COMPLETE', 'RED') or value['rows_sha256'] != rows_digest(value['rows'])
                        or value['registration_sha256'] != sha(c.OUT / 'registration.json')):
                    raise ValueError('invalid unit receipt/digest')
                wall = value['worker_wall_s']
                if not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall < 0:
                    raise ValueError('invalid unit wall')
                if value['status'] == 'COMPLETE' and (wall > 280 or value['fit_status'] != 'FIT'):
                    raise ValueError('completed unit exceeds work rail')
                check_rows(unit, value['rows'], value['status'] == 'COMPLETE')
            target[identity] = {**value, 'path': str(path.relative_to(c.ROOT)), 'sha256': sha(path)}
    if receipts.keys() - claims.keys():
        raise ValueError('unit receipt without claim')
    unit_states, all_rows = {}, []
    for unit in layout:
        first, second = key(unit, 1), key(unit, 2)
        if second in claims and (first not in receipts or receipts[first]['status'] != 'RED'
                                 or receipts[first]['fit_status'] == 'NON_FIT'):
            raise ValueError('invalid unit retry')
        attempts = [receipts[k] for k in (first, second) if k in receipts]
        pending = any(k in claims and k not in receipts for k in (first, second))
        latest = attempts[-1] if attempts else None
        unit_states[unit['unit_id']] = {'status': 'INCOMPLETE_CLAIM' if pending else latest['status'] if latest else 'PENDING',
             'red_count': sum(r['status'] == 'RED' for r in attempts), 'attempts': attempts,
             'fit_status': latest['fit_status'] if latest else 'UNKNOWN'}
        # Keep partial RED rows as diagnostics, use one latest attempt per unit
        # in the aggregate. Failed units can never unlock the natural phase.
        if latest:
            all_rows.extend(latest['rows'])
    result = c.aggregate(all_rows)
    stop = any(s['red_count'] >= 2 or s['status'] == 'INCOMPLETE_CLAIM' for s in unit_states.values())
    red = any(s['red_count'] for s in unit_states.values())
    incomplete = len(claims.keys() - receipts.keys())
    actual_wall = sum(r['worker_wall_s'] for r in receipts.values())
    charged = 570 + actual_wall + 285 * incomplete
    successful = sorted((r for r in receipts.values() if r['status'] == 'COMPLETE'), key=lambda r: (r['completed_unix_ns'], r['unit_id']))
    estimate = successful[0]['worker_wall_s'] if successful else None
    remaining = sum(s['status'] == 'PENDING' for s in unit_states.values())
    projection = 570 + len(layout) * estimate if estimate is not None else None
    remaining_projection = charged + remaining * estimate if estimate is not None else None
    budget_nonfit = charged > 5400 or (projection is not None and max(projection, remaining_projection) > 5400)
    cells = {}
    for cell in c.read(c.FIX / 'cells.json'):
        selected = [unit_states[u['unit_id']] for u in layout if u['cell'] == cell['cell']]
        cells[cell['cell']] = {'status': 'COMPLETE' if all(s['attempts'] for s in selected) else 'PENDING',
                'unit_count': len(selected), 'receipted_units': sum(bool(s['attempts']) for s in selected),
                'red_units': sum(bool(s['red_count']) for s in selected),
                'non_fit_units': sum(s['fit_status'] == 'NON_FIT' for s in selected),
                'worker_wall_s': sum(r['worker_wall_s'] for s in selected for r in s['attempts']),
                'rows': sum(len(s['attempts'][-1]['rows']) for s in selected if s['attempts'])}
    if red or stop:
        result['oracle_positive'] = False
    historical = [c.read(c.ROOT / p) for p in manifest['historical_files']
                  if Path(p).name.startswith('gpu_') and c.read(c.ROOT / p)['status'] == 'RED']
    verdict = 'NON_FIT_BUDGET' if budget_nonfit else 'STOP_SECOND_RED_OR_INCOMPLETE' if stop else 'RED' if red else 'PENDING'
    if not budget_nonfit and not stop and not red and result['oracle_complete']:
        verdict = ('KILLED' if any(result['kill'].values()) else 'ORACLE_NOT_POSITIVE' if not result['oracle_positive']
                   else 'ORACLE_POSITIVE_NATURAL_PENDING' if result['predictions']['S5'] is None
                   else 'COMPLETE_REGISTERED_BENEFIT_RETAINED')
    return {**result, 'active_epoch': 'r3', 'fingerprint': fp, 'units': unit_states, 'cells': cells,
            'cell_status': {k: v['status'] for k, v in cells.items()}, 'historical_red': historical,
            'claims': claims, 'receipts': receipts, 'failed': bool(red or stop), 'stop': stop,
            'campaign_verdict': verdict, 'budget_nonfit': budget_nonfit,
            'reservation_cap_s': 5400, 'historical_consumed_s': 570, 'charged_gpu_s': charged,
            'r3_worker_wall_s': actual_wall, 'outstanding_reserved_s': incomplete * 285,
            'first_completed_unit': successful[0]['unit_id'] if successful else None,
            'planning_estimate_s': estimate, 'projected_total_s': projection,
            'projected_remaining_total_s': remaining_projection, 'unit_count': len(layout),
            'timing_note': 'Per-unit wall unknown until first completed r3 unit; r2 was a cell timeout.'}


def next_unit(requested=None, retry_unit=None):
    current = state()
    layout = units()
    if requested is not None and requested not in current['cells']:
        raise ValueError('unregistered cell')
    if current['budget_nonfit']:
        return None, 'NON_FIT_BUDGET', 1
    if current['stop']:
        return None, 'STOP_SECOND_RED_OR_INCOMPLETE', 1
    if retry_unit:
        selected = [u for u in layout if u['unit_id'] == retry_unit and (requested is None or u['cell'] == requested)]
        if not selected:
            raise ValueError('unregistered retry unit')
        status = current['units'][retry_unit]
        if status['status'] != 'RED' or status['red_count'] != 1 or status['fit_status'] == 'NON_FIT':
            raise ValueError('retry requires exactly one non-timeout unit RED')
        unit, attempt = selected[0], 2
    else:
        unit, attempt = None, 1
        for candidate in layout:
            if requested and candidate['cell'] != requested:
                continue
            if current['units'][candidate['unit_id']]['status'] != 'PENDING':
                continue
            if candidate['phase'] == 'natural' and not current['oracle_positive']:
                if requested:
                    return None, 'NATURAL_LOCKED', 1
                continue
            unit = candidate
            break
    if unit is None:
        return None, 'CELL_COMPLETE' if requested else 'NO_ELIGIBLE_UNIT', 1
    if current['charged_gpu_s'] + 285 > 5400:
        return None, 'NON_FIT_BUDGET', attempt
    return unit, 'READY', attempt


def dry_run():
    manifest = validate()
    current = state()
    return {'evidence_class': 'reasoning: CPU enumeration, zero GPU work',
            'fingerprint': current['fingerprint'], 'unit_scheme': manifest['unit_scheme'],
            'unit_count': len(manifest['units']), 'rails': manifest['rails'], 'budget': manifest['budget'],
            'planning_estimate_s': current['planning_estimate_s'],
            'campaign_verdict': current['campaign_verdict'],
            'cells': [{**cell, 'unit_count': len(cell['query_ids']),
                       'units': [u for u in manifest['units'] if u['cell'] == cell['cell']],
                       'turns': len(cell['query_ids']) * len(cell['arms']) * len(cell['conditions']),
                       'command': f"bash scripts/grm_x1_lead_gpu.sh run {cell['cell']}",
                       'conditional': cell['phase'] == 'natural'} for cell in c.read(c.FIX / 'cells.json')]}


def controller(requested=None, retry_unit=None):
    fp = fingerprint()
    unit, status, attempt = next_unit(requested, retry_unit)
    if unit is None:
        return {'status': 'NON_FIT' if status == 'NON_FIT_BUDGET' else status, 'reason': status, 'cell': requested}
    command = ['flock', '--wait', '250', '--no-fork', '/tmp/forge-gpu.lock', sys.executable,
               str(c.ROOT / 'scripts/grm_x1_campaign.py'), '_worker', unit['cell'],
               '--fingerprint', fp, '--unit', unit['unit_id'], '--attempt', str(attempt)]
    started = time.monotonic()
    run = subprocess.run(command, cwd=c.ROOT, check=False)
    time.sleep(30)  # Foreground lead-only cooldown, one lease per invocation.
    current = state()
    outcome = current['units'][unit['unit_id']]
    status = 'UNIT_COMPLETE' if outcome['status'] == 'COMPLETE' else 'UNIT_RED'
    if outcome['status'] in ('PENDING', 'INCOMPLETE_CLAIM'):
        status = 'STOP_WORKER_NO_RECEIPT'
    elif current['stop']:
        status = 'STOP_SECOND_RED_OR_INCOMPLETE'
    elif current['budget_nonfit']:
        status = 'NON_FIT'
    elif current['cells'][unit['cell']]['status'] == 'COMPLETE':
        status = 'CELL_COMPLETE'
    # Timeout stays a unit-level NON_FIT in summary; continue other missing
    # units, as registered. A completed cell containing it prints NON_FIT.
    if status == 'CELL_COMPLETE' and current['cells'][unit['cell']]['non_fit_units']:
        status = 'NON_FIT'
    wall = time.monotonic() - started
    if wall > 590:
        status = 'STOP_OUTER_OVERRUN'
    result = {'status': status, 'cell': unit['cell'], 'unit_id': unit['unit_id'],
              'attempt': attempt, 'unit_status': outcome['status'], 'unit_fit_status': outcome['fit_status'],
              'returncode': run.returncode, 'fingerprint': fp, 'outer_wall_s': wall,
              'outer_within_rail': wall <= 590, 'reason': current['campaign_verdict'],
              'evidence_class': 'process receipt', 'command': command}
    c.emit_receipt('controller', result, directory=c.OUT / 'receipts/r3')
    return result


def lease_check():
    import fcntl
    inherited = []
    for entry in Path('/proc/self/fd').iterdir():
        try:
            if entry.resolve() == Path('/tmp/forge-gpu.lock'):
                inherited.append(int(entry.name))
        except FileNotFoundError:
            continue
    if not inherited:
        raise RuntimeError('GPU worker requires inherited /tmp/forge-gpu.lock lease')
    fcntl.flock(inherited[0], fcntl.LOCK_EX | fcntl.LOCK_NB)


def execute(unit, rows, directory):
    probe = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                           text=True, capture_output=True, check=False)
    if probe.returncode or probe.stdout.strip():
        raise RuntimeError(f'GPU busy/unavailable; yield to operator: {probe.stdout.strip()} {probe.stderr.strip()}')
    from scripts.grm_x1_gpu import run_unit
    run_unit(unit, rows, directory)


def worker(cell_id, fp, unit_id, attempt):
    lease_check()
    started = time.monotonic()
    if fingerprint() != fp:
        raise ValueError('worker fingerprint mismatch')
    unit, status, selected_attempt = next_unit(cell_id, unit_id if attempt == 2 else None)
    if unit is None or unit['unit_id'] != unit_id or selected_attempt != attempt:
        raise ValueError(f'unit selection changed under lease: {status}')
    claim = {'cell': cell_id, 'unit_id': unit_id, 'attempt': attempt, 'fingerprint': fp,
             'reserved_gpu_s': 285, 'pid': os.getpid(), 'started_unix': time.time(),
             'evidence_class': 'process reservation'}
    create(c.OUT / 'claims/r3' / f'{key(unit, attempt)}_{fp}.json', raw_json(claim))
    rows, error, status, non_fit = [], None, 'COMPLETE', False
    directory = c.OUT / 'sessions/r3' / f'{key(unit, attempt)}_{fp}'
    def expired(_sig, _frame):
        raise TimeoutError('GRM-X1 unit reached its 280 s work rail (285 s worker allowance)')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, 280 - (time.monotonic() - started)))
    try:
        execute(unit, rows, directory)
        check_rows(unit, rows, True)
    except Exception as exc:
        status, non_fit = 'RED', isinstance(exc, TimeoutError)
        error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    wall = time.monotonic() - started
    if wall > 280:
        status, non_fit = 'RED', True
        error = error or {'type': 'WorkerOverrun', 'message': f'{wall} s exceeds 280 s work rail'}
    # Hash small, already-written receipts immediately; never reload payloads.
    digests = {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob('*.json'))}
    wall = time.monotonic() - started
    if wall > 280:
        status, non_fit = 'RED', True
        error = error or {'type': 'WorkerOverrun', 'message': f'{wall} s exceeds 280 s work rail'}
    path = record(unit, attempt, fp, status, rows, wall, error, non_fit, digests)
    print(__import__('json').dumps({'unit_receipt': str(path), 'sha256': sha(path), 'status': status}), flush=True)
    return 0 if status == 'COMPLETE' else 1
