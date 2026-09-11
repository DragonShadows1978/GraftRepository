#!/usr/bin/env python3
"""C2 lead amendment 1: CPU-verifiable registration and foreground execution.

Prior art: C2 r1 / CMC1 / WC1 / RT1.1, project contributors (2026),
locally inspected. Reuse worker, flock lease, battery comparator and receipts.
Taken: create-only reservations and checkpoint dependencies. Ours: budget
amendment validation, battery scheduling and explicit no-retry resume glue.
SHA-256 content binding is established practice, not a signature or novelty;
no prior art known to me for this specific adapter beyond these local systems.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import grm_c2_cells as old
from scripts.grm_c2_profile import (OUT, REGISTRATION, REGISTRY, RT1_FIELDS,
    create, read, sha, verify_registration, compare)

AMENDMENT = OUT / 'amendment_lead_1.json'
# Trusted code anchor: never accept a digest supplied by the amendment itself.
# This detects replaced JSON, including a forger who recomputes its sidecar SHA.
AMENDMENT_SHA256 = '91df018adaa90e9db3fd32981ad72a5564cee2cc8360b462775248945859807c'
BATTERIES = ('sup', 'census', 'longhistory')
WIDTH_NOTE = ('shipped defaults vs proposed profile: defaults width 256, profile '
              'width 96; this is not a width-matched pair.')


def effective_registration():
    if sha(AMENDMENT) != AMENDMENT_SHA256:
        raise ValueError('forged or stale amendment: SHA-256 differs from trusted code anchor')
    a = read(AMENDMENT)
    template = Path(__file__).read_text().replace(AMENDMENT_SHA256, 'PENDING_REGISTRATION')
    if hashlib.sha256(template.encode()).hexdigest() != a['verifier_template_sha256']:
        # Prior art: C2 immutable amendment bindings (project, 2026). Amendment
        # 3 explicitly supersedes this verifier and lead command; validate its
        # fixed-source binding before consulting byte-preserved history.
        from scripts.grm_c2_epoch3 import verify_amendment
        epoch = verify_amendment()
        template = (ROOT / epoch['historical_verifier']).read_text().replace(
            AMENDMENT_SHA256, 'PENDING_REGISTRATION')
        if hashlib.sha256(template.encode()).hexdigest() != a['verifier_template_sha256']:
            raise ValueError('stale amendment verifier source')
    for name, digest in a['bindings'].items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if name == 'artifacts/grm_c2/lead_commands.txt' and sha(path) != digest:
            from scripts.grm_c2_epoch3 import verify_amendment
            path = ROOT / verify_amendment()['historical_lead_commands']
        if sha(path) != digest:
            raise ValueError(f'stale amendment binding: {name}')
    r = verify_registration()
    if a['registration_sha256'] != sha(REGISTRATION):
        raise ValueError('stale r1 registration')
    r = dict(r, budget_seconds=a['budget_seconds'], amendment=a)
    if r['estimated_seconds'] != sum(c['estimate_seconds'] for c in r['cells']):
        raise ValueError('cell estimate mismatch')
    if r['estimated_seconds'] > r['budget_seconds']:
        raise ValueError('NON_FIT_BUDGET')
    return r


def ordered_cells(r):
    # Stable grouping of r1's topologically ordered cells: no changed cell,
    # arm, estimate, probe or dependency. Score each completed battery promptly.
    return [c for b in BATTERIES for c in r['cells'] if c['battery'] == b]


def dry_run():
    r = effective_registration()
    return dict(status='READY_FOR_LEAD', measurement_status='NOT_RUN',
        gpu_executed=False, cell_count=len(r['cells']), cells=ordered_cells(r),
        estimated_seconds=r['estimated_seconds'], budget_seconds=r['budget_seconds'],
        worker_seconds=r['worker_seconds'], outer_seconds=r['outer_seconds'],
        cooldown_seconds=r['cooldown_seconds'], lock_wait_seconds=r['lock_wait_seconds'],
        amendment_sha256=sha(AMENDMENT), registration_sha256=sha(REGISTRATION),
        width_comparison=WIDTH_NOTE, defaults96=r['amendment']['defaults96'],
        accounting=r['amendment']['accounting'], resume_rule=r['amendment']['resume_rule'])


def receipt_bindings():
    return dict(amendment_sha256=sha(AMENDMENT), registration_sha256=sha(REGISTRATION),
                registry_sha256=sha(REGISTRY))


def check_receipt(value):
    if any(value.get(k) != v for k, v in receipt_bindings().items()):
        raise ValueError('stale receipt binding')


def cell_state(cell):
    directory = OUT / 'cells' / cell['id']
    if (directory / 'controller.json').exists():
        value = read(directory / 'controller.json')
        check_receipt(value)
        return value['status']
    return 'STARTED_UNFINISHED' if directory.exists() else 'UNSTARTED'


def charged_seconds(r):
    total = 0.0
    for cell in r['cells']:
        directory = OUT / 'cells' / cell['id']
        controller = directory / 'controller.json'
        if controller.exists():
            value = read(controller); check_receipt(value)
            charge = value['charged_seconds']
            if not isinstance(charge, (float, int)) or not math.isfinite(charge) or not 0 <= charge <= r['worker_seconds']:
                raise ValueError('invalid budget charge')
            total += charge
        elif directory.exists():
            total += r['worker_seconds']  # orphan remains charged at the full cap
    return total


def evidence_rows(cell, value):
    check_receipt(value)
    if value.get('cell') != cell:
        raise ValueError('worker cell mismatch')
    rows = value['rows']
    reg = read(REGISTRY)
    if cell['phase'] == 'restart':
        expected = cell['probes']
    elif cell['battery'] == 'sup':
        expected = [p['probe_id'] for p in reg['sup_plans'][cell['session']]]
    else:
        expected = [f'lh_t{t:03d}' if cell['battery'] == 'longhistory' else
                    next(p['probe_id'] for p in reg['census_probes'] if p['turn'] == t)
                    for t in range(cell['start'], cell['stop'])
                    if reg['session_script'][t]['kind'] == 'probe']
    if sorted(row['probe_id'] for row in rows) != sorted(expected):
        raise ValueError('missing/duplicate/unexpected cell probes')
    if cell['phase'] == 'fresh' and value.get('end_capture', {}).get('valid') is not True:
        raise ValueError('invalid persisted capture evidence')
    for row in rows:
        if any(row.get(k) != cell[k] for k in ('battery', 'side', 'phase')):
            raise ValueError('row cell mismatch')
        seats = row.get('token_seats')
        if (type(seats) is not int or seats < 0 or
            row.get('capture_valid') is not True or
            not all(k in row.get('rt1', {}) for k in RT1_FIELDS) or
            type(row.get('correct')) is not bool):
            raise ValueError('missing/invalid token seats, RT1 or capture evidence')
        if cell['phase'] == 'restart' and not (row.get('new_process') is True and row.get('metadata_retained') is True):
            raise ValueError('restart did not retain metadata in a new process')
    return rows


def cooldown(r):
    # Persisted wall timestamp preserves the rail across a new lead invocation.
    # Unknown completion after an outer timeout gets a full foreground cooldown.
    waits = []
    for c in r['cells']:
        d = OUT / 'cells' / c['id']
        if d.exists():
            p = d / 'controller.json'
            waits.append(max(0, read(p)['finished_at'] + r['cooldown_seconds'] - time.time())
                         if p.exists() else r['cooldown_seconds'])
    if waits and max(waits) > 0:
        time.sleep(min(r['cooldown_seconds'], max(waits)))


@contextmanager
def campaign_lock():
    # CMC1's foreground flock convention, on a separate campaign mutex so two
    # lead invocations cannot both pass accounting or start different cells.
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'campaign.lock').open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('another C2 foreground controller is active') from None
        try:
            yield f.fileno()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def run_cell(cell, r):
    if cell_state(cell) != 'UNSTARTED':
        raise ValueError('cell already started; RED and completed cells are never retried')
    cooldown(r)
    directory = OUT / 'cells' / cell['id']
    directory.mkdir(parents=True, exist_ok=False)
    status, error, elapsed, worker_digest = 'RED', None, 0.0, None
    try:
        for dep in cell['depends']:
            if cell_state(old.cell_by_id(r, dep)) != 'COMPLETE':
                raise ValueError(f'BLOCKED_DEPENDENCY: {dep}; no recapture or retry')
        # Exclude our new empty reservation directory from the existing charge.
        used = charged_seconds(r) - r['worker_seconds']
        if used + r['worker_seconds'] > r['budget_seconds']:
            raise ValueError('budget rail: cannot reserve next 285-second cell')
        create(directory / 'reservation.json', dict(seconds=r['worker_seconds'],
               cell=cell['id'], started_at=time.time(), **receipt_bindings()))
        # Import the existing lease only on the lead's execution path.
        from scripts.grm_cmc1_gpu_arms import gpu_lease
        with gpu_lease(r['worker_seconds'], r['lock_wait_seconds']):
            started = time.monotonic()
            elapsed = r['worker_seconds']  # exceptions retain conservative charge
            env = old.environment(old.flags_for(cell['side']))
            env['GRM_C2_LEASE_PARENT'] = str(os.getpid())
            with (directory / 'worker.log').open('x') as stream:
                result = subprocess.run([sys.executable, __file__, '--worker', cell['id']],
                    cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                    timeout=280)
            elapsed = min(r['worker_seconds'], time.monotonic() - started)
            if result.returncode:
                raise ValueError(f'worker returncode={result.returncode}')
            evidence_rows(cell, read(directory / 'worker.json'))
            worker_digest = sha(directory / 'worker.json')
            status = 'COMPLETE'
    except (Exception, KeyboardInterrupt) as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        create(directory / 'controller.json', dict(status=status, error=error,
            charged_seconds=elapsed, finished_at=time.time(), cell=cell,
            worker_sha256=worker_digest, sources=old.sources(), **receipt_bindings()))
    print(json.dumps(dict(cell=cell['id'], status=status, error=error)), flush=True)
    return 0 if status == 'COMPLETE' else 1


def collect(r, battery=None):
    rows, states = [], {}
    for cell in ordered_cells(r):
        if battery and cell['battery'] != battery:
            continue
        state = cell_state(cell); states[cell['id']] = state
        if state == 'COMPLETE':
            d = OUT / 'cells' / cell['id']
            if sha(d / 'worker.json') != read(d / 'controller.json')['worker_sha256']:
                raise ValueError('worker receipt changed after completion')
            rows.extend(evidence_rows(cell, read(d / 'worker.json')))
    return rows, states


def score_battery(r, battery):
    path = OUT / 'scores' / f'{battery}.json'
    if path.exists():
        prior = read(path); check_receipt(prior)
        return prior  # immutable RED is never re-scored into GREEN
    rows, states = collect(r, battery)
    result = compare(rows)['batteries'][battery]
    complete = all(s == 'COMPLETE' for s in states.values())
    green = complete and result['adoption_acceptance'] and result['fix_validation_exact']
    value = dict(battery=battery, status='GREEN' if green else 'RED',
                 states=states, result=result, **receipt_bindings())
    create(path, value)
    print(f"score {battery}: {value['status']}", flush=True)
    return value


def summary(r):
    rows, states = collect(r)
    results = compare(rows)
    print(WIDTH_NOTE)
    d = r['amendment']['defaults96']
    print(f"defaults96: {d['status']}; +{d['extra_cells']} cells / +{d['extra_estimated_seconds']} s; "
          f"total {d['total_estimated_seconds']} s > {r['budget_seconds']} s cap.")
    print('battery | arm(width) | pre-restart | post-restart | token seats pre/post | RT1 fields present pre/post | restart retained')
    for battery in BATTERIES:
        total = len(read(REGISTRY)['probe_ids'][battery])
        b = results['batteries'][battery]
        for side, width in [('profile', 96), ('defaults', 256)]:
            arm = b['arms'][side]; scores, seats, rt1 = [], [], []
            for phase in ('fresh', 'restart'):
                p = arm['phases'][phase]; rr = list(p['rows'].values())
                scores.append(f"{p['correct']}/{total}" if p['complete'] else f"NOT_RUN/INCOMPLETE ({len(rr)}/{total} probes)")
                valid_seats = p['complete'] and all(type(x.get('token_seats')) is int for x in rr)
                seats.append(str(sum(x['token_seats'] for x in rr)) if valid_seats else 'UNKNOWN')
                rt1.append('YES' if p['complete'] and all(all(k in x.get('rt1', {}) for k in RT1_FIELDS) for x in rr) else 'UNKNOWN/MISSING')
            print(f"{battery} | {side}({width}) | {scores[0]} | {scores[1]} | {'/'.join(seats)} | {'/'.join(rt1)} | {arm['restart_retained']}")
    print('RT1 required fields: ' + ', '.join(RT1_FIELDS))
    complete = all(s == 'COMPLETE' for s in states.values())
    score_red = any(read(p)['status'] != 'GREEN' for p in (OUT / 'scores').glob('*.json'))
    green = complete and results['adopt'] and results['fix_validation'] and not score_red
    status = 'GREEN' if green else ('NOT_RUN' if all(s == 'UNSTARTED' for s in states.values()) else 'RED/INCOMPLETE')
    print(f"summary: {status}; cells complete={sum(s == 'COMPLETE' for s in states.values())}/52; "
          f"charged GPU seconds={charged_seconds(r):.3f}/{r['budget_seconds']}; estimate={r['estimated_seconds']} s")
    return 0 if green else 1


def run_campaign(r, resume=False):
    with campaign_lock() as fd:
        started = [c for c in ordered_cells(r) if cell_state(c) != 'UNSTARTED']
        if started and not resume:
            raise ValueError('existing cell state: use --resume; no started cell is retried')
        for battery in BATTERIES:
            for cell in ordered_cells(r):
                if cell['battery'] != battery or cell_state(cell) != 'UNSTARTED':
                    continue
                # GNU timeout bounds only the process group it starts. No signal
                # is sent by this controller; the worker cap is tighter (280s).
                command = ['timeout', '--signal=TERM', '--kill-after=5', '585',
                           sys.executable, __file__, '--cell', cell['id']]
                env = dict(os.environ, GRM_C2_CAMPAIGN_PARENT=str(os.getpid()))
                env['GRM_C2_CAMPAIGN_FD'] = str(fd)
                result = subprocess.run(command, cwd=ROOT, env=env, pass_fds=(fd,))
                if result.returncode:
                    print(f"STOP_ON_RED: {cell['id']}; returncode={result.returncode}", flush=True)
                    return result.returncode
            # Resume passes recorded RED scoring gates without changing them;
            # the final summary remains RED. Newly observed RED stops at once.
            score_path = OUT / 'scores' / f'{battery}.json'
            existed = score_path.exists()
            value = score_battery(r, battery)
            if value['status'] != 'GREEN' and not (resume and existed):
                return 1
    return 0


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--dry-run', action='store_true')
    g.add_argument('--run', action='store_true')
    g.add_argument('--cell')
    g.add_argument('--worker')
    g.add_argument('--score', choices=BATTERIES)
    g.add_argument('command', nargs='?', choices=['summary'])
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.resume and not args.run:
        p.error('--resume requires --run')
    try:
        r = effective_registration()
        if args.dry_run:
            print(json.dumps(dry_run(), indent=2)); return 0
        if args.command == 'summary':
            return summary(r)
        if args.worker:
            cell = old.cell_by_id(r, args.worker)
            value = old.worker(cell)
            value.update(receipt_bindings())
            create(OUT / 'cells' / cell['id'] / 'worker.json', value)
            return 0
        if args.cell:
            # A direct cell obtains the same campaign mutex. The campaign's
            # child inherits that locked FD through timeout, avoiding relock.
            fd = os.environ.get('GRM_C2_CAMPAIGN_FD')
            if fd is not None and os.fstat(int(fd)).st_ino == (OUT / 'campaign.lock').stat().st_ino:
                return run_cell(old.cell_by_id(r, args.cell), r)
            with campaign_lock():
                return run_cell(old.cell_by_id(r, args.cell), r)
        if args.score:
            return 0 if score_battery(r, args.score)['status'] == 'GREEN' else 1
        return run_campaign(r, args.resume)
    except (ValueError, OSError, KeyError) as exc:
        print(f'RED: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    # Amendment 3 owns all subsequent CLI execution, including legacy entry
    # points; keep old epoch receipts and their charges immutable.
    from scripts.grm_c2_epoch3 import main as epoch_main
    raise SystemExit(epoch_main())
