#!/usr/bin/env python3
"""C8 additive, foreground leased worker and CPU-only evidence reporting.

Prior art: C2 / EB1 / DET1 / CMC1, GRM contributors (2026), inspected locally.
Taken: C2 profile selection, fixture planting, durable continuation and flock
lease; DET1 same-model replay principle. Ours: C8 stage receipts and scheduling.
SHA-256 bindings and create-only reservations are established systems practice;
no prior art known to me for this precise adapter beyond those local systems.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.grm_c2_profile import create, read, sha, REGISTRY
from scripts import grm_c2_cells as c2
from scripts.grm_c8_profiler import TurnProfiler, CudaMeter, summarize
OUT = ROOT / 'artifacts/grm_c8'
REG = OUT / 'registration.json'
ANCHOR = '264ccd493f9f73fd157c47a7fa0ded3be60e0b3530e0ad3b42fae13d872d8345'


def registration():
    if sha(REG) != ANCHOR:
        raise ValueError('registration SHA differs from code anchor')
    value = read(REG)
    template = Path(__file__).read_text().replace(ANCHOR, 'PENDING_REGISTRATION')
    if hashlib.sha256(template.encode()).hexdigest() != value['runner_template_sha256']:
        raise ValueError('runner template changed')
    for name, digest in value['inputs'].items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if sha(path) != digest:
            raise ValueError(f'bound input changed: {name}')
    if value['estimated_seconds'] != sum(c['estimate_seconds'] for c in value['cells']):
        raise ValueError('estimate mismatch')
    return value


def bindings():
    return {'registration_sha256': sha(REG), 'executed_sources': c2.sources()}


def side_b():
    # Prior art: Amdahl (1967) component saving calculation; SP5 (2026)
    # is an external decode microbenchmark receipt, not a GRM E2E gate.
    return {'evidence_class': 'external receipt: synced decode microbenchmark',
        'source': '/mnt/Shared/APA_SP5_GPTOSS20B_Model_Test_Report_2026-09-08.md',
        'receipt_context_tokens': 2048, 'order_claimed_context_tokens': 12288,
        'context_status': 'RED_CONTEXT_MISMATCH', 'standard_ms_tok': 82.6,
        'two_pass_ms_tok': 84.0, 'single_pass_ms_tok': 84.2,
        'two_pass_saving_ms_tok': 82.6 - 84., 'single_pass_saving_ms_tok': 82.6 - 84.2,
        'two_pass_saving_fraction': (82.6 - 84.) / 82.6,
        'single_pass_saving_fraction': (82.6 - 84.2) / 82.6,
        'positive_saving_demonstrated': False,
        'saving_at_12288_or_width96': None,
        'scope': '12,288 is memory ceiling; 96 token seats is not total context length'}


def dry_run():
    r = registration()
    return {'status': 'NON_FIT_BUDGET' if r['estimated_seconds'] > r['budget_seconds'] else 'READY_FOR_LEAD',
        'gpu_executed': False, 'cells': r['cells'], 'cell_count': len(r['cells']),
        'estimated_seconds': r['estimated_seconds'], 'budget_seconds': r['budget_seconds'],
        'worker_seconds': 285, 'outer_seconds': 590, 'cooldown_seconds': 30,
        'measurement_status': 'NOT_RUN', 'side_b': side_b(), **bindings()}


def cell_spec(r, name):
    return next(c for c in r['cells'] if c['id'] == name)


def completed(cell_id):
    directory = OUT / 'cells' / cell_id
    ctl = read(directory / 'controller.json')
    rec = read(directory / 'worker.json')
    if ctl['status'] != 'COMPLETE' or ctl['worker_sha256'] != sha(directory / 'worker.json'):
        raise ValueError(f'dependency RED: {cell_id}')
    if ctl['registration_sha256'] != sha(REG) or rec['registration_sha256'] != sha(REG):
        raise ValueError('stale receipt')
    return rec


def fit(r):
    if r['estimated_seconds'] > r['budget_seconds']:
        raise ValueError(f'NON_FIT_BUDGET: {r["estimated_seconds"]}s > {r["budget_seconds"]}s; never execute/retry this plan')
    if any(c['estimate_seconds'] > 285 or c.get('status') == 'NON_FIT' for c in r['cells']):
        raise ValueError('NON_FIT_CELL')


def worker(cell, *, enabled=False):
    if os.environ.get('GRM_C8_LEASE_PARENT') != str(os.getppid()):
        raise RuntimeError('worker requires its foreground leased parent')
    r = registration(); fit(r)
    if not enabled:
        raise ValueError('measurement requires explicit --profile-turn')
    # GPU imports only below authorization, source binding and budget gates.
    from scripts import grm_e2e_session as e2e
    from scripts.grm_det1_3_gpu import _install_lived_nodes
    from core.graft_repository import GraftRepository
    registry = read(REGISTRY); flags = c2.flags_for('profile')
    flags['demand_ngh'] = cell['battery'] == 'demand'
    if os.environ.get('GRM_DEMAND_NGH') != str(int(flags['demand_ngh'])):
        raise ValueError('demand environment mismatch')
    directory = OUT / 'cells' / cell['id']
    session = directory / 'session'
    constructor = []
    original = e2e.GraftRepository
    def record_ctor(*a, **kw):
        constructor.append((a, kw)); return original(*a, **kw)
    e2e.GraftRepository = record_ctor
    repo = meter = None
    rows = []
    try:
        if cell.get('continuation'):
            previous = OUT / 'cells' / cell['continuation'] / 'session'
            prior = completed(cell['continuation'])
            if c2.tree_hashes(previous) != prior['session_sha256']:
                raise ValueError('continuation content changed')
            shutil.copytree(previous, session)
            c2.assert_payloads(session / 'repository')
        elif cell['battery'] == 'demand':
            previous = OUT / 'cells' / cell['depends'][-1] / 'demand_input'
            checkpoint = c2.validate_checkpoint(previous)
            shutil.copytree(previous / 'repository', session / 'repository')
            create(session / 'state.json', checkpoint['context'])
        else:
            session.mkdir()
        args = c2.args_for(e2e, session, flags)
        model, tokenizer, repo, model_info = e2e.load_model_and_repo(args, session)
        if (repo.arena.width, repo.arena.n_sink, repo.arena.live_shift, repo.arena.ephemeral) != (96, 19, 115, True):
            raise ValueError('C2 geometry mismatch')
        if list(repo.arena.encode(e2e.HARMONY_SINK)) != registry['frame']['sink_token_ids']:
            raise ValueError('sink token identity mismatch')
        meter = CudaMeter()

        def reload_repo(path):
            a, kw = constructor[0]; a = list(a); a[3] = str(path)
            return GraftRepository(*a, **kw)

        def run_event(event, turn, state, selected=True):
            profiler = TurnProfiler(enabled=selected, meter=meter)
            profiler.run(e2e.run_turn, repo, event, turn,
                paths=e2e.stage_paths(session), transcript=state['transcript'],
                turn_records=state['turn_records'], probe_rows=state['probe_rows'],
                args=args, resumed=bool(cell.get('continuation')))
            if profiler.receipt:
                record = profiler.receipt
                record.update(turn=turn, kind=event['kind'], battery=cell['battery'],
                    answer_utf8_hex=state['transcript'][-1]['assistant'].encode().hex())
                rows.append(record)
            return state['transcript'][-1]['assistant']

        if cell['battery'] == 'identity':
            state = {'transcript': [], 'turn_records': {}, 'probe_rows': []}
            for turn in range(5):
                run_event(registry['session_script'][turn], turn, state, selected=False)
            repo.flush_now()
            c2.assert_payloads(session / 'repository')
            shutil.copytree(session / 'repository', directory / 'identity_input')
            repo.close(); repo = None
            values = []
            # One persisted pre-turn repository, same loaded weights; OFF/ON
            # are interleaved within one process. Never reharvest the input.
            for ordinal, selected in enumerate((False, True)):
                target = directory / f'identity_{ordinal}'
                shutil.copytree(directory / 'identity_input', target)
                repo = reload_repo(target)
                logits, tokens = [], []
                forward, finish = repo.arena._forward, repo.arena._finish_attempt
                def capture_forward(*a, **kw):
                    out = forward(*a, **kw)
                    logits.append({'shape': list(out.shape), 'dtype': str(out.dtype),
                                   'sha256': hashlib.sha256(out.tobytes()).hexdigest()})
                    return out
                def capture_finish(*a, **kw):
                    tokens.append(list(kw['out'])); return finish(*a, **kw)
                repo.arena._forward, repo.arena._finish_attempt = capture_forward, capture_finish
                answer = run_event(registry['session_script'][5], 5, copy.deepcopy(state), selected)
                values.append({'answer_utf8_hex': answer.encode().hex(), 'logits': logits, 'tokens': tokens})
                repo.arena._forward, repo.arena._finish_attempt = forward, finish
                repo.close(); repo = None
            create(directory / 'identity.json', {'arms': values, 'pass': values[0] == values[1], **bindings()})
            if not values[0]['logits'] or not values[0]['tokens'] or values[0] != values[1]:
                raise ValueError('GPU_BIT_IDENTITY_RED_OR_VACUOUS')
        elif cell['battery'] == 'sup':
            fixture = read(ROOT / 'tests/fixtures/supersession_battery' / f'{cell["fixture"]}.json')
            _install_lived_nodes(repo, e2e, fixture)
            repo.flush_now(); c2.assert_payloads(session / 'repository')
            shutil.copytree(session / 'repository', directory / 'sup_input')
            repo.close(); repo = None
            for repeat in range(4):
                for probe in registry['sup_plans'][cell['fixture']]:
                    target = directory / f'probe_{len(rows):02d}'
                    shutil.copytree(directory / 'sup_input', target)
                    repo = reload_repo(target)
                    profiler = TurnProfiler(enabled=enabled, meter=meter)
                    answer, info = profiler.run(e2e._probe_ladder_chat, repo, probe['question'],
                        topk=flags['topk'], ngen=flags['ngen'], max_trips=flags['max_trips'],
                        defer_memory=False, demand_ngh=False)
                    row = profiler.receipt
                    row.update(battery='sup', kind='probe', turn=len(rows), probe_id=probe['probe_id'],
                               repeat=repeat, answer_utf8_hex=answer.encode().hex(), info=info)
                    rows.append(row)
                    repo.close(); repo = None
        else:
            state_path = session / 'state.json'
            state = read(state_path) if state_path.exists() else {
                'transcript': [], 'turn_records': {}, 'probe_rows': [], 'next_turn': 0}
            state['turn_records'] = {int(k): v for k, v in state['turn_records'].items()}
            if state.get('next_turn', cell['start']) != cell['start']:
                raise ValueError('continuation turn mismatch')
            for turn in range(cell['start'], cell['stop']):
                if cell['battery'] == 'longhistory' and turn == 33:
                    context = copy.deepcopy(state); context['next_turn'] = 33
                    c2.snapshot(repo, directory / 'demand_input', context, str(os.getpid()))
                run_event(registry['session_script'][turn], turn, state, selected=enabled)
                state['next_turn'] = turn + 1
            # Session state is mutable worker output, not a registration.
            state_path.write_text(json.dumps(state, indent=2) + '\n')
            if cell['battery'] == 'demand':
                instrument = [json.loads(x) for x in e2e.stage_paths(session)['instrumentation'].read_text().splitlines()]
                info = instrument[-1]['info']
                rows[-1]['demand_info'] = info
                create(directory / 'demand_observation.json', {'info': info, **bindings()})
                if info.get('demand_fired') is not True or info.get('demand_trip_taken') is not True:
                    raise ValueError('RED_DEMAND_TRIP_NOT_OBSERVED; no alternative prompt or retry')
        if repo:
            repo.flush_now(); c2.assert_payloads(session / 'repository')
            repo.close(); repo = None
        create(directory / 'worker.json', {'status': 'COMPLETE', 'cell': cell, 'rows': rows,
            'session_sha256': c2.tree_hashes(session), 'model_info': model_info,
            'cuda_runtime_sha256': sha(meter.path), **bindings()})
    except BaseException as exc:
        create(directory / 'failure.json', {'status': 'RED', 'error': f'{type(exc).__name__}: {exc}',
            'partial_rows': rows, **bindings()})
        raise
    finally:
        e2e.GraftRepository = original
        if repo:
            repo.close()
        if meter:
            meter.close()


def run_cell(r, cell):
    fit(r)  # Before reservation, lease, imports or CUDA. NON_FIT never starts.
    OUT.mkdir(exist_ok=True)
    # Prior art: C2/CMC1 (2026), OS flock. Serialize campaign reservations as
    # well as the shared GPU; no lock clearing and no foreign process signals.
    with (OUT / 'campaign.lock').open('a') as campaign:
        fcntl.flock(campaign, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for dep in cell['depends']:
            completed(dep)
        used = 0.
        last_end = 0.
        for reservation in (OUT / 'cells').glob('*/reservation.json'):
            ctl = reservation.with_name('controller.json')
            if not ctl.exists():
                raise ValueError('unfinished reservation: fail closed, never retry')
            record = read(ctl)
            if record['status'] != 'COMPLETE':
                raise ValueError('prior RED: stop campaign, never retry')
            used += record['charged_seconds']
            last_end = max(last_end, record['ended_epoch'])
        if used + 285 > r['budget_seconds']:
            raise ValueError('remaining budget cannot reserve 285s')
        delay = max(0., 30 - (time.time() - last_end))
        if delay:
            time.sleep(delay)
        directory = OUT / 'cells' / cell['id']
        directory.mkdir(parents=True, exist_ok=False)
        create(directory / 'reservation.json', {'seconds': 285, 'cell': cell, **bindings()})
        started = None; status = 'RED'; error = None; charge = 285.
        try:
            from scripts.grm_cmc1_gpu_arms import gpu_lease
            with gpu_lease(285, 240):
                flags = c2.flags_for('profile'); flags['demand_ngh'] = cell['battery'] == 'demand'
                env = c2.environment(flags); env['GRM_C8_LEASE_PARENT'] = str(os.getpid())
                started = time.monotonic()
                with (directory / 'worker.log').open('x') as stream:
                    result = subprocess.run([sys.executable, __file__, '--worker', cell['id'], '--profile-turn'],
                        env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=280)
                charge = time.monotonic() - started
                if result.returncode:
                    raise RuntimeError(f'worker returncode={result.returncode}')
                status = 'COMPLETE'
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        finally:
            create(directory / 'controller.json', {'status': status, 'error': error,
                'charged_seconds': charge, 'ended_epoch': time.time(),
                'worker_sha256': sha(directory / 'worker.json') if (directory / 'worker.json').exists() else None,
                **bindings()})
        return 0 if status == 'COMPLETE' else 1


def summary():
    r = registration(); result = {'batteries': {}, 'side_b': side_b()}
    complete = True
    for battery in ('sup', 'census', 'longhistory', 'demand'):
        rows = []; errors = []
        cells = [c for c in r['cells'] if c['battery'] == battery]
        for cell in cells:
            try:
                rec = completed(cell['id'])
                if len(rec['rows']) != cell['turns']:
                    raise ValueError('turn count mismatch')
                rows.extend(rec['rows'])
            except (OSError, ValueError, KeyError) as exc:
                errors.append(str(exc))
        result['batteries'][battery] = summarize(rows, minimum=1 if battery == 'demand' else 30)
        result['batteries'][battery]['errors'] = errors
        if errors:
            result['batteries'][battery]['status'] = 'INCOMPLETE_CELLS'
            result['batteries'][battery]['decision'] = None
        complete = complete and not errors
    try:
        completed('gpu-identity')
    except (OSError, ValueError, KeyError):
        complete = False
    result['status'] = 'COMPLETE' if complete else 'BLOCKED_NOT_MEASURED'
    # No pooled cross-battery average hides a conflicting allocation result.
    decisions = {result['batteries'][b]['decision'] for b in ('sup', 'census', 'longhistory')}
    result['allocation_decision'] = (next(iter(decisions)) if complete and len(decisions) == 1 else None)
    result['conflicting_batteries'] = complete and len(decisions) > 1
    result.update(bindings())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dry-run', action='store_true')
    group.add_argument('--blocked-report', action='store_true')
    group.add_argument('--preflight', action='store_true')
    group.add_argument('--summary', action='store_true')
    group.add_argument('--cell')
    group.add_argument('--worker')
    parser.add_argument('--profile-turn', action='store_true', default=False)
    args = parser.parse_args(argv)
    if args.dry_run or args.blocked_report:
        value = dry_run()
        if args.blocked_report:
            value.update(status='BLOCKED', reasons=['NO_GPU_IN_SANDBOX', value['status'],
                'SP5_CONTEXT_MISMATCH'], allocation_decision=None)
        print(json.dumps(value, indent=2)); return 0
    if args.summary:
        value = summary(); print(json.dumps(value, indent=2))
        return 0 if value['status'] == 'COMPLETE' else 2
    r = registration()
    if args.preflight:
        fit(r); return 0
    cell = cell_spec(r, args.worker or args.cell)
    if args.worker:
        worker(cell, enabled=args.profile_turn); return 0
    if not args.profile_turn:
        raise ValueError('explicit --profile-turn required')
    return run_cell(r, cell)


if __name__ == '__main__':
    raise SystemExit(main())
