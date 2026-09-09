#!/usr/bin/env python3
"""One registered FIX8 F5 successor; foreground lead execution only.

Prior art: GRM C7/CMC1/FIX8 (contributors, 2026): create-only receipts,
hash binding, checkpoint reader, pessimistic foreground leases, reused here.
NVIDIA nvidia-smi docs (accessed 2026-09-09): XML FB memory and process list,
https://docs.nvidia.com/deploy/nvidia-smi/ . Ours: this exact F5 successor
and 1000 MiB gate composition; no prior art known to me for that composition.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import grm_scout_fix8_replay as original
from scripts.grm_c7_amendment7 import read, sha, write, need

OUT = original.OUT / 'resume_amendment_1'
REG = OUT / 'registration.json'


def verify():
    base = original.verify()  # all original pins, including the prior amendment
    need(sha(REG) == REG.with_suffix('.sha256').read_text().split()[0],
         'RESUME_REGISTRATION_SHA_MISMATCH')
    r = read(REG)
    need(r['registration_sha256'] == sha(original.REG) and
         r['previous_amendment_sha256'] == sha(original.OUT / 'replay_amendment_1.json'),
         'RESUME_CHAIN_MISMATCH')
    for path, digest in r['inputs'].items():
        need(sha(ROOT / path) == digest, 'RESUME_INPUT_SHA_MISMATCH: ' + path)
    for path, digest in r['historical_files'].items():
        need(sha(ROOT / path) == digest, 'HISTORICAL_RECEIPT_CHANGED: ' + path)
    need(set(p.name for p in (original.OUT / 'gpu').iterdir()) ==
         {'F1', 'F2', 'F3', 'F4', 'F5'}, 'UNREGISTERED_ORIGINAL_BATCH')
    for b in ('F1', 'F2', 'F3', 'F4', 'F5'):
        d = original.OUT / 'gpu' / b
        c = read(d / 'controller.json')
        need(c['registration_sha256'] == sha(original.REG) and
             c['status'] == ('FAILED' if b == 'F5' else 'COMPLETE') and
             c['charged_seconds'] == base['lease_seconds'], 'HISTORICAL_STATE_MISMATCH')
    need(read(original.OUT / 'gpu/F5/controller.json')['error'] ==
         'RuntimeError: cudaMalloc failed: out of memory', 'NOT_REGISTERED_OOM')
    need(r['batches'] == {
        'F5-R1': {'probe_ids': base['batches']['F5'], 'lease_seconds': 120},
        'F6': {'probe_ids': base['batches']['F6'], 'lease_seconds': 280}},
        'RESUME_COHORT_OR_LEASE_CHANGED')
    need(r['historical_charged_seconds'] == 1400 and
         r['gpu_cap_seconds'] == base['gpu_cap_seconds'] == 1800 and
         r['memory_limit_mib'] == 1000, 'RESUME_RAIL_CHANGED')
    return r, base


def binding():
    return {'registration_sha256': sha(original.REG),
            'resume_registration_sha256': sha(REG)}


def mib(text):
    parts = (text or '').split()
    need(len(parts) == 2 and parts[1] == 'MiB', 'MEMORY_VALUE_UNAVAILABLE')
    value = int(parts[0])
    need(value >= 0, 'NEGATIVE_MEMORY_VALUE')
    return value


def parse_memory(xml, visible, own_pid):
    # Prior art: NVIDIA nvidia-smi XML framebuffer/process reporting (2026
    # accessed). Total FB used conservatively includes unattributed/graphics
    # memory; never infer an idle card merely from an empty compute list.
    root = ET.fromstring(xml)
    gpus = root.findall('gpu')
    need(len(gpus) == 1, 'REQUIRES_SINGLE_PHYSICAL_GPU')
    gpu = gpus[0]
    device = gpu.findtext('uuid')
    need(bool(device) and device.startswith('GPU-'), 'GPU_UUID_UNAVAILABLE')
    need(visible in (None, '0', device), 'UNSUPPORTED_CUDA_VISIBLE_DEVICES')
    used = mib(gpu.findtext('fb_memory_usage/used'))
    total = mib(gpu.findtext('fb_memory_usage/total'))
    need(total > 0 and used <= total, 'INVALID_DEVICE_MEMORY')
    process_node = gpu.find('processes')
    need(process_node is not None, 'PROCESS_TELEMETRY_UNAVAILABLE')
    need(not (process_node.text or '').strip() or
         (process_node.text or '').strip() == 'No running processes found',
         'PROCESS_TELEMETRY_UNAVAILABLE')
    processes = []
    for p in process_node.findall('process_info'):
        pid = int(p.findtext('pid'))
        need(pid > 0, 'INVALID_PROCESS_PID')
        processes.append({'pid': pid, 'type': p.findtext('type'),
                          'name': p.findtext('process_name'),
                          'memory.used': mib(p.findtext('used_memory'))})
    return {'gpu_uuid': device, 'memory.used': used, 'memory.total': total,
            'unit': 'MiB', 'processes': processes,
            'pids': sorted({p['pid'] for p in processes}), 'own_pid': own_pid,
            'own_memory_mib': sum(p['memory.used'] for p in processes if p['pid'] == own_pid),
            'other_process_pids': sorted({p['pid'] for p in processes if p['pid'] != own_pid}),
            'scope': 'point sample, not peak; device total includes unattributed memory'}


def snapshot(path, stage):
    value = {'stage': stage, 'time_unix': time.time(), **binding()}
    try:
        proc = subprocess.run(['nvidia-smi', '-q', '-x'], capture_output=True,
                              text=True, timeout=5, check=False)
        value.update(raw_xml=proc.stdout, stderr=proc.stderr, returncode=proc.returncode)
        need(proc.returncode == 0, 'NVIDIA_SMI_FAILED')
        value.update(parse_memory(proc.stdout, os.environ.get('CUDA_VISIBLE_DEVICES'), os.getpid()))
        value['status'] = 'OK'
    except Exception as exc:
        value.update(status='ERROR', error=f'{type(exc).__name__}: {exc}')
    if stage in ('before_lease', 'under_lock_before_load'):
        try:
            memory_gate(value, 1000)
            value['admission'] = 'ALLOW'
        except ValueError as exc:
            value.update(admission='REFUSE', refusal_reason=str(exc))
    write(path, value)
    return value


def memory_gate(value, limit):
    need(value['status'] == 'OK', 'DEVICE_MEMORY_PROBE_FAILED')
    need(value['own_memory_mib'] == 0, 'REQUIRES_FRESH_GPU_WORKER')
    need(value['memory.used'] <= limit,
         f"DEVICE_MEMORY_BUSY: memory.used={value['memory.used']} MiB > {limit}; "
         f"pids={value['pids']}")


def campaign_state(r):
    # Prior art: FIX8 pessimistic reservations/O_EXCL and C7 immutable
    # successor chains (GRM, 2026). Exception is only the pinned old F5;
    # no wildcard retry, charge refund, failed successor or orphan bypass.
    charged = r['historical_charged_seconds']
    complete = []
    gpu = OUT / 'gpu'
    if not gpu.exists():
        return charged, complete
    need(all(p.name in r['batches'] and p.is_dir() for p in gpu.iterdir()), 'UNKNOWN_RESUME_BATCH')
    missing = False
    for b, spec in r['batches'].items():
        d = gpu / b
        if not d.exists():
            missing = True
            continue
        need(not missing, 'PRIOR_BATCH_INCOMPLETE')
        need((d / 'reservation.json').exists() and (d / 'controller.json').exists(),
             'ORPHAN_RESERVATION_STOP')
        reservation = read(d / 'reservation.json')
        c = read(d / 'controller.json')
        for value in (reservation, c):
            need(all(value.get(k) == v for k, v in binding().items()) and
                 value.get('batch') == b, 'RESUME_RESULT_BINDING_MISMATCH')
        need(reservation['seconds'] == spec['lease_seconds'], 'INVALID_RESERVATION')
        seconds = c['charged_seconds']
        need(isinstance(seconds, (int, float)) and math.isfinite(seconds) and
             seconds >= spec['lease_seconds'] and seconds >= c['elapsed_seconds'], 'INVALID_CHARGE')
        charged += seconds
        need(c['status'] == 'COMPLETE', 'FAILED_SUCCESSOR_CAMPAIGN_STOP')
        need(c['rows_attempted'] == c['rows_completed'] == spec['probe_ids'], 'INCOMPLETE_BATCH_ROWS')
        need({p.stem for p in d.glob('c7_*.json')} == set(spec['probe_ids']), 'BATCH_ROW_COHORT_MISMATCH')
        for pid in spec['probe_ids']:
            row = read(d / (pid + '.json'))
            need(row['registration_sha256'] == sha(original.REG) and row['probe_id'] == pid,
                 'ROW_BINDING_MISMATCH')
        complete.append(b)
    need(charged <= r['gpu_cap_seconds'], 'GPU_BUDGET_RAIL')
    return charged, complete


def batch(batch_id):
    r, base = verify()
    need(batch_id in r['batches'], 'UNKNOWN_BATCH')
    need(shutil.disk_usage(OUT).free >= base['free_space_min_bytes'], 'FREE_SPACE_BELOW_20_GIB')
    owner = original.OUT / 'replay.active'  # shared with unchanged old controller
    fd = os.open(owner, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(str(os.getpid()))
    try:
        charged, complete = campaign_state(r)
        need(batch_id not in complete, 'SUCCESSOR_ALREADY_CONSUMED')
        need(complete == list(r['batches'])[:list(r['batches']).index(batch_id)],
             'PRIOR_BATCH_INCOMPLETE')
        spec = r['batches'][batch_id]
        need(charged + sum(s['lease_seconds'] for b, s in r['batches'].items() if b not in complete)
             <= r['gpu_cap_seconds'], 'GPU_BUDGET_RAIL')
        preflight = OUT / 'probes' / f'{batch_id}-{uuid.uuid4().hex}.json'
        sample = snapshot(preflight, 'before_lease')
        memory_gate(sample, r['memory_limit_mib'])
        destination = OUT / 'gpu' / batch_id
        destination.mkdir(parents=True, exist_ok=False)
        write(destination / 'reservation.json', {
            **binding(), 'batch': batch_id, 'seconds': spec['lease_seconds'],
            'preflight': str(preflight), 'preflight_sha256': sha(preflight),
            'predecessor_controller_sha256': sha(original.OUT / 'gpu/F5/controller.json')})
        execute(batch_id, spec, destination, base, r)
    finally:
        owner.unlink()  # only the file created by this invocation


def execute(batch_id, spec, destination, base, r):
    started = None
    elapsed = 0.0
    status = 'FAILED'
    error = None
    trace = None
    stage = 'setup'
    attempted, completed, samples = [], [], []
    loaded = None

    def capture(label):
        p = destination / 'memory' / f'{len(samples):03d}.json'
        value = snapshot(p, label)
        samples.append(str(p))
        return value

    try:
        from scripts.grm_c2_cells import environment
        env = environment(base['flags'])
        for k in list(os.environ):
            if k.startswith('GRM_'):
                del os.environ[k]
        os.environ.update({k: v for k, v in env.items() if k.startswith('GRM_')})
        from scripts.grm_cmc1_gpu_arms import gpu_lease
        from scripts.grm_c7_middle_replay import open_copy
        qs = {q['probe_id']: q for q in read(original.OUT / 'requests.json')}
        stage = 'acquire_lease'
        with gpu_lease(spec['lease_seconds'], 0):
            started = time.monotonic()
            try:
                stage = 'under_lock_before_load'
                memory_gate(capture(stage), r['memory_limit_mib'])
                for pid in spec['probe_ids']:
                    attempted.append(pid)
                    repo = None
                    try:
                        stage = f'{pid}:open_copy'
                        capture(stage)
                        repo, loaded = open_copy(qs[pid], destination / 'sessions' / pid, base, loaded)
                        stage = f'{pid}:serve'
                        capture(stage)
                        original.serve(repo, qs[pid], destination)
                        completed.append(pid)
                        capture(f'{pid}:after_serve')
                    except BaseException:
                        # Sample before repo.close; traceback retains actual failure site.
                        trace = traceback.format_exc()
                        capture(f'{stage}:failure_before_close')
                        raise
                    finally:
                        if repo is not None:
                            repo.close()
                    stage = f'{pid}:after_close'
                    capture(stage)
                stage = 'model_receipt'
                write(destination / 'model.json', loaded[2])
                elapsed = time.monotonic() - started
                need(elapsed <= spec['lease_seconds'], 'LEASE_OVERRUN_RED')
                status = 'COMPLETE'
            except BaseException:
                if trace is None:
                    trace = traceback.format_exc()
                    capture(f'{stage}:failure')
                raise
    except BaseException as exc:
        status = 'FAILED'
        error = f'{type(exc).__name__}: {exc}'
        if trace is None:
            trace = traceback.format_exc()
    finally:
        if started is not None:
            elapsed = time.monotonic() - started
        if elapsed > spec['lease_seconds']:
            status = 'FAILED'
            error = error or 'LEASE_OVERRUN_RED'
        write(destination / 'controller.json', {
            **binding(), 'batch': batch_id, 'status': status, 'error': error,
            'traceback': trace, 'stage': stage, 'rows_attempted': attempted,
            'rows_completed': completed, 'memory_receipts': samples,
            'elapsed_seconds': elapsed, 'charged_seconds': max(spec['lease_seconds'], elapsed),
            'historical_F5_status': 'FAILED', 'historical_RED_preserved': True,
            'memory_scope': 'boundary/failure snapshots, not continuous or certified peak'})
        if started is not None:
            time.sleep(30)  # lead-only foreground cooldown outside the lease
    if status != 'COMPLETE':
        raise RuntimeError(error)


def summary():
    r, base = verify()
    stop = None
    try:
        campaign_state(r)
    except (ValueError, KeyError, OSError) as exc:
        stop = str(exc)
    rows = []
    charge = r['historical_charged_seconds']
    complete = stop is None
    for b, spec in r['batches'].items():
        d = OUT / 'gpu' / b
        if not (d / 'controller.json').exists():
            complete = False
        else:
            charge += read(d / 'controller.json')['charged_seconds']
            complete &= read(d / 'controller.json')['status'] == 'COMPLETE'
    for b, ids in base['batches'].items():
        d = original.OUT / 'gpu' / b if b in ('F1', 'F2', 'F3', 'F4') else OUT / 'gpu' / ('F5-R1' if b == 'F5' else b)
        for pid in ids:
            p = d / (pid + '.json')
            if not p.exists():
                complete = False
                continue
            value = read(p)
            need(value['registration_sha256'] == sha(original.REG) and value['probe_id'] == pid,
                 'ROW_BINDING_MISMATCH')
            rows.append(value)
    complete &= len(rows) == 24 and charge <= r['gpu_cap_seconds']
    qs = {q['probe_id']: q for q in read(original.OUT / 'requests.json')}
    groups = {}
    for group in ('fresh', 'folded', 'controls'):
        cohort = [v for v in rows if (qs[v['probe_id']]['probe']['expected'] == 'UNKNOWN'
            if group == 'controls' else qs[v['probe_id']]['probe']['expected'] != 'UNKNOWN'
            and qs[v['probe_id']]['probe']['class'] == group)]
        groups[group] = {'rows': len(cohort), **{k: sum(v['score'][k] for v in cohort) for k in
            ('exact_correct', 'wrong_value_error', 'abstention_error', 'unsupported_answer_error')}}
    admitted = sum(v['admitted'] and not v['identifier_unbound'] and bool(v['mounted_ids']) for v in rows)
    return {**binding(), 'status': 'RED', 'historical_F5_status': 'FAILED',
            'historical_RED_preserved': True, 'campaign_stop': stop,
            'replay_status': ('PASS_ADMISSION' if admitted == 24 else 'RED') if complete else 'NOT_MEASURED',
            'complete': bool(complete), 'rows': len(rows), 'admitted_and_mounted': admitted,
            'charged_seconds': charge, 'groups': groups,
            'historical_partial_F5_rows_used': False,
            'quality_rule': base['score']}


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--batch', choices=['F5-R1', 'F6'])
    mode.add_argument('--summary', action='store_true')
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--pending', action='store_true')
    args = ap.parse_args()
    if args.batch:
        batch(args.batch)
    elif args.summary:
        value = summary()
        print(json.dumps(value, indent=2))
        need(value['complete'], 'RESUME_INCOMPLETE')
    else:
        r, _ = verify()
        charged, complete = campaign_state(r)
        if args.pending:
            print('\n'.join(b for b in r['batches'] if b not in complete))
            return
        print(json.dumps({**binding(), 'status': 'REGISTERED_NOT_RUN' if not complete else 'RESUME_STATE_VERIFIED',
                          'completed_successors': complete, 'charged_seconds': charged,
                          'remaining_reserved_seconds': sum(s['lease_seconds'] for b, s in r['batches'].items() if b not in complete),
                          'gpu_cap_seconds': r['gpu_cap_seconds'],
                          'historical_RED_preserved': True, 'gpu_executed': False,
                          'device_probe_executed': False}, indent=2))


if __name__ == '__main__':
    main()
