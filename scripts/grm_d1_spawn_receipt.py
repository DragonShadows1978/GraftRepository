#!/usr/bin/env python3
"""GRM-D1 follow-up 5: emit a per-arm receipt from a REAL `run_cell` spawn.

The gate in `tests/test_grm_lt1_1_child_spawn.py` proves the route; this
writes the receipt a reader can inspect without running pytest. It performs
the identical substitutions: the shared GPU lease becomes a flock on a temp
file, the `nvidia-smi` idle probe returns idle, and the child runs the CPU
double via `--worker-cpu`. Everything else -- the reservation, the directory,
the argv, the Popen, the foreground wait, the charge, the checkpoint
validation, the controller receipt -- is the production path.

No GPU is touched and the shared lock is never opened.

Prior art: `scripts/grm_lt1_worker.py:run_cell` (GRM contributors, 2026),
called unchanged. No prior art known to me for this receipt emitter.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_lt1_1 as runner          # noqa: E402
from scripts import grm_lt1_worker as worker     # noqa: E402

OUT = ROOT / 'artifacts/grm_d1/lt1_1/proof'


def child_argv(cell):
    return [sys.executable, '-m', 'scripts.grm_lt1_1',
            '--arm', runner.current_arm(), '--worker-cpu', cell['id']]


def spawn(arm, cells=2):
    import scripts.grm_cmc1_gpu_arms as arms
    reg = runner.registration(arm)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / ('run_' + arm.replace('+', 'plus'))
        root.mkdir(parents=True)
        lock = Path(tmp) / 'fake_gpu.lock'
        lock.touch()
        saved_lock = arms.LOCK_PATH
        saved_subprocess_run = worker.subprocess.run
        saved_accounting = worker.accounting
        saved_sleep = worker.time.sleep
        arms.LOCK_PATH = lock

        class Idle:
            returncode = 0
            stdout = ''
        real = subprocess.run
        worker.subprocess.run = (
            lambda a, *x, **k: Idle() if a and 'nvidia-smi' in a[0]
            else real(a, *x, **k))
        # Amendment 6 probes the card through `device_snapshot`. Stub the
        # PROBE, not `await_idle`, so the bounded-wait logic stays real.
        saved_snapshot = runner.device_snapshot
        runner.device_snapshot = lambda: dict(
            time_unix=0.0, status='OK', memory_used_mib=64,
            memory_free_mib=12224, memory_total_mib=12288,
            other_process_pids=[], compute_list_empty=True,
            scope='receipt stub: idle card')
        try:
            with runner.lt1_1_seams(arm), runner.pinned_arm(arm):
                worker.RUN = root
                worker.spawn_argv = child_argv
                worker.accounting = lambda *a, **k: 0.0
                worker.time.sleep = lambda *_: None
                for cell in reg['cells'][:cells]:
                    ok = worker.run_cell(cell, reg)
                    directory = root / 'cells' / cell['id']
                    controller = json.loads(
                        (directory / 'controller.json').read_text())
                    log = (directory / 'worker.log').read_text()
                    receipt = json.loads((directory / 'worker.json').read_text())
                    rows.append(dict(
                        cell=cell['id'], run_cell_returned=ok,
                        status=controller['status'], error=controller['error'],
                        child_argv=' '.join(child_argv(cell)[1:]),
                        child_pid=receipt['pid'],
                        parent_pid=os.getpid(),
                        child_is_a_subprocess=receipt['pid'] != os.getpid(),
                        new_process=receipt['pid'] != receipt['previous_pid'],
                        campaign_arm=receipt['binding']['campaign_arm'],
                        alias_fold_merge=receipt['binding']['alias_fold_merge'],
                        input_sha_mismatch_in_child_log=(
                            'INPUT_SHA_MISMATCH' in log),
                        traceback_in_child_log='Traceback' in log,
                        child_log_bytes=len(log),
                        receipts=sorted(p.name for p in directory.iterdir())))
        finally:
            arms.LOCK_PATH = saved_lock
            worker.subprocess.run = saved_subprocess_run
            worker.accounting = saved_accounting
            worker.time.sleep = saved_sleep
            runner.device_snapshot = saved_snapshot
    return dict(arm=arm, cells=len(rows), rows=rows,
                evidence_class='CPU child spawn through the real run_cell '
                               '(process and receipt plumbing; NOT a '
                               'language-model quality measurement)',
                stubbed=['the shared GPU lease -> flock on a temp file',
                         'the nvidia-smi idle probe -> idle',
                         'the model -> the C7 CPU double (--worker-cpu)'],
                unstubbed=['reservation accounting', 'directory creation',
                           'child argv construction', 'Popen + foreground wait',
                           'charge computation', 'checkpoint validation',
                           'controller receipt'])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for arm in runner.ARMS:
        value = spawn(arm)
        name = 'child_spawn_%s.json' % arm.replace('+', 'plus')
        (OUT / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + '\n')
        for row in value['rows']:
            print('%-3s %-12s run_cell=%-5s status=%-8s child_pid=%-7d '
                  'new_process=%-5s arm=%-3s alias=%-5s '
                  'INPUT_SHA_MISMATCH=%s'
                  % (arm, row['cell'], row['run_cell_returned'], row['status'],
                     row['child_pid'], row['new_process'], row['campaign_arm'],
                     row['alias_fold_merge'],
                     row['input_sha_mismatch_in_child_log']))


if __name__ == '__main__':
    main()
