#!/usr/bin/env python3
"""LT1.1 runner — drives the EXISTING LT1 worker machinery, one arm per run.

WHY THIS EXISTS. The LT1.1 registration, its amendment 1 and its
`dialogue.json` described a campaign that nothing could execute:
`scripts/grm_lt1.py` has no `--run/--arm/--registration/--amendment/--fixture/
--out`; it reads fixed module paths and drives a single hard-wired campaign.
A registration is not runnable until a worker executes it. This module is that
worker entry point.

PARAMETERIZED, NOT FORKED. The cell/lease/checkpoint logic is NOT copied. The
reusable core already exists and is already parameterized:

    scripts/grm_lt1_worker.execute(cell, directory, registration, loader,
                                   *, run=..., deadline=..., fake=...)

`execute` takes its cell, output directory, registration, model loader, run
root and fake flag as arguments -- `scripts/grm_lt1_worker_cpu.py` already
reuses it exactly this way. Only THREE things in that module are bound to
module-level LT1 state rather than passed in:

    lt.FIX      the fixture path `execute` reads for turns/probes/decisions
    lt.binding  the SHA binding stamped into every receipt (via worker.bind)
    RUN         the campaign root -- already a keyword argument

So this runner redirects exactly those three and calls the existing functions.
`run_cell`'s lease/flock/nvidia-smi/charge logic and `pending`'s
resume-and-verify logic are reused through the same redirection, not
reimplemented. The one thing genuinely new here is the A/A+ arm split: the
alias flag pinned AFTER `environment(flags)` and read back (R1 `pin_rule`),
then restored -- LT1 has no concept of two variants of one arm.

WHAT IS NOT REUSED AND WHY. `grm_lt1_worker.run_cell` re-enters the worker as
`-m scripts.grm_lt1_worker --worker`, which would load LT1's module state in
the child. The child is therefore re-entered through THIS module
(`--worker`), so it inherits the same redirection. That is a different entry
point, not a second copy of the lease logic.

ARMS. `A` and `A+` run the same 26 cells over the same frozen conversation and
differ by exactly one environment variable. Their receipts go to different
directories, so either arm may be run, interrupted and resumed alone.

Prior art:
  * `scripts/grm_lt1_worker.py` (execute / run_cell / pending / worker) and
    `scripts/grm_lt1.py` (verify / preflight / score / create / read) — GRM
    contributors, 2026. REUSED by redirection; no cell, lease, checkpoint,
    accounting or scoring logic is reimplemented here.
  * `scripts/grm_lt1_worker_cpu.py` (GRM contributors, 2026) — the pattern of
    calling `execute` with a CPU loader, an arbitrary run root and
    `fake=True`. Taken verbatim as the fake-path shape.
  * R1 `pin_rule` (`scripts/grm_r1_replay.py:242`, GRM contributors 2026) —
    pin the arm variable AFTER `environment(flags)` strips ambient `GRM_*`,
    then read back and assert. Taken; extended to a second variable and to
    restoring the prior value on exit.
  * C2 `environment(flags)` (`scripts/grm_c2_cells.py:47`) — the ambient strip.
  * Mine: the arm-variant split over one registration, the three-seam
    redirection, and the `--arm`-scoped resumable campaign root.
  * No prior art known to me for this exact composition.
"""
from __future__ import annotations
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_lt1 as lt                      # noqa: E402
from scripts import grm_lt1_worker as worker           # noqa: E402

OUT = ROOT / 'artifacts/grm_d1/lt1_1'
REGISTRATION = OUT / 'registration.json'
AMENDMENT1 = OUT / 'amendment1.json'
FIXTURE = OUT / 'dialogue.json'

ALIAS_ENV = 'GRM_ALIAS_FOLD_MERGE'
RULE_ENV = 'GRM_ADMISSION_RULE'
ARMS = ('A', 'A+')

#: Arm -> alias flag value (None = unset). The ONLY difference between arms.
ARM_ALIAS = {'A': None, 'A+': '1'}

#: LT1.1 inherits LT1's restart sentinels; the base registration is a delta
#: over LT1's and does not restate them.
RESTART_SENTINELS = ('recall_1_10', 'recall_4_10')


def out_dir(arm):
    """Per-arm campaign root. Distinct roots are what make arms resumable."""
    return OUT / ('run_A' if arm == 'A' else 'run_Aplus')


def sha(path):
    return lt.sha(Path(path))


# ------------------------------------------------------------------ binding

def binding(arm):
    """The SHA binding stamped into every LT1.1 receipt.

    Same role as `lt.binding` but bound to LT1.1's documents and to the core
    inputs amendment 1 rebound. The arm name is IN the binding, so a receipt
    written under one arm can never validate under the other.
    """
    amendment = json.loads(AMENDMENT1.read_text())
    core = {n: c['after_sha256']
            for n, c in amendment['core_rebind']['changed'].items()}
    core.update({n: e['sha256']
                 for n, e in amendment['core_rebind']['new_inputs'].items()})
    return dict(arm=arm,
                registration_sha256=sha(REGISTRATION),
                amendment1_sha256=sha(AMENDMENT1),
                fixture_sha256=sha(FIXTURE),
                admission_rule='margin_first',
                alias_fold_merge=(ARM_ALIAS[arm] is not None),
                rebound_core_shas=core)


def registration(arm):
    """The registration `execute` consumes, assembled for one arm."""
    if arm not in ARMS:
        raise ValueError('LT11_UNKNOWN_ARM: ' + str(arm))
    if sha(AMENDMENT1) != (OUT / 'amendment1.sha256').read_text().split()[0]:
        raise ValueError('LT11_AMENDMENT1_SHA_MISMATCH')
    amendment = json.loads(AMENDMENT1.read_text())
    if amendment['registration_sha256'] != sha(REGISTRATION):
        raise ValueError('LT11_AMENDMENT1_CHAIN_MISMATCH')
    base = json.loads(REGISTRATION.read_text())
    spec = amendment['arms'][arm]
    if sha(FIXTURE) != spec['fixture']['sha256']:
        raise ValueError('LT11_FIXTURE_SHA_MISMATCH')

    value = dict(base)
    # `execute` indexes `registration['arms'][cell['arm']]`, and every cell
    # carries arm "A" in both variants (same schedule). Expose the running
    # arm's flags under that key rather than renaming 26 cells.
    value['arms'] = {'A': dict(base['arms']['A'], flags=spec['flags'],
                               admission_rule='margin_first')}
    value['cells'] = spec['cells']
    value['effective_admission_rule'] = 'margin_first'
    value['restart_sentinels'] = list(RESTART_SENTINELS)
    value['lt1_1'] = dict(arm=arm, alias_fold_merge=ARM_ALIAS[arm] is not None,
                          out_dir=str(out_dir(arm).relative_to(ROOT)),
                          amendment1_sha256=sha(AMENDMENT1))
    return value


# ------------------------------------------------------------- the arm pin

def arm_environment(arm, flags):
    """`environment(flags)` then the arm pin, in that order (R1 contract).

    `environment(flags)` deletes every ambient `GRM_*`, so the pin MUST come
    after it or the arm silently runs the default.
    """
    from scripts.grm_c2_cells import environment
    env = environment(flags)
    env[RULE_ENV] = 'margin_first'
    env.pop(ALIAS_ENV, None)
    if ARM_ALIAS[arm] is not None:
        env[ALIAS_ENV] = ARM_ALIAS[arm]
    return env


@contextlib.contextmanager
def pinned_arm(arm):
    """Pin the arm in THIS process, read back, and restore on exit.

    Prior art: R1 `pin_rule` (scripts/grm_r1_replay.py:242). The readback is
    the point: a pin that did not take must be RED, not silent.
    """
    from core.grm_admission import admission_rule
    from core.grm_alias_fold import alias_fold_enabled
    previous = {k: os.environ.get(k) for k in (RULE_ENV, ALIAS_ENV)}
    try:
        os.environ[RULE_ENV] = 'margin_first'
        os.environ.pop(ALIAS_ENV, None)
        if ARM_ALIAS[arm] is not None:
            os.environ[ALIAS_ENV] = ARM_ALIAS[arm]
        observed_rule = admission_rule()
        if observed_rule != 'margin_first':
            raise ValueError('LT11_RULE_PIN_FAILED: ' + observed_rule)
        observed_alias = alias_fold_enabled()
        if observed_alias is not (ARM_ALIAS[arm] is not None):
            raise ValueError('LT11_ALIAS_PIN_FAILED: arm=%s observed=%s'
                             % (arm, observed_alias))
        yield dict(arm=arm, admission_rule=observed_rule,
                   alias_fold_merge=observed_alias)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def lt1_1_seams(arm):
    """Redirect the three module seams `execute` reads from LT1 state.

    Everything else in `grm_lt1_worker` -- cells, leases, checkpoints,
    accounting, resume -- is used unchanged through these.
    """
    saved = (lt.FIX, lt.binding, worker.RUN)
    try:
        lt.FIX = FIXTURE
        lt.binding = binding
        worker.RUN = out_dir(arm)
        yield
    finally:
        lt.FIX, lt.binding, worker.RUN = saved


# ------------------------------------------------------------------- loaders

def gpu_loader(session, flags, state):
    """The registered loader: the real model, exactly as LT1's worker uses."""
    from scripts import grm_e2e_session as e2e
    from scripts.grm_c2_cells import args_for
    _, _, repo, info = e2e.load_model_and_repo(args_for(e2e, session, flags),
                                               session)
    return repo, info


def fake_loader(patch):
    """CPU double loader.

    Prior art: scripts/grm_lt1_worker_cpu.py (GRM contributors, 2026), taken
    verbatim, plus the alias-aware dual reader D1 established: the
    consolidation path must receive a faithful extractive digest, or the
    coverage gate in `ArenaCache.consolidate` aborts every alias fold and the
    A+ fake arm measures nothing. Receipt:
    artifacts/grm_d1/alias_cpu_aplus.json.
    """
    from scripts.grm_c7_diagnose import repository, Model
    from scripts.grm_lt1_cpu import visible_answer
    from core.grm_alias_fold import alias_scan_text

    def loader(session, flags, state):
        repo = repository(session / 'repository', patch)
        arena = repo.arena
        arena.width = flags['arena_width']
        arena.recency_mounts = 2
        if state.get('codec_words'):
            arena.m.codec.words = state['codec_words']
            arena.m.codec.ids = {w: i for i, w in enumerate(arena.m.codec.words)}

        consolidating = {'on': False, 'sources': []}

        class ProseModel(Model):
            def __call__(self, ids, kv_caches=None, position_offset=0, **kw):
                if kv_caches is None:
                    text = self.codec.decode(ids[0])
                    if consolidating['on']:
                        self.fold_output = (
                            ' '.join(consolidating['sources']) + '<|end|>')
                    else:
                        visible = ((self.injected if arena.cur_mounts else '')
                                   + '\n' + text)
                        self.fold_output = visible_answer(visible, text)
                return super().__call__(ids, kv_caches, position_offset, **kw)
        arena.m.__class__ = ProseModel

        original = arena.consolidate

        def consolidate(idxs, *args, **kwargs):
            consolidating['on'] = True
            consolidating['sources'] = [
                ' '.join(alias_scan_text(
                    arena.grafts[i].get('text', '')).split()) for i in idxs]
            try:
                return original(idxs, *args, **kwargs)
            finally:
                consolidating['on'] = False
        arena.consolidate = consolidate
        return repo, dict(model='CPU prose fake', gpu=False)
    return loader


# ------------------------------------------------------------------ actions

def preflight(arm):
    """Document-chain checks only. GPU readiness is LT1's own `preflight`."""
    reasons = []
    for path in (REGISTRATION, AMENDMENT1, FIXTURE):
        if not path.exists():
            reasons.append('MISSING_INPUT: ' + path.name)
    if not reasons:
        try:
            registration(arm)
        except ValueError as exc:
            reasons.append(str(exc))
    return dict(status='BLOCKED' if reasons else 'READY', reasons=reasons,
                arm=arm, gpu_executed=False,
                registration_sha256=sha(REGISTRATION) if REGISTRATION.exists() else None,
                amendment1_sha256=sha(AMENDMENT1) if AMENDMENT1.exists() else None,
                fixture_sha256=sha(FIXTURE) if FIXTURE.exists() else None)


def dry_run(arm, *, root=None):
    """Enumerate the arm's cells and estimates. No GPU, no lease, no writes."""
    reg = registration(arm)
    root = Path(root) if root else out_dir(arm)
    with pinned_arm(arm) as pin:
        cells = reg['cells']
        done = [c['id'] for c in cells
                if (root / 'cells' / c['id'] / 'controller.json').exists()]
        # The governing ceiling is the LATEST amendment's. Amendment 2
        # corrected amendment 1's 6120 s, which sat below the reservation sum
        # and would have tripped `run_cell`'s rail mid-campaign.
        amendment2 = OUT / 'amendment2.json'
        source = amendment2 if amendment2.exists() else AMENDMENT1
        budget = json.loads(source.read_text())['arms'][arm]
        lease = sum(c['lease_seconds'] for c in cells)
        return dict(status='PASS', mode='CPU_DRY_RUN', gpu_executed=False,
                    arm=arm, pinned=pin, cells=len(cells),
                    completed_cells=len(done),
                    estimate_seconds=round(
                        sum(c['estimate_seconds'] for c in cells), 3),
                    lease_seconds=lease,
                    budget_gpu_seconds=budget['budget_gpu_seconds'],
                    budget_source=str(source.relative_to(ROOT)),
                    within_budget=lease <= budget['budget_gpu_seconds'],
                    out_dir=str(root),
                    next_cell=next((c['id'] for c in cells
                                    if c['id'] not in done), None),
                    cell_ids=[c['id'] for c in cells],
                    binding=binding(arm))


def fake_cell(arm, cell_id, root):
    """Run ONE cell on the CPU double, in this process. Writes real receipts.

    `execute` enforces `RESTART_REQUIRES_NEW_PROCESS`: a resumed cell must
    observe a different pid/process_id than the checkpoint it loads. So one
    cell per process, exactly as the GPU controller spawns one child per cell.
    `run_fake` is the parent that spawns these.
    """
    import pytest
    reg = registration(arm)
    root = Path(root)
    cell = next(c for c in reg['cells'] if c['id'] == cell_id)
    directory = root / 'cells' / cell['id']
    with pytest.MonkeyPatch.context() as patch:
        loader = fake_loader(patch)
        saved = (lt.FIX, lt.binding, worker.RUN)
        try:
            lt.FIX, lt.binding, worker.RUN = FIXTURE, binding, root
            with pinned_arm(arm):
                directory.mkdir(parents=True)
                lt.create(directory / 'reservation.json',
                          dict(seconds=cell['lease_seconds'], cell=cell,
                               binding=binding(arm), fake=True,
                               admission_rule='margin_first'))
                value = worker.execute(cell, directory, reg, loader,
                                       run=root / 'cells', fake=True)
                lt.create(directory / 'worker.json', value)
                lt.create(directory / 'controller.json',
                          dict(status='COMPLETE', error=None,
                               charged_seconds=0.0, cell=cell,
                               binding=binding(arm), fake=True,
                               admission_rule='margin_first'))
        finally:
            lt.FIX, lt.binding, worker.RUN = saved
    return str(directory)


def run_fake(arm, *, root=None, limit=None):
    """Execute cells on the CPU double, one subprocess per cell.

    Mirrors the GPU controller's structure (one child process per cell, parent
    accumulates receipts) minus the lease, the flock and the GPU check, so the
    receipt plumbing `summary` reads is exercised end to end. Cells already
    COMPLETE are skipped, which is what makes this path resumable.

    Prior art: the one-subprocess-per-cell fake loop in
    tests/test_grm_lt1_fix6.py and tests/test_grm_lt1_amendment1.py
    (GRM contributors, 2026), reused.
    """
    import subprocess
    reg = registration(arm)
    root = Path(root) if root else out_dir(arm)
    written, skipped = [], []
    todo = reg['cells'][:limit] if limit else reg['cells']
    for cell in todo:
        directory = root / 'cells' / cell['id']
        if (directory / 'controller.json').exists():
            skipped.append(cell['id'])
            continue
        env = arm_environment(arm, reg['arms']['A']['flags'])
        env['PATH'] = os.environ.get('PATH', '')
        env['HOME'] = os.environ.get('HOME', '')
        result = subprocess.run(
            [sys.executable, '-m', 'scripts.grm_lt1_1', '--arm', arm,
             '--fake-cell', cell['id'], '--out', str(root)],
            cwd=ROOT, env=env, capture_output=True, text=True)
        if result.returncode:
            raise ValueError('LT11_FAKE_CELL_FAILED: %s\n%s'
                             % (cell['id'], result.stdout + result.stderr))
        written.append(str(directory))
    return dict(arm=arm, mode='CPU_FAKE', gpu_executed=False,
                cells_written=written, cells_skipped=skipped,
                out_dir=str(root),
                evidence_class='CPU fake worker (receipt plumbing; NOT a '
                               'language-model quality measurement)')


def resume(arm):
    """Registered GPU campaign for one arm. Reuses LT1's controller loop."""
    reg = registration(arm)
    root = out_dir(arm)
    with lt1_1_seams(arm), pinned_arm(arm) as pin:
        gate = lt.preflight()
        if gate['status'] != 'READY':
            raise ValueError(json.dumps(gate))
        root.mkdir(parents=True, exist_ok=True)
        owner = root / 'campaign.active'
        fd = os.open(owner, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        print('LT1.1 arm %s pinned %s' % (arm, json.dumps(pin)), flush=True)
        try:
            while True:
                cell = worker.pending(reg, run=root)
                if cell is None:
                    return 0
                print('LT1.1 %s cell %s' % (arm, cell['id']), flush=True)
                if not worker.run_cell(cell, reg):
                    return 2
        finally:
            owner.unlink()


def summary(arm, *, root=None):
    """Score the arm's receipts with LT1's own scorer, unchanged."""
    reg = registration(arm)
    root = Path(root) if root else out_dir(arm)
    fixture = json.loads(FIXTURE.read_text())
    probes = {p['id']: p for p in fixture['probes']}
    rows, complete = [], []
    for cell in reg['cells']:
        controller = root / 'cells' / cell['id'] / 'controller.json'
        if not controller.exists() or lt.read(controller)['status'] != 'COMPLETE':
            continue
        complete.append(cell['id'])
        path = root / 'cells' / cell['id'] / 'probes.jsonl'
        if path.exists():
            rows.extend(json.loads(line) for line in
                        path.read_text().splitlines() if line.strip())
    table = {}
    for cls in ('fresh', 'correction', 'alias'):
        want = [p for p in fixture['probes'] if p['class'] == cls]
        got = [r for r in rows if probes[r['probe_id']]['class'] == cls]
        correct = sum(1 for r in got
                      if lt.score(r['memory']['answer'],
                                  probes[r['probe_id']]['expected'])['exact_correct'])
        table[cls] = dict(expected_n=len(want), n=len(got), correct=correct,
                          exact_rate=(correct / len(got)) if got else None)
    return dict(arm=arm, out_dir=str(root), complete_cells=len(complete),
                total_cells=len(reg['cells']),
                complete=len(complete) == len(reg['cells']),
                completed_cell_ids=complete,
                measured_recalls=len(rows), by_class=table,
                binding=binding(arm),
                evidence_class=('partial raw rows' if rows else 'no rows'))


# ---------------------------------------------------------------------- CLI

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog='scripts/grm_lt1_1.py',
        description='LT1.1 runner: one arm per invocation, over the LT1 '
                    'worker machinery.')
    p.add_argument('--arm', choices=ARMS, required=True,
                   help='A (alias fold OFF) or A+ (GRM_ALIAS_FOLD_MERGE=1)')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--resume', action='store_true',
                      help='registered GPU campaign (takes the shared lease)')
    mode.add_argument('--summary', action='store_true')
    mode.add_argument('--fake', action='store_true',
                      help='CPU double execution; writes real receipts')
    mode.add_argument('--worker', help='leased child entry point (internal)')
    mode.add_argument('--fake-cell', help='one-cell CPU child (internal)')
    p.add_argument('--out', help='override the campaign root (dry-run/fake/summary)')
    p.add_argument('--limit', type=int, help='fake mode: run only the first N cells')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        # Leased child; the parent pinned the arm into our environment.
        with lt1_1_seams(args.arm):
            cell = next(c for c in registration(args.arm)['cells']
                        if c['id'] == args.worker)
            worker.worker(cell)
        return 0
    if args.fake_cell:
        print(fake_cell(args.arm, args.fake_cell,
                        args.out or out_dir(args.arm)))
        return 0
    if args.preflight:
        value = preflight(args.arm)
        print(json.dumps(value, indent=2))
        return 0 if value['status'] == 'READY' else 2
    if args.dry_run:
        gate = preflight(args.arm)
        if gate['status'] != 'READY':
            print(json.dumps(gate, indent=2))
            return 2
        print(json.dumps(dry_run(args.arm, root=args.out), indent=2))
        return 0
    if args.summary:
        print(json.dumps(summary(args.arm, root=args.out), indent=2))
        return 0
    if args.fake:
        print(json.dumps(run_fake(args.arm, root=args.out, limit=args.limit),
                         indent=2))
        return 0
    return resume(args.arm)


if __name__ == '__main__':
    raise SystemExit(main())
