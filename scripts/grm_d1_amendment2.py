#!/usr/bin/env python3
"""GRM-D1 follow-up: LT1.1 amendment 2 — bind the runner, fix the budget.

Amendment 1 registered two arms and rebound the core inputs, but the campaign
it described could not be executed: `scripts/grm_lt1.py` has no
`--run/--arm/--registration/--amendment/--fixture/--out`, and the emitted
`lead_commands.txt` named that non-existent interface. A registration is not
runnable until a worker executes it.

This amendment, sha-chained to amendment 1, does two things:

  1. BIND THE RUNNER as a registered input. `scripts/grm_lt1_1.py` is the
     entry point that drives the existing LT1 worker machinery against these
     documents, one arm per invocation. Its sha is pinned here, so the
     campaign's executable surface is part of the registered chain rather
     than a description of one.

  2. CORRECT THE PER-ARM BUDGET. Amendment 1 registered 6120 s (1.70 GPU-h)
     per arm. Running `--dry-run` against the real cell list showed that is
     BELOW the worst-case reservation: the 26 cells reserve
     `sum(lease_seconds) = 7410 s` (2.06 GPU-h), and `run_cell` charges the
     lease, not the estimate. A ceiling under the reservation sum trips
     `COMBINED_GPU_BUDGET_RAIL` partway through a campaign that was going to
     finish. The projected actual is `sum(estimate_seconds) = 4508 s`
     (1.25 GPU-h); the corrected ceiling is the reservation sum.

     This is a BUDGET INCREASE and the lead should see it as one: 2.06 GPU-h
     per arm, 4.12 for both, against the 1.70 / 3.40 amendment 1 claimed. The
     expected spend is unchanged (1.25 GPU-h per arm); only the ceiling moves,
     to a number the machinery can actually honour. Registered before any run.

Nothing in `core/` is touched. Neither the base registration nor amendment 1
is rewritten.

Prior art:
  * The SHA-chained amendment contract (parent sha, pinned inputs) is LT1's
    own (`scripts/grm_lt1.py:verify`) and C7 r3's, taken unchanged.
  * The reservation-vs-estimate distinction is LT1's `run_cell` accounting
    (`scripts/grm_lt1_worker.py:227`, GRM contributors 2026): `accounting() +
    cell['lease_seconds'] > 10800` is the rail, so leases are what a budget
    must cover. Read, not modified.
  * Mine: binding an executable entry point as a registered input, and
    deriving the ceiling from the cell list rather than asserting a round
    number.
  * No prior art known to me for this exact composition.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / 'artifacts/grm_d1/lt1_1'
AMENDMENT1 = OUT / 'amendment1.json'
AMENDMENT1_SHA = '0a5754cd3e5bc91c4c5428228ed4164534c5d281d2aa0c7410e8c7f8b145ce0e'
ORDER = ROOT / 'orders/GRM_D1_LT1_RESIDUAL_DIAGNOSIS.md'

#: The executable surface this amendment binds.
RUNNER = 'scripts/grm_lt1_1.py'
RUNNER_TESTS = 'tests/test_grm_lt1_1_runner.py'

ALIAS_FLAG = 'GRM_ALIAS_FOLD_MERGE'
RULE_ENV = 'GRM_ADMISSION_RULE'

FOLLOW_UP = (
    'Lead follow-up to GRM-D1 (2026-09-11): the emitted lead_commands did not '
    'run -- they described `grm_lt1.py --run --arm ...`, an interface that '
    'does not exist. Build scripts/grm_lt1_1.py driving the EXISTING LT1 '
    'worker machinery (leased cells, lease-parent contract, create-only '
    'receipts, preflight/dry-run/resume/summary), one arm per invocation, '
    'distinct out dirs, the flag pinned after environment(flags) and read '
    'back; prove it with a dry-run, a fake 2-cell execution per arm and a '
    'resume; bind the runner in amendment 2; regenerate lead_commands from '
    'the ACTUAL argparse with a test that runs each emitted command.')


def sha_path(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, indent=2) + '\n').encode()


def budget_from_cells(cells):
    """The ceiling the machinery can honour, derived from the cell list."""
    lease = sum(int(c['lease_seconds']) for c in cells)
    estimate = sum(float(c['estimate_seconds']) for c in cells)
    return dict(
        reservation_seconds=lease,
        reservation_gpu_hours=round(lease / 3600.0, 2),
        projected_seconds=round(estimate, 3),
        projected_gpu_hours=round(estimate / 3600.0, 2),
        basis='run_cell charges cell["lease_seconds"], not the estimate, and '
              'rails on accounting()+lease. The ceiling is therefore the '
              'reservation sum; the estimate is what the run is expected to '
              'actually spend.')


def amendment():
    if sha_path(AMENDMENT1) != AMENDMENT1_SHA:
        raise ValueError('LT11_AMENDMENT1_SHA_MISMATCH')
    parent = json.loads(AMENDMENT1.read_text())
    cells = parent['arms']['A']['cells']
    budget = budget_from_cells(cells)
    per_arm = budget['reservation_seconds']

    arms = {}
    for name, spec in parent['arms'].items():
        arms[name] = dict(
            arm=name,
            out_dir=spec['out_dir'],
            alias_fold_merge=spec['alias_fold_merge'],
            env_after_environment_flags=spec['env_after_environment_flags'],
            budget_gpu_seconds=per_arm,
            budget_gpu_hours=round(per_arm / 3600.0, 2),
            superseded_budget_gpu_seconds=spec['budget_gpu_seconds'],
            predictions=spec['predictions'],
            invocation=dict(
                preflight='python3 %s --arm %s --preflight' % (RUNNER, name),
                dry_run='python3 %s --arm %s --dry-run' % (RUNNER, name),
                resume='python3 %s --arm %s --resume' % (RUNNER, name),
                summary='python3 %s --arm %s --summary' % (RUNNER, name)))

    correction = dict(budget)
    correction.update(
        superseded_gpu_seconds_per_arm=parent['budget_gpu_seconds_per_arm'],
        superseded_gpu_hours_per_arm=parent['arms']['A']['budget_gpu_hours'],
        corrected_gpu_seconds_per_arm=per_arm,
        corrected_gpu_hours_per_arm=round(per_arm / 3600.0, 2),
        why=('Amendment 1 registered 1.70 GPU-h per arm. `--dry-run` against '
             'the real cell list showed the 26 cells reserve %d s (%.2f '
             'GPU-h) and `run_cell` charges the lease, so the old ceiling '
             'would have tripped COMBINED_GPU_BUDGET_RAIL mid-campaign. '
             'Expected spend is unchanged at %.2f GPU-h per arm; only the '
             'ceiling moves. LEAD DECISION: this is a budget increase to '
             '%.2f GPU-h per arm, %.2f for both.'
             % (per_arm, per_arm / 3600.0, budget['projected_gpu_hours'],
                per_arm / 3600.0, 2 * per_arm / 3600.0)),
        discovered_by='python3 scripts/grm_lt1_1.py --arm A --dry-run '
                      '(within_budget: false)')

    return dict(
        schema='grm.lt1_1.amendment2.v1',
        amendment=2,
        previous_amendment='artifacts/grm_d1/lt1_1/amendment1.json',
        previous_amendment_sha256=AMENDMENT1_SHA,
        registration='artifacts/grm_d1/lt1_1/registration.json',
        registration_sha256=parent['registration_sha256'],
        order='orders/GRM_D1_LT1_RESIDUAL_DIAGNOSIS.md',
        order_sha256=sha_path(ORDER),
        follow_up_instruction=FOLLOW_UP,
        purpose=('Bind the LT1.1 runner as a registered input and correct the '
                 'per-arm budget to the reservation sum the machinery '
                 'actually charges.'),
        runner=dict(
            path=RUNNER,
            sha256=sha_path(ROOT / RUNNER),
            tests=RUNNER_TESTS,
            tests_sha256=sha_path(ROOT / RUNNER_TESTS),
            entry_points=['--preflight', '--dry-run', '--resume', '--summary',
                          '--fake', '--worker', '--fake-cell'],
            reuse=('Parameterized, NOT forked. `grm_lt1_worker.execute` is '
                   'already argument-driven (cell, directory, registration, '
                   'loader, run, fake); the runner redirects the only three '
                   'module-level seams it reads -- `lt.FIX`, `lt.binding`, '
                   '`worker.RUN` -- and reuses execute / run_cell / pending '
                   'unchanged. No cell, lease, checkpoint, accounting or '
                   'scoring logic is reimplemented.'),
            arm_pin=('R1 pin_rule idiom: the alias flag is pinned AFTER '
                     '`environment(flags)` strips every ambient GRM_*, read '
                     'back through `alias_fold_enabled()`, asserted, and '
                     'restored on context exit.')),
        budget_correction=correction,
        arms=arms,
        budget_gpu_seconds_per_arm=per_arm,
        budget_gpu_seconds_total=2 * per_arm,
        budget_gpu_hours_total=round(2 * per_arm / 3600.0, 2),
        proof=dict(
            dry_run='26 cells enumerated per arm with estimates and the '
                    'reservation sum; both arms exit 0',
            fake_execution='2 cells per arm executed on the CPU double, one '
                           'subprocess per cell, writing controller.json / '
                           'worker.json / reservation.json / checkpoint',
            summary='--summary reads those receipts and scores them with '
                    "LT1's own value-span scorer",
            resume='a third invocation skips the COMPLETE cells and runs only '
                   'the next one',
            command_gate='tests/test_grm_lt1_1_runner.py::'
                         'test_every_emitted_command_actually_runs executes '
                         'every emitted command with --dry-run and requires '
                         'exit 0',
            evidence_class='CPU fake worker (receipt plumbing; NOT a '
                           'language-model quality measurement)'),
        process_safety=dict(
            gpu="only --resume takes the shared lease, through LT1's own "
                'run_cell (flock + nvidia-smi idle check + no-kill child '
                'wait); --preflight/--dry-run/--summary/--fake never do',
            arms='A and A+ never run concurrently on the same GPU',
            git='lead commits; the seat never runs git'),
        prior_art=[
            'LT1 worker machinery reused by redirection: '
            'scripts/grm_lt1_worker.py execute/run_cell/pending and '
            'scripts/grm_lt1.py (GRM contributors, 2026)',
            'One-subprocess-per-cell fake loop: tests/test_grm_lt1_fix6.py '
            'and scripts/grm_lt1_worker_cpu.py (GRM contributors, 2026)',
            'Pin-after-strip-then-read-back: R1 pin_rule, '
            'scripts/grm_r1_replay.py:242 (GRM contributors, 2026)',
            'Reservation-vs-estimate accounting: scripts/grm_lt1_worker.py '
            'run_cell budget rail (GRM contributors, 2026)',
            'Binding an executable entry point as a registered input, and '
            'deriving the ceiling from the cell list: no prior art known to me',
        ],
    )


def lead_commands(amend_sha):
    """Emitted from the ACTUAL argparse surface, not from prose."""
    doc = amendment()
    correction = doc['budget_correction']
    lines = [
        '# LT1.1 — arm A and arm A+ (amendment 2)',
        '# registration sha256: %s' % doc['registration_sha256'],
        '# amendment 1  sha256: %s' % doc['previous_amendment_sha256'],
        '# amendment 2  sha256: %s' % amend_sha,
        '# runner            : %s (sha256 %s)'
        % (RUNNER, doc['runner']['sha256']),
        '#',
        '# 26 cells per arm. Reservation ceiling %d s (%.2f GPU-h) PER ARM;'
        % (doc['budget_gpu_seconds_per_arm'],
           doc['budget_gpu_seconds_per_arm'] / 3600.0),
        '# projected spend %.2f GPU-h per arm. Both arms: %.2f GPU-h ceiling.'
        % (correction['projected_gpu_hours'], doc['budget_gpu_hours_total']),
        '# NOTE amendment 2 RAISED the per-arm ceiling from %.2f to %.2f GPU-h;'
        % (correction['superseded_gpu_hours_per_arm'],
           correction['corrected_gpu_hours_per_arm']),
        '# the old figure was below the reservation sum and would have railed.',
        '# The arms are separately resumable: run either alone.',
        '#',
        '# ---- CPU gates (no GPU, no lease) ----',
        'cd %s' % ROOT,
        'python3 -m pytest -q tests/test_grm_d1*.py tests/test_grm_lt1_1*.py',
        'python3 scripts/grm_d1_recap.py          # recap oracle 5/5 PASS',
        'python3 scripts/grm_d1_alias_cpu.py      # A+ alias fold/admit/mount',
        '',
    ]
    for arm in ('A', 'A+'):
        spec = doc['arms'][arm]
        label = ('alias fold OFF' if not spec['alias_fold_merge']
                 else '%s=1 pinned' % ALIAS_FLAG)
        lines += [
            '# ---- arm %s (%s) ----' % (arm, label),
            spec['invocation']['preflight'],
            spec['invocation']['dry_run'],
            '# GPU campaign (takes the shared lease; resumable -- run again '
            'to continue):',
            spec['invocation']['resume'],
            spec['invocation']['summary'],
            '',
        ]
    lines += [
        '# The runner pins the arm itself, AFTER environment(flags) strips',
        '# every ambient GRM_*, and reads it back before any cell runs:',
        '#   arm A   -> admission_rule=margin_first, alias_fold_merge=False',
        '#   arm A+  -> admission_rule=margin_first, alias_fold_merge=True',
        '# A pin that does not take is LT11_ALIAS_PIN_FAILED, never silent.',
        '',
        '# Registered predictions (before any run):',
    ]
    for arm in ('A', 'A+'):
        lines.append('#   %-3s %s' % (arm, json.dumps(
            doc['arms'][arm]['predictions'], sort_keys=True)))
    return '\n'.join(lines) + '\n'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    payload = canonical(amendment())
    (OUT / 'amendment2.json').write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (OUT / 'amendment2.sha256').write_text('%s  amendment2.json\n' % digest)
    (OUT / 'lead_commands.txt').write_text(lead_commands(digest))

    doc = amendment()
    correction = doc['budget_correction']
    print('amendment2 sha256 %s' % digest)
    print('parent amendment1 %s' % AMENDMENT1_SHA)
    print('runner %s sha256 %s' % (RUNNER, doc['runner']['sha256']))
    print('budget per arm: %d s (%.2f GPU-h) reservation; %.2f GPU-h projected'
          % (correction['corrected_gpu_seconds_per_arm'],
             correction['corrected_gpu_hours_per_arm'],
             correction['projected_gpu_hours']))
    print('  SUPERSEDES amendment 1: %.2f -> %.2f GPU-h per arm (INCREASE)'
          % (correction['superseded_gpu_hours_per_arm'],
             correction['corrected_gpu_hours_per_arm']))


if __name__ == '__main__':
    main()
