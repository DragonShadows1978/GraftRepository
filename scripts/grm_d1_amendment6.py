#!/usr/bin/env python3
"""GRM-D1 follow-up 6: LT1.1 amendment 6 — a busy card is not a cell failure.

Two things happened on the card, both lead-caused, and only one of them is a
defect in this runner.

  1. The lead's queue launched `--arm A --resume` while the A1 contrast was
     still leaving the card. The single-probe idle check refused and wrote
     `run_A/cells/A-001-008` as RED `ValueError: GPU_NOT_IDLE: 3336818` --
     controller.json and reservation.json, no worker.log, because no child was
     ever spawned. That leftover then made `--arm A --resume --dry-lease` fail
     with `PRIOR_CELL_RED`. The cell is archived here, create-only, and arm A
     re-arms from cell 1.

  2. `GPU_NOT_IDLE` should never have been a cell failure at all. A busy card
     is a resource somebody else holds, not a fault in our work. This
     amendment makes the check a BOUNDED wait-then-proceed, in the R1
     `await_idle` shape: framebuffer used (never the compute-process list), a
     counted loop with the bound fixed up front, declines only, never signals
     anything. Only after the bound expires is it a RED, and then with the
     memory snapshot attached.

THE SEAM-RESTORE QUESTION the lead asked: `worker.RUN: False` in the audit was
STATE-CAUSED, not a restore bug. That assertion tested that the seam CHANGED
the module global; when a previous test had already left `worker.RUN` equal to
the target, nothing changed and a correct seam read as broken. Restoration was
verified separately and was always correct. The assertion now tests the
invariant that matters -- the seam HOLDS the LT1.1 value inside the window --
plus an identity check on restore. Receipt:
`tests/test_grm_d1_amendment3.py::test_the_seam_audit_matches_what_the_runner_actually_moves`.

Prior art:
  * R1 `await_idle` / `idle_gate` / `device_snapshot`
    (`scripts/grm_r1_replay.py:352-470`, GRM contributors 2026) and, through
    them, FIX-8 `grm_scout_fix8_resume.parse_memory` and the NVIDIA
    nvidia-smi XML framebuffer report. TAKEN: the counted-loop bound, the
    decline-only contract, and the rule that an idle card is never inferred
    from an empty compute list. OURS: the LT1.1 defaults and the receipt
    shape returned to `run_cell`.
  * The seam pattern with a byte-identical default is this campaign's own,
    established for `spawn_argv` in amendment 5.
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
AMENDMENT5 = OUT / 'amendment5.json'
ORDER = ROOT / 'orders/GRM_D1_LT1_RESIDUAL_DIAGNOSIS.md'

RUNNER = 'scripts/grm_lt1_1.py'
WORKER = 'scripts/grm_lt1_worker.py'
RUNNER_TESTS = ('tests/test_grm_lt1_1_runner.py',
                'tests/test_grm_lt1_1_resume_route.py',
                'tests/test_grm_lt1_1_preflight.py',
                'tests/test_grm_lt1_1_child_spawn.py',
                'tests/test_grm_lt1_1_idle_wait.py')

ALIAS_FLAG = 'GRM_ALIAS_FOLD_MERGE'

#: The cell the busy card cost us, archived rather than deleted.
RED_CELL = 'A-001-008'
RED_ARM = 'A'
RED_REASON = 'GPU_NOT_IDLE'

FOLLOW_UP = (
    'Lead follow-up 6 to GRM-D1 (2026-09-11): the queue launched --arm A '
    '--resume while the A1 contrast was leaving the card; the idle check '
    'refused and wrote run_A/cells/A-001-008 RED with GPU_NOT_IDLE, which '
    'then made --dry-lease fail with PRIOR_CELL_RED. Archive that cell '
    'create-only and re-arm arm A from cell 1; make GPU_NOT_IDLE a bounded '
    'wait-then-retry in the R1 await_idle shape (framebuffer used, counted '
    'loop, declines only, never signals), RED with a snapshot only after the '
    'bound; confirm the seam-restore audit failure was state-caused.')


def sha_path(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, indent=2) + '\n').encode()


def amendment():
    if not AMENDMENT5.exists():
        raise ValueError('LT11_AMENDMENT5_MISSING')
    parent_sha = sha_path(AMENDMENT5)
    recorded = (OUT / 'amendment5.sha256').read_text().split()[0]
    if parent_sha != recorded:
        raise ValueError('LT11_AMENDMENT5_SHA_MISMATCH')
    parent = json.loads(AMENDMENT5.read_text())

    from scripts import grm_lt1_1 as runner_mod
    archive = OUT / 'archive' / ('%s_%s_%s' % (RED_ARM, RED_CELL, RED_REASON))

    return dict(
        schema='grm.lt1_1.amendment6.v1',
        amendment=6,
        previous_amendment='artifacts/grm_d1/lt1_1/amendment5.json',
        previous_amendment_sha256=parent_sha,
        registration='artifacts/grm_d1/lt1_1/registration.json',
        registration_sha256=parent['registration_sha256'],
        order='orders/GRM_D1_LT1_RESIDUAL_DIAGNOSIS.md',
        order_sha256=sha_path(ORDER),
        follow_up_instruction=FOLLOW_UP,
        purpose=('Make a busy card a bounded wait instead of a cell failure, '
                 'and clear the spurious RED it already caused.'),
        idle_policy=dict(
            before='one nvidia-smi compute-app probe; any output -> '
                   'ValueError GPU_NOT_IDLE -> the cell is RED and its '
                   'reservation is charged',
            after='bounded wait for the card to fall below a framebuffer '
                  'limit, then proceed; RED with the memory snapshot ONLY '
                  'after the bound expires',
            rationale='a busy card is a resource another process holds, not a '
                      'fault in our cell',
            limit_mib=runner_mod.IDLE_LIMIT_MIB,
            wait_seconds=runner_mod.IDLE_WAIT_SECONDS,
            poll_seconds=runner_mod.IDLE_POLL_SECONDS,
            attempts_allowed=1 + (runner_mod.IDLE_WAIT_SECONDS
                                  // runner_mod.IDLE_POLL_SECONDS),
            keys_on='TOTAL framebuffer used, never the compute-process list '
                    '(FIX-8: never infer an idle card from an empty compute '
                    'list)',
            never='signals, kills or waits on another process; it declines '
                  'and it waits, nothing else',
            bound='structural: attempts_allowed caps the probe count up front, '
                  'so there is no unbounded loop and nothing re-runs a cell',
            seam=dict(
                file=WORKER,
                sha256=sha_path(ROOT / WORKER),
                name='await_idle',
                what='run_cell consults an optional module global; the '
                     'default branch is byte-identical to the two lines it '
                     'replaced, so LT1 own campaign is unaffected',
                lt1_behaviour_unchanged=True)),
        seams=dict(
            count=6,
            added=['worker.await_idle -> await_idle (bounded idle wait)'],
            existing=['lt.FIX', 'worker.bind', 'worker.RUN', 'worker.worker',
                      'worker.spawn_argv', 'worker.spawn_env']),
        seam_restore_finding=dict(
            question='the lead asked whether the audit `worker.RUN: False` '
                     'was a real restore bug',
            answer='NO — state-caused',
            why='the assertion tested that the seam CHANGED the module '
                'global. When an earlier test had already left `worker.RUN` '
                'equal to the target, nothing changed and a correct seam read '
                'as broken. Restoration was verified separately by an '
                'identity check and was always correct.',
            fix='the assertion now tests the invariant that matters -- the '
                'seam HOLDS the LT1.1 value inside the window -- plus an '
                'identity check on restore',
            receipt='tests/test_grm_d1_amendment3.py::'
                    'test_the_seam_audit_matches_what_the_runner_actually_moves'),
        red_cell_archive=dict(
            arm=RED_ARM, cell=RED_CELL, reason=RED_REASON,
            symptom='ValueError: GPU_NOT_IDLE: 3336818; controller.json and '
                    'reservation.json only, no worker.log, because no child '
                    'was ever spawned',
            consequence='`--arm A --resume --dry-lease` failed with '
                        'PRIOR_CELL_RED until the cell was cleared',
            archived_to=str(archive.relative_to(ROOT)),
            policy='create-only: the attempt is copied, never deleted or '
                   'overwritten, and the live cell directory is removed so '
                   'arm %s re-arms from cell 1' % RED_ARM,
            lead_caused=True,
            not_a_cell_failure=True),
        gate=dict(
            tests='tests/test_grm_lt1_1_idle_wait.py',
            fixtures=['busy -> wait -> idle: the cell runs',
                      'busy past the bound: RED with the memory snapshot',
                      'probe failure: declines, with the reason',
                      'the wait is bounded and counted, never unbounded',
                      'nothing is ever signalled'],
            evidence_class='CPU fixture over the idle policy (control flow '
                           'and receipts; no GPU is probed)'),
        runner=dict(
            path=RUNNER,
            sha256=sha_path(ROOT / RUNNER),
            superseded_sha256=parent['runner']['sha256'],
            tests=list(RUNNER_TESTS),
            tests_sha256={name: sha_path(ROOT / name) for name in RUNNER_TESTS
                          if (ROOT / name).exists()}),
        ruling=parent['ruling'],
        arms=parent['arms'],
        budget_gpu_seconds_per_arm=parent['budget_gpu_seconds_per_arm'],
        budget_gpu_seconds_total=parent['budget_gpu_seconds_total'],
        budget_gpu_hours_total=parent['budget_gpu_hours_total'],
        process_safety=dict(
            gpu='the idle wait READS the card and declines; it never signals, '
                'kills or waits on another process. Only `--resume` without '
                '`--dry-lease` takes the shared lease, through LT1 own '
                'run_cell.',
            operator='the operator has absolute right of way: a busy card '
                     'makes this campaign wait, then stand down',
            arms='A and A+ never run concurrently on the same GPU',
            git='lead commits; the seat never runs git'),
        prior_art=[
            'R1 await_idle / idle_gate / device_snapshot, '
            'scripts/grm_r1_replay.py:352-470 (GRM contributors, 2026), and '
            'through them FIX-8 grm_scout_fix8_resume.parse_memory and the '
            'NVIDIA nvidia-smi XML framebuffer report: the counted-loop '
            'bound, the decline-only contract, and the rule that an idle card '
            'is never inferred from an empty compute list',
            'Seam with a byte-identical default: this campaign own pattern, '
            'established for spawn_argv in amendment 5',
            'LT1.1 defaults and the receipt shape returned to run_cell: no '
            'prior art known to me',
        ],
    )


def archive_red_cell():
    """Copy the RED cell aside (create-only) and clear it so arm A re-arms."""
    import shutil
    live = OUT / ('run_%s' % RED_ARM) / 'cells' / RED_CELL
    archive = OUT / 'archive' / ('%s_%s_%s' % (RED_ARM, RED_CELL, RED_REASON))
    if not live.exists():
        return dict(status='NOTHING_TO_ARCHIVE', live=str(live))
    if archive.exists():
        return dict(status='ALREADY_ARCHIVED', archive=str(archive))
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(live, archive)
    shutil.rmtree(live)
    return dict(status='ARCHIVED', archive=str(archive), cleared=str(live),
                note='create-only: the attempt is preserved; the live cell is '
                     'cleared so the arm re-arms from cell 1')


def lead_commands(amend_sha):
    doc = amendment()
    idle = doc['idle_policy']
    lines = [
        '# LT1.1 — arm A and arm A+ (amendment 6)',
        '# registration sha256: %s' % doc['registration_sha256'],
        '# amendment 5  sha256: %s' % doc['previous_amendment_sha256'],
        '# amendment 6  sha256: %s' % amend_sha,
        '# runner            : %s (sha256 %s)'
        % (RUNNER, doc['runner']['sha256']),
        '#',
        '# 26 cells per arm. Reservation ceiling %d s (%.2f GPU-h) PER ARM;'
        % (doc['budget_gpu_seconds_per_arm'],
           doc['budget_gpu_seconds_per_arm'] / 3600.0),
        '# both arms: %.2f GPU-h ceiling. Separately resumable.'
        % doc['budget_gpu_hours_total'],
        '#',
        '# A BUSY CARD IS NOT A CELL FAILURE (amendment 6). The runner now',
        '# waits up to %d s (polling every %d s, %d probes) for framebuffer'
        % (idle['wait_seconds'], idle['poll_seconds'],
           idle['attempts_allowed']),
        '# use to fall to <= %d MiB, then proceeds. Only after the bound is'
        % idle['limit_mib'],
        '# it a RED, and then with the memory snapshot attached. It reads the',
        '# card and declines; it never signals anything. The operator has',
        '# absolute right of way.',
        '# The earlier spurious RED (%s on arm %s) is archived at'
        % (RED_REASON, RED_ARM),
        '#   %s' % doc['red_cell_archive']['archived_to'],
        '# and arm %s re-arms from cell 1.' % RED_ARM,
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
            'python3 %s --arm %s --preflight' % (RUNNER, arm),
            'python3 %s --arm %s --dry-run' % (RUNNER, arm),
            '# Route check on CPU: walks the REAL resume path and stops at',
            '# the lease boundary.',
            'python3 %s --arm %s --resume --dry-lease' % (RUNNER, arm),
            '# GPU campaign (takes the shared lease; resumable -- run again',
            '# to continue). A busy card makes it WAIT, not fail:',
            'python3 %s --arm %s --resume' % (RUNNER, arm),
            'python3 %s --arm %s --summary' % (RUNNER, arm),
            '',
        ]
    lines += [
        '# The runner pins the arm itself, AFTER environment(flags) strips',
        '# every ambient GRM_*, and carries it into the leased child in',
        '# GRM_LT1_1_ARM. The child verifies the LT1.1 chain before any work:',
        '#   arm A   -> admission_rule=margin_first, alias_fold_merge=False',
        '#   arm A+  -> admission_rule=margin_first, alias_fold_merge=True',
        '# A pin that does not take is LT11_ALIAS_PIN_FAILED, never silent.',
        '# A drifted LT1.1 document stops the CHILD too:',
        '#   LT11_CHILD_PREFLIGHT_BLOCKED.',
        '',
        '# Registered predictions (before any run):',
    ]
    for arm in ('A', 'A+'):
        lines.append('#   %-3s %s' % (arm, json.dumps(
            doc['arms'][arm]['predictions'], sort_keys=True)))
    return '\n'.join(lines) + '\n'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archived = archive_red_cell()
    payload = canonical(amendment())
    (OUT / 'amendment6.json').write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (OUT / 'amendment6.sha256').write_text('%s  amendment6.json\n' % digest)
    (OUT / 'lead_commands.txt').write_text(lead_commands(digest))

    doc = amendment()
    idle = doc['idle_policy']
    print('amendment6 sha256 %s' % digest)
    print('parent amendment5 %s' % doc['previous_amendment_sha256'])
    print('runner %s' % doc['runner']['sha256'])
    print('  supersedes %s' % doc['runner']['superseded_sha256'])
    print('idle policy: <= %d MiB, wait <= %d s, %d probes at %d s'
          % (idle['limit_mib'], idle['wait_seconds'],
             idle['attempts_allowed'], idle['poll_seconds']))
    print('seam-restore question: %s' % doc['seam_restore_finding']['answer'])
    print('RED cell archive: %s' % archived['status'])


if __name__ == '__main__':
    main()
