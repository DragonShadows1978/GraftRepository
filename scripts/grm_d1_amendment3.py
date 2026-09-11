#!/usr/bin/env python3
"""GRM-D1 follow-up 3: LT1.1 amendment 3 — rebind the runner after the seam fix.

Amendment 2 bound `scripts/grm_lt1_1.py`. On the card, `--arm A+ --resume`
died before taking any lease:

    grm_lt1_amendment4.apply -> lt.binding('CPU') -> ARM_ALIAS['CPU']
    KeyError: 'CPU'

Two stacked defects, both mine, neither reachable by the gates that existed:

  1. SIGNATURE DRIFT. The redirected `binding` took an ARM. LT1's callers pass
     two different token kinds -- `worker.bind(cell['arm'])` passes an arm,
     while `apply4` / `preflight` / `summary` pass a BACKEND LABEL ('CPU').
     LT1's own `binding` is label-agnostic (it echoes its argument); mine
     indexed `ARM_ALIAS`, so a label was a KeyError.

  2. THE WRONG SEAM. Even with the signature fixed, redirecting `lt.binding`
     is wrong: LT1 uses it to validate ITS OWN chain (`apply4` compares a
     recorded `protocol_binding` against it, and it hashes `lt.FIX`), while
     receipts are stamped through `worker.bind`. Redirecting the former turned
     the KeyError into `AMENDMENT4_PROTOCOL_MISMATCH`.

  Neither `--dry-run` nor the fake path traverses `apply4`, which is why 115
  passing tests said nothing. The fix is three parts: `binding` keeps LT1's
  signature and reads the arm from pinned runner state; the seam narrows to
  `worker.bind`; and LT1's host preflight runs OUTSIDE the redirected window,
  because it asks a question about the host tree, not about LT1.1.

This amendment rebinds the corrected runner and records the gate that now
covers the route.

Prior art:
  * SHA-chained amendment contract: LT1's `scripts/grm_lt1.py:verify` and C7
    r3, taken unchanged.
  * The seam distinction it encodes is a reading of LT1's own code
    (`grm_lt1_worker.bind` vs `grm_lt1.binding`, `grm_lt1_amendment4.apply`),
    GRM contributors 2026.
  * Mine: the lease-boundary gate (`--resume --dry-lease`) that traverses the
    production route on CPU, and the pinned-arm state that lets a
    label-signature function still select an arm.
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
AMENDMENT2 = OUT / 'amendment2.json'
ORDER = ROOT / 'orders/GRM_D1_LT1_RESIDUAL_DIAGNOSIS.md'

RUNNER = 'scripts/grm_lt1_1.py'
RUNNER_TESTS = ('tests/test_grm_lt1_1_runner.py',
                'tests/test_grm_lt1_1_resume_route.py')

ALIAS_FLAG = 'GRM_ALIAS_FOLD_MERGE'

FOLLOW_UP = (
    'Lead follow-up 3 to GRM-D1 (2026-09-11): `--arm A+ --resume` died at '
    "grm_lt1_amendment4.apply -> lt.binding('CPU') -> KeyError: 'CPU'. Fix "
    'the seam so `binding` keeps the LT1 signature (backend label) and takes '
    'the arm from runner state; audit every other redirected seam for the '
    'same drift; add a gate that traverses the REAL --resume route on CPU up '
    'to the lease boundary for both arms, RED-before/GREEN-after; amendment 3 '
    'chained to amendment 2 with the runner rebound.')


def sha_path(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, indent=2) + '\n').encode()


def amendment():
    if not AMENDMENT2.exists():
        raise ValueError('LT11_AMENDMENT2_MISSING')
    parent_sha = sha_path(AMENDMENT2)
    recorded = (OUT / 'amendment2.sha256').read_text().split()[0]
    if parent_sha != recorded:
        raise ValueError('LT11_AMENDMENT2_SHA_MISMATCH')
    parent = json.loads(AMENDMENT2.read_text())

    return dict(
        schema='grm.lt1_1.amendment3.v1',
        amendment=3,
        previous_amendment='artifacts/grm_d1/lt1_1/amendment2.json',
        previous_amendment_sha256=parent_sha,
        registration='artifacts/grm_d1/lt1_1/registration.json',
        registration_sha256=parent['registration_sha256'],
        order='orders/GRM_D1_LT1_RESIDUAL_DIAGNOSIS.md',
        order_sha256=sha_path(ORDER),
        follow_up_instruction=FOLLOW_UP,
        purpose=('Rebind the LT1.1 runner after the seam fix, and register the '
                 'resume-route gate that now covers the path which failed.'),
        defect=dict(
            symptom="KeyError: 'CPU' at grm_lt1_amendment4.py:77 during "
                    '`--arm A+ --resume`, before any GPU lease was taken',
            receipt='artifacts/grm_d1/lt1_1/lead_Ap_resume.log',
            causes=[
                dict(id='D3-C1', kind='signature drift',
                     what='the redirected `binding` took an ARM; LT1 callers '
                          "pass a BACKEND LABEL ('CPU') as well as an arm, and "
                          "LT1's own `binding` is label-agnostic",
                     fix='`binding(label)` keeps the LT1 signature and reads '
                         'the arm from pinned runner state (`current_arm`)'),
                dict(id='D3-C2', kind='seam too wide',
                     what='`lt.binding` was redirected, but LT1 uses it to '
                          'validate ITS OWN chain (`apply4` compares a '
                          'recorded `protocol_binding` against it); receipts '
                          'are stamped through `worker.bind`',
                     fix='redirect `worker.bind` only; `lt.binding` stays LT1'),
                dict(id='D3-C3', kind='ordering',
                     what="LT1's host preflight ran INSIDE the redirected "
                          'window, so `lt.binding` hashed the LT1.1 fixture '
                          'and failed AMENDMENT4_PROTOCOL_MISMATCH',
                     fix='`host_preflight()` runs before any seam moves'),
            ],
            why_the_gates_missed_it=(
                'Neither `--dry-run` nor the fake path traverses '
                '`grm_lt1_amendment4.apply`; only the resume route does. 115 '
                'passing tests covered everything except the one path the '
                'lead actually ran.')),
        seam_audit=dict(
            redirected={
                'lt.FIX': 'PosixPath constant; same type, no signature. Read '
                          'by `execute` for turns/probes/decisions AND by '
                          '`lt.binding` for its fixture sha -- which is why '
                          'the LT1 host preflight must run outside the window.',
                'worker.bind': 'callable(label) -> dict. Keeps the LT1 '
                               'signature; stamps LT1.1 receipts.',
                'worker.RUN': 'PosixPath constant; per-arm campaign root.',
            },
            deliberately_not_redirected={
                'lt.binding': "LT1's self-validation (verify/preflight/apply4). "
                              'Redirecting it breaks the amendment-4 chain.',
                'lt.RUN': "LT1's own campaign root; the runner has its own "
                          '`summary` and never calls `lt.summary`.',
                'lt.REG': "LT1's own registration; `verify` must validate it.",
            },
            method='Every redirected name was checked for arity and argument '
                   'meaning against every call site in grm_lt1.py, '
                   'grm_lt1_worker.py, grm_lt1_amendment3.py and '
                   'grm_lt1_amendment4.py. The two Path constants carry no '
                   'signature; the one callable now matches LT1 exactly.'),
        runner=dict(
            path=RUNNER,
            sha256=sha_path(ROOT / RUNNER),
            superseded_sha256=parent['runner']['sha256'],
            tests=list(RUNNER_TESTS),
            tests_sha256={name: sha_path(ROOT / name) for name in RUNNER_TESTS},
            entry_points=['--preflight', '--dry-run', '--resume',
                          '--resume --dry-lease', '--summary', '--fake',
                          '--worker', '--fake-cell'],
            new_entry_point=dict(
                flag='--resume --dry-lease [--no-host-gate]',
                what='walks the production resume route -- amendment load, '
                     'apply4-bearing preflight, seam redirection, arm pin, '
                     'campaign owner file, `worker.pending` cell selection -- '
                     'and stops at `worker.run_cell`, the lease boundary. '
                     'Nothing before that point is stubbed.',
                why='a registration is not runnable until a worker executes '
                    'it, and a route is not covered until a gate traverses it')),
        gate=dict(
            tests='tests/test_grm_lt1_1_resume_route.py',
            red_before='test_the_shipped_seam_reproduces_the_leads_keyerror '
                       'reconstructs the shipped seam and asserts '
                       "KeyError: 'CPU' surfaces through "
                       'grm_lt1_amendment4.apply',
            green_after='test_resume_route_reaches_the_lease_boundary[A] and '
                        '[A+] traverse the real route and stop at the lease',
            seam_audit_test='test_only_the_documented_seams_move asserts '
                            'lt.binding / lt.RUN / lt.REG never move and that '
                            'every moved seam is restored',
            evidence_class='CPU route traversal (control flow and bindings; '
                           'NOT a language-model quality measurement)'),
        host_blocker=dict(
            status='OPEN — pre-existing, not introduced by this work',
            symptom='`--arm A+ --resume` (without --dry-lease) now reaches '
                    "LT1's host preflight and stops at "
                    'INPUT_SHA_MISMATCH: core/graft_arena.py',
            cause='this worktree forks grm-merge, whose core has drifted past '
                  'the SHAs the LT1 registration pins; `lt.preflight()` fails '
                  'identically with or without the LT1.1 redirection',
            evidence='artifacts/grm_d1/red_before_lt1.log, '
                     'artifacts/grm_d1/core_drift_graft_arena.diff',
            consequence='the GPU campaign cannot start on this tree until the '
                        'lead decides how the LT1 core pins are rebound for '
                        'the host preflight. The LT1.1 route itself is proven '
                        'green up to the lease boundary.',
            lead_decision_required=True),
        arms=parent['arms'],
        budget_gpu_seconds_per_arm=parent['budget_gpu_seconds_per_arm'],
        budget_gpu_seconds_total=parent['budget_gpu_seconds_total'],
        budget_gpu_hours_total=parent['budget_gpu_hours_total'],
        process_safety=dict(
            gpu='only `--resume` without `--dry-lease` takes the shared lease, '
                "through LT1's own run_cell (flock + nvidia-smi idle check + "
                'no-kill child wait). --preflight / --dry-run / --summary / '
                '--fake / --resume --dry-lease never do.',
            arms='A and A+ never run concurrently on the same GPU',
            git='lead commits; the seat never runs git'),
        prior_art=[
            'SHA-chained amendment contract: scripts/grm_lt1.py:verify and C7 '
            'r3 (GRM contributors, 2026)',
            'The seam distinction (worker.bind stamps receipts, lt.binding '
            'validates LT1 own chain): a reading of grm_lt1_worker.py, '
            'grm_lt1.py and grm_lt1_amendment4.py (GRM contributors, 2026)',
            'Pin-after-strip-then-read-back: R1 pin_rule, '
            'scripts/grm_r1_replay.py:242 (GRM contributors, 2026)',
            'Lease-boundary route gate and pinned-arm state for a '
            'label-signature binding: no prior art known to me',
        ],
    )


def lead_commands(amend_sha):
    doc = amendment()
    correction = json.loads(AMENDMENT2.read_text())['budget_correction']
    lines = [
        '# LT1.1 — arm A and arm A+ (amendment 3)',
        '# registration sha256: %s' % doc['registration_sha256'],
        '# amendment 2  sha256: %s' % doc['previous_amendment_sha256'],
        '# amendment 3  sha256: %s' % amend_sha,
        '# runner            : %s (sha256 %s)'
        % (RUNNER, doc['runner']['sha256']),
        '#',
        '# 26 cells per arm. Reservation ceiling %d s (%.2f GPU-h) PER ARM;'
        % (doc['budget_gpu_seconds_per_arm'],
           doc['budget_gpu_seconds_per_arm'] / 3600.0),
        '# projected spend %.2f GPU-h per arm. Both arms: %.2f GPU-h ceiling.'
        % (correction['projected_gpu_hours'], doc['budget_gpu_hours_total']),
        '# The arms are separately resumable: run either alone.',
        '#',
        '# OPEN BLOCKER (pre-existing, needs a lead decision): the real',
        '# --resume stops at INPUT_SHA_MISMATCH: core/graft_arena.py. This',
        "# worktree's core has drifted past the SHAs the LT1 registration",
        '# pins, and the LT1 host preflight fails identically with or without',
        '# LT1.1. The LT1.1 route itself is green to the lease boundary',
        '# (see --dry-lease below).',
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
            '# Route check on CPU: walks the REAL resume path (apply4,',
            '# binding, preflight, cell selection) and stops at the lease.',
            'python3 %s --arm %s --resume --dry-lease --no-host-gate'
            % (RUNNER, arm),
            '# GPU campaign (takes the shared lease; resumable -- run again',
            '# to continue). BLOCKED until the host core pins are resolved:',
            'python3 %s --arm %s --resume' % (RUNNER, arm),
            'python3 %s --arm %s --summary' % (RUNNER, arm),
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
    (OUT / 'amendment3.json').write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (OUT / 'amendment3.sha256').write_text('%s  amendment3.json\n' % digest)
    (OUT / 'lead_commands.txt').write_text(lead_commands(digest))

    doc = amendment()
    print('amendment3 sha256 %s' % digest)
    print('parent amendment2 %s' % doc['previous_amendment_sha256'])
    print('runner %s' % doc['runner']['sha256'])
    print('  supersedes %s' % doc['runner']['superseded_sha256'])
    print('defect causes: %s'
          % ', '.join(c['id'] for c in doc['defect']['causes']))
    print('host blocker: %s' % doc['host_blocker']['symptom'])


if __name__ == '__main__':
    main()
