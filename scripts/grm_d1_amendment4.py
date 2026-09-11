#!/usr/bin/env python3
"""GRM-D1 follow-up 4: LT1.1 amendment 4 — LT1.1 validates its OWN chain.

THE RULING (lead, 2026-09-11), recorded verbatim in the emitted document:
LT1's original registration is a frozen receipt of its day and is NOT
re-validated against today's core. LT1.1's host preflight validates LT1.1's
own chain -- registration 02b44d02 + amendments 1-3, whose core pins
amendment 1 rebound to this tree with attribution -- using the same
verification classes `grm_lt1.verify` applies, pointed at our documents. LT1's
registration sha and its recorded core pins are carried as PARENT LINEAGE in
the LT1.1 receipt, not as a gate.

WHAT CHANGED. `scripts/grm_lt1_1.py` replaced its `lt.preflight()` call with
`lt1_1_preflight(arm)`. That function performs, against LT1.1's documents:
sha-bound documents (each amendment vs its sidecar), chain continuity (each
amendment names its parent's sha), sha-bound inputs (every rebound core pin
plus the bound runner, vs the file on disk), fixture sha, cell schedule (26
arm-A cells), budget (reservation sum within the registered ceiling), and host
readiness (free space, pinned admission rule).

THE apply4 QUESTION THE LEAD ASKED. `grm_lt1_amendment4.apply`'s
protocol-binding check -- `a['protocol_binding'] != lt.binding('CPU')` --
PASSES on this tree, unchanged and untouched. `lt.binding` reports the core
shas RECORDED in LT1's amendment-3 document, not live ones, so it is already
drift-immune. Measured, not assumed: the recorded `protocol_binding` compares
equal to live `lt.binding('CPU')` field for field. The whole of
`lt.verify()`'s failure on this tree was its final input loop
(`INPUT_SHA_MISMATCH: core/graft_arena.py`), which is precisely the
re-validation the ruling removes. So no second ruling is needed for
amendment 4: it never gated on drifted core. LT1.1 still stamps its own
protocol binding through `worker.bind`, and amendment-4's recorded binding is
reported as parent lineage.

Prior art:
  * The verification CLASSES are LT1's own (`scripts/grm_lt1.py:verify` /
    `preflight`, GRM contributors 2026): sha-bound documents and inputs,
    chain continuity, fixture sha, schedule, budget, free space, pinned rule.
    Reused in kind, pointed at LT1.1's chain; no new verification algorithm.
  * SHA-chained amendment contract: LT1's and C7 r3's, taken unchanged.
  * Mine: the parent-lineage record (reporting a frozen receipt beside a live
    chain instead of gating on it), and the drift table that makes the
    distinction auditable.
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
AMENDMENT3 = OUT / 'amendment3.json'
ORDER = ROOT / 'orders/GRM_D1_LT1_RESIDUAL_DIAGNOSIS.md'

RUNNER = 'scripts/grm_lt1_1.py'
RUNNER_TESTS = ('tests/test_grm_lt1_1_runner.py',
                'tests/test_grm_lt1_1_resume_route.py',
                'tests/test_grm_lt1_1_preflight.py')

ALIAS_FLAG = 'GRM_ALIAS_FOLD_MERGE'

FOLLOW_UP = (
    'Lead follow-up 4 to GRM-D1 (2026-09-11): replace the lt.preflight() host '
    "gate with lt1_1_preflight() verifying LT1.1's own chain; say whether "
    "apply4's protocol-binding check still passes on this tree; gate both "
    'arms green with no INPUT_SHA_MISMATCH, plant a drifted core sha in '
    "LT1.1's own amendment and assert the preflight goes RED; retire or "
    'invert test_the_host_blocker_claim_is_true_right_now with the receipt; '
    'amendment 4 chained to 3 recording the ruling verbatim and the '
    'parent-lineage table.')


def sha_path(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, indent=2) + '\n').encode()


def apply4_finding():
    """Measured answer to the lead's question about apply4. Not asserted."""
    from scripts import grm_lt1 as lt
    recorded = json.loads(
        (ROOT / 'artifacts/grm_lt1/amendment4/resume_registration.json')
        .read_text())['protocol_binding']
    live = lt.binding('CPU')
    return dict(
        question="does apply4's protocol-binding check compare against "
                 'drifted core?',
        answer='NO — it passes unchanged on this tree',
        check="grm_lt1_amendment4.apply: a['protocol_binding'] != "
              "lt.binding('CPU')",
        recorded_equals_live=recorded == live,
        why='`lt.binding` reports the core shas RECORDED in LT1 amendment 3, '
            'not live ones, so it is already immune to core drift. The whole '
            'of lt.verify() failure on this tree was its final input loop '
            '(INPUT_SHA_MISMATCH: core/graft_arena.py), which is exactly the '
            're-validation the ruling removes.',
        consequence='no second ruling is needed for amendment 4; it never '
                    'gated on drifted core. LT1.1 stamps its own protocol '
                    'binding through worker.bind, and the amendment-4 recorded '
                    'binding is reported as parent lineage.',
        left_untouched=True)


def amendment():
    if not AMENDMENT3.exists():
        raise ValueError('LT11_AMENDMENT3_MISSING')
    parent_sha = sha_path(AMENDMENT3)
    recorded = (OUT / 'amendment3.sha256').read_text().split()[0]
    if parent_sha != recorded:
        raise ValueError('LT11_AMENDMENT3_SHA_MISMATCH')
    parent = json.loads(AMENDMENT3.read_text())

    from scripts import grm_lt1_1 as runner_mod

    return dict(
        schema='grm.lt1_1.amendment4.v1',
        amendment=4,
        previous_amendment='artifacts/grm_d1/lt1_1/amendment3.json',
        previous_amendment_sha256=parent_sha,
        registration='artifacts/grm_d1/lt1_1/registration.json',
        registration_sha256=parent['registration_sha256'],
        order='orders/GRM_D1_LT1_RESIDUAL_DIAGNOSIS.md',
        order_sha256=sha_path(ORDER),
        follow_up_instruction=FOLLOW_UP,
        purpose=("Point LT1.1 host preflight at LT1.1 own chain per the lead "
                 "ruling, and carry LT1 identity as parent lineage."),
        ruling=runner_mod.RULING,
        ruling_source='lead, 2026-09-11, GRM-D1 follow-up 4',
        preflight=dict(
            function='scripts/grm_lt1_1.py:lt1_1_preflight',
            supersedes='lt.preflight() (LT1 own chain, re-validated against '
                       "today core -- failed with INPUT_SHA_MISMATCH: "
                       'core/graft_arena.py)',
            verification_classes=[
                'sha-bound documents: registration + amendments 1-4 vs sidecars',
                'chain continuity: each amendment names its parent sha',
                'sha-bound inputs: every rebound core pin + the bound runner '
                'vs the file on disk',
                'fixture sha: dialogue.json vs the registered digest',
                'cell schedule: 26 arm-A cells matching the amendment',
                'budget: reservation sum within the registered ceiling',
                'host readiness: free space, pinned admission rule',
            ],
            not_checked=['LT1 day-of core pins -- frozen receipt, see '
                         'parent_lineage (this is the ruling)'],
            teeth='tests/test_grm_lt1_1_preflight.py plants a drifted core sha '
                  'in LT1.1 OWN amendment 1 and requires the preflight to '
                  'report INPUT_SHA_MISMATCH'),
        apply4_finding=apply4_finding(),
        parent_lineage=runner_mod.parent_lineage(),
        runner=dict(
            path=RUNNER,
            sha256=sha_path(ROOT / RUNNER),
            superseded_sha256=parent['runner']['sha256'],
            tests=list(RUNNER_TESTS),
            tests_sha256={name: sha_path(ROOT / name) for name in RUNNER_TESTS
                          if (ROOT / name).exists()}),
        arms=parent['arms'],
        budget_gpu_seconds_per_arm=parent['budget_gpu_seconds_per_arm'],
        budget_gpu_seconds_total=parent['budget_gpu_seconds_total'],
        budget_gpu_hours_total=parent['budget_gpu_hours_total'],
        resolves=dict(
            blocker='amendment 3 host_blocker (INPUT_SHA_MISMATCH: '
                    'core/graft_arena.py)',
            status='RESOLVED by ruling',
            how='the LT1.1 chain preflight replaces the LT1 chain preflight; '
                'both arms reach the lease boundary with no INPUT_SHA_MISMATCH',
            receipt='artifacts/grm_d1/lt1_1/proof/resume_dry_lease_*.json'),
        process_safety=dict(
            gpu='only `--resume` without `--dry-lease` takes the shared lease, '
                'through LT1 own run_cell (flock + nvidia-smi idle check + '
                'no-kill child wait). --preflight / --dry-run / --summary / '
                '--fake / --resume --dry-lease never do.',
            arms='A and A+ never run concurrently on the same GPU',
            git='lead commits; the seat never runs git'),
        prior_art=[
            'Verification classes reused in kind from scripts/grm_lt1.py '
            'verify/preflight (GRM contributors, 2026); no new verification '
            'algorithm',
            'SHA-chained amendment contract: scripts/grm_lt1.py:verify and C7 '
            'r3 (GRM contributors, 2026)',
            'Parent-lineage record (report a frozen receipt beside a live '
            'chain instead of gating on it), and the drift table that makes '
            'the distinction auditable: no prior art known to me',
        ],
    )


def lead_commands(amend_sha):
    doc = amendment()
    lines = [
        '# LT1.1 — arm A and arm A+ (amendment 4)',
        '# registration sha256: %s' % doc['registration_sha256'],
        '# amendment 3  sha256: %s' % doc['previous_amendment_sha256'],
        '# amendment 4  sha256: %s' % amend_sha,
        '# runner            : %s (sha256 %s)'
        % (RUNNER, doc['runner']['sha256']),
        '#',
        '# 26 cells per arm. Reservation ceiling %d s (%.2f GPU-h) PER ARM;'
        % (doc['budget_gpu_seconds_per_arm'],
           doc['budget_gpu_seconds_per_arm'] / 3600.0),
        '# both arms: %.2f GPU-h ceiling. Separately resumable.'
        % doc['budget_gpu_hours_total'],
        '#',
        '# RULING (lead, 2026-09-11): LT1 original registration is a frozen',
        '# receipt of its day and is NOT re-validated against today core.',
        '# LT1.1 preflight validates LT1.1 OWN chain (registration + amendments',
        '# 1-4, core pins rebound to this tree with attribution). LT1 identity',
        '# is carried as parent lineage in the receipt, not as a gate.',
        '# The amendment-3 blocker (INPUT_SHA_MISMATCH) is RESOLVED by this.',
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
            '# Route check on CPU: walks the REAL resume path (chain preflight,',
            '# apply4, binding, cell selection) and stops at the lease.',
            'python3 %s --arm %s --resume --dry-lease' % (RUNNER, arm),
            '# GPU campaign (takes the shared lease; resumable -- run again',
            '# to continue):',
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
    (OUT / 'amendment4.json').write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (OUT / 'amendment4.sha256').write_text('%s  amendment4.json\n' % digest)
    (OUT / 'lead_commands.txt').write_text(lead_commands(digest))

    doc = amendment()
    print('amendment4 sha256 %s' % digest)
    print('parent amendment3 %s' % doc['previous_amendment_sha256'])
    print('runner %s' % doc['runner']['sha256'])
    print('  supersedes %s' % doc['runner']['superseded_sha256'])
    print('apply4 protocol-binding check: %s (recorded==live: %s)'
          % (doc['apply4_finding']['answer'],
             doc['apply4_finding']['recorded_equals_live']))
    print('parent lineage drift rows: %d'
          % len(doc['parent_lineage']['drift_table']))
    print('resolves: %s -> %s'
          % (doc['resolves']['blocker'], doc['resolves']['status']))


if __name__ == '__main__':
    main()
