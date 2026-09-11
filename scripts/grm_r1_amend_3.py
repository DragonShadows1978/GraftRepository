#!/usr/bin/env python3
"""GRM-R1 amendment 3: a satisfied re-arm scope must stop gating the campaign.

Defect (lead, `artifacts/grm_r1/batch_R2_a2.log`): batch R1's re-issue
COMPLETED -- 8/8 cell receipts, both arms, payloads released, lease released
at 79 s -- and then `--batch R2` refused to start:

    ValueError: R1_AMENDMENT_RETAIN_MISMATCH: R1 on_disk=[...8 cells...]
    amendment=[...5 cells...]

Cause, and it is mine: `resume_scope()` re-applies amendment 1's
"R1 retains 5 / reissues 3" cross-check on EVERY invocation, including from
batches that have nothing to do with R1.  That check is a PRECONDITION -- it
asks "is the state I am about to act on still the state this amendment was
written against?"  Once R1's controller says COMPLETE the re-issue has
happened and the amendment's scope is SATISFIED: every registered cell has a
receipt, which is precisely the outcome the amendment existed to produce.
Re-applying the check then compares the finished state against the
pre-re-issue expectation, so the guard closes the campaign on its own
success.  Amendment 2 fixed a self-collision on scratch; this is the same
family of mistake one level up -- a guard that was correct while the batch
was open and became wrong the moment it succeeded.

Fix: the retain/reissue cross-check now applies only while the re-armed
batch's controller is NOT COMPLETE.  Once complete, `resume_scope()` records
`scope_satisfied: true` with the sha256 of every cell receipt and an empty
reissue list, and later batches proceed.  `batch()` treats a satisfied scope
as a RECORD, never as a re-issue instruction.

What is deliberately NOT relaxed: a genuine disagreement still STOPS.  While
the batch is open, retain/reissue must match the amendment exactly
(`R1_AMENDMENT_RETAIN_MISMATCH` / `_REISSUE_MISMATCH`).  And a controller
that says COMPLETE while cells are still missing is a new RED,
`R1_AMENDMENT_SCOPE_UNSATISFIED` -- "complete" is only accepted as satisfying
the amendment when the whole registered cohort really does have receipts.

Scope: rebind the worker sha.  R1's re-arm entry is retained verbatim for the
record (it is now satisfied); R2-R5 unchanged; no budget change.  Neither
admission, routing nor scoring is touched, and the prediction, verdict rule
and parity barrier are unchanged.

Prior art: GRM C7/C2 sha-bound amendment chains (GRM contributors, 2026) --
TAKEN verbatim, including chaining to the previous amendment's hash.  The
"precondition vs postcondition" distinction is ordinary defensive-programming
practice (a guard checked before an action must not be re-evaluated against
the state that action produced); no specific prior art known to me for this
composition, and no new algorithm.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.grm_c7_amendment7 import read, sha, write, need  # noqa: E402
from scripts import grm_r1_replay as r1  # noqa: E402

#: Sources this amendment re-binds (their hashes changed).
REBIND = (
    'scripts/grm_r1_replay.py',
    'scripts/grm_r1_amend_3.py',
)


def build(root=None):
    root = Path(root) if root is not None else r1.OUT
    path = root / 'amendment_3.json'
    need(not path.exists(), 'R1_AMENDMENT_3_IMMUTABLE_ALREADY_EXISTS')
    previous = root / 'amendment_2.json'
    need(previous.is_file(), 'R1_AMENDMENT_3_REQUIRES_AMENDMENT_2')
    a2 = read(previous)
    r = read(r1.REG)
    need(a2['registration_sha256'] == sha(r1.REG),
         'R1_AMENDMENT_3_CHAIN_MISMATCH')
    # Record the state this amendment is written against: R1 finished.
    r1_cells = root / 'gpu/R1/cells'
    receipts = {p.stem: sha(p) for p in sorted(r1_cells.glob('*.json'))}
    controller = read(root / 'gpu/R1/controller.json')
    need(controller['status'] == 'COMPLETE',
         'R1_AMENDMENT_3_EXPECTS_A_COMPLETED_REARM')
    need(len(receipts) == len(r['batches']['R1']),
         'R1_AMENDMENT_3_COHORT_INCOMPLETE')
    value = {
        'schema': 'grm.r1.amendment.v1',
        'amendment': 3,
        'registration_sha256': sha(r1.REG),
        'previous_amendment_sha256': sha(previous),
        'order': 'orders/GRM_R1_MARGIN_FIRST_REPLAY.md',
        'order_sha256': sha(ROOT / 'orders/GRM_R1_MARGIN_FIRST_REPLAY.md'),
        'defects': [
            {'id': 'R1-D4-satisfied-scope-still-gating',
             'evidence': 'artifacts/grm_r1/batch_R2_a2.log',
             'symptom': ('R1_AMENDMENT_RETAIN_MISMATCH raised from '
                         '--batch R2 after R1 completed with 8/8 receipts; '
                         'on_disk=8 vs amendment retain=5.'),
             'cause': ('resume_scope() re-applied the retain/reissue '
                       'cross-check on every invocation. It is a '
                       'PRECONDITION for re-issuing an open batch, so '
                       're-evaluating it against the state the re-issue '
                       'produced closed the campaign on its own success.'),
             'fix': ('The cross-check applies only while the re-armed '
                     "batch's controller is not COMPLETE. Once complete, "
                     'resume_scope records scope_satisfied: true with the '
                     'receipt hashes and an empty reissue list, and later '
                     'batches proceed. batch() treats a satisfied scope as '
                     'a record, never a re-issue instruction.'),
             'still_red': ('A genuine disagreement while the batch is OPEN '
                           'still STOPS. A COMPLETE controller with cells '
                           'still missing is a new RED: '
                           'R1_AMENDMENT_SCOPE_UNSATISFIED.')},
        ],
        'rebound_inputs': {p: sha(ROOT / p) for p in REBIND},
        'rebound_reason': ('The fix lives in the worker, so its hash '
                           'changes; every other registered input keeps its '
                           'original hash and still fails closed.'),
        'rearm': a2['rearm'],
        'rearm_retained_for_the_record': True,
        'rearm_status': {'R1': {'satisfied': True,
                                'controller_status': controller['status'],
                                'charged_seconds': controller['charged_seconds'],
                                'cells': len(receipts),
                                'receipts_sha256': receipts}},
        'retained_receipts_policy': a2['retained_receipts_policy'],
        'unchanged_batches': a2['unchanged_batches'],
        'unchanged_reason': ('The fix changes when a guard is evaluated, not '
                             'what any batch costs.'),
        'prior_charged_seconds': a2['prior_charged_seconds'],
        'added_reserved_seconds': a2['added_reserved_seconds'],
        'total_reserved_seconds': a2['total_reserved_seconds'],
        'gpu_cap_seconds': r['gpu_cap_seconds'],
        'prediction': r['prediction'],
        'verdict_rule': r['verdict_rule'],
        'acceptance_unchanged': True,
        'lessons': [
            'A precondition must not be re-evaluated against the state its '
            'own action produced. Amendment 2 fixed a self-collision on '
            'scratch; this is the same family one level up.',
            'Three of the four defects in this arc were guards that were '
            'correct in the state they were written for and wrong in the '
            'state that followed. Each new guard needs an explicit answer '
            'to "when does this stop applying?".',
        ],
        'prior_art': __doc__,
        'evidence_class': ('Campaign-control amendment. Changes when the '
                           're-arm precondition is evaluated; no admission, '
                           'routing or scoring logic is touched, and the '
                           'prediction and verdict rule are unchanged.'),
    }
    write(path, value)
    path.with_suffix('.sha256').write_text(sha(path) + '  amendment_3.json\n')
    return value


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
