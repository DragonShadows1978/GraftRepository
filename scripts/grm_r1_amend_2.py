#!/usr/bin/env python3
"""GRM-R1 amendment 2: stale-session archiving + environment restoration.

Two defects found by the lead after amendment 1 was committed (c2c8865).
Both are mine; both are fixed in the worker, and this amendment re-binds it.

DEFECT 1 -- re-issue collides with its own scratch.
`artifacts/grm_r1/batch_R1_a1.log`:
    FileExistsError: [Errno 17] File exists:
    '.../gpu/R1/sessions/fix6-replay-defaults-census-restart-1--e2e_t22_mira_seal/off'
Cell 6 is the cell whose arm OFF finished and whose arm ON hit the OOM, so
its `off/` and `on/` session directories survived.  `open_arm` creates the
session with `mkdir(exist_ok=False)` -- correct for a first run, fatal for a
re-issue.  Fix: `archive_stale_sessions()` moves that cell's OWN stale arm
directories to `sessions/<cell>/attempt_<n>/` before the arms run.  Cell
RECEIPTS are create-only evidence and are never touched; session dirs are
scratch, so they are archived (still inspectable), never deleted.  A cell
that already has a receipt is never re-run, so this never sees one.

DEFECT 2 -- the rule pin leaked into the process environment.
`pin_rule` sets `GRM_ADMISSION_RULE`, and `environment(flags)` rewrites the
whole `GRM_` frame, but neither was restored.  `core.grm_admission.
admission_rule()` reads `os.environ` at CALL time, so in one pytest process
every module importing after a R1 test was silently re-ruled to
`margin_first`.  The lead observed 3 FIX-4 and 20 A1 failures that vanish
when R1 runs alone; I reproduced 4 failures with
`pytest tests/test_grm_r1_replay.py tests/test_grm_scout_fix4.py
tests/test_grm_admission.py`.  My own earlier "91 passed" put R1 LAST, which
is exactly why I never saw it -- a green suite whose greenness depends on
file order is not green.  Fix: `scoped_env()` snapshots and restores the
exact prior environment in a `finally` around each arm's pin and around the
batch's frame pin; a key absent before is REMOVED, not set to ''.

Scope: rebind the worker sha only.  R1 keeps amendment 1's retain-5 /
reissue-3 scope and its 134 s lease; R2-R5 unchanged.  Neither fix changes
admission, routing or scoring, and the prediction, verdict rule and parity
barrier are untouched.

Prior art: GRM C7/C2 sha-bound amendment chains (GRM contributors, 2026) --
TAKEN verbatim, including chaining to the PREVIOUS amendment's hash.
FIX8/C7 "archive the prior attempt, never overwrite" (GRM, 2026) -- TAKEN,
applied here to per-cell scratch instead of the batch controller.
GRM-A1 `grm_a1_gpu_contrast.pin_flags` (GRM, 2026) -- independently hit the
same "pin AFTER environment()" trap and records that pinning before it
"fails silently, and that exact mistake produced a wrong reading earlier in
this arc"; A1 pins and asserts but does not restore, which is the half this
amendment adds.  `unittest.mock.patch.dict(os.environ)` / pytest
`monkeypatch.setenv` (Python and pytest contributors) -- the standard
environment snapshot and restore idiom, reproduced in the worker because
production has no pytest fixture.  Nothing novel; no new algorithm.
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
    'scripts/grm_r1_amend_2.py',
)


def build(root=None):
    root = Path(root) if root is not None else r1.OUT
    path = root / 'amendment_2.json'
    need(not path.exists(), 'R1_AMENDMENT_2_IMMUTABLE_ALREADY_EXISTS')
    previous = root / 'amendment_1.json'
    need(previous.is_file(), 'R1_AMENDMENT_2_REQUIRES_AMENDMENT_1')
    a1 = read(previous)
    r = read(r1.REG)
    need(a1['registration_sha256'] == sha(r1.REG),
         'R1_AMENDMENT_2_CHAIN_MISMATCH')
    value = {
        'schema': 'grm.r1.amendment.v1',
        'amendment': 2,
        'registration_sha256': sha(r1.REG),
        'previous_amendment_sha256': sha(previous),
        'order': 'orders/GRM_R1_MARGIN_FIRST_REPLAY.md',
        'order_sha256': sha(ROOT / 'orders/GRM_R1_MARGIN_FIRST_REPLAY.md'),
        'defects': [
            {'id': 'R1-D1-stale-session-collision',
             'evidence': 'artifacts/grm_r1/batch_R1_a1.log',
             'symptom': ('FileExistsError: [Errno 17] File exists: '
                         '.../gpu/R1/sessions/fix6-replay-defaults-'
                         'census-restart-1--e2e_t22_mira_seal/off'),
             'cause': ('open_arm creates the session with '
                       'mkdir(exist_ok=False); the re-issued cell 6 still '
                       'had the off/ and on/ scratch dirs from the attempt '
                       'that OOMed, so the re-issue collided with its own '
                       'debris.'),
             'fix': ("archive_stale_sessions() moves that cell's own stale "
                     'arm dirs to sessions/<cell>/attempt_<n>/ before the '
                     'arms run. Receipts are create-only and untouched; '
                     'session dirs are scratch, archived not deleted.')},
            {'id': 'R1-D2-environment-leak',
             'evidence': ('lead: 3 FIX-4 + 20 A1 failures that vanish when '
                          'R1 runs alone; seat reproduced 4 failures with '
                          'pytest tests/test_grm_r1_replay.py '
                          'tests/test_grm_scout_fix4.py '
                          'tests/test_grm_admission.py'),
             'symptom': ('GRM_ADMISSION_RULE=margin_first survived the run '
                         'and re-ruled every module imported afterwards in '
                         'the same process.'),
             'cause': ('pin_rule and environment(flags) mutate os.environ, '
                       'which admission_rule() reads at call time, and '
                       'neither restored the prior values.'),
             'fix': ('scoped_env() snapshots and restores the exact prior '
                     "environment in a finally around each arm's pin and "
                     'the batch frame pin; a key absent before is REMOVED, '
                     'not set to empty. Tests use monkeypatch so nothing '
                     'outlives a test.')},
        ],
        'rebound_inputs': {p: sha(ROOT / p) for p in REBIND},
        'rebound_reason': ('Both fixes live in the worker, so its hash '
                           'changes; every other registered input keeps its '
                           'original hash and still fails closed.'),
        'rearm': a1['rearm'],
        'rearm_unchanged_from_amendment_1': True,
        'retained_receipts_policy': a1['retained_receipts_policy'],
        'unchanged_batches': a1['unchanged_batches'],
        'unchanged_reason': ('Neither fix changes per-cell cost: one moves '
                             'scratch directories, the other restores '
                             'environment variables.'),
        'prior_charged_seconds': a1['prior_charged_seconds'],
        'added_reserved_seconds': a1['added_reserved_seconds'],
        'total_reserved_seconds': a1['total_reserved_seconds'],
        'gpu_cap_seconds': r['gpu_cap_seconds'],
        'prediction': r['prediction'],
        'verdict_rule': r['verdict_rule'],
        'acceptance_unchanged': True,
        'lessons': [
            'A suite whose greenness depends on file order is not green: '
            "the seat's earlier 91-passed run placed R1 last and hid a real "
            'cross-module defect. Gate batteries should run the suspect '
            'module FIRST, not last.',
            'Create-only is right for evidence and wrong for scratch. The '
            'receipt/session distinction has to be explicit, or the '
            'no-overwrite rule turns into a self-collision on any re-issue.',
        ],
        'prior_art': __doc__,
        'evidence_class': ('Campaign-control and process-hygiene amendment. '
                           'Changes scratch-directory lifecycle and '
                           'environment restoration only; no admission, '
                           'routing or scoring logic is touched, and the '
                           'prediction and verdict rule are unchanged.'),
    }
    write(path, value)
    path.with_suffix('.sha256').write_text(sha(path) + '  amendment_2.json\n')
    return value


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
