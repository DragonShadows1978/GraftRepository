# GRM-C2 amendment 4 (lead, 2026-09-09) — the acceptance rule over-applies "restart retained" to the defaults arm; add a corrected score set; do not touch the running epoch

Lead-run, epoch 3 (scout-fix-2): supersession complete on both sides.
**profile 9/9 fresh, 9/9 restart, 575 seats, RT1 fields present, restart
retained; defaults 4/9 fresh, 3/9 restart, 1,391 seats, restart NOT
retained.** Then `score sup: RED` with `adoption_acceptance: false`,
and the runner returned 1, leaving census and long-history UNSTARTED.

Cause (`scripts/grm_c2_profile.py:126-131`): `acceptance` requires
`all(a['restart_retained'] and a['evidence_complete'] for a in arms.values())`,
i.e. the SHIPPED-DEFAULTS arm must also retain across restart. The
registered acceptance (order §3) is: profile ≥ baseline on every
battery, original-correct controls unchanged, restart retains metadata
and scores — for the PROFILE side. A defaults arm that loses answers
on restart is a finding against the defaults, not a reason to refuse
the profile. The sup score under the flawed rule stays on disk
(immutable RED, by your own rule); the lead has queued
`lead_commands.txt --resume`, which by your line 316 continues past
an existing score, so census and long-history will run under the
epoch-3 source UNCHANGED.

Therefore: **do not edit any file the epoch-3 runner or worker
executes** (`grm_c2_amended.py`, `grm_c2_cells.py`, `grm_c2_profile.py`,
`grm_c2_epoch3.py`, core). Same worktree, same rules (no git, no
subagents, no GPU, foreground, never kill anything). Effort: high.

## Mission
1. New module `scripts/grm_c2_score_a4.py` (+ tests) implementing the
   registered acceptance exactly (profile-side restart retention and
   evidence completeness; defaults-side restart retention reported as a
   finding column), reading the same epoch-3 cell receipts, writing a
   separate immutable score set `artifacts/grm_c2/epochs/scout-fix-2/scores_a4/`
   and a summary table with both rules side by side (original rule
   verdict, corrected rule verdict) so the reader sees why they differ.
   Sha-bound amendment JSON (this order + amendment_lead_3).
2. Pin with tests: the epoch-3 sup receipts score GREEN under the
   corrected rule and RED under the original; a synthetic case where the
   PROFILE loses a probe on restart scores RED under both.
3. `lead_commands_a4.txt`: the single CPU command the lead runs after
   the resumed campaign finishes (`score_a4 --all` then `summary_a4`).

## Done (verbatim)
1. Amendment path + sha; files/lines; test names; the sup table under
   both rules.
2. Exact lead commands.
3. Prior art; deviations; RED; process safety; model id and effort.
