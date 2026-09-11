# GRM-RD2 — why aliases and corrections read as the stale value once the reader answers (worktree wt/grm-rd1, branch grm-rd1)

RD1 (`artifacts/grm_rd1/`, chain log `logs/RD1_chain.log`): with the
fixture's abstention clause removed (arm A2) the oracle reads 16/16 and
memory reads fresh 4/4 and folded 9/10, but **every alias (8) and every
correction (8) is answered with a WRONG value** — the served texts are
in the A2 receipts. Diagnose from receipts + CPU replay only (no GPU,
no core edits; STOP and report with file/lines if the cause is core).
Same rules (no git, no subagents, foreground, never kill anything).
Effort: high.

## Mission
1. For each of the 16 wrong rows: quote question, expected, served,
   and the mount plan at that turn (mounted ids, their texts, which of
   them holds the CURRENT value / the superseded value / the alias
   edge / the alias base). Classify: (a) the current node was not
   mounted (only the superseded one was), (b) both were mounted and
   the model chose the old one, (c) the alias edge node was mounted
   without its base (or vice versa), (d) other.
2. For (a)/(c): trace the ranking/admission/demotion decision in the
   route receipt (`admission_ranking_before_demotion`, RT1 fields,
   L2 supersession) and say why the current node lost.
3. For (b): check the arena order of the two mounts relative to the
   live band (RS3 seat-near-live) and whether the superseded node is
   marked retired/`no_fold` in the repository at that turn — the
   design says corrections retire the old value; if it is retired and
   still mounted, that is a core finding.
4. Compare with C2's supersession battery (9/9 under the profile at
   short distance): what differs here (distance ≥ 30, folds, two
   restarts, the fixture's phrasing "current X value")?
5. State whether the production ladder prompt (`scripts/grm_e2e_session.py`,
   EB1 batteries) contains any abstention clause like the C7 fixture's;
   quote it.

## Done (verbatim)
1. The 16-row table with classes; the dominant class and the
   mechanism with receipts; harness/core ruling.
2. Recommendation (fix candidate, RD3 contrast, or both) with cost.
3. Prior art; deviations; RED; process safety; model id and effort.
