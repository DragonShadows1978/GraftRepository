# GRM-D1 — LT1 residual diagnosis + recap battery redesign (lead order, 2026-09-10)

Seat: Opus 5 (opus-max). YOUR WRITABLE TARGET is the git worktree
`/mnt/ForgeRealm/wt/grm-d1` (branch `grm-d1`, forked from `grm-merge`).
Edits under `scripts/grm_lt1*`, `tests/`, `artifacts/grm_d1/` and the
LT1 fixture files are AUTHORIZED; `core/` is READ-ONLY for this order
(if you find a core cause, STOP on it with a receipt and a proposed
fix, do not implement). Read-only: `/mnt/ForgeRealm/GraftRepository`,
every other `/mnt/ForgeRealm/wt/grm-*` worktree,
`/mnt/Shared/LEAD_TODO_2026-09-10.md`,
`/mnt/Shared/GRM_LT1_200Turn_Conversation_Result_2026-09-09.md`. No git.
No subagents. Foreground only, Bash calls < 10 min, no background waits.
NEVER kill or signal a process you did not start. No GPU: CPU analysis
from receipts/checkpoints; if a GPU replay is genuinely required,
REGISTER it (≤ 0.3 GPU-h, lead-run) instead of running it.

## Context
LT1 arm A (profile, margin_first, `/mnt/ForgeRealm/wt/grm-lt1/
artifacts/grm_lt1/amendment2/run_margin_first/cells/…`, summary in
`comparison.json`): 24/35 exact, 11 WRONG VALUES (fresh 1, corrections
5, aliases 5), 0 abstentions; recap 0/5 on both arms ("under-specified
question"). Oracle 15/35. RD2 on C7 found corrections wrong because the
FIXTURE registered the correction as an ordinary fact (no supersession
edge, old value ranked first) and aliases wrong because the edge is
mounted without its base. LT1 amendment 4 made the harness publish
correction nodes fed through `arena.feed()` at mount time — check
whether those corrections carry production supersession semantics
(L2 edge, retirement of the old value) or are the same fixture trap.

## Mission
1. Per-row cause table for the 11 wrong rows (and the 11 wrong rows of
   arm B for contrast where cheap): question id, distance, expected,
   served value, nodes ranked/mounted (ids + which turn deposited
   them), whether the current value's node exists / was superseded /
   was mounted / was seated where (RS3 position), whether an alias edge
   or base was mounted, route receipt fields (`admission_rule`,
   `served_from`, RT1 provenance). Classify each row: fixture-lineage
   (harness), stale-first ranking (core A-DEC), alias-edge-without-base
   (core composition — A1's target), reader wrong value with the right
   mount, other. Cite the receipt path per row.
2. If the corrections are a fixture-lineage trap: fix the LT1 fixture/
   harness so corrections are real supersessions (as C7 r3 did:
   `wt/grm-c7` commit 9e5533b for the shape), with a CPU fixture proving
   the old value is retired and the new one ranks first on the fake
   model, and re-register LT1.1 (arm A only, same frozen conversation,
   same 52-cell shape, ≤ 1.7 GPU-h, flags: profile + margin_first +
   `GRM_ALIAS_FOLD_MERGE` when present on the merged tree — put the
   alias flag in the registration as a NAMED optional so the lead can
   pin it).
3. Recap battery redesign: replace the five under-specified recap
   questions with five ANSWERABLE ones, each naming the decision it
   targets and a value-span expected answer, scored with the existing
   value-span scorer; keep the fixture gate rules (no probes inside the
   recency window; no instruction-word identifiers). Register them in
   the LT1.1 registration with a CPU gate showing the oracle answers
   5/5 on the fake session (the C7/LT1 oracle path).
4. Gates: `python3 -m pytest -q tests/test_grm_lt1*.py
   tests/test_grm_d1*.py` verbatim last; `artifacts/grm_d1/REPORT.md`,
   `LEDGER.md`, cause table as JSON + Markdown; LT1.1
   `lead_commands.txt` + registration + sha.

## Done (verbatim)
1. The cause table (11 rows, receipt path each) and the class counts.
2. Fixture/harness changes (files + lines) with RED→GREEN lines; what
   is core (STOPPED, with the proposed fix) vs harness (fixed).
3. LT1.1 registration path + sha, budget, the five recap questions
   with expected spans, the optional alias flag hook.
4. Prior art, deviations, RED items, process-safety statement, model id
   and effort. The pytest line LAST.
