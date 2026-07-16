# SUP-WO1: Supersession battery + length-debias (L1) + revision-aware resolution (L2)

YOUR WRITABLE TARGET is the WORKTREE
/home/vader/GraftRepository-supersession (branch grm-supersession) —
edits AUTHORIZED anywhere inside it EXCEPT docs/ plan files (append-only
notes go in your report, not the plans). The main tree at
/mnt/ForgeRealm/GraftRepository is READ-ONLY reference (a sibling seat
is editing it in parallel — never touch it). No git (the lead commits
from the worktree). No subagents. No GPU runs / model loads — CPU
verification only; the lead runs all GPU gates serially. RED honesty;
no monitor-idling.

## Law

docs/GRM_SUPERSESSION_PLAN.md (in your worktree) is the spec — read it
FIRST, plus the seam characterization in docs/GRM_E2E_RECEIPT_LEDGER.md
(2026-07-08 entries: Orion probe failure, product-scoring/max-pool
length-bias decomposition, the E1/E2 verdict letters) and the M5 entry
in docs/GRM_BUG_QUEUE.md (existing explicit-supersede machinery — find
and reuse its metadata, do not invent a parallel scheme).

## Build (three pieces, in dependency order)

1. **Battery fixtures + harness** (tests/fixtures/supersession_battery/
   + tests/test_grm_supersession_battery.py): scripted sessions per the
   plan's G0-SUP scenario list — short correction vs long value-bearing
   competitor, multi-hop supersession A→B→C, correction-then-
   restatement, fresh-fact controls. Harness runs UNPATCHED baseline
   behind --run-gpu (lead executes): route-rank diagnostics per probe
   (rank of correction node, rank of stale node, rank of competitor;
   which mounted; answer text; correct/stale/wrong-fact classification)
   + JSON receipt lines. Baseline floors feed threshold registration —
   print machine-readable, assert nothing about pass/fail at G0.
2. **L1 length-debias** (core/graft_arena.py route scoring): a
   length-debiased latent score, justified in code comments against
   the E2E decomposition (max-pool over pairs = long-key dominance).
   Candidate forms per plan (length-normalized max, mean-over-top-m);
   pick ONE, name why, freeze. Kind-priors are FORBIDDEN (plan
   non-goal). Feature-flag it (default OFF; --debias flag in the
   battery harness) so baseline vs L1 is a clean A/B on one build.
3. **L2 revision-aware mount resolution** (arena mount-set assembly,
   repository lineage plumbing as needed): after routing, before
   injection — if candidate A supersedes candidate B (M5 explicit
   metadata; plus same-lineage detection if the repository already
   tracks it — report what exists), exclude B from the mount set when
   A is present. B stays routable; explicit descent can still reach
   it. Also feature-flagged (--resolve).

## Verify

py_compile everything touched; CPU unit tests for: debias math
(hand-computed score cases, long-vs-short key dominance flips),
resolution logic (pairwise, multi-hop head-finding, no-successor
passthrough, cycle guard), battery fixture validator. Run the shared
regression suites in YOUR worktree:
  python3 -m pytest tests/test_grm_supersession_battery.py tests/test_grm_importance_telemetry.py tests/test_grm_fold_recovered_guard.py -q
All green or explain precisely. Do NOT run --run-gpu anything.

## Done

Final message MUST contain verbatim: diff summary per file; pytest
summary line; the chosen debias form and its one-paragraph
justification; what supersede-lineage metadata you found and reused
(file:line); exact --run-gpu commands for the lead (baseline battery,
--debias, --debias --resolve); spec ambiguities hit; confirmation
"no git, no GPU, main tree untouched".
