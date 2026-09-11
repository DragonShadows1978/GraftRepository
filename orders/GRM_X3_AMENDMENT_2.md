# GRM-X3 amendment 2 (lead, 2026-09-08) — r2: a second 20-snapshot run with the realization scorer and the sham floor registered up front

Amendment 1 is committed (01b8f29). The r1 verdict stands as recorded:
RED_STRATA_OR_INCOMPLETE, lead P1 FAIL (sham 16/20 < 18 under the 0.05
nat rule), P2/P3/Q2/Q3 unevaluable because the frozen whole-answer
exact-equality scorer realized only 3/8 correct, 1/6 decoy, 0/6 refusal
(x3_00 etc. carry prose/Markdown and non-ASCII hyphens around the
value). Nothing in r1 is re-scored, re-thresholded, or reinterpreted.

The descriptive signal r1 leaves on the table is the reason for r2:
removal KL > 1 nat on 8/8 intended-correct, decoy mass high on 6/6
intended-decoy, fixed-classifier accuracy lesion 16/20 vs mass 4/20.
That is the X3 thesis ("the lesion is a better witness than mounted
mass") and it is unproven until it survives registered guards. David's
goal: test both sides; both sides here are the two witnesses.

Same worktree, same rules (no git, no subagents, no GPU, foreground,
never kill anything). Reasoning effort: high.

## r2 registration (new registration JSON, immutable, sha-bound to this order; r1 untouched)
1. **20 NEW snapshot recipes** (8 correct / 6 decoy / 6 refusal by
   intent), disjoint from r1's 20 in entity, value and question text,
   same frame (width 96, capture pin live, seat-near-live, RT1, demand
   OFF, reasoning low), same four-cell layout (≤ 285 s each; r1's cells
   ran inside the rail).
2. **Realization scorer registered BEFORE the run**: a stratum is
   realized by a contiguous, ordered value-span match on the served
   answer after Unicode dash/hyphen and whitespace normalization
   (NFKC, all dash code points → "-", whitespace collapsed, casefold),
   with a registered NEGATIVE control set (changed digit, omitted
   token, swapped relation, negation) that the scorer must reject in a
   CPU test. Report the frozen exact-equality realization beside it,
   separately labeled; the r2 guards use the value-span realization.
   Cross-reference: GRM-C5 arm S registers the same rule for grounding
   in `wt/grm-c5`; do not import from that worktree, implement it here
   and note the duplication for the lead to reconcile.
3. **Sham floor registered BEFORE the run** from r1's own sham
   distribution as a prior-run calibration (mean 0.037, max 0.078 nats):
   P1' = same-payload byte-identical 20/20 AND sham KL < 0.10 nats on
   ≥ 18/20. State plainly that 0.10 was chosen from r1 and that a
   sham that exceeds it in r2 is a FAIL, not a recalibration.
4. P2'/P3'/Q2'/Q3' unchanged in form from r1 (removal KL > 1 nat on
   correct; decoy mass high; lesion accuracy ≥ mass accuracy) but
   evaluated on the value-span strata; strata guard: ≥ 6/8, ≥ 4/6,
   ≥ 4/6 realized, else RED_STRATA again.
5. Kill rule unchanged. Budget: r2 = 4 × 285 s reservation; total X3
   0.64 GPU-h.
6. CPU gates: scorer positive/negative sets; forged/stale r2
   registration refused; r1 receipts and summary byte-unchanged (pin
   the sha); `--dry-run` enumerating r2 cells; refreshed
   `lead_commands.txt` for r2 (`preflight`, four `run`, `summary` with
   the r2 fingerprint).

## Done (verbatim)
1. r2 registration path + sha; recipe manifest sha; scorer file/lines
   and test names; the sham-floor statement.
2. CPU gate results; exact lead commands.
3. Prior art; deviations; RED; process safety; model id and effort.
