# GRM-C7 amendment 4 (lead, 2026-09-09) — r2 completed under FIX-4; diagnose the two things that cap it: the oracle answering "unknown" with the fact in the prompt, and folds still failing under FIX-3

Lead-run r2 (FIX-4): all 39 cells COMPLETE (`artifacts/grm_c7/r2/fix4_attempt_1/cells/`).
Tally over the 52 probes in cells 4–39: memory exact **4/52** (fresh
4/12, alias 0/12, correction 0/13, folded 0/15), abstentions 30, wrong
values 0; **oracle exact 18/52** (the full-information ceiling is only
35%); residency bounded (max 93 token seats); **folds 13/13 rejected**
(`folded_path_exercised: false`). Memory cannot beat the oracle, so
two diagnoses gate any reading of this campaign. CPU only, receipts
only; no core edits without a STOP-and-report if the cause is core.
Same worktree, same rules. Effort: high.

## Mission
1. **Oracle.** For every oracle row that answered `unknown` / abstained
   with its source present: quote the wrapped prompt and the served
   text; classify the failure (model abstains despite the record; the
   question asks for the CURRENT value after a correction and the
   record list contains both; the answer format expected by the scorer
   vs what the model wrote; reasoning-low truncation; anything else).
   Then a CPU-replayable proposal that does not touch core: e.g. ask
   the question in the same Harmony shape the serving path uses, or
   restate the records as "latest value" lines. Register it as an
   oracle-only change (the oracle is the ceiling, not the product).
2. **Folds.** For the 13 rejected folds: quote the digest text the
   fixed path produced, the coverage and the need_count; say whether
   FIX-3's Harmony path was actually taken (receipt evidence), whether
   the 120-token budget or the coverage rule (0.70 of need_count)
   is the binding constraint, and what the digests miss. If the fold
   prompt is core and wrong for this model, STOP and report with
   file/lines (SCOUT-FIX-5 candidate).
3. **Memory.** For the 30 abstentions: split into (a) admission
   refused (no mount), (b) mounted but the reader abstained, (c)
   FIX-4 served-from-recency rows; for aliases and corrections say
   which of these dominates. No fixes here; this is the finding.

## Done (verbatim)
1. Oracle table (rows, classes, quoted examples) and the proposal;
   fold table and ruling; abstention split.
2. Prior art; deviations; RED; process safety; model id and effort.
