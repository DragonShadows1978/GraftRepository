# GRM-SCOUT-FIX-5 — the fold prompt must enumerate what to preserve (worktree wt/grm-c7, branch grm-c7)

David (2026-09-09): "move forward with the best options to resolve the
issues." Origin: C7 amendment 4 (`artifacts/grm_c7/r2/amendment_4/REPORT.md`):
under FIX-3 the Harmony path runs and digests are coherent, but 35 of
39 digest candidates stop on their own after covering ONE fact, so
every fold fails the 0.70 coverage rule (13/13 rejected); the fold
prompt is the binding constraint, not the 120-token budget. Same rules
as FIX-1..4 (minimal core diff, RED-before/GREEN-after, no behaviour
change beyond the finding, no GPU, no git, no subagents, foreground,
never kill anything). Effort: high.

## Mission
1. **Core fix (minimal):** the consolidation prompt enumerates the
   facts the digest must preserve — the fold already computes
   `need_count` from the sources' fact tokens; render those source
   lines (or the extracted fact spans, whichever the fold already
   holds) as an explicit "keep every one of these N facts" list, and
   size the generation budget as max(120, ~24 tokens × N). Coverage
   rule (0.70) and QC unchanged. The digest must remain a single
   prose/archive block (the existing digest format), not a copy of
   the sources.
2. **Pins:** RED before / GREEN after on the 13 r2 fold cases replayed
   on CPU with a fake model that emits exactly the enumerated facts
   (the point is the prompt and budget, not the model); a control that
   a fold whose sources contain zero fact tokens behaves byte-identically;
   the FIX-3 tests still pass.
3. **GPU contrast (lead-run, registered):** replay the 13 r2 folds from
   their bound states (`artifacts/grm_c7/r2/fix4_attempt_1/cells/*/folds/*/start.json`
   and the source payloads) under the new prompt, one lease per fold
   (≤ 0.3 GPU-h), receipts with coverage / accepted / digest text;
   `lead_commands_fix5.txt`.

## Done (verbatim)
1. Fix file/lines; test names with RED/GREEN evidence; budget formula.
2. GPU contrast registration path + sha; exact lead command.
3. Prior art; deviations; RED; process safety; model id and effort.
