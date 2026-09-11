# GRM-SCOUT-FIX-6 + LT1 run — margin-first admission as a flag-gated mode; run the 200-turn conversation under it (worktree wt/grm-lt1, branch grm-lt1)

David (2026-09-09): "move forward with the best options to resolve the
issues." The lead's choice from your amendment-1 table: **Rule 2**
(rank-1 margin first, identifier as tie-breaker): admits 35/35 LT1
natural questions with the smallest historical plan-change footprint
(EB1 7, C2 31, WC1 24). Same rules as FIX-1..4 (minimal core diff,
RED-before/GREEN-after, no default change: the flag is OFF, today's
rule stays the default; no GPU, no git, no subagents, foreground,
never kill anything). Effort: high.

## Mission
1. **Core, flag-gated:** `GRM_ADMISSION_RULE=margin_first` (default
   unset = today's all-tokens-bind rule, byte-identical) selects Rule 2
   exactly as you implemented it offline in `scripts/grm_lt1_admission.py`,
   inside core admission (`core/grm_admission.py`), with the rule
   recorded in every route receipt (additive key). The ladder and
   any duplicate site call the same core decision.
2. **Pins:** flag OFF → C2's 132 recorded plans byte-identical (your
   replay); flag ON → LT1 35/35 admitted on the CPU arena, and the 31
   C2 plan changes enumerated by question id (the registered regression
   set to measure on GPU); FIX-1..4 fixtures pass.
3. **LT1 registration amendment:** both arms (profile / shipped
   defaults) run with the flag ON, admission rule recorded per receipt;
   cap 3.0 GPU-h; `lead_commands.txt` executable/resumable; free-space
   preflight ≥ 20 GB.
4. **C2 regression set:** register (do not run) the 31 changed-plan
   questions as a replay cell list for the grm-c2 harness (ids + the
   recorded states), so the lead can measure them after cherry-picking
   FIX-6 there.

## Done (verbatim)
1. Fix file/lines, flag, tests with RED/GREEN evidence; the 31-question
   regression list.
2. LT1 amendment path + sha; exact lead command.
3. Prior art; deviations; RED; process safety; model id and effort.
