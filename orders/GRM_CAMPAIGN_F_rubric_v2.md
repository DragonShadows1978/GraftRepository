# CAMPAIGN-F: S2 rubric="v2" (preference-aware wording) — registered arm 7c

YOUR WRITABLE TARGETS in /mnt/ForgeRealm/GraftRepository (main tree):
- core/graft_repository.py
- tests/test_grm_importance_salience.py
Everything else READ-ONLY (sibling seat works tests/
test_grm_supersession_battery.py — never touch it, nor
core/graft_arena.py). No git. No subagents. No GPU / no model loads.
RED honesty; no monitor-idling.

## Law
docs/GRM_IMPORTANCE_LEDGER.md — read in full; the 2026-07-17 "S2-v2
REGISTERED" entry IS the spec. v1's frozen rubric + machinery
(primed prefix, strict 0-3 parse, one retry, None on double failure,
recall-kind exclusion, arena snapshot/restore) stay byte-identical;
the ONLY delta is the rubric text, selectable via rubric="v2"
(default "v1" unchanged).

## Build
1. S2_SALIENCE_PROMPT_V2 module constant: reword the 0-3 scale so
   preferences, standing directives, and instructions are explicitly
   keep-worthy classes alongside facts (v1's fact-only wording is the
   registered suspect for the standing-pref 0.0 collapse). Same
   primed-prefix shape, same answer format. Mark FROZEN pre-gate in a
   comment. Report the exact wording verbatim.
2. rubric="v1"|"v2" plumbed through GraftRepository construction to
   the scoring pass; validation like other policy strings.
3. CPU units: v2 selected → v2 prompt used; default v1 byte-identical
   (prove with the existing mocked-generation tests parameterized);
   parse/retry/None behavior identical under v2.

## Verify
py_compile; python3 -m pytest tests/test_grm_importance_salience.py tests/test_grm_s4_fold_order.py tests/test_grm_fold_recovered_guard.py -q — paste summary.

## Done
Verbatim: the frozen v2 rubric text; diff summary; pytest line;
ambiguities; "no git, no GPU, no files outside grant".
