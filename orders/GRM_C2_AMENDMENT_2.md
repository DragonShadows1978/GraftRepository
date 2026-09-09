# GRM-C2 amendment 2 (lead, 2026-09-09) — first GPU cell RED on `invalid persisted capture evidence`; diagnose, fix the harness, register one re-run

Lead-run: `./artifacts/grm_c2/lead_commands.txt` → dry-run OK, then the
first cell `profile-sup-correction_then_restatement` RED after 46 s:
`scripts/grm_c2_amended.py:131-132`
`if cell['phase']=='fresh' and value.get('end_capture',{}).get('valid') is not True: raise ValueError('invalid persisted capture evidence')`.
Receipts: `artifacts/grm_c2/cells/profile-sup-correction_then_restatement/`
(controller.json + whatever the worker wrote). The runner stopped on
RED as registered; charged 46.3 s.

Diagnose from the worker's persisted output what `end_capture` actually
contained (missing key? `valid: False` with a reason? the B3 manifest
`capture` key absent because the fresh-phase save path is not the
ordinary manifest path?). Determine whether the defect is in the C2
harness (your `end_capture` producer / validator) or a real gap in
B3's provenance path for freshly captured repositories. If the latter,
STOP and report it as a SCOUT-FIX-1 finding with the exact file/lines;
do not edit `core/`. If the former, fix it in the C2 scripts only.

Same worktree (r1 + amendment 1 committed), same rules (no git, no
subagents, no GPU, foreground, never kill anything). Effort: high.

## Registered by this amendment
1. Amendment JSON (sha-bound to this order, amendment_lead_1 and the
   registration) recording the diagnosis and the fix.
2. Re-run rule: the RED cell is re-runnable ONCE under this amendment
   (its RED receipt stays; the re-run receipt carries the amendment
   sha); any other RED remains the registered stop. The 46.3 s stays
   charged; total cap unchanged at 3,600 s.
3. CPU gates: a fixture that reproduces the exact `end_capture` shape
   the worker persisted (RED before the fix, GREEN after); forged/stale
   amendment refused; `--resume` starts at the re-runnable cell.
4. Refreshed `lead_commands.txt` (`--resume` form) executable as the
   next lead invocation.

## Done (verbatim)
1. Diagnosis with the persisted `end_capture` value quoted verbatim;
   harness-vs-core ruling; amendment path + sha; files/lines; tests.
2. Exact lead commands.
3. Prior art; deviations; RED; process safety; model id and effort.
