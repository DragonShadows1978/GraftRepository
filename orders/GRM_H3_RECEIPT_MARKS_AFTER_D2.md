# GRM-H3 — campaign-receipt marks after the D2 default flip (lead order, 2026-09-12)

Seat: Opus 5 (opus-max). YOUR WRITABLE TARGET is `/mnt/ForgeRealm/wt/grm-h3`
(branch `grm-h3`, forked from `lc1-wip` 919ad54). `tests/`, `docs/TESTS_CAMPAIGN_RECEIPTS.md`,
`artifacts/grm_h3/` writable; nothing else. No git, no subagents, no GPU,
foreground, Bash calls < 10 min (split suite runs), never kill anything
(another seat may be working in `/mnt/ForgeRealm/wt/grm-xm1` — never touch it);
`--basetemp` under `artifacts/grm_h3/tmp` with cleanup.

## Context
D2 shipped the round-2 defaults and moved `core/grm_admission.py`,
`core/grm_alias_fold.py`, `core/grm_fold_alias_guard.py`,
`core/grm_fold_retain.py`, `scripts/grm_lt1_1.py`. The LT1.1 r3 chain
(registration + amendments 1–13) pins those byte-for-byte: it is a
COMPLETED campaign, so its tests are receipts now (lead ruling: mark, do
not rebind). D2's three-part suite: 38 failures, all receipt class —
16 in `tests/test_grm_f1_r3_receipts.py` (reads gitignored r3 checkpoint
trees absent from fresh worktrees), 22 sha-binding failures across
`test_grm_lt1_1_preflight.py` (11), `test_grm_f1_resume_out_root.py`
(3), `test_grm_lt1_1_child_spawn.py` (2), `test_grm_d1_amendment1.py`
(2, pre-existing lead_commands drift), `test_grm_d1_amendment3.py` (1),
`test_grm_f1_fold_retain.py::test_governing_amendment_rebinds_every_core_pin`
(1), `test_grm_a1_gpu_contrast.py::test_amendment_chain_is_sha_bound…`
(1), `test_grm_r1_amendment2.py` (1). Rules and marker: `tests/conftest.py`,
`docs/TESTS_CAMPAIGN_RECEIPTS.md` (H1/H2/F6 sections).

## Mission
1. Reproduce each failing test in isolation; mark per function with
   `campaign_receipt(registration=…)` those that are sha-bound to the r3
   chain or read absent campaign artifacts; do NOT mark a genuine defect
   (report it). Tests whose assertion is about the CURRENT tree's behaviour
   (not a receipt) get fixed only if the fix is a pin/fixture, never an
   assertion change — list any.
2. Gates: the three-part GRM suite → 0 failed / 0 errors, N skipped
   (report each part verbatim); `-m campaign_receipt` count; the docs
   table + `artifacts/grm_h3/marked_*` updated; the D2 note "an inherited
   default is not a controlled arm" added to the docs rules.

## Done (verbatim)
1. Delta table (marks added, module, reason, registration); genuine defects (if any).
2. The three part lines + the -m line verbatim.
3. Prior art (unchanged), process safety, model + effort; pytest LAST.
