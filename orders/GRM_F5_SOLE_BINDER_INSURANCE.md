# GRM-F5 — sole-binder insurance in the admission plan (lead order, round 2, 2026-09-11)

Seat: Opus 5 (opus-max). YOUR WRITABLE TARGET is `/mnt/ForgeRealm/wt/grm-f5`
(branch `grm-f5`, forked from `lc1-wip` after F3). `core/grm_admission.py`
edits AUTHORIZED under the flag below only. Read-only: canonical
`/mnt/ForgeRealm/GraftRepository`, `docs/GRM_F3_ROUTING_AT_DISTANCE_LEDGER.md`,
`tests/test_grm_f3_routing_at_distance.py` (the DEFECT-PINNED tests),
`artifacts/grm_f3/route_rows.json`. No git, no subagents, no GPU,
foreground, Bash calls < 10 min, never kill anything; `--basetemp` under
`artifacts/grm_f5/tmp` with cleanup; repo-relative pins only.

## The proven defect (F3, receipts)
`core/grm_admission.py:99-117` (`margin_first_plan`, branch
`margin_insurance_k3_identifier_tiebreak`) reorders only within rank-1's
exact-score tie group and returns `ranking[:3]`; `:360-393` (`policy_plan`,
branch `one_off_rank_identifier_insurance_k3`) fires when there is exactly
one off-rank binder and also returns `ranking[:3]`. With ≥ 2 hits,
`declared_synthesis_identified_set` admits every identified candidate
including unranked ones; with exactly one hit it admits none. LT1.1 r2 A+
`recall_3_150`: `identified_candidates=[64]`, margin 0.0077, plan
[168,132,116], node 64 (the intact Breakwater record) never mounted.

## Mission
1. Flag `GRM_ROUTE_SOLE_BINDER_INSURANCE` (default OFF; OFF byte-identical
   — C2 132-plan replay gate, FIX-4/FIX-6 suites, F3 characterization
   suite's pinned-defect tests still pass with the flag OFF). ON: when
   the identifier scan yields exactly one binder, that binder is in the
   plan — reorder-never-score (RT1's stance): substitute it for the last
   plan slot if it is not already present; record `plan_branch` as a new
   name and `sole_binder_inserted: true` in the route receipt; both
   `margin_first_plan` and `policy_plan` paths.
2. Fixtures: the r2 row above from the frozen ranking/margin/scan
   (RED under OFF = plan [168,132,116]; GREEN under ON = 64 in the plan);
   a control where the sole binder is already rank-1 (plan unchanged);
   a control with ≥ 2 binders (existing rule, unchanged); the
   DEFECT-PINNED F3 tests: add ON-arm counterparts asserting the fix
   rather than editing the pinned ones.
3. Register the flag as a named optional on LT1.1 r3 (write
   `artifacts/grm_f5/flag_contract.json`, same shape as F2's).
4. Ledger + REPORT; prior art (RT1 reorder-never-score; MMR as F3 cited;
   unverified marks); `python3 -m pytest -q --basetemp …
   tests/test_grm_f5*.py tests/test_grm_f3_routing_at_distance.py
   tests/test_grm_admission.py tests/test_grm_scout_fix4.py
   tests/test_grm_scout_fix6*.py` verbatim last.

## Done (verbatim)
1. Core diff (file:lines), flag, OFF byte-identity lines (incl. C2 132/132).
2. Fixture RED→GREEN + controls; flag contract path.
3. Prior art, deviations, RED, process safety, model + effort; pytest LAST.
