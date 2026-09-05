# GRM-RT1.1 — Restore the RT1 receipts the lead lost

Lead-authored 2026-09-05 ~03:00. The lead removed the RT1 worktree with
`--force` after merging it (7c0e738 → 29d860b) and the gitignored
`artifacts/grm_rt1/` and the RT1 rule-ON sup receipts under the
worktree's `artifacts/lsr_p2c/` went with it. `scripts/grm_rt1_g2_table.py`
(merged) references those receipts by exact content-hash filename
(`ARMS['rule_on']['receipts']`), so it now reports "receipt missing on
disk", and Astra's WC1 harness marks its sup G2 row BLOCKED_REFERENCE
for the same reason. This order regenerates the receipts on the same
levers, re-points the table at them, and re-produces RT1's G2/G3
tables. Restoration only; no rule change.

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` (branch
`lc1-wip`, HEAD 8ee1301) — scripts (named below), BOUNDED GPU runs, and
`artifacts/grm_rt1/`, `artifacts/lsr_p2c/`, `logs/` AUTHORIZED. A
registered order IS the permission. `core/` and `config/` READ-ONLY.
Another lead-driven process may hold the GPU lock; wait on it.

## Context

1. `orders/GRM_RT1_SPLIT_CHILD_ROUTING.md`; the RT1 seat's report
   numbers (recorded in the 7c0e738 commit message and the board):
   G2 sup rule OFF 8/9 (solace miss), rule ON 9/9; solace ranking
   before `[2,3,4,1,0]` plan `[2,3,4]` → after `[1,0,2,3,4]` plan `[1]`;
   G3 census 9/10, 0 regressions, t33 untouched.
2. `scripts/grm_rt1_g2_table.py` (`ARMS` dict, rule_off / rule_on
   receipts, `--out`), `scripts/grm_rt1_diagnose.py`,
   `scripts/lsr_p2c_replay_gpu.py` (sup battery, arm 1, `capture_pin`,
   `seat_near_live`, receipts under `artifacts/lsr_p2c/` named by
   content hash), `core/grm_admission.py::rt1_enabled` and the
   measurement-only `GRM_RT1_RULE` switch (amendment A1 of RT1: OFF arm
   = `GRM_LSR_FIXES=1 GRM_RT1_RULE=0`; ON arm = rule via the family
   switch), `core/grm_frame.py`.
3. Astra's reference consumer: `/mnt/ForgeRealm/GraftRepository-wt-wc1-astra/scripts/grm_wc1_results.py::references`
   (reads `ARMS['rule_on']['receipts']` from the main checkout) —
   read-only; it must find the receipts after this order without
   edits.

## Mission

1. Re-serve the sup battery (4 families) under the RT1 G2 conditions,
   both arms: rule OFF (`GRM_RT1_RULE=0`) and rule ON, with
   `GRM_LSR_FIXES=1`, `capture_pin=live`, `seat_near_live=1`, spec
   frame, demand OFF. 8 leased runs, bounded.
2. Update `ARMS['rule_off'|'rule_on']['receipts']` in
   `scripts/grm_rt1_g2_table.py` to the regenerated filenames (or make
   the table resolve receipts by a receipt-embedded `rs3_levers` +
   `rt1_rule` marker and glob, if that is cleaner — say which). Re-run
   the table → `artifacts/grm_rt1/grm_rt1_g2_results.json`.
3. Re-run `scripts/grm_rt1_diagnose.py` → `artifacts/grm_rt1/grm_rt1_diagnosis.json`
   (CPU). Write a fresh `artifacts/grm_rt1/registration.json`
   (immutable; state that it is a RESTORATION registration citing the
   original's sha256 `afaa65bb5567c3ace857334ebec2b31f9ad24bdc0531641937cdd1be6dab9b97`
   as reported by the RT1 seat, and this order).
4. Verify: `cd /mnt/ForgeRealm/GraftRepository-wt-wc1-astra && python3
   scripts/grm_wc1_results.py` (read-only worktree; PYTHONPATH as in
   its scripts) now reports `missing_rt1_references: []` and the sup
   G2 row not BLOCKED_REFERENCE. Do not edit anything in that worktree.

## Gates

Register first (`artifacts/grm_rt1/restoration_registration.json`,
immutable): the 8 runs, levers, prediction = OFF 8/9 with solace the
only miss, ON 9/9, served text semantically equal to the RT1 seat's
table. G1: `pytest -q tests/test_grm_rt1_split_child_routing.py
tests/test_grm_admission.py` green. G2: the regenerated table matches
the prediction. G3: the Astra consumer resolves.

GPU conventions: `/tmp/forge-gpu.lock` via `gpu_lease`, single GPU,
≤10 min per run, ≥30 s gaps. NO git. NO subagents. NO background waits
(foreground, explicit `timeout`, every call <10 min). Never kill a
process you did not start. Frozen run tree never written. RED honesty.

## Done

1. pytest line. 2. Restoration registration + sha. 3. G2 table OFF/ON
(9 probes), diagnosis summary, G3 Astra-consumer output.
4. Files modified (exact `ARMS` change). 5. Deviations/risks/RED;
process safety.
