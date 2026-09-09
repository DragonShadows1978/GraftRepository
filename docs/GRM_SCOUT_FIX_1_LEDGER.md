# GRM-SCOUT-FIX-1 ledger

## Registration and inspection
Read HOUSE_RULES.md, AGENTS.md, immutable order and Scout report. Confirmed cwd
and grm-fix1 HEAD by reading worktree metadata (no git command). Frozen
registration and original-source hashes: artifacts/grm_scout_fix1/.
Static evidence: B1 marks final unused prediction as aborted; B2 omits admission
prefix and RT1 projection; B3 omits capture fields; B5 measures mount-list length.
Prior art: local GRM-SC2 / LSR-P2B / RS3 / WC1, project contributors (2026).
Taken: existing abort/resume, additive receipt, manifest, measured sum conventions.
Ours: boundary/projection/report repairs, no algorithm novelty. External prior
art for these specific repairs: no prior art known to me.

## Fixture setup correction (before production edits)
Initial run: 12 failed / 10 passed. Three receipt cases had a fixture TypeError
(missing required session_id/turn_id); corrected arguments, no assertions changed.
Retained red_initial.log. This is harness failure, not B2 evidence.

## RED baseline (unit test evidence)
Command: CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_grm_scout_fix1_*.py
Receipt: artifacts/grm_scout_fix1/red.log. 11 failed, 11 passed.
Exact failures: `DemandError: observer captured 4 answer rows for ngen=3`,
`DemandError: observer captured 2 answer rows for ngen=1`; `KeyError: 'capture'`
(live/off/cache); `KeyError: 'split_child_demoted'` (true/false);
`KeyError: 'mean_resident_token_seats_per_turn'` (three WC1 cases); missing
'grafts'/'token_seats' print labels. Controls already GREEN: first-token
abort/resume (including stop on recovered first/second prediction), stop at each
answer position with abort OFF/ON, legacy receipt, legacy manifest.

## Repairs and treatment effect (unit test evidence)
B1 core/grm_demand.py: restrict suspension to index < ngen. The raw unused flush
mass/prediction stays in observer.records; finish drops it and decision does not
fire. No DemandAbort, stale aborted_at, or DemandError from a flush-only fire.
B2 core/grm_three_pass.py: additive admission_ passthrough plus four RT1 fields
in admission. No new keys on legacy admissions; existing schema/digest rule stays.
B3 core/graft_repository.py: optional capture mapping in manifest, restoring
original top-level capture_* and n_sink/arena_width/live_shift fields on reload.
B5 scripts/grm_wc1_results.py: retain legacy mean_resident_seats_per_turn as
GRAFT count, add measured token sum mean and coverage, label both printed columns.
Prior art at each site: local GRM-SC2 / LSR-P2B / RT1 / RS3 / WC1 (project, 2026).
Taken: existing contracts; ours: small boundary/projection/report fixes.
No new algorithm; no prior art known to me for the specific repairs beyond these.

Same command after fixes: green.log, 22 passed. No threshold or defaults changed.

## WC1 before/after (existing E2E receipts, CPU reporting only)
Command: CUDA_VISIBLE_DEVICES='' python artifacts/grm_scout_fix1/regenerate_wc1.py
before / after. Rebind input root to canonical grm_wc1_opus and outputs to sidecar;
assembler/scoring code otherwise unchanged. Output wc1_before.json/.txt and
wc1_after.json/.txt; exact inputs hashed in wc1_*_input_sha256.json.
Census legacy graft counts 1.1, 0.9, 1.2, 1.2, 1.8 at widths 64,96,128,192,256;
actual measured token sums average 47.1,75.8,80.0,80.0,128.0 respectively.
Reading: 0.9 was never sub-token residency. Smaller width fits more small grafts
at 64 while using fewer token seats than 96; this is descriptive, not a causal
chunking result. Recall, regressions, predictions, timing and split counts are
exactly unchanged (comparison_checks.json). Wall time retains contention caveat.
Coverage finding beyond Scout: the old mean uses only turns with mount_fitted:
10 census and 14 long-horizon observations per width, not every logged filler
turn. New mean uses precisely the same denominator. Sup per-probe receipts lack
cur_mount_n and persisted per-node ntok, so new sup mean is null (0 measured);
no reconstruction or GPU remeasurement. Counts exclude physical sink/live rows.

## Existing batteries (suite evidence)
Command: CUDA_VISIBLE_DEVICES='' python -m pytest -q
 tests/test_grm_lsr_p2b_route_receipt.py tests/test_grm_three_pass.py
 tests/test_grm_sc1_demand_loop.py tests/test_grm_sc2_calibration_early_abort.py
 tests/test_grm_rs3_capture_pin_seat.py tests/test_grm_wc1_sweep.py
After: existing.log, 227 passed / 11 failed. Original-source replay via
run_snapshot_tests.py before: existing_before.log, same 227 / 11, same failed
nodeids. All failures are missing persisted calibration receipts in this worktree;
not claimed fixed, not a clean whole-suite gate. Example verbatim:
`CalibrationError: missing persisted receipt: /mnt/ForgeRealm/wt/grm-fix1/artifacts/grm_det1/run_20260831T160525Z_2/det1_4/campaign/calibration/thresholds_2149a44b6136fd1b.json`.
Existing tests and assertions were not edited, skipped, or weakened.
Six additional ordinary lifecycle tests passed (lifecycle.log): dirty flush/reload,
durability recovery, WAL without manifest, orphan payload adoption, manifest
post-checkpoint forget and correction. Test commands are captured in final report.

## Registered mutations (author baseline, not blind verification)
Command: CUDA_VISIBLE_DEVICES='' python artifacts/grm_scout_fix1/run_mutations.py
Five disposable overlays reintroduce B1 missing boundary, B2 missing prefix,
B3 missing serialization, B3 missing reload, B5 graft-count-as-token-count.
All five killed by assertions / intended DemandError or KeyError; no collection
errors; 5/5 = 1.0 exceeds registered >=0.80 rail. mutations.json and individual
mutation_*.log. Production hashes unchanged throughout; no production restoration
needed because mutations existed only in overlays. Prior art: DeMillo, Lipton,
Sayward (1978), Hints on Test Data Selection, DOI 10.1109/C-M.1978.218136.
Indexed primary-paper excerpt supports mutation testing; full-paper/DOI fetch
failed, so full-text verification remains a lead. Source-overlay harness uses
Python importlib (Python project docs accessed 2026), verified at
https://docs.python.org/3/library/importlib.html. Ours: five chosen faults, not
mutation testing or Python importing.

## Process and scope
No GPU computation, git commands, subagents, shell background jobs, process kills,
live services, or external delivery. Tests ran sequentially, CUDA_VISIBLE_DEVICES
empty; model/attention seams CPU-only. No B4/grounding, default, calibration,
threshold or payload-format changes. Order and registration immutable; lead
commits. A brief tool yield during the first foreground pytest was resumed only
to collect its exit status; no background wait job was launched.
Deviations: fixture argument correction recorded above; missing receipt dependencies
remain RED; WC1 replay uses canonical read-only receipts because local originals
are absent. Independent blind verification and EB1 E2E remain lead-run gates.
Identity: Codex / GPT-6 per supplied system identity; exact API model ID not
exposed. Requested effort high; no claim of changing the runtime effort setting.

## Final boundary control and independent sum cross-check
A1 separately registered a last-valid-answer-position control (AMENDMENT_A1.md).
It is GREEN before and after, preserving real observer suspension at ngen-1.
Final identical 23-case suite against original snapshots: final_red.log,
11 failed / 12 passed. Against fixed source: final_green.log, 23 passed.
Original 22-case logs remain unchanged. No production change since mutations.
Independent WC1 arithmetic: summed manifest ntok for each recorded mount_fitted
list; all 120 observations (50 census, 70 long-horizon) equal recorded cur_mount_n.
Receipt wc1_sum_crosscheck.json. Prior art: existing ArenaCache ntok sum (2026),
reused as a separate reporting check, no new estimator.

## Handoff
Four separate patch files (B1/B2/B3/B5) each include its production diff and
new fixture file; no git operation. C1 spans those four test files. Final report:
docs/GRM_SCOUT_FIX_1_REPORT.md. Order/registration hash checks passed.

Final hygiene check: report links resolve; order/registration hashes match.
Only the four intended production files and four new fixture files have Python
source modification times after registration. final_sha256.json inventories
production sources, tests, report, ledger, patches and receipts.
