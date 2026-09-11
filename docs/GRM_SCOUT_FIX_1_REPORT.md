# GRM-SCOUT-FIX-1 report — 2026-09-08

**B1/B2/B3/B5 fixed; C1 CPU fixtures GREEN. Ready for lead review/commits.**
The final 23-case fixture set is **11 RED / 12 GREEN on original sources →
23 GREEN after fixes**. Five registered mutations are killed. Existing selected
batteries are **227 GREEN / 11 missing-receipt RED on both original and fixed
sources**, with six additional lifecycle tests GREEN. This is author-run CPU
validation, not blind verification or EB1 model E2E certification.

Worktree `/mnt/ForgeRealm/wt/grm-fix1`, HEAD metadata `refs/heads/grm-fix1`.
[Immutable order](../orders/GRM_SCOUT_FIX_1.md),
[registration](../artifacts/grm_scout_fix1/REGISTRATION.md),
[A1](../artifacts/grm_scout_fix1/AMENDMENT_A1.md),
[append-only ledger](GRM_SCOUT_FIX_1_LEDGER.md).

## 1. Per-item changes and RED-before/GREEN-after

| Item | Production file / changed lines (final file) | Named fixture and evidence |
|---|---|---|
| B1 | [core/grm_demand.py](../core/grm_demand.py#L401), 401–408 | `test_final_flush_only_fire_keeps_completed_answer`, `test_single_token_flush_boundary`: RED `DemandError: observer captured 4 answer rows for ngen=3` / `2 answer rows for ngen=1` → GREEN. Raw final-forward mass/prediction retained in `observer.records`, no abort state, unused row excluded from answer decision; completed text and bookkeeping equal abort OFF. |
| B2 | [core/grm_three_pass.py](../core/grm_three_pass.py#L450), 450–452 and 692–700 | `test_rt1_receipt_projection[demoted0/1]`: RED `KeyError: 'split_child_demoted'` → GREEN. All four RT1 fields reach named admission section and generic info passthrough; additive `admission_` prefix only. Legacy receipt shape and digest fixture pass unchanged. |
| B3 | [core/graft_repository.py](../core/graft_repository.py#L4620), 4620–4625, 4641–4642, 4792–4794 | `test_capture_save_reload_round_trip[live/off/cache]`: RED `KeyError: 'capture'` → GREEN. Optional manifest `capture` mapping restores original top-level capture fields and geometry; payload/text/ntok preserved. `test_old_manifest_without_capture_loads_unchanged` GREEN before/after. |
| B5 | [scripts/grm_wc1_results.py](../scripts/grm_wc1_results.py#L71), 17–21, 71–97, 141–145, 161, 169, 196, 215, 336–342, 417–419, 441 | `test_wc1_token_seats_and_resume_dedup`, `test_wc1_missing_token_count_is_unknown_not_zero_or_partial_mean`, `test_wc1_nested_receipt_fallback`, `test_wc1_print_labels_both_columns`: missing token key/labels RED → GREEN. Legacy graft-count key retained; additive token mean/coverage/note, both columns labeled. |

New tests, each paired with a separate commit-sized patch for the lead:

- B1/C1: [test_grm_scout_fix1_demand.py](../tests/test_grm_scout_fix1_demand.py), [B1.patch](../artifacts/grm_scout_fix1/patches/B1.patch).
- B2/C1: [test_grm_scout_fix1_receipt.py](../tests/test_grm_scout_fix1_receipt.py), [B2.patch](../artifacts/grm_scout_fix1/patches/B2.patch).
- B3/C1: [test_grm_scout_fix1_manifest.py](../tests/test_grm_scout_fix1_manifest.py), [B3.patch](../artifacts/grm_scout_fix1/patches/B3.patch).
- B5/C1: [test_grm_scout_fix1_wc1.py](../tests/test_grm_scout_fix1_wc1.py), [B5.patch](../artifacts/grm_scout_fix1/patches/B5.patch).

C1 controls GREEN on both sources: first-token abort/resume for no stop and stop
on first/second prediction (3); stop at every answer position with abort OFF/ON
(6); last valid answer position still suspends and resumes (1); legacy receipt
(1); legacy manifest (1). Resume assertions check exact text, forward count and
live-segment bookkeeping; a resumed suspension cannot be consumed twice.
The model and attention operands are CPU fakes; observer, attempt, resume,
receipt builder, ordinary repository save and load are production methods.

Receipts: [final RED log](../artifacts/grm_scout_fix1/final_red.log),
[final GREEN log](../artifacts/grm_scout_fix1/final_green.log),
[mutation results](../artifacts/grm_scout_fix1/mutations.json),
[existing original](../artifacts/grm_scout_fix1/existing_before.log),
[existing fixed](../artifacts/grm_scout_fix1/existing.log),
[lifecycle](../artifacts/grm_scout_fix1/lifecycle.log).

Reproduction commands, from this worktree (CPU only):

```sh
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_grm_scout_fix1_*.py
CUDA_VISIBLE_DEVICES='' python artifacts/grm_scout_fix1/run_snapshot_tests.py artifacts/grm_scout_fix1/before -q tests/test_grm_scout_fix1_*.py
CUDA_VISIBLE_DEVICES='' python artifacts/grm_scout_fix1/run_mutations.py
CUDA_VISIBLE_DEVICES='' python artifacts/grm_scout_fix1/regenerate_wc1.py before
CUDA_VISIBLE_DEVICES='' python artifacts/grm_scout_fix1/regenerate_wc1.py after
```

Six extra existing lifecycle tests: `test_dirty_flush_and_reload_lifecycle`,
`test_durability_mode_recovers_from_manifest_and_wal`,
`test_wal_recovers_text_metadata_without_manifest`,
`test_wal_recovery_adopts_orphaned_payload_before_manifest`,
`test_manifest_load_replays_post_checkpoint_forget`,
`test_manifest_load_replays_post_checkpoint_correction`, all in
`tests/test_grm_runtime_lifecycle.py`. Run with `python -m pytest -q` and these
six `file::test_name` node IDs. No existing test was modified.

## 2. WC1 before and after B5

These tables are reassembled from existing canonical `grm_wc1_opus` receipts.
They are historical E2E observations processed on CPU, not new GPU measurements.
Printed graft/token-seat columns use census observations (10 per width).

Before, original misleading label:

```text
 width    sup   census     lh  lh_all  splits   seats   wall_ms  regressions vs 96
----------------------------------------------------------------------------------
    64    8/9    10/10    4/4   14/14      72     1.1   8,633.9  sup:sup_reserve_juniper_pass
    96    9/9     9/10    4/4   13/14       0     0.9   5,936.4  (none)
   128    7/9     9/10    4/4   13/14       0     1.2   6,662.5  sup:sup_reserve_juniper_pass, sup:sup_reserve_tundra_ledger
   192    8/9     9/10    4/4   13/14       0     1.2   5,780.4  sup:sup_orion_current
   256    7/9     8/10    4/4   12/14       0     1.8   7,653.5  sup:sup_lumen_head, sup:sup_orion_current, census:e2e_t09_cypher_bridge
```

After, both columns explicitly labeled:

```text
 width    sup   census     lh  lh_all  splits  grafts token_seats   wall_ms  regressions vs 96
----------------------------------------------------------------------------------------------
    64    8/9    10/10    4/4   14/14      72     1.1        47.1   8,633.9  sup:sup_reserve_juniper_pass
    96    9/9     9/10    4/4   13/14       0     0.9        75.8   5,936.4  (none)
   128    7/9     9/10    4/4   13/14       0     1.2        80.0   6,662.5  sup:sup_reserve_juniper_pass, sup:sup_reserve_tundra_ledger
   192    8/9     9/10    4/4   13/14       0     1.2        80.0   5,780.4  sup:sup_orion_current
   256    7/9     8/10    4/4   12/14       0     1.8       128.0   7,653.5  sup:sup_lumen_head, sup:sup_orion_current, census:e2e_t09_cypher_bridge
```

**Reading:** width 96 averages 0.9 fitted grafts but 75.8 mounted token seats.
Width 64 fits more, smaller grafts (1.1; 47.1 token seats). Width 256 averages
1.8 grafts and 128.0 token seats. This is a residency description, not evidence
that chunking alone caused recall changes. Sink and live rows are excluded.
Recall, regressions, split counts, predictions and timing values remain exactly
the same. Wall time still includes historical lock contention.

[Before JSON](../artifacts/grm_scout_fix1/wc1_before.json),
[after JSON](../artifacts/grm_scout_fix1/wc1_after.json),
[comparison checks](../artifacts/grm_scout_fix1/comparison_checks.json),
[120/120 manifest token-sum checks](../artifacts/grm_scout_fix1/wc1_sum_crosscheck.json).
Long-horizon token-seat means at widths 64/96/128/192/256 are respectively
46.4286 / 78.6429 / 81.6429 / 81.6429 / 141.0714, over 14 observations each.
Sup token-seat means are null: its nine per-probe receipts lack token sums.
The reader must not interpret null as zero.

## 3. Findings beyond the Scout report

- Flush-only failure also reproduces at the smallest positive budget, `ngen=1`.
- WC1's existing means exclude logged turns without a `mount_fitted` decision,
  including many filler turns. Both old and new means preserve this denominator.
- Supersession receipts do not retain the token measurements needed for its new
  column; token means are available for census and long-horizon only. All 120
  available recorded sums equal an independent sum over persisted manifest ntok.
- This worktree lacks calibration artifacts needed by 11 existing tests. Both
  source versions fail on the same node IDs; this is a dependency failure, not
  evidence of a new regression. Example verbatim:
  `CalibrationError: missing persisted receipt: /mnt/ForgeRealm/wt/grm-fix1/artifacts/grm_det1/run_20260831T160525Z_2/det1_4/campaign/calibration/thresholds_2149a44b6136fd1b.json`.

## Prior art

Production repairs reuse local **GRM-SC2/SC1**, **LSR-P2B/RT1**, **RS3**, and
**WC1**, project contributors (2026), verified in the source sites cited above.
Taken: answer-position/flush convention, prefix-based additive receipts, capture
metadata and ordinary manifests, serving-time sum of mounted ntok, and existing
turn deduplication/means. Ours: boundary guard, missing field plumbing and accurate
report columns. No new algorithm or novelty claim; **no prior art known to me**
for these specific repairs beyond those local systems.

Tests reuse SC2 `_ResumeArena` and the repository lifecycle `FakeArena` (project,
2026). Fault reintroduction follows DeMillo, Lipton and Sayward (1978),
[*Hints on Test Data Selection*](https://doi.org/10.1109/C-M.1978.218136).
The indexed primary-paper excerpt describes program mutation; full-paper/DOI
fetch failed, so full-text verification is **unverified — lead to check**
(search: DeMillo Lipton Sayward 1978 Hints on Test Data Selection).
The source-overlay harness uses documented Python importlib hooks (Python project,
docs accessed 2026); only the selected source overlays are task-specific.
[Python importlib documentation](https://docs.python.org/3/library/importlib.html).
Prior-art annotations also appear at code sites and in the ledger.

## 4. Deviations, RED, process safety, identity

- Initial receipt fixtures omitted required session/turn IDs; corrected before
  production changes, with the original failed run retained in `red_initial.log`.
  No assertion or tolerance was weakened. A1 adds the opposite B1 boundary control
  in a separate registration amendment; the initial registration is unchanged.
- WC1 receipts are read from canonical checkout because this worktree lacks them;
  all regenerated files are local sidecars. Input SHA-256 inventories match before
  and after. Four patches are ready for the lead; no commits were made.
- **Remaining RED:** 11 existing missing-receipt tests, not claimed fixed. Sup
  token-seat measurement unavailable. Independent blind verification and the
  lead's later EB1 E2E gate are not run or claimed. Author-run 5/5 mutation kills
  establish sensitivity only to those five faults, not broad correctness.
- No GPU runs, git commands, subagents, shell background jobs/waits, kills, service
  changes, external delivery, B4 edits, default/calibration changes or payload
  format changes. One foreground pytest tool yield was collected normally.
  Order and registration hashes verified unchanged.
- Model identity supplied by system: **Codex / GPT-6**; exact API serving model ID
  is not exposed. Reasoning effort requested: **high**. No runtime setting change
  is claimed.
