## GRM-H4 delta: the XM1/XM2/XM3 campaigns on the post-D2 tree (2026-09-13)

XM1 (cross-model graft-read parity), XM2 (Qwen final channel) and XM3
(Trinity) were run on branches forked from `lc1-wip` BEFORE the D2 default
flip, and merged at fdd6880, ff5c9f4, aa226e7. Their registrations pin
`core/grm_admission.py`, `core/grm_alias_fold.py`,
`core/grm_fold_alias_guard.py`, `core/grm_fold_retain.py`,
`scripts/grm_xm1_cpu.py`, `scripts/grm_xm1_parity.py`,
`scripts/grm_xm1_x2_run.py` and their own test files byte-for-byte; D2 moved
them, so the registrations fail closed and the tests that drive them are
receipts.

Method as in H1/H2/H3: every failure reproduced in an **isolated
single-module process** and then per-test, marks on test FUNCTIONS, no
assertion changed anywhere, nothing marked that passes.

### 18 failures, but only 14 receipts

The naive tree-wide reading of these six modules was **18 failed**. Four of
those were NOT receipts and are NOT marked. They were this worktree being
under-provisioned with gitignored `artifacts/`, and they went green the
moment the ten pinned files the XM1 registration names were copied in from
the canonical repository (`/mnt/ForgeRealm/GraftRepository`), where all ten
are present and **sha-MATCH the pin**:

```
artifacts/grm_det1/run_20260831T160525Z_2/runtime_frame_28b3196f8fb04a41.json
artifacts/grm_rs3/registration.json
artifacts/grm_rs4/registration.json
artifacts/grm_rs4/grm_rs4_results.json
artifacts/grm_rs4/grm_rs4_C3l_correction_then_restatement_c9a6fbacd986c992.json
artifacts/grm_rs4/grm_rs4_C3l_fresh_fact_controls_f42f8da7fd3a817c.json
artifacts/grm_rs4/grm_rs4_C3l_multi_hop_a_b_c_9f03781dac09b680.json
artifacts/grm_rs4/grm_rs4_C5_correction_then_restatement_4e04252d50ee8338.json
artifacts/grm_rs4/grm_rs4_C5_fresh_fact_controls_488e7894ee4bbd2a.json
artifacts/grm_rs4/grm_rs4_C5_multi_hop_a_b_c_0cb8f693ae114649.json
```

The four that recovered were
`test_grm_xm1_amendment_1.py::test_worker_scratch_without_pytest_basetemp`,
`::test_failed_reference_does_not_block_next_gpt`,
`::test_mismatched_reference_stops_qwen_but_not_gpt` and
`test_grm_xm1_parity.py::test_00_rs4_historical_reassembly_is_float_equal`.

**A missing file is not automatically a receipt.** The artifact-bound class
means the campaign's data is *gone*; a file that is merely absent from the
tree you happen to be standing on, while present and sha-matching at the
canonical repository, is a provisioning gap, and marking it would file a
recoverable file under a label meaning "never coming back". The test that
separates the two is one line: does the pinned digest still resolve
somewhere. This is the same discipline as the GRM-F6 dead-glob rule — an
absence must be proved, not assumed.

### The 14 marks

| Test module | Marks | Node ids | Class | Node id | Reason | Registration |
|---|---:|---:|---|---|---|---|
| `tests/test_grm_xm1_amendment_1.py` | 2 | 2 | sha-bound | `::test_ten_gpu_execute_cells_then_deleted_reference_stops_qwen` | `registration source drift: core/grm_admission.py`; the expected `STOP: GPT-OSS RS4 parity barrier` never arrives because validation fails first | `artifacts/grm_xm1/registration.json + artifacts/grm_xm1/registration_amendment_1.json` |
| | | | sha-bound | `::test_amendment_binds_worker_and_preserves_original_registration` | same drift, raised out of `xm.validate_registration()` | same |
| `tests/test_grm_xm1_parity.py` | 2 | 2 | sha-bound | `::test_registration_and_pins_are_immutable_and_relative` | same drift, raised out of `xm.validate_registration()` | `artifacts/grm_xm1/registration.json` |
| | | | sha-bound | `::test_lead_commands_all_accept_appended_dry_run` | all 60 pinned `--dry-run` commands exit 1 on the same drift | same |
| `tests/test_grm_xm2_final.py` | 2 | 2 | artifact-bound | `::test_recorded_trace_continuation_selects_input_queries` | reads `artifacts/grm_xm2/recorded_trace_cpu.json`, one of the 19 lost files | `artifacts/grm_xm2/registration.json + artifacts/grm_xm2/registration_amendment_1.json` |
| | | | sha-bound | `::test_registration_matrix_budget_and_dry_run` | `registration source drift: core/grm_admission.py` | same |
| `tests/test_grm_xm2_loader.py` | 4 | 4 | artifact-bound | `::test_original_and_current_loader_bodies_equal` | reads `artifacts/grm_xm2/amendment_1/evidence_before.json`, one of the 19 lost files | `artifacts/grm_xm2/registration.json + artifacts/grm_xm2/registration_amendment_1.json` |
| | | | sha-bound | `::test_amendment_caps_and_immutable_originals` | `registration source drift: core/grm_admission.py` | same |
| | | | sha-bound | `::test_amendment_parent_and_source_drift_fail_closed` | same drift; the expected `wrong registration binding` is pre-empted by it | same |
| | | | sha-bound | `::test_wrong_weight_mode_rejected_before_gpu` | same drift; the expected `TC_WEIGHT_BITS=4` is pre-empted by it | same |
| `tests/test_grm_xm3_trinity.py` | 4 | 4 | sha-bound | `::test_registration_dry_run_all_commands_no_gpu` | `X3 source drift: scripts/grm_xm1_cpu.py` (D2 fallout) | `artifacts/grm_xm3/registration.json + artifacts/grm_xm3/implementation_amendment_3.json` |
| | | | sha-bound | `::test_prerequisites_reject_missing_wrong_control_budget_gap` | same X3 drift | same |
| | | | sha-bound | `::test_pin_drift_fails_closed` | same X3 drift | same |
| | | | sha-bound | `::test_gpu_entrypoint_mock_owns_lease_and_reservation` | same X3 drift, raised out of the real `x3.validate()` the test deliberately keeps unmocked | same |
| **5 modules** | **14** | **14** | | | | |

`tests/test_grm_xm1_results.py` is clean on this tree and needs no marks
(2 passed).

### The 19 lost XM2 process files — artifact-bound, not recoverable

These were lost when the XM2 worktree was removed (lead error, ledgered).
Unlike the ten above they are absent from the canonical repository too, so
their pins can never be satisfied again. Recorded here with the sha the XM2
registration chain pins them at, so the loss is a receipt rather than a
silence.

Pinned by `artifacts/grm_xm2/registration.json`:

| sha256 (first 16) | Path |
|---|---|
| `cc48995f1356b885` | `artifacts/grm_xm2/recorded_trace_cpu.json` |

Pinned by `artifacts/grm_xm2/registration_amendment_1.json`:

| sha256 (first 16) | Path |
|---|---|
| `0f08878ceb8e7e1c` | `artifacts/grm_xm2/amendment_1/evidence_before.json` |
| `72bd29ffd194b96e` | `artifacts/grm_xm2/amendment_1/evidence_concurrent.json` |
| `f76a015559ae4e86` | `artifacts/grm_xm2/lead_C3l__sup_harbor_restatement.log` |
| `963fe214332dd8cc` | `artifacts/grm_xm2/lead_C3l__sup_praxis_fresh.log` |
| `481f39dac872163e` | `artifacts/grm_xm2/lead_C3l__sup_reserve_meridian_docket.log` |
| `fc9cbe63a04fbb0c` | `artifacts/grm_xm2/lead_C3l__sup_reserve_tundra_ledger.log` |
| `fc9cbe63a04fbb0c` | `artifacts/grm_xm2/lead_C3l__sup_solace_fresh.log` |
| `fc9cbe63a04fbb0c` | `artifacts/grm_xm2/lead_C3l_pin_live_seat_0__sup_praxis_fresh.log` |
| `fc9cbe63a04fbb0c` | `artifacts/grm_xm2/lead_C3l_pin_live_seat_0__sup_solace_fresh.log` |
| `fc9cbe63a04fbb0c` | `artifacts/grm_xm2/lead_C3l_pin_off_seat_0__sup_praxis_fresh.log` |
| `fc9cbe63a04fbb0c` | `artifacts/grm_xm2/lead_C3l_pin_off_seat_0__sup_solace_fresh.log` |
| `fc9cbe63a04fbb0c` | `artifacts/grm_xm2/lead_C3l_pin_off_seat_1__sup_praxis_fresh.log` |
| `fc9cbe63a04fbb0c` | `artifacts/grm_xm2/lead_C3l_pin_off_seat_1__sup_solace_fresh.log` |
| `481f39dac872163e` | `artifacts/grm_xm2/lead_C5__sup_harbor_restatement.log` |
| `bcdc49a2d5e8e603` | `artifacts/grm_xm2/lead_C5__sup_praxis_fresh.log` |
| `902f5b873ff2537d` | `artifacts/grm_xm2/lead_C5__sup_reserve_meridian_docket.log` |
| `fc9cbe63a04fbb0c` | `artifacts/grm_xm2/lead_C5__sup_reserve_tundra_ledger.log` |
| `fc9cbe63a04fbb0c` | `artifacts/grm_xm2/lead_C5__sup_solace_fresh.log` |

The repeated `fc9cbe63a04fbb0c` is not an error: eleven of the sixteen
first-attempt `lead_*.log` files were byte-identical (the same short
fail-closed message), so they share one digest. Nine other `lead_*.log`
files were recovered byte-identical in a commit after aa226e7 and are
present; they are not listed here.

Only two of the 19 are actually reached by a test on this tree
(`recorded_trace_cpu.json` and `amendment_1/evidence_before.json`); the other
17 are pinned but not read by any currently collected node id. They are
recorded anyway, because the registration still binds them and a later
re-validation would hit them.

### Failures that are NOT receipts

No genuine defect was found in the six XM modules. Every one of the 14 marked
failures resolves to a registration binding that D2 moved, or to one of the
two lost XM2 files; the other four resolved to a provisioning gap and are
unmarked and green.

### Blocker: this worktree is under-provisioned tree-wide

The XM scope is clean, but the FULL suite on `grm-h4` is not, and the cause
is the same one the four unmarked XM failures had — at a much larger scale.
`/mnt/ForgeRealm/wt/grm-h4/artifacts/` holds 755 MB against 120 GB at
`/mnt/ForgeRealm/GraftRepository/artifacts/`, and 69 campaign directories are
short or entirely empty (`grm_det1` 0 of 65307 files, `grm_f1` 292 of 18017,
`grm_x1` 91 of 397, `grm_sc1`/`grm_sc1_1`/`grm_sc1_2`/`grm_sc2`/`grm_rs1`-`rs4`
/`grm_scout_fix4` 0 of theirs, and so on).

Consequence: 313 RED node ids across the three parts, in 41 modules, none of
them XM, all tracing to `FileNotFoundError` / `missing persisted receipt` on
paths that are present and sha-matching at the canonical repository. H3
recorded this same three-part split as 2894 passed / 0 failed / 0 errors on
2026-09-12, so this is a property of the worktree, not a regression of the
code.

This is not markable and was not marked: these are recoverable files, and
marking them would forge exactly the "never coming back" label the section
above warns against. The remedy is provisioning the worktree's gitignored
`artifacts/` from the canonical repository, which is a lead action — a seat
must not move tens of gigabytes of campaign data on its own judgement.

### Gates (GRM-H4, XM scope, post-D2 tree, 2026-09-13)

XM scope, clean:

```
$ python3 -m pytest -q tests/test_grm_xm*.py -p no:cacheprovider --basetemp artifacts/grm_h4/tmp
109 passed, 14 skipped, 2 warnings in 8.18s

$ python3 -m pytest -q -m campaign_receipt tests/test_grm_xm*.py -p no:cacheprovider --basetemp artifacts/grm_h4/tmp
14 failed, 109 deselected, 2 warnings in 5.75s
```

Every one of the 14 marks reproduces as a failure; none is inert.

Full suite, three parts — RED on the provisioning gap described above, NOT
on XM:

```
$ python3 -m pytest -q --basetemp artifacts/grm_h4/tmp $(ls tests/test_grm_*.py | sed -n 1,70p)
111 failed, 1004 passed, 363 skipped, 2 warnings, 11 errors in 24.02s

$ python3 -m pytest -q --basetemp artifacts/grm_h4/tmp $(ls tests/test_grm_*.py | sed -n 71,101p)
62 failed, 478 passed, 111 skipped, 2 warnings in 345.24s (0:05:45)

$ python3 -m pytest -q --basetemp artifacts/grm_h4/tmp $(ls tests/test_grm_*.py | sed -n 102,152p)
54 failed, 1139 passed, 184 skipped, 2 warnings, 75 errors in 223.36s (0:03:43)
```

Zero of those 313 RED node ids is in a `test_grm_xm*` module. The part-3
range is `102,152p` here, not H3's `102,146p`, because the six XM modules
sort after `test_grm_x3_r2.py` and extend the tree to 152 modules.

Tree totals after this pass: **370 marked test functions, 583 node ids,
63 modules** (H3 left 356 / 569 / 58).
