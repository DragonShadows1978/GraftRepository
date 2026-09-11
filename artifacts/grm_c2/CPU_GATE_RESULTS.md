# GRM-C2 full existing CPU gate receipts

Aggregate across 66 existing files: {'errors': 20, 'failed': 58, 'passed': 1480, 'skipped': 148}; no timeouts.

Every line below is copied from its actual raw pytest output. Skips and errors are not passes.

| File | Original pytest summary | Log |
|---|---|---|
| tests/test_apamq_fc_ppl.py | 1 failed, 5 passed in 0.22s | [000.log](cpu/000.log) |
| tests/test_deepseek_grm_hooks_static.py | 4 passed, 2 warnings in 0.25s | [001.log](cpu/001.log) |
| tests/test_gpt_oss20b_e14_backend_fix.py | 3 passed in 0.25s | [002.log](cpu/002.log) |
| tests/test_gpt_oss20b_grm_output_eval.py | 11 passed in 0.06s | [003.log](cpu/003.log) |
| tests/test_gpt_oss20b_realtext_ppl_gate.py | 4 passed in 0.04s | [004.log](cpu/004.log) |
| tests/test_graft_quant_format.py | 5 failed, 17 passed, 2 warnings in 0.35s | [005.log](cpu/005.log) |
| tests/test_grm_adm1_3_dual_frame.py | 9 passed in 0.16s | [006.log](cpu/006.log) |
| tests/test_grm_adm1_probe_adjudication.py | 1 failed, 1 passed in 0.12s | [007.log](cpu/007.log) |
| tests/test_grm_adm1_snapshot.py | 1 passed, 2 warnings in 0.20s | [008.log](cpu/008.log) |
| tests/test_grm_admission.py | 11 passed, 2 warnings in 0.19s | [009.log](cpu/009.log) |
| tests/test_grm_arena_step_rollback.py | 2 passed, 2 warnings in 0.17s | [010.log](cpu/010.log) |
| tests/test_grm_det1_11_achieved.py | 41 passed in 0.16s | [011.log](cpu/011.log) |
| tests/test_grm_det1_11_census.py | 10 failed, 8 passed in 0.61s | [012.log](cpu/012.log) |
| tests/test_grm_det1_11_envelope_lineage.py | 4 failed, 19 passed, 2 skipped in 0.17s | [013.log](cpu/013.log) |
| tests/test_grm_det1_3_snapshot.py | 39 passed in 1.12s | [014.log](cpu/014.log) |
| tests/test_grm_det1_5_campaign.py | 63 passed in 0.20s | [015.log](cpu/015.log) |
| tests/test_grm_det1_7_registry.py | 23 passed in 0.37s | [016.log](cpu/016.log) |
| tests/test_grm_det1_7_source_auth.py | 6 passed in 0.14s | [017.log](cpu/017.log) |
| tests/test_grm_det1_8_source_auth.py | 6 passed in 0.23s | [018.log](cpu/018.log) |
| tests/test_grm_det1_9_findings.py | 5 passed in 0.20s | [019.log](cpu/019.log) |
| tests/test_grm_det1_9_source_auth.py | 7 passed in 0.31s | [020.log](cpu/020.log) |
| tests/test_grm_det1_baseline_registry.py | 7 failed, 1 passed in 0.29s | [021.log](cpu/021.log) |
| tests/test_grm_eb1_ephemeral_frame.py | 44 passed, 2 warnings in 4.74s | [022.log](cpu/022.log) |
| tests/test_grm_ensure_h_error.py | 3 passed, 2 warnings in 0.19s | [023.log](cpu/023.log) |
| tests/test_grm_fold_recovered_guard.py | 11 passed, 2 warnings in 0.31s | [024.log](cpu/024.log) |
| tests/test_grm_importance_counterfactual.py | 15 passed in 0.16s | [025.log](cpu/025.log) |
| tests/test_grm_importance_g1g2.py | 56 passed in 0.25s | [026.log](cpu/026.log) |
| tests/test_grm_importance_salience.py | 29 passed, 2 warnings in 0.25s | [027.log](cpu/027.log) |
| tests/test_grm_importance_telemetry.py | 5 failed, 7 passed, 2 warnings in 0.40s | [028.log](cpu/028.log) |
| tests/test_grm_lsr_p2a_fit_honesty.py | 20 passed, 2 warnings in 0.19s | [029.log](cpu/029.log) |
| tests/test_grm_lsr_p2a_probe_path.py | 9 passed, 2 warnings in 4.23s | [030.log](cpu/030.log) |
| tests/test_grm_lsr_p2b_e2e_fixture.py | 5 passed, 2 warnings in 4.29s | [031.log](cpu/031.log) |
| tests/test_grm_lsr_p2b_route_receipt.py | 14 passed, 2 warnings in 0.26s | [032.log](cpu/032.log) |
| tests/test_grm_lsr_p2c_split_descent.py | 39 passed, 2 warnings in 4.86s | [033.log](cpu/033.log) |
| tests/test_grm_native_runtime.py | 121 passed, 2 warnings in 280.19s (0:04:40) | [034.log](cpu/034.log) |
| tests/test_grm_probe_ladder.py | 15 passed in 0.05s | [035.log](cpu/035.log) |
| tests/test_grm_route_seams_gate.py | 2 passed in 0.04s | [036.log](cpu/036.log) |
| tests/test_grm_router_baseline.py | 21 passed in 84.11s (0:01:24) | [037.log](cpu/037.log) |
| tests/test_grm_rs1_read_strength.py | 34 skipped in 0.15s | [038.log](cpu/038.log) |
| tests/test_grm_rs2_mount_read.py | 32 skipped in 0.21s | [039.log](cpu/039.log) |
| tests/test_grm_rs3_capture_pin_seat.py | 64 passed, 2 warnings in 0.28s | [040.log](cpu/040.log) |
| tests/test_grm_rs3_tables.py | 24 skipped in 0.16s | [041.log](cpu/041.log) |
| tests/test_grm_rs4_baseline_and_gates.py | 2 failed, 18 passed, 5 skipped in 0.26s | [042.log](cpu/042.log) |
| tests/test_grm_rs4_row_split.py | 27 passed, 5 skipped in 0.17s | [043.log](cpu/043.log) |
| tests/test_grm_rt1_split_child_routing.py | 38 passed, 2 warnings in 0.25s | [044.log](cpu/044.log) |
| tests/test_grm_runtime_lifecycle.py | 101 passed, 2 warnings in 133.05s (0:02:13) | [045.log](cpu/045.log) |
| tests/test_grm_s4_demotion.py | 16 passed, 2 warnings in 0.21s | [046.log](cpu/046.log) |
| tests/test_grm_s4_fold_order.py | 15 passed, 2 warnings in 0.23s | [047.log](cpu/047.log) |
| tests/test_grm_s4_ledger.py | 23 passed, 2 warnings in 0.29s | [048.log](cpu/048.log) |
| tests/test_grm_sc1_1_grounding_glyphs.py | 164 passed, 2 warnings in 4.82s | [049.log](cpu/049.log) |
| tests/test_grm_sc1_2_e2e_pairs.py | 12 failed, 15 passed, 20 errors in 1.59s | [050.log](cpu/050.log) |
| tests/test_grm_sc1_demand_loop.py | 1 failed, 50 passed, 2 warnings in 4.36s | [051.log](cpu/051.log) |
| tests/test_grm_sc2_calibration_early_abort.py | 10 failed, 53 passed, 2 warnings in 0.57s | [052.log](cpu/052.log) |
| tests/test_grm_scout_fix1_demand.py | 12 passed, 2 warnings in 0.20s | [053.log](cpu/053.log) |
| tests/test_grm_scout_fix1_manifest.py | 4 passed, 2 warnings in 0.24s | [054.log](cpu/054.log) |
| tests/test_grm_scout_fix1_receipt.py | 3 passed in 0.04s | [055.log](cpu/055.log) |
| tests/test_grm_scout_fix1_wc1.py | 4 passed in 0.14s | [056.log](cpu/056.log) |
| tests/test_grm_supersession_battery.py | 21 passed, 2 warnings in 0.25s | [057.log](cpu/057.log) |
| tests/test_grm_three_pass.py | 7 passed, 2 warnings in 0.20s | [058.log](cpu/058.log) |
| tests/test_grm_wc1_sweep.py | 39 passed in 0.07s | [059.log](cpu/059.log) |
| tests/test_lsr_p2c_1_census.py | 46 skipped in 0.11s | [060.log](cpu/060.log) |
| tests/test_qwen38_cpu.py | 38 passed, 2 warnings in 1.55s | [061.log](cpu/061.log) |
| tests/test_gpt_oss20b_context_ladder.py | 6 passed in 0.07s | [062.log](cpu/062.log) |
| tests/test_gqa_ragged_cuda_bank.py | 20 passed, 2 warnings in 0.25s | [063.log](cpu/063.log) |
| tests/test_qwen35_translation_poc.py | 39 passed, 2 warnings in 0.51s | [064.log](cpu/064.log) |
| tests/test_gpt_oss20b_scaffold.py | 6 passed, 2 warnings in 0.21s | [065.log](cpu/065.log) |

## Failures and errors (verbatim)

### tests/test_apamq_fc_ppl.py

```text
FAILED tests/test_apamq_fc_ppl.py::test_summary_is_paired_and_contains_no_verdict
E               FileNotFoundError: missing apa_gemm receipt: /tmp/pytest-of-vader/pytest-808/test_summary_is_paired_and_con0/apa_gemm.json
```

### tests/test_graft_quant_format.py

```text
FAILED tests/test_graft_quant_format.py::test_pack_node_default_path_unchanged
FAILED tests/test_graft_quant_format.py::test_pack_node_opt_in_produces_packed_payload
FAILED tests/test_graft_quant_format.py::test_unpack_node_roundtrips_default_and_packed
FAILED tests/test_graft_quant_format.py::test_unpack_node_fails_closed_on_unknown_format_version
FAILED tests/test_graft_quant_format.py::test_pack_node_save_load_npz_cycle_with_storage_bits
E       RuntimeError: cudaMalloc failed: no CUDA-capable device is detected
```

### tests/test_grm_adm1_probe_adjudication.py

```text
FAILED tests/test_grm_adm1_probe_adjudication.py::test_completed_diag_capture_is_exact_anchor_and_l2_noop
E       AssertionError: assert False
```

### tests/test_grm_det1_11_census.py

```text
FAILED tests/test_grm_det1_11_census.py::test_census_partitions_and_counts_the_population
FAILED tests/test_grm_det1_11_census.py::test_wrong_value_with_multiple_mounts_is_not_counted_as_single_mount
FAILED tests/test_grm_det1_11_census.py::test_writing_twice_keeps_exactly_one_receipt
FAILED tests/test_grm_det1_11_census.py::test_changed_evidence_authors_a_successor_naming_its_predecessor
FAILED tests/test_grm_det1_11_census.py::test_a_successor_is_authored_only_when_evidence_actually_drifted
FAILED tests/test_grm_det1_11_census.py::test_lineage_extends_across_several_drifts
FAILED tests/test_grm_det1_11_census.py::test_lineage_rejects_a_receipt_outside_the_chain
FAILED tests/test_grm_det1_11_census.py::test_lineage_rejects_two_originals
FAILED tests/test_grm_det1_11_census.py::test_lineage_rejects_a_fork - FileNo...
FAILED tests/test_grm_det1_11_census.py::test_census_receipt_is_titled_and_scoped_to_receipts_only
E       FileNotFoundError: [Errno 2] No such file or directory: '/mnt/ForgeRealm/wt/grm-c2/artifacts/grm_det1/run_20260831T160525Z_2/registration_62cb6c09cbec211d.json'
```

### tests/test_grm_det1_11_envelope_lineage.py

```text
FAILED tests/test_grm_det1_11_envelope_lineage.py::test_lineage_contains_every_authored_envelope
FAILED tests/test_grm_det1_11_envelope_lineage.py::test_current_envelope_is_the_newest_lineage_member
FAILED tests/test_grm_det1_11_envelope_lineage.py::test_every_superseded_envelope_is_still_accepted
FAILED tests/test_grm_det1_11_envelope_lineage.py::test_a_tampered_record_for_a_real_member_is_rejected
E       AssertionError: expected an append-only series of envelopes
E       IndexError: list index out of range
```

### tests/test_grm_det1_baseline_registry.py

```text
FAILED tests/test_grm_det1_baseline_registry.py::test_live_harbor_registration_is_post_adm2
FAILED tests/test_grm_det1_baseline_registry.py::test_served_comparison_distinguishes_live_retired_and_leak
FAILED tests/test_grm_det1_baseline_registry.py::test_orion_cross_model_uses_det_fixture_semantics_not_producer_bytes
FAILED tests/test_grm_det1_baseline_registry.py::test_cross_model_value_semantics_normalize_emphasis_and_hyphen_glyphs
FAILED tests/test_grm_det1_baseline_registry.py::test_same_model_e2e_contract_remains_exact_bytes_and_ordered_mounts
FAILED tests/test_grm_det1_baseline_registry.py::test_registry_source_hash_drift_fails_closed
FAILED tests/test_grm_det1_baseline_registry.py::test_comparator_and_fixture_identity_tamper_fail_closed
E           scripts.grm_det1_baseline_registry.BaselineRegistryError: active registration receipt hash/size record does not validate
```

### tests/test_grm_importance_telemetry.py

```text
FAILED tests/test_grm_importance_telemetry.py::test_telemetry_flag_defaults_off_and_accumulator_starts_none
FAILED tests/test_grm_importance_telemetry.py::test_reset_telemetry_clears_accumulator
FAILED tests/test_grm_importance_telemetry.py::test_accumulate_telemetry_grows_without_copying_stale_tail
FAILED tests/test_grm_importance_telemetry.py::test_accumulate_telemetry_shrink_restarts_clean_no_crash
FAILED tests/test_grm_importance_telemetry.py::test_accumulate_telemetry_shrink_then_grow_sequence
E       RuntimeError: cudaMalloc failed: no CUDA-capable device is detected
```

### tests/test_grm_rs4_baseline_and_gates.py

```text
FAILED tests/test_grm_rs4_baseline_and_gates.py::test_residual_table_reports_both_framings
FAILED tests/test_grm_rs4_baseline_and_gates.py::test_residual_table_ignores_non_registered_rows
E       FileNotFoundError: [Errno 2] No such file or directory: '/mnt/ForgeRealm/wt/grm-c2/artifacts/grm_rs3/registration.json'
```

### tests/test_grm_sc1_2_e2e_pairs.py

```text
FAILED tests/test_grm_sc1_2_e2e_pairs.py::test_every_pair_has_both_arms_in_the_frozen_rows
FAILED tests/test_grm_sc1_2_e2e_pairs.py::test_pair_selection_requires_node_counts_and_never_guesses_them
FAILED tests/test_grm_sc1_2_e2e_pairs.py::test_sc1_1_rows_carry_into_the_table_unchanged
FAILED tests/test_grm_sc1_2_e2e_pairs.py::test_sc1_1_unmeasurable_arm_is_excluded_not_counted_as_a_miss
FAILED tests/test_grm_sc1_2_e2e_pairs.py::test_ten_pair_table_has_ten_rows_when_all_seven_are_measured
FAILED tests/test_grm_sc1_2_e2e_pairs.py::test_an_unreproduced_pair_is_excluded_from_numerator_and_denominator
FAILED tests/test_grm_sc1_2_e2e_pairs.py::test_the_floor_rule_never_pads_the_denominator_to_ten
FAILED tests/test_grm_sc1_2_e2e_pairs.py::test_pass_rule_is_recall_ge_0_9_and_fp_le_1
FAILED tests/test_grm_sc1_2_e2e_pairs.py::test_recovery_is_reported_never_gated
FAILED tests/test_grm_sc1_2_e2e_pairs.py::test_table_names_the_two_instruments_rather_than_blending_them
FAILED tests/test_grm_sc1_2_e2e_pairs.py::test_registration_exists_and_carries_the_race_threshold
FAILED tests/test_grm_sc1_2_e2e_pairs.py::test_registration_names_the_cross_process_weakening
ERROR tests/test_grm_sc1_2_e2e_pairs.py::test_seven_e2e_pairs_are_selected_and_all_certified
ERROR tests/test_grm_sc1_2_e2e_pairs.py::test_pair_selection_refuses_a_missing_arm
ERROR tests/test_grm_sc1_2_e2e_pairs.py::test_pair_selection_refuses_a_non_certified_family
ERROR tests/test_grm_sc1_2_e2e_pairs.py::test_pair_selection_refuses_a_planted_miss_that_is_not_a_positive
ERROR tests/test_grm_sc1_2_e2e_pairs.py::test_pair_selection_refuses_a_plant_alias_outside_its_prefix
ERROR tests/test_grm_sc1_2_e2e_pairs.py::test_probe_node_counts_match_the_hand_derived_values
ERROR tests/test_grm_sc1_2_e2e_pairs.py::test_fixture_id_number_is_NOT_the_node_index
ERROR tests/test_grm_sc1_2_e2e_pairs.py::test_every_plant_alias_lies_inside_its_own_prefix
ERROR tests/test_grm_sc1_2_e2e_pairs.py::test_every_lived_ranking_lies_inside_its_own_prefix
ERROR tests/test_grm_sc1_2_e2e_pairs.py::test_lineage_derivation_reproduces_every_frozen_manifest
ERROR tests/test_grm_sc1_2_e2e_pairs.py::test_lineage_derivation_is_prefix_sensitive
ERROR tests/test_grm_sc1_2_e2e_pairs.py::test_supersedes_is_deposit_final_across_every_frozen_shard
ERROR tests/test_grm_sc1_2_e2e_pairs.py::test_truncated_manifest_keeps_the_prefix_and_rebuilds_the_back_edges
ERROR tests/test_grm_sc1_2_e2e_pairs.py::test_truncated_manifest_drops_the_native_checkpoint
ERROR tests/test_grm_sc1_2_e2e_pairs.py::test_truncated_manifest_refuses_a_prefix_longer_than_its_source
ERROR tests/test_grm_sc1_2_e2e_pairs.py::test_truncated_manifest_applies_the_no_fold_pins
ERROR tests/test_grm_sc1_2_e2e_pairs.py::test_unpinned_no_fold_pairs_are_named
ERROR tests/test_grm_sc1_2_e2e_pairs.py::test_snapshot_manifests_for_all_seven_pairs_exist_and_are_lived
ERROR tests/test_grm_sc1_2_e2e_pairs.py::test_lived_rows_match_the_snapshot_the_pair_forks_from
ERROR tests/test_grm_sc1_2_e2e_pairs.py::test_planted_miss_arms_are_all_single_mount_withholdings
E       FileNotFoundError: [Errno 2] No such file or directory: '/mnt/ForgeRealm/wt/grm-c2/artifacts/grm_det1/run_20260831T160525Z_2/det1_4/campaign/eval/mechanistic_shards/e2e-4/attempt_001/session/instrumentation.jsonl'
E       FileNotFoundError: [Errno 2] No such file or directory: '/mnt/ForgeRealm/wt/grm-c2/artifacts/grm_det1/run_20260831T160525Z_2/det1_4/campaign/eval/mechanistic_shards/e2e-1/attempt_001/session/repository/manifest.json'
E       FileNotFoundError: [Errno 2] No such file or directory: '/mnt/ForgeRealm/wt/grm-c2/artifacts/grm_det1/run_20260831T160525Z_2/det1_4/campaign/eval/mechanistic_rows.jsonl'
E       FileNotFoundError: [Errno 2] No such file or directory: '/mnt/ForgeRealm/wt/grm-c2/artifacts/grm_sc1_1/sc1_1_g2_summary_c1baaea3f682e7cc.json'
E       FileNotFoundError: [Errno 2] No such file or directory: '/mnt/ForgeRealm/wt/grm-c2/artifacts/grm_sc1_2/registration.json'
```

### tests/test_grm_sc1_demand_loop.py

```text
FAILED tests/test_grm_sc1_demand_loop.py::test_registered_threshold_is_the_race_constant_not_a_refit
E       FileNotFoundError: [Errno 2] No such file or directory: '/mnt/ForgeRealm/wt/grm-c2/artifacts/grm_det1/run_20260831T160525Z_2/det1_4/campaign/calibration/thresholds_2149a44b6136fd1b.json'
```

### tests/test_grm_sc2_calibration_early_abort.py

```text
FAILED tests/test_grm_sc2_calibration_early_abort.py::test_envelope_rule_reproduces_the_frozen_det1_threshold_exactly
FAILED tests/test_grm_sc2_calibration_early_abort.py::test_minima_verdict_equals_detector_decision_on_every_frozen_row
FAILED tests/test_grm_sc2_calibration_early_abort.py::test_fire_index_matches_first_strictly_below_on_frozen_rows
FAILED tests/test_grm_sc2_calibration_early_abort.py::test_registered_split_is_deterministic_across_repeated_runs
FAILED tests/test_grm_sc2_calibration_early_abort.py::test_split_sets_are_non_empty_and_match_the_registration
FAILED tests/test_grm_sc2_calibration_early_abort.py::test_held_out_contains_the_current_false_fire
FAILED tests/test_grm_sc2_calibration_early_abort.py::test_sc1_2_arm0_rows_fold_into_the_race_rows_losslessly
FAILED tests/test_grm_sc2_calibration_early_abort.py::test_the_solace_instrument_disagreement_is_carried_not_collapsed
FAILED tests/test_grm_sc2_calibration_early_abort.py::test_planted_rows_are_all_exactly_zero_mass
FAILED tests/test_grm_sc2_calibration_early_abort.py::test_the_candidate_is_the_envelope_over_the_calibration_set
E           scripts.grm_sc2_calibration.CalibrationError: missing persisted rows: /mnt/ForgeRealm/wt/grm-c2/artifacts/grm_det1/run_20260831T160525Z_2/det1_4/campaign/calibration/mechanistic_rows.jsonl
E           scripts.grm_sc2_calibration.CalibrationError: missing persisted receipt: /mnt/ForgeRealm/wt/grm-c2/artifacts/grm_det1/run_20260831T160525Z_2/det1_4/campaign/calibration/thresholds_2149a44b6136fd1b.json
E       FileNotFoundError: [Errno 2] No such file or directory: '/mnt/ForgeRealm/wt/grm-c2/artifacts/grm_sc2/registration.json'
```
