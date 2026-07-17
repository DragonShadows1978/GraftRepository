# CAMPAIGN-D: importance diagnostics — KL re-rank (7b) + per-layer capture (7a)

YOUR WRITABLE TARGETS in /mnt/ForgeRealm/GraftRepository (main tree):
- tests/test_grm_importance_g1g2.py
- scripts/grm_importance_diagnostics.py (new)
Everything else READ-ONLY (sibling seats in this tree — never touch
core/, tests/test_grm_s4_demotion.py, or
tests/test_grm_runtime_lifecycle.py). No git. No subagents. No GPU /
no model loads. RED honesty; no monitor-idling.

## Law
docs/GRM_IMPORTANCE_LEDGER.md — read in full, ESPECIALLY the
2026-07-17 correction + exploratory registration: this work is
HYPOTHESIS-GENERATING, not a gate. Your outputs are diagnostics
reports; no pass/fail language anywhere. The sealed artifacts
(tests/fixtures/importance_convos/artifacts/*.json) carry per
candidate: s1_mass, s2_salience, s3_dep_dlogit, s3_dep_kl. There is
NO per-layer data in them (verified) — that is what part 2 adds.

## Part 1 (7b, CPU-only): scripts/grm_importance_diagnostics.py
reads the sealed artifacts and reports, per probe and aggregated:
S1 and S2 rank agreement (Spearman + top-1, reusing the harness's
own tie-aware functions by import — never reimplement) against S3
ranks computed BOTH ways: dlogit (the registered primary, = the
published verdict, reproduce it as a cross-check) and KL. Also
report dlogit-rank vs KL-rank agreement of S3 with itself (how much
does the arbiter change?). Output: human-readable table + one JSON
line (schema grm_importance_diag_v1) marked "exploratory": true.

## Part 2 (7a, driver extension): extend the g1g2 driver to record
per-candidate per-layer S1 (arena.s1_mass(per_layer=True) diagnostics
dict) into the artifact under a new optional key s1_mass_per_layer
(schema bump to grm_importance_g1g2_convo_v2; loader stays
backward-compatible with v1 artifacts — analysis must not require
the new key). The lead re-runs GPU legs; your CPU tests prove the
schema round-trip and loader compatibility with mixed v1/v2 sets,
and part 1's script reports per-layer rank agreement per layer
WHEN the key is present (marked exploratory, best-single-layer
explicitly labeled post-hoc).

## Verify
py_compile; python3 -m pytest tests/test_grm_importance_g1g2.py -q;
run part 1 against the existing sealed artifacts and PASTE its
human-readable table in your report.

## Done
Verbatim: diff summary; pytest line; the part-1 diagnostics table
(this is the actual deliverable — I read it directly); exact lead
commands (GPU re-run legs + rerun diagnostics); ambiguities;
"no git, no GPU, no files outside grant".
