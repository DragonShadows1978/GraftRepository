# GRM-H4 — campaign_receipt marks for the XM1/XM2/XM3 test modules on the post-D2 tree

**Seat:** Opus 5 (opus-max). **Lead:** Fable 5.1 (plans, verifies, commits). **Date:** 2026-09-13.

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/wt/grm-h4` (branch `grm-h4`, forked from `lc1-wip` aa226e7+) — edits under `tests/` and `docs/TESTS_CAMPAIGN_RECEIPTS.md` AUTHORIZED, pytest runs AUTHORIZED. Read-only everywhere else. `core/` and `scripts/` are NOT to be edited. No git (lead commits). No subagents. No GPU. Never kill or signal a process you did not start. Every pytest run uses `--basetemp /mnt/ForgeRealm/wt/grm-h4/artifacts/grm_h4/tmp` and removes it afterwards.

## Situation
The XM1 (cross-model graft-read parity), XM2 (Qwen final channel) and XM3 (Trinity) campaigns were run on branches forked from `lc1-wip` BEFORE the D2 default flip (round-2 flags default ON; `core/grm_admission.py` etc. moved). They are merged now (fdd6880, ff5c9f4, aa226e7). On the merged tree `python3 -m pytest tests/test_grm_xm*.py` gives **24 failed, 96 passed, 3 errors**. Two causes, both known classes (see `docs/TESTS_CAMPAIGN_RECEIPTS.md`, GRM-H2/H3 deltas):

1. **sha-bound receipts** — the XM registrations pin `core/grm_admission.py`, `scripts/grm_xm1_cpu.py`, `scripts/grm_xm1_parity.py` … at their campaign shas; the post-D2 tree drifts, so the registrations fail closed (`registration source drift`, `X3 source drift`, dry-run/immutability tests).
2. **artifact-bound receipts** — 19 pinned XM2 process files were lost when the worktree was removed (lead error, ledgered): `artifacts/grm_xm2/amendment_1/evidence_before.json`, `evidence_concurrent.json`, `recorded_trace_cpu.json`, and 16 first-attempt `lead_*.log`. Nine others were recovered byte-identical (commit after aa226e7). Tests that read the lost files error (`OSError`/`FileNotFoundError`).

## Task
Reproduce each of the 27 failures/errors in isolation, classify it (sha-bound / artifact-bound / file-set-bound / genuine defect), and add `@pytest.mark.campaign_receipt(registration='<path>')` marks per the rules at the top of `docs/TESTS_CAMPAIGN_RECEIPTS.md`. A genuine defect (a test whose own probe is wrong, not a receipt) gets pinned with NO assertion change, as H3 did. Do NOT loosen any assertion; do NOT mark a test that passes.

Add a "GRM-H4 delta" section to `docs/TESTS_CAMPAIGN_RECEIPTS.md` (table: module, marks, node ids, class, reason, registration) and record the 19 lost XM2 files there as an artifact-bound class with their registration sha pins.

## Done — paste verbatim in your final message
1. The delta table; any genuine defect and its pin (line cited).
2. Verbatim last lines of: (a) `python3 -m pytest -q tests/test_grm_xm*.py -p no:cacheprovider --basetemp …` (must be 0 failed / 0 errors); (b) the same with `-m campaign_receipt` (every new mark must reproduce as a failure/error); (c) the full suite in the three parts H3 used (`artifacts/grm_h3/part*.txt` show the split) — 0 failed / 0 errors.
3. `grep -n 'GRM-H4' tests/*.py` output; confirmation core/ and scripts/ untouched (`git status --short core scripts` empty).
4. Prior art (mark mechanism is the pytest idiom; say so, or name anything else taken).
5. Model + effort you ran at; confirmation of no GPU / no git / no subagents / nothing killed / basetemp removed.
