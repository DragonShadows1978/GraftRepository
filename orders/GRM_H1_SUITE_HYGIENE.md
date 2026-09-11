# GRM-H1 — test-suite hygiene: campaign-receipt markers + env isolation (lead order, 2026-09-11)

Seat: Opus 5 (opus-max). YOUR WRITABLE TARGET is the git worktree
`/mnt/ForgeRealm/wt/grm-h1` (branch `grm-h1`, forked from `grm-merge`
170e3b3). Edits under `tests/`, `conftest.py`/`pytest.ini`/`pyproject`
test config, `docs/`, `artifacts/grm_h1/` are AUTHORIZED. `core/` and
`scripts/` are READ-ONLY (test files only; do not edit the campaign
scripts the tests exercise). No git. No subagents. Foreground only, Bash
calls < 10 min, no background waits. NEVER kill or signal a process you
did not start. No GPU.

## Findings you are fixing (lead receipts, 2026-09-10)
1. On the merged tree the GRM suite (`tests/test_grm_*.py`) is 1,861
   passed / 102 failed / 63 errors. Every failure is a CAMPAIGN-RECEIPT
   test: it reads gitignored campaign artifacts (checkpoints, receipts)
   or asserts `INPUT_SHA_MISMATCH` against core shas frozen on the
   campaign's day (`test_grm_c2_amendment/epoch3/budget_a5/score_a4`,
   `test_grm_c7_r2_registration/r3/amendment7/lead_*`,
   `test_grm_lt1_amendment*`, `test_grm_scout_fix5_runner`,
   `test_grm_scout_fix8_replay/resume`, `test_grm_rd1_replay`,
   `test_grm_det1_*`, `test_grm_sc*`, `test_grm_r1_*` on any tree but
   grm-r1, …). They already fail on their own source branches once the
   core moved. Such tests are receipts, valid only at their registration
   sha; a tree-wide run must not report them as regressions.
2. Env leakage between test modules: workers pin `GRM_ADMISSION_RULE`
   (R1 `pin_rule`, LT1, P1 smoke) via `os.environ` and some tests leave
   it set, re-ruling every later module in the same pytest process
   (P1 seat found it in its smoke; the lead reproduced it with
   `test_grm_r1_*` before `test_grm_scout_fix4.py`: 3 FIX-4 failures and
   20 A1 failures that vanish when R1 runs in its own process).

## Mission
1. A pytest marker `campaign_receipt(registration=<path>)` applied to
   every test module or test that is sha-bound to a campaign
   registration or reads campaign artifacts under `artifacts/`. Default
   collection SKIPS them with a reason naming the registration; `-m
   campaign_receipt` or `--campaign-receipts` runs them. Decide per
   file by reading it (grep `INPUT_SHA_MISMATCH`, `registration`,
   `artifacts/`), not by name; list every marked file with its reason in
   `docs/TESTS_CAMPAIGN_RECEIPTS.md`. Do NOT edit the tests' assertions.
2. An autouse session/function fixture in `tests/conftest.py` that
   snapshots every `GRM_*` environment variable before each test and
   restores exactly after (delete keys that did not exist), plus a
   check that FAILS the test that leaked (name the key) so the leak is
   attributed, not hidden. Prove it with a fixture test that plants a
   leak and asserts the attribution.
3. Gate: on this tree, `python3 -m pytest -q tests/test_grm_*.py` in ONE
   process must be 0 failed / 0 errors with N skipped campaign receipts
   (report N and the list); then `python3 -m pytest -q -m
   campaign_receipt tests/test_grm_*.py` reproduces the campaign
   failures (report counts; they are expected). Also run the four fix
   suites in the order that leaked before (`tests/test_grm_r1_*.py
   tests/test_grm_scout_fix4.py tests/test_grm_a1_alias_fold.py
   tests/test_grm_admission.py`) — must be green now. Bash calls < 10
   min: split the suite run if needed and report each line.

## Done (verbatim)
1. Marker + conftest (files/lines); marked files table with reasons.
2. Gate lines verbatim (the full one-process line, the -m line, the
   leak-order line).
3. Prior art (pytest markers/monkeypatch idioms; say what is reused),
   deviations, RED items, process safety, model + effort. pytest line LAST.
