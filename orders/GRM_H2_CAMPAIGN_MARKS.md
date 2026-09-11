# GRM-H2 — campaign-receipt marks for the eight newly merged campaign branches (lead order, 2026-09-11)

Seat: Opus 5 (opus-max). YOUR WRITABLE TARGET is the git worktree
`/mnt/ForgeRealm/wt/grm-h2` (branch `grm-h2`, forked from `lc1-wip`
afae419). Only `tests/`, `docs/TESTS_CAMPAIGN_RECEIPTS.md` and
`artifacts/grm_h2/` are writable; `core/` and `scripts/` read-only. No
git, no subagents, no GPU, foreground, every Bash call < 10 min, never
kill or signal a process you did not start. Every pytest run uses
`--basetemp /mnt/ForgeRealm/wt/grm-h2/artifacts/grm_h2/tmp` and removes
it after (the two disk fills last night came from tests copying trees
into /tmp).

## Context
`lc1-wip` just absorbed the Scout Part C / X campaign branches
grm-c3, grm-c4, grm-c5, grm-c6, grm-c8, grm-x1, grm-x2, grm-x3 (their
receipts were reported 2026-09-09 and never merged; the lead merged
them so the forks could be pruned). Their test modules are
campaign-receipt tests of the class `docs/TESTS_CAMPAIGN_RECEIPTS.md`
describes (sha-bound to their day's core or reading gitignored
artifacts). The H1 marker (`campaign_receipt(registration=…)`) and the
`GRM_*` env guard are on this tree.

## Mission
1. Enumerate the test modules those eight merges added or changed
   (`git log` is read-only for you: use `git diff --stat afae419~8..afae419 -- tests/`
   equivalent via the lead-provided list in `artifacts/grm_h2/merged_tests.txt`,
   which the lead writes before dispatch). Reproduce each module in an
   isolated process under `--campaign-receipts`; mark per function what
   fails for receipt reasons (registration string per the sibling
   convention); do NOT mark genuine defects — report those.
2. Gate lines: ONE-process `python3 -m pytest -q --basetemp … tests/test_grm_*.py`
   (0 failed / 0 errors, N skipped), split the run if it exceeds 10 min
   per call and report each part; `-m campaign_receipt` counts; the
   leak-order line. Update the docs table and `artifacts/grm_h2/marked_*`.

## Done (verbatim)
1. Delta table (marks added, module, reason, registration).
2. Gate lines verbatim; genuine defects found (if any) with evidence.
3. Prior art (unchanged from H1 unless new), process safety, model + effort; pytest line LAST.
