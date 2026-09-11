# GRM-D1 LEDGER — receipts as they happened (2026-09-10)

Seat: Opus 5 (`claude-opus-5[1m]`), effort MAX.
Order: `orders/GRM_D1_LT1_RESIDUAL_DIAGNOSIS.md`.
Writable target: `/mnt/ForgeRealm/wt/grm-d1` (branch `grm-d1`).

## Commands run, in order

| # | command | result |
|---|---|---|
| 1 | orient: `ls`, `ls scripts/ tests/ artifacts/` | LT1 harness + 26 arm-A cells located |
| 2 | locate the 35 recall rows: `find ... -name probes.jsonl` | 19 files; 10 arm-A cells with `controller.json` status COMPLETE = 35 rows |
| 3 | grade all 35 arm-A rows with `scripts.grm_lt1.score` | 24 correct, 11 wrong_value, 0 abstention — matches `comparison.json` |
| 4 | read `core/graft_arena.py:_route_cand_base` | retired nodes are excluded from routing — the load-bearing fact |
| 5 | read `scripts/grm_lt1_worker.py:146-148` | every turn fed via bare `arena.feed()`; corrections carry no edge |
| 6 | read `core/graft_repository.py:correct_memory` (1953-1998) | production path retires, sets `superseded_by`, bumps route epoch |
| 7 | `grep -rn 'alias' core/` | one comment + one unrelated docstring: NO alias mechanism in core |
| 8 | `grep -rn 'GRM_ALIAS_FOLD_MERGE' core/ scripts/ tests/` | no match — the named optional flag is ABSENT on this tree |
| 9 | `python3 -m pytest -q tests/test_grm_lt1*.py` (baseline) | **49 failed / 16 passed** — gitignored artifacts missing in worktree |
| 10 | `rsync -a --ignore-existing ../grm-lt1/artifacts/ artifacts/` | restored LT1 receipts; baseline improves to **19 failed / 58 passed** |
| 11 | diff core between `grm-lt1` and `grm-d1` | `graft_arena.py` (50-line diff) + `grm_admission.py` (28) have drifted; the 19 failures are `INPUT_SHA_MISMATCH`, pre-existing and correct |
| 12 | `python3 scripts/grm_d1_cause_table.py` | arm A 11 rows: alias-edge-without-base (core composition) 5; fixture-lineage (harness) 5; stale-first ranking (core A-DEC) [arm-A instance is ladder-drop, not mis-rank] 1 |
| 13 | `python3 scripts/grm_d1_supersession.py` | RED feed_only ranking `[0,1,2]` retired `[]`; GREEN supersede ranking `[2]` retired `[0,1]` |
| 14 | `python3 scripts/grm_d1_recap.py` | **oracle 5/5 PASS** on the five new named questions |
| 15 | `python3 scripts/grm_d1_register_lt11.py` | LT1.1 registration sha256 `02b44d02e3fbb82c...` |
| 16 | `python3 -m pytest -q tests/test_grm_d1*.py` | **42 passed** |
| 17 | `python3 -m pytest -q tests/test_grm_lt1*.py tests/test_grm_d1*.py` | **19 failed, 100 passed** — failing set byte-identical to baseline |
| 18 | `comm` on before/after FAILED lists | **zero new failures, zero D1 failures** |
| 19 | `find core -newer artifacts/grm_d1/red_before_lt1.log` | empty — core provably untouched |

## Decisions taken

1. **Classified the corrections as harness, not core**, because the
   candidate base excludes retired nodes: a real supersession removes
   the stale node from the ranking entirely. Verified by CPU contrast,
   not asserted.
2. **Stopped on the alias class** rather than proposing a core patch,
   per the order. `recall_6_*` succeeds with the same base-only mount,
   so the necessity of a co-mount is unproven — an A1 call.
3. **Qualified the order's third class name.** The one arm-A row was a
   ladder drop of a correctly-ranked plan head, not a mis-rank. Kept the
   order's label for mapping, appended the qualifier.
4. **Wrote the LT1.1 fixture under `artifacts/`, not `fixtures/`**, to
   avoid tripping the SHA-frozen LT1 fixture manifest.
5. **Recorded the absent alias flag as absent**, with a test that
   re-verifies the claim against the tree, rather than silently omitting
   it or silently pinning it.

## Failures and how they were handled

- The LT1 suite was RED on arrival (49F/16P). Root-caused to missing
  gitignored receipts, then to genuine core SHA drift from `grm-merge`.
  Restored the receipts; reported the drift as the remaining, correct,
  pre-existing RED rather than working around it.
- Two writes were intercepted by the AfterImage hook and the bash write
  guard; both retried through the Write tool as required.

## Evidence classes

- Cause table: **frozen GPU run receipts**, read-only, no re-execution.
- Supersession proof: **CPU lineage fixture** — a claim about route
  eligibility only, explicitly NOT a model-quality claim.
- Recap battery: **CPU oracle gate** — a claim about question
  answerability only.
- LT1.1 outcome: **unmeasured**. Registered (1.70 GPU-h), not run.

