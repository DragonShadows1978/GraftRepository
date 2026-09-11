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


---

## Follow-up 9 — LT1.1 r2 analysis + second scoring column (amendment 9)

**Date:** 2026-09-11. **Seat:** Opus 5 (`claude-opus-5[1m]`), effort MAX.
**Inputs:** frozen r2 campaign receipts, both arms 26/26 cells COMPLETE.
No GPU lease taken; no new campaign run; `core/` read-only throughout.

### Commands

- `run_A/cells/*/probes.jsonl`, `run_Aplus/cells/*/probes.jsonl` — 40 rows
  per arm, read only.
- `run_*/cells/A-197-200/session/repository/manifest.json` — node store with
  full `text`, `kind`, `sources`, `retired` per node.
- `artifacts/grm_lt1/amendment2/run_margin_first/cells/A-*/probes.jsonl` —
  archived LT1 parent rows, rescored under the LT1.1 class map.
- `python3 -m pytest -q --basetemp artifacts/grm_d1/tmp tests/test_grm_d1*.py
  tests/test_grm_lt1_1*.py`

### File changes

- **NEW** `scripts/grm_d1_scorer_v2.py` — the second scoring column.
- **NEW** `artifacts/grm_d1/lt1_1/amendment9.json` + `.sha256` — chained to
  amendment 8, restates amendment 8's `core_rebind` verbatim (pins unchanged).
- **NEW** `artifacts/grm_d1/lt1_1/miss_causes.json` — per-row cause table.
- `artifacts/grm_d1/REPORT.md` — new §3.1-3.5.

### Results

Two columns, col1 = registered scorer (PRIMARY, unmodified), col2 =
`value_span_glyph_tolerant_v2` (sha-bound in amendment 9):

| | A+ col1/col2 | A col1/col2 | LT1 parent col1/col2 |
|---|---|---|---|
| fresh | 8/15 → 10/15 | 10/15 → 13/15 | 16/17 → 16/17 |
| correction | 10/10 → 10/10 | 9/10 → 9/10 | 5/10 → 5/10 |
| alias | 10/10 → 10/10 | 10/10 → 10/10 | 5/10 → 5/10 |
| recap | 4/5 → 4/5 | 4/5 → 4/5 | n/a |
| total | 32/40 → 34/40 | 33/40 → 36/40 | 26/37 → 26/37 |

Column 2 is a verified strict superset (0 rows pass col1 and fail col2). It
moves 5 rows, all in LT1.1, and 0 rows in the LT1 parent.

### Findings

1. **The "U+2011" framing was wrong for the dash.** `grm_lt1.normalize`
   already NFKC-folds and already maps U+2010..U+2014/U+2212 to `-`. The
   real unfolded glyph is **U+202F**; the `9 voxels` misses are
   hyphen-for-space plus singular/plural. Measured, not assumed.
2. **The Beacon capture (arm A correctness defect).** Arm A digest 24 folds
   sources including turn 18 ("call Lantern 'the Beacon'") and rebinds
   Commtower's crew and Breakwater's coordinates onto "the Beacon". Era 61
   inherits it. Arm A+ digest 28 keeps the correct entities. "Breakwater"
   appears in 3 A+ nodes and 1 A node.
3. **Routing, not storage, is the top residual.** In every Breakwater miss in
   both arms the mounted node is unrelated to the question.
4. **Source nodes are never mounted, even for hits** (0/28 A+, 0/29 A).
   Correct answers come through fold digests and eras.
5. **Fresh regression is real and negative for A1.** A+ 8/15 vs A 10/15
   (10 vs 13 col2). Both arms share the worker and funnel, so the residual
   difference is the fold itself enlarging the digest population (38/11 vs
   31/8). Recorded as a negative result, not smoothed.
6. **Receipt gap:** `alias_fold_history` is not persisted to campaign
   receipts; only `artifacts/grm_d1/alias_cpu_aplus.json` itemizes the
   10 A+ merges (A: 0).

### Evidence classes

- Two-column table, cause table, alias attribution: **frozen GPU campaign
  receipts**, read-only, no re-execution.
- Fold decision itemization: **CPU replay** (`alias_cpu_aplus.json`) — a
  claim about fold firing only, NOT a model-quality claim.
- LT1 parent column: **archived LT1 receipts**, rescored; the lead's 14/15
  fresh and 0/5 recap are not reproducible under this fixture's class map and
  stand as the campaign record.

### Prior art

- `scripts/grm_d1_scorer_v2.py`: SQuAD `normalize_answer` (Rajpurkar et al.
  2016) for the declared-normalization-pipeline idea. Mine: the Unicode-space
  fold, optional-parenthesis rule, unit singular/plural fold. Deliberately NOT
  taken: SQuAD's strip-all-punctuation, which would erase the minus signs.
  Unverified — no network in sandbox; lead to check. Search terms: "SQuAD
  normalize_answer exact match", "answer span normalization unicode".
- Beacon capture: the entity/coreference error class from abstractive
  summarization faithfulness work (e.g. Maynez et al. 2020). Observed here,
  not imported. Unverified — lead to check. Search terms: "entity
  hallucination abstractive summarization", "coreference error summarization
  faithfulness".

### Process safety

No git, no subagents, no GPU lease, no core writes, no foreign process
killed or signalled. pytest `--basetemp` on NVMe, deleted after.

### Follow-up 9 addendum — test defects found by the gate

The verbatim gate went RED (4 failed) before going green. All four were MY
test defects surfacing against the lead's newer r2 receipts, not campaign
regressions:

1. **`_stage()` copied 28 GB to tamper with 320 KB.**
   `tests/test_grm_lt1_1_preflight.py::_stage` ran
   `shutil.copytree(runner.OUT, ...)`, and `runner.OUT` now contains
   `run_A/` and `run_Aplus/` — a 112 MB native store per cell across 52
   cells. Six tamper tests staged 93 GB and hit ENOSPC on NVMe; earlier in
   this task the same pattern filled the root filesystem twice. Fixed to copy
   only the files. Measured after: the whole preflight file runs in **0.35 s
   on 1.9 MB** of scratch, against ~15 min and 53-93 GB before. Same 19
   tests, same assertions. **This was the single cause of every disk incident
   in this task.**
2. **Three tests asserted a one-time state as an invariant.**
   `test_the_archived_red_cell_is_preserved`,
   `test_the_spurious_red_cell_is_archived_and_arm_a_rearmed` and
   `test_dry_run_enumerates_twentysix_cells_per_arm` required the arms to be
   RE-ARMED (live cell absent, `next_cell == 'A-001-008'`). That was true
   right after the r1 archive; the lead has since run r2 to completion
   (26/26 both arms), so the live cells legitimately exist and `next_cell` is
   `None`. Each now skips (or accepts `None`) when the arm has been re-run.
   The archive-integrity assertions they guard are unchanged and still
   enforced — that is the part that must always hold.

Gate after the fixes: **209 passed, 5 skipped**, 12 MB scratch. The 2 extra
skips versus the earlier 211/3 are exactly the two re-arm tests, now
correctly inapplicable to a completed tree.
