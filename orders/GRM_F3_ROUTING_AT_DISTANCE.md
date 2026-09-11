# GRM-F3 — routing at distance: why the intact record is never ranked (lead order, round 2, 2026-09-11)

Seat: Opus 5 (opus-max). YOUR WRITABLE TARGET is `/mnt/ForgeRealm/wt/grm-f3`
(branch `grm-f3`, forked from `lc1-wip` 8edfae4). `core/` is READ-ONLY
for the diagnosis; a fix is authorized ONLY if item 3 proves a defect,
and then flag-gated (`GRM_ROUTE_…`, default OFF, OFF byte-identical).
Read-only: `/mnt/ForgeRealm/GraftRepository` (canonical; the r2 receipts
under `artifacts/grm_d1/lt1_1/run_A` and `run_Aplus`: `probes.jsonl`
route receipts, `residency.jsonl`, session repositories),
`/mnt/Shared/LEAD_TODO_2026-09-11_R2.md`. No git, no subagents, no GPU,
foreground, Bash calls < 10 min, never kill anything; `--basetemp` under
`artifacts/grm_f3/tmp` with cleanup; repo-relative pins only.

## The receipts
Arm A misses at distance: recall_3_100 served "(64, -17, 8)",
recall_3_150 "(-9, -44, 71)", recap_3 "(19, 83, -22)" for Breakwater
(expected (-31, 48, 12)); arm A+ recall_2_100/150 fabricated a Commtower
crew. In every one the mounted node is an unrelated fact (Nacre's ticket
price, Saffron's galley stock, Crag's tug price — `miss_causes.json`),
while a digest holding the correct value exists (A+: 3 nodes name
Breakwater; A: 1, the Beacon-captured one). The question is why the
ranker never reaches it at d ≥ 100 while it does at d ≤ 50.

## Mission
1. Per-row route table for the 7 rows above + the 3 matching correct
   rows at d=50 (same questions): ranking ids with scores/margins,
   identifier scan (`identified_candidates`, FIX-8 normalized identifier
   set of every node that carries the value), admission branch, recency
   set, era membership (does an era shadow the digest?), the digest's
   `cent` provenance (rebuilt child keys? FIX-9 hold?), width/seat state.
   Use the recorded session repositories on the CPU (routing, admission,
   identifier scan are real core code; only the model is absent — say
   what the frozen ranking receipts allow you to reproduce exactly and
   what needs the GPU).
2. Classify each miss: identifier not in the digest's set (FIX-8/FIX-5
   naming), digest present but out-ranked (score/margin; by what), era
   shadowing, recency exclusion (FIX-4 interaction), or seat/width. One
   cause per row with the receipt line.
3. If a single mechanism explains ≥ 5 of 7 rows, propose the fix with a
   CPU RED→GREEN on the frozen state and land it flag-gated; otherwise
   deliver the table and a registered proposal only.
4. F4 (scorer secondary, small): register `scripts/grm_d1_scorer_v2.py`'s
   column as round 2's secondary scorer and extend the abstention regex
   with "we're still working on that" (+ variants seen in r2 rows);
   re-score r2 A and A+ under it and report both columns (registered
   scorer stays primary).
5. Ledger + REPORT; prior art; `python3 -m pytest -q --basetemp …
   tests/test_grm_f3*.py tests/test_grm_admission.py
   tests/test_grm_scout_fix4.py tests/test_grm_scout_fix8.py` verbatim last.

## Done (verbatim)
1. The 10-row route table and the per-row cause line.
2. Fix (if any): files:lines, flag, OFF byte-identity, RED→GREEN; or the registered proposal.
3. F4 scorer registration path; r2 two-column rescore.
4. Prior art, deviations, RED, process safety, model + effort; pytest LAST.
