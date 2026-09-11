# GRM-F1 — source retention after folds (lead order, round 2, 2026-09-11)

Seat: Opus 5 (opus-max). YOUR WRITABLE TARGET is the git worktree
`/mnt/ForgeRealm/wt/grm-f1` (branch `grm-f1`, forked from `lc1-wip`
8edfae4). `core/` edits AUTHORIZED under the flag below only. Read-only:
`/mnt/ForgeRealm/GraftRepository` (canonical; its `artifacts/grm_d1/lt1_1/run_A`
and `run_Aplus` hold the LT1.1 r2 receipts and session repositories),
`/mnt/Shared/LEAD_TODO_2026-09-11_R2.md`. No git, no subagents, no GPU,
foreground, every Bash call < 10 min, never kill or signal a process you
did not start. Tests that copy repositories use `--basetemp
/mnt/ForgeRealm/wt/grm-f1/artifacts/grm_f1/tmp` and clean up. Every
registration you write pins REPO-RELATIVE paths only (round-1 lesson:
absolute `wt/` pins became permanent receipts when the forks were pruned).

## Receipts you are working from
LT1.1 r2 (both arms): source nodes appear in `route_info.mounts` in 0/57
correct rows — every answer is read from a fold digest or era
(`artifacts/grm_d1/REPORT.md` §3, `miss_causes.json`). FIX-5
consolidation retires a fold's sources when the digest's coverage ≥ 0.70
(`core/graft_arena.py` consolidate / `core/graft_repository.py` fold
job, `MIN_FOLD_KEEP`); after that the only routable record of a fact is
its digest, and fresh facts at d ≥ 100 are confabulated (Breakwater
coordinates: three different wrong tuples across arms).

## Mission
1. Flag `GRM_FOLD_RETAIN_SOURCES` (default OFF; OFF byte-identical —
   prove it with the C2 132-plan replay gate and the FIX-3/5/8/9 and A1
   suites). ON: a fold's sources stay ACTIVE and routable after the
   digest is deposited (the digest is added, not substituted); lineage
   records `digest_of` / `retained_sources`; supersession semantics
   unchanged (a correction still retires the old value and any digest
   carrying it — check the A1 `_alias_extend_correction_targets` path and
   the plain `correct_memory` path); width guard unchanged; residency
   accounting reports retained-source seats separately.
2. Measure on the CPU double through the REAL LT1.1 worker (`scripts/
   grm_lt1_1.py --arm A --fake …`, production semantics, opt-in
   full-campaign fakes stay OFF; use the child-spawn/production-turns
   gates + a 4-cell fake run): retained sources are routable after a
   fold; the OFF arm byte-identical to r2's fake path. Report the node
   count / seat count delta.
3. Register LT1.1 r3 arm A (flag ON) against the r2 A baseline: same
   frozen conversation, same worker, ≤ 1.3 GPU-h, repo-relative pins,
   prediction from the plan verbatim (fresh c2 ≥ 14/15; corrections ≥
   9/10; aliases 10/10; residency bounded, max seats reported); the
   registration must accept F2's flag as a named optional so the lead can
   pin both. `artifacts/grm_f1/lead_commands.txt` (each command must run
   with `--dry-run` appended — add that gate).
4. Ledger + REPORT; prior art (say what is reused: FIX-5, LSR-P2C, A1
   lineage); `python3 -m pytest -q --basetemp … tests/test_grm_f1*.py
   tests/test_grm_scout_fix5*.py tests/test_grm_scout_fix9*.py
   tests/test_grm_a1_alias_fold.py tests/test_grm_lt1_1_production_turns.py`
   verbatim last.

## Done (verbatim)
1. Core diff (files:lines), the flag, OFF byte-identity lines, lineage
   rules, supersession proof.
2. CPU measurement (routable sources after fold; seat/node deltas).
3. Registration path + sha; lead commands (dry-run gate line).
4. Prior art, deviations, RED, process safety, model + effort; pytest LAST.
