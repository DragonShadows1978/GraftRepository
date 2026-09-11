# GRM-F6 — dead absolute worktree paths in scripts (lead order, round 2, 2026-09-11)

Seat: Opus 5 (opus-max). YOUR WRITABLE TARGET is `/mnt/ForgeRealm/wt/grm-f6`
(branch `grm-f6`, forked from `lc1-wip` c609e96). `scripts/`, `tests/`,
`docs/`, `artifacts/grm_f6/` writable; `core/` read-only. No git, no
subagents, no GPU, foreground, Bash calls < 10 min, never kill anything;
`--basetemp` under `artifacts/grm_f6/tmp` with cleanup.

## Finding
Round-1 seat worktrees `/mnt/ForgeRealm/wt/grm-*` were pruned on
2026-09-11 (receipts live in the canonical `artifacts/`). Several
scripts still hard-code absolute paths into them, and some fail
SILENTLY (a glob over a dead path returns 0 rows and the gate passes
vacuously): `scripts/grm_lt1_offline.py:18` (`C2 = Path('/mnt/ForgeRealm/wt/grm-c2/…')`,
found by F5), `scripts/grm_rd1.py:25` (`SOURCE = Path('/mnt/ForgeRealm/wt/grm-c7')`,
found by H2), `scripts/grm_scout_fix8_cpu.py:19` (`C2 = …wt/grm-c2…`,
found by H2). There are likely more (`grep -rn "/mnt/ForgeRealm/wt/" scripts/ tests/`).

## Mission
1. Inventory every absolute `/mnt/ForgeRealm/wt/…` (and `/home/vader/GraftRepository-…`)
   path in `scripts/` and `tests/`; classify: (a) a data path whose
   receipts now live under the canonical `artifacts/` at a known
   relative path, (b) a path with no surviving copy (say so; do not
   invent one), (c) a comment/doc only.
2. For (a): rebind to `ROOT / 'artifacts/…'` (repo-relative, resolved
   from the script's own location, overridable by an env var named per
   script); add a vacuous-zero guard wherever a glob/list feeds a
   gate (`rows == 0` → RED with the path named, never a pass). For (b):
   the script fails loud with the missing path and the receipt that
   named it. Keep sha-bound registrations untouched — they are receipts;
   only the consumer's path resolution changes.
3. Gates: for each rebound script, a CPU run that reproduces the number
   its receipt recorded (e.g. the C2 replay 132/132 through
   `grm_lt1_offline`); a test per script that plants a dead path and
   asserts the loud failure; `python3 -m pytest -q --basetemp …
   tests/test_grm_f6*.py tests/test_grm_lt1_fix6.py
   tests/test_grm_scout_fix8.py tests/test_grm_rd1_replay.py` verbatim
   last (receipt-marked tests may skip by default; run them once under
   `-m campaign_receipt` and report).
4. `docs/TESTS_CAMPAIGN_RECEIPTS.md`: add the rule "scripts and
   registrations pin repo-relative paths; a dead-path glob is RED, never
   zero rows". Ledger + REPORT.

## Done (verbatim)
1. Inventory table (path, script:line, class, action).
2. Per-script reproduction line vs its receipt; the dead-path RED lines.
3. Prior art, deviations, RED, process safety, model + effort; pytest LAST.
