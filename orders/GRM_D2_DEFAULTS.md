# GRM-D2 — ship the proven settings as defaults (lead order, round 2, 2026-09-11)

David (2026-09-11): "You can make the best decision in your judgement. I
can roll it back if I have to." Lead's decision: the settings below become
the shipped defaults; each keeps its OFF/old value as an explicit setting
so rollback is one environment variable.

Seat: Opus 5 (opus-max). YOUR WRITABLE TARGET is `/mnt/ForgeRealm/wt/grm-d2`
(branch `grm-d2`, forked from `lc1-wip`). `core/`, `scripts/`, `tests/`,
`docs/` writable for THIS change only. No git, no subagents, no GPU,
foreground, Bash calls < 10 min, never kill anything; `--basetemp` under
`artifacts/grm_d2/tmp` with cleanup; repo-relative pins only.

## The flips (evidence class per line)
1. `GRM_ADMISSION_RULE` default → `margin_first` (R1: 31/31 replay,
   0 correct→wrong; LT1/LT1.1 natural questions 0/35 admitted under the
   old rule). Old rule stays selectable as `all_tokens_bind`.
2. The C2 profile (`eb1_c2`: width 96, live capture pin, seat-near-live,
   RT1) becomes the shipped default (C2: beats shipped defaults on every
   battery and survives restart; every LT1.x run used it). The old
   defaults stay selectable as a named profile (`legacy_256`).
3. `GRM_FOLD_RETAIN_SOURCES`, `GRM_FOLD_ALIAS_GUARD`,
   `GRM_ROUTE_SOLE_BINDER_INSURANCE`, `GRM_ALIAS_FOLD_MERGE` default → ON
   **as a set** (LT1.1 r3: control 36/40 → all four 38/40; three without
   the alias merge regress aliases 10→7 — never ship a subset). Each keeps
   `=0` as its OFF setting.

## Mission
1. Flip the defaults at their resolvers (`core/grm_admission.py`,
   `scripts/grm_profile.py` / `grm_c2_profile`, `core/grm_fold_retain.py`,
   `core/grm_fold_alias_guard.py`, `core/grm_alias_fold.py`); unknown
   tokens still fail closed — to the NEW default, and say so. Add one
   umbrella `GRM_LEGACY_DEFAULTS=1` that restores every pre-round-2
   default at once (the one-line rollback; documented).
2. Byte-identity: with `GRM_LEGACY_DEFAULTS=1`, the C2 132-plan replay
   gate (`scripts/grm_f5_c2_replay_gate.py`), the FIX-4 frozen receipt
   test, the F1 all-off fingerprint and the OFF arms of every flag test
   must reproduce their recorded values. With the new defaults and NO
   env set, the F1/F2/F5/A1 ON fixtures must pass and `grm_chat.py
   --print-flags` must show the r3 A+′ flag set with no pins.
3. Tests whose expectations encode the old defaults: change them ONLY
   by pinning `GRM_LEGACY_DEFAULTS=1` in a fixture (their assertions
   stay); list every such test. Campaign-receipt tests untouched.
4. Docs: `docs/GRM_CHAT.md` and the primer's defaults section updated;
   a `docs/GRM_DEFAULTS_2026-09-11.md` recording the decision, the
   evidence per flip, the rollback line, and the r3 numbers.
5. Gates: `python3 -m pytest -q --basetemp … tests/test_grm_f1*.py
   tests/test_grm_f2*.py tests/test_grm_f5*.py tests/test_grm_a1_alias_fold.py
   tests/test_grm_admission.py tests/test_grm_scout_fix4.py
   tests/test_grm_scout_fix6*.py tests/test_grm_chat*.py
   tests/test_grm_lt1_1_production_turns.py` verbatim last; then the
   three-part full GRM suite (split as H2 did; report each part).

## Done (verbatim)
1. Files:lines per flip; the umbrella switch; fail-closed direction.
2. Byte-identity lines under LEGACY; new-default gate lines; the
   `--print-flags` block.
3. Tests re-pinned (list); docs paths.
4. Prior art, deviations, RED, process safety, model + effort; pytest LAST.
