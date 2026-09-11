# GRM-F2 — chronicle folds must not rebind facts onto an alias (lead order, round 2, 2026-09-11)

Seat: Opus 5 (opus-max). YOUR WRITABLE TARGET is `/mnt/ForgeRealm/wt/grm-f2`
(branch `grm-f2`, forked from `lc1-wip` 8edfae4). `core/` edits AUTHORIZED
under the flag below only. Read-only: `/mnt/ForgeRealm/GraftRepository`
(canonical; `artifacts/grm_d1/lt1_1/run_A/cells/*/session/repository` holds
the r2 arm-A session repositories — copy what you need, never write there),
`/mnt/Shared/LEAD_TODO_2026-09-11_R2.md`. No git, no subagents, no GPU,
foreground, Bash calls < 10 min, never kill anything; `--basetemp` under
`artifacts/grm_f2/tmp` with cleanup; repo-relative pins only.

## The receipt
LT1.1 r2 arm A digest 24 folded sources [turns 13, 14, 17, 18] — turn 18
is the alias turn "call Lantern 'the Beacon' from now on" — and reads:
"the maintenance crew of the Beacon is located at Iona Vale, and the map
position of the Beacon is (-31, 48, 12)". Commtower's crew and
Breakwater's coordinates were rebound onto Lantern's alias; era 61
inherited it. A+ digest 28 (no alias turn in its window) kept "comm
tower" / "breakwater" correct. (`artifacts/grm_d1/REPORT.md` §3.4.)

## Mission
1. Reproduce on the CPU double: rebuild the fold from those four source
   texts through the production consolidation path (FIX-3 wrapper,
   FIX-5 enumeration) with the model stubbed by a faithful extractive
   double AND show the failure mode with a recorded-output double that
   replays digest 24's text — the fixture must fail on the entity
   check, not on the stub. Entity check: every (entity, attribute, value)
   in the sources must appear in the digest with the SAME entity (or its
   registered alias for that entity only).
2. Fix, flag `GRM_FOLD_ALIAS_GUARD` (default OFF; OFF byte-identical):
   pick by evidence and state both: (a) alias/rename turns are excluded
   from chronicle fold windows (they are handled by A1's parse or stay
   as turn nodes), and/or (b) the FIX-5 enumeration names the entity per
   fact and the coverage/QC check rejects a digest whose facts are
   attributed to a different entity than their source. Fixtures RED→GREEN
   on digest 24; controls: non-alias folds byte-identical ON vs OFF; the
   13 FIX-5 GPU-contrast folds' recorded digests still pass QC.
3. Register the guard as a named optional flag on the LT1.1 r3
   registration F1 owns (write `artifacts/grm_f2/flag_contract.json`
   stating the env name, default, and what the lead pins); no GPU run of
   your own.
4. Ledger + REPORT; prior art (FIX-5, SC1.1 grounding, A1 parse; entity
   faithfulness literature — cite what you know, mark unverified);
   `python3 -m pytest -q --basetemp … tests/test_grm_f2*.py
   tests/test_grm_scout_fix3*.py tests/test_grm_scout_fix5*.py
   tests/test_grm_a1_alias_fold.py` verbatim last.

## Done (verbatim)
1. Reproduction (both doubles) lines; the entity check definition.
2. Core diff (files:lines), the flag, OFF byte-identity, the choice (a)/(b) with evidence.
3. Fixture RED→GREEN lines; controls; flag contract path.
4. Prior art, deviations, RED, process safety, model + effort; pytest LAST.
