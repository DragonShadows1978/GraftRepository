# SCOUT-FIX-9 — degenerate fold digests and the route-eligibility guard (lead order, 2026-09-11)

Seat: Opus 5 (opus-max). YOUR WRITABLE TARGET is the git worktree
`/mnt/ForgeRealm/wt/grm-fix9` (branch `grm-fix9`, forked from
`grm-merge`). `core/` edits AUTHORIZED for this defect only (minimal
diff, same shape as FIX-3/5/8). Read-only: `/mnt/ForgeRealm/GraftRepository`,
every other `/mnt/ForgeRealm/wt/grm-*` worktree (receipts you may copy
FROM). No git. No subagents. Foreground only, every Bash call < 10 min,
no background waits. NEVER kill or signal a process you did not start.
No GPU: CPU fixtures + a registered replay the lead runs.

## The receipt (P1 GPU smoke r2, 2026-09-10, `wt/grm-p1/artifacts/grm_p1/gpu_smoke_r2.log`
and `gpu_session/` — copy them in)
Production ladder, GPT-OSS-20B, `GRM_PROFILE=eb1_c2` (width 96, capture
pin, seat-near-live, RT1, `GRM_ADMISSION_RULE=margin_first`), turn 19
(`/recap`):
```
core.grm_admission.AdmissionPolicyError: production route ranking differs from
frozen A-DEC score reconstruction: backend=native
production=[18, 21, 13, 8, 15, 16] reference=[18, 21, 20, 13, 8, 15]
```
Node 20 is a fold digest: `kind=digest`, `ntok=4`, text exactly
`"ARCHIVE NOTE."`, `cent=None`, `rare=['archive','note']`, with a live
`native_node_id`. The native router omits a node with no route key; the
Python reconstruction scores it from `_lex_bonus` alone and ranks it
third; the integrity guard (correctly) refuses; the turn died. This
reconstruction only runs under `margin_first`, which is why no battery
saw it before; LT1 (200 turns, margin_first) did not produce such a
digest by luck.

Two defects, both core:
1. A digest of four tokens with no fact content was ACCEPTED by the fold
   QC (FIX-3 punctuation-collapse QC + FIX-5 coverage) and deposited as
   an active node. Determine which check let it through (coverage with
   `need` empty? a fold with zero enumerated facts? the QC applied
   before the Harmony final-channel strip?) and close it: a digest that
   carries none of its sources' facts, or has no route key, is REJECTED
   (sources stay active; receipt says why) — never deposited.
2. Route eligibility must be one rule: a node without a route key
   (`cent is None`) is ineligible in BOTH the native router and the
   Python A-DEC reconstruction (and the CPU fake), so the guard compares
   identical candidate sets. Lead ruling: a node you cannot route to is
   not a candidate; lexical bonus alone does not make it one.

## Mission
1. Reproduce on the CPU fake: build the r2 state (copy the session
   repository from `gpu_session/` or synthesize the minimal state: a
   digest node with `cent=None`, 4 tokens) and drive the ladder under
   `margin_first` → `AdmissionPolicyError` RED-before.
2. Fix both defects with the minimal core diff; RED→GREEN on that
   fixture; a second fixture proves a legitimate fold with facts is
   unchanged; the C2 132 recorded plans stay byte-identical (the
   FIX-6/FIX-8 replay gate); FIX-3/4/5/6/8 suites and
   `tests/test_grm_a1_alias_fold.py` green; `GRM_ADMISSION_RULE` unset
   = byte-identical behaviour on every existing fixture.
3. Register (do not run) a replay of the r2 recap turn from the saved
   `gpu_session/` repository on GPT-OSS-20B under `eb1_c2` (≤ 0.1
   GPU-h): prediction — the turn no longer raises, and the recap answer
   is produced (content reported, not gated). `lead_commands.txt`.
4. Ledger + report; prior art; `python3 -m pytest -q
   tests/test_grm_scout_fix9*.py tests/test_grm_scout_fix3*.py
   tests/test_grm_scout_fix5*.py tests/test_grm_scout_fix6*.py
   tests/test_grm_scout_fix8*.py tests/test_grm_admission.py
   tests/test_grm_a1_alias_fold.py` verbatim last.

## Done (verbatim)
1. Which QC check admitted the digest (file:line) and the fix; the
   eligibility rule change (file:line, both paths).
2. RED-before/GREEN-after lines; the C2 plan byte-identity line;
   suites green.
3. Replay registration path + sha + lead commands.
4. Prior art, deviations, RED items, process safety, model + effort.
