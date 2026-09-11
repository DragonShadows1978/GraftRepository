# GRM-X1 amendment 4 (lead, 2026-09-09) — r4: reduced grid under a 3 GPU-h authorization

r3 ran exactly as registered: the first unit (`oracle_m1_s0 / alias_00`)
COMPLETED in 106 s, and the controller projected 144 units × ~106 s
against the 5,400 s budget and stopped with `NON_FIT_BUDGET`
(`artifacts/grm_x1/receipts/r3/controller_15479065….json`). The stop is
correct. Lead decision: **r4 runs a reduced grid** that still answers
the question both sides:
- oracle cells `oracle_m1_s0` and `oracle_m100_s0` (12 units each),
- natural cells `natural_m1` and `natural_m100` (24 units each),
- `m10` and the second seed dropped (registered as NOT RUN, with the
  cost they would add);
- total 72 units, planning estimate 72 × 106 s ≈ 7,632 s; the lead
  authorizes **3.0 GPU-h (10,800 s) total for X1** across r1–r4
  (historical 570 s + r3's 106 s charged).
Same worktree (r3 committed), same rules (no git, no subagents, no
GPU, foreground, never kill anything). Effort: high.

## Registered by this amendment (continuation_04)
1. The reduced cell list and the new budget; the r3 unit receipt for
   `oracle_m1_s0 / alias_00` stays valid under r4 iff the unit definition
   is byte-identical (state it; do not re-run it if valid).
2. Natural-arm gating unchanged (natural cells run only if the oracle
   arm is positive on the reduced oracle set; define "positive" as
   registered in r1, restated here).
3. Per-unit receipt additionally records the served text class
   (answer / abstention / refusal-style text such as "I'm sorry, but I
   can't help with that", which r3's first unit served for arm A on an
   alias-present query) so the summary can report refusals separately
   from abstentions; no scoring change.
4. CPU gates as in r3 (continuation accepted / forged / stale / r3
   receipts untouched / dry-run enumerates 72 units) and refreshed
   `lead_commands.txt` in the r3 loop form.

## Done (verbatim)
1. continuation_04 path + sha; unit count; validity ruling for the r3
   unit; test names and results.
2. Exact lead commands.
3. Prior art; deviations; RED; process safety; model id and effort.
