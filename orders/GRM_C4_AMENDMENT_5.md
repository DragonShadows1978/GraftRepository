# GRM-C4 amendment 5 (lead, 2026-09-09) — lh-12 exceeds the rail; split it, and decouple the remaining cells from c64_w96's score

Lead-run under amendment 4's cap: the A2 recovery re-ran `c64_w96`
long-history segments 06–11 PASS from bound state, then **`c4-lh-12`
RED `Worker rail exceeded: NON_FIT`** (285 s;
`artifacts/grm_c4/lead_a2/runs/c64_w96/longhorizon_c4-lh-12.json`);
lh-13 not run; `score --cell c64_w96` not reached; the A3 stage then
refused with `Prior cell score missing`. Both stops are correct.

Lead decisions:
1. **Split, don't lengthen.** Register sub-segments for the long-history
   turn ranges that do not fit (lh-12, and lh-13 if its plan is the
   same size): halves (or thirds, from the measured walls of lh-06..11
   and the partial lh-12 events, state the numbers) each with a
   planning estimate < 200 s, bound-state resume between them, the
   same scoring. The lh-12 RED stays as recorded; the sub-segments are
   new units (create-only), one authorized run each. Apply the SAME
   split to `c96_w64` and `c64_w64` long-history plans up front (they
   run the same battery; do not wait for them to hit the rail).
2. **Decouple.** The A3 runner runs `c96_w64` and `c64_w64` without
   requiring `c64_w96`'s score first (cells are independent; the
   ordering guard was a convenience). The cross summary must handle a
   NON_FIT or partial cell honestly (print the cell as NON_FIT with the
   segments that completed, never a partial score as a battery score).
3. Cap: recorded + registered under 5,400 s; if the split pushes the
   projection over, say so and register the overflow as NON_FIT rather
   than trimming.

Same worktree (a4 committed 3a915b0), same rules (no git, no subagents,
no GPU, foreground, never kill anything). Effort: high.

## Mission
Amendment JSON (sha-bound to this order and amendment_a6) with the
sub-segment registration and the decoupling; `lead_commands_a5.txt`:
c64_w96 sub-segments of lh-12 then lh-13 (+ score), then c96_w64 and
c64_w64 full unit lists with the split applied (+ scores), then the
cross summary. CPU gates: sub-segment resume on a synthetic layout;
forged/stale refused; existing receipts byte-unchanged; dry-run.

## Done (verbatim)
1. Amendment path + sha; the split with its estimates; unit counts;
   files/lines; tests.
2. Exact lead commands (`lead_commands_a5.txt`).
3. Prior art; deviations; RED; process safety; model id and effort.
