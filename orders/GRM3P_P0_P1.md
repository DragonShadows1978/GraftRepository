# ORDER GRM3P-P0/P1 — scaffold the three-pass driver + extract pass 3

You are an implementation agent on the GRM three-pass program.
YOUR WRITABLE TARGET is /home/vader/GraftRepository-three-pass —
edits, builds, and GPU runs there are AUTHORIZED. Production
/mnt/ForgeRealm/GraftRepository is READ-ONLY reference. Read
docs/GRM_THREE_PASS_PLAN.md FIRST — immutable registered intent;
this order executes its P0 and P1 exactly. Append scope to
docs/GRM_THREE_PASS_LEDGER.md (create) before working, results
after. No git commits/pushes (lead commits), no subagents, no
network, no pip, no service actions. Red results reported verbatim.
Do not idle-monitor; work to completion.

## P0 — scaffold + baseline
1. `turn_pipeline` selector on the session driver: `single`
   (default — byte-identical to HEAD, gated) | `three_pass`.
2. Baseline stage-timing table for the current single-pass turn on
   the registered dev frame (route / mount / infer / deposit /
   supersession / importance bookkeeping), artifact JSON.

## P1 — extract pass 3
1. Move ALL turn-time arena mutation (deposit, supersession,
   importance bookkeeping) into a pass-3 unit running AFTER output.
   Pass 2 becomes arena-read-only (instrument + assert).
2. MEMORY LEDGER receipt: one structured JSON per turn — every
   mutation with reasons + provenance hashes. Freeze the schema in
   the ledger before consumers.
3. Gates per the plan: G-EQUIV (single byte-identical; three_pass
   final arena byte-equal to single for the same scripted session),
   G-SERVE (pass-2 visible memory overhead <= 5ms; total <= 1.25x
   baseline), G-COMPACT completeness audit. Run full existing GRM
   suites both pipelines.

## Done
Report: baseline stage table, gate table verbatim, memory-ledger
schema, suite lines, honest residuals — verbatim in the final
message.
