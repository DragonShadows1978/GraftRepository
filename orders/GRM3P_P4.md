# ORDER GRM3P-P4 — Composed E2E at Scale (Opus 4.8 seat)

YOUR WRITABLE TARGET is /home/vader/GraftRepository-three-pass — edits,
builds, and GPU runs are AUTHORIZED in this worktree. Production
/mnt/ForgeRealm/GraftRepository is READ-ONLY reference. Do not run git
(the lead commits). Do not spawn subagents. Do not idle on monitors.
GPU: flock --wait conventions; operator has right of way.

## Premise (pre-nailed — not open for renegotiation)

The three-step turn procedure (prep / inference / cleanup) is LANDED and
gated: P0/P1 @8ea4b77, P2 @1fa95dd — G-EQUIV/G-ROUTE/G-PREP/G-SERVE/
G-COMPACT all PASS lead-verified on the smoke frame. Your job is NOT to
redesign, refactor, or "improve" any of it. Your job is to run the
composed E2E at scale and report what the receipts say. Read
docs/GRM_THREE_PASS_PLAN.md (immutable) and docs/GRM_THREE_PASS_LEDGER.md
(both lead-verification sections) before touching anything.

## What P4 must establish (the smoke could not)

The 4-node smoke never fired a cold page-in and never stressed the
CUDA route ABI top-16 envelope. P4 runs the composed session at a
scale where the memory system actually works for a living.

## Work items

1. **Full-mode composed run, both pipelines.** Use the existing
   `--mode full` driver frame (see docs/GRM_E2E_RECEIPT_PLAN.md and the
   existing full-run driver conventions). If full mode's node count can
   exceed the CUDA top-16 full-ranking envelope, the documented
   fail-closed applies — report which turns fell back and why; do NOT
   widen the ABI (kernel changes forbidden).
2. **Cold-start leg**: run a session against a repository large enough
   (and with VRAM paging budget bounded, per the existing LRU paging
   conventions) that step-1 prep MUST page in cold grafts. The step_io
   receipts must show those page-ins in step 1, zero in step 2.
3. **Step-3 think-time table**: per-turn step-3 (cleanup) wall-ms
   distribution (mean/p50/p95/max) from stage timing, both legs. This
   is the P3-decision receipt: whether cleanup fits between turns
   without overlap machinery. Report the numbers; the verdict is the
   lead's.
4. **No new mechanisms.** If a gate cannot pass with the landed
   machinery, it is RED with the failing receipt verbatim. Do not add
   flags, policies, or code paths beyond what items 1–3 strictly
   require (instrumentation/driver plumbing only).

## Registered frames (ALL pinned; deviations are FAILURES)

- F-FULL: `--mode full`, all other CLI defaults, env
  `GRM_GQA_CUDA_ROUTE=1 GRM_GRAFT_STORAGE_BITS=8`, both
  `--turn-pipeline single` and `three_pass`.
- F-COLD: same env, session/paging parameters chosen ONCE before any
  run and stated in the report (node count, VRAM budget, probe set);
  both pipelines, identical parameters.

## Gates (registered here, before any run; comparative — no invented numbers)

- **G-E2E-EQ**: F-FULL `single` vs `three_pass` — transcripts
  byte-identical, canonical arena byte-equal.
- **G-E2E-RECALL**: F-COLD `three_pass` probe recall ≥ `single` recall
  on the identical frame; per-probe table verbatim.
- **G-E2E-IO**: F-COLD step-2 page-ins == 0 AND step-2 uploads == 0 on
  every turn; all page-ins in steps 1/3; working_set + step_io receipts
  present for every turn.
- **G-E2E-LEDGER**: memory-ledger receipts complete (schema v1) on all
  three_pass turns, both frames.
- **G-E2E-SERVE**: three_pass total wall ≤ 1.25× single on each frame.
- Full suites both pipelines green (446 passed baseline).

## Honesty

RED results are results — verbatim failing lines. No threshold or frame
adjustment after seeing results. No vacuous passes: a gate that cannot
run is RED with the reason. Your final report will be verified against
disk by the lead; fabricated specifics have been caught before and are
career-ending for a seat.

## Done

Final message MUST contain verbatim: both frames' gate tables, the
step-3 think-time table, the F-COLD parameters statement, per-probe
recall table, page-in/upload step attribution summary, both suite
lines, and honest residuals.
