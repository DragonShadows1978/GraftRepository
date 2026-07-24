# ORDER GRM3P-P2 — Step-1 Prep: Search + Prefetch Staging

YOUR WRITABLE TARGET is /home/vader/GraftRepository-three-pass — edits,
builds, and GPU runs are AUTHORIZED in this worktree. Production
/mnt/ForgeRealm/GraftRepository is READ-ONLY reference. Do not run git
(the lead commits). Do not spawn subagents. Do not idle on monitors.
GPU: use flock --wait conventions; the operator has right of way.

## Canonical framing (supersedes "three-pass" prose; identifiers keep the name)

ONE sequential turn procedure, three steps:
1. PREP — router search + stage the working set (this order).
2. INFERENCE — the only KV-building step; visible window ends at last token.
3. CLEANUP — deposit/supersession/eviction/ledger (landed in P0/P1 as
   `turn_pipeline=three_pass` pass 3).

## Context you must read first

- docs/GRM_THREE_PASS_PLAN.md (immutable) — P2 scope.
- docs/GRM_THREE_PASS_LEDGER.md — P0/P1 receipts INCLUDING the lead
  verification section: the seat's G-COMPACT RED was a frame artifact
  (`--restart-after 99` deviated from the smoke default 5 and kept
  live-window echo alive). LESSON BINDING ON YOU: every frame parameter
  in this order is pinned; deviations are FAILURES even if disclosed.
- Baseline stage table (p0_single_instrumented): route 1420.8ms/turn
  MEAN under route_backend=python. The exact ragged CUDA router (merged
  to main 2e61708, default-off) measured p50 1.59ms at 512 nodes —
  three orders of magnitude sit on the table.

## Work items

1. **Step-1 prep stage in `three_pass`**: before inference, run router
   search over the full repository and STAGE the winners in a per-turn
   working set (device-resident, page-ins done, mount surgery prepped).
   Inference resolves mounts L1 = staged set first, L2 = repository on
   miss (count misses). Prep emits `prep` stage timing rows and a
   `working_set.json` receipt per turn: probe used, ranking, staged ids,
   page-in count, prep wall ms.
2. **Engage the exact ragged CUDA router in prep.** P0/P1 sessions
   reported `route_backend=python` despite `GRM_GQA_CUDA_ROUTE=1`.
   Diagnose why the opt-in fell back; fix the wiring (flag plumbing
   only — do NOT modify router kernels). If engagement is impossible
   without kernel changes, STOP and report; do not tune around it.
3. **Page-ins off the visible path**: instrument page-in/upload events
   per step; all must occur in step 1 (or step 3). Zero during step 2.
4. **Probe enrichment — EXPLORATORY, report-only**: compare bare-message
   probes (E4 hygiene law) vs recency-context-augmented probes on the
   smoke fixtures. Report rankings/recall deltas both ways. NO gate —
   the 2-probe smoke fixture is too thin to license a verdict; say so
   in the report rather than claiming one.

## Registered frames (ALL parameters pinned — no deviations)

Frame F1 (gate frame): `--mode smoke --skip-gpu-idle-check`, all other
CLI at defaults (restart_after=5 — resume EXERCISED), env
`GRM_GQA_CUDA_ROUTE=1 GRM_GRAFT_STORAGE_BITS=8`.
Frame F2 (informational, echo lens): identical plus `--restart-after 99`.
Gates are judged on F1 ONLY. F2 results reported verbatim, no tuning.

## Gates (thresholds registered here, before any run)

- **G-EQUIV-P2**: `single` transcript byte-identical to committed
  8ea4b77 receipts (sha 68da84b8…) on F1; `three_pass` transcript
  byte-identical to `single`; canonical arena byte-equal.
- **G-ROUTE**: F1 three_pass prep reports `route_backend=cuda` on every
  routed turn AND per-turn ranking ids identical to the python backend
  ranking (run both, compare verbatim).
- **G-PREP**: staged-set recall == direct-route recall on F1 (2/2), the
  fact-bearing source node present in the staged set for every probe,
  L1 misses == 0 on F1.
- **G-SERVE-P2**: step-2 visible memory overhead ≤ 5ms; total latency
  ratio vs P0 baseline ≤ 1.25; report the new route wall-ms row.
- **G-COMPACT unchanged**: 10/10 receipts, ledger complete on F1.
- Full suites both pipelines green (443 passed baseline).

## Honesty

RED results are results — report them verbatim with the failing line.
No threshold adjustment after seeing results. No vacuous passes: if a
gate cannot run, it is RED with the reason, not skipped.

## Done

Final message MUST contain verbatim: the F1 gate table, F2 probe results,
prep stage-timing rows (mean/max per turn), working_set.json for one
probe turn, both suite lines, the route-backend diagnosis (what was
broken, what changed), enrichment comparison table, and honest residuals.
