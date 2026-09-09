# GRM-C4 — Was width 64's benefit actually chunking? (worktree wt/grm-c4, branch grm-c4)

Origin: `docs/GRM_SCOUT_2026-09-08.md` Part C rank 4 and the Part A
finding that "splitting helped, not narrowing" is a hypothesis (WC1
changed chunk length, capture geometry and admission opportunities
together). David's goal: test both sides. The two sides: **the benefit
is chunking** (64-token children in a 96-seat arena recover t33 and
reach 14/14 without losing Juniper) vs **the benefit is narrowing**
(only a 64-seat arena recovers t33).

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/wt/grm-c4`.

## Mission
1. Cross chunk length {64, 96} × arena width {64, 96} at FIXED
   capture / seat / live geometry (the RS3 capture pin and near-live
   seating as in the WC1 profile, pinned separately and recorded in
   every receipt). Deposit-time chunking is a NEW additive knob on the
   split-at-deposit guard (`core/graft_repository.py` /
   `core/grm_three_pass.py`, find the guard the WC1 ledger names),
   flag-OFF by default, parent text preserved.
2. Score all three batteries (supersession 9, census 10, long-history
   14 of `scripts/grm_eb1_longhorizon_gpu.py` / `scripts/grm_wc1_sweep_gpu.py`
   reused by import) per cell, plus split counts, routes, and ACTUAL
   token residency (summed token seats, not graft count; the Scout's
   B5 finding). The WC1 (64,64) and (96,96) cells are the existing
   receipts (`artifacts/grm_wc1_opus/grm_wc1_results.json`); re-run
   them only if the fingerprint says the harness changed, otherwise
   cite them and run the two new off-diagonal cells.
3. Registered prediction (Scout's): (64-chunk, 96-width) recovers t33
   and keeps Juniper. Rejection rule (registered): the chunking claim
   is rejected if the improvement needs the narrower seating.
4. Budget ≤ 1 GPU-h; per-battery cells under the 285 s rail (split the
   long-history battery into leased segments with a resumable session
   state if needed; state how).

## Done (verbatim)
1. The chunk knob (file/lines, flag name, default OFF pin test); cell
   list with estimates; registration + sha; blocked-report; exact lead
   commands.
2. CPU gate results (chunking splits at 64 on a synthetic deposit,
   parent text preserved, default OFF unchanged behaviour byte-identical
   on an existing manifest).
3. Prior art; deviations; RED; process safety; model id and effort.

## COMMON RULES (all Scout follow-up orders)
- Forked from `lc1-wip`. Additive only: new scripts under `scripts/`, new
  registration JSON under `artifacts/<campaign>/`, new tests; no edits to
  existing kernels, batteries, registries, `config/`, or flag defaults.
  Reuse the existing harness modules by import, never by copy-edit.
- Registration IMMUTABLE once written (fixtures, counts, thresholds,
  acceptance bars, predictions), sha-bound; later changes are amendment
  JSONs. Every receipt fingerprints the files its cell executes.
- Your sandbox has NO GPU. Build, CPU-gate (pytest, `--dry-run` enumerating
  every cell with wall estimates), write the blocked-report and exact
  `lead_commands.txt` in dependency order. The lead runs GPU cells through
  the leased runner (`/tmp/forge-gpu.lock`, ≤285 s worker / 590 s outer,
  30 s cooldown; a cell that cannot fit the rail is registered non-fit,
  never retried into a longer lease). Total GPU budget for this order
  ≤0.75 GPU-h unless stated. Never kill or signal any process you did
  not start; never clear a lock.
- Both sides of every decision are cells with equal standing; the order
  registers the prediction, the receipts decide. RED is a result.
- No git (lead commits), no subagents, no background waits, foreground
  only, < 10 min per call. Prior Art Directive at every code site.
- Reasoning effort: high.
