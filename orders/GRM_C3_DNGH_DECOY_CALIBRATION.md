# GRM-C3 — Does D-NGH detect missing knowledge when something plausible is mounted? (worktree wt/grm-c3, branch grm-c3)

Origin: `docs/GRM_SCOUT_2026-09-08.md` Part C rank 3. David's goal: the
Scout report is worked to completion; where there is a decision, test
both sides. The decision here is the demand threshold rule, which the
primer lists as David's open item: **carried 0.338** (`config/grm_demand_registered.json`,
min of served-turn minima, carried not refit) vs **candidate 0.318**
(SC2 race envelope) vs **a structural near-zero rule** (SC2 finding:
every planted miss sat at mounted mass exactly 0.0, served controls
never below 0.29) vs the **zero-cost empty-band baseline** (fire iff
the routed band is empty). All four are arms.

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/wt/grm-c3`.

## Mission
1. Freeze 48 new cases, balanced 12/12/12/12 across: correct mount
   (fact present, right node mounted), plausible decoy (a wrong but
   topically matching node mounted, the right one absent from the
   repository), unavailable fact (nothing relevant deposited), and
   weak-but-correct read (right node mounted but the value is a
   number/name the model reads weakly; take the t30/t33 class as the
   model of weakness). Reuse the DET1/SC2 planting machinery
   (`scripts/grm_det1_5_workers.py`, `scripts/grm_sc2_calibration.py`,
   `scripts/grm_sc1_2_session.py`) by import; cases are NEW text, none
   of the existing fixture ids, and none of the SC2 fit fixtures
   (`e2e_t09_cypher_bridge`, `e2e_t16_lyra_dock`).
2. One GPU pass per case records the per-token full-layer mean mounted
   mass trace and the served answer with the detector OFF (so the trace
   is the same for every rule); the four rules are then evaluated
   offline on the same traces (CPU), so the GPU cost is one trace per
   case. Estimate the wall; split into leased cells (≤ 285 s each).
3. Registered acceptance for a demand flip (from the Scout): miss
   detection ≥ 90% on decoy+unavailable, false fires ≤ 5% on
   correct+weak, no lost correct answers. Registered prediction (the
   Scout's): the near-zero rule detects empty bands but misses at
   least one nonempty wrong-memory case; the carried 0.338 creates
   avoidable false fires on weak-but-correct reads.
4. Report per rule: TP/FP/TN/FN, which cases separate the rules, and
   the recommendation with its evidence class. If no rule meets the bar,
   say so; that is the result.

## Done (verbatim)
1. Case set (ids, class, sha); registration path + sha; cell list with
   estimates; blocked-report; exact lead commands.
2. CPU gate results (fixture validity, offline evaluator pinned against
   a hand-computed trace).
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
