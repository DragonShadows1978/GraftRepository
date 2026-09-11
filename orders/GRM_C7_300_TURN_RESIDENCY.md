# GRM-C7 — Does bounded residency survive corrections, folding, cold eviction and restart together? (worktree wt/grm-c7, branch grm-c7, forked from grm-c2)

Origin: `docs/GRM_SCOUT_2026-09-08.md` Part C rank 7 ("tests the actual
product promise"). This branch carries SCOUT-FIX-1 (B1/B2/B3/B5) and the
C2 profile registry `config/grm_eb1_profile_registered.json`. David's
goal: work the Scout report to completion; test both sides of decisions.

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/wt/grm-c7`.

## Mission
1. Register a **300-turn session** on GPT-OSS-20B under the **C2
   profile** (width 96, live capture pin, seat-near-live, RT1; EB1 frame;
   demand OFF): unseen entities (≥ 40), aliases (≥ 12), contradictions /
   corrections (≥ 12, some correcting earlier corrections), **two
   restarts** (persist, new process, reload) at registered turns, and
   **forced paging** (a registered cold-eviction pressure that pushes
   old grafts out of the hot tier and back). Probes: ≥ 60, placed at
   registered distances (5, 30, 60, 120, 250 turns back), including
   probes that require the folded digest/era rather than the raw turn.
2. **Full-information oracle**: the same probes answered with the exact
   source text in the live window (upper bound). Report exact-answer
   errors and abstention errors SEPARATELY per distance and per class
   (fresh / alias / correction / folded), plus active residency per
   turn (summed token seats), fold events with their coverage ratio
   (the 0.70 rule), and every failed fold retained on disk.
3. **Both sides**: the registered comparison side is the same 300-turn
   script under **today's shipped defaults** (pin OFF, near-live OFF,
   width per shipped config); register it as arm B with its own cells so
   the lead can run it if the budget allows; state its cost. If it does
   not fit the cap below, register it NON_FIT with the cost.
4. Registered prediction (Scout's): residency stays bounded but at least
   one attribution / admission / fold failure appears that repeated
   needles never showed. Registered acceptance for "product promise":
   exact-answer error ≤ oracle + 15 points at every distance, no
   unbounded residency growth (max token seats ≤ 2× the 96-seat arena
   plus recency mounts), restarts retain scores.
5. Budget: arm A ≤ 2 GPU-h, split into resumable leased cells (a
   persisted session state per cell boundary; one cell = a turn range;
   every cell under the 285 s rail: choose the turns-per-cell from the
   measured EB1 turn wall in `artifacts/grm_eb1/` and say the number).

## Done (verbatim)
1. Registration path + sha; fixture manifest sha; cell list with
   estimates and the turns-per-cell derivation; arm B cost.
2. CPU gate results (fixture integrity, probe placement, oracle
   scorer pinned against hand cases, resume from a cell boundary on a
   synthetic state); blocked-report; exact lead commands.
3. Prior art; deviations; RED; process safety; model id and effort.

## COMMON RULES
As in `orders/GRM_C3_DNGH_DECOY_CALIBRATION.md` §COMMON: additive only (new scripts/tests/registration JSON; no edits to kernels, batteries, registries, config or flag defaults), registration immutable and sha-bound, NO GPU in the sandbox (build, CPU gates, `--dry-run`, blocked-report, exact `lead_commands.txt` for the leased runner: ≤285 s worker / 590 s outer / 30 s cooldown, non-fit registered never retried), never kill anything, no git, no subagents, foreground, < 10 min per call, Prior Art Directive. Reasoning effort: high.
