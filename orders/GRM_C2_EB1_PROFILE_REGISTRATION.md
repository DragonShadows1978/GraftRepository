# GRM-C2 — Can the best measured GPT-OSS EB1 profile become a durable default? (worktree wt/grm-c2, branch grm-c2, forked from grm-fix1)

Origin: `docs/GRM_SCOUT_2026-09-08.md` Part C rank 2. This worktree is
forked from `grm-fix1` (SCOUT-FIX-1 landed: B1/B2/B3/B5), so this order
is ALSO the EB1 E2E re-validation gate for those fixes. David's goal:
test both sides. The two sides: **profile ON** (width 96 + live capture
pin + seat-near-live + RT1 split-child routing, exactly the WC1/RT1.1
geometry) vs **today's defaults OFF** (the shipped flag values), each on
freshly captured repositories, each across a restart.

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/wt/grm-c2`.

## Mission
1. **New registry** `config/grm_eb1_profile_registered.json` (create-only,
   additive; the existing `grm_live_registered_baselines.json` is NOT
   edited): bound to the exact model revision, frame, flag values,
   capture/seat geometry, and payload provenance (the B3 manifest
   `capture` key). A CPU test pins that loading the profile sets exactly
   those flags and nothing else, and that the profile is OFF unless
   explicitly selected.
2. **Cells** (leased, ≤ 285 s each; split the batteries into resumable
   segments as WC1/EB1 did): for each side {profile, defaults}: fresh
   capture of the three batteries' repositories; supersession (9),
   census (10), long-history (14); then a **restart** (persist, new
   process, reload from manifest) and re-score the same probes. Record
   summed token seats (B5) and the RT1 provenance fields (B2) in every
   receipt; an inherited/unpinned payload must be graded explicitly,
   never silently reharvested.
3. **Registered prediction** (Scout's): profile reproduces 9/9, 9/10,
   13/14 on fresh repositories and preserves them across restart;
   defaults reproduce the EB1 baseline (5/9, 9/10, 13/14). Registered
   acceptance for adoption as default: profile ≥ baseline on every
   battery, original-correct controls unchanged, restart retains
   metadata and scores. Registered fix-validation gate: the profile
   side reproduces the WC1 width-96 row exactly (9/9, 9/10, 13/14);
   any drop is attributed to the SCOUT-FIX-1 diffs first (bisect by
   reverting the four item commits one at a time, CPU-side reasoning
   only; the lead runs any GPU bisect).
4. Budget ≤ 0.8 GPU-h. Blocked-report, `--dry-run`, `lead_commands.txt`.

## Done (verbatim)
1. Registry path + sha; flag pin test names; cell list with estimates;
   blocked-report; exact lead commands.
2. CPU gate results (including the full existing CPU battery on this
   branch, counts pasted; the 11 missing-calibration-receipt failures
   are known and may be listed as such).
3. Prior art; deviations; RED; process safety; model id and effort.

## COMMON RULES
As in `orders/GRM_C3_DNGH_DECOY_CALIBRATION.md` §COMMON (additive only,
registration immutable, no GPU in the sandbox, leased-runner rails,
never kill anything, no git, no subagents, foreground, effort high).
