# GRM-C6 — What prevents lc1-wip → main integration? (worktree wt/grm-c6, branch grm-c6; REVIEW, 0 GPU)

Origin: `docs/GRM_SCOUT_2026-09-08.md` Part C rank 6. `lc1-wip` is 141
commits / 243 files ahead of `main` (`git diff --stat main lc1-wip`).
Registered prediction (Scout's): the first blockers are stale references
and an obsolete baseline frame, not a missing capability.

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/wt/grm-c6` — for the review
document and any CPU test you add; NO changes to `core/` or
`scripts/` in this order (findings become follow-up orders). You may
read `main` via `git show main:<path>` / `git diff main lc1-wip`
(read-only git is allowed here; no checkout, commit, merge, rebase or
branch operations).

## Mission
1. Produce `docs/GRM_LC1WIP_MAIN_INTEGRATION_REVIEW_2026-09-08.md`:
   (a) every default/flag whose value differs between the two branches
   and which receipt justifies the lc1-wip value; (b) stale references
   (paths, receipt names, orders, docs) that would break on main; (c)
   the registry/baseline frame each battery expects vs what the
   branches ship (`config/grm_live_registered_baselines.json` predates
   EB1 per Part A); (d) escapes: tests skipped, marked xfail, or gated
   on files only present in worktrees; (e) which CPU batteries pass on
   lc1-wip HEAD right now (run them; paste the counts), and which need
   the GPU acceptance profile (the Part C rank 2 profile: width 96 +
   live capture pin + near-live seating + RT1 reproducing 9/9, 9/10,
   13/14 across restart) before a merge is defensible.
2. Rank the blockers; state which are edits the lead can commit
   directly (docs/paths), which need a seat order, and which need a GPU
   acceptance run (≤ 0.4 GPU-h, cells listed with estimates).
3. Do not merge, rebase, or touch `main`.

## Done (verbatim)
1. The review doc path; the blocker table; battery counts as run.
2. Prior art; deviations; RED; process safety; model id and effort.

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
