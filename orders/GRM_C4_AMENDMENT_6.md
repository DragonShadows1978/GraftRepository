# GRM-C4 amendment 6 (lead, 2026-09-09) — do not attempt registered NON_FIT units; cap 6,600 s

Your amendment 5 is committed. Its finding stands on its own: under
64-token chunking the late long-history turns cost 225–348 s each
(turn 90 alone 225.5 s), so those turn ranges cannot run inside the
standard 285 s lease. The lead does NOT authorize long leases for
them. Lead decisions:
1. Units whose registered planning estimate exceeds the rail (lh-12a,
   lh-13b, and their counterparts in `c96_w64` / `c64_w64`) are
   **registered NON_FIT and never attempted** (no 285 s charge for an
   attempt that cannot complete); the A5 runner skips them and the
   cross summary reports the cell's long-history battery as NON_FIT
   with the completed segments listed, exactly as amendment 5 already
   prints partial cells.
2. Cap raised to **6,600 s** so every fitting unit of all three cells
   runs (state the new projection: recorded + fitting units).
3. Everything else unchanged; `lead_commands_a5.txt` may be regenerated
   as `lead_commands_a6.txt` (the lead has not queued a5 yet).
Same worktree, same rules (no git, no subagents, no GPU, foreground,
never kill anything). Effort: high.

## Done (verbatim)
1. Amendment path + sha; skip list; projection vs 6,600; files/lines;
   tests (skip semantics, forged/stale refused, receipts unchanged).
2. Exact lead commands (`lead_commands_a6.txt`).
3. Prior art; deviations; RED; process safety; model id and effort.
