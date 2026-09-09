# GRM-C4 amendment 4 (lead, 2026-09-09) — cap raised to 5,400 s so the cross completes

Your amendment 3 projects 4,924 s of recorded + registered work against
the 4,800 s cap, which would stop the runner inside `c64_w64`'s
long-history battery, leaving the (64,64) fixed-geometry diagonal
unscored. Lead decision: **cap raised to 5,400 s (1.5 GPU-h) for C4**,
nothing else changes. Same worktree (a3 committed), same rules (no git,
no subagents, no GPU, foreground, never kill anything). Effort: high.

## Mission
1. Sha-bound amendment JSON (this order + amendment_a5) carrying the
   new cap; BOTH the A2 recovery runner and the A3 remaining runner
   read it (the A2 recovery chain is already queued using
   `lead_commands_a2.txt` and the A3 chain using
   `lead_commands_a3.txt`; those files must stay valid unchanged, so
   the cap must be read from the amendment at run time, not baked into
   the command files). Forged / stale amendment refused (tests).
2. Refresh the accounting statement in the ledger (recorded + registered
   vs 5,400).

## Done (verbatim)
1. Amendment path + sha; files/lines; test names and results;
   confirmation that `lead_commands_a2.txt` and `lead_commands_a3.txt`
   are byte-unchanged.
2. Prior art; deviations; RED; process safety; model id and effort.
