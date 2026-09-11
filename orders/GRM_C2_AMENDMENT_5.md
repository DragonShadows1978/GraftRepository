# GRM-C2 amendment 5 (lead, 2026-09-09) — budget rail at 42/52 cells; cap raised to 4,800 s so long-history completes

Lead-run resume (epoch 3): 42/52 cells COMPLETE, 3,420.8 s charged of
3,600; `profile-longhistory-7` RED `budget rail: cannot reserve next
285-second cell` (registered stop). Results so far: sup profile 9/9 →
9/9 restart (575 seats, retained) vs defaults 4/9 → 3/9 (1,391 seats,
not retained); **census profile 9/10 → 9/10 restart (758 seats,
retained) vs defaults 5/10 → 5/10 (1,802 seats)**; long-history 13/14
probes served on both sides pre-restart, then the rail.

Lead decision: **cap raised to 4,800 s** (measured ~81 s per cell so
far → ~800 s for the 10 remaining cells; headroom for the restart
cells), nothing else changes. Sha-bound amendment read at run time by
the epoch-3 runner WITHOUT editing any file the runner or worker
executes (if the cap is baked into an executed file, say so and
register the amendment the way amendment 3's epoch did: a new epoch
directory is NOT wanted; the 42 completed receipts must remain the
evidence). Do not re-run the RED cell itself if it only failed on the
reservation check (its worker never started): state whether it is
re-eligible under the raised cap by the existing resume rule, or
register a one-time re-eligibility for reservation-only REDs. Refresh
`lead_commands.txt --resume` semantics accordingly. Same worktree,
same rules (no git, no subagents, no GPU, foreground, never kill
anything). Effort: high.

## Done (verbatim)
1. Amendment path + sha; where the cap is read; the RED cell's
   eligibility ruling; tests.
2. Exact lead command.
3. Prior art; deviations; RED; process safety; model id and effort.
