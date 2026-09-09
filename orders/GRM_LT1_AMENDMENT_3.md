# GRM-LT1 amendment 3 (lead, 2026-09-09) — rebind the registration to the cherry-picked FIX-4 and FIX-5 core

Your FIX-6 report said this checkout lacked FIX-4 and preflight stopped
on it. The lead cherry-picked SCOUT-FIX-4 (recency binder served from
mount) and SCOUT-FIX-5 (fold prompt enumerates facts) from grm-c7 onto
this branch (see `git log -3`). Now `--preflight` fails
`INPUT_SHA_MISMATCH: core/graft_arena.py` because the LT1 registration
pins the pre-cherry-pick core shas. Register a sha-bound amendment
(this order + amendment 2 + old/new shas of every pinned core file)
that binds the new source state; receipts carry the amended shas; the
FIX-4 preflight check must now PASS; forged/stale amendment refused
(tests). Nothing else changes: fixture, arms, flag, cap, cells. Run the
FIX-4/5/6 + LT1 CPU suites on this branch and paste counts. Same
worktree, same rules (no GPU, no git, no subagents, foreground, never
kill anything). Effort: high.

## Done (verbatim)
1. Amendment path + sha; old/new core shas; test names; suite counts.
2. Exact lead command (must pass `--preflight` and `--dry-run` on CPU).
3. Prior art; deviations; RED; process safety; model id and effort.
