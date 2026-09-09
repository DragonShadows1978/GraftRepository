# GRM-C2 amendment 1 (lead, 2026-09-09) — budget raised to 3,600 s; make the lead commands executable

Your r1 is committed (see `git log`). The 52-cell plan estimates
3,140 s against the order's 2,880 s (0.8 GPU-h); you correctly refused
to execute. Lead decision: **cap raised to 3,600 s (1.0 GPU-h) for
C2**, no arm dropped, no cell shortened. Record the decision in a
sha-bound registration amendment and the ledger. Same worktree, same
rules (no git, no subagents, no GPU, foreground, never kill anything).
Reasoning effort: high.

## Mission
1. Amendment JSON binding this order, the r1 registration and the new
   cap; the runner's budget check reads the amended cap; a forged or
   stale amendment is refused (tests).
2. `lead_commands.txt` becomes executable in order: dry-run, then the
   52 cells one lease each (≤ 285 s worker / 590 s outer / 30 s
   cooldown between cells, foreground flock on `/tmp/forge-gpu.lock`),
   scoring per battery as soon as its cells complete, restart cells
   after their persist cells, and a final `summary` that prints the
   profile-vs-defaults table (sup 9 / census 10 / long-history 14, pre-
   and post-restart, token seats, RT1 provenance fields present).
   Stop-on-RED with a registered resume rule (next unstarted cell;
   a RED cell is not retried).
3. State the width question plainly in the summary: the defaults arm
   runs the shipped generic width 256 and the profile arm width 96, so
   the comparison is "shipped defaults vs proposed profile", not a
   width-matched pair; if a width-matched defaults arm (96, pin OFF,
   near-live OFF) fits inside the raised cap as extra cells, register it
   as arm `defaults96`; if not, register it as NON_FIT and say what it
   would cost.

## Done (verbatim)
1. Amendment path + sha; test names; cell count and total estimate.
2. Exact lead commands.
3. Prior art; deviations; RED; process safety; model id and effort.
