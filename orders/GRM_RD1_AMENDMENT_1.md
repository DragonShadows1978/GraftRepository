# GRM-RD1 amendment 1 (lead, 2026-09-09) — cut the contrast to the informative minimum and implement the replay worker

Your r1 registered 520 requests (5 arms × memory+oracle rows × 46
probes) at 3.28 GPU-h, NON_FIT at 0.8, and left the numerical
checkpoint replay unimplemented (transport only). Lead decisions:
1. **Arms:** A0 (baseline reproduction), A2 (abstention instruction
   removed), A4 (reasoning medium + instruction removed). Drop A1 and
   A3 (registered NOT RUN with their estimates). Rows: memory rows for
   the 30 mounted-but-unread probes; oracle rows for the 16 oracle
   failures only. That is 46 × 3 = 138 requests; project it from your
   per-request estimate and state it.
2. **Cap 1.0 GPU-h** (lead authorization). If 138 requests exceed it,
   register the overflow NON_FIT, do not trim further.
3. **Implement the replay worker for real:** load the checkpoint
   preceding the probe turn, restore the exact mounts recorded in the
   probe's residency row (assert equality), run the probe turn through
   the production `_attempt` path with the arm's prompt/reasoning
   changes only, write the receipt. A0 must reproduce r2's served text
   for each probe (byte-equal or explain the difference; that is the
   validity gate for the whole contrast). CPU: prove the worker's
   control flow on the fake model with a fake checkpoint; the
   numerical reproduction is the lead's first GPU cell.
4. `lead_commands.txt` executable: A0 reproduction cells first (stop if
   any A0 row differs from r2), then A2, then A4, per-batch leases,
   then `summary` with the arm × class × {memory, oracle} table and the
   registered verdict rule (≥ +8/30 exact, ≤ 2 wrong).
Same worktree, same rules (no GPU, no git, no subagents, foreground,
never kill anything). Effort: high.

## Done (verbatim)
1. Amendment path + sha; request count and projection; worker
   file/lines; CPU gate names.
2. Exact lead commands.
3. Prior art; deviations; RED; process safety; model id and effort.
