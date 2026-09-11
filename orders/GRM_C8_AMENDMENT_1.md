# GRM-C8 amendment 1 (lead, 2026-09-09) — cap raised to 2,700 s; SP5 citation corrected

Your r1 is committed. The 24-cell forecast is 2,591 s against the
order's 1,800 s; you correctly refused. Lead decision: **cap raised to
2,700 s (0.75 GPU-h)**, no cell dropped. Your SP5 correction is accepted:
cite the SP5 decode table at its actual context (2,048 tokens; APA arms
1.4–1.6 ms/token slower than standard) and say the 12,288-token figure
in the order was the lead's error. Same worktree, same rules (no git,
no subagents, no GPU, foreground, never kill anything). Effort: high.

## Mission
1. Sha-bound amendment JSON with the new cap; preflight reads it;
   forged/stale amendment refused (tests).
2. `lead_commands.txt` executable in order with a `--resume` that skips
   started cells (RED never retried) and a final `summary` printing the
   per-stage table (mean / p50 / p95), decode share of turn wall, the
   demand-trip turn, and the registered decision rule outcome
   (decode < 50% ⇒ session routing/admission wins).
3. Side B restated from SP5 at 2,048 tokens with its evidence class.

## Done (verbatim)
1. Amendment path + sha; test names; total estimate under the cap.
2. Exact lead commands.
3. Prior art; deviations; RED; process safety; model id and effort.
