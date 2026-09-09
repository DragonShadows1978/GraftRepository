# GRM-C4 amendment 1 (lead, 2026-09-09) — register the fixed-geometry (64,64) diagonal; cap raised to 4,800 s

Your r1 finding is accepted: WC1's historical width-64 row was captured
under a different numeric geometry, so it cannot serve as the (64,64)
diagonal of a fixed-geometry cross. The lead's decision: **the cross
is run complete under fixed geometry.** Cap raised by the lead to
4,800 s (1.33 GPU-h) for this order; record the decision in the
registration amendment and the ledger. Same worktree
(`/mnt/ForgeRealm/wt/grm-c4`, your r1 committed as 69b406e), same
rules (no git, no subagents, no GPU, foreground, never kill anything).
Reasoning effort: high.

## Mission
1. Amendment JSON (immutable, sha-bound to this order and to
   `registration.json`) registering cell `c64_w64` (chunk 64, width 64)
   under the SAME fixed capture/seat/live geometry as `c64_w96` and
   `c96_w64`, same three batteries, same per-unit rails (285 s worker /
   590 s outer / 30 s cooldown), planning estimate 1,600 s. The (96,96)
   diagonal: state whether the existing WC1 width-96 row satisfies the
   fixed-geometry requirement (it is the RS3 geometry by construction if
   so; show the pin) or register `c96_w96` too and say what the total
   becomes. Do not silently reuse a mismatched row.
2. Registered prediction unchanged (chunking, not narrowing: (64,96)
   recovers t33 and keeps Juniper; rejection if the improvement needs
   width 64). Add the diagonal comparison the cross makes possible:
   (64,64) vs WC1-historical (64,64) at 8/9, 10/10, 14/14 tells whether
   the historical benefit survives fixed geometry at all.
3. CPU gates for the new cell(s) (dry-run enumerates them; existing r1
   receipts none yet, so no receipt validity question); refreshed
   `lead_commands.txt` with ALL cells in an order that lets the lead
   score each cell as soon as its units complete.

## Done (verbatim)
1. Amendment path + sha; cells and estimates; the (96,96) ruling.
2. CPU gate results; exact lead commands.
3. Prior art; deviations; RED; process safety; model id and effort.
