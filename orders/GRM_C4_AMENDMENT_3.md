# GRM-C4 amendment 3 (lead, 2026-09-09) — run the two remaining cells (c96_w64, c64_w64) under the A4 runner

Amendment 2 is committed (121b091) and its recovery
(`lead_commands_a2.txt`, c64_w96 segments 06–13 + score) is queued on
the card. Your report is right that `c96_w64` and `c64_w64` have no
receipts: their units were refused by the original campaign guard
(`scripts/grm_c4_campaign.py:266` "Campaign RED: no continuation
without amendment") and that guard will keep refusing them because the
lh-06..13 RED receipts stay on disk by design. Lead decision: **the two
remaining cells run under the A4 runner** (`scripts/grm_c4_resume_a2.py`
or a sibling), with the per-cell serving guard, the same batteries,
units, rails, scoring and budget as registered (cap 4,800 s; charge the
c64_w96 RED units' walls as recorded). Same worktree, same rules (no
git, no subagents, no GPU, foreground, never kill anything). Effort:
high.

## Mission
1. Amendment JSON (sha-bound to this order and amendment_a4) that
   registers the A4 runner as the executor for every not-yet-started
   unit of `c96_w64` and `c64_w64`, create-only receipts in the same
   layout the scorer reads, and the per-cell `score` for each.
2. `lead_commands_a3.txt`: all units of both cells in the registered
   order (sup, census, long-history segments with bound-state resume),
   cooldowns, then `score --cell` for each, then a final cross summary
   that prints the 2×2 (chunk × width) with (96,96) from the WC1 row,
   plus the (64,64)-fixed vs (64,64)-historical comparison and the
   registered prediction outcome.
3. CPU gates: dry-run enumerates the units; forged/stale amendment
   refused; existing receipts byte-unchanged; the cross summary on a
   synthetic four-cell layout.

## Done (verbatim)
1. Amendment path + sha; unit count and estimate; files/lines; tests.
2. Exact lead commands.
3. Prior art; deviations; RED; process safety; model id and effort.
