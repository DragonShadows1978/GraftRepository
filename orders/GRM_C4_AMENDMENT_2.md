# GRM-C4 amendment 2 (lead, 2026-09-09) — the per-segment "No serving observed" guard REDs a probe-free long-history segment; register a re-run rule

Lead-run so far (chain still running; do not touch `artifacts/grm_c4/runs/`):
sup and census units for `c64_w96` PASS; long-history segments
`c4-lh-01..05` PASS with bound session state; **`c4-lh-06` RED**:
`scripts/grm_c4_campaign.py:290` `assert any(e['event']=='attempt_residency' ...), 'No serving observed'`
fired for a segment whose turn range apparently contains no probe /
answer turn (receipt `artifacts/grm_c4/runs/c64_w96/longhorizon_c4-lh-06.json`,
events show geometry, capture and deposit-guard events, no serving).
The lead's reading: the guard is a per-CELL sanity check applied
per-SEGMENT, so a filler-only segment cannot pass it; every later
segment of that cell then lacks its predecessor's bound state, and the
same segment will RED in `c96_w64` and `c64_w64`. Confirm or refute
this from the battery's turn plan (which turns carry probes) before
changing anything; if the segment really should have served, say so
and stop.

Same worktree (r1 + amendment 1 committed), same rules (no git, no
subagents, no GPU, foreground, never kill anything). Effort: high.

## Mission (if the lead's reading holds)
1. Amendment JSON (sha-bound to this order and amendment_a3): the
   serving guard becomes per-cell (evaluated when the last long-history
   segment completes: at least one `attempt_residency` across the
   cell's segments; a probe-free segment PASSes if it advanced exactly
   its registered turn range and bound its session state).
2. **Re-run rule**: a long-history unit whose receipt `error` is exactly
   the string above, or whose predecessor segment was such a unit, is
   re-runnable ONCE under this amendment, resuming from the last PASS
   segment's bound session state; the RED receipts stay on disk
   (create-only), the re-run receipt carries the amendment sha. Any
   other RED remains the registered stop.
3. Scoring unchanged. Refreshed `lead_commands_a2.txt` containing ONLY
   the re-runnable units in order (per cell: from the first RED segment
   to the end) plus the per-cell `score`, executable after the current
   chain finishes.
4. CPU gates: guard per-cell semantics on a synthetic three-segment
   layout (probe-free middle segment PASSes; a cell with zero serving
   overall REDs); re-run eligibility exactly once; existing PASS
   receipts and the RED receipts byte-unchanged.

## Done (verbatim)
1. Your confirmation/refutation of the reading with the turn plan
   evidence; amendment path + sha; files/lines; test names.
2. Exact lead commands (`lead_commands_a2.txt`).
3. Prior art; deviations; RED; process safety; model id and effort.
