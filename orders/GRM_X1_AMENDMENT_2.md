# GRM-X1 amendment 2 (lead, 2026-09-08) — the lead continuation: bind the amended sources and run the campaign

Amendment 1 is committed (e76dec5). Your `fingerprint()` correctly
refuses to start on drifted sources, and `next_cell()` correctly stops
on the r1 RED `oracle_m1_s0`. Both blockers are now the lead's decision,
and the decision is: **continue the campaign on the amended sources**,
with the r1 RED preserved as evidence and the retry paid for
explicitly. Same worktree, same rules (no git, no subagents, no GPU,
foreground only, never kill anything). Reasoning effort: high.

## Registered by this amendment
1. **Continuation manifest** `artifacts/grm_x1/continuation_02.json`
   (create-only, sha-bound to this order, to
   `payload_amendment_01_handoff.json`, to `registration.json`, and to
   the amended source shas). `fingerprint()` accepts EITHER the original
   handoff OR a continuation whose recorded source shas match the
   current tree; the receipt fingerprint becomes the continuation sha
   so r2 receipts are never confused with r1 (state the receipt
   directory / naming you use; r1's `oracle_m1_s0` receipt stays where
   it is, untouched).
2. **Retry rule**: a cell whose ONLY receipt is a RED bound to the r1
   fingerprint is eligible under the continuation exactly once; a RED
   under the continuation fingerprint is the registered stop again (no
   second retry). `summary` reports r1 and r2 receipts separately and
   the campaign verdict from r2 only, with the r1 RED listed under
   `historical_red`.
3. **Budget**: the lead authorizes the extra 285 s worker reservation
   for the retried cell: total campaign reservation 2565 + 285 = 2850 s
   (0.79 GPU-h, over the COMMON 0.75 by the lead's explicit decision;
   record it in the continuation manifest and the ledger).
4. CPU gates: continuation accepted / forged continuation refused /
   stale continuation (source sha mismatch) refused / second retry
   refused / r1 receipt untouched; `--dry-run` enumerates the 9 cells in
   order with the retry flagged; refreshed `lead_commands.txt` that is
   actually executable in order (`preflight`, then `run oracle_m1_s0` …
   `run natural_m100`, `summary`).

## Done (verbatim)
1. Continuation manifest path + sha; files/lines changed; the five gate
   test names and results.
2. Exact lead commands.
3. Prior art; deviations; RED; process safety; model id and effort.
