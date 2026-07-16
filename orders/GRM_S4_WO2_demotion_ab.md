# S4-WO2: G2-S4 demotion A/B harness (S4-aware spill vs LRU)

YOUR WRITABLE TARGETS in /mnt/ForgeRealm/GraftRepository (main tree):
- core/graft_repository.py
- tests/test_grm_s4_demotion.py (new)
Everything else READ-ONLY. No git, no subagents, no GPU runs — CPU
verification only; lead runs the gate. RED honesty; no monitor-idling.

## Law
docs/GRM_S4_GROUNDING_LEDGER_PLAN.md + docs/GRM_S4_LEDGER.md (the
G2-S4 thresholds are REGISTERED there — implement exactly that gate,
change nothing about it). Read both in full first.

## Build
1. Policy hook in the VRAM paging path (study _page()/LRU over
   last-mounted with write-back-before-spill): opt-in
   spill_policy="s4" — zero-hit nodes (n_grounded==0) spill FIRST,
   LRU tiebreak within equal hit classes; default remains pure LRU
   (byte-identical behavior when the flag is off — that is a hard
   constraint like the telemetry tap's).
2. Gate harness tests/test_grm_s4_demotion.py behind --run-gpu:
   drive the 4 repeat-probe convos as ONE repository session with
   vram_budget forcing >=3x overcommit (report page-in counts to
   prove paging fired); run twice — policy A (lru) vs policy B (s4)
   — same budget, same session, same probes; JSON receipts per the
   registered PASS conditions (late-probe recall, page-in counts).
   An --analyze mode prints the verdict vs the registered
   thresholds.
3. CPU units: policy ordering logic (zero-hit-first, LRU tiebreak,
   flag-off = pure LRU), no-jam under all-zero-hit and all-hit
   populations, receipts schema round-trip.

## Verify
py_compile; python3 -m pytest tests/test_grm_s4_demotion.py tests/test_grm_s4_ledger.py tests/test_grm_fold_recovered_guard.py tests/test_grm_importance_salience.py -q — paste the summary line.

## Done
Verbatim in final message: diff summary per file; pytest line; exact
--run-gpu commands for the lead; spec ambiguities; "no git, no GPU,
no files outside grant".
