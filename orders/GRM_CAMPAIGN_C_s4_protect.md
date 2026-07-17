# CAMPAIGN-C: spill_policy="s4_protect" (hit-protected LRU) + gate leg

YOUR WRITABLE TARGETS in /mnt/ForgeRealm/GraftRepository (main tree):
- core/graft_repository.py
- tests/test_grm_s4_demotion.py
Everything else READ-ONLY (sibling seats work other files in this
tree — never touch core/graft_arena.py or
tests/test_grm_runtime_lifecycle.py). No git. No subagents. No GPU /
no model loads. RED honesty; no monitor-idling.

## Law
docs/GRM_S4_LEDGER.md — read in full. The 2026-07-17 s4_protect
registration section IS your spec: LRU stays PRIMARY; n_grounded>0
nodes are PROTECTED from spill while any unprotected candidate
remains; protection yields to pure LRU only when all residents are
protected. The G2-S4 FAIL entry above it explains the mechanism your
design must avoid (never punish young zero-hit nodes — they get
default LRU treatment). Also read the existing spill_policy="s4"
implementation and the A/B harness in your two files — s4_protect is
a THIRD policy alongside, changing nothing about "lru" or "s4".

## Build
1. spill_policy="s4_protect" in the paging path, per the registered
   semantics; validated like the existing policy strings.
2. Harness: accept --policy s4_protect; verdict analysis compares
   any named policy pair (default s4_protect vs lru) against the
   REGISTERED pass conditions (recall ≥, page-ins ≤); receipts keep
   the same schema with the policy field distinguishing arms.
3. CPU units: protection ordering (protected survive while
   unprotected exist), all-protected degrades to pure LRU,
   flag-off/other-policies byte-identical behavior, no starvation/
   no-jam cases.

## Verify
py_compile; python3 -m pytest tests/test_grm_s4_demotion.py tests/test_grm_s4_ledger.py tests/test_grm_fold_recovered_guard.py -q — paste summary.

## Done
Verbatim: diff summary; pytest line; exact --run-gpu commands for the
lead (s4_protect arm + analyze pairing vs the EXISTING lru receipts if
the fingerprint allows reuse — state whether it does); ambiguities;
"no git, no GPU, no files outside grant".
