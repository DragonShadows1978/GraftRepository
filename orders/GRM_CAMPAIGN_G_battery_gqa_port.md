# CAMPAIGN-G: supersession battery GQA dialect switch (section 6.2 prep)

YOUR WRITABLE TARGETS in /mnt/ForgeRealm/GraftRepository (main tree):
- tests/test_grm_supersession_battery.py
- tests/fixtures/supersession_battery/ (additions only if needed —
  existing fixture files are gate-registered data, do NOT edit them)
Everything else READ-ONLY (a sibling seat works
core/graft_repository.py — never touch it, nor core/graft_arena.py).
No git. No subagents. No GPU / no model loads. RED honesty; no
monitor-idling.

## Law
docs/GRM_GQA_REGATES_L1_PLAN.md (immutable) Phase 6.2 IS the spec;
also docs/GRM_SUPERSESSION_PLAN.md + LEDGER (the MLA battery you are
porting, its receipt schema, the frozen L1 form already implemented
in core/graft_arena.py, and the L2 resolution flags).

## Build
1. --dialect mla|gqa switch in the battery harness: gqa loads
   Qwen3-4B via the GQA arena (mirror tests/test_graft_gqa_arena.py
   mechanics: GQAArenaCache, layer-0 |q.k| router, live_shift,
   early-stop decoding). Same fixtures, same probes, same receipt
   schema with a dialect field. Baseline/-‑debias/--resolve flags all
   work under gqa (L1 debias and L2 resolution are dialect-surface
   features — verify they engage on the GQA path by code reading and
   state where; if either does NOT plumb through GQAArenaCache,
   STOP on that lever and report exactly what's missing rather than
   patching core/).
2. CPU units: dialect selection, mla default unchanged, receipt
   schema round-trip with dialect field.

## Verify
py_compile; python3 -m pytest tests/test_grm_supersession_battery.py tests/test_gqa_ragged_cuda_bank.py -q — paste summary.

## Done
Verbatim: diff summary; pytest line; exact lead GPU commands for
gqa baseline/debias/resolve legs; the L1/L2-engagement code-reading
receipts (file:line); ambiguities; "no git, no GPU, no files outside
grant".
