# ORDER NC17-P4 — GRM proof on Qwen3-1.7B (multi-turn, not production)

YOUR WRITABLE TARGET is /mnt/ForgeRealm/GraftRepository — GQA-dialect
graft hooks in `core/qwen3_1p7b_tc.py` (mirror the Qwen3-4B pattern),
NEW files under `tests/nc17/` and `logs/nc17/`. /tmp writable. Engine
checkouts READ-ONLY. Weight format: the lead names the best-standing
format from P2/P3 in the dispatch message; default bf16 if unstated.

Read the plan (immutable), the ledger tail, and the existing GQA graft
machinery BEFORE writing code: `core/kv_graft.py`, `core/graft_arena.py`
(GQAArenaCache), `core/qwen3_tc.py` hooks (`_capture`, prefill-only
injection, `graft_seats`, `live_shift`), and the E4-arena test pattern
(`tests/test_graft_gqa_arena.py`).

## Task — Phase P4 (proof, explicitly not production)

1. Wire the GQA graft/arena hooks into the 1.7B adapter (capture,
   inject, live_shift; per-head qk-norm interactions same as the 4B).
2. Gates, in order:
   a. Graft-vs-in-context equivalence: harvested document graft mounted
      at scale 1.0 == in-context logits (margin protocol; bottom-right
      mask law).
   b. STATE: save/restore bit-identical continuation (session
      multiplexing is the product mechanism this proves).
   c. E4-class multi-turn: 20 turns, 6 planted facts, routed mounts
      (layer-0 |q·k| router per the GQA law), swaps + evictions on one
      persistent cache, recall table, amnesia control.
3. Deliverables: printed gate results + `logs/nc17/p4_summary.json`;
   residency numbers per turn (the bounded-residency claim is the
   product-relevant one).

## Rails

Foreground GPU under flock, timeout 590, setsid, no detach; no git; no
subagents; no engine edits; RED honesty. A recall miss is a numbered
result, not a failure to hide — report the table as measured.

## Done — gate results with numbers, recall + residency tables, file
paths, deviations. If a gate is structurally blocked (e.g. an adapter
hook can't mirror the 4B pattern), report BLOCKED with the exact seam.
