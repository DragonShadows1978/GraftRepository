# CAMPAIGN-B: route-latency seams 2/3 — ragged CUDA bank engagement + re-prep profile

YOUR WRITABLE TARGET is the WORKTREE
/home/vader/GraftRepository-route-seams (branch grm-route-seams) —
edits AUTHORIZED anywhere inside it EXCEPT docs/ plan files. The main
tree at /mnt/ForgeRealm/GraftRepository is READ-ONLY reference
(sibling seats are editing it in parallel — never touch it). No git
(lead commits). No subagents. No GPU runs / model loads — CPU
verification only; the lead runs all GPU gates serially. RED honesty;
no monitor-idling.

## Law
docs/GRM_ROUTE_SEAMS_PLAN.md (in your worktree) — the plan is
immutable and IS the spec: scope (seam-2 bank engagement, seam-3
re-prep profile, lex rescore measure-first), the registered E3 gate
(route wall ≤5 ms at ≥37 nodes on the flagged path, byte-exact parity
with python at k∈{1,3,5,16}, all routing regressions green), and the
non-goals (no default flip, no MLA, no GPT-OSS). Also read:
docs/GRM_E2E_RECEIPT_LEDGER.md (the 2026-07-08 seam decomposition —
the ~756 ms breakdown you are attacking), docs/
GRM_GQA_EXACT_RAGGED_CUDA_{PLAN,LEDGER}.md (the merged router you are
wiring: core/grm_cuda_router.py + tests/test_gqa_ragged_cuda_bank.py),
and the GQA route path in core/graft_arena.py (GQAArenaCache).

## Build
1. SEAM-2: opt-in flag on the GQA arena route (e.g.
   route_backend="cuda_ragged") that routes candidate scoring through
   the merged exact ragged bank instead of the python path. Byte-exact
   score/rank parity is the correctness demand; tie handling must
   match the python path exactly (the router's dev gate proved zero
   tie drift — preserve it through your wiring). Bank lifecycle must
   respect the epoch law (_bump_cuda_gqa_epoch invalidation — study
   how swap/rollback bump it).
2. SEAM-3: instrument step()'s route path per-turn (build receipts:
   what gets rebuilt each turn — candidate key marshaling, lex keys,
   centroids, bank uploads); cache what is invariant under the epoch
   discipline; parity-gate every elimination.
3. Lex rescore: measure its share; optimize ONLY if top-2 after 1+2.
4. Gate harness (tests/test_grm_route_seams_gate.py, --run-gpu, lead
   executes): drives a ≥37-node GQA arena session (Qwen3-4B; mirror
   tests/test_graft_gqa_arena.py mechanics), measures route wall per
   turn python vs flagged path, asserts parity, emits JSON receipts
   (schema grm_route_seams_e3_v1) with per-term timing decomposition.
   Include a warm-up per the first-run law before any timed capture.

## Verify
py_compile everything touched; CPU units for the wiring logic (flag
off = python path byte-identical by construction, epoch invalidation,
marshaling round-trip); run in YOUR worktree:
python3 -m pytest tests/test_grm_route_seams_gate.py tests/test_gqa_ragged_cuda_bank.py tests/test_grm_fold_recovered_guard.py -q
(CUDA-allocating constructors fail without a GPU in sandbox — known
artifact, report the exact signature and move on.)

## Done
Verbatim: diff summary per file; pytest line; the seam-3 rebuild
inventory (what step() rebuilds per turn, what you cached, what you
left); exact --run-gpu command(s) for the lead; ambiguities;
"no git, no GPU, main tree untouched".
