# GRM CUDA Bridge Overhead Ledger

This ledger is the execution record for the CUDA bridge overhead work
order. The immutable plan is `docs/GRM_CUDA_BRIDGE_OVERHEAD_PLAN.md`; the
wing's narrative continues in `docs/GRM_GEMV_ROUTER_SYNTHESIS.md`.

## 2026-07-07 23:30 EDT

Action: Work-order opening — plan drafted, shaped, and committed.

Repo state:
- Repository: `/mnt/ForgeRealm/GraftRepository`
- Branch: `grm-cuda-bridge-overhead` (cut from `codex/intn-model-ppl-sweep`
  at bbe74a0 + working-tree state; pre-existing dirty GPT-OSS files outside
  this slice untouched).
- Baseline receipts inherited from GRM_GEMV_ROUTER_LEDGER 2026-07-07: bridge
  3.157/11.218/38.961 ms vs direct route 0.129/0.255/0.770 ms at 32/128/512
  nodes; baseline instrument = scripts/grm_gqa_cuda_bridge_smoke.py with the
  exact ledger commands.

Findings (plan shaping, David 2026-07-07):
- Scope: GQA bridge only — that is where the optimization is needed; MLA
  and CPU route paths untouched.
- Architecture decision: C++ — grow the existing CUDA sidecar into the
  native bridge (hot/cold split, device-pointer route entry for
  forward-pass callers). Rust adapter rejected (adds a toolchain and
  boundary crossings; the concurrency story is already gated in C++). CPU
  C ABI stays CUDA-free per the wing's standing law.
- Doc structure per house precedent: new plan + new ledger per work order,
  one continuous wing synthesis.

Next action:
- P0: profile attribution of the bridge gap (Sonnet agent, flat,
  no-delegation rule).

## 2026-07-08 00:20 EDT

Action: P0 complete — attribution receipt (committed BEFORE any fix, per
plan and per Codex's registered point). Lead spot-checked the load-bearing
claim at source.

Findings (evidence class: profile receipt; cProfile over 300 reused route
calls, idle GPU, contended host — see caveat):
- E1 MET: ≥99% of the reused-call cost attributed to named functions;
  81.6% (512n) / 80.8% (128n) of the gap is ONE mechanism —
  `graft_arena.py:1513` `np.ascontiguousarray(np.stack(rows))` inside
  `_cuda_route_bank_inputs()`, which re-materializes the full dense route
  bank (268 MB at 512 nodes) on EVERY route call, before the signature
  compare that then discards it in steady state. Lead-verified at source:
  `_cuda_route_order` → `_cuda_route_bank_inputs()` unconditional.
- PRIME-SUSPECT CORRECTION (this is why P0 ran first): the plan's suspect
  (a) — per-call signature recomputation — is CHEAP (~0.06%, id-tuple
  compare; the blake2b hash is attach-only). An O(1) epoch check alone
  would NOT have closed the gap: the bank rebuild happens before and
  independent of the staleness check. P1 must cache the STACKED bank and
  rebuild only on staleness, with the epoch making the residual per-node
  walk (~1.7 ms, ~1%) O(1) as well — otherwise E2 (≤2× direct at 512n =
  ≤1.54 ms) is unreachable.
- Residuals: per-node Python loop ~1%; query marshal + ctypes ~0.6-1.2%;
  node-id mapping <0.1%.
- Caveat (registered): absolute ms in this profile run (~160 ms @512n) are
  ~4× the ledger receipts (38.96 ms) due to host memory-bandwidth
  contention (multiple live sessions, swap in use — raw 268 MB numpy copy
  benched at 0.8-1.6 GB/s on this box right now). Mechanism, percentages,
  and linear-in-N scaling are consistent across node counts and
  corroborated by an isolated same-shape np.stack microbench (163 vs
  158 ms). P3's re-receipt runs on the quiet machine with the exact ledger
  commands.
- Receipts: artifacts/grm_gqa_cuda_bridge/profile_{512n,128n}*.{pstats,txt,json},
  smoke_512n_recheck.json; profiling wrapper
  scripts/grm_gqa_cuda_bridge_profile.py (new file, product code untouched).

Next action:
- Commit this receipt. Then P1 (corrected design): cache (route_bank,
  node_ids, signature) — stack only on staleness; mutation-epoch integer
  bumped at every site that today clears/invalidates the CUDA bank plus
  graft add/retire/route-key replace in GQAArenaCache; hot path compares
  one int. Fail-closed equivalence regression REQUIRED: every mutation
  that changes the signature must bump the epoch (test asserts
  epoch-valid ⇒ signature-equal across the mutation battery).
