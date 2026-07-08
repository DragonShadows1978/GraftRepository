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
