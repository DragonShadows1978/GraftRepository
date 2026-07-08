# GRM GEMV Router Synthesis

The GEMV router work is closed as an implementation track for the current GRM
runtime. The original per-node scan has been replaced or bypassed by measured
host routing paths:

- MLA uses a contiguous fp32 route-key arena plus INT4 bulk/refine routing.
- GQA uses native key-bank routing with segment-style raw `|q.k|` scoring.
- Prepared epoch snapshots give clean route reads under mutation.
- CUDA is no longer only a probe: there is a runtime-facing sidecar,
  explicit native bank attachment, invalidation on route/eligibility mutation,
  an opt-in `GRM_GQA_CUDA_ROUTE=1` arena bridge, and route-bank lifecycle
  signatures to avoid stale GPU row reuse. The bridge has now also passed live
  GPU validation on real Qwen3.5-2B layer-3 capture banks at 32, 128, and 512
  nodes.

The practical result is good enough for the GRM paper's routing limitation:
hundreds to millions of memories are now in measured interactive territory on
the host path, and the remaining performance questions are targeted kernel work
rather than missing runtime architecture.

## What Is Still Left

1. Decide whether CUDA remains limited or becomes general routing.

   The current CUDA bridge is intentionally a limited mount-window path with
   `topk <= 16`, dense same-shape single-key banks, and no lexical/filter policy
   in the CUDA sidecar. That is correct for fast mount selection. Replacing
   general `arena.route()` would require a larger-topk/full-rank route contract,
   filter semantics, and broader fallback behavior.

2. Only build another GQA kernel if the product target needs it.

   Host-side GQA full-bank routing improved, but sub-10ms larger real-capture
   routing would need a lower-level GEMM/BLAS-style layout or a packed
   microkernel. Several scalar/transposed host experiments were parity-green
   but slower and should stay diagnostic-only.

3. Broaden model/capture coverage before making a general GQA claim.

   Qwen3.5-2B capture sweeps are green for the current evidence set. A broader
   full-bank correctness claim should include another model or capture family.

## What Is Not Left

- The dependency-free CPU C ABI does not need to take a CUDA dependency.
- The host router no longer needs a rewrite to get out of per-node scan mode.
- ANN/IVF indexing remains out of scope until roughly 10M nodes.
- The current stale CUDA route-bank bug is fixed: route-bank snapshots now force
  reattachment when dense GQA rows or native node-id mappings change.
- A live GPU bridge receipt is no longer missing: 32, 128, and 512-node
  Qwen3.5-2B capture-bank smokes passed parity on the 4070 Super.

## Current Evidence

- Previous non-GPU closure gate:
  `tests/test_grm_router_baseline.py tests/test_grm_native_runtime.py -q` ->
  `109 passed, 2 warnings in 344.94s`.
- 2026-07-07 focused CUDA lifecycle gate:
  six selected native/GQA CUDA-route tests -> `6 passed, 2 warnings in 9.65s`.
- 2026-07-07 full native-runtime gate:
  `tests/test_grm_native_runtime.py` -> `90 passed, 2 warnings in 248.88s`.
- 2026-07-07 live CUDA bridge gate:
  `scripts/grm_gqa_cuda_bridge_smoke.py` on real Qwen3.5-2B layer-3 capture
  banks -> parity true at 32, 128, and 512 nodes. Reused bridge min wall:
  `3.15747ms`, `11.217836ms`, and `38.960699ms`; direct CUDA device/query:
  `0.098496ms`, `0.226304ms`, and `0.740352ms`.

## Decision

Treat the GEMV router implementation as closed at the opt-in CUDA bridge
boundary. The live GPU bridge receipt now exists. The next work item is a
deliberate policy decision on whether CUDA should stay as the mount-window
accelerator or become a full-rank general route backend.

## Wing Continuation: The Bridge Overhead Work Order (opened 2026-07-07)

The closure receipts themselves exposed the next defect: the router is
fast and the bridge is not. The same smoke that proved parity showed the
Python bridge costing 25-50× the device work (3.2/11.2/39.0 ms against
0.13/0.26/0.77 ms direct at 32/128/512 nodes), scaling linearly with node
count — per-call O(N) host work re-deriving staleness facts the mutation
paths already know. A successor work order
(`GRM_CUDA_BRIDGE_OVERHEAD_PLAN.md` + its own ledger) attacks exactly
that: O(1) epoch staleness, then a hot/cold split that promotes the CUDA
sidecar into the native bridge with a device-pointer route entry — David's
architecture call: C++, not a new-language adapter, and the CPU C ABI
keeps its CUDA-free law. The registered target is a bridge that costs no
more than 2× the direct route at any node count, with overhead flat in N.

The work order closed in one night, and the profile-first discipline paid
immediately: P0 attributed 82% of the gap to a single line — the dense
bank re-stacked (268 MB at 512 nodes) on every reused call and then
discarded by the very signature check meant to avoid work — and corrected
the plan's own suspect (the signature compare was cheap; the epoch fix
alone would have missed). P1 cached the bank behind the signature; P2 made
a mutation epoch the sole hot-path gate (every graft_repository mutation
site instrumented through a choke point, paranoid-mode cross-check kept
for tests) and gave the sidecar a device-pointer route entry, so a
forward-pass caller's queries never visit host memory. Steady state now:
one integer compare, one C call, one kernel. Clean-window receipts:
0.185/0.339/0.966 ms bridged at 32/128/512 nodes against
0.128/0.261/0.770 direct — 1.26-1.44×, ratio improving with N. That is
17-40× over the closure receipts that opened this work order, parity
green throughout, 95/95 tests. One checkbox remains: a quiet-machine
stamp of the exact smoke commands (small-node points inflate under host
contention — receipted, not disputed).

Where the wing points next, from David's own framing — routing is
attention over the model's attention states, so every trick that made
attention fast applies one level up: an MLA production-path profile (the
1M-node receipts timed the native call, never the Python wrapper around
it); synthetic GQA route centroids (the payload law from SCRIBE/
Translation protects grafts, not search structures — a centroid is an
index over witnessed keys, exactness-gated two-tier keeps the law); and
the per-graft incremental index, whose choke-point foundation W1 already
laid.

## Wing Continuation II: The MLA Order (opened and closed 2026-07-08)

The third work order asked whether MLA was "CUDA routing properly" and
found it wasn't routing properly at all — on two levels nobody had
receipts for. The production path had never been timed (925.6 ms at 1M,
not the harness's 23.9), and the production library had never been
compiled with optimization (-O0 since inception; -O3 alone returned 21×).
Three profile-guided passes later — limit-aware topk with a conservative
replacement for the completeness law, epoch-cached maps on the mutation
choke points, then a device-resident centroid arena in the sidecar —
the same production call routes a million nodes in 2.22 ms, byte-exact
across the operational envelope, with one flagged fp32 near-tie
reordering at the 1M scaling demonstration (David's ruling: known,
accepted, frozen-model repositories never approach that scale). Both
dialects now route on CUDA under one epoch law, and the wing's original
promise — repository-scale memory at interactive latency — holds where
production actually calls it.
