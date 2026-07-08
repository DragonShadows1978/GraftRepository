# GRM CUDA Bridge Overhead — Implementation Plan

Status: immutable at initial commit. Successor work order to the closed
`GRM_GEMV_ROUTER_PLAN.md` (P0-P6). Tracking artifacts:
- Operational ledger (new, this work order):
  `docs/GRM_CUDA_BRIDGE_OVERHEAD_LEDGER.md`
- Narrative synthesis (continues the wing's existing narrative):
  `docs/GRM_GEMV_ROUTER_SYNTHESIS.md`
House laws in force unchanged from the router plan:
measure don't model; commit-per-phase; gate-per-phase; NaN law (M6); C ABI
contracts unchanged; Unicode/time never cross the ABI; dependency-free CPU
router remains the default — CUDA stays opt-in.

## Problem (receipted, ledger 2026-07-07)

The opt-in `GRM_GQA_CUDA_ROUTE=1` bridge is parity-green but the Python
layer costs 25-50× the device work:

| Nodes | Direct CUDA route wall | Reused bridge min wall | bridge overhead |
| ---: | ---: | ---: | ---: |
| 32   | 0.129 ms | 3.157 ms  | 3.03 ms  |
| 128  | 0.255 ms | 11.218 ms | 10.96 ms |
| 512  | 0.770 ms | 38.961 ms | 38.19 ms |

Overhead scales linearly with node count → O(N) host work per route call.
Prime suspect (P4 lifecycle note): the per-call route-snapshot signature
(iterates native node ids, key shapes, dtypes, route-key object identities)
plus the per-call dense-bank eligibility walk — both re-derive facts the
mutation paths already know at mutation time (every route-key/active/
metadata/revision/expire mutation already explicitly closes the CUDA bank).

## Constraints (binding)

- Invalidation semantics preserved EXACTLY and fail-closed. The 2026-07-07
  stale-route-bank fix must not regress: any staleness scheme must be at
  least as conservative as the current signature — a route may never reuse
  a GPU bank that the current code would reattach.
- Mount-window contract unchanged (topk ≤ 16, dense same-shape single-key
  banks, lexical queries fall back). Growing the contract is a separate,
  later decision (synthesis "What Is Still Left" #1) — out of scope here.
- CPU route path untouched; all changes live in the opt-in bridge layer.

## Phases

P0 — Attribution receipt. Profile one bridged route call (cProfile and/or
py-spy over `scripts/grm_gqa_cuda_bridge_smoke.py`, the existing baseline
instrument, exact ledger commands at 32/128/512 nodes). Deliverable: a
table attributing the 3.03/10.96/38.19 ms gap to named functions, committed
before any fix. If the signature+eligibility walk does NOT dominate, the
plan's prime suspect is wrong — record it and retarget from the profile.

P1 — O(1) staleness. Replace per-call O(N) signature recomputation with a
monotonic mutation epoch on the store/arena: every mutation that today
closes or invalidates the CUDA bank also bumps the epoch (O(1) at mutation
time); a bridged route compares one integer against the epoch captured at
attachment. The existing signature machinery stays for the ATTACH moment
(computing it once per attachment is fine — it is per-call recomputation
that is the defect). Gate: lifecycle selector 6/6 (incl.
`test_gqa_arena_rebuilds_cuda_route_bank_when_rows_change`), full
`tests/test_grm_native_runtime.py` green, plus a NEW regression proving a
mutation that would change the signature also bumps the epoch (fail-closed
equivalence).

P2 — Native hot path (ARCHITECTURE DECISION, David 2026-07-07: C++, not
Rust; CPU C ABI stays CUDA-free). Promote the existing CUDA sidecar into
the native bridge by splitting hot from cold:
- Cold path (Python, unchanged in spirit): bank build, dense-bank
  eligibility walk, signature, attach, invalidate — once per mutation
  epoch, off the route path.
- Hot path (one C call): epoch integer check + route + top-k node ids
  out. Add a sidecar route entry that accepts DEVICE query pointers so a
  forward-pass caller whose queries already live on GPU never round-trips
  host (today's path copies to numpy, marshals, re-uploads). Host-pointer
  entry kept for the smoke/bench callers. Node ids cross the boundary as
  int arrays only (Unicode ABI law).
This makes the route entry callable from tensor_cuda C++ later if routing
moves inside the engine, and CUDA-graph-capturable in principle (parked
kernel-opt Phase 2 machinery becomes applicable). Python-side per-call
work after P2: epoch compare + one ctypes/pybind call.

P3 — Re-receipt. Re-run the exact three ledger smoke commands (same
capture banks, same flags) on idle GPU; parity vs batched Python law and
direct `route_gqa_cuda()` must hold at all three node counts. Additionally
receipt the device-pointer entry against the host-pointer entry (same
results, minus the query H2D).

## Registered expectations (frozen at commit)

- E1: P0 attributes ≥ 80% of the 512-node gap to named functions.
- E2: reused bridge min wall ≤ 2× direct CUDA route wall at every node
  count (≈ 0.26 / 0.51 / 1.54 ms) after P1+P2.
- E3: bridge overhead no longer scales with node count (flat gap across
  32/128/512 within noise).
- Misses are findings, not failures — record and proceed.

## Gates (every phase)

Parity (batched Python raw |q·k| law + direct route match); lifecycle
selector 6/6; full native-runtime suite green; CPU fallback paths byte-
identical in behavior; timing receipts on idle GPU with the measurement
laws from the kernel-opt program (interleaved same-session A/B for any
sub-10% claim).

## Roles

Fable session = planner/ledger/gates; Sonnet agents = implementation, one
flat agent per phase, no delegation cascades.
