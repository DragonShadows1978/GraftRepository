# GRM MLA CUDA Route — Implementation Plan

Status: DRAFT — immutable at initial commit. Third work order of the
router wing. Tracking artifacts:
- Operational ledger (new): `docs/GRM_MLA_CUDA_ROUTE_LEDGER.md`
- Narrative synthesis (continues): `docs/GRM_GEMV_ROUTER_SYNTHESIS.md`
House laws unchanged: measure don't model; commit-per-phase;
gate-per-phase; NaN law (M6: non-finite scores dropped, never sorted —
preserved on any CUDA path); CPU C ABI stays CUDA-free and
dependency-free; Unicode/time never cross the ABI.

## Problem

MLA routing has NO CUDA path — it is host-only (contiguous fp32 arena +
INT4 two-tier, best receipted 1M-node operating point 23.88 ms p50 /
25.49 ms p95, bounded-staging M=128). Two open questions:
1. David's registered suspicion (bridge ledger 2026-07-08): the 1M
   receipts timed the NATIVE call only. The production path —
   `ArenaCache.route()` wrapping it (candidate building, filters,
   lexical prefilter, node-id→graft mapping) — has never been profiled
   at scale. If it carries O(N) Python, production 1M routing is far
   worse than 24 ms.
2. The GQA CUDA sidecar receipts (device route 0.13-0.77 ms; GPU top-k;
   persistent arenas; epoch staleness; device-pointer entries) all
   transfer — an MLA centroid arena is the EASIER shape (one fp32 GEMV,
   no segment reduce). Arithmetic: 1M × 128-dim fp32 = 512 MB @ ~400 GB/s
   ≈ 1.3 ms scan — an order of magnitude under the host operating point,
   before an INT4 device book is even considered.

## Constraints (binding)

- Exactness law: CUDA MLA route must match the native fp32 scan top-k
  (the same parity standard the INT4 two-tier was held to). The INT4
  device book is OUT OF SCOPE for this order unless fp32 misses E2 —
  scope stays tight.
- Contract: limited top-k route (mount-window shape, mirroring the GQA
  bridge contract). Full-rank repository ordering stays host.
- CUDA opt-in (`GRM_MLA_CUDA_ROUTE=1`-style), fail-closed to the CPU
  path on any error/ineligibility, per the GQA bridge precedent.
- Staleness: reuse the mutation-epoch machinery (W1 choke points already
  bump on every graft_repository mutation; `_bump_cuda_gqa_epoch` is
  currently a no-op on MLA ArenaCache — it becomes real here, same
  fail-closed equivalence test pattern).

## Phases

P0 — Receipts before anything:
  a. MLA correctness stands post-GQA-work: router baseline suite +
     native runtime suite green (they were during P2; re-stamp on this
     branch tip).
  b. Production-path profile (the suspicion): cProfile through
     `ArenaCache.route()` — flat MLA and INT4 paths — at 100k and 1M on
     the harvested corpus (`scripts/grm_router_baseline.py` machinery).
     Attribution table: wrapper (candidates/filters/lexical/mapping) vs
     native call. Committed before any fix.
  c. Production wall curve (through-Python) at 1k/10k/100k/1M alongside
     the historic native-only curve.

P1 — Wrapper fixes, strictly profile-guided (the GQA P1 pattern: cache
what is re-derived, epoch-gate what is re-checked). Only items P0 names.

P2 — CUDA MLA route in the sidecar: persistent device centroid arena
(fp32 C[N×D] + row norms), cuBLAS GEMV/GEMM scoring, existing GPU top-k,
epoch-integrated attach/invalidate, host-pointer AND device-pointer
query entries — mirroring the GQA bridge surface exactly. Python hot
path: epoch compare + one ctypes call.

P3 — Re-receipt: exactness battery (CUDA top-k ≡ native fp32 scan top-k
across the dialect gate battery + fuzz), latency curves 1k→1M, and the
production-path curve re-measured. Findings recorded either way.

## Registered expectations (frozen at commit)

- E1: P0 attributes ≥80% of any production-vs-native gap to named
  functions (if the gap is <10% of route time, record that and skip P1).
- E2: 1M-node CUDA MLA route ≤ 5 ms steady-state wall (bank resident,
  bridge included) vs 23.88 ms host p50 — ≥4.8× at the 1M point.
- E3: CUDA route exactness: identical top-k to native fp32 scan on the
  full battery (zero tolerance — same standard as the two-tier gate).
- E4: production-path 1M route ≤ 1.15× the native-only time after P1.
- Misses are findings, not failures — record and proceed.

## Roles

Fable = planner/gates/ledger; Sonnet agents = implementation, flat,
no-delegation rule in every brief; timing gates on the measured-law
instruments (interleaved same-session; isolated device timing where
sub-10% claims are made).
