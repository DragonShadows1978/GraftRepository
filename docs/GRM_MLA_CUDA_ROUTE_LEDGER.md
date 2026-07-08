# GRM MLA CUDA Route Ledger

Execution record for the MLA CUDA route work order. Immutable plan:
`docs/GRM_MLA_CUDA_ROUTE_PLAN.md`. Wing narrative continues in
`docs/GRM_GEMV_ROUTER_SYNTHESIS.md`.

## 2026-07-08 (opening)

Action: Work order opened; plan approved by David ("make it so") and
committed immutable.

Repo state:
- Branch `grm-cuda-bridge-overhead` (continuing on the wing's branch; the
  GQA bridge work order closed on it with parity green and its finding
  ledgered).
- Predecessor receipts inherited: GQA sidecar (persistent device arenas,
  GPU top-k, device-pointer entries), W1 mutation-epoch choke points
  (MLA bump currently a no-op — becomes real here), measurement laws.

Registered note (David, at approval): this is another place the
kernel-opt program's results apply — the MLA route scan is a GEMV-class
memory-bound kernel + top-k, the exact shape optimized in Project-Tensor
2026-07-07 (A5 coalesced loads; int4_gemv patterns if an INT4 device
book is ever in scope). The sidecar implementation should draw on that
vocabulary rather than reinvent.

Next action:
- P0 (Sonnet, flat): correctness re-stamp; production-path cProfile at
  100k/1M through ArenaCache.route(); production wall curve — all
  committed before any fix.
