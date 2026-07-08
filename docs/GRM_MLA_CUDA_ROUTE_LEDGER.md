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

## 2026-07-08 (P0 complete — committed before any fix)

Action: P0 receipts landed. David's suspicion CONFIRMED, worse than
suspected. Lead spot-checked the load-bearing lines at source.

Findings (evidence class: profile receipt + code inspection):
- Correctness re-stamp: router baseline 21 passed (grown from 15 by the
  GQA bridge tests — growth, not drift), native runtime 95 passed.
  116/116 green on this tip.
- THE NUMBER: production 1M-node route through ArenaCache.route() =
  925.6 ms p50 / 953.6 ms p95 (INT4 bounded-staging M=128, the receipted
  operating point) vs the historic native-only receipt 23.88 ms p50 —
  a ~39× gap. The wing's 1M claim was only ever true of the bare ctypes
  call; production has never seen it.
- E1 MET: ≥99.9% of the gap named at every point. Wrapper share 47-65%
  of wall; the rest is INFLATED NATIVE WORK caused by the same defect:
- Root defect (lead-verified, graft_arena.py:453 and :1719):
  `_native_route_order` requests topk=len(self.grafts) — full N —
  regardless of the caller's limit (spy receipt: limit=3 → topk=500 at
  N=500). Deliberate belt-and-suspenders: the full ordering feeds a
  fail-closed completeness check (`len(routed) != len(cand)` → Python
  fallback) before truncating to limit. Secondary: native_to_idx dict
  rebuilt over all N per call (epoch-cacheable, GQA pattern).
- Material verdict: MATERIAL (47-65% ≫ the 10% skip threshold). P1 in
  scope, profile-guided.
- Receipts: artifacts/grm_mla_route/P0_ATTRIBUTION_AND_CURVE.md +
  pstats/JSON; harness scripts scripts/grm_mla_route_{profile,wall_curve}.py.

P1 design constraints (registered before implementation):
- When limit is not None: request topk=limit (plus any slack the
  downstream excludes actually need — read the callers), and REPLACE the
  completeness law with an equally conservative one: native must return
  exactly min(topk, eligible) ids, every id must map via native_to_idx,
  any shortfall or unmappable id → Python fallback (same distrust,
  bounded cost). The check is a law — replaced, never deleted.
- limit=None callers (full-ordering contract, incl. :1719) keep full-N
  semantics untouched.
- native_to_idx epoch-cached (the MLA no-op bump becomes real, same
  fail-closed equivalence test pattern as GQA W1).
- Gate: limited-path results must equal the prefix of the old full-path
  results across the battery + fuzz; suites green; curve re-measured.

Next action: P1 (Sonnet, flat).
