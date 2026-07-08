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

## 2026-07-08 (P1 + P1b complete)

Action: P1 (limit-aware topk + epoch-cached id map) and P1b (cand-base
epoch cache + build-flag audit) implemented, lead-verified, committed.
Production library rebuilt Release; full suite re-run against it.

Findings:
- Production 1M int4 M=128 route: 925.6 → 260.9 (P1) → 127.5 ms p50
  (P1b) = 7.3× cumulative. 100k → 13.9 ms. Prefix-parity, byte-identity
  fuzz, and epoch equivalence batteries all green; full-rank contract
  untouched; M6 preserved under truncation. Suites 129/129 (then 108/108
  native re-stamped on the Release library).
- Receipt discrepancy resolved honestly: the "131 ms native" was the
  P1 agent's own micro-bench missing INT4 env — with the op-point flags
  set, production native matches the historic 23.88 ms receipt to the
  hundredth. Pipeline exonerated.
- REAL FIND (build-flag audit): production cpp/build/libgrm_runtime.so
  had been compiling with -fPIC only — no -O flag at all. -O3 alone is
  21× on the router scan (222→10.5 ms int4 at 100k). cpp/CMakeLists.txt
  now defaults CMAKE_BUILD_TYPE=Release with an explanatory comment;
  production artifact rebuilt and re-gated (108/108). Every historic
  receipt assumed an optimized build; production had never had one.
- OpenMP left OPT-IN deliberately (further 1.75×): it changes threading
  behavior (tsan-gate domain) — DECISION FOR DAVID registered here.
- E4 (≤1.15× native-only) still missed at 5.3×; residual is one named
  cost (O(cand) subset pass in _native_route_order, needed because cand
  varies with exclude). DISPOSITION: absorbed into P2's design rather
  than another host pass — the CUDA arena contract returns top-k node
  ids (O(k) mapping), and eligibility/exclude reconciliation follows the
  GQA bridge pattern (dense eligible attach + fallback on shortfall).

Next action: P2 — device-resident MLA centroid arena in the sidecar.

## 2026-07-08 (P2 complete — E2 met 57×; E3 ruling by David)

Action: P2 device-resident MLA CUDA route implemented (Sonnet, flat),
lead-reviewed, committed. Opt-in (GRM_MLA_CUDA_ROUTE=1), dormant by
default.

Findings:
- E2 MET WITH MARGIN: 1M-node route through production ArenaCache.route()
  = 2.22 ms p50 (target ≤5 ms; host P1b 127.5 ms; 57.3×). Within ~10% of
  the measured raw kernel floor. Curve: 0.184/0.210/0.398/2.22 ms at
  1k/10k/100k/1M. Suites 117/117 (9 new lifecycle tests mirror the GQA
  selector + epoch fail-closed + out-of-range-row fallback).
- Scoring semantics replicated from RouterIndex::exact_mla_entry_score
  (grm_runtime.cpp:3518-3545) incl. the 1e-8 denominator guard and
  score-desc/node-id-asc tie order (score_node_better, :75-82); M6
  preserved on device (isfinite guard before ranking).
- **E3 AS REGISTERED: MISSED — FLAGGED, KNOWN, ACCEPTED (David's ruling
  2026-07-08).** Byte-identical top-k at 1k/10k/100k across the full
  fuzz matrix (excludes × retirement × exact ties × NaN rows). At 1M,
  1 of 4 sampled queries swaps its LAST TWO ids on a genuine near-tie
  (fp64 refs 0.70710675 vs 0.70710667, Δ≈8.6e-8): cuBLAS fp32 reduction
  order vs the fp64 host reference. Both orderings defensible under
  fp32 data. Disposition (David): flag it, make it known, no tie-epsilon
  law amendment, no rescore engineering — 1M nodes is never happening on
  a frozen-model repository (GPT/Qwen class); the operational envelope
  (≤ thousands of nodes) is byte-exact. The 1M point stands as a scaling
  demonstration carrying this caveat wherever it is cited.
- Agent self-caught two of its own O(N) host bugs pre-report (per-call
  reverse-map rebuild ~460 ms; 1M-tuple == compare ~2.3 ms — identity
  fast path added). Also renamed a shadowed pre-existing test fixture
  (flagged).
- CROSS-TRACK FLAG (board): GQA's _cuda_route_order retains the same
  O(len(cand)) residual MLA just fixed — fine at the mount-window scale
  GQA runs today, unvalidated past a few hundred nodes.

Work-order status: E1 met, E2 met, E3 missed-with-ruling, E4 missed
(residual absorbed into P2's O(k) contract — production path now 2.22 ms
at 1M, moot). P3 formal re-stamp = the gate artifacts
(artifacts/grm_mla_cuda_route/p2_gate.json, device_entry_parity.json);
official curves recorded above.
