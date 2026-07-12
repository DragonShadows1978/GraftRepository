# GRM GQA Exact Ragged CUDA Router Plan

Status: development-closed, `QUALITY-GREEN` on the registered envelope. The
feature remains opt-in and the repository default is unchanged. Tracking artifacts:

- Operational ledger: `docs/GRM_GQA_EXACT_RAGGED_CUDA_LEDGER.md`
- Continuing narrative: `docs/GRM_GEMV_ROUTER_SYNTHESIS.md`

Branch: `codex/gqa-ragged-cuda-router`

Worktree: `/home/vader/GraftRepository-gqa-ragged-cuda-router`

No commits or pushes are authorized for this work order.

## Decision

Preserve every model-native layer-0 GQA key used by the established routing
law. This branch changes representation and execution only: it does not
compress, average, select, synthesize, or otherwise replace routing evidence.

The predecessor fixed-card experiment is stopped and retained in stash commit
`68c9f0a7945e1872aeb9aba5439a0ee09fc806f2`. Its Qwen development result was
negative: full ragged keys reached 18/20 source recall while fixed source cards
reached 2/3/4 and hybrid source-plus-alias cards reached 3/2/3 at 2/4/8 slots.
This successor therefore attacks the CUDA shape constraint, not retrieval
quality.

## House Rules

- Keep the predecessor failure and its receipts intact; do not retune its
  sealed evidence.
- CPU and CUDA must implement the same raw `|q dot k|` score law over the same
  fp32 key values. Layout padding is allowed only when proved score-neutral.
- Treat variable leaf lengths and multi-row hierarchy as different problems.
  Phase 1 covers mixed-length, single-route-row GQA nodes only.
- Generated aliases, route-card lifecycle, and key compression are out of
  scope.
- Existing stable IDs, filters, lexical bonuses, lineage, and mount behavior
  remain authoritative and unchanged.
- CUDA remains explicitly opt-in through `GRM_GQA_CUDA_ROUTE=1`; every
  ineligible or uncertain case falls back to the existing CPU path.
- Stop on a correctness miss. A latency miss may be reported as
  `QUALITY-GREEN / COMPUTE-RED`, but does not authorize an approximation.

## Exact Routing Law

For a query `q[H,Q,D]` and one stored raw key row `k[H,T,D]`, the established
score is:

```text
mean_h max_q max_t abs(dot(q[h,q,:], k[h,t,:]))
```

A zero-padded row `k'[H,Tmax,D]` is exactly equivalent for every non-empty
source row: every added dot product is zero, while the maximum of absolute dot
products is non-negative. All original fp32 values and their ordering remain
unchanged in the prefix of each padded row.

## P0 — Frozen Gates

Correctness and lifecycle:

- CPU/padded-CUDA top-k parity is 100% outside exact score ties; stable node-ID
  tie order is identical.
- CUDA is engaged for every eligible mixed-length leaf semantic route.
- Existing lexical and candidate filter/exclusion behavior is unchanged.
  Non-finite keys, missing native IDs, incompatible geometry, empty rows, and
  nodes with `child_cents` retain their current fail-closed fallback.
- The existing epoch-driven immutable-bank lifecycle remains authoritative:
  no O(N) rebuild or signature walk on a reused route.
- Padding receipts report per-row lengths, raw/padded value counts and bytes,
  and the padding ratio.

Resource rails:

- At 512 real Qwen nodes, resident-bank route latency is p50 <= 5 ms and
  p95 <= 8 ms.
- The registered development length schedule must use no more than 2.5x the
  raw value count after global padding.
- GPU residency must stabilize after attachment; allocator peaks are reported
  separately from sampled resident memory.
- A cold rebuild should complete within 250 ms. Missing this rail is a
  compute result, not permission to weaken parity.

## P1 — Exact Padded Leaf Bank

Extend the existing GQA arena bank builder so eligible leaf rows may have
different token counts while retaining common KV-head and head dimensions.
Build one immutable fp32 bank `[node, kv_head, max_tokens, head_dim]`, zero-fill
it, and copy each original row into its prefix. Keep individual shapes in the
snapshot signature and publish the padded bank through the existing native
CUDA sidecar. Same-shape banks must retain their present fast path.

Reject rather than sanitize non-finite data or malformed shapes. Do not change
the CUDA score kernel: its fixed token extent is the exact padded extent.

## P2 — Focused Verification

Add deterministic and property tests for:

- exact preservation of every source fp32 byte in its padded prefix;
- zero-only tails and correct padding receipts;
- randomized raw-versus-padded score and ranking parity;
- stable ties and epoch-cache object reuse;
- rebuilds after key/shape/retirement changes;
- fail-closed handling of non-finite rows, incompatible geometry, missing IDs,
  filters, lexical routing, and multi-row hierarchy;
- unchanged behavior for existing same-shape CUDA banks and CPU-only routing.

Run the focused tests first, then the existing GQA CUDA bridge, arena,
lifecycle, native-runtime, and repository suites in proportion to the touched
surface.

## P3 — Real Qwen Development Gate

Use existing Qwen3.5 model-native route captures; do not invent synthetic
centroids. Freeze seed `20260712` and the token-length schedule
`{32,48,64,96,128,192,256}` before scoring.

- Node counts: 32, 128, 512.
- Queries: 100 stored model-native query captures at 512 nodes, proportionally
  fewer at smaller sizes.
- Top-k: 1, 3, 5, 16.
- Timing: two warmups and at least 20 resident-bank measurements.
- Compare: ragged Python law, native CPU routing, direct padded CUDA routing,
  and the arena bridge.
- Record top-k parity, exact-tie cases, CUDA engagement, cold build time,
  resident p50/p95, raw/padded bytes, padding ratio, and sampled GPU residency.

If parity or lifecycle fails, stop this branch. If correctness passes but the
latency or padding rail fails, document `QUALITY-GREEN / COMPUTE-RED` and move
to P4 without presenting padding as the production answer.

## P4 — Successor Choice, Not Prepaid Scope

Only after P3 may the evidence choose between:

1. retaining exact global padding when all rails pass;
2. exact length buckets, if padding alone misses the memory rail; or
3. a segmented-offset kernel, if buckets cannot meet the latency/memory rails.

Multi-row hierarchical nodes require a separate exact reduction: score each
route row with the current law and then take the maximum completed row score
per node. Concatenating child rows before the per-head mean is not generally
equivalent and is forbidden.

## Adoption Boundary

This branch is exploration only. Passing its development gate does not enable
CUDA routing by default. Adoption requires a fresh dual-GQA quality session,
checkpoint/restart evidence, concurrency coverage, and an explicit operator
decision in a separate work order.

## Development Outcome (2026-07-11)

Exact global padding is the development winner. The sealed seed-`20260712`
32/128/512 Qwen ladder passed parity, CUDA-engagement, padding, resident
latency, cold-build, and sampled-residency rails after eliminating a redundant
attach hash and temporary device repack. The load-bearing 512-node values are:
175/175 total ladder queries parity-green, 2.1992x padding, 1.591/2.007 ms
p50/p95 resident route, 147.09 ms cold build, and flat 708 MiB sampled steady
residency. Exact receipts and the harness-correction history are in the ledger.

Do not open length buckets or segmented offsets from this work order. Do not
enable the feature by default. Hierarchical multi-row routing and fresh dual-
GQA adoption remain separate governed work.
