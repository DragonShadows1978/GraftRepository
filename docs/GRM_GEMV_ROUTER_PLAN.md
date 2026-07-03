# GRM GEMV Router — Implementation Plan

**Goal:** retire the per-node routing scan. Routing becomes dense linear
algebra over a contiguous centroid arena — one GEMV (MLA) / one GEMM +
segment-reduce (GQA raw) — with an APA-shaped two-tier precision scheme
(INT4 bulk scan → fp32 refine) and epoch-snapshot reads for concurrency.
Target: "hundreds of memories" → 100k–1M nodes at interactive latency,
on host CPU, with the paper's §5 routing limitation retired in v1.1.

**Status: IN PROGRESS. Translation PoC worktree is committed; P0 baseline
harness has started.**

**House laws in force:** measure, don't model (baseline curve BEFORE any
optimization); commit-per-phase; gate-per-phase (166-test floor); every
phase ships a regression test; NaN law (M6) preserved identically in all
new paths — non-finite scores dropped, never sorted; C ABI contracts
(incl. the m10 route_offsets +1 contract) unchanged; Unicode stays out of
native (lexical prefilter remains Python/ASCII-flag plane).

---

## 0. Current state (verify in Phase 0, don't trust this summary)

- Native `RouterIndex::route` (MLA centroid cosine) and `route_gqa_raw`
  iterate per node; Python fallback plane mirrors it (post-M6: both drop
  non-finite). CUDA route-scan does not exist (board-confirmed).
- Node adds/retires/folds mutate the index while routes may be issued —
  currently serialized by the Python layer's call discipline, not by
  design (the M1 race family, one plane over).
- Route filters (kind/scope/durability/mutability) applied per node
  during the scan.

## 1. Design

### D1 — Contiguous centroid arena (SoA)
- One fp32 matrix `C[N_cap × D]` (D = dialect centroid dim; MLA latent
  centroid today), rows assigned per node; stable node_id → row map;
  tombstone rows on retire; compaction pass when tombstone fraction
  > 25% (row remap table event, mirrors the symbolic-ecology
  address-vs-id law: **stable IDs sacred, rows disposable**).
- Norms precomputed per row (cosine = dot / norms).

### D2 — Two-tier scoring (the APA shape, applied to routing)
- **Bulk:** INT4 centroid book `C4[N_cap × D/2 bytes]` + per-row scale
  (group-wise, reuse tensor_cuda's group-32 packing convention for
  consistency). One pass computes approximate scores for ALL rows.
- **Refine:** top-M candidates by bulk score (M = max(4k, 64), k = route
  top-k) rescored fp32 from `C`. Final top-k from refined scores only.
- **Exactness gate:** two-tier top-k must MATCH full-fp32 scan top-k on
  the entire existing dialect gate battery + a fuzz harness (10k random
  repos). Disagreement → raise M (sweep 2k/4k/8k multipliers); if
  agreement requires M ≈ N, the INT4 tier dies and we keep fp32 GEMV
  only — that is an acceptable outcome, record it.

### D3 — Filters as bitmaps
- kind/scope/durability/mutability → per-node bitmask words maintained
  on mutation. v1: score all rows, mask before top-M selection (GEMV is
  cheap; branchless). v2 (only if profiling demands): gather eligible
  rows first.

### D4 — GQA raw path (|q·k| over stored keys, multiple rows per node)
- Concatenated key bank `K[R_total × D_k]` + segment map (node_id →
  row range). Route = GEMM (query heads × K^T) → per-node segment
  max/mean (dialect-defined) → same two-tier/top-M machinery.
- This is the bigger win: GQA routing currently falls back to Python
  entirely.

### D5 — Epoch snapshots (concurrency, the M1 lesson applied upfront)
- Double-buffered index epochs: writers apply mutation batches to the
  inactive buffer (or copy-on-threshold), publish via one atomic epoch
  pointer; readers pin the epoch for the duration of a route. No locks
  on the read path; writer coalesces.
- v1 scope honesty: GRM today is effectively single-threaded from
  Python — D5 lands as structure + stress test, not as a production
  requirement. It exists so threaded serving (Phase-7 daemon) inherits
  a race-free router instead of retrofitting one.

## 2. Phases

**P0 — Baseline + harness (no optimization).** Synthetic repo generator
(1k/10k/100k/1M nodes, realistic centroid distributions harvested from a
real repo's stats, not gaussian fantasy). Measure CURRENT native scan +
Python fallback latency curves. Commit the harness + `ROUTER_BASELINE.md`
with the numbers. *Everything after is judged against this.*

P0 implementation note: `scripts/grm_router_baseline.py` now harvests route
vectors from real repository source/doc lines by deterministic feature hashing,
builds the native C++ runtime, checks native scan parity against the Python
fallback law, and writes JSON/Markdown curves. `docs/ROUTER_BASELINE.md`
records the smoke command and the full 1k/10k/100k/1M command.

**P1 — Python plane vectorization.** Replace any per-node Python/numpy
loop with matrix-form scoring (one `C @ q`). Cheap, immediate,
independent of native work. Gate: identical route results on the 166
suite; baseline curve re-measured.

P1 implementation note: `ArenaCache.route` now vectorizes the flat MLA
Python fallback path with one stacked matrix `@` query score. Hierarchical
child-centroid grafts and dialects that override route scoring still use the
old scalar path, preserving descent and GQA raw-qk semantics.

**P2 — Native SoA arena + fp32 GEMV scan.** D1 + D3 + OpenMP over row
blocks. Gate: bit-identical top-k vs old scan on full battery; latency
curve; 166 floor.

P2 implementation note: native MLA route now has a lazy contiguous fp32
route-key arena behind the existing C ABI. Uniform-dimension MLA route keys
score through packed rows with precomputed row norms; dimension-mismatch cases
fall back to the original scan semantics. First smoke curve against the P0
1k/10k harness: 1k native p50 0.4985ms → 0.1792ms; 10k native p50
4.0865ms → 2.8098ms, parity green. This is the first packed scan slice, not
the final P2 gate: filters are still applied per entry, and OpenMP row-block
parallelism plus full 100k/1M curves remain.

P2 row-block note: the packed arena now carries per-entry row offsets, so
multi-key nodes score deterministically by entry and can use optional OpenMP
entry-block parallelism without races. The harness can compile with
`--openmp`; the parallel threshold is 32768 entries to avoid 10k-scale thread
overhead. Thresholded OpenMP smoke: 1k p50 0.1806ms, 10k p50 2.8027ms, 50k
p50 13.0213ms, all parity green.

P2 filter note: route entries now carry known-value bitmasks for
kind/scope/durability/mutability. Known filter values use bit checks in the
packed route path; unknown metadata or unknown filter values fall back to the
previous exact string comparison, preserving arbitrary metadata semantics.

P2 scale note: `RouterIndex` now maintains a node_id → entry index map, so
large native route-index population is O(N) instead of linear-search O(N²).
Native-only OpenMP dim128 curve after the map: 100k p50 22.8177ms, 250k p50
57.8062ms, 1M p50 211.3071ms. These large points skip Python reference parity
(`parity=null`) because the Python scan is the bottleneck at this scale.
Finding: fp32 host scan alone is not enough for E3; P3 INT4/two-tier refine is
needed for the 1M interactive target.

**P3 — INT4 books + two-tier refine.** D2. Gates: exactness gate (match
fp32 top-k, M-sweep documented), latency curve, 166 floor.

P3 implementation note: the MLA packed arena now builds a signed INT4
centroid book (`GRM_ROUTER_INT4=1`) with per-row scales and packed nibbles.
Routing bulk-scores all eligible entries through the INT4 book, keeps
`GRM_ROUTER_INT4_REFINE_M` candidates (default 4096), and performs final
ranking only after fp32 rescoring those candidates. Exactness is proven only
when the refine set covers all candidates; M=4096 was exact on the 1k/10k
harness run and remains a measured operating point, not a proof for all 1M
repos. Dim128 native-only OpenMP M=4096 curve: 100k p50 12.8097ms, 250k p50
27.7697ms, 1M p50 76.4786ms. Finding: INT4 gives >2x over fp32 at 250k/1M,
but the current scalar CPU nibble-unpack path still misses E3's 25ms at 1M.
Next P3 work is exactness sweep plus faster unpack/dot kernels or lower-M
policy if exactness holds.

P3 byte-unpack note: the INT4 scorer now unpacks two signed nibbles per byte
inside the dot loop instead of using a helper per dimension. Dim128 native-only
OpenMP M=4096 remeasure: 100k p50 11.9751ms, 1M p50 67.0890ms. This is a
real improvement, but still not enough for E3 without a more vectorized
unpack/dot kernel and/or a lower exactness-safe refine M.

P3 exactness-sweep note: the baseline harness now supports
`--native-fp32-parity` so large native-only INT4 runs can compare against the
same native arena with INT4 temporarily disabled instead of against the slow
Python scan. It also supports `--sweep-refine-m` for one command M sweeps.
On the harvested dim128 route corpus, 100k-node OpenMP INT4 matched native
fp32 for M=16/32/64/128/256/512/1024/2048/4096. The small-M 100k p50 timings
were 8.9106/10.1181/9.0655/9.0242/11.3754ms for M=16/32/64/128/256; the
large-M timings were 8.9430/10.6026/11.8228/9.0604/12.3436ms for
M=256/512/1024/2048/4096. At 1M nodes, M=16 remained exact on the same sampled
queries and measured 65.3449ms p50; M=256 also remained exact and measured
67.6768ms p50. Finding: refine count is not the current 1M limiter. The
bulk all-row INT4 scan dominates, so the next useful P3 work is a more
vectorized unpack/dot kernel or a representation that reduces per-query
decode traffic. Wider fuzz/repo exactness remains required before declaring
M=16 safe generally.

P3 hot-loop note: the INT4 bulk scorer now has a row-level scorer for the
single-route-key-per-entry case, decodes nibbles through a lookup table, applies
the row scale once after the integer-weighted dot, and uses exact-preserving
lexical hashes as a prefilter before string confirmation. This keeps lexical
semantics unchanged while avoiding hot-path string scans for almost all
non-hits. Dim128 OpenMP INT4 M=16 with native-fp32 parity now measures 100k
p50 7.0211ms and 1M p50 49.6927ms on the harvested route corpus. Finding:
this is a real drop from the prior 1M M=16 65.3449ms, but E3 is still missed;
the remaining work needs lower-level vectorized q4 dot or a larger routing
layout change.

**P4 — GQA key-bank GEMM path.** D4. Gates: parity vs Python GQA
routing on the Qwen3-4B gate scenarios, latency, 166 floor.

**P5 — Epoch snapshots + stress.** D5. Gates: race harness (writer churn
@ 1k mutations/s against concurrent routes; TSAN clean; no torn top-k),
166 floor.

**P6 — Report.** `ROUTER_SCALING_REPORT.md`: before/after curves at all
four node counts, both dialects, exactness-gate record, M-sweep table.
Board update; note for GRM paper v1.1 §5 (routing limitation retired —
with measured numbers, not projections).

**Deferred (registered, not scheduled):** CUDA route path (only if host
curves fail interactive targets at 1M — host INT4 GEMV at 1M×512 is
~130MB traffic, expected well under 20ms; measure first); ANN/IVF
structures (only relevant past ~10M nodes; out of scope).

## 3. Registered expectations (freeze before P2)

- E1: fp32 GEMV (P2) ≥ 10× current native scan at 100k nodes.
- E2: INT4 two-tier (P3) ≥ 2× additional over fp32 GEMV at 100k, with
  exactness gate green at M ≤ 4k.
- E3: 1M-node route ≤ 25ms host-side (any tier passing exactness).
- E4: GQA native path (P4) ≥ 20× the current Python fallback at 10k.
- Misses are findings, not failures — record and proceed; the curves
  are the deliverable.

## 4. Open questions (David)

- Centroid dim per dialect at scale: MLA latent centroid is cheap;
  GQA key banks grow with keys-per-node — cap keys-per-node for
  routing (representative-key selection) or carry full banks? (Affects
  D4 memory at 1M nodes.)
- Should the INT4 books live in the checkpoint (recompute-on-load vs
  persist)? Recompute is simpler and load is already IO-bound.
- Does the Phase-7 daemon plan want the router as a separable service
  boundary (route RPC) — if yes, D5's epoch structure should align with
  its threading model now rather than later.
