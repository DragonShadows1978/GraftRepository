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
p50 7.0211ms and 1M p50 49.6927ms on the harvested route corpus. The next
normalization slice precomputes `scale / row_norm` for q4 rows and uses a
single query inverse norm in the bulk scorer; the same parity shape now measures
100k p50 6.2995ms and 1M p50 43.4791ms. Finding: this is a real drop from the
prior 1M M=16 65.3449ms, but E3 is still missed; the remaining work needs
lower-level vectorized q4 dot or a larger routing layout change.

P3 predecode note: the host INT4 arena now keeps a decoded int8 q4-value cache
beside the packed q4 rows. This preserves the quantized values but trades host
RAM for a simpler bulk dot loop. On the harvested 1M dim128 parity run, M=16
was faster but not exact (one top-3 mismatch). M=32/64/128 matched native fp32;
their p50s were 39.6265/37.5690/37.8382ms. Current measured operating point:
predecoded q4, M=64, 1M p50 37.5690ms. E3 is still not met.

P3 bounded-staging note: INT4 route now avoids materializing every eligible
candidate before refine when `M < N`; it keeps a bounded top-M heap after the
parallel bulk score pass. The same harvested 1M dim128 M=64 parity run now
measures 26.0175ms p50 / 27.2486ms p95 against native fp32. This is deep
interactive territory and narrowly misses the original <=25ms E3 target. A
more invasive thread-local heap selection attempt remained parity-green but
slowed to 29.7308ms p50, so it was rejected.

**P4 — GQA key-bank GEMM path.** D4. Gates: parity vs Python GQA
routing on Qwen-family GQA scenarios, latency, 166 floor.

P4 first-slice note: native GQA routing now builds a lazy contiguous key-bank
arena keyed by `(kv_heads, head_dim)`, with entry-to-key and key-to-row offsets.
`route_gqa_raw` scores through this arena when shapes are valid and falls back
to the old scan semantics otherwise. It also parallelizes entry scoring under
OpenMP and hoists query/key finiteness checks out of the inner dot loop while
preserving the M6 law: non-finite key/query scores are dropped. A 10k-node
dim16 synthetic smoke probe (`query_shape=(4,4,16)`, `key_shape=(1,4,16)`, `-O3
-fopenmp`) matched the Python/NumPy raw-score top-k and measured native p50
6.0020ms versus Python 166.9182ms, a 27.81x speedup. This clears E4 for that
smoke shape; broader Qwen-family scenarios and the final P6 curve remain.

P4 Qwen3.5-2B shape note: `scripts/grm_gqa_router_benchmark.py` now measures
native GQA routing on Qwen-shaped harvested route banks. The default preset is
the Qwen3.5-2B source-attention geometry (`8q/2kv/head_dim256`). Representative
1-token route keys with 4-token probes measured 1k p50 2.4416ms and 10k p50
8.6680ms, parity-green against the Python raw q.k law; Python p50 at 10k was
227.7100ms, a 26.27x speedup. The live Qwen3.5-2B translation corpus was
inspected read-only: source shards expose K banks shaped `(1,2,256,256)` for
layers 3/7/11/15/19/23.

P4 real-capture note: the GQA benchmark can now read capture shards in-place
(`--capture-dir`, `--capture-role`, `--capture-layer`) without moving or
deleting generated artifacts. On Qwen3.5-2B source layer 3, full 256-token K
banks first showed a useful failure: 96 nodes measured 56.1185ms native p50 vs
40.9139ms Python p50 because native OpenMP gating only looked at entry count.
`route_gqa_raw` now uses workload-aware gating based on
`entries * query_heads * query_tokens * key_tokens`; the same 96-node full-bank
point improved to 16.6903ms p50, and 127 nodes measured 23.0640ms p50, parity
green. Representative 1-token K from the same captured shards remains much
faster: 96 nodes, 0.1172ms p50, parity green. Finding: captured full-bank GQA
is now usable at small N, but larger N still needs the true GEMM/segment-reduce
layout or explicit representative-key compaction.

P4 full-bank hot-loop note: a tiled scorer that accumulated all query rows for
one KV head against each K row was measured and rejected; it stayed
parity-green but regressed the 127-node full-bank p50 to 29.2260ms. The kept
change leaves the existing segment loop order intact but accumulates arena q.k
dots in `float` instead of promoting every multiply to `double`, matching the
float32 benchmark tensors. On the same Qwen3.5-2B source layer-3 full 256-token
K-bank probe, 32/96/127 nodes now measure 4.4014ms / 9.5322ms / 10.8340ms p50,
parity-green against the Python raw q.k law. This puts exact full-bank routing
under 10ms through 96 captured nodes and within reach at 127. A larger
real-capture run loaded 298 usable source shards and measured 192/256 nodes at
15.1623ms / 18.9169ms p50, still parity-green. Larger-N GQA still needs the
true GEMM/segment-reduce layout or representative-key compaction.

P4 progress/checkpoint note: `scripts/grm_gqa_router_benchmark.py` now supports
`--progress-out` JSONL checkpoints and `--progress` stderr updates after each
node-count result. A native-only real-capture run loaded 1,165 usable Qwen3.5-2B
source shards and extended the exact full-bank curve to 512/768/1,024 nodes at
39.5778ms / 58.3090ms / 76.5800ms p50. These larger points are marked
`parity=null`; they are useful native scaling receipts, not a replacement for a
sampled or batched parity strategy beyond 256 nodes.

P4 sampled-parity note: native-only GQA benchmark runs can now use
`--parity-sample-queries` to check deterministic query samples against the
Python raw q.k reference without including Python in route timing. The first
real-capture sampled check loaded 595 usable Qwen3.5-2B source shards and
matched one sampled Python query at 512 full-bank nodes, measuring 39.6250ms
p50 / 43.4202ms p95. The benchmark also has a batched Python reference mode
that preserves the scalar law's tie order on capture-style float16-derived
keys. With `--parity-reference batched`, 768 and 1,024 full-bank nodes matched
two sampled Python queries at 58.5005ms / 73.5912ms p50. These are sampled
correctness receipts, not exhaustive full-query parity.

P4 qt4 scorer note: native GQA full-bank routing now has a query-token-4 fast
path for the common capture benchmark shape. It keeps the same per-query-row dot
accumulation order but reuses each K row across the four query rows before the
per-head max reduction. Qwen3.5-2B source layer-3 full-bank sampled-parity runs
now measure 512/768/1,024 nodes at 19.0136ms / 33.8389ms / 36.7363ms p50, all
matching the batched Python raw q.k reference on two sampled queries.
`scripts/grm_gqa_router_benchmark.py` now also supports
`--parity-all-queries` for native-only runs, producing explicit exhaustive
parity receipts without timing Python in the route curve. A 512-node Qwen3.5-2B
source layer-3 full-bank run matched all four generated batched-reference
queries and measured 20.4087ms p50 / 21.8582ms p95.
Native GQA selection now uses bounded top-k partial sorting when the caller asks
for fewer routes than the scored candidate count, preserving the existing
score-plus-node-id tie order. A harvested representative 10k-node run measured
5.9959ms p50 with exhaustive five-query parity, and a 512-node Qwen3.5-2B source
full-bank run matched all four generated batched-reference queries at 19.1065ms
p50 / 23.3995ms p95.
The full-bank exhaustive receipt now extends to 768 and 1,024 source-capture
nodes: 768 matched all four batched-reference queries at 26.0228ms p50 /
29.5759ms p95, and 1,024 matched all four at 33.9970ms p50 / 39.3356ms p95.
The no-lexical native route path now skips empty lexical hashing/hit counting
and ranks raw GQA scores directly before top-k selection. It is a measured
cleanup rather than a new scorer: representative 10k measured 5.9905ms p50 with
exhaustive five-query parity, and 512 full-bank measured 19.3370ms p50 /
22.8915ms p95 with all four batched-reference queries matched.

P4 representative-compaction note: the GQA benchmark can now route compacted
capture banks (`--compact-route-tokens`, `--compact-route-mode`) while checking
against the original full-bank Python reference (`--compact-parity-full`). On
Qwen3.5-2B source layer-3 captures, simple stride compaction at 1,024 nodes
failed full-bank sampled parity for 16/32/64/128 representative tokens. The
16-token route was fast (`6.9564ms` p50) but wrong; 128 tokens was still wrong
and slower than the qt4 full-bank path. Simple geometric representative-key
selection is rejected for runtime defaults.

P4 grouped-scorer rejection: an opt-in grouped qt4 scorer that reused each K row
across repeated query heads preserved sampled parity but was slower than the
kept qt4 scorer: 512 nodes regressed to 20.3603ms p50 and 1,024 nodes regressed
to 38.2764ms p50. It was removed rather than kept behind a flag.

P4 repeat-4 head-ratio unroll rejection: a hand-unrolled repeat-4 qt4 scorer
for the Qwen-family `query_heads / kv_heads == 4` shape also preserved sampled
parity, but regressed 512 nodes to 23.1079ms p50 and only matched the
1,024-node point within noise at 36.6391ms p50. This is the `8q/2kv` head
repeat ratio on Qwen3.5-2B source captures, not a 4B model run. It was removed
rather than kept behind a flag.

P4 paired-head rejection: a scorer that reused each K row across two repeated
query heads at a time also preserved parity, but regressed a fresh 512-node
Qwen3.5-2B source full-bank run from 19.8082ms p50 to 22.3257ms p50. It was
removed; the kept runtime remains the prior qt4 full-bank scorer.

**P5 — Epoch snapshots + stress.** D5. Gates: race harness (writer churn
@ 1k mutations/s against concurrent routes; TSAN clean; no torn top-k),
166 floor.

P5 first-slice note: the C ABI handle now wraps `RouterIndex` with a
`std::shared_mutex`. Route-key updates, metadata active/filter changes,
expiry/revision state changes, clears, and checkpoint router rebuilds take the
writer side. Route-key mutations now mark the router arena as needing
preparation instead of rebuilding on every insert. The first route after a
mutation prepares the MLA/GQA arena once under the writer lock; subsequent
prepared route reads use the shared side. Generic MLA-style `route` calls on
GQA stores stay on the writer side because that mixed path can still lazily
build an MLA arena over GQA route keys. Regression
`test_native_router_serializes_concurrent_route_updates` runs three reader
threads against a writer thread that rewrites route keys. This is not yet the
final lock-free double-buffer epoch design from D5, but it lands the first
no-torn-read gate on the current C ABI surface while restoring shared read
concurrency for the normal prepared route paths.

P5 prepared-snapshot note: prepared MLA/GQA router state is now published as a
`std::shared_ptr<const RouterIndex>` snapshot. Writers mutate the live router
under the writer lock and mark the snapshot dirty; the next prepared route
rebuilds the dialect arena once and publishes an immutable copy. Non-GQA
prepared routes and raw GQA `route_gqa` calls copy the snapshot pointer and run
the expensive scoring path outside the mutex. Generic MLA-style `route` calls
on GQA stores remain on the writer-lock path because they can still lazily build
an MLA arena over GQA route keys. A stale-snapshot regression was caught and
fixed by dirtying snapshots on active-state, route-metadata, revision, and
expire mutations. Gates: the focused stale-snapshot/concurrency selector passed
6/6; full `tests/test_grm_native_runtime.py` passed 73/73. Fresh native-only
snapshot receipts: Qwen3.5-2B source layer-3 full-bank 512 nodes matched all
four batched-reference queries at `21.8605ms` p50 / `22.1071ms` p95; the
representative 10k GQA shape matched five batched-reference queries at
`5.9662ms` p50 / `5.9920ms` p95.

P5 atomic-fast-path note: clean prepared routes no longer take the C ABI
shared mutex. The handle now carries an atomic dirty flag next to the published
`shared_ptr<const RouterIndex>`; readers atomically load a clean snapshot and
pin it for the route call, while dirty or missing snapshots fall back to the
writer lock and publish a new immutable copy. Writer-side mutation is still
serialized through the live router lock, so active/filter/revision semantics
stay linear after the mutation call returns. Gates: focused
stale-snapshot/concurrency selector passed 7/7; full
`tests/test_grm_native_runtime.py` passed 73/73. Fresh atomic-fast-path GQA
receipts: Qwen3.5-2B source layer-3 full-bank 512 nodes matched all four
batched-reference queries at `22.0019ms` p50 / `23.8365ms` p95; representative
10k GQA matched five batched-reference queries at `5.9916ms` p50 /
`6.0286ms` p95. Remaining D5 hardening is TSAN or an equivalent sanitizer race
gate; the route hot path itself is now read-lock-free for clean prepared
snapshots.

**P6 — Report.** `ROUTER_SCALING_REPORT.md`: before/after curves at all
four node counts, both dialects, exactness-gate record, M-sweep table.
Board update; note for GRM paper v1.1 §5 (routing limitation retired —
with measured numbers, not projections).

P6 first report note: `docs/ROUTER_SCALING_REPORT.md` now records the current
measured MLA and GQA router state: P0 native/Python baselines, P2 fp32 arena
large points, P3 INT4 M-sweeps and current M=64 operating point, P4 GQA
key-bank smoke plus Qwen3.5-2B representative-key curve, expectation pass/miss
status, and remaining work. The report is intentionally explicit that E3 is
still narrowly missed and that larger real-graft GQA curves remain.

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
