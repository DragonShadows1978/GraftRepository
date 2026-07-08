# GRM GEMV Router — Implementation Plan

**Goal:** retire the per-node routing scan. Routing becomes dense linear
algebra over a contiguous centroid arena — one GEMV (MLA) / one GEMM +
segment-reduce (GQA raw) — with an APA-shaped two-tier precision scheme
(INT4 bulk scan → fp32 refine) and epoch-snapshot reads for concurrency.
Target: "hundreds of memories" → 100k–1M nodes at interactive latency,
on host CPU, with the paper's §5 routing limitation retired in v1.1.

**Status: IMPLEMENTATION CLOSED in the local GraftRepository checkout.** P0-P6
shipped with measured MLA/GQA routing receipts, INT4 bulk/refine host routing,
prepared epoch snapshots, opt-in CUDA GQA route attachment, and the
`GRM_GQA_CUDA_ROUTE=1` limited-route arena bridge. Final non-GPU closure gate:
`PYTEST_ADDOPTS='-p no:cacheprovider' python3 -m pytest
tests/test_grm_router_baseline.py tests/test_grm_native_runtime.py -q` passed
`109 passed, 2 warnings in 344.94s` on 2026-07-03. Live CUDA bridge validation
ran on 2026-07-07 against real Qwen3.5-2B layer-3 capture banks at 32, 128, and
512 nodes; all matched the batched Python raw `|q.k|` reference.

Tracking artifacts:
- Operational ledger: `docs/GRM_GEMV_ROUTER_LEDGER.md`
- Narrative synthesis: `docs/GRM_GEMV_ROUTER_SYNTHESIS.md`
- Scaling report: `docs/ROUTER_SCALING_REPORT.md`

**Successor work order (2026-07-07):** the CUDA bridge overhead exposed by
this plan's closure receipts (bridge 25-50× the device route cost) is being
fixed under `docs/GRM_CUDA_BRIDGE_OVERHEAD_PLAN.md` (own ledger, wing
synthesis continues here) on branch **`grm-cuda-bridge-overhead`**.

**House laws in force:** measure, don't model (baseline curve BEFORE any
optimization); commit-per-phase; gate-per-phase (166-test floor); every
phase ships a regression test; NaN law (M6) preserved identically in all
new paths — non-finite scores dropped, never sorted; C ABI contracts
(incl. the m10 route_offsets +1 contract) unchanged; Unicode stays out of
native (lexical prefilter remains Python/ASCII-flag plane).

---

## 0. Original state (verified in Phase 0)

- Native `RouterIndex::route` (MLA centroid cosine) and `route_gqa_raw`
  iterated per node; Python fallback plane mirrored it (post-M6: both drop
  non-finite). CUDA route-scan did not exist at the start of the work order.
- Node adds/retires/folds mutated the index while routes might be issued,
  serialized by the Python layer's call discipline rather than by an explicit
  snapshot design.
- Route filters (kind/scope/durability/mutability) were applied per node
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

P3 longer exactness update: the shorter M=64 receipt was not strong enough.
Re-running the harvested 1M dim128 route with 10 queries found an M=64
native-fp32 mismatch on query 8 (`[173312, 463984, 963602]` vs
`[173312, 963602, 463984]`). A longer M sweep over the same query set matched
native fp32 for M=96/128/256. Current measured safer operating point is
bounded-staging M=128: 1M nodes, `23.8805ms` p50 / `25.4883ms` p95, 8 measured
queries after warmup, native-fp32 parity green. This clears the original E3
target on p50, but the p95 tail still sits just above 25ms.

P3 no-filter fast-path note: MLA INT4 now skips the full metadata filter helper
when a route call has no kind/scope/durability/mutability filters; inactive
entries are still checked directly. On the current 8192-row harvested q10
shape (`--max-files 32`, M=128), the pre-change baseline measured `27.0773ms`
p50 / `33.4480ms` p95 with native-fp32 parity. The active-only no-filter path
measured `24.9673ms` p50 / `29.1474ms` p95 on the same shape, also parity
green. Finding: this is worth keeping as a narrow hot-path win, but it does
not close the p95 gate; the tail still needs a lower-level q4 dot/layout fix.

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

P4 segment-reduce note: `route_gqa_raw` now has a key-bank segment-reduce
layout. It scores each prepared GQA key independently, stores a key-score bank,
then reduces key ranges back to entries before the existing top-k selection.
The path can be forced with `GRM_ROUTER_GQA_SEGMENT=1` and is selected
automatically when the prepared bank has at least 32 K tokens per key. Focused
GQA selectors passed 6/6. Measured default receipts after the heuristic:
Qwen3.5-2B source layer-3 full-bank 512 nodes matched all four batched-reference
queries at `20.4251ms` p50 / `21.3550ms` p95, improving over the atomic
snapshot pre-segment receipt (`22.0019ms` p50). Representative-key 10k stayed
on the per-entry scorer and matched five batched-reference queries at
`5.9844ms` p50 / `6.0227ms` p95. Forced segment on representative 10k was
parity-green but slightly slower (`6.0345ms` p50), so it is not the default for
single-token key banks. Full native runtime now passes 74/74 and the router
baseline harness passes 15/15.

P4/P6 layer-sweep note: `scripts/grm_gqa_layer_sweep.py` now runs the existing
native GQA benchmark across real Qwen3.5 capture layers with one native C ABI
build. First receipt: Qwen3.5-2B source captures, layers 3/7/11, full 256-token
K-banks, 256 nodes, query shape `(8,4,256)`, lexical off, OpenMP, native-only
timing, and exhaustive six-query batched-reference parity. Results were
parity-green on all three layers: layer 3 `11.5599ms` p50 / `11.8116ms` p95,
layer 7 `11.9944ms` p50 / `12.7472ms` p95, and layer 11 `12.0086ms` p50 /
`12.0181ms` p95. This extends real-capture full-bank evidence across layers
without moving or modifying the translation corpus artifacts.
Follow-up layer coverage extended the same 256-node full-bank exhaustive
batched-reference check to source layers 15/19/23 with `--capture-limit 512`.
All were parity-green: layer 15 measured `9.8055ms` p50 / `10.4419ms` p95,
layer 19 measured `12.0103ms` p50 / `13.2083ms` p95, and layer 23 measured
`12.1050ms` p50 / `16.2180ms` p95. The 256-node full-bank evidence now spans
source layers 3/7/11/15/19/23.
Receipt hygiene note: GQA benchmark and layer-sweep JSON/Markdown now record
the active `GRM_ROUTER_GQA_*` runtime flags and `GRM_BLAS_LIB`. This makes
future attention-state route receipts auditable: row-block, fused segment,
transposed, unroll8, BLAS, and forced key-bank paths are visible in the result
artifact instead of inferred from command history.

P4 fused-single-key experiment: `route_gqa_raw` now has an opt-in fused
single-key segment path behind `GRM_ROUTER_GQA_FUSED_SEGMENT=1`. It skips the
temporary key-score bank and writes the entry score directly when every route
entry has at most one prepared GQA key; `GRM_ROUTER_GQA_KEYBANK_SEGMENT=1`
forces the prior key-score-bank reducer. The focused single-key segment gate
passes against the Python raw-q.k law, but local Qwen3.5-2B layer-3 full-bank
512-node measurements did not show a decisive default win: fused measured
`20.9334ms` p50 / `21.1485ms` p95 with 6/6 batched-reference parity, while the
default key-score-bank confirmation measured `21.2234ms` p50 / `23.2118ms` p95
and an immediate key-score-bank comparison in the same shape measured
`17.7144ms` p50 / `19.4015ms` p95. The fused branch remains a tested diagnostic,
not the default runtime path.
Follow-up current-corpus remeasure kept that decision: at 500 current usable
source-layer-3 full-bank nodes, default measured `21.5515ms` p50 / `22.9931ms`
p95 while fused measured `19.9545ms` p50 / `21.8666ms` p95, both exhaustive
8/8 batched-reference parity green. At 1,000 nodes, default measured
`36.3616ms` p50 / `38.1237ms` p95 while fused regressed to `39.0891ms` p50 /
`41.7008ms` p95, again parity green. Finding: direct single-key writes help the
smaller point but do not scale well enough to become the default heuristic.

P4 GQA no-filter branch rejection: a GQA analogue of the MLA no-filter
active-only fast path was tested and reverted. It preserved exhaustive parity,
but the 500-node current full-bank point regressed from `21.5515ms` p50 to
`23.3020ms` p50 and `22.6285ms` p50 on repeat, while the 1,000-node point only
nudged from `36.3616ms` p50 / `38.1237ms` p95 to `36.1555ms` p50 / `37.4190ms`
p95. The mixed result is not worth a runtime branch; leave the existing
metadata helper in the GQA loops until a lower-level scorer changes the cost
profile.

P4 row-block experiment: `route_gqa_raw` now also has an opt-in query-head
row-block scorer behind `GRM_ROUTER_GQA_ROWBLOCK=1` for single-key, query-token-4
GQA banks. It computes `(entry, query_head)` blocks into a per-head score table
before reducing heads back to raw entry scores, preserving the same Python
raw-q.k law and tie order. Focused row-block parity passes. On Qwen3.5-2B source
layer-3 full-bank captures it measured 512 nodes at `18.4712ms` p50 /
`21.5356ms` p95 with 6/6 batched-reference parity, versus current default
`20.3634ms` p50 / `21.7664ms` p95 in the same code state. Larger points were
not default-worthy: 768 nodes measured `27.8408ms` p50 / `33.6079ms` p95 and
1,024 nodes measured `34.5670ms` p50 / `36.3997ms` p95, both parity-green but
not better than the existing bounded top-k full-bank receipts. The branch stays
opt-in while the default remains the key-score-bank segment reducer. Follow-up
guard hardening now computes `max_keys_per_entry` for row-block requests too, so
multi-key entries cannot accidentally enter the single-key row-block path; the
focused multi-key segment test runs with `GRM_ROUTER_GQA_ROWBLOCK=1` and still
matches the Python raw-q.k law.

P4 dot-kernel experiment: `GRM_ROUTER_GQA_QT4_UNROLL8=1` enables a manual
query-token-4 dot unroll inside the prepared GQA key scorer. It preserves the
raw-q.k law in focused parity, but it is rejected for runtime default use: on
the Qwen3.5-2B source layer-3 full-bank 512-node receipt it measured
`53.2750ms` p50 / `56.3159ms` p95 with 6/6 batched-reference parity, much slower
than the kept qt4 scorer. A follow-up Qwen3-4B preset representative-key check
was parity-green and flat at 512 nodes (`5.9949ms` p50 / `6.0773ms` p95 with
6/6 exhaustive batched-reference parity), but that run used harvested fixture
data, not 4B live capture shards. This confirms the next useful kernel work
should be a true compiler/BLAS-friendly row layout or external GEMM call, not
hand-unrolling the scalar inner loop.

P4 AVX2 dot-kernel note: `GRM_ROUTER_GQA_AVX2=1` now enables an opt-in AVX2
dot4 kernel for the query-token-4 GQA key scorer when the host CPU reports AVX2
support. The default scalar/SIMD-pragmas path remains unchanged, and unsupported
hosts fall back automatically. Focused native GQA parity passed with the AVX2
mode included in the qt4 mode matrix. On Qwen3.5-2B source layer-3 full
256-token K-bank captures, lexical off, OpenMP, exhaustive 8-query batched
parity: 500 nodes measured `15.8914ms` p50 / `17.6990ms` p95, and 1,000 nodes
measured `28.9156ms` p50 / `31.0866ms` p95. Both were parity-green and the
benchmark receipt records `GRM_ROUTER_GQA_AVX2=1`. This is the first
lower-level in-tree dot kernel with a real full-bank win, but it stays opt-in
until broader layer/model receipts prove the accumulation-order change is safe
enough to default. Follow-up layer sweeps: with `GRM_ROUTER_GQA_AVX2=1`,
Qwen3.5-2B source full-bank captures at layers 3/7/11/15/19/23, lexical off,
OpenMP, exhaustive 6-query batched parity, all passed. At 256 nodes, each layer
loaded 509 usable shards (`skipped_shape=3`), p50 ranged `11.2017ms` to
`12.9526ms`, and p95 maxed at `15.4401ms`. At 512 nodes, each layer loaded 747
usable shards (`skipped_shape=21`), p50 ranged `17.3686ms` to `19.8722ms`, and
p95 maxed at `28.6108ms`. At 1,024 nodes, each layer loaded 1,165 usable
shards (`skipped_shape=35`) and all layers remained parity-green, but p50
ranged `23.0302ms` to `30.1175ms` and p95 maxed at `32.3673ms`; only layer 3
cleared 25ms p50. The sweep receipts record the AVX2 runtime flag, and the
1,024-node result keeps AVX2 opt-in: correctness generalizes across the captured
layers, but latency still needs a lower-level GEMM/layout step.

P4 FMA/banked-scorer experiments: `GRM_ROUTER_GQA_FMA=1` enables an opt-in FMA
variant inside the AVX2 dot4 scorer; it requires `GRM_ROUTER_GQA_AVX2=1` and
host FMA support. Focused native GQA parity now includes the FMA mode. On
Qwen3.5-2B source layer-3 full-bank captures, FMA was parity-green but not a
1,024-node win: 512 nodes measured `16.4792ms` p50 / `18.8750ms` p95, while
1,024 nodes measured `28.2471ms` p50 / `29.0913ms` p95. It stays diagnostic
only. `GRM_ROUTER_GQA_BANKED=1` builds a bounded bank-transposed token-block
scorer for query-token-4 GQA banks. It also preserves parity, but is rejected
for runtime default use: 512 nodes measured `38.4846ms` p50 / `41.0115ms` p95,
and 1,024 nodes measured `70.2254ms` p50 / `72.5537ms` p95. This rules out the
simple host transposed-token layout; the next useful layout has to be a true
packed microkernel/GPU path rather than scratch-heavy token-block GEMV.

P4 paired-repeat AVX2 microkernel experiment: `GRM_ROUTER_GQA_PAIR2_AVX2=1`
enables an opt-in repeat-4 scorer for Qwen-style `8q/2kv` full-bank routes. It
computes two repeated query heads against the same K row in one AVX2 helper,
halving K-row streams versus the plain per-query-head AVX2 dot4 path. Focused
native GQA parity includes the mode, and Qwen3.5-2B source layer-3 full-bank
captures stayed parity-green. Runtime rejected it: 512 nodes measured
`23.9675ms` p50 / `25.8050ms` p95, and 1,024 nodes measured `29.1954ms` p50 /
`29.5725ms` p95. The result says repeated-head K reuse alone is not enough on
this CPU; the next meaningful step is GPU/cuBLAS or a tighter hand-packed kernel
that improves arithmetic scheduling as well as memory traffic.

P4 CUDA/cuBLAS probe: `scripts/grm_gqa_cuda_probe.py` now compiles a temporary
CUDA shared library with `nvcc`, pre-packs full captured K banks into
device-resident column-major matrices, runs two `cublasSgemm` calls per query
for the Qwen3.5-2B `8q/2kv` shape, and reduces raw GQA scores on GPU before
checking top-k parity against the existing batched Python law. This is a probe
and not yet wired into the runtime C ABI: one-shot wall time includes allocation,
H2D copy, and K packing. Device-resident route timing is the relevant runtime
target. Qwen3.5-2B source layer-3 full-bank captures were parity-green:
32 nodes measured `0.0580ms` per query, 512 nodes measured `0.9544ms`, and
1,024 nodes measured `1.4769ms`. The 1,024-node one-shot wall was `262.7543ms`,
so the next implementation slice is to make the GPU K arena persistent and
return top-k without copying full score vectors back to host.

P4 CUDA GPU-top-k note: the CUDA probe now performs top-k selection on device
and returns only the winning node IDs to host (`output_mode=gpu_topk`), removing
the full route-score copy from the measured route loop. This remains a
standalone probe and not a runtime C ABI path. Qwen3.5-2B source layer-3
full-bank captures stayed parity-green: 32 nodes measured `0.0819ms` per query,
512 nodes measured `0.8348ms`, and a longer 1,024-node run measured `1.5427ms`.
One short 1,024-node run measured `2.3217ms`, so the kept receipt uses the
longer six-measured-query average. The next implementation slice is now the
persistent GPU K arena plus runtime integration; GPU top-k itself is proven at
probe scope for `topk <= 16`.

P4 persistent CUDA K-arena note: the probe now exposes create/route/destroy
entrypoints and keeps the packed full K bank device-resident across route calls
(`arena_mode=persistent_gpu_k`). The old one-shot C entrypoint remains as a
compatibility wrapper. Sequential Qwen3.5-2B source layer-3 full-bank receipts
with GPU top-k stayed parity-green: 32 nodes measured `0.0911ms` per query with
`126.4656ms` setup / `50.8418ms` route wall, 512 nodes measured `0.8520ms` with
`156.7580ms` setup / `55.7224ms` route wall, and 1,024 nodes measured
`1.4048ms` with `183.2137ms` setup / `61.6378ms` route wall. Those route-wall
numbers still include first-route scratch allocation; the probe no longer
rebuilds or recopies the K arena on each route.

P4 persistent CUDA scratch note: route scratch/query buffers now belong to the
same `GqaDeviceArena` handle and grow only when query/top-k shape increases
(`scratch_mode=persistent_route_buffers`). The CLI can repeat the same route
against one arena and reports the final reused-scratch run. Sequential
Qwen3.5-2B source layer-3 full-bank receipts stayed parity-green: 32 nodes
measured `0.0901ms` per query with reused route wall `0.2558ms`, 512 nodes
measured `0.8530ms` with reused route wall `5.0754ms`, and 1,024 nodes measured
`1.5713ms` with reused route wall `12.6367ms`. Remaining CUDA work is runtime C
ABI integration; at probe scope, K arena, route scratch, and GPU top-k are now
persistent.

P4 CUDA sidecar note: `core/grm_cuda_router.py` now exposes an optional
runtime-facing CUDA sidecar wrapper without changing the dependency-free CPU
router default. `CudaGQARouteSidecar` builds or loads the CUDA shared library,
`CudaGQARouteBank` owns a device-resident route bank and maps CUDA row top-k back
to GRM node IDs, and validation is covered by non-GPU tests. A direct sidecar
smoke over 32 Qwen3.5-2B source layer-3 full-bank nodes with node IDs remapped to
`1000..1031` matched the batched Python law 2/2, measuring `0.0809ms` per query
and `0.2835ms` reused route wall. Remaining work is wiring this sidecar under a
native/router snapshot policy or adding an explicit nvcc-enabled runtime build
mode; the existing CPU C ABI remains unchanged.

P4 native CUDA attachment note: `NativeGraftStore` now has an explicit
`configure_cuda_gqa_route_bank()` / `route_gqa_cuda()` path. The default
`route_gqa()` C ABI remains CPU and dependency-free; CUDA is only used after a
caller attaches a dense GQA route bank and node-id mapping. CPU-only regression
coverage verifies `route_gqa_cuda()` fails closed without that explicit bank. A
native-wrapper CUDA smoke built the normal `libgrm_runtime.so`, added 32 GQA
nodes, attached the Qwen3.5-2B source layer-3 full-bank route bank, and matched
the batched Python law for top-5: `[11, 31, 4, 15, 29]`, with `0.1056ms` per
query and `0.2266ms` reused route wall. Route-key, active-state,
route-metadata, revision, expire, and clear-route mutations now close any
attached CUDA bank so stale explicit GPU route state fails closed; CPU-only
regressions cover route-key and eligibility invalidation.

P4 opt-in CUDA routing policy note: `ArenaCache.route()` now accepts an optional
`limit` so turn execution can ask for only the mount-window ranking it actually
uses instead of forcing a full repository order. `GQAArenaCache` uses that
contract to try `route_gqa_cuda()` when `GRM_GQA_CUDA_ROUTE=1`, the query has no
lexical exact-match keys, and the repository can build a dense same-shape,
single-key GQA bank from native node IDs. Mixed shapes, child-centroid/digest
keys, lexical queries, missing CUDA, incomplete top-k after excludes, or CUDA
errors fall back to the existing CPU native route. CPU-only regressions cover
the CUDA auto-attachment path and the lexical fallback path; the CUDA top-k cap
still limits this path to the step/mount-window contract rather than replacing
full-rank `arena.route()` calls.

P4 CUDA route-bank lifecycle note: `GQAArenaCache` now binds an attached CUDA
route bank to a cheap route-snapshot signature over native node IDs, key shapes,
dtypes, and route-key object identities. A route call reuses the GPU bank only
when that signature still matches; appended/replaced dense GQA rows force a new
`configure_cuda_gqa_route_bank()` attachment instead of reusing stale GPU row
state. Direct `NativeGraftStore.configure_cuda_gqa_route_bank()` attachments
also record a deterministic content signature from the validated dense bank, and
`clear_cuda_gqa_route_bank()` clears that marker with the device bank. CPU-only
regression coverage now checks explicit-bank failure, route-key mutation
invalidation, active-state invalidation, opt-in CUDA routing, lexical fallback,
and row-change reattachment:
```
PYTEST_ADDOPTS='-p no:cacheprovider' python3 -m pytest
tests/test_grm_native_runtime.py::test_native_gqa_cuda_route_requires_explicit_bank
tests/test_grm_native_runtime.py::test_native_gqa_route_mutation_clears_cuda_bank
tests/test_grm_native_runtime.py::test_native_gqa_eligibility_mutation_clears_cuda_bank
tests/test_grm_native_runtime.py::test_gqa_arena_uses_opt_in_cuda_route_bank
tests/test_grm_native_runtime.py::test_gqa_arena_rebuilds_cuda_route_bank_when_rows_change
tests/test_grm_native_runtime.py::test_gqa_arena_skips_cuda_route_for_lexical_queries
```
passed `6 passed, 2 warnings in 9.65s` on 2026-07-07.
The broader native-runtime regression then passed:
```
PYTEST_ADDOPTS='-p no:cacheprovider' python3 -m pytest tests/test_grm_native_runtime.py
```
Result: `90 passed, 2 warnings in 248.88s`.

P4 live CUDA bridge receipt: `scripts/grm_gqa_cuda_bridge_smoke.py` now has a
device-visible validation run on real Qwen3.5-2B source layer-3 capture banks.
The opt-in arena bridge (`GRM_GQA_CUDA_ROUTE=1`) matched both the batched Python
raw `|q.k|` reference and direct `NativeGraftStore.route_gqa_cuda()` at 32, 128,
and 512 full 256-token K-bank nodes. Reused bridge min wall was `3.15747ms`,
`11.217836ms`, and `38.960699ms`; direct CUDA device/query was `0.098496ms`,
`0.226304ms`, and `0.740352ms`.

P4 transposed-bank experiment: `GRM_ROUTER_GQA_TRANSPOSED=1` builds an opt-in
transposed prepared GQA key bank and routes query-token-4 keys through it. The
path preserves the raw-q.k law in focused parity and exhaustive benchmark
parity, but it is rejected for default use on the hard full-bank capture shape:
Qwen3.5-2B source layer-3 full-bank 512 nodes measured `26.3718ms` p50 /
`28.2460ms` p95 with 6/6 batched-reference parity, while the final-code default
measured `19.7473ms` p50 / `23.7554ms` p95. The Qwen3-4B preset
representative-key fixture stayed effectively flat (`5.9930ms` p50 /
`5.9973ms` p95 with 6/6 parity versus default `5.9983ms` p50 / `5.9991ms`
p95), but that fixture has one-token keys and is not 4B live capture-shard
evidence. Disabled runs do not build the duplicate transposed bank.

P4 CBLAS GEMM experiment: the native runtime can now be compiled with
`GRM_ENABLE_CBLAS` and the benchmark helper exposes `--blas`, linking
`${GRM_BLAS_LIB:-/lib/x86_64-linux-gnu/libblas.so.3}`. At runtime
`GRM_ROUTER_GQA_BLAS=1` builds an opt-in per-KV-head prepared key matrix and
uses CBLAS `sgemm` for the query-token-4 GQA segment scorer before reducing the
GEMM result back to graft/key segments. The path is parity-green but rejected
for default use on this host's generic BLAS: Qwen3.5-2B source layer-3 full-bank
512 nodes measured `833.7949ms` p50 / `847.2152ms` p95 with 6/6
batched-reference parity, versus default `20.9562ms` p50 / `21.3305ms` p95.
The Qwen3-4B representative-key fixture also regressed mildly (`6.3741ms` p50 /
`6.3879ms` p95 versus default `5.9956ms` p50 / `6.0177ms` p95). The result is
still useful: the API seam for a true GEMM-backed segment scorer exists, but
generic system BLAS is the wrong backend; future work should target an in-tree
packed kernel, OpenBLAS/MKL receipt, or GPU/cuBLAS path before defaulting.

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
`6.0286ms` p95. D5 sanitizer hardening now has a standalone TSAN stress gate:
`scripts/grm_router_tsan_gate.py` compiles `cpp/grm_runtime.cpp` with
`-fsanitize=thread` and stresses concurrent `route_gqa` readers against route,
active, metadata, revision, and expire writers. On this host GCC TSAN first
hits an ASLR mapping issue, the script retries with `setarch x86_64 -R`, and
the gate passed at 48 nodes, 1,200 writer iterations, and 4 reader threads.
Post-gate validation: focused native router selector passed 5/5 and router
baseline passed 15/15. The route hot path itself is now read-lock-free for clean
prepared snapshots. The TSAN harness uses tiny Qwen3.5_TC-shaped data for race
coverage; it is not a Qwen3-4B or Qwen3.5-2B performance benchmark.

**P6 — Report.** `ROUTER_SCALING_REPORT.md`: before/after curves at all
four node counts, both dialects, exactness-gate record, M-sweep table.
Board update; note for GRM paper v1.1 §5 (routing limitation retired —
with measured numbers, not projections).

P6 report note: `docs/ROUTER_SCALING_REPORT.md` records the measured MLA and
GQA router state: P0 native/Python baselines, P2 fp32 arena large points, P3
INT4 M-sweeps and bounded-staging operating point, P4 GQA key-bank and
real-capture curves, CUDA/cuBLAS probe receipts, runtime CUDA sidecar/bridge
status, P5 snapshot gates, expectation pass/miss status, and follow-up work.
The report is intentionally explicit about misses: host E3 clears p50 but not
p95, generic BLAS is rejected, and simple representative-key compaction is not
correct on the measured capture set.

**Deferred / future work:** ANN/IVF structures are only relevant past roughly
10M nodes and remain out of scope. CUDA is no longer merely deferred: probe,
sidecar, explicit native attachment, invalidation, and opt-in arena policy are
implemented. The remaining CUDA item is live bridge validation on a process
with actual NVIDIA device access, plus a larger-top-k/full-rank route contract
only if CUDA should replace general `arena.route()` calls.

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
