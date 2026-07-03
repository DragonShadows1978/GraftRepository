# GRM Router Scaling Report

Status: GEMV-router implementation checkpoint on branch
`grm-ram-tiered-runtime`.

This report records measured router scaling after replacing the original
per-node route scan with contiguous host arenas:

- MLA: fp32 SoA arena plus INT4 bulk/refine route book.
- GQA: contiguous key-bank arena with segment-style per-entry reduction and
  real-capture layer-sweep coverage.
- Concurrency: route-key mutations mark arenas dirty, the next route prepares
  once under the writer side of the C ABI shared mutex, and clean prepared
  route reads atomically pin immutable snapshots without a read-side mutex.

All numbers below are from local measurements on harvested repository-derived
route vectors unless otherwise noted. Missing direct baselines are marked as
missing rather than inferred.

## MLA Route Curves

P0 measured the original native scan and Python fallback at 1k/10k:

| nodes | dim | native p50 ms | native p95 ms | Python p50 ms | parity |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 1,000 | 128 | 0.4985 | 0.5405 | 25.2087 | true |
| 10,000 | 128 | 4.0865 | 4.7191 | 248.3771 | true |

P2 fp32 arena, native-only OpenMP:

| nodes | dim | fp32 arena p50 ms | fp32 arena p95 ms | parity |
| ---: | ---: | ---: | ---: | :---: |
| 100,000 | 128 | 22.8177 | 23.8262 | not run |
| 250,000 | 128 | 57.8062 | 58.5520 | not run |
| 1,000,000 | 128 | 211.3071 | 217.1046 | not run |

P3 INT4 bulk/refine checkpoints:

| nodes | dim | mode | refine M | p50 ms | p95 ms | parity reference | parity |
| ---: | ---: | --- | ---: | ---: | ---: | --- | :---: |
| 100,000 | 128 | predecoded q4 | 16 | 6.2995 | 10.4405 | native fp32 | true |
| 1,000,000 | 128 | q4 norm-scale | 16 | 43.4791 | 49.3438 | native fp32 | true |
| 1,000,000 | 128 | predecoded q4 | 16 | 39.9895 | 41.4288 | native fp32 | false |
| 1,000,000 | 128 | predecoded q4 | 32 | 39.6265 | 41.7902 | native fp32 | true |
| 1,000,000 | 128 | predecoded q4 | 64 | 37.5690 | 40.2792 | native fp32 | true |
| 1,000,000 | 128 | predecoded q4 | 128 | 37.8382 | 39.1211 | native fp32 | true |
| 1,000,000 | 128 | predecoded q4 + bounded candidate staging | 64 | 26.0175 | 27.2486 | native fp32 | true |
| 1,000,000 | 128 | bounded staging, longer query check | 64 | 23.4215 | 25.9012 | native fp32 | false |
| 1,000,000 | 128 | bounded staging, longer query check | 96 | 24.7049 | 27.0196 | native fp32 | true |
| 1,000,000 | 128 | bounded staging, longer query check | 128 | 23.8805 | 25.4883 | native fp32 | true |
| 1,000,000 | 128 | bounded staging, longer query check | 256 | 26.1488 | 29.1560 | native fp32 | true |
| 1,000,000 | 128 | bounded staging, current corpus rerun | 128 | 27.0773 | 33.4480 | native fp32 | true |
| 1,000,000 | 128 | no-filter active fast path, current corpus | 128 | 24.9673 | 29.1474 | native fp32 | true |

Current measured MLA operating point: bounded-staging predecoded q4, `M=128`,
1M nodes, 23.8805ms p50 with native-fp32 parity over the longer 10-query
receipt. `M=64` is no longer considered safe generally: it passed the shorter
receipt but flipped ranks 2/3 on query 8 of the longer check.

## INT4 Exactness Sweep

Earlier 100k exactness sweeps matched native fp32 for all measured M values:
16, 32, 64, 128, 256, 512, 1024, 2048, and 4096.

The later predecoded 1M sweep changed the safe operating point:

| refine M | 1M parity | p50 ms |
| ---: | :---: | ---: |
| 16 | false | 39.9895 |
| 32 | true | 39.6265 |
| 64 | true | 37.5690 |
| 128 | true | 37.8382 |
| 64 + bounded staging | true | 26.0175 |
| 64 + bounded staging, longer check | false | 23.4215 |
| 96 + bounded staging, longer check | true | 24.7049 |
| 128 + bounded staging, longer check | true | 23.8805 |
| 256 + bounded staging, longer check | true | 26.1488 |

Conclusion: refine count is not the dominant runtime limiter, but too-small
M can still lose exact top-k after the predecoded q4 bulk approximation. Use
M=128 with bounded candidate staging for the current harvested-corpus operating
point. This clears E3 on p50 (`23.8805ms`) but not p95 (`25.4883ms`). A rejected
thread-local heap selection attempt stayed parity-green but slowed the older
M=64 point to 29.7308ms p50, so it was not kept.

The later no-filter active fast path is a same-corpus hot-path comparison, not
a replacement for the older operating-point receipt. On the current q10
8192-row harvested corpus, M=128 improved from 27.0773ms p50 / 33.4480ms p95
to 24.9673ms p50 / 29.1474ms p95 while preserving native-fp32 parity. The tail
is still above 25ms.

## GQA Key-Bank Probe

P4 first slice replaced the GQA per-entry route scan with a contiguous key-bank
arena and OpenMP entry scoring. Smoke probe:

- nodes: 10,000
- query shape: `(4, 4, 16)`
- key shape: `(1, 4, 16)`
- top-k: 5
- build flags: `-O3 -fopenmp`

| backend | p50 ms | parity |
| --- | ---: | :---: |
| native GQA key-bank | 6.0020 | true |
| Python/NumPy reference | 166.9182 | true |

Measured speedup: 27.81x on this smoke shape. Broader non-capture GQA gate
scenarios still need a full P6 curve.

Qwen3.5-2B attention-shaped representative-key probe:

- preset: `qwen35-2b-attn`
- query shape: `(8, 4, 256)`
- key shape: `(2, 1, 256)`
- keys per node: 1
- top-k: 5
- build flags: `-O3 -fopenmp`

| nodes | native p50 ms | native p95 ms | Python p50 ms | parity |
| ---: | ---: | ---: | ---: | :---: |
| 1,000 | 2.4416 | 2.5654 | 22.0276 | true |
| 10,000 | 8.6680 | 9.0765 | 227.7100 | true |
| 10,000, bounded top-k selection | 5.9959 | 5.9989 | n/a | true (5 queries) |

This curve uses the Qwen3.5-2B source attention geometry, not Qwen3-4B.

Qwen3.5-2B real-capture source K-bank probe:

- source: `/mnt/ForgeRealm/qwen35_graft_translation_poc/captures`
- role/layer/key: source, layer 3, `l3_k`
- full K-bank shape: `(2, 256, 256)` per route node
- representative K-bank shape: `(2, 1, 256)` per route node
- query source: deterministic probes derived from captured K banks
- lexical channel: off
- build flags: `-O3 -fopenmp`

| route keys | nodes | native p50 ms | native p95 ms | Python p50 ms | parity |
| --- | ---: | ---: | ---: | ---: | :---: |
| captured full 256-token K | 32 | 4.4014 | 4.9848 | 14.7914 | true |
| captured full 256-token K | 96 | 9.5322 | 14.4400 | 42.2292 | true |
| captured full 256-token K | 127 | 10.8340 | 12.9397 | 53.8978 | true |
| captured full 256-token K | 192 | 15.1623 | 16.1852 | 80.1290 | true |
| captured full 256-token K | 256 | 18.9169 | 21.9172 | 107.5580 | true |
| captured full 256-token K, pre-qt4 native-only | 512 | 39.5778 | 42.3518 | n/a | n/a |
| captured full 256-token K, pre-qt4 sampled parity | 512 | 39.6250 | 43.4202 | n/a | true (1 query) |
| captured full 256-token K, qt4 sampled parity | 512 | 19.0136 | 21.9898 | n/a | true (2 queries) |
| captured full 256-token K, qt4 exhaustive parity | 512 | 20.4087 | 21.8582 | n/a | true (4 queries) |
| captured full 256-token K, qt4 bounded top-k exhaustive parity | 512 | 19.1065 | 23.3995 | n/a | true (4 queries) |
| captured full 256-token K, pre-qt4 native-only | 768 | 58.3090 | 63.2833 | n/a | n/a |
| captured full 256-token K, pre-qt4 batched sampled parity | 768 | 58.5005 | 64.6167 | n/a | true (2 queries) |
| captured full 256-token K, qt4 batched sampled parity | 768 | 33.8389 | 34.8920 | n/a | true (2 queries) |
| captured full 256-token K, qt4 bounded top-k exhaustive parity | 768 | 26.0228 | 29.5759 | n/a | true (4 queries) |
| captured full 256-token K, pre-qt4 native-only | 1,024 | 76.5800 | 80.9969 | n/a | n/a |
| captured full 256-token K, pre-qt4 batched sampled parity | 1,024 | 73.5912 | 75.1033 | n/a | true (2 queries) |
| captured full 256-token K, qt4 batched sampled parity | 1,024 | 36.7363 | 38.6939 | n/a | true (2 queries) |
| captured full 256-token K, qt4 bounded top-k exhaustive parity | 1,024 | 33.9970 | 39.3356 | n/a | true (4 queries) |
| captured representative 1-token K | 32 | 0.0465 | 0.0473 | 0.6651 | true |
| captured representative 1-token K | 96 | 0.1172 | 0.1696 | 2.0503 | true |
| captured stride-16 K vs full K reference | 1,024 | 6.9564 | 7.9222 | n/a | false |
| captured stride-32 K vs full K reference | 1,024 | 16.5056 | 23.0100 | n/a | false |
| captured stride-64 K vs full K reference | 1,024 | 19.3099 | 25.0374 | n/a | false |
| captured stride-128 K vs full K reference | 1,024 | 37.7707 | 45.3476 | n/a | false |

The live translation-corpus source shards were inspected and read in-place only;
no generated graft/capture files were moved or deleted. Full-bank routing first
measured slower than NumPy at 96 nodes (`56.1185ms` native p50 vs `40.9139ms`
Python p50) because the native GQA OpenMP threshold keyed only on entry count.
The runtime now uses a workload-aware GQA threshold based on
`entries * query_heads * query_tokens * key_tokens`, improving the same 96-node
full-bank point to `16.6903ms` p50 while preserving parity. A tiled scorer that
computed all query rows for a KV head against one K row was tested next and
rejected: it stayed parity-green but regressed the 127-node full-bank p50 to
`29.2260ms`. The kept hot-loop change accumulates arena q.k dots in `float`,
matching the float32 benchmark tensors instead of promoting every multiply to
`double`; it improves the 127-node full-bank p50 to `10.8340ms` with parity
green. A larger real-capture run loaded 298 usable source shards and measured
192/256 full-bank nodes at `15.1623ms` / `18.9169ms` p50, still parity-green.
The benchmark now supports `--progress-out` JSONL checkpoints and `--progress`
stderr updates; a native-only 1,165-usable-shard run extended the full-bank
curve to 512/768/1,024 nodes at `39.5778ms` / `58.3090ms` / `76.5800ms` p50.
Those larger points are native-only (`parity=null`) because Python full-bank
reference timing dominates at that scale. Native-only capture runs can now add
`--parity-sample-queries`; the first sampled-parity 512-node full-bank check
loaded 595 usable Qwen3.5-2B source shards, matched one deterministic Python
raw-q.k reference query, and measured `39.6250ms` p50 / `43.4202ms` p95.
The Python reference also has a batched mode that preserves the scalar law's
tie order; using that mode, 768 and 1,024 full-bank nodes matched two sampled
Python queries and measured `58.5005ms` / `73.5912ms` p50.
The native full-bank scorer now has a query-token-4 fast path that reuses each
K row across the four query rows while preserving per-row dot accumulation
order. On the same Qwen3.5-2B source captures, 512/768/1,024 full-bank sampled
parity points now measure `19.0136ms` / `33.8389ms` / `36.7363ms` p50.
Native-only capture runs now also support `--parity-all-queries`, which runs the
Python raw-q.k reference for every generated query without timing Python in the
route curve. A 512-node full-bank Qwen3.5-2B source check matched all four
generated batched-reference queries and measured `20.4087ms` p50 /
`21.8582ms` p95.
Native GQA result selection now uses bounded top-k partial sorting when `topk`
is smaller than the scored candidate set, while preserving the existing
score-plus-node-id tie order. The harvested representative 10k-node GQA point
measured `5.9959ms` p50 with exhaustive five-query parity, and the 512-node
full-bank Qwen3.5-2B capture point matched all four generated batched-reference
queries at `19.1065ms` p50 / `23.3995ms` p95.
The same exhaustive full-bank check now extends to 768 and 1,024 nodes: 768
matched all four batched-reference queries at `26.0228ms` p50 / `29.5759ms`
p95, and 1,024 matched all four at `33.9970ms` p50 / `39.3356ms` p95.
The no-lexical hot path now skips empty lexical hashing, lexical hit counting,
and score normalization; it ranks raw GQA scores directly because normalization
is monotonic when lexical bonus is absent. This is a small cleanup, not a new
scorer: representative 10k measured `5.9905ms` p50 with exhaustive five-query
parity, and 512 full-bank measured `19.3370ms` p50 / `22.8915ms` p95 with all
four batched-reference queries matched.
The benchmark can now route compact representative-token capture banks while
checking parity against the original full-bank reference. Simple stride
compaction is not safe on this capture set: 16/32/64/128-token stride banks at
1,024 nodes all failed the two-query full-bank sampled-parity check. The
16-token case is sub-10ms (`6.9564ms`) but changes top-k; 128 tokens is still
wrong and slower than the qt4 full-bank path.
A grouped qt4 scorer that tried to reuse each K row across repeated query heads
was also measured and rejected. It preserved sampled parity, but regressed the
512-node p50 to `20.3603ms` and the 1,024-node p50 to `38.2764ms`, both slower
than the kept qt4 full-bank scorer.
A hand-unrolled repeat-4 head-ratio scorer (`8q/2kv`, not a 4B model run) was
measured next. It also preserved sampled parity, but regressed 512 nodes to
`23.1079ms` and only matched the 1,024-node point within noise (`36.6391ms`),
so it was removed rather than kept behind a flag.
A paired repeated-head scorer that reused each K row across two query heads at a
time was measured next. It also preserved parity, but regressed the fresh
512-node p50 from `19.8082ms` to `22.3257ms`, so no C++ scorer change was kept.

The GQA route path now includes an automatic segment-reduce scorer for larger
prepared key banks. The scorer computes one raw score per prepared key, stores a
key-score bank, and reduces each entry's key segment before the existing top-k
selection. It is forced by `GRM_ROUTER_GQA_SEGMENT=1` and selected
automatically when `max_key_tokens >= 32`; smaller representative-key banks stay
on the per-entry scorer because the extra key-score buffer is not helpful there.
Focused GQA selectors, including the forced segment path, passed 6/6. Full
native runtime passed 74/74, and the router baseline harness passed 15/15.

| route shape | nodes | native p50 ms | native p95 ms | parity |
| --- | ---: | ---: | ---: | --- |
| Qwen3.5-2B source layer-3 full 256-token K-bank, auto segment | 512 | 20.4251 | 21.3550 | true, 4/4 batched-reference queries |
| harvested representative-key GQA, auto per-entry | 10000 | 5.9844 | 6.0227 | true, 5/5 batched-reference queries |
| harvested representative-key GQA, forced segment | 10000 | 6.0345 | 7.7158 | true, 5/5 batched-reference queries |

The runtime also has a tested but non-default fused single-key segment path:
`GRM_ROUTER_GQA_FUSED_SEGMENT=1` writes entry scores directly when every entry
has at most one GQA route key, avoiding the temporary key-score bank and serial
entry reduction. `GRM_ROUTER_GQA_KEYBANK_SEGMENT=1` forces the prior reducer for
comparison. On the Qwen3.5-2B source layer-3 full-bank 512-node case, fused
single-key segment measured `20.9334ms` p50 / `21.1485ms` p95 with exhaustive
6/6 batched-reference parity. The default key-score-bank path stayed parity
green and was not decisively slower (`21.2234ms` p50 / `23.2118ms` p95 in the
default confirmation; `17.7144ms` p50 / `19.4015ms` p95 in an immediate forced
key-score-bank comparison), so fused single-key remains opt-in rather than a
runtime default.

`GRM_ROUTER_GQA_ROWBLOCK=1` enables a second non-default experiment: a
query-head row-block scorer for single-key, query-token-4 GQA banks. It computes
per `(entry, query_head)` blocks into a temporary head-score table and then
reduces heads back to entry raw scores. This is closer to the requested
GEMM/segment-reduce shape than the single-key shortcut, but the first
real-capture result is mixed. On Qwen3.5-2B source layer-3 full-bank captures:

| mode | nodes | native p50 ms | native p95 ms | parity |
| --- | ---: | ---: | ---: | --- |
| default key-score-bank segment | 512 | 20.3634 | 21.7664 | true, 6/6 batched-reference queries |
| row-block opt-in | 512 | 18.4712 | 21.5356 | true, 6/6 batched-reference queries |
| row-block opt-in | 768 | 27.8408 | 33.6079 | true, 6/6 batched-reference queries |
| row-block opt-in | 1,024 | 34.5670 | 36.3997 | true, 6/6 batched-reference queries |

The 512-node point improved p50, but the 768/1,024 points did not beat the
existing bounded top-k full-bank receipts. Row-block therefore remains an
opt-in measured candidate rather than the default runtime path.
The row-block guard now also computes `max_keys_per_entry` when
`GRM_ROUTER_GQA_ROWBLOCK=1`, preventing multi-key routes from entering this
single-key-only path. The focused multi-key segment regression runs with
row-block requested and still matches the Python raw-q.k law.

`GRM_ROUTER_GQA_QT4_UNROLL8=1` enables a manual unroll-8 variant of the
query-token-4 key dot kernel. It is parity-green but rejected for runtime default
use: the Qwen3.5-2B source layer-3 full-bank 512-node receipt measured
`53.2750ms` p50 / `56.3159ms` p95 with 6/6 batched-reference parity, far slower
than the kept qt4 scorer. On the Qwen3-4B preset representative-key fixture, the
same flag stayed flat at 512 nodes (`5.9949ms` p50 / `6.0773ms` p95, 6/6
exhaustive batched-reference parity), but that was harvested fixture data rather
than 4B live capture shards.

`GRM_ROUTER_GQA_TRANSPOSED=1` builds a duplicate transposed prepared key bank and
routes query-token-4 GQA keys over that layout. It is parity-green but rejected
for runtime default use on the hard full-bank capture shape: Qwen3.5-2B source
layer-3 full-bank 512 nodes measured `26.3718ms` p50 / `28.2460ms` p95 with 6/6
batched-reference parity, slower than the final-code default `19.7473ms` p50 /
`23.7554ms` p95. On the Qwen3-4B preset representative-key fixture, transposed
was effectively flat (`5.9930ms` p50 / `5.9973ms` p95, 6/6 parity) against
default (`5.9983ms` p50 / `5.9991ms` p95), but that fixture uses one-token keys
and is not 4B live capture-shard evidence. The duplicate transposed bank is only
built when the flag is enabled.

`GRM_ENABLE_CBLAS` + `GRM_ROUTER_GQA_BLAS=1` adds a true CBLAS `sgemm` segment
experiment for query-token-4 GQA banks. Benchmark helpers expose `--blas`, which
links `${GRM_BLAS_LIB:-/lib/x86_64-linux-gnu/libblas.so.3}` because this host has
the runtime library but not the `libblas.so` development symlink. The path is
parity-green and compile/runtime gated, but generic BLAS is rejected for default
use:

| shape | nodes | default p50 | BLAS p50 | parity |
| --- | ---: | ---: | ---: | --- |
| Qwen3.5-2B source layer-3 full 256-token K-bank | 512 | 20.9562 | 833.7949 | true, 6/6 batched-reference queries |
| Qwen3-4B preset representative-key fixture | 512 | 5.9956 | 6.3741 | true, 6/6 batched-reference queries |

The receipt proves the GEMM/segment-reduce API shape but not a useful backend on
this machine. Future work should use an in-tree packed kernel, a real optimized
CPU BLAS receipt, or GPU/cuBLAS rather than defaulting generic `libblas.so.3`.

## GQA Capture Layer Sweep

`scripts/grm_gqa_layer_sweep.py` wraps the existing GQA benchmark internals and
builds the native C ABI library once before sweeping real Qwen3.5 capture
layers. The first receipt used Qwen3.5-2B source captures, full 256-token
K-banks, query shape `(8,4,256)`, `topk=5`, OpenMP, lexical off, native-only
timing, and exhaustive six-query batched-reference parity.

Command:

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/grm_gqa_layer_sweep.py \
  --capture-dir /mnt/ForgeRealm/qwen35_graft_translation_poc/captures \
  --capture-role source \
  --capture-limit 0 \
  --layers 3 7 11 \
  --node-counts 256 \
  --queries 6 \
  --warmup 2 \
  --query-tokens 4 \
  --topk 5 \
  --openmp \
  --parity-reference batched \
  --out /tmp/grm_gqa_layer_sweep_3_7_11_256.json \
  --markdown-out /tmp/grm_gqa_layer_sweep_3_7_11_256.md \
  --progress
```

| layer | nodes | usable shards | native p50 ms | native p95 ms | parity |
| ---: | ---: | ---: | ---: | ---: | --- |
| 3 | 256 | 9,635 | 11.5599 | 11.8116 | true, 6/6 batched-reference queries |
| 7 | 256 | 9,635 | 11.9944 | 12.7472 | true, 6/6 batched-reference queries |
| 11 | 256 | 9,635 | 12.0086 | 12.0181 | true, 6/6 batched-reference queries |

## Prepared Router Snapshots

The C ABI router now publishes prepared MLA/GQA state as an immutable
`std::shared_ptr<const RouterIndex>` snapshot. Route-key, active-state,
route-metadata, revision, expire, clear, and checkpoint rebuild mutations update
the live router under the writer lock and dirty the prepared snapshot. The next
prepared route rebuilds the dialect arena once and publishes the copy; prepared
MLA/non-GQA routes and raw GQA `route_gqa` calls then score from that snapshot
outside the mutex. Generic MLA-style `route` calls on GQA stores remain on the
writer-lock path because they can still lazily build an MLA arena over GQA
route keys.

The first full native test run caught stale snapshots on active/filter-only
mutations. Dirty marks were added for active-state, route-metadata, revision,
and expire changes. Focused stale-snapshot/concurrency regression:
`6 passed, 67 deselected`. Full native runtime gate:
`73 passed, 2 warnings in 174.95s`.

The follow-up atomic fast path removes the read-side shared lock for clean
prepared snapshots. A dirty atomic bit protects semantics after writer
mutations: if clean, readers atomically load and pin the immutable snapshot for
the route call; if dirty or missing, they enter the writer-side prepare path and
publish a fresh copy. Focused stale-snapshot/concurrency regression:
`7 passed, 66 deselected`. Full native runtime gate:
`73 passed, 2 warnings in 177.02s`.

A standalone sanitizer race gate now covers the atomic snapshot handoff:
`scripts/grm_router_tsan_gate.py` compiles `cpp/grm_runtime.cpp` with
`-fsanitize=thread` and stresses concurrent `route_gqa` readers against route,
active, metadata, revision, and expire writer mutations. On this host GCC TSAN
first reports `ThreadSanitizer: unexpected memory mapping`; the script retries
with `setarch x86_64 -R`, and the gate passed at 48 nodes, 1,200 writer
iterations, and 4 reader threads. This harness uses tiny Qwen3.5_TC-shaped route
data for race coverage only; it does not load Qwen3.5-2B or Qwen3-4B weights and
is not a routing-performance benchmark. Post-gate validation: focused native
router selector passed 5/5 and router baseline passed 15/15.

Fresh post-snapshot GQA receipts:

| route shape | nodes | native p50 ms | native p95 ms | parity |
| --- | ---: | ---: | ---: | --- |
| Qwen3.5-2B source layer-3 full 256-token K-bank | 512 | 21.8605 | 22.1071 | true, 4/4 batched-reference queries |
| harvested representative-key GQA | 10000 | 5.9662 | 5.9920 | true, 5/5 batched-reference queries |
| Qwen3.5-2B source layer-3 full 256-token K-bank, atomic snapshot | 512 | 22.0019 | 23.8365 | true, 4/4 batched-reference queries |
| harvested representative-key GQA, atomic snapshot | 10000 | 5.9916 | 6.0286 | true, 5/5 batched-reference queries |

## Expectations

| expectation | result |
| --- | --- |
| E1: fp32 GEMV 10x current native scan at 100k | not proven; no direct P0 100k run |
| E2: INT4 two-tier 2x over fp32 at 100k | passed on measured points: 22.8177ms -> 6.2995ms |
| E3: 1M route <= 25ms host-side | passed on p50 with bounded-staging M=128: 23.8805ms p50 / 25.4883ms p95, native-fp32 parity green on the longer check |
| E4: GQA native path 20x Python fallback at 10k | passed on smoke shape: 27.81x; Qwen3.5-2B representative-key shape is 26.27x; real captured 256-node full K-bank is 5.69x; layers 3/7/11 full-bank 256-node parity is green |

## Remaining Work

- Treat the 1M dim128 host route as deep-interactive already; if E3 p95 is
  mandatory, replace the scalar q4 dot with a lower-level vectorized dot kernel
  or a larger routing layout change. The no-filter active fast path helps on
  same-corpus reruns but does not close the p95 gate.
- Extend exhaustive real-capture GQA parity beyond the current layer-3
  1,024-node, four-query batched-reference receipt and the 256-node layer 3/7/11
  sweep before claiming broader full-bank correctness across all capture layers.
- Implement a fuller GQA GEMM/segment-reduce layout if sub-10ms routing is
  required past larger full-bank real-capture points; the current segment
  reduce is a key-score-bank pass and improves 512-node full-bank routing, but
  it is not a BLAS/GEMM-backed row-block kernel. Simple stride representative-key
  compaction, grouped repeated-head qt4 scoring, hand-unrolled repeat-4
  head-ratio scoring, paired repeated-head scoring, and fused single-key segment
  defaulting were measured and rejected for runtime default use. The opt-in
  query-head row-block scorer improves the 512-node p50 but does not yet win the
  768/1,024-node full-bank curve, so the remaining kernel work should target a
  lower-level GEMM/BLAS-style layout rather than this temporary head-score table.
  The manual `GRM_ROUTER_GQA_QT4_UNROLL8` dot-kernel experiment was also
  parity-green but much slower, so further work should avoid scalar hand-unrolls.
  The opt-in transposed key-bank experiment was parity-green but slower on the
  2B full-bank capture shape, so it stays diagnostic-only; the next useful slice
  is still a lower-level GEMM/BLAS layout rather than a duplicate host layout
  walked by scalar C++ loops.
  A compile/runtime-gated CBLAS `sgemm` segment scorer now exists and passes
  parity, but generic system BLAS regressed the 512-node 2B full-bank receipt to
  `833.7949ms` p50, so it is rejected for default use. The seam is useful; the
  backend is not.
