# GRM Router Scaling Report

Status: GEMV-router implementation checkpoint on branch
`grm-ram-tiered-runtime`.

This report records measured router scaling after replacing the original
per-node route scan with contiguous host arenas:

- MLA: fp32 SoA arena plus INT4 bulk/refine route book.
- GQA: contiguous key-bank arena with segment-style per-entry reduction.
- Concurrency: route-key mutations mark arenas dirty, the next route prepares
  once under the writer side of the C ABI shared mutex, and normal prepared
  route reads use shared locks while the final lock-free epoch design remains
  open.

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

Current measured MLA operating point: predecoded q4, `M=64`, 1M nodes,
26.0175ms p50 with bounded candidate staging. `M=16` is not acceptable for
the predecoded checkpoint because it produced one top-3 mismatch on the
harvested 1M parity run.

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

Conclusion: refine count is not the dominant runtime limiter, but too-small
M can still lose exact top-k after the predecoded q4 bulk approximation. Use
M=64 with bounded candidate staging for the current harvested-corpus operating
point. A rejected thread-local heap selection attempt stayed parity-green but
slowed the same point to 29.7308ms p50, so it was not kept.

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
| captured full 256-token K, pre-qt4 native-only | 1,024 | 76.5800 | 80.9969 | n/a | n/a |
| captured full 256-token K, pre-qt4 batched sampled parity | 1,024 | 73.5912 | 75.1033 | n/a | true (2 queries) |
| captured full 256-token K, qt4 batched sampled parity | 1,024 | 36.7363 | 38.6939 | n/a | true (2 queries) |
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

## Expectations

| expectation | result |
| --- | --- |
| E1: fp32 GEMV 10x current native scan at 100k | not proven; no direct P0 100k run |
| E2: INT4 two-tier 2x over fp32 at 100k | passed on measured points: 22.8177ms -> 6.2995ms |
| E3: 1M route <= 25ms host-side | narrowly missed; best exact measured point is 26.0175ms |
| E4: GQA native path 20x Python fallback at 10k | passed on smoke shape: 27.81x; Qwen3.5-2B representative-key shape is 26.27x; real captured 256-node full K-bank is 5.69x |

## Remaining Work

- Treat the 1M dim128 host route as deep-interactive already; if E3 remains
  mandatory, replace the scalar q4 dot with a lower-level vectorized dot kernel
  or a larger routing layout change.
- Extend exhaustive real-capture GQA parity beyond the current 512-node,
  four-query batched-reference receipt before claiming broader full-bank
  correctness.
- Implement a fuller GQA GEMM/segment-reduce layout if sub-10ms routing is
  required past the 96-node real-capture point; simple stride representative-key
  compaction, grouped repeated-head qt4 scoring, and hand-unrolled repeat-4
  head-ratio scoring, and paired repeated-head scoring were measured and
  rejected.
- Replace the current C ABI prepare-on-first-route shared-mutex bridge with the
  planned lock-free double-buffer epoch snapshot model if threaded serving
  requires no read-side lock.
- Update the AI research board / paper note with the final measured status.
