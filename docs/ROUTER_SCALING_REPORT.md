# GRM Router Scaling Report

Status: GEMV-router implementation checkpoint on branch
`grm-ram-tiered-runtime`.

This report records measured router scaling after replacing the original
per-node route scan with contiguous host arenas:

- MLA: fp32 SoA arena plus INT4 bulk/refine route book.
- GQA: contiguous key-bank arena with segment-style per-entry reduction.
- Concurrency: C ABI route/read calls now serialize `RouterIndex` access with
  a shared mutex while the final lock-free epoch design remains open.

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
| captured representative 1-token K | 32 | 0.0465 | 0.0473 | 0.6651 | true |
| captured representative 1-token K | 96 | 0.1172 | 0.1696 | 2.0503 | true |

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
- Extend real-capture GQA curves beyond 256 nodes with checkpointed/progress
  output so C ABI population overhead does not hide route timing.
- Implement the true GQA GEMM/segment-reduce path or a representative-key
  compaction policy for full 256-token captured K banks if sub-10ms routing is
  required past the 96-node real-capture point.
- Replace the current C ABI shared-mutex guard with the planned lock-free
  double-buffer epoch snapshot model if threaded serving requires no read-side
  lock.
- Update the AI research board / paper note with the final measured status.
