# GRM Router Baseline

Status: P0 harness committed; smoke baseline and 1k/10k host baseline recorded.
Full 100k/1M native curves remain runnable, but the current Python fallback is
already ~248 ms at 10k nodes, so the full Python 1M pass should be treated as a
long-run measurement instead of a normal edit gate.

The harness measures two current pre-GEMV paths:

- native `RouterIndex::route`, the C++ per-node route scan.
- Python fallback scan with the same cosine plus fractional lexical bonus law.

Centroids are not gaussian placeholders. The harness feature-hashes real
source/doc lines from this repository into normalized route vectors, then
repeats and permutes those harvested rows deterministically for larger node
counts.

## Smoke Run

Command:

```bash
python3 scripts/grm_router_baseline.py --smoke \
  --out /tmp/grm_router_baseline_smoke.json \
  --markdown-out /tmp/grm_router_baseline_smoke.md
```

The smoke run is the CI/sanity target. It proves the benchmark path builds the
native runtime, loads routes, checks native-vs-Python parity, and writes both
JSON and Markdown output.

## Measured P0 Curve

Command:

```bash
python3 scripts/grm_router_baseline.py \
  --node-counts 1000 10000 --queries 8 --warmup 2 --dim 128 \
  --max-vectors 2048 --max-files 256 \
  --out /tmp/grm_router_baseline_p0.json \
  --markdown-out /tmp/grm_router_baseline_p0.md
```

Harvest:

- source_root: `/mnt/ForgeRealm/GraftRepository`
- source_files: 8
- centroid_bank_rows: 2048
- dim: 128
- mean_tokens_per_row: 10.54
- mean_active_dims: 9.63

Results:

| nodes | dim | parity | native p50 ms | native p95 ms | python p50 ms | python p95 ms | native build ms |
| ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 128 | True | 0.4985 | 0.5405 | 25.2087 | 27.6304 | 13.71 |
| 10000 | 128 | True | 4.0865 | 4.7191 | 248.3771 | 249.5266 | 221.08 |

## Full Curve Command

Run this before judging P2/P3 speedups:

```bash
python3 scripts/grm_router_baseline.py \
  --node-counts 1000 10000 100000 1000000 \
  --queries 24 --warmup 4 --dim 512 \
  --out /tmp/grm_router_baseline_full.json \
  --markdown-out docs/ROUTER_BASELINE.md
```

If the 1M point exceeds available host RAM with the current vector-backed
native store, record that as the P0 result. That failure is itself part of the
case for the SoA arena in P2.
