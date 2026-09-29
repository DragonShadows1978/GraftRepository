# GRM GQA Route-Card Database Postmortem

Date: 2026-07-11 (experiment receipts written 2026-07-12 UTC)

Status: stopped and stashed; do not merge.

## Verdict

The native route-card database implementation is technically sound, but its
search representation failed. Fixed `{2,4,8}` fp32 cards cannot preserve the
measured full-ragged GQA routing law, and validated generated-query K does not
recover the lost coverage under the registered half-source/half-alias policy.

This is not a prompt-quality near miss. The cards also fail as a coarse
shortlist unless they retain most of the repository, which removes their
computational purpose. The line should not be tuned or merged.

## What Was Tested

The implementation added a non-mountable card record to the existing native
`HostGraftStore`, with atomic immutable router publication, GQA checkpoint v11,
C/Python audit APIs, immediate provisional cards, deferred validated aliases,
cancellation, cache isolation, stale/supersession rules, and dense CPU/CUDA
routing.

The development experiment then reused exactly:

- 32 existing Qwen3.5-9B target capture shards;
- layer-3 native K/Q;
- 20 deterministic stored query locations;
- seed `20260711`;
- fp32 cards at `{2,4,8}` slots;
- selector `raw-maxnorm-diverse-v1`;
- prompt `gqa-route-card-v1`, greedy Qwen3.5-9B generation;
- strict exact-evidence, answer-containment, and identifier validation.

The hybrid run was preregistered before alias generation. No prompt, output
rail, selector, slot split, or query was changed after seeing results.

## Results

### Retrieval

| Slots | Full ragged recall@3 | Provisional recall@3 | Hybrid recall@3 |
| ---: | ---: | ---: | ---: |
| 2 | 18/20 | 2/20 | 3/20 |
| 4 | 18/20 | 3/20 | 2/20 |
| 8 | 18/20 | 4/20 | 3/20 |

No hybrid result matched the full bank's top three on any query. Four- and
eight-slot hybrid cards were worse than their provisional controls.

### Librarian yield

- 42 accepted aliases across 12/32 nodes;
- zero accepted unsupported identifiers;
- 20/32 nodes received no alias upgrade;
- 14 outputs exceeded the frozen 384-token rail with incomplete JSON;
- five returned an empty JSON array;
- one returned other malformed JSON;
- two candidates failed answer containment and one failed exact evidence.

### Coarse-shortlist diagnostic

The sealed ranks were read post hoc to test whether the cards could at least
generate candidates for exact ragged reranking:

| Slots | Hybrid recall@8 | Hybrid recall@16 | Hybrid recall@24 |
| ---: | ---: | ---: | ---: |
| 2 | 4/20 | 11/20 | 17/20 |
| 4 | 6/20 | 13/20 | 16/20 |
| 8 | 6/20 | 12/20 | 19/20 |

Keeping 24 of 32 nodes still loses sources. That is 75% retention for less
than full recall, so the card does not provide a useful pruning stage.

## Why It Failed

The following interpretation separates direct evidence from engineering
impressions.

### Directly supported

1. **The max-dot-product surface is not represented by a tiny geometric
   coreset.** Full routing takes a maximum over every source-token key. The
   highest-norm/diverse selector retains only a few witnesses, while many
   queries are won by omitted token-specific keys. Expanding from two to eight
   slots improved provisional recall only from 2/20 to 4/20.

2. **The fixed budget combines two competing jobs.** Hybrid cards spend half
   their already-small capacity on source coverage and half on aliases. With
   only one to four source slots per head, aliases can improve a few
   alias-ready sources while further reducing general source coverage.

3. **Alias competition creates false positives.** Seven of the 20 queried
   source nodes had accepted aliases. Within that subset, hybrid cards gained
   one or two top-three hits depending on size. Across the other 13 queries,
   alias rows from competing nodes displaced correct sources, erasing the
   local gains. The problem is global ranking competition, not merely missing
   aliases on the queried node.

4. **The cards are not a viable coarse stage.** Top-16 recall remained only
   11-13/20, and top-24 remained 16-19/20. A reranker cannot recover a source
   that the coarse card excluded.

5. **V1 librarian reliability is insufficient.** Only 12/32 nodes obtained
   aliases. This is a real operational defect, although it is not the primary
   explanation because the upgraded cards still failed the routing gate.

### My engineering impressions

1. **Norm plus pairwise diversity optimizes the wrong objective.** It spreads
   card vectors through K-space, but recall depends on the future query
   distribution and on which individual key wins a max operation. A
   query-aware set-cover or learned sketch would be a different research line,
   not a repair to this selector.

2. **Synthetic-question K is not a query prototype.** The router scores live
   query Q against stored K. Harvesting K from a generated question does not
   guarantee that another question's Q will align with it; Q and K projections
   are asymmetric. Document-expansion intuition transfers cleanly to lexical
   or shared-embedding retrieval, but only weakly to raw model-native Q-to-K
   geometry.

3. **The no-regression gate was necessary.** Natural future questions might
   benefit on a specially matched benchmark, but an index that gains those
   queries while losing existing source addressability is not safe for GRM.
   The hybrid cards failed precisely that requirement.

4. **More slots would change the economic claim.** The shortlist diagnostic
   suggests that substantially larger cards would be needed. At that point,
   padding or segmenting the original ragged keys is simpler and preserves the
   routing law exactly.

## What Did Not Fail

- fp32 checkpoint bytes and MLA v10 compatibility;
- native atomic snapshot publication and lifecycle filtering;
- CPU/CUDA stable top-k parity on the synthetic dense-card receipt;
- cancellation, cache isolation, alias validation, and audit redaction;
- experiment layout: parameterized tests proved the harness's 2/4/8 hybrid
  builder byte-identical to the repository implementation.

The failure belongs to the compressed search representation, not the native
database plumbing.

## Recommended Successor

Do not continue on this branch. Start a clean worktree from `main` for an exact
ragged GQA CUDA router. The most credible designs are:

- one concatenated K arena plus per-node offsets and a segmented exact maximum;
- length-bucketed dense banks with masked padding and exact per-node maxima;
- stable native top-k over those exact scores, retaining existing lexical and
  lifecycle filters.

That keeps the empirically successful raw `|q.k|` law and attacks only its
storage/layout cost. It should be governed by parity first, then latency and
memory; no generated aliases are required.

## Receipts

```text
artifacts/gqa_route_card/dev/qwen35_slot_sweep.json
artifacts/gqa_route_card/dev/cpu_cuda_parity.json
artifacts/gqa_route_card/dev/qwen35_hybrid_v1/protocol.json
  sha256 91a3c8f776e5e4accc7792513f7c4adb41de56c828464c8fac220860b49c04f6
artifacts/gqa_route_card/dev/qwen35_hybrid_v1/qualification.json
  sha256 47f59a2a2dfde42f6d6e81dcbecad05d7aac6f986903d08b0bcaff81b146b73b
artifacts/gqa_route_card/dev/qwen35_hybrid_v1/qwen35_hybrid_sweep.json
  sha256 5f138a81b1de0161731945acfd515d7075fdc74849ac00e426f227d037c03b95
```

Final verification after adding the experiment-equivalence tests:

- focused route-card module: 23 passed;
- existing router baseline: 21 passed;
- existing runtime lifecycle: 101 passed;
- existing native runtime: 121 passed;
- Release native build and TSAN publication stress: passed;
- no commits and no pushes.

Intended stash label:

```text
gqa-route-card-db stopped: implementation + negative Qwen hybrid receipts
```
