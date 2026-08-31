# GRM Results Index

This page is the complete public index for the gated-result table, failures,
corrections, and measurement laws moved out of the top-level README on
2026-08-04. It changes their location, not their verdicts. A status of
**unconfirmed** means the former README supplied a claim whose exact value was
not found in a `docs/` receipt during this rework; the claim is retained here
rather than silently deleted or upgraded.

The primary early receipt record is
[GraftRepository Memory Architecture](GraftRepository_Memory_Architecture.md),
whose Foundations table records the original gates. The
[GRM Methodology](GRM_Methodology.md#8-experiment-ledger) is the readable
synthesis. Later programs use the linked append-only ledgers.

## Foundational gates

| Result formerly in README | Recorded result | Receipt |
|---|---|---|
| Graft ≡ in-context (lossless mounting) | Top-1 identical; maximum logit difference at the bf16 noise floor | [Architecture](GraftRepository_Memory_Architecture.md), [methodology](GRM_Methodology.md#8-experiment-ledger) |
| MLA latent graft size | 288 values per token per layer, about 22× smaller than full K/V | [Methodology](GRM_Methodology.md#3-what-a-graft-is), [injection record](KV-Graft_Document-Injection.md) |
| E1 router recall | Qwen3 routed top-3 10/10 vs mount-all 4/10; MiniCPM3 latent-centroid routing tied mount-all at 7/10 | [Architecture](GraftRepository_Memory_Architecture.md), [methodology](GRM_Methodology.md#8-experiment-ledger) |
| E2 digest fidelity, chained | D0 9/10 → D1 8/10 → D2 8/10; about 0.89 once, then a fixed point | [Methodology](GRM_Methodology.md#7-the-librarian-folding-digests-eras-descent) |
| E4 conversation memory | 6/6, equal to full-transcript baseline, at about 25% residency; amnesia 0/6 | [Architecture](GraftRepository_Memory_Architecture.md), [methodology](GRM_Methodology.md#8-experiment-ledger) |
| Persistent arena | 6/6 on one never-rebuilt cache through routed swaps and evictions | [Architecture](GraftRepository_Memory_Architecture.md), [methodology](GRM_Methodology.md#8-experiment-ledger) |
| Consolidation (E4-C) | 6/6 through QC'd digest grafts; routing pool 14→8 | [Architecture](GraftRepository_Memory_Architecture.md), [methodology](GRM_Methodology.md#8-experiment-ledger) |
| Shuttling | One-mount arena plus grounded trips reached 6/6, equal to a three-mount arena | [Architecture](GraftRepository_Memory_Architecture.md), [methodology](GRM_Methodology.md#8-experiment-ledger) |
| CORPUS-100 | 20/20 across 100 near-duplicate documents; 50 KB index; 1.3 s/probe | [Architecture](GraftRepository_Memory_Architecture.md), [methodology](GRM_Methodology.md#8-experiment-ledger) |
| Cross-session resume | Fresh-process recall later reached 7/7 from disk artifacts. The former README's exact `26.4MB` size has no matching `docs/` receipt; the earlier architecture row records 24.9 MB at the 6/7 stage. | [Architecture](GraftRepository_Memory_Architecture.md) |
| Long-history ephemeral boat | 42 turns at at-most 456 resident seats; 8/8 including era-folded facts and anaphora | [Methodology](GRM_Methodology.md#8-experiment-ledger) |
| Decode speed, fast stack | 675 → 21.6 ms/token, 31× | [MiniCPM3 results](MiniCPM3-MLA_Results.md#decode-speed-pass-2026-06-10-675--216-mstoken-31) |
| Deferred librarian | 42 turns; hot path at most 0.27 s and flat; recall 8/8 unchanged | [Architecture](GraftRepository_Memory_Architecture.md), [methodology](GRM_Methodology.md#8-experiment-ledger) |
| Fidelity-gated folding | A candidate retaining less than 70% of registered facts aborts; sources stay resident | [Architecture](GraftRepository_Memory_Architecture.md), [methodology](GRM_Methodology.md#7-the-librarian-folding-digests-eras-descent) |
| GQA arena, Qwen3-4B | MLA regression suite unchanged; GQA arena, trips, and E4-C each 6/6 | [Architecture](GraftRepository_Memory_Architecture.md) |
| VRAM paging | LRU write-back pager: 100 docs at 64 MB budget, 20/20, about +0.1 s/probe | [Architecture](GraftRepository_Memory_Architecture.md), [methodology](GRM_Methodology.md#8-experiment-ledger) |

## Later composed, routing, storage, and control receipts

| Result formerly in README | Recorded result | Receipt |
|---|---|---|
| GPT-OSS-20B composed E2E | 34-turn production `chat()` → `step()` session witnessed deposit → evict → route → mount → recall; fresh facts were 7/7 exact across restart and VRAM was 10.8→11.0 GB. Full probe score was 7/9: supersession-under-competition and the route-wall control remained RED, so this is not a certification claim. | [E2E ledger](GRM_E2E_RECEIPT_LEDGER.md), [synthesis](GPT_OSS_20B_APA_GRM_SYNTHESIS.md) |
| CUDA route, MLA | One million nodes in 2.22 ms, from 925.6 ms (417×); at most 100K byte-exact vs Python | [MLA CUDA ledger](GRM_MLA_CUDA_ROUTE_LEDGER.md), [router synthesis](GRM_GEMV_ROUTER_SYNTHESIS.md) |
| GQA CUDA bridge | Route entry 0.19–0.97 ms, 1.26–1.44× direct, after a 25–50× baseline overhead | [bridge ledger](GRM_CUDA_BRIDGE_OVERHEAD_LEDGER.md), [router synthesis](GRM_GEMV_ROUTER_SYNTHESIS.md) |
| Exact ragged GQA CUDA router | 175/175 semantic parity; 512 nodes p50 1.59 ms; disabled by default | [ragged ledger](GRM_GQA_EXACT_RAGGED_CUDA_LEDGER.md) |
| Graft storage quantization | INT8 was free at 1.88× disk compression; INT6 was the last green depth at 2.46×; packed on-disk format landed | [quantization ledger](GRM_GRAFT_QUANT_LEDGER.md) |
| Trinity NoPE grafts | Former README claim retained: recall at `live_shift=789`, twice GPT-OSS's word-salad depth, with carriage controls (`fdc478c`). **Planned / unconfirmed:** no matching `docs/` receipt was found. | unconfirmed |
| Supersession L2 | Stale answers 2/5 → 0/5; multi-hop mounts lineage head only; resolve-only diagnostic attributes 100% of the gain to L2. Default ON by operator decision 2026-08-30; `GRM_SUP_RESOLVE=0` restores legacy behavior. L1 length-debias remains default off. | [supersession ledger](GRM_SUPERSESSION_LEDGER.md) |
| S4 grounding-hit importance | Median Spearman 0.7556 and top-1 87.5% vs teacher-forced counterfactual arbiter, with no extra forward passes | [S4 ledger](GRM_S4_LEDGER.md) |
| M11 fold-after-recovery guard | Crash-recovered placeholder fold bug and guard are recorded. Former README's exact “11 regressions crash pre-fix” count has no matching `docs/` receipt. | [bug queue](GRM_BUG_QUEUE.md); exact count unconfirmed |

## Negative, corrected, open, and refuted receipts

| Finding formerly in README | Full disposition retained | Receipt |
|---|---|---|
| S4-aware paging loses to LRU | **FAIL / RED.** Recall tied 14/16, but S4 required 112 vs 68 page-ins at 4.595× overcommit. The importance signal remains valid; both S4 paging policies remain non-winning, default LRU. | [S4 ledger](GRM_S4_LEDGER.md) |
| S1 attention-mass result corrected | Original 0.47 RED used fixture grades contrary to the registered gate. Correct counterfactual-arbiter result is Spearman 0.8286 (published shorthand 0.83), top-1 9/18, PASS. The earlier “Attention-is-not-Explanation” interpretation is withdrawn; the human-label gap remains the finding. | [importance ledger](GRM_IMPORTANCE_LEDGER.md) |
| 4B self-report salience | **FAIL / RED.** Rankable on 7/18; standing-preference median 0.0 against a 2.0 bar under the frozen fact-worded rubric. | [importance ledger](GRM_IMPORTANCE_LEDGER.md) |
| Co-mount confusion | **OPEN.** Routing and mounting can select the right memory while readout returns a co-mounted sibling; fresh control 1/2. | [supersession ledger](GRM_SUPERSESSION_LEDGER.md) |
| L1 route length-debias on MLA | **UNDECIDABLE on MLA.** Baseline had zero inversions; G1 was vacuous and inert. Its decisive test was registered for GQA. | [supersession ledger](GRM_SUPERSESSION_LEDGER.md), [GQA re-gate plan](GRM_GQA_REGATES_L1_PLAN.md) |
| Route latency at session scale | **OPEN.** About 756 ms at 37 nodes on the GPT-OSS Python path; per-turn arena prep, lexical rescore, and ragged-bank non-engagement were named seams. | [E2E ledger](GRM_E2E_RECEIPT_LEDGER.md) |
| Lifecycle suite RED from 2026-07-08 | **Historical RED, later fixed.** FakeArena lacked `_bump_cuda_gqa_epoch`; 91/101 failed and the rule-2 gate had been silently dead. The current bug queue says all verified major items are fixed. | [bug queue](GRM_BUG_QUEUE.md), [importance ledger](GRM_IMPORTANCE_LEDGER.md) |
| `_ensure_h` silent fall-through | **OPEN in the cited queue entry.** An unbacked payload can fall through instead of raising a named error; the fold guard excludes the known recovery path. | [bug queue](GRM_BUG_QUEUE.md) |
| GQA re-gates | **OPEN / paused in the original receipt.** Repository resume 6/7 and descent 5/8 were measured before the early-stop fix; cross-model migration gate was written and never run; first-generation Qwen3 digests failed the 0.70 fidelity bar and correctly aborted. | [architecture](GraftRepository_Memory_Architecture.md), [re-gate plan](GRM_GQA_REGATES_L1_PLAN.md) |
| Era-depth at 4B | **REFUTED as readable era text.** Multi-digest era prose invents relations and list form strips them. Eras remain routing-only index nodes; descent expands children. | [methodology](GRM_Methodology.md#9-the-38--68--88-descent-story) |

## Measurement laws retained from the README

| Law | Recorded disposition | Receipt |
|---|---|---|
| First-run effect | First forward in a process differs by at most 0.5 logit from later runs; warm runs are bit-identical. Same-process A/B gates warm before side A. | [importance ledger](GRM_IMPORTANCE_LEDGER.md) |
| Seating-epoch invariant | Per-seat attention telemetry is valid only within a stable seating epoch; cache surgery discards the accumulator to under-attribute rather than misattribute. | [importance ledger](GRM_IMPORTANCE_LEDGER.md) |
| Teacher-forced cache comparisons | Cache-equivalence comparisons are teacher-forced because generation A/B ceases to be aligned after the first greedy divergence. | [methodology](GRM_Methodology.md) |
| K/V irreducibility | SCRIBE, sub-floor storage quantization, route-card, and BABEL experiments found compressed proxies of contextualized K/V lose the effect; exact payloads remain the standing engineering law. | [architecture](GraftRepository_Memory_Architecture.md), [translation finding](FINDING_translation_hits_scribe_dialect_lock.md), [quantization ledger](GRM_GRAFT_QUANT_LEDGER.md) |

## Reading status labels

The public status vocabulary is reproduced in the top-level
[certification matrix](../README.md#grm-certification-matrix). A model file or
cache-save gate establishes at most **GRM adapter** status. Only a receipt that
closes deposit, route, mount, recall, persistence/restart, and relevant controls
establishes **GRM certified** status.
