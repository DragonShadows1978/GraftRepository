# GRM-C7 amendment 7 — diagnosis complete; core repair STOP; prompt contrast NOT_RUN

**Core glyph-normalization mismatch explains the returned fresh refusals and the folded identifier failures.** Accepted r3 folds retire the ASCII raw sources. Their generated digests preserve identifiers using U+2011 NON-BREAKING HYPHEN. Routing/coverage normalize that glyph, while A-DEC's own-text binding does not. Thus a digest can rank first, contain the requested identifier, and still have zero identifier hits. No core repair is claimed or implemented. The 40-row middle-prompt reader contrast is registered and CPU-checked; no GPU work ran in this seat.

Evidence classes: lead E2E receipts for historical answers, mounts, rankings and fold events; CPU predicate evaluation for the normalization mismatch; source reasoning for causality and proposed repair; NOT_RUN for the prompt treatment and any repaired-core model result. Seat: **gpt-6-astra, high**, live identity receipt `logs/grm_c7_a7.log:6,10`.

## Per-probe comparison and mechanisms

[PER_PROBE.md](PER_PROBE.md) contains **all 30 fresh/folded rows**, including the 20 answerable probes and ten controls: r2/r3 mounted IDs, identified candidates, full recorded ranking window, source retired flags at that probe, mechanism, and exact receipt paths/line numbers. [per_probe.json](per_probe.json) additionally carries source text, source-turn joins, retirement-event receipts, policy branches and frozen scores. Node IDs are joined through each revision's `turn_records`; the turn-2 correction adds a node in r3, so r2/r3 numeric IDs are not equated.

The retirement column is an event-based reconstruction, not a mislabeled final snapshot: an accepted fold strictly before the probe retires that source; each selected source's cell-end retired flag is checked against the event history through cell end. These selected raw sources have no `superseded_by` edge. Rankings/mounts/identified candidates come directly from the probe receipt and are never reconstructed from later metadata. The r2 comparison uses original cells through turn 23 and FIX4 continuation cells from turn 24 onward; historical quarantines remain in force for old oracles and alias probes.

- **Fresh-0/-1, d005 (turns 11/12):** r2 nodes 5/6 and r3 nodes 6/7 are active ASCII records. Both r3 rows bind and mount their raw source, yielding 2/2 exact. In r2, fresh-0 is exact and fresh-1 abstains.
- **Fresh-0/-1, d030/d060/d120/d250:** accepted fold at turn **18**, sources `[6,7,8,9]`, produces digest **16** and retires r3 raw sources 6/7. Digest 16 has the requested values, 73 tokens, normalized routing rare keys, but no raw admission binding. It ranks first on every one of these eight rows; all eight return `Not in memory: no stored record matches c7-fresh-0.` or `...c7-fresh-1.` with zero mounts. R2 has no accepted digest; nodes 5/6 stay active and mounted at every distance. R2 fresh-0 is exact 5/5 and fresh-1 abstains 5/5: **fresh 5/10**, not ten mounted-but-unread failures. The earlier 30-row failure census was a selected failure cohort.
- **Folded-0, d005 (turn 25):** turn-9 fold retired outbound raw node 3 into digest **10**. Ranking starts `[10,16,12,13,11,15]`, identified `[]`. Digest 10 does not bind. Inbound node **19** is in excluded recency; FIX4 finds its ASCII identifier and serves `[19]`. It returns `Cedar-1011 | Cedar-1011`, with no outbound record mounted. This is a successful FIX4 recency rescue of incomplete information, not a FIX4 regression.
- **Folded-1, d005 (turn 31):** outbound raw node 4 is retired into digest 10; ASCII inbound node **20** binds. Ranking `[10,20,21,19,22,16]`, identified `[20]`; A-DEC initially plans `[10,20,21]`. Digest 10 is **114 tokens** against width 96. The receipt records split children `[26,27]`, descended head `[26]`, `fit_dropped_filler=[10]`, final plan/mount `[20,21]`. Child 26 contains the outbound identifiers but fails the same own-text predicate; because node 20 binds, the split is dropped. Output: `Cedar-1012 | Cedar-1012`.
- **Folded-0/-1, d030 (turns 50/56):** only inbound nodes 19/20 bind and mount; outputs again duplicate the inbound value. Outbound digest 10 and split 26 carry Unicode identifiers and fail binding.
- **Folded-0/-1, d060/d120/d250:** turn-**60** fold retires inbound nodes 19/20 into digest **48**. Neither outbound nor inbound derived text binds. At turn 100, accepted era **89** retires 48 and contains inbound tags with the same Unicode glyphs. All six rows refuse before generation. The later era can outrank digest 10 without repairing binding.
- **Fresh/folded controls:** their base entity can bind even though its inspection password was never stored. The table includes these ten controls. Fresh control d005 and folded controls d005/d030 generate unsupported answers; later rows refuse when their base identifiers stop binding. An entity hit is not evidence for every requested attribute.

The existing broad r3 results were recomputed from the receipts in [cpu_supplement.json](cpu_supplement.json): answerable fresh 2 exact / 0 wrong / 8 abstentions; alias 0/6/4; correction 9/1/0; folded 0/4/6. Unsupported controls remain fresh 1/5, alias 5/5, correction 2/5, folded 2/5: **10/20**. All 20 historical control answers fail the frozen full-string exact UNKNOWN comparison, including ten deterministic refusal strings; unsupported-answer error and exact-control accuracy are distinct metrics.

## Digest text and token evidence

[DIGESTS.md](DIGESTS.md) quotes the complete stored text of nodes **10,16,26,48,89**, its checkpoint SHA, stored routing keys, recomputed own-text rare keys, and the actual admission tokenizer output. [identifier_receipt.json](identifier_receipt.json) is the machine-readable CPU receipt. [fold_receipts.json](fold_receipts.json) retains the exact generated digest texts, original source lists, acceptance and coverage receipts.

For example, digest 16 contains this verbatim text (the apparent dashes inside identifiers are U+2011):

> ARCHIVE NOTE. For the archive: the  current C7‑Fresh‑0 value is Basalt‑811, the current C7‑Fresh‑1 value is Basalt‑812, the current C7‑Fresh‑2 value is Basalt‑813, and C7‑Signal‑0 is an alias for C7‑AliasBase‑0.

Digest 10 contains, verbatim:

> The current C7‑Archive‑0 outbound tag is Reed‑611 and the current C7‑Archive‑1 outbound tag is Reed‑612.

Question identifier tokens are `['c7-fresh-0']` or `['c7-archive-0']`. Own-text `normalized_words` fragments `C7‑Archive‑0` into `['c7','archive','0']`. The stored rare keys and `_rare_tokens(own_text)` both contain the whole `c7-archive-0` token. **The stored digest routing identifier set is a union of inherited source keys AND normalized own-text keys**, not exclusively inherited or exclusively FIX5 output. **The admission identifier set is freshly scanned from the candidate's own text**, ignoring that union. Both lower/casefold these ASCII letters consistently; the divergent operation is dash projection, not case.

CPU evaluation of the unchanged predicate: all five texts fail raw binding and all five bind when the existing `normalize_glyphs` is applied only to the input. Uppercase normalized input still binds; wrong numeric identifiers remain rejected. This establishes a predicate-level mechanism, not a repaired-core end-to-end result.

## Harness/core ruling — STOP before repair

- `core/graft_arena.py:2135` `_deposit_consolidation` deposits generated text, unions source rare keys with its own normalized rare keys at 2146–2155, and retires sources at **2156–2157**. Eligibility excludes retired nodes at **1447–1449**. FIX5 accepted more folds, exposing an existing binding inconsistency; fold coverage 1.0 can coexist with admission failure.
- `core/graft_arena.py:2024–2060` routes `_rare_tokens` through the LSR glyph normalizer; `core/grm_text_norm.py:26–28,36–65` specifies U+2010/U+2011 → ASCII hyphen. `core/grm_admission.py:73–79` tokenizes without that projection; **98–109** judges candidate own text against the question's whole rare tokens; **404–412** performs that scan over eligible candidates. This is a **core** normalization defect.
- `core/grm_admission.py:690–705` implements the FIX4 live-binder rescue and then `identifier_unbound_abstention` at **708–741**. `scripts/grm_e2e_session.py:1235–1278` returns the deterministic refusal before any `_attempt`. Prefixing a prompt cannot fix that outcome under fixed admission.
- `core/graft_arena.py:3356–3365` reuses the predicate for split children. `scripts/grm_e2e_session.py:1374–1393` drops nonbinding split chunks when another plan member binds. This explains the turn-31 fit receipt; simply widening the rank window would not repair it.
- `scripts/grm_c7_register_r3.py:40–50` changes only turn-2 correction lineage. It does not supersede the fresh or archive raw nodes. `scripts/grm_c7_run.py:128–148` removes the prompt clause. Those changes are **harness** changes. The observed fresh retirement is explicitly caused by fold 18, not a turn-2 supersession edge. Removing the clause affects generated controls/readout, not this ASCII question's rare-token identity. No isolated counterfactual of fold scheduling versus the turn-2 extra node was run; a secondary scheduling effect remains unmeasured.

**Registered successor recommendation, not implemented:** a minimal flag-gated core repair should align own-text identifier tokenization with the existing SC1.1 glyph projection for admission, recency binding and child binding, keeping own-text evidence rather than treating inherited keys as proof that a digest contains a fact. Prior art: GRM SC1.1/DET1.4 and RT1 (GRM contributors, 2026), locally verified; this applies their existing rules at the missed boundary. Validate wrong-digit negatives, glyph/case pairs, source-retirement probes, split children, and existing flag-OFF behavior before any model claim. Core edits require the lead's next order. **Not claimed fixed.**

## Middle-prompt contrast

Registration: `artifacts/grm_c7/r3/amendment_7/registration.json`

SHA-256: `f4f1aa27f4db79e1fd95c995d4e72e8b70ea0f29140effaa56462bcc31a294f1`

Plain example:

```text
What is the current C7-Fresh-0 value? Reply only with the answer.
```

Middle example:

```text
Answer using only the mounted memory records. If those records do not contain the requested information, reply unknown. What is the current C7-Fresh-0 value? Reply only with the answer.
```

Both use the unmodified production `harmony_turn`, sink/system (`Reasoning: low. Valid channel: final.`), stop strings and 32-token answer budget. The added instruction uses lowercase `unknown`; CPU gates confirm it adds no rare identifier token. All 40 exact requests and checkpoint/payload hashes are in [requests.json](requests.json). The frozen `scripts/grm_c7_common.py:181` scorer is imported unchanged and source-pinned.

**Scope:** fixed recorded mounts from each probe's **end-of-cell** r3 checkpoint, clean ephemeral cache per side, production `_attempt`, no live source-text insertion or rerouting. Sixteen rows have recorded mounts: at most **32 paired model calls**. Twenty-four rows were admission refusals: they retain the exact policy output with zero forwards, visibly classified as carry-forwards rather than generation. The cohort includes all 20 controls, 10 answerable fresh and 10 answerable folded rows. This is a reader contrast conditional on r3 residency, not a replay of all prior turns or an evaluation of the new prompt's routing effects.

Every generated A0 answer must byte-match the historical r3 text before its paired A1 runs. A mismatch is written to disk, stops the entire campaign, and leaves quality NOT_MEASURED. Pair interleaving uses identical loaded payloads; no expected value reaches the model. All 40 pairs, all six controllers and baseline parity are required for a cohort verdict. Earlier pairs may have run before a later mismatch; they cannot support a campaign verdict in that case.

Registered prediction: unsupported controls **≤5/20** (r3 10/20); fresh exact **≥2/10** with eight unchanged admission refusals; folded exact **0/10** with six unchanged admission refusals. Prompt success requires complete parity-valid data, controls ≤5 unsupported, and retention of at least two fresh exact answers. Report all other errors and folded residuals independently; this gate is not product acceptance.

Budget: six foreground batches × 280-second pessimistic reservations = **1680 seconds / 0.4667 GPU-h**, below 1800 seconds / 0.5 GPU-h. Each lease is 280 seconds (≤285); 30-second foreground cooldown is outside the lease. Busy lease, insufficient space, failed/orphaned reservation, timeout, source/hash drift, or baseline mismatch stops without retries or cohort trimming. Allocation/decoding throughput is unmeasured for this replay: fitting the reservation cap is not proof it will finish. Python's existing lease alarm cannot hard-preempt an uninterruptible native call; an overrun is RED and charged in full, never hidden by clipping. There is no kill-based outer watchdog; the registered 590-second outer envelope is not a hard OS bound. This limitation is retained openly from the RD1 in-process lease pattern.

Exact lead command (GPU execution belongs to the lead):

```bash
bash /mnt/ForgeRealm/wt/grm-c7/artifacts/grm_c7/r3/amendment_7/lead_commands.txt
```

CPU-only command, already executed:

```bash
CUDA_VISIBLE_DEVICES='' bash /mnt/ForgeRealm/wt/grm-c7/artifacts/grm_c7/r3/amendment_7/lead_commands.txt --check-only
```

## Validation, RED and deviations

CPU baseline: **5 tests passed**, `cpu_gates.log`. These cover all cohort IDs/counts, prompt identifier parity, payload/metadata/registration SHA, frozen scorer hand cases, the actual Unicode predicate failure and normalizer counterfactual, fake numerical-boundary cache/mount isolation, raw A0 mismatch receipt-before-stop, and missing-result refusal to report PASS. Six additional launch checks pass in `cpu_supplement.json`: existing owner rejected/preserved, existing batch rejected, own owner cleaned, orphan reservation stop, failed campaign stop. CPU doubles establish control flow only, not decoder fidelity; blind validation remains the lead's responsibility.

Deviations/limits: fixed-residency **cell-end** payload replay is explicitly narrower than exact pre-probe state restoration. Cell-end metadata and omitted intermediate ladder attempts are not claimed historically identical. Two selected rows were FIX4 recency reads rather than ordinary clean-room reads; the strict A0 byte-parity rail tests whether the proposed reconstruction is valid. No full routing/fold replay fits this chosen reader-only design. The 20-GiB free-space check is slightly stricter than the earlier r3 20-decimal-GB preflight. The cooperative outer timeout limitation is stated above. None of the original orders, registrations, scorer, core or existing harness files were edited.

RED: normalization repair pending; prompt model comparison NOT_RUN; fresh/folded admission failures remain; alias residual unchanged; hypothetical normalization input success is not an E2E fix; prompt control-success predictions are untested. No GPU result or numerical baseline parity is claimed from these CPU gates.

Process safety: no GPU/model load, git, subagents, background processes/waits, service operations, signals or process termination in this seat. Only foreground CPU/read-only checks and additive files in the assigned worktree. Temporary guard-test owner files were created and removed solely by that CPU test. Core/source integrity is compared with the pinned r3 source manifest and sealed in `final_integrity.json`.

## Prior art

Locally verified systems: **GRM contributors (2026), SC1.1/DET1.4** supplied the exact glyph projection; **A-DEC/RT1 and LSR-P2C** supplied own-text binding and split-child semantics; **C7/FIX4/C2/CMC1** supplied receipt joins, checkpoint payloads, frozen scoring, environment and foreground leases; **RD1 amendment-1**, `/mnt/ForgeRealm/wt/grm-rd1/scripts/grm_rd1_replay.py`, supplied fixed-residency prompt contrast and baseline-byte-parity stopping. Reused unchanged where applicable. Our contribution is this r2/r3 diagnostic join, selected middle-prompt wording/cohort, and additive replay orchestration. **No prior art known to me for this exact composition.** No external-literature or algorithmic novelty claim. Prior-art annotations appear at additive code sites, in the ledger and here.
