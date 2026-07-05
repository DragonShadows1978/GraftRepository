# Qwen3.5 Translation Reliability Plan

**Status:** R3 full V2 binding gate complete. R4 translator reliability tuning
is next.

**House rules for this track:**

- Register protocol changes before running new gates.
- Keep a completion ledger from day one:
  `docs/QWEN35_TRANSLATION_RELIABILITY_LEDGER.md`.
- Update the ledger after every completed implementation, eval, artifact, or
  analysis step.
- Record commands, artifact paths, and hashes for every result-bearing run.
- Commit per completed phase or stable implementation checkpoint.
- Do not treat partial or toy gates as full evidence.

## Starting Point

The original PoC completed and survived as **qualified binding transfer**:

- G1 passed: average key recall@16 `0.637847516976028`, minimum
  key/shuffled ratio `10.688533519012191`.
- G2 passed: value-output cosine range `0.9094160406409361` to
  `0.9932101022103578`, translated/wrong-layer MSE ratio max
  `0.1447668773166899`.
- G3 translated passed: `25 / 32` positive margins.
- Source-native and target-native ceilings were both `32 / 32`.

The weaknesses to fix are:

- Live G0 logit identity failed the strict max-delta threshold:
  `0.1875` vs `0.002`, though top-1 flip rate stayed `0.0`.
- The G3 amnesia floor was high at `20 / 32`, so the probe set is not a clean
  no-memory floor.
- Translated mode missed `7 / 32` probes and had a low mean margin
  (`1.3904626497223576`) compared with native ceilings.

## Objective

Make Qwen3.5 2B-to-9B graft translation more reliable and more defensible by:

1. tightening the binding probe floor,
2. diagnosing translated misses,
3. rerunning binding under a harder registered probe set,
4. improving translator training only after the harness is clean enough to
   distinguish translator failure from test leakage.

## Phase R0: Baseline Miss And Floor Analysis

Analyze the existing `binding_eval_metrics.json` without changing the model or
translator.

Required outputs:

- Per-probe table for all modes with:
  - probe id
  - gold candidate
  - best decoy
  - gold-minus-best-decoy margin
  - success/failure
  - token lengths for fact, query, gold, and decoys
  - whether amnesia and translated agree
- Summary of:
  - translated misses
  - amnesia successes
  - probes where translated beats amnesia
  - probes where amnesia beats translated

Exit gate:

- Analysis artifact exists under
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/`.
- Ledger records command, hash, and the actionable failure pattern.

Current status:

- Complete.
- Artifact:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_eval_analysis_v1.json`
- Translated misses: `7 / 32`
- Amnesia successes: `20 / 32`
- Translated beats amnesia by margin on `28 / 32` probes.
- Amnesia beats translated by margin on `4 / 32` probes.
- Actionable pattern: V1 is not a clean floor, but translated usually improves
  margin over amnesia. R1 should reduce prompt/entity leakage and make decoys
  more tightly matched.

## Phase R1: Binding Probe V2 Generator

Add a stricter binding probe generator that reduces prior/prompt leakage.

Design requirements:

- Opaque facts only: invented names and random code-like values.
- Matched decoys:
  - same token-count bucket as gold when possible
  - same surface class as gold
  - no semantic hints
- Multiple query templates per binding.
- Deterministic seed and manifest metadata.
- Schema version distinct from V1, likely
  `qwen35_graft_translation_binding_probes_v2`.
- Keep V1 generator/evaluator compatibility intact.

Initial V2 probe shape:

- `32` bindings.
- `2` query templates per binding.
- Evaluation can either flatten to `64` probe rows or keep grouped rows with
  per-binding aggregation. The first implementation should flatten to preserve
  the existing evaluator path.

Exit gate:

- Unit tests prove deterministic output, schema, decoy count, and no duplicate
  gold/decoy values.
- Probe artifact is written under
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_probes_v2.json`.

Current status:

- Complete.
- Added schema `qwen35_graft_translation_binding_probes_v2`.
- V2 emits `32` opaque bindings, flattened to `64` probe rows with `2` query
  templates per binding.
- Candidate values are random `3x3` code strings; every gold and decoy code is
  unique across bindings.
- Artifact:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_probes_v2.json`
- sha256:
  `4cc97ff2d230692e8c0b21bc265bd63132605d16147c5b5623c4ba224728f4a5`

## Phase R2: Amnesia Floor Gate

Run V2 in amnesia mode before interpreting translated transfer.

Frozen floor thresholds:

- `<= 12 / 32` positive margins for one-query-per-binding mode, or
- `<= 24 / 64` positive margins for flattened two-query mode.

If amnesia remains above the floor:

- Do not tune translator yet.
- Revise probe generator with harder values/decoys and record the failure.

Exit gate:

- Amnesia floor artifact exists.
- Ledger states whether V2 is clean enough for translated evaluation.

Current status:

- Complete.
- Artifact:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_eval_v2_amnesia.json`
- sha256:
  `c96291f61a0001680f2c65fce9d0703e020187cf907da73b015c4278dc838234`
- Result: amnesia positive margins `9 / 64`, mean margin
  `-2.255242589061309`, min margin `-6.48096410594205`.
- Decision: V2 passes the frozen flattened floor threshold of `<= 24 / 64`,
  so R3 can run translated/source/target binding evaluation against the frozen
  V2 set.

## Phase R3: V2 Full Binding Gate

Run V2 across:

- `amnesia`
- `source-native`
- `target-native`
- `translated`

Frozen translated pass thresholds:

- Source-native and target-native should stay at or near ceiling:
  `>= 28 / 32` or `>= 56 / 64`.
- Translated must beat amnesia by at least `+8` positive margins and pass the
  original binomial-style threshold:
  - `>= 14 / 32`, or
  - `>= 28 / 64` for flattened two-query mode.
- Report mean margin and min margin for every mode.

Exit gate:

- V2 full binding artifact exists.
- Claim level is updated:
  - no clean transfer,
  - geometry only,
  - binding transfer signal,
  - stronger binding transfer.

Current status:

- Complete.
- Artifact:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_eval_v2_full.json`
- sha256:
  `e8157316292e88ab872144354e5f54ee68b409c81d7334f4a190b13d1b9a2df7`
- Analysis artifact:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_eval_v2_full_analysis.json`
- analysis sha256:
  `123543a4377c3c8068b17215b47f97707904b074bcc4634549777c0990f3ba23`
- Mode summary:
  - amnesia: `9 / 64`, mean margin `-2.255242589061309`, min margin
    `-6.48096410594205`
  - source-native: `64 / 64`, mean margin `38.7234434921245`, min margin
    `31.493913498901925`
  - target-native: `64 / 64`, mean margin `34.43096542845171`, min margin
    `29.44889459450494`
  - translated: `40 / 64`, mean margin `0.41503181978418846`, min margin
    `-5.668900343599823`
- Decision: V2 establishes a stronger binding transfer signal. Translated
  exceeds the frozen `>= 28 / 64` pass threshold and beats amnesia by `+31`
  positive margins, but the `24 / 64` translated misses make R4 tuning
  worthwhile before stronger reliability claims.

## Phase R4: Translator Reliability Tuning

Only start after R2/R3 prove the probe floor is clean enough.

Candidate interventions, in order:

1. Layer selection policy:
   - test all layers,
   - skip weak early layer,
   - weight layers by held-out G1/G2 quality.
2. Ridge sweep using train-split diagnostics only:
   - `1e-5`, `3e-5`, `1e-4`, `3e-4`, `1e-3`.
3. More paired corpus:
   - `5M` tokens,
   - `10M` tokens.
4. Residual translator:
   - ridge baseline plus small low-rank residual,
   - do not approach "rerun the model" capacity.
5. Objective upgrade:
   - K score/top-k preservation loss,
   - V attention-output reconstruction loss.

Exit gate:

- Every intervention reports G1/G2 and V2 G3 against the same frozen V2 probe
  set.
- The winner must improve translated positive margins and/or mean margin
  without worsening G1/G2.

Current R4.1 protocol:

- Scope: ridge-only sweep using the existing paired capture corpus.
- No probe changes, no corpus recapture, no architecture changes.
- Frozen V2 probe set:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_probes_v2.json`
- Baseline translator:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator`
  - ridge lambda: `1e-4`
  - V2 translated result: `40 / 64`, mean margin `0.41503181978418846`
- New ridge candidates:
  - `1e-5`
  - `3e-5`
  - `3e-4`
  - `1e-3`
- Output directories:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_ridge_<lambda>`
- Required per candidate:
  - `fit-translator` on train split,
  - `eval-translator` on held-out split with `topk=16`,
  - `eval-binding-probes` on frozen V2 with `modes=translated`.
- Selection rule:
  - prefer higher translated positive margins,
  - break ties by translated mean margin,
  - reject candidates that materially worsen held-out G1/G2 against baseline.

Current R4.1 status:

- Complete.
- Added shared ridge fitting and shared held-out evaluation commands so the
  capture corpus is scanned once per sweep rather than once per candidate.
- CUDA/CuPy did accelerate the ridge math path, but the current compressed
  `.npz` capture format is the practical limiter. The GPU remained lightly
  used during fit accumulation because Python/zip decompression starved it.
- Full held-out G1/G2 for all four lambdas projected to roughly `100+`
  minutes, so the recorded held-out geometry result is an explicitly bounded
  `128`-pair diagnostic, not a full held-out gate.
- Bounded held-out diagnostic:
  - output name: `eval_metrics_heldout_128.json`
  - progress artifact:
    `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/ridge_eval_sweep_heldout_128_progress.json`
  - all four lambdas were effectively tied:
    mean key recall@16 about `0.6367017`, translated-output cosine about
    `0.90448316`, translated-output MSE about `0.17619949`.
- Frozen V2 translated binding results:
  - baseline `1e-4`: `40 / 64`, mean `0.41503181978418846`, min
    `-5.668900343599823`
  - `1e-5`: `38 / 64`, mean `0.4035796222422129`, min
    `-5.658042730095097`
  - `3e-5`: `36 / 64`, mean `0.4045612544007525`, min
    `-5.607635710753115`
  - `3e-4`: `39 / 64`, mean `0.42209077022321806`, min
    `-5.747188284219789`
  - `1e-3`: `38 / 64`, mean `0.4057173672102292`, min
    `-5.668216921540541`
- Decision: ridge lambda tuning does not improve reliability on this corpus.
  No new candidate recovered a baseline miss; candidates only lost baseline
  successes. Keep the original `1e-4` translator as the R4.1 winner.
- Next tuning move should not be scalar ridge lambda. Prefer layer policy,
  more paired corpus, or a translator objective that directly preserves K
  score/top-k and V attention-output behavior.

Current R4.2 protocol:

- Scope: layer-policy sweep using the existing `1e-4` baseline translator.
- No probe changes, no corpus recapture, no ridge refit, no objective changes.
- Frozen V2 probe set:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_probes_v2.json`
- Baseline translator:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator`
  - V2 translated result: `40 / 64`, mean margin `0.41503181978418846`
- Candidate layer policies:
  - `drop_l3`: remove source `3` to target `3`
  - `drop_l11_l15`: remove source `11` to target `15`
  - `strong4`: keep `7->7`, `15->19`, `19->27`, `23->31`
  - `late3`: keep `15->19`, `19->27`, `23->31`
- Candidate output directories:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_layer_<policy>`
- Required per candidate:
  - write a filtered translator manifest from the frozen baseline translator,
  - run `eval-binding-probes` on frozen V2 with `modes=translated`,
  - compare gained/lost probes against baseline translated rows.
- Selection rule:
  - prefer higher translated positive margins,
  - break ties by translated mean margin,
  - reject candidates that only trade away baseline successes without
    recovering baseline misses.

Current R4.2 status:

- Complete.
- Added reproducible filtered-translator manifest tooling:
  `filter-translator-layers`.
- Frozen V2 translated binding results:
  - baseline `1e-4`: `40 / 64`, mean `0.41503181978418846`, min
    `-5.668900343599823`
  - `drop_l3`: `38 / 64`, mean `0.35193815798529215`, min
    `-5.564677949742489`
  - `drop_l11_l15`: `38 / 64`, mean `0.4061522761956001`, min
    `-5.4681166409774065`
  - `strong4`: `36 / 64`, mean `0.32240649378745834`, min
    `-5.3586791078400395`
  - `late3`: `36 / 64`, mean `0.3056605327202693`, min
    `-5.266425911516741`
- Decision: simple layer filtering does not improve aggregate reliability.
  `drop_l3`, `strong4`, and `late3` recovered a small number of baseline
  misses, but all policies lost more baseline successes than they gained.
  Keep the original all-layer `1e-4` translator as the R4.2 winner.
- Next tuning move should be either more paired corpus or an objective-level
  translator upgrade. Layer omission by itself is too blunt.

Current R4.3 protocol:

- Scope: expand paired capture corpus from the completed `2.5M` token baseline
  to a `5M` token train/heldout cap, then refit the same ridge translator.
- No probe changes and no translator architecture changes.
- Keep the `2.5M` baseline artifacts intact; write separate R4.3 artifacts:
  - corpus plan:
    `/mnt/ForgeRealm/qwen35_graft_translation_poc/corpus_plan_5m.json`
  - captures:
    `/mnt/ForgeRealm/qwen35_graft_translation_poc/captures_5m`
  - translator:
    `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_corpus_5m`
  - gates:
    `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_eval_v2_translated_corpus_5m.json`
- Corpus sources remain the same as the baseline:
  - `/mnt/ForgeRealm/scribe_mint_v1/manifest.jsonl`
  - `/mnt/ForgeRealm/HumanBaselineCorpus/stories`
- Actual generated plan total from those sources is `3,408,241` tokens, not
  `5M`, because the source set is exhausted under the current filters.
- Reuse existing overlapping captures only if shard identity is exact under
  the new plan. Do not copy or relabel mismatched shards.
- Overlap seed status:
  - exact reusable old chunks: `9860`
  - mismatched overlap chunks: `1`
  - new-only chunks: `3959`
  - hardlinked files: `39440`
  - remaining capture work after seed: `3960` source chunks and `3960` target
    chunks.
- Exit gate:
  - source and target captures complete for the `5M` plan,
  - `translator_corpus_5m` fit on train split,
  - held-out G1/G2 recorded,
  - frozen V2 translated binding recorded.
- Selection rule:
  - prefer higher translated positive margins over the `2.5M` baseline
    `40 / 64`,
  - break ties by translated mean margin,
  - reject if held-out G1/G2 materially regress.

## Phase R5: Live G0 Repair

Investigate live capture/reseat numerical mismatch separately from translator
quality.

Work items:

- Compare static capture identity vs live logit identity layer-by-layer.
- Check mount offset, RoPE seat, dtype, cache update path, and logits after
  each mounted layer subset.
- Add a debug artifact that records max/mean logit delta per layer subset.

Exit gate:

- Live G0 max abs delta approaches the frozen floor or the remaining delta is
  explained and registered as the runtime noise floor.

## Open Queue

1. Run R4.3 5M corpus expansion and evaluate against the frozen V2 probe set.
2. Run R5 live G0 repair in parallel only when GPU/runtime time is available.
