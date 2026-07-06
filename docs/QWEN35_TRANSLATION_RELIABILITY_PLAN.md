# Qwen3.5 Translation Reliability Plan

**Status:** R4.7 fine residual-scale sweep complete. `s0p5_kv` remains the
frozen-safe default at frozen V2 `63 / 64`; `s0p6875_kv` is the current
holdout-tuned candidate at fresh holdout `60 / 64`. No new corpus is required
for the next step.

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

Current R4.3 status:

- Complete.
- Expanded capture closed at `3,408,241` paired tokens because the source set
  was exhausted under the current filters.
- Final capture:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/captures_5m`
  - source completed: `13820 / 13820`
  - target completed: `13820 / 13820`
  - train shards/tokens: `12437` / `3067954`
  - heldout shards/tokens: `1383` / `340287`
- Translator:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_corpus_5m`
  - ridge lambda: `1e-4`
  - backend: `cupy`
  - artifacts: `12` K/V maps
- Frozen V2 translated binding result:
  - baseline `2.5M`: `40 / 64`, mean margin `0.41503181978418846`
  - R4.3 expanded corpus: `44 / 64`, mean margin
    `0.9933952155426934`
  - R4.3 gained `5` previously failing probes and lost `1` prior success.
- Decision: R4.3 succeeds under the registered selection rule, but it does not
  close reliability. The remaining `20 / 64` misses point at objective-level
  translator training or targeted hard-negative training as the next move.

Current R4.4 status:

- Complete.
- Added reproducible layer-sweep tooling:
  - `make-translator-layer-sweep`
  - `eval-binding-translator-sweep`
  - `analyze-binding-translator-sweep`
- Added batched gold/decoy scoring for binding probes. For grafted batches,
  the translated K/V graft is repeated to match the candidate batch size.
- Candidate set:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_r44_layer_sweep/layer_sweep_manifest.json`
  - candidates: `13`
  - policies: parent `all`, six single-layer translators, six leave-one-out
    translators
- Frozen V2 translated binding result:
  - R4.3 all-layer baseline: `44 / 64`, mean margin
    `0.9933952155426934`
  - R4.4 best global: `drop_l3_to_l3`, `46 / 64`, mean margin
    `0.935202663147657`
  - R4.4 diagnostic per-probe oracle: `48 / 64`, mean margin
    `1.325906873199195`
- Decision: layer routing gives real but limited headroom. Dropping the first
  layer pair is the best tested global policy, but the remaining misses still
  require an objective-level translator or targeted hard-negative training.

Current R4.5 prep status:

- Complete.
- Promoted the best tested R4.4 global policy into a stable selected
  translator directory:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_r45_selected_drop_l3`
  - policy: drop `3->3`
  - artifact count: `10`
  - frozen V2 translated score from R4.4: `46 / 64`
- Added CPU-only hard-negative plan tooling:
  `make-binding-hard-negative-plan`.
- Wrote the R4.5 hard-negative plan:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_hard_negative_plan_r45_drop_l3.json`
  - selected label: `drop_l3_to_l3`
  - hard-negative items: `18`
  - oracle-recoverable items: `2`
  - oracle-hard items: `16`
  - selected recovered baseline misses: `4`
  - selected lost baseline successes: `2`
- Decision: do not generate more corpus yet. The next bottleneck is objective
  quality, not capture volume. Use CUDA/CuPy for the next fit/eval because the
  full existing capture set is large enough that CPU training would waste time.

Current R4.5 residual-focus status:

- Complete.
- Focus capture:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/captures_r45_hard_negative`
  - paired train shards/tokens: `18` / `814`
- Focus plan:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/corpus_plan_r45_hard_negative.json`
- Base translator:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_r45_selected_drop_l3`
- Residual translator:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_r45_residual_focus_s025`
  - residual scale: `0.25`
  - ridge lambda: `1e-4`
  - refined kinds: `k,v`
  - backend: `cupy`
  - artifact count: `10`
- Frozen V2 result:
  - selected drop-l3 baseline: `46 / 64`, mean margin
    `0.935202663147657`
  - residual `s0.25` K+V: `53 / 64`, mean margin
    `2.9803028386585195`, min margin `-3.0140718657008208`
- Fresh holdout result:
  - selected drop-l3 baseline: `39 / 64`, mean margin
    `0.9628342859065595`, min margin `-5.031213117271335`
  - residual `s0.25` K+V: `54 / 64`, mean margin
    `2.98475187406485`, min margin `-5.2945312158358675`
- Decision: hard-negative residual training is not overfitting to the frozen
  set. It improves the fresh holdout by `+15` positives over selected drop-l3.

Current R4.6 residual/KV split sweep status:

- Complete.
- Protocol:
  - Same pre-RoPE capture plane as production grafts.
  - No probe changes.
  - No post-RoPE target or loss.
  - No new broad corpus capture.
  - Frozen probe:
    `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_probes_v2.json`
  - Fresh holdout:
    `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_probes_v2_holdout_r45.json`
  - Candidate residual scales: `0.125`, `0.25`, `0.5`
  - Candidate kind specs: `k`, `v`, `both`
- Residual sweep manifest:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_r46_residual_sweep/residual_sweep_manifest.json`
  - candidates: `9`
  - labels: `s0p125_k`, `s0p125_v`, `s0p125_kv`, `s0p25_k`,
    `s0p25_v`, `s0p25_kv`, `s0p5_k`, `s0p5_v`, `s0p5_kv`
- Frozen V2 result:
  - prior best `s0p25_kv`: `53 / 64`, mean margin
    `2.9803028386585195`
  - new best `s0p5_kv`: `63 / 64`, mean margin
    `5.931814594442409`, min margin `-0.9922356751544044`
  - diagnostic oracle over the 9 candidates: `64 / 64`, mean margin
    `6.135559159773022`, min margin `1.78372954010959`
  - best global recovered `11` prior frozen misses and lost `1` prior frozen
    success (`bind-v2-011-q0`)
- Fresh holdout result:
  - selected drop-l3 baseline: `39 / 64`, mean margin
    `0.9628342859065595`
  - prior best `s0p25_kv`: `54 / 64`, mean margin
    `2.98475187406485`
  - new best `s0p5_kv`: `58 / 64`, mean margin
    `5.003952600746849`, min margin `-3.777912148347035`
  - `s0p5_kv` recovered `4` prior holdout misses and lost no prior holdout
    successes versus `s0p25_kv`.
- Split result:
  - `s0p5_k`: frozen `58 / 64`, mean margin `4.130273350212388`
  - `s0p5_v`: frozen `51 / 64`, mean margin `1.8453519269352137`
  - `s0p5_kv`: frozen `63 / 64`, mean margin `5.931814594442409`
- Decision: K correction is load-bearing, V-only is weaker, and K+V wins.
  The pre-RoPE residual pivot is the active path. The next work should focus
  on learned gating, remaining miss analysis, or deeper live/open-generation
  reliability tests before generating more broad corpus.

Current R4.7 fine residual-scale status:

- Complete.
- Protocol:
  - Keep the R4.6 base, hard-negative focus capture, and pre-RoPE target
    plane unchanged.
  - Do not generate new broad corpus.
  - First fit fine scales `0.375`, `0.5`, `0.625`, `0.75`, and `0.875` across
    K/V split candidates.
  - Abort the full 15-candidate frozen binding eval after `5 / 64` probes
    because candidate scoring was too expensive for the immediate question.
  - Use result-bearing K+V-only fine and micro sweeps.
- Fine K+V sweep:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_r47_residual_fine_sweep`
  - K+V scales: `0.375`, `0.5`, `0.625`, `0.75`, `0.875`
- Micro K+V sweep:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_r47_residual_micro_sweep`
  - K+V scales: `0.6875`, `0.71875`
- Frozen V2 result:
  - `s0p5_kv`: `63 / 64`, mean margin `5.931814594442409`, min margin
    `-0.9922356751544044`
  - `s0p625_kv`: `62 / 64`, mean margin `7.610320836733261`, min margin
    `-3.597667722569625`
  - `s0p6875_kv`: `62 / 64`, mean margin `8.457185386698411`, min margin
    `-5.453743171778854`
  - `s0p75_kv`: `60 / 64`, mean margin `9.10674637797445`, min margin
    `-7.48717160149144`
- Fresh holdout result:
  - `s0p5_kv`: `58 / 64`, mean margin `5.003952600746849`, min margin
    `-3.777912148347035`
  - `s0p625_kv`: `59 / 64`, mean margin `5.281574454660446`, min margin
    `-2.6199292050993606`
  - `s0p6875_kv`: `60 / 64`, mean margin `5.165943132279072`, min margin
    `-2.0714553694315327`
  - `s0p75_kv`: `60 / 64`, mean margin `4.996703390102546`, min margin
    `-2.0782226450213415`
- Gained/lost versus R4.6 `s0p5_kv`:
  - `s0p6875_kv` recovers holdout `bind-v2-016-q0` and
    `bind-v2-016-q1` with no holdout losses.
  - `s0p6875_kv` loses frozen `bind-v2-011-q1` and recovers no frozen
    misses.
- Decision:
  - Keep `s0p5_kv` as the conservative/frozen-safe default.
  - Register `s0p6875_kv` as the holdout-tuned secondary candidate.
  - The fact that different scales dominate different probes means the next
    refinement should be learned routing/gating over residual candidates, not
    more blind scale growth.

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

1. Implement a minimal learned gate/router over `s0p5_kv` and `s0p6875_kv`,
   with `s0p625_kv` as an optional intermediate candidate.
2. Validate the router against frozen V2 and fresh holdout before adding any
   new broad corpus.
3. Run a deeper live/open-generation reliability gate using both the
   conservative default and any routed policy before making production-grade
   claims.
4. Run R5 live G0 repair in parallel only when GPU/runtime time is available.
