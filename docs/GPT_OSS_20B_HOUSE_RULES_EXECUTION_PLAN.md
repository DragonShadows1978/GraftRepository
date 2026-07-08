# GPT-OSS-20B House Rules Execution Plan

Status: immutable after initial commit.
Created: 2026-07-06
Branch: `codex/intn-model-ppl-sweep`

## Objective

Control the remaining `openai/gpt-oss-20b` APA/GRM work under the repo house
rules so every claim is tied to the right evidence tier.

The model track already has a registered intent plan:
`docs/GPT_OSS_20B_APA_GRM_PLAN.md`.

This document does not replace that plan. It defines how the rest of the work is
allowed to advance, what counts as evidence, and where results must be recorded.

## House Rules

- The registered intent remains fixed in
  `docs/GPT_OSS_20B_APA_GRM_PLAN.md`.
- This execution plan is fixed after its initial commit.
- The operational trail goes in
  `docs/GPT_OSS_20B_APA_GRM_LEDGER.md`.
- The narrative interpretation goes in
  `docs/GPT_OSS_20B_APA_GRM_SYNTHESIS.md`.
- Every result-bearing run records command, artifact path, model source, mode,
  and interpretation.
- Failure is a result.
- OOM is a result.
- Bad prompts, truncated prompts, and wrong protocol runs are results, but they
  do not prove model behavior.
- Toy PPL smokes are wiring receipts, not benchmark results.
- Synthetic context fill is not context-extension evidence.
- Single-token top-k is not greedy generation evidence.
- Greedy generation is not GRM continuity evidence.
- Graft mount tests are not successful unless the answer depends on facts that
  are absent from the live prompt/context.
- Do not call APA enabled unless the attention path actually routes through
  the APA implementation for the tested layers.
- Do not call GPT-OSS APA correct unless learned attention sinks are included in
  the softmax denominator.
- Do not call long-context support proven until the tested prompt uses real
  token fill and reports the OOM boundary.
- Commit per stable implementation checkpoint or completed evidence gate.

## Evidence Tiers

Tier 0: source and metadata.
- Config, tokenizer, safetensor, model-card, and local snapshot facts.
- This can justify implementation planning only.

Tier 1: unit and kernel receipts.
- Focused tests for TensorCUDA kernels and local wrapper math.
- This proves local primitives, not model behavior.

Tier 2: layer and streamed-forward smokes.
- Real model tensors through one layer or all layers.
- This proves the custom path is connected, not that quality is acceptable.

Tier 3: short behavior receipts.
- Top-k, tiny PPL, short greedy, and Harmony protocol smokes.
- This proves the path can produce plausible outputs under narrow prompts.

Tier 4: real-text PPL and memory gates.
- Teacher-forced scoring over real text windows with memory reporting.
- This is the first tier allowed to support quality and quantization claims.

Tier 5: real-text context extension gates.
- Real-token fill, measured VRAM, and an OOM ladder.
- This is the first tier allowed to support context-window claims.

Tier 6: GRM cold-KV continuity gates.
- KV cleared per turn, grafts mounted per turn, and facts omitted from the live
  prompt after capture.
- This is the first tier allowed to support GPT-OSS GRM usefulness.

## Current Locked Receipts

These are the current highest evidence tiers already reached:

- Tier 1: Project-Tensor sink-aware APA blend softmax exists and passes focused
  tests.
- Tier 2: GraftRepository streamed TensorCUDA path can run all 24 GPT-OSS
  layers with resident packed MXFP4 experts one layer at a time.
- Tier 3: Plain prompt top-k produces ` Paris` for
  `The capital of France is`.
- Tier 3: Tiny six-target PPL smoke runs for standard and APA paths.
- Tier 3: Harmony-formatted prompt can run through the streamed path and emits
  the expected protocol transition token.

The current receipts do not prove:

- corpus PPL,
- usable long context,
- APA memory flattening,
- greedy Harmony content quality,
- GRM capture/remount,
- cold-KV multi-turn recall.

## Remaining Execution Phases

### Phase H0: Checkpoint Hygiene

Goal: finish the current sink-aware APA checkpoint cleanly.

Required work:
- Ensure the Project-Tensor sink-aware APA commit hash is recorded.
- Ensure GraftRepository records code, command, artifact, and result receipts.
- Run focused compile/test checks.
- Commit the GraftRepository APA integration and docs.

Gate:
- Passes when the worktree has a stable commit with tests recorded in the
  ledger.

### Phase H1: APA Correctness Consolidation

Goal: make APA selectable and auditable for GPT-OSS.

Required work:
- Keep `standard` and `apa_selective` attention modes explicitly selectable.
- Log refine percentile and bulk bit width in every artifact.
- Add a failure if `apa_selective` is requested without sink-aware TensorCUDA
  support.
- Add an artifact field that states which attention layers used APA.

Gate:
- A standard-vs-APA short prompt comparison runs and records both artifacts.

### Phase H2: Real-Text PPL Gate

Goal: move from toy PPL to real quality evidence.

Protocol:
- Use a fixed real text corpus.
- Score enough tokens to make the number meaningful.
- Compare at minimum:
  - standard attention,
  - APA r0.15,
  - APA r0.10 if r0.15 is stable.
- Keep quantization and prompt formatting constant.

Evidence required:
- corpus source,
- scored token count,
- mean NLL,
- PPL,
- load memory,
- post-run memory,
- peak observed memory,
- artifact path.

Gate:
- APA remains viable only if PPL does not collapse versus standard.

### Phase H3: Tiled Sink-Aware APA Memory Path

Goal: remove the current cuBLAS full-score-matrix limitation.

Required work:
- Replace the smoke-only score-matrix APA path for long-context probes with an
  O(L) or tiled sink-aware APA path.
- Preserve learned sink denominator behavior.
- Preserve `VD != D` correctness.
- Run focused tests against a reference for masks, GQA, sinks, and value width.

Gate:
- Focused tests pass and the implementation can run a real-token context ladder
  without materializing the full `[B, H, L, S]` score matrix.

### Phase H4: Real-Token Context Ladder

Goal: establish the real usable context boundary.

Protocol:
- Fill with real tokens, not repeated synthetic tokens.
- Use the same methodology as the prior MiniCPM/Mistral/Qwen context tests.
- Test standard and APA where possible.
- Record the first clean pass, first failure, and first OOM.

Evidence required:
- fill source,
- target token length,
- actual token length,
- attention mode,
- refine percentile,
- memory before load,
- memory after load,
- peak during prefill,
- error/OOM text,
- artifact path.

Gate:
- Only this phase can support a GPT-OSS context-extension claim.

### Phase H5: GPT-OSS GRM Capture And Mount

Goal: attach graft capture/remount semantics to GPT-OSS.

Required work:
- Capture pre-RoPE K/V graft material at the correct GPT-OSS attention boundary.
- Preserve YARN/RoPE remount semantics.
- Handle alternating sliding/full attention layers deliberately.
- Record graft geometry, layer IDs, token spans, and dtype/quantization.
- Keep the live KV cache clear between turns in GRM continuity tests.

Gate:
- A same-model graft remount test recalls a fact that is not present in the
  live prompt.

### Phase H6: Cold-KV Multi-Turn Needle

Goal: test the actual operating mode.

Protocol:
- Turn 1 captures facts into grafts.
- Later turns clear KV and remount only grafts needed for continuity.
- Probe exact facts at late turns, including turn 50 if runtime permits.
- Include an amnesia/no-graft control.

Evidence required:
- turn count,
- facts planted,
- facts asked,
- graft count and byte size,
- live prompt token count,
- answer text,
- exact hit/miss,
- degeneration notes,
- artifact path.

Gate:
- GPT-OSS GRM is useful only if it beats the amnesia control on facts omitted
  from the live prompt.

### Phase H7: Existing-Model Comparison

Goal: decide whether GPT-OSS deserves more optimization time.

Compare against existing local operating points:
- Qwen3.5,
- Gemma,
- DeepSeek-V2-Lite,
- MiniCPM3 where relevant.

Decision criteria:
- memory footprint,
- usable context,
- PPL or behavior quality,
- cold-KV GRM recall,
- implementation complexity,
- runtime speed.

Gate:
- Keep GPT-OSS only if it gives a defensible advantage or a new research result.

## Stop Conditions

Stop or pivot if:

- sink-aware APA cannot be made memory-efficient,
- real-text PPL collapses under the required quantization path,
- Harmony decode cannot be made protocol-correct,
- GRM remount cannot recall facts absent from the live prompt,
- the final operating point is worse than existing local models without a
  compensating research finding.

