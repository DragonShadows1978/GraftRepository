# GPT-OSS-20B APA/GRM Evaluation Plan

Status: immutable after the initial house-rule commit.
Created: 2026-07-06
Branch: `codex/intn-model-ppl-sweep`

## Objective

Evaluate whether `openai/gpt-oss-20b` is worth bringing into the local
TensorCUDA / APA / GRM stack.

The question is not only whether the model can run. The question is whether its
architecture creates a practical operating point on a 12GB-class GPU when
weights stay resident, KV cache is cleared per turn, and GRM mounts continuity
state per turn.

## House Rules

- This plan is the fixed source of intent after its initial commit.
- The ledger records commands, artifacts, source facts, measurements, failures,
  OOMs, and decisions.
- The synthesis explains the ledger in narrative form.
- Failure is a result.
- Do not call a metadata estimate a load result.
- Do not call an official vLLM/Ollama smoke a TensorCUDA validation.
- Do not dequantize the MXFP4 expert body into BF16 and then call the result a
  viable 12GB path.
- Do not evaluate gpt-oss behavior outside the Harmony chat format.
- Do not show raw chain-of-thought to end users in application-facing tests.

## Known Source Facts

Official / model-card facts to preserve before implementation:

- Model: `openai/gpt-oss-20b`
- License: Apache 2.0
- Architecture: MoE transformer
- Total parameters: about 21B / 22B depending on source display
- Active parameters: about 3.6B per token
- Layers: 24
- Experts: 32 local experts
- Experts active per token: 4
- Hidden size: 2880
- Attention heads: 64
- KV heads: 8
- Head dim: 64
- Attention: alternating `sliding_attention` and `full_attention`
- Sliding window: 128
- Max position embeddings: 131072
- Positional scheme: RoPE with YARN scaling
- Initial context length: 4096
- Vocab size: 201088
- `tie_word_embeddings`: false
- Published expert weights: MXFP4 blocks/scales
- Non-converted modules include attention, router, embeddings, and lm head.
- Harmony response format is required for correct operation.

Source links:
- OpenAI open models: `https://openai.com/open-models/`
- OpenAI gpt-oss repository: `https://github.com/openai/gpt-oss`
- Hugging Face model card: `https://huggingface.co/openai/gpt-oss-20b`
- HF config: `https://huggingface.co/openai/gpt-oss-20b/raw/main/config.json`
- OpenAI implementation notes:
  `https://developers.openai.com/cookbook/articles/gpt-oss/verifying-implementations`

## Initial Memory Model

Remote safetensors header inspection reported:

- Total payload: `13,761,264,768` bytes = `12.816 GiB`
- MXFP4/U8 expert tensors: `10,152,345,600` bytes = `9.455 GiB`
- BF16 non-expert tensors: `3,608,919,168` bytes = `3.361 GiB`
- `model.embed_tokens.weight`: about `1104.61 MiB`
- `lm_head.weight`: about `1104.61 MiB`

Estimated if the MXFP4 expert body remains packed and only the BF16 non-expert
body is requantized:

- Non-experts at INT4-ish `4.25 bits/param`: total about `10.348 GiB`
- Non-experts at INT3-ish `3.25 bits/param`: total about `10.138 GiB`

These are estimates, not load receipts.

## Why This Model Is Interesting

GPT-OSS-20B combines several traits that matter for this stack:

- RoPE/YARN means pre-RoPE graft extraction should remain structurally viable.
- GQA with 8 KV heads gives APA room to reduce context-side memory pressure.
- Alternating sliding/full attention means only half the layers need full-context
  pressure management; sliding layers are capped at 128.
- MoE gives a large total model with relatively low active compute per token.
- The published MXFP4 expert body is already aggressively compressed.

## Main Risks

- The stock HF payload is larger than the 4070 Super's usable 12GB VRAM budget
  once runtime buffers are included.
- Existing TensorCUDA INTN kernels do not implement MXFP4 expert matmul.
- Converting MXFP4 experts to BF16 would likely destroy the memory case.
- Converting experts to local affine INT3/INT4 may be possible but is a new
  quality-risk axis.
- MoE top-4 dispatch is a different hot path than the current dense loaders.
- Harmony formatting is mandatory and must be supported in every behavior test.
- Attention sinks appear in the tensor map and must be handled correctly.

## Phase 0: Source And Local Asset Gate

Goal: establish exact local assets and source metadata without running a full
model path.

Required checks:

1. Confirm whether `openai/gpt-oss-20b` is already present in local HF cache.
2. Record exact commit/revision if present or chosen for download.
3. Record `config.json`, `generation_config.json`, tokenizer files, and
   safetensors index metadata.
4. Confirm the local stack can parse Harmony chat format, or identify the
   package/API needed to render it.
5. Record source facts in the ledger before any load experiment.

Gate:
- Phase passes if all model metadata and formatting requirements are recorded.
- Phase fails if the model cannot be acquired or source metadata is ambiguous.

## Phase 1: Official Runtime Smoke

Goal: determine whether the model can run on this machine before building a
TensorCUDA port.

Candidate routes:

- Ollama: `ollama pull gpt-oss:20b`
- vLLM: `vllm serve openai/gpt-oss-20b`
- Transformers only if the installed version supports the model cleanly.

Required measurements:

- Load success/failure.
- Peak VRAM after load.
- Peak VRAM for a short prompt and short decode.
- Prompt format used.
- Generated answer sanity.
- Any OOM or unsupported-op error.

Gate:
- Phase passes if an official runtime can produce a short valid answer on the
  4070 Super.
- Phase fails usefully if it proves stock runtime is over budget or unsupported.

## Phase 2: TensorCUDA Feasibility Design

Goal: choose a viable implementation path before touching kernels.

Decision branches:

1. Native MXFP4 path:
   - Keep expert blocks/scales in their published form.
   - Implement MXFP4 expert GEMV/GEMM or adapt from the reference code.
   - Quantize attention/router/embed/lm_head through existing INTN wrappers.
2. Requantized expert path:
   - Convert expert MXFP4 to local affine INT4/INT3 only if memory and quality
     gates justify it.
   - This path requires PPL and behavior checks because it changes the expert
     quantization, not just the runtime kernel.
3. Hybrid path:
   - Preserve MXFP4 experts.
   - Quantize only BF16 non-experts.
   - Add APA only to full-attention layers first.

Preferred first implementation path:
- Hybrid path, because it preserves the official expert quantization and attacks
  the known BF16 memory overhead.

## Phase 3: Minimal TensorCUDA Loader

Goal: produce a correctness-first loader.

Required features:

- Config parser for `gpt_oss`.
- Embedding and lm_head handling; decide whether to quantize both.
- RMSNorm path.
- Q/K/V/O projections with bias.
- RoPE/YARN support.
- Alternating sliding/full attention.
- Attention sinks.
- Router logits and top-4 selection.
- Expert dispatch over `gate_up_proj` and `down_proj`.
- Harmony prompt rendering in the test harness.

Non-goal for the first loader:
- No long-context claim.
- No GRM claim.
- No APA claim.
- No production throughput claim.

Gate:
- A small teacher-forced or greedy sanity check must run before any optimization
  claim is made.

## Phase 4: APA Integration

Goal: attach APA only where it can matter.

Rules:

- Full-attention layers are the primary APA target.
- Sliding-window layers stay on the simple bounded path unless profiling proves
  a reason to change them.
- Start with standard attention parity before APA.
- Then add APA r0.15, r0.10, and r0.05.

Required measurements:

- PPL on real text.
- Peak VRAM during prefill.
- Peak VRAM during decode.
- Context-length OOM ladder.
- Greedy output sanity.

Gate:
- APA is useful only if it increases usable context / graft budget without a
  behavior collapse.

## Phase 5: GRM Conversation Continuity Gate

Goal: test the actual operating mode.

Protocol:

- Keep weights resident.
- Clear KV cache per turn.
- Mount grafts per turn for continuity.
- Compare:
  - stock/runtime baseline if available
  - TensorCUDA standard
  - TensorCUDA APA
  - equal graft budget
  - expanded graft budget enabled by memory savings

Required probes:

- Early-turn fact recall.
- Exact code/name/number recall.
- User preference recall.
- Correction/supersession handling.
- Multi-turn instruction retention.
- Repetition/drift.
- OOM boundary with graft payload.

Gate:
- This model is worth keeping only if its MoE/APA path improves cold-KV
  continuity economics versus existing Qwen/Gemma/DeepSeek choices.

## Stop Conditions

Stop or pivot if any of these happen:

- Harmony formatting cannot be represented correctly in local tests.
- Official runtime cannot load and TensorCUDA estimates leave less than a safe
  scratch margin.
- MXFP4 experts require dequantizing into BF16 for the hot path.
- Standard attention parity fails in a way that cannot be localized.
- PPL or usability collapses after non-expert quantization.
- GRM continuity does not improve versus the current Qwen3.5/Gemma/DeepSeek
  operating points.

## First Concrete Next Step

Run Phase 0 only:

1. Create a local source snapshot record from HF metadata.
2. Decide whether to download the model.
3. If downloaded, run a load-only official runtime smoke before writing
   TensorCUDA code.
