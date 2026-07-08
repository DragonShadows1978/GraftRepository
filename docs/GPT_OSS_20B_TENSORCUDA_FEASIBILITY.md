# GPT-OSS-20B TensorCUDA Feasibility Design

Status: Phase 2 design checkpoint.
Created: 2026-07-06.
Branch: `codex/intn-model-ppl-sweep`.

This document is not the immutable implementation plan. It is the Phase 2
engineering design produced from the Phase 0/1 receipts.

## Verdict

GPT-OSS-20B is feasible as a TensorCUDA target, but not by treating it like the
existing dense INT4 loaders.

The viable path is hybrid:

1. Preserve the published MXFP4 expert body in packed form.
2. Quantize the BF16 attention/router/lm-head body through the local INTN path.
3. Keep embeddings out of the persistent GPU budget unless a GPU-resident
   embedding scheme proves affordable.
4. Bring up standard attention first.
5. Add sink-aware APA only to `full_attention` layers after standard parity.
6. Treat sliding layers as bounded 128-token local attention until profiling or
   behavior tests prove a reason to complicate them.

The non-viable path is dequantizing the expert body to BF16 and calling that a
12GB operating point. Exact dequantization may be useful as a correctness
fallback, but it invalidates the memory case.

## Source Evidence

Local artifact receipts:

- Phase 0 artifact:
  `artifacts/gpt_oss_20b/phase0_snapshot_20260706_182416.json`
- Cold Ollama smoke:
  `artifacts/gpt_oss_20b/ollama_smoke_20260706_183047.json`
- Warm `think:false` Ollama smoke:
  `artifacts/gpt_oss_20b/ollama_smoke_20260706_183232.json`

Local implementation references:

- HF model implementation:
  `/home/vader/.local/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py`
- HF MXFP4 integration:
  `/home/vader/.local/lib/python3.12/site-packages/transformers/integrations/mxfp4.py`
- HF MXFP4 quantizer:
  `/home/vader/.local/lib/python3.12/site-packages/transformers/quantizers/quantizer_mxfp4.py`
- Existing TensorCUDA low-bit wrappers:
  `/mnt/ForgeRealm/Project-Tensor/tensor_cuda/tensor_cuda/__init__.py`
  and `core/mistral7b_tc.py`
- Existing correctness-first MoE dispatch pattern:
  `core/deepseek_v2_lite_tc.py`

## Tensor Map

Representative layer tensor names and shapes from safetensors header range
inspection:

| Tensor | Dtype | Shape |
| --- | --- | --- |
| `model.embed_tokens.weight` | BF16 | `[201088, 2880]` |
| `lm_head.weight` | BF16 | `[201088, 2880]` |
| `model.norm.weight` | BF16 | `[2880]` |
| `model.layers.N.input_layernorm.weight` | BF16 | `[2880]` |
| `model.layers.N.post_attention_layernorm.weight` | BF16 | `[2880]` |
| `model.layers.N.self_attn.q_proj.weight` | BF16 | `[4096, 2880]` |
| `model.layers.N.self_attn.q_proj.bias` | BF16 | `[4096]` |
| `model.layers.N.self_attn.k_proj.weight` | BF16 | `[512, 2880]` |
| `model.layers.N.self_attn.k_proj.bias` | BF16 | `[512]` |
| `model.layers.N.self_attn.v_proj.weight` | BF16 | `[512, 2880]` |
| `model.layers.N.self_attn.v_proj.bias` | BF16 | `[512]` |
| `model.layers.N.self_attn.o_proj.weight` | BF16 | `[2880, 4096]` |
| `model.layers.N.self_attn.o_proj.bias` | BF16 | `[2880]` |
| `model.layers.N.self_attn.sinks` | BF16 | `[64]` |
| `model.layers.N.mlp.router.weight` | BF16 | `[32, 2880]` |
| `model.layers.N.mlp.router.bias` | BF16 | `[32]` |
| `model.layers.N.mlp.experts.gate_up_proj_blocks` | U8 | `[32, 5760, 90, 16]` |
| `model.layers.N.mlp.experts.gate_up_proj_scales` | U8 | `[32, 5760, 90]` |
| `model.layers.N.mlp.experts.gate_up_proj_bias` | BF16 | `[32, 5760]` |
| `model.layers.N.mlp.experts.down_proj_blocks` | U8 | `[32, 2880, 90, 16]` |
| `model.layers.N.mlp.experts.down_proj_scales` | U8 | `[32, 2880, 90]` |
| `model.layers.N.mlp.experts.down_proj_bias` | BF16 | `[32, 2880]` |

Important shape implications:

- `q_proj` output is `64 * 64 = 4096`.
- `k_proj` and `v_proj` output are `8 * 64 = 512`.
- `o_proj` input is 4096 and output is 2880.
- Expert MXFP4 blocks use 32-wide input groups; `2880 / 32 = 90`.
- `gate_up_proj` is interleaved gate/up at `2 * intermediate_size = 5760`.
- There are 32 local experts and 4 experts active per token.

## Model Semantics To Preserve

### RMSNorm

GPT-OSS RMSNorm is plain T5-style RMSNorm:

- compute variance in fp32,
- multiply by the learned weight directly,
- no `(1 + weight)` bake.

Use `RMSNormTC` with `eps = 1e-5` unless config says otherwise.

### Attention

Attention is conventional GQA, but not plain Llama attention:

- Q heads: 64.
- KV heads: 8.
- Head dim: 64.
- RoPE applies to full Q/K head dim.
- Layer types alternate `sliding_attention`, `full_attention`.
- Sliding window is 128.
- Every attention layer has learned `sinks` with shape `[64]`.

The sink behavior is load-bearing:

1. compute attention logits over real keys,
2. append one learned sink logit per Q head,
3. softmax over `S + 1`,
4. drop the sink probability before multiplying by V.

The sink therefore changes the denominator even though it contributes no value.
Any TensorCUDA attention path must reproduce that denominator.

### Router

Router semantics from HF:

- `router_logits = linear(hidden, router.weight, router.bias)`
- select top 4 experts by raw router logits,
- softmax only over the selected top-4 values,
- route weights multiply expert outputs.

DeepSeek's token-by-token MoE dispatch is a valid correctness-first pattern.
Grouped dispatch is an optimization, not a Phase 3 prerequisite.

### Expert Activation

GPT-OSS expert activation is not the existing `SwiGLU_TC`.

HF expert path:

- `gate, up = gate_up[..., ::2], gate_up[..., 1::2]`
- `gate = clamp(gate, max=7)`
- `up = clamp(up, min=-7, max=7)`
- `glu = gate * sigmoid(gate * 1.702)`
- `activated = (up + 1) * glu`
- `out = activated @ down + down_bias`

This needs a dedicated GPT-OSS expert MLP implementation.

## MXFP4 Expert Format

The local HF integration defines the exact fallback dequantization:

- FP4 codebook:
  `0, 0.5, 1, 1.5, 2, 3, 4, 6` plus signed variants.
- Each byte stores two FP4 codes:
  - low nibble is the even element,
  - high nibble is the odd element.
- Scale bytes are interpreted as exponents:
  `exponent = uint8_scale - 127`.
- Dequantized values are produced with `ldexp(fp4_value, exponent)`.
- Blocks have shape `[..., G, B]`; output expands to `[..., G * B * 2]`.
- HF then transposes the expanded tensor for the normal expert matmul layout.

This exact dequant path is useful for one-layer parity and small CPU/GPU
diagnostics. It is not the viable resident path for the 4070 Super.

## Current TensorCUDA Support Matrix

Already reusable:

- INT2/INT3/INT4 affine weight packing and fused/two-stage linear kernels.
- BF16 compute path used by Qwen/DeepSeek.
- RMSNorm fused path.
- RoPE kernel.
- GQA attention scaffolding.
- APA selective attention kernels.
- KV INT4 pack/unpack.
- Top-k primitive.
- DeepSeek-style simple MoE dispatch.
- GRM pre-RoPE capture/injection patterns for GQA and MLA families.

Missing or requiring GPT-OSS-specific work:

- Biased quantized linear wrapper. `QuantLinearTC` currently has no bias.
- YARN RoPE parameter parity with HF's `rope_parameters` path.
- Sink-aware standard attention.
- Sink-aware APA attention.
- Sliding-window cache/mask path with window 128.
- Native packed MXFP4 expert matmul/GEMV.
- Dedicated GPT-OSS expert activation.
- Harmony-aware TensorCUDA generation harness.
- GRM dialect that skips or bounds sliding layers and treats full-attention
  layers as primary graft carriers.

## Implementation Strategy

### Phase 3A: Correctness Scaffold

Goal: build the smallest loader that can execute a few layers correctly.

Steps:

1. Create `core/gpt_oss20b_tc.py`.
2. Add `GptOss20BConfig.from_model_dir()`.
3. Add `BiasedQuantLinearTC` as a local wrapper around `QuantLinearTC`.
4. Add host-backed embedding first; do not pin the BF16 embedding on GPU.
5. Quantize `lm_head` through INT4 or INT3 for the first full-logit path.
6. Implement RMSNorm, attention projections, RoPE, sink-aware standard attention.
7. Implement the router and a correctness-first MoE loop.
8. Implement exact MXFP4 dequant for one layer only as a parity fallback.
9. Run `max_layers=1`, `load_lm_head=false` hidden-state smoke before full load.

This phase may dequantize one layer's experts for parity, but it must not be
reported as a viable 12GB model path.

### Phase 3B: Packed Expert Operating Point

Goal: avoid BF16 expert expansion.

Required Project-Tensor primitive:

- `mxfp4_expert_linear(x, blocks, scales, bias, hidden_size, out_size)`

Minimum acceptable behavior:

- decode/GEMV path first,
- one selected expert at a time is acceptable initially,
- no full expert matrix materialization,
- output matches the exact dequant fallback within a registered tolerance.

For prefill, the correctness-first route can process chunks or selected expert
batches. Throughput is secondary until parity and memory are proven.

### Phase 4: APA

APA should target only `full_attention` layers first:

- Layer 0 is `sliding_attention`.
- Odd layers are `full_attention`.
- Even layers are `sliding_attention`.

Sink-aware APA is mandatory. Existing `apa_selective_attention` does not model
the extra sink denominator, so either:

1. add a sink-aware variant of the fused kernel, or
2. use the cuBLAS blend path where sink logits can be appended before softmax.

Do not claim GPT-OSS APA parity until sink behavior is included.

### Phase 5: GRM

GPT-OSS is graftable because it uses RoPE/YARN. The GRM dialect should:

- capture pre-RoPE K/V from attention layers,
- preserve sink parameters as part of the model, not as graft payload,
- treat full-attention layers as primary graft payload carriers,
- either skip sliding layers or mount only within their 128-token local window,
- record whether a graft came from a full or sliding layer,
- clear KV per turn and mount grafts per turn in the same operating mode used
  for Qwen/Gemma/DeepSeek tests.

The first GRM gate should be a multi-turn greedy recall gate, not a synthetic
context-length claim.

## Memory Expectations

Official Ollama cold smoke:

- baseline: 275 MiB,
- peak: 11,392 MiB,
- model residency: 20% CPU / 80% GPU,
- context shown by Ollama: 4096.

This proves short-prompt official runtime viability, not full-resident
TensorCUDA viability.

TensorCUDA memory case:

- Expert U8/MXFP4 payload: about 9.46 GiB.
- BF16 non-expert payload: about 3.36 GiB.
- Embedding plus lm head BF16 together: about 2.16 GiB.
- Fully resident packed experts plus compressed non-experts is plausible.
- Dequantized experts are not plausible on 12GB.

## Stop Conditions

Stop or pivot if:

- exact one-layer MXFP4 dequant parity cannot be reproduced,
- sink-aware standard attention cannot match HF/Ollama sanity behavior,
- packed MXFP4 expert matmul cannot be implemented without full expert
  materialization,
- TensorCUDA cannot fit the packed expert body plus compressed non-experts with
  enough headroom for cache and scratch,
- GRM recall fails in the real multi-turn operating mode after standard
  attention and Harmony formatting are correct.
