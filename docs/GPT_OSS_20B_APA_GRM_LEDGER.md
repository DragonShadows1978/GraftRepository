# GPT-OSS-20B APA/GRM Ledger

This ledger records the operational trail for evaluating
`openai/gpt-oss-20b` as a TensorCUDA / APA / GRM target.

The implementation plan is immutable after its initial commit.

## 2026-07-06

Action: Opened the GPT-OSS-20B evaluation track.

Repo state:
- Repository: `/mnt/ForgeRealm/GraftRepository`
- Branch: `codex/intn-model-ppl-sweep`
- Starting point: `da13e20 docs: record qwen35 9b int3 usability gate`

Reason for this track:
- The user asked whether `gpt-oss-20b` is worth investigating.
- The model has a potentially attractive shape for APA/GRM: RoPE/YARN,
  GQA, MoE, low active parameters, and 131k context.
- The model also has non-trivial implementation risk because the expert weights
  are published in MXFP4 and require Harmony formatting.

Source facts recorded:
- HF config says `model_type = gpt_oss`.
- `num_hidden_layers = 24`
- `hidden_size = 2880`
- `num_attention_heads = 64`
- `num_key_value_heads = 8`
- `head_dim = 64`
- `num_local_experts = 32`
- `num_experts_per_tok = 4`
- `layer_types` alternate `sliding_attention` and `full_attention`
- `sliding_window = 128`
- `max_position_embeddings = 131072`
- RoPE scaling uses YARN with factor `32.0`.
- `vocab_size = 201088`
- `tie_word_embeddings = false`
- `quantization_config.quant_method = mxfp4`
- Modules not converted by MXFP4 include attention, router, embeddings, and
  lm_head.

Remote metadata commands used:
- `curl -sS https://huggingface.co/openai/gpt-oss-20b/raw/main/config.json`
- `curl -sS https://huggingface.co/openai/gpt-oss-20b/raw/main/model.safetensors.index.json`
- HTTP range reads of the three safetensors headers to inspect tensor dtypes,
  shapes, and byte counts without downloading full weights.

Memory metadata:

| Component | Bytes | GiB |
| --- | ---: | ---: |
| MXFP4/U8 expert tensors | 10,152,345,600 | 9.455 |
| BF16 non-expert tensors | 3,608,919,168 | 3.361 |
| Total safetensors payload | 13,761,264,768 | 12.816 |

Large BF16 tensors:

| Tensor | Shape | Size |
| --- | --- | ---: |
| `model.embed_tokens.weight` | `[201088, 2880]` | 1104.61 MiB |
| `lm_head.weight` | `[201088, 2880]` | 1104.61 MiB |

Derived estimates:
- If MXFP4 experts stay packed and BF16 non-experts move to INT4-ish
  `4.25 bits/param`, total resident weight floor is about `10.348 GiB`.
- If MXFP4 experts stay packed and BF16 non-experts move to INT3-ish
  `3.25 bits/param`, total resident weight floor is about `10.138 GiB`.

Interpretation:
- Stock HF payload is likely too tight for a 12GB 4070 Super once runtime
  buffers and scratch are included.
- A local path may become viable only if the expert body remains packed and the
  non-expert BF16 body is compressed.
- Dequantizing experts to BF16 is not a viable 12GB strategy.
- The model is graftable in principle because it uses RoPE/YARN, but the
  implementation must capture/inject at the correct pre-RoPE boundary.

Action: Added and ran the Phase 0 metadata snapshot script.

Script:
- `scripts/gpt_oss_20b_phase0_snapshot.py`

Compile check:
- `env PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/gpt_oss_20b_phase0_snapshot.py`

Run command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss_20b_phase0_snapshot.py`

Artifact:
- `artifacts/gpt_oss_20b/phase0_snapshot_20260706_182416.json`

Snapshot result:
- Resolved HF revision:
  `6cee5e81ee83917806bbde320786a8fb61efebee`
- Metadata files were downloaded into the local HF cache, but full weight
  bodies were not downloaded through the HF path.
- Runtime packages:
  - `transformers = 5.12.0`
  - `huggingface_hub = 1.13.0`
  - `safetensors = 0.7.0`
  - `vllm = None`
  - `openai_harmony = None`
  - `ollama_path = /usr/local/bin/ollama`
- GPU baseline in the artifact:
  `NVIDIA GeForce RTX 4070 SUPER, 275, 12282, 37`

Config facts confirmed by artifact:
- `model_type = gpt_oss`
- `architectures = ["GptOssForCausalLM"]`
- `num_hidden_layers = 24`
- `hidden_size = 2880`
- `intermediate_size = 2880`
- `num_attention_heads = 64`
- `num_key_value_heads = 8`
- `head_dim = 64`
- `num_local_experts = 32`
- `num_experts_per_tok = 4`
- `experts_per_token = 4`
- `sliding_window = 128`
- `max_position_embeddings = 131072`
- `initial_context_length = 4096`
- `rope_theta = 150000`
- `rope_scaling.rope_type = yarn`
- `rope_scaling.factor = 32.0`
- `vocab_size = 201088`
- `tie_word_embeddings = false`
- `layer_types` alternate `sliding_attention` and `full_attention` across all
  24 layers.
- `quantization_config.quant_method = mxfp4`
- MXFP4 does not convert attention, router, embeddings, or lm head.

Safetensors facts confirmed by artifact:
- `metadata.total_size = 13,761,264,768`
- `tensor_count = 459`
- `weight_file_count = 3`
- Dtype payload:
  - `BF16 = 3.361068 GiB`
  - `U8 = 9.455109 GiB`
- Component payload:
  - `attention = 1.186884 GiB`
  - `embed_lmhead = 2.157440 GiB`
  - `experts = 9.467468 GiB`
  - `norm = 0.000263 GiB`
  - `router = 0.004121 GiB`

Harmony/tokenizer result:
- `tokenizer_class = TokenizersBackend`
- `has_chat_template = true`
- `contains_start_token = true`
- `contains_message_end = true`
- `contains_channel_token = false`
- The rendered prefix uses `<|start|>system<|message|>...<|end|>` and
  includes `Reasoning: medium` by default.

Interpretation:
- Phase 0 passes. Metadata is exact, the model revision is pinned, the memory
  model is confirmed by an artifact, and the installed tokenizer can render the
  required Harmony-style template.
- The missing `openai_harmony` package is not a Phase 0 blocker because the
  tokenizer template can render the prompt format; it remains a possible later
  dependency if local tests need lower-level Harmony APIs.

Action: Pulled and smoke-tested the official Ollama runtime.

Pull command:
- `ollama pull gpt-oss:20b`

Pull result:
- Completed successfully.
- `ollama list` row:
  - `gpt-oss:20b`
  - ID `17052f91a42e`
  - Size `13 GB`

Cold official-runtime smoke script:
- `scripts/gpt_oss_20b_ollama_smoke.py`

Compile check:
- `env PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/gpt_oss_20b_ollama_smoke.py`

Cold smoke command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss_20b_ollama_smoke.py`

Cold smoke artifact:
- `artifacts/gpt_oss_20b/ollama_smoke_20260706_183047.json`

Cold smoke result:
- `status = ok`
- `wall_seconds = 49.409`
- GPU memory:
  - baseline `275 MiB`
  - peak `11392 MiB`
  - final `11392 MiB`
  - peak delta `11117 MiB`
- Peak GPU utilization during probe: `51%`
- Ollama durations:
  - `load_duration = 36.174s`
  - `prompt_eval_duration = 9.750s`
  - `eval_duration = 3.211s`
  - `total_duration = 49.402s`
- Token counts:
  - `prompt_eval_count = 80`
  - `eval_count = 64`
- Response: `The capital`
- `done_reason = length`

Cold smoke interpretation:
- The official runtime loaded and generated without OOM on the 4070 Super.
- The answer was not a valid behavior pass because the default Ollama template
  enabled medium reasoning and a 64-token generation cap ended before the final
  visible answer completed.
- This is a prompt/runtime-options finding, not a load failure.

Warm behavior sanity command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss_20b_ollama_smoke.py --num-predict 256 --think false`

Warm behavior artifact:
- `artifacts/gpt_oss_20b/ollama_smoke_20260706_183232.json`

Warm behavior result:
- `status = ok`
- `wall_seconds = 4.331`
- GPU memory:
  - baseline `11392 MiB`
  - peak `11392 MiB`
  - final `11392 MiB`
- Peak GPU utilization during probe: `36%`
- Ollama durations:
  - `load_duration = 0.327s`
  - `prompt_eval_duration = 0.141s`
  - `eval_duration = 3.713s`
  - `total_duration = 4.326s`
- Token counts:
  - `prompt_eval_count = 75`
  - `eval_count = 74`
- Response: `The capital of France is Paris.`
- `done_reason = stop`

Ollama residency observation:
- After the cold smoke, `ollama ps` reported:
  - size `14 GB`
  - processor split `20%/80% CPU/GPU`
  - context `4096`
- This means the official smoke passes, but it is not a full-GPU/full-context
  viability claim.

Cleanup:
- `ollama stop gpt-oss:20b`
- GPU returned to baseline:
  `NVIDIA GeForce RTX 4070 SUPER, 275, 12282, 17`

Phase 1 gate:
- Passes for official short-prompt runtime sanity.
- Does not replace TensorCUDA validation.
- Does not prove 131k operation.
- Does not prove full weight residency on GPU.
- Confirms the hybrid TensorCUDA path is still the relevant next step because
  stock Ollama needed a CPU/GPU split and only showed a 4096 active context.

Action: Built the Phase 2 TensorCUDA feasibility design.

Design artifact:
- `docs/GPT_OSS_20B_TENSORCUDA_FEASIBILITY.md`

Local source files inspected:
- `/home/vader/.local/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py`
- `/home/vader/.local/lib/python3.12/site-packages/transformers/integrations/mxfp4.py`
- `/home/vader/.local/lib/python3.12/site-packages/transformers/quantizers/quantizer_mxfp4.py`
- `core/mistral7b_tc.py`
- `core/deepseek_v2_lite_tc.py`
- `core/gemma4_tc.py`
- `/mnt/ForgeRealm/Project-Tensor/tensor_cuda/tensor_cuda/__init__.py`
- `/mnt/ForgeRealm/Project-Tensor/tensor_cuda/tensor_cuda/quantization/affine.py`

Header inventory facts added:
- Attention projections are BF16 and biased:
  - `q_proj.weight [4096, 2880]`, `q_proj.bias [4096]`
  - `k_proj.weight [512, 2880]`, `k_proj.bias [512]`
  - `v_proj.weight [512, 2880]`, `v_proj.bias [512]`
  - `o_proj.weight [2880, 4096]`, `o_proj.bias [2880]`
- Every attention layer has `self_attn.sinks [64]`.
- Router tensors are BF16:
  - `router.weight [32, 2880]`
  - `router.bias [32]`
- Expert tensors are packed MXFP4/U8:
  - `gate_up_proj_blocks [32, 5760, 90, 16]`
  - `gate_up_proj_scales [32, 5760, 90]`
  - `down_proj_blocks [32, 2880, 90, 16]`
  - `down_proj_scales [32, 2880, 90]`

Implementation facts from local HF source:
- Expert activation is GPT-OSS-specific, not the existing SwiGLU:
  - interleaved `gate_up[..., ::2]` / `gate_up[..., 1::2]`
  - gate clamp max `7`
  - up clamp range `[-7, 7]`
  - `glu = gate * sigmoid(gate * 1.702)`
  - activated value is `(up + 1) * glu`
- Attention appends a learned sink logit per head before softmax, then drops
  the sink probability before multiplying by V.
- Router softmax is over the selected top-4 router logits, not all experts.
- MXFP4 exact fallback dequant uses the local HF codebook:
  `0, 0.5, 1, 1.5, 2, 3, 4, 6` plus signed variants, with scale exponent
  `uint8_scale - 127`.

Phase 2 conclusion:
- GPT-OSS is feasible but requires GPT-OSS-specific pieces before any honest
  TensorCUDA claim.
- The immediate missing pieces are biased quantized linear, YARN parity,
  sink-aware attention, the GPT-OSS expert activation, and a native packed
  MXFP4 expert matmul/GEMV path.
- Exact dequantization of experts is allowed only as a one-layer or diagnostic
  parity fallback. It is not a viable 12GB resident path.

Action: Added the Phase 3A TensorCUDA scaffold primitives and tests.

Files added:
- `core/gpt_oss20b_tc.py`
- `tests/test_gpt_oss20b_scaffold.py`

Implemented scaffold pieces:
- `GptOss20BConfig.from_model_dir()`
- `BiasedQuantLinearTC`
- exact NumPy MXFP4 block dequantization helper
- GPT-OSS expert activation helpers
- sink-aware TensorCUDA standard attention helper

Why these pieces:
- GPT-OSS attention projections have bias, but the shared `QuantLinearTC`
  wrapper is weight-only.
- GPT-OSS experts use clipped gate/up activation, not the existing SwiGLU
  helper.
- GPT-OSS attention appends learned sink logits before softmax.
- MXFP4 exact dequant is needed as a diagnostic parity fallback before a packed
  expert kernel can be trusted.

Compile check:
- `env PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile core/gpt_oss20b_tc.py tests/test_gpt_oss20b_scaffold.py`

First test command:
- `PYTEST_ADDOPTS='-p no:cacheprovider' pytest tests/test_gpt_oss20b_scaffold.py -q`

First test result:
- Failed in the sandbox for TensorCUDA allocations:
  `cudaMalloc failed: no CUDA-capable device is detected`
- Pure Python checks passed before the TensorCUDA allocation failures.
- Interpretation: sandbox/device-access issue, not a model math result.

Escalated GPU test command:
- `PYTEST_ADDOPTS='-p no:cacheprovider' pytest tests/test_gpt_oss20b_scaffold.py -q`

Escalated test result:
- Initial escalated run found a real wrapper bug:
  `ew_binary dtype mismatch` when adding bias to quantized-linear output.
- Fix: `BiasedQuantLinearTC.__call__()` now casts bias to the actual output
  dtype before addition, matching the existing `LinearTC` pattern.

Final test result:
- `7 passed, 2 warnings in 0.48s`

GPU cleanup result:
- GPU returned to baseline:
  `NVIDIA GeForce RTX 4070 SUPER, 275, 12282, 35`

Action: Downloaded the full pinned HF safetensors snapshot for loader work.

Download command:
- `python3 - <<'PY' ... snapshot_download(repo_id='openai/gpt-oss-20b', revision='6cee5e81ee83917806bbde320786a8fb61efebee', allow_patterns=[...])`

Download result:
- Completed successfully.
- Wall time from progress display: about `4:45`.
- Snapshot path:
  `/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee`

Shard verification:

| Shard | Tensors | Bytes |
| --- | ---: | ---: |
| `model-00000-of-00002.safetensors` | 196 | 4,792,272,488 |
| `model-00001-of-00002.safetensors` | 197 | 4,798,702,184 |
| `model-00002-of-00002.safetensors` | 66 | 4,170,342,232 |

Verification commands:
- Opened each shard with `safetensors.safe_open`.
- Confirmed total tensor count across shards remains `459`.
- Confirmed HF snapshot entries are symlinks to cache blobs.

Disk state after download:
- `/home/vader` had `542G` available.
- GPT-OSS HF cache directory reported `26G`, which includes blobs plus cached
  snapshot metadata/symlinks.

Action: Added and ran the one-layer real-tensor TensorCUDA attention smoke.

Files changed:
- `core/gpt_oss20b_tc.py`
- `scripts/gpt_oss20b_layer_smoke.py`

Implemented loader pieces:
- safetensors tensor-name to shard map
- row-sliced host embedding gather using `safe_open().get_slice()`
- HF-compatible YARN RoPE table builder for GPT-OSS config
- real biased attention projection loading from GPT-OSS safetensors
- sink-aware standard attention inside a GPT-OSS attention block
- attention/residual block smoke harness

Compile/test checks:
- `env PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile core/gpt_oss20b_tc.py scripts/gpt_oss20b_layer_smoke.py tests/test_gpt_oss20b_scaffold.py`
- `PYTEST_ADDOPTS='-p no:cacheprovider' pytest tests/test_gpt_oss20b_scaffold.py -q`
- Result: `7 passed, 2 warnings in 0.58s`

Layer 0 smoke command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_layer_smoke.py --layer 0 --max-tokens 16`

Layer 0 artifact:
- `artifacts/gpt_oss_20b/layer_smoke_20260706_185630.json`

Layer 0 result:
- `status = ok`
- `layer_type = sliding_attention`
- `input_ids_shape = [1, 7]`
- `hidden_shape = [1, 7, 2880]`
- `output_shape = [1, 7, 2880]`
- `kv_shapes = [[1, 8, 7, 64], [1, 8, 7, 64]]`
- `wall_seconds = 2.670`
- GPU before: `NVIDIA GeForce RTX 4070 SUPER, 275, 12282, 0`
- GPU after inside script: `NVIDIA GeForce RTX 4070 SUPER, 497, 12282, 0`

Layer 1 smoke command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_layer_smoke.py --layer 1 --max-tokens 16`

Layer 1 artifact:
- `artifacts/gpt_oss_20b/layer_smoke_20260706_185700.json`

Layer 1 result:
- `status = ok`
- `layer_type = full_attention`
- `input_ids_shape = [1, 7]`
- `hidden_shape = [1, 7, 2880]`
- `output_shape = [1, 7, 2880]`
- `kv_shapes = [[1, 8, 7, 64], [1, 8, 7, 64]]`
- `wall_seconds = 2.456`
- GPU before: `NVIDIA GeForce RTX 4070 SUPER, 275, 12282, 0`
- GPU after inside script: `NVIDIA GeForce RTX 4070 SUPER, 497, 12282, 0`

Post-run GPU state:
- `NVIDIA GeForce RTX 4070 SUPER, 275, 12282, 0`

Interpretation:
- Real GPT-OSS safetensors can now feed a TensorCUDA attention/residual block.
- Both layer families have been smoke-tested: sliding attention and full
  attention.
- This is not a full model loader and not a behavior/PPL result because MoE and
  lm_head are still skipped.
- Next hard gate is GPT-OSS MoE: exact diagnostic dequant for selected experts,
  then packed MXFP4 expert math for the viable path.

Pending:
- Build the GPT-OSS MoE diagnostic path on top of the attention scaffold.
- Add packed MXFP4 expert math before making any viable 12GB operating-point
  claim.

Action: Added and ran the selected-expert GPT-OSS MoE diagnostic smoke.

Files changed:
- `core/gpt_oss20b_tc.py`
- `scripts/gpt_oss20b_moe_diag_smoke.py`

Implemented diagnostic pieces:
- real GPT-OSS BF16 router load from HF safetensors
- router top-4 selection using raw logits, followed by softmax over selected
  top-4 values only
- one-layer post-attention RMSNorm load
- selected-expert exact MXFP4 dequantization for `gate_up_proj` and
  `down_proj`
- GPT-OSS clipped gate/up expert activation on TensorCUDA tensors
- attention plus selected-expert MoE diagnostic block
- artifacted one-token smoke harness

Important limitation:
- This is not the final memory-viable MoE path. It dequantizes only selected
  experts for the tiny diagnostic token batch. The real operating point still
  requires packed MXFP4 expert GEMV/GEMM.

First MoE diagnostic command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_moe_diag_smoke.py --layer 0 --max-tokens 1`

First MoE diagnostic artifact:
- `artifacts/gpt_oss_20b/moe_diag_smoke_20260706_190333.json`

First MoE diagnostic result:
- Failed with `RuntimeError: matmul dtype mismatch`
- Failure point: router matmul in `GptOssMoEDiagnosticTC._route()`
- Cause: diagnostic code promoted the hidden state to FP32 for routing while
  the router weight had been cast to BF16.
- Fix: keep diagnostic router weight and bias in FP32. The router is small, and
  this also keeps routing numerics conservative for the diagnostic path.

Compile/test checks after the fix:
- `env PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile core/gpt_oss20b_tc.py scripts/gpt_oss20b_moe_diag_smoke.py`
- `PYTEST_ADDOPTS='-p no:cacheprovider' pytest tests/test_gpt_oss20b_scaffold.py -q`
- Result: `7 passed, 2 warnings in 0.51s`

Layer 0 MoE diagnostic command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_moe_diag_smoke.py --layer 0 --max-tokens 1`

Layer 0 MoE diagnostic artifact:
- `artifacts/gpt_oss_20b/moe_diag_smoke_20260706_190408.json`

Layer 0 MoE diagnostic result:
- `status = ok`
- `layer_type = sliding_attention`
- `input_ids_shape = [1, 1]`
- `hidden_shape = [1, 1, 2880]`
- `output_shape = [1, 1, 2880]`
- `kv_shapes = [[1, 8, 1, 64], [1, 8, 1, 64]]`
- `unique_experts = [13, 17, 21, 29]`
- `top_indices = [[13, 21, 17, 29]]`
- `top_weights = [[0.4329625666, 0.3586286008, 0.1111120209, 0.0972967297]]`
- `wall_seconds = 3.458`
- GPU before: `NVIDIA GeForce RTX 4070 SUPER, 275, 12282, 37`
- GPU after inside script: `NVIDIA GeForce RTX 4070 SUPER, 497, 12282, 7`

Layer 1 MoE diagnostic command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_moe_diag_smoke.py --layer 1 --max-tokens 1`

Layer 1 MoE diagnostic artifact:
- `artifacts/gpt_oss_20b/moe_diag_smoke_20260706_190431.json`

Layer 1 MoE diagnostic result:
- `status = ok`
- `layer_type = full_attention`
- `input_ids_shape = [1, 1]`
- `hidden_shape = [1, 1, 2880]`
- `output_shape = [1, 1, 2880]`
- `kv_shapes = [[1, 8, 1, 64], [1, 8, 1, 64]]`
- `unique_experts = [12, 15, 25, 26]`
- `top_indices = [[12, 26, 15, 25]]`
- `top_weights = [[0.3335072100, 0.2424832731, 0.2141666114, 0.2098428458]]`
- `wall_seconds = 3.526`
- GPU before: `NVIDIA GeForce RTX 4070 SUPER, 275, 12282, 35`
- GPU after inside script: `NVIDIA GeForce RTX 4070 SUPER, 497, 12282, 7`

Post-run GPU state:
- `NVIDIA GeForce RTX 4070 SUPER, 275, 12282, 0`

Interpretation:
- The TensorCUDA GPT-OSS path now has real HF-safetensor receipts through
  embedding, YARN RoPE, sink-aware attention, post-attention norm, router, and
  selected-expert MoE for both sliding and full attention layer families.
- This is still not a full model loader, not PPL, not generation, and not a
  claim that the exact-dequant MoE path is viable at full context or full layer
  count.

Pending:
- Implement packed MXFP4 expert GEMV/GEMM so GPT-OSS experts can stay packed
  end to end.
- Add lm_head/output path and then run short behavior/PPL checks.
- Only after the standard path is validated, attach APA/GRM and run the real
  context/recall tests.

Action: Added the Project-Tensor packed MXFP4 expert linear kernel and wired it
into the GPT-OSS MoE smoke as an explicit expert mode.

Project-Tensor branch:
- `codex/gpt-oss-mxfp4-kernel`

Project-Tensor commit:
- `e4d39d1 feat: add gpt oss mxfp4 linear kernel`

Project-Tensor implementation:
- Added `tc.mxfp4_linear(x, blocks, scales)`.
- Kernel input:
  - `x = (..., K)` fp32/fp16/bf16
  - `blocks = [out_features, groups, 16]` uint8
  - `scales = [out_features, groups]` uint8
  - `K = groups * 32`
- Dequant rule:
  - GPT-OSS FP4 codebook
  - low/high nibbles
  - E8M0 scale exponent `scale = 2 ** (uint8_scale - 127)`
- Added a decode-shaped GEMV path and a tiled GEMM fallback.
- The op is frozen/inference-only; backward intentionally throws.

Project-Tensor verification:
- Build command:
  `./build.sh 89`
- Build result:
  succeeded with CUDA toolkit `12.6, V12.6.85`
- Test command:
  `PYTHONPATH=/mnt/ForgeRealm/Project-Tensor/tensor_cuda PYTEST_ADDOPTS='-p no:cacheprovider' pytest tensor_cuda/tests/test_mxfp4_linear.py tensor_cuda/tests/test_intn_linear.py -q`
- Test result:
  `13 passed in 0.79s`
- GPU returned to baseline:
  `NVIDIA GeForce RTX 4070 SUPER, 275, 12282`

GraftRepository changes:
- `GptOssMoEDiagnosticTC` now supports:
  - `expert_mode = "dequant"`
  - `expert_mode = "packed_mxfp4"`
- `scripts/gpt_oss20b_moe_diag_smoke.py` now accepts:
  - `--expert-mode dequant`
  - `--expert-mode packed_mxfp4`
- Smoke artifacts now include `output_stats` for compact A/B comparison.

GraftRepository verification:
- Compile command:
  `env PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile core/gpt_oss20b_tc.py scripts/gpt_oss20b_moe_diag_smoke.py tests/test_gpt_oss20b_scaffold.py`
- Test command:
  `PYTEST_ADDOPTS='-p no:cacheprovider' pytest tests/test_gpt_oss20b_scaffold.py -q`
- Test result:
  `7 passed, 2 warnings in 0.51s`

Refactor regression smoke:
- Command:
  `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_moe_diag_smoke.py --layer 0 --max-tokens 1 --expert-mode dequant`
- Artifact:
  `artifacts/gpt_oss_20b/moe_diag_smoke_20260706_191540.json`
- Result:
  - `status = ok`
  - `expert_mode = dequant`
  - `layer_type = sliding_attention`
  - `output_shape = [1, 1, 2880]`
  - `unique_experts = [13, 17, 21, 29]`
  - `wall_seconds = 3.368`
  - `output_stats.sum = 24.8995056152`
  - GPU before/after inside script: `275 MiB -> 497 MiB`

Packed MXFP4 layer 0 smoke:
- Command:
  `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_moe_diag_smoke.py --layer 0 --max-tokens 1 --expert-mode packed_mxfp4`
- Artifact:
  `artifacts/gpt_oss_20b/moe_diag_smoke_20260706_191558.json`
- Result:
  - `status = ok`
  - `expert_mode = packed_mxfp4`
  - `layer_type = sliding_attention`
  - `output_shape = [1, 1, 2880]`
  - `unique_experts = [13, 17, 21, 29]`
  - `top_indices = [[13, 21, 17, 29]]`
  - `wall_seconds = 2.312`
  - `output_stats.sum = 24.8838806152`
  - GPU before/after inside script: `275 MiB -> 497 MiB`

Packed MXFP4 layer 1 smoke:
- Command:
  `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_moe_diag_smoke.py --layer 1 --max-tokens 1 --expert-mode packed_mxfp4`
- Artifact:
  `artifacts/gpt_oss_20b/moe_diag_smoke_20260706_191614.json`
- Result:
  - `status = ok`
  - `expert_mode = packed_mxfp4`
  - `layer_type = full_attention`
  - `output_shape = [1, 1, 2880]`
  - `unique_experts = [12, 15, 25, 26]`
  - `top_indices = [[12, 26, 15, 25]]`
  - `wall_seconds = 2.367`
  - `output_stats.sum = -134.4730834961`
  - GPU before/after inside script: `275 MiB -> 497 MiB`

Packed-vs-dequant interpretation:
- Layer 0 used the same router selection in both modes.
- Output stats are very close but not bit-identical. This is expected because
  the older diagnostic path dequantizes on CPU and then casts the dequantized
  matrix to BF16 before TensorCUDA matmul, while the packed kernel decodes FP4
  values inside the kernel and accumulates directly.
- The packed path is the first path that avoids materializing selected expert
  `[K, N]` dequantized matrices.

Remaining limitation:
- The smoke still uploads selected expert packed blocks from CPU for the tiny
  diagnostic call. A full loader must make expert blocks resident in packed
  form and dispatch selected experts without CPU re-upload.
- This is still not a full model loader, not lm_head behavior, not PPL, and not
  APA/GRM evidence.

Pending:
- Build the resident packed-expert loader path.
- Add lm_head/output path and short generation/PPL checks.
- Then attach APA/GRM and run real context/recall tests.

Action: Added resident one-layer packed-expert dispatch for GPT-OSS MoE.

Reason:
- The previous `packed_mxfp4` smoke proved the kernel but still uploaded the
  selected expert blocks from CPU for each tiny diagnostic call.
- The next production-facing gate was to keep a layer's packed expert tensors
  resident and dispatch selected experts from that resident packed tensor.

First resident attempt:
- Command:
  `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_moe_diag_smoke.py --layer 0 --max-tokens 1 --expert-mode resident_packed_mxfp4`
- Artifact:
  `artifacts/gpt_oss_20b/moe_diag_smoke_20260706_191920.json`
- Result:
  failed with `RuntimeError: op supports float32/float16/bfloat16 only`
- Cause:
  TensorCUDA `slice()` does not support uint8 tensors, so Python-side slicing of
  resident `[experts, out_features, groups, 16]` packed blocks failed.

Fix:
- Added a second Project-Tensor kernel API:
  `tc.mxfp4_linear_expert(x, blocks4d, scales3d, expert_idx)`
- This selects the expert by pointer offset inside the TensorCUDA op instead
  of slicing uint8 tensors in Python.

Project-Tensor branch:
- `codex/gpt-oss-mxfp4-kernel`

Project-Tensor commit:
- `7cfd55e feat: add mxfp4 resident expert selector`

Project-Tensor verification:
- Build command:
  `./build.sh 89`
- Test command:
  `PYTHONPATH=/mnt/ForgeRealm/Project-Tensor/tensor_cuda PYTEST_ADDOPTS='-p no:cacheprovider' pytest tensor_cuda/tests/test_mxfp4_linear.py tensor_cuda/tests/test_intn_linear.py -q`
- Test result:
  `14 passed in 0.75s`
- GPU returned to baseline:
  `NVIDIA GeForce RTX 4070 SUPER, 275, 12282`

GraftRepository changes:
- `resident_packed_mxfp4` mode now uploads one layer's packed expert blocks,
  scales, and biases once during block construction.
- Per selected expert, it calls:
  - `tc.mxfp4_linear_expert(xt, gate_up_blocks_tc, gate_up_scales_tc, expert)`
  - `tc.mxfp4_linear_expert(act, down_blocks_tc, down_scales_tc, expert)`
- Only FP32 bias tensors are sliced in Python.

GraftRepository verification:
- Compile command:
  `env PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile core/gpt_oss20b_tc.py scripts/gpt_oss20b_moe_diag_smoke.py`
- Test command:
  `PYTEST_ADDOPTS='-p no:cacheprovider' pytest tests/test_gpt_oss20b_scaffold.py -q`
- Test result:
  `7 passed, 2 warnings in 0.50s`

Resident packed MXFP4 layer 0 smoke:
- Command:
  `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_moe_diag_smoke.py --layer 0 --max-tokens 1 --expert-mode resident_packed_mxfp4`
- Artifact:
  `artifacts/gpt_oss_20b/moe_diag_smoke_20260706_192407.json`
- Result:
  - `status = ok`
  - `expert_mode = resident_packed_mxfp4`
  - `layer_type = sliding_attention`
  - `output_shape = [1, 1, 2880]`
  - `unique_experts = [13, 17, 21, 29]`
  - `top_indices = [[13, 21, 17, 29]]`
  - `wall_seconds = 2.387`
  - `output_stats.sum = 24.8838806152`
  - GPU before/after inside script: `275 MiB -> 905 MiB`

Resident packed MXFP4 layer 1 smoke:
- Command:
  `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_moe_diag_smoke.py --layer 1 --max-tokens 1 --expert-mode resident_packed_mxfp4`
- Artifact:
  `artifacts/gpt_oss_20b/moe_diag_smoke_20260706_192428.json`
- Result:
  - `status = ok`
  - `expert_mode = resident_packed_mxfp4`
  - `layer_type = full_attention`
  - `output_shape = [1, 1, 2880]`
  - `unique_experts = [12, 15, 25, 26]`
  - `top_indices = [[12, 26, 15, 25]]`
  - `wall_seconds = 3.072`
  - `output_stats.sum = -134.4730834961`
  - GPU before/after inside script: `275 MiB -> 905 MiB`

Post-run GPU state:
- `NVIDIA GeForce RTX 4070 SUPER, 275, 12282`

Interpretation:
- Resident one-layer packed expert dispatch now works for both GPT-OSS layer
  families.
- The layer 0 resident output stats match the CPU-selected packed path,
  confirming that expert-offset dispatch did not change math.
- The memory increase from `497 MiB` to `905 MiB` inside the script is expected:
  the resident path holds the whole layer's packed expert blocks/scales on GPU
  instead of only transient selected expert tensors.

Remaining limitation:
- This is one-layer residency, not full-model residency.
- The full expert body is about `9.46 GiB` packed, before non-expert weights,
  activations, KV, APA state, lm_head, or allocator margin.
- The next full-loader design needs explicit tiering and/or non-expert
  compression, not a blind all-GPU resident load.

Pending:
- Add a standard output path/lm_head route for short behavior and PPL tests.
- Decide the full-loader residency policy for experts and non-experts.
- Then attach APA/GRM and run real context/recall tests.
