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

Action: Added and ran a streamed full-stack GPT-OSS TensorCUDA forward smoke.

File added:
- `scripts/gpt_oss20b_stream_forward_smoke.py`

Purpose:
- Stream all decoder layers one at a time.
- Keep only one layer's packed experts resident on GPU at a time.
- Use `resident_packed_mxfp4` MoE dispatch.
- Attach final RMSNorm.
- Optionally attach a TensorCUDA INT4-quantized `lm_head` for a next-token
  top-k receipt.

Important limitation:
- This is still a smoke, not PPL and not generation-quality evidence.
- The lm_head is quantized locally through TensorCUDA `QuantLinearTC`, so this
  is the custom TensorCUDA quantized path, not an official BF16 reference path.
- The prompt is plain tokenizer input, not a Harmony chat template.
- APA and GRM are not attached in this streamed smoke.

Compile check:
- `env PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/gpt_oss20b_stream_forward_smoke.py core/gpt_oss20b_tc.py`
- Result: passed.

Two-layer shakeout:
- Command:
  `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_stream_forward_smoke.py --max-layers 2 --max-tokens 1 --expert-mode resident_packed_mxfp4 --skip-lm-head`
- Artifact:
  `artifacts/gpt_oss_20b/stream_forward_20260706_192828.json`
- Result:
  - `status = ok`
  - `completed_layers = 2`
  - `input_ids_shape = [1, 1]`
  - `wall_seconds = 2.947`
  - layer resident GPU footprint inside script: `905 MiB`

Full 24-layer streamed stack without lm_head:
- Command:
  `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_stream_forward_smoke.py --max-tokens 1 --expert-mode resident_packed_mxfp4 --skip-lm-head`
- Artifact:
  `artifacts/gpt_oss_20b/stream_forward_20260706_192857.json`
- Result:
  - `status = ok`
  - `completed_layers = 24`
  - `input_ids_shape = [1, 1]`
  - `final_hidden_shape = [1, 1, 2880]`
  - `final_hidden_stats.sum = 54.2272949219`
  - `wall_seconds = 26.430`
  - GPU before: `275 MiB`
  - GPU after each resident layer: about `905 MiB`
  - GPU before lm_head stage, with final hidden still live: `485 MiB`

Full 24-layer streamed stack with lm_head, one-token cap:
- Command:
  `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_stream_forward_smoke.py --max-tokens 1 --expert-mode resident_packed_mxfp4 --top-k 8`
- Artifact:
  `artifacts/gpt_oss_20b/stream_forward_20260706_192950.json`
- Result:
  - `status = ok`
  - `completed_layers = 24`
  - `input_ids_shape = [1, 1]`
  - `wall_seconds = 20.166`
  - `gpu_before_lm_head = 485 MiB`
  - `gpu_after = 799 MiB`
  - top tokens:
    - rank 0: token `3490`, text ` code`, logit `9.5`
    - rank 1: token `5787`, text ` article`, logit `8.625`
    - rank 2: token `2700`, text `` ` ``, logit `8.3125`
- Interpretation:
  one-token prompt cap is a stack/output projection smoke only, not a useful
  language behavior check.

Full 24-layer streamed stack with lm_head, five-token prompt:
- Command:
  `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_stream_forward_smoke.py --prompt 'The capital of France is' --max-tokens 8 --expert-mode resident_packed_mxfp4 --top-k 8`
- Artifact:
  `artifacts/gpt_oss_20b/stream_forward_20260706_193035.json`
- Result:
  - `status = ok`
  - `completed_layers = 24`
  - `input_ids_shape = [1, 5]`
  - `input_ids = [[976, 9029, 328, 10128, 382]]`
  - `final_hidden_shape = [1, 5, 2880]`
  - `final_hidden_stats.sum = -71.9653320312`
  - `wall_seconds = 21.007`
  - `gpu_before_lm_head = 483 MiB`
  - `gpu_after = 797 MiB`
  - top tokens:
    - rank 0: token `12650`, text ` Paris`, logit `15.875`
    - rank 1: token `25`, text `:`, logit `12.5`
    - rank 2: token `392`, text ` "`, logit `12.1875`
    - rank 3: token `261`, text ` a`, logit `12.0625`
    - rank 4: token `290`, text ` the`, logit `12.0`

Post-run GPU state:
- `NVIDIA GeForce RTX 4070 SUPER, 275, 12282`

Interpretation:
- The streamed TensorCUDA path now exercises all 24 GPT-OSS layers with
  resident packed expert dispatch.
- The output projection path is connected and can produce next-token top-k
  logits.
- The short plain-text sanity prompt predicts ` Paris` as the top next token,
  which is a useful behavior smoke.
- This still does not establish PPL, long-context behavior, Harmony chat
  behavior, APA behavior, or GRM behavior.

Pending:
- Add a real PPL harness for this streamed path.
- Add a short greedy decode loop using KV cache or repeated streamed forwards.
- Add Harmony-formatted prompt tests.
- Only then attach APA/GRM and run real context/recall tests.

Action: Added and ran streamed-path next-token PPL smoke.

Script change:
- `scripts/gpt_oss20b_stream_forward_smoke.py` now accepts `--score-ppl`.
- In PPL mode:
  - tokenized prompt is capped by `--max-tokens`
  - inputs are all tokens except the final token
  - targets are the next-token shift
  - full streamed forward runs over the input tokens
  - quantized TensorCUDA lm_head scores every input position
  - mean NLL and PPL are computed on CPU from the logits

Compile check:
- `env PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/gpt_oss20b_stream_forward_smoke.py`
- Result: passed.

PPL smoke command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_stream_forward_smoke.py --prompt 'The capital of France is Paris.' --max-tokens 8 --expert-mode resident_packed_mxfp4 --top-k 8 --score-ppl`

PPL smoke artifact:
- `artifacts/gpt_oss_20b/stream_forward_20260706_193416.json`

PPL smoke result:
- `status = ok`
- `completed_layers = 24`
- `input_ids_shape = [1, 6]`
- full tokenized ids:
  `[[976, 9029, 328, 10128, 382, 12650, 13]]`
- scored target texts:
  `[" capital", " of", " France", " is", " Paris", "."]`
- `token_count = 6`
- `mean_nll = 2.9856215566`
- `ppl = 19.7988044911`
- `wall_seconds = 18.556`
- `gpu_before_lm_head = 485 MiB`
- `gpu_after = 803 MiB`
- post-run GPU returned to baseline:
  `NVIDIA GeForce RTX 4070 SUPER, 275, 12282`

Final-position top tokens from the PPL run:
- rank 0: token `3692`, text `."`, logit `15.875`
- rank 1: token `14396`, text `."\n`, logit `15.375`
- rank 2: token `13`, text `.`, logit `15.1875`
- rank 3: token `6635`, text `."\n\n`, logit `14.5625`
- rank 4: token `364`, text `.\n\n`, logit `14.4375`

Interpretation:
- This is the first streamed-path PPL-style receipt.
- It is a tiny six-target smoke, not a benchmark and not comparable to standard
  corpus PPL.
- The result is still useful because it proves the full streamed path can score
  multiple shifted targets through all 24 layers and the lm_head.

Pending:
- Run a real fixed corpus PPL suite with enough tokens to compare settings.
- Add a short greedy decode loop.
- Add Harmony-formatted prompt tests.
- Then attach APA/GRM and run real context/recall tests.

Action: Added and ran a short streamed greedy decode smoke.

File added:
- `scripts/gpt_oss20b_stream_greedy_smoke.py`

Method:
- The harness calls `scripts/gpt_oss20b_stream_forward_smoke.py` once per
  generated token.
- It appends the rank-0 decoded token text to the prompt after each step.
- It does not use KV cache.
- It is intentionally slow and receipt-oriented.

Compile check:
- `env PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/gpt_oss20b_stream_greedy_smoke.py`
- Result: passed.

Greedy smoke command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_stream_greedy_smoke.py --prompt 'The capital of France is' --steps 2 --max-tokens 16 --top-k 8 --expert-mode resident_packed_mxfp4`

Greedy smoke artifact:
- `artifacts/gpt_oss_20b/stream_greedy_20260706_193701.json`

Greedy smoke result:
- `status = ok`
- `wall_seconds = 46.849`
- step 0:
  - child artifact:
    `artifacts/gpt_oss_20b/stream_greedy_20260706_193701_steps/step_00.json`
  - top token: token `12650`, text ` Paris`, logit `15.875`
  - step wall: `24.628s`
- step 1:
  - child artifact:
    `artifacts/gpt_oss_20b/stream_greedy_20260706_193701_steps/step_01.json`
  - top token: token `3692`, text `."`, logit `15.875`
  - step wall: `22.220s`
- final text:
  `The capital of France is Paris."`
- post-run GPU returned to baseline:
  `NVIDIA GeForce RTX 4070 SUPER, 275, 12282`

Interpretation:
- The streamed TensorCUDA path can be driven in a greedy loop.
- The first generated token is the expected factual continuation, ` Paris`.
- The second token includes an extra quote, so this is not polished generation
  behavior.
- This still does not use KV cache and does not prove interactive decode speed.

Pending:
- Add Harmony-formatted prompt tests.
- Add KV-cache/reuse decode instead of repeated full streamed forwards.
- Run real PPL and recall/context tests after APA/GRM attachment.

Action: Added and ran Harmony-formatted streamed prompt smoke.

Script change:
- `scripts/gpt_oss20b_stream_forward_smoke.py` now accepts
  `--use-chat-template`.
- The artifact stores:
  - original user prompt
  - rendered chat-template prompt
  - raw token count
  - whether the input was truncated by `--max-tokens`

Compile check:
- `env PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/gpt_oss20b_stream_forward_smoke.py`
- Result: passed.

First Harmony smoke command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_stream_forward_smoke.py --prompt 'What is the capital of France? Answer with one word.' --use-chat-template --max-tokens 64 --expert-mode resident_packed_mxfp4 --top-k 8`

First Harmony smoke artifact:
- `artifacts/gpt_oss_20b/stream_forward_20260706_193944.json`

First Harmony smoke result:
- `status = ok`
- rendered prompt length was `79` tokens
- input was capped to `64` tokens
- top tokens were conversational text starts:
  - rank 0: `We`
  - rank 1: `Hey`
  - rank 2: `You`
- Interpretation:
  this was a bad behavior smoke because the cap truncated the rendered Harmony
  prompt before the full assistant prefix was present.

Second Harmony smoke command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_stream_forward_smoke.py --prompt 'What is the capital of France? Answer with one word.' --use-chat-template --max-tokens 128 --expert-mode resident_packed_mxfp4 --top-k 8`

Second Harmony smoke artifact:
- `artifacts/gpt_oss_20b/stream_forward_20260706_194041.json`

Second Harmony smoke result:
- `status = ok`
- `input_ids_shape = [1, 79]`
- full rendered prompt fit under the cap
- `wall_seconds = 20.041`
- top tokens:
  - rank 0: token `200005`, text `<|channel|>`, logit `42.5`
  - rank 1: token `220`, text space, logit `16.875`
  - rank 2: token `200003`, text `<|constrain|>`, logit `15.8125`
- post-run GPU returned to baseline:
  `NVIDIA GeForce RTX 4070 SUPER, 275, 12282`

Interpretation:
- The streamed TensorCUDA path can run a Harmony-rendered prompt through all 24
  layers.
- With the full template included, the top next token is the expected protocol
  transition token `<|channel|>`, not content text yet.
- A content-answer Harmony test needs a greedy protocol decode that emits
  channel/control tokens and then content tokens.

Pending:
- Add Harmony-aware greedy decode, not just single-token top-k.
- Add KV-cache/reuse decode instead of repeated full streamed forwards.
- Run APA/GRM context and recall tests only after the standard streamed path is
  stable.

Action: Added sink-aware APA support for GPT-OSS attention and ran streamed APA
smokes.

Reason:
- GPT-OSS attention has learned sink logits.
- The sink logit participates in the softmax denominator but contributes no
  value vector.
- Existing APA kernels did not account for that denominator, so directly
  enabling generic APA would silently score the wrong attention distribution.

Project-Tensor branch:
- `codex/gpt-oss-mxfp4-kernel`

Project-Tensor commit:
- `c9584df feat: add sink-aware apa blend softmax`

Project-Tensor implementation:
- Added `tc.apa_blend_softmax_sink(bulk, rank, sinks, zthr)`.
- Inputs:
  - `bulk = [B, H, L, S]`
  - `rank = [B, H, L, S]`
  - `sinks = [H]`
- Behavior:
  - selection threshold is computed over valid key columns only
  - blended key scores use APA bulk/rank selection
  - sink logit is included in the softmax denominator
  - returned weights have shape `[B, H, L, S]`; no sink column is returned

Project-Tensor verification:
- Build command:
  `./build.sh 89`
- Test command:
  `PYTHONPATH=/mnt/ForgeRealm/Project-Tensor/tensor_cuda PYTEST_ADDOPTS='-p no:cacheprovider' pytest tensor_cuda/tests/test_apa_selective.py tensor_cuda/tests/test_apa_value_dim.py tensor_cuda/tests/test_mxfp4_linear.py -q`
- Test result:
  `16 passed in 1.38s`

GraftRepository implementation:
- Added `sink_apa_blend_attention_tc(...)`.
- `GptOssAttentionTC` now has:
  - `attention_mode = "standard" | "apa_selective"`
  - `refine_percentile`
  - `bulk_bits`
  - `attn_block`
- In `apa_selective` mode:
  - exact K/V remain the standard GPT-OSS projected tensors
  - bulk K is generated with TensorCUDA APA quantization
  - GPT-OSS attention masks are applied to both bulk and exact scores
  - `tc.apa_blend_softmax_sink` handles selection and sink denominator

GraftRepository verification:
- Compile command:
  `env PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile core/gpt_oss20b_tc.py tests/test_gpt_oss20b_scaffold.py`
- Test command:
  `PYTEST_ADDOPTS='-p no:cacheprovider' pytest tests/test_gpt_oss20b_scaffold.py -q`
- Test result:
  `8 passed, 2 warnings in 0.55s`
- New focused test:
  `test_sink_apa_blend_attention_matches_numpy_reference`

Stream harness change:
- `scripts/gpt_oss20b_stream_forward_smoke.py` now accepts:
  - `--attention-mode standard|apa_selective`
  - `--refine-percentile`
  - `--bulk-bits`

Two-layer APA shakeout:
- Command:
  `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_stream_forward_smoke.py --prompt 'The capital of France is' --max-layers 2 --max-tokens 8 --attention-mode apa_selective --refine-percentile 0.15 --bulk-bits 8 --expert-mode resident_packed_mxfp4 --skip-lm-head`
- Artifact:
  `artifacts/gpt_oss_20b/stream_forward_20260706_195250.json`
- Result:
  - `status = ok`
  - `completed_layers = 2`
  - `input_ids_shape = [1, 5]`
  - `attention_mode = apa_selective`
  - `refine_percentile = 0.15`
  - `bulk_bits = 8`
  - `wall_seconds = 3.795`

Full streamed APA top-k smoke:
- Command:
  `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_stream_forward_smoke.py --prompt 'The capital of France is' --max-tokens 8 --attention-mode apa_selective --refine-percentile 0.15 --bulk-bits 8 --expert-mode resident_packed_mxfp4 --top-k 8`
- Artifact:
  `artifacts/gpt_oss_20b/stream_forward_20260706_195315.json`
- Result:
  - `status = ok`
  - `completed_layers = 24`
  - `input_ids_shape = [1, 5]`
  - `wall_seconds = 16.815`
  - `gpu_before_lm_head = 490 MiB`
  - `gpu_after = 804 MiB`
  - top tokens:
    - rank 0: token `12650`, text ` Paris`, logit `15.9375`
    - rank 1: token `25`, text `:`, logit `12.5625`
    - rank 2: token `392`, text ` "`, logit `12.1875`

Tiny streamed APA PPL smoke:
- Command:
  `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_stream_forward_smoke.py --prompt 'The capital of France is Paris.' --max-tokens 8 --attention-mode apa_selective --refine-percentile 0.15 --bulk-bits 8 --expert-mode resident_packed_mxfp4 --top-k 8 --score-ppl`
- Artifact:
  `artifacts/gpt_oss_20b/stream_forward_20260706_195400.json`
- Result:
  - `status = ok`
  - `completed_layers = 24`
  - `input_ids_shape = [1, 6]`
  - scored target texts:
    `[" capital", " of", " France", " is", " Paris", "."]`
  - `mean_nll = 2.9740070154`
  - `ppl = 19.5701807106`
  - `wall_seconds = 16.968`
  - top final-position token remains period-like:
    token `3692`, text `."`, logit `15.875`

Comparison to prior standard tiny PPL smoke:
- Prior standard streamed smoke:
  - artifact `artifacts/gpt_oss_20b/stream_forward_20260706_193416.json`
  - `mean_nll = 2.9856215566`
  - `ppl = 19.7988044911`
- New APA streamed smoke:
  - `mean_nll = 2.9740070154`
  - `ppl = 19.5701807106`
- Interpretation:
  This is only a six-target smoke. It proves the APA path is wired and not
  catastrophically broken on the toy sentence. It is not a corpus PPL result.

Post-run GPU state:
- `NVIDIA GeForce RTX 4070 SUPER, 278, 12282`

Remaining limitation:
- The current GPT-OSS APA path uses cuBLAS score matrices plus sink-aware blend.
- It is correct for sink denominator behavior but not the final long-context
  memory path.
- A true context-extension implementation still needs an O(L) or tiled
  sink-aware fused APA attention path, plus GRM graft capture/injection.

Pending:
- Add long-context APA memory probes on real text.
- Add GPT-OSS KV/GRM capture and remount semantics.
- Add context/recall tests after APA and GRM are both attached.

Action: Added the GPT-OSS-20B house-rules execution plan.

Reason:
- The original GPT-OSS APA/GRM plan is already registered as the fixed intent.
- The remaining work now needs a stricter execution document so future results
  cannot blur smokes, PPL, context, and GRM continuity claims.

New document:
- `docs/GPT_OSS_20B_HOUSE_RULES_EXECUTION_PLAN.md`

Registered evidence tiers:
- Tier 0: source and metadata.
- Tier 1: unit and kernel receipts.
- Tier 2: layer and streamed-forward smokes.
- Tier 3: short behavior receipts.
- Tier 4: real-text PPL and memory gates.
- Tier 5: real-token context extension gates.
- Tier 6: GRM cold-KV continuity gates.

Registered remaining phases:
- H0: checkpoint hygiene.
- H1: APA correctness consolidation.
- H2: real-text PPL gate.
- H3: tiled sink-aware APA memory path.
- H4: real-token context ladder.
- H5: GPT-OSS GRM capture and mount.
- H6: cold-KV multi-turn needle.
- H7: existing-model comparison.

Interpretation:
- Current GPT-OSS receipts reach Tier 3 at best.
- No corpus PPL, long-context, APA memory-flattening, GRM, or cold-KV recall
  claim is allowed until the corresponding registered gate runs.

Action: Completed H1 APA correctness consolidation.

Reason:
- The previous APA artifacts recorded `attention_mode = apa_selective`, but did
  not prove which GPT-OSS layers actually used the APA path.
- GPT-OSS alternates sliding and full attention; the registered plan targets
  full-attention layers first and keeps bounded sliding-window layers on the
  simpler standard path unless explicitly overridden.

Implementation:
- Added `resolve_gpt_oss_attention_mode(...)`.
- Added `--apa-layer-scope full|all` to
  `scripts/gpt_oss20b_stream_forward_smoke.py`.
- Default APA scope is `full`.
- `--attention-mode apa_selective --apa-layer-scope full` now routes:
  - full-attention layers through `apa_selective`
  - sliding-window layers through `standard`
- The harness now fails early if `apa_selective` is requested without
  `tc.apa_blend_softmax_sink`.
- Artifacts now record:
  - `sink_aware_apa_available`
  - `attention_audit`
  - `attention_layers_planned_apa`
  - `attention_layers_planned_standard`
  - `attention_layers_used_apa`
  - `attention_layers_used_standard`
  - per-layer requested/effective attention mode and skip reason

Focused verification:
- Compile command:
  `env PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile core/gpt_oss20b_tc.py scripts/gpt_oss20b_stream_forward_smoke.py tests/test_gpt_oss20b_scaffold.py`
- Diff hygiene:
  `git diff --check`
- Test command:
  `PYTEST_ADDOPTS='-p no:cacheprovider' pytest tests/test_gpt_oss20b_scaffold.py -q`
- Test result:
  `9 passed, 2 warnings in 0.51s`
- New focused test:
  `test_resolve_gpt_oss_attention_mode_scopes_apa_to_full_layers`

H1 standard comparison command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_stream_forward_smoke.py --prompt 'The capital of France is' --max-tokens 8 --attention-mode standard --expert-mode resident_packed_mxfp4 --top-k 8 --output artifacts/gpt_oss_20b/h1_standard_stream_forward.json`

H1 standard artifact:
- `artifacts/gpt_oss_20b/h1_standard_stream_forward.json`

H1 standard result:
- `status = ok`
- `completed_layers = 24`
- `attention_layers_used_apa = []`
- `attention_layers_used_standard` length = `24`
- top token:
  - token `12650`, text ` Paris`, logit `15.875`
- `wall_seconds = 16.351192983041983`
- `gpu_after = NVIDIA GeForce RTX 4070 SUPER, 796, 12282, 7`

H1 APA r0.15 full-scope comparison command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_stream_forward_smoke.py --prompt 'The capital of France is' --max-tokens 8 --attention-mode apa_selective --apa-layer-scope full --refine-percentile 0.15 --bulk-bits 8 --expert-mode resident_packed_mxfp4 --top-k 8 --output artifacts/gpt_oss_20b/h1_apa_r015_full_scope_stream_forward.json`

H1 APA r0.15 full-scope artifact:
- `artifacts/gpt_oss_20b/h1_apa_r015_full_scope_stream_forward.json`

H1 APA r0.15 full-scope result:
- `status = ok`
- `completed_layers = 24`
- `attention_layers_used_apa = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]`
- `attention_layers_used_standard = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]`
- top token:
  - token `12650`, text ` Paris`, logit `15.8125`
- `wall_seconds = 17.014232303015888`
- `gpu_after = NVIDIA GeForce RTX 4070 SUPER, 798, 12282, 8`

Post-run GPU state:
- `NVIDIA GeForce RTX 4070 SUPER, 274, 12282, 36`

Interpretation:
- H1 passes.
- The artifacts now prove whether APA actually ran on each tested layer.
- The full-scope APA r0.15 path still ranks ` Paris` first on the short prompt.
- This remains Tier 3 evidence only; it is not corpus PPL, context extension,
  GRM, or cold-KV recall evidence.

Next:
- H2 real-text PPL gate.

Action: Added and ran the first GPT-OSS H2 real-text PPL gate.

Reason:
- The prior PPL receipts were six-target toy smokes.
- H2 requires fixed real text, scored token counts, memory reporting, and
  standard-vs-APA comparison.

Implementation:
- Added `scripts/gpt_oss20b_realtext_ppl_gate.py`.
- The runner:
  - tokenizes fixed real text into windows
  - invokes `scripts/gpt_oss20b_stream_forward_smoke.py --score-ppl` per window
  - records one artifact per window/setting
  - aggregates weighted mean NLL and PPL
  - parses max observed GPU memory from the per-layer artifact fields
- Added pure tests for:
  - GPU memory parsing from `nvidia-smi` rows
  - max memory extraction from stream artifacts
  - weighted NLL/PPL aggregation
  - setting parsing
- Updated stream-forward artifact notes so `--score-ppl` is described as a
  teacher-forced window receipt while still not implying generation, context,
  or GRM evidence.

Verification:
- Compile command:
  `env PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/gpt_oss20b_realtext_ppl_gate.py scripts/gpt_oss20b_stream_forward_smoke.py tests/test_gpt_oss20b_realtext_ppl_gate.py`
- Diff hygiene:
  `git diff --check`
- Pure test command:
  `PYTEST_ADDOPTS='-p no:cacheprovider' pytest tests/test_gpt_oss20b_realtext_ppl_gate.py -q`
- Pure test result:
  `4 passed in 0.06s`

H2 small-gate command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_realtext_ppl_gate.py --corpus docs/GRM_Primer.md --window-tokens 64 --n-windows 2 --stride-tokens 64 --settings standard,apa_r0.15,apa_r0.10 --apa-layer-scope full --bulk-bits 8 --expert-mode resident_packed_mxfp4 --output artifacts/gpt_oss_20b/h2_realtext_ppl_small.json`

H2 small-gate artifact:
- Aggregate:
  `artifacts/gpt_oss_20b/h2_realtext_ppl_small.json`
- Per-window sub-artifacts:
  - `artifacts/gpt_oss_20b/h2_realtext_ppl_small/standard_w00.json`
  - `artifacts/gpt_oss_20b/h2_realtext_ppl_small/standard_w01.json`
  - `artifacts/gpt_oss_20b/h2_realtext_ppl_small/apa_r0.15_w00.json`
  - `artifacts/gpt_oss_20b/h2_realtext_ppl_small/apa_r0.15_w01.json`
  - `artifacts/gpt_oss_20b/h2_realtext_ppl_small/apa_r0.10_w00.json`
  - `artifacts/gpt_oss_20b/h2_realtext_ppl_small/apa_r0.10_w01.json`

Corpus:
- Path:
  `/mnt/ForgeRealm/GraftRepository/docs/GRM_Primer.md`
- sha256:
  `992c6b602d64e19100e941d10626fe44b08bd5c051bd89f6e95b344ddd796498`
- Windows:
  - window 0: token start `0`, token count `64`
  - window 1: token start `64`, token count `64`

H2 small-gate result:
- Aggregate status:
  `ok`
- Standard:
  - windows ok: `2 / 2`
  - scored tokens: `126`
  - mean NLL: `3.4090559656677732`
  - PPL: `30.236686311708425`
  - max observed memory: `908 MiB`
- APA r0.15:
  - windows ok: `2 / 2`
  - scored tokens: `126`
  - mean NLL: `3.3852607499288823`
  - PPL: `29.525690534212117`
  - max observed memory: `910 MiB`
  - APA layers used:
    `[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]`
- APA r0.10:
  - windows ok: `2 / 2`
  - scored tokens: `126`
  - mean NLL: `3.3960288977553046`
  - PPL: `29.84534549173959`
  - max observed memory: `910 MiB`
  - APA layers used:
    `[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]`

Post-run GPU state:
- `NVIDIA GeForce RTX 4070 SUPER, 274, 12282, 35`

Interpretation:
- This is the first GPT-OSS real-text PPL receipt on the streamed TensorCUDA
  path.
- APA r0.15 and r0.10 did not collapse on this small real-text gate.
- Both APA settings were slightly better than standard on this small sample, but
  the sample is only `126` scored tokens per setting; this should be treated as
  a smoke-scale real-text gate, not a broad benchmark.

Next:
- Scale H2 to more and/or larger real-text windows after the current checkpoint
  is committed.

Action: Ran the broader GPT-OSS H2 real-text PPL gate.

Reason:
- The first H2 gate scored only `126` tokens per setting.
- A larger receipt is needed before treating APA quality as more than a toy
  smoke.

Command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_realtext_ppl_gate.py --corpus docs/GRM_Primer.md --window-tokens 128 --n-windows 4 --stride-tokens 128 --settings standard,apa_r0.15,apa_r0.10 --apa-layer-scope full --bulk-bits 8 --expert-mode resident_packed_mxfp4 --output artifacts/gpt_oss_20b/h2_realtext_ppl_128x4.json`

Artifact:
- Aggregate:
  `artifacts/gpt_oss_20b/h2_realtext_ppl_128x4.json`
- Per-window sub-artifacts:
  `artifacts/gpt_oss_20b/h2_realtext_ppl_128x4/`
- Sub-artifact count:
  `12`

Corpus:
- Path:
  `/mnt/ForgeRealm/GraftRepository/docs/GRM_Primer.md`
- sha256:
  `992c6b602d64e19100e941d10626fe44b08bd5c051bd89f6e95b344ddd796498`
- Windows:
  - window 0: token start `0`, token count `128`
  - window 1: token start `128`, token count `128`
  - window 2: token start `256`, token count `128`
  - window 3: token start `384`, token count `128`

Result:
- Aggregate status:
  `ok`
- Standard:
  - windows ok: `4 / 4`
  - scored tokens: `508`
  - mean NLL: `3.2033194321851637`
  - PPL: `24.61409957453896`
  - max observed memory: `909 MiB`
  - APA layers used: `[]`
- APA r0.15:
  - windows ok: `4 / 4`
  - scored tokens: `508`
  - mean NLL: `3.1885441383478828`
  - PPL: `24.253092580566708`
  - max observed memory: `909 MiB`
  - APA layers used:
    `[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]`
- APA r0.10:
  - windows ok: `4 / 4`
  - scored tokens: `508`
  - mean NLL: `3.19293471357478`
  - PPL: `24.35981171578557`
  - max observed memory: `909 MiB`
  - APA layers used:
    `[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]`

Post-run GPU state:
- `NVIDIA GeForce RTX 4070 SUPER, 276, 12282, 39`

Interpretation:
- The broader H2 gate passes.
- APA r0.15 and r0.10 did not collapse versus standard across `508` scored
  real-text tokens per setting.
- On this corpus slice, APA r0.15 was `0.3610` PPL lower than standard and APA
  r0.10 was `0.2543` PPL lower than standard.
- Max observed memory was effectively identical across settings at `909 MiB`.
- This still does not prove long-context memory flattening, because the current
  GPT-OSS APA implementation uses the score-matrix smoke path. H3 remains
  required for context-extension evidence.

Next:
- H3 tiled sink-aware APA memory path.

Action: Implemented and wired the H3 fused sink-aware APA memory path.

Reason:
- The previous GPT-OSS APA path used cuBLAS score matrices plus
  `apa_blend_softmax_sink`.
- That path was correct for GPT-OSS sink normalization but was still a
  score-matrix smoke path.
- H3 requires a path that does not materialize full `[B, H, L, S]` score
  matrices for the full-attention layers.

Project-Tensor branch:
- `codex/gpt-oss-mxfp4-kernel`

Project-Tensor commit:
- `1c2e8b0 feat: add fused sink-aware apa attention`

Project-Tensor implementation:
- Added `tc.apa_selective_attention_sink(q, k, kq, v, sinks, scale, zthr, is_causal)`.
- The kernel mirrors the existing fused `tc.apa_selective_attention` path:
  - no materialized score matrix
  - GQA-aware KV head indexing
  - bottom-right causal masking for cache/continuation layouts
  - `VD != D` support
- GPT-OSS learned sinks are folded into the online softmax denominator with zero
  value contribution.

Project-Tensor verification:
- Build command:
  `cd /mnt/ForgeRealm/Project-Tensor/tensor_cuda && ./build.sh 89`
- Test command:
  `PYTHONPATH=/mnt/ForgeRealm/Project-Tensor/tensor_cuda PYTEST_ADDOPTS='-p no:cacheprovider' pytest tensor_cuda/tests/test_apa_selective.py tensor_cuda/tests/test_apa_value_dim.py -q`
- Test result:
  `14 passed in 1.21s`
- New tests:
  - `test_selective_sink_matches_reference_gqa_rectangular_value_dim`
  - `test_selective_sink_large_sink_reduces_value_mass`

GraftRepository implementation:
- `GptOssAttentionTC` now prefers `tc.apa_selective_attention_sink` for
  full-attention GPT-OSS APA layers.
- Sliding-window layers stay on standard attention under the default
  `--apa-layer-scope full`.
- The prior sink-aware blend path remains as fallback and for explicit
  non-default experiments.
- Stream artifacts now record per-layer `attention_backend`, with values such
  as:
  - `standard_sink`
  - `apa_selective_sink_fused`
  - `apa_selective_sink_blend`
- Stream artifacts also record `fused_sink_apa_available`.

GraftRepository verification:
- Compile command:
  `env PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile core/gpt_oss20b_tc.py scripts/gpt_oss20b_stream_forward_smoke.py`
- Diff hygiene:
  `git diff --check`
- Test command:
  `PYTEST_ADDOPTS='-p no:cacheprovider' pytest tests/test_gpt_oss20b_scaffold.py -q`
- Test result:
  `9 passed, 2 warnings in 0.50s`

H3 two-layer fused APA smoke:
- Command:
  `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_stream_forward_smoke.py --prompt 'The capital of France is' --max-layers 2 --max-tokens 8 --attention-mode apa_selective --apa-layer-scope full --refine-percentile 0.15 --bulk-bits 8 --expert-mode resident_packed_mxfp4 --skip-lm-head --output artifacts/gpt_oss_20b/h3_fused_apa_two_layer.json`
- Artifact:
  `artifacts/gpt_oss_20b/h3_fused_apa_two_layer.json`
- Result:
  - `status = ok`
  - `completed_layers = 2`
  - `attention_layers_used_apa = [1]`
  - layer 0 backend: `standard_sink`
  - layer 1 backend: `apa_selective_sink_fused`

H3 full fused APA top-k smoke:
- Command:
  `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_stream_forward_smoke.py --prompt 'The capital of France is' --max-tokens 8 --attention-mode apa_selective --apa-layer-scope full --refine-percentile 0.15 --bulk-bits 8 --expert-mode resident_packed_mxfp4 --top-k 8 --output artifacts/gpt_oss_20b/h3_fused_apa_full_topk.json`
- Artifact:
  `artifacts/gpt_oss_20b/h3_fused_apa_full_topk.json`
- Result:
  - `status = ok`
  - `completed_layers = 24`
  - backend counts:
    - `standard_sink = 12`
    - `apa_selective_sink_fused = 12`
  - `attention_layers_used_apa = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]`
  - top token:
    - token `12650`, text ` Paris`, logit `15.875`
  - `wall_seconds = 16.763880875019822`
  - `gpu_after = NVIDIA GeForce RTX 4070 SUPER, 797, 12282, 8`

H3 fused APA PPL smoke:
- Command:
  `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_realtext_ppl_gate.py --corpus docs/GRM_Primer.md --window-tokens 64 --n-windows 2 --stride-tokens 64 --settings apa_r0.15 --apa-layer-scope full --bulk-bits 8 --expert-mode resident_packed_mxfp4 --output artifacts/gpt_oss_20b/h3_fused_apa_ppl_smoke.json`
- Artifact:
  `artifacts/gpt_oss_20b/h3_fused_apa_ppl_smoke.json`
- Result:
  - status: `ok`
  - windows ok: `2 / 2`
  - scored tokens: `126`
  - mean NLL: `3.3905333574251033`
  - PPL: `29.68177904657413`
  - max observed memory: `907 MiB`
  - sub-artifact backend counts:
    - `standard_sink = 12`
    - `apa_selective_sink_fused = 12`

Post-run GPU state:
- `NVIDIA GeForce RTX 4070 SUPER, 276, 12282, 35`

Interpretation:
- H3 implementation slice passes.
- GPT-OSS full-attention APA no longer needs the score-matrix blend path when
  the fused sink-aware primitive is available.
- This is still not a context-extension result. H4 must run a real-token
  context ladder to establish usable context and OOM boundaries.

Next:
- H4 real-token context ladder.

Action: Added and ran the first H4 real-token context ladder through 16K.

Reason:
- H4 requires real-token fill and a pass/fail/OOM ladder before any
  context-extension claim is allowed.
- The first small ladder also exposed that GPT-OSS sliding-window layers were
  mask-bounded but not memory-bounded in the TensorCUDA path.

Implementation:
- Added `--prompt-file` support to
  `scripts/gpt_oss20b_stream_forward_smoke.py`.
- Added `scripts/gpt_oss20b_context_ladder.py`.
- The H4 runner:
  - builds prompt files from real tokenized docs
  - runs one process per setting/context rung
  - records command, prompt file, artifact path, return code, classification,
    actual token count, raw token count, completed layers, backend counts,
    artifact memory, sampled peak memory, stdout/stderr tails, and wall time
  - stops a setting at first failure/OOM
- Added parent-process GPU memory sampling around each rung.
- Added chunked GPT-OSS sliding-window sink attention:
  `sliding_sink_attention_tc(...)`.
- Default GPT-OSS standard sliding layers now use
  `standard_sink_sliding_chunked` instead of materializing full `L x S` scores.
- GPT-OSS APA full-attention layers continue to use
  `apa_selective_sink_fused`.

Verification:
- Compile command:
  `env PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile core/gpt_oss20b_tc.py tests/test_gpt_oss20b_scaffold.py scripts/gpt_oss20b_stream_forward_smoke.py scripts/gpt_oss20b_context_ladder.py`
- Diff hygiene:
  `git diff --check`
- Pure ladder tests:
  `PYTEST_ADDOPTS='-p no:cacheprovider' pytest tests/test_gpt_oss20b_context_ladder.py -q`
  - result: `6 passed in 0.04s`
- GPT-OSS scaffold tests:
  `PYTEST_ADDOPTS='-p no:cacheprovider' pytest tests/test_gpt_oss20b_scaffold.py -q`
  - result: `10 passed, 2 warnings in 0.52s`
- New focused scaffold test:
  `test_sliding_sink_attention_matches_full_mask_reference`

Corpus source:
- Directory:
  `/mnt/ForgeRealm/GraftRepository/docs`
- File count:
  `40`
- Directory content hash:
  `96fa1e59a01388ba3d84424904768aa89fa17cf108d33bb6145c08398ec27a09`
- Token count:
  `222843`

Initial H4 smoke command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_context_ladder.py --corpus-dir docs --lengths 128,256,512 --settings standard,apa_r0.15 --apa-layer-scope full --bulk-bits 8 --expert-mode resident_packed_mxfp4 --output artifacts/gpt_oss_20b/h4_context_ladder_smoke.json`

Initial H4 smoke result:
- Standard:
  - pass: `128`, `256`, `512`
  - max artifact memory: `911 MiB`
- APA r0.15:
  - pass: `128`, `256`, `512`
  - max artifact memory: `911 MiB`
- Interpretation:
  - This run happened before chunked sliding attention was added; it was useful
    as a sanity check but not the final H4 memory shape.

Post-sliding-fix H4 smoke command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_context_ladder.py --corpus-dir docs --lengths 128,256,512 --settings standard,apa_r0.15 --apa-layer-scope full --bulk-bits 8 --expert-mode resident_packed_mxfp4 --output artifacts/gpt_oss_20b/h4_context_ladder_chunked_sliding_smoke.json`

Post-sliding-fix H4 smoke result:
- Standard:
  - pass: `128`, `256`, `512`
  - actual token counts match targets
  - max artifact memory: `907`, `909`, `911 MiB`
  - backend counts:
    `standard_sink = 12`, `standard_sink_sliding_chunked = 12`
- APA r0.15:
  - pass: `128`, `256`, `512`
  - actual token counts match targets
  - max artifact memory: `907`, `909`, `911 MiB`
  - backend counts:
    `apa_selective_sink_fused = 12`, `standard_sink_sliding_chunked = 12`

H4 1K/2K command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_context_ladder.py --corpus-dir docs --lengths 1024,2048 --settings standard,apa_r0.15 --apa-layer-scope full --bulk-bits 8 --expert-mode resident_packed_mxfp4 --output artifacts/gpt_oss_20b/h4_context_ladder_1k_2k.json`

H4 1K/2K result:
- Standard:
  - `1024`: pass, actual `1024`, artifact max `913 MiB`, wall `69.48s`
  - `2048`: pass, actual `2048`, artifact max `919 MiB`, wall `122.39s`
  - backend counts:
    `standard_sink = 12`, `standard_sink_sliding_chunked = 12`
- APA r0.15:
  - `1024`: pass, actual `1024`, artifact max `913 MiB`, wall `70.60s`
  - `2048`: pass, actual `2048`, artifact max `919 MiB`, wall `125.22s`
  - backend counts:
    `apa_selective_sink_fused = 12`, `standard_sink_sliding_chunked = 12`

H4 APA 4K/8K command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_context_ladder.py --corpus-dir docs --lengths 4096,8192 --settings apa_r0.15 --apa-layer-scope full --bulk-bits 8 --expert-mode resident_packed_mxfp4 --output artifacts/gpt_oss_20b/h4_context_ladder_apa_4k_8k.json`

H4 APA 4K/8K result:
- APA r0.15:
  - `4096`: pass, actual `4096`, artifact max `937 MiB`, wall `236.88s`
  - `8192`: pass, actual `8192`, artifact max `969 MiB`, wall `473.46s`
  - backend counts:
    `apa_selective_sink_fused = 12`, `standard_sink_sliding_chunked = 12`

H4 sampled APA 8K command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_context_ladder.py --corpus-dir docs --lengths 8192 --settings apa_r0.15 --apa-layer-scope full --bulk-bits 8 --expert-mode resident_packed_mxfp4 --output artifacts/gpt_oss_20b/h4_context_ladder_apa_8k_sampled.json`

H4 sampled APA 8K result:
- APA r0.15:
  - pass, actual `8192`
  - artifact max memory: `967 MiB`
  - monitor peak memory: `1891 MiB`
  - monitor samples: `909`
  - backend counts:
    `apa_selective_sink_fused = 12`, `standard_sink_sliding_chunked = 12`
  - wall: `474.62s`

H4 sampled APA 16K command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_context_ladder.py --corpus-dir docs --lengths 16384 --settings apa_r0.15 --apa-layer-scope full --bulk-bits 8 --expert-mode resident_packed_mxfp4 --output artifacts/gpt_oss_20b/h4_context_ladder_apa_16k_sampled.json`

H4 sampled APA 16K result:
- APA r0.15:
  - pass, actual `16384`
  - artifact max memory: `1031 MiB`
  - monitor peak memory: `2039 MiB`
  - monitor samples: `1945`
  - backend counts:
    `apa_selective_sink_fused = 12`, `standard_sink_sliding_chunked = 12`
  - wall: `1015.97s`

Post-run GPU state:
- `NVIDIA GeForce RTX 4070 SUPER, 274, 12282, 35`

Interpretation:
- H4 has not found an APA OOM boundary yet.
- Standard has been measured through `2048` tokens with no OOM.
- APA r0.15 has been measured through `16384` real tokens with no OOM.
- The sampled peak rose from `1891 MiB` at 8K to `2039 MiB` at 16K, about
  `+148 MiB` for a 2x context increase on this streamed path.
- Artifact snapshots rose from roughly `911 MiB` at 512 to `1031 MiB` at 16K.
- Runtime, not VRAM, is currently the limiting practical cost.
- This is still the streamed layer-resident path. It proves context-side
  viability for this runner, not full-model-resident serving.

Next:
- Continue H4 ladder at `32768`, then `65536` if 32K passes.
- After H4 context fit is established, move to H5 GPT-OSS GRM capture/mount.

Action: Investigated H4 runtime speed and added compact MoE routing receipts.

Reason:
- The 16K H4 rung passed on VRAM but took `1015.97s`.
- Before running 32K, the runner needed a check for avoidable overhead that
  does not change model math or downgrade the real-token methodology.

Implementation:
- Added `--route-detail {full,summary}` to
  `scripts/gpt_oss20b_stream_forward_smoke.py`.
- Added `--expert-empty-cache-interval N` to
  `scripts/gpt_oss20b_stream_forward_smoke.py`.
- The default stream runner remains conservative:
  - `route_detail = full`
  - `expert_empty_cache_interval = 1`
- The H4 context ladder now defaults to:
  - `route_detail = summary`
  - `expert_empty_cache_interval = 0`
- Summary routing receipts keep:
  - route detail mode
  - token count
  - experts per token
  - unique experts
  - expert histogram
  - slot weight means
  - slot weight max values
  - empty-cache interval
- Full routing receipts still keep per-token `top_indices` and `top_weights`.

Failure caught:
- First fast-mode smoke failed with:
  `AttributeError: 'GptOss20BConfig' object has no attribute 'num_experts'`
- Root cause:
  - GPT-OSS config uses `num_local_experts`, not `num_experts`.
- Fix:
  - Summary histogram now uses `cfg.num_local_experts`.
  - Scaffold config test now asserts `num_local_experts == 32` and
    `num_experts_per_tok == 4`.

Verification:
- Compile command:
  `env PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile core/gpt_oss20b_tc.py scripts/gpt_oss20b_stream_forward_smoke.py scripts/gpt_oss20b_context_ladder.py tests/test_gpt_oss20b_scaffold.py`
  - result: pass
- Pure ladder tests:
  `PYTEST_ADDOPTS='-p no:cacheprovider' pytest tests/test_gpt_oss20b_context_ladder.py -q`
  - result: `6 passed in 0.04s`
- GPT-OSS scaffold tests:
  `PYTEST_ADDOPTS='-p no:cacheprovider' pytest tests/test_gpt_oss20b_scaffold.py -q`
  - sandbox result: failed because CUDA was not visible:
    `cudaMalloc failed: no CUDA-capable device is detected`
  - escalated GPU result: `10 passed, 2 warnings in 0.52s`

Speed smoke A/B:
- Prompt:
  `artifacts/gpt_oss_20b/h4_context_ladder_chunked_sliding_smoke/prompts/prompt_512.txt`
- Shape:
  - real prompt tokens: `512`
  - layers: `2`
  - attention mode: `apa_selective`
  - APA scope: `full`
  - refine percentile: `0.15`
  - expert mode: `resident_packed_mxfp4`
  - LM head skipped

Old/full receipt command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_stream_forward_smoke.py --prompt-file artifacts/gpt_oss_20b/h4_context_ladder_chunked_sliding_smoke/prompts/prompt_512.txt --max-tokens 512 --max-layers 2 --attention-mode apa_selective --apa-layer-scope full --refine-percentile 0.15 --bulk-bits 8 --expert-mode resident_packed_mxfp4 --skip-lm-head --output artifacts/gpt_oss_20b/h4_speed_smoke_old_512_2layers.json`

Old/full receipt result:
- status: `ok`
- completed layers: `2`
- artifact: `artifacts/gpt_oss_20b/h4_speed_smoke_old_512_2layers.json`
- route detail: `full`
- empty cache interval: `1`
- artifact wall seconds: `5.843513193947729`
- artifact size: `268871` bytes

Fast/summary receipt command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_stream_forward_smoke.py --prompt-file artifacts/gpt_oss_20b/h4_context_ladder_chunked_sliding_smoke/prompts/prompt_512.txt --max-tokens 512 --max-layers 2 --attention-mode apa_selective --apa-layer-scope full --refine-percentile 0.15 --bulk-bits 8 --expert-mode resident_packed_mxfp4 --route-detail summary --expert-empty-cache-interval 0 --skip-lm-head --output artifacts/gpt_oss_20b/h4_speed_smoke_fast_512_2layers.json`

Fast/summary receipt result:
- status: `ok`
- completed layers: `2`
- artifact: `artifacts/gpt_oss_20b/h4_speed_smoke_fast_512_2layers.json`
- route detail: `summary`
- empty cache interval: `0`
- artifact wall seconds: `5.914710729965009`
- artifact size: `24010` bytes

Interpretation:
- The compact receipt reduced the 512-token two-layer artifact from `268871`
  bytes to `24010` bytes, about an 11x reduction.
- The tiny A/B did not improve wall time because startup and layer load
  overhead dominate at 512 tokens and 2 layers.
- The remaining H4 wall-clock bottleneck is the token-by-token selected MoE
  path: each token routes to `4` experts, repeated across up to `24` layers.
- A material speedup requires a batched or fused routed MXFP4 MoE kernel, not
  more runner cleanup.
- The compact receipt mode is still useful for 32K+ because it prevents
  long-context artifacts from ballooning with per-token route lists.

Action: Ran H4 APA r0.15 real-token 32K context rung.

Command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_context_ladder.py --corpus-dir docs --lengths 32768 --settings apa_r0.15 --apa-layer-scope full --bulk-bits 8 --expert-mode resident_packed_mxfp4 --output artifacts/gpt_oss_20b/h4_context_ladder_apa_32k_sampled.json`

Artifacts:
- Ladder:
  `artifacts/gpt_oss_20b/h4_context_ladder_apa_32k_sampled.json`
- Run:
  `artifacts/gpt_oss_20b/h4_context_ladder_apa_32k_sampled/apa_r0.15_32768.json`

Result:
- ladder status: `ok`
- classification: `pass`
- actual input tokens: `32768`
- completed layers: `24`
- first failure: `null`
- first OOM: `null`
- max pass tokens: `32768`
- run wall seconds: `2216.4767407539766`
- stream artifact wall seconds: `2210.947730574`
- layer time sum: `2203.3906139629544`
- average layer time: `91.80794224845643`
- artifact max observed memory: `1155 MiB`
- monitor peak memory: `2449 MiB`
- monitor samples: `4246`
- final hidden shape: `[1, 32768, 2880]`
- route detail: `summary`
- expert empty cache interval: `0`
- backend counts:
  - `apa_selective_sink_fused = 12`
  - `standard_sink_sliding_chunked = 12`
- run artifact size:
  `1067619` bytes
- run directory size:
  `1.2M`
- prompt directory size:
  `116K`

Post-run GPU state:
- `NVIDIA GeForce RTX 4070 SUPER, 251, 12282, 33`

Comparison to prior H4 rungs:
- APA 16K:
  - monitor peak memory: `2039 MiB`
  - artifact max memory: `1031 MiB`
  - wall: `1015.97s`
- APA 32K:
  - monitor peak memory: `2449 MiB`
  - artifact max memory: `1155 MiB`
  - wall: `2216.48s`
- Delta from 16K to 32K:
  - monitor peak: `+410 MiB`
  - artifact max: `+124 MiB`
  - wall: about `2.18x`

Interpretation:
- APA r0.15 has now passed a real-token `32768` context rung through all 24
  streamed GPT-OSS layers.
- No OOM boundary has been found yet for the APA operating point.
- Memory continues to scale modestly in this streamed path; the practical
  limiter remains runtime.
- This is still prefill/context-fit evidence with `--skip-lm-head`; it is not
  yet a generation, recall, or GRM continuity result.

Next:
- Run H4 APA r0.15 at `65536` if continuing the context ladder.
- If `65536` passes, decide whether to attempt `131072` or move to H5 GRM
  capture/mount first.

Action: Ran H4 APA r0.15 real-token 64K context rung.

Command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_context_ladder.py --corpus-dir docs --lengths 65536 --settings apa_r0.15 --apa-layer-scope full --bulk-bits 8 --expert-mode resident_packed_mxfp4 --output artifacts/gpt_oss_20b/h4_context_ladder_apa_64k_sampled.json`

Artifacts:
- Ladder:
  `artifacts/gpt_oss_20b/h4_context_ladder_apa_64k_sampled.json`
- Run:
  `artifacts/gpt_oss_20b/h4_context_ladder_apa_64k_sampled/apa_r0.15_65536.json`

Result:
- ladder status: `ok`
- classification: `pass`
- actual input tokens: `65536`
- completed layers: `24`
- first failure: `null`
- first OOM: `null`
- max pass tokens: `65536`
- run wall seconds: `5576.632580742997`
- stream artifact wall seconds: `5571.350367329025`
- layer time sum: `5562.851192067028`
- average layer time: `231.78546633612618`
- artifact max observed memory: `1421 MiB`
- monitor peak memory: `4834 MiB`
- monitor samples: `10684`
- final hidden shape: `[1, 65536, 2880]`
- route detail: `summary`
- expert empty cache interval: `0`
- backend counts:
  - `apa_selective_sink_fused = 12`
  - `standard_sink_sliding_chunked = 12`
- run artifact size:
  `2076839` bytes
- run directory size:
  `2.3M`
- prompt directory size:
  `228K`

Post-run GPU state:
- `NVIDIA GeForce RTX 4070 SUPER, 279, 12282, 39`

Comparison to prior H4 rungs:
- APA 32K:
  - monitor peak memory: `2449 MiB`
  - artifact max memory: `1155 MiB`
  - wall: `2216.48s`
- APA 64K:
  - monitor peak memory: `4834 MiB`
  - artifact max memory: `1421 MiB`
  - wall: `5576.63s`
- Delta from 32K to 64K:
  - monitor peak: `+2385 MiB`
  - artifact max: `+266 MiB`
  - wall: about `2.52x`

Interpretation:
- APA r0.15 has now passed a real-token `65536` context rung through all 24
  streamed GPT-OSS layers.
- No APA OOM boundary has been found through 64K.
- The sampled monitor peak is still below half of the RTX 4070 Super's
  `12282 MiB` reported memory, but the wall time is now the dominant cost.
- This remains prefill/context-fit evidence with `--skip-lm-head`; it is not a
  generation, recall, PPL, or GRM continuity result.

Next:
- H4 has cleared the user's stated minimum target of 64K.
- A 131072-token rung is now plausible on memory but would likely take several
  hours on the current token-routed MoE path.
- Before spending that runtime, decide whether to run the 128K context rung or
  pivot to H5 GRM capture/mount at the confirmed 64K operating point.

Action: Started H4 APA r0.15 real-token 128K context rung, then pivoted H5
implementation so the next long prefill can mint reusable GPT-OSS grafts.

Live H4 command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_context_ladder.py --corpus-dir docs --lengths 131072 --settings apa_r0.15 --apa-layer-scope full --bulk-bits 8 --expert-mode resident_packed_mxfp4 --output artifacts/gpt_oss_20b/h4_context_ladder_apa_128k_sampled.json`

Current live status at implementation update:
- artifact:
  `artifacts/gpt_oss_20b/h4_context_ladder_apa_128k_sampled/apa_r0.15_131072.json`
- status: `running_layers`
- completed layers: `1`
- completed layer receipt: layer `0`, `sliding_attention`,
  `standard_sink_sliding_chunked`
- layer 0 wall seconds: `279.4224069789634`
- latest GPU sample: about `6613 MiB` used, `100%` utilization, `76 C`

H5 implementation update:
- `core/gpt_oss20b_tc.py`
  - Added GPT-OSS attention capture hooks for pre-RoPE K/V and pre-RoPE query
    tensors.
  - Added GPT-OSS graft injection with the established GQA remount law:
    captured keys are re-RoPE'd at graft seats, live tokens are shifted after
    the mounted graft, and cached decode does not re-inject the graft.
  - Added `gpt_oss_grm_dialect_kwargs()` to register GPT-OSS as a native GRM
    GQA-style dialect with `position_law=rope_full_yarn`,
    `state_kind=kv`, `graftability=seat_remountable`, `num_kv_heads=8`,
    `head_dim=64`, and `vals_per_tok_layer=1024`.
- `scripts/gpt_oss20b_stream_forward_smoke.py`
  - Added `--capture-graft-dir` to write per-layer pre-RoPE K/V shards during
    a streamed forward pass.
  - Added `--mount-graft-dir` to mount those shards before a live prompt.
  - Added `--input-ids-file` so exact token-count capture prompts can bypass
    text decode/re-tokenize drift.
  - Added `--candidate-text` for first-token gold/decoy logit scoring, so an
    H5 bulk-graft gate can test information access without slow greedy decode.
  - Added graft manifest receipts, GPT-OSS GRM dialect receipts, mounted token
    counts, per-layer capture geometry, and per-layer mount byte counts.
- `scripts/gpt_oss20b_bulk_graft_gate.py`
  - Added a repeatable H5 helper that builds an exact real-token capture prompt,
    plants a needle near the end without repeating corpus tokens, captures
    graft shards, runs an amnesia control, remounts the graft, and compares a
    single-token gold candidate against single-token decoys.
  - Capture now feeds exact token IDs to the streamed runner and writes the
    decoded text only as an audit artifact. This avoids tokenizer
    decode/re-tokenize drift that made a 4096-token text prompt come back as
    4097 tokens.
  - Default needle now uses `BLUE` because GPT-OSS tokenizes `" BLUE"` as a
    single token; default decoys are `" RED"`, `" GREEN"`, `" BLACK"`, and
    `" WHITE"`, also single-token continuations.

Native GRM clarification:
- The C++ GRM runtime already implements the host runtime and byte-level
  arena oracles:
  - `DialectDescriptor`
  - `HostGraftStore`
  - `RouterIndex`
  - `DirtyQueue`
  - `DurabilityWriter`
  - `DeviceArena` swap/evict planner and host tensor swap/evict references
  - C ABI plus `core/grm_native.py`
- The C++ README explicitly says the CUDA arena is not implemented yet. For
  GPT-OSS, the native runtime is therefore authoritative for host payloads,
  dialect metadata, route bookkeeping, dirty/durable state, and mount commits;
  the actual device splice still happens through the TensorCUDA GPT-OSS
  attention adapter.

Validation:
- `env PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile core/gpt_oss20b_tc.py scripts/gpt_oss20b_stream_forward_smoke.py tests/test_gpt_oss20b_scaffold.py`
  passed.
- `python3 scripts/gpt_oss20b_stream_forward_smoke.py --help | rg -n 'capture-graft|mount-graft|route-detail'`
  confirmed the new CLI flags.
- `env PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/gpt_oss20b_bulk_graft_gate.py`
  passed.
- `env PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_bulk_graft_gate.py --target-tokens 512 --dry-run --output /tmp/gpt_oss_h5_bulk_graft_dryrun.json`
  passed:
  - exact capture tokens: `512`
  - query tokens: `16`
  - candidate tokenization:
    - `" BLUE"` -> `[110151]`
    - `" RED"` -> `[45309]`
    - `" GREEN"` -> `[107361]`
    - `" BLACK"` -> `[71730]`
    - `" WHITE"` -> `[94026]`
- `env PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTEST_ADDOPTS='-p no:cacheprovider' pytest -q tests/test_gpt_oss20b_context_ladder.py`
  passed: `6 passed`.
- `env PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTEST_ADDOPTS='-p no:cacheprovider' pytest -q tests/test_grm_native_runtime.py -k 'dialect_profile or lifecycle_via_ctypes'`
  passed: `2 passed, 88 deselected`.
- `env PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTEST_ADDOPTS='-p no:cacheprovider' pytest -q tests/test_gpt_oss20b_scaffold.py -k 'grm_dialect or captures_prerope or does_not_reinject'`
  passed in the sandbox as `1 passed, 2 skipped`; the skipped tests require a
  visible TensorCUDA GPU device, which the sandboxed pytest process did not
  have while the real H4 GPU job was active.
- Real GPU execution of the two TensorCUDA GPT-OSS graft-splice tests remains
  pending until the H4 128K GPU job is free.
- After freeing the GPU, real GPU execution passed:
  `env PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTEST_ADDOPTS='-p no:cacheprovider' pytest -q tests/test_gpt_oss20b_scaffold.py tests/test_gpt_oss20b_context_ladder.py`
  -> `19 passed, 2 warnings`.

Interpretation:
- This does not claim GPT-OSS GRM recall yet.
- It does make future long GPT-OSS forwards reusable as graft-minting passes
  and creates the fast H5 remount path once a graft directory exists.
- Sliding layers remain architecturally bounded by GPT-OSS's sliding window
  even when a large graft is mounted; full-attention layers are the path that
  can attend over the whole mounted graft.

Queued H5 command for the next free GPU slot:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_bulk_graft_gate.py --target-tokens 131072 --attention-mode apa_selective --apa-layer-scope full --refine-percentile 0.15 --bulk-bits 8 --expert-mode resident_packed_mxfp4 --route-detail summary --expert-empty-cache-interval 0 --output artifacts/gpt_oss_20b/h5_bulk_graft_128k_candidate_gate.json`

Cheaper H5 shakeout command before the full 128K gate:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_bulk_graft_gate.py --target-tokens 4096 --attention-mode apa_selective --apa-layer-scope full --refine-percentile 0.15 --bulk-bits 8 --expert-mode resident_packed_mxfp4 --route-detail summary --expert-empty-cache-interval 0 --output artifacts/gpt_oss_20b/h5_bulk_graft_4k_candidate_gate.json`

Action: Stopped the pre-graft 128K H4 live-prefill job and ran the 4K H5
bulk-graft candidate gate.

128K H4 stop reason:
- The job was launched before GPT-OSS graft capture existed, so it could only
  produce H4 live-prefill evidence and could not mint reusable H5 grafts.
- It had completed only layer `0` after a long wait:
  - status: `running_layers`
  - completed layers: `1`
  - layer 0 type/backend: `sliding_attention`,
    `standard_sink_sliding_chunked`
  - layer 0 wall: `279.4224069789634s`
- It was stopped with `KeyboardInterrupt`, leaving no hidden child stream
  process and returning the GPU to idle.

4K H5 command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_bulk_graft_gate.py --target-tokens 4096 --attention-mode apa_selective --apa-layer-scope full --refine-percentile 0.15 --bulk-bits 8 --expert-mode resident_packed_mxfp4 --route-detail summary --expert-empty-cache-interval 0 --output artifacts/gpt_oss_20b/h5_bulk_graft_4k_candidate_gate.json`

Artifacts:
- Summary:
  `artifacts/gpt_oss_20b/h5_bulk_graft_4k_candidate_gate.json`
- Run directory:
  `artifacts/gpt_oss_20b/h5_bulk_graft_4k_candidate_gate`
- Capture:
  `artifacts/gpt_oss_20b/h5_bulk_graft_4k_candidate_gate/capture_forward.json`
- Control:
  `artifacts/gpt_oss_20b/h5_bulk_graft_4k_candidate_gate/control_forward.json`
- Mounted-graft:
  `artifacts/gpt_oss_20b/h5_bulk_graft_4k_candidate_gate/mount_forward.json`
- Graft manifest:
  `artifacts/gpt_oss_20b/h5_bulk_graft_4k_candidate_gate/graft/manifest.json`

Result:
- status: `ok`
- classification: `pass`
- capture target/actual tokens: `4096 / 4096`
- input mode: `direct_token_ids`
- needle offset: `4077`
- needle tokens: `19`
- query tokens: `16`
- capture run:
  - completed layers: `24`
  - wall seconds: `215.35076313297031`
  - run wrapper wall seconds: `220.43408841098426`
  - final GPU receipt: `NVIDIA GeForce RTX 4070 SUPER, 515, 12282, 12`
- graft payload:
  - layers: `24`
  - token count: `4096`
  - total host bytes: `201326592`
  - run directory size: `193M`
- amnesia control run:
  - completed layers: `24`
  - live prompt tokens: `16`
  - wall seconds: `17.975686279998627`
  - wrapper wall seconds: `23.084253881010227`
  - gold candidate `" BLUE"` logit: `2.015625`
  - best decoy `" RED"` logit: `3.796875`
  - gold minus best decoy: `-1.78125`
- mounted-graft run:
  - completed layers: `24`
  - live prompt tokens: `16`
  - mounted graft tokens: `4096`
  - RoPE table length: `4112`
  - wall seconds: `18.54849335399922`
  - wrapper wall seconds: `23.509232072974555`
  - gold candidate `" BLUE"` logit: `11.75`
  - best decoy `" GREEN"` logit: `7.90625`
  - gold minus best decoy: `3.84375`

Interpretation:
- This is the first GPT-OSS same-model cold-graft access pass.
- The live query did not contain the answer. The no-graft control preferred a
  decoy by `1.78125` logits; the mounted-graft run preferred the gold answer
  by `3.84375` logits.
- This is H5 candidate-logit evidence, not open greedy recall yet.
- The mounted pass was only about `0.57s` slower than the no-graft control at
  the streamed-runner layer-wall level, because the live MoE workload remained
  16 tokens while the graft was mounted as K/V.

Action: Scaled the H5 bulk-graft candidate gate to 16K real tokens.

16K H5 command:
- `env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/gpt_oss20b_bulk_graft_gate.py --target-tokens 16384 --attention-mode apa_selective --apa-layer-scope full --refine-percentile 0.15 --bulk-bits 8 --expert-mode resident_packed_mxfp4 --route-detail summary --expert-empty-cache-interval 0 --output artifacts/gpt_oss_20b/h5_bulk_graft_16k_candidate_gate.json`

Artifacts:
- Summary:
  `artifacts/gpt_oss_20b/h5_bulk_graft_16k_candidate_gate.json`
- Run directory:
  `artifacts/gpt_oss_20b/h5_bulk_graft_16k_candidate_gate`
- Capture:
  `artifacts/gpt_oss_20b/h5_bulk_graft_16k_candidate_gate/capture_forward.json`
- Control:
  `artifacts/gpt_oss_20b/h5_bulk_graft_16k_candidate_gate/control_forward.json`
- Mounted-graft:
  `artifacts/gpt_oss_20b/h5_bulk_graft_16k_candidate_gate/mount_forward.json`
- Graft manifest:
  `artifacts/gpt_oss_20b/h5_bulk_graft_16k_candidate_gate/graft/manifest.json`

Result:
- status: `ok`
- classification: `pass`
- capture target/actual tokens: `16384 / 16384`
- input mode: `direct_token_ids`
- corpus fill tokens: `16365`
- needle offset: `16365`
- needle tokens: `19`
- query tokens: `16`
- attention split:
  - APA layers: `[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]`
  - standard sliding layers: `[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]`
- capture run:
  - completed layers: `24`
  - input shape: `[1, 16384]`
  - RoPE table length: `16384`
  - wall seconds: `931.0161463900004`
  - run wrapper wall seconds: `936.0960545630078`
  - final hidden shape: `[1, 16384, 2880]`
  - final GPU receipt: `NVIDIA GeForce RTX 4070 SUPER, 585, 12282, 39`
- graft payload:
  - layers: `24`
  - token count: `16384`
  - total host bytes: `805306368`
  - per-layer host bytes: `33554432`
  - run directory size: `769M`
- amnesia control run:
  - completed layers: `24`
  - live prompt tokens: `16`
  - RoPE table length: `16`
  - wall seconds: `19.052695906953886`
  - wrapper wall seconds: `24.094125150004402`
  - gold candidate `" BLUE"` logit: `2.015625`
  - best decoy `" RED"` logit: `3.796875`
  - gold minus best decoy: `-1.78125`
- mounted-graft run:
  - completed layers: `24`
  - live prompt tokens: `16`
  - mounted graft tokens: `16384`
  - RoPE table length: `16400`
  - wall seconds: `19.183853538008407`
  - wrapper wall seconds: `24.281130205024965`
  - gold candidate `" BLUE"` logit: `11.0`
  - best decoy `" RED"` logit: `5.90625`
  - gold minus best decoy: `5.09375`

Interpretation:
- The 16K same-model cold-graft candidate gate passes.
- The live query still did not contain the answer. The no-graft control again
  preferred a decoy by `1.78125` logits; the mounted-graft run preferred the
  gold answer by `5.09375` logits.
- The 16K graft payload is exactly 4x the 4K payload, as expected from the
  captured BF16 K/V geometry.
- The mounted pass added only about `0.13s` at the streamed-runner layer-wall
  level versus the no-graft control, because the live prompt remained 16
  tokens and the expensive MoE route was not replaying the 16K capture text.
- This is still H5 candidate-logit evidence. H6 greedy/open multi-turn recall
  remains unproven.
