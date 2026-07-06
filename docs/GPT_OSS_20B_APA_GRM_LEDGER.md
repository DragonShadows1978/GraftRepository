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

Pending:
- Begin Phase 2 TensorCUDA feasibility design from these receipts.
- Decide how to preserve the packed MXFP4 expert body.
- Decide how to quantize the BF16 attention/embed/lm_head body without breaking
  Harmony-format behavior.
