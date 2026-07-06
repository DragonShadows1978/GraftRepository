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

Pending:
- Decide whether to download the model.
- If downloaded, run an official runtime load smoke before TensorCUDA work.
- If official runtime fails on memory, design the TensorCUDA hybrid path.
