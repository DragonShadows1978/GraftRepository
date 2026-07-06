# GPT-OSS-20B APA/GRM Synthesis

GPT-OSS-20B is worth investigating, but it should be treated as a new
architecture project rather than another straightforward low-bit loader.

The attractive part is the shape. The model is MoE, with about 21B total
parameters but about 3.6B active parameters per token. It uses RoPE/YARN with a
131k maximum context window, GQA with 64 query heads and 8 KV heads, and an
alternating sliding/full attention schedule. The sliding layers are capped at a
window of 128, so the long-context pressure should concentrate in the
full-attention layers. That is exactly the kind of structure where APA might
matter.

The hard part is the weight format. The expert body is published as MXFP4
blocks and scales. That is not the same as the local affine INT4/INT3 path used
for Qwen3.5. The official safetensors payload is about 12.816 GiB, which is
already too close to a 12GB card before runtime memory. The expert body accounts
for about 9.455 GiB of that payload, while the non-expert BF16 body accounts for
about 3.361 GiB. The embedding and lm_head are untied and together cost about
2.16 GiB in BF16.

This produces a clear design law: the expert body must remain packed, or the
memory case dies. If the TensorCUDA port requires dequantizing MXFP4 experts
into BF16, it is not the right path for this hardware. The plausible route is a
hybrid port: preserve the official MXFP4 expert representation, quantize the
BF16 attention/router/embed/lm_head body, then attach APA to the full-attention
layers.

The model is structurally graftable because it uses RoPE/YARN, but that does not
mean GRM support is automatic. The loader must identify the correct pre-RoPE
state boundary, handle alternating full/sliding layers, preserve attention
sinks, and respect the Harmony chat format. Behavior outside Harmony is not a
valid test because the model card and implementation notes explicitly warn that
the format is required.

The first sensible experiment is not a TensorCUDA port. It is a Phase 0/Phase 1
gate: record exact local metadata, decide whether to download the model, and
try an official runtime load smoke. If the official runtime cannot fit on the
4070 Super, that failure is still useful because it validates the need for a
hybrid TensorCUDA path. If it can fit, it gives a baseline for prompt formatting,
memory, and output sanity before any custom kernels are written.

Current recommendation: proceed cautiously. GPT-OSS-20B has the right macro
shape for APA/GRM and may be more interesting than a dense 20B-30B model, but
only if the MXFP4 expert path can stay packed end to end.

The Phase 0 local snapshot moved this from source-estimate territory to pinned
metadata. The exact HF revision under test is
`6cee5e81ee83917806bbde320786a8fb61efebee`. The installed stack can render the
Harmony-style tokenizer template through Transformers, even though the separate
`openai-harmony` package is not installed. The local artifact also confirmed
the payload split: about 9.46 GiB of expert U8/MXFP4 tensors and about 3.36 GiB
of BF16 non-expert tensors.

The Phase 1 Ollama smoke is useful but narrow. `gpt-oss:20b` pulled
successfully and produced a short correct answer on the 4070 Super when run
with `think:false`: `The capital of France is Paris.` The cold load peaked at
11,392 MiB from a 275 MiB baseline, so the official runtime can fit a short
prompt without OOM.

That should not be over-read. Ollama reported the resident model as a 20% CPU /
80% GPU split with a 4096 context, so this is not evidence that a fully GPU
resident 131k operating point exists. The first cold prompt also showed why the
Harmony/reasoning controls matter: with the default medium reasoning behavior
and only 64 generated tokens, the visible answer truncated at `The capital`.
With `think:false` and a larger cap, the answer completed normally.

The next design conclusion is unchanged, but now it is supported by a load
receipt: stock official runtime is barely inside the 12GB card for a short
prompt only because it can split work between CPU and GPU. The TensorCUDA path
still needs to preserve packed MXFP4 experts, compress or otherwise manage the
BF16 non-expert body, and attach APA only after a standard-attention sanity
path exists.
