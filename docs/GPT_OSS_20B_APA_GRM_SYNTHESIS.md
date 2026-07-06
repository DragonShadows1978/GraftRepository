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

The Phase 2 feasibility pass sharpened the implementation boundary. GPT-OSS is
not just "Mistral plus MoE." Attention has learned sink logits that change the
softmax denominator, all attention projections have biases, and the expert MLP
uses a GPT-OSS-specific clipped gate/up activation rather than the existing
SwiGLU helpers. Those details have to be in the standard path before APA or GRM
results mean anything.

The MXFP4 expert format is now concrete: packed nibbles map through a small FP4
codebook and uint8 scales are exponent offsets. Exact dequantization is
available as a diagnostic route, but using it broadly would expand the expert
body and break the 12GB premise. The next real implementation gate is therefore
not "load all weights"; it is a two-track scaffold: one-layer exact-dequant
parity for math, followed by a packed MXFP4 expert GEMV/GEMM path for the
actual operating point.

The first Phase 3A scaffold is now in code and tested. The new GPT-OSS module
parses the pinned config, wraps low-bit linears with bias support, reproduces
MXFP4 exact dequantization for diagnostics, implements the GPT-OSS expert
activation, and adds a sink-aware TensorCUDA attention helper. The focused test
selector passed 7/7 with GPU access. One useful bug was caught immediately:
quantized-linear bias must be cast to the output dtype at call time.

This still is not a model loader. It is the math base the loader needs. The next
gate is a one-layer loader/smoke using the full HF safetensors, followed by the
packed MXFP4 expert kernel work required for a real 12GB operating point.

The full pinned HF safetensors snapshot is now local as well. That matters
because the Ollama model proved official runtime behavior, but it is not the
source layout the TensorCUDA loader consumes. The next gate can now read real
GPT-OSS tensors directly from the pinned HF shards.

The first real TensorCUDA layer smokes now pass. The loader can row-slice real
GPT-OSS embeddings, build GPT-OSS YARN tables, load biased attention projections
from the HF shards, apply sink-aware attention, and run attention/residual for
both a sliding layer and a full-attention layer. The receipts are shape/runtime
receipts only: MoE and lm_head are still skipped, so this is not PPL, generation,
or model-behavior evidence.

That moves the hard problem exactly where expected: GPT-OSS MoE. The next step
is selected-expert exact dequant as a diagnostic bridge, then native packed
MXFP4 expert math for the viable path.

The selected-expert MoE diagnostic bridge now passes for both layer families.
The first run usefully failed at the router boundary with a TensorCUDA dtype
mismatch: the hidden state was promoted to FP32 while the router weight had been
cast to BF16. Keeping the small diagnostic router in FP32 fixed the issue and
keeps route selection numerically conservative.

After that fix, one-token real-tensor smokes passed for layer 0
(`sliding_attention`) and layer 1 (`full_attention`). Each smoke runs
row-sliced embedding, YARN RoPE, sink-aware attention, post-attention norm, the
real router, and exact MXFP4 dequantization of only the selected experts. Layer
0 routed through experts `[13, 17, 21, 29]`; layer 1 routed through
`[12, 15, 25, 26]`. Both produced `[1, 1, 2880]` outputs and returned the GPU to
the 275 MiB baseline after process exit.

This closes the diagnostic MoE correctness bridge, but not the production
route. The exact-dequant path is intentionally tiny and token-limited. The next
hard requirement is native packed MXFP4 expert GEMV/GEMM; without that, GPT-OSS
cannot keep its expert body packed end to end on a 12GB card. Only after that
standard path exists should lm_head behavior, PPL, APA, GRM, and real context
extension tests be trusted.

That packed MXFP4 kernel now exists in Project-Tensor on
`codex/gpt-oss-mxfp4-kernel` at commit `e4d39d1`. It exposes
`tc.mxfp4_linear(x, blocks, scales)` for GPT-OSS expert tensors shaped
`[out_features, groups, 16]` with uint8 E8M0 scales. The Project-Tensor build
passed, and the new MXFP4 tests plus the existing INT2/INT3 regression tests
passed 13/13.

GraftRepository now wires that kernel into the GPT-OSS MoE smoke through an
explicit `--expert-mode packed_mxfp4` path. The old exact-dequant bridge still
passes, which matters because it remains the conservative A/B reference. On the
same layer 0 one-token prompt, dequant mode took about `3.37s`; packed MXFP4
took about `2.31s`, routed to the same experts `[13, 17, 21, 29]`, and produced
very close output stats. Layer 1 packed mode also passed, routing to
`[12, 15, 25, 26]`.

This is a meaningful gate: selected experts can now be consumed without
materializing their dense dequantized matrices. It is still not the finished
loader. The smoke harness uploads selected expert packed blocks from CPU for a
tiny diagnostic call. The next production step is a resident packed-expert
loader/dispatcher so the full expert body can live packed on GPU/host tiers
without per-call CPU upload or BF16 expansion.

The first resident-dispatch attempt found another useful TensorCUDA boundary:
uint8 tensors cannot currently be sliced through the generic Python `slice()`
op. That blocked Python-side selection from a resident
`[experts, out_features, groups, 16]` packed tensor. The fix went into
Project-Tensor as `tc.mxfp4_linear_expert(...)`, which selects the expert by
offset inside the MXFP4 kernel path instead of slicing uint8 tensors in Python.

With that fix, `resident_packed_mxfp4` passes for both GPT-OSS layer families.
Layer 0 resident mode routes to `[13, 17, 21, 29]` and produces the same compact
output stats as the CPU-selected packed path. Layer 1 resident mode routes to
`[12, 15, 25, 26]`. The inside-script VRAM rises from the previous `497 MiB`
smoke footprint to about `905 MiB`, which is expected because one full layer's
packed expert body is resident on GPU.

That closes the one-layer resident dispatch gate. It does not close full-model
residency: the packed expert body alone is about `9.46 GiB`, before BF16
non-expert weights, lm_head/embed, KV, APA state, or allocator margin. The next
loader decision is therefore a residency policy, not just another kernel:
which tensors stay GPU resident, which are quantized further, and which can be
tiered without destroying decode latency.

The first streamed full-stack smoke now works. Instead of trying to keep the
whole model resident, the harness streams one decoder layer at a time, keeps
that layer's packed experts resident, runs the block, frees it, and continues.
All 24 layers completed on the 4070 Super with the resident packed expert path.
With only the final hidden state live, the script sat around `485 MiB`; while a
layer's packed experts were resident, it reported about `905 MiB`.

The streamed path also attaches final RMSNorm and a locally quantized
TensorCUDA lm_head. A one-token cap proved only that the output projection is
connected. The better smoke was the plain prompt `The capital of France is`,
which tokenized to five tokens and produced ` Paris` as the top next token. That
is a real behavior sanity receipt for the custom streamed TensorCUDA path, but
it is still not PPL, not Harmony chat behavior, not greedy generation, and not
APA/GRM evidence.

The project has therefore crossed from one-layer math receipts into a complete
streamed forward receipt. The next meaningful gates are PPL, a short greedy
decode loop, Harmony-formatted prompting, and then APA/GRM context/recall tests.

A tiny PPL-style smoke now exists on the streamed path as well. On
`The capital of France is Paris.`, capped to seven tokens and scored as six
next-token targets, the streamed TensorCUDA path reported mean NLL
`2.9856` and PPL `19.80`. That number should not be treated as a benchmark; the
sample is far too small and the path uses the local quantized lm_head. The value
is useful only as a receipt that shifted-target scoring works through all 24
layers and the output projection.
