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

A repeated-forward greedy smoke also works. Starting from
`The capital of France is`, the first generated token was ` Paris`, and the
second was `."`, producing `The capital of France is Paris."`. The important
part is not generation polish; it is that the streamed TensorCUDA path can now
be driven in a greedy loop. It is still slow because each token reruns the full
streamed forward and does not reuse KV cache.

Harmony prompting now has a first streamed receipt. The first attempt was
usefully wrong: `--max-tokens 64` truncated a 79-token rendered Harmony prompt,
so the top tokens were ordinary conversational starts. Rerunning with
`--max-tokens 128` kept the full template and made the top next token
`<|channel|>`, the expected protocol transition before content. That means the
streamed path can carry Harmony-formatted input, but a real Harmony answer test
needs protocol-aware greedy decode rather than a one-token content expectation.

Sink-aware APA is now wired into the GPT-OSS streamed path. The important
architecture detail is that GPT-OSS attention has learned sink logits: the sink
participates in the softmax denominator but has no value vector. Generic APA
would therefore be wrong if it normalized only over key columns. Project-Tensor
now exposes a sink-aware APA blend primitive, and GraftRepository can route
GPT-OSS attention through `apa_selective` with explicit refine percentile and
bulk-bit settings.

The first APA receipts are still Tier 3 evidence. A two-layer APA shakeout and a
full 24-layer streamed top-k smoke both completed, and the plain prompt
`The capital of France is` still ranked ` Paris` first. A six-target toy PPL
smoke reported `19.57` for APA r0.15 versus `19.80` for the prior standard
smoke, but that is only a wiring sanity check, not a corpus benchmark.

The house-rules execution plan now locks the remaining claim boundaries in
`docs/GPT_OSS_20B_HOUSE_RULES_EXECUTION_PLAN.md`. Current receipts do not prove
corpus PPL, long-context support, APA memory flattening, GRM remount, or
cold-KV recall. Those claims require the registered real-text PPL, tiled
sink-aware APA memory path, real-token context ladder, and GRM continuity gates.

H1 has now tightened the APA evidence path. The stream harness separates
requested attention mode from effective per-layer mode and records both planned
and actual APA layers in the artifact. By default, GPT-OSS APA is scoped to the
12 full-attention layers, while the 12 sliding-window layers stay on standard
attention because their window is already bounded.

The H1 comparison run gives a cleaner Tier 3 receipt: standard used no APA
layers and ranked ` Paris` first; APA r0.15 used layers
`[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]`, kept sliding layers standard,
and also ranked ` Paris` first. The logits are close enough for a smoke
receipt, but the next meaningful quality claim still requires H2 real-text PPL.

The first H2 real-text gate now exists. It scores two 64-token windows from
`docs/GRM_Primer.md` through the streamed TensorCUDA path, for 126 scored
tokens per setting. Standard landed at PPL `30.24`; APA r0.15 landed at
`29.53`; APA r0.10 landed at `29.85`. Both APA settings used the 12
full-attention layers and left the 12 sliding-window layers standard. This is
materially stronger than the toy PPL smoke because it uses fixed real text and
aggregated window artifacts, but it is still a small gate. The next H2 step is
to scale token count before making a broad quality claim.

That scale-up now has a stronger receipt: four 128-token windows from the same
fixed corpus, giving 508 scored tokens per setting. Standard measured PPL
`24.61`; APA r0.15 measured `24.25`; APA r0.10 measured `24.36`. The result is
directionally favorable for APA on this slice and, more importantly, shows no
quality collapse on a real-text PPL gate. The memory numbers are not yet the
interesting part because this path still materializes score matrices. H3 is
therefore the next real engineering gate: tiled sink-aware APA that preserves
the sink denominator without paying full score-matrix memory.

H3 now has an implementation receipt. Project-Tensor exposes
`tc.apa_selective_attention_sink`, a fused GQA-aware APA kernel that includes
GPT-OSS learned sinks in the online softmax denominator without adding a value
column. GraftRepository uses that fused path for the 12 full-attention GPT-OSS
layers, while the 12 sliding-window layers remain on standard sink attention by
default.

The full H3 top-k smoke completed all 24 layers with backend counts
`standard_sink = 12` and `apa_selective_sink_fused = 12`, and still ranked
` Paris` first. A small fused-path real-text PPL smoke scored 126 tokens at PPL
`29.68`, consistent with the previous small APA gate. This closes the
score-matrix implementation gap for full-attention GPT-OSS APA. It does not yet
prove a longer context window; H4 has to run the real-token OOM ladder.

H4 is now in motion with real-token fills from the repository docs corpus. The
first important engineering correction was not APA itself but GPT-OSS sliding
attention: the sliding layers were masked to a 128-token window, but the
TensorCUDA path still built full `L x S` score matrices. That would have
polluted any long-context result. The path now uses chunked sliding sink
attention, so the default APA operating point is `12` fused full-attention APA
layers plus `12` chunked sliding-standard layers.

With that fixed, the real-token ladder has passed standard through 2K and APA
r0.15 through 16K. The APA r0.15 sampled peak was `1891 MiB` at 8K and
`2039 MiB` at 16K, while artifact snapshots rose from roughly `911 MiB` at
512 tokens to `1031 MiB` at 16K. No APA OOM boundary has been found yet.
Runtime is currently the expensive part, not VRAM. The next H4 rungs are 32K
and then 64K if 32K passes.

The first H4 speed pass did not change the big conclusion. Compact MoE routing
receipts now exist, and the ladder uses them by default so long-context
artifacts do not store per-token route lists for every layer. A 512-token,
two-layer A/B dropped the receipt from `268871` bytes to `24010` bytes, but it
did not improve wall time on that tiny run (`5.84s` old versus `5.91s`
summary). The real runtime wall is the selected MoE implementation: GPT-OSS is
still routing token-by-token into four experts across the streamed layers. A
true speed breakthrough needs a batched or fused routed MXFP4 MoE kernel. Until
then, the compact receipt mode reduces artifact pressure while preserving the
real-token context ladder methodology.

The 32K H4 rung has now passed. APA r0.15 processed `32768` real tokens through
all 24 streamed GPT-OSS layers with backend counts `12` fused full-attention
APA layers and `12` chunked sliding-standard layers. The monitor peak was
`2449 MiB`, the artifact max was `1155 MiB`, and the final hidden shape was
`[1, 32768, 2880]`. Compared with the 16K rung, the sampled peak rose by
`410 MiB` and wall time rose from about `1016s` to about `2216s`. No APA OOM
boundary has been found yet. This is still context-fit/prefill evidence, not a
generation or GRM continuity result, but the operating point has cleared 32K on
real tokens.

The 64K H4 rung has also passed. APA r0.15 processed `65536` real tokens
through all 24 streamed layers with the same backend split: `12`
`apa_selective_sink_fused` full-attention layers and `12`
`standard_sink_sliding_chunked` sliding layers. The monitor peak was
`4834 MiB`, the artifact max was `1421 MiB`, and the final hidden shape was
`[1, 65536, 2880]`. The 32K-to-64K jump added `2385 MiB` to the sampled peak
and pushed wall time from about `2216s` to about `5577s`. This clears the
practical 64K target on real tokens. A 128K rung now looks plausible on memory,
but the current token-routed MoE path makes it a multi-hour test rather than a
quick confirmation.

The 128K H4 rung is now running, but the faster path for the GRM question is
clearer. A live 128K prefill proves the streamed prompt path. The GRM question
is whether GPT-OSS can remount cold pre-RoPE K/V and use it with a short live
prompt after KV has been cleared. To support that, the streamed GPT-OSS harness
now has a graft capture/mount surface: `--capture-graft-dir` writes per-layer
pre-RoPE K/V shards, and `--mount-graft-dir` seats those shards before the live
prompt.

This work deliberately stays out of the C++/CUDA arena lane. The repository
already has a C++ host runtime with `HostGraftStore`, `RouterIndex`,
durability, dirty planning, route metadata, and byte-level arena swap/evict
references. A separate implementation track is handling the CUDA arena. For
GPT-OSS, this branch owns the model dialect adapter and streamed TensorCUDA
capture/mount harness, while the native layer remains the authority for host
payload accounting, routing state, and mount commits.

The GPT-OSS dialect is now registered as a GQA-style GRM profile:
`rope_full_yarn | kv | seat_remountable | multi_mount`, with `8` KV heads,
`64` head dimension, and `1024` K/V values per token per layer. The remount
law is the same as the proven GQA path: capture K/V before RoPE, re-RoPE keys
at the mounted seats, and shift live-token RoPE positions while the graft is
resident. Sliding layers remain bounded by their sliding window even when a
large graft is mounted; full-attention layers are the part of GPT-OSS that can
see across the whole mounted graft.

This is H5 plumbing, not an H5 pass. The next proof still has to mint real
GPT-OSS graft shards, remount them with the fact omitted from the live prompt,
and beat an amnesia/no-graft control.

The first fast H5 proof should use candidate scoring rather than greedy
generation. The new bulk-graft gate plants a single-token `BLUE` needle into an
exact real-token capture prompt, captures the pre-RoPE K/V shards, then asks a
short live question with no answer in the prompt. It scores the first token of
`BLUE` against single-token decoys (`RED`, `GREEN`, `BLACK`, `WHITE`) in both
an amnesia control and a mounted-graft run. If the mounted run moves the gold
token above the decoys and improves over control, that is direct evidence that
the cold graft is carrying retrievable information. It is still narrower than
open greedy recall, but it is much faster and sharper for closing the first
GPT-OSS remount question.

That first H5 proof is now green at 4K. GPT-OSS captured a 4096-token real-token
prompt with the `BLUE` needle at token offset 4077, wrote 24 layers of pre-RoPE
K/V graft shards totaling `201326592` host bytes, then cleared the live prompt
down to a 16-token query. The amnesia control preferred `RED` over `BLUE` by
`1.78125` logits. With the 4096-token graft mounted, GPT-OSS preferred `BLUE`
over the best decoy by `3.84375` logits. That is the first same-model GPT-OSS
cold-graft access pass: the answer was absent from the live prompt and became
available through the mounted graft. It is still a candidate-logit gate rather
than open greedy recall, but it closes the first mechanical question: the
GPT-OSS pre-RoPE capture/remount path can carry retrievable information.

The H5 gate has now scaled to 16K real tokens. The capture prompt held the same
single-token `BLUE` needle after 16365 real corpus-fill tokens, then wrote 24
layers of pre-RoPE K/V graft shards totaling `805306368` host bytes. The live
query remained only 16 tokens. The amnesia control again failed in the right
direction, preferring `RED` over `BLUE` by `1.78125` logits. With the 16K graft
mounted, GPT-OSS preferred `BLUE` over the best decoy by `5.09375` logits.

This makes the remount story materially stronger: the same candidate-gate
method works at 4K and 16K, the graft payload scales linearly with token count,
and the mounted query path stays cheap because it is not replaying the original
capture prompt through MoE. The result should still be named precisely. It is a
same-model cold-KV candidate-logit access pass, not H6 open greedy recall and
not a turn-50 continuity test. The next hard proof is a greedy/open recall gate
using the mounted graft as the only source of the fact.

That greedy gate now has a first narrow pass, with useful failures around it.
Raw one-token greedy did not recall the fact: the no-graft control generated
` The`, while the mounted run generated ` ...`. Harmony-formatted greedy was
protocol-correct but entered the default medium-reasoning analysis channel:
`<|channel|>analysis<|message|>We need to`. Those are failures for direct
answer recall, not hidden successes.

The pass came from a protocol-aware forced-final probe. The live prompt ended
at `<|channel|>final<|message|>`, the answer itself was still absent, and no
candidate list was supplied. In the no-graft control, the first open greedy
token was `I`. With the 16K pre-RoPE graft mounted, the first open greedy token
was `BLUE`, with `blue` and `Blue` as the next two logits. That is the first
GPT-OSS 16K same-model cold-graft greedy recall pass.

The boundary remains important. This is stronger than candidate scoring because
the model selected the answer as top-1 without a candidate list, but it is a
constrained final-channel test. It does not yet prove unconstrained raw
free-form generation, and it does not prove multi-turn turn-50 continuity with
KV cleared on every turn.

The turn-50 variant now passes under the same forced-final protocol. The live
prompt explicitly said `Conversation turn 50` and `You do not have previous chat
context`; it contained 56 live tokens and no answer text. The no-graft control
again generated `I`. With the same 16K graft mounted, GPT-OSS generated `BLUE`
as greedy top-1. This is not a 50-step sequential dialogue transcript, but it
does exercise the core GRM operating law: short live prompt, cleared prior
context, mounted cold K/V as the only source of the fact.

The H5 candidate-logit gate has now scaled to the confirmed 64K operating
point. GPT-OSS captured a `65536`-token real-token prompt built from the local
docs corpus, with the `BLUE` needle at token offset `65517`, then wrote all 24
layers of pre-RoPE K/V graft shards. The graft manifest records `65536` tokens,
24 layers, `[1, 8, 65536, 64]` K/V shapes, and `3221225472` host bytes. Capture
wall time was about `5806s`, which keeps confirming that token-routed MoE is
the expensive part of this stack.

The live query remained only 16 tokens. The no-graft control again failed in
the expected direction, preferring `RED` over `BLUE` by `1.78125` logits. With
the 64K graft mounted, GPT-OSS preferred `BLUE` over the best decoy by `4.625`
logits. That is a same-model cold-KV candidate-logit access pass at 64K:
the answer was absent from the live prompt and became available through the
mounted pre-RoPE graft.

The boundary is unchanged. This extends H5 to 64K, but H6 open greedy recall is
still only proven at the 16K graft point under the forced-final protocol. The
next aligned hard gate is the forced-final greedy recall probe using the 64K
graft, followed by a turn-labeled variant if that passes.

The 64K forced-final greedy gate now has a narrow pass. With the same 43-token
live prompt used for the 16K forced-final probe, the no-graft control generated
`I`; with the 64K pre-RoPE graft mounted, GPT-OSS generated `BLUE` as greedy
top-1. The margin was small: `BLUE` logged `17.75`, while `The` was next at
`17.625`. This matters because it is a real open-greedy recall pass without a
candidate list, but it is not a wide-margin result.

The 64K turn-labeled variant did not pass. The live prompt explicitly said
`Conversation turn 50` and `You do not have previous chat context`, with no
answer text present. The no-graft control again generated `I`, but the mounted
64K graft generated `The` as greedy top-1. `BLUE` was present in the mounted
top-k at rank 4 with logit `16.0`, so the graft was still influencing the
distribution, but not enough to win the first token under that prompt.

The current 64K state is therefore precise: H5 candidate-logit access passes,
plain forced-final H6 greedy recall passes narrowly, and the turn-labeled
forced-final H6 variant fails top-1. That split is useful. It suggests the
retrieved graft signal is real at 64K, but prompt/protocol priors can still
overpower it when the live prompt adds more instruction context.

A small mounted prompt-sensitivity sweep sharpened that finding. The plain
prompt still returns `BLUE`. The original turn-50 wording returns `The` with
`BLUE` at rank 4, and a compact turn-50 wording also returns `The` with `BLUE`
at rank 3. A stricter turn-50 wording, `Return exactly the stored GPT-OSS vault
keyword. No sentence. No punctuation.`, returns `BLUE` as top-1 with logit
`20.375`; its no-graft control returns `GPT`, not the keyword.

So the 64K conclusion is not "turn 50 works" or "turn 50 fails" as a single
bit. The graft is carrying the fact at 64K, and a tight forced-final retrieval
prompt can extract it even with turn-50/cleared-context wording. But more
generic live wording leaves enough prior probability on answer-like prose that
`The` wins. That is now a concrete reliability target rather than an unknown:
strengthen routing/mount signal, improve recall prompting policy, or test
whether additional/more local grafts reduce the prompt-sensitivity gap.

H7 is now explicit. GPT-OSS should stay in the stack as a research track, not
as the current production default. Qwen3.5-9B still owns the practical serving
slot: lower resident memory than its baseline, ready-to-work acceptance, warm
GRM prefix economics, and much better user-facing speed. MiniCPM3 remains the
cleanest APA/MLA context-extension proof, DeepSeek-V2-Lite remains the strongest
latent-cache quantization result, and Gemma4-12B remains the closed negative APA
selection-rule case.

GPT-OSS earns continued work because it adds a result the others do not have in
this form: 64K same-model cold-KV graft recall on the 4070 Super for a
20B-class MoE with packed MXFP4 experts. The cost is high and the recall is
prompt-sensitive, so the next GPT-OSS work should be reliability and economics,
not another celebratory context rung. The concrete next gates are multi-fact
H6, ordinary-turn prompt robustness, smaller or nearer graft tests, and
faster MoE/mount execution through the native/CUDA arena path.

The first multi-fact H6 gate is now green at the 4K shakeout point. One
real-token capture planted four independent one-token facts: `BLUE`, `EMBER`,
`GRAY`, and `IRON`. GPT-OSS captured all 24 layers into a `4096`-token
pre-RoPE graft manifest with `201326592` host bytes. Each fact was then queried
with a strict forced-final live prompt, once without a graft and once with the
same cold graft mounted.

The controls missed every target: `GPT`, `GPT`, `Blue`, and `metal` were the
four no-graft top-1 tokens. The mounted runs returned all four planted answers
as greedy top-1 with answer rank `0`. That makes the result a real same-model
cold-KV multi-fact recall pass, not just a candidate-logit access result and
not just another repetition of the single `BLUE` keyword.

The boundary is equally important. This does not yet prove long-context
multi-fact reliability, correction/supersession memory, preference recall, or
ordinary conversational prompting. It does prove that the GPT-OSS graft path can
carry more than one independent binding at once under a strict extraction
prompt. The next useful movement is to scale this gate upward and then make the
prompt less laboratory-clean.

Scaling the same multi-fact gate to 16K found the first real multi-binding
failure. The capture itself succeeded: `16384` real tokens, 24 pre-RoPE layers,
and `805306368` host bytes in the graft manifest. Three of four facts recalled
as mounted greedy top-1: `EMBER`, `GRAY`, and `IRON`. The first fact,
`BLUE`, failed top-1 because the mounted run returned `EMBER`; `BLUE` was still
rank 1.

That failure is useful because it is not an OOM, not a missing-graft case, and
not a candidate-only artifact. It looks like cross-fact competition or
addressing weakness. A quick prompt diagnostic against the same 16K graft
confirmed that interpretation: asking for the generic `CLASSIFIED FACT A
answer` still lost to prose, but asking `In CLASSIFIED FACT A, what is the
GPT-OSS vault keyword?` returned `BLUE` top-1 with no-graft control returning
`g`, not the keyword.

So the current GPT-OSS H6 reliability picture is now sharper. The graft can
carry multiple facts at 16K, but the extraction policy is not production-robust.
The next work should not be another larger context rung until addressing is
improved. Better tests are fact-local addressing, metadata-assisted retrieval
prompts, local/smaller graft selection, and correction/supersession cases where
cross-binding errors are expected to become more visible.

The first addressing fix now has a receipt. A reusable fact-local addressing
gate reads the failed 16K multi-fact artifact, reuses the same graft, and asks
each fact with both its classified fact marker and its semantic slot. That turns
the previous `3/4` into `4/4`: `BLUE`, `EMBER`, `GRAY`, and `IRON` all return
as mounted greedy top-1, while the no-graft controls still miss.

This is the important GRM policy result from the failure. GPT-OSS did not lose
the `BLUE` fact at 16K; the original prompt did not isolate the binding well
enough. When the live query says `In CLASSIFIED FACT A, what is the GPT-OSS
vault keyword?`, the same graft returns `BLUE` with logit `22.0`, above `EMBER`
at `19.0`. For multi-binding GPT-OSS GRM, fact-local addressing metadata is not
nice-to-have. It is part of the retrieval contract.

The next reliability work should build on that contract: store stable fact
markers/semantic slots with graft metadata, test the policy in ordinary
conversational wording, and then move to correction/supersession memory where
the model must choose the current binding rather than the nearest or loudest
old one.
