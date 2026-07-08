# GPT-OSS-20B H7 Existing-Model Comparison

Created: 2026-07-07
Branch: `codex/intn-model-ppl-sweep`

This document closes Phase H7 of
`docs/GPT_OSS_20B_HOUSE_RULES_EXECUTION_PLAN.md` against the current local
evidence. It does not replace the immutable GPT-OSS plan or the ledger. It is a
decision artifact: whether GPT-OSS deserves more optimization time compared
with the local Qwen, Gemma, DeepSeek, and MiniCPM operating points.

## Decision

Keep GPT-OSS-20B as a research track, not as the current production default.

The defensible new result is same-model cold-KV recall at 64K on the 4070 Super
with a 64K pre-RoPE graft. That is a real research result because the live query
does not replay the 64K prompt, and the no-graft controls do not recover the
hidden keyword. The practical caveat is also real: capture is slow, prompt
sensitivity is visible, and the current MoE path is not production-fast.

Qwen3.5-9B remains the best local production-style operating point. MiniCPM3-4B
remains the cleanest APA context-extension proof. DeepSeek-V2-Lite remains the
best MLA latent-cache quantization result but is still blocked by prefill
transients for high-context recall. Gemma4-12B is a closed negative APA result.

## Comparison Table

| Model / track | Memory footprint | Usable context evidence | PPL / behavior quality | Cold-KV / GRM recall | Runtime / complexity | H7 decision |
| --- | ---: | --- | --- | --- | --- | --- |
| GPT-OSS-20B TensorCUDA / APA / GRM | Packed MXFP4 expert path plus streamed layer residency; 64K graft payload `3221225472` host bytes | H4 real-token APA passed 64K; H5 graft capture/remount passed 4K/16K/64K; no 128K proof yet | Real-text PPL gates exist; H6 quality is prompt-sensitive | 64K H5 candidate access passes; 64K plain forced-final H6 greedy recall passes narrowly; generic turn-50 wording fails top-1; strict turn-50 wording passes with no-graft control | Capture wall around `5806s` at 64K; mounted query around tens of seconds; MoE routing is the wall | Keep for research; optimize only if MoE/mount economics can improve |
| Qwen3.5-9B APA + GRM | `~4.6GB` resident VRAM in TensorCUDA; baseline Ollama `~6.3GB` | Practical mounted context estimated `~80-100K` on 8GB via state/prefix economics | Final gates pass; ready-to-work gate `10/10` templates in `38s`; decode `38 tok/s` | GRM prefix mounting restores `643/699` tokens by request 3 in the serving workload | Best local serving economics: warm request about `8.8s`, decode `~27.8 tok/s` in workload | Current production-style winner |
| MiniCPM3-4B MLA | INT4 weights about `2.2GB`; loaded `2783MB` on 8GB | APA bulk4/refine0.10 validates full trained `32768` window; residency `2846-2856MB` across 2K to 32K | tensor_cuda INT4 PPL `20.065` vs PyTorch CPU bf16 `17.357`, better than bnb `22.507`; APA bulk4 PPL `19.817` | Arena impact recorded on MiniCPM3 path; recall unchanged in cited speed pass, but not the same GPT-OSS 64K cold-KV gate | Decode speed optimized to `21.6ms/token` / `46.3 tok/s` on 3070 fast stack | Best clean APA/MLA context law proof |
| DeepSeek-V2-Lite MLA latent INT4 | Latent cache drops from `31104 B/token` to `11232 B/token`; INT4 latent reaches configured max allocation/decode rungs | Resident latent allocation/decode passed 64K/96K/128K/163840, but high-context prefill recall hit `_quantize_keys` transient OOM | Six-window PPL: latent4 adds about `+0.42%` to `+0.51%`; open-gen 2K/4K/8K `9/9` | 8K forced-choice and open-gen pass; 32K+ recall not proven due prefill blocker | Strong MLA quantization result; implementation blocker is prefill transient, not resident cache | Keep as latent-quant research; not a GPT-OSS replacement for 64K recall yet |
| Gemma4-12B APA extension | 4070S has full 32K window headroom; APA does not free useful peak VRAM | Standard and APA both hold full `32768` trained window; no context-ceiling win | APA costs PPL `+1.55-1.92%` and runtime; negative result | Gemma adapter/GRM path has green readiness elsewhere, but APA mission closed as not useful on 12B | APA is slower, higher peak VRAM, and MQA geometry is anti-APA | Closed negative APA result; do not spend GPT-OSS-style APA time here |

## Why GPT-OSS Still Matters

GPT-OSS is not winning on speed. The current capture path is dominated by
token-routed MoE, and the mounted query path is still slower than the Qwen3.5
serving stack. It matters because it adds a different research point:

- 20B-class MoE with resident packed MXFP4 experts can run on the 4070 Super
  through the local streamed TensorCUDA harness.
- APA can operate on the full-attention half of the model while sliding layers
  remain bounded.
- Same-model pre-RoPE GQA grafts can carry a fact across a 64K capture into a
  short live prompt with KV cleared.
- The 64K recall result is prompt-sensitive, giving a concrete next reliability
  target rather than a vague failure.

## Current Winner By Axis

- Production serving: Qwen3.5-9B.
- APA context-extension law: MiniCPM3-4B MLA.
- Latent-cache quantization law: DeepSeek-V2-Lite MLA.
- Negative selection rule / anti-APA example: Gemma4-12B.
- 64K same-model cold-KV graft recall on the 4070 Super: GPT-OSS-20B.

## Next GPT-OSS Work If Continued

1. Build a stronger H6 reliability gate with multiple facts and multiple prompt
   styles, not only the single `BLUE` keyword.
2. Test whether nearer or smaller grafts reduce the 64K prompt-sensitivity gap.
3. Improve MoE route/expert execution; otherwise 64K capture remains a
   multi-hour operation.
4. Compare mounted-query economics after any C++/CUDA arena integration rather
   than only through the Python streamed harness.
5. Only attempt 128K if the reliability gate justifies the compute time.

## H7 Verdict

GPT-OSS deserves more optimization time only as a research model for
MoE+APA+GRM and long cold-KV recall. It should not displace Qwen3.5-9B as the
practical local GRM serving target until the following change:

- 64K recall becomes robust across ordinary turn prompts, not just strict
  retrieval prompts; and
- mounted-query/runtime economics improve enough that the MoE cost is not the
  dominant user-facing behavior.
