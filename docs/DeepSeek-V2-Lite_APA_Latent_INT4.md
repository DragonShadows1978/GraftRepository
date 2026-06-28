# DeepSeek-V2-Lite APA + MLA Latent INT4 Results

Date: 2026-06-27

## Verdict

DeepSeek-V2-Lite's MLA latent cache tolerates INT4 storage well enough for
the full configured YaRN resident cache plus one decode token to fit on the
12GB card. That is a cache/decode result, not proof that every prefill path can
ingest a real 163K-token prompt.

The first real-token prefill fix is now landed in the local DeepSeek loader:
`attention_mode="absorbed_exact"` keeps prefill in MLA latent space instead of
expanding the accumulated latent cache through `kv_b(c_n)` every chunk. This is
not the final fused latent-APA kernel, but it proves the optimization diagnosis:
the old expanded APA prefill ceiling was `16,384`; absorbed exact prefill has
now passed real-token `65,536` on the same 12GB card.

The first high-context open-ended greedy needle ladder is stricter than
fill-only. Absorbed exact latent4 hit exact planted codes at `16,384`, `32,768`,
and `49,152` tokens, but OOMed at `65,536` during the greedy test.

The same run now includes an explicit INT2 floor check. A quality-only
group-32 affine INT2 fake-store path does not catastrophically destroy the
model, but it is no longer in the "free" zone: six-window Wikitext PPL rises
by about `+5.8%` versus APA latent16, and the forced-choice needle grid gets
one 8K retrieval rank flip.

This is not a no-prior-art claim. DeepSeek's own V2 paper reports that their
deployed DeepSeek-V2 stack uses KV-cache quantization to compress each KV-cache
element to **6 bits on average**, on top of MLA and FP8 weights. That is the
nearest direct prior art for this line of work.

Our contribution in this run is narrower and lower-level:

- reproduce the latent-cache quantization idea in the local
  `tensor_cuda`/APA DeepSeek-V2-Lite port,
- push the MLA resident latent to **INT4 group-32** rather than 6-bit average,
- measure the PPL and forced-choice retrieval cost on this exact stack, and
- show that a synthetic `163840`-row resident cache plus one decode token fits
  on the 12GB card.

Also note the quantizer caveat: this result is not a generic statement about
"any 4-bit runtime." Prior repo measurements show the local `tensor_cuda`
INT4 stack can be materially better than the default PyTorch bitsandbytes path.
On MiniCPM3-4B, same card and protocol, `tensor_cuda INT4` measured
`ppl@1024 = 20.065` while PyTorch GPU `bnb-4bit` measured `22.507`; the
`tensor_cuda` path also held a larger standard-attention context ceiling
(`3072` vs `2048`). That makes the DeepSeek result a property of this
model/MLA representation **and** this quantization/kernel stack.

The important result is not "V-only" quantization. In DeepSeek MLA there is
no separable resident V cache. The resident decode state is:

- `c_n`: the post-norm MLA latent, shape `(B, S, kv_lora_rank=512)`
- `k_pe`: the roped shared key, shape `(B, 1, S, qk_rope_head_dim=64)`

Absorbed MLA decode uses `c_n` for both:

- key-side scoring, via `(q_nope W_k) dot c_n`
- value-side accumulation, via `softmax(scores) @ c_n`, then projection by
  `W_v`

So the memory-saving experiment quantizes the resident latent itself, not a
separate V tensor. `k_pe` remains bf16.

## Implementation

Code path:

- `core/deepseek_v2_lite_tc.py`
- probe: `/home/vader/deepseek_apa_oom_probe.py`
- PPL runner: `/home/vader/deepseek_latent_ppl.py`
- needle runner: `/home/vader/deepseek_needle_forced_choice.py`

Main implementation points:

- Added mutable `DeepSeekMLACache`.
- Added optional latent cache storage mode:
  - `latent_bits=16`: bf16 latent buffer
  - `latent_bits=4`: `tc.kv_int4_pack` packed latent bytes plus bf16 group
    scales, group size 32
  - `latent_bits=2`: experimental group-32 affine fake-store, dequantized
    back into bf16 cache storage; this tests quality damage only, not memory
    savings
- Added model switches:
  - `set_latent_cache_bits(bits, group=32)`
  - `set_value_int4(enabled=True, group=32)`
- Decode uses absorbed MLA instead of expanding full per-head K/V for the
  whole cache.
- Added `attention_mode="absorbed_exact"` for real prompt prefill. This path
  computes:
  - `q_lat = q_nope @ W_k`
  - `scores = q_lat @ c_n.T + q_pe @ k_pe.T`
  - `ctx_lat = softmax(scores) @ c_n`
  - `out = ctx_lat @ W_v`
  It avoids full-span `kv_b(c_n)` expansion. It still materializes an `L x S`
  score matrix per chunk, so chunk size remains a real transient-memory lever.

The group-32 INT4 latent layout stores, per token per layer:

| Component | bf16 latent cache | INT4 latent cache |
|---|---:|---:|
| latent payload | `512 * 2 = 1024 B` | `512 / 2 = 256 B` |
| latent scales | none | `(512 / 32) * 2 = 32 B` |
| roped `k_pe` | `64 * 2 = 128 B` | `64 * 2 = 128 B` |
| total / layer / token | `1152 B` | `416 B` |
| total / token / 27 layers | `31,104 B` | `11,232 B` |

Net resident cache compression: about `2.77x`.

## Cache/Decode Wall

These probes used DeepSeek-V2-Lite with post-hoc INT4 weights in
`tensor_cuda`, APA r0.15 bulk4, synthetic cache allocation, and one absorbed
MLA decode token. They measure resident-cache/decode capacity only. They do
not measure full prompt prefill.

BF16 latent cache:

- 64K cached tokens passed: after load `9112 MB`, after cache `11164 MB`,
  after decode `11340 MB`.
- 96K cached tokens failed during synthetic cache allocation around layer 21.

INT4 latent cache:

| Cached tokens | After load | After cache | After decode | Result |
|---:|---:|---:|---:|---|
| 64K | `9142 MB` | `9844 MB` | `10020 MB` | OK |
| 96K | `9142 MB` | `10222 MB` | `10430 MB` | OK |
| 128K | `9140 MB` | `10544 MB` | `10752 MB` | OK |
| 163840 | `9140 MB` | `11030 MB` | `11238 MB` | OK |

The INT4 latent cache reaches the model's configured YaRN maximum
(`163840`) with about 1GB of visible VRAM still unused after one decode token.

## Prefill Wall

The overnight high-context needle run showed that the original expanded APA
prefill path does **not** reach the cache/decode limits on real prompts.

Run directory:

`/home/vader/deepseek_needle_runs/overnight_20260627_024912`

Observed behavior:

- Open-gen greedy needle at `2K/4K/8K` passed:
  - APA latent16: `9/9`
  - APA latent4: `9/9`
- Forced-choice latent16 high-context run OOMed on the first `16K` item before
  any score.
- Forced-choice latent4 high-context run completed `16K` at `3/3`, then OOMed
  on the first `32K` item.

The OOM was in the prefill-time APA key quantization path, not in the resident
latent cache:

```text
_quantize_keys(...)
recon = tc.matmul(centroids, Rb) * norms
RuntimeError: cudaMallocAsync failed: out of memory
```

So the current state is:

- cache/decode storage: INT4 reaches the synthetic 163840-token cache rung
- expanded APA real prompt prefill: not solved past 16K
- absorbed exact real prompt prefill: solved through at least 65K

### Real Fill Ceiling - Expanded APA vs Absorbed Exact

Protocol:

- real Wikitext/local-doc token stream, no repeated-token synthetic cache rows
- exact-length prompts
- chunked prefill
- finite-logit check plus short greedy decode smoke
- `latent_bits=4`, group 32

Old expanded APA path, `attention_mode="apa_selective"`, `chunk=256`:

| Context | Result | Prefill | VRAM after prefill/decode |
|---:|---|---:|---:|
| 2,048 | OK | `16.0s` | `9,537 MiB` |
| 4,096 | OK | `32.4s` | `9,953 MiB` |
| 8,192 | OK | `118.6s` | `10,625 MiB` |
| 12,288 | OK | `240.7s` | `11,169 MiB` |
| 16,384 | OK | `399.7s` | `11,617 MiB` |
| 24,576 | OOM | `479.8s` | `11,651 MiB at failure` |

New absorbed exact path, `attention_mode="absorbed_exact"`:

| Context | Chunk | Result | Prefill | VRAM after prefill/decode |
|---:|---:|---|---:|---:|
| 2,048 | 256 | OK | `15.5s` | `9,537 MiB` |
| 4,096 | 256 | OK | `31.0s` | `9,633 MiB` |
| 8,192 | 256 | OK | `63.7s` | `10,145 MiB` |
| 12,288 | 256 | OK | `99.4s` | `10,276 MiB` |
| 16,384 | 256 | OK | `132.3s` | `10,563 MiB` |
| 24,576 | 256 | OK | `204.3s` | `10,787 MiB` |
| 32,768 | 256 | OK | `275.1s` | `11,107 MiB` |
| 49,152 | 256 | OOM | `383.0s` | `11,523 MiB at failure` |
| 49,152 | 128 | OK | `447.4s` | `11,363 MiB` |
| 65,536 | 128 | OK | `621.7s` | `11,715 MiB` |
| 98,304 | 64 | OOM | `891.5s` | `11,713 MiB at failure` |

Branch counters confirmed the new path used absorbed MLA prefill only:

```text
absorbed_exact_prefill > 0
apa_cublas_blend = 0
apa_fused_selective = 0
standard_sdpa = 0
```

The 49K result shows the remaining wall for the first-pass fix: `chunk=256`
OOMs because absorbed exact still materializes an `L x S` score matrix. Halving
the chunk to 128 passes. The 98K chunk64 failure shows that the first-pass
absorbed exact path is still score-transient bound near the top of this card.
The next non-kernel improvement is adaptive prefill chunking; the next kernel
improvement is fused latent APA that keeps the absorbed MLA math but does not
materialize the full score matrix.

## Open-Ended Greedy Context Ladder

Protocol: real-token filler, one planted numeric code, greedy generation answer,
exact-match check. This is a stricter gate than cache allocation or prefill-only
fill because it exercises the long prompt and the answer generation path.

Command:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda:/home/vader PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 /home/vader/deepseek_real_context_ladder.py --contexts 16384,32768,49152,65536 --depths 0.5 --n-per-cell 1 --settings absorbed4 --chunk 128 --max-new 16 --refine 0.15 --bulk-bits 4 --stop-on-oom
```

| Context | Depth | Result | Expected | Generated | Elapsed |
|---:|---:|---|---|---|---:|
| 16,384 | `0.5` | HIT | `81-4520` | `81-4520` | `122.8s` |
| 32,768 | `0.5` | HIT | `67-8619` | `67-8619` | `258.6s` |
| 49,152 | `0.5` | HIT | `38-3329` | `38-3329` | `409.7s` |
| 65,536 | `0.5` | OOM | `51-9776` | `<OOM:RuntimeError: cudaMallocAsync failed: out of memory>` | `539.3s` |

Summary: `absorbed exact latent4 real_ctx_acc 3/4 = 75.0%`
(`[16k:1/1 32k:1/1 48k:1/1 64k:0/1]`).

Evidence:
`/home/vader/deepseek_needle_runs/absorbed4_greedy_real_20260627_65k_middepth`

## PPL Checks

Dataset: cached local Wikitext-2 raw test set.

Protocol: teacher-forced decode scoring with `step=1`, so the measured path
uses the resident MLA cache and the absorbed decode branch. This is the
right path for testing latent cache quantization.

### Short triage

`window=192`, `scored=64`, `windows=1`, 63 scored tokens:

| Mode | NLL | PPL |
|---|---:|---:|
| standard latent16 | `1.698767` | `5.4672` |
| APA r0.15 latent16 | `1.714028` | `5.5513` |
| APA r0.15 latent4 | `1.741862` | `5.7080` |
| APA r0.15 latent2 | `1.807146` | `6.0930` |

This small run was noisy; latent4 looked worse by `+2.82%` PPL versus APA
latent16. Latent2 was already visibly worse at `+9.76%` PPL ratio versus APA
latent16.

### Single 511-token scored pass

`window=640`, `scored=512`, `windows=1`, 511 scored tokens:

| Mode | NLL | PPL |
|---|---:|---:|
| standard latent16 | `1.645527` | `5.1837` |
| APA r0.15 latent16 | `1.647907` | `5.1961` |
| APA r0.15 latent4 | `1.651350` | `5.2140` |
| APA r0.15 latent2 | `1.703844` | `5.4950` |

Latent4 delta versus APA latent16:

- NLL delta: `+0.003443`
- PPL ratio: `1.003449`
- PPL hit: about `+0.345%`

Latent2 delta versus APA latent16:

- NLL delta: `+0.055937`
- PPL ratio: `1.057531`
- PPL hit: about `+5.753%`

### Six-window 1024/512 protocol

`window=1024`, `scored=512`, `windows=6`, 3066 scored tokens:

| Mode | NLL | PPL |
|---|---:|---:|
| standard latent16 | `2.123933` | `8.3640` |
| APA r0.15 latent16 | `2.125756` | `8.3792` |
| APA r0.15 latent4 | `2.129944` | `8.4144` |
| APA r0.15 latent2 | `2.182210` | `8.8659` |

Latent4 delta versus APA latent16:

- NLL delta: `+0.004188`
- PPL ratio: `1.004197`
- PPL hit: about `+0.420%`

Latent2 delta versus APA latent16:

- NLL delta: `+0.056454`
- PPL ratio: `1.058078`
- PPL hit: about `+5.808%`

### Six-window 2048/1024 protocol

`window=2048`, `scored=1024`, `windows=6`, 6138 scored tokens:

| Mode | NLL | PPL |
|---|---:|---:|
| APA r0.15 latent16 | `1.783484` | `5.9506` |
| APA r0.15 latent4 | `1.788551` | `5.9808` |
| absorbed exact latent16 | `1.782938` | `5.9473` |
| absorbed exact latent4 | `1.788475` | `5.9803` |

Latent4 delta versus APA latent16:

- NLL delta: `+0.005068`
- PPL ratio: `1.005080`
- PPL hit: about `+0.508%`

Absorbed exact latent4 delta versus APA latent16:

- NLL delta: `+0.004992`
- PPL ratio: `1.005004`
- PPL hit: about `+0.500%`

Absorbed exact latent4 is effectively tied with expanded APA latent4 on this
protocol (`5.9803` vs `5.9808`), while providing the much higher real-token
prefill ceiling described above.

## Needle Check

Protocol: forced-choice retrieval, not sampled generation.

Each item plants a fact:

`hidden color for <key> is <value>`

Then asks for the value. The correct value and three decoys are all
single-token candidates, so scoring is one clean next-token logits row.

Grid:

- context lengths: `2048`, `4096`, `8192`
- depths: `0.1`, `0.5`, `0.9`
- one item per cell, 9 items total

| Mode | Accuracy | Mean margin | Min margin |
|---|---:|---:|---:|
| APA r0.15 latent16 | `9/9` | `+6.924` | `+2.250` |
| APA r0.15 latent4 | `9/9` | `+6.632` | `+2.875` |
| APA r0.15 latent2 | `8/9` | `+6.535` | `-1.188` |

By context:

| Mode | 2K | 4K | 8K |
|---|---:|---:|---:|
| APA r0.15 latent16 | `3/3` | `3/3` | `3/3` |
| APA r0.15 latent4 | `3/3` | `3/3` | `3/3` |
| APA r0.15 latent2 | `3/3` | `3/3` | `2/3` |

For the INT4 comparison, no retrieval rank flips were observed. Margins
changed, but not in a collapse pattern; one low-margin 8K case improved
slightly under latent4.

Latent2 did produce one rank flip at 8K/depth 0.9:

- correct value: `mango`
- top decoy: `amber`
- rank: `2`
- margin: `-1.188`

So the floor is not a total recall collapse, but it is no longer retrieval
parity.

## Interpretation

This challenges the useful field assumption one step further than the published
DeepSeek deployment point. The prior art already establishes that MLA KV-cache
quantization is viable at about 6 bits/element in deployed DeepSeek-V2. The
new observation here is that the 4-bit bulk law appears to extend beyond APA
key-score bulk to the resident MLA latent cache itself, at least for
DeepSeek-V2-Lite.

The conservative claim:

- DeepSeek-V2-Lite's MLA latent cache can be stored at INT4 group-32 with
  about a half-percent Wikitext PPL hit in the tested decode protocol.
- The same mode preserves a small forced-choice retrieval grid through 8K.
- The memory win is large enough to move the practical decode cache wall from
  failing around 96K bf16-latent cache to passing synthetic resident-cache
  allocation plus one-token decode at the configured 163840-token window.
- The old expanded prefill wall is fixed for absorbed exact through 65K
  fill-only, but usable open-ended greedy recall is only verified through 49K
  in the first high-context real-token ladder; 65K OOMed during the greedy
  test.
- INT2 group-32 affine fake-store is a useful broken-floor control: it costs
  about `+5.8%` PPL and creates an 8K retrieval miss in the tested grid.

What is not proven yet:

- novelty of latent/KV-cache quantization as a general idea
- that any off-the-shelf 4-bit quantizer will match this result
- open-ended generation recall parity beyond the 49K hit in the first
  high-context ladder
- long-context retrieval at 16K/32K/49K/64K/128K across full depth sweeps
- a memory-flat real-prompt prefill path that reaches the resident-cache limit
- multi-needle interference
- cross-model latent-cache quantization law
- whether a real packed INT2 kernel with a different quantization scheme or
  hybrid hot-tail policy can recover enough quality to be useful
- whether group-64 or group-128 scales are still safe

## Next Tests

1. Add adaptive absorbed chunking and/or fused latent APA to remove the current
   score/transient wall; fill-only reaches 65K, but 65K greedy recall OOMs.
2. Repeat the forced-choice needle grid at 16K, 32K, 64K, and 128K.
3. Add multiple simultaneous needles to test interference, not just point
   recall.
4. Run open-ended greedy decode recall as a full depth sweep at 16K, 32K, 49K,
   and 64K after the 65K greedy OOM wall is reduced.
5. Sweep latent groups: 16, 32, 64, 128.
6. Try hybrid hot-tail storage: recent rows bf16, old rows INT4.
7. Treat INT2 latent as the broken-floor control unless a real packed INT2
   kernel plus hybrid policy materially improves the result.

## External Prior Art

- DeepSeek-V2 paper, Section 3.2.3, "Inference Efficiency": deployed
  DeepSeek-V2 uses FP8 weights and KV-cache quantization, compressing each
  KV-cache element to 6 bits on average:
  <https://arxiv.org/html/2405.04434>
- The same paragraph cites KVQuant and Atom as related KV-cache / low-bit
  serving references:
  - KVQuant: <https://arxiv.org/abs/2401.18079>
  - Atom: <https://arxiv.org/abs/2310.19102>
- Local quantizer baseline: `MiniCPM3-MLA_Results.md` records
  `tensor_cuda INT4` beating PyTorch GPU `bnb-4bit` on PPL
  (`20.065` vs `22.507`) and standard-attention context ceiling
  (`3072` vs `2048`) under the same local evaluation setup.
