# ORDER QWEN38-A1 receipt — Qwen3.8-27B INT3 tensor_cuda adapter

Date: 2026-08-15 (America/Detroit)

Claim boundary: implementation and parity-gate receipt only. No model-quality
claim is made. G3 outputs remain characterization receipts when a GPU run is
available.

## Files created or modified

- Modified: `/mnt/ForgeRealm/GraftRepository/core/qwen35_tc.py`
- Created: `/mnt/ForgeRealm/GraftRepository/core/qwen38_tc.py`
- Created: `/mnt/ForgeRealm/GraftRepository/scripts/qwen38_generate.py`
- Created: `/mnt/ForgeRealm/GraftRepository/scripts/qwen38_gates.py`
- Created: `/mnt/ForgeRealm/GraftRepository/tests/test_qwen38_cpu.py`
- Created: `/mnt/ForgeRealm/GraftRepository/docs/QWEN38_PORT_RECEIPT.md`
- Created: `/mnt/ForgeRealm/GraftRepository/artifacts/qwen38_int3_cache/`
  (1 manifest plus 1,204 `.npy` cache arrays; 1,205 files total)

No git write command was run. Project-Tensor, model weights, and the installed
Transformers package were read only.

## Historical A1 `Qwen38Config.from_model_dir` printout (superseded by A3)

```text
QWEN38_CONFIG {"attention_layers": [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63], "conv_kernel": 4, "d_k": 128, "d_v": 128, "deltanet_head_ratio": 3, "eos_token_id": 248044, "full_attention_interval": 4, "gqa_ratio": 6, "head_dim": 256, "hidden_act": "silu", "hidden_dim": 5120, "intermediate_dim": 17408, "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention"], "mamba_ssm_dtype": "float32", "max_position_embeddings": 262144, "model_dir": "/mnt/ForgeRealm/models/Qwen3.8-27B", "n_k_heads": 16, "n_v_heads": 48, "num_heads": 24, "num_kv_heads": 4, "num_layers": 64, "output_gate_type": "swish", "partial_rotary_dim": 64, "rms_norm_eps": 1e-06, "rope_theta": 10000000.0, "tie_word_embeddings": false, "vocab_size": 248320}
```

## Historical A1 output-gate interpretation — REFUTED by A3

A1 incorrectly inferred that checkpoint `output_gate_type="swish"` selected the
full-attention output gate. Lead GPU generation refuted that inference. The A3
resolution and corrected citations are recorded below; this historical section
is retained so the failed interpretation is not silently erased.

## Ratio and config sweep

- DeltaNet 9B ratio 32/16=2: executable expansion was already config-derived as
  `H // cfg.n_k_heads`; it now runs as 48/16=3. The misleading x2-only comment
  was generalized. The installed HF reference does the same division at
  `modeling_qwen3_5.py:519-521`.
- Attention GQA 9B ratio 16/4=4: both standard and APA paths were already
  config-derived as `H // KV`; they now run as 24/4=6. Q/K/V shapes, qk-norm,
  fused per-head `[q|gate]`, repeats, and output flattening all use config dims.
- DeltaNet norm/gate width has no 4096-channel executable assumption: `H*Dv`
  becomes 48*128=6144; `a`, `b`, `A_log`, and `dt_bias` use `H=48`.
- A real silent 9B constant was found outside the requested ratios: causal conv
  state padding and four taps were executable constants 3/4. They now derive
  from `cfg.conv_kernel` at `core/qwen35_tc.py:222-234`; k=4 behavior and
  operation order remain identical for 9B.
- A hardcoded sigmoid attention output gate was isolated behind the existing
  9B method and overridden config-selectably for 27B.
- Parsed-and-consumed fields were swept: vocab/embedding/lm head; hidden/FFN;
  layer count/types/attention fallback interval; attention q/kv heads/head dim;
  theta/partial rotary; DeltaNet k/v heads and dims/conv kernel; RMS epsilon;
  EOS; tied/untied head; hidden activation; output gate; fp32 SSM dtype; maximum
  position setting. `layer_types` is authoritative; interval is only fallback.
- The initial 4096-row RoPE table is an allocation window, not a model constant;
  `extend_rope` grows it from configured theta and partial dimension on demand.

## Quantize-once cache

Command run:

```bash
flock -w 3600 /tmp/forge-gpu.lock timeout 3000 python scripts/qwen38_generate.py --model-dir /mnt/ForgeRealm/models/Qwen3.8-27B --cache-dir artifacts/qwen38_int3_cache --build-cache-only
```

Result:

```text
CACHE_RESULT {"cache_dir": "/mnt/ForgeRealm/GraftRepository/artifacts/qwen38_int3_cache", "elapsed_seconds": 277.86746513296384, "entries": 402, "status": "PASS"}
complete True entries 402 counts {'qlinear': 401, 'host_embedding': 1}
bytes_by {'packed': 9599385600, 'scales': 399974400, 'zeros': 399974400, 'host': 2542796800} total 12942131200 GiB 12.053298950195312
pack_gate {'cases': 4, 'codes_checked': 50688, 'max_byte_diff': 0}
source_files 19
```

The manifest fingerprints bits, group size, config/shard sizes and nanosecond
mtimes. Changed bits/group/source resets the manifest identity. Every qlinear
uses three mmap-loadable arrays. The BF16 embedding is a raw-uint16 host mmap.
Vision and MTP names cannot enter the selected target set. Quantization is
bounded to 1,024 rows at a time and traverses source shard by source shard.

The vector pack is also byte-identical to the engine affine quantizer on a
separate 17x256, five-row-chunk cache test (`packed/scales/zeros` all exact).

## Memory map

No CUDA device was visible, so this is computed exactly from checkpoint/cache
shapes, not presented as a measurement:

```text
PER-COMPONENT MEMORY MAP (GiB; computed from checkpoint shapes)
  int3_body                                     9.2041 GiB  VRAM
  int3_lm_head                                  0.4810 GiB  VRAM
  fp32_norms_qknorm_delta_aux                   0.0978 GiB  VRAM
  bf16_rope_4096                                0.0010 GiB  VRAM
  host_bf16_embedding_ram                       2.3682 GiB  HOST
  fp32_deltanet_state_batch1                    0.1461 GiB  VRAM
  bf16_kv_batch1_seq128                         0.0078 GiB  VRAM
  resident_vram                                 9.7839 GiB  VRAM
  total_vram_batch1_seq128                      9.9378 GiB  VRAM
```

The computed batch-1/128-token total is 1.5622 GiB below the 11.5 GiB order
ceiling. It does not claim allocator/driver overhead is measured; G1 must record
that on the target GPU.

## Gates

### G0 — PASS (CPU)

Command:

```bash
python scripts/qwen38_gates.py --g0
```

Pack gate: 4 cases, 50,688 codes, max byte difference 0.

| case | mixer rel L2 | mixer max abs | block rel L2 | block max abs |
|---|---:|---:|---:|---:|
| random-small layer 0 DeltaNet | 3.161716835133021e-07 | 2.980232238769531e-07 | 3.082156000987091e-07 | 2.980232238769531e-07 |
| random-small layer 3 attention+swish | 2.536298478752779e-07 | 8.940696716308594e-08 | 2.2028064243005206e-07 | 1.4901161193847656e-07 |
| real-weight layer 0 DeltaNet, seq 8 | 6.222185159969209e-07 | 8.58306884765625e-06 | 5.468553683732392e-07 | 6.866455078125e-05 |
| real-weight layer 3 attention+swish, seq 8 | 5.822972921774934e-07 | 5.340576171875e-05 | 5.949565675619561e-07 | 6.103515625e-05 |

All arrays finite; registered relative-L2 threshold 1e-4.

Additional CPU regression run:

```text
PYTHONPATH=. pytest -q tests/test_qwen38_cpu.py tests/test_qwen35_translation_poc.py::test_qwen35_attention_exposes_grm_injection_contract tests/test_qwen35_translation_poc.py::test_qwen35_config_loads_real_2b_and_9b_shapes_when_available
6 passed, 2 warnings in 1.23s
```

### G1 — PENDING-LEAD (no visible GPU)

Visibility error, verbatim:

```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.
```

Exact command (also gates fused INT3 at real projection/lm-head shapes before
enabling `FUSED_DECODE=True`, then loads the already-built cache):

```bash
flock -w 3600 /tmp/forge-gpu.lock timeout 3000 python scripts/qwen38_generate.py --model-dir /mnt/ForgeRealm/models/Qwen3.8-27B --cache-dir artifacts/qwen38_int3_cache --load-only
```

If fused INT3 fails, the loader prints the exception with shape/rel-L2/max-abs,
leaves fused decode disabled, and continues on the two-stage correctness path.
`--skip-fused-gate` exists only as an explicit slow diagnostic fallback and also
disables fused decode. The standalone `--fused-int3` gate raises on failure.

### G2 — PENDING-LEAD (no visible GPU)

```bash
flock -w 3600 /tmp/forge-gpu.lock timeout 3000 python scripts/qwen38_gates.py --model-dir /mnt/ForgeRealm/models/Qwen3.8-27B --cache-dir artifacts/qwen38_int3_cache --g2
```

This compares on-device layers 0 and 3 against the bounded-row CPU oracle using
the dequantized cached INT3 weights.

### G3 — PENDING-LEAD (no visible GPU)

```bash
flock -w 3600 /tmp/forge-gpu.lock timeout 3000 python scripts/qwen38_generate.py --model-dir /mnt/ForgeRealm/models/Qwen3.8-27B --cache-dir artifacts/qwen38_int3_cache --demo --max-new-tokens 64
```

No raw outputs exist yet and none are fabricated.

### G4 — PENDING-LEAD (no visible GPU)

```bash
flock -w 3600 /tmp/forge-gpu.lock timeout 3000 python scripts/qwen38_generate.py --model-dir /mnt/ForgeRealm/models/Qwen3.8-27B --cache-dir artifacts/qwen38_int3_cache --benchmark --max-new-tokens 64
```

The script constructs an exactly 128-token prompt and prints prefill seconds,
decode seconds, and decode tok/s.

## Deviations from the order

1. G1-G4 could not run because no CUDA device/driver was visible. They are
   marked PENDING-LEAD with exact commands; no GPU receipt is simulated.
2. Historical A1 deviation, superseded by A3: G0 replaced the installed sigmoid
   with the incorrectly inferred checkpoint-requested SiLU. A3 removes that
   patch and reruns G0 under sigmoid.
3. The mandatory CPU pack cache was built despite the absent GPU, so its cost is
   already paid. GPU residency, kernel parity, generation, and speed remain
   pending.

---

# ORDER QWEN38-A2 receipt — lm_head OOM fix and generation regates

Date: 2026-08-15 (America/Detroit)

Claim boundary: implementation and CPU regression receipt only. This execution
environment exposed no CUDA device, so no A2 generation quality, performance,
or measured-VRAM claim is made.

## Diff summary of the lm_head fix

- Modified `core/qwen38_tc.py`: `Qwen38_TC.__call__` now explicitly slices the
  normalized final hidden state from `(B, L, 5120)` to `(B, 1, 5120)` before
  `lm_head` whenever `last_token_only=True`.
- Modified `core/qwen38_tc.py`: added
  `PackedRowChunkedQuantLinearTC`. It partitions the mmap-backed packed INT3
  `lm_head` arrays by output row before upload and keeps eight packed device
  chunks resident. Every chunk invokes the validated two-stage
  `tc.intn_linear` separately, and the eight logits pieces are concatenated on
  the vocabulary axis. It never materializes a full BF16 lm_head.
- Default chunk size: 31,040 rows. For the 248,320-row head this is exactly
  eight chunks. Each BF16 dequantized weight transient is 317,849,600 bytes =
  317.850 MB = 303.125 MiB. Configure with
  `--lm-head-chunk-rows` or `QWEN38_LM_HEAD_CHUNK_ROWS`.
- Modified `scripts/qwen38_generate.py`: the known-defective fused INT3 path is
  disabled by default; generation has finite-logit checks, synchronized timing,
  a sigmoid output-gate diagnostic override, and phase-tagged 0.25-second
  `nvidia-smi` sampling plus explicit load/prefill/decode samples.
- Modified `scripts/qwen38_gates.py`: successor G2b is registered with per-layer
  cosine >= 0.999 for layers 0 and 3; relative L2 remains reported alongside.
- Modified `tests/test_qwen38_cpu.py`: tests cover the eight-row-chunk plan,
  last-token slicing before the head, and exactly one two-stage INT3 call per
  packed chunk.

CPU regression result:

```text
.......                                                                  [100%]
7 passed, 2 warnings in 1.18s
```

## GPU availability blocker (verbatim)

Required lock/timeout visibility command:

```bash
flock -w 3600 /tmp/forge-gpu.lock timeout 3000 nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
```

Output and exit code:

```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.

exit code 9
```

Independent locked TensorCUDA allocation probe:

```text
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/mnt/ForgeRealm/Project-Tensor/tensor_cuda/tensor_cuda/__init__.py", line 48, in tensor
    return _C.tensor(arr, device, requires_grad)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: cudaMalloc failed: no CUDA-capable device is detected
```

## Peak VRAM receipt

- Load peak: PENDING-LEAD; no GPU sample exists.
- Prefill peak: PENDING-LEAD; no GPU sample exists.
- Decode peak: PENDING-LEAD; no GPU sample exists.

The sampler prints `VRAM_RECEIPT` with `peak_mib` keys `load`, `prefill`, and
`decode`; phase-specific prompt peaks are retained in `phase_peak_mib`.

## Gates

### G1 — PENDING-LEAD

```bash
flock -w 3600 /tmp/forge-gpu.lock timeout 3000 python scripts/qwen38_generate.py --model-dir /mnt/ForgeRealm/models/Qwen3.8-27B --cache-dir artifacts/qwen38_int3_cache --load-only
```

### G2b — PENDING-LEAD

Registered threshold: on-device vs bounded CPU dequantized-INT3 reference,
cosine >= 0.999 independently for layers 0 and 3, with relative L2 reported.

```bash
flock -w 3600 /tmp/forge-gpu.lock timeout 3000 python scripts/qwen38_gates.py --model-dir /mnt/ForgeRealm/models/Qwen3.8-27B --cache-dir artifacts/qwen38_int3_cache --g2b
```

### G3 — PENDING-LEAD

```bash
flock -w 3600 /tmp/forge-gpu.lock timeout 3000 python scripts/qwen38_generate.py --model-dir /mnt/ForgeRealm/models/Qwen3.8-27B --cache-dir artifacts/qwen38_int3_cache --demo --max-new-tokens 64
```

Raw outputs: PENDING-LEAD. No output is fabricated.

### G3-DIAG — NOT RUN

G3 did not run, so its output could not trigger the diagnostic. If G3 is
degraded or incoherent, rerun registered prompt 1 with the sigmoid override:

```bash
flock -w 3600 /tmp/forge-gpu.lock timeout 3000 python scripts/qwen38_generate.py --model-dir /mnt/ForgeRealm/models/Qwen3.8-27B --cache-dir artifacts/qwen38_int3_cache --prompt 'Write a Python function that returns the first n Fibonacci numbers, with a short explanation.' --output-gate-type sigmoid --max-new-tokens 64
```

### G4 — PENDING-LEAD

```bash
flock -w 3600 /tmp/forge-gpu.lock timeout 3000 python scripts/qwen38_generate.py --model-dir /mnt/ForgeRealm/models/Qwen3.8-27B --cache-dir artifacts/qwen38_int3_cache --benchmark --max-new-tokens 64
```

## Deviations

1. G1, G2b, G3, and G4 were not run because both `nvidia-smi` and a direct
   TensorCUDA allocation confirmed that no CUDA device was available. Peak VRAM,
   raw generation outputs, and timing are therefore PENDING-LEAD.
2. G3-DIAG was not run because it is conditional on degraded G3 output and G3
   could not run.
3. One initial read-only `nvidia-smi` visibility check was accidentally issued
   without the required outer `flock`/`timeout`. It failed with the same driver
   message and performed no allocation. All subsequent GPU-touching probes used
   the mandated wrapper.

---

# ORDER QWEN38-A3 receipt — sigmoid attention-gate adjudication

Date: 2026-08-15 (America/Detroit)

## Empirical adjudication (lead GPU receipts, quoted)

> "with SiLU on the attention output gate, 27B greedy output is multilingual
> token salad on all 3 prompts (G0 5.9e-7 and G2b 0.99998 still passed — the
> reference carried the same gate choice, so parity could not catch it)."

> "With `--output-gate-type sigmoid`: coherent correct text on all 3 prompts
> (code/factual/chat), clean EOS."

This refutes the A1 inference. Sigmoid is now the 27B attention-output-gate
default; `--output-gate-type {swish,silu,sigmoid}` remains a diagnostic override.
No A3 edit was made to `core/qwen35_tc.py`; the 9B path is unchanged by A3.

## `output_gate_type` source resolution

Verdict: **empirically adjudicated, mechanism unresolved**. More precisely, the
checkpoint field governs nothing in the installed Transformers 5.12.0 Qwen3.5
runtime:

- The raw checkpoint contains both `attn_output_gate: true` and
  `output_gate_type: "swish"` at
  `/mnt/ForgeRealm/models/Qwen3.8-27B/config.json:8-17,87-103`.
- The installed text-config field list has `hidden_act` and all attention and
  DeltaNet dimensions, but does not declare or document `output_gate_type`:
  `/home/vader/.local/lib/python3.12/site-packages/transformers/models/qwen3_5/configuration_qwen3_5.py:76-101`.
- Full attention splits the fused query projection into query and gate at
  `/home/vader/.local/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:683-686`, then unconditionally applies
  `torch.sigmoid(gate)` before `o_proj` at the same file's lines 713-716. It does
  not read `config.output_gate_type`.
- The most plausible alternative provenance is the DeltaNet output gate, but
  the installed source does not connect the field there either. DeltaNet creates
  its distinct `in_proj_z` at
  `/home/vader/.local/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:432-435`, passes `z` into its gated norm at lines 552-558,
  and that norm hardcodes `F.silu(gate)` at lines 187-202. DeltaNet separately
  derives its convolution activation from `config.hidden_act` at lines 371-385.
- The model README documents both "Gated DeltaNet" and "Gated Attention" but no
  gate activation selector:
  `/mnt/ForgeRealm/models/Qwen3.8-27B/README.md:32-51`.

Thus the runtime fact is resolved (the metadata is unused); the checkpoint
author's intended meaning for the field is not recoverable from the installed
source or bundled config documentation. Treating it as the DeltaNet selector
would be plausible speculation, not a supported mechanism finding.

## Files changed for A3

- `core/qwen38_tc.py`: default attention gate changed to sigmoid; model-dir
  loading explicitly ignores checkpoint `output_gate_type` for this gate while
  retaining the programmatic diagnostic override.
- `scripts/qwen38_gates.py`: removed the config-aware SiLU reference patch; G0
  calls installed Transformers attention (sigmoid), the independent NumPy
  attention reference uses sigmoid, and the G2b CPU attention oracle uses
  sigmoid.
- `scripts/qwen38_generate.py`: diagnostic override help/default now states the
  27B sigmoid default coherently.
- `tests/test_qwen38_cpu.py`: asserts both the raw checkpoint's historical
  `"swish"` metadata and the adapter's adjudicated `"sigmoid"` attention default.
- `docs/QWEN38_PORT_RECEIPT.md`: marks the A1 interpretation refuted and records
  the source-resolution and rerun receipts.

## Corrected config printout

`python scripts/qwen38_generate.py --budget-only` printed
`"output_gate_type": "sigmoid"` in `QWEN38_CONFIG`.

## G0 — PASS under sigmoid

Command:

```bash
PYTHONPATH=. python scripts/qwen38_gates.py --g0
```

Pack gate: 4 cases, 50,688 codes, max byte difference 0. All arrays were finite;
the registered relative-L2 threshold remained `1e-4`.

| case | mixer rel L2 | mixer max abs | block rel L2 | block max abs |
|---|---:|---:|---:|---:|
| random-small layer 0 DeltaNet | 3.161716835133021e-07 | 2.980232238769531e-07 | 3.082156000987091e-07 | 2.980232238769531e-07 |
| random-small layer 3 attention+sigmoid | 2.3332754524834675e-07 | 1.043081283569336e-07 | 2.525413429578184e-07 | 1.7881393432617188e-07 |
| real-weight layer 0 DeltaNet, seq 8 | 6.222185159969209e-07 | 8.58306884765625e-06 | 5.468553683732392e-07 | 6.866455078125e-05 |
| real-weight layer 3 attention+sigmoid, seq 8 | 5.103872026522143e-07 | 1.049041748046875e-05 | 5.56003057285673e-07 | 7.62939453125e-06 |

## CPU suite — PASS under sigmoid

```bash
PYTHONPATH=. pytest -q tests/test_qwen38_cpu.py tests/test_qwen35_translation_poc.py::test_qwen35_attention_exposes_grm_injection_contract tests/test_qwen35_translation_poc.py::test_qwen35_config_loads_real_2b_and_9b_shapes_when_available
```

```text
.........                                                                [100%]
9 passed, 2 warnings in 1.17s
```

The focused Qwen3.8 file also passed independently: `7 passed, 2 warnings in
1.29s`.

## Deviations

None from the A3 implementation or required CPU reruns. The mechanism-level
meaning intended by the checkpoint author remains unresolved because the
installed source and bundled documentation never consume or define the field;
that is the documented source finding, not a silently assumed deviation.
