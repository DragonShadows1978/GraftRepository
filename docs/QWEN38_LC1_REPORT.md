# ORDER QWEN38-LC1 implementation report

> **Status addendum (lead, 2026-09-09, from the GRM-C6 integration review):**
> the statements below that `BASELINE_TOKEN_IDS` is a `REPLACE-ME`
> placeholder and that the GPU gates are unrun are STALE. The source
> (`scripts/qwen38_longctx_gate.py`, `BASELINE_TOKEN_IDS`) holds the real
> greedy token arrays, and the retained receipt
> `logs/lc1_g0_merged_final.log` (2026-08-20) records G0 `status: PASS`
> with chat 64/64, code 64/64, factual 23/23 tokens matching (loaded
> `attn_path: legacy`, chunk 31040; peak 11,663 MiB decode). That receipt
> covers the default/concat path only; the LC/INT8/host-KV gates and the
> 6,208-row preallocated chunk are not certified by it. The original text
> is left as written below (historical); see the C6 review for the
> remaining drift items.

Status: CPU implementation gates PASS. GPU gates are authored but UNRUN here,
as assigned to the lead. No 8K/16K/32K recall, top-1 agreement, coherence, or
whole-device peak claim is upgraded to PASS until those commands produce GPU
receipts.

## LC1.0 ceiling map

- Native/config ceiling: the adapter reads `max_position_embeddings` from the
  checkpoint and defaults it to 262,144 only when absent
  (`core/qwen38_tc.py:165-196`). The model constructor rejects any configured
  capacity outside `[1, max_position_embeddings]`
  (`core/qwen38_tc.py:830-851`).
- The remaining 4,096 is a compatibility default, not a native-window limit:
  `DEFAULT_MAX_CONTEXT=4096` is paired with independently configurable prefill
  and KV block sizes (`core/qwen38_tc.py:35-42`). The only other behavioral
  4,096 selects the registered BF16 baseline attention call; larger contexts
  use standard online-softmax tiling after the 256-token compatibility corridor
  (`core/qwen38_tc.py:460-472`). The 4,096 in
  the CPU massive-activation pack case is test data, not a length
  (`core/qwen38_tc.py:145-159`).
- RoPE: Qwen3.8 inherits the table builder that materializes cosine/sine through
  the requested sequence length (`core/qwen35_tc.py:444-454`), but now calls it
  once with configured `max_context` during construction
  (`core/qwen38_tc.py:858-862`). The budget charges the same configured table
  length (`core/qwen38_tc.py:1230-1239`).
- Device KV allocation: all 16 full-attention layers receive fixed-capacity
  buffers before prompt execution (`core/qwen38_tc.py:865-878`), and the first
  batch-1 set is allocated immediately after weight load
  (`core/qwen38_tc.py:1129-1133`). BF16 stores K
  and V directly; INT8 stores uint8 K/V plus one scale per token and head, and
  all writes are capacity checked (`core/qwen38_tc.py:265-332`). The exact
  formulas for BF16 and INT8 storage are in the pre-load budget
  (`core/qwen38_tc.py:1244-1249`).
- Host KV allocation: K/V are raw BF16 `uint16` host arrays, capacity checked on
  append, and only requested blocks are reconstructed on device
  (`core/qwen38_tc.py:344-370`). Consequently device KV storage is zero in the
  host budget while the full BF16 bytes are charged to host RAM
  (`core/qwen38_tc.py:1248-1250`, `core/qwen38_tc.py:1267-1282`).
- Prefill behavior before LC1: the Qwen3.5 parent `__call__` makes one layer pass
  over the complete `L` and has no prefill loop (`core/qwen35_tc.py:456-482`),
  so no inherited Qwen3.5 chunker applied. Qwen3.8 now splits at a configurable
  64-token default and empties the allocator pool between chunks
  (`core/qwen38_tc.py:888-912`).
- Positions and masks: the context limit checks absolute `position_offset + L`
  and any graft shift (`core/qwen38_tc.py:888-895`,
  `core/qwen38_tc.py:914-936`). RoPE slices use that absolute offset, while the
  tiled causal mask compares absolute query and key positions
  (`core/qwen38_tc.py:406-416`, `core/qwen38_tc.py:453-458`).
- Standard attention transient: long/device-INT8/host paths reshape GQA without
  repeating the full cache, stage only `kv_block_rows`, and merge each block by
  online log-sum-exp (`core/qwen38_tc.py:392-435`). Thus the attention workspace
  is bounded by prefill-chunk and KV-block sizes rather than total sequence
  length; its explicit budget formula is `core/qwen38_tc.py:1252-1261`.
- Logit transient: prefill slices to the final hidden position before `lm_head`
  (`core/qwen38_tc.py:939-944`). The packed lm-head presents one output-row
  chunk per two-stage INT3 call (`core/qwen38_tc.py:743-805`); the new default
  6,208 rows is 60.625 MiB of BF16 dequantized weight
  (`core/qwen38_tc.py:35-42`).
- Phase peaks: the sampler polls whole-device `nvidia-smi memory.used`, tags
  load/prefill/decode phases, and reduces both family and detailed peaks
  (`scripts/qwen38_generate.py:48-115`). The pre-load map includes persistent
  tensors, the bounded lm-head/attention workspaces, and a fixed 1,185 MiB
  overhead calibrated from the 2026-08-19 peak receipt
  (`core/qwen38_tc.py:45-51`, `core/qwen38_tc.py:1251-1264`). It fails above the
  registered 12,000 MiB ceiling before loading weights
  (`core/qwen38_tc.py:1286-1293`, `scripts/qwen38_generate.py:230-257`).

## INT8 choice

The implementation uses symmetric per-token, per-head K and V scales. This is
the least assumptive device format: every token/head gets an independent scale,
zero is exactly representable, and the K projection is followed by per-head
RMSNorm before RoPE (`core/qwen38_tc.py:444-458`), so scale contamination cannot
cross tokens or heads. The CPU contract deliberately includes a 4,096x outlier
dimension (`core/qwen38_tc.py:145-159`). This does **not** establish that the
real checkpoint lacks within-vector massive-activation dimensions; no GPU K
distribution was measured in this seat. Every INT8 cache records first-prefill-
chunk `abs_p99`, `abs_p999`, `abs_max`, and max/RMS distribution statistics
before packing (`core/qwen38_tc.py:295-322`), and the generation receipt emits
them (`scripts/qwen38_generate.py:159-176`). INT8 therefore remains default-OFF
and G-LC3 top-1 plus recall is the adjudicator, not this rationale. A RED G-LC3
is a valid result and leaves the feature opt-in.

## Registered thresholds

- G-LC0: all three embedded default-demo token-id lists must match; one mismatch
  is RED and stops the ladder. The placeholder refuses to run until replaced
  (`scripts/qwen38_longctx_gate.py:30-37`,
  `scripts/qwen38_longctx_gate.py:211-215`).
- G-LC1: every sampled whole-device peak must be at most 12,000 MiB.
- G-LC2: 8K at least 5/6; 16K at least 2/3; 32K report-only.
- G-LC3: at least 99% teacher-forced top-1 agreement over 64 steps and 8K INT8
  recall count no worse than BF16. The agreement implementation feeds the BF16
  greedy trajectory to the INT8 arm (`scripts/qwen38_longctx_gate.py:122-157`).
- G-LC4: the lead adjudicates the printed 64-token raw continuation
  (`scripts/qwen38_longctx_gate.py:253-261`).

## Known residuals

- GPU runtime behavior is unverified in this CPU-only seat. In particular,
  TensorCUDA broadcasting/reduction behavior in the new online-softmax path,
  INT8 GPU rounding parity, actual allocator high-water, and host-transfer
  correctness require the registered GPU gates.
- A 262,144-token host cache reserves 16 GiB of host virtual address space plus
  the existing 2.37 GiB embedding mmap; committed RAM grows with tokens. This
  rung is correctness-oriented and may page heavily.
- The host implementation uses pageable NumPy memory, not pinned memory.
- Host KV preserves BF16 bytes but online-softmax changes reduction order from
  the 4K baseline. It is mathematically standard attention, not bit-identical
  arithmetic.
- `BASELINE_TOKEN_IDS` remains `REPLACE-ME` by order; G-LC0 cannot run until the
  lead inserts the preserved token IDs.
- The budget's 1,185 MiB fixed term is receipt-calibrated, not a formal allocator
  bound. Any sampled peak above 12,000 MiB is RED regardless of prediction.
