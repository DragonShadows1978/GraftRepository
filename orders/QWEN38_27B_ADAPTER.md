# ORDER QWEN38-A1 — Qwen3.8-27B adapter on tensor_cuda, INT3, DeltaNet and all

YOUR WRITABLE TARGET is /mnt/ForgeRealm/GraftRepository — edits, builds,
and test runs inside it are AUTHORIZED. Also writable: /tmp.
READ-ONLY: /mnt/ForgeRealm/Project-Tensor (the engine you import — do NOT
edit it), /mnt/ForgeRealm/models/Qwen3.8-27B (the weights), the installed
transformers package (your golden reference). Everything else: hands off.
The lead commits; you never run git (read-only git commands are fine).

## Mission

Qwen3.8-27B (released 2026-08-14) generating coherent text through the
house tensor_cuda engine at INT3 weights, on a 12GB card, by morning.
Text-only: the vision tower and the MTP layer are OUT of scope (skip
their tensors at load).

## What already exists — REUSE, do not rebuild

- `core/qwen35_tc.py` — the COMPLETE Qwen3.5-9B hybrid port (Gated
  DeltaNet + gated attention + SwiGLU) on tensor_cuda. Same `qwen3_5`
  arch family as this model. Its docstring lists the hard-won parity
  traps (1+w norm bake vs plain-w DeltaNet norm; fused per-head
  [q|gate] in q_proj; l2norm INSIDE the delta rule, eps on the SUM;
  q scaled 1/sqrt(d_k) after l2norm; decay before delta correction;
  fp32 state; depthwise conv k=4 no-bias SiLU causal pad 3; mRoPE
  no-op for text). `docs/QWEN35_PORT_LEDGER.md` has the receipts.
  READ BOTH FIRST.
- `Qwen35Config.from_model_dir` already reads every dimension from
  config.json — pointed at /mnt/ForgeRealm/models/Qwen3.8-27B it yields
  hidden 5120, 64 layers (48 deltanet + 16 attention, interval 4),
  GQA 24q/4kv head_dim 256, partial rotary 64, DeltaNet 16 qk-heads /
  48 v-heads d=128, FFN 17408, vocab 248320 untied. Lead verified the
  config parse path; trust but re-print it in your receipt.
- `core/mistral7b_tc.py::QuantLinearTC` — bits ∈ {2,3,4} already
  supported; bits==3 routes to the engine's generic `tc.intn_linear` /
  `tc.intn_linear_fused` (lead verified both are exported by the
  canonical engine build). `TC_WEIGHT_BITS=3` or
  `QuantLinearTC.set_weight_bits(3)` selects it.
- transformers 5.12.0 is installed with
  `transformers.models.qwen3_5.modeling_qwen3_5` including
  `Qwen3_5GatedDeltaNet` — this source ON DISK is your golden
  reference for every math question. Cite file:line in receipts when
  you resolve an ambiguity from it.

## Known 9B→27B deltas (the actual work)

1. **Output gate activation**: the 9B port applies a SIGMOID attention
   output gate; the 27B config says `"output_gate_type": "swish"`.
   Verify in the installed modeling source what swish means there
   (SiLU on the gate branch, presumably) and honor the config field —
   selectable, not hardcoded, cited.
2. **Head-count ratios**: DeltaNet v-heads/qk-heads = 48/16 = 3 (9B:
   32/16 = 2); GQA groups = 24/4 = 6 (9B: 16/4 = 4). Audit the port
   for anything that assumed ratio 2 or 4 (reshapes, repeats,
   broadcast tricks) and generalize from config. Same audit for the
   deltanet norm/gate shapes (6144 = 48×128 channels).
3. **Config-vs-code sweep**: diff every field `from_model_dir` returns
   for the 27B against what the layer classes actually consume; any
   silent 9B constant left in the code is a bug class, hunt them.
4. **Memory budget** — hard ceiling 11.5 GiB VRAM total:
   - all qlinears INT3 (linears ≈ 9.1 GiB packed + ~0.8 GiB
     scales/zeros), lm_head INT3 included;
   - embedding stays HOST-side (HostEmbedding pattern; ~2.5 GiB RAM,
     never VRAM);
   - fp32 DeltaNet states + bf16 KV for 16 attention layers are small;
   - print a per-component VRAM map at load and include it verbatim
     in the receipt.
5. **Quantize-once pack cache** (MANDATORY, build it FIRST): quantizing
   52 GB with NumPy on every load makes iteration impossible. First
   load writes packed/scales/zeros per tensor to
   `artifacts/qwen38_int3_cache/` (npz or npy tree + manifest with
   shapes, bits, group, source-file mtimes); later loads mmap/load the
   cache directly to GPU. Cache invalidates on bits/group/source
   change. NOTE: `affine.py`'s `_pack_int3` has a per-row Python loop —
   too slow for 248320-row lm_head. Write a vectorized 3-bit pack IN
   THIS REPO (cols%8==0 ⇒ 8 codes → 3 bytes reshape trick), gated
   bit-exact against the engine's `pack_lowbit(..., 3)` on random
   matrices before first use. The engine stays untouched.
6. **Decode path**: enable the shape-aware fused path
   (`QuantLinearTC.FUSED_DECODE = True`) so M=1 decode reads packed
   INT3 directly instead of dequantizing 9 GiB per token. Unit-gate
   `intn_linear_fused` bits=3 vs `intn_linear` parity at real shapes
   first — if fused-INT3 has a defect, report it and fall back to
   two-stage (slow but correct beats fast but wrong).

## Deliverable shape

`core/qwen38_tc.py` (or a cleanly parameterized extension of
qwen35_tc — your call, but the 9B path must remain byte-identical in
behavior) + `scripts/qwen38_generate.py` (prompt in, greedy tokens out,
loads from cache, prints tok/s) + gates below. Tokenizer: use the HF
tokenizer files from the model dir via transformers (tokenizer-only
import is cheap); chat template applied for the demo prompts.

## Gates

- **G0 (CPU, run it)**: vectorized pack bit-exact vs engine
  `pack_lowbit` bits=3; fp32 NumPy reference of YOUR DeltaNet step /
  attention block (incl. swish gate, partial RoPE, qk-norm) vs the
  transformers modules run on CPU with (a) random small tensors and
  (b) REAL weights for one deltanet layer (0) and one attention layer
  (3) pulled from the shards — teacher-forced, seq 8, rel err 1e-4
  class fp32. This is the arch-correctness gate; it needs no GPU.
- **G1 (GPU if visible)**: full 27B INT3 load from cache; VRAM map
  printed; ceiling 11.5 GiB respected.
- **G2 (GPU)**: per-layer parity ON-DEVICE vs your own G0 CPU
  reference dequantized-INT3 weights (isolates kernel path from quant
  error), layers {0, 3}, rel err 1e-3 class.
- **G3 (GPU)**: greedy generation, 3 prompts (code, factual, chat),
  64 tokens each: no NaN, no repetition collapse, text COHERENT —
  print the raw outputs verbatim in the receipt. INT3 quality is
  characterization, not pass/fail; incoherence with clean G0/G2 is
  still a reportable result (quantization, not arch).
- **G4 (GPU)**: decode tok/s + prefill seconds for a 128-token prompt,
  reported.
- If the sandbox has NO GPU: G0 is still MANDATORY and run-to-green;
  build everything else and mark G1–G4 PENDING-LEAD with exact
  commands (use `flock -w 3600 /tmp/forge-gpu.lock` and generous
  `timeout 3000` for load-bearing steps — first cache build reads
  52 GB). That is a complete delivery; say so plainly, never fake a
  GPU receipt.

## Rails

- NO git writes. NO subagents. RED honesty — a failing gate or a
  design dead-end is a result, verbatim errors.
- Engine (`Project-Tensor`), model dir, transformers install:
  READ-ONLY. All new code lives in GraftRepository.
- Keep RAM under ~50 GiB during cache build (stream shard-by-shard,
  free as you go — the box has 62 total and the lead's session is
  alive on it).
- No monitor-idling; run gates synchronously to completion.
- SCOPE LAW: no model-quality CLAIMS from this order — G3 outputs are
  characterization receipts, nothing more.

## Done — final message must contain, verbatim

- Files created/modified (exact paths).
- The 27B config printout from `from_model_dir`.
- Each gate: PASS with printed numbers, FAIL with error text, or
  PENDING-LEAD with the exact command.
- The per-component VRAM budget table (measured if GPU, computed if
  not).
- The output-gate (swish) resolution with modeling-source file:line.
- Every ratio-2/ratio-4 assumption you found and generalized.
- Any deviation from this order.
