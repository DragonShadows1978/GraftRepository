# Qwen3.5 2B-to-9B Graft Translation PoC Plan

**Status:** implementation plan, no experiments run.

This is the execution plan for the attention-only graft translation PoC
registered in `docs/GRAFT_TRANSLATION_PRIMER.md`.

## Goal

Test whether attention graft state harvested from a smaller Qwen3.5 model can
be translated into the Qwen3.5-9B attention dialect well enough for the 9B
model to read it.

The first PoC is deliberately **attention-only**:

- translate pre-RoPE attention K and V
- evaluate translated 2B state under 9B queries
- do not translate DeltaNet recurrent state
- do not claim full hybrid GRM portability

Full Qwen3.5 hybrid GRM portability remains a later project because the
DeltaNet state is currently prefix-restore-only, not seat-remountable memory.

## Source Weight Law

Use unquantized source weights for both models, then quantize locally.

Allowed source weights:

- BF16/FP16 safetensors
- official Qwen-family model directories with `config.json`, tokenizer files,
  and `.safetensors` shards

Rejected source weights:

- GGUF (`Q4`, `Q8`, `Q8_0`, etc.)
- already-quantized HF/bitsandbytes checkpoints
- any checkpoint without the original unquantized tensors

Known local state at plan time:

- Qwen3.5-9B unquantized safetensors exist at:
  `/home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a/`
- `/home/vader/models/Qwen3.5-9B.Q8_0.gguf` exists but is **not valid** for
  this experiment.
- Qwen3.5-2B unquantized safetensors were not found locally and must be
  downloaded or supplied before implementation begins.

## Implementation Phases

### Phase 0: weight acquisition and validation

1. Download or locate the official unquantized Qwen3.5-2B safetensors.
2. Verify both 2B and 9B source directories contain:
   - `config.json`
   - tokenizer files
   - one or more `.safetensors` shards
3. Compute and record:
   - model directory path
   - model revision/hash if available
   - tokenizer hash
   - safetensors shard list and sizes
4. Abort if either side is GGUF-only or already quantized.

### Phase 1: local INT4 parity for both models

1. Generalize `core/qwen35_tc.py` so Qwen3.5 dimensions come from
   `config.json` instead of the 9B constants where required.
2. Keep the existing tensor_cuda INT4 path as the only quantization path.
3. Bring up `Qwen35_TC.from_pretrained(model_dir=...)` for both 2B and 9B.
4. Run parity gates per model:
   - teacher-forced logits against HF/reference prompts
   - margin-based top-1 disagreement accounting
   - attention-layer cache shape checks
   - state save/restore smoke for the hybrid cache

Phase 1 exit criterion: both models load from unquantized safetensors,
quantize locally, and pass their own engine/reference sanity gates.

### Phase 2: attention capture

Add a capture mode for Qwen3.5 attention layers that records, per selected
text span:

- attention layer ordinal
- normalized pre-RoPE K
- V
- token ids
- position offsets
- for 9B only: live pre-attention queries used for scoring translated keys

Do not capture or fit DeltaNet recurrent state in this PoC.

Run capture sequentially on the 4070 Super:

1. load 2B INT4, harvest source attention K/V, write shards, clear GPU
2. load 9B INT4, harvest target attention K/V and 9B queries, write shards,
   clear GPU

Shard format:

- `.npz` arrays for tensors
- sidecar `manifest.json`
- one shard per corpus chunk or small batch
- no GPU-resident state required after each shard is written

### Phase 3: translator fitting

Fit maps from 2B attention space into 9B attention space.

Default translator:

- one ridge linear map per attention layer for K
- one ridge linear map per attention layer for V
- default ridge lambda: `1e-4`
- fit on CPU/RAM using streaming normal equations or chunked least squares

Layer alignment:

- map by attention-layer ordinal, not absolute block index
- if attention counts match, use `attn_0 -> attn_0`, etc.
- if counts differ, use fractional attention ordinal alignment and record it
  in the manifest

Baselines:

- identity 9B-to-9B map
- shuffled document-pair map
- wrong-layer map
- K-only map
- V-only map

### Phase 4: evaluation gates

Run held-out evaluation in this order:

1. **G0 identity:** 9B-to-9B identity path reproduces 9B-native attention
   behavior within numerical noise.
2. **G1 key fidelity:** translated 2B K scored against 9B queries preserves
   9B-native top-k attention targets better than shuffled and wrong-layer
   baselines.
3. **G2 value fidelity:** translated 2B V under 9B attention weights produces
   attention outputs closer to 9B-native V than the negative controls.
4. **G3 binding probe:** planted facts rank above decoys after translated K/V
   substitution.
5. **Routing-only check:** if full K/V binding fails but key fidelity is real,
   test translated states as cross-model routing keys with native 9B re-harvest
   on promotion.

Success claims:

- If G1 and G2 pass but G3 fails: translation carries attention geometry but
  not usable bindings.
- If G3 passes: attention-plane memory portability is real for this pair.
- No result here proves full Qwen3.5 hybrid-state portability.

## Artifacts

Translator output directory:

`/mnt/ForgeRealm/qwen35_graft_translation_poc/`

Required outputs:

- `weights_manifest.json`
- `capture_manifest.json`
- `translator_manifest.json`
- `fit_metrics.json`
- `eval_metrics.json`
- translator `.npz` files
- plain-text summary suitable for appending to the primer or research board

Every manifest must record:

- source model path and revision
- target model path and revision
- tokenizer identity/hash
- quantizer settings
- attention layer map
- tensor shapes
- train/held-out corpus split
- command line used to produce the artifact

## Acceptance Criteria

The PoC is complete when:

1. both unquantized source checkpoints are verified
2. both models are quantized locally through tensor_cuda INT4
3. both models pass their parity/smoke gates
4. attention captures exist for train and held-out sets
5. linear translator artifacts are produced
6. G0/G1/G2/G3 metrics are reported with negative controls
7. the final write-up states exactly which claim survived:
   - no signal
   - routing-only signal
   - attention geometry transfer
   - binding transfer

## Notes

This PoC intentionally avoids the tempting shortcut of using the local Q8 GGUF.
The experiment is about learned-state translation, not comparing unknown
quantization artifacts. Both sides must start from unquantized safetensors so
the only quantization variable is the local INT4 stack.

---

## Review Addendum (Fable, 2026-07-02) — required before Phase 0 starts

Plan APPROVED with the following amendments. None change the design;
all close holes a hostile reader (or a misleading result) would exploit.

**R1 — Register NUMBERS for every gate before any fitting.** The gates
currently say "better than shuffled" / "closer than controls" — beating a
shuffled baseline is a floor, not a pass. Required pre-registered
thresholds (fill in the blanks and freeze them in this doc before
Phase 3):
- G0 identity: max abs Δlogit ≤ ___ (bf16-noise scale, same style as the
  dialect surface gates).
- G1 key fidelity: recall@16 of 9B-native top-16 attention targets ≥ ___%
  AND ≥ ___× the shuffled baseline, per layer band (early/mid/late).
- G2 value fidelity: attention-output cosine ≥ ___ / MSE ≤ ___ vs
  9B-native, same banding.
- G3 binding: ≥ ___ of ≥32 probes with positive gold-minus-decoy margin.

**R2 — Add the two missing baselines that make G3 interpretable.**
(a) **2B-native ceiling:** run the same binding probes on the 2B itself.
Translated-into-9B state cannot contain more than the 2B encoded; without
this ceiling, a G3 failure is ambiguous between "translation failed" and
"2B never bound it." (b) **9B-native mount (upper bound):** standard GRM
harvest→re-seat of the same spans. The full ladder every metric reports
against: shuffled floor < translated-2B ≤ 2B-native ceiling ≤ 9B-native
mount. G0's identity map doubles as the INT4-noise yardstick — state this
explicitly and subtract it when judging fit residuals.

**R3 — Specify the map SHAPE (the head-mixing decision).** "One ridge
linear map per attention layer for K" must state dimensions: full
KV-width per layer — (KVH_2B x Dh_2B) → (KVH_9B x Dh_9B) per token
position, allowing cross-head mixing (heads do not correspond 1:1 across
sizes). Record the shapes and per-layer R² in fit_metrics.json — the
fit-quality-by-depth curve is a scientific result on its own.

**R4 — Tokenizer identity is an ABORT gate, not a manifest note.**
Per-token pair alignment requires bit-identical tokenization. Phase 0
must assert 2B and 9B tokenizer hashes are EQUAL and abort otherwise
(fall back to span-level alignment only as a documented plan change).
Also specify the capture corpus (source, size ~2-5M tokens is plenty for
per-layer ridge) and make the train/held-out split DOCUMENT-level, not
token-level — token-level splits leak.

**R5 — Name the pair-choice tradeoff in writing.** This PoC answers
geometry/binding transfer on the hybrid pair, whose 9B side cannot yet
exercise full GRM arena semantics (DeltaNet plane is prefix-restore-only).
A G3 pass here CANNOT be demonstrated end-to-end in a live GRM session on
this pair. The dense pair (Qwen3-1.7B → Qwen3-4B, 4B already ported with
GRM descent gates green) is the designated FOLLOW-UP where a pass converts
into a full route→mount→recall demonstration. Acceptable to run 3.5 first
(it is the production serving model — portability TO it is the valuable
direction); the limitation just has to be stated in the final write-up's
claims section, which already gestures at it ("no result here proves full
hybrid portability") — make it explicit that end-to-end GRM proof is
deferred to the dense pair.

Gate-name mapping note for cross-doc readers: this plan's G1/G2 are new
(key/value fidelity); its G3 corresponds to SCRIBE-G3 (bindings, the
decisive gate). The primer's G2 (texture) is subsumed by this plan's G2.
