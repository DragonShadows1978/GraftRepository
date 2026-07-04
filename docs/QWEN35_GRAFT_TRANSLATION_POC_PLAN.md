# Qwen3.5 2B-to-9B Graft Translation PoC Plan

**Status:** Phase 0 source validation passed for the real 2B/9B pair; Phase 1
Qwen3.5 config-generalized loader smoke passed for both models; Phase 2
capture hook/shard smoke plus resumable corpus/pipeline runners passed, and
the real source capture is complete. The real target capture is still running:
latest refreshed status is target `8552 / 9861` chunks, with `1,944,308`
paired train tokens and `239,415` paired held-out tokens. Phase 3/4 ridge fit,
negative-control fit, G0/G1/G2 evaluator, and G3 binding harness commands are
implemented and smoke-tested. The final real fit/gate ladder waits for target
capture completion and the frozen `>= 2M` paired train-token gate.

**Completion ledger:** operational completion entries and evidence live in
`docs/QWEN35_TRANSLATION_IMPLEMENTATION_LEDGER.md`. Update that ledger after
each completed implementation, capture, fit, eval, or control-baseline step.

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

- Qwen3.5-2B unquantized safetensors exist at:
  `/home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc/`
- Qwen3.5-9B unquantized safetensors exist at:
  `/home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a/`
- `/home/vader/models/Qwen3.5-9B.Q8_0.gguf` exists but is **not valid** for
  this experiment.

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

Implemented validator command:

```bash
PYTHONPATH=. python3 scripts/qwen35_graft_translate_poc.py validate-weights \
  --source /path/to/Qwen3.5-2B-safetensors \
  --target /path/to/Qwen3.5-9B-safetensors \
  --out /mnt/ForgeRealm/qwen35_graft_translation_poc/weights_manifest.json
```

Current implementation artifacts:

- `core/qwen35_translation_poc.py` validates HF safetensors directories,
  rejects GGUF and already-quantized sources, records tokenizer/config hashes,
  records shard counts and sizes, and aborts on tokenizer mismatch.
- `scripts/qwen35_graft_translate_poc.py` exposes the validator as a CLI.
- `tests/test_qwen35_translation_poc.py` is the focused Phase 0 test gate.
- `/mnt/ForgeRealm/qwen35_graft_translation_poc/weights_manifest.json` records:
  - source `Qwen/Qwen3.5-2B`
    revision `15852e8c16360a2fea060d615a32b45270f8a8fc`, 1 safetensors shard,
    4,548,221,488 bytes, hidden 2048, 24 layers, attention layers
    `[3, 7, 11, 15, 19, 23]`
  - target `Qwen/Qwen3.5-9B`
    revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, 4 safetensors shards,
    19,306,310,880 bytes, hidden 4096, 32 layers, attention layers
    `[3, 7, 11, 15, 19, 23, 27, 31]`
  - matching `tokenizer.json` hash
    `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`
- Ledger entry:
  `docs/QWEN35_TRANSLATION_IMPLEMENTATION_LEDGER.md`

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

Current Phase 1 implementation status:

- `core/qwen35_tc.py` now reads Qwen3.5 text dimensions from `config.json`
  instead of assuming the 9B constants, including hidden width, layer count,
  attention head/KV-head count, DeltaNet dimensions, RoPE dimensions, attention
  layer indices, repository, revision, and tied/untied output-head mode.
- 2B INT4 GPU smoke passed from the real safetensors:
  - logits shape `(1, 1, 248320)`
  - cache count `24`
  - layer-0 DeltaNet cache `(1, 3, 6144)` and `(1, 16, 128, 128)`
  - first attention cache at layer 3: K/V `(1, 2, 3, 256)`
- 9B INT4 GPU smoke passed from the real safetensors after the same loader
  refactor:
  - logits shape `(1, 1, 248320)`
  - cache count `32`
  - layer-0 DeltaNet cache `(1, 3, 8192)` and `(1, 32, 128, 128)`
  - first attention cache at layer 3: K/V `(1, 4, 3, 256)`
- Still required before Phase 1 is complete: HF/reference teacher-forced parity,
  margin-based top-1 accounting, attention cache checks across all attention
  layers, and state save/restore smoke per model.

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

Current Phase 2 implementation status:

- `core/qwen35_tc.py` exposes Qwen3.5 attention capture hooks compatible with
  `kv_graft.harvest_kv()` and `kv_graft.capture_queries()`:
  - `_capture` records post-qk-norm, pre-RoPE K and V.
  - `_capture_q` records post-qk-norm, pre-RoPE queries.
- `core/kv_graft.py` now finds attention modules either at `layer.self_attn`
  or at Qwen3.5 hybrid `layer.mixer` for full-attention layers.
- `core/qwen35_translation_poc.py` can write capture shards with token ids,
  layer ids, K/V arrays, and optional target-side query arrays.
- `scripts/qwen35_graft_translate_poc.py capture-smoke` writes real smoke
  shards from the local INT4 model loaders.
- `scripts/qwen35_graft_translate_poc.py plan-corpus` writes a document-level
  train/held-out corpus plan with token ids frozen into `corpus_plan.json`.
- `scripts/qwen35_graft_translate_poc.py capture-corpus` runs resumable,
  single-role sequential capture from `corpus_plan.json`; existing complete
  shards are skipped, and `capture_manifest.json` is refreshed after each
  batch.
- Real first-layer smoke artifacts exist at:
  - `/mnt/ForgeRealm/qwen35_graft_translation_poc/capture_smoke/source_docsmoke_chunk000000.npz`
    with layer-3 source K/V shapes `(1, 2, 4, 256)`
  - `/mnt/ForgeRealm/qwen35_graft_translation_poc/capture_smoke/target_docsmoke_chunk000000.npz`
    with layer-3 target K/V shapes `(1, 4, 4, 256)` and query shape
    `(1, 16, 4, 256)`
- Tiny CLI rehearsal passed from `/tmp/qwen35_translation_tiny_corpus`:
  - `plan-corpus` wrote a 2-document, 2-chunk, 49-token plan.
  - `capture-corpus --role source --max-chunks 1` wrote one source shard.
  - `capture-corpus --role target --max-chunks 1` wrote one target shard with
    queries.
  - rerunning source with `--max-chunks 1` skipped one existing shard and
    completed the second source chunk, proving resume-forward behavior.
- Operational runbook for Claude/cron:
  `docs/QWEN35_TRANSLATION_CORPUS_RUNBOOK.md`
- Operational completion ledger:
  `docs/QWEN35_TRANSLATION_IMPLEMENTATION_LEDGER.md`
- Real corpus plan is frozen, source capture is complete, and target capture is
  in progress under the pipeline runner. `capture_manifest.json` and
  `pipeline_status.json` now record paired source/target progress so preview
  work can be separated from final gate artifacts.
- Still required before Phase 2 is complete: finish all-attention-layer target
  capture and produce the final corpus-scale `capture_manifest.json`.

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

Current Phase 3 implementation status:

- `scripts/qwen35_graft_translate_poc.py fit-translator` streams paired
  source/target capture shards and fits full-width ridge maps for K and V.
- Maps are saved as `translator_l{source}_to_l{target}_{k,v}.npz`, with
  separate `weight` and `bias` arrays.
- `translator_manifest.json` records the fractional attention-layer alignment
  and artifact shapes.
- `fit_metrics.json` records train-token counts, MSE, R², and mean row cosine.
- Tiny real capture rehearsal passed on one paired layer-3 shard:
  - K artifact shape `512 -> 1024`
  - V artifact shape `512 -> 1024`
  - smoke fit R² was effectively 1.0 on the tiny same-shard rehearsal
- Wrong-layer, shuffled-docs, K-only, and V-only fit controls are implemented.
- R1 thresholds are frozen below. The first final real fit waits for target
  capture completion and at least `2M` paired train tokens. Partial preview
  fits must use separate output directories and must not populate the final
  `translator/` path.

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

Current Phase 4 implementation status:

- `scripts/qwen35_graft_translate_poc.py eval-translator` evaluates fitted
  translators on paired held-out capture shards.
- Implemented metrics:
  - G1 key recall@k against native 9B query-to-key top-k sets
  - shuffled-key recall baseline
  - G2 value-output MSE/cosine under native 9B attention weights
- Tiny real capture rehearsal passed:
  - `key_recall_at_8 = 1.0`
  - `shuffled_key_recall_at_8 = 0.20703125`
  - `value_output_cosine = 0.999999999999926`
- G0 capture identity, live G0 logit smoke, wrong-layer value/key controls,
  K-only/V-only controls, deterministic G3 binding probes, and G3 binding
  evaluation modes are implemented.
- G0 capture identity now uses a structural exact-identity path instead of
  recomputing target attention against itself, making the final post-capture
  gate linear in shard count.
- Still required for complete Phase 4: run the final G0/G1/G2/G3 gate ladder
  on completed target capture and final translator artifacts, including
  2B-native and 9B-native binding baselines.

## Artifacts

Translator output directory:

`/mnt/ForgeRealm/qwen35_graft_translation_poc/`

Required outputs:

- `weights_manifest.json`
- `capture_manifest.json`
- `corpus_plan.json`
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
thresholds — **FROZEN 2026-07-02 (Fable proposed, David ratified). No
edits after the first real corpus fit:**
- G0 identity (runs BEFORE any fitting — translator-independent): 9B→9B
  capture→re-inject path: max abs Δlogit ≤ 2e-3 AND top-1 flip rate
  ≤ 0.1%. G0's measured values double as the INT4-noise floor for
  interpreting all downstream gates.
- G1 key fidelity: recall@16 of 9B-native top-16 attention targets
  ≥ 60% averaged, AND ≥ 3× the shuffled baseline in EVERY layer band
  (with only 6 source attention layers, bands = pairs in ordinal order).
- G2 value fidelity: attention-output cosine ≥ 0.90 per band, AND MSE
  ≤ 25% of the wrong-layer baseline's MSE.
- G3 binding: ≥ 14 of 32 probes with positive gold-minus-decoy margin.
  RATIONALE: gold-vs-3-decoys chance = 25% = 8/32; 14/32 is binomial
  p < 0.05 against chance. Results of 9-13/32 are reported as "signal,
  not significant" — door ajar, not open. The 2B-native ceiling (R2)
  is reported alongside G3 regardless of outcome.
- Fit protocol (frozen with the gates): ≥ 2M paired tokens for the fit;
  held-out split at DOCUMENT level, 10%; ridge lambda 1e-4 as planned
  (a lambda sweep, if any, uses train-split diagnostics only — never
  held-out gate data).

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
