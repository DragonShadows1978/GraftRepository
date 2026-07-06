# Gemma 4-12B APA Extension — Mission Plan (Fable, registered 2026-07-04)

> **MISSION CLOSED 2026-07-04 (same day) — negative result with a
> law.** A0 re-probe: NO OOM anywhere on the 4070S 12GB — both modes
> hold the full 32K trained window, both phases (June walls were 8GB
> artifacts); B3's gate is untestable by context ceiling. APA as-built
> is a net cost on the 12B (+peak VRAM, 3.5× prefill, +8% decode,
> ppl +1.5-1.9% engaged). **Law extracted: APA is a multi-KV-head
> thing** — statistics need key POPULATION (sliding fails), economics
> need KV-HEAD MULTIPLICITY (MQA fails: coherent noise, D=512 blocks
> the fused kernel, KV slack pre-spent at training time). The 12B
> fails both axes independently. **Architect's closure: "to try and
> save memory on storage is not APA"** — B4/prefill-into-rings are a
> separate storage project, not surviving APA tickets; their 12B case
> fails on pool size (~0.5-1GB at 32K, ~zero at the ~700-tok serving
> shape). Selection rule: APA belongs on GQA-family models (26B-A4B
> qualifies). Phases B4/S below are retained for the record only.
> Results: GEMMA4_PORT_LEDGER.md "A0 re-probe" + "A0-B" sections;
> verdict on the AI_Research_Board (Gemma track).

**Mission (Architect, queued 2026-07-02):** make APA truly EXTEND
Gemma-4 — restore the design law (APA outruns standard's context
ceiling) that holds on every other ported architecture. The Opus-era
APA implementation is DISTRUSTED: every prior claim gets re-verified by
measurement on the current hardware (4070 SUPER 12GB — the June
contract was built for the 3070 8GB) before being believed.

**Known root cause (June investigation, to re-verify):** the blend's
GQA-era 16× MQA expansion (`_repeat_kv(k, 16)`) violates APA's
bounded-transient design on Gemma's 1-KV-head globals. Registered
ticket order (ledger line ~306): ring buffers → blend de-expansion →
re-probe (expect APA > standard) → V-side APA.

**Audit finding at registration (2026-07-04), CORRECTED by A1
(docs/GEMMA4_APA_AUDIT_A1.md):** the registration's "16× expansion
still live" claim was WRONG — the surviving `_repeat_kv` calls
(gemma4_tc.py:612,617, spot-verified) are on the SLIDING path, bounded
≤1024 by ring-trim; all four GLOBAL branches are expansion-free
(traced to the de-expanded blend / fused kernel). A1 verdicts: ring
buffers DONE; global de-expansion DONE; fused D=512 kernel REAL and
dispatched (prefill-only above fast_max_seq=4096; decode always
blends by design; stale twin engine tree exists but imports pin to
the current one); **K8+V4 "best mode" (−3.56% ppl) has ZERO real
implementation** — measured via external test hook only, never built
into KVRing (B4 starts from scratch, not near-done). Consequence:
B1/B2 are substantially closed; the mission's center of gravity is
Phase S (sliding — exactly the Architect's independent read) plus B4
(real quantized-storage arc) and the A0/B3 probes on the 12GB card.
B2's "grep clean" gate is re-scoped: global path only; the sliding
`_repeat_kv` is Phase S's object (bounded, and its de-expansion is a
compute/resident win, not an OOM fix).

## Phase A — Audit first (no fixes until measured)

- **A0. Re-probe on the 4070S, current HEAD:** one-process-per-mode
  OOM ladders (the June discipline — fragmentation fakes cross-context
  OOMs), standard vs APA-r0.10, prefill and decode ceilings. June
  numbers (3070): standard 6,144 vs APA 3,072 decode; both-modes 12K
  prefill wall. The 12GB card moves every wall — get TODAY's numbers
  or nothing downstream means anything.
- **A1. Code audit:** enumerate every surviving expansion site
  (`_repeat_kv` callers, private mask copies, whole-cache requantize
  paths), verify TC_APA_MAXD status for D=512 against the engine as
  built (the June D=512 fused kernel claim gets a direct kernel-launch
  test, not a changelog read), and check which June fixes (incremental
  kq ring, chunked cold-start quantize, bounded _grow_cap) are present
  and actually reached from the current call graph.
- **A2. Gate inventory:** run the existing Gemma gates (parity, state,
  APA bulk4, rect-with-cache, apa_incremental) on HEAD before touching
  anything — the baseline must be green or the mission starts with a
  repair, not a feature.

## Phase B — Tickets, in registered order, each with a gate

- **B1. Ring buffers (complete the June start):** in-place row writes,
  bias-masked invalid rows, zero steady-state copies; ownership
  contract (live caches owned by their decode loop; sharing = explicit
  copy; GRM's host round-trip already complies). Gate: decode-step
  transient allocations flat with S (live allocator trace, not
  inference).
- **B2. Blend de-expansion, TOTAL:** no `_repeat_kv` materialization
  anywhere in the attention path — grouped-batch GEMMs against
  unexpanded K/V on both prefill and decode, both modes. Gate: grep
  clean + parity vs QAT GT unchanged + the June ppl sweep numbers
  reproduced (r0.10 within noise of standard).
- **B3. THE MISSION GATE — re-probe:** one-process-per-mode ladders,
  post-fix. Frozen expectation: **APA ceiling > standard ceiling on
  the 12GB card, both decode and prefill.** If APA merely ties, the
  mission is NOT done — ties mean a hidden resident cost (June: the
  kqb ring's +50%) still cancels the design win; find it or store it
  quantized (tail-law arc: K8+V4 resident storage, the measured
  −3.56% asymmetric config).
- **B4. V-side APA (research extension):** Gemma's scale-free v_norm
  predicts 4-bit V bulk is safe by the same tail law (June measured:
  V4-global +5% but V4-sliding fatal — the split design must respect
  that asymmetry). Registered prediction to freeze before measuring.

## Phase S — The sliding/global split (Architect's direction, 2026-07-04)

Architect: globals work fine under APA; SLIDING layers are the
problem — the mission includes making sliding as effective as global.
June's own measurement is the mechanism: V4 cost +5% ppl on globals
vs +1696% on sliding — with ≤1024 keys each V carries high attention
weight and quant noise is not crushed by aggregation. IME framing:
sliding layers sit in the small-N regime where the geometric bulk has
not formed; the bulk-bits floor is higher there. "Same tiers
everywhere" is the wrong design; the floor must be measured per layer
class.

- **S1. Depth-distribution study** (instrument already staged:
  tests/gemma4_attn_dist.py — needs a GPU window): interaction-depth
  distributions per layer class, GHOST_PRECISION style.
  **Registered prediction (frozen pre-data): global layers show the
  geometric bulk (APA-shaped); sliding layers show a suppressed bulk
  and a higher stabilization floor.** If sliding shows NO usable bulk,
  the honest conclusion is "sliding is tail-only — do not force bulk
  economics"; quantized STORAGE (K8, V per the June asymmetry) is
  then the sliding win, not blend compute.
- **S2. K-side-first sliding APA:** the +1696% catastrophe was
  V-side (output-direct). K-side quantization perturbs selection
  only — APA's robust premise. Design: INT8-K scoring bulk + fp
  refine (~r0.10 of 1024 keys) on sliding layers; V untouched in v1.
- **S3. The sliding prize is compute + resident, not context:**
  40/48 layers × 1024 keys/token dominates decode below ~5K context;
  plus the fixed 40-layer resident cache. Success metric for sliding:
  decode ms/tok and resident GB, at unchanged ppl (within noise, the
  June sweep protocol) — NOT context ceiling (that's the globals'
  trophy, Phase B).

## Coordination

- GPU is shared: translation-corpus target capture finishing
  (~9.85K/9.86K shards); the translation fit queue (Codex) claims the
  card next per the frozen PoC protocol. APA probe windows slot
  between; ladders are one-process-each and resumable by design.
- All measurements go in GEMMA4_PORT_LEDGER.md per house style;
  registered predictions frozen before data; inconclusive is a result.
