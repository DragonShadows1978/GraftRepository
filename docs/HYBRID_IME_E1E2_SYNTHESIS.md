# HYBRID IME E1/E2 — SYNTHESIS (2026-07-20)

Plan: `HYBRID_IME_E1E2_PLAN.md` (immutable, 593a964). Receipts:
`HYBRID_IME_E1E2_LEDGER.md` + `docs/ime_e1e2_results/`. Evidence class
throughout: **instrument measurement (distribution characterization)** —
no model-quality or speed claims. All four capture runs parity-clean
(instrumented vs uninstrumented final-step logits max|Δ| = 0.000e+00,
bit-identical).

## E1 — CONFIRMED (directional hypothesis holds, modestly)

The Qwen3.5-9B hybrid's 8 full-attention GQA layers show the APA
bulk+tail structure at least as strongly as the dense control's
full-context layers.

Aggregate (mean over Q-heads × 64 decode steps × 3 prompts, ~2000-key
populations both):

| | top5% mass | top10% | eff_keys | entropy |
|---|---|---|---|---|
| qwen35 hybrid GQA (8 layers) | **0.834** | 0.902 | ~120 | 4.19 |
| gemma4 dense GLOBAL (8 layers) | 0.807 | 0.873 | 149 | 4.16 |

Reading: at APA r=0.05 the refine set already covers ~83% of attention
mass on the hybrid (~90% on the retrieval workload). The hybrid is
*modestly* sharper than the dense control (+2.7pt top5%, ~20% fewer
effective keys) — real, not dramatic. **Confound, stated:** Gemma's
globals are MQA 1-kv / hd512 vs qwen's GQA 4-kv / hd256; the pair
differs in more than hybrid-vs-dense. Gemma's 40 sliding layers are
tighter (top5 0.862) but sit on 1024-key windows — different population
size, not comparable, reported separately per plan.

Secondary structure (both models): the retrieval-heavy synthetic prompt
sharpens full-attention layers hard and *converges* them — gemma global
top5 0.741→0.882 prose→synthetic; qwen35 GQA ≈0.872 on synthetic.
Under genuine retrieval load, full-attention distributions look alike
across architectures. Qwen35's first attention layer (L3) is the
broad-mixer outlier (365 eff_keys on prose → 104 on synthetic).

APA consequence: the selection law's GQA-layer prediction is supported
on the hybrid class; nothing here moves the law.

## E2 — REFUTED at the registered thresholds (clean negative, with shape)

Per-channel readout contributions c_i = ‖q_i·S[i,:]‖ of the 24 GDN
layers' delta-rule states, vs registered nulls, 32 steps × 3 prompts:

- **THRESHOLD 1 (heavy-tail): FAIL — 0.0% of 2304 (layer,head) cells**
  reached ≥2× the N1 permutation-null share (needed ≥75%).
- **THRESHOLD 2 (stable-tail / state-APA license): FAIL — median
  consecutive-step Jaccard 0.444** (needed ≥0.5).

The result is not a weak null but a directional reversal: real top-10%
channel share (0.43–0.65) sits *below* the permutation null (0.50–0.66)
in **24/24 layers** (r/N1 0.79–0.99). The trained q–S alignment is
mildly ANTI-concentrated — the readout uses the state's channels more
uniformly than a chance pairing of the same tensors would.

Interpretation (labeled as such): the delta rule is an online
least-squares update; it actively decorrelates channel usage. The
superposed state is a dense code — the two-geometry mismatch that
generates IME heavy tails per-pair in softmax attention does not appear
per-channel in the state. Same model, same prompts: attention layers put
~85% of mass in 5% of keys while the state spreads mass near-uniformly
across 128 channels. **The hybrid architecture concentrates its tail in
attention and flattens it in the state.**

Scope of the negative (hold-the-middle): not detected under — GDN
(scalar-per-head decay; KDA's channel-wise gating unmeasured), the
native channel basis (a rotated basis, e.g. eigenvectors of S, is
unsearched and could hide a tail this basis flattens), c_i as the
contribution functional, 32 steps × 3 prompts, INT4 weights, one model.

Consequence: **state-APA is not licensed.** Structural analysis (no
per-pair operand in a GDN layer) and measurement (no per-channel tail
either) now agree from independent directions.

## Method laws earned in passing

1. `tests/gemma4_attn_dist.py` was broken from the day it was written —
   the KVRing rework landed the same day (2026-06-12) and the instrument
   still tuple-indexed the cache. Law (existing, re-confirmed): a fix
   isn't done while stale consumers of the old interface survive;
   instruments rot silently because nothing gates them.
2. Never background a child inside a flocked shell without `setsid` —
   an orphaned monitor inherits the lock fd and deadlocks the queue
   (happened once, cleared, pattern fixed).
3. Detached `nohup` trees launched from inside a seat's turn are reaped
   when the turn ends; long GPU runs must be harness-tracked background
   jobs (both E2 and the Gemma control died to this once each).

## Open successors (registered, not started)

- S1: E2 in a rotated basis (eigenbasis of S) — does the flattening
  survive a basis change, or is it a property of the channel frame?
- S2: same measurement on a KDA (channel-gated) checkpoint when one is
  locally runnable — does learned channel-wise decay re-introduce
  structure the delta rule alone removes?
- S3: 64-step / longer-context replication if any successor needs it;
  32-step reduction was a budget precaution (52.7s actual — 64 fits).
