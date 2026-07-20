# HYBRID IME E1/E2 — Attention-Tail Laws on a GDN Hybrid (Qwen3.5-9B)

**Status: PLAN — IMMUTABLE after initial commit** (house rules). Execution
detail lives in the ledger (`HYBRID_IME_E1E2_LEDGER.md`); meaning lives in
the synthesis. This document is the fixed source of intent.

Date: 2026-07-20. Lead: Fable (planner/ledger). Implementation: Opus 4.8
seats (window override of 2026-07-20), one order each, worktree-isolated.

## Context

Qwen3.6-27B (and its Bonsai ternary crush) put the GDN-hybrid layout in
front of us: ~75% Gated DeltaNet linear-attention sublayers + ~25% GQA
full attention carrying the entire growing KV cache. Analysis (this
session) established: APA's protected function has no operand in a GDN
layer at decode (keys are superposed into a fixed state), while the GQA
layers pass both axes of the APA selection law (population: full context;
multiplicity: 4 KV heads). Two measurable questions fall out, and the
local substrate already exists: the Qwen3.5-9B tensor_cuda port
(`core/qwen35_tc.py`, 32L = 24 GDN + 8 GQA 16q/4kv hd256), gated and
serving since 2026-06-12.

## E1 — Do a hybrid's retrieval layers show a different attention tail?

**Hypothesis (directional, registered):** the 8 GQA layers, having
delegated recency/position work to the GDN layers, show attention-mass
distributions at least as concentrated as dense-model layers — i.e. the
APA bulk+tail structure holds, possibly sharpened.

**Method:** port the `tests/gemma4_attn_dist.py` instrument to the Qwen3.5
port. Per GQA layer, per Q-head, at the last 64 query positions of each
workload prompt: post-softmax entropy, and fraction of probability mass in
the top 5% / 10% / 15% of keys by score. Capture by recomputing
scores from cached K and the live q at the capture point (the Gemma
instrument's pattern); do not modify the serving path.

**Comparator:** the same instrument run on the Gemma 4 12B port
(`tests/gemma4_attn_dist.py` as-is) in the same session, same-class
prompts — dense(-window)/MQA architecture vs hybrid GQA, one harness.
If the Gemma run does not fit the GPU window, E1 reports the hybrid
characterization alone and the control run becomes a successor.

## E2 — Does the tail law survive superposition into a delta-rule state?

**Hypothesis (registered):** per-channel readout contributions of the GDN
state show IME structure — geometric bulk + heavy tail — despite GDN's
scalar-per-head decay providing no gate-imposed channel differentiation.
If confirmed, this is a new IME domain instance (per-channel, inside a
superposed memory) and licenses a state-APA follow-up.

**Method:** at each captured decode step, for each GDN layer and head,
with post-norm query q and post-update state S (composed path, not the
fused kernel — the fused kernel internalizes the normalized q): compute
per-channel contributions c_i = ||q_i * S[i, :]||_2 for key-channels
i = 1..d_k. Characterize the distribution of {c_i} per (layer, head,
step): sorted-rank decay curve, top-10% channel mass share, and
persistence of the top-10% channel set across steps (Jaccard between
consecutive steps).

**Registered nulls:**
- N1 (alignment null): permute S's key-channel rows relative to q within
  each head (marginals preserved, q–S alignment broken).
- N2 (moment null): S replaced by Gaussian with matched per-head mean/var.

**Registered thresholds (fixed now, before any capture):**
- *Heavy-tail confirmed* iff the top-10% channel share is ≥ 2× its share
  under N1 in ≥ 75% of (layer, head) cells, sustained across the captured
  decode steps.
- *Stable-tail (state-APA license)* iff median consecutive-step Jaccard of
  the top-10% set ≥ 0.5.
- Below-threshold ≠ "no structure": curves are reported in full either
  way; a null result is stated as "not detected under [these prompts,
  this model, these stats]" (hold-the-middle).

## Shared workload (both experiments, identical prompts)

Three prompts, ~2048 tokens prefill each, then 64 greedy decode steps:
1. natural prose (public-domain text or a repo README on disk),
2. code (a real source file from this repo),
3. synthetic retrieval: ~40 key–value facts planted early, queries at the
   end (the retrieval-heavy case the hybrid design routes to GQA layers).
Deterministic: greedy decode, fixed prompts committed to the worktree,
seeds fixed where sampling exists.

## Evidence class + scope laws

Both experiments are **instrument measurements (distribution
characterization)**. No model-quality claims, no speed claims, no
storage-quant claims (APA scope law: computation ≠ storage). Instrument
must not perturb serving outputs: each capture script asserts final
logits of an instrumented run match an uninstrumented run on prompt 1
(composed-path-vs-composed-path).

## Constraints

- GPU: single-GPU, runs ≤ 10 min each (power-safety program rule); GPU
  work serialized via `flock -w 3600 /tmp/forge-gpu.lock`.
- Seats: no git, no subagents, RED honesty, receipts verbatim in the
  final message. Lead commits.
- Port semantics untouched: capture is additive (new scripts + opt-in
  env-flagged hooks only); parity assertion mandatory.

## Deliverables

Per experiment: capture script under `tests/`, raw arrays (`.npz`) +
JSON summary under `logs/`, printed summary table, ledger-ready receipt
block. Lead writes the synthesis and the board entry after verification.
