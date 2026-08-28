# ORDER MOE-RT2.1 — key refinement with hard negatives (bolt-on expert ABI, receipt #2.1) — CPU-ONLY

## Grant

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` — edits and CPU
runs AUTHORIZED. This order needs NO GPU: all inputs are the MOE-RT2
captures already on disk under `artifacts/moe_rt2/` (router-input fp16
npy, [24, 8, 512, 2880] per chunk, chunk0 = window indices 0–7, chunk1 =
8–15). Run everything yourself to completion in-sandbox.

READ-ONLY: `artifacts/moe_rt2/*` (never overwrite RT2 artifacts),
`scripts/gpt_oss20b_synthetic_key.py`, `scripts/gpt_oss20b_router_telemetry.py`
(import helpers freely, modify neither). New code in
`scripts/gpt_oss20b_key_refinement.py`; new artifacts in
`artifacts/moe_rt2_1/`. FORBIDDEN: git, subagents, network.

## Context (one paragraph)

MOE-RT2 met its registered rule (H-RT2 SUPPORTED, 16/24 qualifying
layers) but the code-anchor column exposed coarse addressability: the
domain-vs-generic difference-of-means key K1 fires on ~100% of CODE
tokens at every layer, and at late layers code outscores domain
(AUC_dc 0.33). K1 learned "not-wikitext", not "GRM domain". A
lead-verified single-layer probe (L12) showed a key fit against pooled
generic+code negatives reaches AUC_dc 0.887, code FPR 0.018, generic
FPR 0.000 — at recall 0.206 with a naive τ. RT2.1 sweeps key
constructions and τ policy across all 24 layers to find whether a
fine-grained, expert-usable operating point exists. Still zero model
runs: pure linear analysis over captured activations.

## Splits (same law as RT2; violation = G2 RED)

FIT = even window indices (0,2,...,14); EVAL = odd (1,3,...,15). Keys
and τ are fit on FIT only; every reported metric comes from EVAL only.
Record index provenance in the analysis JSON.

## Key arms (per layer; all fit on FIT splits only)

- K3 hard-negative diff-means: unit_norm(mean(h_dom) − mean(h_gen ∪ h_code)).
- K4 Fisher LDA: (Σ_w + λI)⁻¹ (mean(h_dom) − mean(h_gen ∪ h_code)),
  shared within-class covariance over the three FIT corpora, shrinkage λ
  chosen by a small grid validated WITHIN FIT (e.g. split FIT in half);
  unit-normalize the resulting direction.
- K5 logistic probe: linear logistic regression, domain=1 vs
  generic+code=0, FIT tokens only, L2-regularized (C grid validated
  within FIT). LABEL this arm honestly as a trained linear probe — it
  bounds what any linear gate could do; K3/K4 remain the closed-form
  story.

## τ policy (registered)

For each layer × arm, choose τ on FIT only: sweep the quantile grid
p90/p95/p99/p99.5 of pooled FIT negative scores, pick the τ maximizing
FIT recall subject to FIT generic FPR ≤ 0.02 AND FIT code FPR ≤ 0.05,
then FREEZE it and evaluate on EVAL. No eval-side tuning. Report the
full EVAL operating curve (recall vs FPRs across the grid) descriptively.

## Registered decision rule (fixed before any run)

H-RT2.1 SUPPORTED iff at least one layer × arm achieves ALL of the
following on EVAL at its frozen τ: domain recall ≥ 0.50, generic
FPR ≤ 0.02, code FPR ≤ 0.05. Report every qualifying (layer, arm), the
best row per arm, and bootstrap 95% CIs (resample over windows, ≥2000
resamples). If nothing qualifies: "fine-grained linear addressability
NOT DETECTED under these limits (linear keys, 8-window fits, token-level
gating, frozen-τ policy)" — a valid result; do not widen the rule
post-hoc. Window-majority firing rates are reported descriptively.

## Gates (registered; RED with receipts if failed)

- G0 input integrity: capture npy shapes/dtype match RT2 receipts;
  RT2 artifacts untouched (hash `analysis.json` before/after). 
- G1 math truth: reproduce the RT2 K1 layer-12 EVAL row (AUC_dg 0.9996,
  recall 0.995, generic FPR 0.011, code FPR 0.998) from raw captures
  with your own pipeline before running new arms — exact-match
  tolerance 1e-3. This proves your loader/split/AUC machinery.
- G2 split integrity: fit/τ/eval provenance recorded, zero overlap.
- G3 report `artifacts/moe_rt2_1/MOE_RT2_1_REPORT.md` with the
  registered verdict, per-layer × arm tables, operating curves, and a
  short mechanism note (what the hard-negative/LDA directions do that
  K1 did not).

## Constraints

CPU-only; do not import tensor_cuda or touch the GPU. Memory care:
mmap the npy files and process per layer (each layer slice is ~47 MB
fp16 per corpus per chunk). RED honesty; no monitor-idling. Evidence
class: linear analysis over captured activations of one model — no
model-quality or expert-viability claims.

## Done

Final message MUST contain verbatim:
1. G0/G1/G2/G3 verdicts, one line each, with key numbers.
2. The H-RT2.1 registered verdict sentence.
3. Best row per arm: layer, AUC_dg, AUC_dc, recall@τ, generic FPR,
   code FPR (EVAL, with CIs).
4. Count of qualifying (layer, arm) pairs.
5. Exact paths of every file created or modified.
6. Anything you could not do, stated plainly.
