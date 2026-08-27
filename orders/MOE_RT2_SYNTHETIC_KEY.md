# ORDER MOE-RT2 — synthetic-key addressability on the frozen GPT-OSS-20B router (bolt-on expert ABI, receipt #2)

## Grant (read this first)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` — edits, builds,
and GPU runs inside this repo are AUTHORIZED. A registered order IS the
permission: run first, report after. Do not ask permission for anything
non-destructive.

READ-ONLY paths (import/read in place, never write):
- `/mnt/ForgeRealm/Project-Tensor` (tensor_cuda engine)
- `/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/...`
- `/home/vader/.cache/huggingface/hub/datasets--wikitext`

FORBIDDEN: git (lead commits), subagents, network, editing anything under
Project-Tensor. `scripts/gpt_oss20b_router_telemetry.py` (MOE-RT1 harness,
committed at 557abcf) is READ-ONLY — import its helpers freely; do not
modify it. New code goes in `scripts/gpt_oss20b_synthetic_key.py`
(+ optional `_analyze` sibling).

EXPECTED SANDBOX LIMIT: your worker likely exposes no `/dev/nvidia*`
(MOE-RT1 precedent). If CUDA is unavailable: build and validate
everything that can run on CPU, emit a blocked-report with ALL gates RED
as NOT_MEASURED plus exact GPU resume commands (the MOE-RT1 pattern —
it was correct). Never substitute a missing measurement with a value.

## Context (one paragraph)

MOE-RT1 (artifacts/moe_rt1/, all gates GREEN, lead-verified) established
that GPT-OSS-20B's pretrained routing geometry separates domains:
expert-utilization JS(domain, generic) beats within-generic nulls by
7–24× at all spot-checked layers, and per-layer τ-calibration tables of
native logit quantiles exist. RT2 tests the GATE half of the bolt-on
expert ABI: can a NEW key vector, constructed in closed form from the
frozen router-INPUT geometry (no training anywhere), address the GRM
domain with useful precision — firing on domain tokens, silent on
generic/code? This is the "external expert router" of the design with
the expert itself stubbed out. Zero training. Zero model modification.

## Data + splits (registered; violation = G2 RED)

Reuse the EXACT MOE-RT1 windows: load
`artifacts/moe_rt1/corpus_windows.npz` and verify corpus sha256s against
`artifacts/moe_rt1/corpora_manifest.json` (mismatch = stop, report).
Each corpus has 16 windows × 512 tokens. Split by window index parity:
- FIT split = even-index windows (8 per corpus)
- EVAL split = odd-index windows (8 per corpus)
Key construction may touch ONLY FIT-split data (domain + generic).
τ-calibration may touch ONLY GENERIC-EVAL... no — τ must not touch the
same tokens it is evaluated against. Registered correction: calibrate τ
on GENERIC-FIT scores; evaluate FPR/recall on EVAL splits only. All
reported metrics come from EVAL splits exclusively. Leakage receipts:
the analysis JSON must record which window indices fed key, τ, and eval.

## Capture

New GPU mode: per layer (all 24), per token, dump the router INPUT
hidden state h (the exact tensor the native router multiplies,
post-attention-norm, 2880-dim) as fp16, chunked to keep each GPU
invocation ≤590 s (mirror the RT1 chunking; ~1.2 GB total on disk is
fine, write under `artifacts/moe_rt2/`). Reuse RT1's loading/forward
path via import.

## Key arms (both closed-form; registered)

- K1 (PRIMARY): difference of means — k = unit_norm(mean(h_domain_FIT)
  − mean(h_generic_FIT)), one k per layer.
- K2 (secondary): domain centroid — k = unit_norm(mean(h_domain_FIT)).

Score s(h) = h·k. Per layer per arm: τ = p99 of s over GENERIC-FIT
tokens.

## Metrics (per layer × arm, EVAL splits only, bootstrap 95% CIs over windows)

1. Token-level AUC: domain-eval vs generic-eval; domain-eval vs
   code-eval.
2. Domain recall at τ; generic FPR at τ (should be ≈1% by construction
   — report the realized value, drift is a finding); code FPR at τ.
3. Window-majority firing: fraction of eval windows where >50% of
   tokens fire, per corpus.
4. Score-scale context: quantiles of s on each corpus alongside the
   native top-1 logit quantiles from RT1's calibration table (same
   layer), so the report can show where a bolt-on gate sits relative
   to native routing scale.

## Registered decision rule (fixed before any run)

H-RT2 SUPPORTED iff, for PRIMARY arm K1, at least one layer achieves
BOTH on EVAL splits: (a) domain-vs-generic token-level AUC ≥ 0.90 with
bootstrap 95% CI lower bound ≥ 0.85, AND (b) domain recall at τ ≥ 0.30
with realized generic FPR ≤ 0.02. Report the count of qualifying layers
and the best layer. Code FPR and K2 are descriptive (reported, not part
of the rule). If no layer qualifies: "synthetic-key addressability NOT
DETECTED under these limits (closed-form keys, 8-window fits, token-level
gating)" — a valid result; do not soften it, do not widen the rule
post-hoc.

## Gates (registered; a failed gate is a RESULT — report RED with receipts)

- G0 window identity: corpus sha256s and window tensor match RT1
  manifest exactly. Receipt: sha comparison lines.
- G1 capture truth: on ≥1 small batch, captured h re-multiplied against
  the safetensors router weights in NumPy reproduces RT1-captured
  router logits (top-4 indices exact; state tolerance). This proves the
  captured h is the true router input, not a nearby tensor. Receipt:
  comparison numbers.
- G2 split integrity: fit/τ/eval window-index provenance recorded; no
  index appears on both sides. Receipt: the index lists.
- G3 report `artifacts/moe_rt2/MOE_RT2_REPORT.md` with the registered
  verdict in the registered vocabulary, per-layer tables, and the
  score-vs-native-logit-scale comparison.

## Run constraints (standing house laws — mandatory)

- Every GPU invocation wrapped in `flock -w 7200 /tmp/forge-gpu.lock`,
  ≤10 minutes wall-clock each, idle gaps between invocations, single
  GPU, no sustained continuous load. Operator has absolute right of way.
- No monitor-idling; run gates synchronously.
- Evidence class: inference measurement + closed-form linear analysis on
  one model. No expert-viability, no model-quality claims.

## Done

Your final message MUST contain, verbatim:
1. G0/G1/G2/G3 verdicts, one line each, GREEN/RED with the key number.
2. The H-RT2 registered verdict sentence (or NOT_MEASURED + resume
   commands if GPU-blocked).
3. Best-layer row: layer, K1 AUC (dom-vs-gen and dom-vs-code), recall@τ,
   realized generic FPR, code FPR.
4. Exact paths of every file created or modified.
5. Anything you could not do, stated plainly.
