# ORDER MOE-RT1 — GPT-OSS-20B native router telemetry (bolt-on expert ABI, receipt #1)

## Grant (read this first)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` — edits, builds,
and GPU runs inside this repo are AUTHORIZED. A registered order IS the
permission: run the experiment first, report after. Do not ask permission
for anything non-destructive.

READ-ONLY paths (import/read in place, never write):
- `/mnt/ForgeRealm/Project-Tensor` (tensor_cuda engine — already on sys.path
  in the existing gpt_oss20b scripts; use it exactly as they do)
- `/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/...`
  (model snapshot; path constant `SNAPSHOT` in
  `scripts/gpt_oss20b_moe_diag_smoke.py`)
- `/home/vader/.cache/huggingface/hub/datasets--wikitext` (generic corpus)

FORBIDDEN: git (the lead commits everything), subagents, network access,
editing anything under Project-Tensor, editing existing gate scripts'
behavior (new script only; shared helpers may be imported, not modified).

## Context (one paragraph)

We are designing an ABI for bolting addressable expert modules onto a
frozen MoE (GRAPA/GRM consolidation arc). The design bet is that the
pretrained router's logit space is a usable address space: a bolt-on
expert would gate on `h·k_new` calibrated against native logit
quantiles. Before designing anything further we want receipts from the
one frozen MoE running on this stack — GPT-OSS-20B (24 layers, 32
experts/layer, top-4, real router in `core/gpt_oss20b_tc.py`:
`router_weight`/`router_bias`). This order is a READ-ONLY-with-respect-
to-model-behavior diagnostic: instrument the router, sweep three
corpora, produce the calibration tables and the domain-signal
comparison. No model behavior changes. No expert is built in this order.

## Registered hypothesis + decision rule (fixed before any run)

H-RT1: the pretrained routing geometry carries domain signal — i.e.
routing statistics distinguish a niche domain (GRM docs) from generic
English beyond within-corpus noise.

Registered comparison: for each layer L, compute expert-utilization
distributions (over the 32 experts, counting top-4 selections) for:
  (a) DOMAIN vs GENERIC → JS divergence `JS_dg(L)`
  (b) GENERIC split-half A vs B (disjoint windows) → `JS_null(L)`
     (the within-corpus null; use ≥8 random split-half resamples to get a
     null distribution with spread)
H-RT1 is SUPPORTED at layer L iff `JS_dg(L)` exceeds the 95th percentile
of the null resamples at that layer. Report the count of supporting
layers out of 24. If few/no layers clear the bar, the verdict is
"domain signal NOT DETECTED under these limits (corpus sizes, JS on
top-4 utilization)" — that is a valid result; do not soften or inflate
it. Secondary (descriptive, no pass/fail): routing-entropy and margin
deltas domain-vs-generic per layer with bootstrap 95% CIs.

## Corpora (all local; record sha256 + token counts in artifacts)

1. GENERIC — wikitext from the HF datasets cache (test/validation split
   text is fine; concatenate to plain text). If the cached layout is
   awkward, extract the raw text file(s) directly from the cache — no
   network, no `datasets` download call that would touch the net.
2. CODE — concatenation of `.py` + `.cpp`/`.h` sources from THIS repo
   (`core/`, `scripts/`, `cpp/`). A second contrast corpus.
3. DOMAIN — GRM/GRAPA documentation: `docs/GRM_Primer.md` plus other
   `docs/*.md` from this repo (list the exact files used). This is the
   niche domain a future consolidated expert would serve.

Minimum sample per corpus: ≥8,192 routed tokens (≥16 windows × 512
tokens through the full 24-layer forward). More is better if it fits the
run-length constraint below; report actuals per corpus.

## Work

1. New script `scripts/gpt_oss20b_router_telemetry.py`, patterned on
   `gpt_oss20b_stream_forward_smoke.py` / `gpt_oss20b_realtext_ppl_gate.py`
   (reuse their loading/window-building approach; prefill-only, no
   generation needed). Telemetry captures, per layer per token: the full
   32-expert router logit vector (fp32), top-4 indices, and post-softmax
   gate weights. Dump to `artifacts/moe_rt1/` as compressed npz/json
   with full provenance (corpus sha256, window offsets, config, commit).
2. Analysis (same script or a sibling `_analyze.py`), producing per
   layer × corpus:
   - full-softmax routing entropy over 32 experts: mean/median/p90;
   - CALIBRATION TABLE: quantiles (p5/p25/p50/p75/p95) of the top-1
     logit and of the 4th-place (selection-boundary) logit — this is the
     table a bolt-on gate threshold τ would be set against; also margins
     top1−top2 and top4−top5;
   - expert-utilization histogram + the registered JS comparison;
   - mean max-gate weight (post-softmax concentration);
   - consecutive-token expert-set churn (|set(t) Δ set(t+1)|/4 mean).
3. Report `artifacts/moe_rt1/MOE_RT1_REPORT.md`: methods, corpora
   provenance, the calibration tables (per-layer, readable), the
   registered H-RT1 verdict, secondary descriptives, and any anomalies.

## Gates (registered; a failed gate is a RESULT — report it RED with receipts)

- G0 non-invasiveness: with telemetry ON vs OFF, final-layer hidden
  states (or logits) for the same window are BIT-IDENTICAL. Receipt:
  byte-equality check output in the report.
- G1 telemetry truth: on ≥1 small batch, router logits re-derived in
  NumPy directly from the safetensors router weights match captured
  telemetry (top-4 indices exact; logits within dtype-appropriate
  tolerance, state it). Receipt: comparison numbers.
- G2 sweep completeness: all 3 corpora ≥ minimum token counts, artifacts
  written with provenance. Receipt: file list + token counts + sha256s.
- G3 report delivered with the registered H-RT1 verdict stated exactly
  in the registered vocabulary (SUPPORTED at N/24 layers / NOT DETECTED
  under limits).

## Run constraints (standing house laws — mandatory)

- GPU right-of-way: wrap every GPU invocation in
  `flock -w 7200 /tmp/forge-gpu.lock <cmd>`. The operator has absolute
  right of way.
- POWER SAFETY (standing, non-negotiable): each GPU run ≤10 minutes
  wall-clock; split the sweep into per-corpus (or per-corpus-chunk)
  short runs with idle gaps between them; no sustained continuous GPU
  load, single GPU only. If the minimum token counts cannot be met
  within these bounds, meet what fits, report actuals, and say so —
  that is the honest result.
- No monitor-idling: run gates synchronously to completion.
- Evidence classes: everything here is inference measurement on one
  model; make no model-quality or expert-viability claims — this order
  characterizes routing statistics only.

## Done

Your final message MUST contain, verbatim:
1. G0/G1/G2/G3 verdicts, one line each, GREEN/RED with the key number.
2. The H-RT1 registered verdict sentence.
3. Per-corpus actual routed-token counts and corpus sha256s.
4. The calibration-table row for layer 12 (as a sample), pasted from the
   report.
5. Exact paths of every file you created or modified.
6. Anything you could not do, stated plainly.
