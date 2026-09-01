# GRM-DET2 — D-LQR margin salvage (post-hoc, CPU-only, receipts-only)

## Provenance

Order authored by the analysis seat from David's queued question, per the
lead's brief. This order file is the seat's own capture of that brief and is
written BEFORE any analysis work, so the intent is fixed before results are
seen.

## Question (David-queued)

The GRM-DET1.5 detector race
(`artifacts/grm_det1/run_20260831T160525Z_2/det1_4/campaign/analysis/analysis_receipt_89bcd331ae7f9ef4.json`)
refuted D-LQR as a viable detector: 100% recall at 100% FPR, F1 0.6667,
`viable: false`. The stated reason is that "an unmounted graft outranks every
mounted graft" is true for essentially every query at this repository size, so
the detector never says no.

**Is D-LQR salvageable with a margin threshold, or is it structurally a
topic-matcher at repository scale?**

## Authorization and boundaries

- **YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository`** — specifically
  `orders/GRM_DET2_LQR_SALVAGE.md` and everything under
  `artifacts/grm_det2/`. Writes there are AUTHORIZED.
- RUN-FIRST AUTHORIZATION: proceed without further confirmation.
- **CPU-ONLY.** No GPU work of any kind. No model loads. No re-runs of any
  DET1 stage.
- **No network.**
- **No git.** The lead commits.
- **No subagents.**
- **No modification to any DET1.x code.** DET1 sources and receipts are
  read/import ONLY.
- Analysis operates exclusively on already-persisted receipts. If the raw
  D-LQR scores turn out not to be persisted at sufficient granularity, STOP
  and report `NOT_ANSWERABLE_FROM_RECEIPTS` rather than re-running anything.

## Method

1. Locate the persisted per-turn D-LQR observations that the race's trigger
   compared. Confirm granularity is sufficient (per-token
   mounted/unmounted routing scores on the same 10 planted-miss + 10 served
   evaluation turns the race scored).

2. Sweep a **margin-threshold family**: the decision statistic is
   `unmounted_best_score - mounted_best_score` per token, aggregated to a
   turn-level trigger. Grid the threshold over quantiles of the observed
   margin distribution.

3. Additionally evaluate any other **cheap decision rules the persisted data
   supports**, e.g.:
   - rank-gap / rank-position rules,
   - top-k overlap between the mounted set and the routed top-k,
   - ratio and normalized-margin variants,
   - aggregation variants (any-token vs fraction-of-tokens vs sustained run).

4. For each rule and operating point compute **recall, FPR, precision, F1** on
   the same 10 + 10 turns, using the same positive/negative definition the race
   used (positive = planted_miss, negative = served).

## Honesty requirements (binding)

- This is **post-hoc exploration on the race data**. The thresholds are being
  fit and scored on the same 20 turns. Any apparent winner is an in-sample
  optimum and **does not count** until a fresh, pre-registered confirmation run
  on held-out turns. This must be stated in the report, not buried.
- RED honesty: a negative or degenerate result is still a result. If the data
  says D-LQR is structurally undiscriminating, say so plainly.
- Any data quirk that caps achievable specificity (missing scores, degenerate
  fields, label-side artifacts) must be reported, not silently dropped or
  imputed.
- Spec is law: no threshold is adjusted after seeing the gate outcome, because
  no gate is being claimed here — this order registers **no** pass/fail gate.
  Its output is exploratory evidence plus a recommendation.

## Evidence class

Every claim in the report names its evidence class. The class available to this
order is **post-hoc re-analysis of persisted detector telemetry** — it can
establish separability (or its absence) of the persisted statistic on these 20
turns, and nothing about model quality, generalization, or repository-scale
behavior beyond the observed index sizes.

## Deliverable

Artifacts under `artifacts/grm_det2/`, content-addressed (payload SHA-256 in
the filename, receipt records its own inputs' hashes).

Report contains:
1. A **table of rule x operating points** with recall / FPR / precision / F1.
2. A **one-paragraph structural verdict**, one of:
   `salvageable-with-margin` / `structurally-undiscriminating` /
   `not-answerable-from-receipts`.
3. The **recommended registered follow-up**, if any — stated as something that
   would have to be registered and run fresh before any salvage claim counts.

## Done

Final message contains, verbatim:
- the rule x operating-point table,
- the structural verdict paragraph,
- the list of files created (absolute paths, with SHA-256),
- anything in this order that was NOT done, stated plainly.
