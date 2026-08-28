# GLC — Addendum 1 (2026-08-28): HF degrades too — instrument-validity fork

Registered re-plan under the plan's Phase-0 trigger ("If HF also
degrades, the protocol or model is implicated"). The plan itself remains
immutable; this addendum extends Phase 0.

## Receipts that fired the trigger

- HG1 cells (HF transformers 5.12.0, BF16 dequant, alignment verified
  by lead code-read of `glc_p0_common.ramp_arm` + `logits_to_keep`
  scoring): 2048 → delta +2.510 nats DEGRADED; 2560 → delta +2.727
  nats DEGRADED. Same targets, contiguous-wikitext history (seams
  lead-verified as continuous corpus text).
- P0.c had already excluded table-level divergence; P0.b port-side
  captures on disk; remaining P0.a cells (1024/1536/2304) STOPPED as
  low-value post-trigger (receipts of completed cells preserved).

## Reframe hypothesis (H-GLC-2, registered)

GPT-OSS-20B is a harmony-format post-trained model with no base-model
release. Raw-text completion at long range is out-of-distribution for
its post-training; likelihood collapses (degenerate "…" top-1
predictions in E1.2-DIAG receipts) while CAPABILITY remains intact —
consistent with the port's 96k recall gates passing while raw-text ppl
collapses at 2k. Under H-GLC-2 the port is exonerated for this
phenomenon; the raw-completion long-ppl instrument is invalid for this
model, and long-context quality must be gated in-format or task-based.

## Phase 0.d — discriminating experiment (order GLC-P0D)

Same targets, same long/short protocol, but each arm wrapped in the
model's native harmony format (chat template) with the wikitext
presented as in-conversation document content. Registered readings:
- H-GLC-2 SUPPORTED: harmony-wrapped long-vs-short delta ≤ 0.30 nats
  at 2048 and 2560 (an order of magnitude below the raw deltas), with
  raw arms reproducing ≥ 2.0 nats in the same runs.
- H-GLC-2 REFUTED: harmony-wrapped deltas remain ≥ 1.0 nats — the
  degradation is format-independent and the hunt returns to shared
  execution machinery (P0.b HF capture becomes next).
- Intermediate (0.30–1.0): partial mitigation — report as MIXED, both
  successors stay open.

## Consequences if SUPPORTED

- Port parity (PG2) remains worth one receipt (port-vs-HF at matched
  protocol) but PG1's "ramp health" gate is re-scoped to in-format
  scoring; Phase 2's institutionalized long-window gates become
  in-format gates.
- MOE-E1's teacher arm must present guides in harmony format — the
  E1 teacher explosions and E1.3's content-severity pattern are
  re-read as format-OOD amplification, and E1.4 re-runs with an
  in-format teacher.
