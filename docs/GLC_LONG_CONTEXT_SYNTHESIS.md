# GLC — Synthesis (Phase 0 CONCLUDED 2026-08-28 late night)

## Verdict

**The port is fully exonerated. The instrument was invalid.**
GPT-OSS-20B — the model itself — has degenerate raw-text likelihood
that worsens with context length. Likelihood-based long-context
instruments (raw or harmony-wrapped) are INVALID on this model.
Long-context quality on GPT-OSS is measurable only by task-based
instruments (recall/QA/retrieval gates) — which the port already
holds at 96k.

## The receipt chain (each step lead-verified)

1. E1 behavioral eval showed teacher ppl up to 186k (suspected bug).
2. E1.2-DIAG: eval code exonerated (established path reproduces
   exactly); per-position receipts showed flat-high NLL with
   degenerate "…" top-1.
3. E1.3b: BOTH port backends (standard, apa_selective) degrade
   identically on contiguous wikitext (long/short ppl ratio ~8× @1024,
   30–40× @2048–2560); YaRN/rope/sink/interleave audit vs HF = exact
   match.
4. GLC-P0 (HG1): HF transformers BF16 reproduces the collapse on the
   same cells (+2.51 nats @2048, +2.73 @2560). Stream contiguity and
   scoring alignment verified by lead code-read.
5. GLC-P0D: harmony-format wrap REFUTED as rescue — deltas +4.66/+3.92
   nats, WORSE than raw (registered H-GLC-2 sentence: REFUTED).
6. Confidence signature (port receipts, w11 teacher arm): top-1 "…"
   at every sampled position, logits 14–22, true tokens at ranks
   82–47,032 — a high-confidence degenerate attractor, not noise.
7. External literature (evidence class: external, URLs in ledger):
   huggingface/transformers#40990 "Extremely high perplexity on
   openai/gpt-oss-20b with WikiText-2 (raw)" (~394);
   ggml-org/llama.cpp#15155 "gpt-oss-20b perplexity broken" —
   independent implementations worldwide observe the same behavior.

## Laws landed

- **Likelihood ≠ capability on distilled instruct-only models.**
  GPT-OSS-20B passes 96k recall while its raw NLL collapses at 2k.
  Never gate long-context quality on raw-text ppl for a model with no
  base-release; use task-based instruments.
- **The E1 "premise finding" and every ppl-based E1 verdict are void
  as measurements of the consolidation premise** — the instrument
  could not have measured it on this model. The premise itself
  (in-context guides → distillable hidden-state shift) remains
  UNTESTED, neither supported nor refuted.
- Phase 1 (fix wave) is CANCELLED — nothing to fix. Phase 2's
  "institutionalized long-window ppl gates" are cancelled for this
  model; the recall-class gates remain the standard.
- The standard full-attention backend remains less-validated than
  apa_selective at long S (no independent receipt distinguishes them
  beyond parity with each other here); parity receipt optional,
  low priority.

## Open decision (David): where E1 goes next

(a) Redesign E1's behavioral gate task-based on GPT-OSS (teacher-
    agreement probes / in-format generation judged externally);
(b) Move the consolidation testbed to a base-class MoE with healthy
    likelihood — OLMoE-1B-7B is already in the HF cache (64 experts,
    top-8, true base model): ppl instruments valid there, and the
    RT1/RT2 key machinery ports directly;
(c) Park the consolidation arc; the addressability receipts
    (RT1/RT2/RT2.1, L22 install row, ABI G0) stand on GPT-OSS
    regardless.
Lead recommendation: (b) — it restores a valid instrument stack for
the E-series while GPT-OSS remains the addressing/ABI testbed.
