# ORDER GLC-P0D — harmony-format discriminator (program GLC, Phase 0.d per Addendum 1)

## Grant

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` — edits + CPU
runs AUTHORIZED; no GPU in your sandbox: build + self-test + emit run
scripts; lead executes. READ-ONLY as before; artifacts append-only; new
code `scripts/glc_p0d_*.py`, artifacts `artifacts/glc_p0d/`. FORBIDDEN:
git, subagents, network, modifying existing defaults.

## Context

Read `docs/GLC_LONG_CONTEXT_PLAN.md` then
`docs/GLC_LONG_CONTEXT_ADDENDUM_1.md` (the registered fork you are
executing). Summary: HF transformers reproduces the long-vs-short
raw-text collapse (+2.51 nats @2048, +2.73 @2560, receipts
`artifacts/glc_p0/hf_reference/`), so the port is not implicated by the
raw instrument. H-GLC-2: raw-text completion is format-OOD for this
harmony post-trained model; likelihood collapses while capability
stands (96k recall gates pass). Your experiment discriminates.

## Work

1. Harmony-wrapped arms: for lengths {2048, 2560}, construct long and
   reference arms where the SAME wikitext token spans (from
   `artifacts/moe_e1_3b/controls.npz`, `wikitext_0_2560`) are presented
   inside the model's native chat template (use the tokenizer's own
   chat template / harmony conventions from the snapshot — document
   the exact rendered scaffold and its token count). The scored
   targets MUST remain the same 511 wikitext token ids (sha-matched to
   the raw receipts); the scaffold wraps the context, never the
   targets: scaffold + context-tokens + targets, scoring only target
   positions. Long arm wraps the full history; reference wraps only
   the short window. Any unavoidable deviation (e.g., scaffold tokens
   between context and targets breaking rawtext adjacency) must be
   designed AWAY (targets must directly continue their true preceding
   wikitext tokens inside the wrapped document block); state the
   design in the report.
2. Raw arms in the same runner (2 lengths × long/short) reusing the
   existing construction — same-run reproduction so the comparison is
   internal.
3. Runner constraints: HF BF16 CPU path as `glc_p0_hf_reference.py`
   (reuse its loader/scorer via import), --cpu-memory default 30GiB,
   threads 12, one cell per invocation, offload dir per cell. Emit
   `CPU_GLC_P0D_COMMANDS.sh` running: harmony-2048, harmony-2560,
   raw-2048, raw-2560 (raw cells may instead verify against the
   existing attempt01 receipts by sha and skip recompute — prefer
   skip-with-verification to save ~7 h; state which).
4. CPU self-tests: template rendering determinism, target-sha
   invariance, alignment math on synthetic tokens.

## Registered readings (from Addendum 1 — do not restate differently)

SUPPORTED: harmony deltas ≤ 0.30 nats at both lengths (raw ≥ 2.0
reproduced/verified). REFUTED: harmony deltas ≥ 1.0 at either length.
MIXED: between. Report the sentence exactly.

## Gates

- DG1: harmony cells built, target shas match raw receipts, scaffold
  documented (token counts + rendered text excerpt).
- DG2: self-tests pass.
- DG3: analyzer computes the four deltas + the registered sentence
  once lead-run receipts exist.

## Done

Final message verbatim: DG statuses (or NOT_MEASURED + script path);
the scaffold design paragraph; whether raw cells recompute or
verify-by-sha; files created/modified; anything you could not do.
