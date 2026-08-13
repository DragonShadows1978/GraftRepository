# APAMQ-FC — Engaged-Scoring Perplexity Arms (G-C)

YOUR WRITABLE TARGET is this git worktree (GraftRepository branch
`apamq-fb`) — `scripts/`, `tests/`, `docs/` AUTHORIZED (`core/`
read-only this order; the FB/FBD1 wiring is already in place). Run
first, report after; a registered order IS the permission.

HARD BOUNDARIES: canonical repos + FA worktree
(/mnt/ForgeRealm/wt/apamq-fa) READ-ONLY (the int4 arm imports the FA
engine via TENSOR_CUDA_ROOT — mechanism already in
scripts/apamq_fbd1_diag.py / tests). No git, no subagents, no
network, models read-only. Sandbox has NO GPU: every GPU leg = one
self-contained command the lead runs. RED honesty.

## Context (pre-nailed)

Plan: /mnt/ForgeRealm/Project-Tensor/docs/APA_MQA_FIX_PLAN.md (F-C)
and the APAMQ ledger's G-C registration (2026-08-13). Facts: G-B1
failed because the BLEND materializes bf16 score matrices (force_all
receipt: identical selection, Δattention 0.125); the fused kernel is
fp32-faithful (11/11 reference gates). The June +1.55–1.92% ppl
regression was measured over blend segments. G-C decides whether the
fixed stack (fused decode / int4 no-ring) ships default-ON, by
perplexity, not logit parity.

Scoring protocol laws (violations voided results before): APA must be
ENGAGED during scoring (June's 2048-window sweep scored without
engagement and read "within noise" — that scar is documented in
GEMMA4_MQA_ADJUDICATION.md); chunked scoring silently no-ops ring
quantization — decode-scored tokens must be L=1 with dtype asserted;
one arm per process; template-bound -it model → raw-text ppl absolute
values are high, only RELATIVE deltas between arms are meaningful —
same token stream, same protocol, every arm.

## Task

Build `scripts/apamq_fc_ppl.py --arm <name> --output <json>` with arms:

- `standard` — attention_mode standard.
- `apa_blend` — status quo APA (blend decode, mixed prefill), the
  baseline the ship-gate compares against.
- `apa_fused` — FB wiring, GEMMA4_APA_DECODE_FUSED=1, canonical
  engine.
- `apa_int4` — GEMMA4_APA_INT4=1 under TENSOR_CUDA_ROOT=FA worktree.

Protocol: a fixed public-text scoring set (reuse whatever wikitext
slice / scoring corpus the June sweep machinery used if present in
repo history/scripts; else a deterministic seeded slice of any local
plain-text corpus ≥ 200K tokens — name it), prefill S=8192 context
then score ≥ 2048 tokens per document via TRUE L=1 decode NLL
(teacher-forced), ≥ 25 documents, identical stream across arms
(seeded). Report per arm: mean NLL, ppl, relative Δ vs standard and
vs apa_blend with per-doc paired std/sem, decode ms/tok, and the
engagement receipts (APA-active call census per arm, ring dtype or
ring-absence assertion, branch census fused vs blend).

Also emit `--arm summary` that reads the four receipts and prints the
G-C table WITHOUT verdicts (adjudication is the lead's, thresholds
already registered).

Runtime budget: each arm must stay under ~45 min on the 4070S at
these shapes (scale document count down to fit, state the count);
4 arms + summary.

## Done

Final message verbatim: exact per-arm commands for the lead, CPU
check outputs, corpus identity + token counts, the engagement-receipt
mechanism (how each arm proves APA was scoring-engaged), and any
deviation. No GPU numbers; do not fabricate.
