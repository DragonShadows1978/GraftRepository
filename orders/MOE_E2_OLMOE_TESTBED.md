# ORDER MOE-E2 — E-series restart on OLMoE-1B-7B (base): easy-expert-add testbed

## Grant

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` — edits + CPU
runs AUTHORIZED (OLMoE at 7B bf16 fits this sandbox's CPU/RAM — you MAY
self-run CPU forwards for small gates; keep individual invocations
bounded and state walls). No GPU in your sandbox; emit GPU scripts for
the heavy sweeps, lead runs them. READ-ONLY: HF cache
(`models--allenai--OLMoE-1B-7B-0924` — the BASE model, NOT -Instruct),
wikitext cache, the narrative guides corpus paths granted in
`orders/MOE_E1_NARRATIVE_EXPERT.md` (same three Active_* folders ONLY,
same exclusions), all existing artifacts (append-only). New code
`scripts/olmoe_e2_*.py`; artifacts `artifacts/moe_e2/`. FORBIDDEN: git,
subagents, network, touching GPT-OSS scripts.

## Context (one paragraph)

David's adjudication 2026-08-29: the E-series is an experiment in
GROWING a frozen MoE by adding experts — iteration ease beats
capability. Testbed moves to OLMoE-1B-7B-0924 (base): 64 experts top-8
per layer, 1.3B active, ctx 4096, and — decisive — a true base model,
so likelihood instruments are VALID (unlike GPT-OSS, see
`docs/GLC_LONG_CONTEXT_SYNTHESIS.md`; its laws bind this order). This
order stands up the testbed on an HF-transformers harness (no
tensor_cuda port needed for research receipts; the port question comes
later if the arc graduates), re-establishes the address-space receipts
on the new model, and re-runs the full consolidation experiment with
the E1/E1.1 gate semantics.

## Registered design (carried from the GPT-OSS arc, receipts logged)

- ABI: threshold-gated residual side-path at install layer L\*; expert
  reads the router's input hidden state h; adds `B·silu(A·h)` iff
  `h·k ≥ τ`; A: d→64, B: 64→d zero-init; unfired tokens and
  zero-install forwards bit-identical to base (G0).
- Key: shrinkage Fisher LDA on router-input h, narrative (guides) vs
  pooled DIVERSE negatives {wikitext, repo code, GRM docs}, fit/eval
  window-parity splits, frozen fit-side τ (RT2.1 policy).
- Teacher: base + 2,048-token guide prefix in context (P0 raw-excerpt
  rotation is PRIMARY again — valid on a base model). Registered
  fallback: if the G4 precondition fails (teacher gap ≤ 0 on held-out
  windows), run the P1 prose-filtered prefix arm once before any
  premise sentence.
- Corpus: the same 35 guide files, same file-level 70/30 TRAIN/HELDOUT
  split rules as E1 (fresh split receipts for this order).

## Stages (modes; self-run what fits CPU, script the rest)

1. `bringup`: load base OLMoE bf16, deterministic forward config
   documented (GPU runs may place layers via accelerate with an 11GiB
   VRAM cap; quantized backends FORBIDDEN — injection fidelity and G0
   bit-identity require bf16). Sanity receipt: short-window wikitext
   ppl in a plausible base-model range AND long-vs-short delta on
   contiguous wikitext {2048 vs 512} ≤ 0.15 nats — the GLC lesson as
   a permanent bring-up gate (E2-G-1). If E2-G-1 fails, STOP: the
   testbed is invalid too.
2. `capture-keys` + `fit-key`: router-input captures (all layers) on
   narrative/wikitext/code/GRM-doc windows; K4 fit; address gate
   E2-G2 = RT2.1 rule + grm FPR ≤ 0.05 (recall ≥ 0.50, generic FPR
   descriptive per David's E1.1 in-remit adjudication, code ≤ 0.05).
3. `capture-pairs` (teacher/student at L\*), `train` (r=64, MSE vs
   zero-predictor ≥ 10% = E2-G3), `eval-gates`: E2-G0 identity;
   E2-G4 behavioral primary — held-out ppl recovery ≥ 25% of the
   teacher gap (precondition: gap > 0, fallback arm registered
   above); E2-G5 non-interference — wikitext ppl delta ≤ 0.5% with
   gate live, code fire ≤ 5%, per-window deltas on firing windows
   descriptive.
4. `analyze`: tables, bootstrap CIs over windows, report
   `artifacts/moe_e2/MOE_E2_REPORT.md`, ExpertPack
   `artifacts/moe_e2/expertpack_narrative_olmoe_v0/`.

## Constraints

GPU discipline for emitted scripts: flock `/tmp/forge-gpu.lock`,
≤590 s per invocation, 30 s+ gaps, single GPU. CPU self-runs: state
walls, stay bounded per invocation. RED honesty; a failed gate is a
result; no post-hoc widening. Evidence classes: behavioral inference
measurement, one model, one corpus; likelihood instruments valid on
this model per E2-G-1 receipt only.

## Done

Final message verbatim: E2-G-1/G0/G2/G3/G4/G5 statuses with key
numbers (or NOT_MEASURED + exact lead script paths); the install row;
the G4 row (ppl_base / ppl_teacher / ppl_expert / recovery% + CI) once
measured; corpus split receipts; files created/modified; anything you
could not do, stated plainly.
