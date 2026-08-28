# ORDER MOE-E1.4 — backend re-specification: E1 long forwards move to the proven apa_selective path; teacher re-sweep + chain re-run

## Grant

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` — edits + CPU
runs AUTHORIZED; no GPU in your sandbox: build, self-test, emit scripts;
lead runs GPU. READ-ONLY as in prior MOE orders; all moe_e1* artifacts
append-only (`_e14` suffixes). FORBIDDEN: git, subagents, network.

## Context (one paragraph)

E1.3b closed the mechanism hunt: A1 found E1 forces the `standard`
full-attention backend (expert_e1.py:3341 → explicit causal mask +
standard sink softmax) for every forward, while all proven long-context
receipts on this model (context ladder, 96k) run `apa_selective` (fused
causal sink attention); the BG2 ramp then showed the standard backend
progressively collapsing on IDENTICAL wikitext targets as sequence
length grows (long/reference ppl ratio ~8× at 1024, 30–40× at
2048–2560), while 512-token scoring is healthy. Verdict MIXED:
backend defect dominant, content modulates severity via the healthy
sliding layers. Consequences: E1's 512-token key captures remain VALID;
every 2,560-token artifact (teacher scores, pair captures, the trained
adapter, G4/G5' evals, and E1.3's C0/prefix conclusions) is built on
the broken backend and is void. This order re-specifies the backend and
re-runs what is void. Gate semantics (G0/G1/G3/G4/G5' and E1.1's G2')
are UNCHANGED; only the forward backend and, if needed, the teacher
construction are re-registered.

## Work

1. Backend switch: E1 forward paths (capture-pairs, eval-gates, diag
   scoring) run `apa_selective` exactly as the context-ladder/96k
   lineage invokes it (same kernel entry, same parameters). New flag
   `--backend e14-apa` (default remains the historical standard path
   for receipt reproducibility). ABI note: the expert residual add and
   the G0 identity contract are backend-independent — verify G0 still
   holds bit-identical under the new backend (it is a registered gate
   in the re-run).
2. Backend sanity gate (registered, runs FIRST): repeat the BG2 ramp's
   2048 and 2560 cells under `--backend e14-apa`. PASS iff
   long/reference mean-NLL delta ≤ 0.15 nats at both lengths. FAIL →
   stop, report (the proven backend not reproducing health at 2.5k
   would be a new finding).
3. Teacher re-adjudication under the fixed backend: re-run E1.3's C0
   and the 16-cell P-sweep PLUS a P0 arm = the ORIGINAL registered E1
   teacher construction (raw-excerpt rotation). Selection rule as
   E1.3 registered (highest mean gap > 0 wins; P0 winning means no
   redesign was needed and the E1.1 teacher stands).
4. Chain re-run script: from capture-pairs onward under the winning
   teacher + e14-apa backend — pairs, train, G0/G1/G3, G4, G5'
   (G5' = E1.1 amended semantics), analyze. Append-only `_e14`
   receipts; ExpertPack v0_e14.
5. Engine successor note (report only, no action): the standard
   full-attention mask/sink backend in core/gpt_oss20b_tc.py degrades
   with S — file it as a Project-Tensor-adjacent open defect with the
   BG2 receipts; do not attempt to fix the backend in this order.

## Gates

- FG0 backend sanity (step 2) with the two deltas.
- FG1 teacher selection table (17 cells + P0) + winner sentence.
- FG2 chain scripts emitted, CPU self-tests pass, historical paths
  untouched by default.
(The re-run chain itself carries the unchanged E1.1 gates; lead runs
it and evaluates.)

## Constraints

Standing GPU laws in emitted scripts (flock, ≤590 s, 30 s+ gaps,
single GPU). RED honesty. Evidence class: behavioral inference
measurement, one model, one corpus. ## Done: FG verdicts or
NOT_MEASURED + script paths; the P0-vs-P1..P4 table verbatim once
lead-run; files created/modified; anything you could not do.
