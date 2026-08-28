# ORDER MOE-E1.3b — decisive position controls + long-context path audit

## Grant

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` — edits + CPU
runs AUTHORIZED; no GPU in your sandbox: build modes, self-test on
synthetic data, emit the run script; lead executes. READ-ONLY as in
prior MOE orders; all moe_e1* artifacts append-only; new artifacts in
`artifacts/moe_e1_3b/`. FORBIDDEN: git, subagents, network, changing
default behavior of any existing mode.

## Context (one paragraph)

E1.3's C0 (contiguous same-file guide stream) exploded (ppl 3,912) and
correctly reopened MECHANISM=BUG — but C0 is confounded: its prefix is
guide text, so content pathology and positional breakage both predict
the explosion. Current receipt set is contradictory under every simple
theory: w15 teacher at 2,560 tokens scores nearly sane (98 vs 58), yet
clean-prose wikitext prefixes hurt guide windows 6–49×, and 768-token
sequences are mildly hurt (w11 guide256: 3.7×). Lead's reconciling
hypothesis: the E1/diag forward path damages FULL-ATTENTION layers for
positions past ~2048 (RoPE/YARN table or mask construction) while the
interleaved 128-token sliding-window layers keep functioning — so
degradation severity = the window's dependence on long-range context
(open prose dies; locally-predictable formatted text survives on iSWA
layers alone). This order gets the discriminating receipts.

## Controls (registered; base arm only, no teacher)

- C1 contiguous WIKITEXT 2,560: first 2,560 tokens of the frozen RT1
  wikitext corpus stream; score the last 511 targets. Reference R1:
  the SAME 511 targets scored in a 512-token window (tokens
  2048..2559 alone). Positional theory predicts C1/R1 mean-NLL ratio
  >> 1; content theory predicts ≈ 1.
- C2 contiguous WIKITEXT 768: tokens 0..767, score last 511. Should be
  sane under both theories (all-sub-2048 sanity anchor).
- C3 ramp: contiguous wikitext lengths {1024, 1536, 2048, 2304, 2560},
  each scoring its last 511 targets against a same-tokens short-window
  reference. Localizes the break position if C1 is hot.
- C4 guide-window cross: w11's 512 window with a 256-token WIKITEXT
  prefix (768 total; short splice, clean prose). Compares against the
  guide256 arm (3.7×) to size splice-derail vs position effects below
  2,048.

## Audit (registered)

A1: line-level comparison of RoPE/YARN table construction, full-attn
mask, and sink handling between (a) the E1/diag forward and (b) the
long-context path that PASSED the GPT-OSS 96k bench and context-ladder
gates (`scripts/gpt_oss20b_context_ladder.py` lineage). D5 compared
E1-vs-capture-pairs only — this compares against the path with a
proven long-context receipt. Name every divergence file:line, or state
none exists. If a defect is found: fix behind `--eval-fix e13b`
(default off), original preserved.

## Gates

- BG1: C1/R1 + C2 numbers with the ratio stated.
- BG2: C3 ramp table (5 lengths).
- BG3: C4 number in context with guide256.
- BG4: A1 audit verdict verbatim (divergences or none) + mechanism
  sentence: POSITIONAL-BUG (with file:line) / CONTENT-REAL /
  MIXED (state the split), justified only from BG1–BG3 receipts.
(Evaluate after the lead GPU run; your CPU portion = build, self-test,
static audit A1, emit `GPU_E13B_COMMANDS.sh` with standing flock/590s/
sleep discipline. ~12 bounded runs.)

## Done

DG-style final message: BG verdicts or NOT_MEASURED + script path; A1
verdict verbatim; files created/modified; anything you could not do.
