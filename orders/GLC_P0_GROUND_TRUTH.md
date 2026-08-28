# ORDER GLC-P0 — ground truth + localization (program GLC, phase 0)

## Grant

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` — edits + CPU
runs AUTHORIZED; no GPU in your sandbox: build, self-test, emit run
scripts; lead executes GPU. READ-ONLY: `/mnt/ForgeRealm/Project-Tensor`
(engine, import in place), the HF snapshot
(`models--openai--gpt-oss-20b/snapshots/6cee5e81...`), the wikitext
cache, all existing artifacts (append-only). New code:
`scripts/glc_p0_*.py`; artifacts: `artifacts/glc_p0/`. FORBIDDEN: git,
subagents, network, modifying any existing script's default behavior.

## Program context

Read `docs/GLC_LONG_CONTEXT_PLAN.md` (immutable) first. You are
executing Phase 0: P0.a reference arm, P0.b first-divergence hunt,
P0.c shared-machinery audit under H-GLC-1 (YaRN/RoPE application).
The ramp cells and their receipts are in `artifacts/moe_e1_3b/`
(control_manifest.json + controls.npz — REUSE these exact token
sequences; do not re-derive).

## Work

1. P0.a HF reference runner: score the five ramp cells (long + short
   arms, identical target sha256s as the port receipts) through
   HuggingFace transformers on the pinned snapshot. Constraints: the
   lead's GPU is 12 GB — design for it (MXFP4/quantized load if the
   installed transformers supports it, else CPU with accelerate
   offload; measure and chunk to respect the ≤590 s ceiling per
   invocation — if a single HF forward cannot fit 590 s, split by
   ramp cell and document per-cell walls; if CPU-only is the viable
   route, bounded CPU runs are exempt from the GPU ceiling but state
   walls honestly). Emit `GPU_GLC_P0A_COMMANDS.sh` (or CPU
   equivalent). Registered HG1 threshold is in the plan.
2. P0.b First-divergence runner: for one short (512) and one long
   (2560) sequence, capture per-layer hidden states from BOTH the port
   forward and the HF reference at aligned positions
   {100, 500, 1100, 2100, 2500 (long only)}; report per-layer relative
   Frobenius deviation and the FIRST layer where long-position
   deviation exceeds 10× the short-position deviation at the same
   layer. Emit its run script.
3. P0.c Audit (CPU, run it yourself): numerically compare the port's
   YaRN/RoPE construction (`core/gpt_oss20b_tc.py` builder at ~554)
   against the HF modeling code for this architecture: scaling factor,
   original-context constant, attention-scaling/concentration term,
   inv-freq table, and the actual cos/sin values at positions
   {0, 511, 1023, 2047, 2559} (max abs and rel diffs). Also diff sink
   semantics and the sliding/full interleave map against the HF config.
   Report every mismatch with numbers; "no numeric mismatch" is a
   valid finding.

## Gates (registered in the plan; restated)

- HG1: per-cell HF verdicts with numbers.
- HG2: first-divergence localization (layer/op/position band) or
  "port matches HF".
- HG3: mechanism sentence with file:line, or explicit unresolved
  statement listing exclusions.

## Constraints

Standing GPU discipline in emitted scripts (flock `/tmp/forge-gpu.lock`,
≤590 s per GPU invocation, 30 s+ gaps, GPU 0). HF runner must set
HF_HUB_OFFLINE=1 (snapshot only, no network). RED honesty; a control
that fails is a result. Evidence classes per the plan.

## Done

Final message MUST contain verbatim: HG1/HG2/HG3 status (numbers or
NOT_MEASURED + script paths); the P0.c audit verdict with every numeric
mismatch (or none) listed; files created/modified; anything you could
not do, stated plainly.
