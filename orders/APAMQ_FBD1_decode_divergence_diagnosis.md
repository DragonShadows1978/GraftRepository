# APAMQ-FB-D1 — Diagnose Blend-vs-Fused Decode Divergence (+ Bits Ladder)

YOUR WRITABLE TARGET is this git worktree (GraftRepository branch
`apamq-fb`) — `core/`, `scripts/`, `tests/`, `docs/` edits AUTHORIZED.
Run first, report after; a registered order IS the permission.

HARD BOUNDARIES: canonical repos at /mnt/ForgeRealm/GraftRepository and
/mnt/ForgeRealm/Project-Tensor are READ-ONLY (the FA worktree at
/mnt/ForgeRealm/wt/apamq-fa is also read-only reference — its built
engine at wt/apamq-fa/tensor_cuda is what the int4 leg will import).
No git, no subagents, no network, models read-only. Your sandbox has NO
GPU: structure every GPU leg as one runnable command the lead executes;
CPU checks only in-sandbox. RED honesty.

## Measured facts (lead-run, 2026-08-13, do not re-derive)

Gate `tests/gemma4_apa_decode_fused.py` on the canonical engine:
64-step greedy tokens MATCH at S=8192, but max|Δlogit| = 12.852391
vs the registered 0.5 tolerance → FAIL. Census confirmed 512 blend +
512 fused calls (8 global layers × 64 steps, both paths driven on the
same state). Also: the INT4 leg reported "engine entry absent" even
with PYTHONPATH pointing at the FA worktree — your test hardcodes
`TC_ROOT = /mnt/ForgeRealm/Project-Tensor/tensor_cuda` and inserts it
at sys.path[0], defeating the override.

## Pre-registered hypothesis (test it, don't assume it)

H-DIV: both paths implement `thr = mean(|bulk|)+z·std(|bulk|)`,
`score = selected ? exact : bulk` — but the BLEND computes bulk/rank
as cuBLAS bf16 MATRICES and takes stats over bf16 values
(apa_blend_softmax_kernel2), while the FUSED kernel keeps bulk scores
fp32 in-register (apa_selective_kernel). Same statistics at different
precisions → near-threshold keys flip selection between paths; at
D=512/bulk_bits=4 the bulk-score noise makes the flip population and
per-flip score jumps large. Prediction: divergence shrinks strongly
as bulk bits rise and selection-set overlap → 1.

## Tasks

**D1 — attribute the divergence.** Instrument one decode-gate run to
capture, per global layer per step, from BOTH paths on identical
state: refined-key count, selection-set overlap (Jaccard + flip
count), and max|Δ attention output|. Correlate: does Δ track flips,
or does it persist with identical selection (→ bf16 score rounding of
non-selected keys)? Include the control: force-select-ALL keys (zthr
→ −∞ / refine_percentile → 1.0) on both paths — with everything
refined both paths score exact, so residual Δ isolates
non-selection noise; report it.

**D2 — bits ladder (this doubles as the F-C quality probe).** Rerun
the 64-step parity comparison at bulk_bits ∈ {4, 6, 8} (keep
refine_percentile 0.15): report max|Δlogit|, mean|Δlogit|, selection
overlap, refined fraction per rung. The H-DIV prediction is a strong
monotone improvement; a flat curve refutes it and points back at a
semantic bug — if flat, bisect the semantics (stats population, mask
range s_max vs count, scale application, kq slice alignment) until
you name the divergent term with a receipt.

**D3 — mechanics.** Make TC_ROOT env-overridable
(`TENSOR_CUDA_ROOT`, default canonical) in the gate AND
scripts/apamq_e3_attrib.py, so the FA-engine int4 leg actually loads
`wt/apamq-fa/tensor_cuda`. Add an `--int4` leg to the gate that
compares int4-fused decode vs blend the same way (it will run under
the FA engine only; capability-skip elsewhere).

**D4 — verdict material (no unilateral respec).** Based on D1/D2:
state which implementation matches the selective REFERENCE semantics
(the fused kernel mirrors the Rust reference; the blend is the
composed approximation), what tolerance G-B1 could honestly carry and
at what operating point, and list the options (stabilize selection;
raise bulk bits; gate against task-level behavior instead of raw
logits) WITHOUT choosing — the lead and David adjudicate.

Deliver every GPU leg as: `python3 scripts/apamq_fbd1_diag.py
--leg <name> --output artifacts/apamq_fbd1/<name>.json` (one command
per leg, self-contained, prints PASS/FAIL/measured summary lines).
Write findings to docs/APAMQ_FBD1_REPORT.md (facts + receipts, no
verdicts).

## Done

Final message MUST contain verbatim: the leg list with exact commands
for the lead to run, CPU-check outputs, file diffs summarized, the
D3 override behavior matrix, and any deviation. No GPU numbers — do
not fabricate; the lead measures.
