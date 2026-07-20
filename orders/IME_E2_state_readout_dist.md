# ORDER IME-E2 — Per-channel state-readout distribution in the GDN layers

YOUR WRITABLE TARGET is this worktree (a checkout of GraftRepository) —
edits, builds, and GPU runs inside it are AUTHORIZED. Also writable:
`logs/` in this worktree and `/tmp`. Everything else on the machine is
READ-ONLY reference (Project-Tensor engine, model weight dirs, the
canonical GraftRepository checkout).

Read `docs/HYBRID_IME_E1E2_PLAN.md` first. It is immutable — where this
order and the plan conflict, the plan wins; report the conflict, do not
resolve it silently. The registered nulls and thresholds in the plan are
FIXED: do not adjust them after seeing data (spec-is-law).

## Task

Build and run `tests/qwen35_state_readout_dist.py`: measure whether the
per-channel readout contributions of the Gated DeltaNet state show
geometric-bulk + heavy-tail structure (IME), against the plan's two
registered nulls.

Port under study: `core/qwen35_tc.py` — GDN mixer `__call__` ~line 209.
CRITICAL PATH DETAIL: the fused `tc.gated_delta_step` kernel (~line 240)
internalizes the l2-normalized q — for capture you must force the
COMPOSED fallback path (the non-fused branch in the same method), where
post-norm q and post-update state S are materialized as tensors. The
composed path was parity-gated against the fused path at port time; if
you observe divergence, report it, don't chase it.

## Requirements

1. Workload: import the identical prompt set from
   `tests/ime_e1e2_prompts.py` (written by the E1 seat in its worktree —
   a copy is acceptable if it hasn't landed when you need it; flag which
   you used). Greedy decode, capture all 64 decode steps.
2. Per GDN layer × head × decode step: with post-norm query q and
   post-update state S (shapes per the port; verify and report them),
   compute c_i = ||q_i * S[i, :]||_2 for each key-channel i. Store the
   full c vectors.
3. Nulls, computed from the SAME captured tensors (no extra GPU runs
   needed): N1 = permute S's key-channel rows relative to q within each
   head (fixed seed); N2 = Gaussian S with matched per-head mean/var
   (fixed seed).
4. Statistics per plan: sorted-rank decay curves; top-10% channel mass
   share (real vs N1 vs N2); fraction of (layer, head) cells where real
   ≥ 2× N1; median consecutive-step Jaccard of the top-10% channel set.
   Evaluate the two registered thresholds and print PASS/FAIL per the
   plan's definitions — and print the underlying numbers either way.
5. Parity assertion (mandatory, in-script): final-step logits of an
   instrumented composed-path run == uninstrumented composed-path run on
   prompt 1. Print the result.
6. Outputs: raw arrays `logs/ime_e2_qwen35.npz` (c vectors, null draws,
   per-cell stats), summary `logs/ime_e2_summary.json`, printed summary
   table (per-layer: top-10% share real/N1/N2, threshold verdicts,
   Jaccard).

## Rails

- GPU runs: wrap every GPU-touching invocation in
  `flock -w 3600 /tmp/forge-gpu.lock <cmd>`. Each run ≤ 10 minutes.
  Composed-path decode is slower than fused — if 3 prompts × 64 steps
  exceeds the run budget, reduce decode steps to 32 uniformly (all
  prompts, stated in the receipt) rather than dropping a prompt.
- NO git operations of any kind — the lead commits.
- NO subagents.
- RED honesty: a failed gate, an OOM, a parity mismatch, or a
  below-threshold measurement is a RESULT — report it plainly. The
  registered criteria decide "confirmed", not your judgment; never
  soften a FAIL into qualitative language. Never idle waiting on a
  monitor; if blocked, stop and report.
- Evidence class: instrument measurement only. No model-quality claims,
  no speed claims, no claims about what a state-APA would achieve.

## Done

Final message must contain, verbatim (no summaries in place of them):
- The parity assertion line as printed.
- The printed summary table, including both threshold verdicts with
  their measured values.
- The tensor shapes you actually found for q and S, per layer.
- Exact paths of every file you created or modified.
- Wall-clock of each GPU run; which prompt-set source you used (import
  or copy).
- Any deviation from this order or the plan, stated as a deviation.
