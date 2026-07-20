# ORDER IME-E1 — GQA attention-distribution instrument on the Qwen3.5-9B hybrid

YOUR WRITABLE TARGET is this worktree (a checkout of GraftRepository) —
edits, builds, and GPU runs inside it are AUTHORIZED. Also writable:
`logs/` in this worktree and `/tmp`. Everything else on the machine is
READ-ONLY reference (Project-Tensor engine, model weight dirs, the
canonical GraftRepository checkout).

Read `docs/HYBRID_IME_E1E2_PLAN.md` first. It is immutable — where this
order and the plan conflict, the plan wins; report the conflict, do not
resolve it silently.

## Task

Build and run `tests/qwen35_attn_dist.py`: the attention-distribution
instrument for the 8 GQA layers of the Qwen3.5-9B tensor_cuda port, then
run the existing `tests/gemma4_attn_dist.py` unmodified as the dense-model
control.

Template: `tests/gemma4_attn_dist.py` (read it fully). Port under study:
`core/qwen35_tc.py` — attention layers via `cfg.attention_layer_indices()`,
attention forward at `Qwen35AttentionTC.__call__` (~line 312; cuBLAS blend
at ~375, SDPA fallback ~387). Harness patterns for loading/GT:
`tests/qwen35_gt.py`, `tests/qwen35_ready_gate.py`.

## Requirements

1. Workload: the three shared prompts per the plan (prose ~2048 tok, code
   ~2048 tok, synthetic 40-fact retrieval ~2048 tok). Write them (or
   their generator with fixed seed) into `tests/ime_e1e2_prompts.py` so
   E2 imports the identical set. Greedy decode, 64 steps.
2. Capture, per GQA layer × Q-head × last-64 query positions ×
   prompt: post-softmax entropy; fraction of attention mass in top 5% /
   10% / 15% of keys by score. Recompute scores from cached K + live q at
   the capture point (Gemma instrument pattern) — do NOT modify the
   serving path's outputs. Opt-in env-flagged hooks only if hooks are
   unavoidable.
3. Parity assertion (mandatory, in-script): final-step logits of an
   instrumented run == uninstrumented run on prompt 1. Print the result.
4. Outputs: raw arrays `logs/ime_e1_qwen35.npz`, summary
   `logs/ime_e1_summary.json`, and a printed per-layer table (mean
   entropy, mean top-5/10/15 mass) with the Gemma control alongside.
5. Gemma control: run `tests/gemma4_attn_dist.py` AS-IS (it may need its
   GT npz at `/mnt/ForgeRealm/gemma4_gt_qat/gt_long.npz`; if the file or
   VRAM budget makes the run impossible, SAY SO and deliver the
   qwen35-only result — that is a valid outcome, do not fake or
   approximate the control).

## Rails

- GPU runs: wrap every GPU-touching invocation in
  `flock -w 3600 /tmp/forge-gpu.lock <cmd>`. Each run ≤ 10 minutes.
  One model resident at a time (the card is 12 GB; qwen35 INT4 ~4.6 GB,
  Gemma INT4 ~6.8 GB — never both).
- NO git operations of any kind — the lead commits.
- NO subagents.
- RED honesty: a failed gate, an OOM, a parity mismatch, or a control you
  could not run is a RESULT — report it plainly with the error text.
  Never idle waiting on a monitor; if blocked, stop and report.
- Evidence class: instrument measurement only. No model-quality claims,
  no speed claims.

## Done

Final message must contain, verbatim (no summaries in place of them):
- The parity assertion line as printed.
- The printed per-layer summary table(s) for qwen35 (and Gemma if run).
- Exact paths of every file you created or modified.
- Wall-clock of each GPU run and peak VRAM if measured.
- Any deviation from this order or the plan, stated as a deviation.
