# ORDER NC17-P0 — Qwen3-1.7B stock-HF baseline (download, GT, OOM ladder, ppl, peaks)

YOUR WRITABLE TARGET is /mnt/ForgeRealm/GraftRepository — specifically
NEW files under `tests/nc17/` and `logs/nc17/` (create both), plus the
HF cache (~/.cache/huggingface) for the download and /tmp. Do not modify
any existing file. Everything else on the machine is read-only reference.

Read `docs/QWEN3_1P7B_NAMECHECKER_PLAN.md` first — it is immutable; its
"Measurement protocol" section governs every number you produce. Plan
wins on conflict; report conflicts, don't resolve them silently.

Environment (lead-verified tonight): torch 2.11.0+cu130 CUDA TRUE,
transformers 5.12.0, GPU free (RTX 4070 SUPER 12282 MiB), 516G disk.

## Task — Phase P0 of the plan

1. **Download** `Qwen/Qwen3-1.7B` (bf16 safetensors) via
   huggingface_hub to the standard cache. Record revision hash.
2. **GT mint FIRST** (`tests/nc17/p0_gt_mint.py`): on GPU, bf16,
   greedy — capture logits for a fixed prompt set (≥8 prompts,
   deterministic, committed in the script: short factual, multilingual
   incl. the 8 coverage languages, code-ish, and 2 name-verdict-shaped
   prompts) at final positions + first 64 decode steps' logits. Save
   `logs/nc17/p0_gt.npz` + the exact tokenized ppl corpus (see 4) token
   ids as `logs/nc17/p0_ppl_tokens.npz`. Then write the sentinel
   `logs/nc17/P0_GT_READY` (P1 gates depend on these files).
3. **OOM / context ladder** (`tests/nc17/p0_oom_ladder.py` driving ONE
   PROBE PER SUBPROCESS): stock transformers, bf16, default SDPA
   attention, no quantization, no offload.
   - Prefill ceiling: binary-search prompt length (synthetic token
     repeat is fine) until CUDA OOM; report last-solid / first-OOM.
   - Decode ceiling: prefill S then decode 64 steps with cache;
     binary-search S similarly.
   - Each probe: fresh subprocess, flock-wrapped, ≤10 min, 1s
     nvidia-smi poller logging to its own file; record poller peak AND
     torch.cuda.max_memory_allocated.
4. **Perplexity** (`tests/nc17/p0_ppl.py`): wikitext-2-raw test split
   (datasets lib; if unavailable offline, download once — record
   source+sha), sliding-window, stride 512, windows 2048 / 4096 / 8192
   / 16384 / 32768 (each window a separate flock-wrapped subprocess;
   skip rungs above the measured ceiling and SAY so). Identical
   tokenization saved in step 2. Report ppl per window + wall-clock +
   poller peak per run.
5. **Deliverable tables** printed AND written to
   `logs/nc17/p0_summary.json`: (a) ceilings, (b) ppl×window,
   (c) peak-fill×context per the plan's memory-accounting table.

## Rails

- GPU work: FOREGROUND commands only (blocking), each wrapped in
  `flock -w 3600 /tmp/forge-gpu.lock`, each ≤10 min (`timeout 590`).
  NEVER nohup/detach/background GPU runs — detached trees get reaped
  (three incidents on record). If a single rung can't fit 10 min,
  split the rung, don't raise the timeout.
- One model resident at a time. No other GPU work exists tonight
  except sibling seats queued on the same flock — waiting on the lock
  is normal, never a hang.
- NO git operations (lead commits). NO subagents. Do not edit any
  existing repo file.
- OOM is NOT a failure — it is the measurement. Report walls plainly.
  RED honesty for real failures (import errors, download failure,
  nonsense ppl): report verbatim, do not paper over.
- Evidence classes per plan: memory shape + ppl measurement only. No
  capability claims.

## Done — final message must contain, verbatim

- Model revision hash + download size.
- Ceiling table (prefill last-solid/first-OOM, decode last-solid/
  first-OOM) with poller peaks.
- ppl table per window with wall-clocks.
- Exact paths of every file created; confirmation that
  `P0_GT_READY` sentinel + GT npz exist.
- Any deviation from this order or the plan, stated as a deviation.
