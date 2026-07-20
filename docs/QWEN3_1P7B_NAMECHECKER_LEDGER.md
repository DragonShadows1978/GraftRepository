# QWEN3-1.7B NAME-CHECKER — LEDGER

Receipts only. Plan: `QWEN3_1P7B_NAMECHECKER_PLAN.md` (immutable).

## 2026-07-20 — Track opened
- Plan committed this commit. Model decided in [[project-frontier-npc-llm]]
  session: Qwen/Qwen3-1.7B (config verified from HF: 28L, 16Q/8KV,
  hd128, RoPE theta 1e6, tied embeds, vocab 151936, max_position 40960).
- Seats per David: kernels = Sol max (Project-Tensor fork
  `int6-weights`); all other code = Opus 4.8 at MAX effort.
- Phases P0 (HF baseline) → P1 (adapter+APA r0.15) → P2 (INT4) →
  P3 (INT6, conditional) → P4 (GRM proof). Receipts append below.
