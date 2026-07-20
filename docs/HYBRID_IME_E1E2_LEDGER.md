# HYBRID IME E1/E2 — LEDGER

Receipts only. Plan: `HYBRID_IME_E1E2_PLAN.md` (immutable).

## 2026-07-20 — Track opened

- Session: Fable lead, David-directed ("The graphics card is effectively
  open right now. We can run that experiment").
- Papers pulled earlier this session (proxy artifacts, SHA-256 in cache):
  Bonsai 27B whitepaper (06451897…), Kimi Linear/KDA arXiv 2510.26692
  (e2e23a44…), Gated DeltaNet-2 arXiv 2605.22791 (b0c577f4…).
- Tine 1 (own ternary crush vs Bonsai) CLOSED by David — mechanism
  unpublished anywhere (whitepapers disclose format/kernels/benchmarks
  only; Caltech patents unpublished as of today; ThakiCloud repro
  confirms proprietary). Periodic re-check for published patent
  applications is a standing note, not a track.
- Preconditions verified before dispatch: GPU 228 MiB used (open);
  GraftRepository clean at 924b826; `core/qwen35_tc.py` +
  `tests/gemma4_attn_dist.py` + qwen35 gate suite present.
- Seat directive in force: Opus 4.8 implementation seats (window
  override 2026-07-20, routing memory), order-file prompts,
  worktree-isolated, GPU serialized via flock.

(Seat receipts, gate results, and follow-up decisions append below as
they happen.)
