# GRM Composed End-To-End Receipt Ledger

Execution record for the composed end-to-end receipt work order.
Immutable plan: `docs/GRM_E2E_RECEIPT_PLAN.md`. Narrative continues in
`docs/GPT_OSS_20B_APA_GRM_SYNTHESIS.md`.

## 2026-07-08 (opening)

Action: Work order opened per David's session goal ("Once completed
[P3], begin implementing the composed End-To-End receipt"). Plan
committed immutable alongside this entry.

Inherited state:
- P3 packed format shipped and closed (630a581): INT8/INT6 at rest,
  bit-identical recall, measured 1.79×/2.33× zlib disk — and the honest
  seam this receipt must price live: packed mounts 3.76× slower/layer
  (CPU dequant). E4's stable-VRAM leg uses the packed store; the mount
  cadence cost is one of the seams P2's read is FOR.
- Both route dialects on CUDA (GQA bridge 1.26-1.44× direct; MLA 2.22 ms
  at 1M), epoch staleness law, supersession machinery, 96k envelope.

Next action: P0 composition map (Sonnet, flat, read-only) — what the
live loop needs vs what exists; the key unknown is live per-turn
witnessed deposit (all existing gates capture offline).
