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

## 2026-07-20 — Runs complete (all four parity-clean)

- E1 qwen35 (seat: Opus): tests/qwen35_attn_dist.py, exit 0, 2:52.8,
  peak 5558 MiB. PARITY max|Δ|=0.000e+00 PASS. Artifacts:
  wt-e1/logs/ime_e1_qwen35.npz (511 KB), summaries copied to
  docs/ime_e1e2_results/. Seat deviations (accepted): synthetic prompt
  1875 tok (intact retrieval structure > round number); prompt-1 pass
  reused for parity+tables; prose wall-clock field nan (unfixed,
  cosmetic).
- Gemma control: first run CRASHED — instrument bit-rot
  (KVRing not subscriptable; broken since 2026-06-12 ring rework).
  Order IME-E1b (c95c70a); repair verified against live KVRing API by
  second Opus seat (ordered() accessor; core untouched). Re-run under
  lead harness: exit 0, peak 7540 MiB, PARITY 0.000e+00 PASS.
  Artifacts: wt-e1/logs/ime_e1_gemma.npz (3.1 MB) + summaries copied.
- E2 (seat-authored script, run under lead harness after two reaped
  launches): exit 0, capture 52.7s, peak 5780 MiB, PARITY bit-identical
  PASS. 2304 cells. THRESHOLD 1 FAIL (0.0% ≥2×N1, need 75%);
  THRESHOLD 2 FAIL (Jaccard 0.444, need 0.5); real < N1 in 24/24
  layers. Artifacts: wt-e2/logs/ime_e2_qwen35.npz (34.7 MB, NOT
  committed — size; path is the receipt) + summaries copied.
- Session notes: pre-restart session killed all processes on David's
  order (mid-run); relaunches from fresh session. Three process-reaping
  incidents (seat-detached nohup trees) → method law in synthesis §3.
- Instruments + prompts merged from worktrees to canonical tests/;
  repaired gemma4_attn_dist.py replaces the bit-rotted original.
  Worktrees -wt-e1/-wt-e2 retained (hold the .npz raws).
