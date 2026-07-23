# ORDER MERGE-3P — Reconcile codex/grm-three-pass into merge-train (Opus 4.8 seat)

YOUR WRITABLE TARGET is /home/vader/GraftRepository-merge-train (branch
`merge-train` = main @797b6d2: route-seams merged, resolver doc landed).
Production tree /mnt/ForgeRealm/GraftRepository and all other worktrees
are READ-ONLY. Do not spawn subagents. GPU: flock --wait conventions.

## Git authorization — EXACTLY this and nothing more

You may run: `git merge codex/grm-three-pass` (it WILL conflict), edit
conflicted files, `git add` resolved files, `git status`/`git diff`
freely. You may NOT run `git commit`, push, rebase, reset, or checkout
of other branches — the lead reviews the staged resolution and commits.
Leave the merge IN PROGRESS (resolved + staged) when done.

## Premise (pre-nailed)

Both sides are landed, gated programs. BOTH must survive intact:
- **route-seams side (in merge-train HEAD)**: epoch-cached CUDA
  bank/marshal/reverse-map, seam-3 per-turn rebuild inventory + caches,
  E3 gate harness (tests/test_grm_route_seams_gate.py — 21 green at
  HEAD together with the bank tests).
- **three-pass side (codex/grm-three-pass @ccb8a8b)**: turn_pipeline
  selector (single default byte-identical | three_pass), step-1
  prep/staging with fail-closed CUDA engagement + python parity arms,
  step-attributed I/O receipts, pass-3 extraction + memory ledger,
  --vram-budget-mb plumbing.
Conflicts: 5 hunks in core/graft_arena.py (one spans ~260 lines — both
sides' CUDA route machinery). Reconcile so route-seams' epoch-cached
bank is the shared substrate and three-pass's engagement/receipts layer
on top of it. Do NOT rewrite either side's logic beyond what the
reconciliation strictly requires; no new features, flags, or policies.
Read docs/GRM_THREE_PASS_LEDGER.md (in the three-pass branch) and the
route-seams commit message f512e4d before resolving.

## Build

The worktree lacks cpp/build. Build the existing CMake project (same
procedure the P0/P1 seat used in the three-pass worktree) or invoke
sessions with --native-lib pointing at your OWN build — never at
another worktree's artifact.

## Gates (registered before any resolution; ALL must pass)

- **G-M-SUITE**: `pytest -q tests/test_grm_*.py tests/test_gqa_ragged_cuda_bank.py`
  green under BOTH `GRM_TURN_PIPELINE=single` and `three_pass`
  (baseline: 460 at HEAD without three-pass tests; expect more with
  tests/test_grm_three_pass.py added; ZERO failures).
- **G-M-EQUIV**: smoke E2E at the DEFAULT frame (`--mode smoke
  --skip-gpu-idle-check`, env `GRM_GQA_CUDA_ROUTE=1
  GRM_GRAFT_STORAGE_BITS=8`), both pipelines: transcripts byte-identical
  to each other AND to sha256
  `68da84b8824eb79a14f65a692a602215bc53d84232cd041002410f51657bba4f`;
  probes 2/2 both.
- **G-M-IO**: three_pass smoke working-set receipts show
  route_backend=cuda on every routed turn, step-2 page-ins/uploads 0.
- **G-M-SEAMS**: tests/test_grm_route_seams_gate.py green post-merge.

Frames are PINNED — any deviation is a FAILURE. RED results are
results: report failing lines verbatim; never adjust a gate after
seeing results. The lead verifies everything against disk.

## Done

Final message MUST contain verbatim: the conflict-resolution summary
(per hunk: what each side wanted, what survived), all four gate lines
with receipts paths, both suite lines, and honest residuals. Leave the
merge staged, uncommitted.
