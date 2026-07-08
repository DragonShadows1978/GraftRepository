# GRM CUDA Bridge Overhead Ledger

This ledger is the execution record for the CUDA bridge overhead work
order. The immutable plan is `docs/GRM_CUDA_BRIDGE_OVERHEAD_PLAN.md`; the
wing's narrative continues in `docs/GRM_GEMV_ROUTER_SYNTHESIS.md`.

## 2026-07-07 23:30 EDT

Action: Work-order opening — plan drafted, shaped, and committed.

Repo state:
- Repository: `/mnt/ForgeRealm/GraftRepository`
- Branch: `grm-cuda-bridge-overhead` (cut from `codex/intn-model-ppl-sweep`
  at bbe74a0 + working-tree state; pre-existing dirty GPT-OSS files outside
  this slice untouched).
- Baseline receipts inherited from GRM_GEMV_ROUTER_LEDGER 2026-07-07: bridge
  3.157/11.218/38.961 ms vs direct route 0.129/0.255/0.770 ms at 32/128/512
  nodes; baseline instrument = scripts/grm_gqa_cuda_bridge_smoke.py with the
  exact ledger commands.

Findings (plan shaping, David 2026-07-07):
- Scope: GQA bridge only — that is where the optimization is needed; MLA
  and CPU route paths untouched.
- Architecture decision: C++ — grow the existing CUDA sidecar into the
  native bridge (hot/cold split, device-pointer route entry for
  forward-pass callers). Rust adapter rejected (adds a toolchain and
  boundary crossings; the concurrency story is already gated in C++). CPU
  C ABI stays CUDA-free per the wing's standing law.
- Doc structure per house precedent: new plan + new ledger per work order,
  one continuous wing synthesis.

Next action:
- P0: profile attribution of the bridge gap (Sonnet agent, flat,
  no-delegation rule).

## 2026-07-08 00:20 EDT

Action: P0 complete — attribution receipt (committed BEFORE any fix, per
plan and per Codex's registered point). Lead spot-checked the load-bearing
claim at source.

Findings (evidence class: profile receipt; cProfile over 300 reused route
calls, idle GPU, contended host — see caveat):
- E1 MET: ≥99% of the reused-call cost attributed to named functions;
  81.6% (512n) / 80.8% (128n) of the gap is ONE mechanism —
  `graft_arena.py:1513` `np.ascontiguousarray(np.stack(rows))` inside
  `_cuda_route_bank_inputs()`, which re-materializes the full dense route
  bank (268 MB at 512 nodes) on EVERY route call, before the signature
  compare that then discards it in steady state. Lead-verified at source:
  `_cuda_route_order` → `_cuda_route_bank_inputs()` unconditional.
- PRIME-SUSPECT CORRECTION (this is why P0 ran first): the plan's suspect
  (a) — per-call signature recomputation — is CHEAP (~0.06%, id-tuple
  compare; the blake2b hash is attach-only). An O(1) epoch check alone
  would NOT have closed the gap: the bank rebuild happens before and
  independent of the staleness check. P1 must cache the STACKED bank and
  rebuild only on staleness, with the epoch making the residual per-node
  walk (~1.7 ms, ~1%) O(1) as well — otherwise E2 (≤2× direct at 512n =
  ≤1.54 ms) is unreachable.
- Residuals: per-node Python loop ~1%; query marshal + ctypes ~0.6-1.2%;
  node-id mapping <0.1%.
- Caveat (registered): absolute ms in this profile run (~160 ms @512n) are
  ~4× the ledger receipts (38.96 ms) due to host memory-bandwidth
  contention (multiple live sessions, swap in use — raw 268 MB numpy copy
  benched at 0.8-1.6 GB/s on this box right now). Mechanism, percentages,
  and linear-in-N scaling are consistent across node counts and
  corroborated by an isolated same-shape np.stack microbench (163 vs
  158 ms). P3's re-receipt runs on the quiet machine with the exact ledger
  commands.
- Receipts: artifacts/grm_gqa_cuda_bridge/profile_{512n,128n}*.{pstats,txt,json},
  smoke_512n_recheck.json; profiling wrapper
  scripts/grm_gqa_cuda_bridge_profile.py (new file, product code untouched).

Next action:
- Commit this receipt. Then P1 (corrected design): cache (route_bank,
  node_ids, signature) — stack only on staleness; mutation-epoch integer
  bumped at every site that today clears/invalidates the CUDA bank plus
  graft add/retire/route-key replace in GQAArenaCache; hot path compares
  one int. Fail-closed equivalence regression REQUIRED: every mutation
  that changes the signature must bump the epoch (test asserts
  epoch-valid ⇒ signature-equal across the mutation battery).

## 2026-07-08 01:30 EDT

Action: P1 implemented (Sonnet, flat), lead-verified structure, committed.

Findings (evidence class: interleaved same-session A/B, contended host —
load 4.27; ratios load-bearing, absolutes deferred to P3 quiet gate):
- Bridge reused wall: 7.76→0.44 ms (32n, 17.7×), 41.80→0.78 ms (128n,
  53.9×), 155.78→2.12 ms (512n, 73.6×). Parity true at all node counts;
  direct CUDA route wall unchanged (isolation confirmed).
- Mechanism: signature walk split from bank build; np.stack runs only on
  signature change (cache on GQAArenaCache). The P0-receipted 82% cost is
  gone from the steady-state path.
- CONSERVATIVE AUTHORITY DECISION (agent-flagged, lead-endorsed): the
  per-call O(N) signature walk REMAINS the correctness authority; the
  epoch is bookkeeping + equivalence proof. Cause: core/graft_repository.py
  mutates arena.grafts fields directly at ~15+ sites (retire via forget/
  correct_memory, cent replace, WAL replay, migrate, cull) which the
  epoch does not observe. Fail-closed holds because the walk sees those
  writes regardless (verified: raw external retired=True write with no
  epoch bump was caught). CONSEQUENCE REGISTERED: the epoch may not
  become the sole staleness gate until every graft_repository.py mutation
  site is instrumented (or all mutations flow through a choke point —
  David's ledger/index concept). This is P2 scope.
- Tests: lifecycle 6/6; full native runtime 92 passed (90 baseline + 2
  new: epoch/signature fail-closed equivalence battery + stale-bank
  behavior battery), zero regressions (stash-verified baseline).
- Residual per-call cost at 512n (~2.1 ms contended) ≈ the signature walk
  + marshal — the P2 targets, as planned.

Next action:
- Commit P1. P2: native hot/cold split — sidecar route entry with
  device-pointer queries; epoch becomes sole hot-path gate WITH the
  graft_repository.py mutation sites instrumented (scope expansion
  registered here); signature walk moves to attach/cold path only.

## 2026-07-08 02:40 EDT

Action: P2 implemented (Sonnet, flat), lead-verified, committed e8906dc.
P3 receipts taken; formal quiet-window confirmation registered as the
one open checkbox.

Findings:
- W1: epoch is now the SOLE hot-path staleness gate. All
  graft_repository.py signature-relevant mutation sites instrumented
  (choke-point `_mark_mutations` + `_rebuild_child_keys` +
  `_native_sync_node` first-sync + `_fold_once` + expire/supersede/
  WAL-rehydrate/load-identity-mapping paths — full list in the P2 agent
  report and code comments). Ambiguous sites examined and ruled out with
  reasons (metadata-only writes). `GRM_GQA_BRIDGE_PARANOID=1` re-enables
  the per-call walk with an agreement assert; a test proves it catches
  injected under-invalidation. CONTRACT NOW LOAD-BEARING: any direct
  arena.grafts mutation (including test code) must bump the epoch — a
  pre-existing P1 test doing a raw append was caught by its own failure
  and fixed; paranoid mode is the dev safety net.
- W2: sidecar route entry with DEVICE-pointer queries
  (`grm_gqa_cuda_arena_route_device` + Python surface through
  CudaGQARouteBank.route_topk_device / NativeGraftStore.route_gqa_cuda_device).
  Host-vs-device entry: byte-identical top-k on real capture banks at
  32/512 nodes. CPU C ABI untouched (CUDA-free law holds).
- Profile receipt: steady-state hot path has NO O(N) component —
  `_cuda_route_bank_inputs` flat at ~0.005 ms/call at BOTH 32n and 512n;
  signature walk absent from steady-state profiles.
- Timing:
  - P2 agent clean-window interleaved (loadavg 2.3-2.8, idle GPU):
    0.185/0.339/0.966 ms at 32/128/512n vs direct 0.128/0.261/0.770 =
    1.44×/1.30×/1.26×. E2 (≤2× direct) MET at all node counts; E3 MET
    (ratio improves with N — overhead flat, no longer scales).
  - Lead P3 runs under load (loadavg ~3.2, other sessions active,
    exact ledger commands): parity true all counts; 512n 1.52× (met),
    128n 2.03×, 32n 2.7× — small-node fixed Python cost inflates under
    CPU contention exactly as P0's caveat predicted. Receipts:
    artifacts/grm_gqa_cuda_bridge/p3*_*.json.
  - Vs the original baseline: 3.157→0.185, 11.218→0.339, 38.961→0.966 ms
    = 17×/33×/40× on the clean-window instrument.
- Tests: 95 passed (92 + 3 new: graft_repository mutation battery,
  migrate battery, paranoid under-invalidation catch), two independent
  full runs, zero regressions. Lifecycle selector + P1 regressions green.

OPEN CHECKBOX: one quiet-machine run of the three exact ledger smoke
commands to stamp E2 formally at 32/128n free of contention. No code
change rides on it.

Next action:
- Wing synthesis + board update. Candidate successors (David-raised):
  MLA production-path Python overhead profile (P0 method, 1M harness);
  synthetic GQA route centroids (two-tier, exactness-gated); per-graft
  incremental index / ledger-derived staleness (choke-point already
  half-built by W1).

## 2026-07-08 (quiet-window stamp) — WORK ORDER CLOSED

Action: E2 stamp run, GPU idle, loadavg ~2.3-2.9 (ambient interactive
load; the box is never fully quiet in practice). 32n re-run with
queries=10 for real statistics (9 reused samples, spread 0.352-0.448 ms —
tight, not contention).

Verdicts (final):
- E3 MET decisively: bridge overhead flat at ~0.20-0.40 ms across
  32/128/512 nodes (16× node range; was 3.03→38.19 ms linear-in-N).
  The O(N) defect class is eliminated. Receipts: stamp_*.json.
- E2 MET at 512n (1.173 ms vs 0.771 direct = 1.52×); NOT MET at 32n
  (0.352 ms vs 0.150 = 2.35×, solid statistics), marginal at 128n
  (2.23×). FINDING per the registered misses-are-findings clause: the
  residual is the fixed Python invocation floor (ctypes + wrapper +
  result mapping under ambient load), not algorithmic overhead. It
  cannot go below ~1× of a 0.15 ms direct route while the caller is
  Python. The remedy is already built but unexercised: the
  device-pointer route entry invoked from native/forward-pass code
  (plan P2's stated future). Registered as the natural first receipt of
  any forward-pass integration work order.
- Practical outcome vs opening baseline: 3.157→0.352, 11.218→0.563,
  38.961→1.173 ms = 9-33× at ambient load (17-40× in the P2 agent's
  clean window). Parity green at every node count in every run.

WORK ORDER CLOSED. Successors on the board.
