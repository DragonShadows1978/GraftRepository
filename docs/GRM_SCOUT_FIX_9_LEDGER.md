# GRM-SCOUT-FIX-9 ledger

Order is immutable: `orders/GRM_SCOUT_FIX_9_DEGENERATE_DIGEST.md`, SHA256
`7fd45c958490a02ac7cbcf94ae81c2745d53a87a7a71bc404f7c6f97d9a7e412`. Registered replay + gates:
`artifacts/grm_scout_fix9/registration.json`.

Seat: Opus 5 (model id `claude-opus-5[1m]`), reasoning effort max.
Writable target: `/mnt/ForgeRealm/wt/grm-fix9` (branch `grm-fix9`).
No git. No subagents. No GPU run. No process signalled or killed.
Every command foreground and under 10 minutes.

## Prior art

- **Cycle-safe graph traversal with a visited set** — textbook DFS
  (Tarjan 1972; any standard treatment). TAKEN: the visited-set idiom.
  OURS: recognizing that the width guard's parent<->child `sources` edge
  makes the descent-key walk cyclic.
- **Write-through invalidation at a single mutation choke point** —
  standard cache practice; this file already applies it to
  `_bump_cuda_route_epoch`. TAKEN: the choke-point discipline. OURS:
  noticing the native route store is a SECOND reader of `child_cents`
  that the rebuild never served.
- **Minimum-information chunk filters in retrieval pipelines** —
  LlamaIndex / LangChain drop sub-threshold splits (2023-2025).
  TAKEN: the "reject rather than deposit a lossy derivative" stance,
  which is also FIX-3/FIX-5's own fold-QC stance. OURS: keying the
  rejection on the project's existing `_fact_set` fidelity vocabulary
  plus the binary-split fragment shape, rather than a raw token floor.
- **The harness** — `scripts/grm_c7_diagnose.py` CPU doubles and the
  LSR-P2B E2E fixtures (GRM contributors, 2026), reused wholesale.
  OURS: loading a LIVED GPU session's repository into that pattern so a
  GPU-recorded crash reproduces host-only.
- UNVERIFIED against the wider literature (no network in this sandbox)
  — lead to check. Search terms: "hierarchical retrieval index stale key
  invalidation", "parent-child chunk cycle descent keys", "degenerate
  chunk rejection", "retrieval chunk quality gate".

## Audit — the order's hypotheses vs what the receipt actually shows

The order named two defects and offered a mechanism for each. The
mechanisms are REPORTED WRONG by the receipt; the defects are real. All
three corrections below are evidenced, not asserted.

1. **"A digest of four tokens ... was ACCEPTED by the fold QC (FIX-3
   punctuation-collapse QC + FIX-5 coverage)."** It was not. Node 20's
   manifest metadata is `culled_from: 18, cull_index: 0, cull_total: 2,
   width_guard: true, width_guard_child: true`. It was produced by the
   WIDTH GUARD (`_guard_deposit_width`), which never consults the fold
   QC. Node 18 was an era fold of ntok=97 against a width budget of 96 —
   ONE token over. No fold QC check let it through, because none ran.

2. **"`cent=None`, ... The native router omits a node with no route
   key."** Node 20 HAS a route key: `index.npz` holds `rkey_0020` of
   shape (8, 4, 64), and it loads as a live `cent`. The native router
   returned ALL 11 candidates, omitting nothing (verified:
   `native OMITTED graft idx: []`).

3. **"the Python reconstruction scores it from `_lex_bonus` alone."**
   Its `_lex_bonus` is 0.0. The recap query's content tokens are
   {recap, five, biggest, decisions, made}; node 20's rare set is
   {archive, note}; the intersection is empty. Its score came entirely
   from the centroid channel.

**What actually diverged**, reproduced host-only on the saved
`gpu_session/` repository (`scripts/grm_scout_fix9_repro.py`):

- `_rebuild_child_keys` walks `sources` recursively with a depth cap but
  NO visited set. The width guard sets `parent["sources"] = children`
  while each child keeps `child["sources"] = [parent]`, so 18 <-> {20, 21}
  is a CYCLE. Measured on the receipt's own repository: node 18 collected
  13 descent keys (its own key seven times, node 20's three times); nodes
  20 and 21 collected 10 each. Every member of the split family therefore
  held the family's maximal key, and all three scored IDENTICALLY
  (11.435731). A 4-token header routed as the 97-token era it was cut from.
- `_rebuild_child_keys` mutates `child_cents` for every node, and
  `_native_set_route` publishes exactly `[cent] + child_cents`. But the
  callers re-synced only SOME nodes (`_guard_deposit_width` syncs the
  parent; `load()` syncs only WAL-replayed nodes). The native store kept
  each child's deposit-time SINGLE key while Python held the rebuilt
  cyclic set. **The two routers ranked from different key multisets** —
  precisely what the integrity guard exists to catch. PROVEN by
  experiment: forcing a full `_native_set_route` over every node made the
  two orders identical and the guard pass, with no other change.

## Treatment

Two source files, minimal diff.

- `core/graft_repository.py`
  - `_width_guard_degenerate_spans` (new) + its call in
    `_guard_deposit_width`: defect 1. A split plan is rejected when it
    cuts the parent in TWO and one of the two is a fact-less FRAGMENT
    (<= `WIDTH_GUARD_FRAGMENT_FRACTION` = 0.25 of the budget), or when
    NO child carries any of the parent's facts. Also rejects the
    `_decode_token_span` whole-parent placeholder. On rejection the
    parent is left exactly as it was and
    `last_width_guard_rejection` carries the receipt.
    SCOPE MATTERS: an ordinary fact-less chunk of a many-way split is a
    legitimate reader (LSR-P2C pins "Third section is filler prose ..."
    and a two-token "dock schedule." tail), so a fact-less chunk alone is
    NOT the defect. A first draft that rejected any fact-less child broke
    two LSR-P2C tests; that was a REAL regression, caught by comparing
    against the `grm-merge` baseline, and the rule was rescoped to the
    receipt's binary-split shape.
  - `_rebuild_child_keys`: defect 2. Visited set closes the cycle (the
    depth cap is KEPT — it is the generational reach the descent contract
    promises), and nodes whose keys actually moved are republished to the
    native router. `_republish_route_keys` + `_same_cent_list` (new)
    make that a single choke point; `load()` holds the republish until
    the native checkpoint has adopted its nodes, then drains the queue.
- `core/graft_arena.py`
  - `_route_cand_base`: the lead's ruling. A node with `cent is None` is
    no longer a candidate, matching the native router, which drops a
    keyless entry outright (`route_gqa_raw` never sets `have[]` for it).
    This closes a SECOND, independent path to the same guard. NOTE: the
    r2 receipt did NOT exercise it — node 20 had a key.

## Gates

- RED-before (pre-fix core taken from the read-only `wt/grm-merge` fork
  point): `AdmissionPolicyError ... backend=native
  production=[13, 8, 20, 21, 15, 16] reference=[13, 8, 15, 16, 18, 20]`;
  `orders_agree: false`; descent-key counts 18/20/21 = 12/9/9.
  `artifacts/grm_scout_fix9/red.json`.
- GREEN-after: no exception; ranking `[13, 8, 15, 16, 18, 20]`, plan
  `[13, 8, 15]`; `orders_agree: true` (native RAW order == Python RAW
  order element for element); descent-key counts 18/20/21 = 2/2/2.
  `artifacts/grm_scout_fix9/green.json`.
  Absolute ids differ from the GPU log because the probe key is an RNG
  draw, not a GPT-OSS harvest; what reproduces is the DIVERGENCE, a
  property of the stored route keys.
- C2 replay gate (FIX-6/FIX-8): 132 recorded A-DEC plans, 132
  byte-identical, and the full row set equal to the frozen
  `artifacts/grm_scout_fix8/green.json` `c2` block. No frozen golden moved.
- `GRM_ADMISSION_RULE` unset: the reconstruction branch does not run and
  `_route_cand_base` only drops keyless nodes (none exist in any existing
  fixture), so behaviour is unchanged — evidenced by the zero-new-failure
  diffs below rather than asserted.

## Regression discipline

Every failure was diffed against the SAME suite run in the read-only
`wt/grm-merge` fork point before being attributed.

- Required battery: 22 failed / 280 passed. The `grm-merge` baseline is
  22 failed / 285 passed on the same selection minus the new FIX-9 file.
  NEW failures introduced by FIX-9: **none**. The 22 are pre-existing
  sha-drift and missing-artifact gates (`FIX8_AMENDMENT_SCOPE_MISMATCH`,
  and `artifacts/grm_c7/r2/fix4_attempt_1/.../index.npz` absent from
  `grm-merge` AND from the canonical repository).
- Adjacent CPU suites (`test_grm_lsr_p2c_split_descent`,
  `test_grm_rt1_split_child_routing`, `test_grm_c2_amendment`,
  `test_grm_importance_salience`): 18 pre-existing on both trees, NEW:
  **none**.
- Worktree artifact gap repaired, NOT a code change: 104 gitignored
  `.npz` payload files under `artifacts/grm_c7/r2/` were absent from this
  worktree and present in `grm-merge`; they were copied in (no symlinks).
  Without them 2 FIX-5 runner tests failed for `FileNotFoundError`, which
  would have been misread as a regression.

## RED / blocked

- `tests/test_graft_repository.py` is **BLOCKED, not passed**. It
  allocates GPU memory at import and every attempt failed with
  `cudaMalloc failed: out of memory`: a FOREIGN process (pid 3315428)
  held 10.3 GB of the 12 GB card for the session. Per house rule the seat
  did not signal or kill it. LEAD TO RUN.
- The GPU replay itself is REGISTERED, NOT RUN (no GPU in this sandbox):
  `artifacts/grm_scout_fix9/lead_commands.txt`. Its argv was VERIFIED
  against this worktree's `scripts/grm_chat.py` — `--repo` takes the
  SESSION directory, `--resume` is a bare flag, `/recap` is driven by
  `--transcript`, and `--profile eb1_c2 --print-flags` was executed and
  reproduces the r2 log's env block including `admission rule:
  margin_first`.

## Deviations from the order

- The order's stated mechanisms for both defects are contradicted by the
  receipt (see Audit). The DEFECTS were fixed; the stated CAUSES were
  replaced with the measured ones. The lead's ruling on route eligibility
  was implemented as written, as an additional independent path.
- The order asks which FOLD QC check admitted the digest. The honest
  answer is NONE: no fold QC ran. The admitting site is the width guard.

Model: Opus 5 (`claude-opus-5[1m]`), reasoning effort max.
