# GRM Verified Bug Queue — Work Order

Adversarially verified findings from the July 2026 review fleets (each
survived multi-lens verification against the code). The three CRITICALS from
that review are already fixed (`cec8c8a`, `964439b`). This queue is the
remaining majors/minors, ordered by severity.

**RULES (non-negotiable):**
1. **Commit after EVERY individual fix** — do not batch, do not leave work
   in the working tree at a stopping point.
2. **Run the non-GPU gate after every fix**: `python3 -m pytest
   tests/test_grm_runtime_lifecycle.py tests/test_grm_native_runtime.py
   tests/test_deepseek_grm_hooks_static.py -q` — currently **166 passed**.
   A fix that breaks the gate is not a fix.
3. **Every fix ships a regression test** that fails on the pre-fix code.
4. **House law: Unicode and time never cross the ABI.** Native code prunes
   by exactly-comparable state; Python decides case folding and clocks.
   Do not add temporal parsing or Unicode case logic to C++.
5. Line numbers below are approximate (the tree has moved) — re-locate by
   function/symbol, not line.
6. Do NOT edit `docs/GRM_PAPER_DRAFT.md` or the other release documents.

---

## Fixed from this queue

**M1 — `flush_async` wal_lsn race.** Fixed 2026-07-02. `_append_wal`
now serializes LSN assignment and fsync under a WAL lock. `flush_now`
records a checkpoint boundary before the long payload/index/native/manifest
write phase and publishes that boundary in `manifest.json`; WAL records
that land during the flush window now replay after the manifest instead of
being skipped. Dirty cleanup is generation-aware so foreground mutations
that happen during the async flush remain queued for the next checkpoint.
Regression:
`test_async_flush_manifest_lsn_does_not_skip_concurrent_wal`.

**M2 — WAL-replay gap placeholders brick future flushes.** Fixed
2026-07-02. Manifest-plus-WAL replay now creates missing node-id gaps as
`payload_pending=True` recovered placeholders, so the next checkpoint treats
them as text-authoritative gaps instead of trying to synthesize nonexistent
payload tensors. Regression:
`test_wal_gap_placeholders_can_checkpoint_after_replay`.

**M3 — Durability-mode upgrade doesn't protect pre-existing state.**
Fixed 2026-07-02. WAL-off to WAL-on transitions now append `NODE_UPSERT`
snapshots for all dirty in-RAM nodes immediately after the mode `CONFIG`
record, while leaving those nodes dirty for the next full checkpoint. WAL
replay honors snapshot state so tags/source/no-fold metadata survive the
no-manifest crash path. Regression:
`test_wal_upgrade_snapshots_existing_dirty_nodes`.

**M4 — Crash between payload writes and manifest orphans `.npz` files.**
Fixed 2026-07-02. WAL recovery now scans WAL-known node ids for orphaned
`nodes/NNNN.npz` payloads and re-adopts them before native sync, both with
and without a manifest. Missing payload files now downgrade the graft to
text-authoritative `payload_pending` recovery state instead of surfacing a
raw `FileNotFoundError` through descent. Regressions:
`test_wal_recovery_adopts_orphaned_payload_before_manifest` and
`test_missing_payload_load_degrades_to_pending`.

**M5 — Explicit authoritative supersede targets silently dropped.**
Fixed 2026-07-02. Extraction supersession now builds an ordered de-duplicated
union of detected conflicts and explicit requested supersede targets for
authoritative candidates, in both native-planned and Python-planned policy
paths. Regression:
`test_authoritative_extraction_unions_conflicts_and_requested_supersedes`.

**M6 — NaN comparator UB in `RouterIndex::route` (C++).**
Fixed 2026-07-02. Native cosine routing and GQA raw routing now skip
non-finite per-key scores and drop nodes with no finite score before sorting;
Python fallback routing applies the same finite-score filter before
normalization and ranking. Regressions:
`test_native_route_drops_non_finite_scores`,
`test_native_gqa_route_drops_non_finite_scores`, and
`test_arena_python_route_drops_non_finite_scores`.

**M7 — Native command parser keyword rebinding (C++).**
Fixed 2026-07-02. `command_suffix_after_keyword` now binds only a
whitespace-preceded command keyword token and returns failure on malformed
token boundaries instead of scanning forward into free-text payloads. The
valid body `edit review 7 text new text goes here` still preserves the inner
`text`, while `edit review 5,text new text goes here` now fails. Regression:
`test_native_memory_command_parser`.

**M8 — Section-cull spans drift on real BPE.**
Fixed 2026-07-02. Section cull planning now derives child token edges from
progressively encoded prefixes of the original text at section boundaries,
rather than summing stripped/rejoined chunk encodings. Regression:
`test_plan_cull_sections_uses_original_prefix_token_edges`.

**M9 — `load()` ignores the VRAM budget at peak.**
Fixed 2026-07-02. Manifest load now restores host payloads first and only
unpacks device tensors while the running `vram_budget_mb` budget can absorb
the node, leaving the rest host-only for later mount/page-in. Regression:
`test_load_respects_vram_budget_before_device_unpack`.

## Majors

**M11 — fold-after-recovery bricks the librarian (2026-07-16,
lead-reproduced with instrumentation).** Librarian fold-source
selection never excludes WAL-recovery placeholder nodes (kind=turn,
ntok=0, h=None, recovered/payload_pending flags — the state a repo
boots into from wal/ with payloads missing). Placeholders count as
foldable turns → threshold crosses → `_fold_once` →
`arena.consolidate` (graft_arena.py:1252) indexes `h=None` →
TypeError; `_ensure_h` cannot heal an unbacked placeholder and falls
through silently. Production impact: any crash-recovered session
bricks its librarian on the FIRST idle(). Sibling of M2 (flush path
guarded 2026-07-02; fold path never was). Fix in flight 2026-07-16
(fold eligibility = resolvable payload only, both deferred and
backpressure paths, counter-jam guard per 816a0a0). Secondary,
unrouted: `_ensure_h` silent fall-through deserves a named error
(graft_arena.py ownership).

**M10 — lifecycle suite RED: FakeArena drift vs epoch API (2026-07-16,
lead-verified).** `tests/test_grm_runtime_lifecycle.py` fails 91/101 on
clean main: `GraftRepository` now calls `arena._bump_cuda_gqa_epoch()`
(landed `e8906dc`, merged with the CUDA bridge `bcd5f51` 2026-07-08) and
the suite's `FakeArena` double never grew the method. Rule 2's "166
passed" gate has been silently red since that merge — every fix
committed against rule 2 since 2026-07-08 ran a broken gate. Fix: add
the epoch method (no-op) to the test double; audit other FakeArena gaps
vs the current ArenaCache surface while there. Verified: quiet-machine
run + stash A/B (same failures with/without 2026-07-16 working-tree
changes); first failure signature `AttributeError:
'FakeArena' object has no attribute '_bump_cuda_gqa_epoch'` at
`core/graft_repository.py:3779`.

All verified major queue items are fixed.

## Minors

**m10 — C ABI hygiene.** Fixed 2026-07-02. Profile constructors now hold
the C handle in `std::unique_ptr` until `HostGraftStore` construction
succeeds, and `grm_store_set_route_list` now documents and validates the
`key_count + 1` offsets contract, including terminal offset equality with
`value_count`. Regression:
`test_native_route_list_rejects_terminal_offset_mismatch`.

**m11 — Reinforcement metadata overwrite.** Fixed 2026-07-02.
`_reinforce_extraction_target` now reads the existing write-intent rank
before merging non-identity candidate metadata, so lower-trust duplicates
can reinforce confidence/source history without rewriting protected fields
such as `pinned` or `notes`. Regression:
`test_low_trust_duplicate_does_not_overwrite_reinforcement_metadata`.

**m12 — Native text-scan and dirty-plan cleanup.** Fixed 2026-07-02.
`_native_active_text_matches` now uses a repository-level ASCII corpus flag
maintained by mutation tracking and checkpoint/WAL load instead of scanning
all graft texts per command. `flush_now()` now consumes the native dirty
flush order for payload writes while keeping manifest node order stable.
Regressions:
`test_native_text_scan_uses_ascii_cache_not_full_corpus_iteration` and
`test_flush_now_uses_native_dirty_flush_order_without_reordering_manifest`.

---

When the queue is empty: update the board (`/mnt/ForgeRealm/AI_Research_Board.md`,
GRM track — move items from "VERIFIED MAJORS STILL OPEN" to fixed, with
commit hashes) and note the new gate count.
