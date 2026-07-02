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
   tests/test_deepseek_grm_hooks_static.py -q` — currently **151 passed**.
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

## Majors

**M5 — Explicit authoritative supersede targets silently dropped.** In
`_apply_extraction_candidate_direct`, `supersedes = conflicts if conflicts
else list(requested_supersedes)` — when an identity conflict coexists with
explicit user/system-asserted targets, the explicit targets are discarded
and stay active. *Fix direction:* for authoritative intents, UNION requested
targets with detected conflicts.

**M6 — NaN comparator UB in `RouterIndex::route` (C++).** `std::sort` with
`a.first > b.first` over float scores, no finite check (same in
`route_gqa_raw`). NaN breaks strict weak ordering → UB. *Fix direction:*
drop non-finite scores before sorting (and mirror the same guard in the
Python fallback ranking for parity).

**M7 — Native command parser keyword rebinding (C++).**
`command_suffix_after_keyword` advances `pos = low.find(needle, pos+1)`
into free text when the first occurrence fails its delimiter check —
`edit review 5,text new text goes here` binds the *second* "text" and
stores a truncated body. *Fix direction:* bind only the first
grammar-position occurrence; if its delimiter check fails, the parse
fails — never re-search inside the free-text payload.

**M8 — Section-cull spans drift on real BPE.** `_section_cull_spans` sums
`len(encode(chunk))` over stripped/rejoined chunks; separators and
merge-boundary effects make those counts diverge from the original token
stream, so child payload slices shift off section boundaries.
*Fix direction:* derive spans from the ORIGINAL text — encode
progressively larger prefixes ending at each section boundary and use those
prefix lengths as span edges.

**M9 — `load()` ignores the VRAM budget at peak.** Load unpacks device
tensors for every non-retired node and only pages afterward — a repository
larger than free VRAM cannot resume even with `vram_budget_mb` set.
*Fix direction:* respect the budget during load — defer device
materialization (host-only restore) and page in on mount, or page
incrementally as nodes are restored.

## Minors

**m10 — C ABI hygiene:** `grm_store_create_*_profile` leaks the handle when
the store constructor throws (use RAII/unique_ptr before release);
`grm_store_set_route_list` reads `route_offsets[key_count]` — one past the
declared count — undocumented; document the +1 contract in
`grm_runtime_c.h` AND validate, or change the signature.

**m11 — Reinforcement metadata overwrite:** `_reinforce_extraction_target`
copies all non-identity candidate metadata keys onto the existing node
BEFORE the trust-ranked plan runs — a low-trust duplicate can rewrite
`pinned`/notes. Gate the metadata merge on write-intent rank.

**m12 — (optimization, optional):** the corpus-wide `isascii()` scan in
`_native_active_text_matches` is O(total text) per command — cache a
per-repo flag maintained at node-add and checkpoint-load. Native
`dirty_plan` remains test-only — either consume it in `flush_now` ordering
or note it as ABI-surface-only in the build plan.

---

When the queue is empty: update the board (`/mnt/ForgeRealm/AI_Research_Board.md`,
GRM track — move items from "VERIFIED MAJORS STILL OPEN" to fixed, with
commit hashes) and note the new gate count.
