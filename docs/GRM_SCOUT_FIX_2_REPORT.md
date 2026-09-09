# GRM-SCOUT-FIX-2 report

**Fix and required CPU gates GREEN; C2 GPU epoch NOT_RUN.** The required suite has 95 passed; the additional split/routing/lifecycle suite has 178 passed (273 across the two final runs), including all C2 tests, all SCOUT-FIX-1 fixtures and the new provenance tests. Executable dry-run reports READY_FOR_LEAD with all 52 cells UNSTARTED. The original RED remains RED and its exact 46.348182500805706-second charge is retained against the unchanged 3,600-second cap.

Evidence classes: author-run CPU unit/suite tests, CPU fake campaign, static source audit, and SHA checks. No new model-quality, GPU, blind-adversary or certification result is claimed.

## 1. Fix and every child-construction/restore site

Production change: `core/graft_repository.py:1005–1015` (plus deepcopy import at line38). At `_cull_graft_direct` child construction, copy exactly B3's `capture_*`, `n_sink`, `arena_width`, `live_shift` projection. Nested values are independently copied. Add `capture_inherited_from_parent: true` and `capture_parent_graft_id: idx`; B3 places both inside manifest `capture`, and load restores them to the graft top level. The identity is the immediate parent's repository graft ID, also used by `sources`/`culled_from` and preserved by ordinary manifest reload.

Recorded hashes, pin, shifts and geometry describe the parent's capture; they are not recomputed from the child's size or current arena. Existing child token spans describe the slice. A parent's cache-slice start remains parent evidence, not a new child harvest claim. Uncaptured parents contribute no new capture fields. Non-split deposits and B3 serialization/load code are unchanged.

| Site | Ruling |
|---|---|
| `core/graft_repository.py:927`, child dictionary at980, append1017 | **Fixed.** Central persistent payload-slice constructor. Copies recorded capture before lifecycle/native synchronization. Covers direct cull, section cull, selected span, `split_graft` alias1041, native span slicing and packed payload slicing. |
| `core/graft_repository.py:563` add_document; `remember:1385`; `_guard_deposit_width:1218` | **Covered by central fix.** LSR deposit guard calls direct cull, then makes parent an index. No second child constructor. |
| `core/graft_repository.py:1304` `_split_for_fit`; new-node sweep1322; `split_oversized:1344` | **Covered by central fix.** Fit/deposit/legacy repair all call the same width guard and cull. |
| `core/graft_arena.py:3135` `_split_unseatable`, repository callback3154; fit driver3512 | **Covered by central fix on persistent branch.** Calls repository splitter. Existing already-split families reuse IDs, not new objects. |
| `core/graft_arena.py:3178` fallback `self.deposit(chunk)` | **Directly captured, unchanged.** Bare-arena/failed-persistence fallback harvests each child anew; deposit at837/864 attaches its own geometry. Copying old parent capture would mislabel that new payload. `ephemeral_split_of` separately records lineage. CPU fakes may omit capture because they do no harvest. |
| `core/grm_admission.py:221,269`; `core/graft_arena.py:3235` and RT1 fit routing | **No constructor.** RT1 identifies/ranks existing family IDs and tests each member's own text. The family originates in one of the two split branches above. |
| `core/graft_arena.py:837,870` deposit/deposit_from_cache; consolidation2106; feed2617/2629; step4274/4454; deferred commit4625 | **Direct capture, unchanged.** New harvest or live-cache slice records its own capture. Digest/era consolidation deposits a new note; `child_cents` is a routing-key list, not new graft children. |
| `core/graft_repository.py:4770` load, capture restore4806, append4823 | **Already correct via B3.** Restores the whole optional capture dictionary, including new inheritance fields; no inferential repair of old manifests. Save/reload and old-manifest tests pass. |
| `core/graft_repository.py:4191` `_wal_placeholder_graft`; append sites4270/4280/4410 | **Legitimately unattested.** WAL reconstruction has text/metadata and payload-pending state, without captured-parent payload evidence. Gap placeholders have no payload. Do not fabricate capture from lineage or current arena geometry. WAL-only recovery is not claimed to restore capture evidence absent from its records. |
| `core/graft_arena.py:3953,4061,4215` rollback/restore | **No child reconstruction.** Restores saved graft objects/list entries, retaining their dictionary fields. Host/device paging restores payload onto existing grafts; no independent parent-to-child constructor. |

The audit searched graft append/extend/list assignments and split/restore methods across `core/*.py`. Only the persistent cull constructor required a production change. Native slice helpers return payload arrays, not graft dictionaries.

### RED-before / GREEN-after tests

Unchanged test source after fixture repair: `tests/test_grm_scout_fix2_capture.py`.

| Test | Before | After |
|---|---|---|
| `test_split_children_inherit_exact_capture[deposit,cull,fit,repair]` | 4 failures: `AssertionError: assert None == {...capture_inherited_from_parent...}` | 4 passed; exact parent capture plus mark, payload slice, strict C2 grade, parent/sibling mutation isolation |
| `test_split_capture_save_reload_round_trip[live,off,cache]` | 3 failures: missing manifest capture (`None`) | 3 passed; exact capture and payload survive save/load/save |
| `test_nested_split_names_immediate_parent` | Missing capture (`None`) | Passed; immediate parent ID retained |
| `test_split_without_capture_does_not_invent_it`; `test_non_split_capture_is_unchanged` | 2 passed | 2 passed |

Raw receipts: [corrected RED](../artifacts/grm_scout_fix_2/red_before_corrected.log), [GREEN](../artifacts/grm_scout_fix_2/green_after.log), [required full CPU suite](../artifacts/grm_scout_fix_2/cpu_full_initial.log). The first draft fixture also had `KeyError: 'rare'` because FakeSliceArena omits that field before save; [initial log](../artifacts/grm_scout_fix_2/red_before.log) is preserved. Adding rare in the fake deposit corrected that fixture before any production edit. Only the corrected 8-fail/2-pass run is treatment evidence.

Required suite command (95 passed, 2 deprecation warnings, no skips):

```sh
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_grm_c2_*.py tests/test_grm_scout_fix1_*.py tests/test_grm_scout_fix2_capture.py
```

This includes B3 `test_old_manifest_without_capture_loads_unchanged`, all four SCOUT-FIX-1 fixture files, all prior C2 test files unchanged, and new epoch tests for 52-cell ordering/dependencies/scoring, resume, retained accounting, budget refusal, orphan charge, RED no-retry, stale source/order/receipt rejection and forged amendment refusal even with a recomputed sidecar. This claim covers the named C2 suite, not every unrelated test in the repository's older 66-file inventory.

Additional audit suite: **178 passed, 2 deprecation warnings, no skips**, 134.09 seconds. [Raw CPU receipt](../artifacts/grm_scout_fix_2/audit_cpu.log).

```sh
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_grm_lsr_p2c_split_descent.py tests/test_grm_rt1_split_child_routing.py tests/test_grm_runtime_lifecycle.py
```

## 2. C2 amendment 3 and exact lead commands

[artifacts/grm_c2/amendment_lead_3.json](../artifacts/grm_c2/amendment_lead_3.json)

SHA-256: `218cbc8cff58762dee57f4fe399d4440f13db74956d6a5cc18bd6859708d9b22`

Order SHA-256 is recorded in that JSON and in the initial SCOUT-FIX-2 registration. Amendment3 binds the immutable order, r1, amendments1/2, historical RED receipts and checkpoints, fixed executable source set, new tests and refreshed executable command. Its verifier has a pinned amendment digest and a registered template hash to avoid self-hash recursion. Added executable source paths are also rejected as a changed epoch.

New runner: `scripts/grm_c2_epoch3.py:29–99`. It configures the existing scheduler/worker in-process to use `artifacts/grm_c2/epochs/scout-fix-2` for cells, checkpoints, scores and campaign lock. Every child invocation uses the same entry point. The old amended CLI redirects there. r1 input verification (`scripts/grm_c2_profile.py:52–67`) permits only amendment3's exact before/after replacements after amendment3 verification. Historical amendment verification (`scripts/grm_c2_amended.py:39–60`) validates archived original command/verifier bytes under the new binding. No earlier registration or order was edited.

All 52 cells are preserved; the estimate remains 3,140 seconds. Including the original charge gives 3,186.3481825008057 seconds, leaving an estimated 413.6518174991943 seconds under the cap. This is the registered heuristic, not a runtime completion guarantee. Reservations still require room for a full 285 seconds; old charge plus new measured charges/orphan reservations count toward the same cap. No second fixed-source epoch is authorized. New RED stops and is not retried; explicit resume follows amendment1's next-unstarted rule within this epoch. Historical checkpoints are not reused.

Exact next lead invocation, from this worktree:

```sh
./artifacts/grm_c2/lead_commands.txt --resume
```

CPU-only preflight:

```sh
./artifacts/grm_c2/lead_commands.txt --dry-run
```

The executable wrapper runs, in order:

```sh
cd /mnt/ForgeRealm/wt/grm-c2
python scripts/grm_c2_epoch3.py --dry-run
python scripts/grm_c2_epoch3.py --run --resume
python scripts/grm_c2_epoch3.py summary
```

It preserves a nonzero run/summary status. The scheduler uses `timeout --signal=TERM --kill-after=5 585 python scripts/grm_c2_epoch3.py --cell CELL_ID`, one foreground `/tmp/forge-gpu.lock` lease per cell, lock wait240s, worker timeout280s, lease285s, outer590s, cooldown30s; score after each battery, restart after persist. Width comparison remains shipped defaults256 versus proposed profile96; defaults96 remains NON_FIT (+26 cells/+1570s), unchanged.

[dry-run](../artifacts/grm_scout_fix_2/dry_run.json) and [integrity receipt](../artifacts/grm_scout_fix_2/final_integrity.json): READY_FOR_LEAD, all52 UNSTARTED, exact old charge, executable command, legacy CLI equality, historical artifacts unchanged. Dry-run did not create the new epoch directory. GPU cells were not executed by this seat.

## 3. Prior art

B3/RS3 capture records and ordinary manifest projection; LSR-P2C librarian payload slicing and width guards; RT1 family routing; C2 r1/amendment1 and CMC1 create-only reservations, SHA bindings, foreground lease/scheduling. Project contributors, **2026**, inspected in local source and orders. Taken: those schemas, payload operations and scheduling/verification conventions. New here: preserving the existing capture projection on sliced children, additive immediate-parent inheritance markers, and a fixed-source epoch that retains historical charges. Deep copying isolates mutable metadata; SHA-256 provides content integrity, not authentication. No prior art known to me for this specific repair/adapter beyond these local systems. No novel algorithm or literature-wide priority claim; no external literature verification was needed for the locally verified lineage.

## Deviations, RED, process safety, identity

The initial fake fixture's missing `rare` field was repaired and logged before the production treatment. Registration compatibility required exact source supersession and archived command/verifier bytes because both r1 and amendment1 bind the replaced inputs. Changes remain within the authorized core finding plus C2 harness/registration scope. Existing tests and validators were not weakened.

Historical RED stays `ValueError: invalid persisted capture evidence`, with the original worker's `end_capture.valid: false` for children4/5 preserved. **Not claimed fixed:** the old persisted manifests, historical RED receipt, or C2 GPU outcome. Fixed behavior is established by the author-run CPU tests; lead GPU validation and any independent blind review remain outstanding.

No git commands, subagents, GPU execution, background waits, process signals/kills, services or external delivery. All tests/dry-runs ran as foreground commands with GPU visibility empty; native fixture libraries were built into pytest temporary directories. Existing native test helpers reference the main repository C++ source; hashes match this worktree's `.cpp` and `.hpp` ([identity receipt](../artifacts/grm_scout_fix_2/native_test_source_identity.json)). No order or earlier registration was modified. Branch name is supplied by the order; no git inspection was performed.

Model: Codex / GPT-6; exact serving API model ID unavailable in this session. Reasoning effort: high, as requested. No delegated model was used.
