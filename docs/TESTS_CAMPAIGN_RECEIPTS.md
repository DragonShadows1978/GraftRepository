# Campaign-receipt tests (GRM-H1, re-derived 2026-09-11 on `grm-merge`)

A **campaign receipt** is a test that is valid only at the sha its campaign
registered. It either

* asserts `INPUT_SHA_MISMATCH`-class binding against `core/` and `scripts/`
  shas that the campaign froze on its own day, or
* reads gitignored campaign artifacts under `artifacts/` (checkpoints,
  receipts, controller state) that a given tree does not carry complete.

Such a test fails on **its own source branch** once the core moves past its
registration. It is a receipt, not a regression detector, and a tree-wide run
must not report it as a regression of the tree under test.

A third binding, added by GRM-H3: it may assert a **file set** a campaign
froze — an attribution split, a pinned input list — that a later campaign grew
past. Same rule; the receipt is of what the campaign registered, not of what
the tree happens to contain today.

**An inherited default is not a controlled arm** (GRM-D2, 2026-09-12). A test
that is silent about a setting reads whatever the tree currently defaults to,
so a default flip moves it without any assertion changing. A campaign receipt
must PIN the arm it was registered under; an arm is a PIN, never an absence.
This is why a completed campaign is marked rather than rebound to new
defaults — rebinding would re-label a finished run as having been done under
settings it never saw. See the GRM-H3 section.

## The marker

```python
@pytest.mark.campaign_receipt(registration='artifacts/grm_r1/registration.json')
def test_...():
```

Registered in `tests/conftest.py`. Behaviour:

| Invocation | Campaign receipts |
|---|---|
| `pytest tests/test_grm_*.py` | **SKIPPED**, with a reason naming the registration |
| `pytest -m campaign_receipt tests/test_grm_*.py` | run (and expected to fail) |
| `pytest --campaign-receipts tests/test_grm_*.py` | run alongside everything else |

**No test assertion was changed to obtain the marker.** Every marked test still
asserts exactly what its campaign registered; only its default *collection*
changed.

## Marked files

Decided per file by reading it (`INPUT_SHA_MISMATCH`, `registration`,
`artifacts/`) and by reproducing every failure **in an isolated single-module
process** — all 113 `tests/test_grm_*.py` modules were run one at a time under
`--campaign-receipts` for this re-derivation, so no mark rests on a mark.
Marks sit on the **test function**, not the module, so healthy coverage in a
mixed module keeps running.

"Tests" counts marked test functions; "Node ids" counts collected test ids
(parametrised cases expand).

| Test module | Tests | Node ids | Class | Registration | Why it is a receipt |
|---|---:|---:|---|---|---|
| `tests/test_grm_a1_gpu_contrast.py` | 2 | 2 | worktree-path | `artifacts/grm_a1/gpu_contrast_registration.json + gpu_contrast_amendment_1..4.json (pins absolute /mnt/ForgeRealm/wt/grm-a1/ paths)` | asserts the amendment pins keyed by ABSOLUTE paths inside the grm-a1 worktree; cannot pass on any other tree by construction |
| `tests/test_grm_c2_amendment.py` | 12 | 18 | artifact-bound | `artifacts/grm_c2/registration.json + orders/GRM_C2_AMENDMENT_1.md` | reads gitignored campaign artifacts under artifacts/ that this tree does not carry complete |
| `tests/test_grm_c2_budget_a5.py` | 15 | 28 | artifact-bound | `artifacts/grm_c2/a5/ + orders/GRM_C2_AMENDMENT_5.md` | reads gitignored campaign artifacts under artifacts/ that this tree does not carry complete |
| `tests/test_grm_c2_epoch3.py` | 7 | 12 | sha-bound | `artifacts/grm_c2/epochs/scout-fix-2/ + orders/GRM_SCOUT_FIX_2.md` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_c2_profile.py` | 1 | 1 | artifact-bound | `artifacts/grm_c2/checkpoints/profile/` | reads gitignored campaign artifacts under artifacts/ that this tree does not carry complete |
| `tests/test_grm_c2_score_a4.py` | 8 | 8 | sha-bound | `artifacts/grm_c2/a4_staging/ + artifacts/grm_c2/epochs/scout-fix-2/` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_c7.py` | 4 | 4 | sha-bound | `artifacts/grm_c7/registration.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_c7_amendment7.py` | 3 | 3 | sha-bound | `artifacts/grm_c7/amendment_7.json + orders/GRM_C7_AMENDMENT_7.md` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_c7_lead_1.py` | 1 | 1 | sha-bound | `artifacts/grm_c7/amendment_lead_1.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_c7_lead_2_diagnosis.py` | 6 | 6 | artifact-bound | `artifacts/grm_c7/diagnosis_lead_2/registration.json` | reads gitignored campaign artifacts under artifacts/ that this tree does not carry complete |
| `tests/test_grm_c7_r2_launch.py` | 1 | 4 | sha-bound | `artifacts/grm_c7/r2/ (C7 r2 registration)` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_c7_r2_registration.py` | 2 | 12 | sha-bound | `artifacts/grm_c7/r2/ (C7 r2 registration)` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_c7_r3.py` | 5 | 5 | sha-bound | `artifacts/grm_c7/r3/registration.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_d1_amendment3.py` **(new)** | 1 | 1 | sha-bound | `artifacts/grm_d1/lt1_1/registration.json (LT1.1 chain, core pins rebound at D1 amendment 4)` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_lt1.py` | 3 | 4 | sha-bound | `artifacts/grm_lt1/registration.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_lt1_1_preflight.py` **(new)** | 8 | 11 | sha-bound | `artifacts/grm_d1/lt1_1/registration.json + artifacts/grm_lt1/amendment4/resume_registration.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_lt1_amendment1.py` | 3 | 4 | sha-bound | `artifacts/grm_lt1/amendment1/registration_amendment.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_lt1_amendment3.py` | 3 | 3 | sha-bound | `artifacts/grm_lt1/amendment3/registration_amendment.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_lt1_amendment4.py` | 9 | 14 | artifact-bound | `artifacts/grm_lt1/amendment2/run_margin_first/cells/*/checkpoint` | reads gitignored campaign artifacts under artifacts/ that this tree does not carry complete |
| `tests/test_grm_lt1_controller.py` | 2 | 2 | sha-bound | `artifacts/grm_lt1/registration.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_lt1_fix6.py` | 3 | 4 | sha-bound | `artifacts/grm_lt1/ (FIX-6 amendment)` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_r1_amendment1.py` **(re-marked)** | 11 | 11 | sha-bound | `artifacts/grm_r1/amendment_1.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_r1_amendment2.py` **(new)** | 6 | 6 | sha-bound | `artifacts/grm_r1/amendment_2.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_r1_amendment3.py` **(new)** | 10 | 10 | sha-bound | `artifacts/grm_r1/amendment_3.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_r1_amendment4.py` **(new)** | 8 | 8 | sha-bound | `artifacts/grm_r1/amendment_4.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_r1_cpu_gate.py` | 1 | 1 | sha-bound | `artifacts/grm_r1/registration.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_r1_replay.py` | 15 | 15 | sha-bound | `artifacts/grm_r1/registration.json + orders/GRM_R1_MARGIN_FIRST_REPLAY.md` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_scout_fix4_continuation.py` | 3 | 10 | sha-bound | `artifacts/grm_scout_fix4/registration.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_scout_fix5_runner.py` | 4 | 16 | artifact-bound | `artifacts/grm_scout_fix5/registration.json + artifacts/grm_c7/r2/cells/*/checkpoint` | reads gitignored campaign artifacts under artifacts/ that this tree does not carry complete |
| `tests/test_grm_scout_fix8_replay.py` | 5 | 5 | sha-bound | `artifacts/grm_scout_fix8/registration.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_scout_fix8_resume.py` | 2 | 3 | sha-bound | `artifacts/grm_scout_fix8/resume_amendment_1/registration.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| **31 modules** | **164** | **232** | | | |

## Delta from the first pass (pre-merge tree, 100 GRM modules -> 113)

| Change | Module | Tests | Reason |
|---|---|---:|---|
| **removed** | `tests/test_grm_chat.py` | 1 | The merge resolved this file to the later version: the "`GRM_ALIAS_FOLD_MERGE` is absent from this tree" test was replaced by a present-flag test once A1 landed. It is no longer a receipt; it passes. |
| **removed** | `tests/test_grm_d1_registration.py` | 1 | Same cause, same resolution. Passes. |
| **removed** | `tests/test_grm_r1_replay.py` | 3 | `test_parity_barrier_stops_the_run`, `test_parity_barrier_passes_on_the_recorded_plan`, `test_both_arms_share_one_verified_state` now PASS — R1 amendments 2-4 rebound R1's pins to this tree. Verified individually before unmarking. |
| **re-marked** | `tests/test_grm_r1_amendment1.py` | 11 | The merge resolved this file to the later version (R1 amendments 2-4 rewrote it), which dropped the first pass's marks. Re-derived: still `R1_INPUT_SHA_MISMATCH: core/graft_arena.py`. |
| **added** | `tests/test_grm_r1_amendment2.py` | 6 | New module. `R1_INPUT_SHA_MISMATCH: core/graft_arena.py`. |
| **added** | `tests/test_grm_r1_amendment3.py` | 10 | New module. Same binding. |
| **added** | `tests/test_grm_r1_amendment4.py` | 8 | New module. Same binding. |
| **added, then cut to 2** | `tests/test_grm_a1_gpu_contrast.py` | 16 -> 2 | Marked 16 during the re-derivation, when the module failed wholesale on `A1_INPUT_SHA_MISMATCH`. A1 amendment 4 then REBOUND the pins (`pinned inputs verified: 27`), so that binding now passes and 14 of those marks were stale. See the A1 section below. |
| **added** | `tests/test_grm_lt1_1_preflight.py` | 8 | New module. LT1.1's own chain gate returns `BLOCKED` with `INPUT_SHA_MISMATCH` on `core/graft_arena.py`, `core/graft_repository.py`, `core/grm_alias_fold.py`. |
| **added** | `tests/test_grm_d1_amendment3.py` | 1 | New module; one test (`test_the_host_blocker_claim_was_true_and_is_now_resolved`) asserts LT1.1's gate is `READY`. It is `BLOCKED` by the same three-file drift, so the assertion is bound to LT1.1's registration sha. |
| **NOT marked** | `tests/test_grm_scout_fix2_capture.py` | 3 | A genuine merge-drift RED — fixed, not marked. See below. |

Net after the A1 amendment 4 re-derivation: 164 marked test functions
(232 node ids) across 31 modules -- DOWN from 178/246, because A1 amendment 4
rebound its pins and retired 14 marks. See the A1 section below.
Modules that newly landed and are **clean** on this tree, needing no marks:
`test_grm_d1_amendment1/2`, `test_grm_d1_alias_cpu`, `test_grm_lt1_1_runner`,
`test_grm_lt1_1_resume_route`, `test_grm_scout_fix9`.

### A test that pins its own worktree path is always a receipt

If a test (or the script it drives) verifies shas against **absolute paths
inside the worktree it was authored in** -- `/mnt/ForgeRealm/wt/grm-a1/...`,
`/mnt/ForgeRealm/wt/grm-c7/...` -- it **cannot pass on any other tree**, by
construction, no matter how the core moves or how the campaign is re-pinned.
That is the strongest form of campaign receipt: it is bound not just to a sha
but to a filesystem location that exists on exactly one branch. It is also the
most durable: rebinding the campaign's pins (as A1 amendment 4 did) retires a
sha-bound mark, but never a worktree-path one.

Such a test must be marked `campaign_receipt` **in the same change that lands
it**, not later by a hygiene pass. A seat adding one to an existing campaign
module should mark it the way its siblings are marked; `grep campaign_receipt`
in the file gives the registration string to reuse. The cost of not doing so is
that the next tree-wide run reports it as a regression of whatever unrelated
work happens to be in flight.

Receipt for the rule: A1 amendment 4 added four tests to
`tests/test_grm_a1_gpu_contrast.py` after GRM-H1's re-derivation, and they
surfaced as four unexplained failures in the lead's next spot-check.


## A1 amendment 4: why 16 marks became 2

The lead reported four new failures in `tests/test_grm_a1_gpu_contrast.py`
(A1 amendment 4's lease/summary tests) and asked for them to be marked like
their 16 siblings. Reproducing each one in its own process showed that would
have been wrong twice over, so the whole module was re-derived instead.

**1. The sha binding no longer fails.** A1 amendment 4 rebound the campaign's
pins; the worker now prints `pinned inputs verified: 27` on this tree. The
`A1_INPUT_SHA_MISMATCH` that justified all 16 marks during the re-derivation
is gone. Running each of the 16 alone: 3 pass outright, 2 fail on the
worktree-path assertion, 11 fail for the reason below. Only the 2 are still
receipts.

**2. The four "new failures" were caused by this guard, not by a campaign.**
All four -- and 12 of the 16 -- failed with `GRM_ENV_LEAK`, because
`W.main()` calls `scripts/grm_a1_gpu_contrast.pin_flags()`, which does
`os.environ.clear()` and repopulates from `environment(flags)`. That is
deliberate: the worker normally runs as a one-shot process that owns its
environment. Marking those tests `campaign_receipt` would have filed a bug in
this guard under a label that means "not our problem" -- the exact
mis-classification the marker exists to prevent.

The fix is a second, narrower marker:

```python
pytestmark = pytest.mark.grm_env_owned_by_entrypoint(
    reason='scripts/grm_a1_gpu_contrast.pin_flags() owns the process '
           'environment (os.environ.clear() + environment(flags))')
```

It suppresses **attribution only**. The restore still happens, so no later
test is mis-ruled -- proved by
`test_grm_h1_env_isolation.py::test_entrypoint_owned_environments_are_restored_but_not_attributed`,
with a companion test asserting the declaration does not leak to other
modules. Declare it per module, naming the entry point; never relax the guard
globally.

Result on this module: `--campaign-receipts` goes from 11 failed / 35 passed /
5 errors to **2 failed / 58 passed**, and the two failures are the genuine
worktree-path receipts.


## Failures that are NOT receipts

Two modules failed on a merged tree because a **test double went stale against
a merge**, not because a campaign sha moved. Marking either would have hidden a
real regression, so both were fixed in the test fixture. No assertion changed;
`core/` and `scripts/` untouched.

1. `tests/test_grm_native_runtime.py::test_gqa_cuda_epoch_bump_covers_graft_repository_mutation_battery`
   (first pass). A1 added `self.alias_fold_merge` to `GraftRepository.__init__`
   and a read of it in `correct_memory()`; the fixture builds the repository via
   `GraftRepository.__new__`, bypassing `__init__`, so the battery died with
   `AttributeError` before exercising the epoch contract. Fixed in
   `_fake_gqa_repo_for_epoch_battery` by listing the three attributes A1 added,
   at `__init__`'s defaults with the flag OFF.
2. `tests/test_grm_scout_fix2_capture.py::test_split_children_inherit_exact_capture[deposit|fit|repair]`
   (this pass). SCOUT-FIX-9's degenerate-split-child guard calls
   `self.arena._fact_set(...)`; `FakeArena` in
   `tests/test_grm_runtime_lifecycle.py` is not an `ArenaCache` subclass and
   never inherited it — `AttributeError: 'CapturedArena' object has no
   attribute '_fact_set'`. Fixed by borrowing the real
   `ArenaCache._fact_set` onto the double (the pattern that class already uses
   for `_rare_tokens`), so the fixture measures FIX-9's actual fidelity
   vocabulary and cannot drift from it.

## Environment isolation

`tests/conftest.py` snapshots every `GRM_*` environment variable at setup,
restores it exactly (deleting keys that did not exist), and **fails the test
that leaked**, naming the key. See the module docstring there for the
mechanism, and `tests/test_grm_h1_env_isolation.py` (11 tests) for the proof.

Leakers found, across two full sweeps of the suite:

| Test | Leak | Status |
|---|---|---|
| `test_grm_r1_replay.py::test_parity_barrier_stops_the_run` | `GRM_ADMISSION_RULE: unset -> 'margin_first'` | Fixed at source by R1 amendment 2; no longer leaks. |
| `test_grm_a1_gpu_contrast.py::test_pin_failure_is_red` | `GRM_A1_FLAG_THAT_NOTHING_READS: unset -> '1'` | Origin is `scripts/grm_a1_gpu_contrast.pin_flags()` (read-only here), which writes `os.environ` directly and raises `A1_FLAG_NOT_IN_FORCE_AFTER_PIN` before any cleanup. No longer attributed: the module now declares `grm_env_owned_by_entrypoint` (see the A1 section). Still restored. |

That second test also calls `os.environ.clear()` inside `pin_flags()` and so
**wiped 86 non-`GRM_` variables** — `PATH`, `HOME`, `DISPLAY` and the rest —
for every later test in the process. That is outside the `GRM_*` contract the
guard *attributes* on, but leaving it unrepaired would mis-rule far more than a
pinned admission rule would, so `_restore_non_grm()` repairs pre-existing
non-`GRM_` keys unconditionally and silently; only the `GRM_*` leak fails a
test. Proved by
`test_grm_h1_env_isolation.py::test_a_wholesale_environ_clear_is_repaired_for_the_next_test`.

## Gates (merged tree, 2026-09-11) — GRM-H1, SUPERSEDED

These are H1's numbers, recorded **before** the eight Scout Part C / X
branches were merged and their worktrees pruned. Kept as the H1 receipt; the
current gates are in the GRM-H2 section below.

```
$ python3 -m pytest -q tests/test_grm_*.py
2309 passed, 246 skipped, 2 warnings in 585.77s (0:09:45)

$ python3 -m pytest -q -m campaign_receipt tests/test_grm_*.py
176 failed, 2 passed, 2309 deselected, 2 warnings, 68 errors in 59.00s

$ python3 -m pytest -q tests/test_grm_r1_*.py tests/test_grm_scout_fix4.py \
      tests/test_grm_a1_alias_fold.py tests/test_grm_admission.py
204 passed, 51 skipped, 2 warnings in 9.22s
```

176 + 68 + 2 = 246, matching the skip count exactly. The 2 that pass under `-m`
are two parametrised cases of
`test_grm_c2_epoch3.py::test_stale_epoch_input_refused` whose other cases fail;
the mark is on the function, so all its cases travel together.

## GRM-H2 delta: the eight Scout Part C / X merges (2026-09-11)

`lc1-wip` absorbed grm-c3, grm-c4, grm-c5, grm-c6, grm-c8, grm-x1, grm-x2,
grm-x3, **and the eight forks were then pruned**. Pruning is what makes this
pass large: it converted every absolute-worktree pin across the tree from
"file present, sha moved" into "file absent", which is the durable
worktree-path receipt class this document already names as the strongest
and least re-bindable one.

Method as in H1: each module reproduced in an **isolated single-module
process** under `--campaign-receipts`, marks on test FUNCTIONS, no assertion
changed anywhere. 166 marks added across 27 modules.

### In scope -- the 21 modules the eight merges added or changed

| Test module | Marks added | Marked fns | Node ids | Why it is a receipt |
|---|---:|---:|---:|---|
| `tests/test_grm_c4_accounting_a7.py` | 5 | 5 | 20 | same `binding()` chain (a7 -> a6 -> a5 -> ...) |
| `tests/test_grm_c4_campaign.py` | 7 | 7 | 10 | C4 amendments record `registration` as `{path, sha256, bytes}` with an ABSOLUTE path; sha and bytes still match byte-for-byte, only `/mnt/ForgeRealm/wt/grm-c4/` differs, so `grm_c4_campaign.binding()` raises `amendment registration mismatch` |
| `tests/test_grm_c4_cap_a4.py` | 4 | 4 | 13 | same `binding()` chain; one case also reads `/mnt/ForgeRealm/wt/grm-c4/artifacts/grm_c4/` |
| `tests/test_grm_c4_recovery_a8.py` | 6 | 6 | 12 | same `binding()` chain (a8 -> ...) |
| `tests/test_grm_c4_recovery_a8_sequence.py` | 1 | 1 | 1 | same `binding()` chain, reached through the a5 `layout` fixture it imports |
| `tests/test_grm_c4_remaining_a3.py` | 9 | 9 | 22 | same `binding()` chain (a3 -> a2 -> campaign) |
| `tests/test_grm_c4_resume_a2.py` | 19 | 19 | 27 | same `binding()` chain (a2 -> campaign) |
| `tests/test_grm_c4_skip_a6.py` | 7 | 7 | 14 | same `binding()` chain; one case also reads `/mnt/ForgeRealm/wt/grm-c4/AGENTS.md` |
| `tests/test_grm_c4_split_a5.py` | 10 | 10 | 23 | same `binding()` chain (a5 -> a3 -> a2 -> campaign) |
| `tests/test_grm_c5_grounding_receipt.py` | 1 | 1 | 1 | holds the collectable half of the C5 receipt; `test_grm_c5_grounding.py` verifies the registration at IMPORT time and now skips at module level (see below) |
| `tests/test_grm_c8_amendment.py` | 10 | 10 | 18 | `grm_c8_cells` raises `bound input changed: core/graft_arena.py`; `lead_commands.txt` also `cd`s to `/mnt/ForgeRealm/wt/grm-c8` |
| `tests/test_grm_c8_profiler.py` | 4 | 4 | 4 | `bound input changed: core/graft_arena.py` |
| `tests/test_grm_x1_campaign.py` | 1 | 1 | 1 | `frozen source/fixture drift: docs/GRM_SCOUT_2026-09-08.md` (and `scripts/grm_e2e_session.py`) |
| `tests/test_grm_x1_reduced.py` | 13 | 13 | 33 | `r4 source drift outside harness scope` -- 8 pinned `core/` and `scripts/` files moved, 2 new `core/` files appeared; the `r4_tree` fixture also copies the campaign`s real `artifacts/grm_x1/claims/r4/` (23 gitignored claims), so the create-only writes collide |
| `tests/test_grm_x2_runner.py` | 5 | 5 | 5 | `registered input drift: core/graft_arena.py` |
| `tests/test_grm_x2_witnesses.py` | 1 | 1 | 1 | `registered input drift: core/graft_arena.py` |
| `tests/test_grm_x3_lesions.py` | 9 | 9 | 13 | `frozen input drift: core/graft_arena.py` |
| `tests/test_grm_x3_r2.py` | 8 | 8 | 10 | `frozen r2 input drift: core/graft_arena.py`; `r1_evidence_before.json` also pins 5161 files of which 20 gitignored `runs/*/donor_payload.npz` are absent |
| **18 modules** | **120** | **120** | **228** | |

Clean on this tree, needing no marks: `test_grm_c3.py`,
`test_grm_x1_addresses.py`, `test_grm_x1_units.py` (29 + 31 + 22 pass).
grm-c6 added no test module.

### Fallout -- pruning the forks broke pins in modules the merges never touched

These were green for H1 **because the campaign worktrees were still on
disk**. They are not new defects and not new drift; they are the same
receipts, now unable to resolve their own pinned paths.

| Test module | Marks added | Marked fns | Node ids | Why it is a receipt |
|---|---:|---:|---:|---|
| `tests/test_grm_a1_gpu_contrast.py` | 25 | 27 | 27 | **all 27** effective pins are absolute paths in the pruned `grm-a1`, `grm-c7` and `grm-rd1` worktrees; every one now hashes to `null` |
| `tests/test_grm_c7_lead_1.py` | 1 | 2 | 4 | same single pruned C3 order pin |
| `tests/test_grm_c7_r2_registration.py` | 2 | 4 | 18 | `artifacts/grm_c7/registration.json` `immutable_inputs` pins `/mnt/ForgeRealm/wt/grm-c3/orders/GRM_C3_DNGH_DECOY_CALIBRATION.md`; `verify()` cannot reach its sha comparison |
| `tests/test_grm_c7_r3.py` | 1 | 6 | 10 | same single pruned C3 order pin |
| `tests/test_grm_r1_amendment2.py` | 2 | 8 | 8 | same pruned C2 checkpoint, reached through `run.run_cell` |
| `tests/test_grm_r1_replay.py` | 8 | 23 | 23 | cell checkpoints pinned under `/mnt/ForgeRealm/wt/grm-c2/artifacts/grm_c2/epochs/scout-fix-2/` -> `R1_CHECKPOINT_MISSING` |
| `tests/test_grm_rd1_replay.py` | 5 | 5 | 5 | `scripts/grm_rd1.py` hardcodes `SOURCE = Path('/mnt/ForgeRealm/wt/grm-c7')` |
| `tests/test_grm_scout_fix8_replay.py` | 1 | 6 | 6 | A7 archive pins `/mnt/ForgeRealm/wt/grm-c7/artifacts/grm_c7/r3/amendment_7/` |
| `tests/test_grm_scout_fix9.py` | 1 | 1 | 1 | `scripts/grm_scout_fix8_cpu.py` hardcodes `C2 = Path('/mnt/ForgeRealm/wt/grm-c2/...')`; the glob returns 0 rows where 132 are asserted |
| **9 modules** | **46** | **82** | **102** | |

Three of the eight `test_grm_r1_replay.py` marks are **re-marks**: H1
removed them because "R1 amendments 2-4 rebound R1's pins to this tree".
That rebinding was real but not durable -- the pins it rebound to were
absolute paths in `grm-c2`. This document's own rule predicted it: a
worktree-path receipt is never retired by re-pinning. The same correction
applies to `test_grm_a1_gpu_contrast.py`, whose "A1 amendment 4 rebound the
pins, `pinned inputs verified: 27`" result held only while `/mnt/ForgeRealm/
wt/grm-a1/` existed; all 27 of those pins are absolute worktree paths, so
the module is 27/27 worktree-path receipts, not 2.

`test_grm_d1_amendment3.py` and `test_grm_lt1_1_preflight.py` went the other
way: D1 amendment 8 rebound them to this tree and removed their marks. Both
now pass outright (37 passed, 0 skipped) -- correctly unmarked, left alone.

### A receipt whose binding runs at IMPORT time

`tests/test_grm_c5_grounding.py` calls `load_fixtures()` at module scope,
because what it returns IS the `parametrize` argument list; it cannot be
deferred into a fixture without changing what the campaign registered. That
call verifies C5's registration, which pins
`/mnt/ForgeRealm/wt/grm-c5/artifacts/grm_c5/fixtures.json` and gitignored
EB1 session artifacts. Unguarded it does not merely fail -- it aborts
collection of the whole run (`Interrupted: 1 error during collection`), and
`-m campaign_receipt` aborts identically, so the receipt gate could not run
either. A function-level mark cannot reach it: the import never completes.

The fix is a pair, `campaign_receipt_module()` in `tests/conftest.py`:

* the module skips itself at import, unconditionally -- there is no
  invocation in which that import can succeed off the grm-c5 tree;
* `tests/test_grm_c5_grounding_receipt.py` carries ONE
  `campaign_receipt`-marked test calling the same `load_fixtures()` binding,
  so the receipt stays collectable, deselected by default and reproduced
  under `-m campaign_receipt` like every other one.

The skip alone would have converted a receipt into silence; the pair is what
keeps it honest. Use this shape only for modules whose binding is evaluated
by the import itself.

### Failures that are NOT receipts

None. Every failure in all 21 in-scope modules, and every fallout failure,
resolved to a registration binding or a pruned absolute path. No genuine
defect was found, and no test double went stale against these merges.

### Gates (GRM-H2, merged + pruned tree, 2026-09-11)

The one-process tree-wide run is ~705 s, over the 600 s ceiling this seat
runs under, so it is reported in three parts over the same 138 modules.

```
$ python3 -m pytest -q --basetemp artifacts/grm_h2/tmp $(ls tests/test_grm_*.py | sed -n 1,70p)
1114 passed, 321 skipped, 2 warnings in 25.15s

$ python3 -m pytest -q --basetemp artifacts/grm_h2/tmp $(ls tests/test_grm_*.py | sed -n 71,101p)
600 passed, 102 skipped, 2 warnings in 531.27s (0:08:51)

$ python3 -m pytest -q --basetemp artifacts/grm_h2/tmp $(ls tests/test_grm_*.py | sed -n 102,138p)
1014 passed, 99 skipped, 2 warnings in 147.07s (0:02:27)

$ python3 -m pytest -q --basetemp artifacts/grm_h2/tmp -m campaign_receipt tests/test_grm_*.py
314 failed, 8 passed, 1 skipped, 2745 deselected, 2 warnings, 182 errors in 73.54s (0:01:13)

$ python3 -m pytest -q --basetemp artifacts/grm_h2/tmp tests/test_grm_r1_*.py \
      tests/test_grm_scout_fix4.py tests/test_grm_a1_alias_fold.py tests/test_grm_admission.py
194 passed, 61 skipped, 2 warnings in 7.83s
```

2728 passed, 522 skipped, **0 failed / 0 errors** across the three parts.

Skip arithmetic: 314 failed + 182 errors + 8 passed = **504** collected
receipt node ids; + 1 module-level C5 skip + 17 pre-existing non-receipt
skips (12 `test_grm_d1_cause_table.py` `skipif`, 5 LT1.1 archive/full-run
`skipif`) = **522**, matching the default skip count exactly.

Tree totals after this pass: **321 marked test functions, 504 node ids,
49 modules** (H1 left 164 / 232 / 31).

## GRM-F6 rule: repo-relative paths, and a dead glob is RED (2026-09-11)

**Scripts and registrations pin repo-relative paths; a dead-path glob is RED,
never zero rows.**

Round-1 seat worktrees `/mnt/ForgeRealm/wt/grm-*` were pruned on 2026-09-11.
Several scripts still resolved receipt data through absolute paths into them,
and the failure mode that matters is the SILENT one:

```python
C2 = Path('/mnt/ForgeRealm/wt/grm-c2/artifacts/grm_c2/epochs/scout-fix-2')
for worker in sorted(C2.glob('cells/*/worker.json')):   # dead root -> 0 rows
    ...
```

A glob over a dead path raises nothing. The census loop produces zero rows,
and any gate phrased "every row agrees" then passes over **nothing**. A gate
that cannot fail is not a gate.

The rule has three parts.

1. **Resolve from `__file__`, never from the CWD and never from a worktree
   name.** `scripts/grm_repo_paths.repo_root()` is
   `Path(__file__).resolve().parents[1]`; a receipt root is
   `repo_root() / 'artifacts/...'`. A seat worktree name in a path constant is
   a bug with a fuse on it — `scripts/grm_f5_c2_replay_gate.py` "fixed" the
   dead `grm-c2` path by hard-coding `grm-f5`, which is the same bug one
   worktree later.

2. **A missing receipt root fails LOUD, naming everything.**
   `receipt_root(relative, env, pinned_by, dead_absolute)` raises
   `GRM_F6_DEAD_RECEIPT_PATH` carrying the resolved path, the env var that
   overrides it, the receipt that pinned the original, and the pruned path it
   replaced — so the RED is diagnosable without reading the script. Each
   script gets its own override var (`GRM_C2_EPOCH_ROOT`,
   `GRM_C7_SOURCE_ROOT`, `GRM_LT1_CELLS_ROOT`), naming a ROOT that the
   relative path is appended to.

3. **Zero rows feeding a gate is RED.** Every glob/list that feeds a gate
   goes through `require_rows(rows, path, pattern, what)`, which raises
   `GRM_F6_VACUOUS_ZERO` naming the path and the pattern. "No disagreements"
   must never be reachable from "no rows".

**Registrations are NOT rewritten.** A `registration.json` that sha-binds
`/mnt/ForgeRealm/wt/grm-c7/...` is a receipt of what was hashed on the day;
editing it would forge the receipt. Only the *consumer's* path resolution
moves. `scripts/grm_c7_register.py` keeps its absolute C3 order pin for
exactly this reason, even though that order survives in-repo: it is a
registration BUILDER, and rebinding it would change what a future
registration hashes.

`registration=` marker text that says "pins absolute /mnt/ForgeRealm/wt/...
paths" is the receipt staying honest about what it was true at. It is prose,
not a path, and must not be scrubbed.

### Not everything can be rebound

`scripts/apamq_fc_ppl.py` / `scripts/apamq_fbd1_diag.py` default the
`apa_int4` leg to `/mnt/ForgeRealm/wt/apamq-fa/tensor_cuda` and *refuse* any
other root; `artifacts/apamq_fc/apa_int4.json` pins
`engine.compiled_module` to a `.so` inside that pruned worktree.
`/mnt/ForgeRealm/wt/apamq-fa2` is a different Project-Tensor worktree, not
that build. **No surviving copy** — reported, not repaired, and gated by
`test_apamq_int4_leg_has_no_surviving_copy` so the classification fails
loudly if a copy ever reappears.

### Gates

`tests/test_grm_f6_repo_relative_paths.py` walks the **AST** of every rebound
script (comments and docstrings excluded, so the history stays readable) and
asserts no `/mnt/ForgeRealm/wt/` path is bound as live code; plants a dead
path per script and asserts the loud failure; and asserts the vacuous-zero
guard. `tests/test_grm_f6_reproductions.py` pins each rebound script's number
against its frozen receipt.

## GRM-H3 delta: the D2 default flip (2026-09-12)

GRM-D2 shipped the round-2 defaults, and in doing so moved
`core/grm_admission.py`, `core/grm_alias_fold.py`,
`core/grm_fold_alias_guard.py`, `core/grm_fold_retain.py` and
`scripts/grm_lt1_1.py` — plus, as fallout of the same flip,
`scripts/grm_r1_replay.py` and `scripts/grm_a1_gpu_contrast.py`.

The **LT1.1 r3 chain** (registration `02b44d02` + amendments 1–13) pins the
first five byte-for-byte. That campaign is COMPLETE: its r3 receipts are
bound and its finding is registered. So its tests are receipts now, and the
lead's ruling is **mark, do not rebind** — re-pinning a finished campaign's
inputs to today's core would forge the receipt of what actually ran.

Method as in H1/H2: every failure reproduced in an **isolated single-module
process** first, marks on test FUNCTIONS, no assertion changed anywhere.
D2's three-part suite reported 38 failures; 37 were receipts and got 23
marks (a parametrised function travels under one mark), and 1 was not — see
below.

| Test module | Marks added | Marked fns | Node ids | Class | Why it is a receipt |
|---|---:|---:|---:|---|---|
| `tests/test_grm_f1_r3_receipts.py` | 8 | 8 | 16 | artifact-bound | reads the gitignored r3 cell trees under `artifacts/grm_f1/lt1_1_r3/run_*/cells/`. The 26 cell DIRECTORIES survive, but 0 of 26 `checkpoint/repository/manifest.json` and only 11 of 26 `probes.jsonl` do, so `manifest(arm)` dies `IndexError: list index out of range` on `cells[-1]` |
| `tests/test_grm_lt1_1_preflight.py` | 8 | 8 | 11 | sha-bound | the chain preflight returns `BLOCKED` with `INPUT_SHA_MISMATCH` on all four moved core files and `RUNNER_SHA_MISMATCH: scripts/grm_lt1_1.py` |
| `tests/test_grm_f1_resume_out_root.py` | 1 | 1 | 3 | sha-bound | drives the REAL resume route; the spawned child dies `LT11_CHILD_PREFLIGHT_BLOCKED` on those same five mismatches → `WORKER_EXIT_1` → rc 2 |
| `tests/test_grm_lt1_1_child_spawn.py` | 1 | 1 | 2 | sha-bound | same child preflight, same five mismatches → `run_cell` returns False |
| `tests/test_grm_d1_amendment3.py` | 1 | 1 | 1 | sha-bound | asserts LT1.1's governing gate is `READY`; it is `BLOCKED` by the same five |
| `tests/test_grm_f1_fold_retain.py` | 1 | 1 | 1 | sha-bound | the governing amendment (13) pins `core/grm_admission.py` at `7912f291`; D2 moved it to `bc5f7996` |
| `tests/test_grm_r1_amendment2.py` | 1 | 1 | 1 | sha-bound | `artifacts/grm_r1/amendment_2.json` pins `scripts/grm_r1_replay.py` at `9301c7f8`; D2 edited that script for the `margin_first` default and the `GRM_LEGACY_DEFAULTS` pin |
| `tests/test_grm_a1_gpu_contrast.py` | 1 | 1 | 1 | sha-bound | the A1 amendment chain pins its worker `scripts/grm_a1_gpu_contrast.py` at `47f2f7d1`; D2 edited it (the verification repair at its line 654) |
| `tests/test_grm_d1_amendment1.py` | 1 | 1 | 1 | file-set-bound | D1 amendment 1 froze an A1-vs-pre-A1 attribution split that a later campaign's docstring grew past — see below |
| **9 modules** | **23** | **23** | **37** | | |

The 38th failure, the other one in `test_grm_d1_amendment1.py`, was NOT a
receipt and was pinned instead.

### A receipt can be bound to a FILE SET, not only a sha

`test_a1_files_are_attributed_to_a1_and_the_others_are_not` is marked, and it
is worth saying why, because it is not an `INPUT_SHA_MISMATCH`.

D1 amendment 1 froze a two-way attribution split:
`A1_FILES = ('core/graft_repository.py', 'core/grm_runtime.py')`, everything
else "pre-A1 drift, not attributable to A1". The test makes that attribution
CHECKABLE: an A1 file must contain the string `alias_fold`, a non-A1 file must
not. `core/graft_arena.py` is attributed pre-A1 — and a later campaign added
the word `alias_fold` to a **GRM-A1 docstring** inside it (line 2195,
describing why A1 pins `retain=False`). Prose, not live code; the substring
scan cannot tell the difference.

So the receipt here is bound to the FILE SET the amendment froze, which a
later campaign's documentation grew past. Rebinding would mean editing
`A1_FILES` in a completed amendment's generator — forging what that amendment
attributed on its day. Same rule as a sha: a campaign's registration is a
receipt of what was true when it ran.

### Failures that are NOT receipts

One of the 38.

`test_grm_d1_amendment1.py::test_amendment1_no_longer_owns_lead_commands`
asserts a fact about the CURRENT tree — `artifacts/grm_d1/lt1_1/lead_commands.txt`
is owned by some amendment that is not amendment 1, names
`scripts/grm_lt1_1.py`, and no longer carries the dead
`grm_lt1.py --run --registration …` shape. **Every one of those assertions
still passes.** What broke is the test's own OWNER PROBE:

```python
latest = max(int(p.stem.replace('amendment', ''))
             for p in am.OUT.glob('amendment[0-9]*.json'))
assert 'amendment %d' % latest in text
```

It assumed the highest-numbered amendment always regenerates the file.
Amendments 10–13 are campaign RECORDS — bound receipts of runs that already
happened, written by `scripts/grm_f1_register_lt11_r3_amd13.py` and friends —
and none of them rewrites `lead_commands.txt`. The last that did was
amendment 9, which the file says in its own first line. So the probe asked the
file to name an amendment that never wrote it. Marking that would have filed a
stale probe under a label meaning "not our problem".

Fixed as a PIN, not an assertion change: `latest` is now read from the file's
own header. Two notes on the shape:

* The match is an **anchored regex on the header line**, not the original
  `'amendment %d' % latest in text` substring. That substring passes off any
  sha line that happens to carry the number — `# amendment 7  sha256: …` — so
  a *wrong* `latest` could be confirmed by an unrelated line. Verified: with
  the wrong answer 7, the old probe returns True and the new one returns
  False. A gate a wrong answer can satisfy is not a gate (the GRM-F6
  vacuous-zero rule, one line up from a dead glob).
* Globbing `scripts/grm_d1_amendment[0-9]*.py` instead would have been the
  same bug: those scripts stop at 7, so it would have "passed" at 7 purely by
  that substring accident, while the real owner is 9.

### The D2 rule: an inherited default is not a controlled arm

D2 flipped `GRM_ADMISSION_RULE`, the C2 profile, and the four fold/alias flags
to ON **as shipped defaults**. A test that was silent about those settings
used to get the old values for free; it now gets the new ones for free. In
both cases it is reading an INHERITED default, not a pinned arm — and an
inherited default is not a controlled arm.

The consequence for this document: a campaign receipt must pin the arm it was
registered under, never rely on the tree's default agreeing with it.
`scripts/grm_r1_replay.py` is the worked example — D2 gave it an explicit
`R1_RULE_PIN_FAILED: arm=off observed=margin_first` rather than letting an
absent variable stand for "off", because *an arm is a PIN, never an absence*.
The same reasoning is why D2's own tests were re-pinned with
`GRM_LEGACY_DEFAULTS=1` in a fixture instead of having their assertions
relaxed, and why rebinding the LT1.1 chain to the new defaults would have been
wrong: the r3 receipts record a run under the OLD defaults with the four flags
pinned ON explicitly. Re-pinning would silently re-label that run as having
been done under settings it never saw.

### Gates (GRM-H3, D2 tree, 2026-09-12)

```
$ python3 -m pytest -q --basetemp artifacts/grm_h3/tmp $(ls tests/test_grm_*.py | sed -n 1,70p)
1129 passed, 360 skipped, 2 warnings in 26.05s

$ python3 -m pytest -q --basetemp artifacts/grm_h3/tmp $(ls tests/test_grm_*.py | sed -n 71,101p)
540 passed, 111 skipped, 2 warnings in 461.64s (0:07:41)

$ python3 -m pytest -q --basetemp artifacts/grm_h3/tmp $(ls tests/test_grm_*.py | sed -n 102,146p)
1225 passed, 104 skipped, 2 warnings in 239.39s (0:03:59)

$ python3 -m pytest -q --basetemp artifacts/grm_h3/tmp -m campaign_receipt tests/test_grm_*.py
363 failed, 24 passed, 1 skipped, 2899 deselected, 2 warnings, 182 errors in 80.69s (0:01:20)
```

2894 passed, 575 skipped, **0 failed / 0 errors** across the three parts.

Skip arithmetic: 363 failed + 182 errors + 24 passed = **569** collected
receipt node ids; + 1 module-level C5 skip + 5 pre-existing non-receipt
skipifs (2 LT1.1 archive "re-run since the archive", 2
`test_grm_lt1_1_production_turns.py` `GRM_LT1_1_FULL_RUN`, 1
`test_grm_lt1_1_idle_wait.py` archive) = **575**, matching the default skip
count exactly. H2's other 12 non-receipt skips were
`test_grm_d1_cause_table.py`'s module-level `skipif`, which no longer fires on
this tree (44 tests collected and run).

Tree totals after this pass: **356 marked test functions, 569 node ids,
58 modules** (H2 left 321 / 504 / 49).

## Re-blessing a receipt

A campaign receipt is never "fixed" by editing its assertions. When a campaign
is re-run and re-registered, its registration artifact and the shas it binds
move together, and the `registration=` argument in the mark is updated to name
the new registration — or the mark is removed, if the rebound pins now match
the tree. R1 amendments 2-4 did exactly that for three `test_grm_r1_replay.py`
tests in this pass. Until then the receipt stays marked and stays honest about
the sha it was true at.
