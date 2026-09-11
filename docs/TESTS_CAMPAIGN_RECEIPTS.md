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

## Gates (merged tree, 2026-09-11)

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

## Re-blessing a receipt

A campaign receipt is never "fixed" by editing its assertions. When a campaign
is re-run and re-registered, its registration artifact and the shas it binds
move together, and the `registration=` argument in the mark is updated to name
the new registration — or the mark is removed, if the rebound pins now match
the tree. R1 amendments 2-4 did exactly that for three `test_grm_r1_replay.py`
tests in this pass. Until then the receipt stays marked and stays honest about
the sha it was true at.
