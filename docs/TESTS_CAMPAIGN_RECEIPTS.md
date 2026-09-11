# Campaign-receipt tests (GRM-H1, 2026-09-11)

A **campaign receipt** is a test that is valid only at the sha its campaign
registered. It either

* asserts `INPUT_SHA_MISMATCH`-class binding against `core/` and `scripts/`
  shas that the campaign froze on its own day, or
* reads gitignored campaign artifacts under `artifacts/` (checkpoints,
  receipts, controller state) that a given tree does not carry complete, or
* asserts a **tree state** the registration recorded (e.g. "flag X is absent
  from `core/` and `scripts/`").

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

The skip reason is
`campaign receipt (registration=<path>): sha-bound to its campaign; run with -m campaign_receipt`.

**No test assertion was changed to obtain the marker.** Every marked test still
asserts exactly what its campaign registered; only its default *collection*
changed.

## Marked files

Decided per file by reading it (`INPUT_SHA_MISMATCH`, `registration`,
`artifacts/`) and by reproducing the failure in an isolated process, never by
name. Marks sit on the **test function**, not the module, so healthy coverage
in a mixed module keeps running: of the 28 modules that failed on the merged
tree, 27 are marked and in 20 of them only *some* tests are receipts.

"Tests" counts marked test functions; "Node ids" counts collected test ids
(parametrised cases expand).

| Test module | Tests | Node ids | Class | Registration | Why it is a receipt |
|---|---:|---:|---|---|---|
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
| `tests/test_grm_chat.py` | 1 | 1 | tree-state | `GRM-P1 chat registration (records GRM_ALIAS_FOLD_MERGE as ABSENT)` | asserts a TREE STATE (a named flag absent from core/scripts) that the registration recorded |
| `tests/test_grm_d1_registration.py` | 1 | 1 | tree-state | `artifacts/grm_d1/lt1_1/registration.json (records GRM_ALIAS_FOLD_MERGE as ABSENT)` | asserts a TREE STATE (a named flag absent from core/scripts) that the registration recorded |
| `tests/test_grm_lt1.py` | 3 | 4 | sha-bound | `artifacts/grm_lt1/registration.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_lt1_amendment1.py` | 3 | 4 | sha-bound | `artifacts/grm_lt1/amendment1/registration_amendment.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_lt1_amendment3.py` | 3 | 3 | sha-bound | `artifacts/grm_lt1/amendment3/registration_amendment.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_lt1_amendment4.py` | 9 | 14 | artifact-bound | `artifacts/grm_lt1/amendment2/run_margin_first/cells/*/checkpoint` | reads gitignored campaign artifacts under artifacts/ that this tree does not carry complete |
| `tests/test_grm_lt1_controller.py` | 2 | 2 | sha-bound | `artifacts/grm_lt1/registration.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_lt1_fix6.py` | 3 | 4 | sha-bound | `artifacts/grm_lt1/ (FIX-6 amendment)` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_r1_amendment1.py` | 11 | 11 | sha-bound | `artifacts/grm_r1/amendment_1.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_r1_cpu_gate.py` | 1 | 1 | sha-bound | `artifacts/grm_r1/registration.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_r1_replay.py` | 18 | 18 | sha-bound | `artifacts/grm_r1/registration.json + orders/GRM_R1_MARGIN_FIRST_REPLAY.md` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_scout_fix4_continuation.py` | 3 | 10 | sha-bound | `artifacts/grm_scout_fix4/registration.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_scout_fix5_runner.py` | 4 | 16 | artifact-bound | `artifacts/grm_scout_fix5/registration.json + artifacts/grm_c7/r2/cells/*/checkpoint` | reads gitignored campaign artifacts under artifacts/ that this tree does not carry complete |
| `tests/test_grm_scout_fix8_replay.py` | 5 | 5 | sha-bound | `artifacts/grm_scout_fix8/registration.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_scout_fix8_resume.py` | 2 | 3 | sha-bound | `artifacts/grm_scout_fix8/resume_amendment_1/registration.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| **27 modules** | **134** | **199** | | | |

### One module that is NOT marked

`tests/test_grm_native_runtime.py::test_gqa_cuda_epoch_bump_covers_graft_repository_mutation_battery`
also failed on the merged tree but is **not** a campaign receipt — it is a
genuine merge-drift RED, and marking it would have hidden a real regression.
The A1 alias-fold merge added `self.alias_fold_merge` to
`GraftRepository.__init__` and a read of it in `correct_memory()`; the test's
fixture builds the repository via `GraftRepository.__new__`, bypassing
`__init__`, so the battery died with `AttributeError` before exercising the
epoch contract it exists to prove. Fixed in the fixture
(`_fake_gqa_repo_for_epoch_battery`) by listing the three attributes the merge
added, at `__init__`'s defaults with the A1 flag OFF. No assertion changed.

## Environment isolation

`tests/conftest.py` also snapshots every `GRM_*` environment variable at setup,
restores it exactly (deleting keys that did not exist), and **fails the test
that leaked**, naming the key. See the module docstring there for the
mechanism, and `tests/test_grm_h1_env_isolation.py` for the proof.

One real leaker was found across the whole GRM suite:

```
tests/test_grm_r1_replay.py::test_parity_barrier_stops_the_run
  GRM_ADMISSION_RULE: unset -> 'margin_first' (set and not cleaned up)
```

`scripts/grm_r1_replay.pin_rule()` writes `os.environ['GRM_ADMISSION_RULE']`
directly (by design — `environment(flags)` strips every `GRM_` var, so the
rule must be pinned after it), and the `R1_OFF_PLAN_PARITY_RED_STOP` the test
asserts escapes before any caller cleanup. Every later module in the same
pytest process was then re-ruled `margin_first`.

Receipt, before the guard:

```
$ python3 -m pytest -q tests/test_grm_r1_replay.py tests/test_grm_scout_fix4.py \
      tests/test_grm_a1_alias_fold.py tests/test_grm_admission.py
39 failed, 132 passed, 2 warnings in 8.59s
```

with all three downstream modules green in their own process (7, 113 and 11
passed). After the guard, every failure is confined to `test_grm_r1_replay.py`
itself; with the receipts also marked the same command is:

```
$ python3 -m pytest -q tests/test_grm_r1_*.py tests/test_grm_scout_fix4.py \
      tests/test_grm_a1_alias_fold.py tests/test_grm_admission.py
164 passed, 30 skipped, 2 warnings in 8.50s
```

## Gates (this tree, 2026-09-11)

```
$ python3 -m pytest -q tests/test_grm_*.py
2097 passed, 199 skipped, 2 warnings in 542.53s (0:09:02)

$ python3 -m pytest -q -m campaign_receipt tests/test_grm_*.py
134 failed, 2 passed, 2097 deselected, 2 warnings, 63 errors in 58.05s
```

The 2 that pass under `-m` are two parametrised cases of
`test_grm_c2_epoch3.py::test_stale_epoch_input_refused`, whose other cases
fail; the mark is on the function, so all of its cases travel together.

## Re-blessing a receipt

A campaign receipt is never "fixed" by editing its assertions. When a campaign
is re-run and re-registered, its registration artifact and the shas it binds
move together, and the `registration=` argument in the mark is updated to name
the new registration. Until then the receipt stays marked and stays honest
about the sha it was true at.
