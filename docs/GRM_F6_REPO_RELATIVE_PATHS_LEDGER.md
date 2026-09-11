# GRM-F6 — dead absolute worktree paths in scripts (ledger)

Seat: Opus 5 (`claude-opus-5[1m]`), effort MAX. Worktree
`/mnt/ForgeRealm/wt/grm-f6` (branch `grm-f6`). `core/` untouched.
Date: 2026-09-11. No GPU. No git run by this seat.

## Finding confirmed

Round-1 seat worktrees were pruned on 2026-09-11. Verified dead on disk:
`grm-c2`, `grm-c3`, `grm-c4`, `grm-c5`, `grm-c7`, `grm-c8`, `grm-a1`,
`grm-d1`, `grm-lt1`, `grm-x1`, `apamq-fa`, `apamq-fb`. Surviving:
`apamq-fa2`, `grm-f1`, `grm-f2`, `grm-f3`, `grm-f5`, `grm-f6`.

The silent shape, reproduced before any edit:

```
$ python3 -c "from scripts import grm_lt1_offline as old; \
    print(old.C2.exists(), len(list(old.C2.glob('cells/*/worker.json'))))"
False 0
```

Zero rows, no exception. `grm_lt1_offline.c2()` contributed 0 rows to the
census; `grm_lt1_offline_supplement.c2_replay()` returned `[]`, and only the
caller's `identical == len(rows) == 132` stood between that and a vacuous
PASS.

`scripts/grm_f5_c2_replay_gate.py` had "fixed" this by rebinding
`old.C2` to `Path('/mnt/ForgeRealm/wt/grm-f5')/...` — the same bug one
worktree later, with `os.chdir` into a seat worktree on top.

## Changes

New: `scripts/grm_repo_paths.py` — `repo_root()` (from `__file__`),
`receipt_root(relative, env, pinned_by, dead_absolute)` (loud
`GRM_F6_DEAD_RECEIPT_PATH`), `require_rows(...)` (loud
`GRM_F6_VACUOUS_ZERO`), `DeadReceiptPath`.

Rebound (class a — data survives in-repo):

| script | was | now | guard |
|---|---|---|---|
| `grm_lt1_offline.py:18` | `wt/grm-c2/artifacts/grm_c2/epochs/scout-fix-2` | `ROOT/artifacts/grm_c2/epochs/scout-fix-2` | `require_rows` on `cells/*/worker.json` |
| `grm_lt1_offline_supplement.py:64` | (used `old.C2`) | (inherits) | `require_rows` on `cells/*/worker.json` |
| `grm_scout_fix8_cpu.py:19` | same C2 epoch | same | `require_rows` on `cells/*/worker.json` |
| `grm_r1_register.py:28` | same C2 epoch | same | `require_rows` on `cells/*/controller.json` |
| `grm_rd1.py:25` | `wt/grm-c7` | `repo_root()` | `require_rows` on `*/probes.jsonl` |
| `grm_d1_cause_table.py:29` | `wt/grm-lt1` | `repo_root()` + rebound `CELLS` | `require_rows` on `{arm}-*/probes.jsonl` |
| `grm_f5_c2_replay_gate.py:17` | `wt/grm-f5` + `os.chdir` | `Path(__file__).parents[1]`, rebind deleted | (inherits both) |
| `tests/test_grm_a1_gpu_contrast.py:1002` | `wt/grm-c7/.../manifest.json` | `ROOT/artifacts/grm_c7/r3/...` | n/a (single read) |

Env overrides: `GRM_C2_EPOCH_ROOT`, `GRM_C7_SOURCE_ROOT`,
`GRM_LT1_CELLS_ROOT` — each names a ROOT, with the relative path appended.

Not rebound, deliberately:

* `scripts/grm_c7_register.py:222` — registration BUILDER. The C3 order
  survives at `orders/GRM_C3_DNGH_DECOY_CALIBRATION.md`, but rebinding would
  change what a future registration hashes and under which key. Registrations
  are receipts. Gated by `test_c7_registration_builder_keeps_its_absolute_c3_pin`.
* `scripts/apamq_fc_ppl.py:47`, `scripts/apamq_fbd1_diag.py:45` — CLASS (b),
  **no surviving copy**. `artifacts/apamq_fc/apa_int4.json` pins
  `engine.compiled_module` to a `.so` inside the pruned `apamq-fa` worktree,
  and `apamq_fc_ppl.py:509` refuses any other root. `apamq-fa2` is a
  different Project-Tensor worktree, not that build; verifying a substitute
  needs a GPU run this seat does not have. Reported, not invented.
  Gated by `test_apamq_int4_leg_has_no_surviving_copy`.
* `scripts/grm_d1_recap.py:166`, `scripts/grm_d1_emit_report.py` (5 lines),
  `tests/test_grm_x1_reduced.py:325`, `tests/test_grm_c5_grounding*.py` —
  CLASS (c): report prose / `receipt=` JSON text / `.replace()` on frozen
  receipt text. Never opened.
* ~300 `@pytest.mark.campaign_receipt(registration=...)` strings across 20
  test modules — CLASS (c). That prose is the receipt staying honest about
  the absolute pins it was true at; scrubbing it would destroy information.

## Reproductions (each vs its frozen receipt)

```
scripts/grm_f5_c2_replay_gate.py          rows 132, off_byte_identical 132, PASS
                                          == artifacts/grm_f5/c2_replay_off.json
grm_scout_fix8_cpu.c2_plans()             132 rows, 132 identical, 132/132 hex
                                          + checkpoint/manifest/source sha parity
                                          == artifacts/grm_scout_fix8/green.json
                                             + c2_identity.json (count 132)
grm_r1_register.measured_c2_cost()        cost_model float-EXACT both sides,
                                          20 restart controllers
                                          == artifacts/grm_r1/registration.json
grm_rd1.census()                          52 probes / 520 requests / 30 memory
                                          / 16 oracle failures, 16 cells,
                                          100/100 input shas identical
                                          == artifacts/grm_rd1/registration.json
grm_rd1 requests                          520/520 identical modulo ONLY the
                                          source-root prefix on `checkpoint`
                                          == artifacts/grm_rd1/requests.json
grm_d1_cause_table                        12 gates now RUN (were skipif-skipped)
```

## Dead-path REDs (planted)

```
GRM_C2_EPOCH_ROOT=<dead> python3 -c 'import scripts.grm_lt1_offline'     -> rc!=0
GRM_C2_EPOCH_ROOT=<dead> python3 -c 'import scripts.grm_scout_fix8_cpu'  -> rc!=0
GRM_C2_EPOCH_ROOT=<dead> python3 -c 'import scripts.grm_r1_register'     -> rc!=0
GRM_C7_SOURCE_ROOT=<dead> python3 -c 'import scripts.grm_rd1'            -> rc!=0
GRM_LT1_CELLS_ROOT=<dead> python3 -c 'import scripts.grm_d1_cause_table' -> rc!=0
```
each with `GRM_F6_DEAD_RECEIPT_PATH` + the dead path + the env var on stderr,
and `c2_replay()` over a dead root raising `GRM_F6_VACUOUS_ZERO`.

## Prior art

* Repo root from `Path(__file__).resolve().parents[n]` rather than the CWD —
  the standard Python idiom (setuptools / pytest `rootdir` discovery,
  pytest-dev, 2009-; `Path.resolve`, PEP 428, Antoine Pitrou, 2012). Taken
  verbatim. Eleven scripts in this repo already did it for CODE
  (`scripts/grm_rd1.py:21`); F6 only extends it to the DATA roots the same
  scripts still pinned absolutely. Nothing here is ours.
* Env-var override with a repo-relative default — twelve-factor config
  (Adam Wiggins, 2011); matches this repo's own `TENSOR_CUDA_ROOT` /
  `GRM_*` convention. Taken.
* Zero rows is an ERROR, not a vacuous pass — the vacuous-truth hazard;
  operationalised as dbt row-count tests (Fishtown Analytics, 2018), Great
  Expectations `expect_table_row_count_to_be_between` (Superconductive,
  2018), pytest exit code 5 for "no tests collected" (pytest-dev, 2019).
  Taken: the principle. Ours: only the wiring into these receipt consumers,
  with the dead path + naming receipt carried in the message.
* AST-walk lint instead of grep, so a comment recording the dead path is
  distinguishable from a live path constant — `ast` (Python 2.6, 2008),
  flake8/pylint checker style (2010-). Taken verbatim.
* Golden / characterization testing — pin the current output, then refactor
  against it (Michael Feathers, *Working Effectively with Legacy Code*,
  2004; approval tests, Llewellyn Falco, 2011). Taken: the shape. The
  receipts and their numbers are the prior campaigns' (GRM contributors,
  2026), reproduced not re-derived.
* No prior art known to me for "pruned seat worktree leaves a
  vacuously-passing receipt gate" as a NAMED failure mode; it is an
  instance of the vacuous-truth hazard above. Unverified against the
  literature — no network in this seat; lead to check. Search terms:
  "vacuous test pass empty glob", "silent zero-row data test",
  "stale absolute path build reproducibility", "hermetic build path
  independence".
