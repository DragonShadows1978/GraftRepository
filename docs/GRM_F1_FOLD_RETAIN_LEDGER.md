# GRM-F1 — source retention after folds: LEDGER

Order: `orders/GRM_F1_FOLD_RETAIN_SOURCES.md` (lead, round 2, 2026-09-11).
Branch `grm-f1`, forked from `lc1-wip` 8edfae4. Seat: Opus 5 (opus-max),
reasoning effort MAX. CPU only — no GPU was used or available to this seat.

The receipts are the commands and outputs below, in the order they happened,
including the two things that went wrong and what was done about them.

---

## 1. The defect, read from the r2 receipts (not inferred)

Read-only inputs: `/mnt/ForgeRealm/GraftRepository/artifacts/grm_d1/lt1_1/`
(`run_A`, `run_Aplus`, `miss_causes.json`, `lead_A_r2_summary.json`).

r2 arm A measured: fresh 10/15, corrections 9/10, aliases 10/10, recap 4/5
(`lead_A_r2_summary.json`). The lifecycle fact underneath:

    core/graft_arena.py::_deposit_consolidation   grafts[i]["retired"] = True
    core/graft_arena.py::_route_cand_base         excludes every retired node

so after a fold clears `MIN_FOLD_KEEP` (0.70), the digest's prose is the ONLY
routable record of the facts its sources carried.

Measured from the r2 arm-A end-of-campaign checkpoint manifest
(`run_A/cells/A-197-200/checkpoint/repository/manifest.json`):

| quantity | value |
|---|---|
| nodes total | 199 |
| active (routable) | 36 |
| inactive | 163 |
| — retired by a FOLD alone | **148** (124 turns + 24 digests) |
| — retired by a CORRECTION | 15 |
| retained-source `ntok` | mean 52.8, median 53, range 20–73 |

Residency across all 26 arm-A cells: 323 rows, max `summed_token_seats` 94
against `width` 96 — the arena was ALREADY near saturation in r2.

---

## 2. Core change (flag `GRM_FOLD_RETAIN_SOURCES`, default OFF)

`core/grm_fold_retain.py` (NEW, 199 lines). Flag resolver
(explicit > env > permanent OFF; unknown tokens fail CLOSED, the A1 / L2 /
A-DEC precedent), lineage vocabulary, and `retained_source_indices` for the
residency split. No serving path calls anything here with the flag OFF.

`core/graft_arena.py`
* `+from core import grm_fold_retain` (line 47).
* `__init__` gains `fold_retain_sources=None`; resolved ONCE at construction
  (line ~87) so a mid-session env change cannot alter a conversation's fold
  lifecycle halfway through — the freeze A1/L2/A-DEC take.
* `_deposit_consolidation(idxs, text, prefix=..., retain=None)`: `retain`
  defaults to `self.fold_retain_sources`. OFF keeps `grafts[i]["retired"] =
  True` verbatim. ON leaves `retired` unset, sets `no_fold`, and writes
  `metadata.digest_of` on each source / `metadata.retained_sources` on the
  digest.
* `consolidate(idxs, ngen=None, retain=None)`: pass-through only.

`core/graft_repository.py`
* `+from core import grm_fold_retain as _fold_retain`.
* `_fold_once`: records `retained_sources` + `fold_lineage` on the fold
  event; completes the retained lineage into the persisted metadata schema
  **only when sources were actually retained** (see §5, defect 2).
* `_alias_consolidate`: all three `arena.consolidate(...)` calls PIN
  `retain=False`. GRM-A1 always supersedes its alias EDGE — a bare active
  edge that wins admission and answers nothing IS the RD2 defect A1 exists
  to fix — and a window-fold retention policy must not reinstate it.

`scripts/grm_c7_run.py::seats` — residency rows gain
`retained_source_mounted_ids` and `retained_source_token_seats`. Additive:
with the flag OFF no node carries `digest_of`, so both are empty/0 and every
pre-F1 field keeps its bytes.

`scripts/grm_lt1_1.py::arm_environment` — carries `GRM_FOLD_RETAIN_SOURCES`
across `grm_c2_cells.environment()`'s `GRM_*` strip, the rail the arm pin
already rides. See §5, defect 3.

---

## 3. OFF byte-identity

The registered C2 132-plan replay gate
(`tests/test_grm_scout_fix9.py::test_c2_recorded_plans_byte_identical`)
CANNOT RUN on this machine: its source campaign
`/mnt/ForgeRealm/wt/grm-c2/artifacts/grm_c2/epochs/scout-fix-2` was pruned
with the grm-c2 fork, so `c2_plans()` collects 0 rows and asserts at
`scripts/grm_scout_fix8_cpu.py:99`. Verified PRE-EXISTING: it fails
identically with baseline `core/graft_arena.py` + `core/graft_repository.py`
copied in from the canonical repo. This is itself the round-1 absolute-pin
lesson, already landed as a permanent receipt.

Substitute proof, built for this reason:
`artifacts/grm_f1/off_identity/`. A full fold lifecycle on the CPU double —
8 turns, two window folds and one era fold through `repo._fold_once`, then a
`correct_memory` — fingerprinted as canonical JSON over the complete node
table, the routing candidate base, the correction's lineage, the fold return
values and `fold_history`. Run once with BASELINE core, once with F1 core
and the flag unset:

| section | result |
|---|---|
| `nodes` (28 fields × every graft) | IDENTICAL `aed37af295559e72` |
| `route_base` | IDENTICAL `95bce61a71a78185` |
| `correction` (supersedes lineage) | IDENTICAL `913b8117c2b57f74` |
| `folds` (return values) | IDENTICAL `6fb086d658332a43` |
| `fold_history` minus additive fields | IDENTICAL `7aacff00652464ad` |

The ONLY difference between the two dumps is the two additive
`fold_history` receipt fields (`retained_sources`, `fold_lineage`), whose
OFF values are constant `[]` / `digest_supersedes_sources`. They are a
record of which lifecycle ran; no serving path reads them.

Suites green on F1 core with the flag unset: FIX-3, FIX-5, FIX-8, FIX-9
(105 passed, 1 skipped), A1 (`tests/test_grm_a1_alias_fold.py`), LT1.1
child-spawn / preflight / runner / production-turns.

---

## 4. CPU measurement through the REAL LT1.1 worker

`python3 -m scripts.grm_lt1_1 --arm A --fake --limit 4` — the production
`--fake` path (one subprocess per cell, real `worker.execute`, real ladder,
real fold job), turns 1–32, OFF then ON. Receipts:
`artifacts/grm_f1/cpu/{off,on}/`, summary
`artifacts/grm_f1/cpu/measurement.json`.

| quantity (end of cell A-025-032) | OFF | ON |
|---|---|---|
| total nodes | 28 | 28 |
| **ACTIVE / routable nodes** | **12** | **24** (+12) |
| nodes carrying `metadata.digest_of` | 0 | 12 |
| digests carrying `retained_sources` | 0 | 3 |
| residency rows | 48 | 52 |
| **max `summed_token_seats`** (width 96) | **86** | **94** |
| residency rows with a retained source mounted | 0 | 12 |
| **probes mounting a SOURCE node** | **4 / 7** | **6 / 7** |
| **probes mounting a RETAINED source** | **0 / 7** | **2 / 7** |
| exact-correct | 5 / 7 | 5 / 7 |

The mission's question, answered: **retained sources are routable and are
actually mounted after a fold.** In both ON hits (`recall_6_10`,
`recall_7_10`) the retained source was the SOLE mounted node — under OFF
those probes had no source to reach at all. Residency stayed bounded (94 ≤
96, no `RESIDENCY_BOUND_EXCEEDED` on any row).

HONEST: exact-correct did NOT move (5/7 both arms). The CPU double's reader
is a regex, not a language model; this run is evidence about ROUTING
TOPOLOGY and receipt plumbing, and is explicitly NOT a model-quality
measurement. Whether the newly reachable sources change ANSWERS is the GPU
question r3 registers.

---

## 5. What went wrong (three defects, all mine, all fixed)

**Defect 1 — the sha pin blocked the child (expected, by design).**
First run of `tests/test_grm_lt1_1_child_spawn.py` after the core edit:
`LT11_CHILD_PREFLIGHT_BLOCKED: INPUT_SHA_MISMATCH: core/graft_arena.py,
core/graft_repository.py`. Not a behavioural regression — LT1.1's preflight
is sha-bound and F1 moved two pins. Resolved by amendment 10 (§6), after
confirming the gate passes on baseline core.

**Defect 2 — an OFF-arm behavioural break, caught by the §3 fingerprint.**
An earlier draft called `_ensure_lifecycle(didx)` + `_mark_dirty(didx)`
unconditionally in `_fold_once`. That normalizes `metadata.active` on a
digest the pre-F1 path left with NO metadata, and `correct_memory` selects
targets with `meta.get("active", True)` — so an OFF-arm correction began
superseding an extra already-retired digest: `supersedes [0, 10]` became
`[0, 8, 10]`. The lifecycle completion is now gated strictly on `retained`.
Pinned by `test_off_fold_once_leaves_digest_metadata_untouched` and
`test_off_correction_targets_only_active_nodes`. **This is the one that
would have silently corrupted the OFF arm**, and the only reason it was
caught is that the fingerprint compared against baseline core rather than
against my own expectations.

**Defect 3 — the flag would have been stripped from the r3 campaign.**
`grm_c2_cells.environment()` deletes every ambient `GRM_*` name. Without a
re-apply, an r3 campaign launched with `GRM_FOLD_RETAIN_SOURCES=1` exported
would have run the OFF arm and REPORTED it as ON — the exact failure
`arm_environment`'s own docstring already warns about for the arm pin. Fixed
there, carried only when actually set (absent stays absent).

**Not mine, surfaced:** `tests/test_grm_lt1_1_preflight.py::
test_a_planted_drift_in_a_new_input_goes_red` planted its drift into
`new_inputs` but popped only `changed`. `governing_core_pins` applies
`unchanged` LAST, so once amendment 10 filed `core/grm_alias_fold.py` under
`unchanged`, the surviving entry overwrote the planted sha and the teeth
test passed vacuously. The plant now clears all three sections.

---

## 6. Amendment 10 + the r3 registration

`scripts/grm_f1_register_lt11_r3.py`. Writes
`artifacts/grm_d1/lt1_1/amendment10.json` (the CHAIN copy —
`grm_lt1_1.governing_core_rebind` globs exactly that directory, so an
amendment written anywhere else is invisible to the preflight) plus a
self-describing copy and the registration under
`artifacts/grm_f1/lt1_1_r3/`.

APPEND ONLY: amendments 1–9, the r2 registration, the fixture,
`miss_causes.json` and both run directories are byte-for-byte untouched
(verified by md5 before/after). The script REFUSES to overwrite an
amendment 10 written by a different order.

Core rebind: 2 changed, 1 new, 4 unchanged. Every sha is MEASURED from the
tree, never typed; the script raises `F1_UNEXPECTED_CORE_DRIFT` if a file F1
does not touch has moved, `F1_WORKER_DRIFT` if the LT1.1 worker moved, and
`F1_RUNNER_CHANGE_ABSENT` if the runner rebind would claim a change the file
does not carry.

After the rebind: child preflight `READY`, 8 inputs checked, 26 cells;
`tests/test_grm_lt1_1_child_spawn.py` 12 passed, 1 skipped.

**Budget conflict, registered rather than smoothed over.** The order caps r3
at ≤ 1.3 GPU-h. r2's arm-A cell schedule — reused verbatim so r3 stays a
one-variable comparison — reserves 7410 s (2.06 GPU-h) of LEASE. These are
different quantities: the ESTIMATE is 4507.9 s = 1.25 GPU-h and IS under the
cap. Registering `budget_gpu_seconds=4680` would put the budget BELOW the
reservation sum, and LT1.1 amendment 2 exists precisely because that was
already done once (it corrected 6120 → 7410 because the lower figure "sat
below the reservation sum and would have tripped `run_cell`'s rail
mid-campaign"). Resolution: `budget_gpu_seconds` stays at r2's 7410
(reservation ceiling, rail behaves identically); `order_cap_gpu_seconds =
4680` is registered as a SEPARATE binding limit on MEASURED charged GPU
seconds. **Lead decision requested** — the alternative is re-cutting the
arm-A cell schedule, which would break cell-for-cell comparability with the
r2 A baseline and was therefore not done unilaterally.

`artifacts/grm_f1/lead_commands.txt`: every runnable command ends in
`--dry-run`, enforced by `test_lead_commands_dry_run_gated` on the LOGICAL
command (backslash continuations joined first, so a `flock ... \` wrapper
cannot pass trivially while the command it wraps goes unchecked).

---

## 7. Known gap left open (registered, not discovered later)

`correct_memory` matches the query against node TEXT. A fold digest that
PARAPHRASES a fact rather than copying its wording does not match, so the
stale copy inside it is never superseded. Measured on the CPU double in BOTH
arms with a multi-word entity ("Iona Vale") the extractive digest emits as
separate tokens:

* OFF: `supersedes == []` — the correction retires NOTHING. The sources were
  already fold-retired and the digest does not match.
* ON: `supersedes == [source]` — the retained source IS caught; the digest
  still escapes.

So F1 neither causes nor fixes this: under the flag strictly MORE of the
stale record is retired. Owner: **F2** (A1 already carries the entity-scoped
remedy for its own merged digests via
`_alias_extend_correction_targets`; generalising it to ordinary fold digests
changes correctness semantics the F1 order does not authorise). Registered
as `known_gaps` F1-N1 and as the reason the corrections column is predicted
at ≥ 9/10 rather than 10/10.

---

## 8. Prior art

Reused UNCHANGED, all LOCAL and verified at the code site:
* **FIX-5** source-enumerated consolidation, `_fact_set` / `_coverage` and
  `MIN_FOLD_KEEP` = 0.70 (`core/graft_arena.py`). F1 changes only what
  happens to the sources AFTER the digest is accepted.
* **LSR-P2C / SCOUT-FIX-9** width guard, whose rejected-split path ALREADY
  leaves sources ACTIVE and uses `no_fold`. Taken: the precedent that an
  active source alongside a derived node is a supported repository state,
  and `no_fold` itself as the anti-reselect rail — reused, not reinvented.
* **A1 lineage** (`core/grm_alias_fold.py`): the
  `supersedes`/`superseded_by`/`metadata.active` triple and the
  "which of the two happened is RECORDED" discipline. `digest_of` /
  `retained_sources` are the retention-side names for the same idea.
* **D1** registration shape (`scripts/grm_d1_register_lt11.py`) and
  amendment 8's `core_rebind` block, reused field for field.

External, NOT verified in this sandbox (no network) — lead to check:
* **Sarthi et al. (2024), RAPTOR** (arXiv 2401.18059). Its "collapsed tree"
  retrieval searches summary nodes AND original leaves together — that is
  exactly the ON-state topology; its default traversal searches summaries
  only — the OFF state. TAKEN: the framing that leaf retention vs
  substitution is a RETRIEVAL-TOPOLOGY choice with measurable consequences.
  NOT taken: RAPTOR's clustering (GMM/UMAP), tree construction, retrieval
  scoring, or any claim that collapsed-tree wins. Search terms:
  `RAPTOR collapsed tree retrieval`,
  `recursive abstractive summarization retrieval leaves`.
* **Rosenblum & Ousterhout (1992) LFS / Gray & Reuter (1993)** — compaction
  that writes a merged record and frees the inputs on a SEPARATE policy.
  Nothing mechanical taken; named because I am re-deriving a standard
  framing, not inventing it. Search terms:
  `log-structured merge compaction retain inputs`,
  `LSM tombstone vs live key retention`.
* **Haber & Stornetta (1991)** — append-only "latest record wins" chains, in
  kind only, for the amendment chain. Search terms:
  `Haber Stornetta 1991 timestamping digital document`.
* Memory-consolidation literature on gist-plus-trace coexistence: **no prior
  art known to me** that I can name precisely enough to cite. Search terms:
  `fuzzy-trace theory gist verbatim dual storage`,
  `memory consolidation does not erase the trace`.

No prior art known to me for this exact composition: a coverage-gated fold
whose source retirement is an independently flagged policy, with a `no_fold`
anti-reselect rail and split residency accounting.

---

## 9. Process safety

GPU: none used; this seat had no GPU. No lease taken, no flock held.
Processes: none killed or signalled; nothing this seat did not start was
touched. Git: never run — the lead commits. Subagents: none spawned.
Symlinks: none created. Scratch: every pytest run used
`--basetemp /mnt/ForgeRealm/wt/grm-f1/artifacts/grm_f1/tmp`, removed
afterwards. Read-only inputs were read only; `/mnt/ForgeRealm/GraftRepository`
was never written. Every Bash call ran in the foreground under 10 minutes.

---

# FOLLOW-UP (lead, 2026-09-11): merge resolution + three-arm r3

F1 was verified (249 passed) and committed at d45dee3. Merging `grm-f1` onto
`lc1-wip` conflicted with F2, so the lead merged `lc1-wip` INTO this worktree
and left one conflict for this seat. Now also on the tree: F2
(`core/grm_fold_alias_guard.py`), F5 (`GRM_ROUTE_SOLE_BINDER_INSURANCE` in
`core/grm_admission.py`), F3's scorer secondary, H2 marks.

## 10. The conflict, and what it actually was

ONE hunk, `core/graft_repository.py:54-59` — **an import conflict, not a
`_fold_once` conflict**:

    <<<<<<< HEAD
    from core import grm_fold_retain as _fold_retain
    =======
    from core import grm_fold_alias_guard as _fold_guard
    from core.grm_alias_guard import fold_alias_guard_enabled
    >>>>>>> lc1-wip

Resolved as a union (3 lines, `:54-56`). Markers removed; `grep -c
'^<<<<<<<\|^=======\|^>>>>>>>'` returns 0.

Before resolving I verified the composition the lead asked for was
STRUCTURALLY present rather than assuming it:

* F2's `_guard_fold_windows` is called on **both** `_librarian_jobs` return
  paths (`:4113-4117` native plan, `:4128` fallback) — window exclusion runs
  BEFORE job selection.
* F2's `_fold_attribution_ok` returns `(None, None)` on **both** fold paths
  (`core/graft_arena.py:2301` era, `:2417` generated) — always BEFORE
  `_deposit_consolidation`, so an attribution rejection can never leave F1
  lineage on a node.
* F1's `_fold_once` retain lifecycle and A1's `retain=False` pins survived
  the merge intact.

Two annotations added so the composition is stated where it is relied on,
not inferred:
* `_fold_once` records F2's attribution receipt on the fold event
  (`fold_event["attribution"]`), so a reader can tell a COVERAGE abort from
  an ATTRIBUTION abort without correlating two logs. `None` whenever no hook
  is installed, so OFF events keep their values.
* The shared `didx is None` abort branch now states the F1×F2 contract
  explicitly: one branch absorbs both abort classes and returns before any
  F1 lineage is written.
* The F1 lineage block states that `idxs` is already the guarded window, so
  F1 applies to exactly what F2 let through — no second exclusion pass, no
  coupling between the flags.

## 11. Gates on the merged core

    tests/test_grm_f1*.py tests/test_grm_f2*.py tests/test_grm_f5*.py
    tests/test_grm_scout_fix5*.py tests/test_grm_scout_fix9*.py
    tests/test_grm_a1_alias_fold.py tests/test_grm_s4_fold_order.py
    tests/test_grm_lt1_1_production_turns.py

**OFF byte-identity re-run on the merged core** (all three flags unset),
`artifacts/grm_f1/off_identity/merged_core_all_flags_off.json`:

| section | result |
|---|---|
| `nodes` | IDENTICAL `aed37af295559e72` |
| `route_base` | IDENTICAL `95bce61a71a78185` |
| `correction` | IDENTICAL `913b8117c2b57f74` |
| `folds` | IDENTICAL `6fb086d658332a43` |
| `fold_history` minus additive fields | IDENTICAL `7aacff00652464ad` |

The merged-core all-OFF fingerprint hashes to `a1141e67f965419d…` — **the
exact hash the pre-merge F1 core produced with its flag off**. F2 and F5
contribute nothing to node state, routing or supersession when OFF.

Two test adjustments, both because a pin generalised rather than because a
behaviour changed:
* `test_amendment10_rebinds_every_f1_core_pin` →
  `test_governing_amendment_rebinds_every_core_pin`, written against
  `governing_core_rebind()` instead of a hard-coded amendment number (10 was
  F1's, 11 rebinds the merged tree, a later treatment will add another).
* `test_lead_commands_dry_run_gated` now exempts `unset` lines as
  environment lines alongside `export`. A0 must run with
  `GRM_ALIAS_FOLD_MERGE` ABSENT, not `=0`, to exercise the unset branch.

**One restored receipt, not a code change.** `tests/test_grm_f2_alias_guard
.py::test_control_flag_contract_is_registered` asserts every repo-relative
pin in `artifacts/grm_f2/flag_contract.json` exists; `artifacts/grm_d1/lt1_1/
run_A/cells/A-025-032/session/repository/manifest.json` was absent from this
worktree (`artifacts/` is gitignored, and only a partial r2 copy had been
made here). Restored as a byte-identical copy (48 MB, `diff -r` clean) from
the read-only canonical repo. Nothing in the canonical repo was written.

## 12. Amendment 11 — three arms

`scripts/grm_f1_register_lt11_r3_merged.py`. Chains to amendment 10,
supersedes its `core_rebind`, and SUPERSEDES its single-arm registration
(schema `grm.lt1_1_r3.registration.v2`).

Core rebind: **3 changed, 1 new, 4 unchanged**, every sha measured from the
tree, each with a per-file attribution naming the mechanism at the code site:

| file | before → after | attribution |
|---|---|---|
| `core/graft_arena.py` | `7f86633daca7` → `3c97a3ea3afc` | F1 + F2 |
| `core/graft_repository.py` | `c3aa3c92a203` → `758c62dc49ed` | F1 + F2, merge resolved here |
| `core/grm_admission.py` | `ebdfd84af435` → `7912f29108f0` | F5 |
| `core/grm_fold_alias_guard.py` | NEW `45b5640cb6e2` | F2 |

The script raises `F1_UNEXPECTED_CORE_DRIFT` on any pin that moved without a
registered attribution, `F1_RUNNER_DRIFT` if the runner or worker moved, and
`F1_AMENDMENT11_FOREIGN` rather than overwriting another order's receipt.
Amendments 1–10 and every other r2 receipt are untouched.

**Three separately resumable arms**, same frozen conversation, same fixture
sha, same 26-cell schedule, same worker:

| arm | role | A1 | F1 | F2 | F5 | out_dir |
|---|---|---|---|---|---|---|
| `A0` | CONTROL | off | off | off | off | `…/run_A0` |
| `A'` | treatment | off | **on** | **on** | **on** | `…/run_Aprime` |
| `A+'` | treatment | **on** | **on** | **on** | **on** | `…/run_Aplusprime` |

**A0 is what makes the treatment arms readable.** Registered explicitly: if
A0 does not reproduce r2 arm A within ±1 per class, core drift is in play and
NO movement in A′ or A+′ may be attributed to F1/F2/F5 until that is
explained. Within an arm three (or four) flags move TOGETHER — a moved column
belongs to the SET, never to one flag. Both caveats are in the registration
as `attribution_rule` and per-arm `attribution_caveat`, not left to the
reader.

Predictions registered per arm, before the run: A0 within ±1 of r2 arm A on
every class; A′ fresh ≥14/15, corrections ≥9/10 (no-regression, because the
paraphrasing-digest gap stays open), aliases 10/10 (must not move at all),
recap ≥4/5, residency bounded with max seats reported; A+′ the same plus
aliases 10/10 under A1. Falsifiers named for each.

Budget: the lead's decision recorded — each arm reserves r2's 7410 s
per-arm lease sum, with the 4680 s (1.30 GPU-h) order cap binding on
MEASURED charged work. `within_budget` and `within_order_cap` both True for
all three arms.

## 13. Follow-up process safety

No GPU. No process killed or signalled. Git never run. No subagents. No
symlinks. Core edits confined to the conflict hunk and its three composition
annotations, all under the existing flags. `/mnt/ForgeRealm/GraftRepository`
read only — the one copy taken FROM it was verified byte-identical and
written only into this worktree. Every pytest run used
`--basetemp /mnt/ForgeRealm/wt/grm-f1/artifacts/grm_f1/tmp`, removed after.
All Bash calls foreground, under 10 minutes.

---

# FOLLOW-UP 3 (lead, 2026-09-11): `--out` must govern every root

Follow-up 2 was verified (307 passed) and committed at 23d6f0d. The lead's
arm-A' run on the card then died before taking any lease
(`artifacts/grm_f1/lt1_1_r3/lead_Aprime_try1.log`):

    scripts/grm_lt1_1.py:1206  resume -> worker.pending(reg, run=root)
    scripts/grm_lt1_worker.py:240  ValueError: COMPLETED_CELL_BINDING_OR_STATUS

`run_Aprime/cells` was empty — the tell that the campaign never wrote there.

## 14. The cause: NEITHER candidate the lead offered

Settled by receipt, `artifacts/grm_f1/red/`.

`main()` read `return resume(args.arm)`. **`--out` was accepted by argparse
and DROPPED** on the one route that spends GPU. Every other mode
(`--dry-run`, `--summary`, `--fake`, `--dry-lease`) forwarded `args.out`, so
the flag looked honoured everywhere a gate could see it and was discarded
exactly where it mattered. `resume` fell back to `out_dir('A')` = r2's frozen
`artifacts/grm_d1/lt1_1/run_A`, whose 26 COMPLETE cells carry r2's binding
(amendments 10/11 rebound the core), and `pending()` raised.

Reproduced before touching anything
(`artifacts/grm_f1/red/red_resume_drops_out.log`): same error, same line,
and the `--out` root **was never created** — proving the argument was
dropped, not merely disagreeing with a registration.

**Candidate (1) REFUTED by receipt.** `pending()` fails on the FIRST cell,
`A-001-008`, and never reaches `A-025-032`; and its body reads only
`controller.json` / `worker.json` / `checkpoint/` — the string `session`
does not occur in it. The restored directory was irrelevant. Pinned by
`test_red_is_not_caused_by_the_restored_session_directory`.

**Candidate (2) does not apply.** There is no `run_root` binding in the
registration to disagree with; `resume` derives the root from its argument
alone.

Why my own `--dry-lease` gate passed: it is a **different `main()` branch**
that already forwarded `root=args.out`. It exercised the fixed path while the
broken one shipped. That is the reporting failure to name — a gate that
tests the sibling of the code under test is not a gate.

## 15. Two more root leaks, found while fixing the first

Neither would have surfaced today, because the crash happened first.

**Leak 2 — select here, execute there.** `lt1_1_seams` pinned
`worker.RUN = out_dir(arm)` unconditionally. `run_cell`, `accounting` and
the leased child ALL read `worker.RUN`. With `--out` honoured only in
`pending()`, a campaign would have SELECTED a cell from the `--out` root and
EXECUTED it into the arm default.

**Leak 3 — the silently wrong arm.** `run_cell` builds the child environment
from `grm_c2_cells.environment(flags)`, which deletes every ambient `GRM_*`
name; `spawn_env` re-applied only the arm pin. Measured
(`artifacts/grm_f1/red/red_child_env_strip.json`): with all three treatment
flags exported in the parent, **every one was ABSENT from the child** — the
process that actually runs the turns, folds the windows and routes the
probes. An r3 treatment arm would have **executed as the control while its
receipts recorded the flags as on**. Silently wrong, not failed. My
follow-up-2 `arm_environment` fix covered the in-process `--fake` path only;
it did not cross the process boundary, and I did not check that it did.

## 16. The fix (runner only; core untouched)

`scripts/grm_lt1_1.py`:
* `main()` — `return resume(args.arm, root=args.out, host_gate=not args.no_host_gate)`;
  `--out`'s help now states that it governs every root.
* `resolved_roots(arm, root)` — campaign root, cells root, owner file,
  `worker_RUN`, the arm default, and `arm_default_in_use`. Printed into the
  `--dry-lease` receipt AND the operator log (`LT1.1 roots {...}`), because
  the GPU route returns an exit code, not a document.
* `assert_root_isolation(arm, root)` — raises `LT11_OUT_EQUALS_ARM_DEFAULT`,
  `LT11_OUT_INSIDE_ARM_DEFAULT` or `LT11_WORKER_RUN_NOT_ISOLATED`. With
  `--out` given the arm default is never read, and that is now
  unrepresentable rather than merely intended.
* `lt1_1_seams(arm, root=None)` — pins `worker.RUN` to the campaign root.
  Default unchanged, so every pre-existing caller is byte-identical.
* `TREATMENT_FLAGS` + carries in `spawn_env` (cross-process) and
  `arm_environment` (in-process). Carried only when actually set, so a
  caller with none set gets a byte-identical child environment.

## 17. F2's pin moved off the live run root

`artifacts/grm_f2/flag_contract.json` pinned
`artifacts/grm_d1/lt1_1/run_A/cells/A-025-032/session/repository/manifest.json`
— a path INSIDE a live campaign run root, which is what made me restore that
directory in follow-up 2 in the first place. Repointed to
`artifacts/grm_f2/r2_arm_a_A-025-032_session_manifest.json`, a byte-identical
copy (`cmp -s` clean), with a `pin_relocations` entry recording from/to/sha
and why. The restored `session/` subtree was then removed from the run root
after verifying the canonical repo still holds it byte-identically
(`diff -r` clean); the four original r2 receipt files in that cell are
untouched, and no `session/` directories remain anywhere under
`run_A/cells/`. `test_f2_flag_contract_pin_is_off_the_live_run_root` refuses
any pin containing `/run_A` or `/cells/`.

## 18. Gate: the REAL resume route

`tests/test_grm_f1_resume_out_root.py` (383 lines, 20 tests). Drives
`resume()` itself — not the `--dry-lease` branch — through the chain
preflight, seams, arm pin, campaign-owner file, `worker.pending` selection
and ONE real `worker.run_cell` spawn onto the CPU double, for all three
arms. Only the GPU lease and the idle probe are stubbed (the shape reused
verbatim from `tests/test_grm_lt1_1_child_spawn.py`), so reservation
accounting, directory creation, argv construction, Popen, the foreground
wait, charge computation, checkpoint validation and the controller receipt
all run unchanged.

RED-before is pinned in the same file, not merely described:
`test_red_resume_without_out_hits_the_frozen_r2_root` reproduces today's
`COMPLETED_CELL_BINDING_OR_STATUS`, and
`test_main_forwards_out_to_the_real_resume` reads the source so a
re-introduced no-argument call site fails again.

Result: all three arms COMPLETE, `status=COMPLETE error=None`, correct
`campaign_arm` / `alias_fold_merge` per arm, clean child logs, everything
written under `--out` and nothing added to the frozen r2 root.

Worth recording: this gate FAILED first with
`LT11_CHILD_PREFLIGHT_BLOCKED: RUNNER_SHA_MISMATCH` — the sha pin catching
my own runner edit, exactly as designed, before amendment 12 existed.

## 19. Amendment 12

`scripts/grm_f1_register_lt11_r3_amd12.py`. Chains to 11, rebinds the RUNNER
`c9874c41f46e -> 3725dc3fe2a3`, and carries amendment 11's core_rebind
forward VERBATIM — **no core file moves in follow-up 3**, and the script
raises `F1_UNEXPECTED_CORE_DRIFT` if one did (all 8 pins re-measured, all
match). It also raises `F1_WORKER_DRIFT`, `F1_RUNNER_CHANGE_ABSENT` if the
claimed markers are missing, and `F1_RUNNER_DEFECT_PRESENT` if the
argument-dropping call is still in the file. Records the root-isolation
contract, the treatment-flag carry (with the measured before/after), and the
F2 pin relocation.

Amendment 11 is FROZEN: its own generator now re-reads the emitted document
rather than regenerating it, because re-emitting against a moved runner
would rewrite a receipt to say something it did not say on its day. Verified
byte-identical by md5 across the refresh. The three-arm registration stands
unchanged; only the runner sha moves.

Governing amendment: **12**. 8/8 core pins resolve. Preflight READY for both
base arms, 9 inputs checked.

## 20. Follow-up 3 process safety

No GPU (the gate stubs the lease and the idle probe; no card touched). No
process killed or signalled. Git never run. No subagents. No symlinks.
**Core untouched** — follow-up 3 edits the runner, one test file, F2's
contract JSON and the registration scripts only. The 48 MB `session/` copy
taken in follow-up 2 was removed after verifying the canonical repo still
holds it; `/mnt/ForgeRealm/GraftRepository` was never written. Every pytest
run used `--basetemp /mnt/ForgeRealm/wt/grm-f1/artifacts/grm_f1/tmp`,
removed afterwards. All Bash calls foreground, under 10 minutes.

---

# FOLLOW-UP 4 (lead, 2026-09-11): r3 ran; the cause table

Follow-up 3 was committed; the lead ran all three arms to completion on the
card, 26/26 cells each. This entry is the analysis of those receipts. No core
was read-modified; this seat wrote no campaign.

## 21. Scores, re-read from the receipts

Verified against `lead_{A0,Aprime,Aplusprime}_summary.json`, not copied from
the dispatch note: A0 fresh 10/15 corr 9/10 alias 10/10 recap 4/5; A′ 11/15,
9/10, **7/10**, 5/5; A+′ 13/15, 10/10, 10/10, 4/5. `complete=True`, 26/26,
`alias_fold_merge` false/false/true respectively.

**A0 reproduces r2 arm A exactly on every class**, so the registration's
`attribution_rule` is satisfied and the flags are attributable.

## 22. Eleven differing rows, not nine

The dispatch named 9. The receipts show **11**: `recall_1_25` and
`recall_1_50` are two further A′/A+′ gains the class totals absorb. Full
table in `artifacts/grm_f1/REPORT.md` §r3.

## 23. ONE mechanism behind all five A′ losses

`route_info.mount_dropped_for_width`, present on every one. F1's retention
puts the retained raw source into the rank plan, so plans grow from ONE
member to THREE and the seat sum runs **146–188 against width 96**. The fit
seats two and drops the third, which on these rows is the only node holding
the queried identifier.

`recall_7_10`: A0 plans `[12]` (35 seats); A′ plans `[12, 5, 18]` =
35+56+63 = 154, seats `[5, 12]`, **drops 18** — the alias edge; A+′ plans
`[20]` (48). Admission IDENTIFIED 18 in A′ and the fit evicted it, so the
abstention is HONEST: nothing mounted said Beacon IS Lantern. Not a
grounding bug — a width eviction.

`recall_5_50`: A′ seats 13 and 24, neither mentioning Medibay, and answers
"keep it at 12". The "12" is the **z-coordinate of Breakwater's
(-31, 48, 12)** inside node 24 — a cross-entity numeric leak from a
co-mounted node, not a model prior. Node 16 ("16 beds") was identified and
never planned.

`recap_2` (A+′'s only loss): A0/A′ mount the ERA INDEX (61/55) whose prose
names "Iona Vale"; A+′ plans `[13]`, the raw turn, whose assistant half is
"That gives this area a clearer identity". A raw turn is a worse reader than
a digest when the fact lives only in the user half.

## 24. The Beacon node per arm — F2 vindicated by a same-window comparison

A0 holds THREE Beacon nodes, including digest 24 over sources
`[13, 14, 17, 18]`: *"the maintenance crew of the Beacon is located at Iona
Vale, and the map position of the Beacon is (-31, 48, 12)"* — Commtower's
crew and Breakwater's coordinates both re-filed onto "the Beacon", and it
propagated into the ACTIVE era node 61. That is the capture defect F2 exists
to prevent, **present in the control arm**.

In A′ the digest at the same index has sources `[13, 14]` — alias turns 17
and 18 EXCLUDED from the window — and reads "the maintenance crew of
**Commtower** … the map position of **Breakwater**". Same fold, same cell,
correct attribution. A′ holds exactly ONE Beacon node (the raw edge); A+′
holds two (the retired edge and A1's merged digest).

**Correction to my own first reading.** I initially proposed `no_fold=True`
as the marker of F2's exclusion. That is WRONG and I checked it before
relying on it: F1's retention sets `no_fold` on every retained source (174
nodes in A′), so it does not isolate F2 at all. The discriminator is the
fold's SOURCE LIST — `[13,14,17,18]` vs `[13,14]` — which is what the table
above uses.

`fold_guard_history` is not persisted into the cell receipts, so F2's guard
events cannot be counted directly from this campaign. The source-list
comparison is the receipt. **Successor: persist `fold_guard_history` into
`worker.json`** so a future campaign can count exclusions and attribution
rejections without reconstructing them from node state.

## 25. The interaction, stated from receipts

1. F2 removes fact-less alias turns from every fold window (correct; kills
   the capture).
2. Never folded therefore never superseded: in A′ the Beacon edge ends the
   campaign `active=True`, `digest_of=None`, **source of nothing** — a lone
   raw turn with a relation and no value.
3. F1 inflates rank plans from one member to three.
4. The lone edge is admitted, ranked third, evicted for width.

A1 removes exactly that failure: node 20 carries `alias_merge=True`,
`supersedes=[19]`, and states relation AND value in ONE 48-seat node.

**Registered finding: the four flags are a SET at this arena width.** F1+F2
without A1 is not a partial improvement — it is an alias REGRESSION (10/10 →
7/10) produced by two individually-correct treatments interacting. **Do not
merge F1+F2 without A1 on the strength of A′.** What A′ would need instead:
co-mount of edge with base (63+35 = 98 > 96, and FIX-7 was STOPPED as
impossible at this width), plan-head protection so an identifier-bearing
member is never the evicted one, or a two-hop read that A1 deliberately
rejected in favour of the write-time join.

## 26. Residency — bounded everywhere, and the real warning

A0 323 rows max 94; A′ 322 rows max **95**; A+′ 309 rows max 94. Width 96.
No `RESIDENCY_BOUND_EXCEEDED` anywhere: the registered prediction HOLDS.

The warning the prediction could not anticipate: A′'s routable base of 213
active nodes against 96 seats does not overflow the BOUND — it overflows the
PLAN, and the eviction lands on the node the question named. Residency
bounds were the wrong instrument for this failure; `mount_dropped_for_width`
was the right one, and it is only in `route_info`, not in the residency
rows. **Successor: surface plan-eviction counts in the residency row.**

## 27. Verdict against the registered predictions

A0 MET (exact match, all classes). A′: fresh REFUTED as stated (11/15 vs
≥14/15), corrections MET, **aliases FAILED (7/10 vs "must not move at
all")**, recap MET. A+′: fresh MET on c2 (14/15) and missed on the primary
scorer (13/15), corrections MET (10/10), aliases MET, recap MET.

The A′ alias row is the registered falsifier firing verbatim. It fired on
A′, not A+′, and the separation is now diagnosed rather than merely detected.

Honest reading of A+′'s 38/40: real, but FOUR flags moved together. The
receipts license "the SET A1+F1+F2+F5 beats the control on this fixture",
never a single-flag claim.

## 28. Follow-up 4 process safety

No GPU (analysis only; this seat ran no campaign). No process killed or
signalled. Git never run. No subagents. No symlinks. **Core read-only — not
one core file was modified in this follow-up.** `/mnt/ForgeRealm/
GraftRepository` not written. Every pytest run used `--basetemp
/mnt/ForgeRealm/wt/grm-f1/artifacts/grm_f1/tmp`, removed afterwards. All Bash
calls foreground, under 10 minutes.

## 29. ROOT-CAUSED: the follow-up-3 gate was never running the CPU double

RED, then diagnosed. Reported in full because my first two write-ups of this
were wrong and the corrections are the useful part.

**Symptom.** On one run of the full F1 battery,
`tests/test_grm_f1_resume_out_root.py::
test_real_resume_route_through_pending_and_one_run_cell[A0]` failed with
`rc == 2` (`worker.run_cell` returned falsy => the leased child exited
non-zero). It passed on ten other runs. I first recorded it as an
unexplained ~1-in-6 flake, then as "weak evidence of absence" after a
four-iteration hunt came back clean. **Both of those write-ups were wrong.**

**Iteration 5 of the hunt reproduced it with the evidence attached**: all
THREE arms failed, in 32.78 s instead of 102 s, and the captured
`worker.log` files (2074 bytes each, identical) gave the real traceback:

    scripts/grm_lt1_1.py:880   lt1_1_worker -> worker.execute(..., gpu_loader)
    scripts/grm_lt1_1.py:891   gpu_loader -> e2e.load_model_and_repo(...)
    scripts/grm_e2e_session.py:2197  GptOss20B_TC.from_pretrained(...)

**The child was loading the real GPT-OSS-20B model.** It was never running
the CPU double.

**Cause.** The test patched `worker.spawn_argv` BEFORE calling `resume()`.
`resume()` then enters `lt1_1_seams`, which assigns
`worker.spawn_argv = spawn_argv` — the real `--worker` argv — and restores
its own saved value on exit. The seam silently wins over the patch, so the
spawned child took `main()`'s `--worker` branch (real model) instead of
`--worker-cpu`. The passing test in
`tests/test_grm_lt1_1_child_spawn.py` does not have this bug because it
patches INSIDE `with runner.lt1_1_seams(...)` and calls `worker.run_cell`
directly rather than going through `resume()`.

**Why it looked intermittent.** It was not a race and not load sensitivity.
Every "passing" run was also spawning a real 20B load; the cell took ~85 s
doing it and then happened to satisfy the surrounding assertions. Whether
the model-load path failed fast enough to surface as `WORKER_EXIT_1` varied
with page-cache state. The tell was in the timings all along: 102 s per
battery iteration for what should be a CPU-double spawn.

**Fix.** Patch `runner.spawn_argv` — the function the seams INSTALL — so the
patch survives the seams. The gate now runs in **15.9 s instead of 102 s**
and genuinely exercises the CPU double.

**What I got wrong, recorded so the pattern is visible.** I ruled out five
hypotheses by check (ordering, flag values, the SIGALRM deadline, RUN_ENV
leakage, shared roots) and each of those checks was sound — but I never
asked the first question, which is *what did the child actually do*. The
`worker.log` was always there to be read; two of my three isolated
reproduction attempts captured one and I did not open it until the hunt
handed me three at once. "Ruled out by check" is not the same as "diagnosed",
and I twice wrote a status line implying progress toward the latter.

Successor F1-R3-S5: audit every other test that patches a `worker.*` seam
outside `lt1_1_seams`. A patch that the seams overwrite is a test that
silently exercises production behaviour.
## 30. CORRECTION: the column-2 figures are NOT verifiable from the receipts

I carried the dispatch's c2 numbers (A0 fresh 13, A′ 12, A+′ 14/15, A+′ total
38/40) into a first draft of REPORT §r3 before checking whether any r3
artifact contains them. **It does not.** `probes.jsonl` score objects carry
`exact_correct`, `category` and the abstention fields — no `col2_*` key — and
`lead_*_summary.json` carries only `correct` / `n` / `expected_n` /
`exact_rate` per class. Amendment 9 bound c2 as a registered scoring column,
but the campaign left no receipt for it.

Corrected in the REPORT: the c2 figures are now labelled **reported by the
lead, unverified by this seat**, and no conclusion rests on them.

On the column I CAN recompute, the totals are:

| arm | primary total |
|---|---|
| A0 (control) | **33/40** |
| A′ (F1+F2+F5) | **32/40** — a NET REGRESSION against the control |
| A+′ (A1+F1+F2+F5) | **37/40** |

A′ being a net regression on the verifiable column strengthens the
flag-set finding rather than weakening it: the alias column does not merely
fail to improve, it costs more than the fresh/recap gains return.

**Successor F1-R3-S4: persist the column-2 verdict into `probes.jsonl` and
the summary.** A registered scoring column that leaves no receipt cannot be
audited from the artifacts the campaign produced, and a later seat reading
these receipts would reach a different total than the dispatch reported.
