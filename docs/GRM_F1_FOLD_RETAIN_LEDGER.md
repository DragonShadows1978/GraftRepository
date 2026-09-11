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
