# GRM-R1 ledger — margin-first regression replay worker

Seat: Opus 5 (`claude-opus-5[1m]`), reasoning effort **high**.
Order: `orders/GRM_R1_MARGIN_FIRST_REPLAY.md`.
Worktree: `/mnt/ForgeRealm/wt/grm-r1` (branch `grm-r1`). No git run by this seat.
Date: 2026-09-10.

## Commands run, in order, with results

| # | Command | Result |
|---|---|---|
| 1 | hash-verify all 31 cells' `checkpoint` + `checkpoint_files` from `grm-lt1` | 31 checkpoints, **614 files, 0 mismatch, 0 missing** |
| 2 | `cp -a` FIX-6 receipts from `wt/grm-lt1/artifacts/grm_scout_fix6/` into this worktree | `artifacts/` is gitignored, so the registered inputs were absent here; 31/31 `policy_state_sha256` verified after copy |
| 3 | prototype: CPU fake repo + real `decisive_admission_profile`, cell 1 | **RED** — `AdmissionPolicyError: production route ranking differs from frozen A-DEC score reconstruction` |
| 4 | prototype with the FIX-6 `FrozenArena` seam (recorded ranking + synthesized score vector) | OFF `[6,7]` = recorded, ON `[6,7,4]` = recorded |
| 5 | prototype across all 31 cells | **OFF 31/31, ON 31/31** |
| 6 | `python3 scripts/grm_r1_register.py` | registration written, sha `b02593ec…a56cf5`; 5 batches, 1057 s = **0.2936 GPU-h** under the 1.0 GPU-h cap |
| 7 | `python3 scripts/grm_r1_replay.py --dry-run` | `READY_FOR_LEAD`, `gpu_executed: false`, 31 cells enumerated with estimates |
| 8 | `--batch R1..R5 --fake` (CPU) | all 5 batches COMPLETE, 31 cell receipts |
| 9 | `--summary` on the fake run | `NOT_MEASURED`, parity 31/31, `adopt: false` (correct: a fake run measures no answers) |
| 10 | `pytest -q tests/test_grm_r1_replay.py` | first run **2 failed, 38 passed** |
| 11 | diagnose the 2 failures | MY TEST was wrong, not the worker: the pre-flight guards raise `ValueError` *before* the reservation, so they are not wrapped into the controller's `RuntimeError`. Tests corrected to assert the real type, plus new assertions that no reservation and no owner file are left behind. |
| 12 | `pytest -q tests/test_grm_r1_replay.py` | **40 passed** |
| 13 | `pytest -q tests/test_grm_r1_cpu_gate.py` | **3 passed** |
| 14 | `pytest -q tests/test_grm_admission.py tests/test_grm_scout_fix4.py tests/test_grm_scout_fix6*.py` | **1 failed, 27 passed** — `test_nonrecency_existing_fixture_receipt_byte_identical` |
| 15 | diagnose that failure | **Pre-existing worktree artifact gap, not a regression.** The test reads gitignored `artifacts/grm_c7/fixture.json` and `artifacts/grm_scout_fix4/nonrecency_before.json`, neither present here; it imports nothing of mine (`grep -c grm_r1 tests/test_grm_scout_fix4.py` = 0). |
| 16 | copy `artifacts/grm_c7/{fixture,fixture_manifest,registration}.json` + `.sha256` from `wt/grm-c7`, and `artifacts/grm_scout_fix4/` from `wt/grm-lt1` | **28 passed** — including the byte-identical FIX-4 receipt check, which proves R1 perturbed nothing |

## Findings

1. **The CPU double cannot reproduce GPU routing, and that is by design.**
   `grm_c7_diagnose.CPUArena._node_key` returns a constant unit vector, so
   `decisive_admission_profile`'s frozen score-reconstruction guard fires
   (`production=[6,7,4,11,8,1] reference=[6,7,1,4,8,11]`). The CPU gate
   therefore replays the recorded eligibility/ranking/margin through the REAL
   `core.grm_admission` boundary, exactly as FIX-6's `FrozenArena` did. This
   is a TEST SEAM; it never runs on GPU and changes no admission logic.

2. **The 31 registered states carry `ranking` and `margin` but no score
   vector.** `frozen_scores()` synthesizes the vector consistent with both
   (strictly descending in rank order; rank1−rank2 == the recorded margin).
   This is sufficient because none of the 31 cells needs the tie-break path:
   all have `margin > 0` and the FIX-6 `MISSING_TIED_SCORE_GROUP` cases are
   among the 20 UNRESOLVED executions, which are excluded, not imputed.

3. **Both arms must get their own repository copy.** The production
   `run_turn` deposits into the repository (`_mark_dirty`, `_snapshot_state`),
   so a shared repo would leak arm OFF's deposits into arm ON. "Same state"
   is therefore guaranteed by the shared, hash-verified checkpoint — not by a
   shared mutable object. The GPU model is still loaded only ONCE per batch.

4. **The rule flip is a live env read.** `admission_rule()` reads
   `os.environ` at call time, so `GRM_ADMISSION_RULE` is the single
   difference between arms, pinned AFTER `environment(flags)` strips every
   ambient `GRM_` var (FIX-6's state contract) and read back and asserted.

5. **A fake run can never yield a verdict.** `summary()` requires
   `answers_measured` — every arm receipt carrying `answer_measured: true` —
   before it will return `ADOPT`/`DO_NOT_ADOPT`. The CPU run returns
   `NOT_MEASURED` with `adopt: false`, which is the honest ceiling.

6. **Budget came from receipts, not guesses.** The C2 scout-fix-2 restart
   controllers give charged seconds and probe counts; single-probe cells
   isolate (load + 1 probe) and four-probe cells give (load + 4), yielding
   defaults 19.31 s load / 6.91 s per probe and profile 25.87 s / 5.56 s.

## Deviations from the order

* The order says "reuse their lease, receipt and sha-binding code; do not fork
  a third copy of the lease logic if a shared helper exists." Done —
  `grm_cmc1_gpu_arms.gpu_lease` is IMPORTED, and `read/sha/write/need` come
  from `grm_c7_amendment7`, `environment/flags_for/args_for/observe` from
  `grm_c2_cells`. No lease logic is forked.
* The order lists `scripts/grm_r1_replay.py` as the deliverable; I added
  `scripts/grm_r1_register.py` so the registration is BUILT reproducibly from
  the C2 receipts rather than hand-written. The registration remains
  immutable (create-only; `R1_REGISTRATION_IMMUTABLE_ALREADY_EXISTS`).
* Copying gitignored artifacts from `wt/grm-lt1` and `wt/grm-c7` into this
  worktree was required to run the gates at all. The order authorizes those
  worktrees as "receipts you may copy FROM".

## RED items

* None outstanding. The single suite failure (#14) was a missing gitignored
  artifact in this worktree, fixed by copy at #16, and is NOT a code
  regression — the FIX-4 byte-identical receipt check passes.
* The RED at step #3 was a genuine mechanism finding, recorded above, and
  drove the design of the CPU seam.

## Process safety

No process was killed or signalled. No GPU lease was taken, no GPU device was
touched, and GPT-OSS-20B was never loaded. Every Bash call ran in the
foreground and completed; nothing was backgrounded and nothing was left
running. No git command was run.

---

# Amendment 1 — device-memory release + re-arm batch R1 (2026-09-10)

Trigger: the lead's GPU run of batch R1 died with
`RuntimeError('cudaMalloc failed: out of memory')` at turn 22.

## Ground truth read from the receipts (NOT from the brief)

The amendment brief said "completed 7 of 8 cells ... the 8th cell's second
arm died", i.e. one missing cell. **The receipts say otherwise, and I built
the amendment from the receipts:**

| Source | Finding |
|---|---|
| `artifacts/grm_r1/gpu/R1/cells/` | **5** cell receipts, not 7 |
| `artifacts/grm_r1/gpu/R1/sessions/` | **6** session dirs, each with `off` and `on` |
| `artifacts/grm_r1/batch_R1.log` | **12** probe lines: 11 clean + the 12th carrying the OOM |

So: cells 1–5 completed both arms (10 probe lines). Cell 6
(`defaults-census-restart-1--e2e_t22_mira_seal`) completed arm OFF (line 11)
and died on arm ON (line 12) — which is why it has both session dirs but no
receipt. Cells 7–8 (`longhistory-1--lh_t013`, `lh_t016`) never started.
**5 retained, 3 re-issued.** The brief's "one missing cell" would have left
two cells silently unrun. `resume_scope()` now cross-checks the amendment's
lists against disk and STOPS on disagreement rather than re-scoping silently.

## Growth diagnosis, with numbers

Cause, from the code: `GraftRepository.close()`
(`core/graft_repository.py:368-372`) releases only the **native store** —
it sets `native_store = None` and returns. It never touches the arena. But
`ArenaCache._graft_block` (`core/graft_arena.py:2351-2356`) states in its own
docstring: *"Grafts are device-resident tc tensors — no host->device upload
per swap."* So every `grafts[i]['h']` an arm harvests is VRAM that survives
`close()`. With a fresh arena per arm and two arms per cell, residency is
cumulative across the whole batch process.

Node counts per arm-pass, from `batch_R1.log`:

| pass | cell | turn | nodes | cumulative resident |
|---|---|---|---|---|
| 1–2 | census-1 t13 | 13 | 16, 16 | 32 |
| 3–4 | census-1 t16 | 16 | 19, 19 | 70 |
| 5–6 | census-2 t22 | 22 | 26, 26 | 122 |
| 7–8 | census-restart-0 t13 | 13 | 16, 16 | 154 |
| 9–10 | census-restart-0 t16 | 16 | 19, 19 | 192 |
| 11 | census-restart-1 t22 (arm OFF) | 22 | 26 | **218** |
| 12 | census-restart-1 t22 (arm ON) | 22 | 25 → **OOM** | — |

218 node payloads were resident when the 12th pass tried to harvest ~26 more.
Note pass 12 reports `nodes: 25` and `deposit_ms: 0.0` — it died mid-harvest.
The card was otherwise idle (lead: 514 MiB after release), so this is the
batch's own peak, exactly as the lead diagnosed.

Also consistent with the wall clock: the ON arm ran ~7.0 s against the OFF
arm's ~10.2–11.1 s, because ON reused payloads the OFF arm had already pinned.

## Fix

`release_arm(repo)` (`scripts/grm_r1_replay.py:283-345`), called in the
`finally` of each arm **before the next arm allocates**:
1. `arena.reset_live_cache()` — drops the live KV (`self.caches`);
2. `g['h'] = None` for every graft — the repository pager's OWN idiom, used
   verbatim by `_free_retired`, `_page` and `_mark_payload_missing`;
3. `repo.close()` — the native store, as before;
4. `tensor_cuda.empty_cache()` — returns freed blocks to the driver.

Persisted node text, metadata and lineage are untouched; only the device
payload is dropped, and the on-disk checkpoint remains the source of truth.

Per-cell `device_memory` receipt added: MiB before/after each arm and each
cell, plus `payloads_freed` per arm. Point samples, never peaks; nothing
gates on them. `None` off-GPU rather than a fabricated `0`.

I did NOT move to a child process per cell. The in-process release is
sufficient and cheaper (it keeps the one-model-load-per-batch design); the
`device_memory` receipts will show on the next run whether that holds. If
they show climbing peaks, the child-process route remains available and the
estimates would have to be recomputed for a per-cell model load.

## Amendment commands and results

| # | Command | Result |
|---|---|---|
| 1 | read `batch_R1.log`, `controller.json`, `cells/`, `sessions/` | 5 receipts / 3 missing / OOM on cell 6 arm ON |
| 2 | validate the 5 completed receipts | parity 5/5, ON plans match 5/5, all `unchanged_correct`, real measured answers |
| 3 | `release_arm` on a 26-payload stand-in | `payloads_freed: 26`, `pool_emptied: True`, close called, live cache reset |
| 4 | `python3 scripts/grm_r1_amend_1.py` | amendment written, sha `fef08f65…81683`; R1 retain 5 / reissue 3 / lease 134 s |
| 5 | `verify()` + `resume_scope()` + `campaign_state()` | campaign UN-CLOSED: `charged=263.0 complete=[]`, no `R1_FAILED_CAMPAIGN_STOP` |
| 6 | `pytest` R1 suites, several iterations | see findings below; final **63 passed** |
| 7 | `--dry-run`, `--summary` on the real root | `READY_FOR_LEAD`; summary `NOT_MEASURED`, 5/31 measured, parity True |
| 8 | full required battery | **91 passed** |

## Findings from the amendment work

1. **Self-binding is circular and had to be broken explicitly.** The fix
   lives in the worker, which is itself a registered sha-bound input, so
   every edit invalidated the amendment that re-bound it. Resolved by making
   the amendment the LAST artifact built, and by scoping `rebound_inputs` so
   it can only replace a hash for a path the registration already listed
   (plus the amendment's own builder, via `REBINDABLE_NEW`). A test asserts
   an amendment cannot smuggle in a new core or scoring source, and another
   asserts the un-rebound inputs still fail closed.
2. **An amendment is campaign state, not receipt-directory state.** First
   cut had the amendment un-closing any `--out` root, which silently defeated
   two fail-closed tests. Corrected: the re-arm scope and the un-closing
   apply ONLY to the root the amendment was built against, while the re-bound
   SOURCE hashes apply everywhere (a gate elsewhere still runs the amended
   worker). The two tests that caught this were right and are unchanged.
3. **The failed attempt's 263 s charge is not refunded.** It stays on the
   books; the amendment adds 134 s on top. Total across the campaign:
   263 + 134 + 794 = 1191 s = 0.331 GPU-h, still under the 1.0 GPU-h cap.

## RED items

None outstanding. The OOM itself is a genuine RED, diagnosed above with a
mechanism and numbers rather than worked around. The brief's cell count was
corrected against the receipts and is called out in `lead_commands.txt` so
the lead is not surprised by a 3-cell re-issue.

## Process safety (amendment 1)

No process killed or signalled. No GPU lease taken and no GPU work run; the
only device interaction was a read-only `nvidia-smi --query-gpu=memory.used`
point sample (returned 327 MiB, card idle), which allocates nothing. Every
Bash call foreground and completed. No git command run.

---

# Amendment 2 — stale-session archiving + environment restoration (2026-09-10)

Two defects, both mine, both found by the lead AFTER amendment 1 was
verified and committed (c2c8865).

## Defect 1 — the re-issue collided with its own scratch

`artifacts/grm_r1/batch_R1_a1.log`, lease acquired then released in 4.67 s:

```
FileExistsError: [Errno 17] File exists: '.../gpu/R1/sessions/
fix6-replay-defaults-census-restart-1--e2e_t22_mira_seal/off'
```

Verified on disk: that cell (the one whose arm OFF finished and whose arm ON
OOMed) still has BOTH `off/` and `on/`. `open_arm` creates the session with
`mkdir(exist_ok=False)` — correct for a first run, fatal for a re-issue.

I had applied create-only uniformly. That is right for **evidence** and wrong
for **scratch**: amendment 1 taught the campaign to re-issue a cell but left
the cell's own working directory in the way. Fix:
`archive_stale_sessions()` moves that cell's own stale arm dirs to
`sessions/<cell>/attempt_<n>/` before the arms run. Archived, not deleted —
the failed attempt's scratch stays inspectable. Nothing outside that cell's
session directory is touched, and a cell that already has a receipt is never
re-run, so this code never sees a receipt.

## Defect 2 — the rule pin leaked into the process environment

`pin_rule` set `GRM_ADMISSION_RULE`; `environment(flags)` rewrote the whole
`GRM_` frame; neither restored. `admission_rule()` reads `os.environ` at CALL
time, so every module importing after a R1 test in the same process was
silently re-ruled to `margin_first`.

Reproduced before the fix:

```
$ pytest -q tests/test_grm_r1_replay.py tests/test_grm_scout_fix4.py tests/test_grm_admission.py
4 failed, 54 passed
  test_grm_scout_fix4.py::test_identifier_binds_nothing_still_abstains[core]
  test_grm_scout_fix4.py::test_identifier_binds_nothing_still_abstains[ladder]
  test_grm_scout_fix4.py::test_nonrecency_existing_fixture_receipt_byte_identical
  test_grm_admission.py::test_decisive_profile_reuses_route_laws_and_frozen_receipt
```

and directly:

```
after pin_rule(on): 'margin_first'
admission_rule() now: margin_first   <-- every later module sees this
```

**My earlier "91 passed" ran R1 LAST, which is exactly why I never saw it.**
A suite whose greenness depends on file order is not green. The refreshed
`lead_commands.txt` now runs R1 FIRST on purpose, so a future leak turns the
following suites red instead of hiding.

Fix: `scoped_env()` snapshots and restores the EXACT prior environment in a
`finally`, around each arm's pin and around the batch frame pin. A key absent
before is REMOVED, not set to `''`; keys the body ADDED (the whole frame) are
dropped. Tests use `monkeypatch` so no test can itself leak.

After the fix:

```
env after batch: None
leaked GRM_ vars: NONE
admission_rule(): all_tokens_bind
```

## Commands and results

| # | Command | Result |
|---|---|---|
| 1 | read `batch_R1_a1.log`, list `gpu/R1/sessions/` | collision confirmed; cell 6 holds `off/` + `on/` |
| 2 | `pin_rule('on')` then `admission_rule()` | leak confirmed: `margin_first` persists |
| 3 | `pytest r1_replay + scout_fix4 + admission` (R1 first) | **4 failed, 54 passed** — the lead's defect reproduced |
| 4 | implement `scoped_env`, `archive_stale_sessions`; chain-aware `amendment()` | — |
| 5 | env probe after a fake batch | `None` / no leaked `GRM_` vars / `all_tokens_bind` |
| 6 | collision fixture (plant stale scratch, re-issue) | archived to `attempt_1/`, fresh arms created, retained receipts byte-identical, all cells have receipts |
| 7 | `python3 scripts/grm_r1_amend_2.py` | amendment 2 written, sha `e2d4b743…6ddfc`, chains to a1 |
| 8 | `pytest` all four R1 files | **80 passed** |
| 9 | `pytest` R1 FIRST + fix4 + admission (the ordering that failed) | **98 passed** |

## Findings

1. **Create-only is right for evidence and wrong for scratch.** The
   receipt/session distinction now has to be explicit, or the no-overwrite
   rule becomes a self-collision on any re-issue. Receipts: never touched.
   Scratch: archived per attempt.
2. **Process-global state defeats file-scoped gates.** `admission_rule()`
   reading `os.environ` at call time means an unrestored pin is not
   untidiness, it is a correctness defect in OTHER suites. The worker now
   restores in production too, not just under pytest — production has no
   monkeypatch.
3. **Test ordering is part of the gate.** Placing the mutating module last
   hid a real defect through two rounds of review, including mine.
4. Three fixtures of mine that planted only `amendment_1.json` broke once the
   loader required the full chain; they now copy the real chain. That is the
   chain check working, not a false alarm.

## RED items

None outstanding. Both defects are fixed, reproduced-before and
verified-after, with the reproduction commands recorded above.

## Process safety (amendment 2)

No process killed or signalled. No GPU lease taken, no GPU work run, no model
loaded. Every Bash call foreground and completed. No git command run.

## Third defect, found while verifying amendment 2 (accounting)

Checking `campaign_state` against the real receipts after the lead's second
attempt, the campaign reported **134 s charged for a batch that had actually
spent 263 + 134 = 397 s**. Cause: amendment 1 archives the prior attempt's
controller to `controller_attempt_1.json`, but `campaign_state` globbed only
`controller.json` -- so every re-issue silently REFUNDED the failed attempt's
GPU time. On a 1.0 GPU-h cap that is a budget rail that does not hold.

Fix: `campaign_state` now also sums `controller_attempt_*.json` (binding-
checked the same way). Verified on the real receipts: 397.0 s, complete=[].
A test asserts a re-issue ADDS to the prior charge rather than replacing it.

This one was mine too, introduced by amendment 1's archiving and caught only
because I re-read the live numbers instead of trusting the passing suite.

---

# Amendment 3 — a satisfied re-arm scope must stop gating (2026-09-11)

## What the lead's run produced first (the good news)

`--batch R1` re-issue COMPLETED: **8/8 cell receipts, parity 8/8, ON plans
match the registered plans 8/8, every transition `unchanged_correct`,
0 answers changed**, lease released at 79.1 s. The amendment-1 device fix is
visible in the receipts it produced: payloads freed per arm (23/23, 14/14,
17/17) and device memory **flat at 10996 MiB across all three cells** instead
of climbing. The five older receipts predate the probe and carry `None`.

## The defect

`--batch R2` then refused before running
(`artifacts/grm_r1/batch_R2_a2.log`):

```
ValueError: R1_AMENDMENT_RETAIN_MISMATCH: R1 on_disk=[...8 cells...]
amendment=[...5 cells...]
```

Mine again. `resume_scope()` re-applied amendment 1's "R1 retains 5 /
reissues 3" cross-check on EVERY invocation, including from batches with
nothing to do with R1. That check is a **precondition** — "is the state I am
about to act on still the state this amendment was written against?" Once
R1's controller says COMPLETE the re-issue has happened and the scope is
**satisfied**: all 8 registered cells have receipts, which is exactly the
outcome the amendment existed to produce. Re-evaluating the precondition
against the state its own action produced closed the campaign on its own
success.

Amendment 2 fixed a self-collision on scratch. This is the same family one
level up.

## Fix

The cross-check now applies only while the re-armed batch's controller is NOT
COMPLETE. Once complete, `resume_scope()` returns
`scope_satisfied: true`, an empty `reissue`, and `receipts` mapping every
cell id to its receipt sha256; `batch()` treats a satisfied scope as a
RECORD, never a re-issue instruction.

Deliberately NOT relaxed:
* while the batch is OPEN the cross-check is unchanged
  (`R1_AMENDMENT_RETAIN_MISMATCH` / `_REISSUE_MISMATCH`);
* a COMPLETE controller with cells still missing is a NEW red,
  `R1_AMENDMENT_SCOPE_UNSATISFIED`;
* per-receipt binding and parity checks still run on satisfied scopes.

## A fifth defect, found while verifying

`summary()` reported **79.1 s** for a campaign `campaign_state()` scored at
**476.1 s** — amendment 2's accounting fix had been applied to
`campaign_state` only, so `summary` still ignored
`controller_attempt_*.json`. Fixed; both now report 476.1 s, and a test
asserts they agree.

## Commands and results

| # | Command | Result |
|---|---|---|
| 1 | read `batch_R2_a2.log`, R1 controller + cells | R1 COMPLETE, 8 receipts, guard raised from R2 |
| 2 | inspect the 8 R1 receipts | parity 8/8, ON match 8/8, all `unchanged_correct`, device flat at 10996 MiB |
| 3 | implement the precondition scoping + satisfied record | — |
| 4 | `python3 scripts/grm_r1_amend_3.py` | amendment 3, chains to a2 |
| 5 | `resume_scope` / `campaign_state` on live receipts | satisfied True, reissue 0, receipts 8, complete `['R1']` |
| 6 | R2 precondition check | scope None (normal first run), priors complete, budget 476.1+194=670.1 ≤ 3600 |
| 7 | RED-before / GREEN-after probe on one state | with a3: OK; with the batch forced back to open: `R1_AMENDMENT_RETAIN_MISMATCH` — the lead's exact stop |
| 8 | full gate in the leak order | **110 passed** |

## Findings

1. **Every guard needs an explicit answer to "when does this stop applying?"**
   Three of the five defects in this arc were guards that were correct in the
   state they were written for and wrong in the state that followed.
2. **A fix applied to one reader must be applied to every reader of the same
   fact.** Amendment 2's archived-charge fix went into `campaign_state` and
   not `summary`, and the two then disagreed by 397 s.
3. Older test helpers planting a 2-link chain broke once the chain reached 3
   links, and two a2 tests asserted "latest == 2". Both are expected-shape
   drift from a growing chain, not new defects; the helpers now plant every
   link and the a2 tests assert link 2's own contents plus chain membership.
4. Two a1 tests asserted the live retain-5/reissue-3 state that has since
   legitimately advanced to satisfied; they now assert the amendment's
   RECORDED scope plus the current satisfied state, and the two
   disagreement tests construct an OPEN batch (where the check lives).

## RED items

None outstanding.

## Process safety (amendment 3)

No process killed or signalled. No GPU lease taken, no GPU work run, no model
loaded. Every Bash call foreground and completed. No git command run.

---

# Amendment 4 — idle-card pre-check + NON_FIT rail (2026-09-11)

## Item 1: the OOM site and the counts

Site: the controller records `RuntimeError: cudaMalloc failed: out of memory`
at **13.4 s elapsed**, with `cells_run` listing all 8 but **0 cell receipts**
and only the first cell's `off` session dir present. So it died AFTER the
copytree, during model load or payload harvest, before any probe line.

Counts, from the recorded checkpoint manifests:

| batch | cells | max nodes | max npz MiB | batch total MiB |
|---|---|---|---|---|
| R1 | 8 | 25 | 36.7 | 221.5 |
| R2 | 5 | 25 | 36.7 | 130.0 |
| R3 | 8 | 25 | 36.6 | 220.8 |
| **R4** | 8 | 25 | 36.6 | **151.6** |
| R5 | 2 | 3 | 6.1 | 12.3 |

## Item 3: lever (a) DECLINED, with its receipt

* R4's first cell (`profile-longhistory-2--lh_t022`, 25 nodes / 36.6 MiB) is
  the SAME size as `profile-census-2--e2e_t22_mira_seal`, which **R3
  completed** on the same profile frame (freeing 23 payloads per arm).
* R4 is the **smallest profile batch** by total payload (151.6 vs 220.8 MiB).
* Largest single cell = 36.6 MiB against ~1,286 MiB headroom: **2.8%**
  (5.7% for both arms), i.e. **~35x margin**.

So there is no fit problem, and lever (a) would add a page-in path for a
problem the numbers say does not exist. Declined and recorded as
`lever_2a_host_resident_payloads.implemented: false`.

Corroborating: my probe read 10,994 MiB with 6 foreign PIDs mid-analysis —
which the lead confirms were the lead's own runs. And R3's receipts show the
plateau pinned to the MiB across 15- and 25-node cells alike, which is a
shared card, not a leak.

## Item 2: what was implemented

**Idle pre-check.** `device_snapshot()` reads the full nvidia-smi XML;
`idle_gate()` keys on TOTAL framebuffer used and NEVER on the
compute-process list — the R4 holder listed none, and FIX-8's `parse_memory`
states exactly this rule. `await_idle()` polls a read-only probe for a
bounded `--idle-wait`. It only DECLINES to start; it never signals, kills or
clears anything. Recorded as `idle_check` in the controller.

**NON_FIT rail.** An OOM at load/harvest writes a create-only receipt
(reason, stage, memory snapshot incl. `foreign_holder_suspected`) and the
batch CONTINUES. `is_oom` is a deliberately narrow text match; any non-OOM
error and the parity RED still stop the campaign. NON_FIT is UNMEASURED in
`summary()` and never counted as `unchanged`.

## Commands and results

| # | Command | Result |
|---|---|---|
| 1 | read `batch_R4_a3.log`, R4 controller, sessions | died 13.4 s, 0 receipts, first cell `off` only |
| 2 | node/payload counts from every manifest | table above; R4 smallest profile batch |
| 3 | compare R4 cell 1 vs R3's completed cells | identical size (25 nodes / 36.6 MiB) |
| 4 | draft worker + amendment + tests in scratchpad | held unapplied while the lead's run held the lease |
| 5 | apply drafts after the lead reported finished | worker 1200 -> 1408 lines |
| 6 | `python3 scripts/grm_r1_amend_4.py` | R4 re-armed 0 retain / 8 reissue; R5 not re-armed (no controller) |
| 7 | `pytest tests/test_grm_r1_amendment4.py` | 18 passed, then 20 with the zero-receipt fixture |
| 8 | full gate, leak order | **1 failed** — my own no-retry guard caught `while True` in `await_idle` |
| 9 | rewrite `await_idle` as a counted `for` loop | bound is now structural, guard stays strict |
| 10 | rebuild amendment, full gate, leak order | **130 passed** |

## Findings

1. **The guard I wrote in amendment 1 caught my own amendment-4 code.**
   `await_idle` used `while True` with a deadline break; the no-retry test
   forbids that string outright. Rather than loosen the test I rewrote the
   loop with a counted bound, so waiting-on-a-resource is visibly different
   from retrying-our-own-work in the code, not just in a comment. I also
   strengthened the test: `run_cell` must have exactly one call site.
2. **Check the premise against the counts.** "Bigger cells did not fit" was
   the natural reading of an OOM, and it was wrong: R4 is the smallest
   profile batch and R3 had already run an identical-size cell. The
   manifests settled it in one query.
3. **A shared machine needs a pre-flight, not just a lease.** The flock
   coordinates the agents that agreed to use it; it says nothing about a
   display-side program that never did.
4. Chain drift again: a3/a2/a1 helpers planted 3 links, the chain is now 4,
   and two tests asserted "latest == 3" / "only R1 re-armed". Expected-shape
   drift, corrected the same way as before.

## RED items

None outstanding. R4 remains unrun (0 receipts) by design — it is re-armed
and waiting for the lead's GPU run behind the new idle check.

## Process safety (amendment 4)

No process killed or signalled. No GPU lease taken, no GPU work run, no model
loaded; the only device interaction was the read-only `nvidia-smi -q -x`
probe, which allocates nothing. `scripts/grm_r1_replay.py` was deliberately
left unedited until the lead reported the R4/R5 run finished. Every Bash call
foreground and completed. No git command run.
