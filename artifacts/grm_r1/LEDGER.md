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
