# GRM-R1 — margin-first regression replay worker (seat report)

Seat: Opus 5 (`claude-opus-5[1m]`), reasoning effort **high**.
Status: **worker + registration + CPU gates COMPLETE; the 31-cell measurement
is BLOCKED on the lead's GPU run** (this seat is forbidden to touch the GPU).

## What this builds

A resumable, leased, create-only-receipt worker that runs each of the 31
SCOUT-FIX-6 registered execution cells **twice from one recorded C2
checkpoint**: arm OFF (today's `all_tokens_bind`) and arm ON
(`GRM_ADMISSION_RULE=margin_first`), through the PRODUCTION ladder
(`scripts/grm_e2e_session`), scored by the LT1/C5 value-span rule, and
classified into answer transitions.

### Entry points (`scripts/grm_r1_replay.py`)

| Command | Does |
|---|---|
| `--dry-run` | verifies registration + all 31 checkpoints (614 files) and enumerates cells with estimates. No GPU, no model. |
| `--batch <ID>` | runs one batch under ONE GPU lease: both arms of each cell. |
| `--batch <ID> --fake` | the CPU gate: real admission plans, no GPU, no reader. |
| `--summary` | applies the registered verdict rule verbatim to the receipts. |
| (no flag) | prints `REGISTERED_NOT_RUN`. |

### How the flag is pinned per arm

`core.grm_admission.admission_rule()` reads `os.environ` **at call time**, so
one environment variable is the entire difference between the arms:

```
pin_rule('off') -> os.environ.pop('GRM_ADMISSION_RULE')   -> 'all_tokens_bind'
pin_rule('on')  -> os.environ['GRM_ADMISSION_RULE']='margin_first' -> 'margin_first'
```

The pin happens **after** `grm_c2_cells.environment(flags)` has stripped every
ambient `GRM_*` variable — FIX-6's state contract requires exactly this,
because `environment()` would otherwise silently wipe the rule and run both
arms on the default. `pin_rule` then **reads the rule back through the real
core function and asserts it** (`R1_RULE_PIN_FAILED`), so a silent
mis-pin cannot happen. The pinned value is recorded per cell in
`rule_pins`, and each arm receipt carries the `admission_rule` the production
receipt itself observed.

### Proof the two arms share one state

* `verify_cell_inputs()` hashes the checkpoint descriptor **and every file in
  `checkpoint_files`** before either arm runs. A mismatch is RED, never a retry.
* Both arms `copytree` **the same hash-verified recorded repository**. Each
  gets a private copy because the production `run_turn` deposits into the
  repository (`_mark_dirty`); a shared repo would leak arm OFF's deposits into
  arm ON. Sameness is guaranteed by the shared checkpoint **hash**, not by a
  shared mutable object.
* The GPU model is loaded **once per batch** and reused via the ordinary
  `GraftRepository` constructor (C7 `open_copy` shape).
* Each cell receipt carries `same_state_proof`:
  `checkpoint_sha256`, `checkpoint_files_verified`,
  `arms_share_one_checkpoint`, `per_arm_private_copy`, and
  `differing_environment_keys: ["GRM_ADMISSION_RULE"]`.

### The parity barrier (RD1 A0 shape)

Arm OFF must reproduce the FIX-6 recorded `off_plan` **byte for byte**
(`json.dumps(..., separators=(',',':'))`). If it does not, the cell receipt is
written **first** (so the failure is auditable) and then the run **STOPS** with
`R1_OFF_PLAN_PARITY_RED_STOP`. There is no retry path anywhere in the worker;
a test asserts the source contains no retry machinery.

## Verdict machinery

Recorded in the registration **before** the gate runs, verbatim from the
immutable plan:

* Prediction: `<= 2 of 31 executions change their answer; 0 correct->wrong on supersession.`
* Verdict rule: `margin_first is adoptable as the profile default iff correct->wrong = 0 on sup and <= 1 elsewhere.`

`summary()` applies the verdict rule literally and **separately** reports
`prediction_held` — the prediction is an honesty record, not the gate. A test
asserts both directions (3 changed answers → `prediction_held: false` while
`adopt` stays true).

`summary()` also refuses to produce any verdict unless every arm receipt says
`answer_measured: true`. A `--fake` run therefore returns `NOT_MEASURED` /
`adopt: false` — the honest ceiling for a run with no reader.

## Results this seat can honestly claim

CPU only. **No model quality, throughput, or answer-accuracy claim is made.**

| Gate | Result |
|---|---|
| Checkpoint integrity | 31 checkpoints, **614 files, 0 mismatch, 0 missing** |
| Arm-OFF plan parity (CPU) | **31 / 31** byte-identical to the FIX-6 recorded `off_plan` |
| Arm-ON plan parity (CPU) | **31 / 31** equal to the FIX-6 registered `on_plan` |
| Rule pinning | `off → all_tokens_bind`, `on → margin_first`, on all 31 |
| Budget | 1057 s reserved = **0.2936 GPU-h**, cap 3600 s = 1.0 GPU-h |

## Honest limits (what is NOT established)

* **No answers were measured.** Every number above is an admission *plan*.
  Whether `margin_first` changes what the model actually says — the whole
  question R1 exists to answer — requires the lead's GPU run.
* The CPU gate replays the recorded ranking and a **synthesized** score vector
  (`frozen_scores`), because the 31 states record `ranking` and `margin` but
  not the score vector, and the CPU double's `_node_key` is a constant. The
  plans are real; the routing numbers behind them are the recorded GPU ones.
* The 20 FIX-6 executions without exact margins remain **UNRESOLVED and
  excluded**. Nothing is imputed. A test asserts they are not in the cohort.
* The cohort is 31 executions over only **9 distinct questions** — repeats
  across arm/phase are retained as separate executions, which is the FIX-6
  registration's own choice. This is not an independent 31-question sample,
  and the report should not be read as one.

## Prior art

Annotated at each code site, in `LEDGER.md`, and here.

* **GRM C2 restart cells** (`scripts/grm_c2_cells.py`, GRM contributors,
  2026). TAKEN and **imported, not re-implemented**: `environment(flags)`
  frame pinning, `flags_for`, `args_for`, `observe`, and `score_probe`'s
  traced-`run_turn` shape for census/longhistory vs the direct
  `_probe_ladder_chat` call for sup.
* **GRM C7 amendment-7** (`grm_c7_middle_replay.open_copy`, GRM, 2026).
  TAKEN: copy the recorded repository and load the model once per batch,
  reusing the recorded constructor arguments for later opens. **OURS**: two
  arms per cell off one load rather than one.
* **GRM C7 / FIX8 campaign discipline** (GRM, 2026). TAKEN: O_EXCL
  single-owner file, pessimistic reservations charged even on failure,
  create-only receipts, orphan/failed-campaign fail-closed, foreground
  cooldown outside the lease. The lease itself is
  `grm_cmc1_gpu_arms.gpu_lease`, **imported** — order item 1 explicitly
  forbids forking a third copy, and none is forked.
* **GRM RD1 amendment-1 / C7 `A0_R3_BYTE_MISMATCH_STOP`** (GRM, 2026).
  TAKEN: the arm-0 parity barrier concept. **OURS**: comparing the replayed
  OFF `rank_plan` bytes to the FIX-6 recorded `off_plan` bytes, with the RED
  receipt written before the stop.
* **GRM LT1 `grm_lt1_admission.evaluate`** and **FIX-6 `FrozenArena`**
  (GRM, 2026). TAKEN: replaying stored eligibility/ranking/margin through the
  real `core.grm_admission` boundary. **OURS**: `frozen_scores()`, which
  synthesizes the score vector consistent with the recorded ranking and
  margin, because the registered states do not carry one.
* **GRM DET1 comparator** (`lsr_p2c_replay_gpu.answer_verdict` over
  `grm_det1_common.contains_value`) and the DET1.3 refusal grammar
  (`grm_det1_3_gpu._is_refusal`). TAKEN wholesale, **imported**; the scorer is
  not re-implemented. **OURS**: only the five-way transition classification on
  top, and the decision to report abstention transitions *alongside* rather
  than folded into the correctness classes.
* **SQuAD** (Rajpurkar, Zhang, Lopyrev, Liang, 2016, arXiv:1606.05250) and
  **SQuAD 2.0** (Rajpurkar, Jia, Liang, 2018, arXiv:1806.03822) — the
  value-span and answerability/abstention ancestry behind `contains_value`
  and the refusal grammar. **Concept only**; neither the normalization nor
  the transition classes are SQuAD's. **Unverified — no network in this
  sandbox; lead to check.** Search terms: "SQuAD 100,000+ questions machine
  comprehension 2016", "Know What You Don't Know unanswerable questions 2018".
* **ARIES** (Mohan, Haderle, Lindsay, Pirahesh, Schwarz, 1992, IBM Research /
  ACM TODS) — the checkpoint/restart recovery concept behind the C2
  checkpoints this worker consumes. **Concept only**, no WAL algorithm is
  used or claimed. **Unverified — lead to check.** Search terms: "ARIES
  transaction recovery method write-ahead logging 1992".
* **Paired / counterfactual A-B evaluation from one held state** is standard
  experimental practice and I claim no novelty for it. **No prior art known
  to me** for the exact composition of (recorded-plan parity barrier +
  in-process rule flip + per-arm fresh repository copy + five-way transition
  classification with abstention reported separately). Re-deriving a known
  idea here would be fine; I simply do not know of a specific source.

**No new routing, scoring or admission algorithm is introduced by this work.**
Every policy decision is computed by the unmodified `core/grm_admission.py`.

## Process safety

No process was killed or signalled. No GPU lease was taken, no GPU device was
touched, and GPT-OSS-20B was never loaded. Every Bash call ran foreground and
completed; nothing was backgrounded. No git command was run by this seat.

---

# Amendment 1 — device-memory release + re-arm batch R1

## The correction the lead should read first

The amendment brief described batch R1 as "completed 7 of 8 cells ... the
8th cell's second arm died", i.e. **one** missing cell. The receipts say
**5 completed and 3 missing**, and I built the amendment from the receipts:

* `gpu/R1/cells/` holds **5** receipts, not 7.
* `gpu/R1/sessions/` holds **6** directories, each with both `off` and `on`.
* `batch_R1.log` holds **12** probe lines — 11 clean, the 12th carrying the OOM.

Cells 1–5 completed both arms. Cell 6
(`defaults-census-restart-1--e2e_t22_mira_seal`) finished arm OFF and died on
arm ON — hence two session dirs but no receipt. Cells 7–8
(`longhistory-1--lh_t013`, `lh_t016`) never started. Re-arming for one cell
would have left two cells silently unrun, and the campaign would have read as
complete while 2 of 31 executions were missing. `resume_scope()` now
cross-checks the amendment against disk and STOPS on disagreement.

## Growth diagnosis, with numbers

`GraftRepository.close()` (`core/graft_repository.py:368-372`) releases only
the **native store**; it never touches the arena. `ArenaCache._graft_block`
(`core/graft_arena.py:2351-2356`) says in its own docstring that "Grafts are
device-resident tc tensors", so every `grafts[i]['h']` an arm harvests is
VRAM that survives `close()`. A fresh arena per arm, two arms per cell, one
process per batch ⇒ residency accumulates across the batch:

| pass | cell / arm | turn | nodes | cumulative |
|---|---|---|---|---|
| 1–2 | census-1 t13 off, on | 13 | 16, 16 | 32 |
| 3–4 | census-1 t16 off, on | 16 | 19, 19 | 70 |
| 5–6 | census-2 t22 off, on | 22 | 26, 26 | 122 |
| 7–8 | census-restart-0 t13 off, on | 13 | 16, 16 | 154 |
| 9–10 | census-restart-0 t16 off, on | 16 | 19, 19 | 192 |
| 11 | census-restart-1 t22 **off** | 22 | 26 | **218** |
| 12 | census-restart-1 t22 **on** | 22 | 25 → **OOM** | — |

218 payloads resident when the 12th pass tried to harvest ~26 more. Pass 12
shows `nodes: 25` with `deposit_ms: 0.0` — it died mid-harvest. Corroborating
signal: the ON arm consistently ran ~7.0 s against the OFF arm's ~10.2–11.1 s,
because ON reused payloads OFF had already pinned.

## The fix

`release_arm()` runs in each arm's `finally`, **before the next arm
allocates** (not at cell end, which would still let ON pile onto OFF):

1. `arena.reset_live_cache()` — drops the live KV;
2. `g['h'] = None` per graft — the pager's OWN idiom, used verbatim by
   `_free_retired`, `_page`, `_mark_payload_missing`;
3. `repo.close()` — the native store, unchanged;
4. `tensor_cuda.empty_cache()` — returns blocks to the driver.

Node text, metadata and lineage are untouched; the on-disk checkpoint stays
the source of truth. Each cell now records `device_memory`: MiB before/after
each arm and each cell, plus `payloads_freed`. Point samples, never peaks;
nothing gates on them; `None` off-GPU rather than a fabricated `0`.

**I did not switch to a child process per cell.** The in-process release is
sufficient in principle and preserves one model load per batch. The honest
position: that is a mechanism argument, and the `device_memory` receipts on
the next run are what will actually settle it. If peaks still climb, the
child-process route remains open and the estimates must be recomputed for a
per-cell model load.

## Scope and budget

* Re-arms **only** batch R1, and within it **only** the 3 cells with no
  receipt. The 5 completed receipts are retained byte-for-byte, never re-run
  and never rewritten; a test asserts byte equality across a re-issue.
* The prior attempt's controller and reservation are archived as
  `controller_attempt_1.json` / `reservation_attempt_1.json`.
* R2–R5 unchanged: the fix removes cross-cell accumulation rather than
  changing per-cell cost, so the registered estimates stand.
* New R1 lease **134 s**, measured — max both-arm wall (18.12 s) over R1's
  own retained receipts × 3 cells + one model load + 60 s headroom.
* The failed attempt's **263 s is not refunded**. Campaign total
  263 + 134 + 794 = **1191 s = 0.331 GPU-h**, under the 1.0 GPU-h cap.
* Amendment sha `fef08f656af51ef8e46431090f05e69c8c0ce8c9e13a0afcfd9f944d4a781683`,
  bound to registration `b02593ec…a56cf5` and to the order file.

## What the amendment deliberately does NOT relax

* It un-closes **only** batch R1. Any other FAILED batch still stops the
  whole campaign — a test asserts this.
* `rebound_inputs` can only replace a hash for a path the registration
  already listed (plus the amendment's own builder). A test asserts an
  amendment cannot introduce a new core or scoring source, and another
  asserts the un-rebound inputs still fail closed.
* The prediction, the verdict rule and the parity barrier are unchanged.

## Amendment prior art

* **GRM C7 / C2 sha-bound amendment chains** (GRM contributors, 2026) —
  TAKEN verbatim: an amendment is a separate create-only file bound to the
  registration hash, widening scope explicitly, never editing the immutable
  registration; the prior attempt is archived, not overwritten.
* **GRM FIX8 `grm_scout_fix8_resume`** (GRM, 2026) — TAKEN: the shape of a
  registered successor to a single OOM-failed batch, and the nvidia-smi
  framebuffer probe (NVIDIA nvidia-smi XML/query docs, accessed 2026-09-09).
  The lead notes C7 FIX-8 F5 hit this same class yesterday.
* **The repository pager's own free idiom** (`g['h'] = None`, GRM, 2026) —
  TAKEN verbatim rather than inventing a release routine.
* **OURS**: the arena-payload release between arms, the per-cell device
  probe receipt, and the "retain completed cells, re-issue only the missing
  ones, and STOP if disk disagrees" scope. **No prior art known to me** for
  that exact composition. Reference-counted / arena allocator reuse is
  ordinary systems practice and I claim no novelty for the idea of freeing
  what you allocated. **No new algorithm.**

## Process safety (amendment 1)

Nothing killed or signalled. No GPU lease taken and no GPU work run; the only
device interaction was a read-only `nvidia-smi --query-gpu=memory.used` point
sample (327 MiB, card idle), which allocates nothing. Every Bash call
foreground and completed. No git command run.

---

# Amendment 2 — stale-session archiving + environment restoration

Two defects, both mine, both found by the lead after amendment 1 was verified
and committed (c2c8865). Neither touches admission, routing or scoring; the
prediction, verdict rule and parity barrier are unchanged.

## Defect 1 — the re-issue collided with its own scratch

`batch_R1_a1.log`: lease acquired, released 4.67 s later, with
`FileExistsError: .../sessions/fix6-replay-defaults-census-restart-1--e2e_t22_mira_seal/off`.

That is cell 6 — the one whose arm OFF finished and whose arm ON hit the OOM —
so it still holds both `off/` and `on/`. `open_arm` creates the session with
`mkdir(exist_ok=False)`: correct for a first run, fatal for a re-issue.

The underlying mistake is mine and worth naming: **I applied create-only
uniformly.** It is right for *evidence* and wrong for *scratch*. Amendment 1
taught the campaign to re-issue a cell but left that cell's own working
directory in the way. `archive_stale_sessions()` now moves the cell's own
stale arm dirs to `sessions/<cell>/attempt_<n>/` before the arms run —
archived, not deleted, so the failed attempt stays inspectable. Nothing
outside that cell's session directory is touched, and a cell that already has
a receipt is never re-run, so this code never sees a receipt.

## Defect 2 — the rule pin leaked into the process environment

`pin_rule` set `GRM_ADMISSION_RULE`, `environment(flags)` rewrote the whole
`GRM_` frame, and neither restored. `admission_rule()` reads `os.environ` at
CALL time, so every module importing after a R1 test in the same process was
silently re-ruled to `margin_first`.

Reproduced before the fix — the lead's exact symptom:

```
$ pytest -q tests/test_grm_r1_replay.py tests/test_grm_scout_fix4.py tests/test_grm_admission.py
4 failed, 54 passed
```

**My earlier "91 passed" ran R1 LAST, which is precisely why I never saw it.**
A suite whose greenness depends on file order is not green. That is the real
lesson here, and it is now built into the gate: `lead_commands.txt` runs R1
FIRST on purpose, so a future leak turns the following suites red rather than
hiding behind ordering.

`scoped_env()` restores the exact prior environment in a `finally` around each
arm's pin and the batch frame pin: a key absent before is REMOVED (not set to
`''`), and keys the body added are dropped. It lives in the worker, not only
in tests, because production has no `monkeypatch`. After the fix a fake batch
leaves `GRM_ADMISSION_RULE` absent, no `GRM_` vars at all, and
`admission_rule() == 'all_tokens_bind'`.

## Amendment

`artifacts/grm_r1/amendment_2.json`, sha
`e2d4b7439a78f07400ecd12dcfc403adeb46770d368cd01cf798c4a638d6ddfc`, bound to
registration `b02593ec…a56cf5`, to the order file, and to amendment 1's hash.
`amendment()` now walks the chain, checks each link's index and predecessor
hash, and accumulates re-binds (a later link wins; an earlier link's re-bind
is not silently dropped). R1 keeps retain-5 / reissue-3 / 134 s; R2–R5
unchanged — neither fix changes per-cell cost.

## Gate lines

```
BEFORE the fix (the lead's ordering):            4 failed, 54 passed
R1 suites (replay + cpu_gate + a1 + a2):         80 passed
R1 FIRST + scout_fix4 + admission (the gate):    98 passed
env after a fake batch: None | leaked GRM_ vars: NONE | admission_rule(): all_tokens_bind
collision fixture: archived to attempt_1/, fresh arms created,
                   retained receipts byte-identical, all cells have receipts
```

## Prior art (new in amendment 2)

* **GRM C7/C2 sha-bound amendment chains** (GRM contributors, 2026) — TAKEN
  verbatim, extended to chain each link to its predecessor's hash.
* **FIX8/C7 "archive the prior attempt, never overwrite"** (GRM, 2026) —
  TAKEN, applied to per-cell scratch instead of the batch controller.
* **GRM-A1 `grm_a1_gpu_contrast.pin_flags`** (GRM, 2026) — independently hit
  the same "pin AFTER `environment()`" trap, and its comment records that
  pinning before it "fails silently, and that exact mistake produced a wrong
  reading earlier in this arc". A1 pins and asserts but does not RESTORE;
  restoration is the half this amendment adds. Found via the local
  code-similarity index, not searched for — worth the lead knowing two
  independent seats hit one trap.
* **`unittest.mock.patch.dict(os.environ)` / pytest `monkeypatch.setenv`**
  (Python and pytest contributors) — the standard environment snapshot and
  restore idiom, reproduced in the worker because production has no pytest
  fixture. Nothing novel; no new algorithm.

## Process safety (amendment 2)

Nothing killed or signalled. No GPU lease taken, no GPU work run, no model
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

# Amendment 3 — a satisfied re-arm scope must stop gating the campaign

## First, the result the lead's run actually produced

Batch R1's re-issue **COMPLETED**: 8/8 cell receipts, **parity 8/8**, ON plans
match the registered margin_first plans **8/8**, every transition
`unchanged_correct`, **0 answers changed**, lease released at 79.1 s. The
amendment-1 device fix is confirmed by its own receipts — payloads freed per
arm (23/23, 14/14, 17/17) and device memory **flat at 10996 MiB across all
three cells** rather than climbing. That is the first real evidence that the
OOM mechanism is actually fixed, not merely argued.

Campaign now: **8 of 31 executions measured**, parity clean, `NOT_MEASURED`
overall (correct — 23 cells remain).

## The defect

`--batch R2` refused before running:
`R1_AMENDMENT_RETAIN_MISMATCH: R1 on_disk=[8 cells] amendment=[5 cells]`.

`resume_scope()` re-applied amendment 1's retain/reissue cross-check on every
invocation. That check is a **precondition** — "is the state I am about to act
on still the state this amendment was written against?" Once R1 completed,
the scope was **satisfied**: all 8 registered cells had receipts, the exact
outcome the amendment existed to produce. Re-evaluating a precondition against
the state its own action produced closed the campaign on its own success.

Amendment 2 fixed a self-collision on scratch; this is the same family one
level up, and it is mine.

## Fix

The cross-check applies only while the re-armed batch's controller is not
COMPLETE. Once complete, `resume_scope()` returns `scope_satisfied: true`,
an empty `reissue`, and `receipts` mapping every cell id to its receipt
sha256. `batch()` treats a satisfied scope as a record, never a re-issue
instruction.

**Not relaxed:** an OPEN batch's cross-check is unchanged; a COMPLETE
controller with cells still missing is a new RED
(`R1_AMENDMENT_SCOPE_UNSATISFIED`); per-receipt binding and parity checks
still run on satisfied scopes. Four tests cover those.

## A fifth defect, found while verifying

`summary()` reported **79.1 s** where `campaign_state()` reported **476.1 s**:
amendment 2's archived-charge fix had gone into `campaign_state` only. Fixed;
both now report 476.1 s and a test asserts they agree. A fix applied to one
reader of a fact has to be applied to every reader of it.

## Amendment

`artifacts/grm_r1/amendment_3.json`, sha
`e23a91c8d267b185a589cc67b71a1df5fb96dd4adca3fd542519fce00448f349`, chained to
amendment 2, re-binding the worker. R1's re-arm entry is kept verbatim for the
record and annotated `rearm_status.R1.satisfied: true` with all 8 receipt
hashes. R2–R5 unchanged; no budget change.

## Gate lines

```
RED-before / GREEN-after on one state:
  WITH amendment 3   : resume_scope OK, scope_satisfied = True
  WITHOUT amendment 3: R1_AMENDMENT_RETAIN_MISMATCH: R1   <-- the lead's stop

live receipts: scope satisfied True, reissue 0, receipts 8, complete ['R1']
R2 preconditions: scope None (normal first run) | priors complete True
                  budget 476.1 + 194 = 670.1 <= 3600
campaign_state 476.1 s == summary 476.1 s (agree)
summary: NOT_MEASURED, 8/31 measured, parity True
full gate in the leak order: 110 passed
```

## Prior art (new in amendment 3)

* **GRM C7/C2 sha-bound amendment chains** (GRM contributors, 2026) — TAKEN
  verbatim, including chaining to the previous amendment's hash.
* The **precondition vs postcondition** distinction (a guard checked before an
  action must not be re-evaluated against the state that action produced) is
  ordinary defensive-programming practice. **No specific prior art known to
  me** for this composition, and **no new algorithm**.

## Process safety (amendment 3)

Nothing killed or signalled. No GPU lease taken, no GPU work run, no model
loaded. Every Bash call foreground and completed. No git command run.

---

# Amendment 4 — idle-card pre-check + NON_FIT rail

## The counts settle the premise (item 1)

`--batch R4` died **13.4 s** into its lease with `cudaMalloc failed: out of
memory`, after the copytree and before any probe line: 0 cell receipts, only
the first cell's `off` session dir. Node and payload counts from the recorded
checkpoint manifests:

| batch | cells | max nodes | max npz MiB | batch total MiB |
|---|---|---|---|---|
| R1 | 8 | 25 | 36.7 | 221.5 |
| R2 | 5 | 25 | 36.7 | 130.0 |
| R3 | 8 | 25 | 36.6 | 220.8 |
| **R4** | 8 | 25 | 36.6 | **151.6** |
| R5 | 2 | 3 | 6.1 | 12.3 |

**This is not a fit problem.** R4's first cell (25 nodes / 36.6 MiB) is the
same size as `profile-census-2--e2e_t22_mira_seal`, which R3 **completed** on
the same profile frame. R4 is the *smallest* profile batch by total payload.
Against ~1,286 MiB of headroom, the largest single cell is 2.8% (5.7% for
both arms) — roughly **35x margin**.

## Lever chosen (item 2): (b) only; (a) declined with its receipt

Lever (a), host-resident payloads, is **not implemented**. It would add a
page-in path to solve a problem the evidence says does not exist, and every
added path is another guard to get wrong — a lesson this arc has taught
repeatedly. Recorded in the amendment as
`lever_2a_host_resident_payloads.implemented: false` with the reasoning.

**Idle pre-check.** Keyed on TOTAL framebuffer used, never on the
compute-process list — the holder that broke R4 listed none, and FIX-8's
`parse_memory` states exactly that rule. Bounded by `--idle-wait`, read-only,
and it **only declines to start**: it never signals, kills or clears
anything. A test asserts the source contains no `os.kill`/`SIGTERM`/`pkill`.

**NON_FIT rail.** An OOM at load/harvest writes a create-only receipt
(reason, stage, memory snapshot) and the batch continues. `is_oom` is
narrow: any other error, and the parity RED, still stop the campaign. NON_FIT
is **unmeasured** in `summary()` and never counted as `unchanged` — a cell
that did not execute cannot be evidence that the rule left its answer alone.

## Amendment

`artifacts/grm_r1/amendment_4.json`, sha
`003539cd8afdfbac0f78ddae807271184553d73251d90566315b026a806abe6d`, chained to
amendment 3, re-binding the worker. **R4 re-armed with 0 retained / 8
re-issued** (its FAILED record archived). **R5 is not re-armed** — it never
started, so it is a normal first run. Budget unchanged; neither lever changes
per-cell cost.

## The guard that caught me

The full gate came back **1 failed**: my own no-retry test from amendment 1
flagged `while True:` in the new `await_idle`. The loop was genuinely bounded
by a deadline, but the test is right to be strict, so rather than loosen it I
rewrote the wait as a **counted `for` loop** — the bound is now structural,
and "waiting on a resource someone else holds" is visibly different from
"retrying our own failed work" in the code rather than in a comment. I also
tightened the test: `run_cell` must have exactly one call site.

## Gate lines

```
OOM site        : 13.4 s elapsed, 0 receipts, first cell 'off' dir only
counts          : R4 max 25 nodes / 36.6 MiB; R3 completed the same size
headroom        : 36.6 of ~1286 MiB = 2.8% (~35x margin) -> lever (a) declined
amendment 4     : R4 retain 0 / reissue 8; R5 not re-armed (normal first run)
campaign        : charged 988.5 s, complete ['R1','R2','R3']
R4 priors       : all complete; budget 988.5 + 238 + 124 = 1350.5 <= 3600
summary         : NOT_MEASURED, 21/31 measured, parity 21/21, non_fit 0
transitions     : 20 unchanged_correct, 1 wrong_to_correct, 0 correct_to_wrong
amendment tests : 20 passed
full gate (leak order): 130 passed
```

## Prior art (new in amendment 4)

* **GRM FIX-8 `grm_scout_fix8_resume`** (`parse_memory` / `memory_gate` /
  `snapshot`, GRM contributors, 2026) and **NVIDIA nvidia-smi XML
  framebuffer/process reporting** (docs.nvidia.com, accessed 2026-09-09) —
  TAKEN: the XML fields, the conservative total-used reading, the 1000 MiB
  registered limit, and the explicit warning against inferring idleness from
  an empty compute list. That warning is the whole reason this gate is
  correct for the R4 case.
* **GRM C7/FIX8 create-only failure receipts** (GRM, 2026) — TAKEN: record a
  failure as evidence rather than retrying it.
* **OURS**: the NON_FIT class and its unmeasured accounting, and the use of
  the idle probe as a pre-lease gate in this worker. **No prior art known to
  me** for that composition. **No new algorithm.**

## Process safety

Nothing killed or signalled. No GPU lease taken, no GPU work run, no model
loaded; the only device interaction was the read-only `nvidia-smi -q -x`
probe. The worker was deliberately left unedited until the lead reported the
R4/R5 run finished. Every Bash call foreground and completed. No git command
run.
