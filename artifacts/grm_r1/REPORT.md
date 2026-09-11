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
