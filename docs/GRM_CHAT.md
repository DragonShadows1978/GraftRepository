# GRM chat — `scripts/grm_chat.py`

A chat session whose memory is the GRM repository, not the context window.
The chat log is **never** in the model's context: each turn the model reads
the current question and the routed grafts, nothing else. After it answers,
the turn is deposited into the repository so later turns can route to it.

This is the product surface over the ladder the C2 / C7 / LT1 batteries ran.
It does not fork a second serving path.

## Run it

```bash
GRM_PROFILE=eb1_c2 python3 scripts/grm_chat.py --repo ~/grm_sessions/notes
```

That is the whole thing. The directory is created if it does not exist and
resumed if it does — the repository *is* the session, so closing the process
and running the same command tomorrow picks up every fact you told it.

`--resume` makes an existing repository mandatory (exit 2 if there is none),
which is what you want in a script.

Commands inside the session:

| command | what it does |
|---|---|
| `/status` | seats used, arena width, mounted node ids, admission rule, the resolved flag set |
| `/recap`  | the LT1-style five-decision recap, asked over memory; **not deposited** |
| `/restart`| closes and reopens the repository in-process; prints nodes before/after |
| `/help`   | the command list |
| `/quit`   | flush, close, exit |

Batch mode for gates — one user turn per line, `/` commands honoured,
`#` lines ignored:

```bash
GRM_PROFILE=eb1_c2 python3 scripts/grm_chat.py \
    --repo artifacts/grm_p1/gpu_session \
    --transcript fixtures/grm_p1/smoke_transcript.txt
```

No GPU? `--fake-model` swaps in the CPU stub reader (`grm_c7_diagnose`).
Every other object — repository, arena, routing, admission, mounting,
deposit, persistence — is the production one. A `--fake-model` answer is
evidence about plumbing and nothing else.

## What the switch does

`GRM_PROFILE` is one name for the whole registered flag set.

**GRM-D2 (2026-09-11): `eb1_c2` IS the shipped default.** See
[`GRM_DEFAULTS_2026-09-11.md`](GRM_DEFAULTS_2026-09-11.md) for the decision,
the evidence per flip and the rollback. Before D2 the defaults column below
was what an unset environment gave you; it is now the named profile
`legacy_256`.

| | `GRM_PROFILE=legacy_256` | **unset / `eb1_c2`** (default) |
|---|---|---|
| arena width | 256 | **96** |
| capture pin | off | **live** |
| seat near live | off | **on** |
| RT1 rule | on | on |
| admission rule | `all_tokens_bind` | **`margin_first`** |
| F1 fold-retain | off | **on** |
| F2 fold alias guard | off | **on** |
| F5 sole-binder insurance | off | **on** |
| A1 alias fold-merge | off | **on** |
| frame | EB1 ephemeral boat | EB1 ephemeral boat |

`eb1_c2` is the C2 registry entry `gpt-oss-20b-eb1-w96-live-rt1`
(`config/grm_eb1_profile_registered.json`) plus the margin-first admission
rule LT1 registered its receipts under.

**The one-line rollback:**

```bash
GRM_LEGACY_DEFAULTS=1 python3 scripts/grm_chat.py --repo ~/grm_sessions/notes
```

That restores every pre-round-2 default at once — profile, admission rule
and all four flags. Each also keeps its own `=0` / old-value setting, and an
explicitly set flag outranks the umbrella, so
`GRM_LEGACY_DEFAULTS=1 GRM_ALIAS_FOLD_MERGE=1` is legacy-everything-except-A1
(which is how you bisect a regression to one flip).

The four flag flips ship **as a set**: LT1.1 r3 measured three of the four
without the alias merge REGRESSING aliases 10/10 → 7/10. Never ship a
subset; the banner says `MIXED — NOT a measured configuration` if you do.

Two properties worth knowing:

* **Ambient switches are stripped.** Resolving a profile removes every
  `GRM_*` variable already in your shell before pinning the profile's own,
  so a stale `GRM_SEAT_NEAR_LIVE=0` cannot half-apply a profile.
* **An unknown name is an error**, not a fallback to defaults. A profile you
  can silently mistype is a profile you cannot trust a receipt from.

`--pin-flag NAME[=VALUE]` pins an extra named flag. Whether a flag is
*effective* is decided by looking for a reader on the tree, never by a list:
one that nothing reads is reported in the startup notes as absent and is
**not** set, and one that has a reader prints as `(pinned, effective)`.

`--pin-flag GRM_ALIAS_FOLD_MERGE` pins A1's alias fold-merge
(`core/grm_alias_fold.py`). **Since GRM-D2 it is ON by default**, so the pin
is now mostly a way to state the condition explicitly on a receipt; to turn
it OFF, set `GRM_ALIAS_FOLD_MERGE=0` (or the whole umbrella). Before A1
landed this same resolver reported the flag as absent; that transition, and
this one, needed no code change here.

The resolved flag set prints at the top of every session, so any receipt you
paste carries the conditions it was taken under.

## The receipts

Every turn appends one line to `REPO/session_ledger.jsonl`:

* `schema: "grm.chat_turn.v1"` — turn number, the user text, the answer,
  `mounted_ids` (which grafts the model actually read), `deposited_node_id`,
  the repository size, the admission rule and the profile.
* `route_receipt` — a full `grm.route_receipt.v1` record built by
  `core.grm_three_pass.build_route_receipt`, the same schema the batteries
  emit. `route.ranking_ids` is what routing ranked, `admission.policy_branch`
  is which admission branch fired, `fit.seated` is what actually fit in the
  arena. A turn that mounts nothing still gets a receipt with an empty
  ranking — a session ledger has no holes.

`/restart` appends a `grm.chat_event.v1` line with the node count before and
after and the old and new process ids.

Reading a ledger:

```bash
python3 -c "
import json
for line in open('REPO/session_ledger.jsonl'):
    r = json.loads(line)
    if r.get('schema') == 'grm.chat_turn.v1':
        print(r['turn'], r['mounted_ids'],
              r['route_receipt']['admission']['policy_branch'],
              repr(r['answer'][:50]))
"
```

## Why there is no chat log in the context

`ArenaCache._attempt` builds the live prompt as
`self.encode(self._format_step_prompt(user_text))`, and
`_format_step_prompt` with the Harmony template returns
`harmony_turn(user_text, None)` — system prefix, **this turn's question**,
assistant header. There is no parameter through which prior turns could
arrive as text. Under the EB1 ephemeral frame `eb1_begin_turn()` clears the
live window each turn and nothing refeeds it; `refeed_live_window` is the
persistent-frame escape and this surface never calls it.

`tests/test_grm_chat.py` asserts this three ways: structurally on the prompt
builder, over every prompt served during the 20-turn smoke, and — the part
that matters — by proving the assertion can fail, on a deliberately forged
prompt.

## Gates

```bash
python3 -m scripts.grm_p1_smoke          # 20-turn fake-model smoke
python3 -m pytest -q tests/test_grm_chat.py
```

The smoke gates **plumbing**: mounts happen, a route receipt per turn, the
restart retains the repository, post-restart turns still route, no live
history leak, the recap is not deposited. The value-span scorer
(`grm_lt1.score`, the C5 arm-S rule) runs over the recall turns and its
result is **reported, never gated** — the stub reader is a regex that copies
a visible value, so its hit rate says nothing about GRM.

Quality is the registered GPU smoke: `artifacts/grm_p1/lead_commands.txt`
(≤ 0.2 GPU-h per arm, lead-run, gate = fresh recall ≥ 3/4).

The fake smoke runs two arms, `--arm A` (as r2) and `--arm A+`
(`GRM_ALIAS_FOLD_MERGE` pinned); `--arm all` is the default.

## Measured on GPT-OSS-20B (r2, 2026-09-10)

The first clean end-to-end run of this surface on the real model: one
lease, ~6 min, 18 chat turns under `GRM_PROFILE=eb1_c2`.
Receipts: `artifacts/grm_p1/gpu_smoke_r2.log`,
`artifacts/grm_p1/gpu_session/session_ledger.jsonl`.

| | result |
|---|---|
| **fresh facts** | **3/4 — GATE PASS** |
| alias | 1/1 |
| corrections | 0/2 |
| recap | **crashed** (see below) |
| route receipt every turn | yes |
| restart retained | yes (17 → 17 nodes) |
| live-history leak | none |

Per probe: t6 `Auric-4-Alpha` ✓ · t14 refused, "I don't have that
information" · t16 `Nadir-1-Delta` ✓ · t17 `Vortex-3-Sierra` ✓ · alias t18
`Nadir-1-Delta` ✓ · corrections t11 answered the **stale** `Auric-4-Alpha`,
t15 refused.

Read honestly: fresh recall and the alias worked; **both corrections
failed, in the two different ways LT1 predicted** — one stale value, one
refusal. Four fresh probes is a small sample and 3/4 is one miss from the
floor. Corrections remain the residual.

**The recap turn crashed**, and the run had no error isolation, so it left
no ledger row — which is why the r2 scorer died with `KeyError: 19` looking
for a 19th turn among 18:

```
core.grm_admission.AdmissionPolicyError: production route ranking differs
from frozen A-DEC score reconstruction: backend=native
production=[18, 21, 13, 8, 15, 16] reference=[18, 21, 20, 13, 8, 15]
```

Mechanism, diagnosed but **not fixed** (it is in read-only `core/`, and it
is not a defect of this surface): node 20 is a **degenerate fold** —
`kind=digest`, `ntok=4`, text exactly `"ARCHIVE NOTE."`, a librarian
consolidation that emitted its header and no content. Its manifest entry
has `cent = None` (no route centroid) but `rare = ['archive','note']` and a
live `native_node_id`. The native router omits a node with no route key;
the Python reconstruction in `decisive_admission_profile` scores it from
the lexical channel alone and ranks it third. The two disagree and the
integrity guard correctly refuses to proceed. It surfaced here and in no
prior battery because that reconstruction only runs under `margin_first`
(or ≤ 16 eligible nodes) — and `margin_first` is exactly what `eb1_c2`
pins. **Open for the lead:** whether a centroid-less digest should be
route-eligible at all. Until it is answered, a session long enough to
trigger a degenerate fold can abort a turn under `eb1_c2`.

Two surface fixes shipped in response (r3): a turn that raises now writes a
ledger row with `answer: null` and an `error` field instead of escaping the
loop, so the batch completes and the failure is *in* the ledger; and the
interactive loop reports a failed turn and hands the prompt back rather
than ending the session. The registered scorer finds the recap by its
question text, never by a turn index.

## Known residuals

Measured on LT1's 200-turn natural conversation, 2026-09-09, profile arm
(`/mnt/Shared/GRM_LT1_200Turn_Conversation_Result_2026-09-09.md`):

* **Fresh facts are essentially solved: 14/15.** Recall stays flat out to
  150 turns back.
* **Corrections: 5/10.** A value that was stated and later revised is
  recalled correctly about half the time. Superseded values are the residual
  C7 isolated; the mechanism is open.
* **Aliases: 5/10** (LT1). "Call the Vega station the Hub" then asking about
  the Hub requires composing two records. The fold-merge design
  (`ALIAS_DESIGN_OPTIONS.md`, fold-merge first, two-hop read as fallback)
  **has since landed as A1** (`core/grm_alias_fold.py`, default OFF) and is
  reachable here as `--pin-flag GRM_ALIAS_FOLD_MERGE`. The r3 GPU smoke runs
  it as a second arm against the same transcript. r2 (flag off) got the
  single alias probe right, so one probe cannot separate the arms — treat
  the pair as a first exposure, not a measurement.
* **The recap scored 0/5.** The profile listed five real quantities from
  memory, but not the five decisions the fixture had registered. LT1's
  conclusion is that "the five biggest decisions" is under-specified as a
  question, not that recap is broken. Treat `/recap` as a memory dump to
  read, not a scored answer.
* **Zero abstentions, so every miss is a wrong value.** The fixture had no
  abstention clause, and the plain prompt fabricates on unanswerables rather
  than saying it does not know. Whether to add an abstention instruction is
  an open decision (it trades recall against fabrication).
* **`margin_first` IS the default, as of GRM-D2 (2026-09-11).** The gate
  named here — replaying the 31 historical C2 plans the rule changes — was
  run and passed: `artifacts/grm_r1/summary_lead.json` reports status
  `ADOPT`, 31/31 measured, off-plan parity true, **0 correct→wrong**, 1
  wrong→correct. The rule it replaced admitted 0 of LT1's 35 natural
  questions. `GRM_LEGACY_DEFAULTS=1` (or
  `GRM_ADMISSION_RULE=all_tokens_bind`) selects the old rule.

**Native publication (fixed 2026-09-10, first GPU smoke).** The first GPU
run went RED at the first recall with `RuntimeError: graft 0 has no
native_node_id` from `_commit_native_mount`. `native_node_id` is assigned by
`GraftRepository._native_sync_node`, reached via `_mark_mutations` inside
`repo.runtime._finish_turn_event(...)` — the funnel every production deposit
passes through (`grm_e2e_session._probe_finish_deposit` calls exactly it).
The surface deposited its EB1 turn with `arena.feed()` and stopped there, so
fed nodes were never published to the native store and the next turn could
not mount them. **A battery may stop at `feed()` because it replays a frozen
fixture; a product may not.** LT1 hit the same wall on 2026-09-09 and
answered it with a harness workaround
(`grm_lt1_amendment4.install_native_publication` monkey-patches
`_commit_native_mount` to publish lazily) — right for a fixture replayer,
wrong for a product. The fix calls the real funnel; **`core/` is unchanged**.
Receipt: `tests/test_grm_chat_native_publication.py`.

A consequence worth knowing: the funnel also runs `repo._librarian()`, whose
**consolidation folds** (`ArenaCache.DIGEST_PROMPTS`) quote their source
grafts into their own prompt on purpose. That is a separate GRM operation,
not the chat log entering context, so the leak test classifies and reports
folds separately rather than widening the chat-turn rule — the strict rule
stays the default, and an unlabelled prompt is always checked strictly.

One more, found by this surface's own smoke and fixed here rather than in
the frozen fixture: `grm_c7_diagnose.CPUArena.deposit` overrides
`ArenaCache.deposit` without the `_bump_cuda_gqa_epoch()` call the base
makes, so the epoch-cached candidate base never invalidates and `route()`
returns `[]` for the life of a fake session. `grm_c7_diagnose.py` is a
SHA-frozen input of the C7-r3 and LT1 registrations, so `grm_chat.py`
restores the bump in a subclass instead. **Production is unaffected** — it
runs `GptOssGQAArenaCache`, whose inherited `deposit` bumps correctly.

## Prior art

Reused from this tree (GRM contributors, 2026), unchanged: the serving
ladder (`grm_e2e_session.load_model_and_repo`, `_probe_ladder_chat`), the
EB1 complete-turn deposit (`harmony_turn` + `arena.feed`, exactly as
`grm_lt1_worker.execute` does it), the route receipt
(`core.grm_three_pass.build_route_receipt`, `grm.route_receipt.v1`), the
value-span scorer (`grm_lt1.score`, C5 arm S), the C2 profile registry and
env construction (`grm_c2_profile.select_profile`, `grm_c2_cells
.environment`), and the CPU stub reader (`grm_c7_diagnose`).

New here: the interactive surface itself, the `GRM_PROFILE` resolver, the
per-session ledger, the `/status` `/recap` `/restart` commands and the
live-prompt leak assertion. No retrieval, routing, admission or scoring
algorithm is introduced. A REPL over a retrieval memory is not a novel
shape; no prior art known to me for this exact composition — unverified,
lead to check, search terms: "retrieval-augmented chat without conversation
history in context", "KV cache graft memory interactive session".
