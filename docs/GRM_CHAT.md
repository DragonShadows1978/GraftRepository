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

| | `GRM_PROFILE` unset | `GRM_PROFILE=eb1_c2` |
|---|---|---|
| arena width | 256 | **96** |
| capture pin | off | **live** |
| seat near live | off | **on** |
| RT1 rule | on | on |
| admission rule | `all_tokens_bind` | **`margin_first`** |
| frame | EB1 ephemeral boat | EB1 ephemeral boat |

`eb1_c2` is the C2 registry entry `gpt-oss-20b-eb1-w96-live-rt1`
(`config/grm_eb1_profile_registered.json`) plus the margin-first admission
rule LT1 registered its receipts under. **Unset is today's behaviour** —
the shipped defaults stay the defaults and this switch never changes them.

Two properties worth knowing:

* **Ambient switches are stripped.** Resolving a profile removes every
  `GRM_*` variable already in your shell before pinning the profile's own,
  so a stale `GRM_SEAT_NEAR_LIVE=0` cannot half-apply a profile.
* **An unknown name is an error**, not a fallback to defaults. A profile you
  can silently mistype is a profile you cannot trust a receipt from.

`--pin-flag NAME[=VALUE]` pins an extra named flag. If nothing on the tree
reads it, the session says so in its startup notes and does **not** set it —
`GRM_ALIAS_FOLD_MERGE` is in that state today (the alias fold-merge
mechanism is designed but not landed here).

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
(≤ 0.2 GPU-h, lead-run, gate = fresh recall ≥ 3/4).

## Known residuals

Measured on LT1's 200-turn natural conversation, 2026-09-09, profile arm
(`/mnt/Shared/GRM_LT1_200Turn_Conversation_Result_2026-09-09.md`):

* **Fresh facts are essentially solved: 14/15.** Recall stays flat out to
  150 turns back.
* **Corrections: 5/10.** A value that was stated and later revised is
  recalled correctly about half the time. Superseded values are the residual
  C7 isolated; the mechanism is open.
* **Aliases: 5/10.** "Call the Vega station the Hub" then asking about the
  Hub requires composing two records. The fold-merge design exists
  (`ALIAS_DESIGN_OPTIONS.md`, fold-merge first, two-hop read as fallback)
  but is not landed — hence `GRM_ALIAS_FOLD_MERGE` having no reader.
* **The recap scored 0/5.** The profile listed five real quantities from
  memory, but not the five decisions the fixture had registered. LT1's
  conclusion is that "the five biggest decisions" is under-specified as a
  question, not that recap is broken. Treat `/recap` as a memory dump to
  read, not a scored answer.
* **Zero abstentions, so every miss is a wrong value.** The fixture had no
  abstention clause, and the plain prompt fabricates on unanswerables rather
  than saying it does not know. Whether to add an abstention instruction is
  an open decision (it trades recall against fabrication).
* **`margin_first` is not the default.** Today's shipped admission rule
  admits 0 of LT1's 35 natural questions. `GRM_PROFILE=eb1_c2` turns
  margin-first on; adopting it as the default is David's decision and is
  gated on replaying the 31 historical C2 plans it changes.

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
