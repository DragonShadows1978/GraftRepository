# importance_convos — WO-4 labeled conversation fixtures

Ground-truth-labeled conversation set for the GRM importance-weighting
program's **G1** (signal agreement) and **G2** (prospective discriminator)
gates. See `docs/GRM_IMPORTANCE_PLAN.md` at the repo root — that plan is
law; this README documents the fixture format it consumes, nothing more.

This directory contains **static data only**: JSON conversation files, this
README, and a stdlib-only validator (`validate_fixtures.py`). No model
imports, no engine imports, no GPU. Nothing here runs a graft arena or a
transformer.

## Files

- `convo_01_ops_standup.json` — engineering standup chatter
- `convo_02_travel_logistics.json` — personal travel planning
- `convo_03_household_planning.json` — household/family logistics
- `convo_04_worldbuilding_session.json` — fiction worldbuilding brainstorm
- `convo_05_budget_review.json` — personal finance review
- `convo_06_wellness_tracking.json` — wellness/appointment tracking
- `validate_fixtures.py` — schema + cross-field validator, stdlib only

Six conversations, 23-24 turns each, 3 probes each (18 probes total).

## Node classes

Every turn is labeled with exactly one `node_class`:

- **PROBED_LATER** — an ordinary fact stated early, probed by a later
  turn. Graded relevance vs distractors (the G1 rank-correlation class).
- **STANDING_PREF** — the user states a lasting preference or standing
  instruction, which then goes **completely unused for >= 10 turns**
  before a probe exercises it. This is the **G2 discriminator class**:
  it exists to be invisible to a purely retrospective usage signal (S1)
  and to test whether a prospective signal (S2) ranks it above filler
  anyway. The plan pre-registers that S1 is expected to score this class
  at floor — that is a *designed* failure this fixture makes legible,
  not a bug in the fixture.
- **FILLER** — chit-chat with no probe ever targeting it. Present in
  every conversation as the volume that pushes real facts out of any
  naive recency window, matching the house E4 fixture style.
- **SUPERSEDED** — a fact stated, then explicitly corrected at least once
  later in the same conversation under the *same* `node_id`. A probe
  targeting that `node_id` wants the correction's answer, not the stale
  original; the original still shows up as a low-to-mid relevance
  candidate (it is topically on-target, just wrong/outdated).
- **PROBE** — the probe question itself (always the `user` turn that asks
  it; the model's actual answer is not scripted here — the harness that
  consumes this fixture generates it and grades hit/miss or logs the
  importance signals). `node_id` is always `null` for PROBE turns.

`node_id` is a free-form string that names a *fact instance* and is
shared across every turn that touches the same fact (this is how
SUPERSEDED's original + correction turns link up, and how a probe's
`relevance` map points back at the planting turn). Non-fact classes
(`FILLER`, `PROBE`) always carry `node_id: null`.

### Turn-pairing convention (read this before writing a G2 harness)

This repo deposits conversation turns as `(user, assistant)` pairs into a
single graft node — see `tests/test_graft_e4_conversation.py`
(`turn_text(u, a)` / `deposit(text, graft=True)`) and the arena tests that
reuse it. A `STANDING_PREF` user turn and its immediate assistant
acknowledgment (e.g. "Understood, noted...") are **the same node**, not
two touches. The zero-use guarantee is about turns *strictly after* that
pair, up to (not including) the probe turn — not about the assistant's
same-turn acknowledgment. `validate_fixtures.py`'s isolation check only
inspects labels/ids (it cannot read intent out of prose), so this
convention is enforced by authoring discipline, documented here, and
spot-checked by grepping preference keywords across the intervening span
during authoring — all six fixtures were checked this way and the only
hits were each preference's own turn 4 acknowledgment.

## Schema

Top level:

```jsonc
{
  "conversation_id": "convo_01_ops_standup",   // must match the filename
  "description": "...",                          // human-readable summary
  "turns": [ ... ],                               // ordered turn objects
  "probes": [ ... ]                               // probe objects
}
```

Turn object:

```jsonc
{
  "turn_id": 1,              // positive int, strictly ascending, unique
  "role": "user",            // "user" | "assistant"
  "text": "...",             // non-empty string
  "node_class": "PROBED_LATER",  // one of the 5 classes above
  "node_id": "halcyon_codename"  // string for fact classes, else null
}
```

Probe object:

```jsonc
{
  "probe_turn_id": 19,           // turn_id of the PROBE-class turn asking it
  "question": "...",             // must equal that turn's "text" verbatim
  "expected_answer_tokens": ["halcyon"],  // accept-substring list, house
                                            // style: lowercase, matched via
                                            // `any(s in answer.lower() for
                                            // s in expected_answer_tokens)`
                                            // exactly like tests/test_graft_e4_*.py
  "relevance": {                 // candidate turn_id (as string key) -> grade
    "1": 3,
    "3": 0,
    "5": 0,
    "7": 1,
    "9": 0,
    "11": 0
  }
}
```

`relevance` grades are **0-3, graded not binary**:

- `3` — this turn is the direct, current answer to the probe.
- `2` — this turn is moderately relevant (same topic, secondary detail,
  or a stale-but-plausible-sounding value).
- `1` — this turn is weakly/tangentially relevant (same broad context,
  wrong specific fact).
- `0` — this turn is irrelevant to the probe (pure distractor, usually
  FILLER).

Every probe has **>= 4 graded candidates** and **>= 3 distinct grade
values** among them (enforced by the validator) — this is what makes
Spearman rank correlation against S1/S2/S3 well-posed for G1; a candidate
set that was all 3s-and-0s would collapse to a binary agreement check,
which is not what the plan asks G1 to measure.

All `relevance` candidate turn_ids must be **strictly before** the probe
turn (a probe cannot be "relevant" to something that hasn't happened
yet), and the `question` text must match the corresponding turn's `text`
field exactly, so a harness can source the probe prompt from either the
`turns` list or the `probes` list interchangeably.

### Style

Facts follow this repo's E4 house style
(`tests/test_graft_e4_conversation.py`, `test_graft_e4_arena.py`,
`test_graft_e4_consolidated.py`): identifier-shaped plantables (rack/lab
codes like `BX-44`/`LP-2231`, gate codes, dollar amounts, times/dates,
invented proper names for people, hotels, alien factions/worlds) mixed
with relational facts (who quoted what, which rank a character holds),
embedded in otherwise mundane first-person chatter (coffee machines,
parking garages, gym classes, pets). Identifier-shaped tokens were
checked for collisions **across** conversations (not just within one) —
none exist; each conversation's codes are unique to it, keeping this
fixture set out of the corpus-100 cross-conversation-confusability
regime, which is explicitly out of scope for WO-4.

## Validator

`validate_fixtures.py` is stdlib-only (`json`, `glob`, `os`, `sys`) — no
`import torch`, no `tensor_cuda`, no `core.*`, no network, no GPU. It
checks, per file:

- Required top-level keys present; `conversation_id` matches the filename.
- Turn count in `[15, 25]`; `turn_id`s ascending, unique, positive ints.
- `role` in `{user, assistant}`; non-empty `text`; valid `node_class`.
- `node_id` is `null` for `FILLER`/`PROBE`, a string for fact classes;
  `SUPERSEDED` node_ids may repeat (original + correction), other fact
  classes may not reuse a node_id.
- At least 3 probes per file; every `probe_turn_id` refers to a real
  `PROBE`-class turn; `question` matches that turn's text.
- `expected_answer_tokens` is a non-empty list of non-empty strings.
- `relevance` has >= 4 candidates, all referencing real turns strictly
  before the probe, each graded in `[0, 3]`, with >= 3 distinct grades
  present per probe.
- All 5 node classes appear somewhere in each conversation.
- Every `SUPERSEDED` node_id is touched >= 2 times (statement +
  correction).
- Every `STANDING_PREF` node_id: never reused by another turn (single
  touch = zero-use until probed); is the **top-graded** candidate in at
  least one probe's `relevance` map (its "designated" probe — the
  intended answer, not an incidental low-grade distractor mention
  elsewhere); and that designated probe sits **>= 10 turns** after the
  planting turn.

Run it from this directory (or pass a path):

```bash
python3 validate_fixtures.py .
```

Exit code 0 = all files valid (also prints a per-file class-count
summary and a total probe count); exit code 1 = at least one file
failed, with the first violation per file printed to stdout.
