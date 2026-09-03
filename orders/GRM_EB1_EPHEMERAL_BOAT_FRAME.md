# GRM-EB1 — The ephemeral boat is the production frame (spec), re-baseline both batteries

David, 2026-09-02 night, stating the spec: **"The chat log is not kept
in memory context. Any chat recall on facts is pulled via GRM, that way
the chat can grow to any length, limited only by RAM and NVMe."**

The lived frames to date violate that spec. `ArenaCache` defaults to
`ephemeral=False, live_turns=2`: the last two turns persist in the
model's live cache, and the e2e driver's `refeed_live_window` re-feeds
transcript turns into the cache on resume with `deposit=False` — a
chat log in context. P2C.1 measured the consequence on t33: the
just-deposited fact sat in the live window, router-invisible, and the
turn refused. Under the spec, that fact is a graft and routable.

This order (1) makes `ephemeral=True` the production posture on every
serving path, (2) removes the transcript re-feed from the resume path,
(3) re-baselines both lived batteries under the spec frame, and
(4) measures what the spec frame costs. A different frame is a NEW
baseline: no Arm-0 reproduction of the persistent-frame rows is
expected or claimed. Spec is law: a persistent live window is a
FAILURE of this frame, not a fallback.

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` (branch
`lc1-wip`, HEAD 787ef10) — edits, builds, test runs, and BOUNDED GPU
runs AUTHORIZED. A registered order IS the permission: run first,
report after.

## Context (read fully, in order)

1. `core/graft_arena.py::__init__` (~72–95: `ephemeral`, `live_turns`,
   `recency_mounts`, `max_live`), `step()` (~2244: "clear the boat";
   recency `rec`; `live_idx` exclusion; `fit_detail` `rec_budget`;
   `use_rec` and the point-lookup rule), `_trim_live` (~2116).
2. `scripts/grm_e2e_session.py`: `--live-turns`/`--max-live` args
   (~265–268), repository construction (~2150), `refeed_live_window`
   (~2166), the RECENCY LAW site (~1800), `recency_augmented_probe`
   (~702, exploratory), `_probe_ladder_chat`.
3. `core/grm_runtime.py::chat`, `core/graft_repository.py` constructor
   (`arena_kw`), `docs/GRM_E2E_RECEIPT_LEDGER.md` (the recency law and
   clean-room history), `docs/LSR_ADDENDUM_1.md` Phase 0 (recency
   clean-room zeroes recent turns on point lookups).
4. Harnesses that build sessions FRESH (usable for a new frame):
   `scripts/lsr_p2c_replay_gpu.py` (sup battery, deposit-order),
   `scripts/lsr_p2c_e2e_gpu.py` (sharded 34-turn census session with
   save/restore). Fork-based harnesses (SC1.2, P2C.1) restore
   persistent-frame state and are NOT usable here — say so.
5. Receipts to compare against (persistent frame): P2C Arm 1
   (`artifacts/lsr_p2c/lsr_p2c_g2_summary_*.json`), P2C.1 summary
   (`artifacts/lsr_p2c_1/`), the census
   (`artifacts/grm_det1/run_20260831T160525Z_2/det1_11/census/`).

## Mission

### Part 1 — Posture

1. `ephemeral=True` becomes the default on every production serving
   path: `GraftRepository`/`GRMRuntime` construction, the e2e driver's
   `arena_kw`, and any other constructor site (enumerate all). A
   registered escape `GRM_PERSISTENT_BOAT=1` restores the old frame
   for reproduction of frozen receipts ONLY (fail closed to ephemeral
   on an unknown token; a test pins both).
2. `refeed_live_window` is removed from the resume path under the spec
   frame (kept only under the escape). Resume = repository state, not
   transcript re-feed.
3. `live_turns` under the spec frame governs only the CURRENT turn's
   segments (question/answer of this turn); no prior turn survives in
   the live cache across turns. Pin with a test: after N turns, the
   live cache holds only turn N's segments.
4. Recency mounts (`recency_mounts=2`) stay as-is: they are grafts
   pulled from the repository, consistent with the spec. Receipt every
   turn: `frame_ephemeral`, `recency_mounted_ids`, `recency_seats`,
   `live_segments_after_turn`.

### Part 2 — Re-baseline (GPU, bounded)

5. **Sup battery** (`lsr_p2c_replay_gpu.py`-style deposit-order build,
   9 probes, fixes ON, demand OFF): served value, correct?, fit and
   grounding receipts, recency seats consumed. Then demand ON, same 9.
6. **Census session** (`lsr_p2c_e2e_gpu.py` sharded 34-turn build,
   10 probes, fixes ON, demand OFF): same fields. Then demand ON.
7. **Recency cost table**: per probe, arena width 96 minus recency
   budget actually charged (the `rec_budget` path is zero for
   identifier queries — say which probes paid and how much).

### Part 3 — The long-horizon question (bounded, sharded)

8. If it fits the bounds: extend the census session's deposit sequence
   to ≥ 100 turns by cycling its own certified fixture families with
   fresh values (registered generator, deterministic seed), probing
   every 10 turns on a fact deposited ≥ 30 turns earlier. Report
   recall vs turn distance. If it does not fit ≤ 10-min shards with
   save/restore, deliver the sharded script + blocked-report; do not
   exceed the bounds.

## Gates (synchronous, FOREGROUND, in-call waits only; RED is a result)

Register before any gate run (`artifacts/grm_eb1/registration.json`):
posture, escape, predictions below. No numeric constants beyond the
existing width/recency defaults.

G1. `python3 -m pytest -q` sharded under 10 min per call — pre-existing
    set unchanged vs P2C.1's logs; escape-frame byte-identity on a
    fixture turn; live-cache-only-current-turn test; recency receipts.
G2. **Sup battery under the spec frame**: prediction (lead) — the 4
    P2C-corrected probes stay correct (P2C's lever is frame-agnostic);
    the 3 controls correct; the 2 NOT-YET-DEPOSITED unchanged in kind.
    PASS RULE = no probe that was correct under P2C Arm 1 becomes wrong.
G3. **Census under the spec frame**: prediction — the 9 lived-correct
    probes stay correct; **t33 becomes CORRECT (Marble-4-Juliet)**
    because the fact is now a routable graft at probe time; this is the
    order's headline prediction. PASS RULE = no lived-correct probe
    becomes wrong.
G4. Demand ON arms: false fires and any recovery reported; no gate.
G5. Part 3 table or blocked-report.

GPU conventions: `/tmp/forge-gpu.lock` lease via the repo's
`gpu_lease`, single GPU, every run ≤ 10 min wall, ≥30 s gaps inside
the wrapper, operator has absolute right of way. Frozen run tree never
written.

## File boundary

Modify ONLY: `core/graft_arena.py`, `core/graft_repository.py`,
`core/grm_runtime.py`, `scripts/grm_e2e_session.py`, the two fresh-build
harnesses (frame flag + receipts only), `scripts/grm_eb1_*.py` (new),
`tests/`, `artifacts/grm_eb1/`, `logs/`. Read-only: `core/grm_three_pass.py`
(add `"frame_"`/`"recency_"` only if a prefix is needed — one line +
test), every `scripts/grm_det1_*.py`, `scripts/lsr_*.py` other than the
two named, `scripts/grm_sc*.py`, `docs/`, `orders/`, all receipts.

## Principles (binding)

Spec is law: the persistent live window is the deviation; do not
"keep it for safety". Thresholds carried. Determinism. NO git (lead
commits). NO subagents. **NO background processes for waiting**: no
`run_in_background`, no watcher/poll loops, no "block until X exits"
helpers; foreground only, wait in-call, every Bash call under 10
minutes. **Never kill, signal, or interrupt any process you did not
start** (multiple projects share this machine; no kill/pkill/killall/
fuser -k/gpu-reset/systemctl; pgrep/ps hits are off-limits; wait on the
flock or report blocked). Acknowledge process safety in the report.
RED honesty. Verify your own claims against artifacts before writing.

## Done (verbatim in the final message)

1. pytest summary line(s) + pre-existing set status.
2. Registration block as written before gates.
3. Every constructor/driver site changed to the spec posture, and the
   escape's sites.
4. G2 table (9 × served / correct / P2C-Arm-1 value / change /
   recency seats), G3 table (10 × served / correct / lived / change /
   recency seats) with t33 called out, G4 demand-ON summaries, G5
   table or blocked-report.
5. Files created/modified; new receipt fields.
6. Ambiguities, deviations, residual risks, anything RED; process-
   safety acknowledgement.
