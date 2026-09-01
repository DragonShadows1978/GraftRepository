# GRM-LSR-P2B — Route-receipt persistence in the per-turn memory ledger

David's ruling (2026-09-01): "go, all three." This order implements
Phase 2 item 3 of `docs/LSR_ADDENDUM_1.md`: route receipts are not
persisted at serving time — an observability gap that cost LSR Phase 1
a CPU archaeology round. Items 1–2 are GRM-LSR-P2A, running in parallel
in the main checkout; you run in a WORKTREE.

YOUR WRITABLE TARGET is the worktree
`/mnt/ForgeRealm/GraftRepository-wt-lsr-p2b` (branch `lsr-p2b`, forked
from `lc1-wip` @ 70190d0) — edits, builds, and test runs AUTHORIZED.
A registered order IS the permission: run first, report after.
CPU-ONLY: no model loads, no GPU.

## Context (read fully, in order)

1. `docs/LSR_ADDENDUM_1.md` (Phase 1 FINAL, principle 2: "route
   receipts are not persisted at serving time").
2. `core/grm_three_pass.py::MemoryLedgerBuilder` (~438–560) — the
   step-3 memory ledger: frozen-schema mutation records, `finalize()`
   audit that every changed target is receipted.
3. `scripts/grm_e2e_session.py` — `run_step1_prep` (~932–999, the
   per-turn working-set receipt), the step-3 ledger call site
   (~1568–1709), `_probe_ladder_chat` (~1064–1200), and the arena
   `info` consumers.
4. `core/graft_arena.py::step()` (~2422–2660): `attach_route_receipt`,
   `last_route_receipt`, `admission_info_fields`, trip/`no_mount_fit`
   fields. `core/grm_admission.py::admission_info_fields`.
5. `artifacts/lsr_p1/lsr_p1_adjudication_31e5c89453f4aa42.json` —
   the fields Phase 1 had to RECONSTRUCT (admission.ranking,
   identified_candidates, identifier_tokens, policy_branch, rank_plan,
   current_planned/fitted/dropped, final_mounts, cur_mount_n, width).
   Everything in that evidence block that is knowable at serve time
   must be persisted by this order, so no future investigation
   reconstructs it.

## Mission

1. Add a `route_receipt` record to the per-turn memory ledger (step 3),
   written on EVERY turn (including turns with no mutation, abstained
   turns, and `no_mount_fit` turns). Contents, per turn:
   - route: backend, ranking (ids + scores, full bounded window),
     `candidate_count`, exclusions (live/recency), route_limit,
     fallbacks taken (with reason codes);
   - admission: identifier_tokens, identified_candidates,
     policy_branch, rank_plan, margin fields, rule sha;
   - fit/mount: planned, seated, dropped, final mounts, `cur_mount_n`,
     width, trips (mount set + grounded flag per trip), `no_mount_fit`,
     and pass-through of ANY `info` key prefixed `fit_`, `abstain`,
     `served_without_plan_head` (P2A adds these; persist generically
     so P2B needs no re-edit when P2A lands);
   - provenance: repository size at probe (`len(grafts)`), arena state
     sha before serve, `request_sha256`, `output_sha256`.
2. Schema: a NEW versioned schema constant (`grm.route_receipt.v1`),
   canonical-JSON, sha256-stamped; the existing memory-ledger mutation
   schema is FROZEN — do not change `MEMORY_LEDGER_SCHEMA` records or
   the `finalize()` audit semantics. Attach the route receipt alongside
   (e.g. `receipt["route_receipt"]`) so old consumers see identical
   mutation rows. Every existing ledger test must pass unmodified.
3. Persist on both serving paths: `grm_e2e_session.py` (harness turns
   and `_probe_ladder_chat` probes) and `core/grm_runtime.py::chat` /
   `core/graft_repository.py::chat` (the production path) — the
   runtime path writes to the repository's ledger location if one is
   configured, else returns it in `info` (state which).
4. Provide `scripts/lsr_p2b_route_receipt_dump.py`: given a session dir
   or ledger file, prints the Phase-1-style evidence block per turn
   from persisted receipts alone (proof the gap is closed).

## Gates (synchronous, FOREGROUND; RED is a result)

G1. `python3 -m pytest -q tests/` green; existing ledger tests
    (`tests/test_grm_s4_ledger.py` and any three-pass ledger tests)
    pass UNMODIFIED. New tests: receipt written on mutation-free turn;
    receipt written on abstained/`no_mount_fit` turn (simulate via
    `info`); generic `fit_*` pass-through; schema sha stability
    (canonical JSON, fixed fixture → fixed sha); old mutation rows
    byte-identical with/without the route receipt attached.
G2. Replay proof (CPU, model-free): using the persisted DET1.5 lived
    receipts that Phase 1 harvested (`scripts/lsr_p1_harvest.py`
    sources), show the dump script reproduces, field-for-field, the
    `evidence.admission.*` and `arena.*` values in the Phase 1
    adjudication for all 8 probes — or, where the lived receipts lack a
    field that the new receipt WOULD carry, list it explicitly as
    "closed by P2B, not retro-provable". No GPU; no re-serve.
G3. A fixture-only end-to-end: a `__new__`-style or lightweight arena
    fixture turn through the e2e ledger path yields a route receipt
    whose `ranking`, `rank_plan`, `final_mounts` match the arena `info`.

## File boundary

Modify ONLY: `core/grm_three_pass.py`, `core/grm_runtime.py`,
`core/graft_repository.py` (ledger wiring only), `scripts/grm_e2e_session.py`
(receipt attach only — NOT the ladder/fit logic, which P2A owns),
`scripts/lsr_p2b_route_receipt_dump.py`, `tests/`, `artifacts/lsr_p2b/`,
`logs/`. `core/graft_arena.py` and `core/grm_admission.py` are P2A's —
read-only here. `docs/`, `orders/`, all DET1/LSR receipts read-only.

Merge note for the lead: P2A edits `grm_e2e_session.py` too. Keep your
edits in that file to the receipt-attach seams (step-3 call site and
the probe path's info hand-off) and list every hunk in the report so the
lead can rebase onto P2A cleanly.

## Principles (binding)

Frozen schemas stay frozen; new facts get new versioned schemas.
Determinism. NO git (lead commits; you are on a worktree branch the lead
created). NO subagents. NO monitor/watcher idle loops. RED honesty.
Verify your own claims against artifacts before writing them.

## Done (verbatim in the final message)

1. pytest summary line (before and after).
2. The route-receipt schema (field list with types) and its sha rule.
3. G2 field-for-field table for the 8 Phase 1 probes; the
   "closed by P2B, not retro-provable" list.
4. Files created/modified; every hunk in `grm_e2e_session.py` listed.
5. Ambiguities, deviations, residual risks — and anything RED.
