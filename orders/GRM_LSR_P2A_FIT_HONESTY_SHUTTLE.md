# GRM-LSR-P2A — Fit-stage honesty (SHUTTLE) + not-in-memory abstention

David's ruling (2026-09-01): "go, all three, shuttle over fail-loud."
This order implements Phase 2 items 1 and 2 of `docs/LSR_ADDENDUM_1.md`.
Item 3 (route-receipt persistence) is GRM-LSR-P2B, running in parallel
in a separate worktree — do NOT implement it here; do NOT touch
`core/grm_three_pass.py`.

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` (branch
`lc1-wip`) — edits, builds, test runs, and BOUNDED GPU runs AUTHORIZED.
A registered order IS the permission: run first, report after.

## Context (read fully, in order)

1. `docs/LSR_LIVED_SERVING_PLAN.md`, `docs/LSR_ADDENDUM_1.md` (Phase 1
   FINAL: ADMISSION-PRUNE ×4, NOT-YET-DEPOSITED ×2).
2. `artifacts/lsr_p1/lsr_p1_adjudication_31e5c89453f4aa42.json` — the
   per-probe evidence. The four ADMISSION-PRUNE probes share one shape:
   `policy_branch = exactly_one_identifier_decisive_rank1`,
   `rank_plan = [X]`, yet `current_planned = [X, Y, Z]`, fit kept `[Y]`
   (the wrong lineage end, 52–66 seats) and dropped `[X, Z]`. The
   answer-bearing node X was PLANNED and never received seats.
3. `core/graft_arena.py::step()` (~2422–2660): the attempt ladder,
   `fit()` (L2 resolve first, then EXPANSION-ORDER truncation against
   `width`), grounding-driven trips, `no_mount_fit`.
4. `scripts/grm_e2e_session.py::_probe_ladder_chat` (~1064–1200): the
   harness's own ladder/fit path (the lived probes went through it).
   Find where `[X, Y, Z]` came from when `rank_plan = [X]` — that
   expansion is part of the defect anatomy; name the site in the report.
5. `core/grm_admission.py` (A-DEC: `decisive_admission_profile`,
   `policy_plan`, `admission_info_fields`).
6. `docs/GRM_MOUNT_CURATION_DESIGN.md` Stage B (serve declared ties by
   shuttle) and Stage C (abstention / demand loop).

## Ruling 1 — fit-stage honesty, SHUTTLE over fail-loud

Principle: **a planned node is never displaced by an unplanned one, and
a planned set that cannot be co-seated is serialized, never truncated.**

Required behavior (both fit sites: arena `step()` and
`_probe_ladder_chat`):

1. **Seat priority = plan order.** `rank_plan` members are fitted FIRST,
   in plan order, before any expansion/topk/recency filler. Filler only
   consumes seats left over. (L2 resolution still runs first; it must
   not re-sort the plan.)
2. **Plan members that do not co-fit → shuttle.** If plan member(s)
   remain unseated after step 1, the turn becomes a SHUTTLE turn:
   additional trips, one per unseated plan member (in plan order), each
   mounting that member (plus filler that fits). Shuttle trips are
   triggered by the fit-drop itself, NOT by grounding failure — a
   wrong-but-grounded first trip must not end the turn. Compose the
   final answer from the grounded trips per Stage B (reuse the
   existing grounding-notes composition path; if none exists in the
   arena path, the shuttle serves the trip whose mount set contains
   the plan head — state which).
   Trip budget: shuttle trips are additive to `max_trips` only up to a
   hard cap of `len(rank_plan)`; register the cap before running the gate.
3. **Never silent.** Every fit decision lands in `info`:
   `fit_planned`, `fit_seated`, `fit_dropped_planned` (must be `[]`
   unless UNSEATABLE), `fit_dropped_filler`, `fit_shuttle` (bool),
   `fit_shuttle_trips` (list of mount sets), `fit_unseatable` (plan
   members whose own `ntok` exceeds the budget even alone).
4. **UNSEATABLE is explicit degrade, not substitution.** A plan member
   that cannot fit even alone is reported (`fit_unseatable`), and the
   turn serves with `served_without_plan_head = True` in `info` —
   never with a lower-ranked node standing in for it unlabeled.
5. Legacy path (`GRM_ADM_DECISIVE=0`, no admission profile): unchanged
   byte-for-byte. Add a test that pins it.

## Ruling 2 — not-in-memory abstention (structural, no thresholds)

Honest framing first: the two NOT-YET-DEPOSITED probes are NOT this
rule's targets — their identifiers bound an older lineage member that
WAS the repository's best knowledge at that moment; the harness's
deposit ordering produced the "wrong" expectation. Say so in the report;
do not engineer a rule that would make those two pass.

The rule targets the general Stage C class: a point lookup whose
identifier tokens bind NO repository node.

1. When `ordered_identifier_tokens` yields identifier tokens for the
   question AND `identified_candidates` is empty over the full eligible
   repository (not just the bounded ranking window — check the window
   bound and widen the identifier scan if needed; state the cost), the
   turn ABSTAINS: no topical-nearest mount is served as the answer.
   Serve a fixed abstention string (constant in `core/grm_admission.py`,
   e.g. `"Not in memory: no stored record matches <identifier tokens>."`)
   with `info["abstained"] = True`, `info["abstain_reason"] =
   "identifier_unbound"`, and the unbound tokens listed.
2. Deposit behavior on abstention: the user turn still deposits (the
   question is a lived turn); the abstention output does NOT deposit as
   a recall/turn graft that could later be routed as a fact. Pin with a
   test.
3. Non-identifier (ambiguous/topical) questions: NO abstention this
   round (the ambiguous-probe corpus is an open David question; no
   threshold gets registered here).
4. D-NGH is NOT wired as a behavior trigger in this round. Telemetry
   only if trivially available; otherwise leave for the Stage C wiring
   order.

## Gates (run synchronously, FOREGROUND; RED is a result)

G1. `python3 -m pytest -q tests/` — full suite green, plus the new
    tests: plan-order seating; shuttle on non-co-fit plan set
    (declared_synthesis, two 60-seat members, width 96 → two trips,
    both members served); UNSEATABLE explicit degrade; legacy path
    byte-identity; abstention on unbound identifier; abstention
    non-deposit; bound identifier NEVER abstains.
G2. **Lived census replay (GPU, fork-from-snapshot, never rebuild):**
    re-serve the 19 lived probes of the DET1.11 census
    (`artifacts/grm_det1/run_20260831T160525Z_2/det1_11/census/`;
    serving path per `scripts/lsr_p0_gpu.py` / `scripts/grm_det1_5_gpu.py`
    — reuse, don't reimplement) with the fix ON.
    Registered prediction (lead, before the run): the 4 ADMISSION-PRUNE
    probes (meridian docket, juniper pass, falcon registry, tundra
    ledger) serve the expected value; the 2 NOT-YET-DEPOSITED probes and
    the praxis control are unchanged; the 12 passing controls remain
    passing; zero abstentions fire on the 19 (all bind identifiers).
    Pass rule: 16/19 ≥ expected with ZERO control regressions.
    Any control regression = RED, stop, report.
G3. Abstention live probe: one held-out question naming an identifier
    never deposited in that fixture (e.g. "What is the current Zephyr
    manifest value?" against the meridian fixture) → abstention string
    served, `abstained=True`, nothing confabulated. Also one bound
    identifier on the same fixture → normal serve.
G4. Value comparison uses the existing semantic comparator (unified
    comparator with negative guards from DET1) — never glyph equality.

GPU conventions: flock (`--wait`) per repo convention; single GPU;
every run bounded ≤ 10 minutes wall; gaps between runs; the operator has
absolute right of way. Xvfb/display irrelevant here.

## File boundary

Modify ONLY: `core/graft_arena.py`, `core/grm_admission.py`,
`scripts/grm_e2e_session.py`, `tests/` (new/updated tests),
`scripts/lsr_p2a_*.py` (new gate scripts), `artifacts/lsr_p2a/`,
`logs/`. Everything else read-only — in particular
`core/grm_three_pass.py` (P2B's file), `docs/`, `orders/`, all DET1/LSR
receipts and their scripts (import, never edit).

## Principles (binding)

Thresholds/caps registered in the report BEFORE gate runs, never
adjusted after. Determinism. NO git (lead commits). NO subagents.
NO monitor/watcher idle loops. RED honesty: a failed gate is a result;
tune nothing registered. Inspect-don't-trust applies to you too: verify
your own claims against artifacts before writing them.

## Done (verbatim in the final message)

1. pytest summary line (before and after).
2. The `[X, Y, Z]`-from-`rank_plan=[X]` site, named with line numbers.
3. Registered cap + prediction block, then G2 per-probe table
   (probe, expected, served-before, served-after, verdict), G3 outputs.
4. Files created/modified with a one-line purpose each.
5. New `info` fields and their semantics (P2B will persist them).
6. Ambiguities, deviations, residual risks — and anything RED.
