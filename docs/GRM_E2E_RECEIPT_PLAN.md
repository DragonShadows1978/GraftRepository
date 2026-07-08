# GRM Composed End-To-End Receipt — Implementation Plan

Status: DRAFT — immutable at initial commit. Tracking:
- Operational ledger (new): `docs/GRM_E2E_RECEIPT_LEDGER.md`
- Narrative synthesis (continues the live-model wing):
  `docs/GPT_OSS_20B_APA_GRM_SYNTHESIS.md`
House laws in force. Roles: Fable = planner/gates/ledger; Sonnet agents =
implementation, flat, no-delegation. GPU runs bounded and spaced.

## Objective

Every GRM subsystem has receipts in isolation: 96k context (gate PASS),
witnessed graft deposit, supersession, sub-ms CUDA routing (both
dialects), INT8/INT6 storage (recall-proven). MISSING: one receipt of the
composed organism — a live multi-turn session where the model decodes,
turns become witnessed grafts, cold content leaves the live window,
queries route against the repository, and the right memories remount —
continuously, in one process lifetime, with durability across a
checkpoint/reload. This work order produces that receipt.

## Shape of the receipt

A scripted session driver runs GPT-OSS-20B with GRM attached for N
scripted turns (target N ≥ 30) containing planted facts, deliberate
supersessions (facts updated mid-session), and recall probes placed
AFTER their source turns have left the live context window. Per-turn
instrumentation: route wall (CUDA path engaged, receipted), mount
count/changes, deposit wall, live-window token count, VRAM, repository
node count. Mid-session: one checkpoint + process restart + session
continuation (durability leg).

## Phases

P0 — Composition map (read-only): what the live loop needs vs what
exists. The turn-execution machinery (ArenaCache.step, mounts,
deposit paths), the GPT-OSS driver's inject_kv mount surface, the
capture-to-deposit glue (gate scripts capture offline — what does a LIVE
per-turn witnessed deposit require?), eviction/live-window policy as it
exists today, and the CUDA route opt-in path from step(). Deliverable:
gap list with file:line — what P1 must build vs merely wire. Committed
before implementation.

P1 — Session driver (scripts/grm_e2e_session.py): the scripted session,
per-turn instrumentation JSON, probe scorecard, checkpoint/restart leg.
Storage at INT8 packed (P3's format — the first production consumer).
Product-code changes only where P0 names a genuine gap; wiring over
rewriting; every mutation path stays inside existing law machinery
(epoch, WAL, supersession).

P2 — The receipt run: full scripted session on idle GPU, bounded run
segments, receipts committed. Then the honest read: what the seams cost
(route-to-mount latency, deposit overhead per turn, VRAM churn) — each
named cost becomes a registered candidate for the successor queue, with
production justification.

## Registered expectations (frozen at commit)

- E1: every recall probe whose source turn is OUTSIDE the live window is
  answered correctly via route+remount (the composed money shot).
- E2: supersession probes return the CURRENT value, never the stale one,
  including after the checkpoint/reload leg.
- E3: route wall per turn ≤ 5 ms through the production path with the
  CUDA route engaged (receipted per turn).
- E4: session completes with stable VRAM (no monotonic growth across
  turns after steady state) and the INT8 packed store round-trips the
  restart leg.
- Misses are findings: a failed probe or a leaking seam is exactly what
  this receipt exists to surface. Record and proceed.

## Out of scope

Model quality claims beyond the probe scorecard; multi-user/concurrent
sessions; the synthetic-centroid and per-graft-index successors (this
receipt feeds them evidence, not the reverse).
