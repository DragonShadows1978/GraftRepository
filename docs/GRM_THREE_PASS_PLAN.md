# GRM Three-Pass Turn Architecture — Implementation Plan

Status: registered intent, IMMUTABLE after initial commit. Results
and deviations go to docs/GRM_THREE_PASS_LEDGER.md; meaning goes to
docs/GRM_THREE_PASS_SYNTHESIS.md. Branch codex/grm-three-pass,
worktree /home/vader/GraftRepository-three-pass; production main is
untouched by this program until an operator-decided merge.

## The idea (operator, 2026-07-21)

Split the GQA turn into three passes:
- PASS 1 — silent SEARCH/PREP: read the incoming turn, enrich the
  routing probe (the model reaches toward what memory would help),
  route via the exact ragged CUDA router, mount/stage the working
  set within measured walls. No user output.
- PASS 2 — the WORKING OUTPUT pass: user-facing inference over the
  prepped working set. Nothing but route-hit consumption and
  generation on the critical path.
- PASS 3 — silent MEMORY MANAGEMENT: deposit this turn's grafts,
  importance updates (S1 attention-mass harvested under a memory
  objective), supersession adjudication, eviction/tiering,
  consolidation. Emits a structured MEMORY LEDGER receipt per turn.
  No user output.

## Design anchors (receipts)

- Supersession on the turn cost ~756ms python-path (composed E2E
  receipt): pass 3 takes it off the critical path.
- S1 attention-mass importance PASS (corrected 0.8286): pass 3
  harvests it under an explicit memory objective, not as an
  answering side-effect.
- S4-as-paging CLOSED both directions ("eviction is zero-sum"):
  pass 3 makes ACTIVE explicit decisions with receipts — a different
  mechanism class than passive protection.
- Route-card line CLOSED (keys don't compress): pass 1 enriches the
  QUERY side instead; exact ragged router (18/20 dev-frame on raw
  one-token probes) is the search engine — headroom is on the probe.
- Epoch caches (route seams campaign): pass 1 PINS an epoch, pass 3
  BUMPS it — reader/writer overlap of turn N pass 3 with turn N+1
  pass 1, no hot-path locks.
- INT6 chunk-512 walls (p3x/p3y receipts): pass 1 stages against
  registered budgets; transient=wall, chunk=lever.

## Phases (one work order each; letters continue GRM convention)

- P0 SCAFFOLD: three-pass turn driver behind `turn_pipeline=three_pass`
  (default `single` = byte-identical to today, gated). Baseline
  latency receipts for the current single-pass turn (route / mount /
  infer / memory-ops stage table).
- P1 EXTRACT PASS 3: move ALL existing turn-time memory mutation
  (deposit, supersession, importance bookkeeping) into pass 3.
  Pass 2 becomes read-only w.r.t. the arena. Memory-ledger receipt
  format frozen here.
- P2 PASS 1 PREP: probe enrichment (silent prefill; enriched probe =
  registered construction, no tuning after gates) + working-set
  staging. Dev-frame retrieval comparison vs raw probes.
- P3 EPOCH OVERLAP: interleave turn N pass 3 with turn N+1 pass 1
  under epoch pin/bump; two-process interleaved determinism.
- P4 COMPOSED E2E: full session receipt on the 32K INT6 product
  frame; adoption evidence assembled for the operator decision.

## Registered gates (dev-frame; thresholds fixed BEFORE runs)

- G-SERVE (P1): pass-2 user-visible turn overhead from memory ops
  <= 5ms (vs ~756ms baseline supersession tax); end-to-end turn
  latency (all passes) <= 1.25x single-pass baseline.
- G-EQUIV (P0/P1): `single` pipeline byte-identical to HEAD;
  three_pass final arena state after pass 3 byte-equal to the
  single-pass arena state for the same scripted session (same
  decisions, different schedule) — deviations are findings, not
  tuning targets.
- G-PREP (P2): enriched-probe recall@3 >= raw-probe recall@3 on the
  registered dev query set (strictly no worse; report the delta);
  staging never exceeds the registered wall budgets.
- G-COMPACT (P1/P4): recall-quality-at-budget with pass-3 management
  >= without (fixed arena budget, registered query set); L1 decisive
  supersession test runs INSIDE pass 3 and passes; every pass-3
  mutation appears in the memory ledger (completeness audit).
- G-EPOCH (P3): interleaved two-turn scripted sessions byte-equal
  across two processes; a pinned pass-1 epoch never observes a
  concurrent pass-3 mutation (probe test).

## House rules

Standard rails: seat edits only this worktree; no commits/pushes
(lead commits); flags default off; production byte-identical;
thresholds never move after runs; red is a result; every claim names
its evidence class; memory ledger receipts are artifacts, not logs.
Fresh seeds / adoption gates are NOT part of this plan — they are a
separate operator-decided order after P4, per GRM convention.
