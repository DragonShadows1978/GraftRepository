# GRM-SCOUT-1 — scouting the Graft Runtime Memory system: where it stands, what to do next, and what nobody has dared to try

David, 2026-09-08: "run Astra on GRM — the mission is scouting and
recommendations for next steps. The normal 'based on this the next
logical step is…' is fine, but I want crazy ideas too."

This is a READ-AND-THINK order. **NO code changes. NO test runs that
write into the repo. NO GPU.** Your only writable target is ONE new
file: `docs/GRM_SCOUT_2026-09-08.md`. Everything else in
`/mnt/ForgeRealm/GraftRepository` (branch `lc1-wip`, the canonical
home) and `/mnt/ForgeRealm/Project-Tensor` (the engine) is READ-ONLY.
Reading code, receipts, ledgers and artifacts is the whole job.

## What GRM is (in David's words, spec is law)

The chat log is NEVER kept in the model's context. Every recall of a
fact comes through GRM: attention state (K/V) captured at deposit time,
stored in a repository on NVMe, routed by a three-channel router,
mounted into an arena of fixed width when a query needs it, and
unmounted after. The chat can therefore grow to any length limited
only by RAM and NVMe. The ephemeral boat is the intended production
frame (EB1). Read the design intent first, then the receipts.

## Read in this order (all exist; do not skip)

1. `/mnt/Shared/GRM_Primer_2026-09-02.md` — the primer (audited 09-03),
   what it is, how it works, what is being built and why, open
   questions.
2. `docs/GRM_Methodology.md`, `docs/RESULTS_INDEX.md`,
   `/mnt/Shared/GRM_State_Audit_2026-08-29.md` (superseded banner —
   read it anyway for the history) — the foundations; the June
   receipts (42-turn 8/8 flat residency; turn-50 gates; MLA 1M route
   925 ms → 2.2 ms) live behind these.
3. `/mnt/Shared/GRM_Race_Report_2026-09-01.md` with addenda 2–10, and
   `docs/LSR_ADDENDUM_1.md` — the LSR Phase 2 close (plan-order fit,
   shuttle, UNSEATABLE, split-at-deposit), Stage C demand loop (D-NGH
   detector, early abort), the read-strength probes (RS1–RS4: seat
   near live band, capture pin), the width curve (WC1: 96 = plateau
   top), RT1 split-child routing.
4. `orders/GRM_*.md` newest twelve (SUP, WC1, SC1/SC1.1/SC1.2/SC2, RS,
   RT1/RT1.1, S4 WO1/WO2) and their artifacts under `artifacts/`
   (66 GB; read receipts and ledgers, not tensors).
5. `core/` — graft_arena.py, grm_admission.py, grm_demand.py,
   grm_frame.py, grm_three_pass.py, graft_repository.py,
   grm_text_norm.py; `config/grm_demand_registered.json`,
   `config/grm_live_registered_baselines.json`; the MoE bolt-on arc
   (`orders/MOE_*`, `scripts/olmoe_*`, `scripts/gpt_oss20b_expert_*`,
   `/mnt/Shared/MOE_BoltOn_Experts_README.md`) and the graft
   translation primer (`docs/GRAFT_TRANSLATION_PRIMER.md`).
6. `/mnt/ForgeRealm/AI_Research_Board.md` — the cross-track board;
   grep for GRM, LSR, Stage C, EB1, RS, RT1, WC1, D-NGH, MoE.
7. The open decision queue as recorded there and in the primer: the
   seat-near-live default, the capture pin, the demand-threshold rule
   under the spec frame, early-abort default, the demand flip, the t30
   separator rule, the live-window rank / proper-noun identifier
   channel for t33, the lc1-wip → main merge.

## Deliverable: `docs/GRM_SCOUT_2026-09-08.md`, in this shape

**Part A — Where it stands (one page).** What is proven with receipts
(cite the receipt path for every claim), what is flag-OFF and why,
what the open decisions are and what each would change. Distinguish
evidence classes (unit test / kernel gate / E2E session receipt /
model perplexity). Where the primer and the receipts disagree, say so.

**Part B — Quick surface scan (half a page).** Bugs, dead code, drift
between config and code, fragile invariants, anything a fresh pair of
eyes sees in `core/` and the newest orders. Line references. No fixes.

**Part C — The logical next steps (one page).** Ranked by expected
information gain per GPU-hour, each with: the question it answers, the
registered prediction you would make, the gate that would decide it,
the wall estimate, and what it unblocks. Include the boring ones
(flips, merges, calibrations) and rank them honestly against the
interesting ones.

**Part D — The crazy ideas (at least ten, two pages).** This is what
David asked for. Ideas that challenge the premises, cross tracks
(APA single-pass ← SP3/SP4G/SP5 reports on /mnt/Shared; MoE bolt-on
experts; graft translation across models; the GRAPA native-LLM
program), borrow from outside (databases, operating systems, cognitive
science, compression, control theory, immunology, anything), or invert
the architecture. Rules for each idea: state it in two sentences; say
what would be true if it worked; name the cheapest falsifying
experiment (one GPU-hour or less if possible); name what kills it;
give a plausibility score you actually believe (10–90 %). Spec is law
for the current system, but a PROPOSAL may question the spec if it
says explicitly which article it questions and why. Rank Part D by
upside × plausibility, and mark the three you would run first.

**Part E — Prior art (half a page).** For every mechanism GRM relies
on and every idea in Part D, cite what is known (paper/system/year) vs
what is David's or the house's own, or say "none known" / "unverified
— lead to check". No network in your sandbox; give search terms.

## Rules (binding)

NO code changes. NO git. NO subagents. NO GPU. NO background waits;
foreground, every call < 10 minutes. Never kill any process. The one
writable file is named above; write it in one go at the end (draft in
your own head, not in the repo). RED honesty: if a receipt does not
support a claim in the primer, that is a finding for Part A. Model id
and reasoning effort in your final message.

## Done (verbatim)

1. The file path and its section word counts.
2. The three crazy ideas you would run first, one line each.
3. The single most valuable boring next step, one line.
4. Any primer/receipt disagreements found.
5. Prior art coverage statement; process safety; model id and effort.
