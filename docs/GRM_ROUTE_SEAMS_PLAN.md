# GRM Route-Latency Seams 2/3 (PLAN, immutable)

Successors registered in GRM_E2E_RECEIPT_LEDGER.md (E3 RED: ~756 ms
route at 37 nodes, python path, GPT-OSS session). Opened 2026-07-17.
Lead: Fable. Seat: Terra (Sol escalation per HOUSE_RULES §9).
Worktree-isolated (grm-route-seams).

## Scope

- SEAM-2: ragged-bank CUDA route non-engagement — wire the merged
  exact ragged GQA CUDA router (8e11bbd, disabled by default) into
  the GQA arena route path behind an opt-in flag; byte-exact parity
  with the python route is the correctness demand (the router's own
  175/175 four-way parity is the precedent).
- SEAM-3: per-turn arena re-prep — profile what step() rebuilds per
  turn on the route path; eliminate rebuilds that are invariant
  across turns (epoch-cache pattern from the MLA CUDA route work);
  every elimination must be parity-gated.
- Lex rescore cost: measure; optimize only if it is a top-2 term.

## Registered gate (E3 restated)

Route wall ≤ 5 ms at session scale (≥37 resident nodes) on the
flagged path, measured on the local GQA testbed (Qwen3-4B arena
session harness); python-path parity byte-exact at k∈{1,3,5,16};
ALL existing routing regressions green (GQA arena 6/6, trips 6/6,
ragged bank suite). Latency numbers name their path explicitly.

## Non-goals

Default flip (operator decision, adoption evidence = this program's
receipts); MLA path (already 2.22 ms); GPT-OSS re-run (successor —
local GQA evidence first).
