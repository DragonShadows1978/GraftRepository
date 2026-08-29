# ORDER MOE-E4 — closed-form rank-1 consolidation: is the graft→expert step pure algebra?

## Grant

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository`. Rules,
read-onlys, and corpus grants exactly as `orders/MOE_E3_EPISODIC_CONSOLIDATION.md`.
CPU self-runs authorized (bounded, walls stated); emit GPU scripts for
what exceeds your budget. New code `scripts/olmoe_e4_*.py` (import
E2/E3 harnesses; modify nothing existing). Artifacts `artifacts/moe_e4/`.
FORBIDDEN: git, subagents, network.

## Hypothesis (David, 2026-08-29 — the reason this order exists)

A gradient is a sum of keyed rank-1 outer products; summing is what
destroys addressability. Therefore the consolidation step should not
need SGD at all: the MSE-optimal residual map from student hidden
state h to the teacher shift Δ has a CLOSED FORM (ridge regression:
ΔW = Δᵀ·H·(HᵀH + λI)⁻¹), and its truncated SVD gives the asymmetric
A/B expert directly — an expert built by algebra, whose singular
directions are inspectable. If closed-form matches or beats the
SGD-trained adapter at equal rank, the graft→expert bridge is
algebraic and ExpertPacks become per-direction inspectable.

## Dependencies

- Install layer L\* and gate key/τ: consume
  `artifacts/moe_e3_1/` results if present (E3.1 in flight, run
  20260829T230723Z-3971615). If E3.1 yields a qualifying layer under
  its rule, use it. REGISTERED FALLBACK if it does not: use the best
  E3.1 K4 layer by fit-side rank, record "domain-grade address"
  caveat verbatim in every receipt, and keep DOC-B/guides firing
  DESCRIPTIVE. If E3.1 artifacts are absent when you need them, build
  everything else and leave the dependency explicit in your report.
- Pairs: DOC-A PAIR region teacher/student pairs at L\* per the E3
  spec (64 pairs, 2,048-token true-context teacher). These do not
  exist yet — your script captures them (self-run on CPU what fits;
  script the rest).

## Arms (registered; identical pairs, identical rank r=64)

- ARM-CF (primary): ridge closed form on FIT pairs, λ selected on
  validation pairs (grid, fit-side only), truncated SVD → A/B at
  r=64. Report the singular-value spectrum (how concentrated is one
  document's shift?) and per-direction key-alignment (cosine of each
  right singular vector vs the gate key k) — the inspectability
  receipt.
- ARM-SGD (baseline): the E2-style trained adapter, same pairs, same
  r, same early-stopping discipline.
- ARM-CF-FULL (descriptive ceiling): untruncated ridge.

## Gates (registered)

- E4-G3: both arms beat the zero-predictor on validation pairs by
  ≥ 10% MSE (else RED for that arm).
- E4-G4 behavioral (per arm): on DOC-A HELDOUT windows, precondition
  teacher gap ≥ 0.5 ppl (same registered stop as E3); arm SUPPORTED
  iff it recovers ≥ 25% of the gap with bootstrap 95% CI low > 0.
- E4-H (the hypothesis verdict): ALGEBRA-SUFFICES iff ARM-CF recovery
  ≥ ARM-SGD recovery − 5 points (absolute) AND ARM-CF passes E4-G4.
  ALGEBRA-INSUFFICIENT iff ARM-CF fails E4-G4 while ARM-SGD passes.
  BOTH-FAIL: report both numbers; the episodic-consolidation premise
  itself is then the open question, not the construction.
- E4-G5 (per arm): wikitext ppl delta ≤ 0.5% gate-live, code fire
  ≤ 0.05; DOC-B delta/fire descriptive.

## Constraints

Everything standing: bf16 only, content-addressed provenance,
idempotent prepare, fail-loud non-finite writer, GPU discipline in
emitted scripts, RED honesty, no post-hoc widening, adult-content
handling as E3 (data only; nothing excerpted beyond hashes/counts).

## Done

Final message verbatim: E4-G3/G4/G5 per arm + the E4-H sentence (or
NOT_MEASURED + lead scripts); the singular-spectrum summary (top-8
values + energy fraction at r=64); the install row used and which
E3.1 branch supplied it; the G4 rows per arm; files created/modified;
anything you could not do.
