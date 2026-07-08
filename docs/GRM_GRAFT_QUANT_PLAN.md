# GRM Graft Storage Quantization — Implementation Plan

Status: DRAFT — immutable at initial commit. Tracking:
- Operational ledger (new): `docs/GRM_GRAFT_QUANT_LEDGER.md`
- Narrative synthesis (continues the recall-gate wing):
  `docs/GPT_OSS_20B_APA_GRM_SYNTHESIS.md`
House laws unchanged. Scope law (David, registered on the board
2026-07-08): **this is NOT APA** — this is quantizing at-rest grafts to
cut disk usage, and finding where recall degrades. Storage economics
with a fidelity axis. The witnessed-graft law is untouched: quantization
is a storage transform of witnessed K/V banks, never synthesis.

## Question

Quantize stored GQA grafts at 8/6/4/3/2 bits. Where is the recall cliff?
Deliverable: one curve — recall-gate outcomes AND logit margins vs bit
depth — with the disk multiplier alongside (2/2.7/4/5.3/8×), and the
last depth where the battery stays green, named.

## Design

- **Capture once, quantize many.** Witnessing is the expensive half and
  is identical across depths. Capture the graft banks once at fp16
  (or reuse existing captured banks), then per depth: quantize the SAME
  banks at rest → dequantize on mount → run mount+score only. Gate
  scripts must be verified to consume pre-captured graft dirs; if any
  gate re-captures unconditionally, P1 adds the reuse path (harness
  change only, no engine change).
- **Storage transform**: uniform symmetric per-group quantization
  (group-32, the house packing convention; reuse existing INT4/intn
  machinery where it fits). K and V uniformly at each depth. Mount path
  dequantizes to fp16 — attention math unchanged, zero kernel work.
- **Margins, not just verdicts**: every gate records its logit margins
  so the curve shows the cliff's SHAPE (graceful vs sudden), not just
  the break point.
- **K/V asymmetry (K8V4, K4V8) only after the uniform curve** localizes
  the cliff — refinement, not the question (David's framing).

## Phases

P0 — Inventory + baseline receipts:
  a. Graft storage format receipt: where the K/V bytes live on disk,
     dtype, layout, per-graft footprint (file:line map of the store/
     load path in core/).
  b. Gate battery inventory: which gpt_oss20b_*_gate.py scripts are
     runnable against pre-captured banks, their runtime each, what they
     measure (multifact, exact-value, addressing, preference,
     supersession). Pick the sweep battery: fast gates at standard
     context — the 96K H5 monster is NOT in the sweep loop (one capture
     of a moderate-context battery is; the 96K result stands as the
     fp16 anchor).
  c. fp16 baseline re-stamp of the chosen battery ON THE CURRENT ENGINE
     (kernels changed 2026-07-07; old receipts predate them) — margins
     recorded. This is the curve's y-axis origin.
P1 — Quantize-at-rest harness: pack/unpack transform + gate-runner
  wrapper that takes (captured banks, bits) and emits per-gate margins.
  Round-trip receipt: bit-exactness of the 16-bit path (quantize at 16
  ≡ identity) and reconstruction RMSE per depth recorded per bank.
P2 — The sweep: 8/6/4/3/2 bits × battery. GPU runs bounded and spaced
  (power law: no sustained multi-hour continuous draw; each gate run is
  minutes-class, sweep scheduled as separated bounded runs).
P3 — Cliff localization + report: the curve, the named cliff, disk
  multipliers, and (only if the cliff sits between tested depths or
  shows K-vs-V structure worth probing) the asymmetric follow-up sweep.

## Registered expectations (frozen at commit)

- E1: INT8 is free — margins within run-to-run noise of fp16.
- E2: a cliff EXISTS at or above 2 bits (if 2-bit still passes
  everything, that is a finding worth publishing on its own).
- E3: reconstruction RMSE correlates with margin erosion (if margins
  collapse where RMSE is still small, something structural — not
  amplitude noise — is breaking; flag, don't paper over).
- The curve is the deliverable. Misses are findings.

## Roles

Fable = planner/gates/ledger; Sonnet agents = implementation, flat,
no-delegation. GPU runs respect the bounded-run law.
