# GLC — GPT-OSS-20B Port Long-Context Quality Program (PLAN, IMMUTABLE)

Opened 2026-08-28 by David's directive ("open the engine long-context
program"). Plan is immutable after initial commit per house rules;
execution detail lives in the ledger; meaning lives in the synthesis.

## Known facts at open (receipts)

- Long-vs-short scoring of IDENTICAL contiguous-wikitext targets on the
  GraftRepository GPT-OSS-20B port degrades ~8× ppl at total length
  1024 and 30–40× at 2048–2560 (mean-NLL deltas ~2.0 → ~3.4 nats).
  Receipts: `artifacts/moe_e1_3b/runs/` (standard backend ramp),
  `artifacts/moe_e1_4/runs_e14/fg0/` (apa_selective ramp).
- BOTH full-attention backends reproduce it (standard explicit-mask and
  fused apa_selective) ⇒ the defect lives in shared machinery (RoPE/
  YARN tables, sink semantics, layer interleave, scoring protocol) or
  in the model's genuine behavior (to be refuted via reference arm).
- Coverage gap: the port has never held a long-sequence ppl receipt —
  `gpt_oss20b_realtext_ppl_gate.py` defaults to 64-token windows;
  context-ladder/96k receipts are functional/recall-class.
- Prime suspect (unproven, registered as hypothesis H-GLC-1): YaRN/RoPE
  table application error — sliding-window layers (local relative
  positions) stay healthy while full-attention layers (large relative
  distances) break, matching observed content-dependent severity.
- MOE-E1 (consolidation expert) is blocked on this program.

## Phase 0 — Ground truth and localization

- P0.a Reference arm: score the exact ramp cells (contiguous wikitext,
  lengths {1024, 1536, 2048, 2304, 2560}, last-511 targets + same-token
  short-window references) through HuggingFace transformers on the same
  snapshot. REGISTERED EXPECTATION HG1: healthy = long-arm mean NLL ≤
  short-arm + 0.05 nats per cell. If HF also degrades, the protocol or
  model is implicated and the program re-plans via addendum (plan stays
  immutable).
- P0.b First-divergence hunt: port-vs-HF hidden states, layer by layer,
  at short (≤512) and long (2048+) positions; localize the first layer/
  op where long-position states diverge while short positions match.
  (Prior art: trinity_nano_first_divergence pattern.)
- P0.c Static+numeric audit of shared machinery under H-GLC-1: YaRN
  scaling constants and application (factor, original context,
  attn-scaling/concentration), rope table dtype/length, sink semantics
  at long S, sliding/full interleave map — port vs HF modeling code,
  with numeric table diffs at positions {0, 511, 1023, 2047, 2559}.

## Phase 0 registered gates

- HG1: HF reference verdict per cell (healthy/degraded) with numbers.
- HG2: first divergence localized (layer, op, position band) with
  receipts, or "port matches HF" stated (which refutes port-defect).
- HG3: mechanism sentence naming file:line in port/engine code, or an
  explicit unresolved statement listing what was excluded.

## Phase 1 — Fix and parity (entered only on HG2/HG3 implicating the port)

- Fix in the implicated layer (port code or tensor_cuda kernel).
  Standing law: the APA function itself is inviolable; implementation
  rework is permitted.
- REGISTERED PARITY GATES (fixed now, before any fix exists):
  PG1 ramp health: long-vs-short delta ≤ 0.15 nats on all five cells;
  PG2 HF parity: port mean NLL within 0.10 nats of HF on every ramp
  cell (long and short arms);
  PG3 no regression: existing port suites (realtext ppl gate at its
  registered config, context-ladder gates, GRM recall gates on this
  model) all pass unchanged.

## Phase 2 — Institutionalize + unblock

- Extend the realtext ppl gate with registered long-window configs
  ({512, 1024, 2048, 4096} windows) so this receipt class exists
  permanently; wire into the port's standard battery.
- Re-run MOE-E1.4 from FG0 on the healthy forward.

## Roles

Fable = planner/ledger/verification/commits. Implementation seats =
Codex Sol at ultra effort (David's standing directive this program).
Lead runs all GPU work (seat sandbox has no GPU). GPU discipline:
flock, ≤10-min bounded runs, gaps, single GPU.

## Evidence classes

Ramp/parity numbers = inference measurement on one model. Audit
findings = code-reading claims, verified numerically before acted on.
No model-quality claims from kernel-level receipts.
