# ORDER NC17-P2 — Qwen3-1.7B INT4 weights + APA r0.15 battery

YOUR WRITABLE TARGET is /mnt/ForgeRealm/GraftRepository — NEW files
under `tests/nc17/`, `logs/nc17/`, and any quantized-weight artifact
directory you create under `logs/nc17/` or a `weights_nc17/` dir in the
repo root; adapter files only as touched by P1 (extend, don't rewrite).
/tmp writable. Engine READ-ONLY.

Read `docs/QWEN3_1P7B_NAMECHECKER_PLAN.md` (immutable) first.
Prerequisites: P1 adapter landed and parity-gated (its receipt is in
the ledger); P0 GT + ppl tokens on disk.

## Task — Phase P2

1. **Quantize** the 1.7B to INT4 with the HOUSE stack (the same
   int4/symmetric-8 machinery the existing ports use — read how
   qwen3-4B/qwen35 adapters load INT4 and follow that path; record
   group size + symmetric/asymmetric choice + which tensors stay
   higher precision, if any, in the receipt). Self-quantized from the
   bf16 snapshot — no GGUF import. The tied head: quantize consistent
   with P1's resident/host decision.
2. **Parity margin gate** vs P0 GT: same protocol as P1 — INT4 may
   flip only GT near-ties (report every flip + margin). Also report
   INT4-vs-P1-bf16 max|Δlogit| distribution (same engine, isolates
   quant noise from port noise).
3. **Full battery**, INT4 + APA r0.15 (and APA-off control):
   OOM/context ladder (prefill+decode, one probe per subprocess),
   ppl per protocol (engagement asserted), 1s-poller peaks.
4. **Deliverables**: printed tables + `logs/nc17/p2_summary.json` —
   quant config, parity margins, ceilings, ppl×window (vs P0/P1),
   peak×ctx. This feeds the P2→P3 decision row of the ledger.

## Rails

Same as P1: GPU foreground-only under flock, timeout 590, setsid law,
no detaching; NO git; NO subagents; no engine edits; OOM=measurement;
RED honesty verbatim; evidence classes per plan.

## Done — final message must contain, verbatim

- Quant config (bits/group/symmetry/exempt tensors) + artifact paths
  + on-disk and resident sizes.
- Parity table vs GT (flips+margins) and Δ-vs-P1 distribution.
- Ceiling / ppl / peak tables beside P0+P1 numbers.
- APA engagement stats per scored run.
- Deviations stated as deviations.
