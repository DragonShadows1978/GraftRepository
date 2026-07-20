# ORDER NC17-P3m — Qwen3-1.7B INT6 weights + APA r0.15 battery (fork engine)

YOUR WRITABLE TARGET is /mnt/ForgeRealm/GraftRepository — extend
`core/qwen3_1p7b_tc.py` with an INT6 load path, NEW files under
`tests/nc17/` and `logs/nc17/`, weight artifacts beside P2's. /tmp
writable. BOTH engine checkouts are READ-ONLY.

**ENGINE SELECTION — the point of this order:** the INT6 ops exist ONLY
in the fork build at `/mnt/ForgeRealm/Project-Tensor-int6/tensor_cuda`
(branch int6-weights, lead-verified gates: dequant bit-exact, GEMV
rel-fro ~3e-4). Every run in this order sets
`PYTHONPATH=/mnt/ForgeRealm/Project-Tensor-int6/tensor_cuda` (the
harness scripts take the engine path — parameterize, don't hardcode
canonical). Surface: `tc.int6_dequant`, `tc.int6_linear`,
`tc.int6_linear_fused` (x/packed/scales/zeros/group_size naming,
symmetric = empty zeros tensor, g128, 4-codes-per-3-bytes).
Sanity-import before any long work; if the fork engine fails to import
or misbehaves, that is a REPORTABLE RED, not something to patch.

Read the plan (immutable) + the ledger tail first. Prereqs: P2 complete
(its receipt in the ledger; reuse its quant harness structure and its
parity/battery scripts as templates — p2_* files).

## Task — Phase P3 measurement half

1. Quantize the bf16 snapshot to INT6 g128 symmetric with the fork's
   quantizer surface; adapter INT6 path mirrors the INT4 path's
   structure (record what differs). Tied head per P1's decision.
2. Parity: cached-chain vs GT (comparability row) + INT6-vs-P1-bf16
   same-engine delta (the clean quant isolation) + built-in P1b-style
   fresh-refeed adjudication of any hard flips.
3. Full battery, INT6 + APA r0.15 and APA-off control: OOM ladders
   (prefill+decode, one probe per subprocess), ppl per protocol
   (engagement asserted), 1s-poller peaks.
4. Deliverables: printed tables + `logs/nc17/p3_summary.json` with P0/
   P1/P2 columns alongside.

## Rails

Same as P2 (foreground GPU under flock, timeout 590, setsid, no
detach, no git, no subagents, OOM=measurement, RED honesty). Plus:
never mix engines in one process; every log line that reports a number
must name which engine build produced it.

## Done — order's standard receipt set: quant config + sizes, parity
tables + adjudication, ceiling/ppl/peak tables beside P0-P2,
engagement stats, engine-path confirmation per run, deviations.
