# ORDER QWEN38-A3 — attention gate default = sigmoid (adjudicated), doc + gates

YOUR WRITABLE TARGET is /mnt/ForgeRealm/GraftRepository — AUTHORIZED.
Also writable: /tmp. READ-ONLY: /mnt/ForgeRealm/Project-Tensor,
/mnt/ForgeRealm/models/Qwen3.8-27B, installed transformers. No git
writes (read-only git fine). No subagents. RED honesty.

## Adjudication receipts (lead-run, on the real GPU)

Your A1 swish interpretation is REFUTED empirically: with SiLU on the
attention output gate, 27B greedy output is multilingual token salad on
all 3 prompts (G0 5.9e-7 and G2b 0.99998 still passed — the reference
carried the same gate choice, so parity could not catch it). With
`--output-gate-type sigmoid`: coherent correct text on all 3 prompts
(code/factual/chat), clean EOS. Sigmoid is the truth; installed
transformers' hardcoded sigmoid was right.

## Task

1. Make sigmoid the attention-output-gate DEFAULT for the 27B path
   (checkpoint field no longer trusted to mean the attention gate).
   Keep `--output-gate-type` as a diagnostic override. The 9B path
   stays byte-identical.
2. Resolve WHAT `config.output_gate_type: "swish"` actually governs:
   read the installed modeling_qwen3_5 source and any config docs;
   most plausible candidate is the DeltaNet output gate (already
   SiLU). Document the resolution with file:line citations in
   `docs/QWEN38_PORT_RECEIPT.md`, including the empirical adjudication
   above (quote the salad-vs-coherent receipts). If the source is
   genuinely silent, say so — "empirically adjudicated, mechanism
   unresolved" is a valid finding.
3. Fix the G0 CPU reference to use sigmoid (remove the config-aware
   SiLU patch), rerun G0 and the CPU test suite to green, and update
   tests accordingly.
4. Update `scripts/qwen38_gates.py` / `qwen38_generate.py` defaults
   coherently (G2b reference too).

## Done — final message must contain, verbatim

- Files/lines changed.
- The output_gate_type resolution with citations, or the honest
  unresolved verdict.
- G0 + CPU suite results rerun under sigmoid (printed numbers).
- Any deviation.
