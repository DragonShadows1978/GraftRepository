# INT2/INT3 Weight Model PPL Ledger

This ledger records the actual model-validation work for low-bit weight
testing. The implementation plan is immutable after its initial commit.

## 2026-07-06 15:54 EDT

Action: Opened the real-model validation track.

Repo state:
- Repository: `/mnt/ForgeRealm/GraftRepository`
- Branch: `codex/intn-model-ppl-sweep`
- Starting point: `a8c2824 merge: split quant bench work`

Reason for this track:
- The prior Project-Tensor quant sweep only covered kernel correctness and a
  structured linear sweep.
- It did not load a model with INT2/INT3 weights.
- It did not run model memory checks.
- It did not run PPL.

Relevant repo findings:
- Qwen3.5 local unquantized snapshots exist for both 2B and 9B.
- `Qwen35_TC` loads model weights through `QuantLinearTC`.
- `QuantLinearTC` is currently hardwired to INT4.
- Project-Tensor already exposes native INT2/INT3 kernels through
  `tc.intn_linear`, `tc.intn_linear_fused`, and `tc.intn_dequant`.

Next action:
- Commit the house-rule baseline.
- Generalize the real model weight wrapper without changing INT4 defaults.
