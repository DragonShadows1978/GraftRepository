# ORDER GRM-CMC1.2 — micro fix: witness must recompute from the kernel's actual operands

WRITABLE TARGET `/mnt/ForgeRealm/GraftRepository`; rules as CMC1.1.

State: live arms r4 (artifacts/grm_cmc1/live_arms_20260830T090717Z_4045881/)
reproduced the wrong read live (praxis "wrong-fact") and produced
preliminary decisive-head data, but CMC-G2 FAILED: offline fp32 witness
vs tensor_cuda materialized softmax = mean_abs 7.4e-4, max_abs 0.374 >>
0.002 tolerance. The error is localized to extreme entries — lead
hypothesis: the witness recomputes from fp16-exported K while the
engine attends over dequantized INT4 KV (and/or an MLA
absorbed-computation detail differs), so the instrument diverges
exactly at the outlier keys H-OUTLIER is about. Stage correctly
stopped (returncode 5).

Work:
1. Root-cause the max_abs 0.374: identify which operand
   representation differs (INT4-dequant KV vs fp16 export; MLA
   absorbed path; rope application; scale). Name it file:line.
2. Fix the witness to consume the ENGINE'S actual operand values
   (dequantize exactly as the kernel does, or export the kernel's
   post-dequant operands), so CMC-G2 passes at the registered
   tolerance without loosening it. Loosening tolerance is FORBIDDEN.
3. CPU-validate on synthetic + the recorded G2 case if replayable
   CPU-side; then hand back for lead rerun (bare invocation,
   PYTHONPATH=repo-root — note these in the runner header while you
   are there, they cost two comment lines).
4. Do not touch arm semantics, adjudication, or receipts.

Done: root cause file:line, the operand fix, G2 CPU evidence, files
modified, anything not done.
