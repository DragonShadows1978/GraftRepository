# ORDER MOE-E2-F3 — micro fix/diagnosis: inf in eval_abi receipt

WRITABLE TARGET `/mnt/ForgeRealm/GraftRepository`; same rules as E2-F1/F2.

State: chain r4 (logs/moe_e2_gpu_lead_r4.log) reached E2-G2 GREEN
(install L10, 12/16 qualifying) and E2-G3 GREEN (90.7%), then died in
`eval_abi` writing its receipt: `ValueError: Out of range float values
are not JSON compliant: inf` (write_json at script:229, from
eval_abi:3493 via eval_gates:3723). The inf is inside the mount
snapshots / live_gate telemetry structure.

Work:
1. Identify EXACTLY which field goes inf (add a pre-serialization
   walker that names offending paths; reproduce CPU-side if possible).
2. Adjudicate: telemetry artifact (e.g., x/0 in a rate) → sanitize with
   an explicit sentinel string ("inf") and keep going; vs REAL bf16
   score/activation overflow in the gate path → that is a FINDING:
   report it, and make the receipt record it faithfully; do not mask.
3. Make write_json fail-loud-but-named for any future non-finite
   (path-listing error), fix the site, re-run your synthetic contracts
   + a CPU mimic of eval_abi if feasible.
4. Do not regenerate chain scripts unless the fix requires it.

Done: offending field path(s), adjudication (telemetry vs real
overflow) with evidence, diff summary, receipts, anything not done.
