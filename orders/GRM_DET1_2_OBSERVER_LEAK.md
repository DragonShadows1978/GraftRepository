# ORDER GRM-DET1.2 — instrumentation observer-leak: bisect the hook that plants a refusal

WRITABLE TARGET `/mnt/ForgeRealm/GraftRepository`; rules as DET1.x.

Finding (lead-verified from the guard receipt,
det1_1_guard_failure_fefc72a0…): on the SERVED harbor_restatement
counterfactual, the instrumented path answers "I'm sorry, but I can't
help with that." — a REFUSAL — where the live registered baseline
(same mounts [harbor_b], same everything) answers "The current Harbor
token value is Nacre-6-Blue." The observed matches NEITHER the live
nor any retained registration: the DET1.1 live-anchor guard correctly
adjudicated REAL INSTRUMENTATION LEAK. An observer hook is perturbing
model state enough to trigger the distilled model's refusal behavior.

Work:
1. Bisect: run the served harbor turn with each instrumentation hook
   enabled ALONE (D-LQR fingerprint capture / D-NGH witness tap /
   D-ENT logit capture / D-VERB probe pass), then pairwise if needed.
   Convict the leaking hook(s) with file:line and the perturbation
   mechanism (positions, masks, KV mutation, dtype, RNG state, an
   extra forward contaminating cache — name it exactly).
2. READ `docs/GRM_E2E_RECEIPT_LEDGER.md` for the refusal-plant law
   from the E2E program and state whether this is the same mechanism
   resurfacing through the new instrumentation (cite the ledger line)
   or a new one.
3. Fix to pure-observer (byte-identity of the served path with ALL
   hooks enabled is the gate — DET-G1 as registered). If a detector
   CANNOT be made pure (e.g., D-VERB's extra pass inherently touches
   state), isolate it to a fully separate replay pass that shares
   nothing with the scored path, and record that design change.
4. Resume the run to the race verdict.

Done: convicted hook(s) file:line + mechanism; refusal-plant
same-or-new adjudication with ledger citation; the fix; DET-G1
receipt; race status; anything not done.
