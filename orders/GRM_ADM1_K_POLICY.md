# ORDER GRM-ADM1 — admission k-policy: is top-3 a holdover from the weak-router era?

## Grant

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` — edits + CPU
self-runs AUTHORIZED; emit GPU scripts for what exceeds your sandbox
(no CUDA there). GPU work QUEUES BEHIND the in-flight CMC1.1 arms run
(same lock; flock handles it). Append-only artifacts
(`artifacts/grm_adm1/`); new code `scripts/grm_adm1_*.py`; production
byte-identical with flags off. FORBIDDEN: git, subagents, network.

## Context (one paragraph)

Default arena admission mounts top-k=3. That constant was calibrated in
the June single-channel-router era (E1 receipt: routed top-3 10/10 vs
mount-all 4/10 — 3-vs-ALL, never 3-vs-1). The router has since gained
the identifier bonus (CORPUS-100 20/20 over near-duplicates) and
lineage channels; recall@1 on the MODERN production router config has
never been measured. The praxis co-mount miss (supersession ledger
fresh control) is a case where rank-1 was correct and ranks 2–3
injected the competitor that readout then answered. David's
hypothesis, registered: k=3 is insurance against a router that no
longer exists; a decisive-rank-1 admission policy prevents co-mount
ties without recall loss outside declared-synthesis turns.

## Fixture classes (label each probe pre-run; reuse existing certified fixtures)

- POINT-LOOKUP: corpus-100 probe set; supersession-battery fresh
  controls (incl. praxis); DIAG probe set.
- CROSS-FACT / SYNTHESIS: the E2E 34-turn session's cross-fact probes
  and the P4-replication cross-fact set (the known co-mounted-collapse
  class).
- AMBIGUOUS: probes with zero identifier hits (anaphora/vague
  follow-ups) drawn from the E2E session; if fewer than 8 exist, say
  so — do not manufacture new probe text.
Record per probe: identifier-hit count, router rank order, class label.

## Arms (identical router config = production channels; only admission varies)

- A-k1: mount rank-1 only.
- A-k2 / A-k3: fixed k.
- A-DEC (the policy under test): decisive rank-1 → k=1 (decisiveness =
  registered rule: exactly-one identifier-decisive candidate OR
  rank-1/rank-2 route-score margin above a fit-side threshold chosen
  on a held-out split — freeze the rule before eval and report it);
  2+ identifier hits → declared synthesis, mount the identified set;
  zero hits (ambiguous) → k=3.

## Metrics (per class × arm)

Recall (right answer present/derivable), wrong-read rate (co-mount
confusion class), miss rate (needed graft absent), mounts/turn.
Bootstrap CIs over probes.

## Registered predictions + adjudication

- P-HOLDOVER: on POINT-LOOKUP, A-k1 recall is non-inferior to A-k3
  (≥, within CI) AND wrong-read rate strictly lower. If so at k=1
  and A-DEC: **k=3 adjudicated HOLDOVER for point-lookup.**
- P-SYNTH: on CROSS-FACT, A-k1 loses recall (the price of naive
  narrowing) and A-DEC recovers to ≥ A-k3 via the declared branch.
- P-AMB: on AMBIGUOUS, A-k1 drops recall — insurance still earned
  there; A-DEC ties A-k3 by construction.
- ADM-VERDICT vocabulary: DECISIVE-ADMISSION-SUPPORTED iff P-HOLDOVER
  and P-SYNTH both hold (A-DEC ≥ every fixed k on every class, fewer
  wrong reads on lookup); HOLDOVER-ONLY iff P-HOLDOVER holds but
  A-DEC's declared branch underperforms (name why); REFUTED iff
  A-k1/A-DEC lose recall on POINT-LOOKUP (rank-1 is not reliable —
  the insurance was real; report recall@1 vs @3 as the finding).

## Gates

ADM-G0 fixtures reproduce their certified baselines at A-k3 (this IS
the regression anchor). ADM-G1 byte-identity flags-off. ADM-G2 the
decisiveness rule frozen (hash) before eval split runs. ADM-G3 report
with the class × arm table verbatim + the ADM-VERDICT sentence.

## Constraints

Standing laws (GPU discipline queued behind CMC1.1 on the same flock,
provenance, non-finite writer, RED honesty, no post-hoc widening).
## Done: gate statuses; the full table; ADM-VERDICT; the frozen
decisiveness rule; recall@1-vs-@3 headline numbers; files; anything
not done.
