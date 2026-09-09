# GRM-C7 amendment 5 — RED / required STOP

SCOUT-FIX-7's prerequisite is absent. No fixture, scorer, prompt or core changes were made; no r3 campaign was registered or run. **Not claimed fixed.**

The immutable order is `orders/GRM_C7_AMENDMENT_5_R3.md`. Mission 2 explicitly says: “follow the alias edge L2 already records; if it does not record one, say so and STOP”. This is the executed decision fork, not a request for routine implementation approval.

## Evidence

Evidence class: source inspection and existing checkpoint metadata inspection, not a new CPU replay, unit test, treatment experiment or end-to-end gate. `stop_receipt.json` contains all eight RD2 alias rows, exact stored texts, full metadata, checkpoint paths and SHA-256 input bindings.

- `core/graft_arena.py:1282` (`_revision_mount_heads`) traverses only explicit `metadata.supersedes`; `:1337` (`_resolve_revision_mounts`) delegates to it. L2 resolves revision heads. It does not record or traverse alias-to-base links.
- In all eight inspected RD2 predecessor checkpoints, alias node 8 or 9 has `sources=[]`, `metadata.source_grafts=[]`, `metadata.source_turns=[]`, and `metadata.supersedes=[]`. There is no alias-reference field. The alias is only the stored sentence “C7-Signal-0 is an alias for C7-AliasBase-0.” (or Signal-1/Base-1).
- Node 4 contains both Jasper base values and is active. Historical mounted IDs are `[8]` or `[9]`. These are observations from existing receipts, not a newly tested flag-OFF control.
- A second constraint is already recorded by RD2 and confirmed in checkpoint geometry: edge 49 + base 61 = 110 tokens, exceeding the C2 profile's 96-token mount width. Merely extending a mount plan cannot seat both original full payloads at once. This arithmetic is not a new fitting experiment.
- Independent source receipt: `/mnt/ForgeRealm/wt/grm-rd1/artifacts/grm_rd2/REPORT.md:57` explicitly states: “Neither alias edge node has structural source links; the ‘edge’ is natural-language text.”

## Requested deliverables at the stop

1. **Fixture diff and new SHA:** not produced. Existing fixture remains unchanged and its SHA is pinned in `stop_receipt.json`. Turn 2 remains the ordinary fact at `scripts/grm_c7_register.py:38`; production constructs `kind='supersede'` and `correction_command` at `scripts/grm_e2e_session.py:143`. FIX-7 file/lines, flag and RED/GREEN tests: not implemented or run. No default flip.
2. **Prompt quotes:** the registered C7 suffix is “Reply only with the answer; if unspecified, reply UNKNOWN.” (`scripts/grm_c7_register.py:108`). Existing r2 execution lowercases the final UNKNOWN (`scripts/grm_c7_run.py:114`). Production constructs “Recall probe. What is the current {fact_id} value? Reply with only the value.” (`scripts/grm_e2e_session.py:189`). RD2's plain alias query is “What is the current value for C7-Signal-0? Reply only with the answer.” No prompt modification was applied after the stop.
3. **Scorer note:** the requested amendment is casefolded exact comparison for unanswerable abstention controls, preserving the frozen answerable exact-value rule. It is not applied or tested. Existing scores remain historical evidence.
4. **r3 registration path + SHA:** absent; `r3/stop_receipt.json` is a stop receipt, explicitly not an execution registration. Its SHA-256 is `99de4e999b7686a8a18b13da7eb4466e3f4896f977019a52405dcabd02f8a714`.
5. **Exact lead command:** none. No `lead_commands_r3.txt` was created because its mandatory FIX-7 prerequisite failed. The proposed 39 cells, arm A, FIX-3/4/5/7, 2.2 GPU-h cap and predictions have not been registered for execution. No fresh, folded, correction, alias, residency or restart result is claimed for r3.

## Prior art

Verified local system: GRM contributors (2026), M5/L2 explicit supersession and RD2 checkpoint metadata projection/input SHA binding. Taken: their existing metadata semantics and reporting method. Ours: this order-specific stop receipt; **no prior art known to me** for this exact reporting composition. No algorithm, selection rule, data structure, proof technique or optimization is implemented or proposed here. No external literature claim or network lookup.

## Deviations, RED and process safety

Execution ended at mission 2's explicit stop fork before implementation. No tests were registered or run; this receipt must not be presented as a retrospectively registered gate. Source and metadata inspection establishes the failed prerequisite only. The eight alias failures and correction lineage defect remain RED; no treatment effect is established. Further work requires an amended order addressing the missing structural relation and the mount-width constraint.

All commands were foreground CPU reads or receipt writes. GPU time: 0 seconds. No git commands, subagents, background waits, model loads, process kills or service actions. Worktree and branch were verified by reading `.git` and its referenced `HEAD` (`refs/heads/grm-c7`). Existing source, fixture, order and historical receipts were preserved; additions are confined to `artifacts/grm_c7/r3/`.

Agent model: **gpt-6-astra**; effort: **high**, recorded at `logs/grm_c7_a5.log:6` and `:10`. This identifies the engineering seat, not a new inference run.
