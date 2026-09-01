# ORDER GRM-DET1.6 — micro fix: planted-miss delta applies at fork hydration

WRITABLE TARGET `/mnt/ForgeRealm/GraftRepository`; rules as DET1.x.

Defect: campaign G0 worker fails closed — "planted ladder never
applied a registered state delta." The withholding intervention still
operates at the retired reconstruction layer; the fork hydration path
(the only lawful substrate now) consumes lived bytes verbatim and the
delta never engages.

Fix: implement the registered planted-miss withholding AS a fork-
hydration delta — the withheld graft's seats simply not hydrated,
with a delta receipt proving (a) the target graft absent from the
forked arena, (b) every other byte identical to the lived snapshot
(the zero-gate comparator re-run under the delta must show EXACTLY
the withheld fields diverging and nothing else). Self-run the CPU
contracts; hand back for the lead campaign rerun.

Done: diff summary; delta-receipt design; CPU contract results;
anything not done.
