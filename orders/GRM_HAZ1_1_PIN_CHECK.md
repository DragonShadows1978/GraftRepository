# ORDER GRM-HAZ1.1 — micro fix: L2 gate coordination-pin check accepts explicit production pins

WRITABLE TARGET `/mnt/ForgeRealm/GraftRepository`; standing rules.

Defect: `scripts/grm_sup_l2_default_on_gate.py:170` requires ADM
scripts to pin `revision_resolution=False`; ADM1.2 legitimately
migrated ADM to EXPLICIT production L2-on (`--sup-resolve`,
grm_adm1_gpu.py:478/825). The check's intent is "no ACCIDENTAL
default inheritance." Fix: accept an EXPLICIT pin in either
direction (explicit False = old-frame isolation; explicit True /
--sup-resolve = production-aligned); still fail on absent/implicit.
Update the check, keep everything else byte-identical, self-run the
gate's CPU legs, hand back for lead GPU rerun.

Done: diff summary, CPU gate receipt, anything not done.
