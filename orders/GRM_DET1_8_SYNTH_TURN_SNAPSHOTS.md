# ORDER GRM-DET1.8 — micro fix: declared-synthesis turns need complete per-member snapshots

WRITABLE TARGET `/mnt/ForgeRealm/GraftRepository`; rules as DET1.x.

Defect (campaign r3 fail-closed at plant-registration): turn 16 fresh
lived collection raises SnapshotError "fork-hydration delta requires
two complete snapshots." t16 is a declared-synthesis turn (A-DEC
mounts the identified 2-graft set); the fresh-collection path
snapshots as if single-mount, so the per-member withholding delta
cannot be formed.

Fix: the fresh lived collection for multi-mount (declared-synthesis)
turns captures complete snapshots covering every mounted member;
the registry's deterministic member-selection rule (from DET1.7)
then derives the withhold-target; the delta receipt proves exactly
the selected member's seats absent, all other members and bytes
identical. Audit the remaining fresh-collection turns for the same
single-mount assumption (t09 calibration is likely fine as k=1, but
CHECK, don't assume). CPU contracts; hand back for campaign rerun.

Done: diff summary; per-turn audit table (mounts vs snapshot
coverage); CPU receipts; anything not done.
