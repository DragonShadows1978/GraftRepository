# ORDER GRM-DET1.11 — achieved-count amendment; lived-serving reliability census

WRITABLE TARGET `/mnt/ForgeRealm/GraftRepository` — edits/builds/CPU runs
AUTHORIZED. Rules as DET1.x: no git (lead commits), no subagents, no
network, no GPU (CPU self-runs + selftests fine; the lead runs
`./scripts/grm_det1_5_lead.sh`). Append-only artifacts. RED honesty.

State (campaign r7, `logs/grm_det1_5_gpu_lead_r7.log`) fail-closed:

```
DETError: no unused lawful certified campaign-session reserve for
sup_lumen_head: primary_reason='LIVED_SERVED_CONTROL_INCORRECT_OR_REFUSAL'
```

DET1.10 diagnosed the reserve pool as genuinely EMPTY of lawful members
(5/5 candidates fail their own lived controls; the frozen selection rule
is NOT over-constrained). r7 now shows the same failure mode reaching a
second primary slot, `sup_lumen_head`. The campaign-wide pattern across
r4–r7: lived served-control failures include t33 (refusal),
e2e_t30_atlas_tone (separator variant, rescued at source by DET1.10),
`sup_lumen_head`, and four `sup_reserve_*` probes that serve confidently
WRONG values with CORRECT single mounts.

The conclusion the receipts force: the registered 12+12 pair count is
UNREACHABLE on this fixture population. Not a harness defect — the
model's serving baseline is unreliable on a meaningful fraction of
certified probes.

Two threads, kept separate.

## 1. ACHIEVED-COUNT AMENDMENT (lead-authorized)

Authorized under David's delegated keep-going directive. Amend the
campaign to run at the ACHIEVED lawful pair count:

- Every planted turn whose lived control is LAWFUL gets its pair.
- REGISTERED FLOOR: minimum **8** planted/served pairs, or the campaign
  refuses. The floor exists to prevent a hollow race — a two-pair
  "race" would be noise dressed as a result.
- The analyzer reports the achieved N PROMINENTLY in the race table and
  in the verdict sentence — "at N of 12 registered pairs".
- Detector metrics, the thresholds policy, and the
  SUPPORTED / PARTIAL / REFUTED vocabulary are UNCHANGED. This
  amendment changes the population size, never the adjudication.
- Excluded turns are enumerated in the receipts with their
  lived-failure reasons.

## 2. LIVED-FAILURE CENSUS (receipts only — do NOT investigate cause)

Produce a census receipt covering ALL lived-collected probes in this
campaign — planted candidates, reserves, and calibration alike:

- lawful vs `LIVED_SERVED_CONTROL_INCORRECT_OR_REFUSAL`,
- subdivided: refusal / wrong-value / separator-artifact-rescued,
- with the per-probe served answers.

Title it **GRM lived-serving reliability census**.

This census is the foundation for the successor investigation. The
defect class it names is NEW and distinct from co-mount blending:
wrong values served with CORRECT single mounts. Cause analysis is
explicitly OUT OF SCOPE here — receipts only.

## Source declaration (mandatory)

Declare every changed source in the DET1.9/1.10 authorization overlay
pattern. Undeclared drift fail-closes the campaign — two seats have
already been caught by this. Declare BEFORE finishing.

## Self-verify

GRM test battery + campaign/analyzer selftests. Add tests pinning the
floor and the achieved-count reporting.

## Done

Final message carries receipts, verbatim:

- The amendment summary.
- The census table verbatim.
- The declared-sources diff.
- Test counts.
- Files modified.
- Anything not done.
