# GRM-C2 amendment 5, dispatch 2 (lead, 2026-09-09) — the first dispatch died on ENOSPC mid-write; clean your partial staging and redo the same order

Your first run of `orders/GRM_C2_AMENDMENT_5.md` was killed by a full
disk (`No space left on device` while writing
`artifacts/grm_c2/AMENDMENT_5_REPORT.md` and `artifacts/grm_c2/a5/`);
no SHIM-DONE was recorded. The disk has room again (the lead moved a
stale archive off the NVMe). Do the SAME order again (cap 4,800 s read
at run time; RED-cell eligibility ruling; no edits to files the epoch-3
runner or worker executes), but FIRST list every file under
`artifacts/grm_c2/a5/` and any `amendment_lead_5*` you find, decide
per file whether it is complete (parseable, sha-consistent) or a
partial write, keep partials under `a5/partial_first_dispatch/` with
their shas, and rebuild from scratch. Preflight: refuse if
`df /mnt/ForgeRealm` reports < 20 GB free (print it). Same rules (no
git, no subagents, no GPU, foreground, never kill anything). Effort:
high.

## Done (verbatim)
1. Partial-file inventory and disposition; amendment path + sha; where
   the cap is read; the RED cell's eligibility ruling; tests.
2. Exact lead command.
3. Prior art; deviations; RED; process safety; model id and effort.
