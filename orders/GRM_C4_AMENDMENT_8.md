# GRM-C4 amendment 8 (lead, 2026-09-09) — disk-full incident: one truncated receipt and one interrupted unit; make them re-runnable

Lead-run of `lead_resume_a7.txt` (A7 chain) on the card while the NVMe
filled to 0 bytes (SP3/SP4G captures; the lead has since moved a 39 GB
stale archive to `/mnt/Shared/ForgeRealm_offload/`). Consequences in
this worktree:
- `artifacts/grm_c4/lead_a5/runs/c96_w64/census_e2e-2.json` is
  TRUNCATED (invalid JSON at 16,384 bytes) — a create-only receipt
  that cannot be read;
- the lead then stopped the chain (its running worker was killed
  mid-lease), so one further unit may have an `.attempt.json` with no
  completed receipt (an abandoned claim);
- units completed before the incident (c96_w64 sup ×4, census e2e-1)
  are intact; verify each by parsing.
Same worktree (a7 committed 625e79e), same rules (no git, no subagents,
no GPU, foreground, never kill anything). Effort: high.

## Mission
1. Sha-bound amendment: a receipt that fails to parse is classed
   `CORRUPT_DISK_FULL` (keep the bytes under a `corrupt/` sub-path with
   their sha; never overwrite), and an abandoned claim from this
   incident is classed `ABANDONED_DISK_FULL`; both make their unit
   re-runnable ONCE (create-only re-run receipts); the accounting
   charges the abandoned claim's 285 s reservation (unknown usage is
   never zero) and charges nothing for the corrupt one beyond its
   recorded wall if any.
2. Preflight refuses to start while `df /mnt/ForgeRealm` reports
   < 20 GB free (print the number), so this cannot recur silently.
3. `lead_resume_a8.txt`: continue from the first incomplete unit in the
   registered order; projection vs the 6,600 s cap restated.
4. CPU gates: corrupt-receipt classification on the exact truncated
   bytes; abandoned-claim classification; re-run once only; free-space
   preflight refusal (mock df).

## Done (verbatim)
1. Classified files (paths, shas); amendment path + sha; files/lines;
   tests; projection.
2. Exact lead command.
3. Prior art; deviations; RED; process safety; model id and effort.
