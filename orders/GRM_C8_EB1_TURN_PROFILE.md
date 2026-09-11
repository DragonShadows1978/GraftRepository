# GRM-C8 — Which expensive track earns the next GPU allocation? Profile the complete EB1 turn. (worktree wt/grm-c8, branch grm-c8, forked from grm-c2)

Origin: `docs/GRM_SCOUT_2026-09-08.md` Part C rank 8. Registered
prediction (Scout's): session routing / admission work yields more
near-term value than APA decode integration at a 96-seat arena. The
decision has two sides and both get numbers: (A) where the wall of a
complete EB1 turn actually goes (lexical scan, route, admission, cold
fetch, seat/mount, prefill, decode, deposit/harvest, fold), and (B)
what APA decode integration could save at this arena width, taken from
the existing APA receipts (SP5: GPT-OSS decode 82.6 / 84.0 / 84.2
ms/tok standard / 2P / SP at 12,288-token context; `/mnt/Shared/APA_SP5_GPTOSS20B_Model_Test_Report_2026-09-08.md`),
not re-measured.

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/wt/grm-c8`.

## Mission
1. An additive profiler (flag-OFF, selected explicitly) that wraps the
   E2E turn pipeline with per-stage wall + CUDA-synced GPU time +
   peak memory, zero behaviour change (pin: served text byte-identical
   with the profiler on vs off on a CPU fake; and a registered GPU
   bit-identity cell).
2. Cells: 3 batteries × the C2 profile geometry, ≥ 30 turns each,
   profiled; report per-stage mean / p50 / p95, the fraction of turn
   wall in decode vs everything else, and the same for a demand-trip
   turn (demand ON for that cell only, registered separately).
3. Side B: from the SP5 receipts, the decode-only saving APA could
   offer at this context (state the number and its evidence class);
   compare it to the non-decode share of the profiled turn. The
   decision rule (registered): if decode < 50% of turn wall, session
   routing/admission work wins the next allocation; otherwise APA
   decode integration is competitive and the lead reports both.
4. Budget ≤ 0.5 GPU-h; leased cells under the rail.

## Done (verbatim)
1. Profiler file/lines, flag, zero-change pins; cell list + estimates;
   blocked-report; exact lead commands.
2. CPU gate results.
3. Prior art; deviations; RED; process safety; model id and effort.

## COMMON RULES
As in `orders/GRM_C3_DNGH_DECOY_CALIBRATION.md` §COMMON: additive only (new scripts/tests/registration JSON; no edits to kernels, batteries, registries, config or flag defaults), registration immutable and sha-bound, NO GPU in the sandbox (build, CPU gates, `--dry-run`, blocked-report, exact `lead_commands.txt` for the leased runner: ≤285 s worker / 590 s outer / 30 s cooldown, non-fit registered never retried), never kill anything, no git, no subagents, foreground, < 10 min per call, Prior Art Directive. Reasoning effort: high.
