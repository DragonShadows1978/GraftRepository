# ORDER GRM3P-DIAG — Why does the UNBOUNDED arena fail? (Sol-max, diagnostic)

YOUR WRITABLE TARGET is /home/vader/GraftRepository-three-pass — edits,
GPU runs authorized. Production /mnt/ForgeRealm/GraftRepository and the
merge-train worktree are READ-ONLY. No git (lead commits). No
subagents. GPU: `flock -w 7200 /tmp/forge-gpu.lock` — ANOTHER SEAT is
running long suite legs tonight; you WILL wait on the lock sometimes.
Never break or bypass the lock.

## The phenomenon (P4 receipts, lead-verified)

`--mode full` composed E2E, GPT-OSS-20B smoke stack:
- UNBOUNDED (no --vram-budget-mb): recall 7/9, BOTH pipelines,
  identical misses — turn 5 (orion pin probe) answers "The current
  Orion PIN value is Vortex-3…" and turn 13 (orion, post-supersession)
  answers "Vortex-3-Sierra." — both times the CYPHER BRIDGE's value.
- 4MB LRU budget: 9/9 both pipelines. Same script, same env.
Sessions on disk: artifacts/grm_three_pass/p4_ffull_{single,three_pass}
and p4_fcold_{single,three_pass}. Pipeline-independence means the
mechanism is arena/mount/read-level, not scheduler-level.

## Mission: CONVICT the mechanism. Do not fix anything.

This is a diagnostic. The deliverable is a convicted mechanism with a
decisive receipt per conviction, WO-10D style. Candidate hypotheses
(pre-registered; you may add more, labeled, before running arms):
- **H-COMOUNT**: co-mounted distractor collapse (Corpus-100 law:
  co-mounted siblings collapse reads) — unbounded mounts more/other
  nodes at probe turns than the budgeted arena does.
- **H-FIT**: mount width fitting differs with residency — budget
  changes which of the top-k actually fit/mount.
- **H-ECHO**: live-window echo (weakly favored against — F-COLD ran the
  same live-turns=2 and passed).
- **H-ROUTE**: ranking itself differs unbounded vs budgeted.

## Registered arms (run in order; STOP at conviction + one confirmation)

- **ARM A — artifact autopsy (NO GPU).** From the four on-disk P4
  sessions: for turns 5 and 13 (and a passing control turn), table
  route ranking_ids, source_rank, mount_plan, mounted_ids/mount_fitted,
  live-window residency, answer — F-FULL vs F-COLD side by side. State
  which hypothesis the deltas support.
- **ARM B — reproduce (1 GPU session).** Rerun F-FULL `single` frame
  exactly (`--mode full`, defaults, env GRM_GQA_CUDA_ROUTE=1
  GRM_GRAFT_STORAGE_BITS=8): confirm 7/9 with the same two misses.
  Determinism precedent: P4 reruns were byte-identical.
- **ARM C — causal ablation (existing CLI ONLY, no code changes).**
  C1: F-FULL `single` with `--topk 1` (probe mounts rank-1 alone).
  9/9 => H-COMOUNT convicted (co-mounts are the poison).
  Still 7/9 => H-COMOUNT disfavored at topk granularity; proceed.
  C2 (only if C1 does not convict): `--topk 2`, and/or `--live-turns 1`
  as the echo control. One variable per run, each pre-labeled.
- **ARM D — witness (ONLY if A–C leave ambiguity).** Instrument
  per-mounted-node attention mass on the miss turns (S1-style readout).
  Any instrumentation must be additive, default-off, and leave the
  default path byte-identical (verify: rerun default smoke, transcript
  sha must stay 68da84b8824eb79a14f65a692a602215bc53d84232cd041002410f51657bba4f).

## Honesty rails

- Convict ONLY on a decisive receipt; if arms end ambiguous, the
  verdict is "not identified under [these instruments]" + name the
  un-searched space. No mechanism-shopping after results.
- A null is a result. Do not tune frames/thresholds mid-arm; new arms
  may be added only with an explicit pre-registration paragraph in your
  report BEFORE that arm's results.
- No fixes, no recommendations beyond "named successor" one-liners.

## Done

Final message MUST contain verbatim: ARM A table, ARM B result line,
each ablation arm's frame + probe scorecard line, the conviction
statement with its decisive receipt (or the honest ambiguity
statement), session dirs for every run, and residuals.
