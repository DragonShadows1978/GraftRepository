# GRM-C7 amendment 2 (lead, 2026-09-09) — arm A completed 39/39 cells but the harness never mounted anything; diagnose before anything else

Lead-run: all 39 arm-A cells COMPLETE (1.9 GPU-h), `--summary A` =
FAIL. The receipts say the campaign did not measure memory at all:
- every `probes.jsonl` row: memory answer is the abstention template
  `Not in memory: no stored record matches c7-fresh-N, unknown.`,
  `residency.mounted_ids: []`, `arena_cur_mount_n: 0`,
  `summed_token_seats: 0` at every one of 300 post-turn residency
  records, while `route_info.admission_ranking_before_demotion` shows
  ranked candidates (e.g. [5,0,1,4,3,7]) — so routing ranked, admission
  mounted nothing;
- every oracle answer is literally `UNKNOWN` with `mounted_ids: []`
  (the full-information oracle also failed, which cannot be a memory
  result);
- all 13 folds rejected with coverage 0.0–0.1 and digest text like
  `For the archive: the ……………………` (generation collapsed into ellipses);
- probe rows carry `expected: None`, `question: None`.
`max_token_seats: 0` makes `residency_bounded: True` vacuous.

Diagnose from the receipts and a CPU replay (fake model, real
repository/admission/oracle code paths): (1) why admission binds no
candidate for identifiers of the form `c7-fresh-N` (identifier-shaped
token rule? hyphenated synthetic ids? case?) — compare with EB1/WC1
fixtures that DO mount; (2) why the oracle path served `UNKNOWN`
(is the source text actually placed in the live window, and is the
oracle prompt the same shape as EB1's?); (3) why fold generation
produced ellipsis runs (prompt shape / stop tokens / the 32-token
budget?); (4) why probe rows lost `question`/`expected`. For each:
harness (C7 scripts) vs core ruling with file/lines. If any is core,
STOP and report it as a SCOUT-FIX finding; do not edit core.

Same worktree (amendment 1 committed), same rules (no git, no
subagents, no GPU, foreground, never kill anything). Effort: high.

## Mission (only if all four are harness defects)
1. Fix them in the C7 scripts; add a CPU gate that would have caught
   each (a fake-model session where the fixture's identifiers MUST
   mount and the oracle MUST answer from the live window; a fold
   digest pin that rejects degenerate text).
2. Register r2 (sha-bound amendment): same fixtures/probes/oracle,
   same 39-cell layout, receipts under an r2 directory; r1's 39
   receipts stay on disk as the harness-failure record; cap for r2 =
   2 GPU-h (lead authorization; total C7 3.9 GPU-h).
3. Refreshed `lead_commands_r2.txt`.

## Done (verbatim)
1. The four diagnoses with receipts quoted verbatim and the
   harness/core ruling; fixes (file/lines); gate test names.
2. r2 registration path + sha; exact lead commands.
3. Prior art; deviations; RED; process safety; model id and effort.
