# GRM-SCOUT-FIX-4 — a question that binds to a recency-mounted node is served from that mount, never refused (worktree wt/grm-c7, branch grm-c7)

David's ruling (2026-09-09): restricting admission because a fact is
"too soon" is wrong as a user-visible outcome. Read carefully: the
exclusion of recency nominees from ADMISSION stays (they are already
in the live band; mounting them again is waste). What changes is the
consequence: when the identifier scan binds to a node that is already
a recency mount, the turn is served from that mount — no new mount, no
"no stored record" abstention. Rank-visible, mount-exempt.

Evidence: C7 amendment 3 (`artifacts/grm_c7/r2/amendment_3/REPORT.md`):
`core/graft_arena.py:3372–3382` excludes recency nominees from
admission, `:3412–3429` abstains before any mount; the production
ladder repeats it at `scripts/grm_e2e_session.py:1198–1256`; receipt
`A-009-016/probes.jsonl:4` (`excluded_live_ids [8, 9]`,
`recency_mounted_ids [8, 9]`, `identified_candidates []`). C5's t33
(`identifier_unbound`, source node 35 in the excluded-live set) is the
same rule. Same rules as SCOUT-FIX-1/2/3: minimal diff, RED-before /
GREEN-after fixtures, no behaviour change beyond the finding, no GPU,
no git, no subagents, foreground, never kill anything. Effort: high.

## Mission
1. Core fix, minimal: in the admission/abstention decision, an
   identifier that binds ONLY to recency-mounted (or otherwise
   live-excluded) nodes counts as SATISFIED by the live band: the
   receipt records `served_from: recency_mount` with the node ids, the
   route receipt keeps `admission_ranking_before_demotion` as is, and
   the abstention path is not taken. Nothing else about ranking,
   fit, or point-lookup semantics changes. Apply the same rule in the
   production ladder site if it duplicates the logic (prefer making
   the ladder call the core decision so there is one place).
2. Pin with tests (RED before / GREEN after): (a) the C7 alias case
   from the amendment-3 CPU reproduction
   (`test_alias_recency_exclusion_reproduces_in_core_and_ladder`
   inverted: now served, not refused); (b) the C5 t33 shape (source
   node in the excluded-live set, identifier binds, question answered
   from the live band); (c) a control: an identifier that binds to
   NOTHING still abstains with the same message; (d) a control: an
   identifier that binds to a non-recency node still mounts through
   admission exactly as before (byte-identical receipt on an existing
   fixture).
3. Run the SCOUT-FIX-1/2/3 fixtures and the C7 CPU gates on this
   branch; paste counts.
4. Register the C7 r2 continuation as re-armed under this fix (the
   amendment-3 record said it was blocked on this core stop): cells
   1–3 receipts' validity ruling (the alias row in cell 2 was refused
   by the old rule; say whether cell 2 must be re-run or its alias row
   quarantined), resume at A-024-031, `lead_commands_r2_resume.txt`
   executable.

## Done (verbatim)
1. Fix file/lines; the four tests with RED/GREEN evidence; ladder
   ruling.
2. Fixture counts (FIX-1/2/3, C7); continuation registration path +
   sha; exact lead command.
3. Prior art; deviations; RED; process safety; model id and effort.
