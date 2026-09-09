# GRM-RD1 — the mounted-but-unread failure: a bounded reader contrast on the C7 checkpoints (worktree wt/grm-rd1, branch grm-rd1, forked from grm-c7 at amendment 4)

David (2026-09-09): "move forward with the best options to resolve the
issues." The dominant failure across the whole Scout follow-through is
the read step: C7 r2's 30 answerable abstentions were ALL mounted-but-
unread (0 admission refusals; `artifacts/grm_c7/r2/amendment_4/REPORT.md`),
and the full-information oracle itself failed 16/34 with the record in
the prompt (8 fresh, 8 alias; corrections 8/8). X1's served values,
X2's 7/12 abstentions and C5's t33 are the same class. Before anyone
changes serving, measure which lever moves it. Registered predictions
are yours (lead prior: the abstention instruction and reasoning-low
dominate; the answer budget does not). Effort: high; no core edits;
harness only; no GPU in the sandbox; no git; no subagents; foreground;
never kill anything.

## Mission
1. **Replay cells from bound state.** For each of the 30 mounted-but-
   unread probes and the 16 oracle failures, replay the probe turn from
   the r2 checkpoint that precedes it (`fix4_attempt_1/cells/*/checkpoint`,
   `session_state`) with the SAME mounts, under arms:
   A0 baseline (as run); A1 reasoning medium; A2 the abstention
   instruction removed from the probe prompt (the "say unknown if not
   in memory" clause — quote what the prompt actually contains); A3
   answer budget 64 tokens; A4 = A1+A2. Memory and oracle rows both.
   Every arm records served text, exact/wrong/abstain under the frozen
   C7 scorer (with the `unknown`/`UNKNOWN` case contract fixed for
   the control rows only — state it), and `served_from`.
2. **Cells:** one lease per (probe, arm) is too many; batch by cell
   checkpoint (all probes of one checkpoint in one lease, ≤ 285 s;
   estimate from r2 walls); cap 0.8 GPU-h; register NON_FIT rather than
   trim.
3. **Output:** a table arm × class (fresh / alias / correction /
   folded) × {memory, oracle} with exact counts, and the registered
   verdict rule: an arm "moves the reader" if memory exact rises by ≥ 8
   of 30 with wrong values ≤ 2.
4. CPU gates: checkpoint replay reproduces the r2 served text for A0
   on the fake model; arms alter exactly the intended prompt/budget/
   reasoning fields (diff pinned); `--dry-run`; blocked report;
   `lead_commands.txt`.

## Done (verbatim)
1. Registration path + sha; the quoted probe prompt; cells + estimate;
   predictions.
2. CPU gate results; exact lead commands.
3. Prior art; deviations; RED; process safety; model id and effort.
