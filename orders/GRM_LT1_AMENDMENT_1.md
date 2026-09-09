# GRM-LT1 amendment 1 (lead, 2026-09-09) — natural-question admission: evaluate both rules offline; implement the worker; cap 3 GPU-h

Your r1 finding is the important one: on the CPU fake session every
one of the 35 natural recalls was refused by ADMISSION even though
routing ranked the source node first (`first_row`: question "What did
we settle on for Promenade's lamp spacing?", `admission_ranking_before_demotion
[11, ...]`, `source_ids [11]`, refusal "no stored record matches settle,
promenade, lamp, spacing"). The identifier rule
(`core/grm_admission.py:85–119`, all case-folded question tokens must
bind) is tuned for needle ids; a natural question always carries
words the record does not contain. This is a core rule and David
decides it; your job now is to put both sides on the table with
receipts, and to make LT1 runnable. No core edits in this order.
Same worktree (r1 committed), same rules (no GPU, no git, no
subagents, foreground, never kill anything). Effort: high.

## Mission
1. **Both rules offline (CPU, real core admission code).** Rule 0 =
   current (all tokens bind). Rule 1 = only identifier-shaped tokens
   must bind (proper nouns / capitalized names / numbers / hyphenated
   ids, as the lexical scan already distinguishes them; common words
   become non-binding hints). Rule 2 = rank-1 margin path alone (the
   A-DEC threshold as registered) with the identifier rule as a
   tie-breaker, not a gate. Evaluate all three on: (a) LT1's 35
   questions against the frozen repository state at each probe turn
   (mount decision only; no reader); (b) EVERY served question of the
   EB1 / C2 epoch-3 / WC1 batteries with their recorded repository
   states (receipts under `wt/grm-c2/artifacts/grm_c2/epochs/scout-fix-2/`
   and the canonical `artifacts/grm_wc1_opus/`), reporting for each
   rule which currently-correct answers would change their mount
   decision (the regression risk) and which currently-refused ones
   would mount. Implement the alternative rules in a C7/LT1 harness
   module for the offline evaluation only.
2. **Worker.** Implement the resumable GPU worker (8-turn cells, bound
   state, restart cells, the amendment-3 oracle path), fix the
   checkpoint `process_id` omission, and make `lead_commands.txt`
   executable with `--resume`. Cap raised by the lead to **3.0 GPU-h**
   (your 2.504 h projection fits).
3. **Fixture polish (CPU only, no fact changes):** assistant turns
   after fact statements currently repeat one acknowledgment line
   ("That gives us something concrete to design around."); vary them
   naturally (they are not scored) so the transcript reads like a real
   working session; keep every user turn, fact, probe, and the
   manifest's scored content unchanged except where the sha must be
   re-registered (state the new sha and what changed).
4. Register LT1 as runnable under Rule 0 (today's behaviour) so the
   both-sides run can proceed whatever David decides, with the
   admission rule recorded per receipt; if David flips the rule, the
   run is repeated under it (that is the registered plan).

## Done (verbatim)
1. The three-rule table (LT1 mounts per rule; EB1/C2/WC1 regressions
   and gains per rule, with question ids); receipts.
2. Worker: file/lines, tests; new manifest sha if any; registration
   amendment path + sha; exact lead command.
3. Prior art; deviations; RED; process safety; model id and effort.
