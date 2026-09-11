# GRM-X1 amendment 3 (lead, 2026-09-09) — r3: rail-sized units with incremental receipts

r2 `oracle_m1_s0` is RED at the 280 s work rail
(`artifacts/grm_x1/receipts/r2/gpu_oracle_m1_s0_374c534e….json`): the
payload fix held (no TypeError), the alarm fired inside an ordinary
GPT-OSS expert forward (`core/gpt_oss20b_tc.py:1025`), and `rows` is
empty, so the whole cell's work was lost. This is not a native hang; the
cell is simply larger than one lease. The r1 and r2 REDs stay as
recorded (`historical_red`), the registered stop held, and the lead now
decides: **continue as r3 with rail-sized units.** Same worktree (r2
committed as 2c26978), same rules (no git, no subagents, no GPU,
foreground, never kill anything). Reasoning effort: high.

## Registered by this amendment (continuation_03, sha-bound as before)
1. **Units.** Each registered cell becomes an ordered list of units,
   one unit = one query (capture + install + forward + score) or one
   family if a family's queries share the captured template (state
   which; keep the template capture inside the unit so a unit is
   self-contained). Each unit runs under its own lease (worker ≤ 285 s,
   work alarm 280, outer 590, cooldown 30) and writes a create-only
   unit receipt IMMEDIATELY on completion (rows, wall, digests). A cell
   is COMPLETE when all its unit receipts exist; `summary` aggregates
   units; `run <cell>` executes the next missing unit of that cell and
   returns (one unit per invocation, so the lead's loop stays one lease
   per call). A unit RED is a unit RED; the cell continues with the
   next unit and reports the RED in its aggregate (the registered stop
   applies to a unit that REDs twice under r3).
2. **Timing evidence.** The first completed unit's wall becomes the
   planning estimate for the rest; if a single unit cannot fit 280 s,
   that unit is registered NON_FIT (no chunking of the forward). The
   r2 receipt shows one cell exceeded 280 s; you do not know the
   per-unit wall yet, say so.
3. **Budget.** The lead authorizes up to 1.5 GPU-h total for X1 across
   r1–r3 (5,400 s incl. the two consumed 285 s reservations); record
   it. If the unit count × estimate exceeds that after the first unit,
   the `summary` says NON_FIT_BUDGET and the lead decides.
4. CPU gates: unit enumeration and ordering; create-only unit receipts;
   aggregate equals the sum of units on a synthetic layout; resume
   picks the next missing unit; a unit RED does not block the next
   unit; r1/r2 receipts byte-unchanged; `--dry-run`; refreshed
   `lead_commands.txt` (a loop-friendly form: `run <cell>` repeated
   until it prints CELL_COMPLETE or NON_FIT, then the next cell).

## Done (verbatim)
1. continuation_03 path + sha; unit scheme and count per cell;
   files/lines; test names and results.
2. Exact lead commands.
3. Prior art; deviations; RED; process safety; model id and effort.
