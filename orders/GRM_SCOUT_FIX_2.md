# GRM-SCOUT-FIX-2 — split children inherit capture provenance; then re-run C2 under the fixed source (worktree wt/grm-c2, branch grm-c2)

Origin: GRM-C2 amendment 2 diagnosis (committed): width-guard split
children (`core/graft_repository.py:979` child construction) are built
without the capture metadata the parent carries at graft top level, so
SCOUT-FIX-1 B3 (which serializes the key only when present) persists
`capture` for the parent and nothing for children; C2's fresh-phase
validator then sees `end_capture.valid: false` and stops. This is a
production correctness gap of the same class as B3 and gets the same
treatment. Same rules as SCOUT-FIX-1 (minimal diff, pinned by a test
that fails before and passes after, no behaviour change beyond the
finding, no GPU, no git, no subagents, foreground, never kill
anything). Effort: high.

## Mission
1. **Fix**: at child construction, copy the parent's capture metadata
   (the same fields B3 serializes: pin, geometry, shifts, provenance
   hashes as recorded) onto every child, marking the inheritance
   (additive field, e.g. `capture.inherited_from_parent: true` with the
   parent's identity) so a reader can tell inherited from directly
   captured. Old manifests and non-split deposits unchanged. Pin with
   tests: split deposit → children carry capture equal to the parent's
   plus the inheritance mark (RED before / GREEN after); save→reload
   round-trip of a split repository keeps it; the SCOUT-FIX-1 fixtures
   still pass; the B3 old-manifest test still passes.
2. **Where else children are constructed** (RT1 split-child routing,
   LSR split-at-deposit, any restore path): audit and list every site;
   fix the ones that build children from a captured parent; report
   sites that legitimately have no capture (say why).
3. **C2 amendment 3** (sha-bound to this order): the C2 runner's source
   fingerprint now differs; register a new epoch so all 52 cells run
   fresh under the fixed source (the RED cell's receipt and its 46.3 s
   charge stay recorded; cap unchanged 3,600 s), and refresh
   `lead_commands.txt` (executable, `--resume` form) accordingly.
4. CPU gates: full C2 test set + SCOUT-FIX-1 fixtures + the new tests;
   `--dry-run`.

## Done (verbatim)
1. Fix file/lines; every child-construction site with its ruling; test
   names with RED-before/GREEN-after evidence.
2. C2 amendment 3 path + sha; exact lead commands.
3. Prior art; deviations; RED; process safety; model id and effort.
