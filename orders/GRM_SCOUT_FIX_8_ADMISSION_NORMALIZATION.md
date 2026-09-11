# GRM-SCOUT-FIX-8 — admission's identifier scan must normalize like routing does (worktree wt/grm-c7, branch grm-c7)

Origin: C7 amendment 7 (`artifacts/grm_c7/r3/amendment_7/REPORT.md`,
`DIGESTS.md`, `PER_PROBE.md`): FIX-5 folds retire their ASCII source
nodes, and GPT-OSS writes the digests with NON-BREAKING HYPHENS
(U+2011: `C7‑Fresh‑0`); routing normalizes those, admission's
identifier scan does not, so a question about `C7-Fresh-0` never binds
the digest that now holds the value → "no stored record" (24 admission
refusals in r3; fresh 2/10 and folded 0/10 while r2 mounted them).
Same rules as FIX-1..6 (minimal core diff, RED-before/GREEN-after, no
behaviour change beyond the finding, no GPU, no git, no subagents,
foreground, never kill anything). Effort: high.

## Mission
1. **Core fix:** one normalization for identifier tokens shared by
   routing, admission and the lexical scan (NFKC, all Unicode dash
   code points → "-", casefold as today) — reuse routing's existing
   function rather than a second copy. Digest nodes' identifier sets
   are built from the normalized text.
2. **Pins:** the r3 folded/fresh refusals replayed on the CPU fake
   with the recorded digests bind after the fix (RED before / GREEN
   after, per probe id); C2's 132 recorded plans byte-identical
   (ASCII inputs unaffected); FIX-1..6 suites pass.
3. **Replay registration (lead-run, ≤ 0.5 GPU-h):** the 24 admission
   refusals from r3 replayed from their checkpoints under the fix
   (mount decision + read), scored with the frozen scorer, alongside
   the already-registered amendment-7 middle-prompt contrast (do not
   duplicate it; sequence them in one `lead_commands_fix8.txt`).

## Done (verbatim)
1. Fix file/lines; tests with RED/GREEN evidence; C2 byte-identity.
2. Replay registration path + sha; exact lead command.
3. Prior art; deviations; RED; process safety; model id and effort.
