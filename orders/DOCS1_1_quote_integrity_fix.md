# DOCS1.1 — Quote Integrity Fix Round (follow-up to DOCS1)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` — edits to
`docs/GRM_Methodology.md` and `docs/GRM_PAPER_DRAFT.md` AUTHORIZED.
Documentation-only; no code, no GPU runs, no git, no subagents, no network.
RED honesty; no monitor-idling.

## The defect (lead-verified against disk)

DOCS1's sweep edited **inside verbatim quotes while keeping their
citations**, and renamed historically-registered gates. The qualifications
added were correct in substance but must live OUTSIDE quoted material.
Specifics:

1. `docs/GRM_Methodology.md` — the blockquote attributed to
   `GraftRepository_Memory_Architecture.md:8-11` was rewritten. The source
   still reads "unbounded memory at bounded residency, zero training, on a
   frozen model" (verified today). A quote with a line citation must match
   its source verbatim.
2. `docs/GRM_Methodology.md` — the `test_graft_infinite.py` docstring quote
   was rewritten and its "(verbatim)" marker dropped. The test still opens
   with `"""INFINITE-CONTEXT gate: ephemeral boat ("clear the memory window
   at the start of each turn"...` (verified today).
3. `docs/GRM_Methodology.md` §6 heading and the experiment-ledger row
   renamed the INFINITE gate to "LONG-HISTORY". The registered gate name is
   INFINITE-CONTEXT (`tests/test_graft_infinite.py`); public docs must not
   silently rename registered gates.
4. `docs/GRM_PAPER_DRAFT.md` — the evaluation-table rows "Infinite context"
   were silently renamed and results text rewritten in place. The reference
   doc's rule (orders/REFERENCE_PUBLIC_DOC_RECOMMENDATIONS.md): do not
   silently rewrite historical experiments; add dated notes instead.

## Required fixes

1. **Restore both quotes in `GRM_Methodology.md` verbatim** (byte-for-byte
   against their sources, including the "(verbatim)" marker on the test
   docstring quote). Immediately AFTER each restored quote, keep the
   operational qualification as the document's own voice, e.g.: "The
   'unbounded' in this design statement means bounded active residency: it
   does not mean zero host/disk growth, universal recall, or unlimited
   addressable storage."
2. **Restore the registered gate name** in §6's heading context and the
   ledger table: the row is the INFINITE-CONTEXT gate
   (`test_graft_infinite.py`); keep the qualification as an appended
   clause, not a rename. Same for the "standing reference" sentence that
   now says "long-history".
3. **`GRM_PAPER_DRAFT.md` results table**: restore the original row labels
   ("Infinite context") and original result text, then append the
   qualification/adjudication as clearly-dated bracketed notes in the row
   or as a dated footnote block under the table (e.g. "[Adjudication
   2026-08-04: 'infinite context' = bounded active residency; host/disk
   growth and recall quality remain real limits.]"). The abstract/prose
   edits (attention "families" wording, Gemma adapter-vs-certification,
   APA-negative adjudication, the added qualification sentence) are
   current-claims prose and MAY stand as edited.
4. Re-check that no other file from DOCS1 contains rewritten quoted
   material: grep every file you and DOCS1 touched for `>`-blockquotes and
   quotes citing a `file:line`, and verify each against its source. List
   each checked quote and its verdict in ## Done.

## Done

Your final message MUST contain verbatim:
1. The restored quote texts side-by-side with their source lines (proof of
   byte-match).
2. Diff summary (`git diff --stat` equivalent by file description — you do
   not run git; list files + what changed).
3. The quote-audit table from fix 4.
4. Any fix you could NOT complete, stated RED with the reason.
