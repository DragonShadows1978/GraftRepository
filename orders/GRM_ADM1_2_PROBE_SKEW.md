# ORDER GRM-ADM1.2 — adjudicate diag probes 7/9: L2-baseline skew vs real policy regression

WRITABLE TARGET `/mnt/ForgeRealm/GraftRepository`; rules as ADM1/ADM1.1.

State: ADM1 r2 (run_20260830T093538Z_3defaa) completed all 34 diag
turns with full per-arm rows (incl. a live P-SYNTH receipt at turn 13:
A-DEC declared-synthesis recall 1 vs k1/k2 recall 0), then exited
status 2: `probe_failures: probes [7, 9]` in the replay leg. Context:
GRM-SUP-L2 flipped DEFAULT ON tonight (@ the L2 commit), legitimately
re-registering exactly three supersession transcripts. ADM1's frames
pin the OLD default via --no-sup-resolve per the coordination check —
but this replay leg may be un-pinned or graded against mismatched
anchors.

Work:
1. Adjudicate probes 7 and 9: (a) L2-flip baseline skew (frame/anchor
   mismatch — name which side is un-pinned, file:line) vs (b) genuine
   ADM-policy-induced regression. Compare the failing answers against
   BOTH old-default and new-default registered transcripts.
2. If (a): align coherently — EITHER fully pin this leg old-default
   with old anchors, OR migrate ADM1 anchors to the new registered
   baselines (prefer the latter; L2-on is production now). Document
   the choice. If (b): STOP and report — that is a policy finding,
   not a fix target.
3. Rerun analysis over the already-captured rows where valid (the 34
   turns of arm data are on disk; don't recompute what stands) and
   emit the corrected replay/verdict command for the lead.

Done: (a)/(b) verdict with evidence; the alignment choice; whether
captured arm rows remain valid; corrected lead command; files;
anything not done.
