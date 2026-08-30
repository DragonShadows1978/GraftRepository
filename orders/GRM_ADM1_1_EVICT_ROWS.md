# ORDER GRM-ADM1.1 — diag/fix: evict_rows failure under k=1 admission frame

WRITABLE TARGET `/mnt/ForgeRealm/GraftRepository`; rules as ADM1.

State: ADM1 GPU sweep (logs/grm_adm1_gpu_lead.log, run_20260830T091350Z_3dc282)
failed in the diag replay frame, turn 5:
`RuntimeError: evict_rows failed shape=(1, 8, 47, 64) dim=2 head=105 drop=61`
raised through grm_e2e_session.py:2005. head=105 against an 8-head
tensor suggests index-role confusion (head vs row/seat) in an eviction
path that the k=1 arm's arena occupancy exercises and default k=3 may
never reach.

Work:
1. Root-cause: is this (a) ADM1's frame passing malformed args, or
   (b) a LATENT runtime/arena bug in the evict_rows call chain that
   non-default occupancy exposes? Name file:line either way. If (b),
   assess whether any PRODUCTION path can reach it (that assessment
   is a deliverable regardless of fix).
2. Fix minimally at the convicted site. If (b), add a regression test
   pinning the k=1-occupancy shape. No semantics changes elsewhere.
3. CPU-validate (the e2e session machinery runs CPU-side per prior
   receipts); self-run the failing turn if reproducible in-sandbox.
4. Hand back for lead rerun of the ADM1 sweep (bare invocation,
   PYTHONPATH=repo-root).

Done: (a)-vs-(b) verdict with file:line; production-reachability
assessment; diff summary; regression receipt; anything not done.
