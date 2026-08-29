# ORDER MOE-E3.1 — can a linear key separate two same-style novels? (CPU-only)

WRITABLE TARGET `/mnt/ForgeRealm/GraftRepository`; rules as MOE-E3.
CPU-ONLY: OLMoE fits your sandbox — self-run everything to completion.

Context: E3-G2 RED (report artifacts/moe_e3/). The DOC-A key vs
{wikitext, code, GRM, guides} negatives achieves recall 0.996 with
clean code/GRM but fires on 0.997 of DOC-B (same-author novel) and
0.20 of guides — a DOMAIN address, not an EPISODE address. E3.1 asks
the sharp question with a stricter, pre-registered fit.

Work (self-run):
1. Capture router-input h for DOC-B windows (all layers; same
   machinery/provenance as DOC-A).
2. Re-fit per-layer K4 with DOC-B ADDED to the negative pool (five
   corpora), FIT/EVAL parity, frozen fit-side tau.
3. Registered rule E3.1-G2: a layer qualifies iff DOC-A eval recall
   >= 0.50 AND doc_b_fire <= 0.10 AND guides FPR <= 0.10 AND code FPR
   <= 0.05 AND grm FPR <= 0.05 (wikitext descriptive). Report all 16
   layers.
4. Also run the K5 logistic-probe ceiling arm (labeled as trained
   linear) under the same rule — if closed-form fails but the probe
   succeeds, that locates the boundary precisely.
5. Registered sentences: EPISODE-ADDRESSABLE (>=1 K4 layer qualifies);
   PROBE-ONLY (K4 none, K5 >=1); NOT LINEARLY EPISODE-ADDRESSABLE
   under these limits (both none) — in the last case state the law
   candidate: router-geometry keys are domain-grade; episode
   selection belongs to the mount stage (GRM semantic routing).

Done: the sentence, per-layer table (both arms), best rows, files,
receipts, anything not done.
