# ORDER GRM-DET1.4 — replay = fork-from-lived-snapshot; then the race

WRITABLE TARGET `/mnt/ForgeRealm/GraftRepository`; rules as DET1.x.

Adjudication (lead, from comparisons/compare_result: 214 divergent /
1191 equal): the replay harness RECONSTRUCTS arena state and diverges
from lived bytes — arena_kv 96 fields (injection K/V per layer), masks
24, admission dynamic scalars 86. Lived GPT-OSS answers Harbor
correctly ("Nacre‑6‑Blue", harmony formatting); reconstructed replay
refuses. LAW (registered by this order): **a counterfactual replay may
only fork from a lived snapshot's actual bytes; reconstruction is not
a valid counterfactual substrate.**

Work:
1. Convert the DET1 served-counterfactual path to FORK-FROM-SNAPSHOT:
   consume the lived capture's arena K/V, masks, positions, sinks,
   and admission state verbatim (the DET1.3 snapshot format); the
   only permitted deltas are the experiment's registered
   interventions (planted-miss withholding).
2. Gate: forked replay with ZERO interventions byte-matches the lived
   snapshot state AND reproduces the lived answer (comparator
   normalization fix included: strip markdown emphasis + normalize
   U+2011/U+2010 to ASCII hyphen before value comparison — value
   semantics, not glyphs).
3. Chain the required fix/fork source amendment (the DET1.3 receipt
   notes diagnostic-only amendments cannot authorize race resume —
   author the proper amendment).
4. Resume: DET-G0 (planted misses under forked substrate), DET-G1
   all-hooks byte-identity, calibration, the 12+12 race, verdict.
5. One-paragraph law write-up connecting: refusal-basin adjacency
   (GLC), lived-vs-reconstructed divergence receipts, and the
   foundational graft≡in-context gate's scope (mount-level fidelity
   was always proven; RECONSTRUCTION fidelity never was — say it
   plainly for the ledger).

Done: fork-substrate gate receipts; the amendment; race table +
prediction verdict (or honest blockers); the law paragraph; files;
anything not done.
