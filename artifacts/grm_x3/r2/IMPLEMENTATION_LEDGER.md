# GRM-X3 amendment 2 execution ledger

2026-09-09 (order dated 2026-09-08). Immutable plan:
`orders/GRM_X3_AMENDMENT_2.md`; registration references its SHA256. House rules
and active X3 orders read. No git or subagents. No GPU authorized in this seat.

Filesystem evidence: before authoring r2, exclusively created
`r1_evidence_before.json`, 5,161 SHA256 pins including runs/summaries/fixtures and
old lead commands. No r1 scoring invocation against historical results occurred.

Registration chronology: `python3 -m scripts.grm_x3_r2_freeze` exclusively created
20 new 8/6/6 recipes, 372 positive/negative controls, immutable registration and
manifest plus separate trust anchors, BEFORE any CPU outcome gate. Construction
checked only token length/width feasibility with the local tokenizer. Registration
SHA256 99abd4dfbed92df335135cdb08f5180bba32a6cbfa8ff2e37b8d1190d236af33;
manifest b848cf7f1af0c9cf2816acaa8dd253394e2837871677574361e64b9cc38acce6.
No registration or scorer amendment was made after freezing.

Implemented isolated r2 scorer/validator/summary/runner files. Original X3 fork,
logit arithmetic and capture validation are reused. Copied lead/worker shells
are scoped to r2 so the original runner and r1 source remain untouched. Added
explicit runtime fingerprint requirement to lead preflight/run/resume and an
explicit selected fingerprint for summary. Implementation manifests freeze the
current amendment-1 dependencies, not their stale pre-amendment source hashes.

Registered interpretation: value-span changes realization; original exact-error
labels remain classifier targets. All20 exact-error statistics and a separately
labeled value-span error table are descriptive. Predictions use realized subsets
with 6/4/4 minimums; truncated runs stay RED. P3-prime uses amendment 2's inclusive
comparison; Q3-prime retains the opposing r1 comparison. The unchanged overall20
kill rule still demands strict improvement. r1 results are not reinterpreted.
0.10 nats was chosen from r1's prior-run sham distribution, not r2 measurements.
Every r2 sham >=0.10 fails that snapshot; fewer than18 passing or fewer than20
same-payload byte-equal makes P1-prime fail; never recalibrate.

CPU baseline receipt: initial r2 tests 11 passed (2.91 s). Added explicit stale
CLI fingerprint rejection coverage, then ran combined r2/r1 suites: 62 passed,
11.59 s, two SWIG deprecation warnings and shutdown warning. No failed baseline
was suppressed, skipped or retuned. New tests/lines are listed in REPORT.md.
Synthetic full four-cell receipt layout validates real DET1 blobs and raw logits;
it is unit-test evidence, not GPU evidence. All372 registered controls pass;
86 positives,286 negatives; all34 registered Dash/Hyphen code points pass.

Source authorization: implementation manifest contains84 current file pins,
SHA256 a5e13f8337d305a6d6b5e380888d3a6a7c21f7d31d4ce06684f63b6296a94887.
Runtime fingerprint 530dcf174adcfb71b16a7bfba04516b004561911f2afbf43576e7a123494f683.
CPU `--dry-run` enumerates four cells and all arms, exit0. Shell syntax and Python
AST pass. CPU preflight exits2, exact error:
`NO GPU: /dev/nvidia0 absent in dispatched sandbox`.
All5,161 r1 evidence pins still match. CPU_GATE_RECEIPT.json and
r1_evidence_verification.json record these checks. No GPU launch or lease.

Prior art: Unicode Consortium UAX15 (1998 onward), NFKC, primary web source
verified https://www.unicode.org/reports/tr15/ ; Unicode16 PropList (2024),
Dash/Hyphen repertoire including Garay verified
https://www.unicode.org/Public/16.0.0/ucd/PropList.txt . Jain and Wallace (2019),
Attention is not Explanation, attention/intervention distinction, verified
https://aclanthology.org/N19-1357/ . Inherited Meng et al. (2022), arXiv2202.05262,
controlled activation interventions, no fresh literature verification claim.
Inherited Kullback and Leibler (1951) KL, unverified — lead to check DOI
10.1214/aoms/1177729694. House DET1/S3/S4/D-NGH/RS3/RS4/EB1/RT1/X3 (2026)
supplies the capture, geometry, mask forks, accuracy arithmetic, hashes and leased
runner. House C5 arm S (2026) supplies the same ordered-span specification;
independently duplicated locally, no C5 reads/imports; lead reconciliation pending.
No prior art known to me for particular negation/boundary vetoes, relational
recipes, quotas, floor or fixed direction; no novelty claim. Explicit comments
at code sites and the report preserve these distinctions.

Deviations/limits: one completed fact turn per recipe; conservative lexical veto
can reject unrelated negation and does not prove semantic entailment or arbitrary
entity attribution. Ambiguous values realize no class. Exact labels retained and
P3 inclusive choice recorded in registration. Native hang deadline, literal device
byte equivalence, natural routing acceptance, tiny sample, first-token scope,
independent blind verification and new mutation score remain unproven/not claimed
fixed. No serving changes. Report and refreshed commands are in r2; old r1
commands are preserved. Four x285 s =1140 s; total reserved X3 is2280 s=0.6333 h,
within0.64 h. Experimental RED/pending, not a successful X3 thesis test.

Process safety: foreground CPU only; no git, subagents, GPU, lease, background
wait, service change, kill or process signal. Every started CPU process finished.
Model target openai/gpt-oss-20b, frozen revision and Harmony low. Author GPT-6 /
Codex; exact deployment subvariant unavailable; high effort requested, runtime
setting not independently exposed. No claims based on unverified historical memory.
