# GRM-X2 implementation ledger (append-only)

2026-09-08 — Read `/mnt/Shared/HOUSE_RULES.md`, local AGENTS.md, immutable
`orders/GRM_X2_RELATIONAL_WITNESSES.md`, EB1 order, scout Part B/D/E, existing
grounding, frame, deposits and fresh-build drivers. Filesystem worktree pointer
and HEAD name `grm-x2`; no git command used. Memory registry used only to orient
the repository surfaces; all implementation conclusions verified locally.

Registration BEFORE any gate: `python3 scripts/grm_x2_freeze.py` (exit 0).
Registration SHA-256 `92bc8695b2561dc1db68954d5025b2c3d505f24470b73e4985c26c705ae23373`.
40 CPU constructions and 24 live questions frozen; exact hashes in
`artifacts/grm_x2/fixture_manifest.json`. Both prediction sets, kill rail,
five mutation operators, six GPU cells, and budget are registered there.
Evidence class: preregistered design, NOT a test result.

Implementation decision: scoped files allow a new core module, not edits to
existing serving code. `core/grm_x2_witnesses.py` offers opt-in instance deposit
hooks and an explicit post-generation/pre-commit answer gate. Existing serving
paths do not activate merely by setting the environment flag. OFF must be an
identity operation. No new metadata reaches prompt encoding. Unsupported answer
clauses reject. Conjunction joins retain source versions; no transitive reasoning.

Prior art: Fader, Soderland & Etzioni, ReVerb (2011), relation triples and
syntactic constraints, https://aclanthology.org/D11-1142/; Buneman, Khanna & Tan,
Why and Where (2001), source-location provenance,
https://www.pure.ed.ac.uk/ws/files/16509989/Why_and_Where_A_Characterization_of_Data_Provenance.pdf;
Green, Karvounarakis & Tannen, Provenance Semirings (2007), conjunctive evidence,
https://web.cs.ucdavis.edu/~green/papers/pods07.pdf. Primary source records read
through web search in this seat. No ReVerb or semiring code copied. Our additions:
small hand grammar, strict complete-answer coverage, versioned native-K/V
sidecars, and join receipts. House SC1.1/EB1/RS3/RS4/RT1 (2026) supplies existing
normalization, frame, capture/seating and routing. Thompson regex (1968), Fisher
controlled comparisons (1935): unverified — lead to check those author/title/year
search terms. No novelty claim for extraction, provenance, or paired gates.

2026-09-08 — Initial unit gate (foreground, no GPU):
`PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider tests/test_grm_x2_witnesses.py`.
Actual result: `1 failed, 47 passed in 0.15s`. Failure:
`test_reported_or_conditional_source_is_not_affirmative_evidence[The owner of Orion is not known.]`.
The parser emitted `{entity: orion, relation: owner, value: known, negated: true}`.
Named cause: epistemic placeholder classified as an entity/value atom. Before
the frozen 40-case gate, added placeholder rejection. Source audit also added
conditional/reported-conjunction scope preservation, with tests. No fixture or
numeric threshold changed. Follow-up: `48 passed in 0.13s`, then the expanded
witness/runner set: `56 passed in 0.22s`. Evidence class: author unit tests.

2026-09-08 — Existing SC1.1 grounding and EB1 CPU tests, unchanged:
`208 passed, 2 warnings in 4.56s` (SWIG deprecations). No full-suite claim.
This also confirmed normal imports are usable without GPU forwards. Added
actual ArenaCache default-OFF identity and existing native-metadata seam tests.

2026-09-08 — `python3 scripts/grm_x2_validate.py` exit 0. Evidence class:
selected CPU suite + frozen falsifier. `266 passed, 2 warnings in 4.69s`;
`cpu_gate.json`: A TP=10 FP=30 TN=0 FN=0; witness and B TP=10 FP=0 TN=30 FN=0.
P1 A accepted 20/20 swapped/negated; P2 B rejected 20/20 and accepted 10/10
correct paraphrases. Scattered errors rejected 10/10. PASS, registered CPU
kill not hit. Both the original and current CPU receipts preserve all rows.

2026-09-08 — `scripts/grm_x2_mutations.py`: 5/5 non-error mutants killed.
Operators: ignore entity, relation, polarity, unknown clause, or source
version. Copies only under `artifacts/grm_x2/mutants/`; production never
overwritten. Module SHA unchanged:
`357d233ec008a67caaf6c6ba32e7b470f1c520b944ab366433be207d06cae704`.
Evidence class: author mutation baseline; no blind red team. Prior art:
DeMillo/Lipton/Sayward (1978), unverified — lead to check: Hints on Test Data
Selection. House §8 mutation protocol reused. Our contribution is the attack
operators and rejection invariants, not mutation testing itself.

2026-09-08 — Before ANY GPU data, amendment 001 clarified rejection counting:
report raw/A/B scores and both raw-correct and incremental A-correct losses.
P3 uses the stricter raw-correct count; unknown oracle forms block P3. Numeric
thresholds, fixtures, extractor and predictions unchanged. Stored separate
`amendments/001_metric_audit.json`; original registration never edited.
`python3 scripts/grm_x2_validate_001.py`: `267 passed, 2 warnings in 4.75s`,
same CPU matrices, unchanged-module mutation receipt checked and reused.

2026-09-08 — Source audit found the direct `_attempt` driver needed
`ArenaCache.step`'s layer `live_shift` assignment before `eb1_begin_turn`.
`capture_pin` restores the prior shifts; EB1 open alone does not set them.
Added `open_turn` with the production prelude and a CPU ordering test;
layer shifts enter live receipts. Amendment 002 records this mechanical
geometry correction before GPU, preserving the registration. Prior art:
house EB1/RS3 `ArenaCache.step` (2026), directly reused; no new algorithm.

2026-09-08 — Final gate command:
`PYTHONDONTWRITEBYTECODE=1 python3 scripts/grm_x2_validate_001.py 002`.
Actual log: `268 passed, 2 warnings in 4.75s` (60 X2 + 208 unchanged
SC1.1/EB1 checks). CPU matrices unchanged, P1/P2 PASS, original witness
module SHA still matches the 5/5 mutation receipt. Shell syntax PASS.
Active files: `validation_002.json`, `cpu_gate_002.json`, `dry_run_002.json`.
Fingerprint: `1a972a26f78e917cc0bd2ec676139687e2121bc7869ed5560daf1d10febf8171`.
Evidence class: selected CPU suite; neither kernel gate nor E2E receipt.

2026-09-08 — Runner delivered: `scripts/grm_x2_lead_gpu.sh
list|run CELL|resume|summary`; Python also supports `--dry-run`. Every GPU
cell is enumerated in dry-run: 6 cells / 24 questions / 48 paired scores.
The original CLI `--dry-run` and `summary` ran on CPU, exit 0; final
enumeration/summary have separate 002 receipts. No GPU preflight or worker
ran. Model index and 3/3 nonempty shards and native-library existence checked
read-only, recorded in `local_prerequisites.json`; this is NOT model-load
validation or full weight checksum validation.

Runner prior art: established house foreground `flock` lease and EB1/RS3
production seams (2026); POSIX/util-linux flock history unverified — lead
to check: util-linux flock manual. Flat content-addressed receipts use the
existing house SHA-256 pattern; Merkle (1979) is an unverified lead, not a
claim that a Merkle tree was implemented. Python AST method selection reuses
PSF AST tooling (date 2006+ uncertain; unverified — lead to check: Python AST
module history). House receipt-manifest/checksum discipline reused for
`DELIVERY_MANIFEST.json`. Our work is X2-specific orchestration and records.

2026-09-08 — Scope and RED handoff: `REPORT.md`, `BLOCKED_REPORT.md`, exact
`lead_commands.txt`. No GPU access in seat; P3 UNMEASURED. Bounds are a
registered design, not timing evidence: 285-second worker self-timer,
240-second lease wait, 30-second foreground cooldown, 590-second outer
limit, 1710-second total worker reservation. Python cannot hard-bound a
stuck native call while never terminating processes; strict wall compliance
is RED and needs a lead decision, not silently accepted. A is the existing
grounding predicate used as a binary gate; production fallback/lived-session
reproduction is not claimed. Fixed fitting mounts do not exercise RT1 split
children. General prose, equijoins, source authority, full answer-memory
integration and live prompt isolation remain unverified. No blind verifier
was dispatched because the seat may not spawn subagents.

Process safety: no git, subagents, background waits, GPU kernels/model loads,
process termination, service changes, external messages or sibling writes.
All calls foreground and below the per-call limit. Only authorized X2 paths
written. Author model: GPT-6 (system identity); exact backend variant and
author reasoning effort unavailable. Evaluated model: openai/gpt-oss-20b,
Harmony reasoning low, registered only, not run.
