# GRM-C5 — RED: t30 separator rescue; t33 remains an admission failure

**CPU finite validation:** S rescues the recorded SC1 t30 answer with **0/20
new control acceptances**; N does not recover the recorded EB1 t33 answer.
W adds **12/20 new control acceptances**, confirming its registered negative
prediction. Neither S nor N regresses an original baseline grounding
acceptance in the scorable set. The combined acceptance bar is **RED**.
The complete [arm × fixture table](offline_r2.md) and
[machine receipt](offline_r2.json) are the primary results.

| Registered outcome | 0 | S | N | W |
|---|---:|---:|---:|---:|
| SC1 t30 served `Cobalt 1 India` | reject | accept | reject | accept |
| SC1 t30 recorded demand-trip text (not selected as served) | reject | accept | reject | accept |
| EB1 t33 recorded abstention | reject | reject | reject | reject |
| WC1 width 64 t33 `Marble-4-Juliet` | accept | accept | accept | accept |
| New false acceptances, all 20 controls | 0 | 0 | 0 | 12 |
| Total false acceptances, all 20 controls | 4 | 4 | 4 | 16 |

Coverage: 126 fixtures × four arms = **504 cells**; **432 scored**, **72
blocked**. The 84 EB1/WC1 rows cover all 14 final longhorizon probes in EB1
and widths 64, 96, 128, 192, 256. Each includes the ten census probes
(t30/t33 plus the eight other originals) and four additional distant probes.
SC1 contributes the actual spaced t30 served answer and its recorded trip;
RT1 contributes both original arms, 18 served answers. There are ten controls
per class and two constructed positive diagnostics. These are correlated
historical rows and author-created controls, not 126 independent trials.

The current rule already accepts four N-class false controls: swapped
relation, negation, alias collision and source negation. S/N preserve those
pre-existing defects because they are additive alternatives. Zero *new*
false acceptances is the registered criterion, not zero false acceptances.
The full table reports grounding only; JSON carries expected-value hits
separately. In particular, grounding counts must not replace historical
battery correctness: WC1 width 256 has 13 grounded rows but only 12 recorded
correct answers.

## Implementations and original check locations

Original grounding: [`core/graft_arena.py:2670`](../../core/graft_arena.py#L2670),
`_grounding_verdict`; pooled coverage is at lines 2688–2744, and production
attribution selects normalized mode at lines 2777–2796. EB1's inherited
probe ladder calls it in [`scripts/grm_e2e_session.py:1886`](../../scripts/grm_e2e_session.py#L1886)
and demand trips at line 1775. SC calls it in
[`scripts/grm_sc1_recovery_gpu.py:276`](../../scripts/grm_sc1_recovery_gpu.py#L276).
Correctness is separate: E2E `contains_accept` / `score_probe`, lines
515–538, and SC `value_verdict`, lines 167–180. This experiment imports
`contains_value` from `scripts/grm_det1_common.py:240` for value-hit reporting.

All alternatives are in [`scripts/grm_c5_rules.py`](../../scripts/grm_c5_rules.py):

| Surface | Lines | Explicit opt-in / scope |
|---|---|---|
| Default pin / dispatcher | 20, 95–126 | `grounding_verdict(..., rule="0")`; default is `"0"` |
| S span rule | 39–53, 75–92, 124–126 | `rule="S"`; full ordered Word-number-Word answer, source-bound to the asked entity/relation; separators fold only inside the answer value |
| N name rule | 30–36, 39–73, 75–92 | `rule="N"`; full entity phrase/relation and exact hyphenated value; no name dictionary |
| N candidate diagnostic | 56–73 | `binding_candidates(..., rule="N")`; compares supplied eligible sources, does not override eligibility or mount anything |
| W negative control | 113–123 | `rule="W"`; global hyphen/whitespace collapse followed by imported pooled grounding |

No production environment flag, default, battery or admission policy was
changed. The API parameter is the experimental flag. S and N are separate
alternatives, not a combined SN arm. Expected answers are absent from the
grounding API. Outside the registered clause/query/code grammar, S/N add
no acceptance beyond baseline. This deliberately restricted candidate does
not cover arbitrary name spellings, punctuation, paraphrases, values or prose.

Default-OFF pin: [`tests/test_grm_c5_grounding.py:17`](../../tests/test_grm_c5_grounding.py#L17).
Reference parity at line 24 compares both normalized modes on every one of
the 108 scorable fixtures and checks that `GRM_C5_RULE=W` cannot enable an
alternative. Existing SC1.1, admission, EB1 and RT1 tests are included.

## Why t33 is not claimed fixed

The EB1 t33 receipt says, verbatim:
`Not in memory: no stored record matches polaris, mark.`
It records `abstain_reason=identifier_unbound`, no mounted sources and
`infer_calls=0`. Source node 35 is in the excluded-live set. Both the current
ADM1 binder (`core/grm_admission.py:98`) and N bind its full source text when
that source is supplied as eligible. Thus these receipts do not support the
premise that a missing proper-name grounding predicate is the cause of t33.
The same pattern holds at WC1 widths 96–256; width 64 already succeeds.
Evidence: `offline_r2.json:t33_binding_audit` and the hash-bound original
instrumentation listed by `registration.json:inputs`.

Changing eligibility/admission or regenerating after a different mount
would be a different intervention. It has not been performed or counted as
a grounding recovery. **Not claimed fixed:** t33, unrestricted relation
grounding, existing baseline false acceptances, or complete RT1 replay.

## Gates, registration, and handoff

- **520 tests passed in 5.40 s**: [CPU receipt](cpu_gate_r2.json),
  [full log](cpu_gate_r2.log). This is an author-run regression suite.
- **5/5 registered mutants killed**, threshold ≥0.80:
  [mutation receipt](mutations.json). Only temporary copies were mutated;
  production and candidate source were unchanged by the mutation gate.
- [Dry-run](dry_run_r2.json): every one of 504 offline cells plus CPU,
  mutation and integrity gates, total wall estimate **145.2 s**, GPU **0 h**.
- [Integrity](integrity.json): **623 pre-existing files unchanged**.
- Registration SHA-256:
  `8127ddcf3768b2a30fa789abf5f61d0ba55ef58a8e56d4230838d4fc411c8964`.
  It binds fixtures, source receipts, counts, thresholds and predictions.
  Executed local source files are fingerprinted in each successful receipt.
- [Blocked report](blocked_report.json), [exact lead commands](lead_commands.txt),
  [append-only ledger](IMPLEMENTATION_LEDGER.md).

The first CPU invocation failed in its receipt hook after the assertion
progress completed: `FileNotFoundError: [Errno 2] No such file or directory:
'/mnt/ForgeRealm/wt/grm-c5/_classes.py'`. Its full failure log and failure
receipt are retained. Only hashing of synthetic `torch.ops` / `torch.classes`
module metadata was corrected. The repeat passed and explicitly records
those two non-file module declarations. The first offline table is retained;
the repeat after the bookkeeping fix has identical rows/gates. No scientific
threshold, fixture, rule or immutable registration was changed after gates.

RT1's 18 linked served-output rows lack exact mounted text, including split
children. Their value-hit checks ran, but 72 grounding cells remain null.
No parent fixture prose was substituted. The next input needed is exact
historical mounted text with an ID/receipt binding; this is a CPU-data blocker.

GPU cells: **none; 0 GPU-h used and estimated**. Every selected served answer
is already present. The registered missing-served-text regeneration condition
is not met. No GPU command is authorized by this evidence. Any later eligible
regeneration requires an immutable amendment first and the order's ≤0.4 h,
285 s worker / 590 s outer / 30 s cooldown rails; non-fit stays non-fit.
Blind verification remains for the lead; no subagent was launched.

**A passing rule is a narrower grounding rule for a named class, not a
prose-grounding certificate.** S's finite evidence concerns the named
source-bound Word-number-Word separator class. It does not certify prose,
full serving-loop selection, model state identity or a product default flip.

## Prior art

Local verified source: SC1.1 glyph normalization/text-only arena, DET1 value
comparator, ADM1/RT1 ordered binding, EB1/WC1 experiment receipts and Scout C5
adversary classes (GRM, 2026). Imported or reused ideas: glyph projection,
exact spans, identifier binding, paired arms, immutable receipts and controls.

External bibliography is **unverified — lead to check**. Search terms:
Thompson (1968), regular expression search; Codd (1970), relational model;
Pearl (2000), Causality controlled interventions; Gardner et al. (2020),
contrast sets; DeMillo, Lipton and Sayward (1978), mutation testing.
Borrowed concepts: exact matching, entity/relation keys, controlled contrasts
and seeded defects. Ours: restricted grammar, this fixture composition and
the offline adapter/receipt assembly. **No prior art known to me for this
exact composition; no novelty claim.** Matching annotations are at code sites
and in the ledger.

## Deviations, RED, process safety, model and effort

The original spaced t30 evidence is in SC1, so it was added alongside the
requested EB1/WC1/RT1 receipts; their already-hyphenated t30 outputs were not
relabeled as recoveries. t33 contradicts the proposed grounding diagnosis.
RT1 incomplete mount-text evidence and the initial receipt-hook failure are
explicit RED entries, not suppressed failures. No registration amendments
or scope-expanding product changes were made.

Foreground CPU work only; no git commands, subagents, background waits,
GPU/model loads, network, process signals, lock changes or service actions.
Writable changes are new scripts, one new test and `artifacts/grm_c5/`.
Author: **GPT-6 (Codex), high reasoning effort**; the exact backend model ID
is not exposed to this seat. Historical served model: **openai/gpt-oss-20b**,
revision `6cee5e81ee83917806bbde320786a8fb61efebee`, TensorCUDA
`resident_packed_mxfp4`; not loaded in this task.
