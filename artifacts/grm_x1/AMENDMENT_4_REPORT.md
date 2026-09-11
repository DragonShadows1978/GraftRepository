# GRM-X1 amendment 4 — CPU PASS; lead GPU execution pending

Continuation_04 and its source correction are sealed. **245 tests passed** (2 existing SWIG warnings, 14.30 s), including **48 r4 cases across 16 registered gate names**. This is author CPU evidence, not blind verification or a new E2E result. No r4 GPU work or probe occurred.

## 1. Continuation, units, reuse, and CPU results

- Original immutable registration: `artifacts/grm_x1/continuation_04_registration.json`, SHA `288ba57b95fd5bb88e6fa9660fcb4733cddb648867950986bfe6c23d3602b420`.
- Original immutable continuation: `artifacts/grm_x1/continuation_04.json`, SHA `dd6bbf41d33af767936da38eccff279e65359d38cd1690bf5001501acbc6e7a0`.
- Separate source correction registration: `artifacts/grm_x1/continuation_04_source_amendment_01_registration.json`, SHA `98aceb90e852ca5c1de70f94703fdcda662ed9be08e92fc150eae00bc7126269`.
- Separate source correction seal and **effective runtime fingerprint**: `artifacts/grm_x1/continuation_04_source_amendment_01.json`, SHA `44f715a690b5e58c42a675e92df1b997b47598b608ea8c2074f8b13f5d114b02`. Original continuation remains byte-unchanged; validation checks both seals and every unchanged policy field. Fingerprinted r4 claims/receipts use the effective fingerprint.

| Retained cell | Units | Status now |
|---|---:|---|
| oracle_m1_s0 | 12 | 1 reused COMPLETE; 11 pending |
| oracle_m100_s0 | 12 | 12 pending |
| natural_m1 | 24 | Locked until reduced oracle positive |
| natural_m100 | 24 | Locked until reduced oracle positive |

**72 units total; 71 new leases.** Each unit is one query with capture, install, paired forwards and scoring inside its lease. Oracle units have six scored rows; natural units have two: 240 planned rows. Retained units preserve original global ordinal, query, family, arms, conditions, multiplicity, and worker cap.

**Reuse ruling: VALID_BYTE_IDENTICAL.** The canonical `raw_json(unit)` bytes for `oracle_m1_s0__alias_00` equal the sealed r3 bytes (unit-definition SHA `f1debede5d0b7bef9f2150b2089ecea5dc3be9334a16c9679483b4f8c48fc14a`). The GPU driver, all core code, scoring and normalization definitions, original REFUSALS vocabulary, fixtures and dependency inventory remain unchanged. The original receipt at `artifacts/grm_x1/receipts/r3/oracle_m1_s0__alias_00__a1_e8d854f6a6eabbe27602992ba6a804a9b5d4a102417fe726fc8c746d0bcd5b10.json` has SHA `a6fa9cc279ffcd25510348c437190047dfb0f7d8cb119b9e7b3b8e4f31df55d9` and remains untouched. No rerun is scheduled; next unit is `oracle_m1_s0__alias_02`. The original claim and receipt are adopted by explicit SHA-bound paths, rejecting duplicate evidence.

Measured r3 worker wall: **106.258437962 s** (existing E2E receipt). Current charge: **676.258437962 s**, comprising historical reservations 570 s plus reused r3 wall once. r4 charge 0 s. Authorized cap 10,800 s. Planning estimate: **8220.607533 s = 570 + 72 × 106.258438**; remaining 71-unit estimate **7544.349095 s**. This is a projection from one unit, not a runtime bound. Additional 30-second cooldowns total 2,130 s for 71 new invocations, plus lock waits; GPU budget excludes cooldown/lock waits as in r3. Admission still requires room for 285 s; outstanding claims charge 285 s. Actual later cost can stop the reduced grid as NON_FIT_BUDGET.

Dropped cells, explicitly **NOT RUN** (reasoning estimates using the same first-unit wall):

| Cell | Units omitted | Additional estimated GPU seconds |
|---|---:|---:|
| oracle_m1_s1 | 12 | 1275.101256 |
| oracle_m10_s0 | 12 | 1275.101256 |
| oracle_m10_s1 | 12 | 1275.101256 |
| oracle_m100_s1 | 12 | 1275.101256 |
| natural_m10 | 24 | 2550.202511 |

Omitted total: 72 units, 7650.607533 additional seconds. Without double-counting the intersection: m10 drops 48 units (5100.405022 s), then the second oracle seed at m1/m100 drops 24 more (2550.202511 s). All three second-seed oracle cells would cost 36 units; its m10 cell is already included in the m10 group. These seed-named cells partition even/odd query ordinals; dropping the second oracle seed also drops those odd-ordinal oracle queries. Natural retains all 24 queries.

**Positive, restated from r1 on the reduced population:** both reduced oracle cells complete, no active RED/incomplete evidence; B present minimum exact accuracy ≥0.80, B spread ≤0.05 across m1/m100, B100−A100 ≥0.20; B/C present mounted-address coverage=1; C absent false rate=0 at each retained multiplicity; C present accuracy ≥B−0.05 at each. Natural runs only when this is true, without override. Thresholds and scoring unchanged. Original S5 rate comparison retained; natural has 24 queries per multiplicity and oracle 12, so its population is broader. No result is claimed for omitted cells or query strata.

**Served-text diagnostic:** new r4 per-unit rows include `served_text_class` with answer / abstention / refusal-style. Scored abstention takes precedence; an anchored, normalized, limited refusal template recognizes the registered example (including curly apostrophes). Everything else is answer-style, including wrong, empty or verbose output; this label never establishes correctness. Refusal-style remains a false answer under the unchanged r1 scoring unless already scored abstained. Original r3 rows gain derived labels only in the summary, with original receipt provenance. Current counts: 2 answer-style, 1 abstention, 3 refusal-style. The A/present “I’m sorry, but I can’t help with that.” remains false_answer=true, abstained=false. The template is intentionally limited and can miss other refusal phrasings; it is not a semantic refusal detector.

**CPU gate names and results:**

| Test | Cases | Result |
|---|---:|---|
| `test_r4_continuation_accepted` | 1 | PASS |
| `test_r4_forged_continuation` | 8 | PASS |
| `test_r4_stale_continuation` | 3 | PASS |
| `test_r4_historical_bytes` | 1 | PASS |
| `test_r4_dry_run_72` | 1 | PASS |
| `test_r4_reuse_identical_unit` | 1 | PASS |
| `test_r4_unit_definition_drift` | 1 | PASS |
| `test_r4_create_only` | 1 | PASS |
| `test_r4_aggregate_sum` | 1 | PASS |
| `test_r4_resume_next_missing` | 1 | PASS |
| `test_r4_red_advances_second_red_stops` | 1 | PASS |
| `test_r4_budget` | 4 | PASS |
| `test_r4_natural_gate` | 10 | PASS |
| `test_r4_served_text_classes` | 8 | PASS |
| `test_r4_worker_cpu` | 2 | PASS |
| `test_r4_lead_loop` | 4 | PASS |

Receipts: `artifacts/grm_x1/receipts/continuation_04_gates.json` and `artifacts/grm_x1/receipts/continuation_04_lead_checks.json`; full baseline `artifacts/grm_x1/receipts/cpu_baseline_856df4ed4562402db57b2b915aabd86845f112acee2a94455657db83b25fbe2f.json`. Full baseline includes existing r1–r3, address default-OFF, RT1 and RS3 checks, actual local tokenizer, frozen fixtures and shell syntax. Read-only live preflight/dependencies, --dry-run and summary PASS. All **110 frozen historical files**, including r3 claims, receipts, sessions and earlier seals, hash-identical. No r4 GPU claims or receipts. The real shell loop was exercised against a CPU stub: normal/one-RED 71 calls exit 0; budget stop 1 call exit 2; natural locked 24 calls exit 0; registered stop 1 call exit 1. Real controller/worker lifecycle is covered by synthetic CPU execution; stub results are not card evidence.

## 2. Exact lead commands

Run from this checkout:

```bash
cd /mnt/ForgeRealm/wt/grm-x1
bash scripts/grm_x1_lead_gpu.sh preflight
bash scripts/grm_x1_lead_gpu.sh --dry-run
bash artifacts/grm_x1/lead_commands.txt
bash scripts/grm_x1_lead_gpu.sh summary
```

The executable loop file contains exactly:

```bash
#!/usr/bin/env bash
# r4: 72 units, 71 new leases; validated r3 alias_00 reused, never rerun.
# Lead only. Each run below is one foreground lease (<=590 s outer rail).
# Prior art: house X1 foreground controller (2026); new per-query resume loop.
# NON_FIT_BUDGET and registered stops exit for lead decision. A lone UNIT_RED
# continues; a timeout unit is NON_FIT in summary and the next unit is tried.
set -euo pipefail
cd /mnt/ForgeRealm/wt/grm-x1
trap 'bash scripts/grm_x1_lead_gpu.sh summary' EXIT
bash scripts/grm_x1_lead_gpu.sh preflight
for cell in oracle_m1_s0 oracle_m100_s0 natural_m1 natural_m100; do
    while true; do
        result=$(bash scripts/grm_x1_lead_gpu.sh run "$cell")
        printf '%s\n' "$result"
        # Worker progress may precede the final controller JSON. Parse the last
        # JSON object without suppressing malformed/unrecognized controller output.
        status=$(printf '%s\n' "$result" | python3 -c 'import json,sys; s=sys.stdin.read(); i=s.rfind("\n{\n"); v=json.loads(s[i+1:] if i>=0 else s); print(v["status"]); print(v.get("reason", ""))')
        outcome=${status%%$'\n'*}
        reason=${status#*$'\n'}
        if [[ "$reason" == NON_FIT_BUDGET ]]; then
            exit 2
        fi
        case "$outcome" in
            UNIT_COMPLETE|UNIT_RED) ;;
            CELL_COMPLETE|NON_FIT) break ;;
            NATURAL_LOCKED) exit 0 ;;
            *) exit 1 ;;
        esac
    done
done
trap - EXIT
bash scripts/grm_x1_lead_gpu.sh summary
```

Each `run <cell>` executes one next missing unit, then returns. Foreground flock wait 250 s; own work alarm 280 s; worker allowance 285 s; outer rail 590 s; cooldown 30 s. Unit RED advances, explicit non-timeout retry once only; second RED or incomplete claim stops. A timeout is local NON_FIT, without forward chunking. The existing cooperative Python alarm cannot guarantee hard preemption of a non-returning native call; no kill wrapper was introduced.

## Prior art

House X1 r1–r3 (2026), locally verified in existing scripts/registrations, supplies paired query definitions, r1 thresholds, exclusive claims, SHA receipts, unit resume, budget projection and leased foreground execution. Those mechanisms are reused. This amendment integrates the lead-selected subset, historical-unit reuse, separate source correction seal and descriptive served-text labels; no scientific novelty is claimed. No prior art known to me for the exact anchored refusal template. SHA-256: NIST FIPS 180-4 (2015), **unverified — lead to check**, search terms `NIST FIPS 180-4 SHA-256`. The lead handles external literature checks. These annotations also appear at changed code sites and in the ledger.

## Deviations, RED, process safety, model and effort

Mechanical deviation: first author baseline **33 failed, 210 passed**, 2 warnings, 14.46 s. Preserved receipt `artifacts/grm_x1/receipts/cpu_pytest_ee60aefa39f95ecbd934b58b1df4840e129a89a0aacf7621685d5aaaf4c3c7d7.json`. Actual failure: `FileNotFoundError: .../config/grm_demand_registered.json` in isolated fixture; and legacy live-grid assertion `[72, 72, 48, 48] == [72] * 6 + [48] * 3`. Named root causes were missing fixture inputs and testing a legacy matrix against the new live epoch. Separate source correction registration preceded fixes; original continuation/registration not overwritten. Copied the missing real frozen sources into the fixture, kept original matrix assertions in its isolated r2 epoch, and retained a separate real fixture-integrity check. No gate thresholds, scoring or policy weakened. Final suite clears these failures. The correction seal additionally binds validation plumbing and updated test sources.

r1 RED remains `TypeError: 'NoneType' object is not iterable`; r2 RED remains `TimeoutError: GRM-X1 worker reached its 280 s work rail (285 s outer worker allowance)`. r3 remains correctly stopped at NON_FIT_BUDGET under its old cap. r4 live verdict is PENDING; reused output includes wrong/verbose answers and refusals. **Not claimed fixed:** scientific accuracy, natural retention, GPU fit of later units, or the cooperative-timer limitation. No new GPU timing/quality result, independent blind verification or fresh unchanged-core mutation run is claimed.

Only authorized X1 harness/tests/artifacts and append-only ledger changed. No core, GPU driver, order, prior seal, historical receipt, service or sibling edits; no git, subagents, GPU probe/run, background jobs/waits or process kills. CPU subprocesses ran in the foreground. Author model: **GPT-6 (exact deployment variant unavailable), reasoning effort high**. Reader remains **openai/gpt-oss-20b, low**, unloaded here.

Changed source paths and function line numbers are recorded in `continuation_04_delivery.json`.
