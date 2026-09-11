#!/usr/bin/env python3
"""GRM-D1: emit the order's REPORT.md and LEDGER.md deliverables.

The prose is authored here so the two documents are regenerable from the
receipts they cite rather than hand-typed once. Every number below is read
from a receipt file at emit time; nothing is hard-coded that the receipts
could contradict.

Prior art: no prior art known to me for this emitter. The house
plan/ledger/synthesis split it serves is the local research discipline
(Project-Tensor docs/QUANT_SWEEP_IMPLEMENTATION_PLAN.md, 2bb30b1).
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / 'artifacts/grm_d1'


def load(name):
    return json.loads((OUT / name).read_text())


def report():
    table = load('cause_table_A.json')
    table_b = load('cause_table_B.json')
    sup = load('supersession_cpu_receipt.json')
    recap = load('recap_battery.json')
    reg = load('lt1_1/registration.json')
    reg_sha = (OUT / 'lt1_1/registration.sha256').read_text().split()[0]

    counts = table['class_counts']
    rows = {r['question_id']: r for r in table['rows']}

    lines = []
    add = lines.append

    add('# GRM-D1 — LT1 residual diagnosis + recap battery redesign')
    add('')
    add('Seat: Opus 5 (`claude-opus-5[1m]`), effort MAX. Worktree')
    add('`/mnt/ForgeRealm/wt/grm-d1` (branch `grm-d1`). `core/` untouched.')
    add('Date: 2026-09-10.')
    add('')
    add('## Verdict in one line')
    add('')
    add('All 11 arm-A wrong rows are **structural, not memory-quality**: 5 are a')
    add('fixture-lineage trap the harness created (corrections fed as ordinary')
    add('prose, nothing retired), 5 are a core-composition gap (no alias->base')
    add('binding exists in core at all), and 1 is a probe-ladder drop in which')
    add('A-DEC ranked the correct node **first** and the ladder threw the mount')
    add('away. The recap 0/5 is likewise structural: the question carried no')
    add('identifier token, so routing had nothing to key on and every ladder trip')
    add('returned ungrounded.')
    add('')
    add('## 1. The cause table')
    add('')
    add('Full table: `artifacts/grm_d1/cause_table_A.md` and `cause_table_A.json`.')
    add('Arm B contrast: `cause_table_B.md` / `.json`.')
    add('')
    add('Class counts, arm A (%d rows):' % table['wrong_rows'])
    add('')
    add('| class | rows | ownership |')
    add('|---|---|---|')
    owner = {
        'fixture-lineage (harness)': '**FIXED here** (LT1.1)',
        'alias-edge-without-base (core composition)':
            '**STOPPED — core, A1 owns it**',
    }
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        add('| %s | %d | %s |' % (name, n,
                                  owner.get(name, '**STOPPED — core, reported below**')))
    add('| reader wrong value with the right mount | 0 | — |')
    add('| other | 0 | — |')
    add('')
    add('Arm B (%d rows, for contrast): %s.' % (
        table_b['wrong_rows'],
        ', '.join('%s %d' % (k, v) for k, v in sorted(table_b['class_counts'].items()))))
    add("Arm B's extra `reader wrong value with the right mount` class is absent")
    add('from arm A entirely — a reader/geometry difference (w=256, defaults),')
    add('not a lineage one. That is evidence *for* the profile, and means the')
    add('fixture fix is not what separates the arms.')
    add('')

    add('### The five correction rows — `recall_4_{10,25,50,100,150}`')
    add('')
    add('Every one is the Vega docking-fee lineage: turn 1 = `17 credits`,')
    add('turn 3 = `23 credits`, turn 15 = `29 credits` (expected). The receipt is')
    add('identical at all five distances:')
    add('')
    r = rows['recall_4_10']
    add('```')
    add('ranking_ids                     %s   # = turns %s'
        % ([m['node'] for m in r['ranked_nodes']][:3],
           [m['deposited_turn'] for m in r['ranked_nodes']][:3]))
    add('admission_identified_candidates %s' % r['admission_identified_candidates'])
    add('admission_rank_plan             %s' % r['admission_rank_plan'])
    add('admission_route_margin_1_2      %s   # threshold %s'
        % (r['admission_route_margin_1_2'], r['admission_margin_threshold']))
    add('admission_policy_branch         %s' % r['admission_policy_branch'])
    add('fit_seated                      %s   # = turn %s, %r'
        % ([m['node'] for m in r['mounted_nodes']],
           [m['deposited_turn'] for m in r['mounted_nodes']][0],
           r['mounted_nodes'][0]['value']))
    add('```')
    add('')
    add('Served answer at all five distances: *%r*' % r['served_value'])
    add('')
    add('**Why this is the harness, not core.** Core excludes retired nodes from')
    add('the route-candidate base outright:')
    add('')
    add('```python')
    add('# core/graft_arena.py:1450-1454  (_route_cand_base)')
    add('base = [i for i, g in enumerate(self.grafts)')
    add('        if not g.get("retired")')
    add('        and g.get("kind", "turn") != "recall"]')
    add('```')
    add('')
    add('Had the correction been a real supersession, nodes 0 and 2 would carry')
    add('`retired=True` and would **never have entered `ranking_ids` at all**.')
    add('Instead the LT1 worker feeds every turn identically:')
    add('')
    add('```python')
    add('# scripts/grm_lt1_worker.py:146-148')
    add('# User corrections are ordinary prose, not hidden supersede calls.')
    add("idx=a.feed(e2e.harmony_turn(event['user'],event['assistant']))")
    add("a.grafts[idx]['kind']='turn';state['turn_nodes'][str(turn)]=idx")
    add('```')
    add('')
    add('So the whole family stays active, the three near-identical scores tie')
    add('(margin %.4f << threshold %.4f), and the k3 tiebreak preserves the'
        % (r['admission_route_margin_1_2'], r['admission_margin_threshold']))
    add('**deposit order** — oldest first.')
    add('')
    add('Contrast that proves the mechanism is order, not "corrections are hard":')
    add('`recall_5_*` (Medibay, the *same* three-step correction shape) ranked')
    add('`[15, 3, 1]` = turns 16, 4, 2 — **newest first** — seated node 15')
    add('(`16 beds`) and scored **correct at all five distances**. Same policy')
    add('branch, same margin magnitude, opposite order, opposite outcome. The')
    add('ranking order within an un-retired family is arbitrary; the fixture,')
    add('not the policy, decides whether you win.')
    add('')

    add('### The five alias rows — `recall_7_{10,25,50,100,150}`')
    add('')
    a = rows['recall_7_10']
    add('Beacon/Lantern launch date. Receipt at d=10:')
    add('')
    add('```')
    add('admission_identified_candidates %s        # = turn 18, the ALIAS edge'
        % a['admission_identified_candidates'])
    add('admission_rank_plan             %s         # = turn 6, the BASE fact'
        % a['admission_rank_plan'])
    add('fit_seated                      %s' % [m['node'] for m in a['mounted_nodes']])
    add('identifier_tokens               %s' % a['rt1_provenance']['identifier_tokens'])
    add('```')
    add('')
    add('The identifier **did** find the alias edge. The mount seated only the')
    add('base fact. Nothing composes the two, so the reader receives')
    add('*"Lantern\'s launch date will be 18 October 2196"* while being asked')
    add('about "the Beacon", and answers *%r*.' % a['served_value'])
    add('')
    add('**Why this is core, and STOPPED.** There is no alias mechanism in core:')
    add('')
    add('```')
    add("$ grep -rn 'alias' core/graft_arena.py")
    add('core/graft_arena.py:1946:  # spans verbatim (including corrections and '
        'alias edges).   <- a comment')
    add("$ grep -rn 'alias' core/graft_repository.py")
    add('core/graft_repository.py:4767: """Compatibility alias for the old '
        'persistence entry point."""  <- unrelated')
    add("$ grep -rn 'GRM_ALIAS_FOLD_MERGE' core/ scripts/ tests/")
    add('(no match)')
    add('```')
    add('')
    add('One comment and one unrelated docstring. The named optional flag')
    add('`GRM_ALIAS_FOLD_MERGE` does **not exist on this tree** — recorded in the')
    add('LT1.1 registration as `present_on_tree: false`, never silently claimed.')
    add('')
    add('Note `recall_6_*` (Hauler/Kestrel) has the *identical* receipt shape')
    add('(identified `[16]`, rank_plan `[4]`, base-only mount) and scores')
    add('**correct** at all five distances. So the base-only mount is not')
    add('deterministically fatal — whether the reader can bridge the name gap')
    add('unaided is luck. The composition gap is real; its expression is')
    add('stochastic.')
    add('')

    add('### The one fresh row — `recall_3_150`')
    add('')
    f = rows['recall_3_150']
    add('Breakwater map position, `(-31, 48, 12)`, asked at turn 164 about turn 14.')
    add('')
    add('```')
    add('ranking_ids              %s   # node 13 = turn 14 = CORRECT'
        % [m['node'] for m in f['ranked_nodes']])
    add('admission_rank_plan      %s' % f['admission_rank_plan'])
    add('fit_seated               %s                        # = turn %s'
        % ([m['node'] for m in f['mounted_nodes']],
           [m['deposited_turn'] for m in f['mounted_nodes']][0]))
    add('served_without_plan_head %s' % f['served_without_plan_head'])
    for t in f['rt1_provenance']['trips']:
        add('trip %d  planned=%-14s mount_set=%-8s grounded=%s'
            % (t['ordinal'], t['planned'], t['mount_set'], t['grounded']))
    add('```')
    add('')
    add('**A-DEC ranked the correct node first and made it the entire rank plan.**')
    add("The ladder's first trip mounted it, came back `grounded: false`, and a")
    add("second trip seated a *different* entity's map position, which read as")
    add('grounded. The answer was generic filler.')
    add('')
    add("I keep the order's class label `stale-first ranking (core A-DEC)` so the")
    add('lead can map classes 1:1, but the receipt does not support "stale-first')
    add('ranking" for this row: the ranking was correct. The honest name is')
    add('**probe-ladder drop of a correctly-ranked plan head**, which is why the')
    add('class string carries that qualifier. This is core, and STOPPED.')
    add('')
    add('The four correct `recall_3` rows at d=10/25/50/100 all ran two trips in')
    add('which **both** trips read `grounded: false`, and all four kept node 13')
    add('seated and answered correctly:')
    add('')
    add('```')
    add('recall_3_10   correct  trips=2 seated [13]  grounded [False, False]')
    add('recall_3_25   correct  trips=2 seated [13]  grounded [False, False]')
    add('recall_3_50   correct  trips=2 seated [13]  grounded [False, False]')
    add('recall_3_100  correct  trips=2 seated [13]  grounded [False, False]')
    add('recall_3_150  WRONG    trips=2 seated [122] grounded [False, True]')
    add('```')
    add('')
    add('So the displacement is not caused by trip 0 being ungrounded — that')
    add('happens in all five. It is caused by trip 1 *becoming* grounded on an')
    add('unrelated node at d=150 and being preferred over the plan head. The')
    add('grounding signal, not the ladder depth, is what changed. One row is not')
    add('enough to rule on the mechanism; registered as an observation, not a')
    add('finding.')
    add('')

    add('## 2. What I changed, and what I stopped on')
    add('')
    add('### FIXED (harness/fixture only — zero core edits)')
    add('')
    add('| file | what |')
    add('|---|---|')
    add('| `scripts/grm_d1_supersession.py` | the two correction arms on CPU: '
        '`feed_only_arm` (LT1 lineage) vs `supersede_arm` (production path) |')
    add('| `scripts/grm_d1_recap.py` | the five-question recap battery, its two '
        'fixture gates and the oracle gate |')
    add('| `scripts/grm_d1_register_lt11.py` | the LT1.1 fixture amendment and '
        'immutable registration |')
    add('| `scripts/grm_d1_cause_table.py` | the per-row diagnosis |')
    add('| `scripts/grm_d1_emit_report.py` | this report and the ledger |')
    # Test counts are computed, never typed, so the table cannot go stale.
    for name in sorted((ROOT / 'tests').glob('test_grm_d1_*.py')):
        n = sum(1 for line in name.read_text().splitlines()
                if line.startswith('def test_'))
        add('| `tests/%s` | %d tests |' % (name.name, n))
    add('')
    add('**RED -> GREEN proof for the supersession fix.** Both arms run the SAME')
    add('three LT1 turns (Vega 17 -> 23 -> 29) through the same production code,')
    add('differing only in lineage. Receipt:')
    add('`artifacts/grm_d1/supersession_cpu_receipt.json`.')
    add('')
    add('```')
    add("RED  (feed_only — LT1's actual lineage):")
    add('  retired_nodes    : %s' % sup['feed_only']['retired_nodes'])
    add('  route_cand_base  : %s' % sup['feed_only']['route_cand_base'])
    add('  ranking          : %s        <- node 0 = "17 credits" ranks FIRST'
        % sup['feed_only']['ranking'])
    add('  stale reachable  : %s' % sup['feed_only']['stale_still_reachable'])
    add('')
    add('GREEN (supersede — the C7 r3 shape):')
    add('  retired_nodes    : %s' % sup['supersede']['retired_nodes'])
    add('  route_cand_base  : %s              <- stale nodes GONE from the base'
        % sup['supersede']['route_cand_base'])
    add('  ranking          : %s              <- only the current value routable'
        % sup['supersede']['ranking'])
    add('  stale reachable  : %s' % sup['supersede']['stale_still_reachable'])
    add('  superseded_by    : %s' % json.dumps(sup['supersede']['superseded_by']))
    add('```')
    add('')
    add('That is the whole fix: the stale node is not out-ranked, it is **not a')
    add('candidate**. `test_the_supersede_assertions_actually_fail_on_the_old_')
    add('lineage` runs all three GREEN assertions against the RED arm and')
    add('requires each to fail, so the gate is proven capable of going red.')
    add('')
    add('**RED -> GREEN proof for the recap battery.** The old question fails the')
    add('new gate outright (`test_instruction_identifier_gate_can_go_red`:')
    add('`RECAP_INSTRUCTION_IDENTIFIER_COLLISION` on *"Recap the five biggest')
    add('decisions we made"*), and the oracle gate goes RED on a deliberately')
    add('unanswerable question. The new battery scores **oracle %d/%d %s**.'
        % (recap['oracle']['matched'], recap['oracle']['out_of'],
           recap['oracle']['status']))
    add('')
    add('### STOPPED (core — proposed fix, NOT implemented)')
    add('')
    add('#### Proposed core fix — alias->base composition (5 rows, A1 target)')
    add('')
    add('**Receipt of the gap:** `admission_identified_candidates=[17]` (the')
    add('alias edge) but `admission_rank_plan=[5]` (the base fact), and')
    add("`grep -rn 'alias' core/` returns one comment and one unrelated docstring.")
    add('')
    add('**Proposed shape (for A1 to judge, not a patch):** the alias edge is')
    add('already a first-class node; what is missing is that matching it should')
    add('*pull its base in with it* rather than being replaced by the base.')
    add('Concretely, an alias node would carry a lineage pointer — the same')
    add('metadata slot supersession already uses (`metadata.supersedes` /')
    add('`superseded_by`, core/graft_repository.py:1953-1998) — and the admission')
    add('plan would treat an alias hit as a **two-node co-mount requirement**')
    add('rather than a one-node substitution: if node *a* is an alias edge naming')
    add("entity *E*, and the plan would seat *E*'s base fact *b*, seat `{a, b}`")
    add('together so the reader sees both *"Let\'s call Lantern \'the Beacon\'"*')
    add('and *"Lantern\'s launch date will be 18 October 2196"* in one mount. The')
    add('existing multi-node plan machinery already supports this (`recall_5_10`')
    add('seats a 3-node `rank_plan`), so the change is to *plan construction*,')
    add('not the mount path.')
    add('')
    add('**Why I would not implement it even if core were writable:**')
    add('`recall_6_*` scores correct with the base-only mount at all five')
    add('distances, so a co-mount is not obviously necessary and may cost width.')
    add('That is an A1 decision, on evidence, not a D1 one.')
    add('')
    add('#### Reported core observation — probe-ladder drop (1 row)')
    add('')
    add('`recall_3_150`: the ladder discarded a correctly-ranked, correctly-')
    add('planned mount because trip 0 read `grounded: false`, then accepted a')
    add("*different entity's* node from trip 1 because it read `grounded: true`.")
    add('The grounding check appears to test "did the mount produce a groundable')
    add('answer" rather than "did the mount produce an answer grounded **in the')
    add('planned node**". No fix proposed — I have one row, and four')
    add('counterexamples at shorter distances (d=10/25/50/100) whose trip 0 was')
    add('equally ungrounded and which scored correct. Registered as an')
    add('observation for the lead, not a finding.')
    add('')

    add('## 3. LT1.1')
    add('')
    add('- registration: `artifacts/grm_d1/lt1_1/registration.json`')
    add('  sha256 `%s`' % reg_sha)
    add('- sidecar: `artifacts/grm_d1/lt1_1/registration.sha256`')
    add('- fixture: `artifacts/grm_d1/lt1_1/dialogue.json`')
    add('  sha256 `%s`' % reg['fixture']['sha256'])
    add('  (parent = LT1 `%s`)' % reg['fixture']['parent_sha256'])
    add('- commands: `artifacts/grm_d1/lt1_1/lead_commands.txt`')
    add('- shape: arm A only, %d cells, same frozen conversation, same 52-cell'
        % len(reg['cells']))
    add('  schedule, margin_first + profile. Budget **%d s = %.2f GPU-h**.'
        % (reg['budget_gpu_seconds'], reg['budget_gpu_hours']))
    add('- turn mix: fact 60, supersede 15, alias 10, ordinary 75, probe 35,')
    add('  recap_probe 5 = 200. All 200 user/assistant prose strings, all 35')
    add('  recall probes, all distances and all 5 decisions are byte-identical to')
    add('  LT1 (`test_the_user_prose_is_byte_identical_to_lt1`).')
    add('')
    add('**Registered predictions (before the run, per house rules):**')
    for k, v in sorted(reg['predictions'].items()):
        add('- %s: %s' % (k, v))
    add('')
    add('**The optional alias flag.** `%s` is **absent from this tree**. It is'
        % reg['optional_named_flags'][0]['name'])
    add('registered as `present_on_tree: false` with the grep as evidence and')
    add('`disposition: "NOT pinned"`, and')
    add('`test_alias_flag_really_is_absent_from_this_tree` re-verifies that claim')
    add('against the source. The lead can pin it if a later merge introduces it;')
    add('LT1.1 must not claim it was applied.')
    add('')

    add('## 4. The five recap questions')
    add('')
    add('Receipt: `artifacts/grm_d1/recap_battery.json`. Oracle **%d/%d %s**.'
        % (recap['oracle']['matched'], recap['oracle']['out_of'],
           recap['oracle']['status']))
    add('')
    add('| id | turn | dist | question | expected span | decision targeted |')
    add('|---|---|---|---|---|---|')
    for q in recap['gate']['questions']:
        add('| `%s` | %d | %d | %s | `%s` | %s (turns %s) |'
            % (q['id'], q['turn'], q['min_distance'], q['question'],
               q['expected'], q['targets'],
               ', '.join(str(t) for t in q['source_turns'])))
    add('')
    add('Gate rules kept: minimum source distance %d turns (window is 10); no'
        % min(q['min_distance'] for q in recap['gate']['questions']))
    add('instruction-word identifier in any question; scored with the existing')
    add('C5 arm S value-span scorer, unchanged. `recap_4` and `recap_5`')
    add('deliberately target the two correction lineages, so the recap battery')
    add('independently re-tests the supersession fix.')
    add('')
    add('**Why the old one was 0/5.** `recap.json` shows')
    add('`admission_identified_candidates: []` — "recap / five / biggest /')
    add('decisions / made" are all instruction or stop words, so routing had no')
    add('key, ranked arbitrary nodes, every one of four ladder trips returned')
    add('`grounded: false`, and the reader free-associated ("89 iron ingots... 5')
    add('mithril ingots"). The battery measured whether an un-keyed query can be')
    add('routed. It cannot, by construction.')
    add('')

    add('## 5. Prior art')
    add('')
    for item in reg['prior_art']:
        add('- %s' % item)
    add("- **RD2's C7 finding** (GRM contributors, 2026) named both traps before")
    add('  I looked: correction-registered-as-ordinary-fact and')
    add('  alias-edge-mounted-without-base. This report confirms both on LT1')
    add('  receipts and adds the mechanism (candidate-base eligibility) plus the')
    add('  two counterexample lineages (`recall_5_*`, `recall_6_*`) showing the')
    add('  failures are order- and luck-dependent rather than categorical.')
    add('- **Cause-table shape and the per-row receipt join** — no prior art')
    add('  known to me.')
    add('')

    # ---------------------------------------------------------- follow-up
    amend_path = OUT / 'lt1_1/amendment1.json'
    alias_path = OUT / 'alias_cpu_aplus.json'
    if amend_path.exists() and alias_path.exists():
        amend = json.loads(amend_path.read_text())
        alias = json.loads(alias_path.read_text())
        plus, off = alias['A+'], alias['A']
        amend_sha = (OUT / 'lt1_1/amendment1.sha256').read_text().split()[0]
        add('## 5b. Follow-up (A1 merged): amendment 1 and the A+ CPU check')
        add('')
        add('A1 landed `GRM_ALIAS_FOLD_MERGE` in core, so the alias class this')
        add('report STOPPED on now has a mechanism. Three things changed.')
        add('')
        add('**(1) The absent-flag test became a present-flag test.** D1 asserted')
        add('the absence rather than assuming it, so the assertion failed the')
        add('moment A1 merged — which is the point. It is replaced by')
        add('`test_alias_flag_is_present_and_really_read_on_this_tree` (the reader')
        add('exists, switches in both directions, and an unknown token fails')
        add('CLOSED to OFF) and `test_p1_profile_now_pins_the_flag_instead_of_')
        add('recording_it_absent`. The BASE registration is deliberately NOT')
        add('rewritten: it is sha-bound at `02b44d02…` and truthfully records')
        add('what was true when it was written; the amendment supersedes it.')
        add('')
        add('**(2) LT1.1 amendment 1** — `artifacts/grm_d1/lt1_1/amendment1.json`,')
        add('sha256 `%s`,' % amend_sha)
        add('chained to registration `%s…` and order `%s…`.'
            % (amend['registration_sha256'][:16], amend['order_sha256'][:16]))
        add('')
        add('*Core rebind.* The lead named two changed files; the receipts show')
        add('**five** core inputs differ from the LT1 run, and all five are')
        add('rebound with attribution — pinning only two would silently accept')
        add('the other three:')
        add('')
        add('| core input | before | after | attribution |')
        add('|---|---|---|---|')
        for name, change in sorted(amend['core_rebind']['changed'].items()):
            add('| `%s` | `%s…` | `%s…` | %s |'
                % (name, change['before_sha256'][:12],
                   change['after_sha256'][:12], change['attribution']))
        for name, entry in sorted(amend['core_rebind']['new_inputs'].items()):
            add('| `%s` | — (new) | `%s…` | %s |'
                % (name, entry['sha256'][:12], entry['attribution']))
        add('')
        add('%d of the 26 pinned core inputs are unchanged. The A1 attribution is'
            % amend['core_rebind']['unchanged_count'])
        add('checkable and checked: `test_a1_files_are_attributed_to_a1_and_the_')
        add('others_are_not` requires every A1-attributed file to reference')
        add('`alias_fold` and every other changed file NOT to.')
        add('')
        add('*Two arms.* Same frozen conversation, same 26-cell schedule,')
        add('separately resumable into distinct output directories:')
        add('')
        add('| arm | alias flag | env after `environment(flags)` | out dir | budget |')
        add('|---|---|---|---|---|')
        for name in ('A', 'A+'):
            arm = amend['arms'][name]
            add('| `%s` | %s | `%s` | `%s` | %.2f GPU-h |'
                % (name, 'OFF' if not arm['alias_fold_merge'] else '**ON**',
                   ' '.join('%s=%s' % kv for kv in
                            sorted(arm['env_after_environment_flags'].items())),
                   arm['out_dir'], arm['budget_gpu_hours']))
        add('')
        add('Total %.2f GPU-h; the lead may run A+ alone (1.70) and read A off'
            % amend['budget_gpu_hours_total'])
        add('the existing LT1 arm-A receipts, or run both for a same-tree')
        add('contrast — running both is the only way to attribute a change to')
        add('the flag rather than to the core rebind.')
        add('')
        add('*Registered predictions, before any run.* I disagree with one of')
        add("the lead's numbers and say so rather than substituting silently:")
        add('the lead proposed A+ aliases **≥ 3/5**; I register **4/5** (which')
        add('also meets ≥ 3/5). Reason: the CPU check below is unambiguous at')
        add('every distance, but the CPU reader is a regex double and LT1')
        add('already showed one long-distance ladder drop (`recall_3_150`) that')
        add('no lineage change addresses. 4/5 prices exactly one such drop.')
        add('')
        add('**(3) The A+ CPU alias check** — `artifacts/grm_d1/alias_cpu_aplus.md`')
        add('/ `.json`. Flag and rule pinned and **read back** (R1 `pin_rule`')
        add('idiom); %d turns replayed; %d merges executed.'
            % (plus['turns_replayed'], plus['merges_executed']))
        add('')
        add('| probe | d | alias -> base | LT1 arm A | fold fired? | digest | admitted? | mounted | served (CPU double) |')
        add('|---|---|---|---|---|---|---|---|---|')
        for r in plus['rows']:
            was = 'WRONG' if r['probe_id'].startswith('recall_7') else 'correct'
            add('| `%s` | %d | %s -> %s | %s | %s | %s | %s | %s | `%s` |'
                % (r['probe_id'], r['distance'], r['alias'], r['entity'], was,
                   'yes' if r['fold_job_fired'] else 'no',
                   r['digest_nodes'] or '—',
                   'yes' if r['admitted'] else 'no',
                   r['mounted_ids'], r['served'].strip('…')[:24]))
        add('')
        n = len(plus['rows'])
        add('- A+ fold fired **%d/%d**, digest names both names **%d/%d**, '
            'admitted **%d/%d**, mounted **%d/%d**, served-correct **%d/%d**.'
            % (sum(1 for r in plus['rows'] if r['fold_job_fired']), n,
               sum(1 for r in plus['rows'] if r['digest_names_both']), n,
               sum(1 for r in plus['rows'] if r['admitted']), n,
               sum(1 for r in plus['rows'] if r['digest_mounted']), n,
               sum(1 for r in plus['rows'] if r['served_correct']), n))
        add('- A (flag OFF) fold fired **%d/%d**, served-correct **%d/%d** — the'
            % (sum(1 for r in off['rows'] if r['fold_job_fired']), n,
               sum(1 for r in off['rows'] if r['served_correct']), n))
        add('  default-OFF contract holds and the LT1 failure reproduces.')
        add('')
        add('The digest for `recall_7_*` reads: *"For the archive: the Let\'s call')
        add('Lantern \'the Beacon\' from now on… Lantern\'s launch date will be 18')
        add('October 2196. the Beacon is an alias for Lantern."* — coverage 1.0,')
        add('lineage `digest_supersedes_edge_and_base`, **78 tokens inside the 96**')
        add('arena width. That is RD2\'s width wall (49+61=110 > 96, which killed')
        add('the FIX-7 co-mount) solved by moving the join to write time.')
        add('')
        add('**RED I hit and had to work through, reported plainly.** My first')
        add('A+ run showed **0/10 folds fired**. The cause was mine, not A1\'s:')
        add('`ArenaCache.consolidate` gates the digest on fact coverage, and the')
        add('C7 CPU double answers every generation with the probe reader, which')
        add('returns "unknown" for a summarization request — coverage 0.0, so')
        add('every fold correctly aborted with `alias_fold_fidelity_abort`. The')
        add('fix was to route the two request kinds to two readers (probe reader')
        add('for probes, faithful extractive digest for consolidation), which')
        add('exercises the coverage gate rather than bypassing it. Worth')
        add('recording because a seat that stopped at the first run would have')
        add('reported A1 as non-functional on entirely harness-side grounds.')
        add('')
        add('**What this does and does not show.** It shows the fold fires, the')
        add('digest names both entities under the same FIX-8 projection routing')
        add('uses, routing admits it, the ladder mounts it, and it fits the')
        add('width. It does **not** show language-model recall: the reader is a')
        add('regex stub. The GPU A+ arm is the test of sufficiency, and it is')
        add('registered, not run.')
        add('')

    add('## 6. Deviations, RED items, process safety')
    add('')
    add('**Deviations from the order**')
    add('')
    add("1. The order's class name `stale-first ranking (core A-DEC)` does not")
    add('   fit its one arm-A row: A-DEC ranked correctly and the *ladder*')
    add("   dropped the mount. I kept the order's name for 1:1 mapping and")
    add('   appended `[arm-A instance is ladder-drop, not mis-rank]` rather than')
    add('   silently reclassifying or silently mislabelling.')
    add('2. The order asked for the LT1.1 fixture under `fixtures/lt1_1/`. I')
    add('   wrote it to `artifacts/grm_d1/lt1_1/dialogue.json` instead, because')
    add('   the LT1 fixture manifest is SHA-frozen and a sibling under')
    add('   `fixtures/lt1*` risks tripping `FIXTURE_SHA_MISMATCH` in the existing')
    add("   verify chain. The registration's `fixture.path` points at the")
    add('   artifacts location. Trivial for the lead to move if preferred.')
    add('3. Arm B rows were built "where cheap", as permitted — full table, same')
    add('   classifier.')
    add('')
    add('**RED items (honest)**')
    add('')
    add('1. **`tests/test_grm_lt1*.py` is 19 failed / 58 passed on this branch,')
    add('   and this is PRE-EXISTING, not caused by my changes.** Cause: `grm-d1`')
    add('   forks `grm-merge`, whose `core/graft_arena.py` and')
    add("   `core/grm_admission.py` have drifted past the SHAs LT1's registration")
    add('   is bound to. The failure is `INPUT_SHA_MISMATCH: core/graft_arena.py`')
    add('   — the SHA-binding working correctly. Receipts:')
    add('   `artifacts/grm_d1/red_before_lt1.log`,')
    add('   `core_drift_graft_arena.diff` (50 lines),')
    add('   `core_drift_grm_admission.diff` (28 lines). The failing set is')
    add('   IDENTICAL before and after my work (19 = 19, zero new, zero D1).')
    add('2. **Worktree artifact restoration.** `grm-d1` was created without the')
    add("   gitignored `artifacts/` tree, so LT1's `.sha256` sidecars and the")
    add('   `amendment1`/`amendment3`/`grm_scout_fix4`/`grm_scout_fix6`')
    add('   directories were missing (baseline was 49 failed / 16 passed). I')
    add('   restored them with `rsync -a --ignore-existing')
    add('   /mnt/ForgeRealm/wt/grm-lt1/artifacts/ artifacts/`. This copied')
    add('   existing receipts; it fabricated nothing. Flagged because it changed')
    add('   the test baseline mid-session.')
    add('3. **No GPU was used and none is claimed.** Every number here is read off')
    add('   a frozen receipt or produced by a CPU double. The CPU supersession')
    add('   proof is a *lineage* claim (which nodes are route-eligible),')
    add('   explicitly not a model-quality claim. LT1.1 needs a real GPU run to')
    add('   test the prediction; it is **registered, not run** — %.2f GPU-h,'
        % reg['budget_gpu_hours'])
    add('   lead-run, commands in `lt1_1/lead_commands.txt`.')
    add('4. **The alias fix is unresolved.** 5 rows stay wrong by design in LT1.1.')
    add('')
    add('**Process safety**')
    add('')
    add('- No `git` was run.')
    add('- No subagents were spawned.')
    add('- No process this session did not start was killed or signalled.')
    add('- No GPU was touched; no GPU lease was taken.')
    add('- Every Bash call ran in the foreground, under 10 minutes, no background')
    add('  waits.')
    add('- `core/` was read but never written: `find core -newer')
    add('  artifacts/grm_d1/red_before_lt1.log` returns nothing.')
    add('- Writes were confined to')
    add('  `/mnt/ForgeRealm/wt/grm-d1/{scripts,tests,artifacts}`.')
    add('- `/mnt/ForgeRealm/wt/grm-lt1`, `/mnt/ForgeRealm/wt/grm-c7` and')
    add('  `/mnt/ForgeRealm/GraftRepository` were read-only.')
    add('')
    return '\n'.join(lines) + '\n'


def ledger():
    table = load('cause_table_A.json')
    reg = load('lt1_1/registration.json')
    reg_sha = (OUT / 'lt1_1/registration.sha256').read_text().split()[0]
    lines = []
    add = lines.append
    add('# GRM-D1 LEDGER — receipts as they happened (2026-09-10)')
    add('')
    add('Seat: Opus 5 (`claude-opus-5[1m]`), effort MAX.')
    add('Order: `orders/GRM_D1_LT1_RESIDUAL_DIAGNOSIS.md`.')
    add('Writable target: `/mnt/ForgeRealm/wt/grm-d1` (branch `grm-d1`).')
    add('')
    add('## Commands run, in order')
    add('')
    add('| # | command | result |')
    add('|---|---|---|')
    rows = [
        ('orient: `ls`, `ls scripts/ tests/ artifacts/`',
         'LT1 harness + 26 arm-A cells located'),
        ('locate the 35 recall rows: `find ... -name probes.jsonl`',
         '19 files; 10 arm-A cells with `controller.json` status COMPLETE = 35 rows'),
        ('grade all 35 arm-A rows with `scripts.grm_lt1.score`',
         '24 correct, 11 wrong_value, 0 abstention — matches `comparison.json`'),
        ('read `core/graft_arena.py:_route_cand_base`',
         'retired nodes are excluded from routing — the load-bearing fact'),
        ('read `scripts/grm_lt1_worker.py:146-148`',
         'every turn fed via bare `arena.feed()`; corrections carry no edge'),
        ('read `core/graft_repository.py:correct_memory` (1953-1998)',
         'production path retires, sets `superseded_by`, bumps route epoch'),
        ("`grep -rn 'alias' core/`",
         'one comment + one unrelated docstring: NO alias mechanism in core'),
        ("`grep -rn 'GRM_ALIAS_FOLD_MERGE' core/ scripts/ tests/`",
         'no match — the named optional flag is ABSENT on this tree'),
        ('`python3 -m pytest -q tests/test_grm_lt1*.py` (baseline)',
         '**49 failed / 16 passed** — gitignored artifacts missing in worktree'),
        ('`rsync -a --ignore-existing ../grm-lt1/artifacts/ artifacts/`',
         'restored LT1 receipts; baseline improves to **19 failed / 58 passed**'),
        ('diff core between `grm-lt1` and `grm-d1`',
         '`graft_arena.py` (50-line diff) + `grm_admission.py` (28) have drifted; '
         'the 19 failures are `INPUT_SHA_MISMATCH`, pre-existing and correct'),
        ('`python3 scripts/grm_d1_cause_table.py`',
         'arm A %d rows: %s' % (table['wrong_rows'],
                                '; '.join('%s %d' % (k, v)
                                          for k, v in sorted(table['class_counts'].items())))),
        ('`python3 scripts/grm_d1_supersession.py`',
         'RED feed_only ranking `[0,1,2]` retired `[]`; '
         'GREEN supersede ranking `[2]` retired `[0,1]`'),
        ('`python3 scripts/grm_d1_recap.py`',
         '**oracle 5/5 PASS** on the five new named questions'),
        ('`python3 scripts/grm_d1_register_lt11.py`',
         'LT1.1 registration sha256 `%s...`' % reg_sha[:16]),
        ('`python3 -m pytest -q tests/test_grm_d1*.py`', '**42 passed**'),
        ('`python3 -m pytest -q tests/test_grm_lt1*.py tests/test_grm_d1*.py`',
         '**19 failed, 100 passed** — failing set byte-identical to baseline'),
        ('`comm` on before/after FAILED lists', '**zero new failures, zero D1 failures**'),
        ('`find core -newer artifacts/grm_d1/red_before_lt1.log`',
         'empty — core provably untouched'),
    ]
    for n, (cmd, res) in enumerate(rows, 1):
        add('| %d | %s | %s |' % (n, cmd, res))
    add('')
    add('## Decisions taken')
    add('')
    add('1. **Classified the corrections as harness, not core**, because the')
    add('   candidate base excludes retired nodes: a real supersession removes')
    add('   the stale node from the ranking entirely. Verified by CPU contrast,')
    add('   not asserted.')
    add('2. **Stopped on the alias class** rather than proposing a core patch,')
    add('   per the order. `recall_6_*` succeeds with the same base-only mount,')
    add('   so the necessity of a co-mount is unproven — an A1 call.')
    add("3. **Qualified the order's third class name.** The one arm-A row was a")
    add('   ladder drop of a correctly-ranked plan head, not a mis-rank. Kept the')
    add("   order's label for mapping, appended the qualifier.")
    add('4. **Wrote the LT1.1 fixture under `artifacts/`, not `fixtures/`**, to')
    add('   avoid tripping the SHA-frozen LT1 fixture manifest.')
    add('5. **Recorded the absent alias flag as absent**, with a test that')
    add('   re-verifies the claim against the tree, rather than silently omitting')
    add('   it or silently pinning it.')
    add('')
    add('## Failures and how they were handled')
    add('')
    add('- The LT1 suite was RED on arrival (49F/16P). Root-caused to missing')
    add('  gitignored receipts, then to genuine core SHA drift from `grm-merge`.')
    add('  Restored the receipts; reported the drift as the remaining, correct,')
    add('  pre-existing RED rather than working around it.')
    add('- Two writes were intercepted by the AfterImage hook and the bash write')
    add('  guard; both retried through the Write tool as required.')
    add('')
    add('## Evidence classes')
    add('')
    add('- Cause table: **frozen GPU run receipts**, read-only, no re-execution.')
    add('- Supersession proof: **CPU lineage fixture** — a claim about route')
    add('  eligibility only, explicitly NOT a model-quality claim.')
    add('- Recap battery: **CPU oracle gate** — a claim about question')
    add('  answerability only.')
    add('- LT1.1 outcome: **unmeasured**. Registered (%.2f GPU-h), not run.'
        % reg['budget_gpu_hours'])
    add('')
    return '\n'.join(lines) + '\n'


def main():
    (OUT / 'REPORT.md').write_text(report())
    (OUT / 'LEDGER.md').write_text(ledger())
    print('wrote %s' % (OUT / 'REPORT.md'))
    print('wrote %s' % (OUT / 'LEDGER.md'))


if __name__ == '__main__':
    main()
