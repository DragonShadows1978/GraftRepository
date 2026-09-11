#!/usr/bin/env python3
"""GRM-D1: LT1.1 registration -- arm A only, real supersessions, real recap.

LT1.1 re-runs the SAME frozen 200-turn conversation, the SAME 52-cell shape
(arm A's 26 cells), and the SAME margin_first + profile flags, changing exactly
two things, both in the FIXTURE/HARNESS, never in core:

  1. The 15 correction turns become real supersessions. The turn gains
     `kind='supersede'`, `old_value` and `correction_command`, the C7 r3 shape
     (`wt/grm-c7` scripts/grm_c7_register_r3.py:41-49). The worker then runs
     `repo.apply_memory_command(correction_command)` on those turns -- the
     production path `scripts/grm_e2e_session.py:2590-2597` already uses -- so
     the stale node is retired (`retired=True`, `metadata.active=False`,
     `metadata.superseded_by=[new]`) and leaves `_route_cand_base`. Proof:
     tests/test_grm_d1_supersession_fixture.py and
     artifacts/grm_d1/supersession_cpu_receipt.json.

  2. The single un-keyable recap turn becomes five named, value-span recap
     questions (scripts/grm_d1_recap.py). Proof: the CPU oracle gate scores
     5/5 on the fake session.

Everything else -- the 200 turns of prose, the 35 recall probes, the distances,
the arena geometry, the admission rule, the restart points -- is byte-identical
to LT1. The alias rows are NOT addressed here: they are a core-composition gap
(no alias handling exists in core/graft_arena.py or core/graft_repository.py),
STOPPED per the order, and left for A1.

Prior art:
  * C7 r3 `prepare_fixture` / `fixture_bytes` (GRM contributors, 2026) is the
    exact model for a SHA-chained fixture amendment that rewrites only the
    correction turns and records `parent_fixture_sha256`. Taken: the structure,
    the sort_keys/indent=2 canonical byte form, and the amendment note.
  * The SHA-bound immutable-registration contract is LT1's own
    (scripts/grm_lt1_register.py, GRM contributors 2026), reused.
  * Mine: the LT1-specific correction_command synthesis from
    (entity, attribute, old_value), and the absent-flag recording for
    GRM_ALIAS_FOLD_MERGE.
  * No prior art known to me for this exact composition.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_d1_recap as recap  # noqa: E402

LT1_FIXTURE = ROOT / 'fixtures/lt1/dialogue.json'
LT1_REGISTRATION = ROOT / 'artifacts/grm_lt1/registration.json'
OUT = ROOT / 'artifacts/grm_d1/lt1_1'

# The order names this flag as an OPTIONAL the lead may pin. It does not exist
# on this tree -- `grep -rn GRM_ALIAS_FOLD_MERGE core/ scripts/ tests/` returns
# nothing -- so it is recorded as absent, never silently claimed as applied.
ALIAS_FLAG = 'GRM_ALIAS_FOLD_MERGE'

# Registered budget ceiling for LT1.1, arm A only.
BUDGET_GPU_SECONDS = 6120        # 1.7 GPU-h


def sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, indent=2) + '\n').encode()


def correction_command(turn, old_value):
    """The C7 r3 grammar, filled in from an LT1 correction turn.

    Prior art: scripts/grm_c7_register_r3.py:48-49 (GRM contributors, 2026),
    `correct memory: <old> => <new>` verbatim. The query half names the value
    being replaced, so `core/graft_repository.py:correct_memory` matches the
    stale node by its own text and nothing else.
    """
    return 'correct memory: %s => %s' % (old_value, turn['user'])


def fixture_bytes(source=None):
    """LT1.1's fixture: LT1's, with real supersessions and the recap battery."""
    raw = (source or LT1_FIXTURE).read_bytes()
    fixture = json.loads(raw)

    # Every correction turn becomes a production supersede turn. The prose the
    # user says is UNCHANGED -- only the lineage edge is added.
    last_value = {}
    corrections = 0
    for turn in fixture['turns']:
        key = (turn.get('entity'), turn.get('attribute'))
        if turn.get('kind') == 'correction':
            old = last_value.get(key)
            if old is None:
                raise ValueError('LT11_CORRECTION_WITHOUT_PRIOR_VALUE: turn %d'
                                 % turn['turn'])
            turn['kind'] = 'supersede'
            turn['old_value'] = old
            turn['correction_command'] = correction_command(turn, old)
            corrections += 1
        if turn.get('value') is not None:
            last_value[key] = turn['value']
    if corrections != 15:
        raise ValueError('LT11_CORRECTION_COUNT: %d' % corrections)

    # The single recap turn becomes five named recap probes.
    questions = recap.build_questions(fixture)
    fixture['recap_probes'] = questions
    recap_turns = {q['turn'] for q in questions}
    turns = [t for t in fixture['turns'] if t['turn'] not in recap_turns]
    for q in questions:
        turns.append(dict(turn=q['turn'], kind='recap_probe',
                          user=q['question'], recap_probe_id=q['id']))
    fixture['turns'] = sorted(turns, key=lambda t: t['turn'])
    if [t['turn'] for t in fixture['turns']] != list(range(1, 201)):
        raise ValueError('LT11_TURN_COVERAGE')

    fixture['protocol'] = (
        'Frozen user/assistant dialogue replay; recall, supersede and recap '
        'responses generated. Probe answers not deposited. Corrections are '
        'REAL supersessions applied through repo.apply_memory_command '
        '(production path), not ordinary prose.')
    fixture['amendment'] = (
        'GRM-D1: 15 correction turns become kind="supersede" with old_value '
        'and correction_command (C7 r3 shape); the single un-keyable recap '
        'turn 200 becomes five named value-span recap probes at turns '
        '196-200. All 200 user/assistant prose strings, all 35 recall probes, '
        'all distances and the 5 decisions are byte-identical to LT1.')
    fixture['parent_fixture_sha256'] = sha_bytes(raw)
    return canonical(fixture)


def registration(fixture_digest):
    parent = json.loads(LT1_REGISTRATION.read_text())
    cells = [c for c in parent['cells'] if c['arm'] == 'A']
    if len(cells) != 26:
        raise ValueError('LT11_CELL_COUNT: %d' % len(cells))
    return dict(
        schema='grm.lt1_1.registration.v1',
        status='FIT_ESTIMATE',
        derived_from=dict(
            registration='artifacts/grm_lt1/registration.json',
            registration_sha256=sha_bytes(LT1_REGISTRATION.read_bytes()),
            run='artifacts/grm_lt1/amendment2/run_margin_first',
            diagnosis='artifacts/grm_d1/cause_table_A.json'),
        arms={'A': dict(parent['arms']['A'], admission_rule='margin_first')},
        cells=cells,
        model=parent['model'],
        model_frame=parent['model_frame'],
        fixture=dict(path='artifacts/grm_d1/lt1_1/dialogue.json',
                     sha256=fixture_digest,
                     parent='fixtures/lt1/dialogue.json',
                     parent_sha256=sha_bytes(LT1_FIXTURE.read_bytes())),
        budget_gpu_seconds=BUDGET_GPU_SECONDS,
        budget_gpu_hours=round(BUDGET_GPU_SECONDS / 3600.0, 2),
        admission_rule='margin_first',
        required_env=dict(GRM_ADMISSION_RULE='margin_first',
                          GRM_PROFILE='eb1_c2'),
        optional_named_flags=[dict(
            name=ALIAS_FLAG,
            present_on_tree=False,
            evidence="grep -rn 'GRM_ALIAS_FOLD_MERGE' core/ scripts/ tests/ "
                     'returns no match on branch grm-d1',
            disposition='NOT pinned. The lead may pin it only if a later merge '
                        'introduces it; until then LT1.1 must not claim it was '
                        'applied. Alias rows stay RED by design.')],
        changes=[
            dict(id='LT11-C1', kind='fixture',
                 what='15 correction turns become kind="supersede" with '
                      'old_value and correction_command',
                 why='LT1 fed corrections as ordinary prose (arena.feed), so '
                     'nothing was retired and the stale node outranked and '
                     'outseated the current one in all 5 recall_4_* rows',
                 proof='tests/test_grm_d1_supersession_fixture.py; '
                       'artifacts/grm_d1/supersession_cpu_receipt.json',
                 core_change=False),
            dict(id='LT11-C2', kind='fixture',
                 what='one un-keyable recap turn becomes five named value-span '
                      'recap probes',
                 why='the old question carried no identifier token '
                     '(admission_identified_candidates == []), so every ladder '
                     'trip came back ungrounded and the score was 0/5 by '
                     'construction',
                 proof='tests/test_grm_d1_recap.py; '
                       'artifacts/grm_d1/recap_battery.json',
                 core_change=False),
        ],
        not_addressed=[dict(
            id='LT11-N1',
            cause_class='alias-edge-without-base (core composition)',
            rows=5,
            statement='Core has no alias->base composition. The identifier '
                      'matches the alias edge node; the mount seats only the '
                      'base fact; nothing binds them. This is a CORE gap, '
                      'STOPPED under the GRM-D1 order (core/ is read-only). '
                      'A1 owns it. LT1.1 predicts these 5 rows stay wrong.',
            proposed_fix='see artifacts/grm_d1/REPORT.md "Proposed core fix"')],
        predictions=dict(
            corrections='5/5 correct (was 0/5) -- the supersession fix retires '
                        'the stale nodes so they cannot be ranked',
            aliases='0/5 correct (unchanged) -- core gap, not addressed',
            fresh='6/7 unchanged; recall_3_150 is a ladder-drop, not addressed '
                  'by either change',
            recap='5/5 on the oracle (gated on CPU); memory side unpredicted',
            evidence_class='prediction registered BEFORE the run, per house rules'),
        cpu_gates=[
            'python3 -m pytest -q tests/test_grm_d1_supersession_fixture.py',
            'python3 -m pytest -q tests/test_grm_d1_recap.py',
            'python3 -m pytest -q tests/test_grm_d1_registration.py',
            'python3 scripts/grm_d1_recap.py   # oracle 5/5 PASS',
        ],
        process_safety=dict(
            gpu='single lease via flock --wait; operator has absolute right of '
                'way; no process this run did not start is ever signalled',
            display='no windowed gate; CPU gates only',
            git='lead commits; the seat never runs git'),
        prior_art=[
            'C7 r3 supersede fixture shape: wt/grm-c7 '
            'scripts/grm_c7_register_r3.py:41-49 (GRM contributors, 2026)',
            'Production supersede turn path: scripts/grm_e2e_session.py:2590-2597',
            'correct_memory retirement: core/graft_repository.py:1953-1998',
            'Route eligibility excludes retired: core/graft_arena.py:1450-1454',
            'Value-span scorer: C5 arm S via scripts/grm_lt1.py:score',
            'Recap decomposition: no specific prior art known to me; unverified '
            '-- lead to check (search: aggregate question decomposition, '
            'list-recall vs point-recall evaluation)',
        ],
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    payload = fixture_bytes()
    digest = sha_bytes(payload)
    (OUT / 'dialogue.json').write_bytes(payload)

    reg = registration(digest)
    reg_bytes = canonical(reg)
    (OUT / 'registration.json').write_bytes(reg_bytes)
    reg_sha = sha_bytes(reg_bytes)
    (OUT / 'registration.sha256').write_text(
        '%s  registration.json\n' % reg_sha)

    commands = [
        '# LT1.1 -- arm A only, 26 cells, budget %d s (%.2f GPU-h)'
        % (BUDGET_GPU_SECONDS, BUDGET_GPU_SECONDS / 3600.0),
        '# registration sha256: %s' % reg_sha,
        '# fixture     sha256: %s' % digest,
        '#',
        '# The alias flag %s is ABSENT on this tree and is NOT set.' % ALIAS_FLAG,
        '# Pin it ONLY if a later merge introduces it:',
        '#   export %s=1' % ALIAS_FLAG,
        '#',
        'export GRM_ADMISSION_RULE=margin_first',
        'export GRM_PROFILE=eb1_c2',
        'cd %s' % ROOT,
        'python3 -m pytest -q tests/test_grm_d1_supersession_fixture.py \\',
        '  tests/test_grm_d1_recap.py tests/test_grm_d1_registration.py',
        'python3 scripts/grm_d1_recap.py',
        'flock --wait 7200 /var/lock/grm_gpu.lock \\',
        '  python3 scripts/grm_lt1.py --run --arm A \\',
        '    --registration artifacts/grm_d1/lt1_1/registration.json \\',
        '    --fixture artifacts/grm_d1/lt1_1/dialogue.json',
    ]
    (OUT / 'lead_commands.txt').write_text('\n'.join(commands) + '\n')

    print('registration sha256 %s' % reg_sha)
    print('fixture      sha256 %s' % digest)
    print('cells %d  budget %ds (%.2f GPU-h)'
          % (len(reg['cells']), BUDGET_GPU_SECONDS, BUDGET_GPU_SECONDS / 3600.0))
    print('optional flag %s present=%s'
          % (ALIAS_FLAG, reg['optional_named_flags'][0]['present_on_tree']))


if __name__ == '__main__':
    main()
