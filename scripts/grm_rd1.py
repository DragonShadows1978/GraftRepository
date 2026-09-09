#!/usr/bin/env python3
"""RD1 CPU registration and bound reader requests. Never imports a GPU backend.

Prior art: GRM C7/FIX4 (GRM contributors, 2026), locally verified in
grm_c7_common.py, grm_c7_run.py and grm_c7_fix4.py. Reuse immutable checkpoint
bindings, the exact scorer, chronological oracle and cell-level accounting.
Ours: reader-only request contrasts and conservative NON_FIT audit. No prior
art known to me for this exact composition; no new serving algorithm.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.grm_c7_common import normalize, score

OUT = ROOT / 'artifacts/grm_rd1'
SOURCE = Path('/mnt/ForgeRealm/wt/grm-c7')
ARMS = ('A0', 'A1', 'A2', 'A3', 'A4')
CLASSES = ('fresh', 'alias', 'correction', 'folded')
SIDES = ('memory', 'oracle')
SUFFIX = '; if unspecified, reply unknown.'
PREFIX = ('<|start|>system<|message|>You are ChatGPT. Reasoning: low. '
          'Valid channel: final.<|end|><|start|>user<|message|>')
FINAL = '<|end|><|start|>assistant<|channel|>final<|message|>'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    # Prior art: Python hashlib streaming SHA-256 (Python contributors, 2026),
    # C7 tree fingerprints. Bound memory while checking large K/V payloads.
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')


def checkpoint_path(source, cell_id):
    base = source / 'artifacts/grm_c7/r2'
    parent = base / ('cells' if cell_id == 'A-017-023' else 'fix4_attempt_1/cells')
    return parent / cell_id / 'checkpoint'


def census(source):
    """All 52 continuation probes, including controls; no selection by treatment."""
    # Prior art: amendment-4 continuation census (GRM contributors, 2026).
    # Taken unchanged: quarantine first eight probes. Ours: paired arm requests.
    fixture_path = source / 'artifacts/grm_c7/fixture.json'
    fixture = read(fixture_path)
    probes = {p['id']: p for p in fixture['probes']}
    inputs = {str(fixture_path): sha(fixture_path)}
    rows, cells = [], []
    for path in sorted((source / 'artifacts/grm_c7/r2/fix4_attempt_1/cells').glob('*/probes.jsonl')):
        raw = [json.loads(line) for line in path.read_text().splitlines()]
        if not raw:
            continue
        controller_path = path.parent / 'controller.json'
        controller = read(controller_path)
        worker_path = path.parent / 'worker.json'
        worker = read(worker_path)
        cell = controller['cell']
        cp = checkpoint_path(source, cell['depends'])
        descriptor = read(cp / 'checkpoint.json')
        prior_worker = read(cp.parent / 'worker.json')
        for p in (path, controller_path, worker_path, cp.parent / 'worker.json',
                  cp / 'checkpoint.json', cp / 'state.json', cp / 'repository/manifest.json'):
            inputs[str(p)] = sha(p)
        must = 0
        for line, row in enumerate(raw, 1):
            p = probes[row['probe_id']]
            for side in SIDES:
                must += int(bool(p['answerable'] and row[side]['score']['abstention_error']))
            rows.append({'historical': row, 'probe': p, 'cell_id': cell['id'],
                         'receipt': f'{path}:{line}', 'checkpoint': str(cp),
                         'checkpoint_sha256': sha(cp / 'checkpoint.json'),
                         'checkpoint_next_turn': descriptor['next_turn'],
                         'prefix_turns': list(range(descriptor['next_turn'], row['turn']))})
        wall = float(controller['charged_seconds'])
        # Prior art: C7 pessimistic wall reservation (GRM contributors, 2026).
        # No per-probe timers exist. Entire original cell wall proxies one
        # baseline paired sweep, including reload/prefix/fold work. Scaling all
        # of it deliberately overcounts that overhead: a planning estimate,
        # NOT a measured lower bound. Five arms have 1+1+1+2+1=6 budget units.
        cells.append({'id': cell['id'], 'checkpoint': str(cp),
                      'checkpoint_sha256': sha(cp / 'checkpoint.json'),
                      'worker_checkpoint_sha256': prior_worker['checkpoint_sha256'],
                      'start': cell['start'], 'stop': cell['stop'],
                      'historical_controller_status': controller['status'],
                      'historical_binding': descriptor['binding'],
                      'probe_count': len(raw), 'decode_requests': len(raw) * 2 * 5,
                      'mandatory_failure_sides': must,
                      'r2_wall_seconds': wall,
                      'estimate_seconds': wall * 6 * 1.25,
                      'mandatory_only_proxy_seconds': wall * must / (2 * len(raw)) * 6 * 1.25,
                      'lease_limit_seconds': 285,
                      'fit': wall * 6 * 1.25 <= 285,
                      'model_info': worker['model_info']})
    return fixture, rows, cells, inputs


def request(row, side, arm):
    # Prior art: C7 effective_question and EB1 Harmony shape (GRM, 2026).
    # Ablation is literal suffix deletion with punctuation repair; reasoning
    # changes ONLY the system instruction. Valid channel remains final.
    if side not in SIDES or arm not in ARMS:
        raise ValueError('UNKNOWN_SIDE_OR_ARM')
    h = row['historical']
    prompt = h['question'] if side == 'memory' else h['oracle']['live_prompt']
    if not prompt.endswith(SUFFIX):
        raise ValueError('UNRECOGNIZED_ABSTENTION_SUFFIX')
    if arm in ('A2', 'A4'):
        prompt = prompt[:-len(SUFFIX)] + '.'
    reasoning = 'medium' if arm in ('A1', 'A4') else 'low'
    system = PREFIX.replace('Reasoning: low.', f'Reasoning: {reasoning}.')
    return {'probe_id': h['probe_id'], 'turn': h['turn'], 'cell_id': row['cell_id'],
            'side': side, 'checkpoint': row['checkpoint'],
            'checkpoint_sha256': row['checkpoint_sha256'],
            'prefix_turns': row['prefix_turns'],
            'mounted_ids': h['memory']['residency']['mounted_ids'] if side == 'memory' else [],
            'prompt': prompt, 'reasoning': reasoning, 'wrapped_prompt': system + prompt + FINAL,
            'answer_budget': 64 if arm == 'A3' else 32,
            'defer_memory': True, 'deposit': False,
            'stops': ['<|return|>', '<|end|>', '<|start|>user', '<|start|>assistant'],
            'served_from_recorded': h['memory']['route_info'].get('served_from') if side == 'memory' else None,
            'served_from': 'fixed_recorded_mounts' if side == 'memory' else 'oracle_live_records'}


def rd_score(answer, probe):
    # Prior art: frozen C7 scorer (GRM, 2026). Change ONLY the control exact
    # comparison for normalized whole-string unknown, never substring aliases
    # or case of an answerable value. Preserve the raw frozen score alongside.
    frozen = score(answer, probe)
    adjusted = dict(frozen)
    if not probe['answerable'] and normalize(probe['expected']) == 'UNKNOWN' and normalize(answer).casefold() == 'unknown':
        adjusted.update(exact_correct=True, exact_answer_error=0)
    return {'frozen': frozen, 'control_case_fixed': adjusted}


def register(source):
    fixture, rows, cells, inputs = census(source)
    for relative in ('orders/GRM_RD1_MOUNTED_READ_CONTRAST.md', 'scripts/grm_rd1.py',
                     'scripts/grm_c7_common.py', 'scripts/grm_c7_run.py',
                     'scripts/grm_e2e_session.py', 'scripts/grm_c7_diagnose.py',
                     'core/graft_arena.py', 'core/graft_repository.py'):
        p = ROOT / relative
        inputs[str(p)] = sha(p)
    for p in (Path('/mnt/Shared/HOUSE_RULES.md'), ROOT / 'AGENTS.md'):
        inputs[str(p)] = sha(p)
    diffs = []
    requests = []
    for row in rows:
        for side in SIDES:
            base = request(row, side, 'A0')
            for arm in ARMS:
                value = request(row, side, arm)
                requests.append({'arm': arm, **value})
                diffs.append({'probe_id': value['probe_id'], 'side': side, 'arm': arm,
                              'changes': {key: {'before': base[key], 'after': value[key]}
                                          for key in base if base[key] != value[key]}})
    write(OUT / 'cohort.json', rows)
    write(OUT / 'requests.json', requests)
    write(OUT / 'arm_diffs.json', diffs)
    for name in ('cohort.json', 'requests.json', 'arm_diffs.json'):
        inputs[str(OUT / name)] = sha(OUT / name)
    total = sum(c['estimate_seconds'] for c in cells)
    mandatory = sum(c['mandatory_only_proxy_seconds'] for c in cells)
    registration = {
        'schema': 'grm.rd1.registration.v1', 'source_root': str(source),
        'order': 'orders/GRM_RD1_MOUNTED_READ_CONTRAST.md', 'inputs': inputs,
        'status': 'FIT_ESTIMATE' if total <= 2880 and all(c['fit'] for c in cells) else 'NON_FIT',
        'cells': cells, 'total_estimate_seconds': total, 'gpu_cap_seconds': 2880,
        'mandatory_only_proxy_seconds': mandatory,
        'timing_rule': '1.25 * r2 whole-cell charged wall * (1+1+1+2+1); six 32-token budget units. Whole-cell overhead counted six times; conservative planning proxy, not impossibility proof. Medium latency unmeasured.',
        'batch_rule': 'one lease per original cell checkpoint, all probes, both sides and all five arms; sequential, no per-probe lease',
        'cohort_rule': 'all 52 FIX4 continuation probes x both sides x 5 arms; all 34 answerable and 18 controls. 30 memory failure and 16 oracle failure sides separately flagged by frozen scorer; first eight quarantined probes excluded.',
        'counts': {'probes': len(rows), 'requests': len(requests),
                   'memory_failure_sides': sum(bool(r['probe']['answerable'] and r['historical']['memory']['score']['abstention_error']) for r in rows),
                   'oracle_failure_sides': sum(bool(r['probe']['answerable'] and r['historical']['oracle']['score']['abstention_error']) for r in rows)},
        'quoted_probe_prompt': rows[0]['historical']['question'],
        'removed_bytes': SUFFIX, 'replacement_bytes': '.',
        'arms': {'A0': 'as run: low/32/original prompt', 'A1': 'medium only',
                 'A2': 'remove fallback clause only', 'A3': '64 answer tokens only',
                 'A4': 'medium + remove fallback clause'},
        'scoring': 'Frozen C7; only control whole-string normalized unknown matches expected UNKNOWN case-insensitively. Both scores retained. Answerable scoring unchanged; partial unknown remains abstention, possibly hiding wrong components.',
        'verdict_rule': 'On the fixed 30 memory failure probes, arm exact minus replayed A0 exact >= 8 AND arm wrong_value_error sum <= 2. All 30 pairs required. A0 must reproduce r2 text before model verdict; incomplete or discrepant A0 => NOT_MEASURED.',
        'predictions': {'A0': '0/30 memory exact, 0 wrong; all original text reproduced',
                        'A1': '4-10/30 memory exact, 0-2 wrong; possibly moves',
                        'A2': '6-12/30 memory exact, 1-4 wrong; wrong-value rail may bind',
                        'A3': '0-3/30 memory exact, 0-2 wrong; does not move',
                        'A4': '8-16/30 memory exact, 1-4 wrong; largest gain, may fail wrong-value rail',
                        'oracle': 'A2/A4 recover >=8 of 16; A1 4-10; A3 <=3. Subjective forecasts, not confidence intervals.',
                        'alias_limit': 'base value absent from mounts; no prediction of grounded alias recovery. Repeated four underlying questions limit independence.'},
        'gates': ['input SHA + checkpoint descriptor/tree/state/worker binding parity',
                  '52/30/16 census and frozen score parity',
                  'all 520 intended-field request diffs pinned; stops and mount identity fixed',
                  'A0 fake transport replay 104 served strings (scripted tape, not model reproduction)',
                  'real checkpoint A0 fake numerical replay: BLOCKED if prefix/runtime restore unavailable',
                  'control-only case fix adversarial checks',
                  '--dry-run and launch fail closed on NON_FIT'],
        'rails': ['NON_FIT => never launch, never trim, immutable amendment required for any revised estimate or cohort',
                  'no GPU in sandbox; no core edits; no git; no subagents; no background waits; never kill anything',
                  'no post-probe checkpoint substitution: restore preceding checkpoint then replay prefix events including folds/pressure; isolate identical pre-probe state for all arms',
                  'no production model worker is enabled under a NON_FIT registration'],
        'prior_art': 'Local GRM C7/FIX4/EB1, GRM contributors (2026): checkpoint binding, scorer, oracle, Harmony and conservative accounting reused. Ours: this diagnostic request composition. No prior art known to me for this exact composition. No external literature or novelty claim.'}
    write(OUT / 'registration.json', registration)
    with (OUT / 'registration.sha256').open('x') as stream:
        stream.write(sha(OUT / 'registration.json') + '  registration.json\n')
    return {k: registration[k] for k in ('status', 'counts', 'total_estimate_seconds', 'mandatory_only_proxy_seconds')}


def verify_registration():
    path = OUT / 'registration.json'
    if sha(path) != (OUT / 'registration.sha256').read_text().split()[0]:
        raise ValueError('REGISTRATION_SHA_MISMATCH')
    r = read(path)
    for path, digest in r['inputs'].items():
        if sha(path) != digest:
            raise ValueError(f'INPUT_SHA_MISMATCH: {path}')
    return r


def validate_bound_state(cell):
    # Prior art: C7 validate_checkpoint (GRM contributors, 2026). Same exact
    # tree, state and boundary contract, plus pinned predecessor worker SHA.
    cp = Path(cell['checkpoint'])
    descriptor = read(cp / 'checkpoint.json')
    if sha(cp / 'checkpoint.json') != cell['worker_checkpoint_sha256']:
        raise ValueError('PREDECESSOR_WORKER_SHA_MISMATCH')
    actual = {str(p.relative_to(cp)): sha(p) for p in sorted(cp.rglob('*'))
              if p.is_file() and p != cp / 'checkpoint.json'}
    if actual != descriptor['files'] or descriptor['binding'] != cell['historical_binding']:
        raise ValueError('CHECKPOINT_INTEGRITY_OR_BOUNDARY_MISMATCH')
    state = read(cp / 'state.json')
    if not (state['next_turn'] == descriptor['next_turn'] == cell['start']
            and state['process_id'] == descriptor['process_id']):
        raise ValueError('CHECKPOINT_STATE_MISMATCH')
    if cell['historical_controller_status'] != 'COMPLETE':
        raise ValueError('HISTORICAL_CELL_INCOMPLETE')
    return state, read(cp / 'repository/manifest.json'), len(actual)


class TapeModel:
    """Explicit historical-output fixture; input transport test only.

    Prior art: local C7 scripted model doubles (GRM contributors, 2026).
    This exact-input tape intentionally returns receipt text. It CANNOT
    validate numerical checkpoint restoration or rediscover GPT-OSS answers.
    """
    def __init__(self, expected_request, served_text):
        self.expected = copy.deepcopy(expected_request)
        self.text = served_text

    def replay(self, actual):
        if actual != self.expected:
            raise ValueError('FAKE_INPUT_MISMATCH')
        return self.text


def cpu_gates(output):
    r = verify_registration()
    rows = read(OUT / 'cohort.json')
    gates = {}
    # Prior art: amendment-4 AST literal/source parity (GRM, 2026; Python
    # contributors, 2026). Check the actual serving constants without imports.
    literals = {}
    for node in ast.parse((ROOT / 'scripts/grm_e2e_session.py').read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('SYSTEM_PREFIX', 'ASSISTANT_FINAL', 'HARMONY_STOPS'):
                    literals[target.id] = ast.literal_eval(node.value)
    assert PREFIX == literals['SYSTEM_PREFIX'] and FINAL == literals['ASSISTANT_FINAL']
    assert request(rows[0], 'memory', 'A0')['stops'] == list(literals['HARMONY_STOPS'])
    gates['serving_constant_parity'] = {'status': 'PASS', 'constants': list(literals)}
    assert r['counts'] == {'probes': 52, 'requests': 520, 'memory_failure_sides': 30, 'oracle_failure_sides': 16}
    assert len({x['probe']['id'] for x in rows}) == 52
    gates['cohort'] = {'status': 'PASS', **r['counts']}
    bound, missing, trees = {}, [], []
    for cell in r['cells']:
        state, manifest, count = validate_bound_state(cell)
        bound[cell['id']] = (state, manifest)
        trees.append({'cell_id': cell['id'], 'files_verified': count,
                      'state_file': str(Path(cell['checkpoint']) / 'state.json')})
    gates['checkpoint_hashes'] = {'status': 'PASS', 'cells': trees}
    tape_rows, n, mutants = [], 0, 0
    actual_diffs = []
    allowed = {'A0': set(), 'A1': {'reasoning', 'wrapped_prompt'},
               'A2': {'prompt', 'wrapped_prompt'}, 'A3': {'answer_budget'},
               'A4': {'reasoning', 'prompt', 'wrapped_prompt'}}
    for row in rows:
        h, p = row['historical'], row['probe']
        state, manifest = bound[row['cell_id']]
        for i in h['memory']['residency']['mounted_ids']:
            if i >= len(manifest['nodes']):
                missing.append({'probe_id': p['id'], 'node_id': i,
                                'checkpoint': row['checkpoint'], 'prefix_turns': row['prefix_turns']})
        assert h['question'] == p['question'].replace('if unspecified, reply UNKNOWN.', 'if unspecified, reply unknown.')
        for side in SIDES:
            assert score(h[side]['answer'], p) == h[side]['score']
            baseline = request(row, side, 'A0')
            if side == 'oracle':
                assert baseline['wrapped_prompt'] == h[side]['wrapped_prompt']
            else:
                assert baseline['prompt'] == h[side]['route_info']['_deferred_memory']['user_text']
            tape = TapeModel(baseline, h[side]['answer'])
            served = tape.replay(copy.deepcopy(baseline))
            assert served == h[side]['answer']
            tape_rows.append({'probe_id': p['id'], 'side': side, 'arm': 'A0',
                              'served_text': served, 'scores': rd_score(served, p),
                              'served_from': 'historical_tape_fixture',
                              'historical_served_from': baseline['served_from_recorded'],
                              'evidence_class': 'CPU fixture transport; not model inference'})
            # Perturb independently observable inputs; fake must reject all.
            for key, value in [('mounted_ids', [-999]), ('prompt', 'wrong prompt'),
                               ('checkpoint_sha256', '0' * 64), ('answer_budget', 31)]:
                bad = copy.deepcopy(baseline)
                bad[key] = value
                try:
                    tape.replay(bad)
                except ValueError:
                    mutants += 1
                else:
                    raise AssertionError('TAPE_ACCEPTED_MUTANT')
            for arm in ARMS:
                q = request(row, side, arm)
                changes = {key: {'before': baseline[key], 'after': q[key]}
                           for key in baseline if baseline[key] != q[key]}
                assert set(changes) == allowed[arm]
                actual_diffs.append({'probe_id': p['id'], 'side': side, 'arm': arm, 'changes': changes})
                n += 1
    assert actual_diffs == read(OUT / 'arm_diffs.json')
    gates['arm_diffs'] = {'status': 'PASS', 'requests': n, 'sha256': sha(OUT / 'arm_diffs.json')}
    gates['A0_fake_transport'] = {'status': 'PASS', 'served_strings': len(tape_rows),
                                'input_mutants_rejected': mutants, 'limitation': 'Historical answer tape; no numerical restore exercised.'}
    gates['A0_checkpoint_numerical_replay'] = {
        'status': 'BLOCKED_NON_FIT', 'passed': False,
        'reason': 'No numerical checkpoint replay worker enabled after registered NON_FIT rail. Tape equality is not this gate.',
        'prefix_node_absent_at_checkpoint': missing}
    control = {'answerable': False, 'expected': 'UNKNOWN'}
    for text in ('unknown', 'UNKNOWN', 'Unknown.'):
        assert rd_score(text, control)['control_case_fixed']['exact_correct']
    for text in ('not in memory', 'Reed-611 | unknown', 'unknown extra'):
        assert not rd_score(text, control)['control_case_fixed']['exact_correct']
    for text in ('basalt-811', 'Basalt-811', 'Unknown.', 'Reed-611 | unknown'):
        p = {'answerable': True, 'expected': 'Basalt-811'}
        s = rd_score(text, p)
        assert s['frozen'] == s['control_case_fixed']
    gates['control_case_contract'] = {'status': 'PASS', 'cases': 10}
    gates['status'] = 'PARTIAL_CPU_PASS_REPLAY_BLOCKED'
    gates['gpu_executed'] = False
    gates['registration_sha256'] = sha(OUT / 'registration.json')
    write(output / 'cpu_gates.json', gates)
    write(output / 'A0_tape_rows.json', tape_rows)
    return gates


def dry_run():
    r = verify_registration()
    return {'status': 'BLOCKED_NON_FIT' if r['status'] == 'NON_FIT' else 'BLOCKED_NO_GPU_IN_SANDBOX',
            'registration_sha256': sha(OUT / 'registration.json'),
            'gpu_executed': False, 'model_effect': 'NOT_RUN',
            'cells': len(r['cells']), 'requests': r['counts']['requests'],
            'total_estimate_seconds': r['total_estimate_seconds'],
            'mandatory_only_proxy_seconds': r['mandatory_only_proxy_seconds'],
            'gpu_cap_seconds': r['gpu_cap_seconds'],
            'oversize_cells': [c['id'] for c in r['cells'] if not c['fit']],
            'reason': 'Immutable estimate exceeds bound; no trimming or model worker permitted.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    lane = parser.add_mutually_exclusive_group(required=True)
    lane.add_argument('--register', action='store_true')
    lane.add_argument('--dry-run', action='store_true')
    lane.add_argument('--cpu-gates', action='store_true')
    lane.add_argument('--run', action='store_true')
    parser.add_argument('--source-root', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.register:
        result = register(args.source_root)
    elif args.cpu_gates:
        if args.output is None:
            parser.error('--cpu-gates requires a fresh --output directory')
        result = cpu_gates(args.output)
    else:
        result = dry_run()
        if args.output:
            write(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2 if args.run else 0


if __name__ == '__main__':
    raise SystemExit(main())
