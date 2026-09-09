#!/usr/bin/env python3
"""RD1 amendment 1: bound checkpoint reader replay, foreground only.

Prior art: C7/FIX4 checkpoint recovery, EB1 ephemeral turns, production
ArenaCache._attempt, C2 lease accounting and C7 diagnostic doubles (GRM
contributors, 2026), verified in local source. Reuse those contracts; ours is
fixed-residency contrast orchestration. No prior art known to me for this
exact composition. No new serving, selection or numerical algorithm.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import grm_rd1 as old

OUT = old.OUT / 'amendment_1'
REG = OUT / 'amendment.json'
ARMS = ('A0', 'A2', 'A4')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selected(rows):
    # Prior art: frozen C7 failure census (GRM, 2026). Lead specifies these
    # 30 memory and 16 oracle failure sides before treatment observation.
    return [(r, s) for r in rows for s in old.SIDES
            if r['probe']['answerable'] and r['historical'][s]['score']['abstention_error']]


def register():
    previous = old.verify_registration()
    rows = old.read(old.OUT / 'cohort.json')
    pairs = selected(rows)
    requests = [{'arm': arm, **old.request(r, s, arm)}
                for arm in ARMS for r, s in pairs]
    per_request = previous['total_estimate_seconds'] / previous['counts']['requests']
    batches = []
    for arm in ARMS:
        for cell in previous['cells']:
            qs = [q for q in requests if q['arm'] == arm and q['cell_id'] == cell['id']]
            if qs:
                batches.append({'id': arm + '_' + cell['id'], 'arm': arm,
                                'cell': cell, 'requests': len(qs),
                                'estimate_seconds': len(qs) * per_request,
                                'lease_seconds': 285})
    old.write(OUT / 'requests.json', requests)
    diffs = []
    for arm in ARMS:
        for r, s in pairs:
            a, b = old.request(r, s, 'A0'), old.request(r, s, arm)
            diffs.append({'probe_id': r['probe']['id'], 'side': s, 'arm': arm,
                          'changes': {k: {'before': a[k], 'after': b[k]} for k in a if a[k] != b[k]}})
    old.write(OUT / 'arm_diffs.json', diffs)
    source = Path(previous['source_root'])
    c7reg = source / 'artifacts/grm_c7/registration.json'
    c7 = old.read(c7reg)
    paths = [ROOT / 'orders/GRM_RD1_AMENDMENT_1.md', Path(__file__),
             ROOT / 'tests/test_grm_rd1_replay.py', c7reg,
             OUT / 'requests.json', OUT / 'arm_diffs.json',
             OUT / 'lead_commands.txt']
    # Pin the actual executable tree, not just the wrapper. Local artifacts
    # from the immutable r1 registration retain their original pins.
    paths += [p for directory in ('core', 'scripts') for p in (ROOT/directory).rglob('*.py')]
    native = Path('/mnt/ForgeRealm/GraftRepository/cpp/build/libgrm_runtime.so')
    if native.is_file():
        paths.append(native)
    amount = len(requests) * per_request
    value = {
        'schema': 'grm.rd1.amendment.1', 'previous_sha256': old.sha(old.OUT/'registration.json'),
        'order_sha256': old.sha(ROOT/'orders/GRM_RD1_AMENDMENT_1.md'),
        'status': 'FIT_ESTIMATE' if amount <= 3600 else 'NON_FIT',
        'gpu_cap_seconds': 3600, 'lease_seconds': 285,
        'request_count': len(requests), 'memory_rows_per_arm': 30, 'oracle_rows_per_arm': 16,
        'estimate_per_request_seconds': per_request, 'estimate_seconds': amount,
        'estimate_gpu_hours': amount / 3600, 'overflow_seconds': max(0, amount - 3600),
        'estimator': 'r1 total 11793.813152/520 requests, multiplied by 138. Amortized whole-cell planning proxy, not measured latency; medium and repeated loads/prefix work unmeasured.',
        'arms': {**{a: previous['arms'][a] for a in ARMS},
                 'A1': {'status': 'NOT_RUN', 'reason': 'lead dropped', 'requests': 46,
                        'estimate_seconds': 46 * per_request},
                 'A3': {'status': 'NOT_RUN', 'reason': 'lead dropped', 'requests': 46,
                        'estimate_seconds': 92 * per_request}},
        'dropped_estimator': 'A1 one proxy unit per reduced-cohort request; A3 two for its 64-token budget. Original r1 full-cohort estimates remain immutable.',
        'batches': batches, 'flags': c7['arms']['A']['flags'], 'model': c7['model'],
        'model_frame': c7['model_frame'], 'native_library': str(native),
        'inputs': {str(p): old.sha(p) for p in sorted(set(paths))},
        'quoted_probe_prompt': old.request(pairs[0][0], pairs[0][1], 'A0')['prompt'],
        'predictions': {a: previous['predictions'][a] for a in ARMS},
        'verdict_rule': previous['verdict_rule'],
        'scoring': previous['scoring'],
        'gates': ['registration_and_138_intended_diffs', 'checkpoint_tree_and_boundary',
                  'fake_checkpoint_production_attempt_A0', 'fake_prefix_creates_missing_mount',
                  'fake_arm_fields_and_isolation', 'mount_mismatch_rejected_before_forward',
                  'A0_difference_receipted_and_stops', 'A0_campaign_barrier',
                  'budget_and_incomplete_summary_fail_closed', 'dry_run_no_gpu',
                  'registered_mutations_at_least_80_percent'],
        'mutations': ['erase_mount_guard', 'force_A0_equal', 'force_A4_low', 'erase_prompt_suffix_change', 'skip_prefix_advance'],
        'mutation_threshold': 0.8,
        'replay_rule': 'Copy preceding checkpoint repository, restore state and cold nodes; replay original non-probe turns, forced folds and pressure through C7/EB1. Deferred ephemeral probes leave no durable memory: advance their turn_records without extra probe decodes; require historical deferred/no-split contract. Each selected side gets a clean ephemeral cache and the recorded ordered mounts through unmodified _attempt. No post-cell checkpoint substitution. Pin reconstructed mounted payloads and require treatment hashes equal A0.',
        'count_scope': 'Exactly 138 scored probe _attempt calls. Original non-probe prefix turns and folds can generate tokens; they are reconstruction overhead, charged inside leases/cap and separately counted. No unscored probe inference.',
        'validity': 'All 46 A0 UTF-8 served texts must byte-match r2. Raw differences are written before raising; no tolerance or normalization waiver. Checkpoint-to-prefix reconstruction and omitted transient probe history are validated numerically only by this gate; CPU establishes control flow only.',
        'launch_rule': 'A0 batches then verify-a0 then A2 then A4 then summary. One nonblocking lease per checkpoint/arm batch; no retries. Atomic campaign ownership; orphan reservations charged fully. Budget rail stops before next launch, never trims registered cohort.',
        'timeout_rule': 'Foreground in-process GPU lease SIGALRM exception; no subprocess kill or process signals sent by this harness. Python alarm cannot hard-preempt an uninterruptible native call; any overrun is RED and charged in full.',
        'prior_art': 'GRM C7/FIX4/EB1/C2 and diagnostic model (contributors, 2026), local source verified: checkpoint trees, ephemeral lifecycle, _attempt, scorer, prefix folds/pressure, leases and CPU seams reused. Ours: fixed-residency replay and contrast accounting. No prior art known to me for this exact composition.'}
    old.write(REG, value)
    REG.with_suffix('.sha256').write_text(old.sha(REG) + '  amendment.json\n')
    return value


def verify():
    require(old.sha(REG) == REG.with_suffix('.sha256').read_text().split()[0], 'AMENDMENT_SHA_MISMATCH')
    r = old.read(REG)
    require(r['previous_sha256'] == old.sha(old.OUT/'registration.json'), 'AMENDMENT_CHAIN_MISMATCH')
    old.verify_registration()
    for name, digest in r['inputs'].items():
        require(old.sha(name) == digest, 'AMENDMENT_INPUT_SHA_MISMATCH: ' + name)
    return r


def key(q):
    return q['arm'], q['probe_id'], q['side']


def batch_requests(batch):
    return [q for q in old.read(OUT/'requests.json')
            if q['arm'] == batch['arm'] and q['cell_id'] == batch['cell']['id']]


def payload_fingerprints(repo, ids):
    # Prior art: C7 durable NPZ SHA binding (GRM, 2026); flush only newly
    # reconstructed prefix state in the writable session, never source trees.
    repo.flush_now()
    return {str(i): old.sha(Path(repo.path)/f'nodes/{i:04d}.npz') for i in ids}


def replay_attempt(repo, row, q, destination, binding_sha, baseline=None):
    """One real production attempt, with explicit clean ephemeral geometry."""
    # Prior art: C7 oracle cache isolation and EB1 turn-open (GRM, 2026).
    # Changes are prompt text/reasoning only; _attempt, sink and payloads stay
    # production. No expected answer is supplied to the model.
    a = repo.arena
    require(a.ephemeral, 'REPLAY_REQUIRES_EPHEMERAL')
    wanted = list(q['mounted_ids'])
    historical_ids = row['historical']['memory']['residency']['mounted_ids'] if q['side'] == 'memory' else []
    require(wanted == historical_ids, 'REQUEST_RESIDENCY_MISMATCH')
    require(len(wanted) == len(set(wanted)) and all(0 <= i < len(a.grafts) for i in wanted), 'MISSING_OR_DUPLICATE_MOUNT')
    require(list(a._resolve_revision_mounts(wanted)) == wanted, 'RESOLVED_MOUNTS_DIFFER')
    fingerprints = payload_fingerprints(repo, wanted)
    if baseline is not None:
        require(fingerprints == baseline['mounted_payload_sha256'], 'A0_PAYLOAD_MISMATCH')
    template = a.prompt_template
    a.reset_live_cache()
    a.eb1_begin_turn()
    shifts = [(layer.self_attn, hasattr(layer.self_attn, 'live_shift'),
               getattr(layer.self_attn, 'live_shift', None)) for layer in a.m.layers]
    before = len(a.grafts)
    try:
        for att, _, _ in shifts:
            att.live_shift = a.live_shift
        system = old.PREFIX.replace('Reasoning: low.', 'Reasoning: ' + q['reasoning'] + '.')
        def frame(user, answer):
            text = system + user + old.FINAL
            return text if answer is None else text + answer + '<|end|>'
        a.prompt_template = frame
        wrapped = a._format_step_prompt(q['prompt'])
        require(wrapped == q['wrapped_prompt'], 'WRAPPED_PROMPT_MISMATCH')
        require(list(a.stop_sequences) == q['stops'], 'STOP_SEQUENCE_MISMATCH')
        a._ensure_h(wanted)
        require(all(a.grafts[i].get('h') is not None for i in wanted), 'MOUNT_PAYLOAD_NOT_LOADED')
        token_ids = list(a.encode(wrapped))
        answer, info = a._attempt(q['prompt'], wanted, q['answer_budget'], False,
                                  tuple(q['stops']), defer_memory=True)
        require(list(a.cur_mounts) == wanted, 'ACTUAL_MOUNTS_DIFFER')
        require(len(a.grafts) == before, 'PROBE_DEPOSITED_MEMORY')
        expected = row['historical'][q['side']]['answer']
        equal = str(answer).encode('utf-8') == expected.encode('utf-8')
        receipt = {'schema': 'grm.rd1.replay.row.1', 'amendment_sha256': binding_sha,
                   'arm': q['arm'], 'side': q['side'], 'probe_id': q['probe_id'],
                   'class': row['probe']['class'], 'turn': q['turn'], 'cell_id': q['cell_id'],
                   'checkpoint': q['checkpoint'], 'checkpoint_sha256': q['checkpoint_sha256'],
                   'request': q, 'prompt_token_ids': token_ids, 'mounted_ids': list(a.cur_mounts),
                   'mounted_payload_sha256': fingerprints, 'served_text': str(answer),
                   'raw_attempt_text': str(answer), 'r2_served_text': expected,
                   'r2_byte_equal': equal, 'difference': None if equal else {
                       'reason': 'production fixed-mount raw _attempt text differs from historical served text; no normalization waiver',
                       'actual_utf8_hex': str(answer).encode().hex(), 'expected_utf8_hex': expected.encode().hex()},
                   'served_from': 'production_fixed_recorded_mounts' if q['side'] == 'memory' else 'production_oracle_live_records',
                   'historical_served_from': q['served_from_recorded'],
                   'production_attempt_info': info, 'scores': old.rd_score(answer, row['probe']),
                   'evidence_class': 'checkpoint production attempt; numerical backend identified in worker receipt'}
        old.write(destination, receipt)
        if q['arm'] == 'A0':
            require(equal, 'A0_R2_TEXT_DIFFERENCE')
        return receipt
    finally:
        a.prompt_template = template
        a.reset_live_cache()
        # Deferred keys belong only to this non-deposit probe, never a later turn.
        a._deferred_route_keys = {}
        for att, existed, value in shifts:
            if existed:
                att.live_shift = value
            elif hasattr(att, 'live_shift'):
                delattr(att, 'live_shift')


def make_args(e2e, session, r):
    # Prior art: C2 args_for (GRM, 2026); same literal flags, with model path
    # taken from pinned C7 registration instead of an absent C2 registry.
    f = r['flags']
    return e2e.parse_args(['--mode', 'full', '--session-dir', str(session),
        '--model-dir', r['model']['path'], '--native-lib', r['native_library'],
        '--arena-width', str(f['arena_width']), '--ngen', str(f['ngen']),
        '--max-trips', str(f['max_trips']), '--live-turns', str(f['live_turns']),
        '--max-live', str(f['max_live']), '--topk', str(f['topk']),
        '--turn-pipeline', f['turn_pipeline'], '--restart-after', '999',
        '--skip-gpu-idle-check', '--probe-ladder', '--sup-resolve', '--adm-decisive'])


def production_load(session, r):
    from scripts import grm_e2e_session as e2e
    _, _, repo, info = e2e.load_model_and_repo(make_args(e2e, session, r), session)
    return repo, info


def nonprobe(repo, state, event, turn, session, r):
    from scripts import grm_e2e_session as e2e
    records = {int(k): v for k, v in state['turn_records'].items()}
    e2e.run_turn(repo, event, turn, paths=e2e.stage_paths(session),
                 transcript=state['transcript'], turn_records=records,
                 probe_rows=state['probe_rows'], args=make_args(e2e, session, r), resumed=True)
    state['turn_records'] = {str(k): v for k, v in records.items()}


def post_turn(repo, state, fixture, turn, directory):
    # Prior art: C7 worker's exact forced-fold and pressure schedule (2026).
    from scripts import grm_c7_run as c7
    for fold in fixture['forced_folds']:
        if fold['after_turn'] == turn:
            ids = c7.source_ids(state, fold)
            if not any(repo.arena.grafts[i].get('retired') or repo.arena.grafts[i].get('no_fold') for i in ids):
                repo._fold_once(jobs=[('digest', ids)])
    if turn in fixture['pressure_after_turns']:
        result = c7.pressure(repo, state, turn, directory)
        state['cold_nodes'].extend({'pressure_turn': turn, **n} for n in result['cold'] if not n.get('controlled_return'))


def assert_deferred_history(row):
    # Prior art: EB1 ephemeral frame + C7 deferred probe contract (2026).
    # There is no deposited probe history to refeed. Reject split mutations,
    # persistent frames or a non-deferred historical row instead of guessing.
    info = row['historical']['memory']['route_info']
    require(info.get('frame_ephemeral') is True and
            info.get('_deferred_memory', {}).get('deposited') is False and
            not info.get('split_children') and not info.get('split_parent'),
            'PREFIX_PROBE_NOT_READ_ONLY_EPHEMERAL')


def replay_batch(r, batch, directory, requests, rows, fixture,
                 loader=production_load, advance=nonprobe, after=post_turn, baselines=None):
    # Prior art: C7 worker recovery (2026). Same preceding state, copy and cold
    # restoration; callbacks replace only numerical boundaries for CPU gates.
    from scripts import grm_c7_run as c7
    from scripts.grm_c2_cells import manifest_projection, assert_payloads
    state, manifest, files = old.validate_bound_state(batch['cell'])
    cp = Path(batch['cell']['checkpoint'])
    require(all(q['checkpoint'] == str(cp) and q['checkpoint_sha256'] == old.sha(cp/'checkpoint.json') for q in requests), 'REQUEST_CHECKPOINT_MISMATCH')
    session = directory/'session'
    session.mkdir()
    assert_payloads(cp/'repository')
    shutil.copytree(cp/'repository', session/'repository')
    repo, model_info = loader(session, r)
    results, events = [], []
    try:
        a = repo.arena
        require(manifest_projection(manifest['nodes']) == manifest_projection([repo._node_manifest(g) for g in a.grafts]), 'RESTART_METADATA_CHANGED')
        require(a.width == r['flags']['arena_width'] and a.ephemeral, 'PROFILE_GEOMETRY_MISMATCH')
        require(a.n_sink == r['model_frame'].get('n_sink', a.n_sink), 'SINK_GEOMETRY_MISMATCH')
        require(list(a.encode(r['model_frame']['sink_text'])) == r['model_frame']['sink_token_ids'], 'SINK_TOKEN_MISMATCH')
        c7.restore_cold_state(repo, state, directory, state['next_turn'])
        by_turn = {row['historical']['turn']: row for row in rows}
        last = max(q['turn'] for q in requests)
        for turn in range(state['next_turn'], last + 1):
            event = fixture['turns'][turn - 1]
            if event['kind'] == 'probe':
                row = by_turn[turn]
                assert_deferred_history(row)
                for q in [q for q in requests if q['turn'] == turn]:
                    result = replay_attempt(repo, row, q, directory/(q['probe_id']+'_'+q['side']+'.json'),
                                            old.sha(REG), (baselines or {}).get((q['probe_id'], q['side'])))
                    results.append(result)
                state['turn_records'][str(turn)] = {'kind': 'probe', 'chat_node_id': None}
                # No probe cache or deposit survives the next ephemeral turn.
                a.reset_live_cache()
                events.append({'turn': turn, 'kind': 'deferred_probe', 'scored_attempts': sum(q['turn'] == turn for q in requests)})
            else:
                advance(repo, state, event, turn, session, r)
                events.append({'turn': turn, 'kind': 'original_nonprobe_reconstruction'})
            # The target probe's post-turn work is outside its preceding state.
            if turn < last:
                after(repo, state, fixture, turn, directory)
            state['next_turn'] = turn + 1
        return {'status': 'COMPLETE', 'model_info': model_info, 'rows': len(results),
                'checkpoint_files_verified': files, 'events': events,
                'scored_probe_attempts': len(results), 'amendment_sha256': old.sha(REG)}
    finally:
        repo.close()


def collect(r, campaign, arms):
    wanted = {key(q): q for q in old.read(OUT/'requests.json') if q['arm'] in arms}
    found = {}
    for batch in r['batches']:
        if batch['arm'] not in arms:
            continue
        directory = campaign/batch['id']
        controller = directory/'controller.json'
        if not controller.exists() or old.read(controller)['status'] != 'COMPLETE':
            continue
        for q in batch_requests(batch):
            p = directory/(q['probe_id']+'_'+q['side']+'.json')
            if not p.exists():
                continue
            row = old.read(p)
            require(row['amendment_sha256'] == old.sha(REG) and row['request'] == q, 'RESULT_BINDING_MISMATCH')
            original = next(x for x in old.read(old.OUT/'cohort.json') if x['probe']['id'] == q['probe_id'])
            require(row['r2_served_text'] == original['historical'][q['side']]['answer'], 'HISTORICAL_TEXT_BINDING_MISMATCH')
            require(row['scores'] == old.rd_score(row['served_text'], original['probe']), 'RESULT_SCORE_MISMATCH')
            require(row['mounted_ids'] == q['mounted_ids'] and row['class'] == original['probe']['class'], 'RESULT_RESIDENCY_OR_CLASS_MISMATCH')
            require(key(row) not in found, 'DUPLICATE_RESULT')
            found[key(row)] = row
    return wanted, found


def verify_a0(r, campaign):
    wanted, found = collect(r, campaign, ('A0',))
    require(len(wanted) == len(found) == 46, 'A0_INCOMPLETE')
    require(all(x['r2_byte_equal'] and x['served_text'].encode() == x['r2_served_text'].encode() for x in found.values()), 'A0_R2_TEXT_DIFFERENCE')
    return {(p, s): row for (_, p, s), row in found.items()}


def launch(r, batch, campaign):
    # Prior art: C2/C7 atomic owner + reservations + gpu_lease (GRM, 2026).
    # Ours: one campaign-wide owner and dynamic remaining-cap reservation.
    require(r['status'] == 'FIT_ESTIMATE', 'NON_FIT')
    require(os.environ.get('GRM_RD1_LEAD_GPU') == '1', 'BLOCKED_NO_GPU_IN_SANDBOX')
    campaign.mkdir(parents=True, exist_ok=True)
    owner = campaign/'active'
    fd = os.open(owner, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    try:
        index = r['batches'].index(batch)
        for previous in r['batches'][:index]:
            p = campaign/previous['id']/'controller.json'
            require(p.exists() and old.read(p)['status'] == 'COMPLETE', 'PREVIOUS_BATCH_INCOMPLETE')
        baselines = verify_a0(r, campaign) if batch['arm'] != 'A0' else None
        charged = 0
        for p in campaign.glob('*/reservation.json'):
            c = p.with_name('controller.json')
            charged += old.read(c)['charged_seconds'] if c.exists() else old.read(p)['seconds']
        remaining = r['gpu_cap_seconds'] - charged
        require(remaining >= math.ceil(batch['estimate_seconds']), 'BUDGET_RAIL_NON_FIT_REMAINDER')
        lease = min(285, math.floor(remaining))
        directory = campaign/batch['id']
        directory.mkdir()  # Immutable attempts, including failures; never retry.
        old.write(directory/'reservation.json', {'seconds': lease, 'amendment_sha256': old.sha(REG), 'batch': batch['id']})
        from scripts.grm_c2_cells import environment
        env = environment(r['flags'])
        for k in list(os.environ):
            if k.startswith('GRM_'):
                del os.environ[k]
        os.environ.update(env)
        from scripts.grm_cmc1_gpu_arms import gpu_lease
        started, status, error = None, 'FAILED', None
        try:
            with gpu_lease(lease, 0):
                started = time.monotonic()
                result = replay_batch(r, batch, directory, batch_requests(batch),
                                      old.read(old.OUT/'cohort.json'),
                                      old.read(Path(old.SOURCE)/'artifacts/grm_c7/fixture.json'), baselines=baselines)
                old.write(directory/'worker.json', result)
                require(time.monotonic()-started <= lease, 'LEASE_OVERRUN')
                status = 'COMPLETE'
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        finally:
            elapsed = time.monotonic()-started if started is not None else 0
            old.write(directory/'controller.json', {'status': status, 'error': error,
                'charged_seconds': elapsed if status == 'COMPLETE' else max(lease, elapsed),
                'wall_seconds': elapsed, 'lease_seconds': lease, 'amendment_sha256': old.sha(REG)})
        require(status == 'COMPLETE', error or 'BATCH_FAILED')
        return {'status': status, 'batch': batch['id']}
    finally:
        owner.unlink()


def summary(r, campaign):
    # Prior art: frozen C7 tables and RD1 paired verdict (GRM, 2026). Missing
    # observations remain null; no synthetic zeros or incomplete-arm verdict.
    wanted, found = collect(r, campaign, ARMS)
    try:
        verify_a0(r, campaign)
        valid = True
    except ValueError:
        valid = False
    source = {x['probe']['id']: x for x in old.read(old.OUT/'cohort.json')}
    table = []
    for arm in old.ARMS:
        for cls in old.CLASSES:
            for side in old.SIDES:
                planned = [q for q in wanted.values() if q['arm'] == arm and q['side'] == side and source[q['probe_id']]['probe']['class'] == cls]
                observed = [x for x in found.values() if x['arm'] == arm and x['side'] == side and x['class'] == cls]
                table.append({'arm': arm, 'class': cls, 'side': side, 'planned': len(planned), 'observed': len(observed),
                    'status': 'NOT_RUN' if arm not in ARMS or not observed else ('COMPLETE' if len(observed) == len(planned) else 'INCOMPLETE'),
                    **{name: sum(x['scores']['frozen'][field] for x in observed) if observed else None
                       for name, field in [('exact', 'exact_correct'), ('wrong', 'wrong_value_error'), ('abstain', 'abstention_error')]}})
    verdicts = {}
    for arm in old.ARMS:
        baseline = [x for x in found.values() if x['arm'] == 'A0' and x['side'] == 'memory']
        treatment = [x for x in found.values() if x['arm'] == arm and x['side'] == 'memory']
        complete_arm = len([x for x in found.values() if x['arm'] == arm]) == 46
        delta = sum(x['scores']['frozen']['exact_correct'] for x in treatment)-sum(x['scores']['frozen']['exact_correct'] for x in baseline)
        wrong = sum(x['scores']['frozen']['wrong_value_error'] for x in treatment)
        verdicts[arm] = ('NOT_RUN' if arm not in ARMS else 'NOT_MEASURED' if not valid or not complete_arm or len(treatment) != 30 else
                         'BASELINE_REPRODUCED' if arm == 'A0' else 'MOVES_READER' if delta >= 8 and wrong <= 2 else 'DOES_NOT_MEET_RULE')
    return {'status': 'COMPLETE' if valid and len(found) == 138 else 'NOT_MEASURED',
            'A0_valid': valid, 'observed': len(found), 'planned': 138, 'table': table,
            'verdicts': verdicts, 'verdict_rule': r['verdict_rule'], 'gpu_executed_by_summary': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('register', 'dry-run', 'run-batch', 'verify-a0', 'summary'))
    p.add_argument('--batch')
    p.add_argument('--campaign', type=Path, default=OUT/'gpu_1')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    try:
        r = register() if a.command == 'register' else verify()
        if a.command == 'register':
            result = {k: r[k] for k in ('status', 'request_count', 'estimate_seconds', 'estimate_gpu_hours')}
        elif a.command == 'dry-run':
            result = {'status': 'BLOCKED_NO_GPU_IN_SANDBOX' if r['status'] == 'FIT_ESTIMATE' else 'BLOCKED_NON_FIT',
                      'gpu_executed': False, 'fit': r['status'], 'request_count': r['request_count'],
                      'estimate_seconds': r['estimate_seconds'], 'gpu_cap_seconds': r['gpu_cap_seconds'], 'batches': len(r['batches'])}
        elif a.command == 'verify-a0':
            result = {'status': 'A0_REPRODUCED', 'rows': len(verify_a0(r, a.campaign))}
        elif a.command == 'summary':
            result = summary(r, a.campaign)
        else:
            batch = next(b for b in r['batches'] if b['id'] == a.batch)
            result = launch(r, batch, a.campaign)
        if a.output:
            old.write(a.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        result = {'status': 'RED', 'error': f'{type(exc).__name__}: {exc}', 'model_effect': 'NOT_MEASURED'}
        if a.output:
            old.write(a.output, result)
        print(json.dumps(result, indent=2))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
