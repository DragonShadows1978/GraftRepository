#!/usr/bin/env python3
"""Additive C7 leased cells; CPU-only until --worker under its leased parent.

Prior art: C2 cells, EB1 session, RS3 live ceiling, S4 paging and fold receipts
(GRM contributors, 2026), verified local source. Import serving/lease methods
unchanged. Ours: combined stress instrumentation and resumable turn ranges;
no prior art known to me for this exact composition, no new memory algorithm.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.grm_c7_common import (ROOT, OUT as BASE, FIX, REG, DISTANCES, CLASSES, create,
    read, sha, hashes, verify, score, table, checkpoint, validate_checkpoint, folded_evidence)

# Prior art: C7 immutable receipt directories (GRM contributors, 2026).
# r2 is a fresh campaign namespace; original registration/fixture stay in BASE.
OUT = BASE / 'r2' if os.environ.get('GRM_C7_REVISION') == 'r2' else BASE
# Prior art: C7 r2 namespace isolation (GRM, 2026). FIX-4 preserves old cells.
from scripts import grm_c7_fix4 as fix4
if fix4.enabled():
    OUT = fix4.ATTEMPT
if os.environ.get('GRM_C7_REVISION') == 'r3':
    # Prior art: r2 isolated receipt namespaces (GRM contributors, 2026).
    # Fresh r3 has no old checkpoint bridge and uses its amended fixture.
    OUT = BASE / 'r3'
    FIX = OUT / 'fixture.json'
    REG = OUT / 'registration.json'


def binding(arm):
    # Prior art: C7 A1/A2 receipt/checkpoint bindings (GRM, 2026). Extend
    # the same contract with verified lead amendment and effective core SHA.
    r = verify()
    value = {'registration_sha256': sha(REG), 'fixture_sha256': sha(FIX), 'arm': arm}
    if os.environ.get('GRM_C7_REVISION') == 'r3':
        return dict(value, revision='r3', source_manifest_sha256=r['source_manifest_sha256'])
    if (BASE/'amendment_A1.json').exists():
        value['amendment_sha256'] = sha(BASE/'amendment_A1.json')
    if (BASE/'amendment_A2.json').exists():
        value['amendment_A2_sha256'] = sha(BASE/'amendment_A2.json')
    value['amendment_lead_1_sha256'] = r['amendment_lead_1_sha256']
    value['amended_source_sha256'] = r['amended_source_sha256']
    if 'amendment_r2_sha256' in r:
        value.update(revision='r2', amendment_r2_sha256=r['amendment_r2_sha256'],
                     fold_core_sha256=r['fold_core_sha256'])
    if 'amendment_fix4_sha256' in r:
        value['amendment_fix4_sha256'] = r['amendment_fix4_sha256']
    return value


def emit(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(row, sort_keys=True, default=str)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()] if Path(path).exists() else []


def dry_run():
    r = verify()
    if os.environ.get('GRM_C7_REVISION') == 'r3':
        return {'status':'NOT_RUN_NO_GPU_ORDER', 'gpu_executed':False,
                'binding':binding('A'), 'receipt_directory':str(OUT),
                'cells':r['cells'], 'arms':r['arms'],
                'budget_seconds_arm_A':r['budget_seconds_arm_A'],
                'prediction':r['prediction'], 'denominator_note':r['denominator_note']}
    return {'status': 'BLOCKED_NO_GPU_IN_SANDBOX', 'gpu_executed': False,
            'registration_sha256': sha(REG), 'fixture_manifest_sha256': sha(BASE/'fixture_manifest.json'),
            'amendment_sha256': r.get('amendment_sha256'),
            'amendment_A2_sha256': r.get('amendment_A2_sha256'),
            'amendment_lead_1_sha256': r['amendment_lead_1_sha256'],
            'amended_source_sha256': r['amended_source_sha256'],
            'amendment_r2_sha256': r.get('amendment_r2_sha256'),
            'fold_core_sha256': r.get('fold_core_sha256'),
            'receipt_directory': str(OUT),
            'arms': r['arms'], 'cells': r['cells'], 'timing': read(BASE/'timing.json'),
            'B_within_combined_7200_status': r['B_within_combined_7200_status'],
            'model_quality': 'NOT_RUN', 'paging_folding_restart_quality': 'NOT_RUN'}


def reserve_check(r, cell, completed, reserved):
    # Prior art: C2 pessimistic reservation accounting (GRM, 2026). Charge
    # orphaned reservations in full. No automatic retry after controller loss.
    if r['arms'][cell['arm']]['status'] != 'FIT_ESTIMATE':
        raise ValueError('NON_FIT: registered arm never launched/retried')
    if completed + reserved + cell['worker_seconds'] > r.get('budget_seconds_arm_A', 7200):
        raise ValueError('BUDGET_RAIL: next cell cannot be reserved')


def source_ids(state, probe):
    ids = []
    for turn in probe['source_turns']:
        rec = state['turn_records'][str(turn)]
        idx = rec.get('memory_node_id', rec.get('chat_node_id'))
        if idx is None:
            raise ValueError(f'MISSING_SOURCE_NODE: turn {turn}')
        ids.append(int(idx))
    return ids


def seats(arena, info=None):
    mounts = [int(i) for i in arena.cur_mounts]
    if len(set(mounts)) != len(mounts):
        raise ValueError('DUPLICATE_MOUNT')
    rec = set((info or {}).get('recency_mounted_ids',
              getattr(arena, '_eb1_recency_mounted_ids', []))) & set(mounts)
    return {'mounted_ids': mounts,
            'summed_token_seats': sum(int(arena.grafts[i]['ntok']) for i in mounts),
            'actual_recency_token_seats': sum(int(arena.grafts[i]['ntok']) for i in rec),
            'arena_cur_mount_n': int(arena.cur_mount_n),
            'width': int(arena.width), 'live_token_rows': sum(int(n) for _, n in arena.live_segs)}


def effective_question(question):
    # Prior art: RD1 A2 plain-prompt contrast and production e2e template
    # (GRM contributors, 2026). Reuse its exact suffix removal at execution;
    # preserve all historical fixture turns and answer identifiers.
    if os.environ.get('GRM_C7_REVISION') == 'r3':
        for suffix in ('; if unspecified, reply UNKNOWN.', '; if unspecified, reply unknown.'):
            if question.endswith(suffix):
                return question[:-len(suffix)] + '.'
        return question
    # Prior art: C7 lead-2 diagnosis (GRM contributors, 2026), confirmed
    # counterfactual. Lowercase ONLY the frozen fallback instruction suffix;
    # preserve entity identifiers and immutable fixture bytes. Core admission
    # is unchanged. This is a harness repair, not a new selection algorithm.
    suffix = 'if unspecified, reply UNKNOWN.'
    if question.endswith(suffix):
        return question[:-len(suffix)] + 'if unspecified, reply unknown.'
    return question


def probe_row(probe, turn, memory, upper):
    # Prior art: C7 probe JSON receipts (GRM contributors, 2026). Add the
    # actual and registered query bytes plus expected answer for audit.
    return {'probe_id': probe['id'], 'turn': turn, 'class': probe['class'],
            'distance': probe['distance'], 'question': effective_question(probe['question']),
            'registered_question': probe['question'], 'expected': probe['expected'],
            'memory': memory, 'oracle': upper}


def oracle(arena, probe, ngen):
    # Prior art: RS3 C5 / RS4 live-band ceiling (GRM, 2026). Taken: exact
    # source in live input, no mounts, production _attempt. C7 includes ALL
    # relevant revisions in chronological order; no expected-answer hints.
    prompt = 'Exact source records, in chronological order:\n' + '\n'.join(
        f'[turn {t}] {text}' for t, text in zip(probe['source_turns'], probe['oracle_source_texts']))
    question = effective_question(probe['question'])
    prompt += '\nUse only those records.\n' + question
    saved = (arena.caches, arena.pos, list(arena.live_segs), list(arena.cur_mounts), arena.cur_mount_n)
    # Prior art: ArenaCache._capture_geometry and GptOssAttentionTC
    # (GRM contributors, 2026): live_shift is optional, read with getattr and
    # set dynamically. Reuse that contract; our sentinel also restores ABSENCE
    # on a fresh real attention object. No new positioning algorithm.
    missing_shift = object()
    shifts = [getattr(layer.self_attn, 'live_shift', missing_shift)
              for layer in arena.m.layers]
    arena.reset_live_cache()
    try:
        # Prior art: EB1 _probe_ladder_chat layer-position setup (GRM, 2026).
        # Reuse its live_shift contract; restore caller state even on failure.
        for layer in arena.m.layers:
            layer.self_attn.live_shift = arena.live_shift
        wrapped = arena._format_step_prompt(prompt)
        prompt_ids = list(arena.encode(wrapped))
        answer, info = arena._attempt(prompt, [], ngen, False, arena.stop_sequences or (), defer_memory=True)
        if arena.cur_mounts:
            raise ValueError('ORACLE_HAS_MOUNTS')
        return {'answer': str(answer), 'source_texts': probe['oracle_source_texts'],
                'question': question, 'live_prompt': prompt, 'wrapped_prompt': wrapped,
                'prompt_token_ids': prompt_ids, 'layer_live_shift': arena.live_shift,
                'mounted_ids': [], 'score': score(answer, probe)}
    finally:
        arena.reset_live_cache()
        arena.caches, arena.pos, arena.live_segs, arena.cur_mounts, arena.cur_mount_n = saved
        for layer, shift in zip(arena.m.layers, shifts):
            if shift is missing_shift:
                delattr(layer.self_attn, 'live_shift')
            else:
                layer.self_attn.live_shift = shift


def install_observers(repo, directory, context, state=None):
    arena = repo.arena
    original_consolidate = arena.consolidate
    original_loader = repo._load_node
    original_attempt = arena._attempt

    def traced_consolidate(idxs, *args, **kwargs):
        # Prior art: S4 fold_history + MIN_FOLD_KEEP (GRM, 2026). Capture
        # candidates at decode, before product receipts truncate at 1200 chars.
        fid = f'{context["turn"]:03d}-{uuid.uuid4().hex}'
        folder = directory/'folds'/fid
        folder.mkdir(parents=True)
        repo.flush_now()
        sources = [{'id': int(i), 'text': arena.grafts[i]['text'],
                    'kind': arena.grafts[i].get('kind'), 'ntok': arena.grafts[i]['ntok']}
                   for i in idxs]
        for i in idxs:
            src = Path(repo.path)/f'nodes/{int(i):04d}.npz'
            if not src.exists():
                raise ValueError(f'FOLD_SOURCE_NOT_DURABLE: {i}')
            shutil.copy2(src, folder/src.name)
        create(folder/'start.json', {'turn': context['turn'], 'sources': sources,
                                     'prompts': arena._consolidation_prompts(
                                         any(s['kind'] != 'turn' for s in sources),
                                         [s['text'] for s in sources]),
                                     'coverage_threshold': 0.70, 'binding': binding(context['arm'])})
        decode = arena.decode
        def decoded(ids, *a, **kw):
            text = decode(ids, *a, **kw)
            emit(folder/'full_candidates.jsonl', {'text': str(text)})
            return text
        arena.decode = decoded
        error, result = None, None
        # Reset stale receipts so a pre-generation exception cannot inherit an
        # earlier fold's successful coverage.
        arena.last_consolidation_result = {}
        arena.last_consolidation_attempts = []
        try:
            result = original_consolidate(idxs, *args, **kwargs)
            return result
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            arena.decode = decode
            create(folder/'result.json', {'turn': context['turn'], 'error': error,
                'accepted': bool(result and result[0] is not None),
                'result': dict(arena.last_consolidation_result),
                'attempts': list(arena.last_consolidation_attempts),
                'digest_text': result[1] if result else None,
                'source_payload_sha256': {p.name: sha(p) for p in folder.glob('*.npz')}})

    def traced_loader(i):
        origin = 'nvme' if arena.grafts[i].get('host_payload') is None else 'ram'
        result = original_loader(i)
        emit(directory/'page_ins.jsonl', {'turn': context['turn'], 'node': int(i),
              'source': origin, 'success': result is not None, 'phase': context['phase']})
        if result is not None and state is not None:
            state['cold_nodes'] = [n for n in state['cold_nodes'] if n['node'] != int(i)]
        return result

    def traced_attempt(*args, **kwargs):
        answer, info = original_attempt(*args, **kwargs)
        if context['phase'] != 'oracle':
            emit(directory/'residency.jsonl', {'turn': context['turn'],
                'phase': context['phase'], **seats(arena, info)})
        return answer, info

    arena.consolidate = traced_consolidate
    repo._load_node = traced_loader
    arena.node_loader = traced_loader
    arena._attempt = traced_attempt


def restore_cold_state(repo, state, directory, turn):
    # Prior art: C2 restart metadata + S4 NVMe loader (GRM, 2026), and
    # Mohan et al., ARIES (1992), recovery concept only. C7 A1 restores its
    # explicit fault-injection state after ordinary repository construction;
    # no payload recapture, read, preload or changed eviction algorithm.
    restored = []
    for n in state['cold_nodes']:
        i = int(n['node'])
        path = Path(repo.path)/f'nodes/{i:04d}.npz'
        if not path.is_file() or sha(path) != n['payload_sha256']:
            raise ValueError(f'COLD_RESUME_PAYLOAD_CHANGED: {i}')
        g = repo.arena.grafts[i]
        g['h'] = None
        g['host_payload'] = None
        repo._native_evict_device_copy(i)
        restored.append(i)
    emit(directory/'cold_resume.jsonl', {'before_turn': turn, 'nodes': restored,
        'method': 'restore persisted cold state; not a fresh eviction witness'})


def pressure(repo, state, turn, directory, context=None):
    # Prior art: production LRU _page/_load_node (GRM, 2026), reused unchanged.
    # C7 fault injection drops old RAM host copies AFTER a durable flush. It is
    # an explicit workload intervention in both arms, not a shipped default.
    # External antecedent: Denning, The Working Set Model for Program Behavior
    # (1968), DOI 10.1145/363095.363141. Borrow bounded residency/page return
    # reasoning, not a new LRU policy or a claim that working sets equal LRU.
    arena = repo.arena
    # A2: forced roundtrip is a controlled pager exercise, not evidence of
    # successful retrieval. Denning (1968) paging/working-set antecedent; C7
    # uses production _ensure_h/_page and a registered old turn6 payload.
    target = state['turn_records']['6'].get('memory_node_id',
                                          state['turn_records']['6'].get('chat_node_id'))
    if target is None or turn-6 < 30:
        raise ValueError('PRESSURE_TARGET_MISSING_OR_NOT_OLD')
    target = int(target)
    arena._ensure_h([target])
    if arena.grafts[target].get('h') is None:
        raise ValueError('PRESSURE_TARGET_NOT_HOT')
    old_ids = sorted({int(r.get('memory_node_id', r.get('chat_node_id')))
                      for t, r in state['turn_records'].items()
                      if int(t) <= turn-30 and r.get('memory_node_id', r.get('chat_node_id')) is not None})
    # Include derived nodes only when all their source records are old.
    old_set = set(old_ids)
    for i, g in enumerate(arena.grafts):
        src = set(g.get('sources') or [])
        if src and src <= old_set:
            old_set.add(i)
    # The designated target was deliberately warmed by the registered fault
    # protocol. Other hot candidates come from ordinary serving.
    hot = {i for i in old_set if arena.grafts[i].get('h') is not None}
    repo.flush_now()
    saved = repo.vram_budget
    try:
        repo.vram_budget = 0
        freed = repo._page()
    finally:
        repo.vram_budget = saved
    cold = []
    for i in sorted(old_set):
        g = arena.grafts[i]
        path = Path(repo.path)/f'nodes/{i:04d}.npz'
        if not path.exists() or g.get('payload_pending'):
            raise ValueError(f'PRESSURE_MISSING_DURABLE_PAYLOAD: {i}')
        if g.get('h') is None:
            g['host_payload'] = None
            cold.append({'node': i, 'was_hot': i in hot, 'payload_sha256': sha(path)})
    if target not in {x['node'] for x in cold} or arena.grafts[target].get('host_payload') is not None:
        raise ValueError('PRESSURE_TARGET_NOT_COLD')
    previous_phase = context['phase'] if context is not None else None
    if context is not None:
        context['phase'] = 'pressure_return'
    try:
        arena._ensure_h([target])
    finally:
        if context is not None:
            context['phase'] = previous_phase
    if arena.grafts[target].get('h') is None:
        raise ValueError('PRESSURE_TARGET_DID_NOT_RETURN')
    for n in cold:
        n['controlled_return'] = n['node'] == target
    row = {'turn': turn, 'old_hot_before': sorted(hot), 'evicted_total': freed, 'cold': cold,
           'controlled_target': target, 'controlled_source_turn': 6,
           'controlled_return_is_retrieval_success': False}
    emit(directory/'pressure.jsonl', row)
    return row


def run_sentinels(repo, e2e, fixture, r, directory, context, phase):
    context['phase'] = phase
    rows = []
    for pid in r['restart_sentinels']:
        p = next(p for p in fixture['probes'] if p['id'] == pid)
        answer, info = e2e._probe_ladder_chat(repo, effective_question(p['question']), topk=3, ngen=32,
                                            max_trips=1, defer_memory=True)
        rows.append({'probe_id': pid, 'answer': str(answer), 'score': score(answer, p),
                     'seats': seats(repo.arena, info)})
    emit(directory/'restart.jsonl', {'after_turn': context['turn'], 'phase': phase,
         'process_id': context['process_id'], 'pid': os.getpid(), 'rows': rows})
    return rows


def worker(cell):
    if os.environ.get('GRM_C7_LEASE_PARENT') != str(os.getppid()):
        raise ValueError('WORKER_REQUIRES_FOREGROUND_LEASE_PARENT')
    r = verify()
    if r['arms'][cell['arm']]['status'] != 'FIT_ESTIMATE':
        raise ValueError('NON_FIT')
    from scripts import grm_e2e_session as e2e
    from scripts.grm_c2_cells import args_for, strict_capture_grade, manifest_projection, sources
    from scripts.grm_c2_cells import assert_payloads
    flags = r['arms'][cell['arm']]['flags']
    directory = OUT/'cells'/cell['id']
    session = directory/'session'
    session.mkdir()
    f = read(FIX)
    state = {'next_turn': 1, 'turn_records': {}, 'transcript': [], 'probe_rows': [],
             'process_id': None, 'cold_nodes': [], 'fold_failures': []}
    prior = None
    if cell['depends']:
        prior_dir = fix4.cell_directory(OUT, cell['depends'])
        prior = read(prior_dir/'worker.json')
        cp = prior_dir/'checkpoint'
        state = fix4.resume_state(cp, cell['start'], binding(cell['arm']), r)
        assert_payloads(cp/'repository')
        shutil.copytree(cp/'repository', session/'repository')
    old_process = state['process_id']
    process_id = str(uuid.uuid4())
    if old_process == process_id:
        raise ValueError('PROCESS_ID_REUSED')
    state['process_id'] = process_id
    context = {'turn': cell['start']-1, 'phase': 'load', 'process_id': process_id, 'arm': cell['arm']}
    model, tokenizer, repo, model_info = e2e.load_model_and_repo(args_for(e2e, session, flags), session)
    rows, restart_ok = [], None
    try:
        a = repo.arena
        if a.width != flags['arena_width'] or a.n_sink != 19 or not a.ephemeral:
            raise ValueError('PROFILE_GEOMETRY_MISMATCH')
        if list(a.encode(r['model_frame']['sink_text'])) != r['model_frame']['sink_token_ids']:
            raise ValueError('EB1_SINK_TOKEN_MISMATCH')
        if prior:
            expected = read(fix4.cell_directory(OUT, cell['depends'])/'checkpoint/repository/manifest.json')['nodes']
            loaded = [repo._node_manifest(g) for g in a.grafts]
            if manifest_projection(expected) != manifest_projection(loaded):
                raise ValueError('RESTART_METADATA_CHANGED')
        restore_cold_state(repo, state, directory, cell['start'])
        install_observers(repo, directory, context, state)
        if cell['start'] in (101, 201):
            after = run_sentinels(repo, e2e, f, r, directory, context, 'restart_after')
            before = prior['restart_before']
            keys = ('exact_correct','exact_answer_error','wrong_value_error','abstention_error','unsupported_answer_error')
            restart_ok = len(after) == len(before) == 4 and all(
                x['probe_id'] == y['probe_id'] and all(x['score'][k] == y['score'][k] for k in keys)
                for x, y in zip(before, after))
        for turn in range(cell['start'], cell['stop']+1):
            context.update(turn=turn, phase='session')
            event = f['turns'][turn-1]
            if event['kind'] == 'probe':
                p = next(p for p in f['probes'] if p['id'] == event['probe_id'])
                # Use the existing production ladder without depositing Q&A:
                # repeated probes must not shorten the registered source age.
                answer, info = e2e._probe_ladder_chat(repo, effective_question(event['user']), topk=3,
                    ngen=32, max_trips=1, defer_memory=True)
                obs = seats(a, info)
                memory = {'answer': str(answer), 'score': score(answer, p),
                          'residency': obs, 'route_info': info,
                          'folded_path': folded_evidence(a.grafts, obs['mounted_ids'], source_ids(state, p))
                          if p['requires_folded'] else None}
                context['phase'] = 'oracle'
                upper = oracle(a, p, 32)
                row = probe_row(p, turn, memory, upper)
                emit(directory/'probes.jsonl', row)
                rows.append(row)
                state['turn_records'][str(turn)] = {'kind': 'probe', 'chat_node_id': None}
            else:
                records = {int(k): v for k, v in state['turn_records'].items()}
                e2e.run_turn(repo, event, turn, paths=e2e.stage_paths(session),
                    transcript=state['transcript'], turn_records=records,
                    probe_rows=state['probe_rows'], args=args_for(e2e, session, flags),
                    resumed=cell['start'] > 1)
                state['turn_records'] = {str(k): v for k, v in records.items()}
            context['phase'] = 'post_turn'
            emit(directory/'residency.jsonl', {'turn': turn, 'phase': 'post_turn', **seats(a)})
            for fold in f['forced_folds']:
                if fold['after_turn'] == turn:
                    ids = source_ids(state, fold)
                    if any(a.grafts[i].get('retired') or a.grafts[i].get('no_fold') for i in ids):
                        emit(directory/'forced_folds.jsonl', {'turn': turn, 'status': 'NOT_FORCED_ALREADY_RETIRED_OR_EXEMPT', 'sources': ids})
                    else:
                        repo._fold_once(jobs=[('digest', ids)])
            if turn in f['pressure_after_turns']:
                row = pressure(repo, state, turn, directory, context)
                state['cold_nodes'].extend({'pressure_turn': turn, **n} for n in row['cold']
                                          if not n.get('controlled_return'))
            state['next_turn'] = turn+1
        repo.flush_now()
        nodes = read(session/'repository/manifest.json')['nodes']
        capture = strict_capture_grade([g for g in nodes if not g.get('retired')],
            'profile' if cell['arm'] == 'A' else 'defaults', a.width, a.n_sink)
        cp = directory/'checkpoint'
        cp.mkdir()
        shutil.copytree(session/'repository', cp/'repository')
        assert_payloads(cp/'repository')
        checkpoint(cp, state, binding(cell['arm']))
        restart_before = None
        if cell['stop'] in (100, 200):
            context['turn'] = cell['stop']
            restart_before = run_sentinels(repo, e2e, f, r, directory, context, 'restart_before')
        return {'cell': cell, 'binding': binding(cell['arm']), 'process_id': process_id,
                'pid': os.getpid(), 'previous_process_id': old_process,
                'new_process': old_process != process_id, 'metadata_retained': True if prior else None,
                'restart_before': restart_before, 'restart_retained_scores': restart_ok,
                'capture': capture, 'model_info': model_info, 'rows': len(rows),
                'executed_inputs': sources(), 'checkpoint_sha256': sha(cp/'checkpoint.json')}
    finally:
        repo.close()


def run_leased(cell):
    r = verify()
    if os.environ.get('GRM_C7_REVISION') == 'r3':
        from scripts.grm_c7_register_r3 import check_ready, preflight
        check_ready()
        preflight()
    arm = cell['arm']
    if fix4.enabled() and OUT == fix4.ATTEMPT:
        if cell['arm'] != 'A' or cell['start'] < 24:
            raise ValueError('FIX4_RESUME_BOUNDARY_ONLY')
        fix4.check_ready()
        OUT.mkdir(parents=True, exist_ok=True)
    # Allocate campaign ownership atomically; concurrent launches of this arm
    # cannot overspend or race dependencies. Stale owners are RED, never cleared.
    owner = OUT/f'{arm}.active'
    fd = os.open(owner, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    directory = OUT/'cells'/cell['id']
    started = None
    try:
        if cell['depends']:
            previous = read(fix4.cell_directory(OUT, cell['depends'])/'controller.json')
            if previous['status'] != 'COMPLETE':
                raise ValueError('DEPENDENCY_NOT_COMPLETE')
        controllers = [p for d in fix4.accounting_directories(OUT)
                       for p in d.glob(f'{arm}-*/controller.json')]
        done = sum(read(p)['charged_seconds'] for p in controllers)
        reserved = sum(read(p)['seconds'] for d in fix4.accounting_directories(OUT)
                       for p in d.glob(f'{arm}-*/reservation.json')
                       if not p.with_name('controller.json').exists())
        reserve_check(r, cell, done, reserved)
        if directory.exists():
            raise ValueError('STARTED_CELL_NEVER_RETRIED')
        directory.mkdir(parents=True)
        create(directory/'reservation.json', {'seconds': 280, 'cell': cell, 'binding': binding(arm)})
        from scripts.grm_cmc1_gpu_arms import gpu_lease
        from scripts.grm_c2_cells import environment
        status, error, charged = 'FAILED', None, 280.0
        try:
            with gpu_lease(285, 240):
                started = time.monotonic()
                env = environment(r['arms'][arm]['flags'])
                # Prior art: C7 lease-parent propagation (GRM, 2026).
                # C2 strips ambient GRM_* switches; restore the campaign
                # selector explicitly so this owned child uses r2 receipts.
                env['GRM_C7_REVISION'] = os.environ.get('GRM_C7_REVISION', 'r1')
                if fix4.enabled():
                    env['GRM_C7_FIX4'] = '1'
                env['GRM_C7_LEASE_PARENT'] = str(os.getpid())
                with (directory/'worker.log').open('x') as log:
                    # Parent waits in foreground. Registered timeout applies
                    # only to the child this controller starts, never by name.
                    result = subprocess.run([sys.executable, __file__, '--worker', cell['id']],
                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=280)
                charged = time.monotonic()-started
                if result.returncode == 0:
                    status = 'COMPLETE'
                else:
                    error = f'worker returncode={result.returncode}'
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
        finally:
            create(directory/'controller.json', {'status': status, 'error': error,
                'charged_seconds': min(280, charged) if status == 'COMPLETE' else 280,
                'cell': cell, 'binding': binding(arm), 'executed_inputs': r['effective_inputs']})
            time.sleep(30)  # foreground cooldown
        return 0 if status == 'COMPLETE' else 1
    finally:
        # This file was created by this controller; shared GPU lock untouched.
        owner.unlink()


def summary(arm):
    r = verify()
    f = read(FIX)
    cells = [c for c in r['cells'] if c['arm'] == arm]
    paths = [fix4.cell_directory(OUT, c['id']) for c in cells]
    rows = [row for p in paths for row in lines(p/'probes.jsonl')]
    strata = table(rows, f['probes'])
    residency = [row for p in paths for row in lines(p/'residency.jsonl')]
    complete = all((p/'controller.json').exists() and read(p/'controller.json')['status']=='COMPLETE' for p in paths)
    complete = complete and len(rows)==len(f['probes']) and {row['turn'] for row in residency if row['phase']=='post_turn'}==set(range(1,301))
    distance_rows = []
    for distance in DISTANCES:
        rs = [row for row in rows if row['distance']==distance]
        means = {s: sum(score(x[s]['answer'], next(p for p in f['probes'] if p['id']==x['probe_id']))['exact_answer_error'] for x in rs)/len(rs) if rs else None for s in ('memory','oracle')}
        distance_rows.append({'distance': distance, 'n': len(rs), **means,
            'passes': len(rs)==12 and means['memory'] <= means['oracle']+0.15})
    bound = bool(residency) and all(x['summed_token_seats'] <= 192+x['actual_recency_token_seats'] for x in residency)
    folded_ids = {p['id'] for p in f['probes'] if p['requires_folded']}
    folded = len([x for x in rows if x['probe_id'] in folded_ids and x['memory']['folded_path']]) == len(folded_ids)
    restarts = [read(p/'worker.json')['restart_retained_scores'] for p in paths if (p/'worker.json').exists() and read(p/'worker.json')['restart_retained_scores'] is not None]
    pressures = [x for p in paths for x in lines(p/'pressure.jsonl')]
    pageins = [x for p in paths for x in lines(p/'page_ins.jsonl')]
    paging = []
    for t in f['pressure_after_turns']:
        ps = [p for p in pressures if p['turn']==t]
        eligible = {n['node'] for p in ps for n in p['cold'] if n['was_hot']}
        natural = bool(eligible) and any(x['node'] in eligible and x['turn']>t and x['success'] and x['source']=='nvme' and x['phase']=='session' for x in pageins)
        controlled = bool(eligible) and any(x['node'] in eligible and x['turn']==t and x['success'] and x['source']=='nvme' and x['phase']=='pressure_return' for x in pageins)
        paging.append({'pressure_turn': t, 'old_hot_nodes': sorted(eligible),
                       'natural_nvme_return': natural, 'controlled_nvme_return': controlled})
    folds = [read(p) for d in paths for p in (d/'folds').glob('*/result.json')]
    fold_rule = bool(folds) and all(not x['accepted'] or x['result'].get('best_cov', -1)>=0.70 for x in folds)
    captures = [read(p/'worker.json')['capture']['valid'] for p in paths if (p/'worker.json').exists()]
    passed = (complete and all(d['passes'] for d in distance_rows) and bound and folded and
              len(restarts)==2 and all(restarts) and all(x['controlled_nvme_return'] for x in paging)
              and fold_rule and len(captures)==len(cells) and all(captures))
    value = {'arm': arm, 'status': 'PASS' if passed else 'FAIL' if complete else 'NOT_RUN' if not rows else 'INCOMPLETE',
            'binding': binding(arm), 'evidence_class': 'E2E only if complete; otherwise incomplete or no GPU evidence',
            'by_distance_and_class': strata, 'by_distance': distance_rows,
            'residency_bounded': bound if residency else None,
            'max_token_seats': max((x['summed_token_seats'] for x in residency), default=None),
            'folded_path_exercised': folded, 'restart_scores': restarts, 'paging': paging,
            'fold_count': len(folds), 'failed_fold_count': sum(not x['accepted'] for x in folds),
            'fold_coverage_rule': fold_rule if folds else None, 'complete': complete}
    if os.environ.get('GRM_C7_REVISION') == 'r3':
        # Prior art: C7 exhaustive strata (GRM, 2026). Keep alias failures
        # in acceptance; expected failure is a prediction, never an exclusion.
        value['prediction'] = r['prediction']
        value['prediction_evaluation'] = 'UNRESOLVED_DENOMINATORS'
        value['denominator_note'] = r['denominator_note']
        value['by_class'] = {c: {side: {
            'expected_n':sum(p['class']==c for p in f['probes']),
            'n':sum(x['class']==c for x in rows),
            'exact_correct':sum(score(x[side]['answer'], next(p for p in f['probes']
                if p['id']==x['probe_id']))['exact_correct'] for x in rows if x['class']==c)}
            for side in ('memory','oracle')} for c in CLASSES}
    return fix4.quarantine_summary(value) if fix4.enabled() else value


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--cell')
    p.add_argument('--worker')
    p.add_argument('--summary', choices=['A','B'])
    p.add_argument('--synthetic-resume', type=Path)
    p.add_argument('--next-turn', type=int)
    args = p.parse_args()
    if args.synthetic_resume:
        state = validate_checkpoint(args.synthetic_resume, args.next_turn, binding('A'))
        print(json.dumps({'state': state, 'reader_pid': os.getpid()}))
        return 0
    if args.dry_run:
        print(json.dumps(dry_run(), indent=2)); return 0
    if args.summary:
        value = summary(args.summary)
        print(json.dumps(value, indent=2)); return 0 if value['status']=='PASS' else 1
    r = verify()
    cid = args.worker or args.cell
    cell = next((c for c in r['cells'] if c['id']==cid), None)
    if cell is None:
        p.error('select --dry-run, --summary ARM or a registered --cell ID')
    if args.worker:
        create(OUT/'cells'/cell['id']/'worker.json', worker(cell)); return 0
    return run_leased(cell)


if __name__ == '__main__':
    raise SystemExit(main())
