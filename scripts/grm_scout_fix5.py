"""FIX5 bound fold replay; --check is CPU-only, --run is lead-only.

Prior art: GRM C7 start/payload receipts, C2 environment/model loader,
GQA unpack_index/unpack_node, CMC1 foreground flock lease (GRM contributors,
2026), source verified. Reuse those contracts; new source-only fold replay
and prompt/output telemetry. No prior art known to me for this composition.
No source re-harvest, question/probe replay, scorer change or campaign resume.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/grm_scout_fix5'
REG = OUT/'gpu_registration.json'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def create(p, value):
    with Path(p).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, default=str)
        f.write('\n')


def verify():
    if sha(REG) != REG.with_suffix('.sha256').read_text().split()[0]:
        raise ValueError('GPU_REGISTRATION_SHA_MISMATCH')
    r = read(REG)
    for path, expected in r['inputs'].items():
        if sha(ROOT/path) != expected:
            raise ValueError(f'INPUT_SHA_MISMATCH: {path}')
    if len(r['folds']) != 13 or len({x['turn'] for x in r['folds']}) != 13:
        raise ValueError('REQUIRES_13_DISTINCT_FOLDS')
    if r['lease_seconds']*len(r['folds']) > 1080:
        raise ValueError('GPU_BUDGET_EXCEEDED')
    if r['flags'] != read(ROOT/'artifacts/grm_c7/registration.json')['arms']['A']['flags']:
        raise ValueError('FRAME_CHANGED')
    # This check only reads metadata and hashes; no model import/allocation.
    for fold in r['folds']:
        start = read(ROOT/fold['start'])
        baseline = read(ROOT/fold['result'])
        manifest = read(ROOT/fold['manifest'])
        if start['turn'] != fold['turn'] or start['coverage_threshold'] != .70:
            raise ValueError('FOLD_ID_OR_THRESHOLD_CHANGED')
        for src in start['sources']:
            node = manifest['nodes'][src['id']]
            if (node['text'] != src['text'] or node['ntok'] != src['ntok']
                    or node['kind'] != (src['kind'] or 'turn')):
                raise ValueError('CHECKPOINT_SOURCE_MISMATCH')
            payload = ROOT/fold['start']
            payload = payload.parent/f'{src["id"]:04d}.npz'
            if sha(payload) != baseline['source_payload_sha256'][payload.name]:
                raise ValueError('HISTORICAL_PAYLOAD_MISMATCH')
        if baseline['accepted'] or baseline['result']['best_cov'] >= .70:
            raise ValueError('HISTORICAL_BASELINE_NOT_RED')
    return r


def restore_sources(arena, fold):
    """Restore the fold operator's inputs with original K/V and centroids.

    Prior art: GraftRepository.load and GQAArenaCache persistence (2026).
    Same unpackers; subset IDs are remapped in source order solely for this
    isolated operator replay. No routing/live state is used by consolidate.
    """
    import numpy as np
    start = read(ROOT/fold['start'])
    manifest = read(ROOT/fold['manifest'])
    if arena.grafts:
        raise ValueError('REPLAY_ARENA_MUST_BE_EMPTY')
    mapping = []
    with np.load(ROOT/fold['index'], allow_pickle=False) as index:
        for src in start['sources']:
            path = (ROOT/fold['start']).parent/f'{src["id"]:04d}.npz'
            with np.load(path, allow_pickle=False) as z:
                payload = {k:z[k] for k in z.files}
            node = manifest['nodes'][src['id']]
            g = {'text':src['text'], 'ntok':src['ntok'], 'kind':src['kind'] or 'turn',
                 'cent':arena.unpack_index(index, src['id']),
                 'h':arena.unpack_node(payload), 'rare':set(node['rare']),
                 'retired':False, 'sources':[], 'metadata':node.get('metadata', {}),
                 'host_payload':payload, 'durable':False, 'dirty':False}
            if len(arena.encode(g['text'])) != g['ntok']:
                raise ValueError('SOURCE_TOKEN_GEOMETRY_CHANGED')
            for layer in g['h']:
                if layer['k'].shape[2] != g['ntok'] or layer['v'].shape[2] != g['ntok']:
                    raise ValueError('PAYLOAD_SEAT_GEOMETRY_CHANGED')
            mapping.append({'original_id':src['id'], 'replay_id':len(arena.grafts),
                            'payload_sha256':sha(path)})
            arena.grafts.append(g)
    return mapping


class TraceModel:
    """C7 decode receipts extended to exact input/output IDs, no logits edit."""
    def __init__(self, model, arena, folder):
        self.model, self.arena, self.folder = model, arena, folder
        self.traces = []

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, ids, **kwargs):
        generation = kwargs.get('last_token_only', False)
        if generation and kwargs.get('kv_caches') is None:
            prompt_ids = [int(x) for x in ids[0]]
            self.traces.append({'prompt':self.arena.decode(prompt_ids),
                                'prompt_token_ids':prompt_ids, 'output_token_ids':[]})
            create(self.folder/f'prompt_{len(self.traces)-1}.json', self.traces[-1])
        result = self.model(ids, **kwargs)
        if generation:
            token = int(result[0].numpy()[0, -1].argmax())
            self.traces[-1]['output_token_ids'].append(token)
            with (self.folder/'tokens.jsonl').open('a') as f:
                f.write(json.dumps({'attempt':len(self.traces)-1, 'token':token})+'\n')
        return result


def worker(turn):
    if os.environ.get('GRM_FIX5_PARENT') != str(os.getppid()):
        raise ValueError('WORKER_REQUIRES_FOREGROUND_PARENT')
    r = verify()
    fold = next(x for x in r['folds'] if x['turn'] == turn)
    folder = OUT/'gpu'/f'{turn:03d}'
    from scripts.grm_cmc1_gpu_arms import gpu_lease
    # Cooperative in-process deadline raises; no process is killed. A native
    # call that does not return to Python cannot be forcibly bounded without
    # violating the order's never-kill rule; report that limitation explicitly.
    with gpu_lease(r['lease_seconds'], 0):
        started = time.monotonic()
        signal.alarm(r['worker_seconds'])
        repo = model = tracer = None
        error = None
        result = None
        receipt = {'turn':turn, 'registration_sha256':sha(REG), 'status':'FAILED',
                   'accepted':False, 'digest_text':None, 'baseline':read(ROOT/fold['result'])}
        try:
            from scripts import grm_e2e_session as e2e
            from scripts.grm_c2_cells import args_for
            session = folder/'session'
            session.mkdir()
            args = args_for(e2e, session, r['flags'])
            if str(args.model_dir) != r['model']['path']:
                raise ValueError('MODEL_PATH_MISMATCH')
            model, tokenizer, repo, info = e2e.load_model_and_repo(args, session)
            a = repo.arena
            if a.width != 96 or a.n_sink != 19 or not a.ephemeral:
                raise ValueError('FRAME_GEOMETRY_MISMATCH')
            if list(a.encode(r['model_frame']['sink_text'])) != r['model_frame']['sink_token_ids']:
                raise ValueError('SINK_TOKEN_IDS_CHANGED')
            receipt['source_mapping'] = restore_sources(a, fold)
            receipt['model_info'] = info
            need = a._fact_set([g['text'] for g in a.grafts])
            receipt['need'] = sorted(need)
            receipt['budget'] = max(120, 24*len(need))
            assert a.MIN_FOLD_KEEP == .70 and not a.ALLOW_HIGH_COVERAGE_LIST_DIGESTS
            tracer = TraceModel(model, a, folder)
            a.m = tracer
            result = a.consolidate(list(range(len(a.grafts))))
            receipt.update(status='COMPLETE', accepted=result[0] is not None,
                           digest_text=result[1], result=dict(a.last_consolidation_result),
                           attempts=list(a.last_consolidation_attempts))
            for trace in tracer.traces:
                trace['raw_decode'] = a.decode(trace['output_token_ids'])
                trace['generation_tokens'] = len(trace['output_token_ids'])
                trace['finish'] = ('stop' if any(s in trace['raw_decode'] for s in a.stop_sequences)
                                   else 'budget' if trace['generation_tokens'] == receipt['budget']
                                   else 'other')
            receipt['traces'] = tracer.traces
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
        finally:
            receipt['error'] = error
            # Keep the 80s lease alarm active during cleanup, after the 75s
            # worker deadline. No timeout subprocess or kill signal is used.
            remaining = max(1, int(r['lease_seconds']-(time.monotonic()-started)))
            signal.alarm(remaining)
            try:
                if repo is not None:
                    repo.arena.m = model
                    repo.close()
                repo = model = tracer = None
                gc.collect()
            except BaseException as exc:
                receipt['status'] = 'FAILED'
                receipt['cleanup_error'] = f'{type(exc).__name__}: {exc}'
            receipt['elapsed_seconds'] = time.monotonic()-started
            create(folder/'receipt.json', receipt)
        return 0 if receipt['status'] == 'COMPLETE' else 1


def run():
    r = verify()
    # Exclusive directory is immutable attempt ownership. Any started run,
    # including an interrupted controller, requires a separate lead amendment.
    target = OUT/'gpu'
    target.mkdir()
    create(target/'reservation.json', {'seconds':r['lease_seconds']*13,
           'registration_sha256':sha(REG), 'no_retry':True})
    from scripts.grm_c2_cells import environment
    completed = []
    for fold in r['folds']:
        verify()
        folder = target/f'{fold["turn"]:03d}'
        folder.mkdir()
        create(folder/'reservation.json', {'seconds':r['lease_seconds'], 'fold':fold,
                                          'registration_sha256':sha(REG)})
        env = environment(r['flags'])
        env['GRM_FIX5_PARENT'] = str(os.getpid())
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        with (folder/'worker.log').open('x') as log:
            # Foreground owned child, deliberately no subprocess timeout (it
            # kills the child). The child's existing alarm/lease raises instead.
            p = subprocess.run([sys.executable, '-m', 'scripts.grm_scout_fix5',
                '--worker', str(fold['turn'])], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT)
        ok = p.returncode == 0 and (folder/'receipt.json').exists()
        create(folder/'controller.json', {'status':'COMPLETE' if ok else 'FAILED',
                'returncode':p.returncode, 'charged_seconds':r['lease_seconds']})
        if not ok:
            create(target/'summary.json', {'status':'RED_INCOMPLETE', 'complete_turns':completed,
                   'failed_turn':fold['turn'], 'charged_seconds':r['lease_seconds']*(len(completed)+1)})
            return 1
        completed.append(fold['turn'])
    rows = [read(target/f'{t:03d}'/'receipt.json') for t in completed]
    create(target/'summary.json', {'status':'COMPLETE', 'folds':13,
           'accepted':sum(x['accepted'] for x in rows), 'charged_seconds':13*r['lease_seconds'],
           'rows':[{'turn':x['turn'], 'coverage':x['result']['best_cov'],
                    'accepted':x['accepted'], 'digest_text':x['digest_text']} for x in rows]})
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--check', action='store_true')
    g.add_argument('--run', action='store_true')
    g.add_argument('--worker', type=int)
    args = p.parse_args()
    if args.check:
        r = verify()
        print(json.dumps({'status':'CPU_BINDINGS_PASS_GPU_NOT_RUN','folds':len(r['folds']),
                          'registration_sha256':sha(REG)}, sort_keys=True))
        return 0
    return worker(args.worker) if args.worker is not None else run()


if __name__ == '__main__':
    raise SystemExit(main())
