"""LT1 harness-only native publication and one registered failed-cell retry.

Prior art: GRM contributors (2026), repository _native_sync_node, C7/LT1
immutable checkpoints, explicit retry/accounting contracts. Reuse existing
publication, SHA verification and cell order; new: delayed publication only
for the exact objects fed by this worker, at actual native commit. No prior
art known to me for this exact composition. No selection rule is changed.
"""
from pathlib import Path
from scripts.grm_c7_common import read, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/grm_lt1/amendment4'
REG = OUT/'resume_registration.json'
PARENT_SHA = '809f906c5e7d057f8cffcf067d4eee7ec990a996f62a05f789916d2f8d47e193'
FAILED = 'A-033-040'
RETRY = 'A-033-040-retry-amendment4'
OVERRIDES = {'scripts/grm_lt1.py', 'scripts/grm_lt1_worker.py', 'artifacts/grm_lt1/lead_commands.txt'}
NEW = {'orders/GRM_LT1_AMENDMENT_4.md', 'scripts/grm_lt1_amendment4.py',
       'scripts/grm_lt1_amendment4_replay.py', 'tests/test_grm_lt1_amendment4.py',
       'artifacts/grm_lt1/amendment4/DIAGNOSIS_REGISTRATION.json',
       'artifacts/grm_lt1/amendment4/original_run_shas.json',
       'artifacts/grm_lt1/amendment4/diagnosis.json',
       'artifacts/grm_lt1/amendment4/red_before.log',
       'artifacts/grm_lt1/amendment4/route_receipt_analysis.py',
       'artifacts/grm_lt1/amendment4/route_analysis.json'}


def execution_sha():
    return sha(REG)


def install_native_publication(repo, fed, directory, context):
    # Prior art: repository _native_sync_node (GRM, 2026), unchanged publisher.
    # The late hook preserves every prior successful native-commit path: only
    # a missing ID takes treatment, and the old commit would have raised there.
    arena = repo.arena
    original = arena._commit_native_mount
    def commit(picks, mount_tokens):
        store = getattr(arena, 'native_store', None)
        if store is not None and hasattr(store, 'commit_mount'):
            missing = [i for i in picks if arena.grafts[i].get('native_node_id') is None]
            # Validate the WHOLE set first. Never repair checkpoint corruption,
            # rollback/reused ordinals, split children, or missing payloads.
            for i in missing:
                g = arena.grafts[i]
                meta = g.get('metadata', {})
                if (fed.get(i) is not g or i in repo._native_node_ids
                        or g.get('h') is None or g.get('retired')
                        or g.get('sources') or g.get('ephemeral_split_of') is not None
                        or g.get('ephemeral_split') or meta.get('width_guard_child')):
                    raise RuntimeError(f'LT1_UNPUBLISHED_NODE_OUTSIDE_FEED: {i}')
            for i in missing:
                from scripts.grm_c7_run import emit
                native_id = repo._native_sync_node(i)
                if arena.grafts[i].get('native_node_id') != native_id or native_id is None:
                    raise RuntimeError(f'LT1_NATIVE_PUBLICATION_FAILED: {i}')
                emit(Path(directory)/'native_publication.jsonl', dict(context,
                     graft_id=i, native_node_id=native_id, picks=list(picks),
                     text_sha256=__import__('hashlib').sha256(arena.grafts[i]['text'].encode()).hexdigest(),
                     evidence_class='LT1 feed publication at native mount'))
        return original(picks, mount_tokens)
    arena._commit_native_mount = commit


def apply(lt, inputs):
    if not REG.exists() or sha(REG) != REG.with_suffix('.sha256').read_text().split()[0]:
        raise ValueError('AMENDMENT4_SHA_MISMATCH')
    a = read(REG)
    if (a.get('schema') != 'grm.lt1.amendment4.resume.v1'
            or a.get('parent_sha256') != PARENT_SHA or sha(lt.AMEND3) != PARENT_SHA
            or a.get('order_sha256') != sha(ROOT/'orders/GRM_LT1_AMENDMENT_4.md')):
        raise ValueError('AMENDMENT4_CHAIN_MISMATCH')
    if (a.get('failed_cell') != FAILED or a.get('retry_directory') != RETRY
            or a.get('resume_at') != FAILED or a.get('budget_seconds') != 10800
            or a.get('retain_completed') != [f'{arm}-{start:03d}-{start+7:03d}' for start in (1,9,17,25) for arm in 'AB']
            or a.get('protocol_binding') != lt.binding('CPU')):
        raise ValueError('AMENDMENT4_PROTOCOL_MISMATCH')
    if set(a.get('overrides',{})) != OVERRIDES or set(a.get('new_inputs',{})) != NEW:
        raise ValueError('AMENDMENT4_SCOPE_MISMATCH')
    for n,c in a['overrides'].items():
        archive = 'artifacts/grm_lt1/amendment4/before/'+n
        if c.get('before_archive') != archive or c.get('before_sha256') != inputs[n] or sha(ROOT/archive) != inputs[n]:
            raise ValueError('AMENDMENT4_BEFORE_MISMATCH')
        inputs[n] = c['after_sha256']
    inputs.update(a['new_inputs'])
    return a


def cell_directory(cells, cell_id):
    # CPU fake campaigns keep their historical fixture layout. The retry is
    # ONLY for the registered real campaign and exact failed cell.
    real = ROOT/'artifacts/grm_lt1/amendment2/run_margin_first/cells'
    if Path(cells).resolve() == real and cell_id == FAILED:
        return Path(cells)/RETRY
    return Path(cells)/cell_id


def accepted_failed_charge(directory, receipt):
    original = ROOT/'artifacts/grm_lt1/amendment2/run_margin_first/cells'/FAILED
    if Path(directory).resolve() != original or receipt.get('status') != 'RED':
        return None
    manifest = read(OUT/'original_run_shas.json')
    name = str((original/'controller.json').relative_to(ROOT))
    if (sha(original/'controller.json') != manifest[name]
            or receipt.get('error') != 'ValueError: WORKER_EXIT_1'):
        raise ValueError('AMENDMENT4_FAILED_RECEIPT_CHANGED')
    return float(receipt['charged_seconds'])


def check_original_evidence():
    # Called at preflight, not per turn. Pin partial probes as evidence only;
    # summary ignores this registered failed directory to avoid double scoring.
    for n,d in read(OUT/'original_run_shas.json').items():
        if sha(ROOT/n) != d:
            raise ValueError('AMENDMENT4_ORIGINAL_EVIDENCE_CHANGED: '+n)


def cpu_ready():
    p = OUT/'cpu_receipt.json'
    return p.exists() and read(p).get('status') == 'PASS' and read(p).get('execution_amendment_sha256') == execution_sha()
