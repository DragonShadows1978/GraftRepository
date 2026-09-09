"""C7 CPU contracts. Prior art: local C2/EB1/SCOUT-FIX-1 (GRM contributors,
2026): immutable manifests, summed seats and checkpoint hashes. Borrowed:
those contracts. C7 adds the fixture, strict answer grammar and stratification;
no prior art known to me for this exact composition. No routing algorithm.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/grm_c7'
REG = OUT / 'registration.json'
FIX = OUT / 'fixture.json'
DISTANCES = (5, 30, 60, 120, 250)
CLASSES = ('fresh', 'alias', 'correction', 'folded')
SOURCE_PATH = 'core/graft_repository.py'
SOURCE_BEFORE = '2bb38b8efd8f75f3d189ade592133f3a40b245182b98c04b485976d5b2ed4311'
SOURCE_AFTER = 'fc6b9448efb45c29d5d2271fe929e3e4b6519867567cf1c69588b2a99e4773db'
LEAD_ORDER = 'orders/GRM_C7_AMENDMENT_1.md'
LEAD_ORDER_SHA = '5e23a7f9fd06ac9442129fcbf28345a5ff88b2e149a2843171499665ca8d7215'
LEAD_COMMANDS_SHA = 'a0320cf79b5d2ba3f86f16e13673802fa316e65d0d66db2123683dc09f695ee7'


def verify_lead_1(r, inputs):
    # Prior art: C7 A1/A2 and C2 (GRM contributors, 2026), inspected locally.
    # Taken: SHA-bound chain, before archives and exact scope allowlists.
    # Ours: pin the authorized SCOUT-FIX-2 delta and preserve C7 acceptance.
    # No prior art known to me for this exact composition; no new algorithm.
    path = OUT / 'amendment_lead_1.json'
    checksum = path.with_suffix('.sha256')
    if not path.is_file() or not checksum.is_file():
        raise ValueError('LEAD_1_AMENDMENT_REQUIRED')
    if sha(path) != checksum.read_text().split()[0]:
        raise ValueError('LEAD_1_SHA_MISMATCH')
    a = read(path)
    if a['schema'] != 'grm.c7.lead-amendment.v1':
        raise ValueError('LEAD_1_SCHEMA_MISMATCH')
    if (a['registration_sha256'] != sha(REG)
            or a['previous_amendment_sha256'] != r.get('amendment_A2_sha256')
            or a['previous_amendment_sha256'] !=
            '41b70efdf1735684322a8762c2b716348180bb51ec48a6cc05c630b13c2285fb'):
        raise ValueError('LEAD_1_CHAIN_MISMATCH')
    if a['order'] != {'path': LEAD_ORDER, 'sha256': LEAD_ORDER_SHA} or sha(ROOT/LEAD_ORDER) != LEAD_ORDER_SHA:
        raise ValueError('LEAD_1_ORDER_MISMATCH')
    if (set(a['overrides']) != {SOURCE_PATH, 'scripts/grm_c7_common.py', 'scripts/grm_c7_run.py'}
            or set(a['new_inputs']) != {LEAD_ORDER, 'tests/test_grm_c7_lead_1.py',
                                       'tests/test_grm_scout_fix2_capture.py'}):
        raise ValueError('LEAD_1_SCOPE_MISMATCH')
    source = a['overrides'][SOURCE_PATH]
    if source['before_sha256'] != SOURCE_BEFORE or source['after_sha256'] != SOURCE_AFTER:
        raise ValueError('LEAD_1_SOURCE_MISMATCH')
    if (a['acceptance'] != r['acceptance'] or a['acceptance_wording_changed'] is not False
            or a['registered_items'] != {
                'fixtures': 'unchanged', 'probes': 'unchanged', 'oracle': 'unchanged',
                'restart_metadata_scope': 'all persisted nodes including split children',
                'paging': 'A2 controlled roundtrip unchanged; child capture retained',
                'cells_budgets_thresholds': 'unchanged'}):
        raise ValueError('LEAD_1_ACCEPTANCE_MISMATCH')
    if (a['lead_commands_sha256'] != LEAD_COMMANDS_SHA
            or sha(OUT/'lead_commands.txt') != LEAD_COMMANDS_SHA):
        raise ValueError('LEAD_1_COMMANDS_MISMATCH')
    for name, change in a['overrides'].items():
        if (change['before_sha256'] != inputs[name]
                or sha(ROOT/change['before_archive']) != inputs[name]):
            raise ValueError('LEAD_1_BEFORE_MISMATCH')
        inputs[name] = change['after_sha256']
    inputs.update(a['new_inputs'])
    r['amendment_lead_1_sha256'] = sha(path)
    r['amended_source_sha256'] = SOURCE_AFTER


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def create(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')


def hashes(directory):
    return {str(p.relative_to(directory)): sha(p)
            for p in sorted(Path(directory).rglob('*')) if p.is_file()}


def verify():
    if sha(REG) != (OUT / 'registration.sha256').read_text().split()[0]:
        raise ValueError('REGISTRATION_SHA_MISMATCH')
    r = read(REG)
    inputs = dict(r['immutable_inputs'])
    amendment = OUT/'amendment_A1.json'
    if amendment.exists():
        # Prior art: C2 sha-bound amendments (GRM, 2026). A1 changes only
        # campaign code for restoring registered cold state, never the plan.
        if sha(amendment) != (OUT/'amendment_A1.sha256').read_text().split()[0]:
            raise ValueError('AMENDMENT_SHA_MISMATCH')
        a = read(amendment)
        if a['registration_sha256'] != sha(REG):
            raise ValueError('AMENDMENT_REGISTRATION_MISMATCH')
        if set(a['overrides']) != {'scripts/grm_c7_common.py', 'scripts/grm_c7_run.py'}:
            raise ValueError('AMENDMENT_SCOPE_MISMATCH')
        if set(a['new_inputs']) != {'tests/test_grm_c7_a1.py'}:
            raise ValueError('AMENDMENT_NEW_INPUT_SCOPE_MISMATCH')
        for path, change in a['overrides'].items():
            if change['before_sha256'] != inputs[path] or sha(ROOT/change['before_archive']) != inputs[path]:
                raise ValueError('AMENDMENT_BEFORE_MISMATCH')
            inputs[path] = change['after_sha256']
        inputs.update(a['new_inputs'])
        r['amendment_sha256'] = sha(amendment)
    a2path = OUT/'amendment_A2.json'
    if a2path.exists():
        a2 = read(a2path)
        if sha(a2path) != (OUT/'amendment_A2.sha256').read_text().split()[0]:
            raise ValueError('AMENDMENT_A2_SHA_MISMATCH')
        if a2['registration_sha256'] != sha(REG) or a2['previous_amendment_sha256'] != r.get('amendment_sha256'):
            raise ValueError('AMENDMENT_A2_CHAIN_MISMATCH')
        if set(a2['overrides']) != {'scripts/grm_c7_common.py','scripts/grm_c7_run.py'} or set(a2['new_inputs']) != {'tests/test_grm_c7_a2.py'}:
            raise ValueError('AMENDMENT_A2_SCOPE_MISMATCH')
        for path, change in a2['overrides'].items():
            if change['before_sha256'] != inputs[path] or sha(ROOT/change['before_archive']) != inputs[path]:
                raise ValueError('AMENDMENT_A2_BEFORE_MISMATCH')
            inputs[path] = change['after_sha256']
        inputs.update(a2['new_inputs'])
        r['amendment_A2_sha256'] = sha(a2path)
        r['pressure']['controlled_source_turn'] = 6
        r['acceptance']['paging'] = 'each pressure forces registered old source hot -> durable cold -> production loader return; report controlled return separately from probe-selected return'
    verify_lead_1(r, inputs)
    if os.environ.get('GRM_C7_REVISION') == 'r2':
        # Prior art: local C7 amendment chain (GRM contributors, 2026).
        # r2 explicitly binds changed core/harness before live SHA checks.
        from scripts.grm_c7_register_r2 import verify_r2
        verify_r2(r, inputs, ROOT, OUT)
    elif os.environ.get('GRM_C7_REVISION', 'r1') != 'r1':
        raise ValueError('UNKNOWN_C7_REVISION')
    for path, digest in inputs.items():
        p = Path(path) if Path(path).is_absolute() else ROOT / path
        if sha(p) != digest:
            raise ValueError(f'INPUT_SHA_MISMATCH: {path}')
    m = read(OUT / 'fixture_manifest.json')
    if m['fixture_sha256'] != sha(FIX):
        raise ValueError('FIXTURE_SHA_MISMATCH')
    r['effective_inputs'] = inputs
    return r


def normalize(answer):
    # Prior art: exact-match QA scoring, local C2 comparator (2026). C7 uses
    # full-string equality, not substring matching; case remains significant.
    # External antecedent: Rajpurkar et al., SQuAD (2016), arXiv:1606.05250;
    # answerability control: Rajpurkar/Jia/Liang, SQuAD 2.0 (2018), 1806.03822.
    # Borrowed concepts only; our normalization/UNKNOWN grammar is not SQuAD's.
    text = unicodedata.normalize('NFKC', str(answer)).strip()
    for c in '\u2010\u2011\u2012\u2013\u2014\u2212':
        text = text.replace(c, '-')
    text = text.rstrip('.').strip().strip('`*').strip()
    return re.sub(r'\s*\|\s*', '|', text.rstrip('.').strip())


def score(answer, probe):
    text = normalize(answer)
    low = text.casefold().replace('\u2019', "'")
    abstained = bool(re.search(
        r"\b(?:unknown|not in memory|not (?:provided|specified)|"
        r"(?:do not|don't|cannot|can't) (?:know|recall|find)|"
        r"(?:do not|don't) have (?:that|the|this) information)\b", low))
    required = bool(probe['answerable'])
    exact = text == normalize(probe['expected'])
    # Disjoint errors: an answerable abstention is not ALSO an exact-value
    # error. Total exact-answer failure remains available for acceptance.
    return {'exact_correct': exact,
            'exact_answer_error': int(not exact),
            'wrong_value_error': int(required and not exact and not abstained),
            'abstention_error': int(required and abstained),
            'unsupported_answer_error': int(not required and not abstained),
            'abstained': abstained, 'answer': str(answer)}


def table(rows, probes):
    expected = {p['id']: p for p in probes}
    ids = [r['probe_id'] for r in rows]
    if len(ids) != len(set(ids)) or any(i not in expected for i in ids):
        raise ValueError('DUPLICATE_OR_UNKNOWN_PROBE')
    result = []
    for d in DISTANCES:
        for c in CLASSES:
            ps = [p for p in probes if p['distance'] == d and p['class'] == c]
            selected = [r for r in rows if r['probe_id'] in {p['id'] for p in ps}]
            entry = {'distance': d, 'class': c, 'expected_n': len(ps),
                     'n': len(selected), 'complete': len(selected) == len(ps)}
            for side in ('memory', 'oracle'):
                scores = [score(r[side]['answer'], expected[r['probe_id']]) for r in selected]
                entry[side] = {'n': len(scores)}
                for key in ('exact_answer_error', 'wrong_value_error', 'abstention_error',
                            'unsupported_answer_error'):
                    n = sum(s[key] for s in scores)
                    entry[side][key] = n
                    entry[side][key + '_rate'] = n / len(scores) if scores else None
            result.append(entry)
    return result


def checkpoint(directory, state, binding):
    # Prior art: C2 snapshot/tree_hashes (GRM, 2026). Taken: immutable tree
    # fingerprints. Ours: next-turn and parent binding covering session state.
    # Recovery antecedent: Mohan et al., ARIES (1992), IBM Research publication.
    # We reuse the checkpoint/recovery concept, not the ARIES WAL algorithm.
    directory = Path(directory)
    create(directory / 'state.json', state)
    create(directory / 'checkpoint.json', {
        'binding': binding, 'files': hashes(directory),
        'next_turn': state['next_turn'], 'process_id': state['process_id']})


def validate_checkpoint(directory, next_turn, binding):
    directory = Path(directory)
    cp = read(directory / 'checkpoint.json')
    actual = hashes(directory)
    actual.pop('checkpoint.json')
    if actual != cp['files'] or cp['binding'] != binding or cp['next_turn'] != next_turn:
        raise ValueError('CHECKPOINT_INTEGRITY_OR_BOUNDARY_MISMATCH')
    state = read(directory / 'state.json')
    if state['next_turn'] != next_turn or state['process_id'] != cp['process_id']:
        raise ValueError('CHECKPOINT_STATE_MISMATCH')
    return state


def folded_evidence(grafts, mounts, source_ids):
    # Prior art: local digest/era lineage (GRM, 2026); ordinary graph reachability.
    # Taken: sources edges. C7 requires actual parent seating, forbids raw reads.
    # DFS antecedent: Tarjan (1972), unverified — lead to check search terms
    # "Tarjan depth first search linear graph algorithms 1972". No DFS novelty.
    def ancestry(i, seen):
        if i in seen:
            return set()
        seen.add(i)
        return {i}.union(*(ancestry(int(j), seen) for j in grafts[i].get('sources', [])))
    targets = set(source_ids)
    parents = [i for i in mounts if grafts[i].get('kind') in ('digest', 'era')]
    covered = set().union(*(ancestry(i, set()) for i in parents)) if parents else set()
    return bool(targets and targets <= covered and not targets.intersection(mounts)
                and all(grafts[i].get('retired') for i in targets))
