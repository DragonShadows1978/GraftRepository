"""C2 opt-in profile and evidence checks; CPU-only imports.

Prior art: GRM WC1/RT1.1, RS3 and SCOUT-FIX-1 (project contributors, 2026).
Taken: width/capture/seating flags, manifest capture evidence, token-seat sums,
and named admission fields. Ours: additive opt-in registry and paired gate
orchestration. No new routing algorithm; no prior art known to me beyond these
local systems for this particular registration adapter.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/grm_c2'
REGISTRY = ROOT / 'config/grm_eb1_profile_registered.json'
REGISTRATION = OUT / 'registration.json'
PROFILE_ID = 'gpt-oss-20b-eb1-w96-live-rt1'
RT1_FIELDS = ('admission_split_child_demoted', 'admission_split_demoted_ids',
              'admission_split_family_ids', 'admission_ranking_before_demotion')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def select_profile(flags, selection=None, registry=None):
    """Explicit selection only; returns a copy and changes only registered keys."""
    result = copy.deepcopy(flags)
    if selection is None:
        return result
    registry = read(REGISTRY) if registry is None else registry
    if selection != registry['profile_id']:
        raise ValueError(f'unknown explicit profile: {selection}')
    result.update(copy.deepcopy(registry['flag_overrides']))
    return result


def verify_registration():
    registration = read(REGISTRATION)
    for name, digest in registration['immutable_inputs'].items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if sha(path) != digest:
            raise ValueError(f'registered input changed: {name}')
    return registration


def capture_grade(nodes, side, width, n_sink):
    """Keep inherited and unpinned evidence explicit; never invent capture metadata."""
    rows = []
    for i, node in enumerate(nodes):
        c = node.get('capture')
        if not c:
            grade = 'INHERITED_UNATTESTED'
        elif c.get('capture_pin') == 'off':
            grade = 'FRESH_UNPINNED' if side == 'defaults' else 'WRONG_PIN'
        elif (c.get('capture_pin') == 'live' and
              c.get('arena_width') == width and c.get('n_sink') == n_sink and
              c.get('capture_shift') == n_sink + width and
              c.get('capture_shift_observed') == n_sink + width and
              c.get('live_shift') == n_sink + width):
            grade = 'FRESH_LIVE_PINNED' if side == 'profile' else 'WRONG_PIN'
        else:
            grade = 'GEOMETRY_MISMATCH'
        rows.append({'node': i, 'grade': grade, 'capture': c})
    accepted = {'FRESH_LIVE_PINNED'} if side == 'profile' else {'FRESH_UNPINNED'}
    return {'rows': rows, 'valid': bool(rows) and all(r['grade'] in accepted for r in rows)}


def token_seats(grafts, mounts):
    """B5: count mounted token rows, excluding sink/live; missing is unknown."""
    try:
        return sum(int(grafts[int(i)]['ntok']) for i in mounts)
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def compare(rows, registry=None):
    """Matched probe IDs, controls, metadata and scores; missing data cannot pass."""
    registry = read(REGISTRY) if registry is None else registry
    out = {}
    for battery, expected_ids in registry['probe_ids'].items():
        arms = {}
        for side in ('profile', 'defaults'):
            phases = {}
            for phase in ('fresh', 'restart'):
                selected = [r for r in rows if (r['battery'], r['side'], r['phase']) ==
                            (battery, side, phase)]
                ids = [r['probe_id'] for r in selected]
                complete = sorted(ids) == sorted(expected_ids) and len(set(ids)) == len(ids)
                phases[phase] = {'complete': complete, 'correct': sum(r['correct'] for r in selected),
                                 'rows': {r['probe_id']: r for r in selected}}
            f, r = phases['fresh'], phases['restart']
            retained = f['complete'] and r['complete'] and all(
                f['rows'][i]['correct'] == r['rows'][i]['correct'] and
                r['rows'][i].get('metadata_retained') is True and
                r['rows'][i].get('new_process') is True for i in expected_ids)
            evidence = all(x.get('capture_valid') is True and x.get('token_seats') is not None and
                           all(k in x.get('rt1', {}) for k in RT1_FIELDS)
                           for p in phases.values() for x in p['rows'].values())
            arms[side] = {'phases': phases, 'restart_retained': retained, 'evidence_complete': evidence}
        p, d = arms['profile'], arms['defaults']
        controls = all(not d['phases']['fresh']['rows'].get(i, {}).get('correct') or
                       p['phases']['fresh']['rows'].get(i, {}).get('correct', False)
                       for i in expected_ids)
        original = registry['original_correct_controls'][battery]
        controls = controls and all(p['phases']['fresh']['rows'].get(i, {}).get('correct', False)
                                    for i in original)
        exact = p['phases']['fresh']['correct'] == registry['prediction']['profile'][battery]
        acceptance = (all(a['restart_retained'] and a['evidence_complete'] for a in arms.values())
                      and controls and p['phases']['fresh']['correct'] >=
                      registry['prediction']['defaults'][battery] and
                      p['phases']['fresh']['correct'] >= d['phases']['fresh']['correct'])
        out[battery] = {'arms': arms, 'controls_unchanged': controls,
                        'fix_validation_exact': exact and p['restart_retained'],
                        'adoption_acceptance': acceptance}
    return {'batteries': out, 'adopt': all(v['adoption_acceptance'] for v in out.values()),
            'fix_validation': all(v['fix_validation_exact'] for v in out.values())}
