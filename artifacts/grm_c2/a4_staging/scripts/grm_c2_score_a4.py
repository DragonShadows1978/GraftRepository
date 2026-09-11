#!/usr/bin/env python3
"""CPU-only A4 receipt rescoring; never imports a runner or a worker.

Prior art: GRM C2 compare and amendments 1/3 (project contributors, 2026),
locally inspected. Reuse per-probe/control gates, SHA bindings and create-only
receipts. New: profile-scoped acceptance and dual-rule immutable reports.
No prior art known to me beyond these local systems for this adapter.
SHA-256 is integrity checking, not a signature or novelty claim.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'config/grm_eb1_profile_registered.json').is_file())
OUT = ROOT / 'artifacts/grm_c2'
EPOCH = OUT / 'epochs/scout-fix-2'
AMENDMENT = OUT / 'amendment_lead_4_v2.json'
AMENDMENT_SHA256 = 'c805908001127bc9d79b292c63cf6ba8afda118dc83a20717610582c5156def4'
BATTERIES = ('sup', 'census', 'longhistory')
TARGET = ROOT / 'scripts/grm_c2_score_a4.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def immutable(path, data):
    """C2 create-only convention (2026); mismatches never overwrite history."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('xb') as stream:
            stream.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ValueError(f'immutable output differs: {path}')


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def verify_amendment():
    if sha(AMENDMENT) != AMENDMENT_SHA256:
        raise ValueError('stale or forged amendment 4')
    a = read(AMENDMENT)
    template = Path(__file__).read_text().replace(AMENDMENT_SHA256, 'PENDING_REGISTRATION')
    if hashlib.sha256(template.encode()).hexdigest() != a['verifier_template_sha256']:
        raise ValueError('stale A4 verifier source')
    for name, digest in a['bindings'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'stale A4 binding: {name}')
    # Read-only historical source verification, deliberately no epoch3 install()
    # or path-set verifier call: the postcampaign scorer is an additive source.
    a3 = read(OUT / 'amendment_lead_3.json')
    for name in a3['source_paths']:
        digest = a3['bindings'].get(name)
        if digest is not None and sha(ROOT / name) != digest:
            raise ValueError(f'changed epoch3 source: {name}')
    return a


def context():
    a = verify_amendment()
    registry = read(ROOT / 'config/grm_eb1_profile_registered.json')
    registration = read(OUT / 'registration.json')
    bindings = dict(epoch='scout-fix-2',
                    epoch_amendment_sha256=sha(OUT / 'amendment_lead_3.json'),
                    amendment_sha256=sha(OUT / 'amendment_lead_1.json'),
                    registration_sha256=sha(OUT / 'registration.json'),
                    registry_sha256=sha(ROOT / 'config/grm_eb1_profile_registered.json'))
    return a, registry, registration, bindings


def legacy():
    # Load only the frozen, CPU-only original comparator, without sys.path edits
    # or any mutation of the epoch runner's process-local globals.
    spec = importlib.util.spec_from_file_location('c2_a4_original', ROOT / 'scripts/grm_c2_profile.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(rows, registry, battery):
    """C2 comparator (2026) unchanged; A4 changes the acceptance arm scope."""
    p = legacy()
    original = p.compare(rows, registry)['batteries'][battery]
    profile, defaults = (original['arms'][s] for s in ('profile', 'defaults'))
    coverage = all(v['complete'] for arm in original['arms'].values()
                   for v in arm['phases'].values())
    # Strict evidence typing complements the original comparator's key checks.
    profile_evidence = profile['evidence_complete'] and all(
        type(r.get('token_seats')) is int and r['token_seats'] >= 0
        and type(r.get('correct')) is bool
        for v in profile['phases'].values() for r in v['rows'].values())
    baseline = (profile['phases']['fresh']['correct'] >= registry['prediction']['defaults'][battery]
                and profile['phases']['fresh']['correct'] >= defaults['phases']['fresh']['correct'])
    corrected = (coverage and profile['restart_retained'] and profile_evidence
                 and original['controls_unchanged'] and baseline)
    exact = original['fix_validation_exact']
    return dict(original_rule=original, corrected_rule=dict(
        adoption_acceptance=corrected, fix_validation_exact=exact,
        coverage_complete=coverage, profile_evidence_complete=profile_evidence,
        profile_restart_retained=profile['restart_retained'],
        controls_unchanged=original['controls_unchanged'], baseline_floor_met=baseline),
        defaults_findings=dict(restart_retained=defaults['restart_retained'],
                               evidence_complete=defaults['evidence_complete']),
        original_rule_verdict='GREEN' if original['adoption_acceptance'] and exact else 'RED',
        corrected_rule_verdict='GREEN' if corrected and exact else 'RED')


def expected_probes(cell, registry):
    if cell['phase'] == 'restart':
        return cell['probes']
    if cell['battery'] == 'sup':
        return [p['probe_id'] for p in registry['sup_plans'][cell['session']]]
    return [f'lh_t{t:03d}' if cell['battery'] == 'longhistory' else
            next(p['probe_id'] for p in registry['census_probes'] if p['turn'] == t)
            for t in range(cell['start'], cell['stop'])
            if registry['session_script'][t]['kind'] == 'probe']


def check_binding(value, bindings):
    if any(value.get(k) != v for k, v in bindings.items()):
        raise ValueError('stale receipt binding')


def collect(epoch, registration, registry, bindings, battery):
    rows, states, hashes = [], {}, {}
    for cell in registration['cells']:
        if cell['battery'] != battery:
            continue
        directory = epoch / 'cells' / cell['id']
        controller_path = directory / 'controller.json'
        if not controller_path.exists():
            states[cell['id']] = 'STARTED_UNFINISHED' if directory.exists() else 'UNSTARTED'
            continue
        controller_bytes = controller_path.read_bytes()
        controller = json.loads(controller_bytes)
        check_binding(controller, bindings)
        if controller.get('cell') != cell:
            raise ValueError('controller cell mismatch')
        state = controller['status']
        if state not in ('COMPLETE', 'RED'):
            raise ValueError('invalid controller status')
        states[cell['id']] = state
        hashes[str(controller_path.relative_to(epoch))] = hashlib.sha256(controller_bytes).hexdigest()
        if state != 'COMPLETE':
            continue
        path = directory / 'worker.json'
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != controller.get('worker_sha256'):
            raise ValueError('worker receipt changed after completion')
        hashes[str(path.relative_to(epoch))] = digest
        worker = json.loads(data)
        check_binding(worker, bindings)
        if worker.get('cell') != cell:
            raise ValueError('worker cell mismatch')
        rr = worker['rows']
        if sorted(r['probe_id'] for r in rr) != sorted(expected_probes(cell, registry)):
            raise ValueError('missing/duplicate/unexpected cell probes')
        for row in rr:
            if any(row.get(k) != cell[k] for k in ('battery', 'side', 'phase')):
                raise ValueError('row cell mismatch')
            if type(row.get('correct')) is not bool:
                raise ValueError('invalid correctness type')
        # Missing capture in a zero-probe fresh segment must not pass vacuously.
        if (cell['side'] == 'profile' and cell['phase'] == 'fresh'
                and worker.get('end_capture', {}).get('valid') is not True):
            raise ValueError('invalid profile persisted capture evidence')
        rows.extend(rr)
    return rows, states, hashes


def evaluate(epoch, registration, registry, bindings, battery):
    rows, states, hashes = collect(epoch, registration, registry, bindings, battery)
    result = compare(rows, registry, battery)
    complete = bool(states) and all(s == 'COMPLETE' for s in states.values())
    if not complete:
        result['original_rule_verdict'] = result['corrected_rule_verdict'] = 'RED'
        result['corrected_rule']['adoption_acceptance'] = False
    prior_path = epoch / 'scores' / f'{battery}.json'
    recorded = None
    if prior_path.exists():
        data = prior_path.read_bytes()
        prior = json.loads(data)
        check_binding(prior, bindings)
        if prior.get('battery') != battery or prior.get('states') != states:
            raise ValueError('original score cell state mismatch')
        if prior.get('result') != result['original_rule'] or prior.get('status') != result['original_rule_verdict']:
            raise ValueError('original score disagrees with original-rule replay')
        recorded = prior['status']
        hashes[str(prior_path.relative_to(epoch))] = hashlib.sha256(data).hexdigest()
    return dict(schema='grm.c2.score.a4', battery=battery, **bindings,
                amendment_a4_sha256=AMENDMENT_SHA256, input_sha256=hashes,
                states=states, all_cells_complete=complete,
                recorded_original_verdict=recorded, **result,
                status=result['corrected_rule_verdict'])


def score_a4(epoch, registration, registry, bindings, batteries):
    # Evaluate all selected batteries before publishing any; partial live runs
    # cannot permanently acquire a misleading immutable all-battery score set.
    values = [evaluate(epoch, registration, registry, bindings, b) for b in batteries]
    if any(s in ('UNSTARTED', 'STARTED_UNFINISHED') for v in values for s in v['states'].values()):
        raise ValueError('campaign incomplete: selected cells are unstarted or unfinished')
    for value in values:
        immutable(epoch / 'scores_a4' / f"{value['battery']}.json", encoded(value))
        print(f"score_a4 {value['battery']}: original={value['original_rule_verdict']} corrected={value['status']}")
    return values


def table(values, registry):
    lines = ['| Battery | Arm (width) | Fresh | Restart | Seats fresh/restart | RT1 fresh/restart | Restart retained | Evidence complete | Original rule | Corrected rule |',
             '|---|---|---|---|---|---|---|---|---|---|']
    fields = legacy().RT1_FIELDS
    for v in values:
        b = v['battery']
        for side, width in [('profile', 96), ('defaults', 256)]:
            arm = v['original_rule']['arms'][side]
            scores, seats, rt1 = [], [], []
            for phase in ('fresh', 'restart'):
                p = arm['phases'][phase]
                rr = list(p['rows'].values())
                scores.append(f"{p['correct']}/{len(registry['probe_ids'][b])}" if p['complete'] else 'INCOMPLETE')
                seats.append(str(sum(r['token_seats'] for r in rr)) if p['complete'] and
                             all(type(r.get('token_seats')) is int for r in rr) else 'UNKNOWN')
                rt1.append('YES' if p['complete'] and all(all(k in r.get('rt1', {}) for k in fields) for r in rr) else 'MISSING')
            lines.append(f"| {b} | {side} ({width}) | {scores[0]} | {scores[1]} | {'/'.join(seats)} | {'/'.join(rt1)} | {arm['restart_retained']} | {arm['evidence_complete']} | {v['original_rule_verdict']} | {v['corrected_rule_verdict']} |")
    lines += ['', 'Verdicts are battery-level adoption AND exact fix-validation AND complete cell gates.',
              'Defaults restart retention/evidence are findings under A4. Profile width 96 versus shipped defaults width 256; not width matched.',
              'Original RED receipts remain unchanged. All-battery adoption requires every corrected battery GREEN.']
    return '\n'.join(lines) + '\n'


def summary_a4(epoch, registration, registry, bindings):
    values = []
    for battery in BATTERIES:
        path = epoch / 'scores_a4' / f'{battery}.json'
        if not path.exists():
            raise ValueError(f'summary incomplete: missing scores_a4/{battery}.json')
        value = evaluate(epoch, registration, registry, bindings, battery)
        if read(path) != value:
            raise ValueError(f'immutable score disagrees with receipts: {battery}')
        values.append(value)
    report = table(values, registry)
    immutable(epoch / 'scores_a4/summary_a4.md', report.encode())
    print(report)
    return 0 if all(v['status'] == 'GREEN' for v in values) else 1


def install_after_campaign(epoch, registration, bindings):
    """C2 nonblocking flock (2026); A4 defers source addition until completion."""
    with (epoch / 'campaign.lock').open('rb') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for cell in registration['cells']:
            path = epoch / 'cells' / cell['id'] / 'controller.json'
            if not path.exists():
                raise ValueError('cannot install A4: campaign incomplete')
            value = read(path)
            check_binding(value, bindings)
            if value.get('cell') != cell or value.get('status') not in ('COMPLETE', 'RED'):
                raise ValueError('cannot install A4: controller incomplete')
        if not all((epoch / 'scores' / f'{b}.json').is_file() for b in BATTERIES):
            raise ValueError('cannot install A4: original campaign scoring incomplete')
        immutable(TARGET, Path(__file__).read_bytes())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['score_a4', 'summary_a4'])
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all', action='store_true')
    group.add_argument('--battery', choices=BATTERIES)
    parser.add_argument('--install-after-campaign', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'score_a4' and not (args.all or args.battery):
        parser.error('score_a4 requires --all or --battery')
    try:
        _, registry, registration, bindings = context()
        if args.install_after_campaign:
            install_after_campaign(EPOCH, registration, bindings)
        if args.command == 'summary_a4':
            return summary_a4(EPOCH, registration, registry, bindings)
        values = score_a4(EPOCH, registration, registry, bindings,
                          BATTERIES if args.all else [args.battery])
        return 0 if all(v['status'] == 'GREEN' for v in values) else 1
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'RED: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
