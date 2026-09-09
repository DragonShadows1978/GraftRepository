#!/usr/bin/env python3
"""A8 incident-scoped recovery; CPU preflight, explicit lead-only GPU worker.

Prior art: C4/A2/A5/A6/A7 and DET1 (house, 2026), verified local source:
create-only SHA manifests, ordered receipt selection, claim/reserve accounting,
and verified saved-state copies. A8 adds only lead-authorized incident classes
and a df free-space guard. No novel algorithm claimed.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager, ExitStack
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import grm_c4_campaign as c
from scripts import grm_c4_split_a5 as a5
from scripts import grm_c4_skip_a6 as a6
from scripts import grm_c4_accounting_a7 as a7

OUT = c.OUT/'lead_a8'
OLD_OUT = a5.OUT
ORDER = c.ROOT/'orders/GRM_C4_AMENDMENT_8.md'
AMENDMENT = OUT/'amendment_a10.json'
COMMANDS = c.OUT/'lead_resume_a8.txt'
PRE_SHA = '636f222cc9b1606e770044c418922a2560007c3d33c006e37ec9c572748e06a7'
BEFORE_SHA = '676d217f2f4266973afa486ab1eee1ab0bb306e837d2ace904db4a398f89a1bd'
MIN_FREE_BYTES = 20_000_000_000  # Lead's 20 GB, decimal bytes (not GiB).
OLD_SELECTED = a5.selected_path
OLD_HARNESS = a5.split_harness
SOURCES = ('scripts/grm_c4_recovery_a8.py', 'tests/test_grm_c4_recovery_a8.py')


def identity(path):
    name = path.name.removesuffix('.attempt.json').removesuffix('.json')
    battery, spec = name.split('_', 1)
    assert battery in ('sup', 'census', 'longhorizon'), 'Unknown worker filename'
    return path.parent.name, battery, spec


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def recorded_wall(data):
    # Prior art: Python json.JSONDecoder.raw_decode (Python stdlib, local
    # installed implementation); reuse token decoding, not regex inference.
    # Only complete TOP-LEVEL fields before damage count; nested times do not.
    decoder = json.JSONDecoder()
    text = data.decode('utf-8', errors='replace')
    pos = 0
    def space(i):
        while i < len(text) and text[i].isspace(): i += 1
        return i
    pos = space(pos)
    if text[pos:pos+1] != '{': return None
    pos += 1
    wall = None
    try:
        while True:
            key, pos = decoder.raw_decode(text, space(pos))
            pos = space(pos)
            if text[pos:pos+1] != ':': break
            value, pos = decoder.raw_decode(text, space(pos+1))
            if key == 'gpu_seconds':
                assert number(value), 'Invalid recorded corrupt wall'
                wall = value
            pos = space(pos)
            if text[pos:pos+1] != ',': break
            pos += 1
    except ValueError:
        pass
    return wall


def classify(receipt, claim, previous):
    # Prior art: A7 incomplete-receipt/claim guards (house, 2026). Exceptions
    # below are authorized ONLY when their exact pins enter the A8 amendment.
    unit = identity(receipt)
    assert unit in {(u['cell'], u['battery'], u['spec']) for u in a5.schedule(previous)}, 'Unregistered incident unit'
    assert claim == receipt.with_suffix('.attempt.json'), 'Wrong incident claim path'
    marker = c.read(claim)
    assert tuple(marker[k] for k in ('cell','battery','spec')) == unit, 'Incident claim identity mismatch'
    assert marker['registration'] == c.record(c.REG), 'Incident registration drift'
    assert marker['amendment'] == previous['amendment'], 'Incident amendment drift'
    assert marker['budget_amendment'] == previous['budget_amendment'], 'Incident budget drift'
    assert number(marker['started_unix']), 'Invalid incident start'
    base = {'unit': list(unit), 'claim': c.record(claim), 'reruns_allowed': 1}
    if receipt.exists():
        data = receipt.read_bytes()
        try:
            json.loads(data)
        except (ValueError, UnicodeDecodeError) as exc:
            wall = recorded_wall(data)
            return {**base, 'classification': 'CORRUPT_DISK_FULL',
                'receipt': c.record(receipt), 'parse_error': f'{type(exc).__name__}: {exc}',
                'recorded_wall_seconds': wall, 'charge_seconds': wall if wall is not None else 0,
                'accounting_reason': 'Lead A8: recorded top-level wall only; paired claim is not a second abandoned unit.'}
        raise AssertionError('Parseable receipt cannot be disk-full corrupt')
    return {**base, 'classification': 'ABANDONED_DISK_FULL', 'receipt_path': str(receipt),
            'charge_seconds': 285, 'accounting_reason': 'Full interrupted lease reservation; unknown usage never zero.'}


def original_roots():
    return (c.OUT, a5.a2.OUT, a5.a3.OUT, OLD_OUT)


def initial_incidents(previous):
    # Prior art: A7 filename scan (house, 2026), with exact evidence registration.
    result = []
    for path in sorted((OLD_OUT/'runs').glob('*/*.attempt.json')):
        receipt = path.with_name(path.name.replace('.attempt.json', '.json'))
        if receipt.exists():
            try: c.read(receipt)
            except ValueError: pass
            else: continue
        result.append(classify(receipt, path, previous))
    return result


def disk_preflight():
    # Prior art: GNU coreutils df (local system), available-block admission.
    # 20 GB is the lead's fixed threshold; no adaptive threshold or novelty.
    result = subprocess.run(['df', '-B1', '--output=avail', '/mnt/ForgeRealm'],
                            check=True, capture_output=True, text=True)
    lines = result.stdout.splitlines()
    assert len(lines) == 2 and lines[0].strip() == 'Avail', 'Malformed df output'
    free = int(lines[1].strip())
    print(json.dumps({'df_mount': '/mnt/ForgeRealm', 'available_bytes': free,
                      'available_GB': free/1e9, 'minimum_bytes': MIN_FREE_BYTES}), flush=True)
    assert free >= MIN_FREE_BYTES, f'Disk preflight refused: {free} bytes free < {MIN_FREE_BYTES} bytes (20 GB)'
    return free


def old_selected(previous, cell, battery, spec):
    with patch.object(a5, 'OUT', OLD_OUT):
        return OLD_SELECTED(previous, cell, battery, spec)


def old_validate(path, previous, cell, battery, spec):
    with patch.object(a5, 'OUT', OLD_OUT), patch.object(a5, 'selected_path', OLD_SELECTED):
        return a6.validate_receipt(path, previous, cell, battery, spec)


def incident_units(reg):
    return {tuple(r['unit']) for r in reg['a8_incidents']}


def selected_path(reg, cell, battery, spec):
    # Prior art: A2/A5 historical-vs-successor selection (house, 2026), reused.
    # An incident gets exactly ONE new pathname, independent of completion.
    old = old_selected(reg['a8_previous'], cell, battery, spec)
    if (cell, battery, spec) not in incident_units(reg) and old.exists(): return old
    return OUT/'runs'/cell/f'{battery}_{spec}.json'


def validate_receipt(path, reg, cell, battery, spec):
    assert path == selected_path(reg, cell, battery, spec), 'Receipt path mismatch'
    if path.is_relative_to(OUT/'runs'):
        return a6.validate_receipt(path, reg, cell, battery, spec)
    return old_validate(path, reg['a8_previous'], cell, battery, spec)


def accounting(reg, *, reserve=True):
    # Prior art: DET1/A5/A7 wall+reservation accounting (house, 2026).
    # Count each classified incident once, plus all rerun walls. Unregistered
    # corruption and new incomplete claims always block; never auto-reclassify.
    incidents = reg['a8_incidents']
    excluded = {r['claim']['path'] for r in incidents}
    excluded.update(r['receipt']['path'] for r in incidents if 'receipt' in r)
    used = sum(r['charge_seconds'] for r in incidents)
    finished, completed, claims = [], set(), []
    old_pins = {p['path']:p for p in c.read(OLD_OUT/'before.json')['files']}
    for root in (*original_roots(), OUT):
        for path in sorted((root/'runs').glob('*/*.json')):
            if str(path) in excluded:
                print(json.dumps({'accounting_classified': str(path), 'amendment': reg['amendment']['sha256']}), file=sys.stderr)
                continue
            if path.name.endswith('.attempt.json'):
                claims.append(path); continue
            if not path.name.startswith(('sup_', 'census_', 'longhorizon_')): continue
            try: row = c.read(path)
            except ValueError as exc:
                raise AssertionError(f'Unregistered corrupt worker; campaign blocked: {path}: {exc}') from exc
            assert isinstance(row, dict) and row.get('status') in ('PASS','RED','NON_FIT'), f'Incomplete worker; campaign blocked: {path}'
            assert tuple(row.get(k) for k in ('cell','battery','spec')) == identity(path), f'Worker identity mismatch: {path}'
            if root in (OLD_OUT, OUT):
                row = validate_receipt(path, reg, *identity(path))
            else:
                assert c.record(path) == old_pins.get(str(path)), f'Historical receipt drift: {path}'
            if row['status'] == 'NON_FIT':
                assert row.get('gpu_executed') is False and row.get('gpu_seconds') == 0, 'Invalid NON_FIT accounting'
                continue
            assert all(number(row.get(k)) for k in ('gpu_seconds','finished_unix')), f'Invalid wall accounting: {path}'
            used += row['gpu_seconds']; finished.append(row['finished_unix']); completed.add(path)
    for claim in claims:
        receipt = claim.with_name(claim.name.replace('.attempt.json', '.json'))
        assert receipt in completed, f'Unfinished worker claim; campaign blocked: {claim}'
    if reserve:
        assert used+285 <= reg['budget']['gpu_seconds'], 'GPU budget non-fit; no run'
        if finished: assert time.time()-max(finished) >= 30, '30s cooldown not met'
    return used


def remaining(previous, incidents):
    units = {tuple(r['unit']) for r in incidents}
    return [u for u in a5.schedule(previous) if (u['cell'],u['battery'],u['spec']) in units
            or not old_selected(previous,u['cell'],u['battery'],u['spec']).exists()]


def projection(reg):
    # Prior art: A6 fitting-only planning projection (house, 2026), unchanged
    # estimates; replace completed estimates with walls, add incident charges.
    used = accounting(reg, reserve=False)
    running = peak = used
    units = remaining(reg['a8_previous'], reg['a8_incidents'])
    for u in units:
        if u['dispatch'] == 'SKIP': continue
        peak = max(peak, running+max(285,u['estimate_seconds']))
        running += u['estimate_seconds']
    return {'recorded_seconds_including_incident_charges': used,
        'remaining_estimate_seconds': running-used, 'total_seconds': running,
        'cap_seconds': 6600, 'headroom_seconds': 6600-running,
        'peak_reservation_seconds': peak, 'status': 'FIT' if peak <= 6600 else 'NON_FIT',
        'remaining_units': len(units), 'remaining_fitting_units': sum(u['dispatch']=='WORKER' for u in units),
        'limit': 'Planning only; six registered skips and all state dependencies unchanged. Dependency-blocked fitting estimates retained. Long-history NON_FIT; no full score claim.'}


def command_text(previous, incidents):
    # Prior art: A6 foreground sequence (house, 2026), retain order/rails.
    lines = ['#!/bin/bash','set -e',f'cd {c.ROOT}',
             '# A8 only; original receipts and claims remain immutable.',
             'python scripts/grm_c4_recovery_a8.py preflight','sleep 30']
    units = remaining(previous, incidents)
    for cell in a5.CELLS:
        cell_units = [u for u in units if u['cell']==cell]
        for u in cell_units:
            args = f"--cell {cell} --battery {u['battery']} --spec {u['spec']}"
            if u['dispatch']=='SKIP': lines.append(f'python scripts/grm_c4_recovery_a8.py skip {args}')
            else: lines.extend([f'timeout --signal=KILL 590s python scripts/grm_c4_recovery_a8.py worker {args}','sleep 30'])
        # A8 summaries use a single amendment; recreate CPU-only partial scores
        # in the isolated root even for the already completed c64_w96 cell.
        if cell_units: lines.append(f'python scripts/grm_c4_recovery_a8.py score --cell {cell}')
    lines.extend(['python scripts/grm_c4_recovery_a8.py score --cell c64_w96',
                  'python scripts/grm_c4_recovery_a8.py summary'])
    return '\n'.join(lines)+'\n'


def expected_payload(previous):
    pre = c.record(OUT/'pre_registration.json')
    assert pre['sha256'] == PRE_SHA, 'A8 pre-registration drift'
    before = c.read(OUT/'before.json')
    assert c.record(OUT/'before.json')['sha256'] == BEFORE_SHA == c.read(OUT/'registration_pins.json')['before_sha256'], 'A8 before drift'
    for pin in before['files']:
        assert c.record(pin['path']) == pin, f'A8 historical drift: {pin["path"]}'
    old_paths = {str(p) for root in original_roots() for p in (root/'runs').glob('*/*.json')}
    pinned_paths = {p['path'] for p in before['files'] if '/runs/' in p['path']}
    assert old_paths == pinned_paths, 'New unregistered historical evidence; campaign blocked'
    incidents = initial_incidents(previous)
    for row in incidents:
        if 'receipt' in row:
            dest = OUT/'corrupt'/row['receipt']['sha256']/Path(row['receipt']['path']).name
            pin = c.record(dest)
            assert (pin['sha256'],pin['bytes']) == (row['receipt']['sha256'],row['receipt']['bytes']), 'Corrupt archive drift'
            row['archive'] = pin
    assert COMMANDS.read_text() == command_text(previous,incidents), 'A8 commands drift'
    return {'schema':'grm.c4.disk-full-recovery.v1','immutable':True,
        'lead_amendment_number':8,'artifact_sequence':10,'order':c.record(ORDER),
        'previous_amendment':c.record(a7.AMENDMENT),'pre_registration':pre,
        'before':c.record(OUT/'before.json'),'registration_pins':c.record(OUT/'registration_pins.json'),
        'incidents':incidents,'receipt_root':str(OUT/'runs'),
        'sources':[c.record(c.ROOT/p) for p in SOURCES], 'commands':c.record(COMMANDS),
        'budget':previous['budget'],'minimum_free_bytes':MIN_FREE_BYTES,
        'policy':'Exact incident only; original bytes retained; one create-only successor path. Unknown new incomplete claims block. Corrupt charges recorded top-level wall only; abandoned charges 285. No cap/skip/dependency/scoring/lease changes.',
        'prior_art':c.read(OUT/'pre_registration.json')['prior_art']}


def context(previous, row):
    return {**previous, 'a8_previous':previous,'a8_incidents':row['incidents'],
            'amendment':c.record(AMENDMENT),'sources':row['sources']}


def binding():
    with patch.object(a5, 'OUT', OLD_OUT):
        previous = a6.binding()
    pin = c.record(AMENDMENT)
    assert pin['sha256'] == AMENDMENT.with_suffix('.sha256').read_text().split()[0], 'A8 amendment SHA mismatch'
    row = c.read(AMENDMENT)
    assert row == expected_payload(previous), 'A8 amendment content/source mismatch'
    return context(previous,row)


@contextmanager
def executor():
    # Prior art: A6 runtime adapter (house, 2026), preserving all pinned sources.
    with ExitStack() as stack:
        for name, value in (('OUT',OUT),('binding',binding),('selected_path',selected_path),
                            ('validate_receipt',validate_receipt),('accounting',accounting)):
            stack.enter_context(patch.object(a5,name,value))
        yield


def worker(cell, battery, spec):
    disk_preflight()
    reg = binding()
    unit = (cell,battery,spec)
    old = old_selected(reg['a8_previous'],*unit)
    if unit not in incident_units(reg):
        assert not old.exists() and not old.with_suffix('.attempt.json').exists(), 'No retries or overwrite'
    @contextmanager
    def harness(chunk,width,root,events):
        # Prior art: A2/A5 copy-and-verify resume state (house, 2026). Census
        # resumes relative to its run_dir; copy the intact predecessor into
        # the isolated root before calling the unchanged census implementation.
        if battery == 'census':
            units = [u for u in reg['units_per_new_cell'] if u['battery']==battery]
            index = next(i for i,u in enumerate(units) if u['spec']==spec)
            if index:
                _, prior = a5.pass_receipt(reg,cell,battery,units[index-1]['spec'])
                source = a5.a2.state_files(prior)
                dest = root/'census/arm1'/prior['spec']/'session'
                if source != dest:
                    assert not dest.exists(), 'Census seed exists; no overwrite'
                    shutil.copytree(source,dest)
                    for pin in prior['session_state']:
                        copied = c.record(dest/Path(pin['path']).relative_to(source))
                        assert (copied['sha256'],copied['bytes']) == (pin['sha256'],pin['bytes']), 'Copied census state drift'
        with OLD_HARNESS(chunk,width,root,events) as modules: yield modules
    with executor(), patch.object(a5,'split_harness',harness):
        # Guard the disk again at each reservation immediately before the
        # underlying worker can import/acquire a lease; no actual lease here.
        original_ready = a5.ready
        def ready(*args):
            disk_preflight()
            return original_ready(*args)
        with patch.object(a5,'ready',ready): return a5.worker(*unit)


def register():
    previous = a6.binding()
    incidents = initial_incidents(previous)
    assert len(incidents)==1 and incidents[0]['classification']=='CORRUPT_DISK_FULL', 'Incident snapshot differs from diagnosis'
    assert incidents[0]['receipt']['sha256']=='28eb9c9dd370c07f00f2e5e10e413f987c1d80bde3d2ab11f67fe96b008b03c6', 'Unexpected corrupt bytes'
    for row in incidents:
        dest = OUT/'corrupt'/row['receipt']['sha256']/Path(row['receipt']['path']).name
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f: f.write(Path(row['receipt']['path']).read_bytes())
    c.write_once(OUT/'registration_pins.json',{'before_sha256':c.record(OUT/'before.json')['sha256']})
    with COMMANDS.open('x') as f: f.write(command_text(previous,incidents))
    c.write_once(AMENDMENT,expected_payload(previous))
    with AMENDMENT.with_suffix('.sha256').open('x') as f: f.write(c.record(AMENDMENT)['sha256']+'  '+AMENDMENT.name+'\n')
    print(c.record(AMENDMENT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('register','preflight','worker','skip','score','summary'))
    parser.add_argument('--cell',choices=a5.CELLS)
    parser.add_argument('--battery',choices=('sup','census','longhorizon'))
    parser.add_argument('--spec')
    args = parser.parse_args()
    if args.command=='register': register(); return
    if args.command=='worker':
        row=worker(args.cell,args.battery,args.spec)
        if row is not None: print(json.dumps(row,indent=2))
        return
    reg=binding()
    with executor():
        if args.command=='preflight':
            disk_preflight()
            print(json.dumps({'gpu_executed':False,'amendment':reg['amendment'],
                              'incidents':reg['a8_incidents'],'projection':projection(reg)},indent=2))
        elif args.command=='skip':
            assert any(u['dispatch']=='SKIP' and (u['cell'],u['battery'],u['spec'])==
                       (args.cell,args.battery,args.spec) for u in a5.schedule(reg)), 'Not registered skip'
            print(json.dumps(a5.worker(args.cell,args.battery,args.spec),indent=2))
        elif args.command=='score': print(json.dumps(a5.score(args.cell),indent=2))
        else: print(a5.format_summary(a5.cross_summary()))


if __name__=='__main__': main()
