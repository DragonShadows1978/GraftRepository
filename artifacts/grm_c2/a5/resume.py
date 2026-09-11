#!/usr/bin/env python3
"""A5 dispatch 2 same-epoch continuation; unchanged epoch3 scheduler/worker on disk.

Prior art: C2 amendments1/3 and CMC1 (project contributors, 2026), inspected
locally. Taken: process-local adapters, SHA-bound registration, retained charge,
create-only cells and flock. New: cap override and one named atomic relocation
of a reservation-only RED. No prior art known to me beyond these local systems
for this adapter. SHA integrity is not a signature or novelty claim.
Dispatch 2 rebuilds registration and validation; the first-dispatch source design
is reused after its template hash was verified. No historical Python is edited.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'config/grm_eb1_profile_registered.json').is_file())
sys.path.insert(0, str(ROOT))
from scripts import grm_c2_epoch3 as epoch
from scripts import grm_c2_amended as base
from scripts.grm_c2_profile import read, sha

AMENDMENT = ROOT / 'artifacts/grm_c2/amendment_lead_5_r2.json'
AMENDMENT_SHA256 = '2668e7184d9c05126a25a9ac46ff0de12a86ed68a9092343a022382e5f13f967'
LEAD_COMMAND = 'artifacts/grm_c2/lead_commands.txt'
_BASE_LOCK = base.campaign_lock
_BASE_DRY = epoch.dry_run
_BASE_CAMPAIGN = base.run_campaign
_BASE_SUMMARY = base.summary
_RESUMING = False
_INSTALLED = False


def verify_amendment():
    if sha(AMENDMENT) != AMENDMENT_SHA256:
        raise ValueError('forged or stale amendment 5')
    a = read(AMENDMENT)
    template = Path(__file__).read_text().replace(AMENDMENT_SHA256, 'PENDING_REGISTRATION')
    if hashlib.sha256(template.encode()).hexdigest() != a['verifier_template_sha256']:
        raise ValueError('stale amendment 5 adapter source')
    for name, digest in a['bindings'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'stale amendment 5 binding: {name}')
    # Frozen completed receipts remain at their original paths. The one RED
    # may only move to its registered archive; no other historical substitution.
    for name, digest in a['completed_receipts'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'changed completed receipt: {name}')
    red_location(a)
    return a


def reservation_only(directory, a):
    if sorted(p.name for p in directory.iterdir()) != ['controller.json']:
        raise ValueError('reservation-only RED has extra files or worker/reservation evidence')
    p = directory / 'controller.json'
    if sha(p) != a['red_controller_sha256']:
        raise ValueError('reservation-only RED controller hash differs')
    c = read(p)
    if (c['status'] != 'RED' or c['charged_seconds'] != 0
            or c['worker_sha256'] is not None or c['cell']['id'] != a['red_cell']
            or c['error'] != 'ValueError: budget rail: cannot reserve next 285-second cell'):
        raise ValueError('not the registered reservation-only RED')
    return c


def red_location(a):
    original = base.OUT / 'cells' / a['red_cell']
    archive = base.OUT / 'reservation_red_a5' / a['red_cell']
    if archive.exists():
        reservation_only(archive, a)
        # If the original also exists, it is the new attempt, never re-eligible.
        return archive
    reservation_only(original, a)
    return original


def verify_epoch():
    """A3 verification (project, 2026), with one explicit command supersession."""
    a = verify_amendment()
    if sha(epoch.AMENDMENT) != epoch.AMENDMENT_SHA256:
        raise ValueError('forged or stale amendment 3')
    value = read(epoch.AMENDMENT)
    template = Path(epoch.__file__).read_text().replace(epoch.AMENDMENT_SHA256, 'PENDING_REGISTRATION')
    if hashlib.sha256(template.encode()).hexdigest() != value['verifier_template_sha256']:
        raise ValueError('stale amendment 3 verifier source')
    for name, digest in value['bindings'].items():
        path = ROOT / (a['historical_lead_commands'] if name == LEAD_COMMAND else name)
        if sha(path) != digest:
            raise ValueError(f'stale amendment 3 binding: {name}')
    paths = sorted(str(p.relative_to(ROOT)) for d in ('core', 'scripts', 'cpp', 'config')
                   for p in (ROOT / d).rglob('*') if p.is_file()
                   and p.suffix in ('.py', '.cpp', '.hpp', '.h', '.json'))
    if paths != value['source_paths']:
        raise ValueError('stale amendment 3 source path set')
    controller = read(ROOT / value['prior_controller'])
    if controller['status'] != 'RED' or controller['charged_seconds'] != value['prior_charged_seconds']:
        raise ValueError('historical RED charge differs')
    actual = sorted(str(p.relative_to(ROOT / 'artifacts/grm_c2/cells'))
                    for p in (ROOT / 'artifacts/grm_c2/cells').iterdir())
    if actual != value['historical_started_cells']:
        raise ValueError('historical epoch acquired additional started cells')
    return value


def effective_registration():
    r = epoch.effective_registration()
    a = verify_amendment()
    # A1 still validates its original 3600s data. Only this registered field is
    # overridden, after all historical checks, for the existing reservation rail.
    return dict(r, budget_seconds=a['budget_seconds'], budget_amendment=a)


def reeligible_once(r):
    """A3 create-only discipline (2026); atomic archive is the one-time marker."""
    a = r['budget_amendment']
    original = base.OUT / 'cells' / a['red_cell']
    archive = base.OUT / 'reservation_red_a5' / a['red_cell']
    if archive.exists():
        reservation_only(archive, a)
        return False
    reservation_only(original, a)
    cell = base.old.cell_by_id(r, a['red_cell'])
    if any(base.cell_state(base.old.cell_by_id(r, dep)) != 'COMPLETE' for dep in cell['depends']):
        raise ValueError('reservation-only RED dependency incomplete')
    if base.charged_seconds(r) + r['worker_seconds'] > r['budget_seconds']:
        raise ValueError('budget rail: amendment 5 still cannot reserve cell')
    archive.parent.mkdir(parents=True, exist_ok=True)
    original.rename(archive)  # Same filesystem, under campaign flock; bytes preserved.
    return True


@contextmanager
def campaign_lock():
    with _BASE_LOCK() as fd:
        if _RESUMING:
            # Reverify under lock before the only historical mutation (relocation).
            reeligible_once(effective_registration())
        yield fd


def run_campaign(r, resume=False):
    global _RESUMING
    if not resume:
        raise ValueError('amendment 5 requires explicit --run --resume')
    _RESUMING = True
    try:
        return _BASE_CAMPAIGN(r, resume=True)
    finally:
        _RESUMING = False


def dry_run():
    d = _BASE_DRY()
    r = effective_registration()
    a = r['budget_amendment']
    archived = (base.OUT / 'reservation_red_a5' / a['red_cell']).exists()
    d.update(budget_seconds=r['budget_seconds'],
             budget_amendment_sha256=sha(AMENDMENT),
             reservation_only_red=a['red_cell'],
             reservation_only_eligibility='CONSUMED' if archived else 'ONE_TIME_ON_EXPLICIT_RESUME',
             resume_rule=a['resume_rule'])
    return d


def summary(r):
    # A1's defaults96 NON_FIT sentence is historical at3600, not valid at4800.
    # Keep its immutable diagnostic labelled rather than changing arm scope.
    print('A5 cap=4800 s; defaults96 remains excluded by unchanged scope. '
          'The following A1 defaults96 NON_FIT/greater-than line is historical at3600 s.')
    return _BASE_SUMMARY(r)


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    # Every cell and worker child re-enters this SHA-bound adapter. Original
    # receipt identity stays epoch3 so the42 existing cells and A4 remain valid.
    epoch.install()
    epoch.verify_amendment = verify_epoch
    base.effective_registration = effective_registration
    base.__file__ = str(Path(__file__).resolve())
    base.campaign_lock = campaign_lock
    base.run_campaign = run_campaign
    base.dry_run = dry_run
    base.summary = summary
    _INSTALLED = True


def main():
    install()
    # Direct --cell/--worker are descendant interfaces only. The lead must use
    # the campaign mutex and explicit resume to consume the one-time exception.
    if '--cell' in sys.argv and not base.os.environ.get('GRM_C2_CAMPAIGN_FD'):
        print('RED: direct --cell disabled; use --run --resume', file=sys.stderr)
        return 1
    return base.main()


if __name__ == '__main__':
    raise SystemExit(main())
