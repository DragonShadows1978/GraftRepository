#!/usr/bin/env python3
"""C2 amendment 3: isolated fixed-source epoch, retaining all historical charges.

Prior art: C2 amendment1 / CMC1 (project contributors, 2026), locally inspected.
Taken: unchanged cell worker, comparator, leases and create-only scheduler.
New: source-bound epoch directory, explicit historical command supersession,
and retained prior charge. No prior art known to me beyond these local systems.
SHA-256 binding is content integrity, not a signature or novelty claim.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import grm_c2_amended as base
from scripts.grm_c2_profile import read, sha

AMENDMENT = ROOT / 'artifacts/grm_c2/amendment_lead_3.json'
AMENDMENT_SHA256 = '218cbc8cff58762dee57f4fe399d4440f13db74956d6a5cc18bd6859708d9b22'
EPOCH_OUT = ROOT / 'artifacts/grm_c2/epochs/scout-fix-2'
_BASE_REGISTRATION = base.effective_registration
_BASE_CHARGED = base.charged_seconds
_BASE_BINDINGS = base.receipt_bindings
_BASE_DRY_RUN = base.dry_run


def verify_amendment():
    if sha(AMENDMENT) != AMENDMENT_SHA256:
        raise ValueError('forged or stale amendment 3')
    value = read(AMENDMENT)
    template = Path(__file__).read_text().replace(AMENDMENT_SHA256, 'PENDING_REGISTRATION')
    if hashlib.sha256(template.encode()).hexdigest() != value['verifier_template_sha256']:
        raise ValueError('stale amendment 3 verifier source')
    for name, digest in value['bindings'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'stale amendment 3 binding: {name}')
    # Check both hashes and the source path set: an added executable input is
    # also a new source epoch. Runtime-loaded module fingerprints stay in cells.
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
    epoch = verify_amendment()
    r = _BASE_REGISTRATION()
    return dict(r, epoch=epoch)


def receipt_bindings():
    return dict(_BASE_BINDINGS(), epoch='scout-fix-2',
                epoch_amendment_sha256=sha(AMENDMENT))


def charged_seconds(r):
    return r['epoch']['prior_charged_seconds'] + _BASE_CHARGED(r)


def dry_run():
    value = _BASE_DRY_RUN()
    r = effective_registration()
    value.update(epoch='scout-fix-2', epoch_directory=str(EPOCH_OUT),
                 epoch_amendment_sha256=sha(AMENDMENT),
                 prior_charged_seconds=r['epoch']['prior_charged_seconds'],
                 charged_seconds=charged_seconds(r),
                 total_estimated_seconds_including_prior=r['estimated_seconds'] + r['epoch']['prior_charged_seconds'],
                 states={c['id']: base.cell_state(c) for c in base.ordered_cells(r)},
                 resume_rule=r['epoch']['resume_rule'])
    return value


def install():
    # Process-local adapter; every descendant executes this entry point.
    # Legacy amended CLI also dispatches here, sharing the epoch mutex.
    base.OUT = EPOCH_OUT
    base.old.OUT = EPOCH_OUT
    base.__file__ = __file__
    base.effective_registration = effective_registration
    base.charged_seconds = charged_seconds
    base.receipt_bindings = receipt_bindings
    base.dry_run = dry_run


def main():
    install()
    return base.main()


if __name__ == '__main__':
    from scripts.grm_c2_epoch3 import main as epoch_main
    raise SystemExit(epoch_main())
