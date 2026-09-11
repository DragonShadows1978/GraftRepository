#!/usr/bin/env python3
"""GRM-R1 amendment 1: device-memory release + re-arm batch R1's missing cells.

Cause (from the lead's own receipts, `artifacts/grm_r1/batch_R1.log`):
`GraftRepository.close()` releases only the NATIVE STORE
(`core/graft_repository.py:368-372`); it never touches the arena.  But
`ArenaCache._graft_block` documents that "Grafts are device-resident tc
tensors", so every `grafts[i]['h']` an arm harvests stays pinned in VRAM
after `close()`.  With a fresh arena per arm and two arms per cell, resident
payloads accumulate ACROSS cells inside the one batch process: the log shows
16+16+19+19+26+26+16+16+19+19+26 = 218 node payloads resident when the 12th
arm-pass tried to harvest 26 more and hit
`RuntimeError('cudaMalloc failed: out of memory')` at turn 22.

Fix: `grm_r1_replay.release_arm` drops each arm's device payloads using the
repository pager's OWN idiom (`g['h'] = None`, as in `_free_retired` /
`_page` / `_mark_payload_missing`) and then calls `tensor_cuda.empty_cache()`,
BEFORE the next arm allocates.  A per-cell device-memory probe receipt
(`device_memory`, MiB before/after each arm and each cell) makes the next
run's peak profile visible instead of inferred.

Scope: amendment 1 re-arms ONLY batch R1, and within it ONLY the cells that
have no receipt.  The completed cell receipts are retained byte-for-byte and
are never re-run -- they are valid create-only receipts, parity-clean, with
measured answers.  R2-R5 are untouched: the release fix removes accumulation
rather than changing per-cell cost, so their registered estimates stand.

Prior art: GRM C7 / C2 sha-bound amendment chains (GRM contributors, 2026) --
TAKEN verbatim: an amendment is a separate create-only file bound to the
registration hash which widens scope explicitly and never edits the immutable
registration; the prior attempt's controller is archived, not overwritten.
GRM FIX8 `grm_scout_fix8_resume` (GRM, 2026) -- TAKEN: the shape of a
registered successor to a single OOM-failed batch, and the nvidia-smi
framebuffer probe.  OURS: the arena-payload release itself and the
"retain completed cells, re-issue only the missing ones" scope.
No prior art known to me for this exact composition.  No new algorithm.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.grm_c7_amendment7 import read, sha, write, need  # noqa: E402
from scripts import grm_r1_replay as r1  # noqa: E402

#: Sources this amendment explicitly re-binds (their hashes changed).
REBIND = (
    'scripts/grm_r1_replay.py',
    'scripts/grm_r1_amend_1.py',
)


def observed_state(root=None):
    """What is actually on disk for each batch: receipts present vs missing."""
    root = Path(root) if root is not None else r1.OUT
    r = read(r1.REG)
    state = {}
    for batch_id in r['batch_ids']:
        d = root / 'gpu' / batch_id
        if not (d / 'controller.json').exists():
            continue
        controller = read(d / 'controller.json')
        retain, reissue = [], []
        for cell_id in r['batches'][batch_id]:
            (retain if (d / 'cells' / (cell_id + '.json')).exists()
             else reissue).append(cell_id)
        state[batch_id] = {'status': controller['status'],
                           'error': controller.get('error'),
                           'charged_seconds': controller['charged_seconds'],
                           'retain': retain, 'reissue': reissue}
    return state


def build(root=None):
    root = Path(root) if root is not None else r1.OUT
    path = root / 'amendment_1.json'
    need(not path.exists(), 'R1_AMENDMENT_IMMUTABLE_ALREADY_EXISTS')
    r = read(r1.REG)
    state = observed_state(root)
    failed = {b: s for b, s in state.items() if s['status'] != 'COMPLETE'}
    need(failed, 'R1_NOTHING_FAILED_TO_REARM')
    rearm = {}
    for batch_id, s in failed.items():
        need(s['reissue'], 'R1_FAILED_BATCH_HAS_NO_MISSING_CELL: ' + batch_id)
        # Honest re-estimate: per-cell cost from the RETAINED receipts of this
        # very batch (both arms, measured wall), plus one model load and the
        # registered 60 s headroom.  No guess, and no reuse of the original
        # lease, which was sized for eight cells rather than these.
        walls = []
        for cell_id in s['retain']:
            row = read(root / 'gpu' / batch_id / 'cells' / (cell_id + '.json'))
            walls.append(sum(a['wall_seconds'] for a in row['arms'].values()))
        per_cell = (max(walls) if walls
                    else r['per_cell_estimate_seconds'][s['reissue'][0]])
        side = r1.cell_by_id(s['reissue'][0])['side']
        load = r['cost_model'][side]['model_load_seconds']
        lease = int(min(285, max(60, round(load + per_cell * len(s['reissue'])
                                           + 60))))
        rearm[batch_id] = {
            'retain': s['retain'], 'reissue': s['reissue'],
            'lease_seconds': lease,
            'prior_status': s['status'], 'prior_error': s['error'],
            'prior_charged_seconds': s['charged_seconds'],
            'measured_per_cell_seconds': round(per_cell, 2),
            'estimate_basis': ("max both-arm wall over this batch's retained "
                               'receipts, plus one model load and 60 s '
                               'headroom; measured, not guessed'),
        }
    # The failed attempt's charge stays on the books; the re-arm adds to it.
    charged = sum(s['charged_seconds'] for s in state.values())
    added = sum(e['lease_seconds'] for e in rearm.values())
    need(charged + added <= r['gpu_cap_seconds'], 'R1_AMENDMENT_BUDGET_RAIL')
    value = {
        'schema': 'grm.r1.amendment.v1',
        'amendment': 1,
        'registration_sha256': sha(r1.REG),
        'order': 'orders/GRM_R1_MARGIN_FIRST_REPLAY.md',
        'order_sha256': sha(ROOT / 'orders/GRM_R1_MARGIN_FIRST_REPLAY.md'),
        'cause': ('GraftRepository.close() frees only the native store; the '
                  'arena\'s device-resident grafts[i]["h"] payloads survive, '
                  'so resident payloads accumulate across arms and cells in '
                  'one batch process. 218 node payloads were resident when '
                  'the 12th arm-pass harvested 26 more and hit cudaMalloc '
                  'OOM at turn 22. Evidence: artifacts/grm_r1/batch_R1.log.'),
        'fix': ('grm_r1_replay.release_arm drops each arm\'s device payloads '
                'with the pager\'s own g["h"] = None idiom and calls '
                'tensor_cuda.empty_cache() BEFORE the next arm allocates; a '
                'per-cell device_memory probe receipt records MiB before and '
                'after each arm and each cell.'),
        'rebound_inputs': {p: sha(ROOT / p) for p in REBIND},
        'rebound_reason': ('release_arm, device_memory_mib, the amendment '
                           'reader and the re-arm scope live in the worker, '
                           'so its hash necessarily changes; every other '
                           'registered input keeps its original hash.'),
        'rearm': rearm,
        'retained_receipts_policy': (
            'Completed cell receipts are VALID create-only receipts: retained '
            'byte-for-byte, never re-run, never rewritten. Only cells with no '
            'receipt are re-issued. resume_scope() compares the amendment\'s '
            'retain/reissue lists against what is actually on disk and STOPS '
            '(R1_AMENDMENT_RETAIN_MISMATCH / _REISSUE_MISMATCH) if they '
            'disagree, rather than silently re-scoping.'),
        'unchanged_batches': [b for b in r['batch_ids'] if b not in rearm],
        'unchanged_reason': ('The release fix removes cross-cell accumulation '
                             'rather than changing per-cell cost, so the '
                             'registered R2-R5 estimates stand unamended.'),
        'prior_charged_seconds': charged,
        'added_reserved_seconds': added,
        'total_reserved_seconds': charged + added,
        'gpu_cap_seconds': r['gpu_cap_seconds'],
        'prediction': r['prediction'],
        'verdict_rule': r['verdict_rule'],
        'acceptance_unchanged': True,
        'prior_art': __doc__,
        'evidence_class': ('Campaign-control amendment. Changes device memory '
                           'lifecycle and campaign scope only; no admission, '
                           'routing or scoring logic is touched, and the '
                           'prediction and verdict rule are unchanged.'),
    }
    write(path, value)
    path.with_suffix('.sha256').write_text(sha(path) + '  amendment_1.json\n')
    return value


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
