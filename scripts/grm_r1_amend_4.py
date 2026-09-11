#!/usr/bin/env python3
"""GRM-R1 amendment 4: idle-card pre-check + NON_FIT rail; R4/R5 re-armed.

Defect (lead): `--batch R4` died 13.0 s into its lease with
`cudaMalloc failed: out of memory`, before any probe line and with zero cell
receipts (`artifacts/grm_r1/batch_R4_a3.log`).  The A1 seat's worker died the
same way 60 s later at 15 s, and `nvidia-smi` at that moment showed ~3,144
MiB used with NO compute process listed -- a display-side program on :0.
This machine is shared; that holder is never ours to touch.

NOT A FIT PROBLEM -- the counts say so plainly (item 1 of the order):

    batch  cells  max nodes  max npz MiB  batch total MiB
    R1     8      25         36.7         221.5
    R2     5      25         36.7         130.0
    R3     8      25         36.6         220.8   <- all COMPLETE
    R4     8      25         36.6         151.6   <- died on cell 1
    R5     2       3          6.1          12.3

R4's first cell (`profile-longhistory-2--lh_t022`, 25 nodes / 36.6 MiB) is
the SAME size as `profile-census-2--e2e_t22_mira_seal`, which R3 completed on
the same profile frame (freeing 23 payloads per arm).  R4 is in fact the
SMALLEST profile batch by total payload.  Against the ~1,286 MiB of headroom
the receipts show, the largest single cell is 36.6 MiB -- roughly 2.8%, or
5.7% counting both arms.  There is ~35x headroom over the largest cell, so
lever 2(a) (host-resident payloads) is NOT indicated and is NOT implemented:
it would add a page-in path to solve a problem the numbers say does not
exist, and every added path is another guard to get wrong.

What IS implemented:

(2) An idle-card pre-check on the lease path, modelled on FIX-8's replay.
    `device_snapshot()` reads the full nvidia-smi XML (total, used, free,
    compute-process list); `idle_gate()` keys on TOTAL framebuffer used,
    never on the process list, because the holder that broke R4 listed no
    compute process at all -- FIX-8's `parse_memory` states exactly this
    rule ("never infer an idle card merely from an empty compute list").
    `await_idle()` polls a read-only probe for a bounded `--idle-wait` and
    then declines.  It NEVER signals, kills or clears anything; the operator
    has absolute right of way.  The result is recorded as `idle_check`.

(b) A cell that cannot be loaded is recorded as a create-only NON_FIT
    receipt -- reason, stage, and a device-memory snapshot including whether
    a foreign holder is suspected -- and the batch CONTINUES to the next
    cell.  Only an OOM takes this path (`is_oom`, a deliberately narrow text
    match, since the engine raises a plain RuntimeError); anything else
    still stops the campaign.  A NON_FIT cell is UNMEASURED in `summary()`
    and is NEVER counted as `unchanged`: a cell that did not execute cannot
    be evidence that the rule left its answer alone.

Scope: rebind the worker sha; re-arm R4 and R5 (both open, 0 receipts);
budget unchanged, since neither lever changes per-cell cost.  Admission,
routing and scoring are untouched; the prediction, verdict rule and parity
barrier are unchanged.

Prior art: GRM C7/C2 sha-bound amendment chains (GRM contributors, 2026) --
TAKEN verbatim, including chaining to the previous amendment's hash.  GRM
FIX-8 `grm_scout_fix8_resume.parse_memory` / `memory_gate` / `snapshot`
(GRM, 2026) and the NVIDIA nvidia-smi XML framebuffer and process reporting
(docs.nvidia.com, accessed 2026-09-09) -- TAKEN: the XML fields, the
conservative total-used reading, the 1000 MiB registered limit, and the
explicit warning against inferring idleness from an empty compute list.
GRM C7/FIX8 create-only failure receipts (GRM, 2026) -- TAKEN: record a
failure as evidence instead of retrying it.  OURS: the NON_FIT class and its
unmeasured accounting, and the use of the idle probe as a pre-lease gate in
this worker.  No prior art known to me for that exact composition, and no
new algorithm.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.grm_c7_amendment7 import read, sha, write, need  # noqa: E402
from scripts import grm_r1_replay as r1  # noqa: E402

#: Sources this amendment re-binds (their hashes changed).
REBIND = (
    'scripts/grm_r1_replay.py',
    'scripts/grm_r1_amend_4.py',
)

#: Item 1 of the order: recorded node/payload counts from the checkpoint
#: manifests, for every batch. Evidence for the "not a fit problem" finding.
COUNTS = {
    'R1': {'cells': 8, 'max_nodes': 25, 'max_npz_mib': 36.7, 'total_mib': 221.5},
    'R2': {'cells': 5, 'max_nodes': 25, 'max_npz_mib': 36.7, 'total_mib': 130.0},
    'R3': {'cells': 8, 'max_nodes': 25, 'max_npz_mib': 36.6, 'total_mib': 220.8},
    'R4': {'cells': 8, 'max_nodes': 25, 'max_npz_mib': 36.6, 'total_mib': 151.6},
    'R5': {'cells': 2, 'max_nodes': 3, 'max_npz_mib': 6.1, 'total_mib': 12.3},
}


def observed_open(root):
    """Batches with a controller that is not COMPLETE, and their missing cells."""
    r = read(r1.REG)
    out = {}
    for batch_id in r['batch_ids']:
        d = root / 'gpu' / batch_id
        if not (d / 'controller.json').exists():
            continue
        controller = read(d / 'controller.json')
        if controller['status'] == 'COMPLETE':
            continue
        retain, reissue = [], []
        for cell_id in r['batches'][batch_id]:
            (retain if (d / 'cells' / (cell_id + '.json')).exists()
             else reissue).append(cell_id)
        out[batch_id] = {'status': controller['status'],
                         'error': controller.get('error'),
                         'charged_seconds': controller['charged_seconds'],
                         'retain': retain, 'reissue': reissue}
    return out


def build(root=None):
    root = Path(root) if root is not None else r1.OUT
    path = root / 'amendment_4.json'
    need(not path.exists(), 'R1_AMENDMENT_4_IMMUTABLE_ALREADY_EXISTS')
    previous = root / 'amendment_3.json'
    need(previous.is_file(), 'R1_AMENDMENT_4_REQUIRES_AMENDMENT_3')
    a3 = read(previous)
    r = read(r1.REG)
    need(a3['registration_sha256'] == sha(r1.REG),
         'R1_AMENDMENT_4_CHAIN_MISMATCH')
    rearm = dict(a3['rearm'])
    for batch_id, s in observed_open(root).items():
        need(s['reissue'], 'R1_AMENDMENT_4_NOTHING_MISSING: ' + batch_id)
        rearm[batch_id] = {
            'retain': s['retain'], 'reissue': s['reissue'],
            'lease_seconds': r['batch_lease_seconds'][batch_id],
            'prior_status': s['status'], 'prior_error': s['error'],
            'prior_charged_seconds': s['charged_seconds'],
            'estimate_basis': ('registered lease unchanged: neither lever '
                               'changes per-cell cost'),
        }
    value = {
        'schema': 'grm.r1.amendment.v1',
        'amendment': 4,
        'registration_sha256': sha(r1.REG),
        'previous_amendment_sha256': sha(previous),
        'order': 'orders/GRM_R1_MARGIN_FIRST_REPLAY.md',
        'order_sha256': sha(ROOT / 'orders/GRM_R1_MARGIN_FIRST_REPLAY.md'),
        'defects': [
            {'id': 'R1-D6-foreign-device-holder',
             'evidence': 'artifacts/grm_r1/batch_R4_a3.log',
             'symptom': ('cudaMalloc failed: out of memory 13.0 s into the '
                         'lease, before any probe line, 0 cell receipts; the '
                         'full 238 s reservation charged for nothing.'),
             'cause': ('A display-side program on :0 held ~3,144 MiB and '
                       'listed NO compute process. The card was not ours to '
                       'use and the worker started anyway. This machine is '
                       'shared; that holder is never to be touched.'),
             'not_the_cause': ('NOT a payload-fit problem. R4 cell 1 '
                               '(25 nodes / 36.6 MiB) is the same size as a '
                               'cell R3 completed on the same frame, and R4 '
                               'is the smallest profile batch by total '
                               'payload. See node_payload_counts.'),
             'fix': ('An idle-card pre-check on the lease path keyed on '
                     'TOTAL framebuffer used (never on the compute-process '
                     'list), bounded by --idle-wait, recorded as '
                     'idle_check. It only declines to start; it never '
                     'signals, kills or clears anything.')},
            {'id': 'R1-D7-one-unloadable-cell-failed-the-campaign',
             'evidence': 'artifacts/grm_r1/gpu/R4/controller.json',
             'symptom': 'One cell that could not load failed the whole batch.',
             'cause': 'No rail existed between "cell runs" and "campaign dies".',
             'fix': ('An OOM at load/harvest writes a create-only NON_FIT '
                     'receipt (reason, stage, memory snapshot) and the batch '
                     'CONTINUES. NON_FIT is unmeasured in summary() and is '
                     'never counted as unchanged. Only an OOM takes this '
                     'path; any other error still stops the campaign.')},
        ],
        'node_payload_counts': COUNTS,
        'fit_analysis': {
            'headroom_mib_at_observed_plateau': 1286,
            'largest_single_cell_mib': 36.6,
            'largest_cell_both_arms_mib': 73.2,
            'headroom_multiple_over_largest_cell': 35,
            'r3_completed_same_size_cell': (
                'profile-census-2--e2e_t22_mira_seal, 25 nodes / 36.6 MiB, '
                'freed 23 payloads per arm'),
            'conclusion': ('No R4/R5 cell is near a fit boundary. Lever 2(a) '
                           'is NOT indicated and NOT implemented.'),
        },
        'lever_2a_host_resident_payloads': {
            'implemented': False,
            'reason': ('The counts show ~35x headroom over the largest cell '
                       'and R3 already ran an identical-size cell to '
                       'completion. Adding a page-in path would solve a '
                       'problem the evidence says does not exist, and every '
                       'added path is another guard to get wrong.'),
        },
        'idle_limit_mib': 1000,
        'idle_limit_basis': ('FIX-8 registered the same 1000 MiB limit. The '
                             'receipts bracket it cleanly: an idle card in '
                             'this arc reads 314-327 MiB, and the holder '
                             'that broke R4 held ~3,144 MiB.'),
        'rebound_inputs': {p: sha(ROOT / p) for p in REBIND},
        'rebound_reason': ('Both the idle gate and the NON_FIT rail live in '
                           'the worker, so its hash changes; every other '
                           'registered input keeps its original hash.'),
        'rearm': rearm,
        'retained_receipts_policy': a3['retained_receipts_policy'],
        'unchanged_batches': [b for b in r['batch_ids'] if b not in rearm],
        'unchanged_reason': ('Neither lever changes per-cell cost: one '
                             'declines to start on a busy card, the other '
                             'records a cell that could not load.'),
        'prior_charged_seconds': a3['prior_charged_seconds'],
        'added_reserved_seconds': a3['added_reserved_seconds'],
        'total_reserved_seconds': a3['total_reserved_seconds'],
        'gpu_cap_seconds': r['gpu_cap_seconds'],
        'prediction': r['prediction'],
        'verdict_rule': r['verdict_rule'],
        'acceptance_unchanged': True,
        'lessons': [
            'Check the premise against the counts before building the fix '
            'the symptom suggests. "Bigger cells did not fit" was the '
            'natural reading of an OOM; the manifests said R4 was the '
            'smallest profile batch and R3 had already run an '
            'identical-size cell.',
            'A shared machine needs a pre-flight, not just a lease. The '
            'flock coordinates the agents that agreed to use it; it says '
            'nothing about a display-side program that never did.',
        ],
        'prior_art': __doc__,
        'evidence_class': ('Campaign-control amendment. Adds a device '
                           'pre-check and a non-fit rail; no admission, '
                           'routing or scoring logic is touched, and the '
                           'prediction and verdict rule are unchanged.'),
    }
    write(path, value)
    path.with_suffix('.sha256').write_text(sha(path) + '  amendment_4.json\n')
    return value


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
