#!/usr/bin/env python3
"""Build the GRM-R1 immutable registration (create-only; amendments separate).

Prior art: GRM C2 / C7 / FIX8 registration builders (GRM contributors, 2026).
TAKEN: sha-bound immutable inputs, pessimistic per-batch reservations charged
even on success, prediction and verdict rule recorded VERBATIM from the plan
before the gate runs, and a create-only ``registration.json`` +
``registration.sha256`` pair.  OURS: the per-cell estimate derived from the
measured C2 restart-cell receipts, and the two-arm cost model.  No prior art
known to me for this exact composition; no algorithm is introduced here.

Budget derivation is from RECEIPTS, not guesses: the C2 scout-fix-2 restart
controllers give charged seconds and probe counts, from which model load and
per-probe cost are separated (see REPORT.md).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.grm_c7_amendment7 import read, sha, write, need  # noqa: E402
from scripts import grm_r1_replay as r1  # noqa: E402

C2_EPOCH = Path('/mnt/ForgeRealm/wt/grm-c2/artifacts/grm_c2/epochs/scout-fix-2')

#: Immutable code + data inputs, sha-bound at registration time.
INPUT_PATHS = (
    'core/grm_admission.py',
    'core/graft_arena.py',
    'core/graft_repository.py',
    'core/grm_three_pass.py',
    'core/grm_text_norm.py',
    'scripts/grm_r1_replay.py',
    'scripts/grm_c2_cells.py',
    'scripts/grm_c2_profile.py',
    'scripts/grm_e2e_session.py',
    'scripts/grm_c7_diagnose.py',
    'scripts/grm_cmc1_gpu_arms.py',
    'scripts/lsr_p2c_replay_gpu.py',
    'scripts/grm_det1_common.py',
    'scripts/grm_det1_3_gpu.py',
    'config/grm_eb1_profile_registered.json',
    'artifacts/grm_scout_fix6/c2_replay_cells.json',
    'orders/GRM_R1_MARGIN_FIRST_REPLAY.md',
)

#: Verbatim from the immutable plan /mnt/Shared/LEAD_TODO_2026-09-10.md (R1).
PREDICTION = ('Registered prediction: <= 2 of 31 executions change their '
              'answer; 0 correct->wrong on supersession.')
VERDICT_RULE = ('Verdict rule: margin_first is adoptable as the profile '
                'default iff correct->wrong = 0 on sup and <= 1 elsewhere.')


def measured_c2_cost():
    """Separate model-load from per-probe cost using the C2 restart receipts.

    Evidence class: campaign controller receipts (wall clock), not a model
    quality measure.  Single-probe restart cells isolate (load + 1 probe);
    four-probe cells give (load + 4 probes); the difference yields per-probe.
    """
    rows = []
    for p in sorted(C2_EPOCH.glob('cells/*/controller.json')):
        c = read(p)
        if c['cell']['phase'] != 'restart' or c['status'] != 'COMPLETE':
            continue
        rows.append({'cell': p.parent.name, 'side': c['cell']['side'],
                     'probes': len(c['cell']['probes']),
                     'charged_seconds': float(c['charged_seconds'])})
    need(rows, 'R1_NO_C2_RESTART_RECEIPTS')
    model = {}
    for side in ('defaults', 'profile'):
        side_rows = [r for r in rows if r['side'] == side]
        need(side_rows, 'R1_NO_C2_RECEIPTS_FOR_SIDE: ' + side)
        one = min(side_rows, key=lambda r: (r['probes'], r['cell']))
        many = max(side_rows, key=lambda r: (r['probes'], -r['charged_seconds']))
        need(many['probes'] > one['probes'], 'R1_DEGENERATE_C2_COST_BASIS')
        per_probe = ((many['charged_seconds'] - one['charged_seconds'])
                     / (many['probes'] - one['probes']))
        load = one['charged_seconds'] - per_probe * one['probes']
        model[side] = {'per_probe_seconds': round(per_probe, 2),
                       'model_load_seconds': round(load, 2),
                       'basis_one_probe_cell': one['cell'],
                       'basis_many_probe_cell': many['cell'],
                       'basis_one_charged': one['charged_seconds'],
                       'basis_many_charged': many['charged_seconds']}
    model['evidence_class'] = ('C2 scout-fix-2 restart controller wall clock; '
                               'timing only, no model-quality claim')
    model['rows'] = rows
    return model


def build():
    need(not r1.REG.exists(), 'R1_REGISTRATION_IMMUTABLE_ALREADY_EXISTS')
    cells = r1.cells()
    need(len(cells) == 31, 'R1_COHORT_MISMATCH')
    cost = measured_c2_cost()
    by_id = {c['id']: c for c in cells}
    # Per cell we run BOTH arms: two repository copies + two ladder passes.
    # A 1.5x safety factor over the measured per-probe cost, because a
    # replayed turn may mount MORE nodes under the margin_first arm.
    per_cell = {}
    for c in cells:
        probe = cost[c['side']]['per_probe_seconds']
        per_cell[c['id']] = int(round(2 * probe * 1.5)) + 2
    # Batch by side (a batch shares one model load, and environment(flags) is
    # per side), preserving the registered ORDER of the FIX-6 cells file.
    batches = {}
    batch_ids = []
    current = None
    for c in cells:
        if (current is None
                or by_id[batches[current][0]]['side'] != c['side']
                or len(batches[current]) >= 8):
            current = f'R{len(batch_ids) + 1}'
            batch_ids.append(current)
            batches[current] = []
        batches[current].append(c['id'])
    lease = {}
    for b in batch_ids:
        load = max(cost[by_id[i]['side']]['model_load_seconds']
                   for i in batches[b])
        work = sum(per_cell[i] for i in batches[b])
        # Pessimistic ceiling: this is what gets charged, success or not.
        lease[b] = int(min(285, max(60, round(load + work + 60))))
    reserved = sum(lease.values())
    need(reserved <= 3600, 'R1_BUDGET_OVER_ONE_GPU_HOUR')
    registration = {
        'schema': 'grm.r1.replay.v1',
        'order': 'orders/GRM_R1_MARGIN_FIRST_REPLAY.md',
        'plan': '/mnt/Shared/LEAD_TODO_2026-09-10.md (R1)',
        'plan_immutable': True,
        'cell_count': len(cells),
        'distinct_question_ids': len({c['question_id'] for c in cells}),
        'cells_sha256': sha(r1.CELLS),
        'cells_path': str(r1.CELLS.relative_to(ROOT)),
        'batches': batches,
        'batch_ids': batch_ids,
        'batch_lease_seconds': lease,
        'per_cell_estimate_seconds': per_cell,
        'reserved_seconds': reserved,
        'reserved_gpu_hours': round(reserved / 3600.0, 4),
        'gpu_cap_seconds': 3600,
        'gpu_cap_gpu_hours': 1.0,
        'budget': (f'{len(batch_ids)} pessimistic per-batch reservations '
                   f'totalling {reserved}s ({reserved / 3600.0:.4f} GPU-h), '
                   f'charged in full even on success, under the registered '
                   f'1.0 GPU-h cap. Estimates derive from the C2 scout-fix-2 '
                   f'restart controller receipts. No retries and no cohort '
                   f'trimming: inability to finish STOPS the campaign.'),
        'cost_model': cost,
        'free_space_min_bytes': 20 * (1024 ** 3),
        'lease_path': '/tmp/forge-gpu.lock',
        'worker_seconds': 280,
        'outer_seconds': 590,
        'cooldown_seconds': 30,
        'arms': {'off': {'GRM_ADMISSION_RULE': None,
                         'rule': 'all_tokens_bind',
                         'note': "today's shipped default"},
                 'on': {'GRM_ADMISSION_RULE': 'margin_first',
                        'rule': 'margin_first',
                        'note': 'the FIX-6 opt-in rule under test'}},
        'parity_barrier': ('Arm OFF must reproduce the FIX-6 recorded off_plan '
                           'byte-for-byte (RD1 A0 shape). A mismatch is a RED '
                           'receipt and the run STOPS; never a retry.'),
        'scorer': ('lsr_p2c_replay_gpu.answer_verdict over '
                   'grm_det1_common.contains_value (LT1/C5 value-span rule), '
                   'plus grm_det1_3_gpu._is_refusal for abstention. Expected '
                   'values come from the RECORDED checkpoint context only.'),
        'unresolved_policy': ('The 20 FIX-6 executions without exact margins '
                              'stay UNRESOLVED and are NOT in this cohort. '
                              'Nothing is imputed.'),
        'prediction': PREDICTION,
        'verdict_rule': VERDICT_RULE,
        'inputs': {p: sha(ROOT / p) for p in INPUT_PATHS},
        'evidence_class': ('Admission-rule A/B replay on recorded C2 '
                           'checkpoints through the production ladder. '
                           'Establishes answer transitions under one rule '
                           'flip; NOT a fresh battery and NOT a product '
                           'certification.'),
        'prior_art': r1.__doc__,
        'status': 'REGISTERED_NOT_RUN',
        'gpu_executed': False,
    }
    write(r1.REG, registration)
    r1.REG.with_suffix('.sha256').write_text(sha(r1.REG) + '  registration.json\n')
    return registration


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
