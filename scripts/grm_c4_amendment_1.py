#!/usr/bin/env python3
"""Write lead amendment 1 as A3 (r1 already used A1/A2), before CPU gates.

Prior art: RS3/WC1/C4 immutable SHA-bound registrations (house, 2026).
Borrow their source records and amendment chain; add only the lead-authorized
fixed diagonal, cap, per-cell schedule and historical comparison. No new
algorithm claimed. All evidence reads are local; no model/GPU is loaded.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import grm_c4_campaign as c

ORDER = c.ROOT / 'orders/GRM_C4_AMENDMENT_1.md'
AMENDMENT = c.OUT / 'amendment_a3.json'


def diagonal_pin():
    # Prior art: WC1 frame_binding and RS3 derived geometry (house, 2026).
    # Resolve archived relative paths explicitly, retaining original SHA pins.
    rs3_path = c.SHARED / 'artifacts/grm_rs3/registration.json'
    rs3 = c.read(rs3_path)
    geometry = rs3['geometry_REGISTERED_BEFORE_ANY_GATE']
    assert (geometry['n_sink'], geometry['arena_width'], geometry['live_shift']) == (19, 96, 115)
    assert rs3['capture_pin_REGISTERED_BEFORE_ANY_GATE']['pins']['live']['capture_shift'] == 115
    frame_path = c.ARCHIVE / 'frames/runtime_frame_w96_590846a9fbab95c0.json'
    from scripts.grm_wc1_common import unchanged_fields
    assert unchanged_fields(c.read(c.FRAME), c.read(frame_path))
    assert c.read(frame_path)['resolved_flags']['arena_width'] == 96
    fingerprints = c.historic_fingerprint()
    assert all(r['match'] for r in fingerprints), 'WC1 source fingerprint mismatch'
    sources = {rs3_path, frame_path}
    workers = []
    for p in sorted((c.ARCHIVE / 'runs/w96').glob('*.json')):
        row = c.read(p)
        if row.get('schema') not in ('grm.wc1.sup_fixture.v1', 'grm.wc1.census_shard.v1', 'grm.wc1.longhorizon_shard.v1'):
            continue
        fb = row['frame_binding']
        assert fb['single_variable_check'] and fb['derived_width'] == fb['requested_width'] == 96
        assert fb['derived_frame']['sha256'] == c.record(frame_path)['sha256']
        assert fb['frozen_frame']['sha256'] == c.record(c.FRAME)['sha256']
        assert row['registration']['sha256'] == c.record(c.ARCHIVE / 'registration.json')['sha256']
        if row['battery'] == 'sup':
            levers = row['rs3_levers']
            assert row['arena_width_on_receipt'] == 96 and row['frame']['ephemeral']
        elif row['battery'] == 'census':
            levers = row['shard']['rs3_levers']
        else:
            levers = row['rs3_levers_resolved']
        assert levers['capture_pin'] == 'live' and levers['seat_near_live']
        sources.add(p)
        workers.append(c.record(p))
    assert len(workers) == 13  # 4 sup, 4 census, 5 historical LH segments
    for p in sorted((c.ARCHIVE / 'runs/w96').rglob('run_config.json')):
        config = c.read(p)
        assert config['arena_width'] == 96 and config['live_turns'] == 2
        sources.add(p)
    results = c.read(c.ARCHIVE / 'grm_wc1_results.json')
    for width in ('64', '96'):
        for battery in results['detail'][width].values():
            for original in battery['sources']:
                p = c.ARCHIVE / Path(original['path']).relative_to('artifacts/grm_wc1')
                assert c.record(p)['sha256'] == original['sha256']
                sources.add(p)
    assert results['detail']['96']['sup']['score'] == '9/9'
    assert results['detail']['96']['census']['score'] == '9/10'
    assert results['detail']['96']['longhorizon']['all_probes'] == '13/14'
    assert results['detail']['64']['sup']['score'] == '8/9'
    assert results['detail']['64']['census']['score'] == '10/10'
    assert results['detail']['64']['longhorizon']['all_probes'] == '14/14'
    return {
        'cell': 'c96_w96', 'decision': 'CITE_WC1', 'rerun_estimate_seconds': 0,
        'evidence_class': 'archived E2E receipts plus source/registration reasoning',
        'rs3_registration': c.record(rs3_path), 'wc1_derived_frame': c.record(frame_path),
        'n_sink': 19, 'capture_shift': 115, 'live_shift': 115,
        'plan_head_last_position': 114, 'capture_pin': 'live', 'seat_near_live': True,
        'live_turns': 2, 'ephemeral': True,
        'derivation': 'WC1 width96 uses unchanged RS3 band: n_sink+96=115; near-live head ends at 114. Deposit budget equals width96. C4 fixed_geometry(96) retains that geometry.',
        'scores': {'sup': 9, 'census': 9, 'longhorizon': 13},
        'counts': {'sup': 9, 'census': 10, 'longhorizon': 14},
        'worker_pins': workers, 'historical_fingerprints': fingerprints,
        'limits': 'Reuse for geometry/quality diagonal only. Historical actual token residency is missing; graft count is not token seats. No new GPU parity claim.',
    }, sources


def main():
    assert not AMENDMENT.exists(), 'Amendment immutable; do not overwrite'
    assert not any((c.OUT / 'runs').glob('*/*.json')), 'Existing C4 worker state requires lead review'
    base = c.read(c.REG)
    assert c.record(c.REG)['sha256'] == (c.OUT / 'registration.sha256').read_text().split()[0]
    prior = None
    for path in (c.OUT / 'amendment_a1.json', c.OUT / 'amendment_a2.json'):
        assert c.record(path)['sha256'] == path.with_suffix('.sha256').read_text().split()[0]
        assert c.read(path)['registration'] == c.record(c.REG)
        if prior:
            assert c.read(path)['previous_amendment'] == prior
        prior = c.record(path)
    ruling, extra = diagonal_pin()
    cells = copy.deepcopy(base['cells'])
    for cell in cells:
        if cell['id'] == 'c64_w64':
            cell.update(status='NEW', reason='Lead amendment 1 authorizes fixed geometry replacement; historical row remains a separate comparison.')
        if cell['id'] == 'c96_w96':
            assert cell['status'] == 'CITE_WC1'
    budget = {**base['budget'], 'gpu_seconds': 4800,
              'new_cells_estimate_seconds': 4800,
              'full_fixed_geometry_estimate_seconds': 4800,
              'full_fixed_geometry_status': 'PLANNING_FIT_AT_CAP_UNVALIDATED',
              'authorized_by': 'Lead, orders/GRM_C4_AMENDMENT_1.md, 2026-09-09',
              'planning_cooldown_seconds': 63 * 30,
              'reserve_note': '285s full-worker reservation unchanged. At exactly estimated unit times the final worker needs 5005s cumulative reservation, so a 4800s estimate does not guarantee completion under the 4800s cap; stop NON_FIT if reserve fails. Cooldown is outside GPU budget.'}
    comparison = {
        'fixed_cell': 'c64_w64', 'historical_cell': 'WC1-historical c64_w64',
        'historical_results': c.record(c.ARCHIVE / 'grm_wc1_results.json'),
        'historical_scores': {'sup': 8, 'census': 10, 'longhorizon': 14},
        'counts': base['counts'],
        'historical_geometry': {'n_sink': 19, 'capture_shift': 83, 'live_shift': 83, 'plan_head_last_position': 82},
        'fixed_geometry': base['geometry'],
        'question': 'Does the historical width64 benefit survive fixed geometry at all?',
        'interpretation': 'Report all three scores, componentwise deltas and t33/Juniper probe identities against 8/9,10/10,14/14. Retaining census10 and LH14 retains those aggregate benefits; failure loses at least part. Juniper remains separately reported. This comparison alone does not support chunking.',
        'prediction_unchanged': base['prediction'], 'rejection_unchanged': base['rejection'],
    }
    paths = {Path(r['path']) for r in c.inventory()} | extra | {ORDER, c.ROOT/'orders/GRM_C4_WIDTH_VS_CHUNKING.md', c.OUT/'lead_commands.txt'}
    payload = {
        'schema': 'grm.c4.amendment.v1', 'immutable': True,
        'lead_amendment_number': 1, 'artifact_sequence': 3,
        'order': c.record(ORDER), 'registration': c.record(c.REG),
        'previous_amendment': prior,
        'reason': 'Lead accepted r1 geometry mismatch and raised cap to 4800s for a complete fixed-geometry cross.',
        'model_id': 'GPT-6 (Codex; no more specific deployment id exposed)',
        'reasoning_effort': 'high (order)',
        'overrides': {'cells': cells, 'budget': budget, 'diagonal_ruling': ruling,
                      'diagonal_comparison': comparison,
                      'execution_order': ['c64_w96', 'c96_w64', 'c64_w64']},
        'unchanged': ['geometry', 'fixtures', 'counts', 'acceptance', 'prediction', 'rejection', 'units_per_new_cell', '285s worker / 590s outer / 30s cooldown'],
        'gates_before_gpu': ['C4 and P2C CPU suite including added cell', 'amendment binding and order-drift refusal', 'dry-run 3 executable cells / 63 units / 4800s', 'per-cell command dependency and bash syntax check'],
        'prior_art': 'Verified local house RS3/WC1/EB1/DET1/C4 (2026): geometry derivation, immutable SHA chain, schedule enumeration, per-battery deltas, session resume and lease. This amendment adds lead-authorized controls and pins; no novel algorithm claimed. Original external Fisher (1935) lead remains unverified — lead to check: Fisher factorial experiments 1935.',
        'sources': [c.record(p) for p in sorted(paths)],
    }
    c.write_once(AMENDMENT, payload)
    with AMENDMENT.with_suffix('.sha256').open('x') as f:
        f.write(c.record(AMENDMENT)['sha256'] + '  amendment_a3.json\n')
    print(json.dumps({'amendment': c.record(AMENDMENT), 'order': payload['order'],
                      'cells': cells, 'budget': budget}, indent=2))


if __name__ == '__main__':
    main()
