"""GRM-R1 CPU gate: all 31 registered cells replayed on the fake model.

The CPU double has no reader, so this gate measures PLANS, never answers.
It is the pre-GPU proof that (a) every recorded checkpoint still verifies,
(b) arm OFF reproduces the FIX-6 recorded plan on all 31, and (c) arm ON
reproduces the FIX-6 registered margin_first plan on all 31.

Prior art: GRM FIX-6 ``c2_receipt.json`` (GRM contributors, 2026) --
TAKEN: the "OFF byte-identical / ON reference parity" gate shape, replayed
here through the worker instead of the one-off FIX-6 script.  OURS: running
it as the worker's own CPU gate over the batch/lease/receipt machinery.
No prior art known to me for this exact composition.  No quality claim.
"""
from __future__ import annotations

import json

import pytest

from scripts import grm_r1_replay as run


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/registration.json')
def test_all_31_cells_replay_on_cpu_with_both_plans_reproduced(tmp_path):
    r = run.verify()
    for batch_id in r['batch_ids']:
        run.batch(batch_id, fake=True, root=tmp_path)

    rows = {}
    for batch_id in r['batch_ids']:
        for cell_id in r['batches'][batch_id]:
            p = tmp_path / 'gpu' / batch_id / 'cells' / (cell_id + '.json')
            rows[cell_id] = json.loads(p.read_text())
    assert len(rows) == 31

    registered = {c['id']: c for c in run.cells()}
    off_parity = 0
    on_parity = 0
    for cell_id, row in rows.items():
        c = registered[cell_id]
        # (a) the recorded checkpoint tree verified before either arm ran
        assert row['same_state_proof']['checkpoint_sha256'] == c['checkpoint_sha256']
        assert (row['same_state_proof']['checkpoint_files_verified']
                == len(c['checkpoint_files']))
        # (b) arm OFF == the FIX-6 recorded plan, byte for byte
        assert row['off_plan_bytes_hex'] == row['recorded_off_plan_bytes_hex']
        off_parity += bool(row['off_plan_parity'])
        # (c) arm ON == the FIX-6 registered margin_first plan
        on_parity += bool(row['on_plan_matches_registered'])
        # the rule really was the only thing that differed
        assert row['rule_pins'] == {'off': 'all_tokens_bind',
                                    'on': 'margin_first'}
        assert row['arms']['off']['admission_rule'] == 'all_tokens_bind'
        assert row['arms']['on']['admission_rule'] == 'margin_first'
        # and the plan actually changed, which is why the cell was registered
        assert row['arms']['off']['rank_plan'] != row['arms']['on']['rank_plan']

    assert off_parity == 31, f'OFF parity {off_parity}/31'
    assert on_parity == 31, f'ON parity {on_parity}/31'

    value = run.summary(tmp_path)
    assert value['complete'] is True
    assert value['cells_measured'] == 31
    assert value['off_plan_parity'] is True
    # Honest ceiling: a fake run never yields an adoption verdict.
    assert value['answers_measured'] is False
    assert value['status'] == 'NOT_MEASURED'
    assert value['adopt'] is False


def test_cpu_gate_covers_every_battery_and_side(tmp_path):
    cells = run.cells()
    assert {c['battery'] for c in cells} == {'census', 'longhistory', 'sup'}
    assert {c['side'] for c in cells} == {'defaults', 'profile'}
    # sup is the battery the verdict rule treats as absolute; it must be here.
    assert sum(c['battery'] == 'sup' for c in cells) == 7


def test_the_20_unresolved_executions_are_excluded_not_imputed():
    unresolved = run.read(run.FIX6 / 'c2_unresolved.json')
    assert len(unresolved) == 20
    registered = {c['source_execution_id'] for c in run.cells()}
    for row in unresolved:
        assert row['execution_id'] not in registered
        assert row['reason'] in ('MISSING_EXACT_MARGIN',
                                 'MISSING_TIED_SCORE_GROUP',
                                 'MISSING_DECISION_RELEVANT_MARGIN')
