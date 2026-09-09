"""Prior art: LSR-P2B and RT1 (project contributors, 2026), additive projection.
New fixture: RT1 decision evidence through both canonical receipt surfaces.
"""
import json
import pytest
from core.grm_admission import rt1_info_fields
from core.grm_three_pass import build_route_receipt, ROUTE_RECEIPT_INFO_PREFIXES


@pytest.mark.parametrize('demoted', [[], [2]])
def test_rt1_receipt_projection(demoted):
    fields = rt1_info_fields(demoted=demoted, split_members=[2, 3], ranking_before=[2, 0, 3])
    base = build_route_receipt(session_id='fix1', turn_id='0', info={'admission_policy': 'A-DEC'})
    result = build_route_receipt(session_id='fix1', turn_id='0', info={'admission_policy': 'A-DEC', **fields})
    assert result['schema'] == base['schema'] == 'grm.route_receipt.v1'
    for key, value in fields.items():
        assert result['admission'][key.removeprefix('admission_')] == value
        assert result['fit']['info_pass_through'][key] == value
    for key, value in base['admission'].items():
        assert result['admission'][key] == value
    assert json.loads(json.dumps(result)) == result
    assert ROUTE_RECEIPT_INFO_PREFIXES == (
        'fit_', 'abstain', 'demand_', 'grounding_', 'frame_', 'recency_',
        'live_segments_', 'admission_')


def test_legacy_receipt_does_not_invent_rt1_decision():
    result = build_route_receipt(session_id='fix1', turn_id='0', info={})
    assert 'split_child_demoted' not in result['admission']
    assert result['fit']['info_pass_through'] == {}
