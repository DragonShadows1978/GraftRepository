"""GRM-D1: gates for the LT1 arm-A cause table.

These tests hold the diagnosis to its receipts. They are skipped, not failed,
if the frozen LT1 run receipts are not mounted -- a missing receipt is an
environment fact, not a diagnosis error.

Prior art: RD2's C7 finding named the two traps (GRM contributors, 2026); the
classification and the join are described in scripts/grm_d1_cause_table.py.
"""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_d1_cause_table as ct  # noqa: E402

pytestmark = pytest.mark.skipif(
    not ct.CELLS.exists() or not ct.FIXTURE.exists(),
    reason='frozen LT1 run receipts not mounted at %s' % ct.CELLS)


@pytest.fixture(scope='module')
def table():
    return ct.build('A')


def test_exactly_eleven_wrong_rows(table):
    """The LT1 result: 24/35 exact, 11 wrong values, 0 abstentions."""
    assert table['wrong_rows'] == 11
    assert sum(table['class_counts'].values()) == 11
    assert all(r['score_category'] == 'wrong_value' for r in table['rows'])


def test_every_row_cites_a_receipt_that_exists(table):
    for row in table['rows']:
        receipt = Path(row['receipt'])
        assert receipt.exists(), 'missing receipt %s' % receipt
        assert receipt.name == 'probes.jsonl'
        # And the cited receipt really contains that probe row.
        ids = {json.loads(line)['probe_id']
               for line in receipt.read_text().splitlines() if line.strip()}
        assert row['question_id'] in ids


def test_class_counts_are_five_five_one(table):
    counts = table['class_counts']
    assert counts[ct.FIXTURE_LINEAGE] == 5
    assert counts[ct.ALIAS_NO_BASE] == 5
    assert counts[ct.STALE_FIRST] == 1
    assert ct.READER_WRONG not in counts
    assert ct.OTHER not in counts


def test_all_five_correction_rows_are_the_same_unretired_family(table):
    rows = [r for r in table['rows'] if r['cause_class'] == ct.FIXTURE_LINEAGE]
    assert len(rows) == 5
    for row in rows:
        assert row['probe_class'] == 'correction'
        # Nothing was ever retired ...
        assert row['current_value_node_superseded'] is False
        # ... so the current value's node exists but was never mounted ...
        assert row['current_value_node_exists'] is True
        assert row['current_value_node_mounted'] is False
        # ... and the node that WAS mounted is strictly older.
        seated = [m['deposited_turn'] for m in row['mounted_nodes']]
        assert seated and max(seated) < row['current_value_node_turn']
        # The whole family is still in the ranking.
        ranked = {m['node'] for m in row['ranked_nodes']}
        assert row['current_value_node'] in ranked


def test_the_correction_rows_serve_the_original_pre_correction_value(table):
    """The served value is turn 1's `17 credits`, not either later value."""
    rows = [r for r in table['rows'] if r['cause_class'] == ct.FIXTURE_LINEAGE]
    for row in rows:
        assert row['expected'] == '29 credits'
        assert '17 credits' in row['served_value']
        assert '29 credits' not in row['served_value']


def test_all_five_alias_rows_mount_the_base_without_the_edge(table):
    rows = [r for r in table['rows'] if r['cause_class'] == ct.ALIAS_NO_BASE]
    assert len(rows) == 5
    for row in rows:
        assert row['probe_class'] == 'alias'
        assert row['alias_edge_nodes'], 'the alias edge node must exist'
        assert row['alias_edge_mounted'] is False
        assert row['base_fact_mounted'] is True
        # The identifier DID find the alias edge -- routing is not the failure.
        assert set(row['alias_edge_nodes']) & set(
            row['admission_identified_candidates'] or [])


def test_the_single_fresh_row_had_the_right_node_ranked_first(table):
    """Honest naming: A-DEC ranked correctly; the ladder dropped the mount."""
    rows = [r for r in table['rows'] if r['cause_class'] == ct.STALE_FIRST]
    assert len(rows) == 1
    row = rows[0]
    assert row['question_id'] == 'recall_3_150'
    assert row['probe_class'] == 'fresh'
    ranked = [m['node'] for m in row['ranked_nodes']]
    assert ranked[0] == row['current_value_node'], (
        'A-DEC ranked the correct node first')
    assert row['admission_rank_plan'] == [row['current_value_node']]
    assert row['current_value_node_mounted'] is False
    assert row['served_without_plan_head'] is True


def test_the_ladder_drop_is_a_grounding_flip_not_an_ungrounded_trip_zero():
    """Trip 0 is ungrounded in ALL FIVE recall_3 rows; only d=150 is wrong.

    This is the claim the REPORT makes about `recall_3_150`. It rules out
    "trip 0 came back ungrounded" as the cause -- that happens at every
    distance, four of which answer correctly. What is different at d=150 is
    that trip 1 reads `grounded: true` on an unrelated node and displaces the
    correctly-planned head.
    """
    from scripts.grm_lt1 import score
    probes = {p['id']: p for p in json.loads(ct.FIXTURE.read_text())['probes']}
    seen = {}
    for row in ct.load_rows('A'):
        if not row['probe_id'].startswith('recall_3_'):
            continue
        info = row['memory']['route_info']
        trips = info['_route_observation']['trips']
        seen[row['probe_id']] = dict(
            category=score(row['memory']['answer'],
                           probes[row['probe_id']]['expected'])['category'],
            seated=list(info['fit_seated']),
            grounded=[t['grounded'] for t in trips])

    assert len(seen) == 5
    for pid, r in seen.items():
        assert len(r['grounded']) == 2, '%s ran %d trips' % (pid, len(r['grounded']))
        assert r['grounded'][0] is False, '%s trip 0 was grounded' % pid

    for pid in ('recall_3_10', 'recall_3_25', 'recall_3_50', 'recall_3_100'):
        assert seen[pid]['category'] == 'correct'
        assert seen[pid]['seated'] == [13]
        assert seen[pid]['grounded'] == [False, False]

    assert seen['recall_3_150']['category'] == 'wrong_value'
    assert seen['recall_3_150']['seated'] == [122]
    assert seen['recall_3_150']['grounded'] == [False, True]


def test_every_row_carries_the_route_provenance_fields(table):
    for row in table['rows']:
        assert row['admission_rule'] == 'margin_first'
        assert row['admission_policy'] == 'A-DEC'
        assert row['served_from'] == 'grm_e2e_session._probe_ladder_chat'
        prov = row['rt1_provenance']
        assert prov['route_backend'] == 'python'
        assert len(prov['rule_sha256']) == 64
        assert prov['trips'], 'ladder trips must be recorded'
        assert row['rs3_residency'] is not None
        assert row['recency_mounted_ids']


def test_no_source_turn_was_ever_a_recency_nominee(table):
    """The fixture gate's own rule, re-checked on the wrong rows."""
    for row in table['rows']:
        seated = {m['node'] for m in row['mounted_nodes']}
        assert not seated & set(row['recency_mounted_ids'] or [])


def test_arm_b_is_built_for_contrast_and_is_strictly_worse():
    """Arm B (defaults, w=256) is the contrast arm named by the order."""
    b = ct.build('B')
    assert b['wrong_rows'] == 30
    # Arm B fails the same two structural classes ...
    assert b['class_counts'][ct.FIXTURE_LINEAGE] == 5
    assert b['class_counts'][ct.ALIAS_NO_BASE] == 10
    # ... and adds a class arm A does not have at all: the right node mounted
    # and the reader still wrong. That is a reader/geometry difference, not a
    # lineage one, so it is evidence for the profile, not against the fix.
    assert b['class_counts'][ct.READER_WRONG] == 12


def test_markdown_renders_every_row_with_its_receipt(table):
    text = ct.markdown(table)
    for row in table['rows']:
        assert row['question_id'] in text
        assert row['receipt'] in text
    for name, count in table['class_counts'].items():
        assert '%s**: %d' % (name, count) in text
